//! Retrying LLM client with optional provider failover.
//!
//! Wraps one or more [`LlmTransport`]s with the retry/backoff policy ported
//! from the grok sampler. When `fallback_providers` is empty, behaviour is
//! unchanged: a single provider spends the full `max_retries` budget.
//! When a chain is configured, each provider is limited to a small
//! `failover_max_attempts` budget (~6s) before the next endpoint is tried,
//! and a tripped provider sits out for a cooldown so later requests stick
//! to the backup.
//!
//! Retries and failovers only happen before any content has been streamed
//! to the observer, so the UI never sees duplicated deltas.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::config::EngineConfig;
use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver};
use crate::llm::doom_loop_wire::DoomLoopRecoveryPolicy;
use crate::llm::failover::{
    compatibility_key, provider_fingerprint, resolve_provider_group, select_index,
    ProviderHealth, FAILOVER_RETRY_AFTER_INLINE_CAP,
};
use crate::llm::retry::{
    doom_loop_backoff, is_retry_vetoed_message, parse_retry_after, retry_backoff_with_jitter,
    RetryClass, RATE_LIMIT_RETRY_THRESHOLD,
};
use crate::llm::stream::collect_stream;
use crate::llm::transport::{retry_class_for_error, LlmTransport};
use crate::llm::types::{
    drop_colliding_function_tools, ChatCompletionRequest, CollectedTurn, HostedTool,
};

pub struct LlmClient {
    providers: Vec<ProviderSlot>,
    /// Single-provider retry budget (`EngineConfig.max_retries`).
    max_retries: u32,
    /// Chain-mode per-provider attempt budget (total tries, including the
    /// first). Ignored when `providers.len() == 1`.
    failover_max_attempts: u32,
    provider_cooldown_secs: u64,
    /// Independent budget for server doom-loop resamples (default 2).
    doom_loop_max_retries: u32,
    last_success: Mutex<Option<String>>,
}

struct ProviderSlot {
    transport: Arc<dyn LlmTransportObj>,
    /// Desensitized `host/model` for events.
    fingerprint: String,
    /// `{group}/{model}` — same key keeps encrypted thinking on failover.
    compat_key: String,
    model: String,
    api_backend: crate::llm::types::ApiBackend,
    enable_web_search: bool,
    health: Mutex<ProviderHealth>,
}

/// Object-safe wrapper so the client can hold `Arc<dyn ...>`.
pub trait LlmTransportObj: Send + Sync {
    fn request_stream_boxed<'a>(
        &'a self,
        req: &'a ChatCompletionRequest,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>>;
    fn name(&self) -> &str;
}

impl<T: LlmTransport> LlmTransportObj for T {
    fn request_stream_boxed<'a>(
        &'a self,
        req: &'a ChatCompletionRequest,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>> {
        Box::pin(self.request_stream(req))
    }
    fn name(&self) -> &str {
        <Self as LlmTransport>::name(self)
    }
}

enum ProviderAttemptError {
    /// Retryable/fatal-non-veto budget exhausted (or immediate fatal in
    /// chain mode). Caller should trip this provider and try the next.
    Failover {
        error: EngineError,
        cooldown_override: Option<Duration>,
    },
    /// Context overflow / `x-should-retry: false`: do not switch.
    Veto(EngineError),
    /// Empty-response budget spent on this provider: do not switch.
    EmptyExhausted(EngineError),
    /// Doom-loop resample budget spent: do not switch.
    DoomLoop(EngineError),
    /// Mid-stream failure or single-provider hard fail: surface as-is.
    Terminal(EngineError),
}

impl LlmClient {
    pub fn new(transport: Arc<dyn LlmTransportObj>, max_retries: u32) -> Self {
        let fingerprint = transport.name().to_string();
        Self {
            providers: vec![ProviderSlot {
                transport,
                fingerprint: fingerprint.clone(),
                compat_key: fingerprint,
                model: String::new(),
                api_backend: crate::llm::types::ApiBackend::ChatCompletions,
                enable_web_search: false,
                health: Mutex::new(ProviderHealth::default()),
            }],
            max_retries,
            failover_max_attempts: crate::llm::failover::DEFAULT_FAILOVER_MAX_ATTEMPTS,
            provider_cooldown_secs: crate::llm::failover::DEFAULT_PROVIDER_COOLDOWN_SECS,
            doom_loop_max_retries: DoomLoopRecoveryPolicy::DEFAULT_MAX_RETRIES,
            last_success: Mutex::new(None),
        }
    }

    /// Test/CLI helper: assemble a chain of named transports.
    pub fn with_chain(
        providers: Vec<(String, String, Arc<dyn LlmTransportObj>)>,
        max_retries: u32,
        failover_max_attempts: u32,
        provider_cooldown_secs: u64,
    ) -> Self {
        let slots = providers
            .into_iter()
            .map(|(fingerprint, model, transport)| ProviderSlot {
                transport,
                fingerprint: fingerprint.clone(),
                compat_key: fingerprint,
                model,
                api_backend: crate::llm::types::ApiBackend::ChatCompletions,
                enable_web_search: false,
                health: Mutex::new(ProviderHealth::default()),
            })
            .collect();
        Self {
            providers: slots,
            max_retries,
            failover_max_attempts,
            provider_cooldown_secs,
            doom_loop_max_retries: DoomLoopRecoveryPolicy::DEFAULT_MAX_RETRIES,
            last_success: Mutex::new(None),
        }
    }

    pub fn from_http(cfg: &EngineConfig) -> EngineResult<Self> {
        let mut slots = Vec::new();
        slots.push(slot_from_primary(cfg)?);
        for ep in &cfg.fallback_providers {
            slots.push(slot_from_endpoint(cfg, ep)?);
        }
        Ok(Self {
            providers: slots,
            max_retries: cfg.max_retries,
            failover_max_attempts: cfg.failover_max_attempts.max(1),
            provider_cooldown_secs: cfg.provider_cooldown_secs,
            doom_loop_max_retries: DoomLoopRecoveryPolicy::DEFAULT_MAX_RETRIES,
            last_success: Mutex::new(None),
        })
    }

    /// Compatibility key (`group/model`) of the provider that last
    /// produced a successful turn, falling back to the primary. Used to
    /// tag assistant history so same-group failover can keep encrypted
    /// thinking.
    pub fn origin_fingerprint(&self) -> String {
        self.last_success
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| {
                self.providers
                    .first()
                    .map(|p| p.compat_key.clone())
                    .unwrap_or_default()
            })
    }

    fn chain_mode(&self) -> bool {
        self.providers.len() > 1
    }

    fn primary_compat_key(&self) -> &str {
        self.providers
            .first()
            .map(|p| p.compat_key.as_str())
            .unwrap_or("")
    }

    fn select_next(&self, tried: &[usize]) -> Option<usize> {
        let now = Instant::now();
        let n = self.providers.len();
        if n == 0 {
            return None;
        }
        let snapshots: Vec<ProviderHealth> = self
            .providers
            .iter()
            .map(|p| p.health.lock().unwrap().clone())
            .collect();
        // Prefer an untried, not-cooling provider in chain order.
        if let Some(i) = snapshots
            .iter()
            .enumerate()
            .position(|(i, h)| !tried.contains(&i) && !h.is_cooling(now))
        {
            return Some(i);
        }
        // Any untried provider, even if cooling.
        let untried: Vec<usize> = (0..n).filter(|i| !tried.contains(i)).collect();
        if untried.is_empty() {
            return None;
        }
        let subset: Vec<ProviderHealth> = untried.iter().map(|&i| snapshots[i].clone()).collect();
        let local = select_index(&subset, now);
        Some(untried[local])
    }

    fn rewrite_request_for(
        &self,
        req: &ChatCompletionRequest,
        slot: &ProviderSlot,
    ) -> ChatCompletionRequest {
        let mut out = req.clone();
        if !slot.model.is_empty() {
            out.model = slot.model.clone();
        }
        out.hosted_tools = HostedTool::for_request(slot.enable_web_search, slot.api_backend);
        let tools = out.tools.take().unwrap_or_default();
        out.tools = drop_colliding_function_tools(tools, &out.hosted_tools);
        let primary = self.primary_compat_key();
        let target = slot.compat_key.as_str();
        out.messages = out
            .messages
            .iter()
            .map(|m| m.sanitized_for_provider(target, primary))
            .collect();
        out
    }

    /// Run one chat-completion request with retry / failover, streaming
    /// deltas to `observer`, and return the fully collected turn.
    pub async fn complete(
        &self,
        req: &ChatCompletionRequest,
        observer: &dyn AgentObserver,
    ) -> EngineResult<CollectedTurn> {
        let chain_mode = self.chain_mode();
        let mut tried: Vec<usize> = Vec::new();
        let mut last_error: Option<EngineError> = None;
        let mut tried_fps: Vec<String> = Vec::new();

        loop {
            let Some(idx) = self.select_next(&tried) else {
                let err = last_error.unwrap_or_else(|| {
                    EngineError::Llm("no LLM providers configured".into())
                });
                return Err(annotate_tried(err, &tried_fps));
            };
            tried.push(idx);
            let slot = &self.providers[idx];
            tried_fps.push(slot.fingerprint.clone());
            let rewritten = self.rewrite_request_for(req, slot);

            match self
                .try_provider(slot, &rewritten, observer, chain_mode)
                .await
            {
                Ok(turn) => {
                    slot.health.lock().unwrap().recover();
                    *self.last_success.lock().unwrap() = Some(slot.compat_key.clone());
                    return Ok(turn);
                }
                Err(ProviderAttemptError::Veto(e))
                | Err(ProviderAttemptError::EmptyExhausted(e))
                | Err(ProviderAttemptError::DoomLoop(e))
                | Err(ProviderAttemptError::Terminal(e)) => {
                    return Err(e);
                }
                Err(ProviderAttemptError::Failover {
                    error,
                    cooldown_override,
                }) => {
                    last_error = Some(error);
                    if !chain_mode {
                        return Err(last_error.take().unwrap());
                    }
                    let cooldown = slot.health.lock().unwrap().trip(
                        Instant::now(),
                        self.provider_cooldown_secs,
                        cooldown_override,
                    );
                    let Some(next_idx) = self.select_next(&tried) else {
                        return Err(annotate_tried(
                            last_error.take().unwrap(),
                            &tried_fps,
                        ));
                    };
                    let reason = last_error
                        .as_ref()
                        .map(|e| e.to_string())
                        .unwrap_or_default();
                    observer.on_event(AgentEvent::ProviderSwitched {
                        from: slot.fingerprint.clone(),
                        to: self.providers[next_idx].fingerprint.clone(),
                        reason,
                        cooldown_ms: cooldown.as_millis() as u64,
                    });
                }
            }
        }
    }

    async fn try_provider(
        &self,
        slot: &ProviderSlot,
        req: &ChatCompletionRequest,
        observer: &dyn AgentObserver,
        chain_mode: bool,
    ) -> Result<CollectedTurn, ProviderAttemptError> {
        let max_attempts = if chain_mode {
            self.failover_max_attempts.max(1)
        } else {
            self.max_retries.max(1)
        };
        let mut attempt: u32 = 0;
        let mut rate_limit_retries: u32 = 0;
        let mut doom_loop_retries: u32 = 0;
        loop {
            attempt += 1;
            match slot.transport.request_stream_boxed(req).await {
                Ok(stream) => match collect_stream(stream, observer).await {
                    Ok(turn) => {
                        if turn.is_empty() && attempt <= self.max_retries {
                            let wait = retry_backoff_with_jitter(attempt);
                            tracing::warn!(
                                "empty LLM response; retry {attempt} on {}",
                                slot.fingerprint
                            );
                            emit_retrying(observer, slot, attempt + 1, max_attempts, wait, "empty");
                            tokio::time::sleep(wait).await;
                            continue;
                        }
                        if turn.is_empty() {
                            return Err(ProviderAttemptError::EmptyExhausted(
                                EngineError::EmptyResponse,
                            ));
                        }
                        return Ok(turn);
                    }
                    Err(EngineError::DoomLoopServer(ref triggers)) => {
                        if doom_loop_retries < self.doom_loop_max_retries {
                            doom_loop_retries += 1;
                            let wait = doom_loop_backoff(doom_loop_retries);
                            tracing::warn!(
                                "server doom-loop ({triggers}); resample {doom_loop_retries}/{} after {wait:?}",
                                self.doom_loop_max_retries
                            );
                            tokio::time::sleep(wait).await;
                            continue;
                        }
                        return Err(ProviderAttemptError::DoomLoop(
                            EngineError::DoomLoopServer(triggers.clone()),
                        ));
                    }
                    Err(e) => {
                        // Mid-stream: deltas may already have reached the UI.
                        return Err(ProviderAttemptError::Terminal(e));
                    }
                },
                Err(e) => {
                    if is_veto(&e) {
                        return Err(ProviderAttemptError::Veto(e));
                    }
                    let class = retry_class_for_error(&e);
                    match class {
                        RetryClass::Fatal => {
                            if chain_mode {
                                return Err(ProviderAttemptError::Failover {
                                    error: e,
                                    cooldown_override: None,
                                });
                            }
                            return Err(ProviderAttemptError::Terminal(e));
                        }
                        RetryClass::RateLimited => {
                            rate_limit_retries += 1;
                            let wait = retry_after_from_error(&e)
                                .unwrap_or_else(|| retry_backoff_with_jitter(attempt));
                            let budget_exhausted = rate_limit_retries
                                > RATE_LIMIT_RETRY_THRESHOLD
                                || (!chain_mode && attempt > self.max_retries)
                                || (chain_mode && attempt >= max_attempts);
                            let too_long =
                                chain_mode && wait > FAILOVER_RETRY_AFTER_INLINE_CAP;
                            if budget_exhausted || too_long {
                                if chain_mode {
                                    return Err(ProviderAttemptError::Failover {
                                        error: e,
                                        cooldown_override: Some(wait),
                                    });
                                }
                                return Err(ProviderAttemptError::Terminal(e));
                            }
                            tracing::warn!(
                                "rate limited (429) on {}; waiting {wait:?}",
                                slot.fingerprint
                            );
                            emit_retrying(
                                observer,
                                slot,
                                attempt + 1,
                                max_attempts,
                                wait,
                                "status=429",
                            );
                            tokio::time::sleep(wait).await;
                        }
                        RetryClass::Retry => {
                            let exhausted = if chain_mode {
                                attempt >= max_attempts
                            } else {
                                attempt > self.max_retries
                            };
                            if exhausted {
                                if chain_mode {
                                    return Err(ProviderAttemptError::Failover {
                                        error: e,
                                        cooldown_override: None,
                                    });
                                }
                                return Err(ProviderAttemptError::Terminal(e));
                            }
                            let wait = retry_backoff_with_jitter(attempt);
                            tracing::warn!(
                                "LLM request error on {}: {e}; retry in {wait:?}",
                                slot.fingerprint
                            );
                            emit_retrying(
                                observer,
                                slot,
                                attempt + 1,
                                max_attempts,
                                wait,
                                &e.to_string(),
                            );
                            tokio::time::sleep(wait).await;
                        }
                    }
                }
            }
        }
    }
}

fn emit_retrying(
    observer: &dyn AgentObserver,
    slot: &ProviderSlot,
    attempt: u32,
    max_attempts: u32,
    wait: Duration,
    reason: &str,
) {
    observer.on_event(AgentEvent::Retrying {
        provider: slot.fingerprint.clone(),
        attempt,
        max_attempts,
        wait_ms: wait.as_millis() as u64,
        reason: reason.to_string(),
    });
}

fn is_veto(err: &EngineError) -> bool {
    match err {
        EngineError::Llm(msg) => is_retry_vetoed_message(msg),
        _ => false,
    }
}

fn annotate_tried(err: EngineError, fingerprints: &[String]) -> EngineError {
    if fingerprints.is_empty() {
        return err;
    }
    let summary = fingerprints.join(", ");
    match err {
        EngineError::Llm(msg) => EngineError::Llm(format!("{msg} [tried: {summary}]")),
        other => EngineError::Llm(format!("{other} [tried: {summary}]")),
    }
}

fn slot_from_primary(cfg: &EngineConfig) -> EngineResult<ProviderSlot> {
    let transport = Arc::new(http_transport(
        cfg,
        &cfg.base_url,
        &cfg.api_key,
        cfg.api_backend,
        cfg.client_profile,
        cfg.client_version.clone(),
        cfg.client_session_id.clone(),
        cfg.extra_headers.clone(),
        cfg.extra_body.clone(),
        cfg.doom_loop_check_enabled(),
    )?);
    let group = resolve_provider_group(cfg.provider_group.as_deref(), cfg.client_profile);
    Ok(ProviderSlot {
        fingerprint: provider_fingerprint(&cfg.base_url, &cfg.model),
        compat_key: compatibility_key(&group, &cfg.model),
        model: cfg.model.clone(),
        api_backend: cfg.api_backend,
        enable_web_search: cfg.enable_web_search,
        health: Mutex::new(ProviderHealth::default()),
        transport,
    })
}

fn slot_from_endpoint(
    cfg: &EngineConfig,
    ep: &crate::config::ProviderEndpoint,
) -> EngineResult<ProviderSlot> {
    let transport = Arc::new(http_transport(
        cfg,
        &ep.base_url,
        &ep.api_key,
        ep.api_backend,
        ep.client_profile,
        ep.client_version.clone(),
        ep.client_session_id.clone(),
        ep.extra_headers.clone(),
        ep.extra_body.clone(),
        cfg.doom_loop_check_enabled(),
    )?);
    let group = resolve_provider_group(ep.provider_group.as_deref(), ep.client_profile);
    Ok(ProviderSlot {
        fingerprint: provider_fingerprint(&ep.base_url, &ep.model),
        compat_key: compatibility_key(&group, &ep.model),
        model: ep.model.clone(),
        api_backend: ep.api_backend,
        enable_web_search: ep.enable_web_search,
        health: Mutex::new(ProviderHealth::default()),
        transport,
    })
}

fn http_transport(
    cfg: &EngineConfig,
    base_url: &str,
    api_key: &str,
    api_backend: crate::llm::types::ApiBackend,
    client_profile: crate::llm::profiles::ClientProfile,
    client_version: Option<String>,
    client_session_id: Option<String>,
    extra_headers: std::collections::HashMap<String, String>,
    extra_body: std::collections::HashMap<String, serde_json::Value>,
    doom_loop: bool,
) -> EngineResult<crate::llm::transport::HttpTransport> {
    let dumper = crate::llm::dumper::HttpDumper::new(cfg.http_dump.clone(), cfg.http_dumps_dir());
    crate::llm::transport::HttpTransport::new_with_all_options(
        base_url,
        api_key,
        Duration::from_secs(cfg.stream_idle_timeout_secs),
        api_backend,
        client_profile,
        client_version,
        client_session_id,
        extra_headers,
        extra_body,
        doom_loop,
        dumper,
    )
}

/// Best-effort extraction of a `Retry-After` hint embedded by transports in
/// the error message (`status=429 retry-after=<secs>`).
fn retry_after_from_error(err: &EngineError) -> Option<Duration> {
    let EngineError::Llm(msg) = err else {
        return None;
    };
    let marker = "retry-after=";
    let idx = msg.find(marker)?;
    let rest = &msg[idx + marker.len()..];
    let secs: u64 = rest.split_whitespace().next()?.parse().ok()?;
    parse_retry_after(&secs.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::RecordingObserver;
    use crate::llm::types::{ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, Role};
    use std::sync::atomic::{AtomicU32, Ordering};

    struct ScriptedTransport {
        name: String,
        /// Remaining failures before a successful text stream.
        remaining_fails: AtomicU32,
        error: String,
        hits: Arc<AtomicU32>,
    }

    impl ScriptedTransport {
        fn failing(name: &str, fails: u32, error: &str, hits: Arc<AtomicU32>) -> Arc<Self> {
            Arc::new(Self {
                name: name.into(),
                remaining_fails: AtomicU32::new(fails),
                error: error.into(),
                hits,
            })
        }
    }

    impl LlmTransport for ScriptedTransport {
        async fn request_stream(
            &self,
            _req: &ChatCompletionRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            let left = self.remaining_fails.load(Ordering::SeqCst);
            if left > 0 {
                self.remaining_fails.fetch_sub(1, Ordering::SeqCst);
                return Err(EngineError::Llm(self.error.clone()));
            }
            let stream = async_stream::stream! {
                let mut d = ChatChunkDelta::default();
                d.role = Some(Role::Assistant);
                d.content = Some("ok".into());
                yield Ok(ChatCompletionChunk {
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: d,
                        finish_reason: Some("stop".into()),
                    }],
                    ..Default::default()
                });
            };
            Ok(Box::pin(stream))
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    fn req() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "m".into(),
            messages: vec![crate::llm::types::ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            search_parameters: None,
            hosted_tools: Vec::new(),
        }
    }

    #[tokio::test]
    async fn chain_fails_over_after_fast_fail_budget() {
        tokio::time::pause();
        let a_hits = Arc::new(AtomicU32::new(0));
        let b_hits = Arc::new(AtomicU32::new(0));
        let a = ScriptedTransport::failing("a", u32::MAX, "status=503 busy", a_hits.clone());
        let b = ScriptedTransport::failing("b", 0, "", b_hits.clone());
        let client = LlmClient::with_chain(
            vec![
                ("host-a/m".into(), "m".into(), a),
                ("host-b/m".into(), "m".into(), b),
            ],
            5,
            3,
            120,
        );
        let observer = RecordingObserver::new();
        let turn = client.complete(&req(), &observer).await.unwrap();
        assert_eq!(turn.text, "ok");
        assert_eq!(a_hits.load(Ordering::SeqCst), 3);
        assert_eq!(b_hits.load(Ordering::SeqCst), 1);

        let events = observer.snapshot();
        let retrying = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::Retrying { .. }))
            .count();
        assert!(retrying >= 1, "expected Retrying events, got {events:?}");
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ProviderSwitched { from, to, .. } if from == "host-a/m" && to == "host-b/m"
        )));

        // Sticky: A is cooling, next request goes straight to B.
        let turn2 = client.complete(&req(), &observer).await.unwrap();
        assert_eq!(turn2.text, "ok");
        assert_eq!(a_hits.load(Ordering::SeqCst), 3);
        assert_eq!(b_hits.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn fatal_401_fails_over_immediately() {
        tokio::time::pause();
        let a_hits = Arc::new(AtomicU32::new(0));
        let b_hits = Arc::new(AtomicU32::new(0));
        let a = ScriptedTransport::failing("a", u32::MAX, "status=401 unauthorized", a_hits.clone());
        let b = ScriptedTransport::failing("b", 0, "", b_hits.clone());
        let client = LlmClient::with_chain(
            vec![
                ("host-a/m".into(), "m".into(), a),
                ("host-b/m".into(), "m".into(), b),
            ],
            5,
            3,
            120,
        );
        let observer = RecordingObserver::new();
        client.complete(&req(), &observer).await.unwrap();
        assert_eq!(a_hits.load(Ordering::SeqCst), 1);
        assert_eq!(b_hits.load(Ordering::SeqCst), 1);
        assert!(observer.snapshot().iter().any(|e| matches!(
            e,
            AgentEvent::ProviderSwitched { .. }
        )));
    }

    #[tokio::test]
    async fn context_overflow_does_not_failover() {
        let a_hits = Arc::new(AtomicU32::new(0));
        let b_hits = Arc::new(AtomicU32::new(0));
        let a = ScriptedTransport::failing(
            "a",
            u32::MAX,
            "status=400 maximum context length exceeded",
            a_hits.clone(),
        );
        let b = ScriptedTransport::failing("b", 0, "", b_hits.clone());
        let client = LlmClient::with_chain(
            vec![
                ("host-a/m".into(), "m".into(), a),
                ("host-b/m".into(), "m".into(), b),
            ],
            5,
            3,
            120,
        );
        let observer = RecordingObserver::new();
        let err = client.complete(&req(), &observer).await.unwrap_err();
        assert!(err.to_string().contains("context length"));
        assert_eq!(a_hits.load(Ordering::SeqCst), 1);
        assert_eq!(b_hits.load(Ordering::SeqCst), 0);
        assert!(!observer
            .snapshot()
            .iter()
            .any(|e| matches!(e, AgentEvent::ProviderSwitched { .. })));
    }

    #[tokio::test]
    async fn single_provider_keeps_large_retry_budget() {
        tokio::time::pause();
        let hits = Arc::new(AtomicU32::new(0));
        // 5 failures then success. Single-provider max_retries=5 allows
        // attempt 1..=5 to retry (existing `attempt > max_retries` rule),
        // so 5 fails + 1 success = 6 hits.
        let t = ScriptedTransport::failing("p", 5, "status=503", hits.clone());
        let client = LlmClient::new(t, 5);
        let observer = RecordingObserver::new();
        let turn = client.complete(&req(), &observer).await.unwrap();
        assert_eq!(turn.text, "ok");
        assert_eq!(hits.load(Ordering::SeqCst), 6);
        assert!(!observer
            .snapshot()
            .iter()
            .any(|e| matches!(e, AgentEvent::ProviderSwitched { .. })));
    }

    #[test]
    fn sanitize_strips_foreign_reasoning() {
        let item = crate::llm::types::ReasoningItem {
            id: "rs_abc".into(),
            summary: Vec::new(),
            content: None,
            encrypted_content: Some("enc".into()),
            status: None,
        };
        let mut msg = crate::llm::types::ChatMessage::assistant_with_reasoning(
            "hello",
            Some("thoughts".into()),
            vec![item],
            Some("sig".into()),
        );
        msg.origin = Some("grok_build/grok-4.6".into());

        let same_group = msg.sanitized_for_provider(
            "grok_build/grok-4.6",
            "grok_build/grok-4.6",
        );
        assert_eq!(same_group.reasoning_content.as_deref(), Some("thoughts"));
        assert_eq!(same_group.encrypted_reasoning.as_deref(), Some("sig"));
        assert_eq!(same_group.reasoning_items.len(), 1);
        assert_eq!(same_group.reasoning_items[0].id, "rs_abc");

        let group_change = msg.sanitized_for_provider(
            "claude_code/grok-4.6",
            "grok_build/grok-4.6",
        );
        assert!(group_change.reasoning_content.is_none());
        assert!(group_change.encrypted_reasoning.is_none());
        assert!(group_change.reasoning_items.is_empty());
        assert_eq!(group_change.content.as_deref(), Some("hello"));

        let model_change = msg.sanitized_for_provider(
            "grok_build/grok-3",
            "grok_build/grok-4.6",
        );
        assert!(model_change.reasoning_items.is_empty());
    }
}
