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
//! to the observer, so the UI never sees duplicated deltas — except for
//! SSE idle-timeout prefix continuation (same provider once, then failover).

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
use crate::llm::stream::{collect_stream, CollectStreamError};
use crate::llm::transport::{retry_class_for_error, LlmTransport, LlmTurnContext};
use crate::llm::types::{
    drop_colliding_function_tools, CollectedTurn, ConversationRequest, HostedTool,
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

/// A logical main-agent or subagent turn.
///
/// Reusing this value across the tool loop keeps provider-specific ephemeral
/// state available inside the turn. Creating a new value guarantees that the
/// state cannot leak into another user turn or concurrently running subagent.
pub struct LlmTurnSession<'a> {
    client: &'a LlmClient,
    context: LlmTurnContext,
}

impl LlmTurnSession<'_> {
    pub async fn complete(
        &self,
        req: &ConversationRequest,
        observer: &dyn AgentObserver,
    ) -> EngineResult<CollectedTurn> {
        self.client
            .complete_in_context(req, observer, &self.context)
            .await
    }
}

struct ProviderSlot {
    transport: Arc<dyn LlmTransportObj>,
    /// Desensitized `host/model` for events.
    fingerprint: String,
    /// `{group}/{model}` — same key keeps encrypted thinking on failover.
    compat_key: String,
    model: String,
    api_backend: crate::llm::types::ApiBackend,
    reasoning_effort: Option<crate::llm::types::ReasoningEffort>,
    enable_web_search: bool,
    web_search_options: Option<crate::llm::types::WebSearchOptions>,
    enable_x_search: bool,
    x_search_options: Option<crate::llm::types::XSearchOptions>,
    health: Mutex<ProviderHealth>,
}

/// Object-safe wrapper so the client can hold `Arc<dyn ...>`.
pub trait LlmTransportObj: Send + Sync {
    fn request_stream_boxed<'a>(
        &'a self,
        req: &'a ConversationRequest,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>>;
    fn request_stream_in_context_boxed<'a>(
        &'a self,
        req: &'a ConversationRequest,
        context: &'a LlmTurnContext,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>>;
    fn name(&self) -> &str;
}

impl<T: LlmTransport> LlmTransportObj for T {
    fn request_stream_boxed<'a>(
        &'a self,
        req: &'a ConversationRequest,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>> {
        Box::pin(self.request_stream(req))
    }
    fn request_stream_in_context_boxed<'a>(
        &'a self,
        req: &'a ConversationRequest,
        context: &'a LlmTurnContext,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>> {
        self.request_stream_in_context(req, context)
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
        /// Partial turn from an idle-timeout continuation. The next
        /// provider prefixes this instead of regenerating from scratch.
        continue_from: Option<CollectedTurn>,
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
    /// Start an isolated logical agent turn.
    pub fn begin_turn(&self) -> LlmTurnSession<'_> {
        LlmTurnSession {
            client: self,
            context: LlmTurnContext::new(),
        }
    }

    pub fn new(transport: Arc<dyn LlmTransportObj>, max_retries: u32) -> Self {
        let fingerprint = transport.name().to_string();
        Self {
            providers: vec![ProviderSlot {
                transport,
                fingerprint: fingerprint.clone(),
                compat_key: fingerprint,
                model: String::new(),
                api_backend: crate::llm::types::ApiBackend::ChatCompletions,
                reasoning_effort: None,
                enable_web_search: false,
                web_search_options: None,
                enable_x_search: false,
                x_search_options: None,
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
                reasoning_effort: None,
                enable_web_search: false,
                web_search_options: None,
                enable_x_search: false,
                x_search_options: None,
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
        req: &ConversationRequest,
        slot: &ProviderSlot,
    ) -> ConversationRequest {
        let mut out = req.clone();
        if !slot.model.is_empty() {
            out.model = slot.model.clone();
        }
        if let Some(effort) = slot.reasoning_effort {
            out.reasoning_effort = Some(effort);
        }
        out.hosted_tools = HostedTool::for_request_with_options(
            slot.enable_web_search,
            slot.web_search_options.clone(),
            slot.enable_x_search,
            slot.x_search_options.clone(),
            slot.api_backend,
        );
        let tools = out.tools.take().unwrap_or_default();
        out.tools = drop_colliding_function_tools(tools, &out.hosted_tools);
        let primary = self.primary_compat_key();
        let target = slot.compat_key.as_str();
        out.items = crate::llm::failover::sanitize_items_for_provider(&out.items, target, primary);
        out
    }

    /// Run one chat-completion request with retry / failover, streaming
    /// deltas to `observer`, and return the fully collected turn.
    pub async fn complete(
        &self,
        req: &ConversationRequest,
        observer: &dyn AgentObserver,
    ) -> EngineResult<CollectedTurn> {
        self.begin_turn().complete(req, observer).await
    }

    async fn complete_in_context(
        &self,
        req: &ConversationRequest,
        observer: &dyn AgentObserver,
        context: &LlmTurnContext,
    ) -> EngineResult<CollectedTurn> {
        let chain_mode = self.chain_mode();
        let mut tried: Vec<usize> = Vec::new();
        let mut last_error: Option<EngineError> = None;
        let mut tried_fps: Vec<String> = Vec::new();
        let skipper = PrefixSkippingObserver::new(observer);
        let mut continue_from: Option<CollectedTurn> = None;

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
            let rewritten = if let Some(ref partial) = continue_from {
                skipper.set_skip(&partial.text, &partial.reasoning);
                // Response ids are host-bound; keep encrypted reasoning
                // items in the prefix but drop previous_response_id.
                prefix_continue_request(&rewritten, partial, false)
            } else {
                rewritten
            };

            match self
                .try_provider(slot, &rewritten, &skipper, chain_mode, context)
                .await
            {
                Ok(turn) => {
                    let turn = match &continue_from {
                        Some(partial) => merge_continued_turn(partial, turn),
                        None => turn,
                    };
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
                    continue_from: cf,
                }) => {
                    if let Some(partial) = cf {
                        let merged = match continue_from {
                            Some(prev) => merge_continued_turn(&prev, partial),
                            None => partial,
                        };
                        skipper.set_skip(&merged.text, &merged.reasoning);
                        continue_from = Some(merged);
                    }
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
                    skipper.on_event(AgentEvent::ProviderSwitched {
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
        req: &ConversationRequest,
        observer: &PrefixSkippingObserver<'_>,
        chain_mode: bool,
        context: &LlmTurnContext,
    ) -> Result<CollectedTurn, ProviderAttemptError> {
        let max_attempts = if chain_mode {
            self.failover_max_attempts.max(1)
        } else {
            self.max_retries.max(1)
        };
        let mut attempt: u32 = 0;
        let mut rate_limit_retries: u32 = 0;
        let mut doom_loop_retries: u32 = 0;
        let mut live_req = req.clone();
        let mut continued_once = false;
        let mut prefix: Option<CollectedTurn> = None;
        loop {
            attempt += 1;
            match slot
                .transport
                .request_stream_in_context_boxed(&live_req, context)
                .await
            {
                Ok(stream) => match collect_stream(stream, observer).await {
                    Ok(turn) => {
                        let turn = match &prefix {
                            Some(partial) => merge_continued_turn(partial, turn),
                            None => turn,
                        };
                        if turn.is_empty() {
                            // In chain mode use the per-provider failover budget;
                            // in single-provider mode use the global retry budget.
                            let budget = if chain_mode {
                                max_attempts
                            } else {
                                self.max_retries
                            };
                            if attempt < budget {
                                let wait = retry_backoff_with_jitter(attempt);
                                tracing::warn!(
                                    "empty LLM response; retry {attempt}/{budget} on {}",
                                    slot.fingerprint
                                );
                                emit_retrying(
                                    observer,
                                    slot,
                                    attempt + 1,
                                    max_attempts,
                                    wait,
                                    "empty",
                                );
                                tokio::time::sleep(wait).await;
                                continue;
                            }
                            // Budget spent: in chain mode escalate to failover so the
                            // next provider in the chain is tried; otherwise surface as
                            // a terminal empty-response error.
                            if chain_mode {
                                return Err(ProviderAttemptError::Failover {
                                    error: EngineError::EmptyResponse,
                                    cooldown_override: None,
                                    continue_from: None,
                                });
                            }
                            return Err(ProviderAttemptError::EmptyExhausted(
                                EngineError::EmptyResponse,
                            ));
                        }
                        return Ok(turn);
                    }
                    Err(CollectStreamError::Other(EngineError::DoomLoopServer(ref triggers))) => {
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
                    Err(CollectStreamError::IdleTimeout { partial, timeout }) => {
                        let combined = match &prefix {
                            Some(prev) => merge_continued_turn(prev, partial),
                            None => partial,
                        };
                        if should_accept_partial_as_complete(&combined) {
                            return Ok(combined);
                        }
                        if !continued_once && can_prefix_continue(&combined) {
                            continued_once = true;
                            attempt = attempt.saturating_sub(1);
                            prefix = Some(combined.clone());
                            observer.set_skip(&combined.text, &combined.reasoning);
                            live_req = prefix_continue_request(&live_req, &combined, true);
                            tracing::warn!(
                                "idle timeout after {timeout:?} with continuable prefix; retrying once on {}",
                                slot.fingerprint
                            );
                            emit_retrying(
                                observer,
                                slot,
                                attempt + 1,
                                max_attempts,
                                Duration::from_millis(0),
                                "idle-timeout-continue",
                            );
                            continue;
                        }
                        let err = EngineError::StreamIdleTimeout(timeout);
                        if chain_mode {
                            return Err(ProviderAttemptError::Failover {
                                error: err,
                                cooldown_override: None,
                                continue_from: can_prefix_continue(&combined)
                                    .then_some(combined),
                            });
                        }
                        return Err(ProviderAttemptError::Terminal(err));
                    }
                    Err(CollectStreamError::Other(e)) => {
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
                                    continue_from: prefix.clone(),
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
                                        continue_from: prefix.clone(),
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
                                        continue_from: prefix.clone(),
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

/// Drops already-emitted prefix tokens so a continuation / failover stream
/// does not duplicate text the UI has already rendered.
struct PrefixSkippingObserver<'a> {
    inner: &'a dyn AgentObserver,
    skip_text: Mutex<String>,
    skip_reasoning: Mutex<String>,
}

impl<'a> PrefixSkippingObserver<'a> {
    fn new(inner: &'a dyn AgentObserver) -> Self {
        Self {
            inner,
            skip_text: Mutex::new(String::new()),
            skip_reasoning: Mutex::new(String::new()),
        }
    }

    fn set_skip(&self, text: &str, reasoning: &str) {
        *self.skip_text.lock().unwrap() = text.to_string();
        *self.skip_reasoning.lock().unwrap() = reasoning.to_string();
    }
}

impl AgentObserver for PrefixSkippingObserver<'_> {
    fn on_event(&self, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta { text } => {
                let rest = skip_prefix_chunk(&mut self.skip_text.lock().unwrap(), &text);
                if !rest.is_empty() {
                    self.inner.on_event(AgentEvent::TextDelta { text: rest });
                }
            }
            AgentEvent::ReasoningDelta { text } => {
                let rest = skip_prefix_chunk(&mut self.skip_reasoning.lock().unwrap(), &text);
                if !rest.is_empty() {
                    self.inner
                        .on_event(AgentEvent::ReasoningDelta { text: rest });
                }
            }
            other => self.inner.on_event(other),
        }
    }
}

fn skip_prefix_chunk(remaining: &mut String, incoming: &str) -> String {
    if remaining.is_empty() || incoming.is_empty() {
        return incoming.to_string();
    }
    if remaining.starts_with(incoming) {
        remaining.drain(..incoming.len());
        return String::new();
    }
    if incoming.starts_with(remaining.as_str()) {
        let rest = incoming[remaining.len()..].to_string();
        remaining.clear();
        return rest;
    }
    remaining.clear();
    incoming.to_string()
}

fn has_incomplete_tool_calls(turn: &CollectedTurn) -> bool {
    turn.tool_calls.iter().any(|tc| {
        if tc.kind == "server" {
            return false;
        }
        serde_json::from_str::<serde::de::IgnoredAny>(&tc.function.arguments).is_err()
    })
}

/// Idle-timeout partial is already a usable tool-calling hop: execute it.
fn should_accept_partial_as_complete(turn: &CollectedTurn) -> bool {
    !turn.tool_calls.is_empty() && !has_incomplete_tool_calls(turn)
}

/// Prefix-continue only when we have visible text and no tool calls at all.
fn can_prefix_continue(turn: &CollectedTurn) -> bool {
    !turn.text.trim().is_empty() && turn.tool_calls.is_empty()
}

fn prefix_continue_request(
    req: &ConversationRequest,
    partial: &CollectedTurn,
    keep_response_id: bool,
) -> ConversationRequest {
    let mut out = req.clone();
    if keep_response_id {
        out.previous_response_id = partial
            .response_id
            .as_ref()
            .filter(|s| !s.is_empty())
            .cloned();
    } else {
        out.previous_response_id = None;
    }
    let mut prefix_items = partial.items.clone();
    if prefix_items.is_empty() {
        prefix_items = synthesize_partial_items(partial);
    }
    out.items.extend(prefix_items);
    out
}

fn synthesize_partial_items(
    partial: &CollectedTurn,
) -> Vec<crate::conversation::ConversationItem> {
    use crate::conversation::{AssistantItem, ConversationItem};
    let mut items = Vec::new();
    for r in &partial.reasoning_items {
        items.push(ConversationItem::Reasoning(r.clone()));
    }
    items.push(ConversationItem::Assistant(AssistantItem {
        content: partial.text.clone(),
        tool_calls: Vec::new(),
        reasoning_content: if partial.reasoning.is_empty() {
            None
        } else {
            Some(partial.reasoning.clone())
        },
        encrypted_reasoning: partial.encrypted_reasoning.clone(),
        origin: None,
    }));
    items
}

fn merge_reasoning_items(
    old: &[crate::llm::types::ReasoningItem],
    new: &[crate::llm::types::ReasoningItem],
) -> Vec<crate::llm::types::ReasoningItem> {
    crate::llm::types::merge_reasoning_items(old, new)
}

fn merge_continued_turn(partial: &CollectedTurn, cont: CollectedTurn) -> CollectedTurn {
    let text = if cont.text.starts_with(&partial.text) {
        cont.text
    } else if cont.text.is_empty() {
        partial.text.clone()
    } else if partial.text.is_empty() {
        cont.text
    } else {
        format!("{}{}", partial.text, cont.text)
    };
    let reasoning = if cont.reasoning.starts_with(&partial.reasoning) {
        cont.reasoning
    } else if cont.reasoning.is_empty() {
        partial.reasoning.clone()
    } else if partial.reasoning.is_empty() {
        cont.reasoning
    } else {
        format!("{}{}", partial.reasoning, cont.reasoning)
    };
    let mut merged = CollectedTurn {
        items: merge_continued_items(&partial.items, &cont.items, &text, &reasoning),
        text,
        reasoning,
        reasoning_items: merge_reasoning_items(&partial.reasoning_items, &cont.reasoning_items),
        encrypted_reasoning: cont
            .encrypted_reasoning
            .or_else(|| partial.encrypted_reasoning.clone()),
        tool_calls: if cont.tool_calls.is_empty() {
            partial.tool_calls.clone()
        } else {
            cont.tool_calls
        },
        finish_reason: cont.finish_reason.or_else(|| partial.finish_reason.clone()),
        usage: cont.usage.or_else(|| partial.usage.clone()),
        model: if cont.model.is_empty() {
            partial.model.clone()
        } else {
            cont.model
        },
        response_id: cont.response_id.or_else(|| partial.response_id.clone()),
        final_output: None,
    };
    if merged.items.is_empty() {
        merged.items = synthesize_partial_items(&merged);
    }
    merged.sync_derived_views();
    merged
}

fn merge_continued_items(
    partial: &[crate::conversation::ConversationItem],
    cont: &[crate::conversation::ConversationItem],
    text: &str,
    reasoning: &str,
) -> Vec<crate::conversation::ConversationItem> {
    use crate::conversation::ConversationItem;
    let mut out = if partial.is_empty() {
        cont.to_vec()
    } else {
        partial.to_vec()
    };
    // Drop trailing assistant from partial so we can replace it.
    while matches!(out.last(), Some(ConversationItem::Assistant(_))) {
        out.pop();
    }
    // Merge reasoning by id from continuation.
    for item in cont {
        match item {
            ConversationItem::Reasoning(r) => {
                if r.id.is_empty() {
                    if !out.iter().any(|o| matches!(o, ConversationItem::Reasoning(x) if x == r)) {
                        out.push(item.clone());
                    }
                } else if let Some(ConversationItem::Reasoning(existing)) = out.iter_mut().find(|o| {
                    matches!(o, ConversationItem::Reasoning(x) if x.id == r.id)
                }) {
                    if r.encrypted_content.is_some() {
                        existing.encrypted_content = r.encrypted_content.clone();
                    }
                    if !r.summary.is_empty() {
                        existing.summary = r.summary.clone();
                    }
                    if r.content.is_some() {
                        existing.content = r.content.clone();
                    }
                } else {
                    out.push(item.clone());
                }
            }
            ConversationItem::BackendToolCall(_) => {
                out.push(item.clone());
            }
            ConversationItem::Assistant(a) => {
                let mut merged_a = a.clone();
                merged_a.content = text.to_string();
                if merged_a.reasoning_content.is_none() && !reasoning.is_empty() {
                    merged_a.reasoning_content = Some(reasoning.to_string());
                }
                out.push(ConversationItem::Assistant(merged_a));
            }
            _ => {}
        }
    }
    if !out.iter().any(|i| matches!(i, ConversationItem::Assistant(_))) {
        out.push(crate::conversation::ConversationItem::assistant(text));
    }
    out
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
        reasoning_effort: cfg.reasoning_effort,
        enable_web_search: cfg.enable_web_search,
        web_search_options: cfg.web_search_options.clone(),
        enable_x_search: cfg.enable_x_search,
        x_search_options: cfg.x_search_options.clone(),
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
        reasoning_effort: ep.reasoning_effort.or(cfg.reasoning_effort),
        enable_web_search: ep.enable_web_search,
        web_search_options: cfg.web_search_options.clone(),
        enable_x_search: ep.enable_x_search,
        x_search_options: ep.x_search_options.clone().or_else(|| cfg.x_search_options.clone()),
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
            _req: &ConversationRequest,
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

    struct ContextProbeTransport {
        observed: Mutex<Vec<Option<String>>>,
    }

    impl ContextProbeTransport {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                observed: Mutex::new(Vec::new()),
            })
        }
    }

    impl LlmTransport for ContextProbeTransport {
        async fn request_stream(
            &self,
            _req: &ConversationRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            let stream = async_stream::stream! {
                let mut delta = ChatChunkDelta::default();
                delta.role = Some(Role::Assistant);
                delta.content = Some("ok".into());
                yield Ok(ChatCompletionChunk {
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta,
                        finish_reason: Some("stop".into()),
                    }],
                    ..Default::default()
                });
            };
            Ok(Box::pin(stream))
        }

        fn request_stream_in_context<'a>(
            &'a self,
            req: &'a ConversationRequest,
            context: &'a LlmTurnContext,
        ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>>
        {
            self.observed
                .lock()
                .unwrap()
                .push(context.turn_state("probe"));
            context.set_turn_state("probe", "sticky".to_string());
            Box::pin(self.request_stream(req))
        }

        fn name(&self) -> &str {
            "context-probe"
        }
    }

    fn req() -> ConversationRequest {
        ConversationRequest {
            model: "m".into(),
            items: vec![crate::conversation::ConversationItem::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: Vec::new(),
            previous_response_id: None,
            image_bytes: crate::llm::image::ImageBytesStore::default(),
            audio_bytes: crate::llm::image::AudioBytesStore::default(),
        }
    }

    #[tokio::test]
    async fn logical_turn_context_is_reused_but_never_crosses_turns() {
        let transport = ContextProbeTransport::new();
        let client = LlmClient::new(transport.clone(), 1);
        let observer = RecordingObserver::new();
        let request = req();

        // Main agent plus two concurrent subagents each own a context.
        let main = client.begin_turn();
        let subagent_one = client.begin_turn();
        let subagent_two = client.begin_turn();

        let first = tokio::join!(
            main.complete(&request, &observer),
            subagent_one.complete(&request, &observer),
            subagent_two.complete(&request, &observer),
        );
        first.0.unwrap();
        first.1.unwrap();
        first.2.unwrap();

        let second = tokio::join!(
            main.complete(&request, &observer),
            subagent_one.complete(&request, &observer),
            subagent_two.complete(&request, &observer),
        );
        second.0.unwrap();
        second.1.unwrap();
        second.2.unwrap();

        // LlmClient::complete starts a fresh turn and must not inherit tokens.
        client.complete(&request, &observer).await.unwrap();

        let observed = transport.observed.lock().unwrap().clone();
        assert_eq!(observed.len(), 7);
        assert!(
            observed[..3].iter().all(Option::is_none),
            "concurrent first hops must not share turn state: {observed:?}"
        );
        assert!(
            observed[3..6]
                .iter()
                .all(|state| state.as_deref() == Some("sticky")),
            "same-turn hops must reuse turn state: {observed:?}"
        );
        assert_eq!(observed[6], None);
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
        assert_eq!(group_change.content_text(), "hello");

        let model_change = msg.sanitized_for_provider(
            "grok_build/grok-3",
            "grok_build/grok-4.6",
        );
        assert!(model_change.reasoning_items.is_empty());
    }

    // ── Empty-response failover regression tests ──────────────────────────────

    /// A transport that always returns an empty (no-content) stream.
    struct EmptyTransport {
        name: String,
        hits: Arc<AtomicU32>,
    }

    impl EmptyTransport {
        fn new(name: &str, hits: Arc<AtomicU32>) -> Arc<Self> {
            Arc::new(Self {
                name: name.into(),
                hits,
            })
        }
    }

    impl LlmTransport for EmptyTransport {
        async fn request_stream(
            &self,
            _req: &ConversationRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            // Return a stream that yields a stop chunk with no content,
            // so `CollectedTurn::is_empty()` returns true.
            let stream = async_stream::stream! {
                yield Ok(ChatCompletionChunk {
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatChunkDelta::default(),
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

    /// Regression: before the fix, an always-empty second provider (openlux)
    /// consumed the global `max_retries` budget (15+) instead of the
    /// per-provider `failover_max_attempts` budget (3), and returned
    /// `EmptyExhausted` — which the outer loop treated as terminal — so the
    /// third provider (wududu) was never tried.
    ///
    /// After the fix: empty budget in chain mode raises `Failover`, the outer
    /// loop trips the provider and continues to the third one.
    #[tokio::test]
    async fn empty_response_chain_fails_over_to_third_provider() {
        tokio::time::pause();

        let a_hits = Arc::new(AtomicU32::new(0)); // hermes — always 502 (HTTP error)
        let b_hits = Arc::new(AtomicU32::new(0)); // openlux — always empty
        let c_hits = Arc::new(AtomicU32::new(0)); // wududu  — succeeds

        let a = ScriptedTransport::failing("a", u32::MAX, "status=502", a_hits.clone());
        let b = EmptyTransport::new("b", b_hits.clone());
        let c = ScriptedTransport::failing("c", 0, "", c_hits.clone());

        // failover_max_attempts = 3: each provider gets at most 3 attempts.
        let client = LlmClient::with_chain(
            vec![
                ("host-a/m".into(), "m".into(), a as Arc<dyn LlmTransportObj>),
                ("host-b/m".into(), "m".into(), b as Arc<dyn LlmTransportObj>),
                ("host-c/m".into(), "m".into(), c as Arc<dyn LlmTransportObj>),
            ],
            15,  // max_retries (single-provider budget, irrelevant in chain mode)
            3,   // failover_max_attempts
            120,
        );

        let observer = RecordingObserver::new();
        let turn = client.complete(&req(), &observer).await.unwrap();
        assert_eq!(turn.text, "ok", "third provider should have answered");

        // Provider A exhausted its 3-attempt budget (HTTP 502 retries).
        assert_eq!(
            a_hits.load(Ordering::SeqCst),
            3,
            "A should have been tried exactly failover_max_attempts times"
        );
        // Provider B exhausted its 3-attempt budget (empty responses).
        // Before fix: b_hits could be >> 3.
        assert_eq!(
            b_hits.load(Ordering::SeqCst),
            3,
            "B should have been tried exactly failover_max_attempts times (empty-response budget)"
        );
        // Provider C answered on the first attempt.
        assert_eq!(c_hits.load(Ordering::SeqCst), 1);

        let events = observer.snapshot();
        // Two ProviderSwitched events: A→B and B→C.
        let switches: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ProviderSwitched { .. }))
            .collect();
        assert_eq!(switches.len(), 2, "expected two ProviderSwitched events");
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ProviderSwitched { from, to, .. }
                if from == "host-a/m" && to == "host-b/m"
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ProviderSwitched { from, to, .. }
                if from == "host-b/m" && to == "host-c/m"
        )));
    }

    /// Single-provider: empty responses must still retry up to `max_retries`
    /// without triggering a failover (no chain to switch to).
    #[tokio::test]
    async fn single_provider_empty_retries_full_budget() {
        tokio::time::pause();
        let hits = Arc::new(AtomicU32::new(0));
        let t = EmptyTransport::new("p", hits.clone());
        // max_retries = 2, budget = 2: retry while attempt < 2.
        // attempt 1 < 2 → retry; attempt 2 >= 2 → EmptyExhausted.
        // Total hits = 2.
        let client = LlmClient::new(t as Arc<dyn LlmTransportObj>, 2);
        let observer = RecordingObserver::new();
        let err = client.complete(&req(), &observer).await.unwrap_err();
        assert!(
            matches!(err, EngineError::EmptyResponse),
            "single-provider empty should surface as EmptyResponse, got: {err}"
        );
        // attempt 1 (initial, retries) + attempt 2 (no retry) = 2 hits
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        assert!(!observer
            .snapshot()
            .iter()
            .any(|e| matches!(e, AgentEvent::ProviderSwitched { .. })));
    }

    struct IdleContinueTransport {
        name: String,
        hits: Arc<AtomicU32>,
        captured: Arc<Mutex<Option<ConversationRequest>>>,
    }

    impl LlmTransport for IdleContinueTransport {
        async fn request_stream(
            &self,
            req: &ConversationRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            let n = self.hits.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                let stream = async_stream::stream! {
                    let mut d = ChatChunkDelta::default();
                    d.content = Some("Hello ".into());
                    d.reasoning_items.push(crate::llm::types::ReasoningItem {
                        id: "rs_1".into(),
                        summary: Vec::new(),
                        content: None,
                        encrypted_content: Some("enc".into()),
                        status: None,
                    });
                    yield Ok(ChatCompletionChunk {
                        id: "resp_abc".into(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: d,
                            finish_reason: None,
                        }],
                        ..Default::default()
                    });
                    yield Err(EngineError::StreamIdleTimeout(Duration::from_secs(120)));
                };
                return Ok(Box::pin(stream));
            }
            *self.captured.lock().unwrap() = Some(req.clone());
            let stream = async_stream::stream! {
                let mut d = ChatChunkDelta::default();
                d.content = Some("world".into());
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

    #[tokio::test]
    async fn idle_timeout_prefix_continues_once_keeping_response_id_and_encrypted_reasoning() {
        let hits = Arc::new(AtomicU32::new(0));
        let t = Arc::new(IdleContinueTransport {
            name: "p".into(),
            hits: hits.clone(),
            captured: Arc::new(Mutex::new(None)),
        });
        let captured = t.captured.clone();
        let client = LlmClient::new(t as Arc<dyn LlmTransportObj>, 5);
        let observer = RecordingObserver::new();
        let turn = client.complete(&req(), &observer).await.unwrap();
        assert_eq!(turn.text, "Hello world");
        assert_eq!(hits.load(Ordering::SeqCst), 2);

        let cont = captured.lock().unwrap().clone().expect("continuation request");
        assert_eq!(cont.previous_response_id.as_deref(), Some("resp_abc"));
        let last = cont.items.last().expect("prefix assistant");
        match last {
            crate::conversation::ConversationItem::Assistant(a) => {
                assert_eq!(a.content, "Hello ");
            }
            other => panic!("expected assistant prefix, got {other:?}"),
        }
        let reasoning: Vec<_> = cont
            .items
            .iter()
            .filter_map(|i| match i {
                crate::conversation::ConversationItem::Reasoning(r) => Some(r),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0].id, "rs_1");
        assert_eq!(reasoning[0].encrypted_content.as_deref(), Some("enc"));

        let texts: Vec<_> = observer
            .snapshot()
            .into_iter()
            .filter_map(|e| match e {
                AgentEvent::TextDelta { text } => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hello ".to_string(), "world".to_string()]);
    }

    struct AlwaysIdleAfterText {
        name: String,
        hits: Arc<AtomicU32>,
        text: String,
    }

    impl LlmTransport for AlwaysIdleAfterText {
        async fn request_stream(
            &self,
            _req: &ConversationRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            let text = self.text.clone();
            let stream = async_stream::stream! {
                let mut d = ChatChunkDelta::default();
                d.content = Some(text);
                yield Ok(ChatCompletionChunk {
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: d,
                        finish_reason: None,
                    }],
                    ..Default::default()
                });
                yield Err(EngineError::StreamIdleTimeout(Duration::from_secs(120)));
            };
            Ok(Box::pin(stream))
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[tokio::test]
    async fn idle_timeout_second_failure_fails_over_with_prefix() {
        let a_hits = Arc::new(AtomicU32::new(0));
        let b_hits = Arc::new(AtomicU32::new(0));
        let a = Arc::new(AlwaysIdleAfterText {
            name: "a".into(),
            hits: a_hits.clone(),
            text: "Hello ".into(),
        });
        let b = ScriptedTransport::failing("b", 0, "", b_hits.clone());
        let client = LlmClient::with_chain(
            vec![
                ("host-a/m".into(), "m".into(), a as Arc<dyn LlmTransportObj>),
                ("host-b/m".into(), "m".into(), b as Arc<dyn LlmTransportObj>),
            ],
            5,
            3,
            120,
        );
        let observer = RecordingObserver::new();
        let turn = client.complete(&req(), &observer).await.unwrap();
        assert_eq!(turn.text, "Hello ok");
        assert_eq!(a_hits.load(Ordering::SeqCst), 2, "same provider continued once");
        assert_eq!(b_hits.load(Ordering::SeqCst), 1);
        assert!(observer.snapshot().iter().any(|e| matches!(
            e,
            AgentEvent::ProviderSwitched { from, to, .. }
                if from == "host-a/m" && to == "host-b/m"
        )));
    }

    struct IncompleteToolIdle {
        hits: Arc<AtomicU32>,
    }

    impl LlmTransport for IncompleteToolIdle {
        async fn request_stream(
            &self,
            _req: &ConversationRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            let stream = async_stream::stream! {
                let mut d = ChatChunkDelta::default();
                d.tool_calls.push(crate::llm::types::ToolCallDelta {
                    index: 0,
                    id: Some("c1".into()),
                    kind: Some("function".into()),
                    function: Some(crate::llm::types::ToolCallFunctionDelta {
                        name: Some("web_search".into()),
                        arguments: Some("{\"q\":".into()),
                    }),
                    thought_signature: None,
                });
                yield Ok(ChatCompletionChunk {
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: d,
                        finish_reason: None,
                    }],
                    ..Default::default()
                });
                yield Err(EngineError::StreamIdleTimeout(Duration::from_secs(120)));
            };
            Ok(Box::pin(stream))
        }

        fn name(&self) -> &str {
            "p"
        }
    }

    #[tokio::test]
    async fn idle_timeout_with_incomplete_tool_does_not_continue() {
        let hits = Arc::new(AtomicU32::new(0));
        let t = Arc::new(IncompleteToolIdle { hits: hits.clone() });
        let client = LlmClient::new(t as Arc<dyn LlmTransportObj>, 5);
        let observer = RecordingObserver::new();
        let err = client.complete(&req(), &observer).await.unwrap_err();
        assert!(
            matches!(err, EngineError::StreamIdleTimeout(_)),
            "got {err}"
        );
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn skip_prefix_chunk_strips_matching_prefix() {
        let mut remaining = "Hello world".to_string();
        assert_eq!(skip_prefix_chunk(&mut remaining, "Hello "), "");
        assert_eq!(remaining, "world");
        assert_eq!(skip_prefix_chunk(&mut remaining, "world!"), "!");
        assert_eq!(remaining, "");

        let mut remaining = "Hello".to_string();
        assert_eq!(skip_prefix_chunk(&mut remaining, "Hi"), "Hi");
        assert_eq!(remaining, "");
    }
}
