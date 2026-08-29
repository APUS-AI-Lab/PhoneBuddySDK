//! Long-lived process handle owning the LLM router.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::config::EngineConfig;
use crate::conversation::ConversationItem;
use crate::engine::PhoneBuddyEngine;
use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver};
use crate::llm::client::LlmClient;
use crate::llm::dumper::HttpDumpConfig;
use crate::llm::router::{synthesize_legacy_routing, LlmRouter, LlmRoutingConfig, Workload};
use crate::llm::types::{ConversationRequest, ReasoningEffort, ResponseFormat, Usage};

/// Tool-free one-shot generation. Pool id is caller-supplied (e.g. `session_title`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextRequest {
    pub pool_id: String,
    #[serde(default)]
    pub instructions: Option<String>,
    pub input: String,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Structured-output constraint. Backends that cannot express it reject
    /// the request instead of silently returning prose.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

/// One-shot result. Never includes API keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextResult {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub provider_id: String,
    pub model: String,
    /// Provider visits spent producing this result (failover count + 1), not
    /// in-provider HTTP retries.
    pub attempts: u32,
    pub operation_id: String,
    pub pool_id: String,
    /// Always `one_shot`; lets hosts join results with routing diagnostics.
    pub workload: String,
}

/// Forwards one-shot router diagnostics to `tracing`. A one-shot call has no
/// session and no host event stream, so `Retrying` / `ProviderSwitched` would
/// otherwise be invisible; content deltas are dropped.
struct OneShotDiagnostics;

impl AgentObserver for OneShotDiagnostics {
    fn on_event(&self, event: AgentEvent) {
        match event {
            AgentEvent::Retrying {
                provider_id,
                pool_id,
                workload,
                operation_id,
                failure_class,
                attempt,
                max_attempts,
                wait_ms,
                reason,
                ..
            } => {
                tracing::warn!(
                    provider_id = provider_id.as_deref().unwrap_or(""),
                    pool_id = pool_id.as_deref().unwrap_or(""),
                    workload = workload.as_deref().unwrap_or(""),
                    operation_id = operation_id.as_deref().unwrap_or(""),
                    failure_class = failure_class.as_deref().unwrap_or(""),
                    "one-shot retry {attempt}/{max_attempts} in {wait_ms}ms: {reason}"
                );
            }
            AgentEvent::ProviderSwitched {
                from_provider_id,
                to_provider_id,
                pool_id,
                workload,
                operation_id,
                failure_class,
                cooldown_ms,
                reason,
                ..
            } => {
                tracing::warn!(
                    from_provider_id = from_provider_id.as_deref().unwrap_or(""),
                    to_provider_id = to_provider_id.as_deref().unwrap_or(""),
                    pool_id = pool_id.as_deref().unwrap_or(""),
                    workload = workload.as_deref().unwrap_or(""),
                    operation_id = operation_id.as_deref().unwrap_or(""),
                    failure_class = failure_class.as_deref().unwrap_or(""),
                    "one-shot provider switch, cooldown {cooldown_ms}ms: {reason}"
                );
            }
            _ => {}
        }
    }
}

/// Process-scoped routing owner. Longer-lived than a single
/// [`PhoneBuddyEngine`]. Shared interior mutability is not exposed via
/// [`Clone`]; callers hold [`Arc`].
#[derive(Clone, PartialEq, Eq)]
struct OneShotHttpSettings {
    stream_idle_timeout_secs: u64,
    http_dump: HttpDumpConfig,
    enable_doom_loop_check: Option<bool>,
}

impl OneShotHttpSettings {
    fn from_engine(cfg: &EngineConfig) -> Self {
        Self {
            stream_idle_timeout_secs: cfg.stream_idle_timeout_secs,
            http_dump: cfg.http_dump.clone(),
            enable_doom_loop_check: cfg.enable_doom_loop_check,
        }
    }
}

pub struct PhoneBuddyRuntime {
    router: Arc<LlmRouter>,
    root_dir: PathBuf,
    operations: Mutex<HashMap<String, CancellationToken>>,
    http_settings: Mutex<OneShotHttpSettings>,
    one_shot_clients: Mutex<HashMap<String, Arc<LlmClient>>>,
}

impl PhoneBuddyRuntime {
    pub fn new(
        routing_config: LlmRoutingConfig,
        root_dir: impl Into<PathBuf>,
    ) -> EngineResult<Arc<Self>> {
        let root_dir = root_dir.into();
        let _ = rustls::crypto::ring::default_provider().install_default();
        let router = LlmRouter::persist(routing_config, root_dir.clone())?;
        Ok(Arc::new(Self {
            router,
            root_dir,
            operations: Mutex::new(HashMap::new()),
            http_settings: Mutex::new(OneShotHttpSettings::from_engine(&EngineConfig::default())),
            one_shot_clients: Mutex::new(HashMap::new()),
        }))
    }

    /// Compatibility constructor: synthesize `main` + `subagent` pools from
    /// the historic primary + `fallback_providers` chain.
    pub fn from_engine_config(config: &EngineConfig) -> EngineResult<Arc<Self>> {
        let routing =
            synthesize_legacy_routing(config).map_err(EngineError::InvalidRoutingConfig)?;
        let runtime = Self::new(routing, config.root_dir.clone())?;
        runtime.adopt_http_settings(config);
        Ok(runtime)
    }

    /// Replace routing. In-flight operations may finish on a previously
    /// captured visit plan. Health is reconciled by stable `provider_id`.
    pub fn update_routing(&self, new_config: LlmRoutingConfig) -> EngineResult<()> {
        self.router.update_config(new_config)
    }

    pub fn router(&self) -> Arc<LlmRouter> {
        self.router.clone()
    }

    pub fn root_dir(&self) -> &std::path::Path {
        &self.root_dir
    }

    pub fn create_engine(
        self: &Arc<Self>,
        agent_config: EngineConfig,
        main_pool_id: &str,
    ) -> EngineResult<Arc<PhoneBuddyEngine>> {
        self.adopt_http_settings(&agent_config);
        PhoneBuddyEngine::from_runtime(self.clone(), agent_config, main_pool_id)
    }

    fn adopt_http_settings(&self, cfg: &EngineConfig) {
        let next = OneShotHttpSettings::from_engine(cfg);
        let mut settings = self.http_settings.lock().unwrap();
        if *settings != next {
            *settings = next;
            drop(settings);
            self.one_shot_clients.lock().unwrap().clear();
        }
    }

    fn http_engine_config(&self) -> EngineConfig {
        let settings = self.http_settings.lock().unwrap();
        EngineConfig {
            root_dir: self.root_dir.clone(),
            http_dump: settings.http_dump.clone(),
            stream_idle_timeout_secs: settings.stream_idle_timeout_secs,
            enable_doom_loop_check: settings.enable_doom_loop_check,
            ..Default::default()
        }
    }

    fn client_for_pool(&self, pool_id: &str) -> EngineResult<Arc<LlmClient>> {
        if let Some(client) = self.one_shot_clients.lock().unwrap().get(pool_id) {
            return Ok(client.clone());
        }
        let client = Arc::new(
            LlmClient::from_router(self.router(), pool_id, &self.http_engine_config())?
                .with_workload(Workload::OneShot),
        );
        self.one_shot_clients
            .lock()
            .unwrap()
            .insert(pool_id.to_string(), client.clone());
        Ok(client)
    }

    /// One-shot text generation: router + adapters + retry, no session or tools.
    pub async fn generate_text(
        &self,
        request: GenerateTextRequest,
        cancellation: CancellationToken,
    ) -> EngineResult<GenerateTextResult> {
        if !self.router.has_pool(&request.pool_id) {
            self.one_shot_clients
                .lock()
                .unwrap()
                .remove(&request.pool_id);
            return Err(EngineError::RouteNotConfigured {
                pool_id: request.pool_id,
            });
        }
        let client = self.client_for_pool(&request.pool_id)?;
        self.generate_text_on(&client, request, cancellation).await
    }

    /// Blocking wrapper around [`Self::generate_text`] using the process-wide executor.
    pub fn generate_text_blocking(
        &self,
        request: GenerateTextRequest,
        cancellation: CancellationToken,
    ) -> EngineResult<GenerateTextResult> {
        crate::engine::shared_runtime()?.block_on(self.generate_text(request, cancellation))
    }

    /// Start one-shot generation on the process-wide executor. Returns the
    /// operation id immediately; `on_done` runs when the call finishes.
    pub fn generate_text_async(
        self: &Arc<Self>,
        request: GenerateTextRequest,
        on_done: impl FnOnce(String, EngineResult<GenerateTextResult>) + Send + 'static,
    ) -> EngineResult<String> {
        let operation_id = format!("op_{}", uuid::Uuid::new_v4().simple());
        let token = CancellationToken::new();
        self.operations
            .lock()
            .unwrap()
            .insert(operation_id.clone(), token.clone());
        let this = self.clone();
        let op = operation_id.clone();
        crate::engine::shared_runtime()?.spawn(async move {
            let result = AssertUnwindSafe(this.generate_text(request, token))
                .catch_unwind()
                .await;
            let mut result = match result {
                Ok(inner) => inner,
                Err(_) => Err(EngineError::Llm("one-shot worker panicked".into())),
            };
            if let Ok(ref mut ok) = result {
                ok.operation_id = op.clone();
            }
            if let Ok(mut ops) = this.operations.lock() {
                ops.remove(&op);
            }
            on_done(op, result);
        });
        Ok(operation_id)
    }

    /// Cancel a one-shot operation started by [`Self::generate_text_async`].
    pub fn cancel_operation(&self, operation_id: &str) {
        if let Some(token) = self.operations.lock().unwrap().get(operation_id) {
            token.cancel();
        }
    }

    /// Cancel every in-flight one-shot. Does not wait for last-Arc drop, so
    /// FFI `pb_runtime_free` can stop HTTP work while workers still hold `Arc`.
    pub fn cancel_all(&self) {
        if let Ok(ops) = self.operations.lock() {
            for token in ops.values() {
                token.cancel();
            }
        }
    }

    async fn generate_text_on(
        &self,
        client: &LlmClient,
        request: GenerateTextRequest,
        cancellation: CancellationToken,
    ) -> EngineResult<GenerateTextResult> {
        if cancellation.is_cancelled() {
            return Err(EngineError::OperationCancelled);
        }
        if !self.router.has_pool(&request.pool_id) {
            return Err(EngineError::RouteNotConfigured {
                pool_id: request.pool_id,
            });
        }

        let conv = one_shot_conversation_request(&request);

        let work = async {
            let session = client.begin_turn();
            session.complete(&conv, &OneShotDiagnostics).await
        };

        let turn = if let Some(ms) = request.timeout_ms {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    return Err(EngineError::OperationCancelled);
                }
                res = tokio::time::timeout(Duration::from_millis(ms), work) => {
                    match res {
                        Ok(inner) => inner?,
                        Err(_) => return Err(EngineError::OperationTimedOut),
                    }
                }
            }
        } else {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    return Err(EngineError::OperationCancelled);
                }
                res = work => res?,
            }
        };

        let operation_id = if turn.operation_id.is_empty() {
            format!("op_{}", uuid::Uuid::new_v4().simple())
        } else {
            turn.operation_id
        };
        Ok(GenerateTextResult {
            text: turn.text,
            usage: turn.usage,
            provider_id: turn.provider_id,
            model: turn.model,
            attempts: turn.attempts.max(1),
            operation_id,
            pool_id: request.pool_id,
            workload: Workload::OneShot.as_str().to_string(),
        })
    }
}

impl Drop for PhoneBuddyRuntime {
    fn drop(&mut self) {
        self.cancel_all();
    }
}

fn one_shot_conversation_request(request: &GenerateTextRequest) -> ConversationRequest {
    let mut items = Vec::new();
    if let Some(instructions) = request.instructions.as_ref() {
        if !instructions.is_empty() {
            items.push(ConversationItem::system(instructions.clone()));
        }
    }
    items.push(ConversationItem::user(request.input.clone()));
    ConversationRequest {
        model: String::new(),
        items,
        stream: Some(true),
        tools: None,
        tool_choice: Some(serde_json::json!("none")),
        temperature: request.temperature,
        max_tokens: request.max_output_tokens,
        reasoning_effort: request.reasoning_effort,
        response_format: request.response_format.clone(),
        search_parameters: None,
        hosted_tools: Vec::new(),
        previous_response_id: None,
        image_bytes: Default::default(),
        audio_bytes: Default::default(),
    }
}

impl PhoneBuddyRuntime {
    #[cfg(test)]
    async fn generate_text_with_transports(
        &self,
        request: GenerateTextRequest,
        transports: std::collections::HashMap<String, Arc<dyn crate::llm::client::LlmTransportObj>>,
        cancellation: CancellationToken,
    ) -> EngineResult<GenerateTextResult> {
        let client = LlmClient::from_router_with_transports(
            self.router(),
            request.pool_id.clone(),
            transports,
        )?
        .with_workload(Workload::OneShot);
        self.generate_text_on(&client, request, cancellation).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::router::{
        FailureClass, PoolMember, ProviderPool, ProviderTarget, MAIN_POOL_ID, SUBAGENT_POOL_ID,
    };
    use chrono::{TimeZone, Utc};

    fn target(id: &str) -> ProviderTarget {
        ProviderTarget {
            provider_id: id.into(),
            base_url: "https://api.example.com/v1".into(),
            api_key: "k".into(),
            model: "m".into(),
            api_backend: Default::default(),
            client_profile: Default::default(),
            client_version: None,
            client_session_id: None,
            reasoning_compatibility_key: None,
            capabilities: Default::default(),
            extra_headers: Default::default(),
            extra_body: Default::default(),
            enable_web_search: false,
            web_search_options: None,
            enable_x_search: false,
            x_search_options: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn engine_recreation_keeps_health() {
        let dir = tempfile::tempdir().unwrap();
        let mut pools = std::collections::BTreeMap::new();
        let member = PoolMember {
            provider_id: "p1".into(),
            routing_group: "g".into(),
            base_score: 10,
            order: 0,
            enabled: true,
        };
        pools.insert(
            MAIN_POOL_ID.into(),
            ProviderPool {
                members: vec![member.clone()],
                ..Default::default()
            },
        );
        pools.insert(
            SUBAGENT_POOL_ID.into(),
            ProviderPool {
                members: vec![member],
                ..Default::default()
            },
        );
        let routing = LlmRoutingConfig {
            providers: vec![target("p1")],
            pools,
            health: Default::default(),
        };
        let runtime = PhoneBuddyRuntime::new(routing, dir.path()).unwrap();
        let t = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        runtime
            .router()
            .record_trip_at("op1", "p1", FailureClass::RetryableHttp, None, t);

        let mut cfg = EngineConfig::default();
        cfg.api_key = "k".into();
        cfg.root_dir = dir.path().to_path_buf();
        let engine = runtime.create_engine(cfg.clone(), MAIN_POOL_ID).unwrap();
        drop(engine);
        let _engine2 = runtime.create_engine(cfg, MAIN_POOL_ID).unwrap();
        assert!(runtime.router().health_record("p1").unwrap().is_cooling(t));
    }

    #[test]
    fn create_engine_without_subagent_pool_is_not_configured() {
        let dir = tempfile::tempdir().unwrap();
        let mut pools = std::collections::BTreeMap::new();
        pools.insert(
            MAIN_POOL_ID.into(),
            ProviderPool {
                members: vec![PoolMember {
                    provider_id: "p1".into(),
                    routing_group: "g".into(),
                    base_score: 10,
                    order: 0,
                    enabled: true,
                }],
                ..Default::default()
            },
        );
        let routing = LlmRoutingConfig {
            providers: vec![target("p1")],
            pools,
            health: Default::default(),
        };
        let runtime = PhoneBuddyRuntime::new(routing, dir.path()).unwrap();
        let mut cfg = EngineConfig::default();
        cfg.api_key = "k".into();
        cfg.root_dir = dir.path().to_path_buf();
        match runtime.create_engine(cfg, MAIN_POOL_ID) {
            Err(EngineError::RouteNotConfigured { pool_id }) => {
                assert_eq!(pool_id, SUBAGENT_POOL_ID);
            }
            Err(other) => panic!("expected RouteNotConfigured, got {other}"),
            Ok(_) => panic!("expected RouteNotConfigured for missing subagent pool"),
        }
    }
}

#[cfg(test)]
mod generate_text_tests {
    use super::*;
    use crate::error::EngineError;
    use crate::llm::client::LlmTransportObj;
    use crate::llm::router::{
        ExhaustionPolicy, FailureClass, PoolMember, ProviderPool, ProviderTarget,
    };
    use crate::llm::transport::LlmTransport;
    use crate::llm::transport::LlmTurnContext;
    use crate::llm::types::{
        ApiBackend, ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ConversationRequest, Role,
    };
    use std::collections::{BTreeMap, HashMap};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    const TITLE_POOL: &str = "session_title";

    fn target_with(id: &str, backend: ApiBackend, web_search: bool) -> ProviderTarget {
        ProviderTarget {
            provider_id: id.into(),
            base_url: format!("https://{id}.example.com/v1"),
            api_key: "secret-key-must-not-leak".into(),
            model: format!("{id}-model"),
            api_backend: backend,
            client_profile: Default::default(),
            client_version: None,
            client_session_id: None,
            reasoning_compatibility_key: None,
            capabilities: Default::default(),
            extra_headers: Default::default(),
            extra_body: Default::default(),
            enable_web_search: web_search,
            web_search_options: None,
            enable_x_search: false,
            x_search_options: None,
            reasoning_effort: None,
        }
    }

    fn member(id: &str, order: u32) -> PoolMember {
        PoolMember {
            provider_id: id.into(),
            routing_group: "cheap".into(),
            base_score: 10,
            order,
            enabled: true,
        }
    }

    fn title_runtime(
        providers: Vec<ProviderTarget>,
        members: Vec<PoolMember>,
        exhausted: ExhaustionPolicy,
    ) -> (tempfile::TempDir, Arc<PhoneBuddyRuntime>) {
        let dir = tempfile::tempdir().unwrap();
        let mut pools = BTreeMap::new();
        pools.insert(
            TITLE_POOL.into(),
            ProviderPool {
                members,
                when_exhausted: exhausted,
                ..Default::default()
            },
        );
        let routing = LlmRoutingConfig {
            providers,
            pools,
            health: Default::default(),
        };
        let runtime = PhoneBuddyRuntime::new(routing, dir.path()).unwrap();
        (dir, runtime)
    }

    struct CapturingTransport {
        name: String,
        text: String,
        delay: Duration,
        captured: Mutex<Vec<ConversationRequest>>,
        isolated_starts: AtomicU32,
    }

    impl CapturingTransport {
        fn new(name: &str, text: &str) -> Arc<Self> {
            Arc::new(Self {
                name: name.into(),
                text: text.into(),
                delay: Duration::from_millis(0),
                captured: Mutex::new(Vec::new()),
                isolated_starts: AtomicU32::new(0),
            })
        }

        fn slow(name: &str, text: &str, delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                name: name.into(),
                text: text.into(),
                delay,
                captured: Mutex::new(Vec::new()),
                isolated_starts: AtomicU32::new(0),
            })
        }
    }

    impl LlmTransport for CapturingTransport {
        async fn request_stream(
            &self,
            req: &ConversationRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            self.captured.lock().unwrap().push(req.clone());
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            let text = self.text.clone();
            let stream = async_stream::stream! {
                let d = ChatChunkDelta {
                    role: Some(Role::Assistant),
                    content: Some(text),
                    ..Default::default()
                };
                yield Ok(ChatCompletionChunk {
                    model: "stream-model".into(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: d,
                        finish_reason: Some("stop".into()),
                    }],
                    usage: Some(Usage {
                        prompt_tokens: 3,
                        completion_tokens: 5,
                        total_tokens: 8,
                    }),
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
            assert!(
                context.turn_state(&self.name).is_none(),
                "one-shot must start with an isolated turn context"
            );
            self.isolated_starts.fetch_add(1, Ordering::SeqCst);
            context.set_turn_state(&self.name, "oneshot".into());
            Box::pin(self.request_stream(req))
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    struct HangTransport {
        started: tokio::sync::Notify,
        hits: AtomicU32,
    }

    impl HangTransport {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                started: tokio::sync::Notify::new(),
                hits: AtomicU32::new(0),
            })
        }
    }

    impl LlmTransport for HangTransport {
        async fn request_stream(
            &self,
            _req: &ConversationRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            self.started.notify_waiters();
            std::future::pending().await
        }

        fn name(&self) -> &str {
            "hang"
        }
    }

    fn req() -> GenerateTextRequest {
        GenerateTextRequest {
            pool_id: TITLE_POOL.into(),
            instructions: Some("Title the conversation.".into()),
            input: "User talked about shipping the SDK.".into(),
            max_output_tokens: Some(32),
            temperature: Some(0.0),
            reasoning_effort: None,
            response_format: None,
            timeout_ms: None,
        }
    }

    #[tokio::test]
    async fn one_shot_excludes_tools_sessions_and_previous_response_id() {
        let t = CapturingTransport::new("p-cc", "SDK shipping");
        let (dir, runtime) = title_runtime(
            vec![target_with("p-cc", ApiBackend::ChatCompletions, true)],
            vec![member("p-cc", 0)],
            ExhaustionPolicy::FailFast,
        );
        let mut transports: HashMap<String, Arc<dyn LlmTransportObj>> = HashMap::new();
        transports.insert("p-cc".into(), t.clone() as Arc<dyn LlmTransportObj>);

        let result = runtime
            .generate_text_with_transports(req(), transports, CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(result.text, "SDK shipping");
        assert_eq!(result.provider_id, "p-cc");
        assert_eq!(result.model, "stream-model");
        assert_eq!(result.pool_id, TITLE_POOL);
        assert_eq!(result.workload, "one_shot");
        assert!(result.operation_id.starts_with("op_"));
        assert!(result.attempts >= 1);
        let usage = result.usage.expect("usage");
        assert_eq!(usage.total_tokens, 8);

        let captured = t.captured.lock().unwrap();
        assert_eq!(captured.len(), 1);
        let conv = &captured[0];
        assert!(conv.tools.is_none(), "one-shot must not send tools");
        assert_eq!(
            conv.tool_choice.as_ref().and_then(|v| v.as_str()),
            Some("none")
        );
        assert!(conv.hosted_tools.is_empty());
        assert!(conv.previous_response_id.is_none());
        assert!(
            conv.items
                .iter()
                .any(|i| matches!(i, ConversationItem::System(_))),
            "instructions become a system item"
        );
        assert!(conv
            .items
            .iter()
            .any(|i| matches!(i, ConversationItem::User(_))));

        let sessions = dir.path().join(".phonebuddy").join("sessions");
        assert!(
            !sessions.exists() || std::fs::read_dir(&sessions).map(|d| d.count()).unwrap_or(0) == 0,
            "one-shot must not create session files"
        );
        let tasks = dir.path().join(".phonebuddy").join("tasks");
        assert!(!tasks.exists(), "one-shot must not create tasks");
    }

    #[tokio::test]
    async fn one_shot_uses_fresh_turn_context_per_call() {
        let t = CapturingTransport::new("p-cc", "ok");
        let (_dir, runtime) = title_runtime(
            vec![target_with("p-cc", ApiBackend::ChatCompletions, false)],
            vec![member("p-cc", 0)],
            ExhaustionPolicy::FailFast,
        );
        let mut transports: HashMap<String, Arc<dyn LlmTransportObj>> = HashMap::new();
        transports.insert("p-cc".into(), t.clone() as Arc<dyn LlmTransportObj>);

        runtime
            .generate_text_with_transports(req(), transports.clone(), CancellationToken::new())
            .await
            .unwrap();
        runtime
            .generate_text_with_transports(req(), transports, CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(
            t.isolated_starts.load(Ordering::SeqCst),
            2,
            "each one-shot call must start with a clean turn context"
        );
    }

    #[tokio::test]
    async fn one_shot_covers_responses_backend_without_hosted_tools() {
        let t = CapturingTransport::new("p-resp", "A title");
        let (_dir, runtime) = title_runtime(
            vec![target_with("p-resp", ApiBackend::Responses, true)],
            vec![member("p-resp", 0)],
            ExhaustionPolicy::FailFast,
        );
        let mut transports: HashMap<String, Arc<dyn LlmTransportObj>> = HashMap::new();
        transports.insert("p-resp".into(), t.clone() as Arc<dyn LlmTransportObj>);

        let result = runtime
            .generate_text_with_transports(req(), transports, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(result.provider_id, "p-resp");
        let conv = t.captured.lock().unwrap().pop().unwrap();
        assert!(conv.hosted_tools.is_empty());
        assert!(conv.tools.is_none());
        assert_eq!(
            conv.tool_choice.as_ref().and_then(|v| v.as_str()),
            Some("none")
        );
        assert!(conv.previous_response_id.is_none());
    }

    fn assert_payload_has_no_tools(payload: &serde_json::Value, backend: &str) {
        if let Some(tools) = payload.get("tools") {
            if let Some(arr) = tools.as_array() {
                assert!(
                    arr.is_empty(),
                    "{backend} one-shot payload must not list tools: {payload}"
                );
            } else {
                panic!("{backend} tools must be an array or absent: {payload}");
            }
        }
        let encoded = payload.to_string();
        assert!(
            !encoded.contains("web_search") && !encoded.contains("x_search"),
            "{backend} one-shot payload must not attach hosted search: {payload}"
        );
        assert!(
            !encoded.contains("functionDeclarations"),
            "{backend} one-shot payload must not declare functions: {payload}"
        );
    }

    #[test]
    fn one_shot_adapters_emit_no_tools_or_hosted_search() {
        let conv = one_shot_conversation_request(&req());
        let cc = crate::llm::wire::chat_completions::build_chat_completions_payload(&conv).unwrap();
        assert_payload_has_no_tools(&cc, "chat_completions");
        assert_eq!(cc.get("tool_choice").and_then(|v| v.as_str()), Some("none"));

        let responses = crate::llm::wire::responses::build_responses_payload(&conv).unwrap();
        assert_payload_has_no_tools(&responses, "responses");

        let messages = crate::llm::wire::messages::build_messages_payload(&conv).unwrap();
        assert_payload_has_no_tools(&messages, "messages");

        let gemini = crate::llm::wire::gemini::build_gemini_payload(&conv).unwrap();
        assert_payload_has_no_tools(&gemini, "gemini");
    }

    #[test]
    fn one_shot_response_format_reaches_every_supported_backend() {
        let mut request = req();
        request.response_format = Some(ResponseFormat::JsonSchema {
            name: "title".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": { "title": { "type": "string" } },
                "required": ["title"],
            }),
            strict: Some(true),
        });
        let conv = one_shot_conversation_request(&request);

        let cc = crate::llm::wire::chat_completions::build_chat_completions_payload(&conv).unwrap();
        assert_eq!(
            cc["response_format"]["type"].as_str(),
            Some("json_schema"),
            "{cc}"
        );
        assert_eq!(cc["response_format"]["json_schema"]["name"], "title");
        assert_eq!(cc["response_format"]["json_schema"]["strict"], true);

        let responses = crate::llm::wire::responses::build_responses_payload(&conv).unwrap();
        assert_eq!(responses["text"]["format"]["type"], "json_schema");
        assert_eq!(responses["text"]["format"]["name"], "title");
        assert!(
            responses["text"]["format"]["schema"].is_object(),
            "{responses}"
        );

        let gemini = crate::llm::wire::gemini::build_gemini_payload(&conv).unwrap();
        assert_eq!(
            gemini["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert!(gemini["generationConfig"]["responseSchema"].is_object());

        // Anthropic Messages cannot express this without a tool schema, and
        // one-shot generation forbids tools — fail loudly instead of
        // returning prose the caller would try to parse.
        let err = crate::llm::wire::messages::build_messages_payload(&conv).unwrap_err();
        assert_eq!(err.kind(), "ResponseFormatUnsupported");
        assert_eq!(err.envelope_fields()["api_backend"], "messages");
    }

    #[test]
    fn one_shot_json_object_and_text_formats_round_trip() {
        let mut request = req();
        request.response_format = Some(ResponseFormat::JsonObject);
        let conv = one_shot_conversation_request(&request);
        let cc = crate::llm::wire::chat_completions::build_chat_completions_payload(&conv).unwrap();
        assert_eq!(cc["response_format"]["type"], "json_object");
        assert_eq!(
            crate::llm::wire::gemini::build_gemini_payload(&conv).unwrap()["generationConfig"]
                ["responseMimeType"],
            "application/json"
        );

        // `Text` is the documented no-op and stays legal on every backend.
        request.response_format = Some(ResponseFormat::Text);
        let conv = one_shot_conversation_request(&request);
        assert!(crate::llm::wire::messages::build_messages_payload(&conv).is_ok());
        let gemini = crate::llm::wire::gemini::build_gemini_payload(&conv).unwrap();
        assert!(gemini["generationConfig"].get("responseMimeType").is_none());

        let parsed: ResponseFormat =
            serde_json::from_str(r#"{"type":"json_object"}"#).expect("wire form");
        assert_eq!(parsed, ResponseFormat::JsonObject);
    }

    #[tokio::test]
    async fn missing_pool_is_route_not_configured() {
        let dir = tempfile::tempdir().unwrap();
        let routing = LlmRoutingConfig {
            providers: vec![target_with("p-cc", ApiBackend::ChatCompletions, false)],
            pools: BTreeMap::new(),
            health: Default::default(),
        };
        let runtime = PhoneBuddyRuntime::new(routing, dir.path()).unwrap();
        let err = runtime
            .generate_text(req(), CancellationToken::new())
            .await
            .unwrap_err();
        match err {
            EngineError::RouteNotConfigured { ref pool_id } => assert_eq!(pool_id, TITLE_POOL),
            other => panic!("expected RouteNotConfigured, got {other}"),
        }
        assert_eq!(err.kind(), "RouteNotConfigured");
    }

    #[tokio::test]
    async fn fail_fast_pool_exhausted_returns_promptly() {
        let t = CapturingTransport::new("p-cc", "should-not-run");
        let (_dir, runtime) = title_runtime(
            vec![target_with("p-cc", ApiBackend::ChatCompletions, false)],
            vec![member("p-cc", 0)],
            ExhaustionPolicy::FailFast,
        );
        runtime
            .router()
            .record_trip("op-cool", "p-cc", FailureClass::RetryableHttp, None);

        let mut transports: HashMap<String, Arc<dyn LlmTransportObj>> = HashMap::new();
        transports.insert("p-cc".into(), t.clone() as Arc<dyn LlmTransportObj>);
        let started = std::time::Instant::now();
        let err = runtime
            .generate_text_with_transports(req(), transports, CancellationToken::new())
            .await
            .unwrap_err();
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "fail_fast must not wait on cooldown"
        );
        match err {
            EngineError::PoolExhausted { pool_id, .. } => assert_eq!(pool_id, TITLE_POOL),
            other => panic!("expected PoolExhausted, got {other}"),
        }
        assert!(t.captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn timeout_maps_to_operation_timed_out() {
        let t = CapturingTransport::slow("p-cc", "late", Duration::from_secs(5));
        let (_dir, runtime) = title_runtime(
            vec![target_with("p-cc", ApiBackend::ChatCompletions, false)],
            vec![member("p-cc", 0)],
            ExhaustionPolicy::FailFast,
        );
        let mut transports: HashMap<String, Arc<dyn LlmTransportObj>> = HashMap::new();
        transports.insert("p-cc".into(), t as Arc<dyn LlmTransportObj>);
        let mut request = req();
        request.timeout_ms = Some(20);
        let err = runtime
            .generate_text_with_transports(request, transports, CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::OperationTimedOut));
    }

    #[tokio::test]
    async fn cancel_maps_to_operation_cancelled() {
        let hang = HangTransport::new();
        let (_dir, runtime) = title_runtime(
            vec![target_with("p-cc", ApiBackend::ChatCompletions, false)],
            vec![member("p-cc", 0)],
            ExhaustionPolicy::FailFast,
        );
        let mut transports: HashMap<String, Arc<dyn LlmTransportObj>> = HashMap::new();
        transports.insert("p-cc".into(), hang.clone() as Arc<dyn LlmTransportObj>);
        let token = CancellationToken::new();
        let cancel = token.clone();
        let task = tokio::spawn(async move {
            runtime
                .generate_text_with_transports(req(), transports, token)
                .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel.cancel();
        let err = task.await.unwrap().unwrap_err();
        assert!(matches!(err, EngineError::OperationCancelled));
    }

    /// Always fails with a retryable status so the pool fails over.
    struct FailingTransport {
        name: String,
        hits: AtomicU32,
    }

    impl FailingTransport {
        fn new(name: &str) -> Arc<Self> {
            Arc::new(Self {
                name: name.into(),
                hits: AtomicU32::new(0),
            })
        }
    }

    impl LlmTransport for FailingTransport {
        async fn request_stream(
            &self,
            _req: &ConversationRequest,
        ) -> EngineResult<crate::llm::transport::ChunkStream> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            Err(EngineError::Llm("status=503 busy".into()))
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[tokio::test]
    async fn concurrent_main_subagent_and_one_shot_share_health_but_not_state() {
        use crate::events::RecordingObserver;
        use crate::llm::client::LlmClient;
        use crate::llm::router::{Workload, MAIN_POOL_ID, SUBAGENT_POOL_ID};

        let dir = tempfile::tempdir().unwrap();
        let agent_pool = ProviderPool {
            members: vec![member("p-shared-a", 0), member("p-shared-b", 1)],
            // Fail over on the first failure instead of burning the budget.
            retry: crate::llm::router::RetryPolicy {
                failover_max_attempts: 1,
                max_retries: 1,
            },
            when_exhausted: ExhaustionPolicy::ProbeEarliest,
        };
        let mut pools = BTreeMap::new();
        pools.insert(MAIN_POOL_ID.into(), agent_pool.clone());
        pools.insert(SUBAGENT_POOL_ID.into(), agent_pool);
        pools.insert(
            TITLE_POOL.into(),
            ProviderPool {
                members: vec![member("p-title", 0)],
                when_exhausted: ExhaustionPolicy::FailFast,
                ..Default::default()
            },
        );
        let routing = LlmRoutingConfig {
            providers: vec![
                target_with("p-shared-a", ApiBackend::ChatCompletions, false),
                target_with("p-shared-b", ApiBackend::ChatCompletions, false),
                target_with("p-title", ApiBackend::ChatCompletions, false),
            ],
            pools,
            health: Default::default(),
        };
        let runtime = PhoneBuddyRuntime::new(routing, dir.path()).unwrap();

        let dead = FailingTransport::new("p-shared-a");
        let healthy = CapturingTransport::new("p-shared-b", "agent answer");
        let title = CapturingTransport::new("p-title", "A Title");
        let agent_transports: HashMap<String, Arc<dyn LlmTransportObj>> = HashMap::from([
            (
                "p-shared-a".into(),
                dead.clone() as Arc<dyn LlmTransportObj>,
            ),
            (
                "p-shared-b".into(),
                healthy.clone() as Arc<dyn LlmTransportObj>,
            ),
        ]);
        let title_transports: HashMap<String, Arc<dyn LlmTransportObj>> =
            HashMap::from([("p-title".into(), title.clone() as Arc<dyn LlmTransportObj>)]);

        let main_client = LlmClient::from_router_with_transports(
            runtime.router(),
            MAIN_POOL_ID,
            agent_transports.clone(),
        )
        .unwrap()
        .with_workload(Workload::Main);
        let sub_client = LlmClient::from_router_with_transports(
            runtime.router(),
            SUBAGENT_POOL_ID,
            agent_transports,
        )
        .unwrap()
        .with_workload(Workload::Subagent);

        let conv = one_shot_conversation_request(&req());
        let observer = RecordingObserver::new();
        let main_turn = main_client.begin_turn();
        let sub_turn = sub_client.begin_turn();
        let (main_res, sub_res, one_shot_res) = tokio::join!(
            main_turn.complete(&conv, &observer),
            sub_turn.complete(&conv, &observer),
            runtime.generate_text_with_transports(
                req(),
                title_transports,
                CancellationToken::new()
            ),
        );

        assert_eq!(main_res.unwrap().provider_id, "p-shared-b");
        assert_eq!(sub_res.unwrap().provider_id, "p-shared-b");
        let one_shot = one_shot_res.unwrap();
        assert_eq!(one_shot.provider_id, "p-title");
        assert_eq!(one_shot.workload, "one_shot");

        // One trip per logical visit, never per low-level retry. Whether the
        // second agent workload reaches the dead provider depends on whether
        // it captured its plan before the first trip landed, so compare the
        // trip count against the visits that actually happened.
        let visits = dead.hits.load(Ordering::SeqCst);
        assert!(
            (1..=2).contains(&visits),
            "unexpected dead visits: {visits}"
        );
        let shared = runtime.router().health_record("p-shared-a").unwrap();
        assert_eq!(shared.consecutive_trips, visits);
        assert!(shared.is_cooling(chrono::Utc::now()));

        // Health is shared by provider id: the trip suppresses the dead
        // provider in every pool that lists it, not just the one that failed.
        for pool in [MAIN_POOL_ID, SUBAGENT_POOL_ID] {
            let plan = runtime.router().plan_visit(pool).unwrap();
            assert_eq!(plan.provider_ids[0], "p-shared-b", "pool {pool}");
        }
        // The title pool uses a different provider id, so it stays healthy.
        assert_eq!(
            runtime
                .router()
                .plan_visit(TITLE_POOL)
                .unwrap()
                .provider_ids,
            vec!["p-title"]
        );
        assert!(runtime
            .router()
            .health_record("p-title")
            .is_none_or(|h| !h.is_cooling(chrono::Utc::now())));

        // Isolated route-affine state: three concurrent operations each
        // started with a clean turn context on their own transport.
        assert_eq!(healthy.isolated_starts.load(Ordering::SeqCst), 2);
        assert_eq!(title.isolated_starts.load(Ordering::SeqCst), 1);

        // Diagnostics attribute the shared trip to the workload that caused it.
        let workloads: Vec<String> = observer
            .snapshot()
            .into_iter()
            .filter_map(|e| match e {
                crate::events::AgentEvent::ProviderSwitched { workload, .. } => workload,
                _ => None,
            })
            .collect();
        assert_eq!(workloads.len(), visits as usize, "{workloads:?}");
        assert!(
            workloads.iter().all(|w| w == "main" || w == "subagent"),
            "{workloads:?}"
        );
    }
}
