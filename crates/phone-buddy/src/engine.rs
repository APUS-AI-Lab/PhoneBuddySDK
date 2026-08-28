//! The agent engine and turn loop.
//!
//! This is the mobile port of grok's agent turn loop
//! (`xai-grok-shell/src/session/.../turn.rs`), stripped of ACP/leader/stdio
//! transport and TUI concerns: sample from the LLM, dispatch tool calls,
//! append results, repeat until a final answer — with doom-loop detection,
//! cancellation, and history compaction.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::agent::doom_loop::{
    step_signature, stationarity_nudge_message, IdenticalToolCallRun,
};
use crate::config::{EngineConfig, LlmMode};
use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver, NullObserver, UsageSummary};
use crate::llm::client::LlmClient;
use crate::llm::host::{HostLlmHub, HostLlmNotify, HostLlmTransport};
use crate::conversation::{user_assistant_count, ConversationItem};
use crate::llm::types::{
    drop_colliding_function_tools, ConversationRequest, HostedTool, ToolCall, ToolDefinitionWire,
    Usage,
};
use crate::prompt::{build_system_prompt, PromptRuntime};
use crate::session::{SessionStore, StoredSession};
use crate::tools::fs::Sandbox;
use crate::tools::host::{HostToolHub, HostToolNotify};
use crate::tools::plan::PlanState;
use crate::tools::webview::{WebViewFetchNotify, WebViewHost};
use crate::tools::{ToolCtx, ToolRegistry, ToolSpec};

/// Result of a completed turn.
#[derive(Debug, Clone)]
pub struct ChatOutcome {
    pub final_text: String,
    pub turns_used: u32,
    pub usage: Option<UsageSummary>,
    pub plan_items_json: String,
}

/// Approximate tokens = chars / 4 (same heuristic grok uses for quick
/// estimates; avoids pulling in a tokenizer).
fn estimate_image_chars(
    width: u32,
    height: u32,
    detail: Option<crate::conversation::ImageDetail>,
) -> usize {
    use crate::conversation::ImageDetail;
    let tiles = ((width.max(1) + 511) / 512) * ((height.max(1) + 511) / 512);
    let tokens = match detail.unwrap_or(ImageDetail::Auto) {
        ImageDetail::Low => 85,
        ImageDetail::High | ImageDetail::Original => 85 + tiles as usize * 170,
        ImageDetail::Auto => {
            if width <= 512 && height <= 512 {
                85
            } else {
                85 + tiles as usize * 170
            }
        }
    };
    tokens * 4
}

fn estimate_tokens(items: &[ConversationItem]) -> usize {
    let mut chars = 0usize;
    for item in items {
        match item {
            ConversationItem::System(s) => chars += s.content.len(),
            ConversationItem::User(u) => {
                chars += u.text_content().len();
                for p in &u.parts {
                    if let crate::conversation::UserContentPart::Image {
                        width,
                        height,
                        detail,
                        ..
                    } = p
                    {
                        chars += estimate_image_chars(*width, *height, *detail);
                    }
                }
            }
            ConversationItem::Assistant(a) => {
                chars += a.content.len();
                chars += a.reasoning_content.as_deref().map(str::len).unwrap_or(0);
                for tc in &a.tool_calls {
                    chars += tc.function.name.len() + tc.function.arguments.len();
                }
            }
            ConversationItem::ToolResult(t) => chars += t.content.len(),
            ConversationItem::Reasoning(r) => {
                chars += crate::llm::types::reasoning_item_text(r).len();
            }
            ConversationItem::BackendToolCall(b) => {
                chars += crate::conversation::backend_call_summary(b).len();
            }
        }
    }
    chars / 4
}

/// Threshold above which we compact history (tokens).
const COMPACT_THRESHOLD_TOKENS: usize = 24_000;

#[derive(Default)]
struct CancellationState {
    active: HashMap<String, tokio_util::sync::CancellationToken>,
    /// Cancellation requested before the first turn for a session is registered.
    pending: HashSet<String>,
    /// Prevents a late cancel after one turn from poisoning a future turn.
    started: HashSet<String>,
}

/// Process-wide async executor shared by isolated engine instances.
///
/// Mobile hosts create one engine per active run so mutable prompt, plan,
/// tool callback, and cancellation state cannot leak across conversations.
/// Sharing only the executor avoids creating a separate Tokio worker pool for
/// every simultaneous cloud chat while keeping all run state isolated.
fn shared_runtime() -> EngineResult<Arc<tokio::runtime::Runtime>> {
    static RUNTIME: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime.clone());
    }

    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .thread_name("phone-buddy")
            .build()
            .map_err(|e| EngineError::Config(format!("failed to build runtime: {e}")))?,
    );
    let _ = RUNTIME.set(runtime);
    Ok(RUNTIME.get().expect("runtime initialized").clone())
}

pub struct PhoneBuddyEngine {
    config: EngineConfig,
    /// Runtime identity + extra instructions, shared with subagents.
    prompt: Arc<Mutex<PromptRuntime>>,
    runtime: Arc<tokio::runtime::Runtime>,
    tools: Arc<ToolRegistry>,
    sandbox: Arc<Sandbox>,
    client: Arc<LlmClient>,
    sessions: SessionStore,
    plan_state: Arc<PlanState>,
    task_manager: Arc<crate::agent::task_manager::TaskManager>,
    scheduler_manager: Arc<crate::agent::scheduler_manager::SchedulerManager>,
    cancellation: Mutex<CancellationState>,
    /// Present when `llm_mode == Host` (and also allocated for with_transport).
    host_llm: Arc<HostLlmHub>,
    host_tools: Arc<HostToolHub>,
    webview: Arc<WebViewHost>,
}

impl PhoneBuddyEngine {
    /// Build an engine with HTTP or Host transport and the full toolset.
    pub fn new(config: EngineConfig) -> EngineResult<Arc<Self>> {
        config
            .validate()
            .map_err(|e| EngineError::Config(e))?;

        // Force the ring crypto provider (iOS/Android-friendly).
        let _ = rustls::crypto::ring::default_provider().install_default();

        let host_llm = HostLlmHub::new();
        let host_tools = HostToolHub::new();
        let webview = WebViewHost::new();
        let sandbox = Arc::new(Sandbox::new(&config.root_dir)?);
        let client = match config.llm_mode {
            LlmMode::Http => Arc::new(LlmClient::from_http(&config)?),
            LlmMode::Host => Arc::new(LlmClient::new(
                Arc::new(HostLlmTransport::new(host_llm.clone())),
                config.max_retries,
            )),
        };
        Self::assemble(config, client, host_llm, host_tools, webview, sandbox)
    }

    /// Build an engine around a custom transport (used for the mock/demo
    /// and tests).
    pub fn with_transport(
        config: EngineConfig,
        transport: Arc<dyn crate::llm::LlmTransportObj>,
    ) -> EngineResult<Arc<Self>> {
        let host_llm = HostLlmHub::new();
        let host_tools = HostToolHub::new();
        let webview = WebViewHost::new();
        let sandbox = Arc::new(Sandbox::new(&config.root_dir)?);
        let client = Arc::new(LlmClient::new(transport, config.max_retries));
        Self::assemble(config, client, host_llm, host_tools, webview, sandbox)
    }

    fn assemble(
        config: EngineConfig,
        client: Arc<LlmClient>,
        host_llm: Arc<HostLlmHub>,
        host_tools: Arc<HostToolHub>,
        webview: Arc<WebViewHost>,
        sandbox: Arc<Sandbox>,
    ) -> EngineResult<Arc<Self>> {
        let sessions = SessionStore::new(config.sessions_dir())?;
        let plan_state = PlanState::new();
        let prompt = Arc::new(Mutex::new(PromptRuntime::from_config(&config)));

        let scheduler_manager = Arc::new(crate::agent::scheduler_manager::SchedulerManager::new(
            sandbox.clone(),
        ));

        let mut subagent_registry = ToolRegistry::new();
        subagent_registry.register(crate::tools::read_file::arc());
        subagent_registry.register(crate::tools::write_file::arc());
        subagent_registry.register(crate::tools::edit_file::arc());
        subagent_registry.register(crate::tools::list_dir::arc());
        subagent_registry.register(crate::tools::grep::arc());
        subagent_registry.register(crate::tools::busybox::arc());
        subagent_registry.register(crate::tools::script::arc());
        subagent_registry.register(
            crate::tools::web_search::arc_from_engine_config_with_webview(&config, webview.clone()),
        );
        subagent_registry.register(crate::tools::web_fetch::arc_with_allow_local_and_webview(
            config.web_fetch_allow_local,
            webview.clone(),
        ));
        subagent_registry.register(crate::tools::plan::arc(plan_state.clone()));
        subagent_registry.register(crate::tools::scheduler::arc(
            scheduler_manager.clone(),
            host_tools.clone(),
        ));
        subagent_registry.register(crate::tools::notification::arc(host_tools.clone()));

        let task_manager = Arc::new(crate::agent::task_manager::TaskManager::with_prompt(
            config.clone(),
            client.clone(),
            sandbox.clone(),
            Arc::new(subagent_registry),
            prompt.clone(),
        ));

        let mut registry = ToolRegistry::new();
        registry.register(crate::tools::read_file::arc());
        registry.register(crate::tools::write_file::arc());
        registry.register(crate::tools::edit_file::arc());
        registry.register(crate::tools::list_dir::arc());
        registry.register(crate::tools::grep::arc());
        registry.register(crate::tools::busybox::arc());
        registry.register(crate::tools::script::arc());
        registry.register(crate::tools::web_search::arc_from_engine_config_with_webview(
            &config,
            webview.clone(),
        ));
        registry.register(crate::tools::web_fetch::arc_with_allow_local_and_webview(
            config.web_fetch_allow_local,
            webview.clone(),
        ));
        registry.register(crate::tools::plan::arc(plan_state.clone()));
        registry.register(crate::tools::scheduler::arc(
            scheduler_manager.clone(),
            host_tools.clone(),
        ));
        registry.register(crate::tools::notification::arc(host_tools.clone()));
        registry.register(crate::tools::ask_user_question::arc(host_tools.clone()));
        registry.register(crate::tools::task::task_arc(task_manager.clone()));
        registry.register(crate::tools::task::task_output_arc(task_manager.clone()));
        registry.register(crate::tools::task::get_task_output_arc(task_manager.clone()));
        registry.register(crate::tools::task::kill_task_arc(task_manager.clone()));
        registry.register(crate::tools::task::wait_tasks_arc(task_manager.clone()));
        registry.register(crate::tools::monitor::arc(
            task_manager.clone(),
            host_tools.clone(),
        ));
        let registry = Arc::new(registry);

        let runtime = shared_runtime()?;

        Ok(Arc::new(Self {
            config,
            prompt,
            runtime,
            tools: registry,
            sandbox,
            client,
            sessions,
            plan_state,
            task_manager,
            scheduler_manager,
            cancellation: Mutex::new(CancellationState::default()),
            host_llm,
            host_tools,
            webview,
        }))
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn sandbox(&self) -> &Arc<Sandbox> {
        &self.sandbox
    }

    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }

    pub fn host_llm(&self) -> &Arc<HostLlmHub> {
        &self.host_llm
    }

    pub fn host_tools(&self) -> &Arc<HostToolHub> {
        &self.host_tools
    }

    pub fn webview_host(&self) -> &Arc<WebViewHost> {
        &self.webview
    }

    /// Register or clear the host system WebView fetch callback.
    pub fn set_webview_callback(&self, cb: Option<WebViewFetchNotify>) {
        if let Some(cb) = cb {
            self.webview.set_notify(cb);
        } else {
            self.webview.clear_notify();
        }
    }

    pub fn task_manager(&self) -> &Arc<crate::agent::task_manager::TaskManager> {
        &self.task_manager
    }

    pub fn scheduler_manager(&self) -> &Arc<crate::agent::scheduler_manager::SchedulerManager> {
        &self.scheduler_manager
    }

    /// Register host callbacks for LLM streaming and host-tool execution.
    pub fn set_host_callbacks(
        &self,
        llm: Option<HostLlmNotify>,
        tool: Option<HostToolNotify>,
    ) {
        if let Some(cb) = llm {
            self.host_llm.set_notify(cb);
        } else {
            self.host_llm.clear_notify();
        }
        if let Some(cb) = tool {
            self.host_tools.set_notify(cb);
        } else {
            self.host_tools.clear_notify();
        }
    }

    /// Replace host tool schemas (OpenAI function definitions). Built-in
    /// names are skipped.
    pub fn set_host_tools(&self, tools: Vec<ToolSpec>) {
        let skip = self.tools.names();
        self.host_tools.set_tools(tools, &skip);
    }

    /// Update the extra system prompt used on subsequent turns.
    pub fn set_system_prompt_extra(&self, extra: Option<String>) {
        let extra = extra
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        self.prompt.lock().unwrap().extra = extra;
    }

    /// Set the identity used in the system prompt (`You are {name}…`).
    ///
    /// Empty or whitespace resets to [`crate::config::DEFAULT_AGENT_NAME`].
    pub fn set_agent_name(&self, name: Option<String>) {
        let resolved = crate::config::resolve_agent_name(name.as_deref().unwrap_or(""));
        self.prompt.lock().unwrap().agent_name = resolved;
    }

    /// Current system-prompt identity.
    pub fn agent_name(&self) -> String {
        self.prompt.lock().unwrap().agent_name.clone()
    }

    fn merged_tools_wire(&self) -> Vec<ToolDefinitionWire> {
        let mut defs = self.tools.wire_definitions();
        defs.extend(self.host_tools.wire_definitions());
        defs
    }

    fn conversation_request(
        &self,
        model: String,
        items: Vec<ConversationItem>,
        stream: bool,
    ) -> EngineResult<ConversationRequest> {
        let hosted = HostedTool::for_request(self.config.enable_web_search, self.config.api_backend);
        let root = self.config.resolved_attachment_root();
        let image_bytes = if items.iter().any(|i| {
            matches!(i, ConversationItem::User(u) if u.has_images())
        }) {
            crate::llm::image::materialize_items(&items, &root)?
        } else {
            crate::llm::image::ImageBytesStore::default()
        };
        Ok(ConversationRequest {
            model,
            items,
            stream: Some(stream),
            tools: drop_colliding_function_tools(self.merged_tools_wire(), &hosted),
            tool_choice: Some(serde_json::json!("auto")),
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_output_tokens),
            reasoning_effort: self.config.reasoning_effort,
            search_parameters: None,
            hosted_tools: hosted,
            previous_response_id: None,
            image_bytes,
        })
    }

    fn build_prompt(&self) -> String {
        let mut cfg = self.config.clone();
        self.prompt.lock().unwrap().apply_to(&mut cfg);
        build_system_prompt(&cfg)
    }

    // ── Sessions ─────────────────────────────────────────────────────────

    pub fn list_sessions(&self) -> EngineResult<Vec<crate::session::SessionMeta>> {
        self.sessions.list()
    }

    /// Load and return the full session (metadata + message history).
    pub fn get_session(&self, id: &str) -> EngineResult<Option<crate::session::StoredSession>> {
        self.sessions.load(id)
    }

    pub fn delete_session(&self, id: &str) -> EngineResult<()> {
        self.sessions.delete(id)
    }

    /// Cancel an in-flight turn for `session_id`.
    pub fn cancel(&self, session_id: &str) {
        let mut cancellation = self.cancellation.lock().unwrap();
        if let Some(token) = cancellation.active.get(session_id) {
            token.cancel();
        } else if !cancellation.started.contains(session_id) {
            cancellation.pending.insert(session_id.to_string());
        }
        drop(cancellation);
        // Unblock any host LLM stream waiting on the host.
        self.host_llm
            .abort_all(&format!("cancelled session {session_id}"));
    }

    // ── Chat (sync facade over the internal runtime) ─────────────────────

    /// Run one user turn to completion. Blocking; call from a background
    /// thread. `observer` receives streaming events.
    pub fn chat(
        self: &Arc<Self>,
        session_id: &str,
        user_input: &str,
        observer: Option<Arc<dyn AgentObserver>>,
    ) -> EngineResult<ChatOutcome> {
        self.chat_user(
            session_id,
            crate::conversation::UserItem::text(user_input),
            observer,
        )
    }

    /// Versioned structured user turn (`pb_engine_chat_v2`).
    pub fn chat_v2(
        self: &Arc<Self>,
        session_id: &str,
        turn_json: &str,
        observer: Option<Arc<dyn AgentObserver>>,
    ) -> EngineResult<ChatOutcome> {
        let user = crate::conversation::parse_user_turn_v2(turn_json)?;
        self.chat_user(session_id, user, observer)
    }

    fn chat_user(
        self: &Arc<Self>,
        session_id: &str,
        user: crate::conversation::UserItem,
        observer: Option<Arc<dyn AgentObserver>>,
    ) -> EngineResult<ChatOutcome> {
        let observer = observer.unwrap_or_else(|| Arc::new(NullObserver));
        let session_id = session_id.to_string();
        let this = self.clone();
        self.runtime
            .block_on(async move { this.chat_async(&session_id, user, observer).await })
    }

    /// Async version of [`chat`] for Rust consumers already on a runtime.
    pub async fn chat_async(
        self: &Arc<Self>,
        session_id: &str,
        user: crate::conversation::UserItem,
        observer: Arc<dyn AgentObserver>,
    ) -> EngineResult<ChatOutcome> {
        let token = tokio_util::sync::CancellationToken::new();
        let mut cancellation = self.cancellation.lock().unwrap();
        if cancellation.pending.remove(session_id) {
            token.cancel();
        }
        cancellation.started.insert(session_id.to_string());
        cancellation
            .active
            .insert(session_id.to_string(), token.clone());
        drop(cancellation);
        // Wire the plan observer so plan updates stream to the UI.
        *self.plan_state.observer.lock().unwrap() = Some(observer.clone());

        let result = tokio::select! {
            _ = token.cancelled() => {
                Err(EngineError::Cancelled)
            }
            res = self.run_turn(session_id, user, &observer, &token) => {
                res
            }
        };

        self.cancellation
            .lock()
            .unwrap()
            .active
            .remove(session_id);

        match &result {
            Ok(outcome) => {
                observer.on_event(AgentEvent::Completed {
                    final_text: outcome.final_text.clone(),
                    usage: outcome.usage,
                });
            }
            Err(e) => {
                observer.on_event(AgentEvent::Failed {
                    message: e.to_string(),
                });
            }
        }
        result
    }

    async fn run_turn(
        &self,
        session_id: &str,
        user: crate::conversation::UserItem,
        observer: &Arc<dyn AgentObserver>,
        token: &tokio_util::sync::CancellationToken,
    ) -> EngineResult<ChatOutcome> {
        user.validate_shape()?;
        if user.has_images() && !self.config.supports_image_input {
            return Err(EngineError::VisionUnsupported);
        }
        if user.has_images() {
            let root = self.config.resolved_attachment_root();
            crate::llm::image::materialize_user_item(
                &user,
                &root,
                &crate::llm::image::ImageBytesStore::default(),
            )?;
        }

        let title_src = user.text_content();
        let title = if title_src.trim().is_empty() {
            "Image".to_string()
        } else {
            truncate_title(&title_src)
        };

        // Load or create the session.
        let mut session = self
            .sessions
            .load(session_id)?
            .unwrap_or_else(|| StoredSession {
                id: session_id.to_string(),
                title,
                created_at: now_iso(),
                updated_at: now_iso(),
                format_version: 2,
                items: Vec::new(),
            });

        session.items.push(ConversationItem::User(user));
        session.updated_at = now_iso();

        let system_prompt = self.build_prompt();
        let mut total_usage = Usage::default();
        let mut turns_used = 0u32;

        // Shell action-stationarity: identical tool-call step runs.
        // Ported from grok-build `IdenticalToolCallRun` (turn.rs).
        let mut identical = IdenticalToolCallRun::default();

        loop {
            turns_used += 1;
            if token.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            if turns_used > self.config.max_turns {
                return Err(EngineError::MaxTurnsReached(self.config.max_turns));
            }

            // Hard-stop check at loop top (after prior step results committed).
            if identical.run_len >= identical.hard_stop_threshold() {
                return Err(EngineError::DoomLoop(identical.run_len));
            }

            // Compaction keeps long sessions inside a token budget.
            self.maybe_compact(&mut session.items, &system_prompt);

            let items = with_system(&session.items, &system_prompt);
            let request = self.conversation_request(self.config.model.clone(), items, true)?;

            let turn = tokio::select! {
                _ = token.cancelled() => {
                    return Err(EngineError::Cancelled);
                }
                res = self.client.complete(&request, observer.as_ref()) => {
                    res?
                }
            };
            if let Some(u) = &turn.usage {
                total_usage.prompt_tokens += u.prompt_tokens;
                total_usage.completion_tokens += u.completion_tokens;
                total_usage.total_tokens += u.total_tokens;
            }

            let origin = self.client.origin_fingerprint();
            let mut turn_items = if turn.items.is_empty() {
                synthesize_turn_items(&turn)
            } else {
                turn.items.clone()
            };
            stamp_origin(&mut turn_items, &origin);

            let client_calls: Vec<ToolCall> = turn_items
                .iter()
                .rev()
                .find_map(|i| i.as_assistant().map(|a| a.tool_calls.clone()))
                .unwrap_or_default();

            // Backend-only (or empty-call) turns: surface hosted calls and stop.
            if client_calls.is_empty() {
                for item in &turn_items {
                    if let ConversationItem::BackendToolCall(b) = item {
                        observer.on_event(AgentEvent::ToolCallResult {
                            call_id: b.id.clone(),
                            name: crate::conversation::server_tool_function_name(&b.item_type),
                            ok: true,
                            output: b.payload.to_string(),
                        });
                    }
                }
                session.items.extend(turn_items);
                let final_text = turn.text.clone();
                self.sessions.save(&session)?;
                return Ok(ChatOutcome {
                    final_text,
                    turns_used,
                    usage: Some(total_usage.into()),
                    plan_items_json: serde_json::to_string(&self.plan_state.snapshot())?,
                });
            }

            // Observe step signature once per tool batch (upstream shell).
            let sig = step_signature(&client_calls);
            let step_name = client_calls
                .first()
                .map(|c| c.function.name.as_str())
                .unwrap_or("");
            identical.observe(&sig, step_name);

            session.items.extend(turn_items);

            for call in &client_calls {
                if token.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                let name = call.function.name.clone();

                let output = tokio::select! {
                    _ = token.cancelled() => {
                        return Err(EngineError::Cancelled);
                    }
                    res = self.execute_tool(call, token) => {
                        res
                    }
                };
                let (ok, text) = match output {
                    Ok(o) => (true, o.text),
                    Err(e) => (false, format!("Error: {e}")),
                };
                observer.on_event(AgentEvent::ToolCallResult {
                    call_id: call.id.clone(),
                    name: name.clone(),
                    ok,
                    output: truncate_tool_event(&text),
                });
                session
                    .items
                    .push(ConversationItem::tool_result(call.id.clone(), text));
            }

            // Once-per-run nudge after results are committed (upstream latch).
            if identical.take_nudge() {
                let nudge =
                    stationarity_nudge_message(&identical.tool_name, identical.run_len);
                session.items.push(ConversationItem::system(nudge));
            }

            session.updated_at = now_iso();
            self.sessions.save(&session)?;
        }
    }

    async fn execute_tool(
        &self,
        call: &ToolCall,
        token: &tokio_util::sync::CancellationToken,
    ) -> EngineResult<crate::tools::ToolOutput> {
        let name = call.function.name.clone();

        let args: serde_json::Value = if call.function.arguments.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&call.function.arguments).map_err(|e| {
                EngineError::ToolArgs {
                    name: name.clone(),
                    message: format!("invalid JSON arguments: {e}"),
                }
            })?
        };

        // Built-in tools take priority over host tools on name clash.
        if let Some(tool) = self.tools.get(&name) {
            let ctx = ToolCtx {
                sandbox: self.sandbox.clone(),
                cancel: token.clone(),
            };
            let fut = tool.execute(args, &ctx);
            return tokio::select! {
                _ = token.cancelled() => {
                    Err(EngineError::Cancelled)
                }
                res = tokio::time::timeout(std::time::Duration::from_secs(120), fut) => {
                    match res {
                        Ok(res) => res,
                        Err(_) => Err(EngineError::Tool {
                            name,
                            message: "tool timed out after 120s".into(),
                        }),
                    }
                }
            };
        }

        if self.host_tools.has(&name) {
            let fut = self.host_tools.execute(&name, args, token);
            return tokio::select! {
                _ = token.cancelled() => {
                    Err(EngineError::Cancelled)
                }
                res = tokio::time::timeout(std::time::Duration::from_secs(120), fut) => {
                    match res {
                        Ok(res) => res,
                        Err(_) => Err(EngineError::Tool {
                            name,
                            message: "host tool timed out after 120s".into(),
                        }),
                    }
                }
            };
        }

        Err(EngineError::ToolNotFound(name))
    }

    /// Simple sliding-window compaction: when the estimated token count
    /// exceeds the threshold, drop the oldest non-system messages beyond a
    /// recent window and leave a marker. Keeps mobile memory/context small.
    /// Ensures the cut point never leaves orphan tool messages without their
    /// corresponding assistant tool_calls.
    fn maybe_compact(&self, items: &mut Vec<ConversationItem>, system_prompt: &str) {
        if estimate_tokens(items) < COMPACT_THRESHOLD_TOKENS {
            return;
        }
        let keep = 12usize.min(user_assistant_count(items).max(1));
        if items.len() <= keep {
            return;
        }
        let mut cut_point = items.len().saturating_sub(keep);
        while cut_point > 0 {
            match &items[cut_point] {
                ConversationItem::User(_) => break,
                ConversationItem::ToolResult(_)
                | ConversationItem::Reasoning(_)
                | ConversationItem::BackendToolCall(_) => {
                    cut_point -= 1;
                }
                ConversationItem::Assistant(a) if !a.tool_calls.is_empty() => {
                    cut_point += 1;
                    while cut_point < items.len() {
                        if matches!(items[cut_point], ConversationItem::User(_)) {
                            break;
                        }
                        if matches!(&items[cut_point], ConversationItem::Assistant(a) if a.tool_calls.is_empty())
                        {
                            break;
                        }
                        cut_point += 1;
                    }
                    break;
                }
                _ => break,
            }
        }

        if cut_point >= items.len() {
            return;
        }

        let dropped = cut_point;
        let tail: Vec<ConversationItem> = items.split_off(cut_point);
        *items = tail;

        if !matches!(items.first(), Some(ConversationItem::User(_))) {
            let note = format!(
                "(Earlier conversation was compacted to save context. System capabilities unchanged. {} messages were summarized away.)",
                dropped
            );
            items.insert(0, ConversationItem::user(note));
        }
        let _ = system_prompt;
    }
}

fn with_system(items: &[ConversationItem], system_prompt: &str) -> Vec<ConversationItem> {
    let mut out = Vec::with_capacity(items.len() + 1);
    out.push(ConversationItem::system(system_prompt));
    out.extend(items.iter().cloned());
    out
}

fn stamp_origin(items: &mut [ConversationItem], origin: &str) {
    for item in items.iter_mut().rev() {
        if let Some(a) = item.as_assistant_mut() {
            a.origin = Some(origin.to_string());
            break;
        }
    }
}

fn synthesize_turn_items(turn: &crate::llm::types::CollectedTurn) -> Vec<ConversationItem> {
    use crate::conversation::{AssistantItem, BackendToolCallItem};
    let mut items = Vec::new();
    for r in &turn.reasoning_items {
        items.push(ConversationItem::Reasoning(r.clone()));
    }
    let mut client_calls = Vec::new();
    for tc in &turn.tool_calls {
        if tc.kind == "server" {
            let item_type = crate::conversation::server_tool_item_type(&tc.function.name);
            items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                item_type: item_type.clone(),
                id: tc.id.clone(),
                payload: crate::conversation::reconstruct_backend_payload(
                    &item_type,
                    &tc.id,
                    &tc.function.arguments,
                ),
            }));
        } else {
            client_calls.push(tc.clone());
        }
    }
    items.push(ConversationItem::Assistant(AssistantItem {
        content: turn.text.clone(),
        tool_calls: client_calls,
        reasoning_content: if turn.reasoning.is_empty() {
            None
        } else {
            Some(turn.reasoning.clone())
        },
        encrypted_reasoning: turn.encrypted_reasoning.clone(),
        origin: None,
    }));
    items
}

fn truncate_title(input: &str) -> String {
    let t = input.trim();
    let mut s: String = t.chars().take(40).collect();
    if t.chars().count() > 40 {
        s.push('…');
    }
    s
}

fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn truncate_tool_event(s: &str) -> String {
    crate::tools::truncate_chars(s, 2_000)
}

#[cfg(test)]
mod tests {
    use super::shared_runtime;
    use std::sync::Arc;

    #[test]
    fn isolated_engines_share_the_process_runtime() {
        let first = shared_runtime().expect("first runtime");
        let second = shared_runtime().expect("second runtime");
        assert!(Arc::ptr_eq(&first, &second));
    }
}
