//! The agent engine and turn loop.
//!
//! This is the mobile port of grok's agent turn loop
//! (`xai-grok-shell/src/session/.../turn.rs`), stripped of ACP/leader/stdio
//! transport and TUI concerns: sample from the LLM, dispatch tool calls,
//! append results, repeat until a final answer — with doom-loop detection,
//! cancellation, and history compaction.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::agent::doom_loop::{
    step_signature, stationarity_nudge_message, IdenticalToolCallRun,
};
use crate::config::{EngineConfig, LlmMode};
use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver, NullObserver, UsageSummary};
use crate::llm::client::LlmClient;
use crate::llm::host::{HostLlmHub, HostLlmNotify, HostLlmTransport};
use crate::llm::types::{
    drop_colliding_function_tools, ChatCompletionRequest, ChatMessage, HostedTool, Role, ToolCall,
    ToolDefinitionWire, Usage,
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
fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    let mut chars = 0usize;
    for m in messages {
        chars += m.content.as_deref().map(str::len).unwrap_or(0);
        chars += m.reasoning_content.as_deref().map(str::len).unwrap_or(0);
        for tc in &m.tool_calls {
            chars += tc.function.name.len() + tc.function.arguments.len();
        }
    }
    chars / 4
}

/// Threshold above which we compact history (tokens).
const COMPACT_THRESHOLD_TOKENS: usize = 24_000;

pub struct PhoneBuddyEngine {
    config: EngineConfig,
    /// Runtime identity + extra instructions, shared with subagents.
    prompt: Arc<Mutex<PromptRuntime>>,
    runtime: tokio::runtime::Runtime,
    tools: Arc<ToolRegistry>,
    sandbox: Arc<Sandbox>,
    client: Arc<LlmClient>,
    sessions: SessionStore,
    plan_state: Arc<PlanState>,
    task_manager: Arc<crate::agent::task_manager::TaskManager>,
    scheduler_manager: Arc<crate::agent::scheduler_manager::SchedulerManager>,
    cancels: Mutex<HashMap<String, tokio_util::sync::CancellationToken>>,
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

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("phone-buddy")
            .build()
            .map_err(|e| EngineError::Config(format!("failed to build runtime: {e}")))?;

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
            cancels: Mutex::new(HashMap::new()),
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
        messages: Vec<ChatMessage>,
        stream: bool,
    ) -> ChatCompletionRequest {
        let hosted = HostedTool::for_request(self.config.enable_web_search, self.config.api_backend);
        ChatCompletionRequest {
            model,
            messages,
            stream: Some(stream),
            tools: drop_colliding_function_tools(self.merged_tools_wire(), &hosted),
            tool_choice: Some(serde_json::json!("auto")),
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_output_tokens),
            search_parameters: None,
            hosted_tools: hosted,
        }
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
        if let Some(token) = self.cancels.lock().unwrap().get(session_id) {
            token.cancel();
        }
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
        let observer = observer.unwrap_or_else(|| Arc::new(NullObserver));
        let session_id = session_id.to_string();
        let user_input = user_input.to_string();
        let this = self.clone();
        // The internal runtime is only active outside of async context here.
        self.runtime.block_on(async move {
            this.chat_async(&session_id, &user_input, observer).await
        })
    }

    /// Async version of [`chat`] for Rust consumers already on a runtime.
    pub async fn chat_async(
        self: &Arc<Self>,
        session_id: &str,
        user_input: &str,
        observer: Arc<dyn AgentObserver>,
    ) -> EngineResult<ChatOutcome> {
        let token = tokio_util::sync::CancellationToken::new();
        self.cancels
            .lock()
            .unwrap()
            .insert(session_id.to_string(), token.clone());
        // Wire the plan observer so plan updates stream to the UI.
        *self.plan_state.observer.lock().unwrap() = Some(observer.clone());

        let result = tokio::select! {
            _ = token.cancelled() => {
                Err(EngineError::Cancelled)
            }
            res = self.run_turn(session_id, user_input, &observer, &token) => {
                res
            }
        };

        self.cancels.lock().unwrap().remove(session_id);

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
        user_input: &str,
        observer: &Arc<dyn AgentObserver>,
        token: &tokio_util::sync::CancellationToken,
    ) -> EngineResult<ChatOutcome> {
        // Load or create the session.
        let mut session = self
            .sessions
            .load(session_id)?
            .unwrap_or_else(|| StoredSession {
                id: session_id.to_string(),
                title: truncate_title(user_input),
                created_at: now_iso(),
                updated_at: now_iso(),
                messages: Vec::new(),
            });

        session.messages.push(ChatMessage::user(user_input));
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
            self.maybe_compact(&mut session.messages, &system_prompt);

            let messages = with_system(&session.messages, &system_prompt);
            let request = self.conversation_request(self.config.model.clone(), messages, true);

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

            // Record assistant message.
            let reasoning_opt = if turn.reasoning.is_empty() {
                None
            } else {
                Some(turn.reasoning.clone())
            };
            let reasoning_items = turn.reasoning_items.clone();
            let encrypted_reasoning = turn.encrypted_reasoning.clone();

            let origin = self.client.origin_fingerprint();
            let has_client_tools = turn.tool_calls.iter().any(|tc| tc.kind != "server");
            if !has_client_tools {
                // Server-side Responses tools (e.g. server-side web_search) ran inline in the
                // provider and already contributed to `turn.text`. Surface them to the observer
                // as completed and exit the turn without executing locally or calling the LLM again.
                for call in &turn.tool_calls {
                    observer.on_event(AgentEvent::ToolCallResult {
                        call_id: call.id.clone(),
                        name: call.function.name.clone(),
                        ok: true,
                        output: call.function.arguments.clone(),
                    });
                }

                let assistant_msg = if turn.tool_calls.is_empty() {
                    ChatMessage::assistant_with_reasoning(
                        turn.text.clone(),
                        reasoning_opt,
                        reasoning_items,
                        encrypted_reasoning,
                    )
                } else {
                    ChatMessage::assistant_tool_calls_with_reasoning(
                        turn.tool_calls.clone(),
                        if turn.text.is_empty() {
                            None
                        } else {
                            Some(turn.text.clone())
                        },
                        reasoning_opt,
                        reasoning_items,
                        encrypted_reasoning,
                    )
                };

                session
                    .messages
                    .push(assistant_msg.with_origin(origin));
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
            let sig = step_signature(&turn.tool_calls);
            let step_name = turn
                .tool_calls
                .first()
                .map(|c| c.function.name.as_str())
                .unwrap_or("");
            identical.observe(&sig, step_name);

            session.messages.push(
                ChatMessage::assistant_tool_calls_with_reasoning(
                    turn.tool_calls.clone(),
                    if turn.text.is_empty() {
                        None
                    } else {
                        Some(turn.text.clone())
                    },
                    reasoning_opt,
                    reasoning_items,
                    encrypted_reasoning,
                )
                .with_origin(origin),
            );

            // Execute each tool call sequentially (mobile: keep it simple and
            // predictable; parallel dispatch is a future optimization).
            for call in &turn.tool_calls {
                if token.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                let name = call.function.name.clone();
                // ToolCallStart is emitted from collect_stream when the
                // model first names the call. Re-emitting here doubled
                // every tool in the UI (and e2e) with the fully assembled
                // arguments, including any snapshot concatenation bug.

                // Server-side Responses tools (web_search_call, computer_call,
                // …) already ran in the provider. Surface them to the UI but
                // do not try to execute them locally or feed a fake result
                // back to the model.
                if call.kind == "server" {
                    observer.on_event(AgentEvent::ToolCallResult {
                        call_id: call.id.clone(),
                        name: name.clone(),
                        ok: true,
                        output: call.function.arguments.clone(),
                    });
                    continue;
                }

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
                    .messages
                    .push(ChatMessage::tool_result(call.id.clone(), text));
            }

            // Once-per-run nudge after results are committed (upstream latch).
            if identical.take_nudge() {
                let nudge =
                    stationarity_nudge_message(&identical.tool_name, identical.run_len);
                session.messages.push(ChatMessage::system(nudge));
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
    fn maybe_compact(&self, messages: &mut Vec<ChatMessage>, system_prompt: &str) {
        if estimate_tokens(messages) < COMPACT_THRESHOLD_TOKENS {
            return;
        }
        let keep = 12usize.min(messages.len());
        if messages.len() <= keep {
            return;
        }
        let mut cut_point = messages.len() - keep;

        // Walk backwards from cut_point to find a safe boundary (user or
        // assistant message). Tool messages must stay with their assistant
        // tool_calls message to preserve history validity.
        while cut_point > 0 && messages[cut_point].role == Role::Tool {
            cut_point -= 1;
        }

        // Now walk backwards to ensure we don't cut right after an assistant
        // message with tool_calls (which would orphan the tool results).
        if cut_point > 0 && cut_point < messages.len() {
            if let Some(prev) = messages.get(cut_point - 1) {
                if prev.role == Role::Assistant && !prev.tool_calls.is_empty() {
                    // Find the next user/assistant-without-calls boundary.
                    while cut_point < messages.len() {
                        if messages[cut_point].role == Role::User {
                            break;
                        }
                        if messages[cut_point].role == Role::Assistant && messages[cut_point].tool_calls.is_empty() {
                            break;
                        }
                        cut_point += 1;
                    }
                }
            }
        }

        if cut_point >= messages.len() {
            // Safety valve: if we can't find a safe cut point, don't compact.
            return;
        }

        let dropped = cut_point;
        let tail: Vec<ChatMessage> = messages.split_off(cut_point);
        *messages = tail;

        // Ensure we still start from a user message to satisfy providers.
        if messages.first().map(|m| m.role) != Some(Role::User) {
            let note = format!(
                "(Earlier conversation was compacted to save context. System capabilities unchanged. {} messages were summarized away.)",
                dropped
            );
            messages.insert(0, ChatMessage::user(note));
        }
        let _ = system_prompt;
    }
}

fn with_system(messages: &[ChatMessage], system_prompt: &str) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    out.push(ChatMessage::system(system_prompt));
    // Sanitize tool-call arguments on the outbound path so a single
    // malformed historical call cannot 400 every subsequent turn
    // (grok `sanitize_tool_arguments`).
    out.extend(messages.iter().map(ChatMessage::sanitized_for_request));
    out
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
