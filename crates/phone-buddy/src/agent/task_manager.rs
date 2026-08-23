//! In-memory Subagent Task Manager and Coordinator.
//!
//! Replaces external process forks with in-memory Tokio async tasks.
//! Each subagent runs its own turn loop against the LLM client, with
//! an isolated or inherited message history, cancellation tokens,
//! duration tracking, and resume capabilities.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use regex::RegexBuilder;
use tokio_util::sync::CancellationToken;

use crate::config::EngineConfig;
use crate::error::{EngineError, EngineResult};
use crate::events::NullObserver;
use crate::llm::client::LlmClient;
use crate::llm::types::{
    drop_colliding_function_tools, ChatCompletionRequest, ChatMessage, HostedTool,
};
use crate::prompt::{build_subagent_prompt, PromptRuntime};
use crate::tools::fs::Sandbox;
use crate::tools::{ToolCtx, ToolRegistry};

/// Status of a subagent task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// In-memory record for a spawned subagent task.
#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub task_id: String,
    pub description: String,
    pub subagent_type: String,
    pub status: TaskStatus,
    pub started: String,
    pub ended: Option<String>,
    pub duration_secs: f64,
    pub output: String,
    pub tool_calls: u32,
    pub turns: u32,
    pub messages: Vec<ChatMessage>,
    pub logs: Vec<String>,
    pub cancel_token: CancellationToken,
}

/// Task tool input payload.
#[derive(Debug, Clone, Deserialize)]
pub struct TaskInput {
    pub prompt: String,
    pub description: String,
    #[serde(default = "default_subagent_type")]
    pub subagent_type: String,
    #[serde(default = "default_true")]
    pub run_in_background: bool,
    pub resume_from: Option<String>,
    pub model: Option<String>,
}

fn default_subagent_type() -> String {
    "general-purpose".to_string()
}
fn default_true() -> bool {
    true
}

/// Maximum number of task IDs accepted by a single multi-id `get_task_output`
/// call. Ported from grok `xai-tool-types::MAX_MULTI_WAIT_IDS`.
pub const MAX_MULTI_WAIT_IDS: usize = 20;

/// Ported from grok `BACKGROUND_SUBAGENT_CONTINUE_PARENT_WORK`.
pub const BACKGROUND_SUBAGENT_CONTINUE_PARENT_WORK: &str =
    "Do not only poll the child. Continue unfinished parent work now.";

/// Render the model-facing notice for a subagent that was spawned in the
/// background. Ported from grok `xai-tool-types::format_subagent_started_background`.
pub fn format_subagent_started_background(
    subagent_id: &str,
    subagent_type: &str,
    description: &str,
) -> String {
    format_subagent_started_background_full(
        subagent_id,
        subagent_type,
        description,
        "task_output",
        false,
    )
}

/// Full form with tool name + optional continue-parent CTA.
pub fn format_subagent_started_background_full(
    subagent_id: &str,
    subagent_type: &str,
    description: &str,
    task_output_tool_name: &str,
    continue_parent_work: bool,
) -> String {
    let mut text = format!(
        "Subagent started in background.\n\
         subagent_id: {subagent_id}\n\
         type: {subagent_type}\n\
         description: {description}\n\n\
         Use {task_output_tool_name} with task_ids=[\"{subagent_id}\"] and timeout_ms to wait for results."
    );
    if continue_parent_work {
        text.push_str("\n\n");
        text.push_str(BACKGROUND_SUBAGENT_CONTINUE_PARENT_WORK);
    }
    text
}

/// Render the full model-facing completion block.
/// Ported from grok `xai-tool-types::format_subagent_completed`.
pub fn format_subagent_completed(
    output: &str,
    subagent_id: &str,
    subagent_type: &str,
    tool_calls: u32,
    turns: u32,
    duration_ms: u64,
) -> String {
    format_subagent_completed_full(
        output,
        subagent_id,
        subagent_type,
        tool_calls,
        turns,
        duration_ms,
        None,
    )
}

pub fn format_subagent_completed_full(
    output: &str,
    subagent_id: &str,
    subagent_type: &str,
    tool_calls: u32,
    turns: u32,
    duration_ms: u64,
    persona: Option<&str>,
) -> String {
    let footer = format_resume_footer(subagent_id, subagent_type, persona);
    format!(
        "{output}\n\n<subagent_meta>id={subagent_id}, type={subagent_type}, \
         tool_calls={tool_calls}, turns={turns}, duration_ms={duration_ms}</subagent_meta>\n\n\
         {footer}"
    )
}

/// Ported from grok `xai-tool-types::format_resume_footer`.
pub fn format_resume_footer(
    subagent_id: &str,
    subagent_type: &str,
    persona: Option<&str>,
) -> String {
    let mut footer = format!(
        "<subagent_result>\n\
         subagent_id: {subagent_id}\n\
         subagent_type: {subagent_type}\n\
         To continue this subagent's conversation, use resume_from=\"{subagent_id}\"."
    );
    if let Some(persona) = persona {
        footer.push_str(&format!(
            "\nThe subagent used persona=\"{persona}\". Pass the same persona when resuming."
        ));
    }
    footer.push_str("\n</subagent_result>");
    footer
}

/// Task Manager coordinating background subagent tasks.
pub struct TaskManager {
    config: EngineConfig,
    client: Arc<LlmClient>,
    sandbox: Arc<Sandbox>,
    subagent_tools: Arc<ToolRegistry>,
    prompt: Arc<Mutex<PromptRuntime>>,
    tasks: Arc<Mutex<HashMap<String, TaskRecord>>>,
    counter: Arc<Mutex<u64>>,
}

impl TaskManager {
    pub fn new(
        config: EngineConfig,
        client: Arc<LlmClient>,
        sandbox: Arc<Sandbox>,
        subagent_tools: Arc<ToolRegistry>,
    ) -> Self {
        let prompt = Arc::new(Mutex::new(PromptRuntime::from_config(&config)));
        Self::with_prompt(config, client, sandbox, subagent_tools, prompt)
    }

    pub fn with_prompt(
        config: EngineConfig,
        client: Arc<LlmClient>,
        sandbox: Arc<Sandbox>,
        subagent_tools: Arc<ToolRegistry>,
        prompt: Arc<Mutex<PromptRuntime>>,
    ) -> Self {
        Self {
            config,
            client,
            sandbox,
            subagent_tools,
            prompt,
            tasks: Arc::new(Mutex::new(HashMap::new())),
            counter: Arc::new(Mutex::new(1)),
        }
    }

    fn system_prompt(&self) -> String {
        let mut cfg = self.config.clone();
        self.prompt.lock().unwrap().apply_to(&mut cfg);
        build_subagent_prompt(&cfg)
    }

    fn generate_task_id(&self) -> String {
        let mut c = self.counter.lock().unwrap();
        let id = format!("task-{}", *c);
        *c += 1;
        id
    }

    /// Spawn a subagent task either synchronously or in background.
    pub async fn spawn_task(self: &Arc<Self>, input: TaskInput) -> EngineResult<String> {
        let task_id = self.generate_task_id();
        let cancel_token = CancellationToken::new();

        let record = TaskRecord {
            task_id: task_id.clone(),
            description: input.description.clone(),
            subagent_type: input.subagent_type.clone(),
            status: TaskStatus::Running,
            started: Utc::now().to_rfc3339(),
            ended: None,
            duration_secs: 0.0,
            output: String::new(),
            tool_calls: 0,
            turns: 0,
            messages: Vec::new(),
            logs: vec![format!(
                "[{}] Subagent task '{}' spawned (type: {})",
                Utc::now().to_rfc3339(),
                input.description,
                input.subagent_type
            )],
            cancel_token: cancel_token.clone(),
        };

        {
            let mut guard = self.tasks.lock().unwrap();
            guard.insert(task_id.clone(), record);
        }

        if input.run_in_background {
            let this = self.clone();
            let task_id_clone = task_id.clone();
            let prompt = input.prompt.clone();
            let resume_from = input.resume_from.clone();
            let model = input.model.clone();

            tokio::spawn(async move {
                this.run_subagent(&task_id_clone, &prompt, resume_from.as_deref(), model.as_deref()).await;
            });

            Ok(format_subagent_started_background(
                &task_id,
                &input.subagent_type,
                &input.description,
            ))
        } else {
            let prompt = input.prompt.clone();
            let resume_from = input.resume_from.clone();
            let model = input.model.clone();

            self.run_subagent(&task_id, &prompt, resume_from.as_deref(), model.as_deref()).await;

            let guard = self.tasks.lock().unwrap();
            if let Some(res) = guard.get(&task_id) {
                let duration_ms = (res.duration_secs * 1000.0) as u64;
                Ok(format_subagent_completed(
                    &res.output,
                    &task_id,
                    &input.subagent_type,
                    res.tool_calls,
                    res.turns,
                    duration_ms,
                ))
            } else {
                Err(EngineError::ToolArgs {
                    name: "task".into(),
                    message: format!("task {task_id} state lost"),
                })
            }
        }
    }

    /// Run subagent turn loop.
    pub async fn run_subagent(
        &self,
        task_id: &str,
        prompt: &str,
        resume_from: Option<&str>,
        model_override: Option<&str>,
    ) {
        let start_time = Instant::now();
        let mut messages = Vec::new();

        if let Some(rf) = resume_from {
            let guard = self.tasks.lock().unwrap();
            if let Some(prev) = guard.get(rf) {
                messages = prev.messages.clone();
            }
        }

        messages.push(ChatMessage::user(prompt));

        let cancel_token = {
            let guard = self.tasks.lock().unwrap();
            guard.get(task_id).map(|t| t.cancel_token.clone())
        };

        let cancel_token = match cancel_token {
            Some(t) => t,
            None => return,
        };

        let system_prompt = self.system_prompt();
        let max_turns = self.config.max_turns.min(10);
        let mut turns_used = 0u32;
        let mut total_tool_calls = 0u32;
        let mut final_text = String::new();
        let mut failed = false;

        while turns_used < max_turns {
            if cancel_token.is_cancelled() {
                let mut guard = self.tasks.lock().unwrap();
                if let Some(record) = guard.get_mut(task_id) {
                    record.status = TaskStatus::Cancelled;
                    record.ended = Some(Utc::now().to_rfc3339());
                    record.duration_secs = start_time.elapsed().as_secs_f64();
                    record.output = "Task was cancelled.".to_string();
                    record.logs.push(format!("[{}] Task cancelled via token.", Utc::now().to_rfc3339()));
                }
                return;
            }

            turns_used += 1;
            {
                let mut guard = self.tasks.lock().unwrap();
                if let Some(record) = guard.get_mut(task_id) {
                    record.logs.push(format!(
                        "[{}] Turn {}/{} started.",
                        Utc::now().to_rfc3339(),
                        turns_used,
                        max_turns
                    ));
                }
            }

            let mut req_messages = vec![ChatMessage::system(&system_prompt)];
            req_messages.extend(messages.clone());

            let model = model_override.unwrap_or(&self.config.model).to_string();
            let hosted =
                HostedTool::for_request(self.config.enable_web_search, self.config.api_backend);
            let request = ChatCompletionRequest {
                model,
                messages: req_messages,
                stream: Some(false),
                tools: drop_colliding_function_tools(
                    self.subagent_tools.wire_definitions(),
                    &hosted,
                ),
                tool_choice: Some(serde_json::json!("auto")),
                temperature: Some(self.config.temperature),
                max_tokens: Some(self.config.max_output_tokens),
                search_parameters: None,
                hosted_tools: hosted,
            };

            let observer = NullObserver;
            match self.client.complete(&request, &observer).await {
                Ok(turn) => {
                    let reasoning_opt = if turn.reasoning.is_empty() {
                        None
                    } else {
                        Some(turn.reasoning.clone())
                    };
                    let reasoning_items = turn.reasoning_items.clone();
                    let encrypted_reasoning = turn.encrypted_reasoning.clone();

                    let origin = self.client.origin_fingerprint();
                    if turn.tool_calls.is_empty() {
                        final_text = turn.text.clone();
                        messages.push(
                            ChatMessage::assistant_with_reasoning(
                                turn.text,
                                reasoning_opt,
                                reasoning_items,
                                encrypted_reasoning,
                            )
                            .with_origin(origin.clone()),
                        );
                        {
                            let mut guard = self.tasks.lock().unwrap();
                            if let Some(record) = guard.get_mut(task_id) {
                                record.logs.push(format!(
                                    "[{}] Turn completed with text response.",
                                    Utc::now().to_rfc3339()
                                ));
                            }
                        }
                        break;
                    }

                    messages.push(
                        ChatMessage::assistant_tool_calls_with_reasoning(
                            turn.tool_calls.clone(),
                            if turn.text.is_empty() { None } else { Some(turn.text.clone()) },
                            reasoning_opt,
                            reasoning_items,
                            encrypted_reasoning,
                        )
                        .with_origin(origin),
                    );

                    for call in &turn.tool_calls {
                        total_tool_calls += 1;
                        let name = call.function.name.clone();
                        let args: Value = serde_json::from_str(&call.function.arguments)
                            .unwrap_or_else(|_| serde_json::json!({}));

                        {
                            let mut guard = self.tasks.lock().unwrap();
                            if let Some(record) = guard.get_mut(task_id) {
                                record.logs.push(format!(
                                    "[{}] Tool call '{name}': args={}",
                                    Utc::now().to_rfc3339(),
                                    call.function.arguments
                                ));
                            }
                        }

                        let output_text = if let Some(tool) = self.subagent_tools.get(&name) {
                            let ctx = ToolCtx {
                                sandbox: self.sandbox.clone(),
                                cancel: cancel_token.clone(),
                            };
                            match tool.execute(args, &ctx).await {
                                Ok(out) => out.text,
                                Err(e) => format!("Error: {e}"),
                            }
                        } else {
                            format!("Error: tool {name} not found")
                        };

                        {
                            let mut guard = self.tasks.lock().unwrap();
                            if let Some(record) = guard.get_mut(task_id) {
                                record.logs.push(format!(
                                    "[{}] Tool output '{name}': {}",
                                    Utc::now().to_rfc3339(),
                                    crate::tools::truncate_chars(&output_text, 150)
                                ));
                            }
                        }

                        messages.push(ChatMessage::tool_result(call.id.clone(), output_text));
                    }
                }
                Err(e) => {
                    final_text = format!("Subagent failed with error: {e}");
                    failed = true;
                    {
                        let mut guard = self.tasks.lock().unwrap();
                        if let Some(record) = guard.get_mut(task_id) {
                            record.logs.push(format!(
                                "[{}] Subagent error: {e}",
                                Utc::now().to_rfc3339()
                            ));
                        }
                    }
                    break;
                }
            }
        }

        let elapsed = start_time.elapsed();
        let mut guard = self.tasks.lock().unwrap();
        if let Some(record) = guard.get_mut(task_id) {
            if record.status != TaskStatus::Cancelled {
                record.status = if failed { TaskStatus::Failed } else { TaskStatus::Completed };
                record.ended = Some(Utc::now().to_rfc3339());
                record.duration_secs = elapsed.as_secs_f64();
                record.output = final_text;
                record.tool_calls = total_tool_calls;
                record.turns = turns_used;
                record.messages = messages;
                record.logs.push(format!(
                    "[{}] Subagent task finished with status '{:?}' in {:.2}s.",
                    Utc::now().to_rfc3339(),
                    record.status,
                    record.duration_secs
                ));
            }
        }
    }

    /// Fetch task output/status.
    pub async fn get_task_output(
        &self,
        task_ids: &[String],
        timeout_ms: Option<u64>,
    ) -> EngineResult<String> {
        let timeout = timeout_ms.unwrap_or(0);
        if timeout > 0 {
            let start = Instant::now();
            let timeout_dur = std::time::Duration::from_millis(timeout);
            loop {
                let all_done = {
                    let guard = self.tasks.lock().unwrap();
                    task_ids.iter().all(|id| {
                        guard.get(id).map_or(true, |t| t.status != TaskStatus::Running)
                    })
                };
                if all_done || start.elapsed() >= timeout_dur {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }

        let guard = self.tasks.lock().unwrap();
        let mut results = Vec::new();

        for id in task_ids {
            if let Some(record) = guard.get(id) {
                let status_str = match record.status {
                    TaskStatus::Running => "running",
                    TaskStatus::Completed => "completed",
                    TaskStatus::Failed => "failed",
                    TaskStatus::Cancelled => "cancelled",
                };

                let item = serde_json::json!({
                    "task_id": record.task_id,
                    "description": record.description,
                    "subagent_type": record.subagent_type,
                    "status": status_str,
                    "started": record.started,
                    "ended": record.ended,
                    "duration_secs": record.duration_secs,
                    "output": record.output,
                    "turns": record.turns,
                    "tool_calls": record.tool_calls
                });
                results.push(item);
            } else {
                results.push(serde_json::json!({
                    "task_id": id,
                    "status": "not_found",
                    "output": format!("Task '{id}' not found.")
                }));
            }
        }

        if results.len() == 1 {
            Ok(serde_json::to_string_pretty(&results[0]).unwrap())
        } else {
            Ok(serde_json::to_string_pretty(&results).unwrap())
        }
    }

    /// Kill/terminate a running task.
    pub fn kill_task(&self, task_id: &str) -> EngineResult<String> {
        let mut guard = self.tasks.lock().unwrap();
        if let Some(record) = guard.get_mut(task_id) {
            if record.status == TaskStatus::Running {
                record.cancel_token.cancel();
                record.status = TaskStatus::Cancelled;
                record.ended = Some(Utc::now().to_rfc3339());
                record.output = "Task terminated by kill_task.".to_string();
                Ok(format!("Task '{task_id}' was cancelled."))
            } else {
                Ok(format!("Task '{task_id}' is already in terminal state."))
            }
        } else {
            Ok(format!("Task '{task_id}' not found."))
        }
    }

    /// Wait for multiple tasks to complete.
    pub async fn wait_tasks(
        &self,
        task_ids: &[String],
        mode: &str,
        timeout_ms: Option<u64>,
    ) -> EngineResult<String> {
        let timeout = timeout_ms.unwrap_or(60_000);
        let start = Instant::now();
        let timeout_dur = std::time::Duration::from_millis(timeout);
        let wait_any = mode.eq_ignore_ascii_case("wait_any");

        loop {
            let (all_done, any_done) = {
                let guard = self.tasks.lock().unwrap();
                let mut all = true;
                let mut any = false;
                for id in task_ids {
                    if let Some(t) = guard.get(id) {
                        if t.status == TaskStatus::Running {
                            all = false;
                        } else {
                            any = true;
                        }
                    } else {
                        any = true;
                    }
                }
                (all, any)
            };

            if (wait_any && any_done) || (!wait_any && all_done) || start.elapsed() >= timeout_dur {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        self.get_task_output(task_ids, None).await
    }

    /// Fetch log stream for a task with optional offset, max_lines, and regex filter.
    pub fn get_task_logs(
        &self,
        task_id: &str,
        offset: Option<usize>,
        max_lines: usize,
        filter: Option<&str>,
        tail: bool,
    ) -> EngineResult<serde_json::Value> {
        let guard = self.tasks.lock().unwrap();
        let record = match guard.get(task_id) {
            Some(r) => r,
            None => {
                return Ok(serde_json::json!({
                    "target": "task",
                    "task_id": task_id,
                    "status": "not_found",
                    "output": format!("Task '{task_id}' not found.")
                }));
            }
        };

        let status_str = match record.status {
            TaskStatus::Running => "running",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
        };

        let regex_matcher = if let Some(pat) = filter {
            RegexBuilder::new(pat).case_insensitive(true).build().ok()
        } else {
            None
        };

        let filtered_logs: Vec<&str> = record
            .logs
            .iter()
            .map(|s| s.as_str())
            .filter(|line| {
                if let Some(ref re) = regex_matcher {
                    re.is_match(line)
                } else if let Some(pat) = filter {
                    line.contains(pat)
                } else {
                    true
                }
            })
            .collect();

        let total_lines = filtered_logs.len();
        let (start_idx, end_idx) = match offset {
            Some(off) => {
                let start = off.min(total_lines);
                let end = (start + max_lines).min(total_lines);
                (start, end)
            }
            None => {
                if tail {
                    let start = total_lines.saturating_sub(max_lines);
                    (start, total_lines)
                } else {
                    let end = max_lines.min(total_lines);
                    (0, end)
                }
            }
        };

        let selected = &filtered_logs[start_idx..end_idx];
        let content = if selected.is_empty() {
            if record.output.is_empty() {
                "(no log lines recorded yet)".to_string()
            } else {
                record.output.clone()
            }
        } else {
            selected.join("\n")
        };

        Ok(serde_json::json!({
            "target": "task",
            "task_id": record.task_id,
            "description": record.description,
            "subagent_type": record.subagent_type,
            "status": status_str,
            "started": record.started,
            "ended": record.ended,
            "duration_secs": record.duration_secs,
            "turns": record.turns,
            "tool_calls": record.tool_calls,
            "total_log_lines": total_lines,
            "start_line": start_idx + 1,
            "lines_returned": selected.len(),
            "next_offset": end_idx,
            "filter": filter,
            "logs": content,
            "latest_output": record.output
        }))
    }
}
