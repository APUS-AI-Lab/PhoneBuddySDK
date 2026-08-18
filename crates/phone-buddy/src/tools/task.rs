//! Subagent and Task management tools (`task`, `task_output`, `get_task_output`, `kill_task`, `wait_tasks`).

use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value;

use crate::agent::task_manager::{TaskInput, TaskManager, MAX_MULTI_WAIT_IDS};
use crate::error::{EngineError, EngineResult};
use crate::tools::{
    schema_object, s_boolean, s_enum, s_integer, s_string, s_string_array, Tool, ToolCtx, ToolOutput,
    ToolSpec,
};

// ── 1. task (spawn) tool ───────────────────────────────────────────────────

pub struct TaskTool {
    manager: Arc<TaskManager>,
}

impl TaskTool {
    pub fn new(manager: Arc<TaskManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "task".into(),
            description: "Launch a subagent to execute a task autonomously. Can run synchronously or in background.".into(),
            parameters: schema_object(
                vec![
                    ("prompt", s_string(), "The full task prompt for the subagent to execute."),
                    ("description", s_string(), "Short description of the task (3-5 words)."),
                    ("subagent_type", s_string(), "Name of subagent type (default: 'general-purpose')."),
                    ("run_in_background", s_boolean(), "Whether to run subagent in background (default: true)."),
                    ("resume_from", s_string(), "Subagent ID to resume conversation from."),
                    ("model", s_string(), "Optional model override."),
                ],
                &["prompt", "description"],
            ),
        }
    }

    async fn execute(&self, args: Value, _ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let input: TaskInput = serde_json::from_value(args).map_err(|e| EngineError::ToolArgs {
            name: "task".into(),
            message: format!("invalid arguments: {e}"),
        })?;

        let res = self.manager.spawn_task(input).await?;
        Ok(ToolOutput::new(res))
    }
}

// ── 2. task_output / get_task_output tool ──────────────────────────────────

pub struct TaskOutputTool {
    name: String,
    manager: Arc<TaskManager>,
}

impl TaskOutputTool {
    pub fn new(name: &str, manager: Arc<TaskManager>) -> Self {
        Self {
            name: name.to_string(),
            manager,
        }
    }
}

#[async_trait]
impl Tool for TaskOutputTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: "Get output or status from one or more background subagent tasks.".into(),
            parameters: schema_object(
                vec![
                    ("task_ids", s_string_array(), "Task IDs to inspect or wait for."),
                    ("task_id", s_string(), "Single task ID alias for convenience."),
                    ("timeout_ms", s_integer(), "Max wait time in milliseconds. 0 or omitted for non-blocking poll."),
                ],
                &[],
            ),
        }
    }

    async fn execute(&self, args: Value, _ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let mut ids = Vec::new();

        if let Some(arr) = args.get("task_ids").and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    ids.push(s.to_string());
                }
            }
        }
        if ids.is_empty() {
            if let Some(s) = args.get("task_id").and_then(|v| v.as_str()) {
                ids.push(s.to_string());
            }
        }

        if ids.is_empty() {
            return Err(EngineError::ToolArgs {
                name: self.name.clone(),
                message: "task_ids list or task_id string is required".into(),
            });
        }

        // Cap multi-id waits (grok MAX_MULTI_WAIT_IDS = 20).
        if ids.len() > MAX_MULTI_WAIT_IDS {
            return Err(EngineError::ToolArgs {
                name: self.name.clone(),
                message: format!(
                    "too many task_ids ({}): max is {MAX_MULTI_WAIT_IDS}",
                    ids.len()
                ),
            });
        }

        let timeout_ms = args.get("timeout_ms").and_then(|v| v.as_u64());
        let res = self.manager.get_task_output(&ids, timeout_ms).await?;
        Ok(ToolOutput::new(res))
    }
}

// ── 3. kill_task tool ──────────────────────────────────────────────────────

pub struct KillTaskTool {
    manager: Arc<TaskManager>,
}

impl KillTaskTool {
    pub fn new(manager: Arc<TaskManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for KillTaskTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "kill_task".into(),
            description: "Terminate a running background subagent task by ID.".into(),
            parameters: schema_object(
                vec![("task_id", s_string(), "The task ID to terminate.")],
                &["task_id"],
            ),
        }
    }

    async fn execute(&self, args: Value, _ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let task_id = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| EngineError::ToolArgs {
                name: "kill_task".into(),
                message: "task_id is required".into(),
            })?;

        let res = self.manager.kill_task(task_id)?;
        Ok(ToolOutput::new(res))
    }
}

// ── 4. wait_tasks tool ─────────────────────────────────────────────────────

pub struct WaitTasksTool {
    manager: Arc<TaskManager>,
}

impl WaitTasksTool {
    pub fn new(manager: Arc<TaskManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for WaitTasksTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "wait_tasks".into(),
            description: "Wait for multiple background subagent tasks to complete.".into(),
            parameters: schema_object(
                vec![
                    ("task_ids", s_string_array(), "Task IDs to wait for."),
                    ("mode", s_enum(&["wait_any", "wait_all"]), "Wait mode: wait_any or wait_all (default: wait_all)."),
                    ("timeout_ms", s_integer(), "Max wait time in milliseconds."),
                ],
                &["task_ids"],
            ),
        }
    }

    async fn execute(&self, args: Value, _ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let mut ids = Vec::new();
        if let Some(arr) = args.get("task_ids").and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    ids.push(s.to_string());
                }
            }
        }

        if ids.is_empty() {
            return Err(EngineError::ToolArgs {
                name: "wait_tasks".into(),
                message: "task_ids is required".into(),
            });
        }

        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("wait_all");
        let timeout_ms = args.get("timeout_ms").and_then(|v| v.as_u64());

        let res = self.manager.wait_tasks(&ids, mode, timeout_ms).await?;
        Ok(ToolOutput::new(res))
    }
}

// ── Constructors for Arc<dyn Tool> ─────────────────────────────────────────

pub fn task_arc(manager: Arc<TaskManager>) -> Arc<dyn Tool> {
    Arc::new(TaskTool::new(manager))
}

pub fn task_output_arc(manager: Arc<TaskManager>) -> Arc<dyn Tool> {
    Arc::new(TaskOutputTool::new("task_output", manager))
}

pub fn get_task_output_arc(manager: Arc<TaskManager>) -> Arc<dyn Tool> {
    Arc::new(TaskOutputTool::new("get_task_output", manager))
}

pub fn kill_task_arc(manager: Arc<TaskManager>) -> Arc<dyn Tool> {
    Arc::new(KillTaskTool::new(manager))
}

pub fn wait_tasks_arc(manager: Arc<TaskManager>) -> Arc<dyn Tool> {
    Arc::new(WaitTasksTool::new(manager))
}
