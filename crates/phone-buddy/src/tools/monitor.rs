//! Log and Task Output Monitoring Tool (`monitor`).
//!
//! Provides mobile-sandbox safe monitoring of:
//! 1. Subagent task logs and execution status (`target: "task"`)
//! 2. Sandbox log files with offset reading, tailing, and regex filtering (`target: "file"`)
//! 3. Host Native application logs via HostToolHub (`target: "host"`)

use std::sync::Arc;

use async_trait::async_trait;
use regex::RegexBuilder;
use serde_json::Value;

use crate::agent::task_manager::TaskManager;
use crate::error::{EngineError, EngineResult};
use crate::tools::host::HostToolHub;
use crate::tools::{
    arg_opt_str, arg_opt_usize, s_boolean, s_enum, s_integer, s_string, schema_object, Tool,
    ToolCtx, ToolOutput, ToolSpec,
};

pub struct MonitorTool {
    task_manager: Arc<TaskManager>,
    host_tools: Arc<HostToolHub>,
}

impl MonitorTool {
    pub fn new(task_manager: Arc<TaskManager>, host_tools: Arc<HostToolHub>) -> Self {
        Self {
            task_manager,
            host_tools,
        }
    }
}

pub fn arc(task_manager: Arc<TaskManager>, host_tools: Arc<HostToolHub>) -> Arc<dyn Tool> {
    Arc::new(MonitorTool::new(task_manager, host_tools))
}

#[async_trait]
impl Tool for MonitorTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "monitor".into(),
            description: "Monitor logs, subagent task output streams, or host application logs incrementally without blocking. Supports file tailing, task log monitoring, and host log bridging.".into(),
            parameters: schema_object(
                vec![
                    ("target", s_enum(&["auto", "file", "task", "host"]), "Monitoring target: 'file' for log files, 'task' for subagent tasks, 'host' for native app logs, or 'auto' (default)."),
                    ("path", s_string(), "File path relative to sandbox (required for target='file')."),
                    ("task_id", s_string(), "Subagent task ID e.g. 'task-1' (required for target='task')."),
                    ("event_type", s_string(), "Event type or channel filter for host logs (for target='host')."),
                    ("offset", s_integer(), "Line index offset to start reading from (0-indexed). Omitted for tail mode."),
                    ("lines", s_integer(), "Maximum number of lines to retrieve (default: 50, max: 500)."),
                    ("filter", s_string(), "Optional text substring or regex pattern to filter log lines."),
                    ("tail", s_boolean(), "If true (default), returns the most recent lines when offset is omitted."),
                ],
                &[],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let raw_target = arg_opt_str(&args, "target").unwrap_or_else(|| "auto".to_string());
        let path = arg_opt_str(&args, "path");
        let task_id = arg_opt_str(&args, "task_id");
        let event_type = arg_opt_str(&args, "event_type");
        let offset = args
            .get("offset")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let max_lines = arg_opt_usize(&args, "lines", 50).min(500);
        let filter = arg_opt_str(&args, "filter");
        let tail = args.get("tail").and_then(Value::as_bool).unwrap_or(true);

        // Resolve target if auto
        let target = if raw_target == "auto" {
            if task_id.is_some() {
                "task"
            } else if path.is_some() {
                "file"
            } else if event_type.is_some() {
                "host"
            } else {
                "file"
            }
        } else {
            raw_target.as_str()
        };

        match target {
            "task" => {
                self.execute_task_monitor(task_id, offset, max_lines, filter, tail)
                    .await
            }
            "file" => {
                self.execute_file_monitor(ctx, path, offset, max_lines, filter, tail)
                    .await
            }
            "host" => {
                self.execute_host_monitor(ctx, event_type, offset, max_lines, filter)
                    .await
            }
            _ => Err(EngineError::ToolArgs {
                name: "monitor".into(),
                message: format!(
                    "unsupported target '{target}', must be 'file', 'task', or 'host'"
                ),
            }),
        }
    }
}

impl MonitorTool {
    async fn execute_task_monitor(
        &self,
        task_id: Option<String>,
        offset: Option<usize>,
        max_lines: usize,
        filter: Option<String>,
        tail: bool,
    ) -> EngineResult<ToolOutput> {
        let tid = match task_id {
            Some(id) if !id.trim().is_empty() => id,
            _ => {
                return Err(EngineError::ToolArgs {
                    name: "monitor".into(),
                    message: "missing required 'task_id' for target='task'".into(),
                })
            }
        };

        let result =
            self.task_manager
                .get_task_logs(&tid, offset, max_lines, filter.as_deref(), tail)?;
        Ok(ToolOutput::new(
            serde_json::to_string_pretty(&result).unwrap(),
        ))
    }

    async fn execute_file_monitor(
        &self,
        ctx: &ToolCtx,
        path: Option<String>,
        offset: Option<usize>,
        max_lines: usize,
        filter: Option<String>,
        tail: bool,
    ) -> EngineResult<ToolOutput> {
        let p = match path {
            Some(p) if !p.trim().is_empty() => p,
            _ => {
                return Err(EngineError::ToolArgs {
                    name: "monitor".into(),
                    message: "missing required 'path' for target='file'".into(),
                })
            }
        };

        let resolved_path = match ctx.sandbox.resolve(&p) {
            Ok(path_buf) => path_buf,
            Err(e) => {
                return Ok(ToolOutput::new(format!(
                    "Sandbox path error for '{p}': {e}"
                )))
            }
        };

        let content = match std::fs::read_to_string(&resolved_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput::new(format!(
                    "Failed to read log file '{p}': {e}"
                )))
            }
        };

        let regex_matcher = if let Some(ref pat) = filter {
            RegexBuilder::new(pat).case_insensitive(true).build().ok()
        } else {
            None
        };

        let all_lines: Vec<&str> = content
            .lines()
            .filter(|line| {
                if let Some(ref re) = regex_matcher {
                    re.is_match(line)
                } else if let Some(ref pat) = filter {
                    line.contains(pat)
                } else {
                    true
                }
            })
            .collect();

        let total_lines = all_lines.len();
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

        let selected_lines = &all_lines[start_idx..end_idx];
        let next_offset = end_idx;
        let formatted_content = selected_lines
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:4} | {}", start_idx + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        let response = serde_json::json!({
            "target": "file",
            "path": p,
            "total_lines": total_lines,
            "start_line": start_idx + 1,
            "lines_returned": selected_lines.len(),
            "next_offset": next_offset,
            "filter": filter,
            "content": if formatted_content.is_empty() { "(no matching log lines found)".to_string() } else { formatted_content }
        });

        Ok(ToolOutput::new(
            serde_json::to_string_pretty(&response).unwrap(),
        ))
    }

    async fn execute_host_monitor(
        &self,
        ctx: &ToolCtx,
        event_type: Option<String>,
        offset: Option<usize>,
        max_lines: usize,
        filter: Option<String>,
    ) -> EngineResult<ToolOutput> {
        let event_name = event_type.unwrap_or_else(|| "app_logs".to_string());

        // If a host tool named 'monitor' or 'host_monitor' is registered by iOS/Android host app
        if self.host_tools.has("monitor") {
            let host_args = serde_json::json!({
                "event_type": event_name,
                "offset": offset,
                "lines": max_lines,
                "filter": filter,
            });
            return self
                .host_tools
                .execute("monitor", host_args, &ctx.cancel)
                .await;
        } else if self.host_tools.has("host_monitor") {
            let host_args = serde_json::json!({
                "event_type": event_name,
                "offset": offset,
                "lines": max_lines,
                "filter": filter,
            });
            return self
                .host_tools
                .execute("host_monitor", host_args, &ctx.cancel)
                .await;
        }

        // Notify host via HostToolHub notify_event for passive observation
        let notify_payload = serde_json::json!({
            "action": "monitor",
            "event_type": event_name,
            "filter": filter,
        });
        self.host_tools
            .notify_event("monitor", &notify_payload.to_string());

        let res = serde_json::json!({
            "target": "host",
            "event_type": event_name,
            "filter": filter,
            "status": "host_event_dispatched",
            "message": "Host app notified via HostToolHub. Native iOS (os_log) / Android (Logcat) can register a 'host_monitor' tool to supply live native log buffers directly."
        });

        Ok(ToolOutput::new(serde_json::to_string_pretty(&res).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use crate::llm::client::LlmClient;
    use crate::tools::fs::Sandbox;
    use crate::tools::ToolRegistry;
    use tempfile::tempdir;

    fn setup_env() -> (
        Arc<TaskManager>,
        Arc<HostToolHub>,
        Arc<Sandbox>,
        tempfile::TempDir,
    ) {
        let dir = tempdir().unwrap();
        let sandbox = Arc::new(Sandbox::new(dir.path()).unwrap());
        let mut config = EngineConfig::default();
        config.api_key = "test".into();
        config.root_dir = dir.path().to_path_buf();
        let client = Arc::new(LlmClient::from_http(&config).unwrap());
        let subagent_tools = Arc::new(ToolRegistry::new());
        let task_manager = Arc::new(TaskManager::new(
            config,
            client,
            sandbox.clone(),
            subagent_tools,
        ));
        let host_tools = HostToolHub::new();
        (task_manager, host_tools, sandbox, dir)
    }

    #[tokio::test]
    async fn test_monitor_file_tail_and_filter() {
        let (task_manager, host_tools, sandbox, _dir) = setup_env();
        let log_path = sandbox.resolve("app.log").unwrap();
        std::fs::write(&log_path, "INFO: App started\nDEBUG: Loading config\nERROR: Network failed\nINFO: Retry success\n").unwrap();

        let tool = MonitorTool::new(task_manager, host_tools);
        let ctx = ToolCtx {
            sandbox,
            cancel: tokio_util::sync::CancellationToken::new(),
        };

        // Test filtering for ERROR
        let args = serde_json::json!({
            "target": "file",
            "path": "app.log",
            "filter": "ERROR"
        });
        let out = tool.execute(args, &ctx).await.unwrap();
        assert!(out.text.contains("ERROR: Network failed"));
        assert!(out.text.contains("\"lines_returned\": 1"));
    }

    #[tokio::test]
    async fn test_monitor_task_logs() {
        let (task_manager, host_tools, sandbox, _dir) = setup_env();

        // Spawn a background task
        let task_input = crate::agent::task_manager::TaskInput {
            prompt: "Test task prompt".into(),
            description: "Test description".into(),
            subagent_type: "general-purpose".into(),
            run_in_background: true,
            resume_from: None,
        };
        let spawn_res = task_manager.spawn_task(task_input).await.unwrap();
        assert!(spawn_res.contains("subagent_id: task-1"));

        let tool = MonitorTool::new(task_manager, host_tools);
        let ctx = ToolCtx {
            sandbox,
            cancel: tokio_util::sync::CancellationToken::new(),
        };

        let args = serde_json::json!({
            "target": "task",
            "task_id": "task-1"
        });
        let out = tool.execute(args, &ctx).await.unwrap();
        assert!(out.text.contains("task-1"));
        assert!(out.text.contains("Test description"));
    }
}
