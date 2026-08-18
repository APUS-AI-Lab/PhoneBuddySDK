//! Scheduler tool for scheduling background agent tasks on mobile/host platforms.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::scheduler_manager::SchedulerManager;
use crate::error::{EngineError, EngineResult};
use crate::tools::host::HostToolHub;
use crate::tools::{
    arg_bool, arg_opt_str, arg_str, s_boolean, s_enum, s_string, schema_object, Tool, ToolCtx,
    ToolOutput, ToolSpec,
};

pub struct SchedulerTool {
    scheduler_manager: Arc<SchedulerManager>,
    host_tools: Arc<HostToolHub>,
}

impl SchedulerTool {
    pub fn new(
        scheduler_manager: Arc<SchedulerManager>,
        host_tools: Arc<HostToolHub>,
    ) -> Self {
        Self {
            scheduler_manager,
            host_tools,
        }
    }
}

pub fn arc(
    scheduler_manager: Arc<SchedulerManager>,
    host_tools: Arc<HostToolHub>,
) -> Arc<dyn Tool> {
    Arc::new(SchedulerTool::new(scheduler_manager, host_tools))
}

#[async_trait]
impl Tool for SchedulerTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "scheduler".into(),
            description: "Schedule recurring or one-off tasks for background execution on mobile/host platforms. Supports creating, listing, and deleting scheduled agent tasks and notifies host OS schedulers (iOS BGTaskScheduler/Notifications, Android WorkManager/AlarmManager).".into(),
            parameters: schema_object(
                vec![
                    ("action", s_enum(&["create", "list", "delete"]), "Action to perform: 'create' to add a schedule, 'list' to view schedules, 'delete' to cancel a schedule"),
                    ("prompt", s_string(), "Task prompt/instruction to execute when triggered (required for 'create')"),
                    ("cron_or_time", s_string(), "Schedule specification: cron expression (e.g. '*/15 * * * *'), relative duration (e.g. '10m', '1h'), or ISO timestamp (e.g. '2026-08-12T18:00:00Z')"),
                    ("recurring", s_boolean(), "Whether this task is recurring periodically (default: false)"),
                    ("task_id", s_string(), "Task ID to delete (required for 'delete')"),
                ],
                &["action"],
            ),
        }
    }

    async fn execute(&self, args: Value, _ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let action = arg_str(&args, "action")?;
        match action.as_str() {
            "create" => {
                let prompt = arg_str(&args, "prompt")?;
                let cron_or_time = arg_opt_str(&args, "cron_or_time");
                let recurring = arg_bool(&args, "recurring");

                let item = self.scheduler_manager.create_task(
                    prompt,
                    cron_or_time,
                    recurring,
                    &self.host_tools,
                )?;

                let res = serde_json::json!({
                    "status": "scheduled",
                    "id": item.id,
                    "prompt": item.prompt,
                    "cron_or_time": item.cron_or_time,
                    "recurring": item.recurring,
                    "created_at": item.created_at,
                    "notice": "Scheduled task registered. Host OS native scheduler notified."
                });

                Ok(ToolOutput::new(serde_json::to_string_pretty(&res).unwrap()))
            }
            "list" => {
                let tasks = self.scheduler_manager.list_tasks();
                let res = serde_json::json!({
                    "count": tasks.len(),
                    "tasks": tasks
                });
                Ok(ToolOutput::new(serde_json::to_string_pretty(&res).unwrap()))
            }
            "delete" => {
                let task_id = arg_str(&args, "task_id")?;
                let deleted = self.scheduler_manager.delete_task(&task_id, &self.host_tools)?;
                let res = serde_json::json!({
                    "status": "deleted",
                    "id": deleted.id,
                    "prompt": deleted.prompt
                });
                Ok(ToolOutput::new(serde_json::to_string_pretty(&res).unwrap()))
            }
            other => Err(EngineError::ToolArgs {
                name: "scheduler".into(),
                message: format!("unknown action '{other}'; expected 'create', 'list', or 'delete'"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scheduler_tool_flow() {
        let root = tempfile::tempdir().unwrap();
        let sandbox = Arc::new(crate::tools::fs::Sandbox::new(root.path()).unwrap());
        let host_tools = HostToolHub::new();
        let manager = Arc::new(SchedulerManager::new(sandbox.clone()));
        let tool = SchedulerTool::new(manager.clone(), host_tools.clone());

        let ctx = ToolCtx {
            sandbox: sandbox.clone(),
            cancel: tokio_util::sync::CancellationToken::new(),
        };

        // Create task
        let out = tool
            .execute(
                serde_json::json!({
                    "action": "create",
                    "prompt": "Check morning weather",
                    "cron_or_time": "0 8 * * *",
                    "recurring": true
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(out.text.contains("scheduled"));
        assert!(out.text.contains("sched-1"));

        // List tasks
        let out_list = tool
            .execute(serde_json::json!({"action": "list"}), &ctx)
            .await
            .unwrap();
        assert!(out_list.text.contains("Check morning weather"));

        // Delete task
        let out_del = tool
            .execute(
                serde_json::json!({
                    "action": "delete",
                    "task_id": "sched-1"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(out_del.text.contains("deleted"));
        assert!(out_del.text.contains("sched-1"));

        // List tasks again
        let out_list2 = tool
            .execute(serde_json::json!({"action": "list"}), &ctx)
            .await
            .unwrap();
        assert!(out_list2.text.contains("\"count\": 0"));
    }
}
