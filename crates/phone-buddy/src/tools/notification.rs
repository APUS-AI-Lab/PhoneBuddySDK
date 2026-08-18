//! Notification tool for sending or scheduling local user notifications and reminders.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::host::HostToolHub;
use crate::tools::{
    arg_opt_str, arg_str, s_boolean, s_enum, s_string, schema_object, Tool, ToolCtx,
    ToolOutput, ToolSpec,
};

pub struct NotificationTool {
    host_tools: Arc<HostToolHub>,
}

impl NotificationTool {
    pub fn new(host_tools: Arc<HostToolHub>) -> Self {
        Self { host_tools }
    }
}

pub fn arc(host_tools: Arc<HostToolHub>) -> Arc<dyn Tool> {
    Arc::new(NotificationTool::new(host_tools))
}

#[async_trait]
impl Tool for NotificationTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "notification".into(),
            description: "Send or schedule local user notifications and reminders on mobile devices (iOS UNUserNotificationCenter / Android NotificationManager).".into(),
            parameters: schema_object(
                vec![
                    ("action", s_enum(&["send", "schedule"]), "Action to perform: 'send' for immediate notification, 'schedule' for delayed/scheduled reminder"),
                    ("title", s_string(), "Title of the notification or reminder"),
                    ("body", s_string(), "Message body content of the notification"),
                    ("delay_or_time", s_string(), "For 'schedule': delay string (e.g. '10m', '1h') or ISO timestamp (e.g. '2026-08-12T18:00:00Z')"),
                    ("sound", s_boolean(), "Whether to play notification sound (default: true)"),
                ],
                &["action", "title", "body"],
            ),
        }
    }

    async fn execute(&self, args: Value, _ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let action = arg_str(&args, "action")?;
        let title = arg_str(&args, "title")?;
        let body = arg_str(&args, "body")?;
        let delay_or_time = arg_opt_str(&args, "delay_or_time");
        let sound = args.get("sound").and_then(Value::as_bool).unwrap_or(true);

        let event_name = match action.as_str() {
            "send" => "notification_send",
            "schedule" => "notification_schedule",
            other => return Err(EngineError::ToolArgs {
                name: "notification".into(),
                message: format!("unknown action '{other}'; expected 'send' or 'schedule'"),
            }),
        };

        let payload = serde_json::json!({
            "action": action,
            "title": title,
            "body": body,
            "delay_or_time": delay_or_time,
            "sound": sound,
        });

        self.host_tools.notify_event(event_name, &payload.to_string());

        let res = serde_json::json!({
            "status": "dispatched",
            "action": action,
            "title": title,
            "body": body,
            "delay_or_time": delay_or_time,
            "notice": "Notification request dispatched to host OS native notification system."
        });

        Ok(ToolOutput::new(serde_json::to_string_pretty(&res).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_notification_tool() {
        let host_tools = HostToolHub::new();
        let tool = NotificationTool::new(host_tools.clone());

        let root = tempfile::tempdir().unwrap();
        let sandbox = Arc::new(crate::tools::fs::Sandbox::new(root.path()).unwrap());
        let ctx = ToolCtx {
            sandbox,
            cancel: tokio_util::sync::CancellationToken::new(),
        };

        let out = tool
            .execute(
                serde_json::json!({
                    "action": "send",
                    "title": "Meeting Reminder",
                    "body": "Daily sync starts now",
                    "sound": true
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(out.text.contains("dispatched"));
        assert!(out.text.contains("Meeting Reminder"));
    }
}
