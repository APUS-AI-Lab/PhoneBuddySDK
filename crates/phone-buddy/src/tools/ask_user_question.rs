//! Built-in ask_user_question tool for host UI interaction.
//!
//! Enables the agent to ask the host user a question during task execution.
//! Dispatches the request to [`HostToolHub`], which triggers the host app's
//! UI callback and awaits the user's response.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::host::HostToolHub;
use crate::tools::{
    arg_str, s_string, s_string_array, schema_object, Tool, ToolCtx, ToolOutput, ToolSpec,
};

pub struct AskUserQuestionTool {
    host_tools: Arc<HostToolHub>,
}

impl AskUserQuestionTool {
    pub fn new(host_tools: Arc<HostToolHub>) -> Self {
        Self { host_tools }
    }
}

pub fn arc(host_tools: Arc<HostToolHub>) -> Arc<dyn Tool> {
    Arc::new(AskUserQuestionTool::new(host_tools))
}

#[async_trait]
impl Tool for AskUserQuestionTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "ask_user_question".into(),
            description: "Ask the host user a question when clarification, decision, or input is needed. The host UI will present the question and return the user's response.".into(),
            parameters: schema_object(
                vec![
                    ("question", s_string(), "The question to ask the user."),
                    ("options", s_string_array(), "Optional array of predefined choice options for the user to select from."),
                ],
                &["question"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let question = arg_str(&args, "question")?;

        match self
            .host_tools
            .dispatch_host_call("ask_user_question", args, &ctx.cancel)
            .await
        {
            Ok(output) => Ok(output),
            Err(EngineError::Tool { message, .. }) if message.contains("callback is not set") => {
                Ok(ToolOutput::new(format!(
                    "User question could not be presented to host UI (no host UI callback configured). Question asked: '{}'",
                    question
                )))
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;
    use crate::tools::fs::Sandbox;

    #[tokio::test]
    async fn ask_user_question_round_trip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let sandbox = Arc::new(Sandbox::new(temp_dir.path()).unwrap());
        let cancel = CancellationToken::new();
        let ctx = ToolCtx { sandbox, cancel };

        let host_tools = HostToolHub::new();
        let tool = AskUserQuestionTool::new(host_tools.clone());

        // Spec verification
        let spec = tool.spec();
        assert_eq!(spec.name, "ask_user_question");

        // Set host notify callback
        let hub_c = host_tools.clone();
        host_tools.set_notify(Arc::new(move |call_id, name, args_json| {
            assert_eq!(name, "ask_user_question");
            let v: Value = serde_json::from_str(&args_json).unwrap();
            assert_eq!(v["question"], "Which color?");
            let hub = hub_c.clone();
            std::thread::spawn(move || {
                let _ = hub.complete(&call_id, true, "Blue");
            });
        }));

        let args = serde_json::json!({
            "question": "Which color?",
            "options": ["Red", "Blue", "Green"]
        });

        let out = tool.execute(args, &ctx).await.unwrap();
        assert_eq!(out.text, "Blue");
    }

    #[tokio::test]
    async fn ask_user_question_no_callback_fallback() {
        let temp_dir = tempfile::tempdir().unwrap();
        let sandbox = Arc::new(Sandbox::new(temp_dir.path()).unwrap());
        let cancel = CancellationToken::new();
        let ctx = ToolCtx { sandbox, cancel };

        let host_tools = HostToolHub::new();
        let tool = AskUserQuestionTool::new(host_tools);

        let args = serde_json::json!({
            "question": "Confirm deletion?"
        });

        let out = tool.execute(args, &ctx).await.unwrap();
        assert!(out.text.contains("User question could not be presented to host UI"));
        assert!(out.text.contains("Confirm deletion?"));
    }
}
