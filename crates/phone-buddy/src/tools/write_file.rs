//! `write_file` tool — create or overwrite a file inside the sandbox.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::{arg_str, schema_object, s_string, Tool, ToolCtx, ToolOutput, ToolSpec};

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: concat!(
                "Create or overwrite a file with the given content. Parent ",
                "directories are created automatically. For small changes to ",
                "an existing file prefer edit_file."
            )
            .into(),
            parameters: schema_object(
                vec![
                    ("path", s_string(), "File path, relative to the workspace root or absolute."),
                    ("content", s_string(), "Full file content."),
                ],
                &["path", "content"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let path = arg_str(&args, "path")?;
        let content = arg_str(&args, "content")?;

        let abs = ctx.sandbox.resolve(&path)?;
        if abs.is_dir() {
            return Err(EngineError::Tool {
                name: "write_file".into(),
                message: format!("'{}' is a directory", ctx.sandbox.display(&abs)),
            });
        }
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&abs, &content)?;
        Ok(ToolOutput::new(format!(
            "Wrote {} bytes to {}",
            content.len(),
            ctx.sandbox.display(&abs)
        )))
    }
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(WriteFileTool)
}
