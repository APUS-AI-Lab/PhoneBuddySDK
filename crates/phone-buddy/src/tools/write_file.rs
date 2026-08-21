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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_ctx() -> (tempfile::TempDir, ToolCtx) {
        let dir = tempdir().unwrap();
        let sandbox = Arc::new(crate::tools::fs::Sandbox::new(dir.path()).unwrap());
        let ctx = ToolCtx {
            sandbox,
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        (dir, ctx)
    }

    #[tokio::test]
    async fn test_write_file_basic_and_auto_parent_dir() {
        let (dir, ctx) = make_ctx();
        let tool = WriteFileTool;

        // Write with nested subdirectories automatically created
        let res = tool
            .execute(
                serde_json::json!({
                    "path": "sub/deep/test.txt",
                    "content": "sample content"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(res.text.contains("Wrote 14 bytes"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("sub/deep/test.txt")).unwrap(),
            "sample content"
        );

        // Overwrite file
        tool.execute(
            serde_json::json!({
                "path": "sub/deep/test.txt",
                "content": "new content"
            }),
            &ctx,
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("sub/deep/test.txt")).unwrap(),
            "new content"
        );
    }

    #[tokio::test]
    async fn test_write_file_directory_and_escape_rejection() {
        let (dir, ctx) = make_ctx();
        let tool = WriteFileTool;

        std::fs::create_dir(dir.path().join("folder")).unwrap();

        // Reject writing to an existing directory
        let res_dir = tool
            .execute(
                serde_json::json!({
                    "path": "folder",
                    "content": "data"
                }),
                &ctx,
            )
            .await;
        assert!(res_dir.is_err());

        // Reject sandbox escape
        let res_escape = tool
            .execute(
                serde_json::json!({
                    "path": "../../escaped.txt",
                    "content": "data"
                }),
                &ctx,
            )
            .await;
        assert!(res_escape.is_err());
    }
}
