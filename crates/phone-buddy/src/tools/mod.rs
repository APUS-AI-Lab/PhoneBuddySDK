//! Tool abstraction and registry.
//!
//! The `Tool` trait mirrors the shape of grok's tool layer
//! (`xai-tool-runtime` / `xai-grok-tools`): each tool has a JSON-schema
//! definition exposed to the model, and executes against shared resources.
//! The heavy desktop machinery (Resources map, notifications, versioning,
//! gRPC) is replaced by a single [`ToolCtx`] carrying the sandbox root and
//! a cancellation token, which is all a mobile engine needs.

pub mod ask_user_question;
pub mod binary;
pub mod browser;
pub mod busybox;
pub mod edit_file;
pub mod edit_helpers;
pub mod fs;
pub mod grep;
pub mod host;
pub mod list_dir;
pub mod monitor;
pub mod notification;
pub mod plan;
pub mod read_file;
pub mod scheduler;
pub mod script;
pub mod ssrf;
pub mod task;
pub mod unicode_confusables;
pub mod web_fetch;
pub mod web_search;
pub mod webview;
pub mod write_file;

pub use ask_user_question::AskUserQuestionTool;
pub use host::{HostToolHub, HostToolNotify};
pub use webview::{WebViewFetchNotify, WebViewFetchRequest, WebViewHost};
pub use monitor::MonitorTool;
pub use notification::NotificationTool;
pub use scheduler::SchedulerTool;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::llm::types::{FunctionDefinitionWire, ToolDefinitionWire};
use crate::tools::fs::Sandbox;

/// Per-execution context passed to every tool.
pub struct ToolCtx {
    pub sandbox: Arc<Sandbox>,
    pub cancel: tokio_util::sync::CancellationToken,
}

/// Result of a tool execution: text shown to the model.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub text: String,
}

impl ToolOutput {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Static definition of a tool, exposed to the model.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput>;
}

/// Ordered registry of tools.
#[derive(Default)]
pub struct ToolRegistry {
    order: Vec<String>,
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.spec().name.clone();
        if !self.tools.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn names(&self) -> Vec<String> {
        self.order.clone()
    }

    /// Tool definitions in the chat-completions wire format.
    pub fn wire_definitions(&self) -> Vec<ToolDefinitionWire> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| {
                let spec = tool.spec();
                ToolDefinitionWire {
                    kind: "function".into(),
                    function: FunctionDefinitionWire {
                        name: spec.name,
                        description: Some(spec.description),
                        parameters: spec.parameters,
                    },
                }
            })
            .collect()
    }
}

// ── JSON-schema helpers ──────────────────────────────────────────────────

pub fn schema_object(props: Vec<(&str, Value, &str)>, required: &[&str]) -> Value {
    let mut properties = serde_json::Map::new();
    for (name, schema, desc) in props {
        let mut s = schema;
        if let Value::Object(m) = &mut s {
            m.insert("description".into(), Value::String(desc.to_string()));
        }
        properties.insert(name.to_string(), s);
    }
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

pub fn s_string() -> Value {
    serde_json::json!({"type": "string"})
}
pub fn s_integer() -> Value {
    serde_json::json!({"type": "integer"})
}
pub fn s_boolean() -> Value {
    serde_json::json!({"type": "boolean"})
}
pub fn s_string_array() -> Value {
    serde_json::json!({"type": "array", "items": {"type": "string"}})
}
pub fn s_enum(values: &[&str]) -> Value {
    serde_json::json!({"type": "string", "enum": values})
}

/// Helper: pull a required string argument.
pub fn arg_str(args: &Value, key: &str) -> EngineResult<String> {
    match args.get(key).and_then(Value::as_str) {
        Some(s) => Ok(s.to_string()),
        None => Err(EngineError::ToolArgs {
            name: key.into(),
            message: format!("missing or non-string argument '{key}'"),
        }),
    }
}

/// Helper: optional string argument.
pub fn arg_opt_str(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .filter(|s| !s.trim().is_empty())
}

/// Helper: optional string list argument (e.g. `allowed_domains`, `blocked_domains`).
pub fn arg_opt_str_list(args: &Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Helper: optional positive usize.
pub fn arg_opt_usize(args: &Value, key: &str, default: usize) -> usize {
    args.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Helper: optional bool.
pub fn arg_bool(args: &Value, key: &str) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Default preview size before a truncation footer (grok: `PREVIEW_SIZE`).
pub const PREVIEW_SIZE: usize = 2_000;

/// Truncate a line/string to at most `max_chars` characters.
/// Port of grok `truncate_line` marker form:
/// `{prefix} [... truncated ({char_count} chars total)]`
pub fn truncate_line(line: &str, max_chars: usize) -> String {
    if line.len() <= max_chars {
        // Fast path: byte length ≤ max_chars ⇒ char count ≤ max_chars for ASCII.
        return line.to_string();
    }
    let char_count = line.chars().count();
    if char_count <= max_chars {
        return line.to_string();
    }
    let end_byte = line
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(line.len());
    format!(
        "{} [... truncated ({} chars total)]",
        &line[..end_byte],
        char_count
    )
}

/// Truncate text to a char budget, appending a marker when truncated.
/// Uses the same marker convention as grok `truncate_line`.
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    truncate_line(text, max_chars)
}

/// Truncate a string to at most `max_bytes` bytes at a valid UTF-8 boundary.
/// Port of grok `truncate_str` (no marker).
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
