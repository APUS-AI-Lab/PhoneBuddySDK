//! Host-provided tools (App Talents).
//!
//! Schemas are registered from the host via JSON; execution notifies the
//! host and waits for [`HostToolHub::complete`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use uuid::Uuid;

use crate::error::{EngineError, EngineResult};
use crate::llm::types::{FunctionDefinitionWire, ToolDefinitionWire};
use crate::tools::{ToolOutput, ToolSpec};

/// Notifies the host of a tool invocation.
/// Arguments: `(call_id, name, arguments_json)`.
pub type HostToolNotify = Arc<dyn Fn(String, String, String) + Send + Sync>;

/// Dynamic host-tool registry + pending result channels.
pub struct HostToolHub {
    notify: Mutex<Option<HostToolNotify>>,
    /// name → spec
    specs: Mutex<HashMap<String, ToolSpec>>,
    /// call_id → oneshot sender for the tool result text
    pending: Mutex<HashMap<String, tokio::sync::oneshot::Sender<Result<String, String>>>>,
}

impl HostToolHub {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            notify: Mutex::new(None),
            specs: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        })
    }

    pub fn set_notify(&self, cb: HostToolNotify) {
        *self.notify.lock().unwrap() = Some(cb);
    }

    pub fn clear_notify(&self) {
        *self.notify.lock().unwrap() = None;
    }

    /// Fire a host notification event without waiting for a result channel.
    pub fn notify_event(&self, name: &str, args_json: &str) {
        if let Some(cb) = self.notify.lock().unwrap().clone() {
            let call_id = Uuid::new_v4().to_string();
            cb(call_id, name.to_string(), args_json.to_string());
        }
    }

    /// Replace the full host tool set. Names in `skip` (built-in tools) are
    /// ignored so built-ins always win on collision.
    pub fn set_tools(&self, tools: Vec<ToolSpec>, skip: &[String]) {
        let skip: std::collections::HashSet<&str> = skip.iter().map(|s| s.as_str()).collect();
        let mut map = HashMap::new();
        for spec in tools {
            if skip.contains(spec.name.as_str()) {
                tracing::warn!(
                    "host tool '{}' skipped: conflicts with built-in tool",
                    spec.name
                );
                continue;
            }
            map.insert(spec.name.clone(), spec);
        }
        *self.specs.lock().unwrap() = map;
    }

    pub fn has(&self, name: &str) -> bool {
        self.specs.lock().unwrap().contains_key(name)
    }

    pub fn names(&self) -> Vec<String> {
        self.specs.lock().unwrap().keys().cloned().collect()
    }

    pub fn wire_definitions(&self) -> Vec<ToolDefinitionWire> {
        let specs = self.specs.lock().unwrap();
        specs
            .values()
            .map(|spec| ToolDefinitionWire {
                kind: "function".into(),
                function: FunctionDefinitionWire {
                    name: spec.name.clone(),
                    description: Some(spec.description.clone()),
                    parameters: spec.parameters.clone(),
                },
            })
            .collect()
    }

    /// Execute a host tool: notify and wait for [`complete`].
    pub async fn execute(
        &self,
        name: &str,
        args: Value,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> EngineResult<ToolOutput> {
        if !self.has(name) {
            return Err(EngineError::ToolNotFound(name.to_string()));
        }
        self.dispatch_host_call(name, args, cancel).await
    }

    /// Dispatch a call to the host (used for both custom host tools and built-in host UI interactions like `ask_user_question`).
    pub async fn dispatch_host_call(
        &self,
        name: &str,
        args: Value,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> EngineResult<ToolOutput> {
        let notify = self
            .notify
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| EngineError::Tool {
                name: name.to_string(),
                message: "host tool notify callback is not set".into(),
            })?;

        let call_id = Uuid::new_v4().to_string();
        let args_json = serde_json::to_string(&args).unwrap_or_else(|_| "{}".into());
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.lock().unwrap().insert(call_id.clone(), tx);

        notify(call_id.clone(), name.to_string(), args_json);

        tokio::select! {
            _ = cancel.cancelled() => {
                self.pending.lock().unwrap().remove(&call_id);
                Err(EngineError::Cancelled)
            }
            res = rx => {
                match res {
                    Ok(Ok(text)) => Ok(ToolOutput::new(text)),
                    Ok(Err(msg)) => Err(EngineError::Tool {
                        name: name.to_string(),
                        message: msg,
                    }),
                    Err(_) => Err(EngineError::Tool {
                        name: name.to_string(),
                        message: "host tool result channel closed".into(),
                    }),
                }
            }
        }
    }

    /// Host finished a tool call.
    pub fn complete(&self, call_id: &str, ok: bool, output: impl Into<String>) -> Result<(), String> {
        let mut pending = self.pending.lock().unwrap();
        let Some(tx) = pending.remove(call_id) else {
            return Err(format!("unknown host tool call_id: {call_id}"));
        };
        let output = output.into();
        let payload = if ok {
            Ok(output)
        } else {
            Err(output)
        };
        tx.send(payload)
            .map_err(|_| format!("host tool call {call_id} receiver dropped"))?;
        Ok(())
    }

    /// Parse OpenAI-style tool definition array into specs.
    pub fn parse_tool_defs(json: &str) -> Result<Vec<ToolSpec>, String> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| format!("invalid host tools JSON: {e}"))?;
        let arr = value
            .as_array()
            .ok_or_else(|| "host tools JSON must be an array".to_string())?;
        let mut out = Vec::new();
        for item in arr {
            let function = item
                .get("function")
                .or(Some(item))
                .ok_or_else(|| "tool entry missing function".to_string())?;
            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "tool entry missing name".to_string())?
                .to_string();
            let description = function
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
            out.push(ToolSpec {
                name,
                description,
                parameters,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn host_tool_round_trip() {
        let hub = HostToolHub::new();
        hub.set_tools(
            vec![ToolSpec {
                name: "calc".into(),
                description: "calc".into(),
                parameters: serde_json::json!({"type":"object"}),
            }],
            &[],
        );
        let hub_c = hub.clone();
        hub.set_notify(Arc::new(move |call_id, name, _args| {
            assert_eq!(name, "calc");
            let hub = hub_c.clone();
            std::thread::spawn(move || {
                let _ = hub.complete(&call_id, true, "42");
            });
        }));

        let cancel = tokio_util::sync::CancellationToken::new();
        let out = hub
            .execute("calc", serde_json::json!({"x": 1}), &cancel)
            .await
            .unwrap();
        assert_eq!(out.text, "42");
    }

    #[test]
    fn parse_openai_tool_defs() {
        let json = r#"[
          {"type":"function","function":{"name":"datetime","description":"now","parameters":{"type":"object"}}}
        ]"#;
        let specs = HostToolHub::parse_tool_defs(json).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "datetime");
    }

    #[test]
    fn skip_builtin_names() {
        let hub = HostToolHub::new();
        hub.set_tools(
            vec![ToolSpec {
                name: "read_file".into(),
                description: "host".into(),
                parameters: serde_json::json!({}),
            }],
            &["read_file".into()],
        );
        assert!(!hub.has("read_file"));
    }
}
