//! Down-conversion adapters: canonical items → wire protocol JSON.
//!
//! Adapters are the only code that produces provider payloads (I5).

pub mod chat_completions;
pub mod gemini;
pub mod messages;
pub mod responses;

#[cfg(test)]
mod conformance_tests;

use crate::error::EngineResult;
use crate::llm::types::{ChatCompletionChunk, ConversationRequest};

/// One protocol's request builder + SSE parser.
pub(crate) trait WireAdapter {
    fn endpoint(&self, base: &str, model: &str, stream: bool) -> String;
    fn build_payload(&self, req: &ConversationRequest) -> serde_json::Value;
    /// One SSE frame → zero or one internal chunk (unchanged chunk model).
    fn parse_event(&self, event: &str, data: &str) -> EngineResult<Option<ChatCompletionChunk>>;
}

/// Strip JSON-Schema keywords Gemini's OpenAPI subset rejects, and map
/// `oneOf` to `anyOf`. Lossless for the supported subset.
pub fn sanitize_json_schema_for_gemini(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(mut map) => {
            map.remove("additionalProperties");
            map.remove("$schema");
            map.remove("$id");
            map.remove("$comment");
            if let Some(one_of) = map.remove("oneOf") {
                map.entry("anyOf").or_insert(one_of);
            }
            let keys: Vec<String> = map.keys().cloned().collect();
            for k in keys {
                if let Some(v) = map.remove(&k) {
                    map.insert(k, sanitize_json_schema_for_gemini(v));
                }
            }
            serde_json::Value::Object(map)
        }
        serde_json::Value::Array(arr) => serde_json::Value::Array(
            arr.into_iter()
                .map(sanitize_json_schema_for_gemini)
                .collect(),
        ),
        other => other,
    }
}

/// Parse stored tool-call argument JSON into an object. Invalid or
/// non-object JSON degrades to `{}`.
pub fn args_as_object(arguments: &str) -> serde_json::Value {
    let parsed = serde_json::from_str::<serde_json::Value>(arguments).ok();
    match parsed {
        Some(serde_json::Value::Object(_)) => parsed.unwrap(),
        _ => serde_json::json!({}),
    }
}

pub(crate) fn adapter_for(backend: crate::llm::types::ApiBackend) -> Box<dyn WireAdapter + Send + Sync> {
    match backend {
        crate::llm::types::ApiBackend::ChatCompletions => {
            Box::new(chat_completions::ChatCompletionsAdapter)
        }
        crate::llm::types::ApiBackend::Responses => Box::new(responses::ResponsesAdapter),
        crate::llm::types::ApiBackend::Messages => Box::new(messages::MessagesAdapter),
        crate::llm::types::ApiBackend::Gemini => Box::new(gemini::GeminiAdapter),
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    #[test]
    fn gemini_schema_sanitizer() {
        let raw = serde_json::json!({
            "$schema": "https://json-schema.org/draft/07/schema#",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "number"}
                        ],
                        "additionalProperties": true
                    }
                }
            },
            "required": ["items"]
        });
        let out = sanitize_json_schema_for_gemini(raw);
        assert!(out.get("$schema").is_none());
        assert!(out.get("additionalProperties").is_none());
        assert_eq!(out["type"], "object");
        assert_eq!(out["required"], serde_json::json!(["items"]));
        let items_schema = &out["properties"]["items"]["items"];
        assert!(items_schema.get("oneOf").is_none());
        assert!(items_schema.get("anyOf").is_some());
        assert!(items_schema.get("additionalProperties").is_none());
        assert_eq!(items_schema["anyOf"][0]["type"], "string");
    }
}
