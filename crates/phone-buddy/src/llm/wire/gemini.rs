//! Google Gemini generateContent / streamGenerateContent adapter.

use crate::conversation::{backend_call_summary, function_name_for_call, ConversationItem};
use crate::error::EngineResult;
use crate::llm::types::{
    ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ConversationRequest, ToolCallDelta,
    ToolCallFunctionDelta, Usage,
};

use super::{args_as_object, sanitize_json_schema_for_gemini, WireAdapter};

/// `generationConfig.responseMimeType` value that constrains output to JSON.
const JSON_MIME: &str = "application/json";

pub struct GeminiAdapter;

impl WireAdapter for GeminiAdapter {
    fn endpoint(&self, base: &str, model: &str, stream: bool) -> String {
        let method = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let mut url = format!("{base}/models/{model}:{method}");
        if stream {
            url.push_str("?alt=sse");
        }
        url
    }

    fn build_payload(&self, req: &ConversationRequest) -> EngineResult<serde_json::Value> {
        build_gemini_payload(req)
    }

    fn parse_event(&self, event: &str, data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
        parse_gemini_chunk(event, data)
    }
}

pub fn build_gemini_payload(req: &ConversationRequest) -> EngineResult<serde_json::Value> {
    let mut system_parts: Vec<serde_json::Value> = Vec::new();
    let mut contents: Vec<serde_json::Value> = Vec::new();
    let mut pending_model_parts: Vec<serde_json::Value> = Vec::new();
    let mut pending_user_parts: Vec<serde_json::Value> = Vec::new();

    let flush_model = |pending: &mut Vec<serde_json::Value>, out: &mut Vec<serde_json::Value>| {
        if !pending.is_empty() {
            out.push(serde_json::json!({
                "role": "model",
                "parts": std::mem::take(pending)
            }));
        }
    };
    let flush_user = |pending: &mut Vec<serde_json::Value>, out: &mut Vec<serde_json::Value>| {
        if !pending.is_empty() {
            out.push(serde_json::json!({
                "role": "user",
                "parts": std::mem::take(pending)
            }));
        }
    };

    for item in &req.items {
        match item {
            ConversationItem::System(s) => {
                system_parts.push(serde_json::json!({ "text": s.content }));
            }
            ConversationItem::User(u) => {
                flush_model(&mut pending_model_parts, &mut contents);
                flush_user(&mut pending_user_parts, &mut contents);
                contents.push(serde_json::json!({
                    "role": "user",
                    "parts": super::gemini_user_parts(u, req)?
                }));
            }
            ConversationItem::Reasoning(_) => {
                // Gemini signatures ride `thoughtSignature` on functionCall
                // parts, not a sibling reasoning item.
            }
            ConversationItem::BackendToolCall(b) => {
                flush_user(&mut pending_user_parts, &mut contents);
                pending_model_parts.push(serde_json::json!({
                    "text": backend_call_summary(b)
                }));
            }
            ConversationItem::Assistant(a) => {
                flush_user(&mut pending_user_parts, &mut contents);
                if !a.content.is_empty() {
                    pending_model_parts.push(serde_json::json!({ "text": a.content }));
                }
                for tc in &a.tool_calls {
                    let mut part = serde_json::json!({
                        "functionCall": {
                            "name": tc.function.name,
                            "args": args_as_object(&tc.function.arguments)
                        }
                    });
                    if let Some(ref sig) = tc.thought_signature {
                        if !sig.is_empty() {
                            part["thoughtSignature"] = serde_json::Value::String(sig.clone());
                        }
                    }
                    pending_model_parts.push(part);
                }
                flush_model(&mut pending_model_parts, &mut contents);
            }
            ConversationItem::ToolResult(t) => {
                flush_model(&mut pending_model_parts, &mut contents);
                let name = function_name_for_call(&req.items, &t.tool_call_id)
                    .unwrap_or("unknown")
                    .to_string();
                let response = match serde_json::from_str::<serde_json::Value>(&t.content) {
                    Ok(serde_json::Value::Object(_)) => {
                        serde_json::from_str(&t.content).unwrap_or_else(|_| {
                            serde_json::json!({ "result": t.content })
                        })
                    }
                    Ok(other) => serde_json::json!({ "result": other }),
                    Err(_) => serde_json::json!({ "result": t.content }),
                };
                pending_user_parts.push(serde_json::json!({
                    "functionResponse": {
                        "name": name,
                        "response": response
                    }
                }));
            }
        }
    }
    flush_model(&mut pending_model_parts, &mut contents);
    flush_user(&mut pending_user_parts, &mut contents);

    let mut generation_config = serde_json::Map::new();
    if let Some(temp) = req.temperature {
        generation_config.insert("temperature".into(), serde_json::json!(temp));
    }
    if let Some(max) = req.max_tokens {
        generation_config.insert("maxOutputTokens".into(), serde_json::json!(max));
    }
    generation_config.insert(
        "thinkingConfig".into(),
        serde_json::json!({ "includeThoughts": true }),
    );
    match req.response_format.as_ref() {
        None | Some(crate::llm::types::ResponseFormat::Text) => {}
        Some(crate::llm::types::ResponseFormat::JsonObject) => {
            generation_config.insert("responseMimeType".into(), JSON_MIME.into());
        }
        Some(crate::llm::types::ResponseFormat::JsonSchema { schema, .. }) => {
            generation_config.insert("responseMimeType".into(), JSON_MIME.into());
            generation_config.insert(
                "responseSchema".into(),
                sanitize_json_schema_for_gemini(schema.clone()),
            );
        }
    }

    let mut payload = serde_json::json!({
        "contents": contents,
        "generationConfig": generation_config,
    });

    if !system_parts.is_empty() {
        payload["systemInstruction"] = serde_json::json!({ "parts": system_parts });
    }

    if let Some(ref tools) = req.tools {
        let decls: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": sanitize_json_schema_for_gemini(t.function.parameters.clone())
                })
            })
            .collect();
        if !decls.is_empty() {
            payload["tools"] = serde_json::json!([{ "functionDeclarations": decls }]);
            payload["toolConfig"] = serde_json::json!({
                "functionCallingConfig": { "mode": "AUTO" }
            });
        }
    }

    if req.image_bytes.total_bytes() > crate::llm::image::GEMINI_INLINE_MAX_BYTES {
        return Err(crate::error::EngineError::PayloadTooLarge);
    }

    Ok(payload)
}

pub fn parse_gemini_chunk(
    _event_name: &str,
    data: &str,
) -> EngineResult<Option<ChatCompletionChunk>> {
    let raw = data.trim();
    if raw.is_empty() || raw == "[DONE]" {
        return Ok(None);
    }

    let v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(val) => val,
        Err(_) => return Ok(None),
    };

    let mut delta = ChatChunkDelta::default();
    let mut finish_reason = None;

    if let Some(cand) = v
        .get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    {
        if let Some(reason) = cand.get("finishReason").and_then(|s| s.as_str()) {
            finish_reason = Some(map_gemini_finish(reason, cand));
        }
        if let Some(parts) = cand
            .pointer("/content/parts")
            .and_then(|p| p.as_array())
        {
            for (idx, part) in parts.iter().enumerate() {
                let thought = part
                    .get("thought")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);
                if let Some(text) = part.get("text").and_then(|s| s.as_str()) {
                    if thought {
                        let mut existing = delta.reasoning_content.take().unwrap_or_default();
                        existing.push_str(text);
                        delta.reasoning_content = Some(existing);
                    } else {
                        let mut existing = delta.content.take().unwrap_or_default();
                        existing.push_str(text);
                        delta.content = Some(existing);
                    }
                }
                if let Some(fc) = part.get("functionCall") {
                    let name = fc
                        .get("name")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = fc
                        .get("args")
                        .map(|a| a.to_string())
                        .unwrap_or_else(|| "{}".into());
                    let thought_signature = part
                        .get("thoughtSignature")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    if thought_signature.is_some() {
                        delta.encrypted_reasoning = thought_signature.clone();
                    }
                    delta.tool_calls.push(ToolCallDelta {
                        index: idx as u32,
                        id: Some(format!("call_{idx}")),
                        kind: Some("function".into()),
                        function: Some(ToolCallFunctionDelta {
                            name: Some(name),
                            arguments: Some(args),
                        }),
                        thought_signature,
                        ..Default::default()
                    });
                }
            }
        }
    }

    let usage = v.get("usageMetadata").map(|u| Usage {
        prompt_tokens: u
            .get("promptTokenCount")
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32,
        completion_tokens: u
            .get("candidatesTokenCount")
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32,
        total_tokens: u
            .get("totalTokenCount")
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32,
    });

    if delta.content.is_none()
        && delta.reasoning_content.is_none()
        && delta.encrypted_reasoning.is_none()
        && delta.tool_calls.is_empty()
        && usage.is_none()
        && finish_reason.is_none()
    {
        return Ok(None);
    }

    Ok(Some(ChatCompletionChunk {
        id: v
            .get("responseId")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        object: "gemini.chunk".to_string(),
        created: 0,
        model: v
            .get("modelVersion")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
    }))
}

fn map_gemini_finish(reason: &str, cand: &serde_json::Value) -> String {
    let has_calls = cand
        .pointer("/content/parts")
        .and_then(|p| p.as_array())
        .map(|a| a.iter().any(|p| p.get("functionCall").is_some()))
        .unwrap_or(false);
    if has_calls {
        return "tool_calls".into();
    }
    match reason {
        "STOP" => "stop".into(),
        "MAX_TOKENS" => "length".into(),
        other => other.to_ascii_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{AssistantItem, BackendToolCallItem, ToolResultItem};
    use crate::llm::types::{
        FunctionDefinitionWire, ToolCall, ToolCallFunction, ToolDefinitionWire,
    };

    fn req_with(items: Vec<ConversationItem>) -> ConversationRequest {
        ConversationRequest {
            model: "gemini-2.5-pro".into(),
            items,
            stream: Some(true),
            tools: Some(vec![ToolDefinitionWire {
                kind: "function".into(),
                function: FunctionDefinitionWire {
                    name: "read_file".into(),
                    description: Some("read".into()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "$schema": "x",
                        "additionalProperties": false,
                        "properties": { "path": { "type": "string" } }
                    }),
                },
            }]),
            tool_choice: None,
            temperature: Some(0.2),
            max_tokens: Some(1024),
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
            response_format: None,
            image_bytes: crate::llm::image::ImageBytesStore::default(),
            audio_bytes: crate::llm::image::AudioBytesStore::default(),
        }
    }

    #[test]
    fn gemini_args_are_objects() {
        let req = req_with(vec![
            ConversationItem::user("fetch"),
            ConversationItem::Assistant(AssistantItem {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    kind: "function".into(),
                    function: ToolCallFunction {
                        name: "read_file".into(),
                        arguments: r#"{"url":"https://ex"}"#.into(),
                    },
                    thought_signature: Some("sig-1".into()),
                }],
                reasoning_content: None,
                encrypted_reasoning: None,
                origin: None,
            }),
        ]);
        let payload = build_gemini_payload(&req).unwrap();
        let parts = &payload["contents"][1]["parts"];
        assert_eq!(parts[0]["functionCall"]["args"]["url"], "https://ex");
        assert!(parts[0]["functionCall"]["args"].is_object());
        assert_eq!(parts[0]["thoughtSignature"], "sig-1");

        let mut bad = req;
        if let ConversationItem::Assistant(a) = &mut bad.items[1] {
            a.tool_calls[0].function.arguments = "not-json".into();
        }
        let payload = build_gemini_payload(&bad).unwrap();
        let args = &payload["contents"][1]["parts"][0]["functionCall"]["args"];
        assert_eq!(args, &serde_json::json!({}));
    }

    #[test]
    fn gemini_sse_text_thought_and_call() {
        let data = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "planning", "thought": true},
                        {"text": "hello"},
                        {
                            "functionCall": {"name": "read_file", "args": {"path": "a.txt"}},
                            "thoughtSignature": "sig-abc"
                        }
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 7,
                "totalTokenCount": 18
            }
        }"#;
        let chunk = parse_gemini_chunk("", data).unwrap().unwrap();
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("planning")
        );
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert_eq!(chunk.choices[0].delta.tool_calls.len(), 1);
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(
            tc.function.as_ref().and_then(|f| f.name.as_deref()),
            Some("read_file")
        );
        let args = tc.function.as_ref().and_then(|f| f.arguments.as_deref());
        assert!(args.unwrap().contains("a.txt"));
        assert_eq!(
            chunk.choices[0].delta.encrypted_reasoning.as_deref(),
            Some("sig-abc")
        );
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        let u = chunk.usage.unwrap();
        assert_eq!(u.prompt_tokens, 11);
        assert_eq!(u.completion_tokens, 7);
        assert_eq!(u.total_tokens, 18);
    }

    #[test]
    fn gemini_function_response_role_is_user() {
        let req = req_with(vec![
            ConversationItem::user("go"),
            ConversationItem::Assistant(AssistantItem {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    kind: "function".into(),
                    function: ToolCallFunction {
                        name: "read_file".into(),
                        arguments: r#"{"path":"a"}"#.into(),
                    },
                    thought_signature: None,
                }],
                reasoning_content: None,
                encrypted_reasoning: None,
                origin: None,
            }),
            ConversationItem::ToolResult(ToolResultItem {
                tool_call_id: "c1".into(),
                content: "file contents".into(),
            }),
        ]);
        let payload = build_gemini_payload(&req).unwrap();
        let last = payload["contents"].as_array().unwrap().last().unwrap();
        assert_eq!(last["role"], "user");
        assert_eq!(
            last["parts"][0]["functionResponse"]["name"],
            "read_file"
        );
    }

    #[test]
    fn backend_call_degrades_to_model_text() {
        let req = req_with(vec![
            ConversationItem::user("search"),
            ConversationItem::BackendToolCall(BackendToolCallItem {
                item_type: "web_search_call".into(),
                id: "ws_1".into(),
                payload: serde_json::json!({"type":"web_search_call","id":"ws_1"}),
            }),
            ConversationItem::assistant("done"),
        ]);
        let payload = build_gemini_payload(&req).unwrap();
        let model = &payload["contents"][1];
        assert_eq!(model["role"], "model");
        let text = model["parts"][0]["text"].as_str().unwrap();
        assert!(text.contains("web_search_call"));
        assert!(!model["parts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p.get("functionCall").is_some()));
    }
}
