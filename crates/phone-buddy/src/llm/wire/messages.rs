//! Anthropic Messages API adapter.

use crate::conversation::{backend_call_summary, ConversationItem};
use crate::error::{EngineError, EngineResult};
use crate::llm::types::{
    ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ConversationRequest, Role, ToolCallDelta,
    ToolCallFunctionDelta, Usage,
};

use super::{args_as_object, WireAdapter};

pub struct MessagesAdapter;

impl WireAdapter for MessagesAdapter {
    fn endpoint(&self, base: &str, _model: &str, _stream: bool) -> String {
        format!("{base}/messages")
    }

    fn build_payload(&self, req: &ConversationRequest) -> EngineResult<serde_json::Value> {
        build_messages_payload(req)
    }

    fn parse_event(&self, event: &str, data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
        parse_messages_chunk(event, data)
    }

    fn parse_response(&self, data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
        parse_messages_response(data)
    }
}

pub fn build_messages_payload(req: &ConversationRequest) -> EngineResult<serde_json::Value> {
    // Anthropic Messages has no request-level structured-output field; the
    // documented pattern is a tool schema, which one-shot generation forbids.
    if req.response_format.as_ref().is_some_and(|f| !f.is_text()) {
        return Err(EngineError::ResponseFormatUnsupported {
            backend: "messages".into(),
        });
    }
    let mut system_text = String::new();
    let mut messages: Vec<serde_json::Value> = Vec::new();
    let mut pending_assistant: Vec<serde_json::Value> = Vec::new();
    let mut pending_tool_results: Vec<serde_json::Value> = Vec::new();

    let flush_assistant = |pending: &mut Vec<serde_json::Value>, msgs: &mut Vec<serde_json::Value>| {
        if !pending.is_empty() {
            msgs.push(serde_json::json!({
                "role": "assistant",
                "content": std::mem::take(pending)
            }));
        }
    };
    let flush_tools = |pending: &mut Vec<serde_json::Value>, msgs: &mut Vec<serde_json::Value>| {
        if !pending.is_empty() {
            msgs.push(serde_json::json!({
                "role": "user",
                "content": std::mem::take(pending)
            }));
        }
    };

    for item in &req.items {
        match item {
            ConversationItem::System(s) => {
                flush_assistant(&mut pending_assistant, &mut messages);
                flush_tools(&mut pending_tool_results, &mut messages);
                if !system_text.is_empty() {
                    system_text.push_str("\n\n");
                }
                system_text.push_str(&s.content);
            }
            ConversationItem::User(u) => {
                flush_assistant(&mut pending_assistant, &mut messages);
                flush_tools(&mut pending_tool_results, &mut messages);
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": super::messages_user_content(u, req)?
                }));
            }
            ConversationItem::Reasoning(r) => {
                flush_tools(&mut pending_tool_results, &mut messages);
                let text = crate::llm::types::reasoning_item_text(r);
                let sig = r.encrypted_content.clone().unwrap_or_default();
                if !text.is_empty() || !sig.is_empty() {
                    pending_assistant.push(serde_json::json!({
                        "type": "thinking",
                        "thinking": text,
                        "signature": sig
                    }));
                }
            }
            ConversationItem::BackendToolCall(b) => {
                flush_tools(&mut pending_tool_results, &mut messages);
                pending_assistant.push(serde_json::json!({
                    "type": "text",
                    "text": backend_call_summary(b)
                }));
            }
            ConversationItem::Assistant(a) => {
                flush_tools(&mut pending_tool_results, &mut messages);
                // Thinking block must precede tool_use (I1 order). If no
                // Reasoning sibling produced a thinking block but the
                // assistant still carries reasoning_content / signature,
                // emit one now.
                let has_thinking = pending_assistant
                    .iter()
                    .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("thinking"));
                if !has_thinking {
                    if let Some(ref reasoning) = a.reasoning_content {
                        if !reasoning.is_empty() || a.encrypted_reasoning.is_some() {
                            pending_assistant.push(serde_json::json!({
                                "type": "thinking",
                                "thinking": reasoning,
                                "signature": a.encrypted_reasoning.as_deref().unwrap_or("")
                            }));
                        }
                    } else if let Some(ref sig) = a.encrypted_reasoning {
                        if !sig.is_empty() {
                            pending_assistant.push(serde_json::json!({
                                "type": "thinking",
                                "thinking": "",
                                "signature": sig
                            }));
                        }
                    }
                }
                if !a.content.is_empty() {
                    pending_assistant.push(serde_json::json!({
                        "type": "text",
                        "text": a.content
                    }));
                }
                for tc in &a.tool_calls {
                    pending_assistant.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": args_as_object(&tc.function.arguments)
                    }));
                }
                flush_assistant(&mut pending_assistant, &mut messages);
            }
            ConversationItem::ToolResult(t) => {
                flush_assistant(&mut pending_assistant, &mut messages);
                pending_tool_results.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": t.tool_call_id,
                    "content": t.content
                }));
            }
        }
    }
    flush_assistant(&mut pending_assistant, &mut messages);
    flush_tools(&mut pending_tool_results, &mut messages);

    let mut payload = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens.unwrap_or(8192),
        "stream": req.stream.unwrap_or(true),
    });

    if !system_text.is_empty() {
        payload["system"] = serde_json::Value::String(system_text);
    }
    if let Some(temp) = req.temperature {
        payload["temperature"] = serde_json::json!(temp);
    }
    if let Some(ref tools) = req.tools {
        let tools_val: Vec<_> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.function.name,
                    "description": t.function.description,
                    "input_schema": t.function.parameters
                })
            })
            .collect();
        payload["tools"] = serde_json::Value::Array(tools_val);
    }

    if let Some(effort) = req
        .reasoning_effort
        .and_then(|e| e.to_messages_api_for_model(&req.model))
    {
        payload["output_config"] = serde_json::json!({
            "effort": effort
        });
        payload["thinking"] = serde_json::json!({
            "type": "adaptive"
        });
    }

    Ok(payload)
}

/// Parse one complete Anthropic Messages response into the internal chunk
/// model used by the ordinary collector.
pub fn parse_messages_response(data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
    let value: serde_json::Value = serde_json::from_str(data).map_err(|e| {
        EngineError::Stream(format!(
            "failed to parse buffered Messages response: {e}: {data:.120}"
        ))
    })?;
    let mut delta = ChatChunkDelta {
        role: Some(Role::Assistant),
        ..Default::default()
    };
    let mut text = String::new();
    let mut reasoning = String::new();

    if let Some(content) = value.get("content").and_then(|v| v.as_array()) {
        for (index, block) in content.iter().enumerate() {
            match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "text" => {
                    if let Some(part) = block.get("text").and_then(|v| v.as_str()) {
                        text.push_str(part);
                    }
                }
                "thinking" => {
                    if let Some(part) = block.get("thinking").and_then(|v| v.as_str()) {
                        reasoning.push_str(part);
                    }
                    if let Some(signature) = block.get("signature").and_then(|v| v.as_str()) {
                        if !signature.is_empty() {
                            delta.encrypted_reasoning = Some(signature.to_string());
                        }
                    }
                }
                "tool_use" => {
                    delta.tool_calls.push(ToolCallDelta {
                        index: index as u32,
                        id: block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string),
                        kind: Some("function".into()),
                        function: Some(ToolCallFunctionDelta {
                            name: block
                                .get("name")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            arguments: Some(
                                block
                                    .get("input")
                                    .cloned()
                                    .unwrap_or_else(|| serde_json::json!({}))
                                    .to_string(),
                            ),
                        }),
                        ..Default::default()
                    });
                }
                _ => {}
            }
        }
    }
    if !text.is_empty() {
        delta.content = Some(text);
    }
    if !reasoning.is_empty() {
        delta.reasoning_content = Some(reasoning);
    }

    let usage = value.get("usage").map(|usage| {
        let prompt_tokens = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let completion_tokens = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    });
    let finish_reason = value
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .map(str::to_string);

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
        id: value
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        object: value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message")
            .to_string(),
        created: 0,
        model: value
            .get("model")
            .and_then(|v| v.as_str())
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

pub fn parse_messages_chunk(
    event_name: &str,
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

    let event_type = if !event_name.is_empty() {
        event_name
    } else {
        v.get("type").and_then(|s| s.as_str()).unwrap_or("")
    };

    let mut delta = ChatChunkDelta::default();
    let mut usage = None;
    let mut finish_reason = None;

    match event_type {
        "message_start" => {
            if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
                usage = Some(Usage {
                    prompt_tokens: u
                        .get("input_tokens")
                        .and_then(|n| n.as_u64())
                        .unwrap_or(0) as u32,
                    completion_tokens: u
                        .get("output_tokens")
                        .and_then(|n| n.as_u64())
                        .unwrap_or(0) as u32,
                    total_tokens: 0,
                });
            }
        }
        "content_block_start" => {
            if let Some(block) = v.get("content_block") {
                let btype = block.get("type").and_then(|s| s.as_str()).unwrap_or("");
                let index = v.get("index").and_then(|n| n.as_u64()).unwrap_or(0) as u32;
                if btype == "tool_use" {
                    let id = block
                        .get("id")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    let name = block
                        .get("name")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    delta.tool_calls.push(ToolCallDelta {
                        index,
                        id,
                        kind: Some("function".to_string()),
                        function: Some(ToolCallFunctionDelta {
                            name,
                            arguments: Some(String::new()),
                        }),
                        ..Default::default()
                    });
                }
            }
        }
        "content_block_delta" => {
            let index = v.get("index").and_then(|n| n.as_u64()).unwrap_or(0) as u32;
            if let Some(d) = v.get("delta") {
                let dtype = d.get("type").and_then(|s| s.as_str()).unwrap_or("");
                match dtype {
                    "text_delta" => {
                        if let Some(text) = d.get("text").and_then(|s| s.as_str()) {
                            delta.content = Some(text.to_string());
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial) = d.get("partial_json").and_then(|s| s.as_str()) {
                            delta.tool_calls.push(ToolCallDelta {
                                index,
                                id: None,
                                kind: None,
                                function: Some(ToolCallFunctionDelta {
                                    name: None,
                                    arguments: Some(partial.to_string()),
                                }),
                                ..Default::default()
                            });
                        }
                    }
                    "thinking_delta" => {
                        if let Some(thinking) = d.get("thinking").and_then(|s| s.as_str()) {
                            delta.reasoning_content = Some(thinking.to_string());
                        }
                    }
                    "signature_delta" => {
                        if let Some(sig) = d.get("signature").and_then(|s| s.as_str()) {
                            delta.encrypted_reasoning = Some(sig.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        "message_delta" => {
            if let Some(u) = v.get("usage") {
                usage = Some(Usage {
                    prompt_tokens: 0,
                    completion_tokens: u
                        .get("output_tokens")
                        .and_then(|n| n.as_u64())
                        .unwrap_or(0) as u32,
                    total_tokens: 0,
                });
            }
            if let Some(d) = v.get("delta") {
                if let Some(sr) = d.get("stop_reason").and_then(|s| s.as_str()) {
                    finish_reason = Some(sr.to_string());
                }
            }
        }
        _ => {}
    }

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
            .get("id")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 0,
        model: String::new(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
    }))
}
