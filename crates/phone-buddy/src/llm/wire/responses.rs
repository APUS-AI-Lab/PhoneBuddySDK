//! OpenAI / xAI Responses API adapter.

use crate::conversation::ConversationItem;
use crate::error::EngineResult;
use crate::llm::types::{
    parse_output_item, sanitize_tool_arguments, ChatChunkChoice, ChatChunkDelta,
    ChatCompletionChunk, ConversationRequest, OutputItemWire, ReasoningItem, ToolCallDelta,
    ToolCallFunctionDelta, Usage,
};

use super::WireAdapter;

pub struct ResponsesAdapter;

impl WireAdapter for ResponsesAdapter {
    fn endpoint(&self, base: &str, _model: &str, _stream: bool) -> String {
        format!("{base}/responses")
    }

    fn build_payload(&self, req: &ConversationRequest) -> EngineResult<serde_json::Value> {
        build_responses_payload(req)
    }

    fn parse_event(&self, event: &str, data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
        parse_responses_chunk(event, data)
    }
}

/// Inject the `type: "reasoning_text"` discriminator the API requires.
pub fn patch_reasoning_text_types(body: &mut serde_json::Value) {
    let Some(input) = body.get_mut("input").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for item in input.iter_mut() {
        if item.get("type").and_then(|t| t.as_str()) != Some("reasoning") {
            continue;
        }
        let Some(content) = item.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for c in content.iter_mut() {
            if let Some(obj) = c.as_object_mut() {
                obj.entry("type")
                    .or_insert_with(|| serde_json::Value::String("reasoning_text".into()));
            }
        }
    }
}

pub fn build_responses_payload(req: &ConversationRequest) -> EngineResult<serde_json::Value> {
    let mut instructions = String::new();
    let mut input = Vec::new();

    for item in &req.items {
        match item {
            ConversationItem::System(s) => {
                if !instructions.is_empty() {
                    instructions.push_str("\n\n");
                }
                instructions.push_str(&s.content);
            }
            ConversationItem::User(u) => {
                input.push(serde_json::json!({
                    "role": "user",
                    "content": super::responses_user_content(u, req)?
                }));
            }
            ConversationItem::Reasoning(r) => {
                input.push(reasoning_to_input_item(r));
            }
            ConversationItem::BackendToolCall(b) => {
                // Verbatim raw payload (I3). Never rewrite as function_call.
                input.push(b.payload.clone());
            }
            ConversationItem::Assistant(a) => {
                if !a.content.is_empty() {
                    input.push(serde_json::json!({
                        "role": "assistant",
                        "content": a.content
                    }));
                }
                for tc in &a.tool_calls {
                    let arguments =
                        sanitize_tool_arguments(&tc.id, &tc.function.name, &tc.function.arguments);
                    input.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments
                    }));
                }
            }
            ConversationItem::ToolResult(t) => {
                input.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": t.tool_call_id,
                    "output": t.content
                }));
            }
        }
    }

    let mut payload = serde_json::json!({
        "model": req.model,
        "input": input,
        "stream": true,
        "store": false,
        "include": ["reasoning.encrypted_content"],
        "reasoning": { "summary": "concise" },
    });
    if let Some(ref id) = req.previous_response_id {
        if !id.is_empty() {
            payload["previous_response_id"] = serde_json::Value::String(id.clone());
        }
    }

    patch_reasoning_text_types(&mut payload);

    if !instructions.is_empty() {
        payload["instructions"] = serde_json::Value::String(instructions);
    }
    if let Some(temp) = req.temperature {
        payload["temperature"] = serde_json::json!(temp);
    }
    if let Some(max_tokens) = req.max_tokens {
        payload["max_output_tokens"] = serde_json::json!(max_tokens);
    }
    let mut tools_val: Vec<serde_json::Value> = req
        .hosted_tools
        .iter()
        .map(|h| h.to_tool_entry())
        .collect();
    if let Some(ref tools) = req.tools {
        for t in tools {
            if req
                .hosted_tools
                .iter()
                .any(|h| h.wire_name() == t.function.name)
            {
                continue;
            }
            tools_val.push(serde_json::json!({
                "type": "function",
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters
            }));
        }
    }
    if !tools_val.is_empty() {
        payload["tools"] = serde_json::Value::Array(tools_val);
    }

    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{AssistantItem, BackendToolCallItem, ConversationItem};
    use crate::llm::types::{ReasoningItem, ToolCall, ToolCallFunction};

    fn req(items: Vec<ConversationItem>) -> ConversationRequest {
        ConversationRequest {
            model: "grok-4.6".into(),
            items,
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
            image_bytes: crate::llm::image::ImageBytesStore::default(),
        }
    }

    #[test]
    fn responses_payload_replays_backend_call_verbatim() {
        let payload_item = serde_json::json!({
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "apus"}
        });
        let body = build_responses_payload(&req(vec![
            ConversationItem::user("q"),
            ConversationItem::BackendToolCall(BackendToolCallItem {
                item_type: "web_search_call".into(),
                id: "ws_1".into(),
                payload: payload_item.clone(),
            }),
            ConversationItem::Assistant(AssistantItem {
                content: "ok".into(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    kind: "function".into(),
                    function: ToolCallFunction {
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                    thought_signature: None,
                }],
                reasoning_content: None,
                encrypted_reasoning: None,
                origin: None,
            }),
            ConversationItem::tool_result("c1", "out"),
        ])).unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[1], payload_item);
        assert_ne!(input[1]["type"], "function_call");
        let mut calls = Vec::new();
        let mut outputs = Vec::new();
        for item in input {
            if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                calls.push(item["call_id"].as_str().unwrap());
            }
            if item.get("type").and_then(|t| t.as_str()) == Some("function_call_output") {
                outputs.push(item["call_id"].as_str().unwrap());
            }
        }
        assert_eq!(calls, outputs);
    }

    #[test]
    fn responses_payload_sets_store_and_include() {
        let body = build_responses_payload(&req(vec![ConversationItem::user("hi")])).unwrap();
        assert_eq!(body["store"], false);
        assert!(body["include"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "reasoning.encrypted_content"));
    }

    #[test]
    fn responses_payload_preserves_sibling_order() {
        let r1 = ReasoningItem {
            id: "r1".into(),
            summary: Vec::new(),
            content: None,
            encrypted_content: None,
            status: None,
        };
        let r2 = ReasoningItem {
            id: "r2".into(),
            summary: Vec::new(),
            content: None,
            encrypted_content: None,
            status: None,
        };
        let ws = serde_json::json!({"type":"web_search_call","id":"ws1"});
        let body = build_responses_payload(&req(vec![
            ConversationItem::Reasoning(r1),
            ConversationItem::BackendToolCall(BackendToolCallItem {
                item_type: "web_search_call".into(),
                id: "ws1".into(),
                payload: ws,
            }),
            ConversationItem::Reasoning(r2),
            ConversationItem::Assistant(AssistantItem {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "fc1".into(),
                    kind: "function".into(),
                    function: ToolCallFunction {
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                    thought_signature: None,
                }],
                reasoning_content: None,
                encrypted_reasoning: None,
                origin: None,
            }),
        ])).unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[0]["id"], "r1");
        assert_eq!(input[1]["type"], "web_search_call");
        assert_eq!(input[2]["type"], "reasoning");
        assert_eq!(input[2]["id"], "r2");
        assert_eq!(input[3]["type"], "function_call");
        assert_eq!(input[3]["call_id"], "fc1");
    }
}

fn reasoning_to_input_item(r: &ReasoningItem) -> serde_json::Value {
    let mut r_val = serde_json::to_value(r).unwrap_or_default();
    if let Some(obj) = r_val.as_object_mut() {
        obj.remove("status");
    }
    let mut item_obj = serde_json::json!({
        "type": "reasoning",
        "summary": r_val
            .get("summary")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([])),
    });
    if let Some(content) = r_val.get("content") {
        if !content.is_null() {
            item_obj["content"] = content.clone();
        }
    }
    if let Some(enc) = r_val.get("encrypted_content") {
        if !enc.is_null() {
            item_obj["encrypted_content"] = enc.clone();
        }
    }
    if let Some(id) = r_val.get("id").and_then(|s| s.as_str()) {
        if !id.is_empty() {
            item_obj["id"] = serde_json::Value::String(id.to_string());
        }
    }
    item_obj
}

fn output_index_of(v: &serde_json::Value) -> u32 {
    v.get("output_index")
        .or_else(|| v.get("index"))
        .and_then(|n| n.as_u64())
        .unwrap_or(0) as u32
}

fn tool_delta_from_output_item(
    item: &serde_json::Value,
    fallback_index: u32,
) -> Option<ToolCallDelta> {
    let item_type = item.get("type").and_then(|s| s.as_str()).unwrap_or("");
    match item_type {
        "function_call" => {
            let id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let name = item
                .get("name")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let args = item
                .get("arguments")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            Some(ToolCallDelta {
                index: fallback_index,
                id,
                kind: Some("function".to_string()),
                function: Some(ToolCallFunctionDelta {
                    name,
                    arguments: args,
                }),
                thought_signature: None,
            })
        }
        "web_search_call" | "file_search_call" | "computer_call" | "mcp_call"
        | "image_generation_call" | "code_interpreter_call" => {
            let id = item
                .get("id")
                .or_else(|| item.get("call_id"))
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let name = match item_type {
                "web_search_call" => "web_search",
                "file_search_call" => "file_search",
                "computer_call" => "computer",
                "mcp_call" => item.get("name").and_then(|s| s.as_str()).unwrap_or("mcp"),
                "image_generation_call" => "image_generation",
                "code_interpreter_call" => "code_interpreter",
                other => other,
            }
            .to_string();
            let args = item.get("action").or_else(|| item.get("arguments")).map(|a| {
                if let Some(s) = a.as_str() {
                    s.to_string()
                } else {
                    a.to_string()
                }
            });
            Some(ToolCallDelta {
                index: fallback_index,
                id,
                kind: Some("server".to_string()),
                function: Some(ToolCallFunctionDelta {
                    name: Some(name),
                    arguments: args,
                }),
                thought_signature: None,
            })
        }
        _ => None,
    }
}

pub fn parse_responses_chunk(
    event_name: &str,
    data: &str,
) -> EngineResult<Option<ChatCompletionChunk>> {
    let raw = data.trim();
    if raw.is_empty() || raw == "[DONE]" {
        return Ok(None);
    }

    if let Ok(Some(chunk)) = crate::llm::stream::parse_chunk(raw) {
        if !chunk.choices.is_empty() {
            return Ok(Some(chunk));
        }
    }

    let v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(val) => val,
        Err(_) => return Ok(None),
    };

    let mut delta = ChatChunkDelta::default();

    let type_str = event_name.to_lowercase();
    let json_type = v.get("type").and_then(|s| s.as_str()).unwrap_or("");

    let is_completed = type_str.contains("response.completed")
        || json_type.contains("response.completed")
        || type_str.contains("response.incomplete")
        || json_type.contains("response.incomplete");

    if is_completed {
        let output = v
            .pointer("/response/output")
            .or_else(|| v.get("output"))
            .and_then(|o| o.as_array());
        if let Some(output) = output {
            let parsed: Vec<OutputItemWire> = output.iter().filter_map(parse_output_item).collect();
            if !parsed.is_empty() {
                delta.final_output = Some(parsed);
            }
        }
    } else if type_str.contains("reasoning_summary_text.delta")
        || json_type.contains("reasoning_summary_text.delta")
    {
        if let Some(text) = v.get("delta").and_then(|s| s.as_str()) {
            delta.reasoning_content = Some(text.to_string());
        }
    } else if type_str.contains("reasoning_text.delta") || json_type.contains("reasoning_text.delta")
    {
        if let Some(text) = v.get("delta").and_then(|s| s.as_str()) {
            delta.reasoning_content = Some(text.to_string());
        }
    } else if type_str.contains("output_text.delta")
        || json_type.contains("output_text.delta")
        || type_str.contains("text.delta")
        || json_type.contains("text.delta")
    {
        if let Some(text) = v.get("delta").and_then(|s| s.as_str()) {
            delta.content = Some(text.to_string());
        }
    } else if type_str.contains("function_call_arguments.delta")
        || json_type.contains("function_call_arguments.delta")
    {
        let id = v
            .get("call_id")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());
        let name = v.get("name").and_then(|s| s.as_str()).map(|s| s.to_string());
        let args = v
            .get("delta")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());
        delta.tool_calls.push(ToolCallDelta {
            index: output_index_of(&v),
            id,
            kind: Some("function".to_string()),
            function: Some(ToolCallFunctionDelta {
                name,
                arguments: args,
            }),
            thought_signature: None,
        });
    } else if let Some(item) = v.get("item").or_else(|| v.get("reasoning")) {
        let item_type = item.get("type").and_then(|s| s.as_str()).unwrap_or("");
        if item_type == "reasoning" {
            let is_added = type_str.contains("output_item.added")
                || json_type.contains("output_item.added");
            if !is_added {
                if let Ok(ri) = serde_json::from_value::<ReasoningItem>(item.clone()) {
                    delta.encrypted_reasoning = ri.encrypted_content.clone();
                    delta.reasoning_items.push(ri);
                }
            }
        } else if let Some(mut tc) = tool_delta_from_output_item(item, output_index_of(&v)) {
            let is_added = type_str.contains("output_item.added")
                || json_type.contains("output_item.added");
            if is_added {
                if let Some(f) = tc.function.as_mut() {
                    f.arguments = None;
                }
            }
            delta.tool_calls.push(tc);
        }
    } else if let Some(output) = v
        .get("output")
        .or_else(|| v.get("response").and_then(|r| r.get("output")))
        .and_then(|o| o.as_array())
    {
        for item in output {
            if item.get("type").and_then(|s| s.as_str()) == Some("reasoning") {
                if let Ok(ri) = serde_json::from_value::<ReasoningItem>(item.clone()) {
                    delta.encrypted_reasoning = ri.encrypted_content.clone();
                    delta.reasoning_items.push(ri);
                }
            } else if let Some(text) = item.get("content").and_then(|c| c.as_str()) {
                delta.content = Some(text.to_string());
            }
        }
    } else if let Some(d) = v.get("delta").and_then(|s| s.as_str()) {
        delta.content = Some(d.to_string());
    }

    let usage = v.get("usage").or_else(|| v.pointer("/response/usage")).map(|u| Usage {
        prompt_tokens: u
            .get("input_tokens")
            .or_else(|| u.get("prompt_tokens"))
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32,
        completion_tokens: u
            .get("output_tokens")
            .or_else(|| u.get("completion_tokens"))
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32,
        total_tokens: u
            .get("total_tokens")
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32,
    });

    let response_id = v
        .pointer("/response/id")
        .or_else(|| v.get("id"))
        .and_then(|s| s.as_str())
        .filter(|s| s.starts_with("resp_"))
        .unwrap_or("");

    if delta.content.is_none()
        && delta.reasoning_content.is_none()
        && delta.reasoning_items.is_empty()
        && delta.encrypted_reasoning.is_none()
        && delta.tool_calls.is_empty()
        && delta.final_output.is_none()
        && usage.is_none()
        && response_id.is_empty()
    {
        return Ok(None);
    }

    Ok(Some(ChatCompletionChunk {
        id: if !response_id.is_empty() {
            response_id.to_string()
        } else {
            v.get("id")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string()
        },
        object: "response.chunk".to_string(),
        created: 0,
        model: v
            .get("model")
            .or_else(|| v.pointer("/response/model"))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta,
            finish_reason: None,
        }],
        usage,
    }))
}

