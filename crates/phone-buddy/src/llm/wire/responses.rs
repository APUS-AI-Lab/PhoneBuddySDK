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
    let mut call_kinds: std::collections::HashMap<String, String> = std::collections::HashMap::new();

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
                    call_kinds.insert(tc.id.clone(), tc.kind.clone());
                    if tc.kind == "local_shell" {
                        let action_val = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::json!({
                                "type": "exec",
                                "command": [tc.function.arguments.clone()]
                            }));
                        input.push(serde_json::json!({
                            "type": "local_shell_call",
                            "call_id": tc.id,
                            "status": "completed",
                            "action": action_val
                        }));
                    } else if tc.kind == "custom_tool" {
                        input.push(serde_json::json!({
                            "type": "custom_tool_call",
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "input": tc.function.arguments
                        }));
                    } else {
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
            }
            ConversationItem::ToolResult(t) => {
                let is_custom = call_kinds.get(&t.tool_call_id).map(|k| k == "custom_tool").unwrap_or(false);
                if is_custom {
                    let output_val = serde_json::from_str::<serde_json::Value>(&t.content)
                        .unwrap_or_else(|_| serde_json::Value::String(t.content.clone()));
                    input.push(serde_json::json!({
                        "type": "custom_tool_call_output",
                        "call_id": t.tool_call_id,
                        "output": output_val
                    }));
                } else {
                    input.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": t.tool_call_id,
                        "output": t.content
                    }));
                }
            }
        }
    }

    let mut reasoning = serde_json::json!({ "summary": "concise" });
    if let Some(effort) = req.reasoning_effort {
        reasoning["effort"] = serde_json::Value::String(effort.as_str().to_string());
    }

    let mut payload = serde_json::json!({
        "model": req.model,
        "input": input,
        "stream": true,
        "store": false,
        "parallel_tool_calls": true,
        "include": ["reasoning.encrypted_content"],
        "reasoning": reasoning,
        "stream_options": {
            "reasoning_summary_delivery": "sequential_cutoff"
        },
        "text": {
            "verbosity": "medium"
        },
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
        if let Some(ref tc) = req.tool_choice {
            payload["tool_choice"] = tc.clone();
        } else {
            payload["tool_choice"] = serde_json::json!("auto");
        }
    }

    if let Some(format) = &req.response_format {
        payload["text"]["format"] = format.to_responses_text_format();
    }

    Ok(payload)
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

fn opt_nonzero_str(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn json_to_arg_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn function_call_arg_string(item: &serde_json::Value) -> Option<String> {
    item.get("arguments")
        .or_else(|| item.get("input"))
        .map(json_to_arg_string)
        .filter(|s| !s.is_empty())
}

fn function_call_arguments_event(v: &serde_json::Value, args: Option<String>) -> ToolCallDelta {
    ToolCallDelta {
        index: output_index_of(v),
        id: opt_nonzero_str(v, "call_id"),
        item_id: opt_nonzero_str(v, "item_id").or_else(|| opt_nonzero_str(v, "id")),
        kind: Some("function".to_string()),
        function: Some(ToolCallFunctionDelta {
            name: opt_nonzero_str(v, "name"),
            arguments: args,
        }),
        thought_signature: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{AssistantItem, ConversationItem};
    use crate::llm::types::HostedTool;

    fn req(items: Vec<ConversationItem>) -> ConversationRequest {
        ConversationRequest {
            model: "grok-4.6".into(),
            items,
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
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
    fn single_message_history() {
        let r = req(vec![
            ConversationItem::Assistant(AssistantItem {
                content: "ok".into(),
                tool_calls: vec![],
                reasoning_content: None,
                encrypted_reasoning: None,
                origin: None,
            }),
            ConversationItem::User(crate::conversation::UserItem::text("next")),
        ]);
        let p = build_responses_payload(&r).unwrap();
        let inp = p["input"].as_array().unwrap();
        assert_eq!(inp.len(), 2);
        assert_eq!(inp[0]["role"], "assistant");
        assert_eq!(inp[0]["content"], "ok");
        assert_eq!(inp[1]["role"], "user");
        assert_eq!(inp[1]["content"], "next");
    }

    #[test]
    fn previous_response_id_included() {
        let mut r = req(vec![ConversationItem::User(
            crate::conversation::UserItem::text("hi"),
        )]);
        r.previous_response_id = Some("resp_xyz".into());
        let p = build_responses_payload(&r).unwrap();
        assert_eq!(p["previous_response_id"], "resp_xyz");
    }

    #[test]
    fn patch_reasoning_text_types_adds_type_to_content() {
        let mut p = serde_json::json!({
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [{ "text": "thought" }]
                }
            ]
        });
        patch_reasoning_text_types(&mut p);
        assert_eq!(
            p["input"][0]["content"][0]["type"],
            "reasoning_text"
        );
    }

    #[test]
    fn responses_payload_multimodal_audio_and_image() {
        let store_img = crate::llm::image::ImageBytesStore::default();
        store_img.insert(crate::llm::image::MaterializedImage {
            attachment_id: "img_1".into(),
            mime_type: crate::conversation::ImageMimeType::Jpeg,
            bytes: vec![1, 2, 3],
            detail: Some(crate::conversation::ImageDetail::High),
            width: 100,
            height: 100,
        });
        let store_audio = crate::llm::image::AudioBytesStore::default();
        store_audio.insert(crate::llm::image::MaterializedAudio {
            attachment_id: "aud_1".into(),
            mime_type: crate::conversation::AudioMimeType::Wav,
            bytes: vec![4, 5, 6, 7],
            format: Some("wav".into()),
        });

        let mut r = req(vec![
            ConversationItem::User(crate::conversation::UserItem {
                parts: vec![
                    crate::conversation::UserContentPart::Text { text: "listen and look".into() },
                    crate::conversation::UserContentPart::Image {
                        attachment_id: "img_1".into(),
                        local_path: "p.jpg".into(),
                        mime_type: crate::conversation::ImageMimeType::Jpeg,
                        byte_size: 3,
                        width: 100,
                        height: 100,
                        detail: Some(crate::conversation::ImageDetail::High),
                    },
                    crate::conversation::UserContentPart::Audio {
                        attachment_id: "aud_1".into(),
                        local_path: "a.wav".into(),
                        mime_type: crate::conversation::AudioMimeType::Wav,
                        byte_size: 4,
                        format: Some("wav".into()),
                    },
                ],
            }),
        ]);
        r.image_bytes = store_img;
        r.audio_bytes = store_audio;

        let p = build_responses_payload(&r).unwrap();
        let content = p["input"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "input_image");
        assert_eq!(content[0]["detail"], "high");
        assert_eq!(content[1]["type"], "input_audio");
        assert!(content[1]["audio_url"].as_str().unwrap().starts_with("data:audio/wav;base64,"));
        assert_eq!(content[2]["type"], "input_text");
        assert_eq!(content[2]["text"], "listen and look");
    }

    #[test]
    fn responses_payload_x_search_and_web_search_hosted_tools() {
        let mut r = req(vec![ConversationItem::User(crate::conversation::UserItem::text("search x"))]);
        r.hosted_tools = vec![
            HostedTool::WebSearch {
                options: Some(crate::llm::types::WebSearchOptions {
                    allowed_domains: Some(vec!["example.com".into()]),
                    excluded_domains: None,
                }),
            },
            HostedTool::XSearch {
                options: Some(crate::llm::types::XSearchOptions {
                    date_bound: None,
                    from_date: Some("2026-01-01".into()),
                    to_date: Some("2026-08-01".into()),
                }),
            },
        ];

        let p = build_responses_payload(&r).unwrap();
        let tools = p["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["type"], "web_search");
        assert_eq!(tools[0]["filters"]["allowed_domains"][0], "example.com");
        assert_eq!(tools[1]["type"], "x_search");
        assert_eq!(tools[1]["from_date"], "2026-01-01");
        assert_eq!(tools[1]["to_date"], "2026-08-01");
    }

    #[test]
    fn responses_payload_local_shell_and_custom_tool_serialization() {
        let r = req(vec![
            ConversationItem::Assistant(AssistantItem {
                content: String::new(),
                tool_calls: vec![
                    crate::llm::types::ToolCall {
                        id: "shell_1".into(),
                        kind: "local_shell".into(),
                        function: crate::llm::types::ToolCallFunction {
                            name: "local_shell".into(),
                            arguments: serde_json::json!({
                                "command": ["ls", "-la"]
                            }).to_string(),
                        },
                        thought_signature: None,
                    },
                    crate::llm::types::ToolCall {
                        id: "custom_1".into(),
                        kind: "custom_tool".into(),
                        function: crate::llm::types::ToolCallFunction {
                            name: "my_device_sensor".into(),
                            arguments: serde_json::json!({ "mode": "accelerometer" }).to_string(),
                        },
                        thought_signature: None,
                    },
                ],
                reasoning_content: None,
                encrypted_reasoning: None,
                origin: None,
            }),
            ConversationItem::ToolResult(crate::conversation::ToolResultItem {
                tool_call_id: "shell_1".into(),
                content: "file1.txt".into(),
            }),
            ConversationItem::ToolResult(crate::conversation::ToolResultItem {
                tool_call_id: "custom_1".into(),
                content: serde_json::json!({ "x": 0.1, "y": 9.8 }).to_string(),
            }),
        ]);

        let p = build_responses_payload(&r).unwrap();
        let input = p["input"].as_array().unwrap();
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["type"], "local_shell_call");
        assert_eq!(input[0]["call_id"], "shell_1");
        assert_eq!(input[0]["action"]["command"][0], "ls");
        assert_eq!(input[1]["type"], "custom_tool_call");
        assert_eq!(input[1]["call_id"], "custom_1");
        assert_eq!(input[1]["name"], "my_device_sensor");

        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "shell_1");
        assert_eq!(input[2]["output"], "file1.txt");

        assert_eq!(input[3]["type"], "custom_tool_call_output");
        assert_eq!(input[3]["call_id"], "custom_1");
        assert_eq!(input[3]["output"]["y"], 9.8);
    }

    #[test]
    fn parse_responses_chunk_custom_tool_and_local_shell_stream() {
        // 1. Output item added: local_shell_call
        let chunk1 = parse_responses_chunk(
            "response.output_item.added",
            &serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "local_shell_call",
                    "id": "item_sh_1",
                    "call_id": "call_sh_1",
                    "status": "in_progress",
                    "action": {
                        "type": "exec",
                        "command": ["cat", "hello.txt"]
                    }
                }
            }).to_string(),
        ).unwrap().unwrap();
        let tc1 = &chunk1.choices[0].delta.tool_calls[0];
        assert_eq!(tc1.kind.as_deref(), Some("local_shell"));
        assert_eq!(tc1.id.as_deref(), Some("call_sh_1"));

        // 2. Custom tool call input delta
        let chunk2 = parse_responses_chunk(
            "response.custom_tool_call_input.delta",
            &serde_json::json!({
                "type": "response.custom_tool_call_input.delta",
                "output_index": 1,
                "call_id": "call_custom_1",
                "name": "camera_capture",
                "delta": "{\"quality\": \"high\"}"
            }).to_string(),
        ).unwrap().unwrap();
        let tc2 = &chunk2.choices[0].delta.tool_calls[0];
        assert_eq!(tc2.kind.as_deref(), Some("custom_tool"));
        assert_eq!(tc2.function.as_ref().unwrap().name.as_deref(), Some("camera_capture"));

        // 3. Response completed with final_output
        let chunk3 = parse_responses_chunk(
            "response.completed",
            &serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "model": "grok-4.6",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "content": [{ "type": "output_text", "text": "All done." }]
                        },
                        {
                            "type": "local_shell_call",
                            "id": "item_sh_1",
                            "call_id": "call_sh_1",
                            "status": "completed",
                            "action": { "type": "exec", "command": ["cat", "hello.txt"] }
                        }
                    ]
                }
            }).to_string(),
        ).unwrap().unwrap();
        let final_out = chunk3.choices[0].delta.final_output.as_ref().unwrap();
        assert_eq!(final_out.len(), 2);
        assert!(matches!(final_out[0], OutputItemWire::Message { ref text, .. } if text == "All done."));
        assert!(matches!(final_out[1], OutputItemWire::LocalShellCall { ref call_id, .. } if call_id.as_deref() == Some("call_sh_1")));
    }
}

// ── SSE Chunk Parsing ───────────────────────────────────────────────────

fn tool_delta_from_output_item(
    item: &serde_json::Value,
    fallback_index: u32,
) -> Option<ToolCallDelta> {
    let item_type = item.get("type").and_then(|s| s.as_str()).unwrap_or("");
    match item_type {
        "function_call" => {
            let call_id = opt_nonzero_str(item, "call_id").or_else(|| opt_nonzero_str(item, "id"));
            let item_id = opt_nonzero_str(item, "id");
            let name = opt_nonzero_str(item, "name");
            let args = function_call_arg_string(item);
            Some(ToolCallDelta {
                index: fallback_index,
                id: call_id,
                item_id,
                kind: Some("function".to_string()),
                function: Some(ToolCallFunctionDelta {
                    name,
                    arguments: args,
                }),
                thought_signature: None,
            })
        }
        "local_shell_call" => {
            let id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
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
                item_id: None,
                kind: Some("local_shell".to_string()),
                function: Some(ToolCallFunctionDelta {
                    name: Some("local_shell".to_string()),
                    arguments: args,
                }),
                thought_signature: None,
            })
        }
        "custom_tool_call" => {
            let id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let name = item
                .get("name")
                .and_then(|s| s.as_str())
                .unwrap_or("custom_tool")
                .to_string();
            let args = item.get("input").or_else(|| item.get("arguments")).map(|a| {
                if let Some(s) = a.as_str() {
                    s.to_string()
                } else {
                    a.to_string()
                }
            });
            let kind = if name == "x_search" || name.starts_with("x_") {
                "server"
            } else {
                "custom_tool"
            };
            Some(ToolCallDelta {
                index: fallback_index,
                id,
                item_id: None,
                kind: Some(kind.to_string()),
                function: Some(ToolCallFunctionDelta {
                    name: Some(name),
                    arguments: args,
                }),
                thought_signature: None,
            })
        }
        "x_search" | "x_search_call" | "web_search_call" | "file_search_call" | "computer_call"
        | "mcp_call" | "image_generation_call" | "code_interpreter_call" => {
            let id = item
                .get("id")
                .or_else(|| item.get("call_id"))
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let name = match item_type {
                "x_search" | "x_search_call" => "x_search",
                "web_search_call" => "web_search",
                "file_search_call" => "file_search",
                "computer_call" => "computer",
                "mcp_call" => item.get("name").and_then(|s| s.as_str()).unwrap_or("mcp"),
                "image_generation_call" => "image_generation",
                "code_interpreter_call" => "code_interpreter",
                other => other,
            }
            .to_string();
            let args = item.get("action").or_else(|| item.get("arguments")).or_else(|| item.get("input")).map(|a| {
                if let Some(s) = a.as_str() {
                    s.to_string()
                } else {
                    a.to_string()
                }
            });
            Some(ToolCallDelta {
                index: fallback_index,
                id,
                item_id: None,
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
        || json_type.contains("response.incomplete")
        || type_str.contains("response.done")
        || json_type.contains("response.done");

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
    } else if type_str.contains("reasoning_summary_part.added")
        || json_type.contains("reasoning_summary_part.added")
    {
        let summary_index = v.get("summary_index").and_then(|n| n.as_u64()).unwrap_or(0);
        if summary_index > 0 {
            delta.reasoning_content = Some("\n\n".to_string());
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
        let args = v.get("delta").and_then(|s| s.as_str()).map(str::to_string);
        delta
            .tool_calls
            .push(function_call_arguments_event(&v, args));
    } else if type_str.contains("function_call_arguments.done")
        || json_type.contains("function_call_arguments.done")
    {
        // Codex takes the complete snapshot; grok-build ignores this event
        // because it already assembled deltas. Feed it through the merge
        // helper: fill an empty/`{}` buffer, do not concatenate onto a
        // finished JSON object.
        let args = v.get("arguments").map(json_to_arg_string);
        delta
            .tool_calls
            .push(function_call_arguments_event(&v, args));
    } else if type_str.contains("custom_tool_call_input.delta")
        || json_type.contains("custom_tool_call_input.delta")
    {
        let id = v
            .get("call_id")
            .or_else(|| v.get("id"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());
        let name = v.get("name").and_then(|s| s.as_str()).map(|s| s.to_string());
        let args = v
            .get("delta")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());
        let kind = name
            .as_deref()
            .map(|n| if n == "x_search" || n.starts_with("x_") { "server" } else { "custom_tool" })
            .unwrap_or("custom_tool");
        delta.tool_calls.push(ToolCallDelta {
            index: output_index_of(&v),
            id,
            item_id: None,
            kind: Some(kind.to_string()),
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

