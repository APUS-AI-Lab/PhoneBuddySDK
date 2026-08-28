//! Table-driven golden payloads for all four backends.

use crate::conversation::{AssistantItem, BackendToolCallItem, ConversationItem, ToolResultItem};
use crate::llm::types::{
    ConversationRequest, FunctionDefinitionWire, ReasoningItem, SummaryPart, SummaryTextContent,
    ToolCall, ToolCallFunction, ToolDefinitionWire,
};
use crate::llm::wire::chat_completions::build_chat_completions_payload;
use crate::llm::wire::gemini::build_gemini_payload;
use crate::llm::wire::messages::build_messages_payload;
use crate::llm::wire::responses::build_responses_payload;

fn fixture() -> ConversationRequest {
    ConversationRequest {
        model: "test-model".into(),
        items: vec![
            ConversationItem::system("You are helpful."),
            ConversationItem::user("search then read"),
            ConversationItem::Reasoning(ReasoningItem {
                id: "rs_1".into(),
                summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                    text: "plan".into(),
                })],
                content: None,
                encrypted_content: Some("enc".into()),
                status: Some("completed".into()),
            }),
            ConversationItem::BackendToolCall(BackendToolCallItem {
                item_type: "web_search_call".into(),
                id: "ws_1".into(),
                payload: serde_json::json!({
                    "type": "web_search_call",
                    "id": "ws_1",
                    "action": {"type": "search", "query": "apus"}
                }),
            }),
            ConversationItem::Assistant(AssistantItem {
                content: String::new(),
                tool_calls: vec![
                    ToolCall {
                        id: "c1".into(),
                        kind: "function".into(),
                        function: ToolCallFunction {
                            name: "read_file".into(),
                            arguments: r#"{"path":"a"}"#.into(),
                        },
                        thought_signature: Some("g-sig".into()),
                    },
                    ToolCall {
                        id: "c2".into(),
                        kind: "function".into(),
                        function: ToolCallFunction {
                            name: "grep".into(),
                            arguments: r#"{"pattern":"x"}"#.into(),
                        },
                        thought_signature: None,
                    },
                ],
                reasoning_content: Some("plan".into()),
                encrypted_reasoning: Some("enc".into()),
                origin: Some("openai/gpt-5".into()),
            }),
            ConversationItem::ToolResult(ToolResultItem {
                tool_call_id: "c1".into(),
                content: "file-a".into(),
            }),
            ConversationItem::ToolResult(ToolResultItem {
                tool_call_id: "c2".into(),
                content: "hits".into(),
            }),
            ConversationItem::assistant("done"),
        ],
        stream: Some(true),
        tools: Some(vec![ToolDefinitionWire {
            kind: "function".into(),
            function: FunctionDefinitionWire {
                name: "read_file".into(),
                description: Some("read".into()),
                parameters: serde_json::json!({"type": "object"}),
            },
        }]),
        tool_choice: None,
        temperature: Some(0.2),
        max_tokens: Some(1024),
        reasoning_effort: None,
        search_parameters: None,
        hosted_tools: vec![],
        previous_response_id: None,
        image_bytes: crate::llm::image::ImageBytesStore::default(),
    }
}

#[test]
fn conformance_fixture_per_backend() {
    let req = fixture();
    let responses = build_responses_payload(&req).unwrap();
    let cc = build_chat_completions_payload(&req).unwrap();
    let messages = build_messages_payload(&req).unwrap();
    let gemini = build_gemini_payload(&req).unwrap();

    // Responses: native reasoning + verbatim backend payload, no function_call for ws_*.
    let input = responses["input"].as_array().unwrap();
    assert_eq!(input[1]["type"], "reasoning");
    assert_eq!(input[2]["type"], "web_search_call");
    assert_eq!(input[2]["id"], "ws_1");
    assert_eq!(input[3]["type"], "function_call");
    assert_eq!(input[3]["call_id"], "c1");
    assert_eq!(responses["store"], false);
    assert!(responses["include"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v == "reasoning.encrypted_content"));

    // CC: no reasoning item types, backend becomes assistant text.
    let cc_s = cc.to_string();
    assert!(!cc_s.contains("\"type\":\"reasoning\""));
    assert!(!cc_s.contains("\"type\":\"web_search_call\""));
    assert!(cc_s.contains("[backend web_search_call"));
    assert_eq!(cc["messages"].as_array().unwrap().iter().filter(|m| m["role"] == "tool").count(), 2);

    // Messages: thinking before tool_use.
    let asst = &messages["messages"][1];
    assert_eq!(asst["role"], "assistant");
    assert_eq!(asst["content"][0]["type"], "thinking");
    assert_eq!(asst["content"][1]["type"], "text"); // backend summary
    assert_eq!(asst["content"][2]["type"], "tool_use");

    // Gemini: functionCall args are objects; functionResponse role is user.
    let gem_s = gemini.to_string();
    assert!(gemini["contents"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["parts"]
            .as_array()
            .map(|p| p.iter().any(|part| part.get("functionCall").is_some()))
            .unwrap_or(false)));
    assert!(gem_s.contains("functionResponse"));
    assert_eq!(gemini["systemInstruction"]["parts"][0]["text"], "You are helpful.");
}

#[test]
fn pairing_invariants_all_backends() {
    let req = fixture();
    let responses = build_responses_payload(&req).unwrap();
    let cc = build_chat_completions_payload(&req).unwrap();
    let messages = build_messages_payload(&req).unwrap();
    let gemini = build_gemini_payload(&req).unwrap();

    // Responses: every function_call has a matching function_call_output.
    let input = responses["input"].as_array().unwrap();
    let mut calls = Vec::new();
    let mut outputs = Vec::new();
    for item in input {
        if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
            calls.push(item["call_id"].as_str().unwrap().to_string());
        }
        if item.get("type").and_then(|t| t.as_str()) == Some("function_call_output") {
            outputs.push(item["call_id"].as_str().unwrap().to_string());
        }
    }
    assert_eq!(calls, outputs);

    // CC: tool_calls paired with role=tool.
    let mut cc_ids = Vec::new();
    let mut cc_results = Vec::new();
    for m in cc["messages"].as_array().unwrap() {
        if m["role"] == "assistant" {
            if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tcs {
                    cc_ids.push(tc["id"].as_str().unwrap().to_string());
                }
            }
        }
        if m["role"] == "tool" {
            cc_results.push(m["tool_call_id"].as_str().unwrap().to_string());
        }
    }
    assert_eq!(cc_ids, cc_results);

    // Messages: tool_use / tool_result pairing + thinking before tool_use.
    let mut use_ids = Vec::new();
    let mut result_ids = Vec::new();
    for m in messages["messages"].as_array().unwrap() {
        if let Some(blocks) = m.get("content").and_then(|c| c.as_array()) {
            let mut saw_tool = false;
            for b in blocks {
                let ty = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if ty == "thinking" {
                    assert!(!saw_tool, "thinking must precede tool_use");
                }
                if ty == "tool_use" {
                    saw_tool = true;
                    use_ids.push(b["id"].as_str().unwrap().to_string());
                }
                if ty == "tool_result" {
                    result_ids.push(b["tool_use_id"].as_str().unwrap().to_string());
                }
            }
        }
    }
    assert_eq!(use_ids, result_ids);

    // Gemini: functionCall / functionResponse pairing.
    let mut g_calls = 0usize;
    let mut g_resp = 0usize;
    for c in gemini["contents"].as_array().unwrap() {
        if let Some(parts) = c.get("parts").and_then(|p| p.as_array()) {
            for p in parts {
                if p.get("functionCall").is_some() {
                    g_calls += 1;
                }
                if p.get("functionResponse").is_some() {
                    g_resp += 1;
                    assert_eq!(c["role"], "user");
                }
            }
        }
    }
    assert_eq!(g_calls, g_resp);
}

#[test]
fn backend_call_degrades_to_text() {
    let req = fixture();
    let responses = build_responses_payload(&req).unwrap();
    let cc = build_chat_completions_payload(&req).unwrap();
    let messages = build_messages_payload(&req).unwrap();
    let gemini = build_gemini_payload(&req).unwrap();

    let input = responses["input"].as_array().unwrap();
    let backend = input
        .iter()
        .find(|i| i.get("type").and_then(|t| t.as_str()) == Some("web_search_call"))
        .unwrap();
    assert_eq!(
        backend,
        &serde_json::json!({
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "apus"}
        })
    );

    let cc_s = cc.to_string();
    assert_eq!(cc_s.matches("[backend web_search_call").count(), 1);
    assert!(!cc["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m
            .get("tool_calls")
            .and_then(|t| t.as_array())
            .map(|a| a.iter().any(|tc| tc["id"] == "ws_1"))
            .unwrap_or(false)));

    let msg_s = messages.to_string();
    assert_eq!(msg_s.matches("[backend web_search_call").count(), 1);
    assert!(!msg_s.contains("\"id\":\"ws_1\""));

    let gem_s = gemini.to_string();
    assert_eq!(gem_s.matches("[backend web_search_call").count(), 1);
}

#[test]
fn origin_strip_blocks_foreign_signatures() {
    let mut req = fixture();
    req.items = crate::llm::failover::sanitize_items_for_provider(
        &req.items,
        "anthropic/claude",
        "openai/gpt-5",
    );
    let messages = build_messages_payload(&req).unwrap();
    let gemini = build_gemini_payload(&req).unwrap();
    let msg_s = messages.to_string();
    let gem_s = gemini.to_string();
    assert!(!msg_s.contains("\"signature\":\"enc\""));
    assert!(!gem_s.contains("thoughtSignature"));
    assert!(!gem_s.contains("g-sig"));
}

#[test]
fn messages_thinking_block_ordering() {
    let req = fixture();
    let messages = build_messages_payload(&req).unwrap();
    let asst = &messages["messages"][1];
    assert_eq!(asst["content"][0]["type"], "thinking");
    assert_eq!(asst["content"][0]["signature"], "enc");
    let types: Vec<_> = asst["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["type"].as_str().unwrap())
        .collect();
    let think_at = types.iter().position(|t| *t == "thinking").unwrap();
    let tool_at = types.iter().position(|t| *t == "tool_use").unwrap();
    assert!(think_at < tool_at);
}

#[test]
fn host_contract_still_cc() {
    let req = fixture();
    let json = crate::llm::wire::chat_completions::host_request_json(&req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(v.get("messages").is_some());
    assert!(v.get("items").is_none());
    assert!(!json.contains("\"type\":\"reasoning\""));
    assert!(!json.contains("backend_tool_call"));
    let _: crate::llm::types::ChatCompletionRequest = serde_json::from_str(&json).unwrap();
}

#[test]
fn endpoint_and_headers_per_backend() {
    use crate::llm::types::ApiBackend;
    use crate::llm::wire::adapter_for;

    let chat = adapter_for(ApiBackend::ChatCompletions);
    assert_eq!(
        chat.endpoint("https://api.x.ai/v1", "m", true),
        "https://api.x.ai/v1/chat/completions"
    );
    let resp = adapter_for(ApiBackend::Responses);
    assert_eq!(
        resp.endpoint("https://api.openai.com/v1", "m", true),
        "https://api.openai.com/v1/responses"
    );
    let msg = adapter_for(ApiBackend::Messages);
    assert_eq!(
        msg.endpoint("https://api.anthropic.com/v1", "m", true),
        "https://api.anthropic.com/v1/messages"
    );
    let gem = adapter_for(ApiBackend::Gemini);
    assert_eq!(
        gem.endpoint(
            "https://generativelanguage.googleapis.com/v1beta",
            "gemini-2.5-flash",
            true
        ),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    );
}

#[test]
fn multimodal_image_conformance_all_backends() {
    use crate::conversation::{ImageDetail, ImageMimeType, UserContentPart, UserItem};
    use crate::llm::image::{ImageBytesStore, MaterializedImage};

    let png_bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4];
    let store = ImageBytesStore::default();
    store.insert(MaterializedImage {
        attachment_id: "att_1".into(),
        mime_type: ImageMimeType::Png,
        bytes: png_bytes.clone(),
        detail: Some(ImageDetail::High),
        width: 100,
        height: 100,
    });

    let req = ConversationRequest {
        model: "test-model".into(),
        items: vec![ConversationItem::User(UserItem {
            parts: vec![
                UserContentPart::Text {
                    text: "describe this".into(),
                },
                UserContentPart::Image {
                    attachment_id: "att_1".into(),
                    local_path: "/tmp/fake.png".into(),
                    mime_type: ImageMimeType::Png,
                    byte_size: png_bytes.len() as u64,
                    width: 100,
                    height: 100,
                    detail: Some(ImageDetail::High),
                },
            ],
        })],
        stream: Some(true),
        tools: None,
        tool_choice: None,
        temperature: None,
        max_tokens: None,
        reasoning_effort: None,
        search_parameters: None,
        hosted_tools: vec![],
        previous_response_id: None,
        image_bytes: store,
    };

    // 1. OpenAI / xAI Responses API
    let responses = build_responses_payload(&req).unwrap();
    let resp_input = &responses["input"][0];
    assert_eq!(resp_input["role"], "user");
    let resp_content = resp_input["content"].as_array().unwrap();
    // Normalized order: images first, then text
    assert_eq!(resp_content[0]["type"], "input_image");
    assert!(resp_content[0]["image_url"].as_str().unwrap().starts_with("data:image/png;base64,"));
    assert_eq!(resp_content[0]["detail"], "high");
    assert_eq!(resp_content[1]["type"], "input_text");
    assert_eq!(resp_content[1]["text"], "describe this");

    // 2. OpenAI Chat Completions API
    let cc = build_chat_completions_payload(&req).unwrap();
    let cc_msg = &cc["messages"][0];
    assert_eq!(cc_msg["role"], "user");
    let cc_content = cc_msg["content"].as_array().unwrap();
    assert_eq!(cc_content[0]["type"], "image_url");
    assert!(cc_content[0]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png;base64,"));
    assert_eq!(cc_content[0]["image_url"]["detail"], "high");
    assert_eq!(cc_content[1]["type"], "text");
    assert_eq!(cc_content[1]["text"], "describe this");

    // 3. Anthropic Messages API
    let msg = build_messages_payload(&req).unwrap();
    let msg_item = &msg["messages"][0];
    assert_eq!(msg_item["role"], "user");
    let msg_content = msg_item["content"].as_array().unwrap();
    assert_eq!(msg_content[0]["type"], "image");
    assert_eq!(msg_content[0]["source"]["type"], "base64");
    assert_eq!(msg_content[0]["source"]["media_type"], "image/png");
    assert!(!msg_content[0]["source"]["data"].as_str().unwrap().starts_with("data:"));
    assert_eq!(msg_content[1]["type"], "text");
    assert_eq!(msg_content[1]["text"], "describe this");

    // 4. Google Gemini generateContent API
    let gem = build_gemini_payload(&req).unwrap();
    let gem_item = &gem["contents"][0];
    assert_eq!(gem_item["role"], "user");
    let gem_parts = gem_item["parts"].as_array().unwrap();
    assert!(gem_parts[0].get("inlineData").is_some());
    assert_eq!(gem_parts[0]["inlineData"]["mimeType"], "image/png");
    assert!(!gem_parts[0]["inlineData"]["data"].as_str().unwrap().starts_with("data:"));
    assert_eq!(gem_parts[1]["text"], "describe this");
}
