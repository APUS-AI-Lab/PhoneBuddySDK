//! OpenAI-compatible Chat Completions adapter. Also the host-transport contract.

use crate::conversation::{backend_call_summary, chat_messages_from_items, ConversationItem};
use crate::error::EngineResult;
use crate::llm::types::{
    sanitize_tool_arguments, ChatCompletionChunk, ChatCompletionRequest, ConversationRequest, Role,
};

use super::WireAdapter;

pub struct ChatCompletionsAdapter;

impl WireAdapter for ChatCompletionsAdapter {
    fn endpoint(&self, base: &str, _model: &str, _stream: bool) -> String {
        format!("{base}/chat/completions")
    }

    fn build_payload(&self, req: &ConversationRequest) -> EngineResult<serde_json::Value> {
        build_chat_completions_payload(req)
    }

    fn parse_event(&self, _event: &str, data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
        crate::llm::stream::parse_chunk(data)
    }
}

pub fn build_chat_completions_payload(req: &ConversationRequest) -> EngineResult<serde_json::Value> {
    let host = items_to_chat_request(req)?;
    let mut val = serde_json::to_value(&host).unwrap_or_else(|_| serde_json::json!({}));
    val["stream"] = serde_json::Value::Bool(true);
    val["stream_options"] = serde_json::json!({ "include_usage": true });
    if let Some(arr) = val.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for m in arr {
            if let Some(obj) = m.as_object_mut() {
                obj.remove("origin");
                obj.remove("reasoning_items");
                obj.remove("encrypted_reasoning");
            }
        }
    }
    if let Some(format) = &req.response_format {
        val["response_format"] = format.to_chat_completions();
    }
    Ok(val)
}

/// Host-contract Chat Completions request (no item types, origin stripped
/// at serialize time by [`ChatMessage`] serde skip).
pub fn items_to_chat_request(req: &ConversationRequest) -> EngineResult<ChatCompletionRequest> {
    let mut messages = Vec::new();
    let mut pending_backend_text = String::new();

    for item in &req.items {
        match item {
            ConversationItem::BackendToolCall(b) => {
                if !pending_backend_text.is_empty() {
                    pending_backend_text.push('\n');
                }
                pending_backend_text.push_str(&backend_call_summary(b));
            }
            ConversationItem::Assistant(a) => {
                let mut content = String::new();
                if !pending_backend_text.is_empty() {
                    content.push_str(&pending_backend_text);
                    pending_backend_text.clear();
                    if !a.content.is_empty() {
                        content.push('\n');
                    }
                }
                content.push_str(&a.content);
                let mut msg = crate::llm::types::ChatMessage {
                    role: Role::Assistant,
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(crate::llm::types::MessageContent::text(content))
                    },
                    tool_calls: a
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            let mut c = tc.clone();
                            c.function.arguments = sanitize_tool_arguments(
                                &tc.id,
                                &tc.function.name,
                                &tc.function.arguments,
                            );
                            c.thought_signature = None;
                            c
                        })
                        .collect(),
                    tool_call_id: None,
                    reasoning_items: Vec::new(),
                    reasoning_content: a.reasoning_content.clone(),
                    encrypted_reasoning: None,
                    origin: None,
                };
                // Drop encrypted-only reasoning siblings; keep plaintext
                // reasoning_content when present (same-origin CC path).
                let _ = &mut msg;
                messages.push(msg);
            }
            ConversationItem::Reasoning(_) => {
                // Encrypted reasoning items are dropped on CC. Plaintext is
                // carried on the following assistant's reasoning_content.
            }
            ConversationItem::System(s) => {
                flush_backend_as_assistant(&mut messages, &mut pending_backend_text);
                messages.push(crate::llm::types::ChatMessage::system(&s.content));
            }
            ConversationItem::User(u) => {
                flush_backend_as_assistant(&mut messages, &mut pending_backend_text);
                messages.push(user_to_cc_message(u, req)?);
            }
            ConversationItem::ToolResult(t) => {
                flush_backend_as_assistant(&mut messages, &mut pending_backend_text);
                messages.push(crate::llm::types::ChatMessage::tool_result(
                    &t.tool_call_id,
                    &t.content,
                ));
            }
        }
    }
    flush_backend_as_assistant(&mut messages, &mut pending_backend_text);

    Ok(ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        stream: req.stream,
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        reasoning_effort: req.reasoning_effort,
        search_parameters: req.search_parameters.clone(),
        hosted_tools: Vec::new(),
        previous_response_id: None,
    })
}

fn user_to_cc_message(
    u: &crate::conversation::UserItem,
    req: &ConversationRequest,
) -> EngineResult<crate::llm::types::ChatMessage> {
    if !u.has_images() {
        return Ok(crate::llm::types::ChatMessage::user(u.text_content()));
    }
    let content = super::chat_completions_user_content(u, req)?;
    let parts: Vec<crate::llm::types::ChatContentPart> =
        serde_json::from_value(content).map_err(|e| {
            crate::error::EngineError::InvalidUserTurn(format!("cc content parts: {e}"))
        })?;
    Ok(crate::llm::types::ChatMessage {
        role: Role::User,
        content: Some(crate::llm::types::MessageContent::Parts(parts)),
        tool_calls: Vec::new(),
        tool_call_id: None,
        reasoning_items: Vec::new(),
        reasoning_content: None,
        encrypted_reasoning: None,
        origin: None,
    })
}

fn flush_backend_as_assistant(
    messages: &mut Vec<crate::llm::types::ChatMessage>,
    pending: &mut String,
) {
    if pending.is_empty() {
        return;
    }
    messages.push(crate::llm::types::ChatMessage::assistant(std::mem::take(
        pending,
    )));
}

/// Host contract: serialize items as Chat Completions JSON (legacy messages).
pub fn host_request_json(req: &ConversationRequest) -> EngineResult<String> {
    let mut host = req.to_host_chat_request();
    for m in &mut host.messages {
        m.origin = None;
        m.reasoning_items.clear();
    }
    // Prefer the CC adapter's backend-degrade so the host never sees
    // `type: reasoning` / backend items.
    let _ = chat_messages_from_items;
    let degraded = items_to_chat_request(req)?;
    serde_json::to_string(&degraded).map_err(Into::into)
}
