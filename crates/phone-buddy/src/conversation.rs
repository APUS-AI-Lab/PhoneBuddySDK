//! Canonical conversation representation: an ordered list of heterogeneous items.
//!
//! Wire adapters down-convert from this shape. The legacy [`ChatMessage`]
//! list is retained only for host-transport and v1 session compatibility.

use serde::{Deserialize, Serialize};

use crate::llm::types::{
    reasoning_item_text, ChatMessage, ReasoningItem, Role, ToolCall, ToolCallFunction,
};

/// A single item in a conversation — the unified internal representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationItem {
    System(SystemItem),
    User(UserItem),
    Assistant(AssistantItem),
    ToolResult(ToolResultItem),
    /// Sibling reasoning item (Responses `rs_*` / synthesized).
    Reasoning(ReasoningItem),
    /// Server-executed hosted tool call (`web_search_call`, …), stored as the
    /// raw wire item so Responses replay is verbatim.
    BackendToolCall(BackendToolCallItem),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemItem {
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserItem {
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantItem {
    /// Empty when the turn is tool-call-only.
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// ChatCompletions `reasoning_content` / Anthropic thinking text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Anthropic thinking signature / single-blob encrypted reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_reasoning: Option<String>,
    /// Provider compat key (`group/model`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultItem {
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolCallItem {
    /// Wire item type: "web_search_call", "code_interpreter_call", …
    pub item_type: String,
    /// Canonical id (`ws_*`, `ci_*`), empty when the provider omitted it.
    #[serde(default)]
    pub id: String,
    /// Full original `response.output` item, replayed verbatim (Responses).
    pub payload: serde_json::Value,
}

impl ConversationItem {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(SystemItem {
            content: content.into(),
        })
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User(UserItem {
            content: content.into(),
        })
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant(AssistantItem {
            content: content.into(),
            tool_calls: Vec::new(),
            reasoning_content: None,
            encrypted_reasoning: None,
            origin: None,
        })
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult(ToolResultItem {
            tool_call_id: call_id.into(),
            content: content.into(),
        })
    }

    pub fn as_assistant(&self) -> Option<&AssistantItem> {
        match self {
            Self::Assistant(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_assistant_mut(&mut self) -> Option<&mut AssistantItem> {
        match self {
            Self::Assistant(a) => Some(a),
            _ => None,
        }
    }

    pub fn origin(&self) -> Option<&str> {
        self.as_assistant().and_then(|a| a.origin.as_deref())
    }
}

impl AssistantItem {
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self
    }
}

/// Human-readable one-line summary of a backend-executed tool call.
/// Used when down-converting to protocols that cannot replay the native item.
pub fn backend_call_summary(item: &BackendToolCallItem) -> String {
    let id = if item.id.is_empty() {
        String::new()
    } else {
        format!(" {}", item.id)
    };
    let extra = backend_call_detail(&item.payload);
    if extra.is_empty() {
        format!("[backend {}{}]", item.item_type, id)
    } else {
        format!("[backend {}{}: {extra}]", item.item_type, id)
    }
}

fn backend_call_detail(payload: &serde_json::Value) -> String {
    if let Some(action) = payload.get("action") {
        if let Some(q) = action.get("query").and_then(|s| s.as_str()) {
            if !q.is_empty() {
                return truncate_summary(q, 80);
            }
        }
        if let Some(url) = action.get("url").and_then(|s| s.as_str()) {
            if !url.is_empty() {
                return truncate_summary(url, 80);
            }
        }
    }
    if let Some(args) = payload.get("arguments") {
        let s = if let Some(t) = args.as_str() {
            t.to_string()
        } else {
            args.to_string()
        };
        if !s.is_empty() && s != "{}" {
            return truncate_summary(&s, 80);
        }
    }
    String::new()
}

fn truncate_summary(s: &str, max: usize) -> String {
    let t = s.trim();
    let mut out: String = t.chars().take(max).collect();
    if t.chars().count() > max {
        out.push('…');
    }
    out
}

/// Map a client-visible server-tool name (`web_search`) to the Responses
/// output item type (`web_search_call`). Names that already end in `_call`
/// are kept as-is.
pub fn server_tool_item_type(name: &str) -> String {
    if name.ends_with("_call") {
        name.to_string()
    } else {
        format!("{name}_call")
    }
}

/// Inverse of [`server_tool_item_type`]: `web_search_call` → `web_search`.
pub fn server_tool_function_name(item_type: &str) -> String {
    item_type
        .strip_suffix("_call")
        .unwrap_or(item_type)
        .to_string()
}

/// Reconstruct a minimal Responses payload from a legacy `kind == "server"`
/// tool call. Documented lossy: only `type`/`id`/`action`/`arguments` survive.
pub fn reconstruct_backend_payload(item_type: &str, id: &str, arguments: &str) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "type".into(),
        serde_json::Value::String(item_type.to_string()),
    );
    if !id.is_empty() {
        obj.insert("id".into(), serde_json::Value::String(id.to_string()));
    }
    let args_trim = arguments.trim();
    if !args_trim.is_empty() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(args_trim) {
            if item_type == "web_search_call" || item_type == "file_search_call" {
                obj.insert("action".into(), v);
            } else {
                obj.insert("arguments".into(), v);
            }
        } else {
            obj.insert(
                "arguments".into(),
                serde_json::Value::String(arguments.to_string()),
            );
        }
    }
    serde_json::Value::Object(obj)
}

/// Up-conversion from the legacy Chat Completions-shaped list.
///
/// Reasoning fields become a leading `Reasoning` sibling; `kind == "server"`
/// tool calls become `BackendToolCall` items with a reconstructed payload.
pub fn items_from_chat_messages(messages: &[ChatMessage]) -> Vec<ConversationItem> {
    let mut out = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                out.push(ConversationItem::system(
                    msg.content.clone().unwrap_or_default(),
                ));
            }
            Role::User => {
                out.push(ConversationItem::user(
                    msg.content.clone().unwrap_or_default(),
                ));
            }
            Role::Assistant => {
                for r in &msg.reasoning_items {
                    out.push(ConversationItem::Reasoning(r.clone()));
                }
                if msg.reasoning_items.is_empty() {
                    if let Some(item) = crate::llm::types::build_synthetic_reasoning(
                        String::new(),
                        msg.reasoning_content.as_deref(),
                        msg.encrypted_reasoning.as_deref(),
                    ) {
                        out.push(ConversationItem::Reasoning(item));
                    }
                }

                let mut client_calls = Vec::new();
                for tc in &msg.tool_calls {
                    if tc.kind == "server" {
                        let item_type = server_tool_item_type(&tc.function.name);
                        let payload = reconstruct_backend_payload(
                            &item_type,
                            &tc.id,
                            &tc.function.arguments,
                        );
                        out.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                            item_type,
                            id: tc.id.clone(),
                            payload,
                        }));
                    } else {
                        client_calls.push(tc.clone());
                    }
                }

                let content = msg.content.clone().unwrap_or_default();
                out.push(ConversationItem::Assistant(AssistantItem {
                    content,
                    tool_calls: client_calls,
                    reasoning_content: msg.reasoning_content.clone(),
                    encrypted_reasoning: msg.encrypted_reasoning.clone(),
                    origin: msg.origin.clone(),
                }));
            }
            Role::Tool => {
                out.push(ConversationItem::tool_result(
                    msg.tool_call_id.clone().unwrap_or_default(),
                    msg.content.clone().unwrap_or_default(),
                ));
            }
        }
    }
    out
}

/// Down-conversion for legacy consumers (host contract, v1 session compat).
///
/// Documented lossy cases:
/// - sibling order inside a turn collapses onto the assistant message
/// - backend payload keeps only the arguments / action JSON
pub fn chat_messages_from_items(items: &[ConversationItem]) -> Vec<ChatMessage> {
    let mut out = Vec::new();
    let mut pending_reasoning: Vec<ReasoningItem> = Vec::new();
    let mut pending_backend: Vec<ToolCall> = Vec::new();

    let flush_orphans = |out: &mut Vec<ChatMessage>,
                         pending_reasoning: &mut Vec<ReasoningItem>,
                         pending_backend: &mut Vec<ToolCall>| {
        if pending_reasoning.is_empty() && pending_backend.is_empty() {
            return;
        }
        let reasoning_content = first_reasoning_text(pending_reasoning);
        let encrypted = first_encrypted(pending_reasoning);
        out.push(ChatMessage {
            role: Role::Assistant,
            content: None,
            tool_calls: std::mem::take(pending_backend),
            tool_call_id: None,
            reasoning_items: std::mem::take(pending_reasoning),
            reasoning_content,
            encrypted_reasoning: encrypted,
            origin: None,
        });
    };

    for item in items {
        match item {
            ConversationItem::System(s) => {
                flush_orphans(&mut out, &mut pending_reasoning, &mut pending_backend);
                out.push(ChatMessage::system(&s.content));
            }
            ConversationItem::User(u) => {
                flush_orphans(&mut out, &mut pending_reasoning, &mut pending_backend);
                out.push(ChatMessage::user(&u.content));
            }
            ConversationItem::Reasoning(r) => {
                pending_reasoning.push(r.clone());
            }
            ConversationItem::BackendToolCall(b) => {
                pending_backend.push(backend_to_legacy_tool_call(b));
            }
            ConversationItem::Assistant(a) => {
                let mut tool_calls = std::mem::take(&mut pending_backend);
                tool_calls.extend(a.tool_calls.clone());
                let reasoning_items = std::mem::take(&mut pending_reasoning);
                let reasoning_content = a
                    .reasoning_content
                    .clone()
                    .or_else(|| first_reasoning_text(&reasoning_items));
                let encrypted = a
                    .encrypted_reasoning
                    .clone()
                    .or_else(|| first_encrypted(&reasoning_items));
                out.push(ChatMessage {
                    role: Role::Assistant,
                    content: if a.content.is_empty() {
                        None
                    } else {
                        Some(a.content.clone())
                    },
                    tool_calls,
                    tool_call_id: None,
                    reasoning_items,
                    reasoning_content,
                    encrypted_reasoning: encrypted,
                    origin: a.origin.clone(),
                });
            }
            ConversationItem::ToolResult(t) => {
                flush_orphans(&mut out, &mut pending_reasoning, &mut pending_backend);
                out.push(ChatMessage::tool_result(&t.tool_call_id, &t.content));
            }
        }
    }
    flush_orphans(&mut out, &mut pending_reasoning, &mut pending_backend);
    out
}

fn first_reasoning_text(items: &[ReasoningItem]) -> Option<String> {
    let joined: String = items
        .iter()
        .map(reasoning_item_text)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

fn first_encrypted(items: &[ReasoningItem]) -> Option<String> {
    items
        .iter()
        .find_map(|r| r.encrypted_content.clone())
        .filter(|s| !s.is_empty())
}

pub fn backend_to_legacy_tool_call(b: &BackendToolCallItem) -> ToolCall {
    let arguments = b
        .payload
        .get("action")
        .or_else(|| b.payload.get("arguments"))
        .map(|v| {
            if let Some(s) = v.as_str() {
                s.to_string()
            } else {
                v.to_string()
            }
        })
        .unwrap_or_else(|| "{}".into());
    ToolCall {
        id: b.id.clone(),
        kind: "server".into(),
        function: ToolCallFunction {
            name: server_tool_function_name(&b.item_type),
            arguments,
        },
        thought_signature: None,
    }
}

/// Count `User` + `Assistant` items (session list metadata).
pub fn user_assistant_count(items: &[ConversationItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i, ConversationItem::User(_) | ConversationItem::Assistant(_)))
        .count()
}

/// Look up the function name of a client tool call by `call_id`.
pub fn function_name_for_call<'a>(items: &'a [ConversationItem], call_id: &str) -> Option<&'a str> {
    for item in items {
        if let ConversationItem::Assistant(a) = item {
            if let Some(tc) = a.tool_calls.iter().find(|c| c.id == call_id) {
                return Some(tc.function.name.as_str());
            }
        }
    }
    None
}

/// Split items into turn groups. A group is a maximal run of
/// `[Reasoning | BackendToolCall]* Assistant`, or a singleton of any
/// other item.
pub fn turn_groups(items: &[ConversationItem]) -> Vec<&[ConversationItem]> {
    let mut groups = Vec::new();
    let mut i = 0;
    while i < items.len() {
        match &items[i] {
            ConversationItem::Reasoning(_) | ConversationItem::BackendToolCall(_) => {
                let start = i;
                while i < items.len()
                    && matches!(
                        items[i],
                        ConversationItem::Reasoning(_) | ConversationItem::BackendToolCall(_)
                    )
                {
                    i += 1;
                }
                if i < items.len() && matches!(items[i], ConversationItem::Assistant(_)) {
                    i += 1;
                }
                groups.push(&items[start..i]);
            }
            ConversationItem::Assistant(_) => {
                groups.push(&items[i..i + 1]);
                i += 1;
            }
            _ => {
                groups.push(&items[i..i + 1]);
                i += 1;
            }
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{SummaryPart, SummaryTextContent};

    fn reasoning(id: &str, text: &str) -> ReasoningItem {
        ReasoningItem {
            id: id.into(),
            summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                text: text.into(),
            })],
            content: None,
            encrypted_content: None,
            status: None,
        }
    }

    fn client_call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: name.into(),
                arguments: args.into(),
            },
            thought_signature: None,
        }
    }

    #[test]
    fn items_chatmessage_roundtrip_without_backend() {
        let items = vec![
            ConversationItem::system("sys"),
            ConversationItem::user("hi"),
            ConversationItem::Reasoning(reasoning("rs_1", "think")),
            ConversationItem::Assistant(AssistantItem {
                content: "hello".into(),
                tool_calls: vec![client_call("c1", "read_file", r#"{"path":"a"}"#)],
                reasoning_content: Some("think".into()),
                encrypted_reasoning: Some("sig".into()),
                origin: Some("openai/gpt-5".into()),
            }),
            ConversationItem::tool_result("c1", "ok"),
            ConversationItem::assistant("done"),
        ];
        let msgs = chat_messages_from_items(&items);
        let back = items_from_chat_messages(&msgs);
        assert_eq!(back, items);
    }

    #[test]
    fn items_chatmessage_roundtrip_backend_degrades_payload() {
        let payload = serde_json::json!({
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": { "type": "search", "query": "apus", "sources": ["a"] }
        });
        let items = vec![
            ConversationItem::user("search"),
            ConversationItem::BackendToolCall(BackendToolCallItem {
                item_type: "web_search_call".into(),
                id: "ws_1".into(),
                payload: payload.clone(),
            }),
            ConversationItem::assistant("found it"),
        ];
        let msgs = chat_messages_from_items(&items);
        assert_eq!(msgs[1].tool_calls.len(), 1);
        assert_eq!(msgs[1].tool_calls[0].kind, "server");
        assert_eq!(msgs[1].tool_calls[0].id, "ws_1");
        assert_eq!(msgs[1].tool_calls[0].function.name, "web_search");

        let back = items_from_chat_messages(&msgs);
        match &back[1] {
            ConversationItem::BackendToolCall(b) => {
                assert_eq!(b.item_type, "web_search_call");
                assert_eq!(b.id, "ws_1");
                assert_eq!(b.payload["type"], "web_search_call");
                assert_eq!(b.payload["id"], "ws_1");
                assert_eq!(b.payload["action"]["query"], "apus");
                assert!(b.payload.get("status").is_none());
            }
            other => panic!("expected backend call, got {other:?}"),
        }
        match &back[2] {
            ConversationItem::Assistant(a) => assert_eq!(a.content, "found it"),
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    #[test]
    fn serde_roundtrip_every_variant() {
        let items = vec![
            ConversationItem::system("s"),
            ConversationItem::user("你好"),
            ConversationItem::Reasoning(ReasoningItem {
                id: "rs_1".into(),
                summary: Vec::new(),
                content: None,
                encrypted_content: Some("enc".into()),
                status: Some("completed".into()),
            }),
            ConversationItem::BackendToolCall(BackendToolCallItem {
                item_type: "web_search_call".into(),
                id: "ws_1".into(),
                payload: serde_json::json!({"type":"web_search_call","id":"ws_1","nested":{"x":"⌘"}}),
            }),
            ConversationItem::Assistant(AssistantItem {
                content: "ok".into(),
                tool_calls: vec![client_call("c1", "fn", "{}")],
                reasoning_content: None,
                encrypted_reasoning: None,
                origin: Some("g/m".into()),
            }),
            ConversationItem::tool_result("c1", "out"),
        ];
        let json = serde_json::to_string(&items).unwrap();
        let back: Vec<ConversationItem> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, items);
    }
}
