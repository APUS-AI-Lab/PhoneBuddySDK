//! Chat Completions wire types needed by the mobile engine.

use serde::{Deserialize, Serialize};

// ── API Backends ─────────────────────────────────────────────────────────

/// Which API backend protocol to use for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApiBackend {
    /// OpenAI Chat Completions protocol (/v1/chat/completions) - default
    #[default]
    ChatCompletions,
    /// OpenAI Responses protocol (/v1/responses)
    Responses,
    /// Anthropic Messages protocol (/v1/messages)
    Messages,
    /// Google Gemini generateContent / streamGenerateContent protocol
    Gemini,
}

impl ApiBackend {
    pub fn supports_native_schema(&self) -> bool {
        matches!(
            self,
            Self::ChatCompletions | Self::Responses | Self::Gemini
        )
    }

    pub fn requires_reasoning_strip(&self) -> bool {
        matches!(self, Self::Messages)
    }

    pub fn forwards_prompt_cache_key(&self) -> bool {
        matches!(self, Self::Responses)
    }
}

// ── Roles ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

// ── Reasoning items ───────────────────────────────────────────────────────
// Ported from upstream grok `rs::ReasoningItem` and `conversation.rs`.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningItem {
    #[serde(default)]
    pub id: String,
    // Always serialize, even as `[]`. The Responses API rejects
    // `type: "reasoning"` items that omit `summary` (`missing field
    // summary` → 422). Matches async-openai `rs::ReasoningItem` /
    // grok-build `conversation/responses.rs` (empty summary is valid
    // for encrypted-only `tco_*` blobs).
    #[serde(default)]
    pub summary: Vec<SummaryPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ReasoningTextContent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SummaryPart {
    SummaryText(SummaryTextContent),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SummaryTextContent {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningTextContent {
    #[serde(rename = "type", default = "default_reasoning_text_type")]
    pub r#type: String,
    pub text: String,
}

fn default_reasoning_text_type() -> String {
    "reasoning_text".to_string()
}

/// Ordering contract: summary parts come first, then content blocks.
/// Ported from grok-build `conversation.rs::reasoning_item_text`.
pub fn reasoning_item_text(r: &ReasoningItem) -> String {
    let mut parts: Vec<String> = Vec::new();
    for sp in &r.summary {
        match sp {
            SummaryPart::SummaryText(t) => parts.push(t.text.clone()),
        }
    }
    if let Some(ref content) = r.content {
        for c in content {
            parts.push(c.text.clone());
        }
    }
    parts.join("\n")
}

/// Construct an `ReasoningItem` carrying a single `SummaryText` part.
/// Ported from grok-build `conversation.rs::synthesized_reasoning_item`.
pub fn synthesized_reasoning_item(text: impl Into<String>) -> ReasoningItem {
    ReasoningItem {
        id: String::new(),
        summary: vec![SummaryPart::SummaryText(SummaryTextContent {
            text: text.into(),
        })],
        content: None,
        encrypted_content: None,
        status: None,
    }
}

/// Build a `ReasoningItem` from text and optional encrypted content string.
/// Ported from grok-build `conversation.rs::build_synthetic_reasoning`.
pub fn build_synthetic_reasoning(
    id: String,
    text: Option<&str>,
    encrypted: Option<&str>,
) -> Option<ReasoningItem> {
    let text = text.filter(|t| !t.is_empty());
    let encrypted = encrypted.filter(|e| !e.is_empty());
    if text.is_none() && encrypted.is_none() {
        return None;
    }
    let summary = match text {
        Some(t) => vec![SummaryPart::SummaryText(SummaryTextContent {
            text: t.to_string(),
        })],
        None => Vec::new(),
    };
    Some(ReasoningItem {
        id,
        summary,
        content: None,
        encrypted_content: encrypted.map(String::from),
        status: None,
    })
}

/// Merge `new` into `old` by `id`.
///
/// grok-build takes a single canonical list from
/// `ResponseCompleted.response.output`. PhoneBuddy does not keep a
/// `Response` object, so `output_item.done` and `response.output` can
/// both emit the same id; merge instead of appending duplicates.
/// Empty-id items are kept unless they are exact duplicates.
pub fn merge_reasoning_items(old: &[ReasoningItem], new: &[ReasoningItem]) -> Vec<ReasoningItem> {
    let mut out = old.to_vec();
    for n in new {
        if n.id.is_empty() {
            if !out.iter().any(|o| o == n) {
                out.push(n.clone());
            }
            continue;
        }
        if let Some(existing) = out.iter_mut().find(|o| o.id == n.id) {
            if n.encrypted_content.is_some() {
                existing.encrypted_content = n.encrypted_content.clone();
            }
            if !n.summary.is_empty() {
                existing.summary = n.summary.clone();
            }
            if n.content.is_some() {
                existing.content = n.content.clone();
            }
        } else {
            out.push(n.clone());
        }
    }
    out
}

/// Ported from grok-build `conversation.rs::inject_streaming_reasoning_fallback`.
///
/// Called after typed items are collected, when streamed reasoning
/// deltas were observed but those items have no summary text:
///
/// - If any existing item already carries summary text, leave items
///   untouched (the deltas are redundant).
/// - Otherwise, if there is an item with no summary text, append a
///   `SummaryText` part to it (avoids introducing a phantom sibling).
/// - Otherwise, push a new `synthesized_reasoning_item(text)`.
///
/// Grok-build inspects `summary` only (not `content`); encrypted-only
/// `tco_*` blobs keep `summary: []`.
pub fn inject_streaming_reasoning_fallback(items: &mut Vec<ReasoningItem>, text: &str) {
    if text.is_empty() {
        return;
    }
    let any_with_text = items.iter().any(|r| {
        r.summary.iter().any(|sp| match sp {
            SummaryPart::SummaryText(t) => !t.text.is_empty(),
        })
    });
    if any_with_text {
        return;
    }
    if let Some(r) = items.first_mut() {
        r.summary.push(SummaryPart::SummaryText(SummaryTextContent {
            text: text.to_string(),
        }));
        return;
    }
    items.push(synthesized_reasoning_item(text));
}

#[cfg(test)]
mod inject_fallback_tests {
    use super::*;

    fn empty_item(id: &str) -> ReasoningItem {
        ReasoningItem {
            id: id.into(),
            summary: Vec::new(),
            content: None,
            encrypted_content: None,
            status: None,
        }
    }

    #[test]
    fn leaves_items_untouched_when_any_summary_has_text() {
        let mut items = vec![ReasoningItem {
            id: "rs_1".into(),
            summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                text: "already there".into(),
            })],
            content: None,
            encrypted_content: None,
            status: None,
        }];
        inject_streaming_reasoning_fallback(&mut items, "streamed");
        assert_eq!(items.len(), 1);
        assert_eq!(reasoning_item_text(&items[0]), "already there");
    }

    #[test]
    fn appends_summary_to_first_empty_item() {
        let mut items = vec![empty_item("rs_1")];
        inject_streaming_reasoning_fallback(&mut items, "streamed");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "rs_1");
        assert_eq!(reasoning_item_text(&items[0]), "streamed");
    }

    #[test]
    fn synthesizes_item_when_none_exist() {
        let mut items = Vec::new();
        inject_streaming_reasoning_fallback(&mut items, "streamed");
        assert_eq!(items.len(), 1);
        assert!(items[0].id.is_empty());
        assert_eq!(reasoning_item_text(&items[0]), "streamed");
    }

    #[test]
    fn ignores_empty_text() {
        let mut items: Vec<ReasoningItem> = Vec::new();
        inject_streaming_reasoning_fallback(&mut items, "");
        assert!(items.is_empty());
    }
}

// ── Messages ─────────────────────────────────────────────────────────────

/// Chat Completions / host-contract message content: a string or multimodal parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl MessageContent {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    pub fn as_plain_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ChatContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ChatImageUrl,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Present on `role = "tool"` messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Reasoning items associated with this turn (Responses API sibling items).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_items: Vec<ReasoningItem>,
    /// Reasoning/thinking content for ChatCompletions reasoning_content or Anthropic Thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Encrypted reasoning tokens / signature for multi-turn continuation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_reasoning: Option<String>,
    /// Compatibility key (`group/model`) of the provider that produced this
    /// assistant turn. Missing (legacy sessions) is treated as the primary
    /// provider. Same group + same model keeps encrypted thinking across
    /// host failover. Stripped from Chat Completions wire JSON in transport.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

/// A provider that validates `function.arguments` rejects the whole request,
/// so one malformed call from an earlier turn breaks every turn after it.
/// Ported from grok `xai-grok-sampling-types::conversation::sanitize_tool_arguments`.
pub fn sanitize_tool_arguments(id: &str, name: &str, arguments: &str) -> String {
    if arguments.trim().is_empty() {
        return "{}".into();
    }
    // `IgnoredAny` avoids building a DOM on a path that runs for every call.
    if serde_json::from_str::<serde::de::IgnoredAny>(arguments).is_err() {
        tracing::warn!(
            tool_call_id = id,
            tool_name = name,
            "Tool call has invalid JSON arguments; replacing with {{}} to prevent provider 400"
        );
        "{}".into()
    } else {
        arguments.to_string()
    }
}

impl ChatMessage {
    /// Clone this message with tool-call arguments sanitized for outbound
    /// provider requests (invalid JSON → `{}`).
    pub fn sanitized_for_request(&self) -> Self {
        let mut out = self.clone();
        for tc in &mut out.tool_calls {
            tc.function.arguments =
                sanitize_tool_arguments(&tc.id, &tc.function.name, &tc.function.arguments);
        }
        out
    }

    /// Drop encrypted thinking when this message's compatibility key
    /// (`group/model`) differs from `target`. Text, tool calls, and tool
    /// results are always kept so the new model can continue the turn.
    /// Same group + same model (grok-build-style full history replay)
    /// leaves reasoning item ids and encrypted content in place.
    pub fn sanitized_for_provider(&self, target: &str, primary: &str) -> Self {
        let mut out = self.sanitized_for_request();
        if crate::llm::failover::should_strip_origin(out.origin.as_deref(), target, primary) {
            out.reasoning_items.clear();
            out.encrypted_reasoning = None;
            out.reasoning_content = None;
        }
        out
    }

    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self
    }

    pub fn content_text(&self) -> String {
        self.content
            .as_ref()
            .map(|c| c.as_plain_text())
            .unwrap_or_default()
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(MessageContent::text(text)),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning_items: Vec::new(),
            reasoning_content: None,
            encrypted_reasoning: None,
            origin: None,
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(MessageContent::text(text)),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning_items: Vec::new(),
            reasoning_content: None,
            encrypted_reasoning: None,
            origin: None,
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(MessageContent::text(text)),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning_items: Vec::new(),
            reasoning_content: None,
            encrypted_reasoning: None,
            origin: None,
        }
    }

    pub fn assistant_with_reasoning(
        text: impl Into<String>,
        reasoning: Option<String>,
        reasoning_items: Vec<ReasoningItem>,
        encrypted_reasoning: Option<String>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(MessageContent::text(text)),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning_items,
            reasoning_content: reasoning,
            encrypted_reasoning,
            origin: None,
        }
    }

    pub fn assistant_tool_calls(calls: Vec<ToolCall>, text: Option<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: text.map(MessageContent::Text),
            tool_calls: calls,
            tool_call_id: None,
            reasoning_items: Vec::new(),
            reasoning_content: None,
            encrypted_reasoning: None,
            origin: None,
        }
    }

    pub fn assistant_tool_calls_with_reasoning(
        calls: Vec<ToolCall>,
        text: Option<String>,
        reasoning: Option<String>,
        reasoning_items: Vec<ReasoningItem>,
        encrypted_reasoning: Option<String>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: text.map(MessageContent::Text),
            tool_calls: calls,
            tool_call_id: None,
            reasoning_items,
            reasoning_content: reasoning,
            encrypted_reasoning,
            origin: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(MessageContent::text(output)),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
            reasoning_items: Vec::new(),
            reasoning_content: None,
            encrypted_reasoning: None,
            origin: None,
        }
    }
}

// ── Tool calls ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: ToolCallFunction,
    /// Gemini per-part thought signature; never serialized on other wires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

impl Default for ToolCall {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: String::new(),
                arguments: "{}".into(),
            },
            thought_signature: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
}

// ── Tool definitions (request side) ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinitionWire {
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: FunctionDefinitionWire,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinitionWire {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

// ── Request / response ───────────────────────────────────────────────────

/// xAI Chat Completions live-search extension.
///
/// Kept for the type's historical shape. Grok Build never populates this on
/// `/v1/responses` (conversion hard-codes `None`); PhoneBuddy matches that
/// and does not send it on any backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchParameters {
    /// "off" | "on" | "auto"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_citations: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_search_results: Option<i32>,
}

/// Backend-hosted Responses API tools, matching Grok Build `HostedTool`.
///
/// These are spliced into the Responses `tools` array as native types
/// (e.g. `{ "type": "web_search" }`), not as function tools. The client
/// `web_search` function tool is dropped from the same request so the
/// two do not collide (the API rejects duplicates).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostedTool {
    WebSearch,
}

impl HostedTool {
    pub fn wire_name(&self) -> &'static str {
        match self {
            HostedTool::WebSearch => "web_search",
        }
    }

    pub fn to_tool_entry(&self) -> serde_json::Value {
        match self {
            HostedTool::WebSearch => serde_json::json!({ "type": "web_search" }),
        }
    }

    /// Grok Build sends hosted search only on the Responses API when the
    /// model/gateway supports backend search (`supports_backend_search`).
    pub fn for_request(enable_web_search: bool, api_backend: ApiBackend) -> Vec<Self> {
        if enable_web_search && matches!(api_backend, ApiBackend::Responses) {
            vec![Self::WebSearch]
        } else {
            Vec::new()
        }
    }
}

/// Drop function tools whose names collide with hosted tools.
pub fn drop_colliding_function_tools(
    tools: Vec<ToolDefinitionWire>,
    hosted: &[HostedTool],
) -> Option<Vec<ToolDefinitionWire>> {
    let filtered: Vec<_> = tools
        .into_iter()
        .filter(|t| !hosted.iter().any(|h| h.wire_name() == t.function.name))
        .collect();
    if filtered.is_empty() {
        None
    } else {
        Some(filtered)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinitionWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Unused on the wire. Grok Build always sets this to `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_parameters: Option<SearchParameters>,
    /// Responses-only hosted tools. Never serialized on Chat Completions.
    #[serde(skip)]
    pub hosted_tools: Vec<HostedTool>,
    /// Responses `previous_response_id`. Set on same-provider idle-timeout
    /// continuation so encrypted reasoning (`rs_*`) can resume. Never sent
    /// across hosts — the id is bound to the originating gateway.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
}

/// Internal LLM request: canonical items plus sampling / tool config.
///
/// This is the type transports and adapters consume. The host-transport
/// contract still serializes a [`ChatCompletionRequest`] (Chat Completions
/// shape) via `chat_messages_from_items`.
#[derive(Debug, Clone)]
pub struct ConversationRequest {
    pub model: String,
    pub items: Vec<crate::conversation::ConversationItem>,
    pub stream: Option<bool>,
    pub tools: Option<Vec<ToolDefinitionWire>>,
    pub tool_choice: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Unused on the wire. Kept so historical tests and extra-body merges
    /// continue to compile.
    pub search_parameters: Option<SearchParameters>,
    pub hosted_tools: Vec<HostedTool>,
    pub previous_response_id: Option<String>,
    /// Request-scoped image bytes. Never serialized to session storage.
    pub image_bytes: crate::llm::image::ImageBytesStore,
}

impl ConversationRequest {
    pub fn from_chat(req: ChatCompletionRequest) -> Self {
        Self {
            model: req.model,
            items: crate::conversation::items_from_chat_messages(&req.messages),
            stream: req.stream,
            tools: req.tools,
            tool_choice: req.tool_choice,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            search_parameters: req.search_parameters,
            hosted_tools: req.hosted_tools,
            previous_response_id: req.previous_response_id,
            image_bytes: crate::llm::image::ImageBytesStore::default(),
        }
    }

    /// Frozen host-transport contract: Chat Completions JSON, no item types.
    pub fn to_host_chat_request(&self) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: self.model.clone(),
            messages: crate::conversation::chat_messages_from_items(&self.items),
            stream: self.stream,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            search_parameters: self.search_parameters.clone(),
            hosted_tools: self.hosted_tools.clone(),
            previous_response_id: self.previous_response_id.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

// ── Streaming chunk types ────────────────────────────────────────────────
// Shapes ported from grok sampling-types: `ChatCompletionChunk`,
// `ChatChunkChoice`, `ChatChunkDelta`, `ToolCallDelta`.

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatChunkChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ChatChunkChoice {
    #[serde(default)]
    pub index: u32,
    pub delta: ChatChunkDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Streaming delta for a tool call.
///
/// In OpenAI-compatible streaming, tool calls arrive across multiple chunks:
/// the first chunk carries `id` + `function.name` + start of `arguments`,
/// subsequent chunks only carry `index` and an `arguments` fragment.
/// Ported verbatim (with comments) from grok sampling-types.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ToolCallDelta {
    /// Positional index correlating delta chunks of the same tool call.
    #[serde(default)]
    pub index: u32,
    /// Only present in the first chunk for this tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Only present in the first chunk (usually "function").
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// The function name and/or argument fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
    /// Gemini per-part thought signature, captured on the complete call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ToolCallFunctionDelta {
    /// Only present in the first chunk for this tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Argument fragment (may be empty or partial JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ChatChunkDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_items: Vec<ReasoningItem>,
    /// Tool call deltas. Handles `null` in JSON as empty vec.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDelta>,
    /// Terminal Responses `response.output` snapshot. Not a wire field of
    /// Chat Completions chunks; adapters attach it so `finalize_turn` can
    /// treat the completed output as the canonical turn source.
    #[serde(skip)]
    pub final_output: Option<Vec<OutputItemWire>>,
}

/// Typed subset of Responses `response.output[]`. Anything unknown stays
/// raw and becomes a BackendToolCall — never dropped, never guessed at.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputItemWire {
    Message { id: String, text: String },
    Reasoning(ReasoningItem),
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    Backend {
        item_type: String,
        id: String,
        payload: serde_json::Value,
    },
}

/// Parse one Responses `response.output[]` element by `"type"`.
pub fn parse_output_item(v: &serde_json::Value) -> Option<OutputItemWire> {
    let ty = v.get("type").and_then(|s| s.as_str())?;
    match ty {
        "message" => {
            let id = v
                .get("id")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let mut text = String::new();
            if let Some(parts) = v.get("content").and_then(|c| c.as_array()) {
                for part in parts {
                    let part_ty = part.get("type").and_then(|s| s.as_str()).unwrap_or("");
                    if part_ty == "output_text" || part_ty == "text" || part_ty.is_empty() {
                        if let Some(t) = part.get("text").and_then(|s| s.as_str()) {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                    }
                }
            } else if let Some(t) = v.get("content").and_then(|s| s.as_str()) {
                text = t.to_string();
            }
            Some(OutputItemWire::Message { id, text })
        }
        "reasoning" => serde_json::from_value::<ReasoningItem>(v.clone())
            .ok()
            .map(OutputItemWire::Reasoning),
        "function_call" => {
            let call_id = v
                .get("call_id")
                .or_else(|| v.get("id"))
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let name = v
                .get("name")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = v
                .get("arguments")
                .map(|a| {
                    if let Some(s) = a.as_str() {
                        s.to_string()
                    } else {
                        a.to_string()
                    }
                })
                .unwrap_or_default();
            Some(OutputItemWire::FunctionCall {
                call_id,
                name,
                arguments,
            })
        }
        _ => {
            let id = v
                .get("id")
                .or_else(|| v.get("call_id"))
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            Some(OutputItemWire::Backend {
                item_type: ty.to_string(),
                id,
                payload: v.clone(),
            })
        }
    }
}

// ── Collected (non-streaming view) ───────────────────────────────────────

/// Result of accumulating one streamed assistant turn.
#[derive(Debug, Clone, Default)]
pub struct CollectedTurn {
    /// Canonical ordered items for this assistant turn
    /// (`[Reasoning | BackendToolCall]* Assistant`).
    pub items: Vec<crate::conversation::ConversationItem>,
    /// Derived views filled from `items` at `finalize_turn`. Kept during
    /// the dual-representation window so existing call sites compile.
    pub text: String,
    pub reasoning: String,
    pub reasoning_items: Vec<ReasoningItem>,
    pub encrypted_reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    pub model: String,
    /// Responses `resp_*` id captured from `response.created` / chunk `id`.
    pub response_id: Option<String>,
    /// Terminal `response.output` snapshot, consumed by `finalize_turn`.
    pub final_output: Option<Vec<OutputItemWire>>,
}

impl CollectedTurn {
    /// True when the model emitted no text and no tool calls.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.tool_calls.is_empty()
    }

    /// Concatenated assistant text from `items`.
    pub fn text_from_items(&self) -> String {
        self.items
            .iter()
            .filter_map(|i| i.as_assistant().map(|a| a.content.as_str()))
            .collect::<Vec<_>>()
            .join("")
    }

    /// Client function calls from the trailing assistant item.
    pub fn client_tool_calls(&self) -> Vec<ToolCall> {
        self.items
            .iter()
            .rev()
            .find_map(|i| i.as_assistant().map(|a| a.tool_calls.clone()))
            .unwrap_or_default()
    }

    /// Reasoning siblings from `items`.
    pub fn reasoning_from_items(&self) -> Vec<ReasoningItem> {
        self.items
            .iter()
            .filter_map(|i| match i {
                crate::conversation::ConversationItem::Reasoning(r) => Some(r.clone()),
                _ => None,
            })
            .collect()
    }

    /// Refresh derived views from `items` (test 1.5 invariant).
    pub fn sync_derived_views(&mut self) {
        self.text = self.text_from_items();
        let mut calls: Vec<ToolCall> = self
            .items
            .iter()
            .filter_map(|i| match i {
                crate::conversation::ConversationItem::BackendToolCall(b) => {
                    Some(crate::conversation::backend_to_legacy_tool_call(b))
                }
                _ => None,
            })
            .collect();
        calls.extend(self.client_tool_calls());
        self.tool_calls = calls;
        self.reasoning_items = self.reasoning_from_items();
        if self.encrypted_reasoning.is_none() {
            self.encrypted_reasoning = self
                .reasoning_items
                .iter()
                .find_map(|r| r.encrypted_content.clone())
                .or_else(|| {
                    self.items.iter().find_map(|i| {
                        i.as_assistant()
                            .and_then(|a| a.encrypted_reasoning.clone())
                    })
                });
        }
        if self.reasoning.is_empty() {
            self.reasoning = self
                .reasoning_items
                .iter()
                .map(reasoning_item_text)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
        }
    }
}

#[cfg(test)]
mod output_item_tests {
    use super::*;

    #[test]
    fn parse_output_item_known_types() {
        let message = serde_json::json!({
            "type": "message",
            "id": "msg_1",
            "content": [
                {"type": "output_text", "text": "hello"},
                {"type": "output_text", "text": "world"}
            ]
        });
        match parse_output_item(&message) {
            Some(OutputItemWire::Message { id, text }) => {
                assert_eq!(id, "msg_1");
                assert_eq!(text, "hello\nworld");
            }
            other => panic!("expected message, got {other:?}"),
        }

        let reasoning = serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "think"}],
            "encrypted_content": "enc"
        });
        match parse_output_item(&reasoning) {
            Some(OutputItemWire::Reasoning(r)) => {
                assert_eq!(r.id, "rs_1");
                assert_eq!(r.encrypted_content.as_deref(), Some("enc"));
            }
            other => panic!("expected reasoning, got {other:?}"),
        }

        let fc = serde_json::json!({
            "type": "function_call",
            "call_id": "c1",
            "name": "read_file",
            "arguments": "{\"path\":\"a\"}"
        });
        match parse_output_item(&fc) {
            Some(OutputItemWire::FunctionCall { call_id, name, arguments }) => {
                assert_eq!(call_id, "c1");
                assert_eq!(name, "read_file");
                assert_eq!(arguments, "{\"path\":\"a\"}");
            }
            other => panic!("expected function_call, got {other:?}"),
        }

        let ws = serde_json::json!({
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "apus"}
        });
        match parse_output_item(&ws) {
            Some(OutputItemWire::Backend { item_type, id, payload }) => {
                assert_eq!(item_type, "web_search_call");
                assert_eq!(id, "ws_1");
                assert_eq!(payload, ws);
            }
            other => panic!("expected backend, got {other:?}"),
        }

        let future = serde_json::json!({
            "type": "future_call",
            "id": "fut_1",
            "foo": "bar"
        });
        match parse_output_item(&future) {
            Some(OutputItemWire::Backend { item_type, id, payload }) => {
                assert_eq!(item_type, "future_call");
                assert_eq!(id, "fut_1");
                assert_eq!(payload, future);
            }
            other => panic!("expected backend future_call, got {other:?}"),
        }
    }
}
