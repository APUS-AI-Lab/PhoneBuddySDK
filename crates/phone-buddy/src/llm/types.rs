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
}

impl ApiBackend {
    pub fn supports_native_schema(&self) -> bool {
        matches!(self, Self::ChatCompletions | Self::Responses)
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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

// ── Messages ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
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
            tc.function.arguments = sanitize_tool_arguments(
                &tc.id,
                &tc.function.name,
                &tc.function.arguments,
            );
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
        if crate::llm::failover::should_strip_origin(out.origin.as_deref(), target, primary)
        {
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

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(text.into()),
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
            content: Some(text.into()),
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
            content: Some(text.into()),
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
            content: Some(text.into()),
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
            content: text,
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
            content: text,
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
            content: Some(output.into()),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

// ── Collected (non-streaming view) ───────────────────────────────────────

/// Result of accumulating one streamed assistant turn.
#[derive(Debug, Clone, Default)]
pub struct CollectedTurn {
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
}

impl CollectedTurn {
    /// True when the model emitted no text and no tool calls.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.tool_calls.is_empty()
    }
}
