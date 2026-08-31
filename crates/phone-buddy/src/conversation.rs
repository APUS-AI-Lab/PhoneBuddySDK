//! Canonical conversation representation: an ordered list of heterogeneous items.
//!
//! Wire adapters down-convert from this shape. The legacy [`ChatMessage`]
//! list is retained only for host-transport and v1 session compatibility.

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{EngineError, EngineResult};
use crate::llm::types::{
    reasoning_item_text, ChatMessage, MessageContent, ReasoningItem, Role, ToolCall,
    ToolCallFunction,
};

/// Max processed image long edge accepted by the SDK (e.g. 1024x768 landscape or 768x1024 portrait).
pub const MAX_IMAGE_LONG_EDGE: u32 = 1024;
/// Max processed image short edge accepted by the SDK.
pub const MAX_IMAGE_SHORT_EDGE: u32 = 768;

/// Max processed image width accepted by the SDK (landscape baseline).
pub const MAX_IMAGE_WIDTH: u32 = MAX_IMAGE_LONG_EDGE;
/// Max processed image height accepted by the SDK (landscape baseline).
pub const MAX_IMAGE_HEIGHT: u32 = MAX_IMAGE_SHORT_EDGE;
/// Max image parts on a single user turn.
pub const MAX_IMAGES_PER_TURN: usize = 5;
/// Max audio parts on a single user turn.
pub const MAX_AUDIOS_PER_TURN: usize = 5;
/// Max audio bytes accepted per file (25MB).
pub const MAX_AUDIO_BYTES: u64 = 25 * 1024 * 1024;

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
    /// Server-executed hosted tool call (`web_search_call`, `custom_tool_call`, …), stored as the
    /// raw wire item so Responses replay is verbatim.
    BackendToolCall(BackendToolCallItem),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemItem {
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageMimeType {
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/png")]
    Png,
}

impl ImageMimeType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioMimeType {
    #[serde(rename = "audio/wav", alias = "audio/x-wav")]
    Wav,
    #[serde(rename = "audio/mp3", alias = "audio/mpeg")]
    Mp3,
    #[serde(rename = "audio/ogg")]
    Ogg,
    #[serde(rename = "audio/m4a", alias = "audio/mp4", alias = "audio/x-m4a")]
    M4a,
    #[serde(rename = "audio/aac")]
    Aac,
    #[serde(rename = "audio/flac", alias = "audio/x-flac")]
    Flac,
    #[serde(rename = "audio/webm")]
    Webm,
}

impl AudioMimeType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mp3",
            Self::Ogg => "audio/ogg",
            Self::M4a => "audio/m4a",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Webm => "audio/webm",
        }
    }

    pub fn from_mime_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "audio/wav" | "audio/x-wav" | "wav" => Some(Self::Wav),
            "audio/mp3" | "audio/mpeg" | "mp3" => Some(Self::Mp3),
            "audio/ogg" | "ogg" => Some(Self::Ogg),
            "audio/m4a" | "audio/mp4" | "audio/x-m4a" | "m4a" => Some(Self::M4a),
            "audio/aac" | "aac" => Some(Self::Aac),
            "audio/flac" | "audio/x-flac" | "flac" => Some(Self::Flac),
            "audio/webm" | "webm" => Some(Self::Webm),
            _ => None,
        }
    }

    pub fn default_extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Ogg => "ogg",
            Self::M4a => "m4a",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Webm => "webm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
    Original,
}

impl ImageDetail {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::High => "high",
            Self::Original => "original",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContentPart {
    Text {
        text: String,
    },
    Image {
        attachment_id: String,
        local_path: String,
        mime_type: ImageMimeType,
        byte_size: u64,
        width: u32,
        height: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    Audio {
        attachment_id: String,
        local_path: String,
        mime_type: AudioMimeType,
        byte_size: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
}

/// Ordered user content. Legacy sessions with `content: "…"` migrate on load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UserItem {
    pub parts: Vec<UserContentPart>,
}

impl<'de> Deserialize<'de> for UserItem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            parts: Option<Vec<UserContentPart>>,
            #[serde(default)]
            content: Option<String>,
        }
        let raw = Raw::deserialize(deserializer)?;
        if let Some(parts) = raw.parts {
            Ok(UserItem { parts })
        } else if let Some(content) = raw.content {
            Ok(UserItem {
                parts: vec![UserContentPart::Text { text: content }],
            })
        } else {
            Ok(UserItem { parts: Vec::new() })
        }
    }
}

impl UserItem {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            parts: vec![UserContentPart::Text {
                text: content.into(),
            }],
        }
    }

    pub fn text_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                UserContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn has_images(&self) -> bool {
        self.parts
            .iter()
            .any(|p| matches!(p, UserContentPart::Image { .. }))
    }

    pub fn image_count(&self) -> usize {
        self.parts
            .iter()
            .filter(|p| matches!(p, UserContentPart::Image { .. }))
            .count()
    }

    pub fn has_audio(&self) -> bool {
        self.parts
            .iter()
            .any(|p| matches!(p, UserContentPart::Audio { .. }))
    }

    pub fn audio_count(&self) -> usize {
        self.parts
            .iter()
            .filter(|p| matches!(p, UserContentPart::Audio { .. }))
            .count()
    }

    pub fn has_media(&self) -> bool {
        self.has_images() || self.has_audio()
    }

    pub fn normalized_parts(&self) -> Vec<&UserContentPart> {
        let mut media = Vec::new();
        let mut texts = Vec::new();
        for p in &self.parts {
            match p {
                UserContentPart::Image { .. } | UserContentPart::Audio { .. } => media.push(p),
                UserContentPart::Text { text } if !text.trim().is_empty() => texts.push(p),
                UserContentPart::Text { .. } => {}
            }
        }
        media.extend(texts);
        media
    }

    pub fn validate_shape(&self) -> EngineResult<()> {
        let images = self.image_count();
        if images > MAX_IMAGES_PER_TURN {
            return Err(EngineError::TooManyImages(images));
        }
        let audios = self.audio_count();
        if audios > MAX_AUDIOS_PER_TURN {
            return Err(EngineError::TooManyAudio(audios));
        }
        let has_text = self
            .parts
            .iter()
            .any(|p| matches!(p, UserContentPart::Text { text } if !text.trim().is_empty()));
        if images == 0 && audios == 0 && !has_text {
            return Err(EngineError::InvalidUserTurn(
                "turn has no text, image, or audio parts".into(),
            ));
        }
        for p in &self.parts {
            match p {
                UserContentPart::Image {
                    width,
                    height,
                    attachment_id,
                    ..
                } => {
                    if *width == 0 || *height == 0 {
                        return Err(EngineError::AttachmentInvalid(
                            attachment_id.clone(),
                            "invalid dimensions".into(),
                        ));
                    }
                    let max_dim = (*width).max(*height);
                    let min_dim = (*width).min(*height);
                    if max_dim > MAX_IMAGE_LONG_EDGE || min_dim > MAX_IMAGE_SHORT_EDGE {
                        return Err(EngineError::AttachmentInvalid(
                            attachment_id.clone(),
                            format!("exceeds {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}"),
                        ));
                    }
                }
                UserContentPart::Audio {
                    attachment_id,
                    byte_size,
                    ..
                } => {
                    if *byte_size == 0 {
                        return Err(EngineError::AttachmentInvalid(
                            attachment_id.clone(),
                            "empty audio file (0 bytes)".into(),
                        ));
                    }
                    if *byte_size > MAX_AUDIO_BYTES {
                        return Err(EngineError::AttachmentInvalid(
                            attachment_id.clone(),
                            format!(
                                "audio exceeds maximum size ({} > {MAX_AUDIO_BYTES})",
                                byte_size
                            ),
                        ));
                    }
                }
                UserContentPart::Text { .. } => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct UserTurnV2Wire {
    schema_version: u32,
    #[serde(rename = "type")]
    kind: String,
    parts: Vec<UserContentPart>,
}

/// Parse a versioned `pb_engine_chat_v2` turn JSON. No text fallback.
pub fn parse_user_turn_v2(json: &str) -> EngineResult<UserItem> {
    let turn: UserTurnV2Wire =
        serde_json::from_str(json).map_err(|e| EngineError::InvalidUserTurn(e.to_string()))?;
    if turn.schema_version != 1 {
        return Err(EngineError::InvalidUserTurn(format!(
            "unsupported schema_version {}",
            turn.schema_version
        )));
    }
    if turn.kind != "user_turn" {
        return Err(EngineError::InvalidUserTurn(format!(
            "unexpected type {}",
            turn.kind
        )));
    }
    let item = UserItem { parts: turn.parts };
    item.validate_shape()?;
    Ok(item)
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

impl BackendToolCallItem {
    /// Returns the descriptive tool name (e.g. `x_thread_fetch`, `x_keyword_search`, `web_search`, `x_search`).
    pub fn display_name(&self) -> String {
        if let Some(name) = self.payload.get("name").and_then(|v| v.as_str()) {
            if !name.is_empty() {
                return name.to_string();
            }
        }
        server_tool_function_name(&self.item_type)
    }

    pub fn status(&self) -> Option<&str> {
        self.payload.get("status").and_then(|v| v.as_str())
    }

    /// Hosted `{type: web_search}` that the server did not finish.
    ///
    /// grok-build keeps the Responses stream open until search completes.
    /// Buffering proxies often close after the `web_search_call` lands
    /// with `status: in_progress` and empty `sources`. Those turns must
    /// not be treated as a final answer.
    pub fn is_unfinished_hosted_search(&self) -> bool {
        let name = self.display_name();
        if name != "web_search" && self.item_type != "web_search_call" {
            return false;
        }
        match self.status() {
            Some("completed") | Some("failed") => false,
            Some("in_progress") | Some("incomplete") | Some("searching") => true,
            _ => self
                .payload
                .pointer("/action/sources")
                .and_then(|v| v.as_array())
                .is_some_and(|a| a.is_empty()),
        }
    }

    /// Client `web_search` function call that can finish a truncated hosted search.
    pub fn to_client_web_search_call(&self) -> ToolCall {
        let query = self
            .payload
            .pointer("/action/query")
            .or_else(|| self.payload.pointer("/arguments/query"))
            .or_else(|| self.payload.get("query"));
        let arguments = match query {
            Some(q) => serde_json::json!({ "query": q }).to_string(),
            None => self
                .payload
                .get("action")
                .or_else(|| self.payload.get("arguments"))
                .map(|v| {
                    if let Some(s) = v.as_str() {
                        s.to_string()
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_else(|| "{}".into()),
        };
        let id = if self.id.is_empty() {
            "ws_salvage".to_string()
        } else {
            self.id.clone()
        };
        ToolCall {
            id,
            kind: "function".into(),
            function: ToolCallFunction {
                name: "web_search".into(),
                arguments,
            },
            thought_signature: None,
        }
    }
}

/// Rewrite unfinished hosted `web_search_call` items into client function
/// calls on the assistant item so the engine can execute DuckDuckGo /
/// hosted-search fallback instead of stopping the turn.
pub fn salvage_unfinished_hosted_searches(items: &mut Vec<ConversationItem>) -> Vec<ToolCall> {
    let mut salvaged = Vec::new();
    items.retain(|item| {
        if let ConversationItem::BackendToolCall(b) = item {
            if b.is_unfinished_hosted_search() {
                salvaged.push(b.to_client_web_search_call());
                return false;
            }
        }
        true
    });
    if salvaged.is_empty() {
        return salvaged;
    }
    if let Some(a) = items.iter_mut().rev().find_map(|i| i.as_assistant_mut()) {
        a.tool_calls.extend(salvaged.clone());
    } else {
        items.push(ConversationItem::Assistant(AssistantItem {
            content: String::new(),
            tool_calls: salvaged.clone(),
            reasoning_content: None,
            encrypted_reasoning: None,
            origin: None,
        }));
    }
    salvaged
}

impl ConversationItem {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(SystemItem {
            content: content.into(),
        })
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User(UserItem::text(content))
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
pub fn reconstruct_backend_payload(
    item_type: &str,
    id: &str,
    arguments: &str,
) -> serde_json::Value {
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
                out.push(ConversationItem::system(msg.content_text()));
            }
            Role::User => {
                out.push(ConversationItem::user(msg.content_text()));
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
                        let payload =
                            reconstruct_backend_payload(&item_type, &tc.id, &tc.function.arguments);
                        out.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                            item_type,
                            id: tc.id.clone(),
                            payload,
                        }));
                    } else {
                        client_calls.push(tc.clone());
                    }
                }

                let content = msg.content_text();
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
                    msg.content_text(),
                ));
            }
        }
    }
    out
}

fn user_item_to_chat_message(u: &UserItem) -> ChatMessage {
    // Image bytes are injected at wire time from `ImageBytesStore`. This
    // down-conversion keeps plaintext so v1 consumers still compile.
    ChatMessage::user(u.text_content())
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
                out.push(user_item_to_chat_message(u));
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
                        Some(MessageContent::text(a.content.clone()))
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
        .filter(|i| {
            matches!(
                i,
                ConversationItem::User(_) | ConversationItem::Assistant(_)
            )
        })
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
    fn legacy_user_content_migrates_to_text_part() {
        let json = r#"{"type":"user","content":"hello"}"#;
        let item: ConversationItem = serde_json::from_str(json).unwrap();
        match item {
            ConversationItem::User(u) => {
                assert_eq!(u.parts.len(), 1);
                assert_eq!(u.text_content(), "hello");
            }
            other => panic!("expected user, got {other:?}"),
        }
        let encoded = serde_json::to_value(&ConversationItem::user("hello")).unwrap();
        assert!(encoded.get("content").is_none());
        assert_eq!(encoded["parts"][0]["type"], "text");
        assert_eq!(encoded["parts"][0]["text"], "hello");
    }

    #[test]
    fn parse_user_turn_v2_accepts_image_then_text() {
        let json = serde_json::json!({
            "schema_version": 1,
            "type": "user_turn",
            "parts": [
                {
                    "type": "image",
                    "attachment_id": "img_1",
                    "local_path": "/tmp/a.jpg",
                    "mime_type": "image/jpeg",
                    "byte_size": 12,
                    "width": 800,
                    "height": 600,
                    "detail": "auto"
                },
                {"type": "text", "text": "这是什么"}
            ]
        });
        let item = parse_user_turn_v2(&json.to_string()).unwrap();
        assert_eq!(item.image_count(), 1);
        assert_eq!(item.text_content(), "这是什么");
        assert_eq!(item.normalized_parts().len(), 2);
    }

    #[test]
    fn parse_user_turn_v2_rejects_unknown_schema_and_too_many_images() {
        let bad_ver =
            r#"{"schema_version":2,"type":"user_turn","parts":[{"type":"text","text":"hi"}]}"#;
        match parse_user_turn_v2(bad_ver) {
            Err(EngineError::InvalidUserTurn(msg)) => assert!(msg.contains("schema_version")),
            other => panic!("{other:?}"),
        }
        let mut parts = Vec::new();
        for i in 0..6 {
            parts.push(serde_json::json!({
                "type": "image",
                "attachment_id": format!("img_{i}"),
                "local_path": "/tmp/a.jpg",
                "mime_type": "image/jpeg",
                "byte_size": 1,
                "width": 10,
                "height": 10
            }));
        }
        let too_many = serde_json::json!({
            "schema_version": 1,
            "type": "user_turn",
            "parts": parts
        });
        match parse_user_turn_v2(&too_many.to_string()) {
            Err(EngineError::TooManyImages(6)) => {}
            other => panic!("{other:?}"),
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

    #[test]
    fn image_dimension_validation_supports_orientation_adaptability() {
        // Landscape (1024x768) -> Valid
        let landscape_turn = serde_json::json!({
            "schema_version": 1,
            "type": "user_turn",
            "parts": [
                {
                    "type": "image",
                    "attachment_id": "img_landscape",
                    "local_path": "/tmp/landscape.jpg",
                    "mime_type": "image/jpeg",
                    "byte_size": 100,
                    "width": 1024,
                    "height": 768
                },
                {"type": "text", "text": "look at landscape"}
            ]
        });
        assert!(parse_user_turn_v2(&landscape_turn.to_string()).is_ok());

        // Portrait / Rotated 90 degrees (768x1024) -> Valid
        let portrait_turn = serde_json::json!({
            "schema_version": 1,
            "type": "user_turn",
            "parts": [
                {
                    "type": "image",
                    "attachment_id": "img_portrait",
                    "local_path": "/tmp/portrait.jpg",
                    "mime_type": "image/jpeg",
                    "byte_size": 100,
                    "width": 768,
                    "height": 1024
                },
                {"type": "text", "text": "look at portrait"}
            ]
        });
        assert!(parse_user_turn_v2(&portrait_turn.to_string()).is_ok());

        // Exceeds long edge (1025x768 or 768x1025) -> Invalid
        let exceeds_long = serde_json::json!({
            "schema_version": 1,
            "type": "user_turn",
            "parts": [
                {
                    "type": "image",
                    "attachment_id": "img_toolong",
                    "local_path": "/tmp/toolong.jpg",
                    "mime_type": "image/jpeg",
                    "byte_size": 100,
                    "width": 768,
                    "height": 1025
                },
                {"type": "text", "text": "too long"}
            ]
        });
        assert!(matches!(
            parse_user_turn_v2(&exceeds_long.to_string()),
            Err(EngineError::AttachmentInvalid(_, msg)) if msg.contains("exceeds 1024x768")
        ));

        // Exceeds short edge (800x800) -> Invalid
        let exceeds_short = serde_json::json!({
            "schema_version": 1,
            "type": "user_turn",
            "parts": [
                {
                    "type": "image",
                    "attachment_id": "img_toowide",
                    "local_path": "/tmp/toowide.jpg",
                    "mime_type": "image/jpeg",
                    "byte_size": 100,
                    "width": 800,
                    "height": 800
                },
                {"type": "text", "text": "too wide"}
            ]
        });
        assert!(matches!(
            parse_user_turn_v2(&exceeds_short.to_string()),
            Err(EngineError::AttachmentInvalid(_, msg)) if msg.contains("exceeds 1024x768")
        ));
    }

    fn backend_search(id: &str, payload: serde_json::Value) -> ConversationItem {
        ConversationItem::BackendToolCall(BackendToolCallItem {
            item_type: "web_search_call".into(),
            id: id.into(),
            payload,
        })
    }

    #[test]
    fn unfinished_hosted_search_detects_in_progress_and_empty_sources() {
        let in_progress = BackendToolCallItem {
            item_type: "web_search_call".into(),
            id: "ws_1".into(),
            payload: serde_json::json!({
                "type": "web_search_call",
                "id": "ws_1",
                "status": "in_progress",
                "action": {"type": "search", "query": "today news", "sources": []}
            }),
        };
        assert!(in_progress.is_unfinished_hosted_search());
        let call = in_progress.to_client_web_search_call();
        assert_eq!(call.id, "ws_1");
        assert_eq!(call.kind, "function");
        assert_eq!(call.function.name, "web_search");
        assert_eq!(call.function.arguments, r#"{"query":"today news"}"#);

        let completed = BackendToolCallItem {
            item_type: "web_search_call".into(),
            id: "ws_2".into(),
            payload: serde_json::json!({
                "type": "web_search_call",
                "id": "ws_2",
                "status": "completed",
                "action": {"type": "search", "query": "today news", "sources": []}
            }),
        };
        assert!(!completed.is_unfinished_hosted_search());

        let reconstructed_empty_sources = BackendToolCallItem {
            item_type: "web_search_call".into(),
            id: "ws_3".into(),
            payload: serde_json::json!({
                "type": "web_search_call",
                "id": "ws_3",
                "action": {"type": "search", "query": "today news", "sources": []}
            }),
        };
        assert!(reconstructed_empty_sources.is_unfinished_hosted_search());

        let reconstructed_query_only = BackendToolCallItem {
            item_type: "web_search_call".into(),
            id: "ws_4".into(),
            payload: serde_json::json!({
                "type": "web_search_call",
                "id": "ws_4",
                "action": {"query": "today news"}
            }),
        };
        assert!(!reconstructed_query_only.is_unfinished_hosted_search());
    }

    #[test]
    fn salvage_moves_unfinished_hosted_search_onto_assistant_tool_calls() {
        let mut items = vec![
            backend_search(
                "ws_1",
                serde_json::json!({
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "in_progress",
                    "action": {"type": "search", "query": "today news", "sources": []}
                }),
            ),
            ConversationItem::assistant("先搜集今天的新闻。"),
        ];
        let salvaged = salvage_unfinished_hosted_searches(&mut items);
        assert_eq!(salvaged.len(), 1);
        assert_eq!(salvaged[0].function.arguments, r#"{"query":"today news"}"#);
        assert!(items
            .iter()
            .all(|i| !matches!(i, ConversationItem::BackendToolCall(_))));
        let assistant = items
            .iter()
            .find_map(|i| i.as_assistant())
            .expect("assistant");
        assert_eq!(assistant.content, "先搜集今天的新闻。");
        assert_eq!(assistant.tool_calls.len(), 1);
        assert_eq!(assistant.tool_calls[0].id, "ws_1");
        assert_eq!(assistant.tool_calls[0].function.name, "web_search");
    }

    #[test]
    fn salvage_leaves_completed_hosted_search_in_place() {
        let mut items = vec![
            backend_search(
                "ws_ok",
                serde_json::json!({
                    "type": "web_search_call",
                    "id": "ws_ok",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "today news",
                        "sources": [{"type": "url", "url": "https://example.com"}]
                    }
                }),
            ),
            ConversationItem::assistant("Here is the latest news."),
        ];
        let salvaged = salvage_unfinished_hosted_searches(&mut items);
        assert!(salvaged.is_empty());
        assert!(matches!(items[0], ConversationItem::BackendToolCall(_)));
        assert!(items[1].as_assistant().unwrap().tool_calls.is_empty());
    }
}
