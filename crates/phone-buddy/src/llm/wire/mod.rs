//! Down-conversion adapters: canonical items → wire protocol JSON.
//!
//! Adapters are the only code that produces provider payloads (I5).

pub mod chat_completions;
pub mod gemini;
pub mod messages;
pub mod responses;

#[cfg(test)]
mod conformance_tests;

use crate::conversation::{ImageDetail, UserContentPart, UserItem};
use crate::error::{EngineError, EngineResult};
use crate::llm::types::{ChatCompletionChunk, ConversationRequest};

/// One protocol's request builder + SSE parser.
pub(crate) trait WireAdapter {
    fn endpoint(&self, base: &str, model: &str, stream: bool) -> String;
    fn build_payload(&self, req: &ConversationRequest) -> EngineResult<serde_json::Value>;
    /// One SSE frame → zero or one internal chunk (unchanged chunk model).
    fn parse_event(&self, event: &str, data: &str) -> EngineResult<Option<ChatCompletionChunk>>;
}

fn require_image(
    req: &ConversationRequest,
    attachment_id: &str,
) -> EngineResult<crate::llm::image::MaterializedImage> {
    req.image_bytes
        .get(attachment_id)
        .ok_or_else(|| EngineError::AttachmentMissing(attachment_id.to_string()))
}

fn require_audio(
    req: &ConversationRequest,
    attachment_id: &str,
) -> EngineResult<crate::llm::image::MaterializedAudio> {
    req.audio_bytes
        .get(attachment_id)
        .ok_or_else(|| EngineError::AttachmentMissing(attachment_id.to_string()))
}

/// OpenAI Responses user `content`: string, or `[input_image, input_audio, input_text]`.
pub fn responses_user_content(
    u: &UserItem,
    req: &ConversationRequest,
) -> EngineResult<serde_json::Value> {
    if !u.has_media() {
        return Ok(serde_json::Value::String(u.text_content()));
    }
    let mut content = Vec::new();
    for p in u.normalized_parts() {
        match p {
            UserContentPart::Image {
                attachment_id,
                detail,
                ..
            } => {
                let img = require_image(req, attachment_id)?;
                let d = detail.unwrap_or(ImageDetail::Auto);
                content.push(serde_json::json!({
                    "type": "input_image",
                    "image_url": img.data_url(),
                    "detail": d.as_str(),
                }));
            }
            UserContentPart::Audio { attachment_id, .. } => {
                let audio = require_audio(req, attachment_id)?;
                content.push(serde_json::json!({
                    "type": "input_audio",
                    "audio_url": audio.data_url(),
                }));
            }
            UserContentPart::Text { text } => {
                content.push(serde_json::json!({
                    "type": "input_text",
                    "text": text,
                }));
            }
        }
    }
    Ok(serde_json::Value::Array(content))
}

/// Chat Completions user `content`: string, or `[image_url, input_audio, text]`.
pub fn chat_completions_user_content(
    u: &UserItem,
    req: &ConversationRequest,
) -> EngineResult<serde_json::Value> {
    if !u.has_media() {
        return Ok(serde_json::Value::String(u.text_content()));
    }
    let mut content = Vec::new();
    for p in u.normalized_parts() {
        match p {
            UserContentPart::Image {
                attachment_id,
                detail,
                ..
            } => {
                let img = require_image(req, attachment_id)?;
                let d = detail.unwrap_or(ImageDetail::Auto);
                content.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": {
                        "url": img.data_url(),
                        "detail": d.as_str(),
                    }
                }));
            }
            UserContentPart::Audio { attachment_id, .. } => {
                let audio = require_audio(req, attachment_id)?;
                content.push(serde_json::json!({
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio.raw_b64(),
                        "format": audio.format_str(),
                    }
                }));
            }
            UserContentPart::Text { text } => {
                content.push(serde_json::json!({
                    "type": "text",
                    "text": text,
                }));
            }
        }
    }
    Ok(serde_json::Value::Array(content))
}

/// Anthropic Messages user `content` blocks. Images use raw base64.
pub fn messages_user_content(
    u: &UserItem,
    req: &ConversationRequest,
) -> EngineResult<serde_json::Value> {
    if !u.has_media() {
        return Ok(serde_json::Value::String(u.text_content()));
    }
    let mut content = Vec::new();
    for p in u.normalized_parts() {
        match p {
            UserContentPart::Image { attachment_id, .. } => {
                let img = require_image(req, attachment_id)?;
                content.push(serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime_type.as_str(),
                        "data": img.raw_b64(),
                    }
                }));
            }
            UserContentPart::Audio { attachment_id, .. } => {
                let audio = require_audio(req, attachment_id)?;
                content.push(serde_json::json!({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": audio.mime_type.as_str(),
                        "data": audio.raw_b64(),
                    }
                }));
            }
            UserContentPart::Text { text } => {
                content.push(serde_json::json!({
                    "type": "text",
                    "text": text,
                }));
            }
        }
    }
    Ok(serde_json::Value::Array(content))
}

/// Gemini generateContent user parts. Images use camelCase `inlineData`.
pub fn gemini_user_parts(
    u: &UserItem,
    req: &ConversationRequest,
) -> EngineResult<Vec<serde_json::Value>> {
    let mut parts = Vec::new();
    for p in u.normalized_parts() {
        match p {
            UserContentPart::Image { attachment_id, .. } => {
                let img = require_image(req, attachment_id)?;
                parts.push(serde_json::json!({
                    "inlineData": {
                        "mimeType": img.mime_type.as_str(),
                        "data": img.raw_b64(),
                    }
                }));
            }
            UserContentPart::Audio { attachment_id, .. } => {
                let audio = require_audio(req, attachment_id)?;
                parts.push(serde_json::json!({
                    "inlineData": {
                        "mimeType": audio.mime_type.as_str(),
                        "data": audio.raw_b64(),
                    }
                }));
            }
            UserContentPart::Text { text } => {
                parts.push(serde_json::json!({ "text": text }));
            }
        }
    }
    if parts.is_empty() {
        parts.push(serde_json::json!({ "text": "" }));
    }
    Ok(parts)
}

/// Strip JSON-Schema keywords Gemini's OpenAPI subset rejects, and map
/// `oneOf` to `anyOf`. Lossless for the supported subset.
pub fn sanitize_json_schema_for_gemini(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(mut map) => {
            map.remove("additionalProperties");
            map.remove("$schema");
            map.remove("$id");
            map.remove("$comment");
            if let Some(one_of) = map.remove("oneOf") {
                map.entry("anyOf").or_insert(one_of);
            }
            let keys: Vec<String> = map.keys().cloned().collect();
            for k in keys {
                if let Some(v) = map.remove(&k) {
                    map.insert(k, sanitize_json_schema_for_gemini(v));
                }
            }
            serde_json::Value::Object(map)
        }
        serde_json::Value::Array(arr) => serde_json::Value::Array(
            arr.into_iter()
                .map(sanitize_json_schema_for_gemini)
                .collect(),
        ),
        other => other,
    }
}

/// Parse stored tool-call argument JSON into an object. Invalid or
/// non-object JSON degrades to `{}`.
pub fn args_as_object(arguments: &str) -> serde_json::Value {
    let parsed = serde_json::from_str::<serde_json::Value>(arguments).ok();
    match parsed {
        Some(serde_json::Value::Object(_)) => parsed.unwrap(),
        _ => serde_json::json!({}),
    }
}

pub(crate) fn adapter_for(backend: crate::llm::types::ApiBackend) -> Box<dyn WireAdapter + Send + Sync> {
    match backend {
        crate::llm::types::ApiBackend::ChatCompletions => {
            Box::new(chat_completions::ChatCompletionsAdapter)
        }
        crate::llm::types::ApiBackend::Responses => Box::new(responses::ResponsesAdapter),
        crate::llm::types::ApiBackend::Messages => Box::new(messages::MessagesAdapter),
        crate::llm::types::ApiBackend::Gemini => Box::new(gemini::GeminiAdapter),
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    #[test]
    fn multimodal_user_turn_golden_payloads() {
        use crate::conversation::{
            ConversationItem, ImageDetail, ImageMimeType, UserContentPart, UserItem,
        };
        use crate::llm::image::{ImageBytesStore, MaterializedImage};

        let store = ImageBytesStore::default();
        store.insert(MaterializedImage {
            attachment_id: "img_1".into(),
            mime_type: ImageMimeType::Jpeg,
            bytes: b"fake-jpeg-bytes".to_vec(),
            detail: Some(ImageDetail::Auto),
            width: 1024,
            height: 768,
        });
        let item = UserItem {
            parts: vec![
                UserContentPart::Image {
                    attachment_id: "img_1".into(),
                    local_path: "/private/objects/img_1.jpg".into(),
                    mime_type: ImageMimeType::Jpeg,
                    byte_size: 15,
                    width: 1024,
                    height: 768,
                    detail: Some(ImageDetail::Auto),
                },
                UserContentPart::Text {
                    text: "这是什么".into(),
                },
            ],
        };
        let req = ConversationRequest {
            model: "vision".into(),
            items: vec![ConversationItem::User(item.clone())],
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
            audio_bytes: crate::llm::image::AudioBytesStore::default(),
        };

        let responses = super::responses::build_responses_payload(&req).unwrap();
        let resp_content = &responses["input"][0]["content"];
        assert_eq!(resp_content[0]["type"], "input_image");
        assert!(resp_content[0]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,"));
        assert_eq!(resp_content[0]["detail"], "auto");
        assert_eq!(resp_content[1]["type"], "input_text");
        assert_eq!(resp_content[1]["text"], "这是什么");

        let cc = super::chat_completions::build_chat_completions_payload(&req).unwrap();
        let cc_content = &cc["messages"][0]["content"];
        assert_eq!(cc_content[0]["type"], "image_url");
        assert!(cc_content[0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,"));
        assert_eq!(cc_content[1]["type"], "text");

        let messages = super::messages::build_messages_payload(&req).unwrap();
        let msg_content = &messages["messages"][0]["content"];
        assert_eq!(msg_content[0]["type"], "image");
        assert_eq!(msg_content[0]["source"]["type"], "base64");
        assert_eq!(msg_content[0]["source"]["media_type"], "image/jpeg");
        assert!(!msg_content[0]["source"]["data"]
            .as_str()
            .unwrap()
            .starts_with("data:"));
        assert_eq!(msg_content[1]["type"], "text");

        let gemini = super::gemini::build_gemini_payload(&req).unwrap();
        let parts = &gemini["contents"][0]["parts"];
        assert!(parts[0].get("inlineData").is_some());
        assert_eq!(parts[0]["inlineData"]["mimeType"], "image/jpeg");
        assert_eq!(parts[1]["text"], "这是什么");
    }

    #[test]
    fn gemini_schema_sanitizer() {
        let raw = serde_json::json!({
            "$schema": "https://json-schema.org/draft/07/schema#",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "number"}
                        ],
                        "additionalProperties": true
                    }
                }
            },
            "required": ["items"]
        });
        let out = sanitize_json_schema_for_gemini(raw);
        assert!(out.get("$schema").is_none());
        assert!(out.get("additionalProperties").is_none());
        assert_eq!(out["type"], "object");
        assert_eq!(out["required"], serde_json::json!(["items"]));
        let items_schema = &out["properties"]["items"]["items"];
        assert!(items_schema.get("oneOf").is_none());
        assert!(items_schema.get("anyOf").is_some());
        assert!(items_schema.get("additionalProperties").is_none());
        assert_eq!(items_schema["anyOf"][0]["type"], "string");
    }
}
