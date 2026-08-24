//! Image attachment validation, request-scoped materialization, and dump redaction.
//!
//! Base64 / Data URLs are produced only for the current LLM request and must
//! never be written to `StoredSession`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use sha2::{Digest, Sha256};

use crate::conversation::{
    ImageDetail, ImageMimeType, UserContentPart, UserItem, MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH,
    MAX_IMAGES_PER_TURN,
};
use crate::error::{EngineError, EngineResult};

/// Gemini generateContent inline payload ceiling (bytes of decoded image data).
pub const GEMINI_INLINE_MAX_BYTES: u64 = 20 * 1024 * 1024;

const DATA_URL_PREFIX: &str = "data:image/";

/// In-memory bytes for one image, valid only for the current request + retries.
#[derive(Debug, Clone)]
pub struct MaterializedImage {
    pub attachment_id: String,
    pub mime_type: ImageMimeType,
    pub bytes: Vec<u8>,
    pub detail: Option<ImageDetail>,
    pub width: u32,
    pub height: u32,
}

impl MaterializedImage {
    pub fn data_url(&self) -> String {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&self.bytes);
        format!("data:{};base64,{}", self.mime_type.as_str(), b64)
    }

    pub fn raw_b64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.bytes)
    }

    pub fn sha256_prefix(&self) -> String {
        sha256_prefix(&self.bytes)
    }
}

/// Request-scoped image bytes. Cheap to clone (shared map).
#[derive(Debug, Clone, Default)]
pub struct ImageBytesStore {
    inner: Arc<Mutex<HashMap<String, MaterializedImage>>>,
}

impl ImageBytesStore {
    pub fn insert(&self, img: MaterializedImage) {
        self.inner
            .lock()
            .expect("image store lock")
            .insert(img.attachment_id.clone(), img);
    }

    pub fn get(&self, attachment_id: &str) -> Option<MaterializedImage> {
        self.inner
            .lock()
            .expect("image store lock")
            .get(attachment_id)
            .cloned()
    }

    pub fn total_bytes(&self) -> u64 {
        self.inner
            .lock()
            .expect("image store lock")
            .values()
            .map(|i| i.bytes.len() as u64)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().expect("image store lock").is_empty()
    }
}

/// Canonicalize `path` and reject symlink escape / non-files outside `root`.
pub fn assert_path_in_root(path: &Path, root: &Path) -> EngineResult<PathBuf> {
    let root = root.canonicalize().map_err(|e| {
        EngineError::AttachmentInvalid(
            path.display().to_string(),
            format!("attachment root is not accessible: {e}"),
        )
    })?;
    let meta = std::fs::symlink_metadata(path).map_err(|e| {
        EngineError::AttachmentMissing(format!("{} ({e})", path.display()))
    })?;
    if meta.file_type().is_symlink() {
        return Err(EngineError::SandboxEscape(path.display().to_string()));
    }
    if !meta.file_type().is_file() {
        return Err(EngineError::AttachmentInvalid(
            path.display().to_string(),
            "not a regular file".into(),
        ));
    }
    let canon = path.canonicalize().map_err(|e| {
        EngineError::AttachmentMissing(format!("{} ({e})", path.display()))
    })?;
    if !canon.starts_with(&root) {
        return Err(EngineError::SandboxEscape(canon.display().to_string()));
    }
    Ok(canon)
}

/// Detect JPEG / PNG from magic bytes. Extension and caller MIME are ignored.
pub fn detect_image_mime(bytes: &[u8]) -> EngineResult<ImageMimeType> {
    let kind = infer::get(bytes).ok_or_else(|| {
        EngineError::AttachmentInvalid(String::new(), "unrecognized file magic".into())
    })?;
    match kind.mime_type() {
        "image/jpeg" => Ok(ImageMimeType::Jpeg),
        "image/png" => Ok(ImageMimeType::Png),
        other => Err(EngineError::AttachmentInvalid(
            String::new(),
            format!("unsupported mime {other}"),
        )),
    }
}

/// Read width/height from JPEG SOF or PNG IHDR. Does not decode pixels.
pub fn image_dimensions(bytes: &[u8], mime: ImageMimeType) -> EngineResult<(u32, u32)> {
    match mime {
        ImageMimeType::Png => png_dimensions(bytes),
        ImageMimeType::Jpeg => jpeg_dimensions(bytes),
    }
}

fn png_dimensions(bytes: &[u8]) -> EngineResult<(u32, u32)> {
    // signature (8) + length (4) + "IHDR" (4) + width (4) + height (4)
    if bytes.len() < 24 || &bytes[0..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return Err(EngineError::AttachmentInvalid(
            String::new(),
            "invalid PNG header".into(),
        ));
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    if width == 0 || height == 0 {
        return Err(EngineError::AttachmentInvalid(
            String::new(),
            "invalid PNG dimensions".into(),
        ));
    }
    Ok((width, height))
}

fn jpeg_dimensions(bytes: &[u8]) -> EngineResult<(u32, u32)> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return Err(EngineError::AttachmentInvalid(
            String::new(),
            "invalid JPEG header".into(),
        ));
    }
    let mut i = 2usize;
    while i + 3 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        while i < bytes.len() && bytes[i] == 0xFF {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let marker = bytes[i];
        i += 1;
        // Standalone markers with no length.
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if i + 1 >= bytes.len() {
            break;
        }
        let len = u16::from_be_bytes([bytes[i], bytes[i + 1]]) as usize;
        if len < 2 || i + len > bytes.len() {
            break;
        }
        // SOF0 / SOF1 / SOF2
        if marker == 0xC0 || marker == 0xC1 || marker == 0xC2 {
            if len < 7 || i + 6 >= bytes.len() {
                break;
            }
            let height = u16::from_be_bytes([bytes[i + 3], bytes[i + 4]]) as u32;
            let width = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]) as u32;
            if width == 0 || height == 0 {
                return Err(EngineError::AttachmentInvalid(
                    String::new(),
                    "invalid JPEG dimensions".into(),
                ));
            }
            return Ok((width, height));
        }
        i += len;
    }
    Err(EngineError::AttachmentInvalid(
        String::new(),
        "JPEG SOF marker not found".into(),
    ))
}

/// Read, validate, and cache every image part on `item`.
pub fn materialize_user_item(
    item: &UserItem,
    attachment_root: &Path,
    store: &ImageBytesStore,
) -> EngineResult<()> {
    if item.image_count() > MAX_IMAGES_PER_TURN {
        return Err(EngineError::TooManyImages(item.image_count()));
    }
    for part in &item.parts {
        let UserContentPart::Image {
            attachment_id,
            local_path,
            mime_type,
            byte_size,
            width,
            height,
            detail,
        } = part
        else {
            continue;
        };
        if store.get(attachment_id).is_some() {
            continue;
        }
        let canon = assert_path_in_root(Path::new(local_path), attachment_root)?;
        let bytes = std::fs::read(&canon).map_err(|e| {
            EngineError::AttachmentMissing(format!("{attachment_id} ({e})"))
        })?;
        if bytes.len() as u64 != *byte_size {
            return Err(EngineError::AttachmentInvalid(
                attachment_id.clone(),
                format!("byte_size mismatch: declared {byte_size}, file {}", bytes.len()),
            ));
        }
        let detected = detect_image_mime(&bytes).map_err(|e| match e {
            EngineError::AttachmentInvalid(_, msg) => {
                EngineError::AttachmentInvalid(attachment_id.clone(), msg)
            }
            other => other,
        })?;
        if detected != *mime_type {
            return Err(EngineError::AttachmentInvalid(
                attachment_id.clone(),
                format!(
                    "mime mismatch: declared {}, magic {}",
                    mime_type.as_str(),
                    detected.as_str()
                ),
            ));
        }
        let (w, h) = image_dimensions(&bytes, detected).map_err(|e| match e {
            EngineError::AttachmentInvalid(_, msg) => {
                EngineError::AttachmentInvalid(attachment_id.clone(), msg)
            }
            other => other,
        })?;
        if w != *width || h != *height {
            return Err(EngineError::AttachmentInvalid(
                attachment_id.clone(),
                format!("dimension mismatch: declared {width}x{height}, file {w}x{h}"),
            ));
        }
        if w > MAX_IMAGE_WIDTH || h > MAX_IMAGE_HEIGHT {
            return Err(EngineError::AttachmentInvalid(
                attachment_id.clone(),
                format!("exceeds {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}"),
            ));
        }
        store.insert(MaterializedImage {
            attachment_id: attachment_id.clone(),
            mime_type: detected,
            bytes,
            detail: *detail,
            width: w,
            height: h,
        });
    }
    Ok(())
}

/// Materialize every user image in a conversation (current turn + history).
pub fn materialize_items(
    items: &[crate::conversation::ConversationItem],
    attachment_root: &Path,
) -> EngineResult<ImageBytesStore> {
    let store = ImageBytesStore::default();
    for item in items {
        if let crate::conversation::ConversationItem::User(u) = item {
            materialize_user_item(u, attachment_root, &store)?;
        }
    }
    Ok(store)
}

pub fn sha256_prefix(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    hash.iter().take(4).map(|b| format!("{b:02x}")).collect()
}

fn decoded_len_from_data_url(url: &str) -> usize {
    url.split_once("base64,")
        .map(|(_, b64)| {
            let trimmed = b64.trim();
            // 4 base64 chars → 3 bytes; ignore padding.
            trimmed.len() * 3 / 4
        })
        .unwrap_or(0)
}

fn mime_from_data_url(url: &str) -> String {
    url.strip_prefix("data:")
        .and_then(|rest| rest.split(';').next())
        .unwrap_or("image/jpeg")
        .to_string()
}

fn sha_prefix_from_data_url(url: &str) -> String {
    if let Some((_, b64)) = url.split_once("base64,") {
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64.trim()) {
            return sha256_prefix(&bytes);
        }
    }
    "00000000".into()
}

fn redact_signed_url(url: &str) -> Option<String> {
    let (base, query) = url.split_once('?')?;
    if query.is_empty() {
        return None;
    }
    // Drop query + fragment so signed image URLs never land in dumps.
    let base = base.split('#').next().unwrap_or(base);
    Some(format!("{base}?[REDACTED]"))
}

/// Recursively strip image bytes / data URLs / signed image URLs from JSON.
pub fn redact_image_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            redact_object(map);
            for v in map.values_mut() {
                redact_image_json(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                redact_image_json(v);
            }
        }
        serde_json::Value::String(s) => {
            if s.starts_with(DATA_URL_PREFIX) {
                *s = "[REDACTED]".into();
            } else if s.contains(DATA_URL_PREFIX) {
                *s = redact_text(s);
            } else if looks_like_raw_image_b64(s) {
                *s = "[REDACTED]".into();
            }
        }
        _ => {}
    }
}

fn redact_object(map: &mut serde_json::Map<String, serde_json::Value>) {
    match map.get("image_url") {
        Some(serde_json::Value::String(s)) if s.starts_with(DATA_URL_PREFIX) => {
            let mime = mime_from_data_url(s);
            let decoded = decoded_len_from_data_url(s);
            let prefix = sha_prefix_from_data_url(s);
            map.insert(
                "image_url".into(),
                serde_json::json!({
                    "type": "image",
                    "mime_type": mime,
                    "data": "[REDACTED]",
                    "decoded_bytes": decoded,
                    "sha256_prefix": prefix,
                }),
            );
        }
        Some(serde_json::Value::String(s)) => {
            if let Some(redacted) = redact_signed_url(s) {
                map.insert("image_url".into(), serde_json::Value::String(redacted));
            }
        }
        Some(serde_json::Value::Object(_)) => {
            if let Some(serde_json::Value::Object(inner)) = map.get_mut("image_url") {
                redact_url_field(inner, "url");
            }
        }
        _ => {}
    }

    if let Some(serde_json::Value::Object(source)) = map.get_mut("source") {
        let mime = source
            .get("media_type")
            .and_then(|v| v.as_str())
            .unwrap_or("image/jpeg")
            .to_string();
        redact_raw_b64_field(source, "data", &mime);
    }

    for key in ["inlineData", "inline_data"] {
        if let Some(serde_json::Value::Object(inner)) = map.get_mut(key) {
            let mime = inner
                .get("mimeType")
                .or_else(|| inner.get("mime_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("image/jpeg")
                .to_string();
            redact_raw_b64_field(inner, "data", &mime);
        }
    }
}

fn redact_url_field(map: &mut serde_json::Map<String, serde_json::Value>, key: &str) {
    let Some(serde_json::Value::String(s)) = map.get(key).cloned() else {
        return;
    };
    if s.starts_with(DATA_URL_PREFIX) {
        map.insert(key.into(), serde_json::Value::String("[REDACTED]".into()));
    } else if let Some(redacted) = redact_signed_url(&s) {
        map.insert(key.into(), serde_json::Value::String(redacted));
    }
}

fn redact_raw_b64_field(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    mime: &str,
) {
    let Some(serde_json::Value::String(s)) = map.get(key) else {
        return;
    };
    if s == "[REDACTED]" {
        return;
    }
    let decoded = s.len() * 3 / 4;
    let prefix = if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(s.trim()) {
        sha256_prefix(&bytes)
    } else {
        "00000000".into()
    };
    map.insert(
        key.into(),
        serde_json::json!({
            "type": "image",
            "mime_type": mime,
            "data": "[REDACTED]",
            "decoded_bytes": decoded,
            "sha256_prefix": prefix,
        }),
    );
}

fn looks_like_raw_image_b64(s: &str) -> bool {
    // JPEG `/9j/` or PNG `iVBOR` in standard base64. Require a long payload so
    // short identifiers are not mistaken for image data.
    s.len() > 128 && (s.starts_with("/9j/") || s.starts_with("iVBOR"))
}

/// Strip data URLs and long image base64 from free-form text (error strings).
pub fn redact_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find(DATA_URL_PREFIX) {
        out.push_str(&rest[..idx]);
        out.push_str("[REDACTED]");
        let after = &rest[idx + DATA_URL_PREFIX.len()..];
        // Skip until whitespace / quote / end.
        let skip = after
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '}')
            .unwrap_or(after.len());
        rest = &after[skip..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1×1 PNG (red pixel).
    pub fn tiny_png() -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
            .unwrap()
    }

    #[test]
    fn png_magic_and_dimensions() {
        let bytes = tiny_png();
        let mime = detect_image_mime(&bytes).unwrap();
        assert_eq!(mime, ImageMimeType::Png);
        assert_eq!(image_dimensions(&bytes, mime).unwrap(), (1, 1));
    }

    #[test]
    fn rejects_non_image_magic() {
        let err = detect_image_mime(b"hello world this is not an image").unwrap_err();
        match err {
            EngineError::AttachmentInvalid(_, msg) => {
                assert!(msg.contains("unrecognized") || msg.contains("unsupported"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn path_sandbox_rejects_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        std::fs::create_dir(&root).unwrap();
        let outside = tmp.path().join("secret.png");
        std::fs::write(&outside, tiny_png()).unwrap();
        let err = assert_path_in_root(&outside, &root).unwrap_err();
        match err {
            EngineError::SandboxEscape(_) => {}
            other => panic!("expected sandbox escape, got {other:?}"),
        }
    }

    #[test]
    fn materialize_reads_png_under_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("atts");
        std::fs::create_dir(&root).unwrap();
        let file = root.join("a.png");
        let bytes = tiny_png();
        std::fs::write(&file, &bytes).unwrap();
        let item = UserItem {
            parts: vec![UserContentPart::Image {
                attachment_id: "img_1".into(),
                local_path: file.to_string_lossy().into(),
                mime_type: ImageMimeType::Png,
                byte_size: bytes.len() as u64,
                width: 1,
                height: 1,
                detail: Some(ImageDetail::Auto),
            }],
        };
        let store = ImageBytesStore::default();
        materialize_user_item(&item, &root, &store).unwrap();
        let img = store.get("img_1").unwrap();
        assert_eq!(img.width, 1);
        assert!(img.data_url().starts_with("data:image/png;base64,"));
        assert!(!img.raw_b64().starts_with("data:"));
    }

    #[test]
    fn redact_openai_data_url_and_anthropic_raw() {
        let mut v = serde_json::json!({
            "input": [{
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD",
                        "detail": "auto"
                    }
                ]
            }],
            "messages": [{
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRgABAQAAAQABAAD"
                    }
                }]
            }],
            "contents": [{
                "parts": [{
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                    }
                }]
            }]
        });
        redact_image_json(&mut v);
        let dumped = v.to_string();
        assert!(!dumped.contains("/9j/"));
        assert!(!dumped.contains("iVBORw0KGgo"));
        assert!(dumped.contains("[REDACTED]"));
        assert!(dumped.contains("decoded_bytes"));
        assert!(dumped.contains("sha256_prefix"));
    }

    #[test]
    fn redact_signed_remote_url() {
        let mut v = serde_json::json!({
            "image_url": {
                "url": "https://cdn.example.com/x.jpg?X-Amz-Signature=abc&Expires=1"
            }
        });
        redact_image_json(&mut v);
        assert_eq!(
            v["image_url"]["url"],
            "https://cdn.example.com/x.jpg?[REDACTED]"
        );
    }
}
