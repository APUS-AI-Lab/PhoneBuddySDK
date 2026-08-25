//! Session persistence.
//!
//! Conversations are stored as JSON files under
//! `<root>/.phonebuddy/sessions/<id>.json` — deliberately avoiding sqlite
//! (bundled C builds are one of the things cut for mobile v1).
//!
//! Format v2 stores an ordered [`ConversationItem`] list. v1 files
//! (`messages: Vec<ChatMessage>`) are still readable and upgraded on save.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::conversation::{
    items_from_chat_messages, user_assistant_count, ConversationItem,
};
use crate::error::{EngineError, EngineResult};
use crate::llm::types::ChatMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSession {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default = "default_format_version")]
    pub format_version: u32,
    pub items: Vec<ConversationItem>,
}

fn default_format_version() -> u32 {
    2
}

impl Default for StoredSession {
    fn default() -> Self {
        Self {
            id: String::new(),
            title: String::new(),
            created_at: String::new(),
            updated_at: String::new(),
            format_version: 2,
            items: Vec::new(),
        }
    }
}

impl StoredSession {
    pub fn as_chat_messages(&self) -> Vec<ChatMessage> {
        crate::conversation::chat_messages_from_items(&self.items)
    }
}

#[derive(Debug, Deserialize)]
struct StoredSessionV1 {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct StoredSessionHeader {
    #[serde(default)]
    format_version: u32,
}

pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    pub fn new(dir: PathBuf) -> EngineResult<Self> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn path(&self, id: &str) -> PathBuf {
        // Defensive: session ids are our own uuids, but never trust input.
        let safe: String = id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        self.dir.join(format!("{safe}.json"))
    }

    pub fn load(&self, id: &str) -> EngineResult<Option<StoredSession>> {
        let path = self.path(id);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)?;
        parse_session_json(id, &text).map(Some)
    }

    pub fn save(&self, session: &StoredSession) -> EngineResult<()> {
        let mut to_write = session.clone();
        to_write.format_version = 2;
        let text = serde_json::to_string_pretty(&to_write)?;
        let tmp = self.path(&session.id).with_extension("tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, self.path(&session.id))?;
        Ok(())
    }

    pub fn delete(&self, id: &str) -> EngineResult<()> {
        let path = self.path(id);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    pub fn list(&self) -> EngineResult<Vec<SessionMeta>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let Some(entry) = entry.ok() else { continue };
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let Ok(s) = parse_session_json(id, &text) else {
                continue;
            };
            out.push(SessionMeta {
                id: s.id,
                title: s.title,
                created_at: s.created_at,
                updated_at: s.updated_at,
                message_count: user_assistant_count(&s.items),
            });
        }
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(out)
    }
}

fn parse_session_json(id: &str, text: &str) -> EngineResult<StoredSession> {
    let header: StoredSessionHeader = serde_json::from_str(text).map_err(|e| {
        EngineError::SessionNotFound(format!("{id}: corrupt session file: {e}"))
    })?;
    if header.format_version >= 2 {
        match serde_json::from_str::<StoredSession>(text) {
            Ok(s) => Ok(s),
            Err(e) => Err(EngineError::SessionNotFound(format!(
                "{id}: invalid session items (format_version {}): {e}",
                header.format_version
            ))),
        }
    } else {
        let v1: StoredSessionV1 = serde_json::from_str(text).map_err(|e| {
            EngineError::SessionNotFound(format!("{id}: corrupt session file: {e}"))
        })?;
        Ok(StoredSession {
            id: v1.id,
            title: v1.title,
            created_at: v1.created_at,
            updated_at: v1.updated_at,
            format_version: 1,
            items: items_from_chat_messages(&v1.messages),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{AssistantItem, BackendToolCallItem};
    use crate::llm::types::{ReasoningItem, SummaryPart, SummaryTextContent};

    #[test]
    fn v1_session_loads_and_upgrades() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf()).unwrap();

        let v1 = r#"{
            "id": "test-sess-1",
            "title": "Test Reasoning",
            "created_at": "2026-08-17T00:00:00Z",
            "updated_at": "2026-08-17T00:01:00Z",
            "messages": [
                {"role": "user", "content": "Solve math"},
                {
                    "role": "assistant",
                    "content": "Result is 42",
                    "reasoning_content": "Let's calculate...",
                    "encrypted_reasoning": "enc_token_xyz",
                    "reasoning_items": [{
                        "id": "r1",
                        "summary": [{"type": "summary_text", "text": "Let's calculate..."}],
                        "encrypted_content": "enc_token_xyz"
                    }]
                }
            ]
        }"#;
        std::fs::write(tmp.path().join("test-sess-1.json"), v1).unwrap();

        let loaded = store.load("test-sess-1").unwrap().unwrap();
        assert_eq!(loaded.id, "test-sess-1");
        assert!(matches!(loaded.items[0], ConversationItem::User(_)));
        assert!(matches!(loaded.items[1], ConversationItem::Reasoning(_)));
        assert!(matches!(loaded.items[2], ConversationItem::Assistant(_)));

        store.save(&loaded).unwrap();
        let again = store.load("test-sess-1").unwrap().unwrap();
        assert_eq!(again.format_version, 2);
        assert_eq!(again.items, loaded.items);
    }

    #[test]
    fn test_session_backward_compatibility_without_reasoning() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf()).unwrap();

        let legacy_json = r#"{
            "id": "legacy-sess",
            "title": "Legacy",
            "created_at": "2026-08-16T00:00:00Z",
            "updated_at": "2026-08-16T00:01:00Z",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"}
            ]
        }"#;

        std::fs::write(tmp.path().join("legacy-sess.json"), legacy_json).unwrap();

        let loaded = store.load("legacy-sess").unwrap().unwrap();
        assert_eq!(loaded.id, "legacy-sess");
        assert_eq!(loaded.items.len(), 2);
        match &loaded.items[1] {
            ConversationItem::Assistant(a) => {
                assert_eq!(a.content, "hi there");
                assert!(a.reasoning_content.is_none());
            }
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    #[test]
    fn v2_roundtrip_is_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf()).unwrap();
        let session = StoredSession {
            id: "v2".into(),
            title: "All variants".into(),
            created_at: "2026-08-17T00:00:00Z".into(),
            updated_at: "2026-08-17T00:01:00Z".into(),
            format_version: 2,
            items: vec![
                ConversationItem::system("sys"),
                ConversationItem::user("你好"),
                ConversationItem::Reasoning(ReasoningItem {
                    id: "rs_1".into(),
                    summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                        text: "think".into(),
                    })],
                    content: None,
                    encrypted_content: Some("enc".into()),
                    status: None,
                }),
                ConversationItem::BackendToolCall(BackendToolCallItem {
                    item_type: "web_search_call".into(),
                    id: "ws_1".into(),
                    payload: serde_json::json!({
                        "type": "web_search_call",
                        "id": "ws_1",
                        "action": {"query": "⌘"}
                    }),
                }),
                ConversationItem::Assistant(AssistantItem {
                    content: "ok".into(),
                    tool_calls: vec![],
                    reasoning_content: None,
                    encrypted_reasoning: None,
                    origin: Some("openai/gpt-5".into()),
                }),
                ConversationItem::tool_result("c1", "out"),
            ],
        };
        store.save(&session).unwrap();
        let loaded = store.load("v2").unwrap().unwrap();
        assert_eq!(loaded.format_version, 2);
        assert_eq!(loaded.items, session.items);
    }

    #[test]
    fn corrupt_and_unknown_versions() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf()).unwrap();
        std::fs::write(tmp.path().join("bad.json"), "{not json").unwrap();
        let err = store.load("bad").unwrap_err();
        match err {
            EngineError::SessionNotFound(msg) => assert!(msg.contains("corrupt")),
            other => panic!("expected SessionNotFound, got {other:?}"),
        }

        let v3 = r#"{
            "id": "fwd",
            "title": "t",
            "created_at": "t",
            "updated_at": "t",
            "format_version": 3,
            "items": [{"type": "unknown_future_item", "foo": 1}]
        }"#;
        std::fs::write(tmp.path().join("fwd.json"), v3).unwrap();
        let err = store.load("fwd").unwrap_err();
        match err {
            EngineError::SessionNotFound(msg) => {
                assert!(msg.contains("invalid session items"), "{msg}");
            }
            other => panic!("expected SessionNotFound, got {other:?}"),
        }
    }

    #[test]
    fn list_handles_mixed_formats() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf()).unwrap();
        let v1 = r#"{
            "id": "v1s",
            "title": "old",
            "created_at": "2026-08-16T00:00:00Z",
            "updated_at": "2026-08-16T00:01:00Z",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "tool", "content": "c", "tool_call_id": "x"}
            ]
        }"#;
        std::fs::write(tmp.path().join("v1s.json"), v1).unwrap();
        let v2 = StoredSession {
            id: "v2s".into(),
            title: "new".into(),
            created_at: "2026-08-17T00:00:00Z".into(),
            updated_at: "2026-08-17T00:02:00Z".into(),
            format_version: 2,
            items: vec![
                ConversationItem::user("u"),
                ConversationItem::assistant("a"),
                ConversationItem::tool_result("c", "out"),
            ],
        };
        store.save(&v2).unwrap();
        let list = store.list().unwrap();
        assert_eq!(list.len(), 2);
        let v1_meta = list.iter().find(|m| m.id == "v1s").unwrap();
        assert_eq!(v1_meta.message_count, 2); // User + Assistant, not tool
        let v2_meta = list.iter().find(|m| m.id == "v2s").unwrap();
        assert_eq!(v2_meta.message_count, 2);
    }
}
