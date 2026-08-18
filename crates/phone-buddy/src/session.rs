//! Session persistence.
//!
//! Conversations are stored as JSON files under
//! `<root>/.phonebuddy/sessions/<id>.json` — deliberately avoiding sqlite
//! (bundled C builds are one of the things cut for mobile v1).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoredSession {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub messages: Vec<ChatMessage>,
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
        let session: StoredSession = serde_json::from_str(&text).map_err(|e| {
            EngineError::SessionNotFound(format!("{id}: corrupt session file: {e}"))
        })?;
        Ok(Some(session))
    }

    pub fn save(&self, session: &StoredSession) -> EngineResult<()> {
        let text = serde_json::to_string_pretty(session)?;
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
            let Ok(s) = serde_json::from_str::<StoredSession>(&text) else {
                continue;
            };
            out.push(SessionMeta {
                id: s.id,
                title: s.title,
                created_at: s.created_at,
                updated_at: s.updated_at,
                message_count: s.messages.len(),
            });
        }
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatMessage, ReasoningItem, SummaryPart, SummaryTextContent};

    #[test]
    fn test_session_save_and_load_with_reasoning() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf()).unwrap();

        let session = StoredSession {
            id: "test-sess-1".into(),
            title: "Test Reasoning".into(),
            created_at: "2026-08-17T00:00:00Z".into(),
            updated_at: "2026-08-17T00:01:00Z".into(),
            messages: vec![
                ChatMessage::user("Solve math"),
                ChatMessage::assistant_with_reasoning(
                    "Result is 42",
                    Some("Let's calculate...".into()),
                    vec![ReasoningItem {
                        id: "r1".into(),
                        summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                            text: "Let's calculate...".into(),
                        })],
                        content: None,
                        encrypted_content: Some("enc_token_xyz".into()),
                        status: None,
                    }],
                    Some("enc_token_xyz".into()),
                ),
            ],
        };

        store.save(&session).unwrap();

        let loaded = store.load("test-sess-1").unwrap().unwrap();
        assert_eq!(loaded.id, "test-sess-1");
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(
            loaded.messages[1].reasoning_content.as_deref(),
            Some("Let's calculate...")
        );
        assert_eq!(
            loaded.messages[1].encrypted_reasoning.as_deref(),
            Some("enc_token_xyz")
        );
        assert_eq!(loaded.messages[1].reasoning_items.len(), 1);
        assert_eq!(loaded.messages[1].reasoning_items[0].id, "r1");
    }

    #[test]
    fn test_session_backward_compatibility_without_reasoning() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf()).unwrap();

        // Old format json without reasoning fields
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
        assert_eq!(loaded.messages.len(), 2);
        assert!(loaded.messages[1].reasoning_items.is_empty());
        assert!(loaded.messages[1].reasoning_content.is_none());
        assert!(loaded.messages[1].encrypted_reasoning.is_none());
    }
}
