//! `plan` tool — visible task planning (todo write).
//!
//! Ported core logic from grok-build `implementations/grok_build/todo`:
//! `validate_no_duplicate_ids`, `apply_replace`, `apply_merge`,
//! `summarize_todo_state`. Supports full replace and id-based merge so the
//! model can flip status without resending entire contents.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver};
use crate::tools::{arg_bool, arg_opt_str, schema_object, s_boolean, s_string, Tool, ToolCtx, ToolOutput, ToolSpec};

// ── Types (aligned with grok TodoItem / TodoUpdate) ──────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

impl TodoStatus {
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Pending => "[pending]",
            Self::InProgress => "[in_progress]",
            Self::Completed => "[completed]",
            Self::Cancelled => "[cancelled]",
        }
    }

    fn as_wire(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl Default for TodoStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// Wire-facing plan item (backward compatible with prior `PlanItem`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanItem {
    #[serde(default)]
    pub id: Option<String>,
    pub content: String,
    #[serde(deserialize_with = "deserialize_status_wire")]
    pub status: String,
}

fn deserialize_status_wire<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "pending" | "in_progress" | "completed" | "cancelled" => Ok(s),
        _ => Err(serde::de::Error::custom(format!(
            "status must be pending|in_progress|completed|cancelled; got '{s}'"
        ))),
    }
}

fn parse_status(s: &str) -> TodoStatus {
    match s {
        "in_progress" => TodoStatus::InProgress,
        "completed" => TodoStatus::Completed,
        "cancelled" => TodoStatus::Cancelled,
        _ => TodoStatus::Pending,
    }
}

#[derive(Debug, Clone)]
pub struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone)]
pub struct TodoUpdate {
    pub id: String,
    pub content: Option<String>,
    pub status: Option<TodoStatus>,
}

impl TodoUpdate {
    fn has_no_content(&self) -> bool {
        self.content.as_ref().map(|c| c.trim().is_empty()).unwrap_or(true)
    }
}

#[derive(Debug, Default, Clone)]
pub struct TodoState {
    items: IndexMap<String, TodoItem>,
}

impl TodoState {
    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn push(&mut self, id: String, todo: TodoItem) {
        self.items.insert(id, todo);
    }

    /// Update existing item; returns false if id is unknown.
    pub fn update(
        &mut self,
        id: &str,
        content: Option<&str>,
        status: Option<TodoStatus>,
    ) -> bool {
        let Some(item) = self.items.get_mut(id) else {
            return false;
        };
        if let Some(c) = content {
            if !c.trim().is_empty() {
                item.content = c.to_string();
            }
        }
        if let Some(s) = status {
            item.status = s;
        }
        true
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn todo_items_with_ids(&self) -> impl Iterator<Item = (&String, &TodoItem)> + '_ {
        self.items.iter()
    }

    pub fn to_plan_items(&self) -> Vec<PlanItem> {
        self.items
            .iter()
            .map(|(id, t)| PlanItem {
                id: Some(id.clone()),
                content: t.content.clone(),
                status: t.status.as_wire().to_string(),
            })
            .collect()
    }
}

// ── Core pure functions (ported from grok todo/mod.rs) ───────────────────

#[derive(Debug, thiserror::Error)]
pub enum TodoError {
    #[error("Duplicate Todo ID in response: {0}")]
    DuplicateTodoID(String),
}

pub fn validate_no_duplicate_ids(updates: &[TodoUpdate]) -> Result<(), TodoError> {
    use std::collections::HashSet;
    let mut seen = HashSet::with_capacity(updates.len());
    if let Some(dup) = updates.iter().map(|u| &u.id).find(|id| !seen.insert(id.as_str())) {
        return Err(TodoError::DuplicateTodoID(dup.clone()));
    }
    Ok(())
}

/// `merge=false`: full replace. Missing content falls back to id; missing status → Pending.
pub fn apply_replace(state: &mut TodoState, updates: &[TodoUpdate]) -> Result<(), TodoError> {
    state.clear();
    for u in updates {
        let content = if u.has_no_content() {
            u.id.clone()
        } else {
            u.content.clone().unwrap()
        };
        let status = u.status.unwrap_or(TodoStatus::Pending);
        state.push(
            u.id.clone(),
            TodoItem { content, status },
        );
    }
    Ok(())
}

/// `merge=true`: merge by id; content optional for existing items.
pub fn apply_merge(state: &mut TodoState, updates: &[TodoUpdate]) -> Result<(), TodoError> {
    for u in updates {
        if state.update(&u.id, u.content.as_deref(), u.status) {
            continue;
        }
        let content = if u.has_no_content() {
            u.id.clone()
        } else {
            u.content.clone().unwrap()
        };
        let status = u.status.unwrap_or(TodoStatus::Pending);
        state.push(
            u.id.clone(),
            TodoItem { content, status },
        );
    }
    Ok(())
}

pub fn summarize_todo_state(state: &TodoState) -> String {
    if state.is_empty() {
        "No tasks currently tracked.".into()
    } else {
        let mut out = String::new();
        for (id, t) in state.todo_items_with_ids() {
            use std::fmt::Write as _;
            let _ = writeln!(&mut out, "- {} {id}: {}", t.status.tag(), t.content);
        }
        out
    }
}

// ── Engine-facing PlanState ──────────────────────────────────────────────

/// Shared plan state; also readable by the engine after a turn.
#[derive(Default)]
pub struct PlanState {
    pub items: Mutex<Vec<PlanItem>>,
    todo: Mutex<TodoState>,
    pub observer: Mutex<Option<Arc<dyn AgentObserver>>>,
}

impl PlanState {
    pub fn new() -> Arc<Self> {
        Arc::default()
    }
    pub fn snapshot(&self) -> Vec<PlanItem> {
        self.items.lock().unwrap().clone()
    }
}

pub struct PlanTool {
    state: Arc<PlanState>,
}

#[async_trait]
impl Tool for PlanTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "plan".into(),
            description: concat!(
                "Create or update a visible step-by-step plan for the current task. ",
                "Pass steps with optional stable `id` fields. ",
                "Set merge=true to update existing steps by id without resending full content ",
                "(e.g. mark in_progress → completed). merge=false (default) replaces the whole plan. ",
                "Use for non-trivial multi-step tasks; skip for simple requests."
            )
            .into(),
            parameters: schema_object(
                vec![
                    (
                        "steps",
                        serde_json::json!({
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Stable step id for merge updates"},
                                    "content": {"type": "string"},
                                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                                },
                                "required": ["status"],
                            },
                        }),
                        "Ordered plan steps (content required for new items; optional when merging by id).",
                    ),
                    (
                        "merge",
                        s_boolean(),
                        "If true, merge steps by id into existing plan (default false = full replace).",
                    ),
                    ("note", s_string(), "Optional one-line explanation of a plan change."),
                ],
                &["steps"],
            ),
        }
    }

    async fn execute(&self, args: Value, _ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let steps_val = args.get("steps").cloned().unwrap_or(Value::Array(vec![]));
        let steps_arr = steps_val.as_array().ok_or_else(|| EngineError::ToolArgs {
            name: "plan".into(),
            message: "steps must be an array".into(),
        })?;

        let mut updates: Vec<TodoUpdate> = Vec::new();
        for (i, step) in steps_arr.iter().enumerate() {
            let id = step
                .get("id")
                .and_then(Value::as_str)
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("step-{}", i + 1));
            let content = step
                .get("content")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let status = step
                .get("status")
                .and_then(Value::as_str)
                .map(parse_status);
            updates.push(TodoUpdate { id, content, status });
        }

        if updates.is_empty() {
            return Err(EngineError::ToolArgs {
                name: "plan".into(),
                message: "steps must not be empty".into(),
            });
        }

        if let Err(TodoError::DuplicateTodoID(id)) = validate_no_duplicate_ids(&updates) {
            return Err(EngineError::ToolArgs {
                name: "plan".into(),
                message: format!("duplicate step id: {id}"),
            });
        }

        let merge = arg_bool(&args, "merge");
        let mut todo = self.state.todo.lock().unwrap();
        if merge {
            apply_merge(&mut todo, &updates).map_err(|e| EngineError::Tool {
                name: "plan".into(),
                message: e.to_string(),
            })?;
        } else {
            apply_replace(&mut todo, &updates).map_err(|e| EngineError::Tool {
                name: "plan".into(),
                message: e.to_string(),
            })?;
        }

        let plan_items = todo.to_plan_items();
        let summary = summarize_todo_state(&todo);
        drop(todo);

        *self.state.items.lock().unwrap() = plan_items.clone();

        let json = serde_json::to_string(&plan_items)?;
        if let Some(observer) = self.state.observer.lock().unwrap().clone() {
            observer.on_event(AgentEvent::PlanUpdated { items_json: json });
        }

        let done = plan_items
            .iter()
            .filter(|s| s.status == "completed")
            .count();
        let note = arg_opt_str(&args, "note");
        let mut out = format!(
            "Plan updated ({}): {done}/{} steps completed.\n{summary}",
            if merge { "merge" } else { "replace" },
            plan_items.len()
        );
        if let Some(n) = note {
            out.push_str(&format!("Note: {n}\n"));
        }
        Ok(ToolOutput::new(out))
    }
}

pub fn arc(state: Arc<PlanState>) -> Arc<dyn Tool> {
    Arc::new(PlanTool { state })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upd(id: &str, content: Option<&str>, status: Option<TodoStatus>) -> TodoUpdate {
        TodoUpdate {
            id: id.into(),
            content: content.map(|s| s.into()),
            status,
        }
    }

    #[test]
    fn replace_mode_creates_items() {
        let mut state = TodoState::default();
        apply_replace(
            &mut state,
            &[
                upd("a", Some("first"), Some(TodoStatus::Pending)),
                upd("b", Some("second"), Some(TodoStatus::InProgress)),
            ],
        )
        .unwrap();
        assert_eq!(state.items.len(), 2);
        assert_eq!(state.items["a"].content, "first");
    }

    #[test]
    fn merge_updates_status_without_content() {
        let mut state = TodoState::default();
        apply_replace(
            &mut state,
            &[upd("a", Some("do thing"), Some(TodoStatus::InProgress))],
        )
        .unwrap();
        apply_merge(
            &mut state,
            &[upd("a", None, Some(TodoStatus::Completed))],
        )
        .unwrap();
        assert_eq!(state.items["a"].content, "do thing");
        assert_eq!(state.items["a"].status, TodoStatus::Completed);
    }

    #[test]
    fn duplicate_ids_rejected() {
        let err = validate_no_duplicate_ids(&[
            upd("x", Some("1"), None),
            upd("x", Some("2"), None),
        ]);
        assert!(matches!(err, Err(TodoError::DuplicateTodoID(_))));
    }
}
