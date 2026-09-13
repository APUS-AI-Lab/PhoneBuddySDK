//! Visit-level history trimming for a target context window.

use crate::conversation::{ConversationItem, UserContentPart};
use crate::error::{EngineError, EngineResult};

/// Conservative character-to-token ratio used without a tokenizer.
const CHARS_PER_TOKEN: usize = 4;
/// Safety margin reserved for templates / images / generation.
const SAFETY_TOKENS: u32 = 64;

pub fn estimate_item_tokens(item: &ConversationItem) -> u32 {
    let chars = item_chars(item);
    (chars.div_ceil(CHARS_PER_TOKEN)) as u32
}

fn item_chars(item: &ConversationItem) -> usize {
    match item {
        ConversationItem::System(s) => s.content.len(),
        ConversationItem::User(u) => u
            .parts
            .iter()
            .map(|p| match p {
                UserContentPart::Text { text } => text.len(),
                UserContentPart::Image { .. } => 256,
                UserContentPart::Audio { .. } => 256,
            })
            .sum(),
        ConversationItem::Assistant(a) => {
            a.content.len()
                + a.reasoning_content.as_deref().map(str::len).unwrap_or(0)
                + a.tool_calls
                    .iter()
                    .map(|tc| tc.function.name.len() + tc.function.arguments.len())
                    .sum::<usize>()
        }
        ConversationItem::ToolResult(t) => t.content.len() + t.tool_call_id.len(),
        ConversationItem::Reasoning(r) => crate::llm::types::reasoning_item_text(r).len(),
        ConversationItem::BackendToolCall(b) => b.payload.to_string().len(),
    }
}

fn is_protected(item: &ConversationItem) -> bool {
    matches!(
        item,
        ConversationItem::System(_) | ConversationItem::Reasoning(_)
    )
}

/// Trim oldest complete user turns until `items` fit in `window` minus
/// output reserve. Never drops the current user input or trailing tool
/// results that still belong to an open tool cycle. Returns
/// [`EngineError::LocalContextExceeded`] when the protected suffix does
/// not fit.
pub fn trim_history_to_window(
    items: &[ConversationItem],
    window_tokens: u32,
    max_output_tokens: u32,
) -> EngineResult<Vec<ConversationItem>> {
    if window_tokens == 0 {
        return Err(EngineError::LocalContextExceeded);
    }
    let budget = window_tokens.saturating_sub(max_output_tokens.max(1) + SAFETY_TOKENS);
    if total_tokens(items) <= budget {
        return Ok(items.to_vec());
    }

    let keep_from = protected_suffix_start(items);
    let mut out = items.to_vec();
    while total_tokens(&out) > budget {
        let Some(idx) = first_droppable_index(&out, keep_from) else {
            return Err(EngineError::LocalContextExceeded);
        };
        out.remove(idx);
    }
    Ok(out)
}

fn total_tokens(items: &[ConversationItem]) -> u32 {
    items.iter().map(estimate_item_tokens).sum()
}

fn protected_suffix_start(items: &[ConversationItem]) -> usize {
    let mut i = items.len();
    while i > 0 {
        match &items[i - 1] {
            ConversationItem::ToolResult(_) | ConversationItem::BackendToolCall(_) => i -= 1,
            ConversationItem::Assistant(a) if !a.tool_calls.is_empty() => i -= 1,
            ConversationItem::User(_) => {
                i -= 1;
                break;
            }
            ConversationItem::System(_) | ConversationItem::Reasoning(_) => i -= 1,
            ConversationItem::Assistant(_) => break,
        }
    }
    i
}

fn first_droppable_index(items: &[ConversationItem], keep_from: usize) -> Option<usize> {
    items
        .iter()
        .enumerate()
        .find(|(i, item)| *i < keep_from && !is_protected(item))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ConversationItem;

    #[test]
    fn does_not_drop_current_user_input() {
        let items = vec![
            ConversationItem::system("sys"),
            ConversationItem::user("old ".repeat(200)),
            ConversationItem::assistant("old-a"),
            ConversationItem::user("current-question"),
        ];
        let trimmed = trim_history_to_window(&items, 120, 16).unwrap();
        assert!(matches!(
            trimmed.last(),
            Some(ConversationItem::User(_))
        ));
        assert!(trimmed.iter().any(|i| matches!(i, ConversationItem::System(_))));
        assert!(!trimmed.iter().any(|i| match i {
            ConversationItem::User(u) => u
                .parts
                .iter()
                .any(|p| matches!(p, UserContentPart::Text { text } if text.starts_with("old "))),
            _ => false,
        }));
    }

    #[test]
    fn exceeds_when_protected_suffix_does_not_fit() {
        let items = vec![
            ConversationItem::system("sys"),
            ConversationItem::user("x".repeat(4000)),
        ];
        match trim_history_to_window(&items, 40, 16) {
            Err(EngineError::LocalContextExceeded) => {}
            other => panic!("expected LocalContextExceeded, got {other:?}"),
        }
    }
}
