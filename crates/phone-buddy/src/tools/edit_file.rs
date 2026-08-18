//! `edit_file` tool — exact string replacement in a file.
//!
//! Ported semantics from grok's `search_replace` tool
//! (`implementations/grok_build/search_replace`):
//! - `old_string` must match exactly once unless `replace_all` is set;
//! - empty `old_string` creates a new file (and never overwrites a
//!   non-empty one);
//! - `old_string == new_string` is rejected;
//! - confusable-aware (unicode homoglyph) matching via
//!   [`crate::tools::unicode_confusables`] when exact match fails;
//! - whitespace-tolerant fallback as an extra mobile-friendly pass;
//! - the result shows the edited region with `LINE_NUMBER→` prefixes and
//!   ±3 lines of context (`render_snippet`).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::edit_helpers::{
    find_normalized_match_positions, replace_normalized_matches, replace_spans,
    replace_using_positions, render_snippet, tolerant_match_positions, CONTEXT_LINES,
    NormalizedMatchResult,
};
use crate::tools::{
    arg_bool, arg_str, schema_object, s_boolean, s_string, Tool, ToolCtx, ToolOutput, ToolSpec,
};

pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".into(),
            description: concat!(
                "Replace an exact string in a file. `old_string` must match ",
                "exactly one place in the file (line-number prefixes shown by ",
                "read_file are not part of the file). If it appears more than ",
                "once, add surrounding lines to make it unique, or set ",
                "`replace_all`. To create a new file, pass an empty ",
                "`old_string` (it cannot overwrite an existing non-empty file)."
            )
            .into(),
            parameters: schema_object(
                vec![
                    (
                        "path",
                        s_string(),
                        "File path, relative to the workspace root or absolute.",
                    ),
                    (
                        "old_string",
                        s_string(),
                        "Exact text to find. Empty string creates a new file.",
                    ),
                    ("new_string", s_string(), "Replacement text."),
                    (
                        "replace_all",
                        s_boolean(),
                        "Replace every occurrence (default false).",
                    ),
                ],
                &["path", "old_string", "new_string"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let path = arg_str(&args, "path")?;
        let old_string = arg_str(&args, "old_string")?;
        let new_string = arg_str(&args, "new_string")?;
        let replace_all = arg_bool(&args, "replace_all");

        let abs = ctx.sandbox.resolve(&path)?;
        let display = ctx.sandbox.display(&abs);

        // Upstream: reject no-op edits early.
        if !old_string.is_empty() && old_string == new_string {
            return Err(EngineError::Tool {
                name: "edit_file".into(),
                message: "old_string and new_string are identical; nothing to change".into(),
            });
        }

        // New-file creation path (grok: empty old_string).
        if old_string.is_empty() {
            if abs.exists() {
                let existing = std::fs::read_to_string(&abs).unwrap_or_default();
                if !existing.is_empty() {
                    return Err(EngineError::Tool {
                        name: "edit_file".into(),
                        message: format!(
                            "cannot create '{display}' with an empty old_string: file exists and is non-empty"
                        ),
                    });
                }
            }
            if let Some(parent) = abs.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&abs, &new_string)?;
            let snippet = render_snippet(&new_string, &new_string, 0, CONTEXT_LINES).0;
            return Ok(ToolOutput::new(format!(
                "Created {display} ({new_len} bytes):\n{snippet}",
                new_len = new_string.len()
            )));
        }

        if !abs.exists() {
            return Err(EngineError::Tool {
                name: "edit_file".into(),
                message: format!("file not found: {display}"),
            });
        }

        let text = std::fs::read_to_string(&abs).map_err(|e| EngineError::Tool {
            name: "edit_file".into(),
            message: format!("cannot read {display} as text: {e}"),
        })?;

        // Pass 1: exact matches (upstream default path).
        let positions: Vec<usize> = text.match_indices(&old_string).map(|(i, _)| i).collect();

        let (new_text, new_positions, count) = if !positions.is_empty() {
            if positions.len() > 1 && !replace_all {
                return Err(EngineError::Tool {
                    name: "edit_file".into(),
                    message: format!(
                        "old_string matches {} places in {display}; add surrounding lines to make it unique or set replace_all",
                        positions.len()
                    ),
                });
            }
            let use_positions = if replace_all {
                positions
            } else {
                vec![positions[0]]
            };
            let count = use_positions.len();
            let (new_text, new_positions) =
                replace_using_positions(&text, &use_positions, &old_string, &new_string);
            (new_text, new_positions, count)
        } else {
            // Pass 2: confusable-normalized matching (upstream helpers).
            match find_normalized_match_positions(&text, &old_string) {
                NormalizedMatchResult::Matches(matches) => {
                    if matches.len() > 1 && !replace_all {
                        return Err(EngineError::Tool {
                            name: "edit_file".into(),
                            message: format!(
                                "old_string matches {} places in {display} (confusable-normalized); add surrounding lines to make it unique or set replace_all",
                                matches.len()
                            ),
                        });
                    }
                    let use_matches = if replace_all {
                        matches
                    } else {
                        vec![matches[0].clone()]
                    };
                    let count = use_matches.len();
                    let (new_text, new_positions) =
                        replace_normalized_matches(&text, &use_matches, &new_string);
                    (new_text, new_positions, count)
                }
                NormalizedMatchResult::Ambiguous => {
                    return Err(EngineError::Tool {
                        name: "edit_file".into(),
                        message: format!(
                            "old_string match is ambiguous in {display} after confusable normalization; add surrounding context"
                        ),
                    });
                }
                NormalizedMatchResult::NoMatch => {
                    // Pass 3: whitespace-tolerant fallback (mobile extra).
                    if let Some(spans) = tolerant_match_positions(&text, &old_string) {
                        if spans.len() > 1 && !replace_all {
                            return Err(EngineError::Tool {
                                name: "edit_file".into(),
                                message: format!(
                                    "old_string matches {} places in {display} (whitespace-tolerant); add surrounding lines to make it unique or set replace_all",
                                    spans.len()
                                ),
                            });
                        }
                        let use_spans = if replace_all {
                            spans
                        } else {
                            vec![spans[0]]
                        };
                        let count = use_spans.len();
                        let new_text = replace_spans(&text, &use_spans, &new_string);
                        // Approximate positions for snippet (start of each replacement).
                        let mut new_positions = Vec::new();
                        let mut cursor = 0usize;
                        let mut last = 0usize;
                        for &(start, end) in &use_spans {
                            cursor += start - last;
                            new_positions.push(cursor);
                            cursor += new_string.len();
                            last = end;
                        }
                        (new_text, new_positions, count)
                    } else {
                        return Err(EngineError::Tool {
                            name: "edit_file".into(),
                            message: format!("old_string not found in {display}"),
                        });
                    }
                }
            }
        };

        std::fs::write(&abs, &new_text)?;

        let first_pos = new_positions.first().copied().unwrap_or(0);
        let (snippet, _, _) = render_snippet(&new_text, &new_string, first_pos, CONTEXT_LINES);
        Ok(ToolOutput::new(format!(
            "Replaced {count} occurrence(s) in {display}:\n{snippet}"
        )))
    }
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(EditFileTool)
}
