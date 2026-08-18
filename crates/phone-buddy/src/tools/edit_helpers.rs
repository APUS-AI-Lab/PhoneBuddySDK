//! Shared string-edit helpers — ported from grok-build
//! `implementations/grok_build/search_replace/helpers.rs`.
//!
//! Pure logic: exact replace, confusable-normalized matching, and numbered
//! snippets. No filesystem or process dependencies.

use crate::tools::unicode_confusables::{build_offset_map, normalize_confusables};

/// Lines of context shown around each edit (grok: `CONTEXT_LINES`).
pub const CONTEXT_LINES: usize = 3;

/// Render a snippet of the file with line numbers around the edit.
/// Port of grok `render_snippet`.
pub fn render_snippet(
    new_text: &str,
    new_string: &str,
    start_pos: usize,
    context_size: usize,
) -> (String, String, String) {
    let LineRange {
        start_line,
        end_line,
    } = compute_line_range(new_text, start_pos, new_string);
    let total_lines_count = new_text.split_inclusive('\n').count();
    let lines = new_text.split_inclusive('\n').collect::<Vec<_>>();

    let snippet_start = start_line.saturating_sub(context_size);
    let snippet_end = (end_line + context_size).min(total_lines_count.saturating_sub(1));

    let before_context = if snippet_start < start_line {
        lines[snippet_start..start_line].join("")
    } else {
        String::new()
    };

    let after_context = if end_line < snippet_end {
        lines[(end_line + 1)..=snippet_end].join("")
    } else {
        String::new()
    };

    let snippet = lines
        .iter()
        .enumerate()
        .map(|(line_num, line)| format!("{}→{}", line_num + 1, line))
        .skip(snippet_start)
        .take(snippet_end - snippet_start + 1)
        .collect::<String>();

    (snippet, before_context, after_context)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LineRange {
    pub start_line: usize,
    pub end_line: usize,
}

/// Compute the 0-based inclusive line range of the inserted text.
pub fn compute_line_range(text: &str, start_pos: usize, inserted_text: &str) -> LineRange {
    let start_pos = start_pos.min(text.len());
    let start_line = text[..start_pos].matches('\n').count();
    let lines_in_inserted = inserted_text.split_inclusive('\n').count().max(1);
    let end_line = start_line + lines_in_inserted - 1;
    LineRange {
        start_line,
        end_line,
    }
}

/// Replace text at specific positions and return new text with new positions.
pub fn replace_using_positions(
    text: &str,
    match_positions: &[usize],
    old_string: &str,
    new_string: &str,
) -> (String, Vec<usize>) {
    let mut new_text = String::new();
    let mut new_positions: Vec<usize> = Vec::with_capacity(match_positions.len());
    let mut last_end: usize = 0;

    for &pos in match_positions {
        new_text.push_str(&text[last_end..pos]);
        new_positions.push(new_text.len());
        new_text.push_str(new_string);
        last_end = pos + old_string.len();
    }

    new_text.push_str(&text[last_end..]);
    (new_text, new_positions)
}

/// A single match found via confusable-normalized comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedMatch {
    pub original_start: usize,
    pub original_len: usize,
}

/// Result of normalized match search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedMatchResult {
    NoMatch,
    Matches(Vec<NormalizedMatch>),
    Ambiguous,
}

/// Find match positions using confusable-normalized comparison.
/// Port of grok `find_normalized_match_positions`.
pub fn find_normalized_match_positions(text: &str, pattern: &str) -> NormalizedMatchResult {
    let (norm_text, offset_map) = build_offset_map(text);
    let norm_pattern = normalize_confusables(pattern);

    if norm_pattern.is_empty() {
        return NormalizedMatchResult::NoMatch;
    }

    let mut validated = Vec::new();
    let mut had_rejected_candidates = false;

    for (norm_start, _) in norm_text.match_indices(&norm_pattern) {
        let norm_end = norm_start + norm_pattern.len();
        let orig_start = offset_map[norm_start];
        let orig_end = offset_map[norm_end];

        if orig_end <= orig_start {
            had_rejected_candidates = true;
            continue;
        }

        let orig_slice = &text[orig_start..orig_end];
        if normalize_confusables(orig_slice) != norm_pattern {
            had_rejected_candidates = true;
            continue;
        }

        validated.push(NormalizedMatch {
            original_start: orig_start,
            original_len: orig_end - orig_start,
        });
    }

    if validated.is_empty() {
        return if had_rejected_candidates {
            NormalizedMatchResult::Ambiguous
        } else {
            NormalizedMatchResult::NoMatch
        };
    }

    for window in validated.windows(2) {
        let end_of_prev = window[0].original_start + window[0].original_len;
        if end_of_prev > window[1].original_start {
            return NormalizedMatchResult::Ambiguous;
        }
    }

    NormalizedMatchResult::Matches(validated)
}

/// Replace text at normalized-match positions.
pub fn replace_normalized_matches(
    text: &str,
    matches: &[NormalizedMatch],
    new_string: &str,
) -> (String, Vec<usize>) {
    let mut result = String::new();
    let mut new_positions: Vec<usize> = Vec::with_capacity(matches.len());
    let mut last_end: usize = 0;

    for m in matches {
        result.push_str(&text[last_end..m.original_start]);
        new_positions.push(result.len());
        result.push_str(new_string);
        last_end = m.original_start + m.original_len;
    }

    result.push_str(&text[last_end..]);
    (result, new_positions)
}

/// Whitespace-tolerant matching (mobile extra pass): compare line-by-line with
/// trailing whitespace removed. Not a substitute for confusable matching.
pub fn tolerant_match_positions(text: &str, pattern: &str) -> Option<Vec<(usize, usize)>> {
    let pattern_lines: Vec<&str> = pattern.split('\n').collect();
    if pattern_lines.is_empty() || pattern_lines.iter().all(|l| l.trim().is_empty()) {
        return None;
    }
    let text_lines: Vec<(usize, &str)> = line_offsets(text);
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i + pattern_lines.len() <= text_lines.len() {
        let ok = pattern_lines
            .iter()
            .enumerate()
            .all(|(j, pl)| text_lines[i + j].1.trim_end() == pl.trim_end());
        if ok {
            let start = text_lines[i].0;
            let last_idx = i + pattern_lines.len() - 1;
            let (off, line) = text_lines[last_idx];
            let end = off + line.len();
            spans.push((start, end.min(text.len())));
            i += pattern_lines.len();
        } else {
            i += 1;
        }
    }
    if spans.is_empty() {
        None
    } else {
        Some(spans)
    }
}

fn line_offsets(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    for line in text.split('\n') {
        out.push((offset, line));
        offset += line.len() + 1;
    }
    if text.ends_with('\n') {
        out.pop();
    }
    out
}

/// Replace by original-text spans (start, end).
pub fn replace_spans(text: &str, spans: &[(usize, usize)], new_string: &str) -> String {
    let mut out = String::new();
    let mut last = 0;
    for &(start, end) in spans.iter() {
        out.push_str(&text[last..start]);
        out.push_str(new_string);
        last = end;
    }
    out.push_str(&text[last..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_snippet_middle_of_file_contexts() {
        let new_text = "one\ntwo NEW here\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n";
        let inserted = "NEW here";
        let start_pos = new_text.find(inserted).unwrap();

        let (snippet, before_context, after_context) =
            render_snippet(new_text, inserted, start_pos, 3);

        let expected_snippet = "1→one\n2→two NEW here\n3→three\n4→four\n5→five\n";
        assert_eq!(snippet, expected_snippet);
        assert_eq!(before_context, "one\n");
        assert_eq!(after_context, "three\nfour\nfive\n");
    }

    #[test]
    fn confusable_smart_quotes_match() {
        let text = "He said \u{201C}hello\u{201D}.";
        let pattern = "He said \"hello\".";
        match find_normalized_match_positions(text, pattern) {
            NormalizedMatchResult::Matches(m) => {
                assert_eq!(m.len(), 1);
                let (new_text, _) = replace_normalized_matches(text, &m, "He said \"hi\".");
                assert!(new_text.contains("hi"));
            }
            other => panic!("expected Matches, got {other:?}"),
        }
    }
}
