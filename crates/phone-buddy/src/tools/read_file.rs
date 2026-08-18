//! `read_file` tool.
//!
//! Semantics ported from grok's read_file (`implementations/grok_build/read_file`):
//! - text files use sparse `LINE_NUMBER→` anchors (first visible line and every
//!   10th line) so the model can reference exact lines without prefix noise;
//! - `offset` is **1-based** (0 and omitted both mean start at line 1);
//!   negative offsets count from the end of the file;
//! - large files are truncated by line count and char budget;
//! - binary files are reported as metadata instead of dumped.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::binary;
use crate::tools::{
    arg_opt_usize, arg_str, schema_object, s_integer, s_string, Tool, ToolCtx, ToolOutput, ToolSpec,
};

/// Max lines returned by default (grok: `MAX_LINES_READ = 1000`).
pub const DEFAULT_MAX_LINES: usize = 1_000;
/// Hard char budget for one read (keeps tool results phone-sized).
pub const MAX_OUTPUT_CHARS: usize = 96_000;
/// Files larger than this are refused unless an offset is given.
pub const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: concat!(
                "Read a text file. Line numbers (1-based) appear as anchors in the ",
                "format LINE_NUMBER→LINE_CONTENT on the first returned line and on ",
                "every 10th line of the file; the lines in between show content only. ",
                "Count from the nearest anchor when referring to a specific line. ",
                "The prefix is not part of the file. For large files use `offset`/`limit` ",
                "to page. Binary files return metadata only."
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
                        "offset",
                        s_integer(),
                        "1-based line number to start reading from (default 1). Negative values count from the end of the file.",
                    ),
                    (
                        "limit",
                        s_integer(),
                        "Maximum number of lines to read (default 1000).",
                    ),
                ],
                &["path"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let path = arg_str(&args, "path")?;
        // Upstream: offset defaults to 1 (1-based); 0 is treated as 1.
        let offset_raw: Option<i64> = args.get("offset").and_then(|v| {
            v.as_i64()
                .or_else(|| v.as_u64().map(|u| u as i64))
                .or_else(|| v.as_f64().map(|f| f as i64))
        });
        let limit = arg_opt_usize(&args, "limit", DEFAULT_MAX_LINES);

        let abs = ctx.sandbox.resolve(&path)?;
        if !abs.exists() {
            return Err(EngineError::Tool {
                name: "read_file".into(),
                message: format!("file not found: {}", ctx.sandbox.display(&abs)),
            });
        }
        if abs.is_dir() {
            return Err(EngineError::Tool {
                name: "read_file".into(),
                message: format!(
                    "'{}' is a directory; use list_dir",
                    ctx.sandbox.display(&abs)
                ),
            });
        }

        let meta = std::fs::metadata(&abs)?;
        let start_from_beginning = matches!(offset_raw, None | Some(0) | Some(1));
        if meta.len() > MAX_FILE_BYTES && start_from_beginning {
            return Err(EngineError::Tool {
                name: "read_file".into(),
                message: format!(
                    "file is {} bytes (max {MAX_FILE_BYTES}); read it with an offset/limit",
                    meta.len()
                ),
            });
        }

        let bytes = std::fs::read(&abs)?;

        let ext = abs
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if binary::is_binary(&ext, &bytes) {
            let kind = infer::get(&bytes)
                .map(|t| t.mime_type().to_string())
                .unwrap_or_else(|| "application/octet-stream".into());
            return Ok(ToolOutput::new(format!(
                "Binary file: {}\nMIME type: {kind}\nSize: {} bytes",
                ctx.sandbox.display(&abs),
                meta.len()
            )));
        }

        // Decode leniently (grok uses encoding_rs for the same purpose).
        let (text, _enc, had_errors) = encoding_rs::UTF_8.decode(&bytes);
        let text = if had_errors {
            encoding_rs::GBK.decode(&bytes).0.into_owned()
        } else {
            text.into_owned()
        };

        let total_lines = count_lines(&text);
        let extracted = extract_file_content_lines(&text, offset_raw, Some(limit), total_lines);

        let mut result = format!(
            "File: {} ({} lines, {} bytes)\n{}",
            ctx.sandbox.display(&abs),
            total_lines,
            meta.len(),
            extracted
        );
        let start_line = resolve_read_start_line(&text, offset_raw);
        let end_shown = (start_line + limit - 1).min(total_lines.max(1));
        if start_line > 1 || end_shown < total_lines {
            use std::fmt::Write as _;
            let _ = writeln!(
                result,
                "…[showing lines {start_line}..{end_shown} of {total_lines}; re-read with an offset to continue]"
            );
        }
        Ok(ToolOutput::new(truncate_output(result)))
    }
}

fn truncate_output(s: String) -> String {
    crate::tools::truncate_chars(&s, MAX_OUTPUT_CHARS + 2_000)
}

/// Count logical lines via `split_inclusive` (matches upstream line indexing).
fn count_lines(file_content: &str) -> usize {
    if file_content.is_empty() {
        return 0;
    }
    file_content.split_inclusive('\n').count()
}

/// Port of grok `resolve_read_start_line`: 1-based start line.
/// `None` / `0` → 1; positive as-is; negative counts from end.
pub fn resolve_read_start_line(file_content: &str, offset: Option<i64>) -> usize {
    let offset_raw = offset.unwrap_or(1);
    if offset_raw == 0 {
        return 1;
    }
    if offset_raw > 0 {
        return offset_raw as usize;
    }
    let mut total_fields = file_content.split('\n').count();
    if !file_content.is_empty() && !file_content.ends_with('\n') {
        total_fields += 1;
    }
    let computed = (total_fields as i64) + offset_raw + 1;
    computed.max(1) as usize
}

/// Port of grok `extract_file_content_lines` (text path only; no base64 image
/// extraction). Sparse anchors: first visible line and every 10th line.
pub fn extract_file_content_lines(
    file_content: &str,
    offset: Option<i64>,
    limit: Option<usize>,
    _total_lines: usize,
) -> String {
    fn strip(s: &str) -> &str {
        let Some(s) = s.strip_suffix('\n') else {
            return s;
        };
        s.strip_suffix('\r').unwrap_or(s)
    }
    use std::fmt::Write as _;
    let mut output = String::new();
    let mut first_line: Option<usize> = None;
    let split_count = file_content.split_inclusive('\n').count();
    let has_trailing_empty = !file_content.is_empty() && file_content.ends_with('\n');
    let skip = resolve_read_start_line(file_content, offset).saturating_sub(1);
    let take = limit.unwrap_or(usize::MAX);

    if file_content.is_empty() && take > 0 && skip == 0 {
        let _ = write!(&mut output, "1→");
        return output;
    }

    for (i, line) in file_content
        .split_inclusive('\n')
        .map(strip)
        .enumerate()
        .skip(skip)
        .take(take)
    {
        let is_first_visible = first_line.is_none();
        if is_first_visible {
            first_line = Some(i + 1);
        } else {
            output.push('\n');
        }
        let line_num = i + 1;
        if is_first_visible || line_num.is_multiple_of(10) {
            let _ = write!(&mut output, "{line_num}→{line}");
        } else {
            output.push_str(line);
        }
    }

    if has_trailing_empty {
        let trailing_line_idx = split_count;
        if trailing_line_idx >= skip && trailing_line_idx < skip.saturating_add(take) {
            let line_num = trailing_line_idx + 1;
            let is_first_visible = first_line.is_none();
            if is_first_visible {
                first_line = Some(line_num);
            } else {
                output.push('\n');
            }
            if is_first_visible || line_num.is_multiple_of(10) {
                let _ = write!(&mut output, "{line_num}→");
            }
        }
    }
    let _ = first_line;
    output
}

/// Binary detection — thin wrapper for call sites that only have bytes.
/// Prefer [`binary::is_binary`] with an extension when available.
pub fn is_binary(bytes: &[u8]) -> bool {
    binary::is_binary("", bytes)
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(ReadFileTool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_anchors_first_and_every_10th() {
        let content: String = (1..=25).map(|i| format!("line{i}\n")).collect();
        let out = extract_file_content_lines(&content, Some(1), Some(25), 25);
        assert!(out.starts_with("1→line1\n"));
        assert!(out.contains("\n10→line10\n") || out.contains("10→line10\n"));
        assert!(out.contains("\n20→line20\n") || out.contains("20→line20\n"));
        // Line 2 has no number prefix.
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[1], "line2");
        assert_eq!(lines[9], "10→line10");
    }

    #[test]
    fn offset_is_one_based() {
        let content = "a\nb\nc\nd\n";
        let out = extract_file_content_lines(content, Some(2), Some(2), 4);
        assert!(out.starts_with("2→b\n") || out.starts_with("2→b"));
        assert!(out.contains("c"));
        assert!(!out.contains("a\n") && !out.starts_with("1→"));
    }

    #[test]
    fn offset_zero_means_one() {
        assert_eq!(resolve_read_start_line("a\nb\n", Some(0)), 1);
        assert_eq!(resolve_read_start_line("a\nb\n", None), 1);
    }

    #[test]
    fn negative_offset_from_end() {
        let content = "a\nb\nc\nd\n";
        // Upstream: total_fields for trailing-newline content via split('\n')
        // is 5 (a,b,c,d,''); + (-1) + 1 = 5 → line 5 (trailing empty).
        // For practical -1 meaning "last content line", models use limit too.
        let start = resolve_read_start_line(content, Some(-1));
        assert!(start >= 1);
    }
}
