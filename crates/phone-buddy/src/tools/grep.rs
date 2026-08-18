//! `grep` tool — content search.
//!
//! Grok's grep tool shells out to a ripgrep binary, which iOS cannot do.
//! This implementation uses ripgrep's own pure-Rust library crates
//! (`ignore`, `globset`, `regex`) instead, keeping the input schema and
//! output formatting faithful to the desktop tool
//! (`format_content_output`, mode-aware head limits, workspace_result wrapper).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::{
    arg_opt_str, arg_str, schema_object, s_enum, s_integer, s_string, Tool, ToolCtx, ToolOutput,
    ToolSpec,
};

/// Hard max when the model passes an explicit `head_limit` (content lines).
const CONTENT_LINE_LIMIT: usize = 2_000;
/// Default when `head_limit` is omitted (content).
const CONTENT_LINE_DEFAULT: usize = 200;
const FILE_COUNT_LIMIT: usize = 10_000;
/// Default when `head_limit` is omitted (files/count modes).
const FILE_COUNT_DEFAULT: usize = 500;
/// Per-line char trim (grok: `DEFAULT_MAX_CHARS_PER_LINE`).
const DEFAULT_MAX_CHARS_PER_LINE: usize = 1_000;
/// Overall card body byte budget.
const MAX_OUTPUT_BYTES: usize = 60_000;
/// Skip files larger than this (upstream rg `--max-filesize` is 5M).
const MAX_FILE_BYTES: u64 = 5 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Content,
    FilesWithMatches,
    Count,
}

impl OutputMode {
    fn parse(s: &str) -> Self {
        match s {
            "files_with_matches" => Self::FilesWithMatches,
            "count" => Self::Count,
            _ => Self::Content,
        }
    }
}

/// Port of grok `resolve_effective_head_limit`.
fn resolve_effective_head_limit(head_limit: Option<usize>, mode: OutputMode) -> usize {
    let (default, cap) = match mode {
        OutputMode::Content => (CONTENT_LINE_DEFAULT, CONTENT_LINE_LIMIT),
        OutputMode::FilesWithMatches | OutputMode::Count => (FILE_COUNT_DEFAULT, FILE_COUNT_LIMIT),
    };
    head_limit.unwrap_or(default).min(cap)
}

/// Port of grok `trim_line` for match context.
fn trim_line(line: &str, max_chars: usize) -> String {
    crate::tools::truncate_line(line, max_chars)
}

fn count_content_matches(lines: &[String]) -> usize {
    lines
        .iter()
        .filter(|l| {
            let parts: Vec<&str> = l.splitn(3, ':').collect();
            parts.len() >= 3 && parts[1].chars().all(|c| c.is_ascii_digit()) && !parts[1].is_empty()
        })
        .count()
}

fn first_idx_exceed_cum_limit(lines: &[String], max_bytes: usize) -> usize {
    let mut cum = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let add = line.len() + 1;
        if cum + add > max_bytes {
            return i;
        }
        cum += add;
    }
    lines.len()
}

/// Port of grok `format_content_output`.
fn format_content_output(output_lines: Vec<String>, is_truncated: bool) -> String {
    let is_truncated_str = if is_truncated { "at least " } else { "" };
    let num_matching_lines = count_content_matches(&output_lines);
    let mut final_output_lines = vec![format!(
        "Found {is_truncated_str}{num_matching_lines} matching lines"
    )];

    let trimmed_lines: Vec<String> = output_lines
        .iter()
        .map(|line| trim_line(line, DEFAULT_MAX_CHARS_PER_LINE))
        .collect();

    let cut_idx = first_idx_exceed_cum_limit(&trimmed_lines, MAX_OUTPUT_BYTES);
    final_output_lines.extend_from_slice(&trimmed_lines[..cut_idx]);

    let remaining_matches = count_content_matches(&trimmed_lines[cut_idx..]);
    if remaining_matches > 0 {
        final_output_lines.push(format!(
            "... [{is_truncated_str}{remaining_matches} lines truncated] ..."
        ));
    }

    final_output_lines.join("\n")
}

/// Port of grok `format_files_with_matches_output`.
fn format_files_with_matches_output(output_lines: Vec<String>, is_truncated: bool) -> String {
    let is_truncated_str = if is_truncated { "at least " } else { "" };
    let mut final_output_lines = vec![format!(
        "Found {is_truncated_str}{} files",
        output_lines.len()
    )];

    let trimmed_lines: Vec<String> = output_lines
        .iter()
        .map(|line| trim_line(line, DEFAULT_MAX_CHARS_PER_LINE))
        .collect();

    let cut_idx = first_idx_exceed_cum_limit(&trimmed_lines, MAX_OUTPUT_BYTES);
    final_output_lines.extend_from_slice(&trimmed_lines[..cut_idx]);

    if output_lines.len() > cut_idx {
        final_output_lines.push(format!(
            "... [{is_truncated_str}{} lines truncated] ...",
            output_lines.len() - cut_idx
        ));
    }

    final_output_lines.join("\n")
}

/// Port of grok `format_count_output`.
fn format_count_output(output_lines: Vec<String>, is_truncated: bool) -> String {
    let is_truncated_str = if is_truncated { "at least " } else { "" };
    let mut sum_matches = 0usize;
    for line in &output_lines {
        if let Some(count_str) = line.split(':').next_back() {
            if let Ok(count) = count_str.parse::<usize>() {
                sum_matches += count;
            }
        }
    }
    let mut final_output_lines = vec![format!(
        "Found {is_truncated_str}{sum_matches} matches across {} files",
        output_lines.len()
    )];
    let trimmed_lines: Vec<String> = output_lines
        .iter()
        .map(|line| trim_line(line, DEFAULT_MAX_CHARS_PER_LINE))
        .collect();
    let cut_idx = first_idx_exceed_cum_limit(&trimmed_lines, MAX_OUTPUT_BYTES);
    final_output_lines.extend_from_slice(&trimmed_lines[..cut_idx]);
    final_output_lines.join("\n")
}

fn finalize_grep(
    output_lines: Vec<String>,
    is_truncated: bool,
    mode: OutputMode,
    workspace_path: &str,
) -> String {
    if output_lines.is_empty() {
        return format!(
            "<workspace_result workspace_path=\"{workspace_path}\">\nNo matches found\n</workspace_result>"
        );
    }
    let body = match mode {
        OutputMode::Content => format_content_output(output_lines, is_truncated),
        OutputMode::FilesWithMatches => format_files_with_matches_output(output_lines, is_truncated),
        OutputMode::Count => format_count_output(output_lines, is_truncated),
    };
    format!("<workspace_result workspace_path=\"{workspace_path}\">\n{body}\n</workspace_result>")
}

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep".into(),
            description: concat!(
                "Search file contents with a regular expression (ripgrep-style). ",
                "Respects .gitignore. Use output_mode to get matching lines, ",
                "file paths, or counts."
            )
            .into(),
            parameters: schema_object(
                vec![
                    ("pattern", s_string(), "Regular expression to search for."),
                    (
                        "path",
                        s_string(),
                        "File or directory to search in (default: workspace root).",
                    ),
                    (
                        "glob",
                        s_string(),
                        "Glob filter, e.g. \"*.rs\" or \"*.{ts,tsx}\".",
                    ),
                    (
                        "-i",
                        serde_json::json!({"type": "boolean"}),
                        "Case-insensitive search.",
                    ),
                    ("-A", s_integer(), "Lines to show after each match."),
                    ("-B", s_integer(), "Lines to show before each match."),
                    (
                        "-C",
                        s_integer(),
                        "Lines to show before and after each match.",
                    ),
                    (
                        "multiline",
                        serde_json::json!({"type": "boolean"}),
                        "Enable multiline mode (default true for line-by-line search).",
                    ),
                    (
                        "output_mode",
                        s_enum(&["content", "files_with_matches", "count"]),
                        "Output format (default content).",
                    ),
                    (
                        "head_limit",
                        s_integer(),
                        "Limit output entries (content default 200, files/count default 500).",
                    ),
                ],
                &["pattern"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let pattern = arg_str(&args, "pattern")?;
        let path = arg_opt_str(&args, "path").unwrap_or_else(|| ".".into());
        let glob = arg_opt_str(&args, "glob");
        let case_insensitive = args.get("-i").and_then(Value::as_bool).unwrap_or(false);
        let after = args.get("-A").and_then(Value::as_u64).unwrap_or(0) as usize;
        let before = args.get("-B").and_then(Value::as_u64).unwrap_or(0) as usize;
        let context = args.get("-C").and_then(Value::as_u64).unwrap_or(0) as usize;
        let multiline = args
            .get("multiline")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let mode = OutputMode::parse(
            &arg_opt_str(&args, "output_mode").unwrap_or_else(|| "content".into()),
        );
        let head_raw = args
            .get("head_limit")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .filter(|v| *v > 0);
        let head_limit = resolve_effective_head_limit(head_raw, mode);

        let (before, after) = if context > 0 {
            (context.max(before), context.max(after))
        } else {
            (before, after)
        };

        let re = regex::RegexBuilder::new(&pattern)
            .case_insensitive(case_insensitive)
            .multi_line(multiline)
            .build()
            .map_err(|e| EngineError::ToolArgs {
                name: "grep".into(),
                message: format!("invalid regex: {e}"),
            })?;

        let glob_matcher = match &glob {
            Some(g) => Some(
                globset::GlobSetBuilder::new()
                    .add(globset::Glob::new(g).map_err(|e| EngineError::ToolArgs {
                        name: "grep".into(),
                        message: format!("invalid glob: {e}"),
                    })?)
                    .build()
                    .map_err(|e| EngineError::ToolArgs {
                        name: "grep".into(),
                        message: format!("invalid glob: {e}"),
                    })?,
            ),
            None => None,
        };

        let abs = ctx.sandbox.resolve(&path)?;
        if !abs.exists() {
            return Err(EngineError::Tool {
                name: "grep".into(),
                message: format!("path not found: {}", ctx.sandbox.display(&abs)),
            });
        }

        let workspace = ctx.sandbox.display(ctx.sandbox.root());

        if abs.is_file() {
            let display = ctx.sandbox.display(&abs);
            let mut found = search_file_lines(&abs, &re, mode, before, after, &display)?;
            let truncated = found.len() > head_limit;
            found.truncate(head_limit);
            return Ok(ToolOutput::new(finalize_grep(
                found, truncated, mode, &workspace,
            )));
        }

        let mut lines_out: Vec<String> = Vec::new();
        let mut entries = 0usize;
        let mut truncated = false;

        let walker = ignore::WalkBuilder::new(&abs)
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .require_git(false)
            .follow_links(false)
            .build();

        'walk: for entry in walker {
            let Ok(entry) = entry else { continue };
            let p = entry.path();
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            if let Some(gm) = &glob_matcher {
                if !gm.is_match(p) {
                    continue;
                }
            }
            if entry
                .metadata()
                .map(|m| m.len() > MAX_FILE_BYTES)
                .unwrap_or(false)
            {
                continue;
            }

            match search_file_lines(p, &re, mode, before, after, &ctx.sandbox.display(p)) {
                Ok(mut found) => {
                    if found.is_empty() {
                        continue;
                    }
                    for line in found.drain(..) {
                        lines_out.push(line);
                        entries += 1;
                        if entries >= head_limit {
                            truncated = true;
                            break 'walk;
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        Ok(ToolOutput::new(finalize_grep(
            lines_out, truncated, mode, &workspace,
        )))
    }
}

fn search_file_lines(
    path: &std::path::Path,
    re: &regex::Regex,
    mode: OutputMode,
    before: usize,
    after: usize,
    display: &str,
) -> EngineResult<Vec<String>> {
    let Ok(bytes) = std::fs::read(path) else {
        return Ok(Vec::new());
    };
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if super::binary::is_binary(&ext, &bytes) {
        return Ok(Vec::new());
    }
    let (cow, _, _) = encoding_rs::UTF_8.decode(&bytes);
    let text = cow.into_owned();
    let lines: Vec<&str> = text.lines().collect();

    let mut out: Vec<String> = Vec::new();

    let match_line_indices: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| re.is_match(l))
        .map(|(i, _)| i)
        .collect();

    if match_line_indices.is_empty() {
        return Ok(out);
    }

    match mode {
        OutputMode::FilesWithMatches => {
            out.push(display.to_string());
        }
        OutputMode::Count => {
            out.push(format!("{display}:{}", match_line_indices.len()));
        }
        OutputMode::Content => {
            let mut last_end: Option<usize> = None;
            for &i in &match_line_indices {
                let start = i.saturating_sub(before);
                let end = (i + after).min(lines.len().saturating_sub(1));

                let actual_start = match last_end {
                    Some(le) if start <= le => le + 1,
                    _ => start,
                };

                if actual_start > i {
                    last_end = Some(i.max(last_end.unwrap_or(0)));
                    continue;
                }

                for j in actual_start..=end {
                    let sep = if j == i { ':' } else { '-' };
                    out.push(format!("{}:{}{}{}", display, j + 1, sep, lines[j]));
                }
                last_end = Some(end);
            }
        }
    }
    Ok(out)
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(GrepTool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_limit_defaults_by_mode() {
        assert_eq!(
            resolve_effective_head_limit(None, OutputMode::Content),
            200
        );
        assert_eq!(
            resolve_effective_head_limit(None, OutputMode::FilesWithMatches),
            500
        );
        assert_eq!(
            resolve_effective_head_limit(Some(9999), OutputMode::Content),
            2000
        );
    }

    #[test]
    fn finalize_wraps_workspace_result() {
        let out = finalize_grep(
            vec!["src/a.rs:1:hello".into()],
            false,
            OutputMode::Content,
            "/tmp/ws",
        );
        assert!(out.starts_with("<workspace_result "));
        assert!(out.contains("Found 1 matching lines"));
        assert!(out.contains("</workspace_result>"));
    }
}
