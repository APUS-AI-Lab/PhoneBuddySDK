//! `busybox` tool — built-in applets for basic file management.
//!
//! iOS has no busybox and forbids spawning processes, so the applets are
//! implemented in pure Rust and jailed to the sandbox. This replaces the
//! desktop `bash` tool for the shell-style chores an agent normally needs
//! (`cat`, `head`, `tail`, `wc`, `sort`, `uniq`, `find`, `mkdir`, `cp`,
//! `mv`, `rm`, `touch`, `du`, `echo`, `ls`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::{arg_str, s_string, s_string_array, schema_object, Tool, ToolCtx, ToolOutput, ToolSpec};

pub struct BusyboxTool;

const APPLET_DOC: &str = "\
Applets: ls, cat, head, tail, wc, sort, uniq, find, mkdir, cp, mv, rm, touch, du, echo, pwd, basename, dirname.\n\
All paths are resolved inside the workspace sandbox. Examples:\n\
  busybox applet=\"head\" args=[\"-n\", \"20\", \"notes.md\"]\n\
  busybox applet=\"find\" args=[\".\", \"-name\", \"*.csv\"]\n\
  busybox applet=\"wc\" args=[\"-l\", \"report.md\"]";

#[async_trait]
impl Tool for BusyboxTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "busybox".into(),
            description: format!(
                "Built-in file-management applets (no shell available on mobile). {}",
                APPLET_DOC
            ),
            parameters: schema_object(
                vec![
                    ("applet", s_string(), "Applet name, e.g. head/tail/wc/find/cp/mv/rm."),
                    ("args", s_string_array(), "Arguments, busybox-style (flags then paths)."),
                ],
                &["applet", "args"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let applet = arg_str(&args, "applet")?;
        let argv: Vec<String> = args
            .get("args")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();

        match applet.as_str() {
            "ls" => applet_ls(&argv, ctx),
            "cat" => applet_cat(&argv, ctx),
            "head" => applet_head_tail(&argv, ctx, true),
            "tail" => applet_head_tail(&argv, ctx, false),
            "wc" => applet_wc(&argv, ctx),
            "sort" => applet_sort(&argv, ctx),
            "uniq" => applet_uniq(&argv, ctx),
            "find" => applet_find(&argv, ctx),
            "mkdir" => applet_mkdir(&argv, ctx),
            "cp" => applet_cp(&argv, ctx),
            "mv" => applet_mv(&argv, ctx),
            "rm" => applet_rm(&argv, ctx),
            "touch" => applet_touch(&argv, ctx),
            "du" => applet_du(&argv, ctx),
            "echo" => Ok(ToolOutput::new(argv.join(" "))),
            "pwd" => Ok(ToolOutput::new(ctx.sandbox.root().display().to_string())),
            "basename" => {
                let p = argv.first().cloned().unwrap_or_default();
                Ok(ToolOutput::new(
                    Path::new(&p)
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or(p),
                ))
            }
            "dirname" => {
                let p = argv.first().cloned().unwrap_or_default();
                Ok(ToolOutput::new(
                    Path::new(&p)
                        .parent()
                        .map(|n| n.display().to_string())
                        .unwrap_or_default(),
                ))
            }
            other => Err(EngineError::Tool {
                name: "busybox".into(),
                message: format!("unknown applet '{other}'. {APPLET_DOC}"),
            }),
        }
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

fn fail(applet: &str, message: String) -> EngineError {
    EngineError::Tool {
        name: format!("busybox:{applet}"),
        message,
    }
}

/// Split argv into flag→optional-value pairs and positional paths.
/// `value_flags` lists flags that consume the next argument (e.g. `-n 3`).
/// Supports both `-n 3` and `-n3` spellings.
fn parse_argv<'a>(
    argv: &'a [String],
    value_flags: &[&str],
) -> (Vec<(String, Option<String>)>, Vec<&'a str>) {
    let mut flags: Vec<(String, Option<String>)> = Vec::new();
    let mut paths: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if let Some(rest) = a.strip_prefix('-') {
            if rest.is_empty() {
                paths.push(a.as_str());
                i += 1;
                continue;
            }
            if let Some((f, attached)) = value_flags
                .iter()
                .find_map(|f| rest.strip_prefix(f.trim_start_matches('-')).map(|v| (*f, v)))
            {
                // Attached value spelling: -n3
                if !attached.is_empty() {
                    flags.push((f.to_string(), Some(attached.to_string())));
                } else {
                    let value = argv.get(i + 1).map(String::as_str);
                    if value.is_some() {
                        i += 1;
                    }
                    flags.push((f.to_string(), value.map(str::to_string)));
                }
            } else {
                flags.push((a.clone(), None));
            }
        } else {
            paths.push(a.as_str());
        }
        i += 1;
    }
    (flags, paths)
}

fn flag_get<'a>(flags: &'a [(String, Option<String>)], name: &str) -> Option<&'a str> {
    flags
        .iter()
        .find(|(f, _)| f == name)
        .and_then(|(_, v)| v.as_deref())
}

fn flag_has(flags: &[(String, Option<String>)], name: &str) -> bool {
    flags.iter().any(|(f, _)| f == name)
}

fn first_path<'a>(paths: &'a [&'a str], applet: &str) -> EngineResult<&'a str> {
    paths
        .first()
        .copied()
        .ok_or_else(|| fail(applet, "no path argument".into()))
}

fn read_text(abs: &Path, applet: &str) -> EngineResult<String> {
    let bytes = std::fs::read(abs).map_err(|e| fail(applet, format!("{abs:?}: {e}")))?;
    let ext = abs
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if super::binary::is_binary(&ext, &bytes) {
        return Err(fail(applet, format!("{abs:?}: binary file")));
    }
    let (cow, _, _) = encoding_rs::UTF_8.decode(&bytes);
    Ok(cow.into_owned())
}

fn applet_ls(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (flags, paths) = parse_argv(argv, &[]);
    let target = paths.first().copied().unwrap_or(".");
    let abs = ctx.sandbox.resolve(target)?;
    if !abs.is_dir() {
        return Err(fail("ls", format!("not a directory: {target}")));
    }
    let long = flag_has(&flags, "-l");
    let mut rows = Vec::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&abs)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let name = if p.is_dir() { format!("{name}/") } else { name };
        if long {
            let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            rows.push(format!("{size:>10}  {name}"));
        } else {
            rows.push(name);
        }
    }
    Ok(ToolOutput::new(if rows.is_empty() {
        "(empty directory)".to_string()
    } else {
        rows.join("\n")
    }))
}

fn applet_cat(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (_flags, paths) = parse_argv(argv, &[]);
    let target = first_path(&paths, "cat")?;
    let abs = ctx.sandbox.resolve(target)?;
    let text = read_text(&abs, "cat")?;
    Ok(ToolOutput::new(crate::tools::truncate_chars(&text, 96_000)))
}

fn applet_head_tail(argv: &[String], ctx: &ToolCtx, head: bool) -> EngineResult<ToolOutput> {
    let applet = if head { "head" } else { "tail" };
    let (flags, paths) = parse_argv(argv, &["-n"]);
    let target = first_path(&paths, applet)?;
    let n: usize = flag_get(&flags, "-n").and_then(|v| v.parse().ok()).unwrap_or(10);
    let abs = ctx.sandbox.resolve(target)?;
    let text = read_text(&abs, applet)?;
    let lines: Vec<&str> = text.lines().collect();
    let sel: Vec<&str> = if head {
        lines.iter().take(n).copied().collect()
    } else {
        lines.iter().rev().take(n).copied().collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    };
    Ok(ToolOutput::new(sel.join("\n")))
}

fn applet_wc(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (flags, paths) = parse_argv(argv, &[]);
    let target = first_path(&paths, "wc")?;
    let abs = ctx.sandbox.resolve(target)?;
    let text = read_text(&abs, "wc")?;
    let lines = text.lines().count();
    let words = text.split_whitespace().count();
    let chars = text.chars().count();
    let want_lines = flag_has(&flags, "-l");
    let want_words = flag_has(&flags, "-w");
    let want_chars = flag_has(&flags, "-c") || flag_has(&flags, "-m");
    let out = if !want_lines && !want_words && !want_chars {
        format!("{lines:>7} {words:>7} {chars:>7} {}", ctx.sandbox.display(&abs))
    } else {
        let mut parts = Vec::new();
        if want_lines {
            parts.push(lines.to_string());
        }
        if want_words {
            parts.push(words.to_string());
        }
        if want_chars {
            parts.push(chars.to_string());
        }
        parts.push(ctx.sandbox.display(&abs));
        parts.join(" ")
    };
    Ok(ToolOutput::new(out))
}

fn applet_sort(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (flags, paths) = parse_argv(argv, &[]);
    let target = first_path(&paths, "sort")?;
    let abs = ctx.sandbox.resolve(target)?;
    let text = read_text(&abs, "sort")?;
    let mut lines: Vec<&str> = text.lines().collect();
    if flag_has(&flags, "-r") {
        lines.sort_unstable_by(|a, b| b.cmp(a));
    } else {
        lines.sort_unstable();
    }
    if flag_has(&flags, "-u") {
        lines.dedup();
    }
    Ok(ToolOutput::new(lines.join("\n")))
}

fn applet_uniq(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (flags, paths) = parse_argv(argv, &[]);
    let target = first_path(&paths, "uniq")?;
    let abs = ctx.sandbox.resolve(target)?;
    let text = read_text(&abs, "uniq")?;
    let count = flag_has(&flags, "-c");
    let mut out = Vec::new();
    let mut prev: Option<&str> = None;
    let mut n = 0usize;
    let flush = |out: &mut Vec<String>, prev: Option<&str>, n: usize| {
        if let Some(p) = prev {
            if count {
                out.push(format!("{n:>7} {p}"));
            } else {
                out.push(p.to_string());
            }
        }
    };
    for line in text.lines() {
        if prev == Some(line) {
            n += 1;
        } else {
            flush(&mut out, prev, n);
            prev = Some(line);
            n = 1;
        }
    }
    flush(&mut out, prev, n);
    Ok(ToolOutput::new(out.join("\n")))
}

fn applet_find(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (flags, paths) = parse_argv(argv, &["-name"]);
    let target = first_path(&paths, "find")?;
    let abs = ctx.sandbox.resolve(target)?;
    // -name PATTERN
    let name_pat = flag_get(&flags, "-name");
    let glob_matcher = match name_pat {
        Some(g) => Some(
            globset::Glob::new(g)
                .map_err(|e| fail("find", format!("bad -name pattern: {e}")))?
                .compile_matcher(),
        ),
        None => None,
    };
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(&abs).into_iter().filter_map(|e| e.ok()) {
        if out.len() >= 500 {
            out.push("…[truncated at 500 entries]".into());
            break;
        }
        let p = entry.path();
        if p == abs {
            continue;
        }
        if let Some(gm) = &glob_matcher {
            let file = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            if !gm.is_match(&file) && !gm.is_match(p) {
                continue;
            }
        }
        out.push(ctx.sandbox.display(p));
    }
    Ok(ToolOutput::new(if out.is_empty() {
        "No matches.".to_string()
    } else {
        out.join("\n")
    }))
}

fn applet_mkdir(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (_flags, paths) = parse_argv(argv, &[]);
    let target = first_path(&paths, "mkdir")?;
    let abs = ctx.sandbox.resolve(target)?;
    std::fs::create_dir_all(&abs)?;
    Ok(ToolOutput::new(format!("created {}", ctx.sandbox.display(&abs))))
}

fn applet_cp(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (_flags, paths) = parse_argv(argv, &[]);
    if paths.len() != 2 {
        return Err(fail("cp", "usage: cp SRC DST".into()));
    }
    let src = ctx.sandbox.resolve(paths[0])?;
    let dst = ctx.sandbox.resolve(paths[1])?;
    if !src.exists() {
        return Err(fail("cp", format!("source not found: {}", paths[0])));
    }
    if src.is_dir() {
        return Err(fail("cp", "directory copy not supported; use find+read/write".into()));
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(&src, &dst)?;
    Ok(ToolOutput::new(format!(
        "copied {} -> {}",
        ctx.sandbox.display(&src),
        ctx.sandbox.display(&dst)
    )))
}

fn applet_mv(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (_flags, paths) = parse_argv(argv, &[]);
    if paths.len() != 2 {
        return Err(fail("mv", "usage: mv SRC DST".into()));
    }
    let src = ctx.sandbox.resolve(paths[0])?;
    let dst = ctx.sandbox.resolve(paths[1])?;
    if !src.exists() {
        return Err(fail("mv", format!("source not found: {}", paths[0])));
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::rename(&src, &dst)?;
    Ok(ToolOutput::new(format!(
        "moved {} -> {}",
        ctx.sandbox.display(&src),
        ctx.sandbox.display(&dst)
    )))
}

fn applet_rm(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (flags, paths) = parse_argv(argv, &[]);
    let target = first_path(&paths, "rm")?;
    let recursive = flag_has(&flags, "-r") || flag_has(&flags, "-rf") || flag_has(&flags, "-R");
    let abs = ctx.sandbox.resolve(target)?;
    // Never allow removing the sandbox root itself.
    if abs == ctx.sandbox.root() {
        return Err(fail("rm", "refusing to remove the workspace root".into()));
    }
    if !abs.exists() {
        return Err(fail("rm", format!("not found: {}", ctx.sandbox.display(&abs))));
    }
    if abs.is_dir() {
        if !recursive {
            return Err(fail("rm", "is a directory (use -r)".into()));
        }
        std::fs::remove_dir_all(&abs)?;
    } else {
        std::fs::remove_file(&abs)?;
    }
    Ok(ToolOutput::new(format!("removed {}", ctx.sandbox.display(&abs))))
}

fn applet_touch(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (_flags, paths) = parse_argv(argv, &[]);
    let target = first_path(&paths, "touch")?;
    let abs = ctx.sandbox.resolve(target)?;
    if !abs.exists() {
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&abs, "")?;
    } else {
        // Rewrite in place to bump mtime without adding a filetime dep.
        let content = std::fs::read(&abs)?;
        std::fs::write(&abs, content)?;
    }
    Ok(ToolOutput::new(format!("touched {}", ctx.sandbox.display(&abs))))
}

fn applet_du(argv: &[String], ctx: &ToolCtx) -> EngineResult<ToolOutput> {
    let (_flags, paths) = parse_argv(argv, &[]);
    let target = first_path(&paths, "du")?;
    let abs = ctx.sandbox.resolve(target)?;
    let mut total = 0u64;
    let mut rows = Vec::new();
    if abs.is_file() {
        total = std::fs::metadata(&abs)?.len();
        rows.push(format!("{:>10}  {}", total, ctx.sandbox.display(&abs)));
    } else {
        for entry in walkdir::WalkDir::new(&abs).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                total += size;
            }
        }
        rows.push(format!("{:>10}  {} (total)", total, ctx.sandbox.display(&abs)));
    }
    Ok(ToolOutput::new(rows.join("\n")))
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(BusyboxTool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_ctx() -> (tempfile::TempDir, ToolCtx) {
        let dir = tempdir().unwrap();
        let sandbox = Arc::new(crate::tools::fs::Sandbox::new(dir.path()).unwrap());
        let ctx = ToolCtx {
            sandbox,
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        (dir, ctx)
    }

    #[tokio::test]
    async fn test_busybox_ls_and_cat() {
        let (dir, ctx) = make_ctx();
        let tool = BusyboxTool;

        std::fs::write(dir.path().join("a.txt"), "hello world\n").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        // ls
        let res = tool
            .execute(serde_json::json!({"applet": "ls", "args": ["."]}), &ctx)
            .await
            .unwrap();
        assert!(res.text.contains("a.txt"));
        assert!(res.text.contains("subdir/"));

        // ls -l
        let res_l = tool
            .execute(serde_json::json!({"applet": "ls", "args": ["-l", "."]}), &ctx)
            .await
            .unwrap();
        assert!(res_l.text.contains("a.txt"));

        // cat
        let res_cat = tool
            .execute(serde_json::json!({"applet": "cat", "args": ["a.txt"]}), &ctx)
            .await
            .unwrap();
        assert_eq!(res_cat.text, "hello world\n");

        // cat binary file error
        std::fs::write(dir.path().join("bin.dat"), [0, 158, 255, 0]).unwrap();
        let res_bin = tool
            .execute(serde_json::json!({"applet": "cat", "args": ["bin.dat"]}), &ctx)
            .await;
        assert!(res_bin.is_err());
    }

    #[tokio::test]
    async fn test_busybox_head_tail_wc() {
        let (dir, ctx) = make_ctx();
        let tool = BusyboxTool;

        let content = "line1\nline2\nline3\nline4\nline5\n";
        std::fs::write(dir.path().join("lines.txt"), content).unwrap();

        // head -n 2
        let res_head = tool
            .execute(
                serde_json::json!({"applet": "head", "args": ["-n", "2", "lines.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(res_head.text, "line1\nline2");

        // head -n2 attached flag
        let res_head2 = tool
            .execute(
                serde_json::json!({"applet": "head", "args": ["-n2", "lines.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(res_head2.text, "line1\nline2");

        // tail -n 2
        let res_tail = tool
            .execute(
                serde_json::json!({"applet": "tail", "args": ["-n", "2", "lines.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(res_tail.text, "line4\nline5");

        // wc
        let res_wc = tool
            .execute(
                serde_json::json!({"applet": "wc", "args": ["-l", "lines.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(res_wc.text.contains("5"));
    }

    #[tokio::test]
    async fn test_busybox_sort_uniq() {
        let (dir, ctx) = make_ctx();
        let tool = BusyboxTool;

        let content = "banana\napple\nbanana\ncherry\n";
        std::fs::write(dir.path().join("fruits.txt"), content).unwrap();

        // sort
        let res_sort = tool
            .execute(
                serde_json::json!({"applet": "sort", "args": ["fruits.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        let lines: Vec<&str> = res_sort.text.lines().collect();
        assert_eq!(lines, vec!["apple", "banana", "banana", "cherry"]);

        // sort -r
        let res_sort_r = tool
            .execute(
                serde_json::json!({"applet": "sort", "args": ["-r", "fruits.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        let lines_r: Vec<&str> = res_sort_r.text.lines().collect();
        assert_eq!(lines_r, vec!["cherry", "banana", "banana", "apple"]);

        // sort -u
        let res_sort_u = tool
            .execute(
                serde_json::json!({"applet": "sort", "args": ["-u", "fruits.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        let lines_u: Vec<&str> = res_sort_u.text.lines().collect();
        assert_eq!(lines_u, vec!["apple", "banana", "cherry"]);

        // uniq
        let uniq_content = "a\na\nb\nc\nc\n";
        std::fs::write(dir.path().join("uniq.txt"), uniq_content).unwrap();
        let res_uniq = tool
            .execute(
                serde_json::json!({"applet": "uniq", "args": ["uniq.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(res_uniq.text, "a\nb\nc");

        // uniq -c
        let res_uniq_c = tool
            .execute(
                serde_json::json!({"applet": "uniq", "args": ["-c", "uniq.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(res_uniq_c.text.contains("2 a"));
        assert!(res_uniq_c.text.contains("1 b"));
        assert!(res_uniq_c.text.contains("2 c"));
    }

    #[tokio::test]
    async fn test_busybox_find_mkdir_cp_mv_rm_touch_du() {
        let (dir, ctx) = make_ctx();
        let tool = BusyboxTool;

        // mkdir
        tool.execute(
            serde_json::json!({"applet": "mkdir", "args": ["nested/dir"]}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(dir.path().join("nested/dir").is_dir());

        // touch
        tool.execute(
            serde_json::json!({"applet": "touch", "args": ["nested/dir/file.txt"]}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(dir.path().join("nested/dir/file.txt").is_file());

        // cp
        tool.execute(
            serde_json::json!({"applet": "cp", "args": ["nested/dir/file.txt", "copied.txt"]}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(dir.path().join("copied.txt").is_file());

        // mv
        tool.execute(
            serde_json::json!({"applet": "mv", "args": ["copied.txt", "moved.txt"]}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(!dir.path().join("copied.txt").exists());
        assert!(dir.path().join("moved.txt").is_file());

        // find
        let res_find = tool
            .execute(
                serde_json::json!({"applet": "find", "args": [".", "-name", "*.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(res_find.text.contains("moved.txt"));
        assert!(res_find.text.contains("file.txt"));

        // du
        let res_du = tool
            .execute(serde_json::json!({"applet": "du", "args": ["."]}), &ctx)
            .await
            .unwrap();
        assert!(res_du.text.contains("(total)"));

        // rm
        tool.execute(
            serde_json::json!({"applet": "rm", "args": ["moved.txt"]}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(!dir.path().join("moved.txt").exists());

        // rm -r
        tool.execute(
            serde_json::json!({"applet": "rm", "args": ["-r", "nested"]}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(!dir.path().join("nested").exists());

        // rm sandbox root rejected
        let rm_root = tool
            .execute(
                serde_json::json!({"applet": "rm", "args": ["-r", "."]}),
                &ctx,
            )
            .await;
        assert!(rm_root.is_err());
    }

    #[tokio::test]
    async fn test_busybox_misc_applets() {
        let (dir, ctx) = make_ctx();
        let tool = BusyboxTool;

        // echo
        let res_echo = tool
            .execute(
                serde_json::json!({"applet": "echo", "args": ["hello", "world"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(res_echo.text, "hello world");

        // pwd
        let res_pwd = tool
            .execute(serde_json::json!({"applet": "pwd", "args": []}), &ctx)
            .await
            .unwrap();
        assert_eq!(res_pwd.text, ctx.sandbox.root().display().to_string());

        // basename
        let res_base = tool
            .execute(
                serde_json::json!({"applet": "basename", "args": ["foo/bar/baz.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(res_base.text, "baz.txt");

        // dirname
        let res_dir = tool
            .execute(
                serde_json::json!({"applet": "dirname", "args": ["foo/bar/baz.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(res_dir.text, "foo/bar");

        // unknown applet
        let res_unknown = tool
            .execute(
                serde_json::json!({"applet": "invalid_applet", "args": []}),
                &ctx,
            )
            .await;
        assert!(res_unknown.is_err());
    }
}
