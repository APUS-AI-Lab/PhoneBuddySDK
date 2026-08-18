//! `list_dir` tool — budgeted directory listing.
//!
//! Pure tree algorithm ported from grok-build
//! `implementations/grok_build/list_dir`: `DirNode` + seed depth-1 + deep walk
//! + `budget_expand` with collapsed extension summaries.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::{
    arg_opt_usize, arg_str, schema_object, s_integer, s_string, Tool, ToolCtx, ToolOutput, ToolSpec,
};

const DEFAULT_MAX_OUTPUT_CHARS: usize = 10_000;
const TOP_K_EXTENSIONS: usize = 3;
const ROOT_TRUNCATION_NOTICE: &str = "    ...\n\n\
    Note: this directory is too large to list fully. Try list_dir on a narrower path, or \
    use grep / busybox.";
/// Hard cap on deep-walk items (depth ≥ 2). Upstream desktop uses 100_000;
/// mobile keeps the same algorithm with a phone-friendly default that still
/// far exceeds typical app sandboxes.
const MAX_GLOBAL_ITEMS: usize = 100_000;
const MAX_SEED_ITEMS: usize = 100_000;
const _: () = assert!(MAX_SEED_ITEMS == MAX_GLOBAL_ITEMS);

#[derive(Debug, Default)]
struct DirAccum {
    total_files: usize,
    by_ext: HashMap<String, usize>,
}

impl DirAccum {
    fn add_ext(&mut self, ext: &str) {
        self.total_files += 1;
        *self.by_ext.entry(ext.to_owned()).or_default() += 1;
    }

    /// Render a summary like `[3 files in subtree: 2 *.rs, 1 *.toml]`.
    fn to_summary(&self, top_n: usize) -> String {
        if self.by_ext.is_empty() {
            return String::new();
        }
        let mut items: Vec<(String, usize)> =
            self.by_ext.iter().map(|(k, v)| (k.clone(), *v)).collect();
        items.sort_by(|a, b| match b.1.cmp(&a.1) {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        });
        let mut parts: Vec<String> = Vec::new();
        let mut top_sum: usize = 0;
        for (ext, count) in items.iter().take(top_n) {
            top_sum += *count;
            if ext == "no-ext" {
                parts.push(format!("{count} *no-ext"));
            } else {
                parts.push(format!("{count} *.{ext}"));
            }
        }
        let ellipsis = if top_sum < self.total_files {
            ", ..."
        } else {
            ""
        };
        let file_word = if self.total_files == 1 {
            "file"
        } else {
            "files"
        };
        format!(
            "[{} {} in subtree: {}{}]",
            self.total_files,
            file_word,
            parts.join(", "),
            ellipsis
        )
    }
}

fn ext_key_from_path(path: &Path) -> String {
    path.extension().map_or("no-ext".to_string(), |s| {
        s.to_string_lossy().to_ascii_lowercase()
    })
}

#[derive(Debug)]
struct DirNode {
    depth: usize,
    files: Vec<String>,
    subdirs: Vec<String>,
    children: HashMap<String, DirNode>,
    subtree: DirAccum,
    is_expanded: bool,
}

impl DirNode {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            files: Vec::new(),
            subdirs: Vec::new(),
            children: HashMap::new(),
            subtree: DirAccum::default(),
            is_expanded: false,
        }
    }

    fn add_item(&mut self, rel_parts: &[&str], is_dir: bool) {
        if rel_parts.is_empty() {
            return;
        }
        if rel_parts.len() == 1 {
            let name = rel_parts[0].to_owned();
            if is_dir {
                let key = format!("{name}/");
                if !self.children.contains_key(&key) {
                    self.children
                        .insert(key.clone(), DirNode::new(self.depth + 1));
                    self.subdirs.push(key);
                }
            } else {
                let ext = ext_key_from_path(Path::new(&name));
                self.files.push(name);
                self.subtree.add_ext(&ext);
            }
            return;
        }
        let subdir = rel_parts[0];
        let key = format!("{subdir}/");
        if !self.children.contains_key(&key) {
            self.children
                .insert(key.clone(), DirNode::new(self.depth + 1));
            self.subdirs.push(key.clone());
        }
        let child = self.children.get_mut(&key).expect("just inserted");
        child.add_item(&rel_parts[1..], is_dir);
        if !is_dir {
            let ext = ext_key_from_path(Path::new(rel_parts.last().unwrap()));
            self.subtree.add_ext(&ext);
        }
    }

    fn sort_recursive(&mut self) {
        self.files.sort_by_key(|a| a.to_ascii_lowercase());
        self.subdirs.sort_by_key(|a| a.to_ascii_lowercase());
        for child in self.children.values_mut() {
            child.sort_recursive();
        }
    }

    fn all_subitems_sorted(&self) -> Vec<&str> {
        let mut items: Vec<&str> = self
            .files
            .iter()
            .map(String::as_str)
            .chain(self.subdirs.iter().map(String::as_str))
            .collect();
        items.sort_by_key(|a| a.to_ascii_lowercase());
        items
    }

    fn subitem_line(&self, name: &str) -> String {
        let indent = "  ".repeat(self.depth + 1);
        format!("{indent}- {name}")
    }

    fn summary_str(&self, top_k: usize) -> String {
        self.subtree.to_summary(top_k)
    }

    fn summary_char_cost(&self, top_k: usize) -> usize {
        let s = self.summary_str(top_k);
        if s.is_empty() {
            return 0;
        }
        (self.depth + 1) * 2 + s.len() + 1
    }

    fn render_expanded(&self, top_k: usize) -> String {
        let mut out = String::new();
        for name in self.all_subitems_sorted() {
            out.push_str(&self.subitem_line(name));
            out.push('\n');
            if let Some(child) = self.children.get(name) {
                out.push_str(&child.render_subtree(top_k));
            }
        }
        out
    }

    fn render_subtree(&self, top_k: usize) -> String {
        if self.is_expanded {
            return self.render_expanded(top_k);
        }
        let summary = self.summary_str(top_k);
        if summary.is_empty() {
            return String::new();
        }
        let indent = "  ".repeat(self.depth + 1);
        format!("{indent}{summary}\n")
    }
}

fn list_dir_walk_builder(root: &Path, respect_gitignore: bool) -> ignore::WalkBuilder {
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .standard_filters(true)
        .git_ignore(respect_gitignore)
        .git_global(respect_gitignore)
        .git_exclude(respect_gitignore);
    builder
}

fn seed_depth1_children(
    root: &Path,
    root_node: &mut DirNode,
    respect_gitignore: bool,
    max_seed: usize,
) -> bool {
    let walker = list_dir_walk_builder(root, respect_gitignore)
        .max_depth(Some(1))
        .build();
    let mut seed_count: usize = 0;
    for entry in walker {
        let Ok(entry) = entry else { continue };
        if entry.depth() != 1 {
            continue;
        }
        let Some(ft) = entry.file_type() else {
            continue;
        };
        let Some(name) = entry.file_name().to_str() else {
            continue;
        };
        seed_count += 1;
        if seed_count > max_seed {
            return true;
        }
        root_node.add_item(&[name], ft.is_dir());
    }
    false
}

fn build_tree_with_limit(
    root: &Path,
    respect_gitignore: bool,
    max_items: usize,
) -> (DirNode, bool) {
    let mut root_node = DirNode::new(0);
    let seed_truncated =
        seed_depth1_children(root, &mut root_node, respect_gitignore, MAX_SEED_ITEMS);
    let walker = list_dir_walk_builder(root, respect_gitignore).build();
    let mut item_count: usize = 0;
    let mut walk_truncated = false;
    for entry in walker {
        let Ok(entry) = entry else { continue };
        let Some(ft) = entry.file_type() else {
            continue;
        };
        if entry.depth() <= 1 {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(root) else {
            continue;
        };
        let parts: Vec<&str> = rel.iter().filter_map(|c| c.to_str()).collect();
        if parts.is_empty() {
            continue;
        }
        item_count += 1;
        if item_count > max_items {
            walk_truncated = true;
            break;
        }
        root_node.add_item(&parts, ft.is_dir());
    }
    root_node.sort_recursive();
    (root_node, seed_truncated || walk_truncated)
}

fn budget_expand(
    root: &mut DirNode,
    max_chars: usize,
    top_k: usize,
    truncated: bool,
    truncation_notice: &str,
) -> String {
    let cutoff_msg = if truncated {
        format!(
            "\nNote: there are more than {MAX_GLOBAL_ITEMS} items in the directory, \
             so not all files may be shown.\n"
        )
    } else {
        String::new()
    };
    if root.files.is_empty() && root.subdirs.is_empty() {
        return cutoff_msg;
    }
    root.is_expanded = true;
    let root_expanded = root.render_expanded(top_k);
    if root_expanded.len() > max_chars {
        let mut out = render_truncated_root(root, max_chars, top_k, truncation_notice);
        out.push_str(&cutoff_msg);
        return out;
    }
    let mut remaining = max_chars - root_expanded.len();
    let mut queue: std::collections::VecDeque<Vec<String>> = std::collections::VecDeque::new();
    for name in &root.subdirs {
        queue.push_back(vec![name.clone()]);
    }
    while let Some(node_path) = queue.pop_front() {
        let Some(node) = navigate_mut(root, &node_path) else {
            continue;
        };
        node.is_expanded = true;
        let expanded = node.render_expanded(top_k);
        let summary_cost = node.summary_char_cost(top_k);
        if expanded.len() > remaining + summary_cost {
            node.is_expanded = false;
            continue;
        }
        remaining += summary_cost;
        remaining -= expanded.len();
        let child_names: Vec<String> = node.subdirs.clone();
        for child_name in child_names {
            let mut child_path = node_path.clone();
            child_path.push(child_name);
            queue.push_back(child_path);
        }
    }
    let mut out = root.render_expanded(top_k);
    out.push_str(&cutoff_msg);
    out
}

fn navigate_mut<'a>(root: &'a mut DirNode, path: &[String]) -> Option<&'a mut DirNode> {
    let mut node = root;
    for key in path {
        node = node.children.get_mut(key)?;
    }
    Some(node)
}

fn render_truncated_root(root: &DirNode, max_chars: usize, top_k: usize, notice: &str) -> String {
    let mut out = String::new();
    let mut remaining = max_chars;
    let child_summary_indent = "  ".repeat(root.depth + 2);
    for name in root.all_subitems_sorted() {
        let mut chunk = root.subitem_line(name);
        chunk.push('\n');
        if let Some(child) = root.children.get(name) {
            let summary = child.summary_str(top_k);
            if !summary.is_empty() {
                chunk.push_str(&format!("{child_summary_indent}{summary}\n"));
            }
        }
        if chunk.len() > remaining {
            break;
        }
        out.push_str(&chunk);
        remaining -= chunk.len();
    }
    out.push_str(notice);
    out
}

/// Public entry for tests / reuse.
pub fn list_directory(root: &Path, max_output_chars: usize) -> String {
    let (mut tree, truncated) = build_tree_with_limit(root, true, MAX_GLOBAL_ITEMS);
    let body = budget_expand(
        &mut tree,
        max_output_chars,
        TOP_K_EXTENSIONS,
        truncated,
        ROOT_TRUNCATION_NOTICE,
    );
    let header = format!("{}/", root.file_name().and_then(|n| n.to_str()).unwrap_or("."));
    if body.is_empty() {
        header
    } else {
        format!("{header}\n{body}")
    }
}

pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".into(),
            description: concat!(
                "List a directory tree (files and folders) with indentation. ",
                "Respects .gitignore. Large trees are truncated with extension ",
                "summaries and a notice."
            )
            .into(),
            parameters: schema_object(
                vec![
                    (
                        "path",
                        s_string(),
                        "Directory path, relative to the workspace root or absolute.",
                    ),
                    (
                        "max_output_chars",
                        s_integer(),
                        "Char budget for the listing (default 10000).",
                    ),
                ],
                &["path"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let path = arg_str(&args, "path")?;
        let budget = arg_opt_usize(&args, "max_output_chars", DEFAULT_MAX_OUTPUT_CHARS);

        let abs = ctx.sandbox.resolve(&path)?;
        if !abs.exists() || !abs.is_dir() {
            return Err(EngineError::Tool {
                name: "list_dir".into(),
                message: format!("not a directory: {}", ctx.sandbox.display(&abs)),
            });
        }

        let (mut tree, truncated) = build_tree_with_limit(&abs, true, MAX_GLOBAL_ITEMS);
        let body = budget_expand(
            &mut tree,
            budget,
            TOP_K_EXTENSIONS,
            truncated,
            ROOT_TRUNCATION_NOTICE,
        );
        let display = ctx.sandbox.display(&abs);
        let mut out = format!("{display}/");
        if !body.is_empty() {
            out.push('\n');
            out.push_str(&body);
        }
        Ok(ToolOutput::new(out))
    }
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(ListDirTool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_accum_summary() {
        let mut a = DirAccum::default();
        a.add_ext("rs");
        a.add_ext("rs");
        a.add_ext("toml");
        let s = a.to_summary(3);
        assert!(s.contains("3 files"));
        assert!(s.contains("*.rs"));
    }

    #[test]
    fn seed_and_expand_small_tree() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "x").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/b.rs"), "y").unwrap();
        let listed = list_directory(dir.path(), 10_000);
        assert!(listed.contains("a.rs") || listed.contains("- a.rs"));
        assert!(listed.contains("sub/") || listed.contains("*.rs"));
    }
}
