//! Sandboxed filesystem access.
//!
//! All file tools resolve paths through [`Sandbox`], which jails every
//! operation to the engine's `root_dir`. On mobile the root is the app's
//! own documents/files directory, so the agent can never touch files
//! outside the app sandbox even if a model hallucinates an absolute path.

use std::path::{Component, Path, PathBuf};

use crate::error::{EngineError, EngineResult};

#[derive(Debug)]
pub struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    /// Create a sandbox rooted at `root` (canonicalized; created if needed).
    pub fn new(root: &Path) -> EngineResult<Self> {
        if !root.exists() {
            std::fs::create_dir_all(root)?;
        }
        let root = root.canonicalize()?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a model-supplied path against the sandbox root.
    ///
    /// Relative paths join onto the root; absolute paths are accepted only
    /// when they stay inside the root. Lexically normalizes `.`/`..` first,
    /// then re-checks the real (symlink-resolved) prefix when the path
    /// exists.
    pub fn resolve(&self, raw: &str) -> EngineResult<PathBuf> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err(EngineError::SandboxEscape("(empty path)".into()));
        }
        let joined = {
            let p = Path::new(raw);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                self.root.join(p)
            }
        };
        let normalized = normalize_lexical(&joined);

        // Lexical containment check.
        if !starts_with_path(&normalized, &self.root) {
            return Err(EngineError::SandboxEscape(raw.to_string()));
        }

        // Real containment check for existing paths (symlink escape).
        if normalized.exists() {
            let real = normalized.canonicalize()?;
            if !starts_with_path(&real, &self.root) {
                return Err(EngineError::SandboxEscape(raw.to_string()));
            }
            return Ok(real);
        }

        // Non-existing path: canonicalize the deepest existing ancestor and
        // re-verify containment, then append the remaining components.
        let mut ancestor = normalized.clone();
        let mut tail: Vec<std::ffi::OsString> = Vec::new();
        loop {
            if ancestor.exists() {
                let real = ancestor.canonicalize()?;
                if !starts_with_path(&real, &self.root) {
                    return Err(EngineError::SandboxEscape(raw.to_string()));
                }
                let mut out = real;
                for part in tail.into_iter().rev() {
                    out.push(part);
                }
                return Ok(out);
            }
            match ancestor.file_name() {
                Some(name) => {
                    tail.push(name.to_os_string());
                    ancestor.pop();
                }
                None => return Err(EngineError::SandboxEscape(raw.to_string())),
            }
        }
    }

    /// Render an absolute path relative to the sandbox root for model-facing
    /// output (mirrors grok's `display_cwd_or_cwd` convention).
    pub fn display(&self, abs: &Path) -> String {
        match abs.strip_prefix(&self.root) {
            Ok(rel) if !rel.as_os_str().is_empty() => rel.display().to_string(),
            _ => abs.display().to_string(),
        }
    }
}

/// Lexically normalize `.` and `..` without touching the filesystem.
pub fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                // Keep leading `..` for absolute-path checks; they fail the
                // containment check anyway, but keep behavior predictable.
                if !matches!(
                    out.components().next_back(),
                    Some(Component::Normal(_))
                ) {
                    if out.has_root() && !out.is_absolute() {
                        out.pop();
                    } else {
                        out.push(comp);
                    }
                } else {
                    out.pop();
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn starts_with_path(path: &Path, prefix: &Path) -> bool {
    path.starts_with(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_removes_dotdot() {
        let p = normalize_lexical(Path::new("/a/b/../c/./d"));
        assert_eq!(p, PathBuf::from("/a/c/d"));
    }

    #[test]
    fn sandbox_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path()).unwrap();
        assert!(sb.resolve("../outside").is_err());
        assert!(sb.resolve("/etc/passwd").is_err());
        let ok = sb.resolve("sub/file.txt").unwrap();
        assert!(ok.starts_with(sb.root()));
    }

    #[test]
    fn sandbox_allows_absolute_inside() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path()).unwrap();
        let inner = sb.root().join("x.txt");
        std::fs::write(&inner, "hi").unwrap();
        let got = sb.resolve(inner.to_str().unwrap()).unwrap();
        assert_eq!(got, inner.canonicalize().unwrap());
    }
}
