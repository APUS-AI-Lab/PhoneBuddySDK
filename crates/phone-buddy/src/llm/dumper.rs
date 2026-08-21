//! Raw HTTP request and response dumper for debugging and diagnostics.
//!
//! Captures complete HTTP exchanges (request URL, headers, JSON body,
//! response status, response headers, response body) to structured JSON
//! files on disk. Useful for diagnosing upstream proxy / gateway failures,
//! authentication errors, and unexpected HTTP issues in mobile sandboxes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Mode determining when HTTP exchanges are dumped to disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HttpDumpMode {
    /// Dumping disabled (default).
    #[default]
    Off,
    /// Dump only when HTTP request fails (non-2xx status or network/timeout error).
    OnError,
    /// Dump every HTTP exchange (both successful and failed).
    All,
}

/// Configuration for HTTP traffic dumping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpDumpConfig {
    /// Dump mode. Default: `Off`.
    #[serde(default)]
    pub mode: HttpDumpMode,

    /// Directory where dump files are stored.
    /// If `None`, defaults to `<root_dir>/.phonebuddy/http_dumps`.
    #[serde(default)]
    pub dump_dir: Option<PathBuf>,

    /// Whether to mask sensitive headers (Authorization, x-api-key, cookie, etc.).
    /// Default: `true`.
    #[serde(default = "default_mask_sensitive")]
    pub mask_sensitive: bool,

    /// Maximum number of dump files to retain before rotating out oldest files.
    /// Default: 30.
    #[serde(default = "default_max_dump_files")]
    pub max_files: usize,
}

fn default_mask_sensitive() -> bool {
    true
}

fn default_max_dump_files() -> usize {
    30
}

impl Default for HttpDumpConfig {
    fn default() -> Self {
        Self {
            mode: HttpDumpMode::Off,
            dump_dir: None,
            mask_sensitive: default_mask_sensitive(),
            max_files: default_max_dump_files(),
        }
    }
}

/// Structured record of a dumped HTTP request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequestDump {
    pub method: String,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: serde_json::Value,
}

/// Structured record of a dumped HTTP response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponseDump {
    pub status: u16,
    pub status_text: String,
    pub headers: BTreeMap<String, String>,
    pub body_text: String,
}

/// Full exchange dump document saved to disk as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpDumpRecord {
    pub schema_version: String,
    pub request_id: String,
    pub timestamp: String,
    pub duration_ms: u64,
    pub request: HttpRequestDump,
    pub response: Option<HttpResponseDump>,
    pub error: Option<String>,
}

/// HTTP Traffic Dumper engine component.
#[derive(Debug, Clone)]
pub struct HttpDumper {
    config: HttpDumpConfig,
    resolved_dir: PathBuf,
}

impl HttpDumper {
    pub fn new(config: HttpDumpConfig, default_dir: PathBuf) -> Self {
        let resolved_dir = config.dump_dir.clone().unwrap_or(default_dir);
        Self {
            config,
            resolved_dir,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.mode != HttpDumpMode::Off
    }

    pub fn should_dump_error(&self) -> bool {
        matches!(self.config.mode, HttpDumpMode::OnError | HttpDumpMode::All)
    }

    pub fn should_dump_success(&self) -> bool {
        matches!(self.config.mode, HttpDumpMode::All)
    }

    pub fn dump_dir(&self) -> &Path {
        &self.resolved_dir
    }

    /// Check if a header name is considered sensitive and needs masking.
    pub fn is_sensitive_header(name: &str) -> bool {
        let lower = name.to_ascii_lowercase();
        lower == "authorization"
            || lower == "x-api-key"
            || lower == "api-key"
            || lower == "cookie"
            || lower == "set-cookie"
            || lower == "proxy-authorization"
            || lower == "x-auth-token"
    }

    /// Mask sensitive values (e.g. `Bearer sk-1234567890abcdef` -> `Bearer sk-1***cdef`).
    pub fn mask_header_value(&self, key: &str, val: &str) -> String {
        if !self.config.mask_sensitive || !Self::is_sensitive_header(key) {
            return val.to_string();
        }

        let trimmed = val.trim();
        if let Some(token) = trimmed.strip_prefix("Bearer ") {
            format!("Bearer {}", mask_token(token))
        } else if let Some(token) = trimmed.strip_prefix("bearer ") {
            format!("Bearer {}", mask_token(token))
        } else {
            mask_token(trimmed)
        }
    }

    /// Format and mask headers from a reqwest HeaderMap.
    pub fn extract_headers(&self, headers: &reqwest::header::HeaderMap) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        for (name, val) in headers {
            let key = name.as_str().to_string();
            let val_str = val.to_str().unwrap_or("<binary/non-utf8>");
            let masked = self.mask_header_value(&key, val_str);
            map.insert(key, masked);
        }
        map
    }

    /// Dump an HTTP exchange record to a JSON file in the dump directory.
    ///
    /// Automatically manages file retention (FIFO rotation) so the total number
    /// of files does not exceed `config.max_files`.
    pub fn dump(&self, record: &HttpDumpRecord) -> Option<PathBuf> {
        if !self.is_enabled() {
            return None;
        }

        if let Err(e) = std::fs::create_dir_all(&self.resolved_dir) {
            tracing::warn!("failed to create HTTP dump directory {:?}: {e}", self.resolved_dir);
            return None;
        }

        // Clean up oldest files if capacity exceeded
        self.rotate_old_files();

        // Generate file name: dump_{timestamp}_{status}_{request_id}.json
        let ts_clean = chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f").to_string();
        let status_str = record
            .response
            .as_ref()
            .map(|r| r.status.to_string())
            .unwrap_or_else(|| "err".to_string());
        let safe_req_id = sanitize_filename_part(&record.request_id);

        let file_name = format!("dump_{}_{}_{}.json", ts_clean, status_str, safe_req_id);
        let target_path = self.resolved_dir.join(file_name);

        let json_str = match serde_json::to_string_pretty(record) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("failed to serialize HTTP dump record: {e}");
                return None;
            }
        };

        if let Err(e) = std::fs::write(&target_path, json_str) {
            tracing::warn!("failed to write HTTP dump file {:?}: {e}", target_path);
            return None;
        }

        Some(target_path)
    }

    /// Enforce `max_files` FIFO rotation in `resolved_dir`.
    fn rotate_old_files(&self) {
        if self.config.max_files == 0 {
            return;
        }

        let Ok(entries) = std::fs::read_dir(&self.resolved_dir) else {
            return;
        };

        let mut dump_files: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with("dump_") && name.ends_with(".json") {
                    let modified = entry
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    dump_files.push((path, modified));
                }
            }
        }

        if dump_files.len() >= self.config.max_files {
            // Sort by modified time ascending (oldest first)
            dump_files.sort_by_key(|(_, time)| *time);
            let to_remove = dump_files.len() - self.config.max_files + 1;
            for (path, _) in dump_files.into_iter().take(to_remove) {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

fn mask_token(token: &str) -> String {
    if token.len() <= 8 {
        "***[MASKED]***".to_string()
    } else {
        let prefix = &token[..4.min(token.len())];
        let suffix = &token[token.len().saturating_sub(4)..];
        format!("{}***{}", prefix, suffix)
    }
}

fn sanitize_filename_part(raw: &str) -> String {
    raw.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_mask_token_and_sensitive_headers() {
        let dumper = HttpDumper::new(
            HttpDumpConfig {
                mode: HttpDumpMode::OnError,
                dump_dir: None,
                mask_sensitive: true,
                max_files: 10,
            },
            PathBuf::from("/tmp/dummy"),
        );

        assert_eq!(
            dumper.mask_header_value("Authorization", "Bearer sk-proj-1234567890abcdef"),
            "Bearer sk-p***cdef"
        );
        assert_eq!(
            dumper.mask_header_value("x-api-key", "secret123456"),
            "secr***3456"
        );
        assert_eq!(
            dumper.mask_header_value("X-Custom-Header", "unmasked-value"),
            "unmasked-value"
        );
    }

    #[test]
    fn test_http_dump_file_creation_and_rotation() {
        let tmp = tempfile::tempdir().unwrap();
        let dump_dir = tmp.path().join("http_dumps");

        let dumper = HttpDumper::new(
            HttpDumpConfig {
                mode: HttpDumpMode::OnError,
                dump_dir: Some(dump_dir.clone()),
                mask_sensitive: true,
                max_files: 3,
            },
            dump_dir.clone(),
        );

        assert!(dumper.is_enabled());
        assert!(dumper.should_dump_error());
        assert!(!dumper.should_dump_success());

        // Create 4 dump records to test FIFO rotation with limit of 3
        for i in 1..=4 {
            let record = HttpDumpRecord {
                schema_version: "1.0".into(),
                request_id: format!("req_{i}"),
                timestamp: "2026-08-21T12:00:00Z".into(),
                duration_ms: 100,
                request: HttpRequestDump {
                    method: "POST".into(),
                    url: "https://api.example.com/v1/chat/completions".into(),
                    headers: BTreeMap::new(),
                    body: serde_json::json!({"model": "test"}),
                },
                response: Some(HttpResponseDump {
                    status: 502,
                    status_text: "Bad Gateway".into(),
                    headers: BTreeMap::new(),
                    body_text: format!("Error {i}"),
                }),
                error: Some(format!("status=502 Error {i}")),
            };

            let path = dumper.dump(&record).expect("dump should succeed");
            assert!(path.exists());
            // Small sleep to ensure different modified timestamps on fast file systems
            std::thread::sleep(Duration::from_millis(15));
        }

        // Check that at most 3 files remain in the directory
        let files: Vec<_> = std::fs::read_dir(&dump_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();
        assert_eq!(files.len(), 3);
    }
}
