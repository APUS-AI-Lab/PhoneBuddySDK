//! Client profile presets for 1:1 emulation of official coding agents.
//!
//! Provides out-of-the-box profiles that accurately mirror the HTTP headers,
//! user agents, beta flags, and request formatting of:
//! - **xAI Grok Build** (`grok-cli`)
//! - **OpenAI Codex** (`codex-cli`)
//! - **Anthropic Claude Code** (`claude-cli`)

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use crate::llm::types::ApiBackend;

/// Client emulation profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ClientProfile {
    /// Standard PhoneBuddy client identity (default).
    #[default]
    Default,
    /// 1:1 emulation of xAI Grok Build (`grok-cli`).
    GrokBuild,
    /// 1:1 emulation of OpenAI Codex CLI (`codex-cli`).
    Codex,
    /// 1:1 emulation of Anthropic Claude Code (`claude-cli`).
    ClaudeCode,
}

impl ClientProfile {
    /// Wire name used as the default provider-group id when the host does
    /// not set an explicit `provider_group`.
    pub fn group_name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::GrokBuild => "grok_build",
            Self::Codex => "codex",
            Self::ClaudeCode => "claude_code",
        }
    }
}

impl std::str::FromStr for ClientProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "default" | "generic" | "phone_buddy" | "phonebuddy" => Ok(Self::Default),
            "grok" | "grok_build" | "grokbuild" => Ok(Self::GrokBuild),
            "codex" | "openai_codex" => Ok(Self::Codex),
            "claude" | "claude_code" | "claudecode" => Ok(Self::ClaudeCode),
            other => Err(format!(
                "unknown client profile '{other}'. Supported: default, grok_build, codex, claude_code"
            )),
        }
    }
}

/// Metadata definition for a client profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientProfileDefinition {
    /// Profile identifier.
    pub profile: ClientProfile,
    /// Default model endpoint URL.
    pub default_base_url: String,
    /// Default API backend protocol.
    pub default_backend: ApiBackend,
    /// Default version string if not overridden.
    pub default_version: String,
    /// Anthropic API protocol version string (`anthropic-version`).
    pub anthropic_version: Option<String>,
    /// Anthropic beta headers (`anthropic-beta`).
    pub anthropic_betas: Vec<String>,
    /// Default static HTTP headers.
    pub default_headers: BTreeMap<String, String>,
}

/// Current OS identifier formatted for CLI User-Agent strings.
pub fn platform_os() -> &'static str {
    match std::env::consts::OS {
        "macos" => "macos",
        "windows" => "windows",
        "linux" => "linux",
        "ios" => "ios",
        "android" => "android",
        other => other,
    }
}

/// Current CPU architecture formatted for CLI User-Agent strings.
pub fn platform_arch() -> &'static str {
    match std::env::consts::ARCH {
        "aarch64" | "arm64" => "aarch64",
        "x86_64" => "x86_64",
        "x86" => "x86",
        "arm" => "arm",
        other => other,
    }
}

/// Get the static metadata definition for a given client profile.
pub fn get_profile_definition(profile: ClientProfile) -> ClientProfileDefinition {
    match profile {
        ClientProfile::Default => ClientProfileDefinition {
            profile,
            default_base_url: "https://api.x.ai/v1".to_string(),
            default_backend: ApiBackend::ChatCompletions,
            default_version: crate::VERSION.to_string(),
            anthropic_version: Some("2023-06-01".to_string()),
            anthropic_betas: Vec::new(),
            default_headers: BTreeMap::new(),
        },
        ClientProfile::GrokBuild => {
            let mut headers = BTreeMap::new();
            headers.insert("x-grok-client-identifier".to_string(), "grok-cli".to_string());
            ClientProfileDefinition {
                profile,
                default_base_url: "https://api.x.ai/v1".to_string(),
                default_backend: ApiBackend::Responses,
                default_version: "0.1.0".to_string(),
                anthropic_version: None,
                anthropic_betas: Vec::new(),
                default_headers: headers,
            }
        }
        ClientProfile::Codex => {
            let mut headers = BTreeMap::new();
            headers.insert("originator".to_string(), "codex_cli_rs".to_string());
            headers.insert("openai-beta".to_string(), "responses=true".to_string());
            ClientProfileDefinition {
                profile,
                default_base_url: "https://api.openai.com/v1".to_string(),
                default_backend: ApiBackend::Responses,
                default_version: "0.1.0".to_string(),
                anthropic_version: None,
                anthropic_betas: Vec::new(),
                default_headers: headers,
            }
        }
        ClientProfile::ClaudeCode => {
            let mut headers = BTreeMap::new();
            headers.insert("x-app".to_string(), "cli".to_string());
            headers.insert(
                "anthropic-version".to_string(),
                "2023-06-01".to_string(),
            );
            headers.insert(
                "anthropic-beta".to_string(),
                "ccr-byoc-2025-07-29,prompt-caching-2024-07-31,effort-2025-11-24,claude-code-20250219".to_string(),
            );
            ClientProfileDefinition {
                profile,
                default_base_url: "https://api.anthropic.com/v1".to_string(),
                default_backend: ApiBackend::Messages,
                default_version: "2.1.238".to_string(),
                anthropic_version: Some("2023-06-01".to_string()),
                anthropic_betas: vec![
                    "ccr-byoc-2025-07-29".to_string(),
                    "prompt-caching-2024-07-31".to_string(),
                    "effort-2025-11-24".to_string(),
                    "claude-code-20250219".to_string(),
                ],
                default_headers: headers,
            }
        }
    }
}

/// Render the 1:1 official User-Agent string for the selected profile.
pub fn render_user_agent(profile: ClientProfile, custom_version: Option<&str>) -> String {
    let def = get_profile_definition(profile);
    let version = custom_version.unwrap_or(&def.default_version);
    let os = platform_os();
    let arch = platform_arch();

    match profile {
        ClientProfile::Default => {
            format!("PhoneBuddy/{version} (Mobile SDK; LLM Client; {os}; {arch})")
        }
        ClientProfile::GrokBuild => {
            // Mirrors upstream grok-build client.rs: user_agent_string_for
            // e.g. "grok-cli/0.1.0 (macos; aarch64)"
            format!("grok-cli/{version} ({os}; {arch})")
        }
        ClientProfile::Codex => {
            // Mirrors upstream codex-rs login/src/auth/default_client.rs: get_codex_user_agent
            // e.g. "codex_cli_rs/0.1.0 (macos; aarch64) codex_cli"
            format!("codex_cli_rs/{version} ({os}; {arch}) codex_cli")
        }
        ClientProfile::ClaudeCode => {
            // Mirrors upstream cc-src utils/http.ts: getUserAgent
            // e.g. "claude-cli/2.1.238 (external, cli)"
            format!("claude-cli/{version} (external, cli)")
        }
    }
}

/// Generate default headers for the given profile and context.
pub fn build_profile_headers(
    profile: ClientProfile,
    api_key: &str,
    session_id: Option<&str>,
    custom_version: Option<&str>,
    doom_loop_enabled: bool,
) -> BTreeMap<String, String> {
    let def = get_profile_definition(profile);
    let mut headers = def.default_headers.clone();

    headers.insert("accept".to_string(), "text/event-stream".to_string());
    headers.insert(
        "user-agent".to_string(),
        render_user_agent(profile, custom_version),
    );

    let resolved_session_id = session_id
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    match profile {
        ClientProfile::Default => {
            if !api_key.is_empty() {
                headers.insert("authorization".to_string(), format!("Bearer {api_key}"));
            }
        }
        ClientProfile::GrokBuild => {
            if !api_key.is_empty() {
                headers.insert("authorization".to_string(), format!("Bearer {api_key}"));
            }
            if doom_loop_enabled {
                headers.insert(
                    crate::llm::doom_loop_wire::DOOM_LOOP_CHECK_HEADER.to_string(),
                    "1".to_string(),
                );
            }
        }
        ClientProfile::Codex => {
            if !api_key.is_empty() {
                headers.insert("authorization".to_string(), format!("Bearer {api_key}"));
            }
            headers.insert("originator".to_string(), "codex_cli_rs".to_string());
            headers.insert("session-id".to_string(), resolved_session_id.clone());
            headers.insert("thread-id".to_string(), resolved_session_id.clone());
            headers.insert("x-client-request-id".to_string(), resolved_session_id.clone());
            headers.insert("x-codex-installation-id".to_string(), resolved_session_id.clone());
            headers.insert("x-codex-window-id".to_string(), resolved_session_id);
        }
        ClientProfile::ClaudeCode => {
            if !api_key.is_empty() {
                headers.insert("x-api-key".to_string(), api_key.to_string());
                headers.insert("authorization".to_string(), format!("Bearer {api_key}"));
            }
            headers.insert(
                "x-claude-code-session-id".to_string(),
                resolved_session_id,
            );
        }
    }

    headers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_user_agents() {
        let ua_grok = render_user_agent(ClientProfile::GrokBuild, None);
        assert!(ua_grok.starts_with("grok-cli/"));
        assert!(ua_grok.contains('('));

        let ua_codex = render_user_agent(ClientProfile::Codex, Some("1.2.3"));
        assert_eq!(
            ua_codex,
            format!("codex_cli_rs/1.2.3 ({}; {}) codex_cli", platform_os(), platform_arch())
        );

        let ua_claude = render_user_agent(ClientProfile::ClaudeCode, None);
        assert!(ua_claude.starts_with("claude-cli/2.1.238"));
    }

    #[test]
    fn test_profile_headers() {
        let h_claude = build_profile_headers(
            ClientProfile::ClaudeCode,
            "test_key",
            Some("sess-123"),
            None,
            false,
        );
        assert_eq!(h_claude.get("anthropic-version").unwrap(), "2023-06-01");
        assert_eq!(h_claude.get("x-app").unwrap(), "cli");
        assert_eq!(h_claude.get("x-claude-code-session-id").unwrap(), "sess-123");
        assert_eq!(h_claude.get("x-api-key").unwrap(), "test_key");
        assert!(h_claude.get("anthropic-beta").unwrap().contains("effort-2025-11-24"));

        let h_grok = build_profile_headers(
            ClientProfile::GrokBuild,
            "test_key",
            None,
            None,
            true,
        );
        assert_eq!(h_grok.get("x-grok-client-identifier").unwrap(), "grok-cli");
        assert_eq!(h_grok.get("x-grok-doom-loop-check").unwrap(), "1");
        assert_eq!(h_grok.get("authorization").unwrap(), "Bearer test_key");

        let h_codex = build_profile_headers(
            ClientProfile::Codex,
            "test_key",
            Some("sess-456"),
            None,
            false,
        );
        assert_eq!(h_codex.get("originator").unwrap(), "codex_cli_rs");
        assert_eq!(h_codex.get("session-id").unwrap(), "sess-456");
        assert_eq!(h_codex.get("thread-id").unwrap(), "sess-456");
        assert_eq!(h_codex.get("x-client-request-id").unwrap(), "sess-456");
        assert_eq!(h_codex.get("x-codex-installation-id").unwrap(), "sess-456");
        assert_eq!(h_codex.get("x-codex-window-id").unwrap(), "sess-456");
        assert_eq!(h_codex.get("openai-beta").unwrap(), "responses=true");
        assert_eq!(h_codex.get("authorization").unwrap(), "Bearer test_key");
    }
}
