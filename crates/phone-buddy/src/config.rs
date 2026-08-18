//! Engine configuration.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::llm::types::ApiBackend;

pub use crate::llm::types::ApiBackend as ConfigApiBackend;

/// How the engine reaches the LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LlmMode {
    /// OpenAI-compatible HTTP `/chat/completions` (default).
    #[default]
    Http,
    /// Host-provided transport via FFI callbacks (local llama.rn, etc.).
    Host,
}

/// Configuration for one engine instance.
///
/// `root_dir` is the file sandbox root (on mobile: the app's Documents or
/// files dir). Every file tool resolves paths against it and refuses to
/// escape it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// API key for the LLM provider (xAI / OpenAI / any compatible endpoint).
    /// Required when [`llm_mode`] is [`LlmMode::Http`]; ignored for Host.
    #[serde(default)]
    pub api_key: String,
    /// Base URL of an OpenAI-compatible API, e.g. `https://api.x.ai/v1`.
    /// Required when [`llm_mode`] is [`LlmMode::Http`]; ignored for Host.
    #[serde(default = "default_base_url")]
    pub base_url: String,
    /// Model id, e.g. `grok-4` or `gpt-4o`.
    pub model: String,
    /// File sandbox root. All file tools are jailed to this directory.
    pub root_dir: PathBuf,
    /// UI locale; affects the language the agent is told to answer in.
    #[serde(default = "default_locale")]
    pub locale: String,
    /// Max tool-loop turns per user message.
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_output_tokens: u32,
    /// Enable xAI live web search (`search_parameters` on the request).
    /// Only effective against xAI endpoints with a search-capable model.
    #[serde(default)]
    pub enable_web_search: bool,
    /// Identity used in the system prompt (`You are {agent_name}…`).
    /// Empty or whitespace falls back to [`DEFAULT_AGENT_NAME`].
    #[serde(default = "default_agent_name")]
    pub agent_name: String,
    /// Extra text appended to the system prompt (product persona etc).
    #[serde(default)]
    pub system_prompt_extra: Option<String>,
    /// Idle timeout for the LLM stream, in seconds (default: 120s).
    #[serde(default = "default_idle_timeout_secs")]
    pub stream_idle_timeout_secs: u64,
    /// Max API retries per LLM request (default: 5 attempts).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// LLM transport mode. Default: HTTP.
    #[serde(default)]
    pub llm_mode: LlmMode,
    /// API backend protocol (ChatCompletions, Responses, Messages). Default: ChatCompletions.
    #[serde(default)]
    pub api_backend: ApiBackend,
    /// Extra HTTP headers sent with every LLM request (e.g. X-App-Version, X-Client-Platform).
    #[serde(default)]
    pub extra_headers: std::collections::HashMap<String, String>,
    /// Extra custom fields merged into the LLM request JSON payload (e.g. packageName, pid, clientId, etc.).
    #[serde(default)]
    pub extra_body: std::collections::HashMap<String, serde_json::Value>,
    /// Opt into server-side doom-loop recovery on the Responses API
    /// (`x-grok-doom-loop-check`). Default: on when `api_backend == Responses`.
    #[serde(default)]
    pub enable_doom_loop_check: Option<bool>,
    /// Allow web_fetch to hit explicit loopback hosts only. Default false.
    #[serde(default)]
    pub web_fetch_allow_local: bool,
}

/// Default system-prompt identity when the host does not set [`EngineConfig::agent_name`].
pub const DEFAULT_AGENT_NAME: &str = "PhoneBuddy";
const AGENT_NAME_MAX_CHARS: usize = 80;

fn default_agent_name() -> String {
    DEFAULT_AGENT_NAME.into()
}

fn default_locale() -> String {
    "zh".into()
}
fn default_base_url() -> String {
    "https://api.x.ai/v1".into()
}
fn default_max_turns() -> u32 {
    24
}
fn default_temperature() -> f32 {
    0.2
}
fn default_max_tokens() -> u32 {
    8192
}
fn default_idle_timeout_secs() -> u64 {
    120
}
fn default_max_retries() -> u32 {
    5
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: default_base_url(),
            model: "grok-4".into(),
            root_dir: std::env::temp_dir().join("phone-buddy"),
            locale: default_locale(),
            max_turns: default_max_turns(),
            temperature: default_temperature(),
            max_output_tokens: default_max_tokens(),
            enable_web_search: false,
            agent_name: default_agent_name(),
            system_prompt_extra: None,
            stream_idle_timeout_secs: default_idle_timeout_secs(),
            max_retries: default_max_retries(),
            llm_mode: LlmMode::Http,
            api_backend: ApiBackend::ChatCompletions,
            extra_headers: std::collections::HashMap::new(),
            extra_body: std::collections::HashMap::new(),
            enable_doom_loop_check: None,
            web_fetch_allow_local: false,
        }
    }
}

/// Sanitize a host-supplied identity for the system prompt.
///
/// Takes the first line, trims it, caps length, and falls back to
/// [`DEFAULT_AGENT_NAME`] when empty.
pub fn resolve_agent_name(raw: &str) -> String {
    let first = raw.lines().next().unwrap_or("").trim();
    if first.is_empty() {
        return DEFAULT_AGENT_NAME.to_string();
    }
    first.chars().take(AGENT_NAME_MAX_CHARS).collect()
}

impl EngineConfig {
    /// Identity interpolated into the system prompt.
    pub fn resolved_agent_name(&self) -> String {
        resolve_agent_name(&self.agent_name)
    }

    /// Whether to send the doom-loop opt-in header and act on server signals.
    pub fn doom_loop_check_enabled(&self) -> bool {
        self.enable_doom_loop_check.unwrap_or(matches!(
            self.api_backend,
            ApiBackend::Responses
        ))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.model.trim().is_empty() {
            return Err("model is empty".into());
        }
        if self.root_dir.as_os_str().is_empty() {
            return Err("root_dir is empty".into());
        }
        match self.llm_mode {
            LlmMode::Http => {
                if self.api_key.trim().is_empty() {
                    return Err("api_key is empty".into());
                }
                if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://")
                {
                    return Err(format!(
                        "base_url must be an http(s) URL: {}",
                        self.base_url
                    ));
                }
            }
            LlmMode::Host => {
                // Host mode uses FFI callbacks; no HTTP credentials required.
            }
        }
        Ok(())
    }

    /// Canonicalized sandbox root (created if missing).
    pub fn resolved_root(&self) -> std::io::Result<PathBuf> {
        let root = &self.root_dir;
        if !root.exists() {
            std::fs::create_dir_all(root)?;
        }
        root.canonicalize()
    }

    /// Directory where sessions are persisted: `<root>/.phonebuddy/sessions`.
    pub fn sessions_dir(&self) -> PathBuf {
        self.root_dir.join(".phonebuddy").join("sessions")
    }

    pub fn root_as_path(&self) -> &Path {
        &self.root_dir
    }
}
