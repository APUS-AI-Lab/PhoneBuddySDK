//! Engine configuration.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use crate::llm::dumper::{HttpDumpConfig, HttpDumpMode};
pub use crate::llm::profiles::{
    build_profile_headers, get_profile_definition, render_user_agent, ClientProfile,
    ClientProfileDefinition,
};
pub use crate::llm::types::{
    ApiBackend, ReasoningEffort, SearchDateBound, WebSearchOptions, XSearchOptions,
};

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
    /// API key for the LLM provider (xAI / OpenAI / Anthropic / any compatible endpoint).
    /// Required when [`llm_mode`] is [`LlmMode::Http`]; ignored for Host.
    #[serde(default)]
    pub api_key: String,
    /// Base URL of an API endpoint, e.g. `https://api.x.ai/v1`, `https://api.anthropic.com/v1`.
    /// Required when [`llm_mode`] is [`LlmMode::Http`]; ignored for Host.
    #[serde(default = "default_base_url")]
    pub base_url: String,
    /// Model id, e.g. `grok-4.6`, `claude-opus-5`, `gpt-4o`.
    pub model: String,

    /// Client profile for 1:1 emulation of official clients (Default, GrokBuild, Codex, ClaudeCode).
    #[serde(default)]
    pub client_profile: ClientProfile,
    /// Optional custom version string for User-Agent (defaults to official client version).
    #[serde(default)]
    pub client_version: Option<String>,
    /// Optional custom session identifier for vendor session tracking headers.
    #[serde(default)]
    pub client_session_id: Option<String>,
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
    /// Reasoning effort level for thinking models (e.g. Low, Medium, High).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// This model/gateway exposes hosted `{ "type": "web_search" }` on
    /// Responses. Matches grok-build: the agent conversation inlines that
    /// tool so server-side search runs inside the same SSE. Gateways
    /// without hosted search (PackyAPI, chat-completions models) must
    /// leave this off. Truncated `in_progress` hosted searches are
    /// salvaged as client `web_search` (DuckDuckGo, then a separate
    /// hosted request).
    #[serde(default)]
    pub enable_web_search: bool,
    /// Optional configuration options for WebSearch (allowed/excluded domains).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    /// Attach Grok-style backend-hosted XSearch (`{ "type": "x_search" }`) on the Responses API.
    ///
    /// Because not all models and gateways support XSearch, this is controlled via an explicit
    /// toggle. When true and [`api_backend`] is [`ApiBackend::Responses`], `{ "type": "x_search" }`
    /// is added to the request `tools` array.
    #[serde(default)]
    pub enable_x_search: bool,
    /// Optional configuration options for XSearch (date bounds etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_search_options: Option<XSearchOptions>,
    /// Identity used in the system prompt (`You are {agent_name}…`).
    /// Empty or whitespace falls back to [`DEFAULT_AGENT_NAME`].
    #[serde(default = "default_agent_name")]
    pub agent_name: String,
    /// Extra text appended to the system prompt (product persona etc).
    #[serde(default)]
    pub system_prompt_extra: Option<String>,
    /// Idle timeout for the LLM stream, in seconds (default: 300s, matching grok-build).
    #[serde(default = "default_idle_timeout_secs")]
    pub stream_idle_timeout_secs: u64,
    /// Max API retries per LLM request (default: 5 attempts).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// LLM transport mode. Default: HTTP.
    #[serde(default)]
    pub llm_mode: LlmMode,
    /// API backend protocol (ChatCompletions, Responses, Messages, Gemini). Default: ChatCompletions.
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
    /// Configuration for dumping raw HTTP requests and responses to disk for debugging.
    #[serde(default)]
    pub http_dump: HttpDumpConfig,
    /// Ordered fallback endpoints tried after the primary provider degrades.
    /// Empty (the default) keeps single-provider behaviour: the full
    /// [`max_retries`] budget is spent on the primary endpoint.
    ///
    /// The legacy adapter copies this chain into both `main` and `subagent`
    /// pools. A direct [`crate::llm::router::LlmRoutingConfig`] must declare
    /// `subagent` itself; it is not inherited from `main`.
    #[serde(default)]
    pub fallback_providers: Vec<ProviderEndpoint>,
    /// Chain mode only: total attempts per provider per LLM request before
    /// failing over (default 3 = initial + 2 retries).
    #[serde(default = "default_failover_max_attempts")]
    pub failover_max_attempts: u32,
    /// Cooldown (secs) a degraded provider sits out before re-probing
    /// (default 120; doubles on consecutive trips, capped at 600).
    #[serde(default = "default_provider_cooldown_secs")]
    pub provider_cooldown_secs: u64,
    /// Compatibility group for encrypted reasoning / reasoning-item ids.
    /// Empty falls back to the client-profile name (`grok_build`, `codex`,
    /// `claude_code`, `default`). Same group + same model keeps those
    /// artifacts when failing over to another host.
    #[serde(default)]
    pub provider_group: Option<String>,
    /// App-private directory that image attachments must live under.
    /// Empty falls back to `<root_dir>/image_attachments`.
    #[serde(default)]
    pub attachment_root: Option<PathBuf>,
    /// When false, a user turn with images fails with [`crate::error::EngineError::VisionUnsupported`].
    #[serde(default = "default_supports_image_input")]
    pub supports_image_input: bool,
    /// When false, a user turn with audio attachments fails with [`crate::error::EngineError::AudioUnsupported`].
    #[serde(default = "default_supports_audio_input")]
    pub supports_audio_input: bool,
}

fn default_supports_image_input() -> bool {
    true
}

fn default_supports_audio_input() -> bool {
    true
}

/// One backup LLM HTTP endpoint. Fields mirror the primary EngineConfig
/// HTTP identity so a fallback can use a different vendor, model, and
/// client profile.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderEndpoint {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub api_backend: ApiBackend,
    #[serde(default)]
    pub client_profile: ClientProfile,
    #[serde(default)]
    pub client_version: Option<String>,
    #[serde(default)]
    pub client_session_id: Option<String>,
    #[serde(default)]
    pub extra_headers: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub extra_body: std::collections::HashMap<String, serde_json::Value>,
    /// Hosted `{type: web_search}` on Responses, matching grok-build
    /// backend search. Also used as the function-tool fallback after
    /// DuckDuckGo when a truncated hosted search is salvaged.
    #[serde(default)]
    pub enable_web_search: bool,
    /// Optional configuration options for WebSearch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    /// Hosted `{type: x_search}` on Responses.
    #[serde(default)]
    pub enable_x_search: bool,
    /// Optional configuration options for XSearch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_search_options: Option<XSearchOptions>,
    /// Reasoning effort level for thinking models on this fallback provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Compatibility group; empty inherits this endpoint's `client_profile`.
    #[serde(default)]
    pub provider_group: Option<String>,
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
    300
}
fn default_max_retries() -> u32 {
    5
}
fn default_failover_max_attempts() -> u32 {
    crate::llm::failover::DEFAULT_FAILOVER_MAX_ATTEMPTS
}
fn default_provider_cooldown_secs() -> u64 {
    crate::llm::failover::DEFAULT_PROVIDER_COOLDOWN_SECS
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: default_base_url(),
            model: "grok-4.6".into(),
            client_profile: ClientProfile::Default,
            client_version: None,
            client_session_id: None,

            root_dir: std::env::temp_dir().join("phone-buddy"),
            locale: default_locale(),
            max_turns: default_max_turns(),
            temperature: default_temperature(),
            max_output_tokens: default_max_tokens(),
            reasoning_effort: None,
            enable_web_search: false,
            web_search_options: None,
            enable_x_search: false,
            x_search_options: None,
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
            http_dump: HttpDumpConfig::default(),
            fallback_providers: Vec::new(),
            failover_max_attempts: default_failover_max_attempts(),
            provider_cooldown_secs: default_provider_cooldown_secs(),
            provider_group: None,
            attachment_root: None,
            supports_image_input: true,
            supports_audio_input: true,
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
    /// Create a fluent builder for [`EngineConfig`].
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::new()
    }

    /// Create a builder pre-configured for 1:1 xAI Grok Build (`grok-cli`) emulation.
    pub fn for_grok(api_key: impl Into<String>, model: impl Into<String>) -> EngineConfigBuilder {
        EngineConfigBuilder::new()
            .client_profile(ClientProfile::GrokBuild)
            .url("https://api.x.ai/v1")
            .api_key(api_key)
            .model(model)
    }

    /// Create a builder pre-configured for 1:1 Anthropic Claude Code (`claude-cli`) emulation.
    pub fn for_claude_code(
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> EngineConfigBuilder {
        EngineConfigBuilder::new()
            .client_profile(ClientProfile::ClaudeCode)
            .url("https://api.anthropic.com/v1")
            .api_key(api_key)
            .model(model)
    }

    /// Create a builder pre-configured for 1:1 OpenAI Codex CLI (`codex-cli`) emulation.
    pub fn for_codex(api_key: impl Into<String>, model: impl Into<String>) -> EngineConfigBuilder {
        EngineConfigBuilder::new()
            .client_profile(ClientProfile::Codex)
            .url("https://api.openai.com/v1")
            .api_key(api_key)
            .model(model)
    }

    /// Builder pre-configured for the Gemini generateContent protocol.
    pub fn for_gemini(api_key: impl Into<String>, model: impl Into<String>) -> EngineConfigBuilder {
        EngineConfigBuilder::new()
            .url("https://generativelanguage.googleapis.com/v1beta")
            .api_backend(ApiBackend::Gemini)
            .api_key(api_key)
            .model(model)
    }

    /// True when hosted `{type: web_search}` should ride the Responses request.
    pub fn backend_search_active(&self) -> bool {
        self.enable_web_search && matches!(self.api_backend, ApiBackend::Responses)
    }

    /// True when hosted `{type: x_search}` should ride the Responses request.
    pub fn backend_x_search_active(&self) -> bool {
        self.enable_x_search && matches!(self.api_backend, ApiBackend::Responses)
    }

    /// Identity interpolated into the system prompt.
    pub fn resolved_agent_name(&self) -> String {
        resolve_agent_name(&self.agent_name)
    }

    /// Whether to send the doom-loop opt-in header and act on server signals.
    pub fn doom_loop_check_enabled(&self) -> bool {
        self.enable_doom_loop_check
            .unwrap_or(matches!(self.api_backend, ApiBackend::Responses))
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_pool_bound()?;
        if matches!(self.llm_mode, LlmMode::Http) {
            if self.api_key.trim().is_empty() {
                return Err("api_key is empty".into());
            }
            if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
                return Err(format!(
                    "base_url must be an http(s) URL: {}",
                    self.base_url
                ));
            }
        }
        for (i, ep) in self.fallback_providers.iter().enumerate() {
            if ep.model.trim().is_empty() {
                return Err(format!("fallback_providers[{i}].model is empty"));
            }
            if ep.api_key.trim().is_empty() {
                return Err(format!("fallback_providers[{i}].api_key is empty"));
            }
            if !ep.base_url.starts_with("http://") && !ep.base_url.starts_with("https://") {
                return Err(format!(
                    "fallback_providers[{i}].base_url must be an http(s) URL: {}",
                    ep.base_url
                ));
            }
        }
        Ok(())
    }

    /// Validation for an engine bound to a [`crate::llm::router`] pool.
    ///
    /// The pool's [`crate::llm::router::ProviderTarget`]s own `base_url`,
    /// `api_key`, and the per-provider model, so those engine fields are inert
    /// and may be left empty. Hosts that pass a declarative routing config
    /// therefore keep every credential in one place instead of duplicating one
    /// into the agent config just to satisfy a check.
    ///
    /// `model` stays required: it is the request model until the router picks
    /// a provider, and it drives the web-search fallback and token budgets.
    pub fn validate_pool_bound(&self) -> Result<(), String> {
        if self.model.trim().is_empty() {
            return Err("model is empty".into());
        }
        if self.root_dir.as_os_str().is_empty() {
            return Err("root_dir is empty".into());
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

    /// Directory where raw HTTP request/response dumps are persisted: `<root>/.phonebuddy/http_dumps`.
    pub fn http_dumps_dir(&self) -> PathBuf {
        if let Some(ref d) = self.http_dump.dump_dir {
            d.clone()
        } else {
            self.root_dir.join(".phonebuddy").join("http_dumps")
        }
    }

    /// Directory image attachments must live under.
    pub fn resolved_attachment_root(&self) -> PathBuf {
        self.attachment_root
            .clone()
            .unwrap_or_else(|| self.root_dir.join("image_attachments"))
    }

    pub fn root_as_path(&self) -> &Path {
        &self.root_dir
    }
}

/// Fluent builder for creating and configuring [`EngineConfig`].
#[derive(Debug, Clone, Default)]
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
        }
    }

    /// Set the LLM API base URL / endpoint URL (e.g. `https://api.anthropic.com/v1`, `https://api.x.ai/v1`).
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.config.base_url = url.into();
        self
    }

    /// Alias for [`Self::url`].
    pub fn base_url(self, url: impl Into<String>) -> Self {
        self.url(url)
    }

    /// Set the LLM API key.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = key.into();
        self
    }

    /// Set the LLM model name (e.g. `grok-4.6`, `claude-opus-5`, `gpt-4o`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Set the client emulation profile (Default, GrokBuild, Codex, ClaudeCode).
    ///
    /// Automatically applies the profile's default backend protocol, default endpoint URL
    /// (if still at default), and standard headers.
    pub fn client_profile(mut self, profile: ClientProfile) -> Self {
        self.config.client_profile = profile;
        let def = crate::llm::profiles::get_profile_definition(profile);
        if self.config.base_url == default_base_url() {
            self.config.base_url = def.default_base_url;
        }
        self.config.api_backend = def.default_backend;
        if profile == ClientProfile::GrokBuild {
            self.config.enable_web_search = true;
            self.config.enable_doom_loop_check = Some(true);
        }
        self
    }

    /// Set custom client version string to report in User-Agent.
    pub fn client_version(mut self, version: impl Into<String>) -> Self {
        self.config.client_version = Some(version.into());
        self
    }

    /// Set custom session identifier for vendor session tracking headers.
    pub fn client_session_id(mut self, id: impl Into<String>) -> Self {
        self.config.client_session_id = Some(id.into());
        self
    }

    /// Set the file sandbox root directory.
    pub fn root_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.root_dir = dir.into();
        self
    }

    /// Set the UI locale (e.g. "zh", "en").
    pub fn locale(mut self, locale: impl Into<String>) -> Self {
        self.config.locale = locale.into();
        self
    }

    /// Set max tool-loop turns per user message.
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.config.max_turns = turns;
        self
    }

    /// Set sampling temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Set max output tokens per turn.
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.config.max_output_tokens = tokens;
        self
    }

    /// Set reasoning effort level for thinking models (e.g. Low, Medium, High).
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.config.reasoning_effort = Some(effort);
        self
    }

    /// Set optional reasoning effort level for thinking models.
    pub fn reasoning_effort_opt(mut self, effort: Option<ReasoningEffort>) -> Self {
        self.config.reasoning_effort = effort;
        self
    }

    /// Enable or disable backend-hosted search on Responses API.
    pub fn enable_web_search(mut self, enable: bool) -> Self {
        self.config.enable_web_search = enable;
        self
    }

    /// Set options for backend-hosted WebSearch (allowed/excluded domains).
    pub fn web_search_options(mut self, options: WebSearchOptions) -> Self {
        self.config.web_search_options = Some(options);
        self
    }

    /// Enable or disable backend-hosted XSearch on Responses API.
    pub fn enable_x_search(mut self, enable: bool) -> Self {
        self.config.enable_x_search = enable;
        self
    }

    /// Set options for backend-hosted XSearch.
    pub fn x_search_options(mut self, options: XSearchOptions) -> Self {
        self.config.x_search_options = Some(options);
        self
    }

    /// Set whether audio input is supported by the engine / model.
    pub fn supports_audio_input(mut self, supports: bool) -> Self {
        self.config.supports_audio_input = supports;
        self
    }

    /// Set agent identity name.
    pub fn agent_name(mut self, name: impl Into<String>) -> Self {
        self.config.agent_name = name.into();
        self
    }

    /// Set extra system prompt instructions.
    pub fn system_prompt_extra(mut self, extra: impl Into<String>) -> Self {
        self.config.system_prompt_extra = Some(extra.into());
        self
    }

    /// Set streaming idle timeout in seconds.
    pub fn stream_idle_timeout_secs(mut self, secs: u64) -> Self {
        self.config.stream_idle_timeout_secs = secs;
        self
    }

    /// Set max API retries on transient errors.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Set LLM transport mode (Http or Host).
    pub fn llm_mode(mut self, mode: LlmMode) -> Self {
        self.config.llm_mode = mode;
        self
    }

    /// Set the API backend protocol (ChatCompletions, Responses, Messages).
    pub fn api_backend(mut self, backend: ApiBackend) -> Self {
        self.config.api_backend = backend;
        self
    }

    /// Add an extra HTTP header.
    pub fn extra_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.extra_headers.insert(key.into(), value.into());
        self
    }

    /// Extend extra HTTP headers.
    pub fn extra_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.config.extra_headers.extend(headers);
        self
    }

    /// Add an extra field to merge into the request JSON payload.
    pub fn extra_body(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.extra_body.insert(key.into(), value);
        self
    }

    /// Extend extra request JSON payload fields.
    pub fn extra_body_map(
        mut self,
        map: impl IntoIterator<Item = (String, serde_json::Value)>,
    ) -> Self {
        self.config.extra_body.extend(map);
        self
    }

    /// Opt into or out of server-side doom loop checking.
    pub fn enable_doom_loop_check(mut self, enable: bool) -> Self {
        self.config.enable_doom_loop_check = Some(enable);
        self
    }

    /// Allow web_fetch to hit explicit loopback addresses.
    pub fn web_fetch_allow_local(mut self, allow: bool) -> Self {
        self.config.web_fetch_allow_local = allow;
        self
    }

    /// Set HTTP dump configuration for request/response diagnostics.
    pub fn http_dump(mut self, dump: HttpDumpConfig) -> Self {
        self.config.http_dump = dump;
        self
    }

    /// Replace the ordered fallback provider list.
    pub fn fallback_providers(mut self, providers: Vec<ProviderEndpoint>) -> Self {
        self.config.fallback_providers = providers;
        self
    }

    /// Push one fallback endpoint onto the chain.
    pub fn fallback_provider(mut self, endpoint: ProviderEndpoint) -> Self {
        self.config.fallback_providers.push(endpoint);
        self
    }

    /// Set per-provider attempt budget used when fallbacks are configured.
    pub fn failover_max_attempts(mut self, attempts: u32) -> Self {
        self.config.failover_max_attempts = attempts;
        self
    }

    /// Set the starting cooldown (seconds) after a provider trip.
    pub fn provider_cooldown_secs(mut self, secs: u64) -> Self {
        self.config.provider_cooldown_secs = secs;
        self
    }

    /// Set the encrypted-reasoning compatibility group for the primary
    /// provider. Fallbacks use [`ProviderEndpoint::provider_group`].
    pub fn provider_group(mut self, group: impl Into<String>) -> Self {
        self.config.provider_group = Some(group.into());
        self
    }

    /// Validate and build the [`EngineConfig`].
    pub fn build(self) -> Result<EngineConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without running validation.
    pub fn build_unvalidated(self) -> EngineConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_search_active_only_on_responses() {
        let mut cfg = EngineConfig::default();
        assert!(!cfg.backend_search_active());

        cfg.enable_web_search = true;
        assert!(!cfg.backend_search_active());

        cfg.api_backend = ApiBackend::Responses;
        assert!(cfg.backend_search_active());

        cfg.api_backend = ApiBackend::Messages;
        assert!(!cfg.backend_search_active());
    }

    #[test]
    fn test_engine_config_builder_fluent_and_presets() {
        let grok_cfg = EngineConfig::for_grok("xai-key-123", "grok-4.6")
            .url("https://custom.x.ai/v1")
            .extra_header("X-Custom", "Val")
            .build()
            .unwrap();

        assert_eq!(grok_cfg.client_profile, ClientProfile::GrokBuild);
        assert_eq!(grok_cfg.base_url, "https://custom.x.ai/v1");
        assert_eq!(grok_cfg.api_backend, ApiBackend::Responses);
        assert_eq!(grok_cfg.api_key, "xai-key-123");
        assert_eq!(grok_cfg.model, "grok-4.6");
        assert!(grok_cfg.enable_web_search);
        assert_eq!(grok_cfg.enable_doom_loop_check, Some(true));
        assert_eq!(grok_cfg.extra_headers.get("X-Custom").unwrap(), "Val");

        let claude_cfg = EngineConfig::for_claude_code("sk-ant-123", "claude-opus-5")
            .client_session_id("sess-custom-uuid")
            .build()
            .unwrap();

        assert_eq!(claude_cfg.client_profile, ClientProfile::ClaudeCode);
        assert_eq!(claude_cfg.base_url, "https://api.anthropic.com/v1");
        assert_eq!(claude_cfg.api_backend, ApiBackend::Messages);
        assert_eq!(
            claude_cfg.client_session_id.as_deref(),
            Some("sess-custom-uuid")
        );

        let gemini_cfg = EngineConfig::for_gemini("AIza-test", "gemini-2.5-flash")
            .build()
            .unwrap();
        assert_eq!(gemini_cfg.api_backend, ApiBackend::Gemini);
        assert_eq!(
            gemini_cfg.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert_eq!(gemini_cfg.model, "gemini-2.5-flash");
    }

    #[test]
    fn fallback_fields_default_and_round_trip() {
        let json = r#"{
            "api_key": "k",
            "base_url": "https://api.example.com/v1",
            "model": "m",
            "root_dir": "/tmp/pb"
        }"#;
        let cfg: EngineConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.fallback_providers.is_empty());
        assert_eq!(cfg.failover_max_attempts, 3);
        assert_eq!(cfg.provider_cooldown_secs, 120);

        let mut cfg2 = EngineConfig::default();
        cfg2.api_key = "k".into();
        cfg2.fallback_providers.push(ProviderEndpoint {
            base_url: "https://api.openai.com/v1".into(),
            api_key: "sk-test".into(),
            model: "gpt-4o".into(),
            ..Default::default()
        });
        let encoded = serde_json::to_string(&cfg2).unwrap();
        let decoded: EngineConfig = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.fallback_providers.len(), 1);
        assert_eq!(decoded.fallback_providers[0].model, "gpt-4o");
    }
}
