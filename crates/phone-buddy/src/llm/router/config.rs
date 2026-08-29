//! Routing configuration types, validation, and the legacy EngineConfig adapter.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::config::{EngineConfig, ProviderEndpoint};
use crate::llm::failover::{compatibility_key, resolve_provider_group};
use crate::llm::profiles::ClientProfile;
use crate::llm::types::{ApiBackend, ReasoningEffort, WebSearchOptions, XSearchOptions};

/// Default pool used by the main agent engine.
pub const MAIN_POOL_ID: &str = "main";
/// Default pool used by subagents. HTTP engines bind TaskManager here.
/// The legacy adapter copies `main` into this pool; a direct routing config
/// that omits it is [`crate::error::EngineError::RouteNotConfigured`].
pub const SUBAGENT_POOL_ID: &str = "subagent";
/// Synthesized id for the EngineConfig primary endpoint.
pub const LEGACY_PRIMARY_PROVIDER_ID: &str = "legacy-primary";
/// Routing group assigned to every member of a synthesized legacy pool.
pub const DEFAULT_ROUTING_GROUP: &str = "default";
/// Default member `base_score` (Tianyan's historic starting score).
pub const DEFAULT_BASE_SCORE: i32 = 10;
/// Default failure-penalty window (one hour).
pub const DEFAULT_PENALTY_WINDOW_SECS: u64 = 3600;
/// Keep health for provider ids absent from config this long.
pub const DEFAULT_ABSENT_PROVIDER_RETENTION_SECS: u64 = 24 * 3600;

/// Which kind of work a routed LLM operation belongs to.
///
/// Pool ids are app-chosen and arbitrary, so diagnostics report the workload
/// separately: two pools may share a `provider_id`, and a `provider_id` may be
/// visited by an agent turn and a one-shot call in the same second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Workload {
    /// Main agent turn (tool loop, session, compaction).
    #[default]
    Main,
    /// Subagent task turn (tool loop, task-owned history).
    Subagent,
    /// Tool-free, session-free [`crate::runtime::PhoneBuddyRuntime::generate_text`].
    OneShot,
}

impl Workload {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Subagent => "subagent",
            Self::OneShot => "one_shot",
        }
    }
}

impl std::fmt::Display for Workload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Declarative routing snapshot owned by the host app.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmRoutingConfig {
    #[serde(default)]
    pub providers: Vec<ProviderTarget>,
    #[serde(default)]
    pub pools: BTreeMap<String, ProviderPool>,
    #[serde(default)]
    pub health: RouterHealthConfig,
}

/// One concrete routable target. `provider_id` is the stable, secret-free key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderTarget {
    pub provider_id: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub api_backend: ApiBackend,
    #[serde(default)]
    pub client_profile: ClientProfile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_session_id: Option<String>,
    /// When unset or empty, encrypted reasoning is treated as unique to
    /// this `provider_id` (never replayed onto another target).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_compatibility_key: Option<String>,
    #[serde(default)]
    pub capabilities: ProviderCapabilities,
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    #[serde(default)]
    pub extra_body: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub enable_web_search: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    #[serde(default)]
    pub enable_x_search: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_search_options: Option<XSearchOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// Declared model capabilities for a target. Unused by selection itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    #[serde(default = "default_true")]
    pub supports_image_input: bool,
    #[serde(default = "default_true")]
    pub supports_audio_input: bool,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            supports_image_input: true,
            supports_audio_input: true,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Named pool of providers with a shared retry/exhaustion policy.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderPool {
    #[serde(default)]
    pub members: Vec<PoolMember>,
    #[serde(default)]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub when_exhausted: ExhaustionPolicy,
}

/// One membership row inside a pool. Health is keyed by `provider_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolMember {
    pub provider_id: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    #[serde(default = "default_base_score")]
    pub base_score: i32,
    #[serde(default)]
    pub order: u32,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_routing_group() -> String {
    DEFAULT_ROUTING_GROUP.into()
}

fn default_base_score() -> i32 {
    DEFAULT_BASE_SCORE
}

/// Per-pool attempt budgets. Single-member pools use `max_retries`;
/// multi-member pools use `failover_max_attempts` per visit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    #[serde(default = "default_failover_max_attempts")]
    pub failover_max_attempts: u32,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            failover_max_attempts: default_failover_max_attempts(),
            max_retries: default_max_retries(),
        }
    }
}

fn default_failover_max_attempts() -> u32 {
    crate::llm::failover::DEFAULT_FAILOVER_MAX_ATTEMPTS
}

fn default_max_retries() -> u32 {
    5
}

/// What to do when no member is normally eligible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExhaustionPolicy {
    /// Visit the enabled provider whose cooldown expires first.
    #[default]
    ProbeEarliest,
    /// Fail immediately with [`crate::error::EngineError::PoolExhausted`].
    FailFast,
}

/// SDK-owned scoring / cooldown policy. Values, not host-side constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterHealthConfig {
    #[serde(default = "default_penalty_window")]
    pub penalty_window_secs: u64,
    #[serde(default = "default_cooldown_base")]
    pub cooldown_base_secs: u64,
    #[serde(default = "default_cooldown_cap")]
    pub cooldown_cap_secs: u64,
    #[serde(default = "default_retention")]
    pub absent_provider_retention_secs: u64,
}

impl Default for RouterHealthConfig {
    fn default() -> Self {
        Self {
            penalty_window_secs: default_penalty_window(),
            cooldown_base_secs: default_cooldown_base(),
            cooldown_cap_secs: default_cooldown_cap(),
            absent_provider_retention_secs: default_retention(),
        }
    }
}

fn default_penalty_window() -> u64 {
    DEFAULT_PENALTY_WINDOW_SECS
}
fn default_cooldown_base() -> u64 {
    crate::llm::failover::DEFAULT_PROVIDER_COOLDOWN_SECS
}
fn default_cooldown_cap() -> u64 {
    crate::llm::failover::MAX_PROVIDER_COOLDOWN_SECS
}
fn default_retention() -> u64 {
    DEFAULT_ABSENT_PROVIDER_RETENTION_SECS
}

impl ProviderTarget {
    /// Compatibility key used when stripping encrypted reasoning.
    ///
    /// A missing or blank key defaults to the unique `provider_id` so
    /// artifacts never leak across unverified targets.
    pub fn resolved_compat_key(&self) -> String {
        resolve_compat_key(
            self.reasoning_compatibility_key.as_deref(),
            &self.provider_id,
        )
    }
}

/// Resolve a reasoning-compatibility key, defaulting to `provider_id`.
pub fn resolve_compat_key(explicit: Option<&str>, provider_id: &str) -> String {
    explicit
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| provider_id.to_string())
}

impl LlmRoutingConfig {
    pub fn provider(&self, id: &str) -> Option<&ProviderTarget> {
        self.providers.iter().find(|p| p.provider_id == id)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut seen = HashSet::new();
        for (i, p) in self.providers.iter().enumerate() {
            validate_provider_id(&p.provider_id).map_err(|e| format!("providers[{i}].{e}"))?;
            if !seen.insert(p.provider_id.clone()) {
                return Err(format!("duplicate provider_id '{}'", p.provider_id));
            }
            if p.model.trim().is_empty() {
                return Err(format!("providers[{i}].model is empty"));
            }
            if p.api_key.trim().is_empty() {
                return Err(format!("providers[{i}].api_key is empty"));
            }
            if !p.base_url.starts_with("http://") && !p.base_url.starts_with("https://") {
                return Err(format!(
                    "providers[{i}].base_url must be an http(s) URL: {}",
                    p.base_url
                ));
            }
        }
        for (pool_id, pool) in &self.pools {
            if pool_id.trim().is_empty() {
                return Err("pool id is empty".into());
            }
            if pool.members.is_empty() {
                return Err(format!("pools.{pool_id} has no members"));
            }
            let mut in_pool = HashSet::new();
            for (i, member) in pool.members.iter().enumerate() {
                if member.provider_id.trim().is_empty() {
                    return Err(format!("pools.{pool_id}.members[{i}].provider_id is empty"));
                }
                if !in_pool.insert(member.provider_id.clone()) {
                    return Err(format!(
                        "pools.{pool_id} duplicate provider_id '{}'",
                        member.provider_id
                    ));
                }
                if !seen.contains(&member.provider_id) {
                    return Err(format!(
                        "pools.{pool_id}.members[{i}] references unknown provider_id '{}'",
                        member.provider_id
                    ));
                }
                if member.routing_group.trim().is_empty() {
                    return Err(format!(
                        "pools.{pool_id}.members[{i}].routing_group is empty"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_provider_id(id: &str) -> Result<(), String> {
    if id.trim().is_empty() {
        return Err("provider_id is empty".into());
    }
    if id.trim() != id {
        return Err("provider_id must not have leading or trailing whitespace".into());
    }
    if provider_id_looks_like_secret(id) {
        return Err("provider_id must not contain a secret".into());
    }
    Ok(())
}

fn provider_id_looks_like_secret(id: &str) -> bool {
    let lower = id.to_ascii_lowercase();
    lower.contains("sk-")
        || lower.contains("api_key")
        || lower.contains("bearer ")
        || lower.contains("authorization")
        || id.contains("://")
}

pub fn legacy_fallback_provider_id(index: usize) -> String {
    format!("legacy-fallback-{index}")
}

/// Build a routing snapshot from the historic primary + `fallback_providers` chain.
///
/// Synthesizes a `main` pool and a `subagent` copy of it. Does **not** create
/// a `session_title` pool.
pub fn synthesize_legacy_routing(cfg: &EngineConfig) -> Result<LlmRoutingConfig, String> {
    let mut providers = Vec::new();
    providers.push(target_from_primary(cfg));
    for (i, ep) in cfg.fallback_providers.iter().enumerate() {
        providers.push(target_from_endpoint(ep, i, cfg));
    }

    let members: Vec<PoolMember> = providers
        .iter()
        .enumerate()
        .map(|(i, p)| PoolMember {
            provider_id: p.provider_id.clone(),
            routing_group: DEFAULT_ROUTING_GROUP.into(),
            base_score: DEFAULT_BASE_SCORE,
            order: i as u32,
            enabled: true,
        })
        .collect();

    let pool = ProviderPool {
        members,
        retry: RetryPolicy {
            failover_max_attempts: cfg.failover_max_attempts.max(1),
            max_retries: cfg.max_retries.max(1),
        },
        when_exhausted: ExhaustionPolicy::ProbeEarliest,
    };

    let mut pools = BTreeMap::new();
    pools.insert(MAIN_POOL_ID.to_string(), pool.clone());
    pools.insert(SUBAGENT_POOL_ID.to_string(), pool);

    let routing = LlmRoutingConfig {
        providers,
        pools,
        health: RouterHealthConfig {
            cooldown_base_secs: cfg.provider_cooldown_secs,
            ..RouterHealthConfig::default()
        },
    };
    routing.validate()?;
    Ok(routing)
}

fn target_from_primary(cfg: &EngineConfig) -> ProviderTarget {
    let group = resolve_provider_group(cfg.provider_group.as_deref(), cfg.client_profile);
    ProviderTarget {
        provider_id: LEGACY_PRIMARY_PROVIDER_ID.to_string(),
        base_url: cfg.base_url.clone(),
        api_key: cfg.api_key.clone(),
        model: cfg.model.clone(),
        api_backend: cfg.api_backend,
        client_profile: cfg.client_profile,
        client_version: cfg.client_version.clone(),
        client_session_id: cfg.client_session_id.clone(),
        reasoning_compatibility_key: Some(compatibility_key(&group, &cfg.model)),
        capabilities: ProviderCapabilities {
            supports_image_input: cfg.supports_image_input,
            supports_audio_input: cfg.supports_audio_input,
        },
        extra_headers: cfg.extra_headers.clone(),
        extra_body: cfg.extra_body.clone(),
        enable_web_search: cfg.enable_web_search,
        web_search_options: cfg.web_search_options.clone(),
        enable_x_search: cfg.enable_x_search,
        x_search_options: cfg.x_search_options.clone(),
        reasoning_effort: cfg.reasoning_effort,
    }
}

fn target_from_endpoint(ep: &ProviderEndpoint, index: usize, cfg: &EngineConfig) -> ProviderTarget {
    let group = resolve_provider_group(ep.provider_group.as_deref(), ep.client_profile);
    ProviderTarget {
        provider_id: legacy_fallback_provider_id(index),
        base_url: ep.base_url.clone(),
        api_key: ep.api_key.clone(),
        model: ep.model.clone(),
        api_backend: ep.api_backend,
        client_profile: ep.client_profile,
        client_version: ep.client_version.clone(),
        client_session_id: ep.client_session_id.clone(),
        reasoning_compatibility_key: Some(compatibility_key(&group, &ep.model)),
        capabilities: ProviderCapabilities {
            supports_image_input: cfg.supports_image_input,
            supports_audio_input: cfg.supports_audio_input,
        },
        extra_headers: ep.extra_headers.clone(),
        extra_body: ep.extra_body.clone(),
        enable_web_search: ep.enable_web_search,
        web_search_options: ep
            .web_search_options
            .clone()
            .or_else(|| cfg.web_search_options.clone()),
        enable_x_search: ep.enable_x_search,
        x_search_options: ep
            .x_search_options
            .clone()
            .or_else(|| cfg.x_search_options.clone()),
        reasoning_effort: ep.reasoning_effort.or(cfg.reasoning_effort),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_target(id: &str) -> ProviderTarget {
        ProviderTarget {
            provider_id: id.into(),
            base_url: "https://api.example.com/v1".into(),
            api_key: "k".into(),
            model: "m".into(),
            api_backend: ApiBackend::ChatCompletions,
            client_profile: ClientProfile::Default,
            client_version: None,
            client_session_id: None,
            reasoning_compatibility_key: None,
            capabilities: ProviderCapabilities::default(),
            extra_headers: HashMap::new(),
            extra_body: HashMap::new(),
            enable_web_search: false,
            web_search_options: None,
            enable_x_search: false,
            x_search_options: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn missing_compat_key_defaults_to_provider_id() {
        let t = sample_target("hermes-grok-main");
        assert_eq!(t.resolved_compat_key(), "hermes-grok-main");
        assert_eq!(resolve_compat_key(Some("  "), "p1"), "p1");
        assert_eq!(
            resolve_compat_key(Some("grok_build/grok-4.6"), "p1"),
            "grok_build/grok-4.6"
        );
    }

    #[test]
    fn rejects_duplicate_and_secret_provider_ids() {
        let mut cfg = LlmRoutingConfig {
            providers: vec![sample_target("a"), sample_target("a")],
            pools: BTreeMap::new(),
            health: RouterHealthConfig::default(),
        };
        assert!(cfg.validate().unwrap_err().contains("duplicate"));

        cfg.providers[1].provider_id = "sk-abc123".into();
        assert!(cfg.validate().unwrap_err().contains("secret"));

        cfg.providers[1].provider_id = "https://evil.example/p".into();
        assert!(cfg.validate().unwrap_err().contains("secret"));
    }

    #[test]
    fn pool_member_must_exist() {
        let mut pools = BTreeMap::new();
        pools.insert(
            "main".into(),
            ProviderPool {
                members: vec![PoolMember {
                    provider_id: "missing".into(),
                    routing_group: "g".into(),
                    base_score: 10,
                    order: 0,
                    enabled: true,
                }],
                retry: RetryPolicy::default(),
                when_exhausted: ExhaustionPolicy::ProbeEarliest,
            },
        );
        let cfg = LlmRoutingConfig {
            providers: vec![sample_target("a")],
            pools,
            health: RouterHealthConfig::default(),
        };
        assert!(cfg.validate().unwrap_err().contains("unknown provider_id"));
    }

    #[test]
    fn rejects_empty_and_duplicate_pool_members() {
        let mut pools = BTreeMap::new();
        pools.insert(
            "main".into(),
            ProviderPool {
                members: vec![],
                retry: RetryPolicy::default(),
                when_exhausted: ExhaustionPolicy::ProbeEarliest,
            },
        );
        let mut cfg = LlmRoutingConfig {
            providers: vec![sample_target("a")],
            pools,
            health: RouterHealthConfig::default(),
        };
        assert!(cfg.validate().unwrap_err().contains("no members"));

        cfg.pools.get_mut("main").unwrap().members = vec![
            PoolMember {
                provider_id: "a".into(),
                routing_group: "g".into(),
                base_score: 10,
                order: 0,
                enabled: true,
            },
            PoolMember {
                provider_id: "a".into(),
                routing_group: "g".into(),
                base_score: 5,
                order: 1,
                enabled: true,
            },
        ];
        assert!(cfg
            .validate()
            .unwrap_err()
            .contains("duplicate provider_id"));
    }

    #[test]
    fn legacy_synthesizes_main_and_subagent_not_session_title() {
        let mut cfg = EngineConfig::default();
        cfg.api_key = "k".into();
        cfg.base_url = "https://primary.example/v1".into();
        cfg.model = "grok-4.6".into();
        cfg.client_profile = ClientProfile::GrokBuild;
        cfg.provider_group = Some("packy".into());
        cfg.fallback_providers.push(ProviderEndpoint {
            base_url: "https://api.openai.com/v1".into(),
            api_key: "sk-test".into(),
            model: "gpt-4o".into(),
            client_profile: ClientProfile::Codex,
            ..Default::default()
        });

        let routing = synthesize_legacy_routing(&cfg).unwrap();
        assert!(routing.pools.contains_key(MAIN_POOL_ID));
        assert!(routing.pools.contains_key(SUBAGENT_POOL_ID));
        assert!(!routing.pools.contains_key("session_title"));

        assert_eq!(routing.providers[0].provider_id, LEGACY_PRIMARY_PROVIDER_ID);
        assert_eq!(routing.providers[1].provider_id, "legacy-fallback-0");
        assert_eq!(routing.providers[0].resolved_compat_key(), "packy/grok-4.6");
        assert_eq!(routing.providers[1].resolved_compat_key(), "codex/gpt-4o");

        let main = &routing.pools[MAIN_POOL_ID];
        let sub = &routing.pools[SUBAGENT_POOL_ID];
        assert_eq!(main.members.len(), 2);
        assert_eq!(sub.members.len(), 2);
        assert_eq!(main.members[0].order, 0);
        assert_eq!(main.members[1].order, 1);
        assert_eq!(main.members[0].routing_group, DEFAULT_ROUTING_GROUP);
        assert_eq!(main.members[0].base_score, DEFAULT_BASE_SCORE);
        assert_eq!(main.when_exhausted, ExhaustionPolicy::ProbeEarliest);
    }
}
