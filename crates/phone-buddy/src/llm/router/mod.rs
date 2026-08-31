//! SDK-owned LLM router: named pools, shared health, deterministic selection.

mod config;
mod health;
mod select;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};

pub use config::{
    synthesize_legacy_routing, ExhaustionPolicy, LlmRoutingConfig, PoolMember,
    ProviderCapabilities, ProviderPool, ProviderTarget, RetryPolicy, RouterHealthConfig, Workload,
    DEFAULT_BASE_SCORE, DEFAULT_ROUTING_GROUP, LEGACY_PRIMARY_PROVIDER_ID, MAIN_POOL_ID,
    SUBAGENT_POOL_ID,
};
pub use health::{health_file_path, FailureClass, ProviderHealthRecord};
pub use select::VisitPlan;

use config::resolve_compat_key;
use health::{load_health_file, reconcile_health, save_health_file};
use select::{select_visit_order, SelectError};

use crate::error::{EngineError, EngineResult};

/// Shared router state. Longer-lived than an engine; not [`Clone`].
pub struct LlmRouter {
    inner: Mutex<RouterInner>,
    persist_path: Option<PathBuf>,
}

struct RouterInner {
    generation: u64,
    config: LlmRoutingConfig,
    health: HashMap<String, ProviderHealthRecord>,
}

impl LlmRouter {
    /// In-memory router (tests and injected transports). Does not persist.
    pub fn in_memory(config: LlmRoutingConfig) -> EngineResult<Arc<Self>> {
        Self::open(config, None)
    }

    /// Router that loads and stores health under `root_dir`.
    pub fn persist(
        config: LlmRoutingConfig,
        root_dir: impl Into<PathBuf>,
    ) -> EngineResult<Arc<Self>> {
        Self::open(config, Some(root_dir.into()))
    }

    fn open(config: LlmRoutingConfig, root_dir: Option<PathBuf>) -> EngineResult<Arc<Self>> {
        config
            .validate()
            .map_err(EngineError::InvalidRoutingConfig)?;
        let persist_path = root_dir.as_ref().map(|d| health_file_path(d));
        let mut health = persist_path
            .as_ref()
            .map(|p| load_health_file(p))
            .unwrap_or_default();
        let now = Utc::now();
        reconcile_health(&mut health, &config, now);
        let router = Arc::new(Self {
            inner: Mutex::new(RouterInner {
                generation: 0,
                config,
                health,
            }),
            persist_path,
        });
        router.flush_health();
        Ok(router)
    }

    pub fn generation(&self) -> u64 {
        self.inner.lock().unwrap().generation
    }

    pub fn snapshot_config(&self) -> LlmRoutingConfig {
        self.inner.lock().unwrap().config.clone()
    }

    /// Config snapshot paired with the generation it was taken from.
    pub fn snapshot(&self) -> (u64, LlmRoutingConfig) {
        let inner = self.inner.lock().unwrap();
        (inner.generation, inner.config.clone())
    }

    pub fn has_pool(&self, pool_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap()
            .config
            .pools
            .contains_key(pool_id)
    }

    pub fn health_record(&self, provider_id: &str) -> Option<ProviderHealthRecord> {
        self.inner.lock().unwrap().health.get(provider_id).cloned()
    }

    pub fn resolved_compat_key(&self, provider_id: &str) -> String {
        let inner = self.inner.lock().unwrap();
        inner
            .config
            .provider(provider_id)
            .map(|p| p.resolved_compat_key())
            .unwrap_or_else(|| resolve_compat_key(None, provider_id))
    }

    /// Replace routing policy. In-flight operations may finish on a
    /// previously captured visit plan; health is reconciled by `provider_id`.
    pub fn update_config(&self, config: LlmRoutingConfig) -> EngineResult<()> {
        config
            .validate()
            .map_err(EngineError::InvalidRoutingConfig)?;
        let mut inner = self.inner.lock().unwrap();
        let now = Utc::now();
        reconcile_health(&mut inner.health, &config, now);
        inner.generation = inner.generation.wrapping_add(1);
        inner.config = config;
        self.write_health(&inner);
        Ok(())
    }

    /// Capture a visit order for one operation. Health timestamps are pruned
    /// under the router lock before ranking.
    pub fn plan_visit(&self, pool_id: &str) -> EngineResult<VisitPlan> {
        self.plan_visit_at(pool_id, Utc::now())
    }

    pub fn plan_visit_at(&self, pool_id: &str, now: DateTime<Utc>) -> EngineResult<VisitPlan> {
        let mut inner = self.inner.lock().unwrap();
        let RouterInner {
            generation,
            config,
            health,
        } = &mut *inner;
        let pool =
            config
                .pools
                .get(pool_id)
                .cloned()
                .ok_or_else(|| EngineError::RouteNotConfigured {
                    pool_id: pool_id.to_string(),
                })?;
        reconcile_health(health, config, now);
        match select_visit_order(pool_id, &pool, health, &config.health, now) {
            Ok(mut plan) => {
                plan.generation = *generation;
                plan.retry = pool.retry.clone();
                emit_ranked_server_list(&plan, config, health, now);
                Ok(plan)
            }
            Err(SelectError::FailFast { retry_after_ms }) => Err(EngineError::PoolExhausted {
                pool_id: pool_id.to_string(),
                retry_after_ms,
            }),
        }
    }

    pub fn record_trip(
        &self,
        operation_id: &str,
        provider_id: &str,
        class: FailureClass,
        retry_after: Option<Duration>,
    ) -> Duration {
        self.record_trip_at(operation_id, provider_id, class, retry_after, Utc::now())
    }

    pub fn record_trip_at(
        &self,
        operation_id: &str,
        provider_id: &str,
        class: FailureClass,
        retry_after: Option<Duration>,
        now: DateTime<Utc>,
    ) -> Duration {
        let mut inner = self.inner.lock().unwrap();
        let (wait, trips, failures_count) = {
            let RouterInner { config, health, .. } = &mut *inner;
            let rec = health.entry(provider_id.to_string()).or_default();
            rec.last_seen_in_config = Some(now);
            let wait = rec.trip(now, operation_id, class, &config.health, retry_after);
            let window = Duration::from_secs(config.health.penalty_window_secs.max(1));
            let failures = rec
                .recent_failures
                .iter()
                .filter(|t| **t > (now - window))
                .count();
            (wait, rec.consecutive_trips, failures)
        };
        tracing::warn!(
            target: "phone_buddy::router",
            "Provider '{}' TRIPPED (class={}, op={}): cooldown={}s, consecutive_trips={}, penalty_failures={}",
            provider_id,
            class.as_str(),
            operation_id,
            wait.as_secs(),
            trips,
            failures_count
        );
        self.write_health(&inner);
        wait
    }

    pub fn record_success(&self, provider_id: &str) {
        self.record_success_at(provider_id, Utc::now());
    }

    pub fn record_success_at(&self, provider_id: &str, now: DateTime<Utc>) {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner.health.entry(provider_id.to_string()).or_default();
        let had_trips = rec.consecutive_trips > 0 || rec.cooldown_until.is_some();
        rec.recover(now);
        rec.last_seen_in_config = Some(now);
        if had_trips {
            tracing::info!(
                target: "phone_buddy::router",
                "Provider '{}' probe SUCCEEDED, health recovered (cooldown cleared, consecutive_trips reset to 0)",
                provider_id
            );
        }
        self.write_health(&inner);
    }

    fn flush_health(&self) {
        let inner = self.inner.lock().unwrap();
        self.write_health(&inner);
    }

    fn write_health(&self, inner: &RouterInner) {
        let Some(path) = &self.persist_path else {
            return;
        };
        if let Err(e) = save_health_file(path, &inner.health) {
            tracing::warn!("failed to persist router health at {}: {e}", path.display());
        }
    }
}

/// Ranked catalog dumped immediately before an LLM HTTP visit plan is used.
///
/// Goes through [`crate::diag`] so release `.so` builds still reach logcat;
/// `tracing::info!` is compiled out of those binaries.
fn emit_ranked_server_list(
    plan: &VisitPlan,
    config: &LlmRoutingConfig,
    health: &HashMap<String, ProviderHealthRecord>,
    now: DateTime<Utc>,
) {
    let window = Duration::from_secs(config.health.penalty_window_secs.max(1));
    let pool = config.pools.get(&plan.pool_id);
    let lines: Vec<String> = plan
        .provider_ids
        .iter()
        .enumerate()
        .map(|(rank, pid)| {
            let member = pool.and_then(|p| p.members.iter().find(|m| m.provider_id == *pid));
            let target = config.provider(pid);
            let base = member.map(|m| m.base_score).unwrap_or(DEFAULT_BASE_SCORE);
            let rec = health.get(pid);
            let score = rec
                .map(|r| r.effective_score(base, now, window))
                .unwrap_or(base);
            let group = member
                .map(|m| m.routing_group.as_str())
                .unwrap_or("default");
            let url = target.map(|t| t.base_url.as_str()).unwrap_or("");
            let model = target.map(|t| t.model.as_str()).unwrap_or("default");
            let mut tags = String::new();
            if rec.map(|r| r.is_cooling(now)).unwrap_or(false) {
                let remaining = rec
                    .map(|r| r.cooldown_remaining(now).as_secs())
                    .unwrap_or(0);
                tags.push_str(&format!(" [cooling: {remaining}s left]"));
            }
            if rank == 0 {
                tags.push_str(" ★ PRIMARY");
            }
            format!(
                "  #{rank} [score: {score} base={base}] {pid} ({url} | model: {model} | group: {group}){tags}"
            )
        })
        .collect();
    let msg = format!(
        "[LLM Request] Pool '{}' Server List ({} servers, ranked by live score):\n{}",
        plan.pool_id,
        plan.provider_ids.len(),
        lines.join("\n")
    );
    tracing::info!(target: "phone_buddy::router", "{msg}");
    crate::diag::info("phone_buddy::router", &msg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::router::config::{
        PoolMember, ProviderPool, ProviderTarget, DEFAULT_ROUTING_GROUP,
    };
    use chrono::TimeZone;

    fn target(id: &str, url: &str) -> ProviderTarget {
        ProviderTarget {
            provider_id: id.into(),
            base_url: url.into(),
            api_key: "k".into(),
            model: "m".into(),
            api_backend: Default::default(),
            client_profile: Default::default(),
            client_version: None,
            client_session_id: None,
            reasoning_compatibility_key: None,
            capabilities: Default::default(),
            extra_headers: HashMap::new(),
            extra_body: HashMap::new(),
            enable_web_search: false,
            web_search_options: None,
            enable_x_search: false,
            x_search_options: None,
            reasoning_effort: None,
        }
    }

    fn member(id: &str, order: u32) -> PoolMember {
        PoolMember {
            provider_id: id.into(),
            routing_group: DEFAULT_ROUTING_GROUP.into(),
            base_score: 10,
            order,
            enabled: true,
        }
    }

    fn two_provider_config(same_url: bool) -> LlmRoutingConfig {
        let url_a = "https://api.example.com/v1";
        let url_b = if same_url {
            url_a
        } else {
            "https://backup.example.com/v1"
        };
        let mut pools = std::collections::BTreeMap::new();
        pools.insert(
            "main".into(),
            ProviderPool {
                members: vec![member("p-a", 0), member("p-b", 1)],
                ..Default::default()
            },
        );
        pools.insert(
            "subagent".into(),
            ProviderPool {
                members: vec![member("p-a", 0)],
                ..Default::default()
            },
        );
        LlmRoutingConfig {
            providers: vec![target("p-a", url_a), target("p-b", url_b)],
            pools,
            health: RouterHealthConfig::default(),
        }
    }

    #[test]
    fn same_provider_id_in_two_pools_shares_trips() {
        let router = LlmRouter::in_memory(two_provider_config(false)).unwrap();
        let t = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        router.record_trip_at("op1", "p-a", FailureClass::RetryableHttp, None, t);
        assert!(router.health_record("p-a").unwrap().is_cooling(t));
        let main = router.plan_visit_at("main", t).unwrap();
        assert_eq!(main.provider_ids[0], "p-b");
        let sub = router.plan_visit_at("subagent", t).unwrap();
        // Only member is cooling; probe_earliest still returns it.
        assert_eq!(sub.provider_ids[0], "p-a");
        assert!(router.health_record("p-a").unwrap().is_cooling(t));
    }

    #[test]
    fn distinct_ids_same_url_have_independent_health() {
        let router = LlmRouter::in_memory(two_provider_config(true)).unwrap();
        let t = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        router.record_trip_at("op1", "p-a", FailureClass::RetryableHttp, None, t);
        assert!(router.health_record("p-a").unwrap().is_cooling(t));
        assert!(!router
            .health_record("p-b")
            .map(|h| h.is_cooling(t))
            .unwrap_or(false));
        let plan = router.plan_visit_at("main", t).unwrap();
        assert_eq!(plan.provider_ids[0], "p-b");
    }

    #[test]
    fn config_update_bumps_generation() {
        let router = LlmRouter::in_memory(two_provider_config(false)).unwrap();
        assert_eq!(router.generation(), 0);
        router.update_config(two_provider_config(true)).unwrap();
        assert_eq!(router.generation(), 1);
        let plan = router.plan_visit("main").unwrap();
        assert_eq!(plan.generation, 1);
    }

    #[test]
    fn config_update_reconciles_by_stable_id() {
        let router = LlmRouter::in_memory(two_provider_config(false)).unwrap();
        let t = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        router.record_trip_at("op1", "p-a", FailureClass::Fatal, None, t);
        let trips = router.health_record("p-a").unwrap().consecutive_trips;

        let mut next = two_provider_config(false);
        next.providers[0].model = "m2".into();
        router.update_config(next).unwrap();
        assert_eq!(
            router.health_record("p-a").unwrap().consecutive_trips,
            trips
        );
    }

    #[test]
    fn persistence_reloads_cooldown() {
        let dir = tempfile::tempdir().unwrap();
        let t = Utc::now();
        let router = LlmRouter::persist(two_provider_config(false), dir.path()).unwrap();
        router.record_trip_at("op1", "p-a", FailureClass::RetryableHttp, None, t);
        drop(router);

        let reloaded = LlmRouter::persist(two_provider_config(false), dir.path()).unwrap();
        let rec = reloaded.health_record("p-a").unwrap();
        assert_eq!(rec.consecutive_trips, 1);
        assert!(rec.is_cooling(t));
    }

    #[test]
    fn missing_pool_is_not_configured() {
        let router = LlmRouter::in_memory(two_provider_config(false)).unwrap();
        let err = router.plan_visit("session_title").unwrap_err();
        match err {
            EngineError::RouteNotConfigured { pool_id } => {
                assert_eq!(pool_id, "session_title");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn plan_visit_emits_ranked_server_list_to_diag() {
        let _ = crate::diag::take_test_messages();
        let router = LlmRouter::in_memory(two_provider_config(false)).unwrap();
        let t = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        router.record_trip_at("op1", "p-a", FailureClass::RetryableHttp, None, t);
        let plan = router.plan_visit_at("main", t).unwrap();
        assert_eq!(plan.provider_ids[0], "p-b");
        let logs = crate::diag::take_test_messages();
        let dump = logs
            .iter()
            .find(|line| line.contains("[LLM Request] Pool 'main' Server List"))
            .expect("ranked server list should be dumped before the visit");
        assert!(dump.contains("p-b"), "{dump}");
        assert!(dump.contains("p-a"), "{dump}");
        assert!(dump.contains("★ PRIMARY"), "{dump}");
        assert!(dump.contains("[cooling:"), "{dump}");
    }
}
