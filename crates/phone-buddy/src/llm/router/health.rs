//! Provider health records, scoring, cooldown, and disk persistence.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::config::{LlmRoutingConfig, RouterHealthConfig};
use crate::llm::failover::cooldown_duration;

/// Classification stored on a health record and emitted in routing events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    RetryableHttp,
    RateLimited,
    Fatal,
    Connection,
    StreamIdle,
    EmptyResponse,
    Other,
}

impl FailureClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RetryableHttp => "retryable_http",
            Self::RateLimited => "rate_limited",
            Self::Fatal => "fatal",
            Self::Connection => "connection",
            Self::StreamIdle => "stream_idle",
            Self::EmptyResponse => "empty_response",
            Self::Other => "other",
        }
    }
}

/// Non-secret per-provider health. Keyed by `provider_id`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderHealthRecord {
    #[serde(default)]
    pub recent_failures: Vec<DateTime<Utc>>,
    #[serde(default)]
    pub consecutive_trips: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown_until: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_success_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_failure_class: Option<FailureClass>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_trip_operation_id: Option<String>,
    /// Last time this id was present in the live routing config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen_in_config: Option<DateTime<Utc>>,
}

impl ProviderHealthRecord {
    pub fn is_cooling(&self, now: DateTime<Utc>) -> bool {
        self.cooldown_until.map(|t| t > now).unwrap_or(false)
    }

    pub fn cooldown_remaining(&self, now: DateTime<Utc>) -> Duration {
        self.cooldown_until
            .and_then(|t| (t - now).to_std().ok())
            .unwrap_or(Duration::ZERO)
    }

    pub fn prune_failures(&mut self, now: DateTime<Utc>, penalty_window: Duration) {
        let cutoff = now - penalty_window;
        self.recent_failures.retain(|t| *t > cutoff);
        if self.cooldown_until.map(|t| t <= now).unwrap_or(false) {
            self.cooldown_until = None;
        }
    }

    pub fn effective_score(
        &self,
        base_score: i32,
        now: DateTime<Utc>,
        penalty_window: Duration,
    ) -> i32 {
        let cutoff = now - penalty_window;
        let n = self.recent_failures.iter().filter(|t| **t > cutoff).count() as i32;
        base_score.saturating_sub(n)
    }

    /// Record one logical visit trip. A second call with the same
    /// `operation_id` is idempotent (no extra penalty).
    pub fn trip(
        &mut self,
        now: DateTime<Utc>,
        operation_id: &str,
        class: FailureClass,
        health: &RouterHealthConfig,
        retry_after: Option<Duration>,
    ) -> Duration {
        if self.last_trip_operation_id.as_deref() == Some(operation_id) {
            return self.cooldown_remaining(now);
        }
        self.consecutive_trips = self.consecutive_trips.saturating_add(1);
        self.recent_failures.push(now);
        self.last_failure_class = Some(class);
        self.last_trip_operation_id = Some(operation_id.to_string());
        let computed = capped_cooldown(
            health.cooldown_base_secs,
            self.consecutive_trips,
            health.cooldown_cap_secs,
        );
        let wait = match retry_after {
            Some(after) if after > computed => after,
            _ => computed,
        };
        self.cooldown_until = Some(now + wait);
        wait
    }

    /// Success clears consecutive-trip escalation but leaves historical
    /// failures to age out of the penalty window.
    pub fn recover(&mut self, now: DateTime<Utc>) {
        self.consecutive_trips = 0;
        self.cooldown_until = None;
        self.last_success_at = Some(now);
        self.last_trip_operation_id = None;
    }
}

fn capped_cooldown(base_secs: u64, consecutive_trips: u32, cap_secs: u64) -> Duration {
    let raw = cooldown_duration(base_secs, consecutive_trips);
    raw.min(Duration::from_secs(cap_secs.max(1)))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HealthSnapshotFile {
    version: u32,
    #[serde(default)]
    records: BTreeMap<String, ProviderHealthRecord>,
}

const HEALTH_FILE_VERSION: u32 = 1;

pub fn health_file_path(root_dir: &Path) -> PathBuf {
    root_dir
        .join(".phonebuddy")
        .join("router")
        .join("health-v1.json")
}

/// Load a health snapshot. Corrupt or unsupported files fail open (empty).
pub fn load_health_file(path: &Path) -> HashMap<String, ProviderHealthRecord> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return HashMap::new();
    };
    let Ok(file) = serde_json::from_str::<HealthSnapshotFile>(&text) else {
        tracing::warn!("discarding corrupt router health file {}", path.display());
        return HashMap::new();
    };
    if file.version != HEALTH_FILE_VERSION {
        tracing::warn!(
            "discarding unsupported router health version {} at {}",
            file.version,
            path.display()
        );
        return HashMap::new();
    }
    file.records.into_iter().collect()
}

/// Atomic persist of non-secret health. Callers must serialize writers.
pub fn save_health_file(
    path: &Path,
    records: &HashMap<String, ProviderHealthRecord>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = HealthSnapshotFile {
        version: HEALTH_FILE_VERSION,
        records: records
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    };
    let text = serde_json::to_string_pretty(&file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Drop expired timestamps, bump last-seen for live ids, and forget stale
/// records after the retention window.
pub fn reconcile_health(
    records: &mut HashMap<String, ProviderHealthRecord>,
    config: &LlmRoutingConfig,
    now: DateTime<Utc>,
) {
    let live: HashSet<&str> = config
        .providers
        .iter()
        .map(|p| p.provider_id.as_str())
        .collect();
    let window = Duration::from_secs(config.health.penalty_window_secs.max(1));
    let retention = Duration::from_secs(config.health.absent_provider_retention_secs.max(1));

    for (id, rec) in records.iter_mut() {
        rec.prune_failures(now, window);
        if live.contains(id.as_str()) {
            rec.last_seen_in_config = Some(now);
        }
    }

    records.retain(|id, rec| {
        if live.contains(id.as_str()) {
            return true;
        }
        match rec.last_seen_in_config {
            Some(seen) => now
                .signed_duration_since(seen)
                .to_std()
                .map(|d| d < retention)
                .unwrap_or(false),
            None => false,
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::router::config::{
        LlmRoutingConfig, PoolMember, ProviderPool, ProviderTarget, DEFAULT_BASE_SCORE,
        MAIN_POOL_ID,
    };
    use chrono::TimeZone;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
    }

    fn health_cfg() -> RouterHealthConfig {
        RouterHealthConfig {
            penalty_window_secs: 3600,
            cooldown_base_secs: 120,
            cooldown_cap_secs: 600,
            absent_provider_retention_secs: 3600,
        }
    }

    #[test]
    fn score_decays_with_failures_in_window() {
        let now = t0();
        let mut rec = ProviderHealthRecord::default();
        rec.recent_failures = vec![
            now - chrono::Duration::minutes(10),
            now - chrono::Duration::minutes(20),
            now - chrono::Duration::hours(2),
        ];
        rec.prune_failures(now, Duration::from_secs(3600));
        assert_eq!(rec.recent_failures.len(), 2);
        assert_eq!(
            rec.effective_score(DEFAULT_BASE_SCORE, now, Duration::from_secs(3600)),
            8
        );
    }

    #[test]
    fn cooldown_escalates_and_success_does_not_erase_failures() {
        let cfg = health_cfg();
        let now = t0();
        let mut rec = ProviderHealthRecord::default();
        let w1 = rec.trip(now, "op1", FailureClass::RetryableHttp, &cfg, None);
        assert_eq!(w1, Duration::from_secs(120));
        let w2 = rec.trip(
            now + chrono::Duration::seconds(1),
            "op2",
            FailureClass::RetryableHttp,
            &cfg,
            None,
        );
        assert_eq!(w2, Duration::from_secs(240));
        let w3 = rec.trip(
            now + chrono::Duration::seconds(2),
            "op3",
            FailureClass::RetryableHttp,
            &cfg,
            None,
        );
        assert_eq!(w3, Duration::from_secs(480));
        let w4 = rec.trip(
            now + chrono::Duration::seconds(3),
            "op4",
            FailureClass::RetryableHttp,
            &cfg,
            None,
        );
        assert_eq!(w4, Duration::from_secs(600));

        rec.recover(now + chrono::Duration::seconds(4));
        assert_eq!(rec.consecutive_trips, 0);
        assert!(!rec.is_cooling(now + chrono::Duration::seconds(4)));
        assert_eq!(rec.recent_failures.len(), 4);
        assert_eq!(
            rec.effective_score(
                10,
                now + chrono::Duration::seconds(4),
                Duration::from_secs(3600)
            ),
            6
        );
    }

    #[test]
    fn trip_is_idempotent_for_one_operation() {
        let cfg = health_cfg();
        let now = t0();
        let mut rec = ProviderHealthRecord::default();
        let first = rec.trip(now, "op-same", FailureClass::RetryableHttp, &cfg, None);
        let second = rec.trip(
            now + chrono::Duration::seconds(5),
            "op-same",
            FailureClass::RetryableHttp,
            &cfg,
            None,
        );
        assert_eq!(rec.consecutive_trips, 1);
        assert_eq!(rec.recent_failures.len(), 1);
        assert_eq!(first, Duration::from_secs(120));
        assert!(second <= first);
    }

    #[test]
    fn persist_reload_and_corrupt_file_fail_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = health_file_path(dir.path());
        let mut records = HashMap::new();
        let mut rec = ProviderHealthRecord::default();
        rec.consecutive_trips = 2;
        rec.last_failure_class = Some(FailureClass::Fatal);
        records.insert("p1".into(), rec);
        save_health_file(&path, &records).unwrap();

        let loaded = load_health_file(&path);
        assert_eq!(loaded["p1"].consecutive_trips, 2);

        std::fs::write(&path, "{not json").unwrap();
        let recovered = load_health_file(&path);
        assert!(recovered.is_empty());

        std::fs::write(&path, r#"{"version":99,"records":{}}"#).unwrap();
        assert!(load_health_file(&path).is_empty());
    }

    #[test]
    fn reconcile_drops_absent_ids_after_retention() {
        let now = t0();
        let mut records = HashMap::new();
        let mut stale = ProviderHealthRecord::default();
        stale.last_seen_in_config = Some(now - chrono::Duration::hours(2));
        records.insert("gone".into(), stale);
        let mut live_rec = ProviderHealthRecord::default();
        live_rec.last_seen_in_config = Some(now - chrono::Duration::hours(2));
        records.insert("keep".into(), live_rec);

        let mut cfg = LlmRoutingConfig {
            providers: vec![ProviderTarget {
                provider_id: "keep".into(),
                base_url: "https://api.example.com/v1".into(),
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
            }],
            pools: BTreeMap::new(),
            health: health_cfg(),
        };
        cfg.pools.insert(
            MAIN_POOL_ID.into(),
            ProviderPool {
                members: vec![PoolMember {
                    provider_id: "keep".into(),
                    routing_group: "g".into(),
                    base_score: 10,
                    order: 0,
                    enabled: true,
                }],
                ..Default::default()
            },
        );
        reconcile_health(&mut records, &cfg, now);
        assert!(records.contains_key("keep"));
        assert!(!records.contains_key("gone"));
    }
}
