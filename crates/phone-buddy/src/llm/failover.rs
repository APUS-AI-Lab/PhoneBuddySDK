//! Provider health table and chain-mode selection.
//!
//! When `EngineConfig.fallback_providers` is non-empty, a degraded provider
//! sits out for a cooldown (circuit-breaker half-open). Subsequent requests
//! stick to the first healthy provider in chain order until the cooldown
//! expires, at which point the primary is passively re-probed.

use std::time::{Duration, Instant};

use url::Url;

/// Default per-provider attempt budget in chain mode (initial + 2 retries).
pub const DEFAULT_FAILOVER_MAX_ATTEMPTS: u32 = 3;

/// Default cooldown after the first trip, in seconds.
pub const DEFAULT_PROVIDER_COOLDOWN_SECS: u64 = 120;

/// Hard cap on exponential cooldown growth.
pub const MAX_PROVIDER_COOLDOWN_SECS: u64 = 600;

/// In chain mode, wait in place for a 429 only when the wait is at most this
/// long. Longer `Retry-After` values trip the provider and switch.
pub const FAILOVER_RETRY_AFTER_INLINE_CAP: Duration = Duration::from_secs(10);

/// Per-provider circuit-breaker state. Lives for the lifetime of an
/// [`crate::llm::client::LlmClient`] (i.e. one engine instance).
#[derive(Debug, Clone, Default)]
pub struct ProviderHealth {
    /// Degraded until this instant; `None` = healthy.
    pub cooldown_until: Option<Instant>,
    /// Consecutive trips; drives 120s → 240s → 480s → 600s (cap).
    pub consecutive_trips: u32,
}

impl ProviderHealth {
    pub fn is_cooling(&self, now: Instant) -> bool {
        self.cooldown_until.map(|t| t > now).unwrap_or(false)
    }

    /// Record a trip. Returns the cooldown duration applied.
    ///
    /// `retry_after`, when present, raises the cooldown to
    /// `max(computed, retry_after)` so a long 429 does not re-probe early.
    pub fn trip(
        &mut self,
        now: Instant,
        base_secs: u64,
        retry_after: Option<Duration>,
    ) -> Duration {
        self.consecutive_trips = self.consecutive_trips.saturating_add(1);
        let computed = cooldown_duration(base_secs, self.consecutive_trips);
        let wait = match retry_after {
            Some(after) if after > computed => after,
            _ => computed,
        };
        self.cooldown_until = Some(now + wait);
        wait
    }

    /// Probe succeeded: reset the breaker.
    pub fn recover(&mut self) {
        self.consecutive_trips = 0;
        self.cooldown_until = None;
    }
}

/// Exponential cooldown: `base * 2^(trips-1)`, capped at
/// [`MAX_PROVIDER_COOLDOWN_SECS`].
pub fn cooldown_duration(base_secs: u64, consecutive_trips: u32) -> Duration {
    let shift = consecutive_trips.saturating_sub(1).min(10);
    let secs = base_secs
        .saturating_mul(1u64 << shift)
        .min(MAX_PROVIDER_COOLDOWN_SECS);
    Duration::from_secs(secs)
}

/// Pick the first provider that is not cooling. If every provider is in
/// cooldown, pick the one whose cooldown expires soonest so a request always
/// has somewhere to go.
pub fn select_index(health: &[ProviderHealth], now: Instant) -> usize {
    if health.is_empty() {
        return 0;
    }
    if let Some(i) = health.iter().position(|h| !h.is_cooling(now)) {
        return i;
    }
    health
        .iter()
        .enumerate()
        .min_by_key(|(_, h)| h.cooldown_until.unwrap_or(now))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Resolve the provider group used for encrypted-reasoning compatibility.
///
/// An explicit non-empty `provider_group` wins; otherwise the client
/// profile name (`grok_build` / `codex` / `claude_code` / `default`).
/// Providers in the same group that share a model name can replay
/// reasoning item ids and encrypted thinking the way grok-build replays
/// full Responses history (`previous_response_id` stays unset).
pub fn resolve_provider_group(
    explicit: Option<&str>,
    profile: crate::llm::profiles::ClientProfile,
) -> String {
    explicit
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| profile.group_name().to_string())
}

/// Compatibility key: `{group}/{model}`. Same key ⇒ keep encrypted thinking.
pub fn compatibility_key(group: &str, model: &str) -> String {
    format!("{group}/{model}")
}

/// Desensitized provider id (`host/model`) for events. Never includes an API key.
pub fn provider_fingerprint(base_url: &str, model: &str) -> String {
    let host = Url::parse(base_url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_else(|| {
            base_url
                .trim()
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap_or(base_url)
                .to_string()
        });
    format!("{host}/{model}")
}

/// True when encrypted thinking must be stripped before encoding for `target`.
///
/// `origin` / `target` / `primary` are compatibility keys (`group/model`).
/// A missing origin is treated as a product of `primary`. Same group and
/// same model keep reasoning item ids + encrypted content even if the
/// HTTP host changed.
pub fn should_strip_origin(origin: Option<&str>, target: &str, primary: &str) -> bool {
    let origin = origin.filter(|s| !s.is_empty()).unwrap_or(primary);
    origin != target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooldown_doubles_and_caps() {
        assert_eq!(cooldown_duration(120, 1), Duration::from_secs(120));
        assert_eq!(cooldown_duration(120, 2), Duration::from_secs(240));
        assert_eq!(cooldown_duration(120, 3), Duration::from_secs(480));
        assert_eq!(cooldown_duration(120, 4), Duration::from_secs(600));
        assert_eq!(cooldown_duration(120, 8), Duration::from_secs(600));
    }

    #[test]
    fn select_skips_cooling_then_falls_back_to_earliest() {
        let now = Instant::now();
        let mut a = ProviderHealth::default();
        let b = ProviderHealth::default();
        a.trip(now, 120, None);
        let health = [a.clone(), b.clone()];
        assert_eq!(select_index(&health, now), 1);

        let mut b2 = ProviderHealth::default();
        b2.trip(now, 120, None);
        // Both cooling; A tripped first so its deadline is equal-or-earlier
        // depending on the exact Instant, but both expire ~120s out. Index 0
        // wins the min_by_key tie because enumerate order is stable for equal
        // keys only if cooldown_until differs. Force B to expire later.
        let mut late = ProviderHealth::default();
        late.cooldown_until = Some(now + Duration::from_secs(400));
        late.consecutive_trips = 1;
        let both = [a, late];
        assert_eq!(select_index(&both, now), 0);
    }

    #[test]
    fn recover_resets_trips() {
        let now = Instant::now();
        let mut h = ProviderHealth::default();
        h.trip(now, 120, None);
        assert_eq!(h.consecutive_trips, 1);
        assert!(h.is_cooling(now));
        h.recover();
        assert_eq!(h.consecutive_trips, 0);
        assert!(!h.is_cooling(now));
    }

    #[test]
    fn retry_after_raises_cooldown() {
        let now = Instant::now();
        let mut h = ProviderHealth::default();
        let wait = h.trip(now, 120, Some(Duration::from_secs(200)));
        assert_eq!(wait, Duration::from_secs(200));
    }

    #[test]
    fn fingerprint_uses_host_and_model() {
        assert_eq!(
            provider_fingerprint("https://cf.api.fan/v1", "grok-4.6"),
            "cf.api.fan/grok-4.6"
        );
        assert_eq!(
            provider_fingerprint("https://api.openai.com/v1", "gpt-5.6"),
            "api.openai.com/gpt-5.6"
        );
    }

    #[test]
    fn missing_origin_is_primary() {
        assert!(!should_strip_origin(None, "grok_build/m", "grok_build/m"));
        assert!(should_strip_origin(None, "claude_code/m", "grok_build/m"));
        assert!(should_strip_origin(
            Some("grok_build/m"),
            "claude_code/m",
            "grok_build/m"
        ));
        assert!(!should_strip_origin(
            Some("grok_build/m"),
            "grok_build/m",
            "grok_build/m"
        ));
    }

    #[test]
    fn same_group_same_model_keeps_thinking_across_hosts() {
        let key = compatibility_key("grok_build", "grok-4.6");
        assert!(!should_strip_origin(Some(&key), &key, &key));
    }

    #[test]
    fn group_or_model_change_strips() {
        assert!(should_strip_origin(
            Some("grok_build/grok-4.6"),
            "grok_build/grok-3",
            "grok_build/grok-4.6"
        ));
        assert!(should_strip_origin(
            Some("grok_build/grok-4.6"),
            "codex/grok-4.6",
            "grok_build/grok-4.6"
        ));
    }

    #[test]
    fn explicit_group_overrides_profile() {
        assert_eq!(
            resolve_provider_group(Some("packy"), crate::llm::profiles::ClientProfile::GrokBuild),
            "packy"
        );
        assert_eq!(
            resolve_provider_group(None, crate::llm::profiles::ClientProfile::GrokBuild),
            "grok_build"
        );
        assert_eq!(
            resolve_provider_group(Some("  "), crate::llm::profiles::ClientProfile::Codex),
            "codex"
        );
    }
}
