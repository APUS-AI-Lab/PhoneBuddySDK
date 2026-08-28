//! Deterministic group/score provider selection.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};

use super::config::{ExhaustionPolicy, PoolMember, ProviderPool, RetryPolicy, RouterHealthConfig};
use super::health::ProviderHealthRecord;

/// Ordered visit plan for one operation against one pool.
#[derive(Debug, Clone)]
pub struct VisitPlan {
    pub pool_id: String,
    pub provider_ids: Vec<String>,
    pub chain_mode: bool,
    /// Router config generation this plan was computed against.
    pub generation: u64,
    pub retry: RetryPolicy,
}

#[derive(Debug, Clone)]
pub enum SelectError {
    FailFast { retry_after_ms: u64 },
}

#[derive(Clone, Copy)]
struct ScoredMember<'a> {
    member: &'a PoolMember,
    score: i32,
    cooling: bool,
    cooldown_until: Option<DateTime<Utc>>,
}

/// Compute the visit order for `pool` at `now`.
///
/// Eligible members (`enabled`, `effective_score > 0`, not cooling) are
/// ranked by routing-group max score, then declared `order`. Remaining
/// enabled members are appended for `probe_earliest` so a request still
/// has somewhere to go when every preferred member is suppressed.
pub fn select_visit_order(
    pool_id: &str,
    pool: &ProviderPool,
    health: &HashMap<String, ProviderHealthRecord>,
    health_cfg: &RouterHealthConfig,
    now: DateTime<Utc>,
) -> Result<VisitPlan, SelectError> {
    let window = Duration::from_secs(health_cfg.penalty_window_secs.max(1));
    let mut scored: Vec<ScoredMember<'_>> = pool
        .members
        .iter()
        .filter(|m| m.enabled)
        .map(|member| {
            let rec = health.get(&member.provider_id);
            let score = rec
                .map(|r| r.effective_score(member.base_score, now, window))
                .unwrap_or(member.base_score);
            let cooling = rec.map(|r| r.is_cooling(now)).unwrap_or(false);
            let cooldown_until = rec.and_then(|r| r.cooldown_until);
            ScoredMember {
                member,
                score,
                cooling,
                cooldown_until,
            }
        })
        .collect();

    let mut eligible: Vec<ScoredMember<'_>> = scored
        .iter()
        .filter(|s| s.score > 0 && !s.cooling)
        .cloned()
        .collect();

    let mut provider_ids = rank_eligible(&mut eligible);

    if provider_ids.is_empty() {
        match pool.when_exhausted {
            ExhaustionPolicy::FailFast => {
                let retry_after_ms = scored
                    .iter()
                    .filter_map(|s| s.cooldown_until)
                    .filter(|t| *t > now)
                    .min()
                    .and_then(|t| (t - now).to_std().ok())
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                return Err(SelectError::FailFast { retry_after_ms });
            }
            ExhaustionPolicy::ProbeEarliest => {
                provider_ids = rank_probe_earliest(&mut scored, now);
            }
        }
    } else if pool.when_exhausted == ExhaustionPolicy::ProbeEarliest {
        // After the preferred sequence, still allow remaining enabled
        // members so an in-operation failover can reach a cooling backup.
        let preferred: std::collections::HashSet<&str> =
            provider_ids.iter().map(|s| s.as_str()).collect();
        let mut rest: Vec<ScoredMember<'_>> = scored
            .iter()
            .filter(|s| !preferred.contains(s.member.provider_id.as_str()))
            .cloned()
            .collect();
        provider_ids.extend(rank_probe_earliest(&mut rest, now));
    }

    Ok(VisitPlan {
        pool_id: pool_id.to_string(),
        chain_mode: provider_ids.len() > 1,
        provider_ids,
        generation: 0,
        retry: pool.retry.clone(),
    })
}

fn rank_eligible(members: &mut [ScoredMember<'_>]) -> Vec<String> {
    let mut groups: HashMap<&str, (i32, u32)> = HashMap::new();
    for s in members.iter() {
        let entry = groups
            .entry(s.member.routing_group.as_str())
            .or_insert((s.score, s.member.order));
        if s.score > entry.0 {
            entry.0 = s.score;
        }
        if s.member.order < entry.1 {
            entry.1 = s.member.order;
        }
    }
    let mut group_rank: Vec<(&str, i32, u32)> = groups
        .into_iter()
        .map(|(g, (score, order))| (g, score, order))
        .collect();
    group_rank.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(b.0)));

    let mut out = Vec::with_capacity(members.len());
    for (group, _, _) in group_rank {
        let mut in_group: Vec<&ScoredMember<'_>> = members
            .iter()
            .filter(|s| s.member.routing_group == group)
            .collect();
        in_group.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then(a.member.order.cmp(&b.member.order))
                .then(a.member.provider_id.cmp(&b.member.provider_id))
        });
        out.extend(in_group.into_iter().map(|s| s.member.provider_id.clone()));
    }
    out
}

fn rank_probe_earliest(members: &mut [ScoredMember<'_>], now: DateTime<Utc>) -> Vec<String> {
    members.sort_by(|a, b| {
        let a_exp = a.cooldown_until.unwrap_or(now);
        let b_exp = b.cooldown_until.unwrap_or(now);
        a_exp
            .cmp(&b_exp)
            .then(b.score.cmp(&a.score))
            .then(a.member.order.cmp(&b.member.order))
            .then(a.member.provider_id.cmp(&b.member.provider_id))
    });
    members
        .iter()
        .map(|s| s.member.provider_id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::router::config::{ExhaustionPolicy, RetryPolicy};
    use chrono::TimeZone;

    fn member(id: &str, group: &str, score: i32, order: u32) -> PoolMember {
        PoolMember {
            provider_id: id.into(),
            routing_group: group.into(),
            base_score: score,
            order,
            enabled: true,
        }
    }

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, 12, 0, 0).unwrap()
    }

    fn cfg() -> RouterHealthConfig {
        RouterHealthConfig::default()
    }

    #[test]
    fn group_rank_uses_max_member_score_then_order() {
        let pool = ProviderPool {
            members: vec![
                member("cheap-a", "cheap", 5, 0),
                member("cheap-b", "cheap", 9, 2),
                member("pref-a", "preferred", 8, 1),
                member("pref-b", "preferred", 8, 3),
            ],
            retry: RetryPolicy::default(),
            when_exhausted: ExhaustionPolicy::ProbeEarliest,
        };
        let plan = select_visit_order("main", &pool, &HashMap::new(), &cfg(), t0()).unwrap();
        // cheap max=9 > preferred max=8, so cheap group first; within cheap, b (9) then a (5).
        assert_eq!(
            plan.provider_ids,
            vec!["cheap-b", "cheap-a", "pref-a", "pref-b"]
        );
    }

    #[test]
    fn group_tie_breaks_by_lowest_declared_order() {
        let pool = ProviderPool {
            members: vec![
                member("g2-a", "g2", 10, 5),
                member("g1-a", "g1", 10, 1),
                member("g1-b", "g1", 7, 2),
            ],
            retry: RetryPolicy::default(),
            when_exhausted: ExhaustionPolicy::ProbeEarliest,
        };
        let plan = select_visit_order("main", &pool, &HashMap::new(), &cfg(), t0()).unwrap();
        assert_eq!(plan.provider_ids[0], "g1-a");
        assert_eq!(plan.provider_ids[1], "g1-b");
        assert_eq!(plan.provider_ids[2], "g2-a");
    }

    #[test]
    fn within_group_score_then_order() {
        let pool = ProviderPool {
            members: vec![
                member("a", "g", 5, 0),
                member("b", "g", 9, 2),
                member("c", "g", 9, 1),
            ],
            retry: RetryPolicy::default(),
            when_exhausted: ExhaustionPolicy::ProbeEarliest,
        };
        let plan = select_visit_order("main", &pool, &HashMap::new(), &cfg(), t0()).unwrap();
        assert_eq!(plan.provider_ids, vec!["c", "b", "a"]);
    }

    #[test]
    fn fail_fast_when_all_cooling() {
        let pool = ProviderPool {
            members: vec![member("a", "g", 10, 0)],
            retry: RetryPolicy::default(),
            when_exhausted: ExhaustionPolicy::FailFast,
        };
        let mut health = HashMap::new();
        health.insert(
            "a".into(),
            ProviderHealthRecord {
                cooldown_until: Some(t0() + chrono::Duration::seconds(30)),
                consecutive_trips: 1,
                ..Default::default()
            },
        );
        match select_visit_order("title", &pool, &health, &cfg(), t0()) {
            Err(SelectError::FailFast { retry_after_ms }) => {
                assert_eq!(retry_after_ms, 30_000);
            }
            other => panic!("expected fail_fast, got {other:?}"),
        }
    }

    #[test]
    fn probe_earliest_picks_soonest_cooldown() {
        let pool = ProviderPool {
            members: vec![member("a", "g", 10, 0), member("b", "g", 8, 1)],
            retry: RetryPolicy::default(),
            when_exhausted: ExhaustionPolicy::ProbeEarliest,
        };
        let mut health = HashMap::new();
        health.insert(
            "a".into(),
            ProviderHealthRecord {
                cooldown_until: Some(t0() + chrono::Duration::seconds(120)),
                consecutive_trips: 1,
                ..Default::default()
            },
        );
        health.insert(
            "b".into(),
            ProviderHealthRecord {
                cooldown_until: Some(t0() + chrono::Duration::seconds(40)),
                consecutive_trips: 1,
                ..Default::default()
            },
        );
        let plan = select_visit_order("main", &pool, &health, &cfg(), t0()).unwrap();
        assert_eq!(plan.provider_ids[0], "b");
        assert_eq!(plan.provider_ids[1], "a");
    }

    #[test]
    fn disabled_members_are_skipped() {
        let mut disabled = member("dead", "g", 10, 0);
        disabled.enabled = false;
        let pool = ProviderPool {
            members: vec![disabled, member("live", "g", 5, 1)],
            retry: RetryPolicy::default(),
            when_exhausted: ExhaustionPolicy::ProbeEarliest,
        };
        let plan = select_visit_order("main", &pool, &HashMap::new(), &cfg(), t0()).unwrap();
        assert_eq!(plan.provider_ids, vec!["live"]);
        assert!(
            !plan.chain_mode,
            "one enabled member must use the single-provider retry budget"
        );
    }

    #[test]
    fn probe_earliest_appends_cooling_after_healthy() {
        let pool = ProviderPool {
            members: vec![member("a", "g", 10, 0), member("b", "g", 8, 1)],
            retry: RetryPolicy::default(),
            when_exhausted: ExhaustionPolicy::ProbeEarliest,
        };
        let mut health = HashMap::new();
        health.insert(
            "b".into(),
            ProviderHealthRecord {
                cooldown_until: Some(t0() + chrono::Duration::seconds(120)),
                consecutive_trips: 1,
                ..Default::default()
            },
        );
        let plan = select_visit_order("main", &pool, &health, &cfg(), t0()).unwrap();
        assert_eq!(plan.provider_ids, vec!["a", "b"]);
        assert!(plan.chain_mode);
    }

    #[test]
    fn fail_fast_does_not_append_cooling_when_healthy_exists() {
        let pool = ProviderPool {
            members: vec![member("a", "g", 10, 0), member("b", "g", 8, 1)],
            retry: RetryPolicy::default(),
            when_exhausted: ExhaustionPolicy::FailFast,
        };
        let mut health = HashMap::new();
        health.insert(
            "b".into(),
            ProviderHealthRecord {
                cooldown_until: Some(t0() + chrono::Duration::seconds(120)),
                consecutive_trips: 1,
                ..Default::default()
            },
        );
        let plan = select_visit_order("title", &pool, &health, &cfg(), t0()).unwrap();
        assert_eq!(plan.provider_ids, vec!["a"]);
        assert!(!plan.chain_mode);
    }
}
