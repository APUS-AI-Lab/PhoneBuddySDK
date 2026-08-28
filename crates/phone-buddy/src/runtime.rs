//! Long-lived process handle owning the LLM router.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::EngineConfig;
use crate::engine::PhoneBuddyEngine;
use crate::error::{EngineError, EngineResult};
use crate::llm::router::{synthesize_legacy_routing, LlmRouter, LlmRoutingConfig};

/// Process-scoped routing owner. Longer-lived than a single
/// [`PhoneBuddyEngine`]. Shared interior mutability is not exposed via
/// [`Clone`]; callers hold [`Arc`].
pub struct PhoneBuddyRuntime {
    router: Arc<LlmRouter>,
    root_dir: PathBuf,
}

impl PhoneBuddyRuntime {
    pub fn new(
        routing_config: LlmRoutingConfig,
        root_dir: impl Into<PathBuf>,
    ) -> EngineResult<Arc<Self>> {
        let root_dir = root_dir.into();
        let router = LlmRouter::persist(routing_config, root_dir.clone())?;
        Ok(Arc::new(Self { router, root_dir }))
    }

    /// Compatibility constructor: synthesize `main` + `subagent` pools from
    /// the historic primary + `fallback_providers` chain.
    pub fn from_engine_config(config: &EngineConfig) -> EngineResult<Arc<Self>> {
        let routing =
            synthesize_legacy_routing(config).map_err(EngineError::InvalidRoutingConfig)?;
        Self::new(routing, config.root_dir.clone())
    }

    /// Replace routing. In-flight operations may finish on a previously
    /// captured visit plan. Health is reconciled by stable `provider_id`.
    pub fn update_routing(&self, new_config: LlmRoutingConfig) -> EngineResult<()> {
        self.router.update_config(new_config)
    }

    pub fn router(&self) -> Arc<LlmRouter> {
        self.router.clone()
    }

    pub fn root_dir(&self) -> &std::path::Path {
        &self.root_dir
    }

    pub fn create_engine(
        self: &Arc<Self>,
        agent_config: EngineConfig,
        main_pool_id: &str,
    ) -> EngineResult<Arc<PhoneBuddyEngine>> {
        PhoneBuddyEngine::from_runtime(self.clone(), agent_config, main_pool_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::router::{
        FailureClass, PoolMember, ProviderPool, ProviderTarget, MAIN_POOL_ID,
    };
    use chrono::{TimeZone, Utc};

    fn target(id: &str) -> ProviderTarget {
        ProviderTarget {
            provider_id: id.into(),
            base_url: "https://api.example.com/v1".into(),
            api_key: "k".into(),
            model: "m".into(),
            api_backend: Default::default(),
            client_profile: Default::default(),
            client_version: None,
            client_session_id: None,
            reasoning_compatibility_key: None,
            capabilities: Default::default(),
            extra_headers: Default::default(),
            extra_body: Default::default(),
            enable_web_search: false,
            web_search_options: None,
            enable_x_search: false,
            x_search_options: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn engine_recreation_keeps_health() {
        let dir = tempfile::tempdir().unwrap();
        let mut pools = std::collections::BTreeMap::new();
        pools.insert(
            MAIN_POOL_ID.into(),
            ProviderPool {
                members: vec![PoolMember {
                    provider_id: "p1".into(),
                    routing_group: "g".into(),
                    base_score: 10,
                    order: 0,
                    enabled: true,
                }],
                ..Default::default()
            },
        );
        let routing = LlmRoutingConfig {
            providers: vec![target("p1")],
            pools,
            health: Default::default(),
        };
        let runtime = PhoneBuddyRuntime::new(routing, dir.path()).unwrap();
        let t = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();
        runtime
            .router()
            .record_trip_at("op1", "p1", FailureClass::RetryableHttp, None, t);

        let mut cfg = EngineConfig::default();
        cfg.api_key = "k".into();
        cfg.root_dir = dir.path().to_path_buf();
        let engine = runtime.create_engine(cfg.clone(), MAIN_POOL_ID).unwrap();
        drop(engine);
        let _engine2 = runtime.create_engine(cfg, MAIN_POOL_ID).unwrap();
        assert!(runtime.router().health_record("p1").unwrap().is_cooling(t));
    }
}
