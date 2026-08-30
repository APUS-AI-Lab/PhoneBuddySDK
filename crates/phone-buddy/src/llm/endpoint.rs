//! Runtime LLM credentials for tool HTTP fallbacks.
//!
//! Analog of grok-build `SharedApiKeyProvider`, extended to the selected
//! pool member's URL and backend: PhoneBuddy routes those per provider, so
//! a key-only provider would still hit the wrong host after failover.

use std::collections::HashMap;
use std::sync::Arc;

use crate::llm::router::ProviderTarget;
use crate::llm::types::ApiBackend;

/// Credentials and wire settings of one routable provider.
#[derive(Debug, Clone)]
pub struct LlmEndpoint {
    pub provider_id: String,
    pub api_key: String,
    pub base_url: String,
    pub api_backend: ApiBackend,
    pub model: String,
    pub extra_headers: HashMap<String, String>,
    pub extra_body: HashMap<String, serde_json::Value>,
}

impl LlmEndpoint {
    pub fn from_target(target: &ProviderTarget) -> Self {
        Self {
            provider_id: target.provider_id.clone(),
            api_key: target.api_key.clone(),
            base_url: target.base_url.clone(),
            api_backend: target.api_backend,
            model: target.model.clone(),
            extra_headers: target.extra_headers.clone(),
            extra_body: target.extra_body.clone(),
        }
    }
}

/// Fresh read of the currently selected pool member.
///
/// `None` means this client has no HTTP routing credentials (injected or
/// host transports). Callers then fall back to a static snapshot / env.
pub trait LlmEndpointProvider: Send + Sync {
    fn current_endpoint(&self) -> Option<LlmEndpoint>;
}

pub type SharedLlmEndpointProvider = Arc<dyn LlmEndpointProvider>;
