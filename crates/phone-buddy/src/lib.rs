//! # PhoneBuddy SDK
//!
//! Core LLM Agent engine library for mobile:
//! LLM calls (OpenAI-compatible + SSE streaming), a tool loop (plan/search/execute/report),
//! file tools, built-in busybox applets, subagent task execution, and an embedded JS scripting engine.
//!
//! Design constraints (for iOS / Android cross-compilation):
//! - No fork / exec: every "command" runs in-process (busybox applets + JS engine);
//! - TLS via rustls + ring; CA via webpki-roots (reqwest `rustls-tls`);
//! - No jemalloc / git2 / rusqlite / tree-sitter / gcloud-storage / tonic;
//! - All file operations are confined to the `EngineConfig::root_dir` sandbox.
//!
//! ## Quick start
//!
//! ```no_run
//! use phone_buddy::prelude::*;
//!
//! let cfg = EngineConfig {
//!     api_key: "xai-...".into(),
//!     base_url: "https://api.x.ai/v1".into(),
//!     model: "grok-3".into(),
//!     root_dir: std::env::temp_dir().join("phone-buddy-demo"),
//!     ..Default::default()
//! };
//! let engine = PhoneBuddyEngine::new(cfg).unwrap();
//! let events = std::sync::Arc::new(RecordingObserver::new());
//! let outcome = engine
//!     .chat("session-1", "list the files in the working directory", Some(events.clone()))
//!     .unwrap();
//! println!("{}", outcome.final_text);
//! ```

pub mod agent;
pub mod config;
pub mod conversation;
pub mod diag;
pub mod engine;
pub mod error;
pub mod events;
pub mod llm;
pub mod prompt;
pub mod runtime;
pub mod session;
pub mod tools;

pub mod prelude {
    pub use crate::config::{
        ClientProfile, ClientProfileDefinition, EngineConfig, EngineConfigBuilder,
        ProviderEndpoint, DEFAULT_AGENT_NAME,
    };
    pub use crate::engine::{ChatOutcome, PhoneBuddyEngine};
    pub use crate::error::{EngineError, EngineResult};
    pub use crate::events::{
        AgentEvent, AgentObserver, NullObserver, RecordingObserver, UsageSummary,
    };
    pub use crate::llm::router::{
        ExhaustionPolicy, LlmRoutingConfig, PoolMember, ProviderPool, ProviderTarget, RetryPolicy,
        RouterHealthConfig, Workload, MAIN_POOL_ID, SUBAGENT_POOL_ID,
    };
    pub use crate::llm::types::{ApiBackend, ReasoningEffort, ResponseFormat};
    pub use crate::runtime::{GenerateTextRequest, GenerateTextResult, PhoneBuddyRuntime};
    pub use crate::session::SessionMeta;
}

pub use prelude::*;

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
