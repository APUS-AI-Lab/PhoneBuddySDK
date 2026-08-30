//! LLM layer: wire types, transports, streaming, retry, client.

pub mod client;
pub mod doom_loop_collector;
pub mod endpoint;
pub mod doom_loop_wire;
pub mod dumper;
pub mod failover;
pub mod host;
pub mod image;
pub mod profiles;
pub mod retry;
pub mod router;
pub mod stream;
pub mod transport;
pub mod types;
pub mod wire;

pub use client::{LlmClient, LlmTransportObj, LlmTurnSession};
pub use endpoint::{LlmEndpoint, LlmEndpointProvider, SharedLlmEndpointProvider};
pub use dumper::{HttpDumpConfig, HttpDumpMode, HttpDumper};
pub use host::{HostLlmHub, HostLlmNotify, HostLlmTransport};
pub use profiles::{
    build_profile_headers, get_profile_definition, render_user_agent, ClientProfile,
    ClientProfileDefinition,
};
pub use transport::{HttpTransport, LlmTurnContext, MockTransport, MockTurn};
pub use types::*;
