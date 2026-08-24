//! LLM layer: wire types, transports, streaming, retry, client.

pub mod client;
pub mod doom_loop_collector;
pub mod doom_loop_wire;
pub mod dumper;
pub mod image;
pub mod failover;
pub mod host;
pub mod profiles;
pub mod retry;
pub mod stream;
pub mod transport;
pub mod types;
pub mod wire;

pub use client::{LlmClient, LlmTransportObj};
pub use dumper::{HttpDumpConfig, HttpDumpMode, HttpDumper};
pub use host::{HostLlmHub, HostLlmNotify, HostLlmTransport};
pub use profiles::{
    build_profile_headers, get_profile_definition, render_user_agent, ClientProfile,
    ClientProfileDefinition,
};
pub use transport::{HttpTransport, MockTransport, MockTurn};
pub use types::*;

