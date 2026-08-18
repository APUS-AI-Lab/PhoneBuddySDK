//! LLM layer: wire types, transports, streaming, retry, client.

pub mod client;
pub mod doom_loop_collector;
pub mod doom_loop_wire;
pub mod host;
pub mod retry;
pub mod stream;
pub mod transport;
pub mod types;

pub use client::{LlmClient, LlmTransportObj};
pub use host::{HostLlmHub, HostLlmNotify, HostLlmTransport};
pub use transport::{HttpTransport, MockTransport, MockTurn};
pub use types::*;
