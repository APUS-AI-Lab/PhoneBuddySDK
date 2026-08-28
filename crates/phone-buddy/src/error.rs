//! Engine errors.

use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("LLM request failed: {0}")]
    Llm(String),

    #[error("LLM stream interrupted: {0}")]
    Stream(String),

    #[error("LLM stream interrupted: idle timeout after {0:?} with no SSE events")]
    StreamIdleTimeout(Duration),

    #[error("LLM returned an empty response (no text or tool calls)")]
    EmptyResponse,

    #[error("tool '{name}' failed: {message}")]
    Tool { name: String, message: String },

    #[error("tool '{0}' was not found")]
    ToolNotFound(String),

    #[error("invalid tool arguments for '{name}': {message}")]
    ToolArgs { name: String, message: String },

    #[error("path '{0}' escapes the sandbox root")]
    SandboxEscape(String),

    #[error("session '{0}' not found")]
    SessionNotFound(String),

    #[error("cancelled")]
    Cancelled,

    #[error("agent stopped after {0} turns without a final answer")]
    MaxTurnsReached(u32),

    #[error("doom loop detected: the same tool call repeated {0} times")]
    DoomLoop(u32),

    /// Server-reported generation doom-loop (Responses API). Pure signal for
    /// the client retry layer to resample; not a hard user-facing failure
    /// until the recovery budget is spent.
    #[error("server doom-loop signal: {0}")]
    DoomLoopServer(String),

    #[error("script execution failed: {0}")]
    Script(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("invalid user turn: {0}")]
    InvalidUserTurn(String),

    #[error("attachment '{0}' is missing or evicted")]
    AttachmentMissing(String),

    #[error("attachment '{0}' is invalid: {1}")]
    AttachmentInvalid(String, String),

    #[error("too many images: {0} (max 5)")]
    TooManyImages(usize),

    #[error("too many audio attachments: {0} (max 5)")]
    TooManyAudio(usize),

    #[error("current model does not support image input")]
    VisionUnsupported,

    #[error("current model does not support audio input")]
    AudioUnsupported,

    #[error("inline request payload too large")]
    PayloadTooLarge,

    #[error("invalid routing configuration: {0}")]
    InvalidRoutingConfig(String),

    #[error("route not configured: pool '{pool_id}'")]
    RouteNotConfigured { pool_id: String },

    #[error("pool '{pool_id}' exhausted (retry-after {retry_after_ms}ms)")]
    PoolExhausted {
        pool_id: String,
        retry_after_ms: u64,
    },

    #[error(
        "provider attempts exhausted in pool '{pool_id}' (tried: {})",
        tried_provider_ids.join(", ")
    )]
    ProviderAttemptsExhausted {
        pool_id: String,
        tried_provider_ids: Vec<String>,
    },

    #[error("operation timed out")]
    OperationTimedOut,

    #[error("operation cancelled")]
    OperationCancelled,
}

impl EngineError {
    /// Stable kind name for FFI / wrapper matching. Does not include secrets.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Config(_) => "Config",
            Self::Llm(_) => "Llm",
            Self::Stream(_) => "Stream",
            Self::StreamIdleTimeout(_) => "StreamIdleTimeout",
            Self::EmptyResponse => "EmptyResponse",
            Self::Tool { .. } => "Tool",
            Self::ToolNotFound(_) => "ToolNotFound",
            Self::ToolArgs { .. } => "ToolArgs",
            Self::SandboxEscape(_) => "SandboxEscape",
            Self::SessionNotFound(_) => "SessionNotFound",
            Self::Cancelled => "Cancelled",
            Self::MaxTurnsReached(_) => "MaxTurnsReached",
            Self::DoomLoop(_) => "DoomLoop",
            Self::DoomLoopServer(_) => "DoomLoopServer",
            Self::Script(_) => "Script",
            Self::Io(_) => "Io",
            Self::Serde(_) => "Serde",
            Self::InvalidUserTurn(_) => "InvalidUserTurn",
            Self::AttachmentMissing(_) => "AttachmentMissing",
            Self::AttachmentInvalid(_, _) => "AttachmentInvalid",
            Self::TooManyImages(_) => "TooManyImages",
            Self::TooManyAudio(_) => "TooManyAudio",
            Self::VisionUnsupported => "VisionUnsupported",
            Self::AudioUnsupported => "AudioUnsupported",
            Self::PayloadTooLarge => "PayloadTooLarge",
            Self::InvalidRoutingConfig(_) => "InvalidRoutingConfig",
            Self::RouteNotConfigured { .. } => "RouteNotConfigured",
            Self::PoolExhausted { .. } => "PoolExhausted",
            Self::ProviderAttemptsExhausted { .. } => "ProviderAttemptsExhausted",
            Self::OperationTimedOut => "OperationTimedOut",
            Self::OperationCancelled => "OperationCancelled",
        }
    }

    /// Extra JSON fields for a versioned one-shot error envelope. Never includes API keys.
    pub fn envelope_fields(&self) -> serde_json::Value {
        match self {
            Self::RouteNotConfigured { pool_id } => serde_json::json!({ "pool_id": pool_id }),
            Self::PoolExhausted {
                pool_id,
                retry_after_ms,
            } => serde_json::json!({
                "pool_id": pool_id,
                "retry_after_ms": retry_after_ms,
            }),
            Self::ProviderAttemptsExhausted {
                pool_id,
                tried_provider_ids,
            } => serde_json::json!({
                "pool_id": pool_id,
                "tried_provider_ids": tried_provider_ids,
            }),
            _ => serde_json::json!({}),
        }
    }
}

pub type EngineResult<T> = Result<T, EngineError>;
