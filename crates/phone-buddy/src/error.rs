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
}

pub type EngineResult<T> = Result<T, EngineError>;

