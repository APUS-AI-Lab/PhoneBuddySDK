//! Retrying LLM client.
//!
//! Wraps an [`LlmTransport`] with the retry/backoff policy ported from the
//! grok sampler: exponential backoff with jitter for 5xx/connection errors,
//! `Retry-After`-honoring bounded retries for 429, and one retry for empty
//! responses. Retries only happen before any content has been streamed to
//! the observer, so the UI never sees duplicated deltas.

use std::sync::Arc;
use std::time::Duration;

use crate::config::EngineConfig;
use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver};
use crate::llm::doom_loop_wire::DoomLoopRecoveryPolicy;
use crate::llm::retry::{
    doom_loop_backoff, parse_retry_after, retry_backoff_with_jitter, RetryClass,
    RATE_LIMIT_RETRY_THRESHOLD,
};
use crate::llm::stream::collect_stream;
use crate::llm::transport::{retry_class_for_error, LlmTransport};
use crate::llm::types::{ChatCompletionRequest, CollectedTurn};

pub struct LlmClient {
    transport: Arc<dyn LlmTransportObj>,
    max_retries: u32,
    /// Independent budget for server doom-loop resamples (default 2).
    doom_loop_max_retries: u32,
}

/// Object-safe wrapper so the client can hold `Arc<dyn ...>`.
pub trait LlmTransportObj: Send + Sync {
    fn request_stream_boxed<'a>(
        &'a self,
        req: &'a ChatCompletionRequest,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>>;
    fn name(&self) -> &str;
}

impl<T: LlmTransport> LlmTransportObj for T {
    fn request_stream_boxed<'a>(
        &'a self,
        req: &'a ChatCompletionRequest,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<crate::llm::transport::ChunkStream>> {
        Box::pin(self.request_stream(req))
    }
    fn name(&self) -> &str {
        <Self as LlmTransport>::name(self)
    }
}

impl LlmClient {
    pub fn new(transport: Arc<dyn LlmTransportObj>, max_retries: u32) -> Self {
        Self {
            transport,
            max_retries,
            doom_loop_max_retries: DoomLoopRecoveryPolicy::DEFAULT_MAX_RETRIES,
        }
    }

    pub fn from_http(cfg: &EngineConfig) -> EngineResult<Self> {
        let t = crate::llm::transport::HttpTransport::new_with_doom_loop_and_extra_body(
            &cfg.base_url,
            &cfg.api_key,
            Duration::from_secs(cfg.stream_idle_timeout_secs),
            cfg.api_backend,
            cfg.extra_headers.clone(),
            cfg.extra_body.clone(),
            cfg.doom_loop_check_enabled(),
        )?;
        Ok(Self::new(Arc::new(t), cfg.max_retries))
    }

    /// Run one chat-completion request with retry, streaming deltas to
    /// `observer`, and return the fully collected turn.
    pub async fn complete(
        &self,
        req: &ChatCompletionRequest,
        observer: &dyn AgentObserver,
    ) -> EngineResult<CollectedTurn> {
        let mut attempt: u32 = 0;
        let mut rate_limit_retries: u32 = 0;
        let mut doom_loop_retries: u32 = 0;
        loop {
            attempt += 1;
            match self.transport.request_stream_boxed(req).await {
                Ok(stream) => {
                    match collect_stream(stream, observer).await {
                        Ok(turn) => {
                            if turn.is_empty() && attempt <= self.max_retries {
                                // EmptyResponse: model returned no content/tool
                                // calls. Retry (nothing was streamed).
                                tracing::warn!("empty LLM response; retry {attempt}");
                                let wait = retry_backoff_with_jitter(attempt);
                                tokio::time::sleep(wait).await;
                                continue;
                            }
                            if turn.is_empty() {
                                return Err(EngineError::EmptyResponse);
                            }
                            return Ok(turn);
                        }
                        Err(EngineError::DoomLoopServer(ref triggers)) => {
                            // Server thinking-loop: resample with independent budget.
                            if doom_loop_retries < self.doom_loop_max_retries {
                                doom_loop_retries += 1;
                                let wait = doom_loop_backoff(doom_loop_retries);
                                tracing::warn!(
                                    "server doom-loop ({triggers}); resample {doom_loop_retries}/{} after {wait:?}",
                                    self.doom_loop_max_retries
                                );
                                tokio::time::sleep(wait).await;
                                continue;
                            }
                            // Budget spent: surface as hard error (caller may
                            // still present partial UI from streamed deltas).
                            return Err(EngineError::DoomLoopServer(triggers.clone()));
                        }
                        Err(e) => {
                            // Other mid-stream failures: do not retry.
                            return Err(e);
                        }
                    }
                }
                Err(e) => {
                    let class = retry_class_for_error(&e);
                    match class {
                        RetryClass::Fatal => return Err(e),
                        RetryClass::RateLimited => {
                            rate_limit_retries += 1;
                            if rate_limit_retries > RATE_LIMIT_RETRY_THRESHOLD
                                || attempt > self.max_retries
                            {
                                return Err(e);
                            }
                            // Honor Retry-After when present; bounded budget.
                            let wait = retry_after_from_error(&e)
                                .unwrap_or_else(|| retry_backoff_with_jitter(attempt));
                            tracing::warn!("rate limited (429); waiting {wait:?}");
                            tokio::time::sleep(wait).await;
                        }
                        RetryClass::Retry => {
                            if attempt > self.max_retries {
                                return Err(e);
                            }
                            let wait = retry_backoff_with_jitter(attempt);
                            tracing::warn!("LLM request error: {e}; retry in {wait:?}");
                            tokio::time::sleep(wait).await;
                        }
                    }
                }
            }
        }
    }
}

/// Best-effort extraction of a `Retry-After` hint embedded by transports in
/// the error message (`status=429 retry-after=<secs>`).
fn retry_after_from_error(err: &EngineError) -> Option<Duration> {
    let EngineError::Llm(msg) = err else {
        return None;
    };
    let marker = "retry-after=";
    let idx = msg.find(marker)?;
    let rest = &msg[idx + marker.len()..];
    let secs: u64 = rest.split_whitespace().next()?.parse().ok()?;
    parse_retry_after(&secs.to_string())
}

#[allow(dead_code)]
fn _unused_event_ref(_: &AgentEvent) {}
