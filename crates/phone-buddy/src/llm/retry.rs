//! Retry classification and backoff trimmed to the generic OpenAI-compatible path.
//!
//! Ported pure logic from grok `xai-grok-sampler/src/retry.rs` +
//! `xai-grok-sampling-types` veto helpers.
//!
//! Behavior summary:
//! - Retried: 429 and any 5xx except 525/526 (broken origin certs never
//!   clear on their own), connection errors, mid-stream failures, and empty
//!   responses.
//! - 429 additionally honors the server `Retry-After` but is capped by a
//!   small attempt budget ([`RATE_LIMIT_RETRY_THRESHOLD`]).
//! - Not retried (fatal immediately): 400/401/403/404/408/422, 525/526,
//!   `x-should-retry: false`, context-length overflow messages.

use std::time::Duration;

/// After this many rate-limit (429) retries, escalate to the caller.
pub const RATE_LIMIT_RETRY_THRESHOLD: u32 = 2;

/// Default retry budget when no override is set (grok sampler default).
/// Mobile `EngineConfig::max_retries` may still choose a lower value for
/// battery/latency; this constant is the upstream default.
pub const DEFAULT_MAX_RETRIES: u32 = 15;

/// Longest single wait on the generic retry path — the exponential-backoff
/// ceiling, and the clamp for a server `Retry-After`. Ported from grok.
pub const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(30);

/// Parse-level cap for `Retry-After` values (seconds). Ported from grok.
pub const RETRY_AFTER_PARSE_CAP_SECS: u64 = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    /// Retry with exponential backoff.
    Retry,
    /// Rate-limited: honor Retry-After, bounded attempt budget.
    RateLimited,
    /// Do not retry.
    Fatal,
}

/// True when an error message indicates a context-window overflow.
/// Ported from grok `is_context_length_error` — deterministic, never retry.
pub fn is_context_length_error(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("too long for this model")
        || m.contains("prompt is too long")
        || m.contains("maximum prompt length")
        || m.contains("maximum context length")
        || m.contains("context_length_exceeded")
        || (m.contains("current message") && m.contains("exceeds budget"))
}

/// Server `x-should-retry: false` veto (CCP header). Ported from grok.
pub fn is_retry_vetoed_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("x-should-retry=false")
        || lower.contains("x-should-retry: false")
        || is_context_length_error(message)
}

/// Classify an HTTP status for retry. Ported rule set from grok
/// (`RetryPolicy::edge_client` + 525/526 exclusion).
pub fn classify_status(status: u16) -> RetryClass {
    match status {
        429 => RetryClass::RateLimited,
        // Cloudflare origin TLS handshake / invalid cert: never clears.
        525 | 526 => RetryClass::Fatal,
        s if (500..600).contains(&s) => RetryClass::Retry,
        // Client errors
        400 | 401 | 403 | 404 | 408 | 422 => RetryClass::Fatal,
        _ => RetryClass::Fatal,
    }
}

/// Backoff for doom-loop resamples: near-immediate with small jitter.
/// Ported from grok `doom_loop_backoff` (0–250ms).
pub fn doom_loop_backoff(retry_count: u32) -> Duration {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static JITTER_SEQ: AtomicU64 = AtomicU64::new(0);

    let mut hasher = std::hash::DefaultHasher::new();
    JITTER_SEQ.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    retry_count.hash(&mut hasher);
    Duration::from_millis(hasher.finish() % 251)
}

/// Exponential backoff (2s, 4s, 8s, ..., capped at [`MAX_RETRY_BACKOFF`])
/// with ±20% jitter to prevent thundering-herd retry storms. Ported from
/// grok `retry_backoff_with_jitter`.
pub fn retry_backoff_with_jitter(retry_count: u32) -> Duration {
    let shift = retry_count.saturating_sub(1);
    let base_ms = 2000u64
        .checked_shl(shift)
        .unwrap_or(u64::MAX)
        .min(MAX_RETRY_BACKOFF.as_millis() as u64);
    jittered(Duration::from_millis(base_ms))
}

/// ±20% jitter around `base`, de-syncing clients that failed at the same
/// instant. Ported from grok.
fn jittered(base: Duration) -> Duration {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static JITTER_SEQ: AtomicU64 = AtomicU64::new(0);

    let base_ms = base.as_millis() as u64;
    let jitter_range = base_ms / 5;
    let mut hasher = std::hash::DefaultHasher::new();
    JITTER_SEQ.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let jitter = if jitter_range == 0 {
        0
    } else {
        hasher.finish() % (jitter_range * 2 + 1)
    };
    let delta = jitter as i64 - jitter_range as i64;
    Duration::from_millis(base_ms.saturating_add_signed(delta))
}

/// Clamp + parse a `Retry-After` header value (seconds form only; HTTP-date
/// form is ignored), capped at [`RETRY_AFTER_PARSE_CAP_SECS`]. Ported from
/// grok.
pub fn parse_retry_after(header_value: &str) -> Option<Duration> {
    let secs: u64 = header_value.trim().parse().ok()?;
    Some(Duration::from_secs(secs.min(RETRY_AFTER_PARSE_CAP_SECS)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_classification() {
        assert_eq!(classify_status(429), RetryClass::RateLimited);
        assert_eq!(classify_status(500), RetryClass::Retry);
        assert_eq!(classify_status(503), RetryClass::Retry);
        assert_eq!(classify_status(525), RetryClass::Fatal);
        assert_eq!(classify_status(526), RetryClass::Fatal);
        assert_eq!(classify_status(401), RetryClass::Fatal);
        assert_eq!(classify_status(400), RetryClass::Fatal);
        assert_eq!(classify_status(200), RetryClass::Fatal);
    }

    #[test]
    fn context_length_is_fatal_veto() {
        assert!(is_context_length_error(
            "This model's maximum context length is 128000 tokens"
        ));
        assert!(is_retry_vetoed_message(
            "status=400 maximum context length exceeded"
        ));
        assert!(is_retry_vetoed_message(
            "status=500 x-should-retry=false something"
        ));
        assert!(!is_retry_vetoed_message("status=503 temporary"));
    }

    #[test]
    fn backoff_is_capped() {
        for i in 1..20 {
            let d = retry_backoff_with_jitter(i);
            assert!(d <= MAX_RETRY_BACKOFF + MAX_RETRY_BACKOFF / 5);
        }
    }

    #[test]
    fn retry_after_capped() {
        assert_eq!(parse_retry_after("10"), Some(Duration::from_secs(10)));
        assert_eq!(
            parse_retry_after("9999"),
            Some(Duration::from_secs(RETRY_AFTER_PARSE_CAP_SECS))
        );
        assert_eq!(parse_retry_after("garbage"), None);
    }
}
