//! Events streamed to the host UI while a turn runs.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::llm::types::Usage;

/// Everything the UI may want to render while the agent works.
/// Delivered through [`AgentObserver`] (FFI maps this to a callback).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    /// Assistant text token delta.
    TextDelta { text: String },
    /// Reasoning/thinking delta (models that stream it).
    ReasoningDelta { text: String },
    /// A tool call is about to execute.
    ToolCallStart {
        call_id: String,
        name: String,
        arguments_json: String,
    },
    /// A tool call finished.
    ToolCallResult {
        call_id: String,
        name: String,
        ok: bool,
        output: String,
    },
    /// The planning tool updated the visible plan.
    PlanUpdated { items_json: String },
    /// Turn finished successfully.
    Completed {
        final_text: String,
        usage: Option<UsageSummary>,
    },
    /// Turn failed.
    Failed { message: String },
    /// The current provider failed and the engine is waiting before retrying
    /// the same provider. `provider` is a desensitized host/model id.
    Retrying {
        provider: String,
        attempt: u32,
        max_attempts: u32,
        wait_ms: u64,
        reason: String,
    },
    /// The current provider was marked degraded and the engine switched to
    /// the next endpoint in the fallback chain. `from` / `to` are
    /// desensitized host/model ids; they never carry an API key.
    ProviderSwitched {
        from: String,
        to: String,
        reason: String,
        cooldown_ms: u64,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct UsageSummary {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<Usage> for UsageSummary {
    fn from(u: Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

/// Sink for [`AgentEvent`]s. Implementations must be cheap: they run on the
/// engine's worker thread.
pub trait AgentObserver: Send + Sync {
    fn on_event(&self, event: AgentEvent);
}

/// An observer that discards everything (used by tests / fire-and-forget).
pub struct NullObserver;

impl AgentObserver for NullObserver {
    fn on_event(&self, _event: AgentEvent) {}
}

/// Collects events into a Vec (tests, CLI pretty-printer).
#[derive(Default)]
pub struct RecordingObserver {
    events: Mutex<Vec<AgentEvent>>,
}

impl RecordingObserver {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn snapshot(&self) -> Vec<AgentEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl AgentObserver for RecordingObserver {
    fn on_event(&self, event: AgentEvent) {
        self.events.lock().unwrap().push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrying_and_switched_serialize_externally_tagged() {
        let retrying = AgentEvent::Retrying {
            provider: "cf.api.fan/grok-4.6".into(),
            attempt: 2,
            max_attempts: 3,
            wait_ms: 2000,
            reason: "status=503".into(),
        };
        let json = serde_json::to_string(&retrying).unwrap();
        assert!(json.contains("\"Retrying\""));
        assert!(json.contains("cf.api.fan/grok-4.6"));
        assert!(!json.to_ascii_lowercase().contains("sk-"));

        let switched = AgentEvent::ProviderSwitched {
            from: "cf.api.fan/grok-4.6".into(),
            to: "api.openai.com/gpt-5.6".into(),
            reason: "status=503".into(),
            cooldown_ms: 120_000,
        };
        let json = serde_json::to_string(&switched).unwrap();
        assert!(json.contains("\"ProviderSwitched\""));
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::ProviderSwitched { cooldown_ms, .. } => {
                assert_eq!(cooldown_ms, 120_000);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
