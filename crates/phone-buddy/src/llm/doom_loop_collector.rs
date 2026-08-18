//! Per-request collector for server-reported doom-loop signals.
//!
//! Ported from grok-build `xai-grok-sampler/src/doom_loop.rs` (simplified:
//! no mid-stream abort of typed parsers beyond swallowing check events).

use std::sync::{Arc, Mutex};

use crate::llm::doom_loop_wire::{
    DOOM_LOOP_CHECK_EVENT_TYPE, DoomLoopPeek, DoomLoopRecoveryPolicy, DoomLoopSignal, peek_doom_loop,
};

/// Cheap-to-clone accumulator shared between the SSE decode loop and the
/// client retry layer for one request attempt.
#[derive(Clone, Debug, Default)]
pub struct DoomLoopSignalCollector {
    inner: Arc<Mutex<CollectorState>>,
}

#[derive(Debug, Default)]
struct CollectorState {
    signals: Vec<DoomLoopSignal>,
    malformed_logged: bool,
    policy: DoomLoopRecoveryPolicy,
    abort_disarmed: bool,
}

impl DoomLoopSignalCollector {
    pub fn new(policy: DoomLoopRecoveryPolicy) -> Self {
        let collector = Self::default();
        if let Ok(mut state) = collector.inner.lock() {
            state.policy = policy;
        }
        collector
    }

    pub fn disarm_abort(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.abort_disarmed = true;
        }
    }

    /// Confident trigger labels, if armed and any exist.
    pub fn abort_triggers(&self) -> Option<Vec<String>> {
        let state = self.inner.lock().ok()?;
        if state.abort_disarmed {
            return None;
        }
        let confident = state.policy.confident_triggers(&state.signals);
        (!confident.is_empty()).then_some(confident)
    }

    /// Inspect a raw SSE frame. Returns `true` when the frame is the
    /// non-standard check event that the caller must swallow.
    pub fn absorb(&self, event_name: &str, data: &str) -> bool {
        let named = event_name == DOOM_LOOP_CHECK_EVENT_TYPE;
        let (signals, swallow) = match peek_doom_loop(data) {
            DoomLoopPeek::CheckEvent(signals) => (signals, true),
            DoomLoopPeek::ResponseField(signals) => (signals, false),
            DoomLoopPeek::None => {
                if named {
                    self.log_malformed_once();
                }
                return named;
            }
        };
        if signals.is_empty() {
            self.log_malformed_once();
        } else {
            self.record(signals);
        }
        swallow || named
    }

    pub fn take(&self) -> Vec<DoomLoopSignal> {
        match self.inner.lock() {
            Ok(mut state) => std::mem::take(&mut state.signals),
            Err(_) => Vec::new(),
        }
    }

    fn record(&self, signals: Vec<DoomLoopSignal>) {
        let Ok(mut state) = self.inner.lock() else {
            return;
        };
        for signal in signals {
            if !state.signals.iter().any(|s| s.raw == signal.raw) {
                state.signals.push(signal);
            }
        }
    }

    fn log_malformed_once(&self) {
        let Ok(mut state) = self.inner.lock() else {
            return;
        };
        if !state.malformed_logged {
            state.malformed_logged = true;
            tracing::debug!("doom-loop check payload malformed or empty; ignoring");
        }
    }
}
