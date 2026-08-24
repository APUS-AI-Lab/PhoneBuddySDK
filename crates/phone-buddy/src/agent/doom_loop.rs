//! Action-stationarity (identical tool-call) detection.
//!
//! Ported from grok-build shell turn loop
//! (`xai-grok-shell/.../acp_session_impl/turn.rs` — `IdenticalToolCallRun`).
//!
//! This is **not** the server-side reasoning-channel doom-loop wire protocol
//! (`xai-grok-sampling-types::doom_loop` / sampler SSE collector). That path
//! needs the Responses API + `x-grok-doom-loop-check` and is orthogonal.
//!
//! Client-side stationarity: hash each model step's tool-call batch
//! (`name\u{1f}args` joined by `\u{1e}`). Escalate consecutive identical
//! steps:
//! - at [`NUDGE_AFTER_IDENTICAL_TOOL_CALLS`] → inject a one-shot nudge;
//! - at [`MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS`] → abort with
//!   `EngineError::DoomLoop`.
//!
//! Mobile deliberately omits the bash `true` noop special-case (no process
//! shell).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::llm::types::ToolCall;

/// Hard stop after this many consecutive identical tool steps.
pub const MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS: u32 = 16;
/// Nudge once when the identical-run length reaches this value.
pub const NUDGE_AFTER_IDENTICAL_TOOL_CALLS: u32 = 8;

const _: () = assert!(NUDGE_AFTER_IDENTICAL_TOOL_CALLS < MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS);

/// Aliases kept for call-site readability (same values as upstream shell).
pub const WARN_AFTER: u32 = NUDGE_AFTER_IDENTICAL_TOOL_CALLS;
pub const FAIL_AFTER: u32 = MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS;

/// Build the step signature for a batch of tool calls (upstream wire form).
pub fn step_signature(calls: &[ToolCall]) -> String {
    calls
        .iter()
        .map(|tc| format!("{}\u{1f}{}", tc.function.name, tc.function.arguments))
        .collect::<Vec<_>>()
        .join("\u{1e}")
}

fn hash_step_signature(signature: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    signature.hash(&mut hasher);
    hasher.finish()
}

/// Tracks consecutive identical tool-call steps within one turn loop.
#[derive(Default, Debug)]
pub struct IdenticalToolCallRun {
    last_signature_hash: Option<u64>,
    /// Name of the (first) tool in the current identical run.
    pub tool_name: String,
    /// Length of the current identical run (1 = first / different step).
    pub run_len: u32,
    nudged: bool,
}

impl IdenticalToolCallRun {
    /// Observe one model step's tool-call signature. Returns the new run length.
    pub fn observe(&mut self, signature: &str, tool_name: &str) -> u32 {
        let hash = hash_step_signature(signature);
        if self.last_signature_hash == Some(hash) {
            self.run_len += 1;
        } else {
            self.run_len = 1;
            self.last_signature_hash = Some(hash);
            self.nudged = false;
        }
        self.tool_name = tool_name.to_string();
        self.run_len
    }

    /// Once per identical run at/after the nudge threshold.
    /// Call only after tool results for the step are committed.
    pub fn take_nudge(&mut self) -> bool {
        let fire = self.run_len >= NUDGE_AFTER_IDENTICAL_TOOL_CALLS && !self.nudged;
        self.nudged |= fire;
        fire
    }

    pub fn hard_stop_threshold(&self) -> u32 {
        MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS
    }
}

/// Model-facing stationarity nudge (plain format!; no minijinja / tool_bridge).
pub fn stationarity_nudge_message(tool_name: &str, run_len: u32) -> String {
    format!(
        "You have called the same tool (`{tool_name}`) with the exact same arguments \
         {run_len} times in a row — you appear to be stuck in a polling loop. \
         Stop repeating this call. If you are waiting on a long-running job, use a \
         background task or the `monitor` tool, or wait once and check once — do not \
         poll in a tight loop. If you cannot make progress, stop and tell the user \
         what you are waiting for. This turn will be halted automatically if the \
         identical call keeps repeating."
    )
}

/// Backward-compatible alias used by older call sites.
pub type DoomLoopGuard = IdenticalToolCallRun;

/// Legacy single-call hash (tests / callers that only have one ToolCall).
pub fn hash_call(call: &ToolCall) -> u64 {
    hash_step_signature(&format!("{}\u{1f}{}", call.function.name, call.function.arguments))
}

pub fn nudge_message(repeats: usize) -> String {
    stationarity_nudge_message("tool", repeats as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::ToolCallFunction;

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: "x".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: name.into(),
                arguments: args.into(),
            },
            thought_signature: None,
        }
    }

    #[test]
    fn identical_resets_and_caps_at_16() {
        let mut run = IdenticalToolCallRun::default();
        let a = step_signature(&[call("read_file", "{\"path\":\"a\"}")]);
        assert_eq!(run.observe(&a, "read_file"), 1);
        assert_eq!(run.observe(&a, "read_file"), 2);
        let b = step_signature(&[call("grep", "{}")]);
        assert_eq!(run.observe(&b, "grep"), 1);
        let mut last = 0;
        let same = step_signature(&[call("same", "{}")]);
        for _ in 0..MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS {
            last = run.observe(&same, "same");
        }
        assert_eq!(last, MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS);
        assert_eq!(run.hard_stop_threshold(), MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS);
    }

    #[test]
    fn nudge_latch_fires_once_per_run_after_threshold() {
        let mut run = IdenticalToolCallRun::default();
        let sig = step_signature(&[call("get_task_output", "{\"task_ids\":[\"t1\"]}")]);
        for i in 1..NUDGE_AFTER_IDENTICAL_TOOL_CALLS {
            assert_eq!(run.observe(&sig, "get_task_output"), i);
            assert!(
                !run.take_nudge(),
                "must not nudge before threshold; run_len={i}"
            );
        }
        assert_eq!(
            run.observe(&sig, "get_task_output"),
            NUDGE_AFTER_IDENTICAL_TOOL_CALLS
        );
        assert!(run.take_nudge());
        assert!(!run.take_nudge());
        assert_eq!(
            run.observe(&sig, "get_task_output"),
            NUDGE_AFTER_IDENTICAL_TOOL_CALLS + 1
        );
        assert!(!run.take_nudge());
        let other = step_signature(&[call("bash", "{}")]);
        assert_eq!(run.observe(&other, "bash"), 1);
        assert!(!run.nudged);
        assert!(!run.take_nudge());
    }

    #[test]
    fn multi_tool_step_is_one_signature() {
        let mut run = IdenticalToolCallRun::default();
        let batch = step_signature(&[
            call("read_file", "{\"path\":\"a\"}"),
            call("grep", "{\"pattern\":\"x\"}"),
        ]);
        assert_eq!(run.observe(&batch, "read_file"), 1);
        assert_eq!(run.observe(&batch, "read_file"), 2);
        let single = step_signature(&[call("read_file", "{\"path\":\"a\"}")]);
        assert_eq!(run.observe(&single, "read_file"), 1);
    }
}
