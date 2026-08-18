//! SSE chat-completions stream accumulation: delta-accumulation semantics (text deltas,
//! reasoning deltas, positional tool-call assembly keyed by `index`, usage,
//! finish reason).

use std::collections::BTreeMap;

use futures_util::StreamExt;

use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver};
use crate::llm::types::{
    ChatCompletionChunk, CollectedTurn, ToolCall, ToolCallFunction, Usage,
};

/// Consume a raw chunk stream, streaming deltas to `observer` and returning
/// the fully assembled turn.
///
/// Tool call assembly follows the OpenAI-compatible streaming contract:
/// the first chunk for an index carries `id`+`name` and starts the argument
/// buffer; subsequent chunks append argument fragments.
pub async fn collect_stream(
    mut stream: std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<ChatCompletionChunk, EngineError>> + Send>,
    >,
    observer: &dyn AgentObserver,
) -> EngineResult<CollectedTurn> {
    let mut turn = CollectedTurn::default();

    // Tool-call accumulators keyed by positional index:
    // (id, name, arguments_buffer). Mirrors grok's `tool_call_acc`.
    let mut tool_call_acc: BTreeMap<u32, (String, String, String)> = BTreeMap::new();

    while let Some(next) = stream.next().await {
        let chunk = next?;
        if !chunk.model.is_empty() {
            turn.model = chunk.model.clone();
        }
        if let Some(usage) = &chunk.usage {
            turn.usage = Some(usage.clone());
        }

        for choice in &chunk.choices {
            if let Some(reason) = &choice.finish_reason {
                turn.finish_reason = Some(reason.clone());
            }
            let delta = &choice.delta;

            if let Some(text) = &delta.content {
                if !text.is_empty() {
                    turn.text.push_str(text);
                    observer.on_event(AgentEvent::TextDelta { text: text.clone() });
                }
            }
            if let Some(reasoning) = &delta.reasoning_content {
                if !reasoning.is_empty() {
                    turn.reasoning.push_str(reasoning);
                    observer.on_event(AgentEvent::ReasoningDelta {
                        text: reasoning.clone(),
                    });
                }
            }
            if let Some(enc) = &delta.encrypted_reasoning {
                turn.encrypted_reasoning = Some(enc.clone());
            }
            for ri in &delta.reasoning_items {
                turn.reasoning_items.push(ri.clone());
            }

            for tc in &delta.tool_calls {
                let entry = tool_call_acc.entry(tc.index).or_default();
                if let Some(id) = &tc.id {
                    entry.0 = id.clone();
                }
                if let Some(f) = &tc.function {
                    if let Some(name) = &f.name {
                        entry.1 = name.clone();
                    }
                    if let Some(args) = &f.arguments {
                        entry.2.push_str(args);
                    }
                }
            }
        }
    }

    // Assemble tool calls in index order.
    for (id, name, arguments) in tool_call_acc.into_values() {
        if name.is_empty() {
            continue;
        }
        turn.tool_calls.push(ToolCall {
            id: if id.is_empty() {
                // Some providers omit the id; synthesize a stable one.
                format!("call_{}", uuid::Uuid::new_v4().simple())
            } else {
                id
            },
            kind: "function".to_string(),
            function: ToolCallFunction {
                name,
                arguments: if arguments.trim().is_empty() {
                    "{}".to_string()
                } else {
                    arguments
                },
            },
        });
    }

    // If no typed reasoning items arrived but reasoning text or encrypted content was
    // collected, synthesize a ReasoningItem (matching grok-build `inject_streaming_reasoning_fallback`).
    if turn.reasoning_items.is_empty() && (!turn.reasoning.is_empty() || turn.encrypted_reasoning.is_some()) {
        if let Some(item) = crate::llm::types::build_synthetic_reasoning(
            String::new(),
            if turn.reasoning.is_empty() { None } else { Some(&turn.reasoning) },
            turn.encrypted_reasoning.as_deref(),
        ) {
            turn.reasoning_items.push(item);
        }
    }

    Ok(turn)
}

/// Parse one SSE `data:` payload into a chunk. `[DONE]` terminator yields
/// `None`.
pub fn parse_chunk(data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    let chunk: ChatCompletionChunk = serde_json::from_str(data).map_err(|e| {
        EngineError::Stream(format!("failed to parse SSE chunk: {e}: {data:.120}"))
    })?;
    if chunk.choices.is_empty() && chunk.usage.is_none() {
        return Ok(None);
    }
    Ok(Some(chunk))
}

#[allow(dead_code)]
fn _assert_usage_send() {
    fn _take(_: Usage) {}
}
