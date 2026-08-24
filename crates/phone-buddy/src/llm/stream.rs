//! SSE chat-completions stream accumulation: delta-accumulation semantics (text deltas,
//! reasoning deltas, positional tool-call assembly keyed by `index`, usage,
//! finish reason).

use std::collections::BTreeMap;
use std::time::Duration;

use futures_util::StreamExt;

use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver};
use crate::llm::types::{
    ChatCompletionChunk, CollectedTurn, ToolCall, ToolCallFunction, Usage,
};

/// Error from [`collect_stream`]. Idle timeout keeps the partial turn so
/// the retry layer can prefix-continue; every other stream error is
/// wrapped as [`CollectStreamError::Other`].
#[derive(Debug)]
pub enum CollectStreamError {
    IdleTimeout {
        partial: CollectedTurn,
        timeout: Duration,
    },
    Other(EngineError),
}

impl From<CollectStreamError> for EngineError {
    fn from(err: CollectStreamError) -> Self {
        match err {
            CollectStreamError::IdleTimeout { timeout, .. } => {
                EngineError::StreamIdleTimeout(timeout)
            }
            CollectStreamError::Other(e) => e,
        }
    }
}

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
) -> Result<CollectedTurn, CollectStreamError> {
    let mut turn = CollectedTurn::default();

    // Tool-call accumulators keyed by positional index:
    // (id, name, arguments_buffer, kind). Mirrors grok's `tool_call_acc`.
    // `id_to_idx` merges Responses events that share `call_id` but
    // disagree on `output_index` (arguments.delta often omits it).
    let mut tool_call_acc: BTreeMap<u32, (String, String, String, String)> = BTreeMap::new();
    let mut id_to_idx: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut started: std::collections::HashSet<String> = std::collections::HashSet::new();

    while let Some(next) = stream.next().await {
        let chunk = match next {
            Ok(c) => c,
            Err(e) => {
                finalize_turn(&mut turn, std::mem::take(&mut tool_call_acc));
                return Err(match idle_timeout_of(&e) {
                    Some(timeout) => CollectStreamError::IdleTimeout {
                        partial: turn,
                        timeout,
                    },
                    None => CollectStreamError::Other(e),
                });
            }
        };
        if chunk.id.starts_with("resp_") {
            turn.response_id = Some(chunk.id.clone());
        }
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
                let idx = if let Some(id) = &tc.id {
                    if !id.is_empty() {
                        *id_to_idx.entry(id.clone()).or_insert(tc.index)
                    } else {
                        tc.index
                    }
                } else {
                    tc.index
                };
                let entry = tool_call_acc.entry(idx).or_default();
                if let Some(id) = &tc.id {
                    if !id.is_empty() {
                        entry.0 = id.clone();
                    }
                }
                if let Some(kind) = &tc.kind {
                    if !kind.is_empty() {
                        entry.3 = kind.clone();
                    }
                }
                if let Some(f) = &tc.function {
                    if let Some(name) = &f.name {
                        if !name.is_empty() {
                            entry.1 = name.clone();
                        }
                    }
                    if let Some(args) = &f.arguments {
                        if entry.3 == "server" || tc.kind.as_deref() == Some("server") {
                            if !args.is_empty() {
                                entry.2 = args.clone();
                            }
                        } else {
                            append_tool_argument_fragment(&mut entry.2, args);
                        }
                    }
                }
                if !entry.1.is_empty() {
                    let call_id = if entry.0.is_empty() {
                        format!("idx_{idx}")
                    } else {
                        entry.0.clone()
                    };
                    if started.insert(call_id.clone()) {
                        observer.on_event(AgentEvent::ToolCallStart {
                            call_id,
                            name: entry.1.clone(),
                            arguments_json: if entry.2.is_empty() {
                                "{}".to_string()
                            } else {
                                entry.2.clone()
                            },
                        });
                    }
                }
            }
        }
    }

    finalize_turn(&mut turn, tool_call_acc);
    Ok(turn)
}

fn idle_timeout_of(err: &EngineError) -> Option<Duration> {
    match err {
        EngineError::StreamIdleTimeout(d) => Some(*d),
        EngineError::Stream(msg) if msg.contains("idle timeout") => {
            Some(Duration::from_secs(120))
        }
        _ => None,
    }
}

fn finalize_turn(
    turn: &mut CollectedTurn,
    tool_call_acc: BTreeMap<u32, (String, String, String, String)>,
) {
    // Assemble tool calls in index order.
    for (id, name, arguments, kind) in tool_call_acc.into_values() {
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
            kind: if kind.is_empty() {
                "function".to_string()
            } else {
                kind
            },
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
}

fn is_complete_json(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    serde_json::from_str::<serde::de::IgnoredAny>(s).is_ok()
}

/// Accumulate a tool-call argument fragment.
///
/// grok-build streams `function_call_arguments.delta` fragments and
/// *ignores* the later full snapshot (`function_call_arguments.done` /
/// `output_item.done`). PhoneBuddy's Responses parser used to feed both
/// into this buffer; `push_str` of a complete JSON onto a complete JSON
/// is exactly `{...}{...}`, which then fails with
/// `invalid JSON arguments: trailing characters`.
fn append_tool_argument_fragment(buffer: &mut String, incoming: &str) {
    if incoming.is_empty() {
        return;
    }
    if buffer.is_empty() {
        buffer.push_str(incoming);
        return;
    }
    if is_complete_json(buffer) && is_complete_json(incoming) {
        return;
    }
    buffer.push_str(incoming);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::NullObserver;
    use crate::llm::types::{
        ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ToolCallDelta,
        ToolCallFunctionDelta,
    };
    use futures_util::stream;

    fn chunk_with_tool(tc: ToolCallDelta) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "c".into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: "m".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatChunkDelta {
                    tool_calls: vec![tc],
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    fn tc(
        index: u32,
        id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> ToolCallDelta {
        ToolCallDelta {
            index,
            id: id.map(str::to_string),
            kind: Some("function".into()),
            function: Some(ToolCallFunctionDelta {
                name: name.map(str::to_string),
                arguments: arguments.map(str::to_string),
            }),
        }
    }

    #[tokio::test]
    async fn argument_fragments_concatenate() {
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc(
                0,
                Some("nav"),
                Some("browser_navigate"),
                Some("{\"url\":"),
            ))),
            Ok(chunk_with_tool(tc(0, None, None, Some("\"https://news.cctv.com/\"}")))),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(
            turn.tool_calls[0].function.arguments,
            "{\"url\":\"https://news.cctv.com/\"}"
        );
    }

    #[tokio::test]
    async fn complete_json_snapshot_is_not_appended_onto_the_same_json() {
        // Hermes Responses: a complete arguments delta followed by
        // function_call_arguments.done / output_item.done carrying the
        // same snapshot. grok-build ignores the snapshot; appending it
        // produces `{...}{...}` and the tool dies with trailing characters.
        let json = r#"{"url": "https://news.cctv.com/"}"#;
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc(
                0,
                Some("nav"),
                Some("browser_navigate"),
                Some(json),
            ))),
            Ok(chunk_with_tool(tc(0, Some("nav"), Some("browser_navigate"), Some(json)))),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.arguments, json);
    }

    #[tokio::test]
    async fn snapshot_fills_buffer_when_no_deltas_arrived() {
        let json = r#"{"direction": "down"}"#;
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc(
                0,
                Some("scroll"),
                Some("browser_scroll"),
                Some(""),
            ))),
            Ok(chunk_with_tool(tc(0, Some("scroll"), None, Some(json)))),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.tool_calls[0].function.arguments, json);
    }

    #[tokio::test]
    async fn idle_timeout_keeps_partial_text_and_response_id() {
        let stream = stream::iter(vec![
            Ok(ChatCompletionChunk {
                id: "resp_abc".into(),
                object: "response.chunk".into(),
                created: 0,
                model: "m".into(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatChunkDelta {
                        content: Some("hello ".into()),
                        ..Default::default()
                    },
                    finish_reason: None,
                }],
                usage: None,
            }),
            Err(EngineError::StreamIdleTimeout(Duration::from_secs(120))),
        ]);
        let err = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap_err();
        match err {
            CollectStreamError::IdleTimeout { partial, timeout } => {
                assert_eq!(partial.text, "hello ");
                assert_eq!(partial.response_id.as_deref(), Some("resp_abc"));
                assert_eq!(timeout, Duration::from_secs(120));
            }
            other => panic!("expected IdleTimeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn idle_timeout_with_incomplete_tool_json_is_still_partial() {
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc(
                0,
                Some("c1"),
                Some("web_search"),
                Some("{\"query\":"),
            ))),
            Err(EngineError::StreamIdleTimeout(Duration::from_secs(30))),
        ]);
        let err = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap_err();
        match err {
            CollectStreamError::IdleTimeout { partial, .. } => {
                assert_eq!(partial.tool_calls.len(), 1);
                assert_eq!(partial.tool_calls[0].function.arguments, "{\"query\":");
            }
            other => panic!("expected IdleTimeout, got {other:?}"),
        }
    }
}

#[allow(dead_code)]
fn _assert_usage_send() {
    fn _take(_: Usage) {}
}
