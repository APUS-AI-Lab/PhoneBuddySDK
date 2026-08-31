//! SSE chat-completions stream accumulation: delta-accumulation semantics (text deltas,
//! reasoning deltas, positional tool-call assembly keyed by `index`, usage,
//! finish reason).

use std::collections::BTreeMap;
use std::time::Duration;

use futures_util::StreamExt;

use crate::error::{EngineError, EngineResult};
use crate::events::{AgentEvent, AgentObserver};
use crate::llm::types::{ChatCompletionChunk, CollectedTurn, ToolCall, ToolCallFunction, Usage};

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
    // (id, name, arguments_buffer, kind, thought_signature).
    //
    // Identity merge is grok-build's output_index map plus Codex's
    // item_id/call_id: argument deltas often carry `item_id` and omit
    // `call_id`, and some proxies omit `output_index` (it defaults to 0).
    let mut tool_call_acc: BTreeMap<u32, (String, String, String, String, Option<String>)> =
        BTreeMap::new();
    let mut id_to_idx: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut output_to_idx: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut started: std::collections::HashSet<String> = std::collections::HashSet::new();
    // gro-build drops unmatched fragments outright because its canonical
    // `ResponseCompleted` output carries the authoritative arguments.
    // Hermes / LiteLLM-style translation gateways renumber argument
    // deltas relative to `output_item.added` and often send no usable
    // canonical snapshot, so fragments that resolve nowhere are parked
    // here and salvaged at finalize (see `salvage_parked_fragments`).
    let mut parked: Vec<ParkedFragment> = Vec::new();

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
            if delta.final_output.is_some() {
                turn.final_output = delta.final_output.clone();
            }
            if !delta.reasoning_items.is_empty() {
                turn.reasoning_items = crate::llm::types::merge_reasoning_items(
                    &turn.reasoning_items,
                    &delta.reasoning_items,
                );
            }

            for tc in &delta.tool_calls {
                let Some(idx) = resolve_tool_call_index(tc, &id_to_idx, &output_to_idx) else {
                    // grok-build drops FunctionCallArgumentsDelta when no
                    // OutputItemAdded mapped that output_index; it can
                    // afford to because its canonical response snapshot
                    // owns the final arguments. Hermes-style translation
                    // gateways misnumber deltas relative to the added
                    // item, so park the fragment and salvage it before
                    // finalize instead of losing the arguments. Parked
                    // fragments graft onto known tool entries only, so
                    // they can never land on slot 0 (reasoning / text).
                    if let Some(args) = tc.function.as_ref().and_then(|f| f.arguments.as_deref()) {
                        if !args.is_empty() {
                            parked.push(ParkedFragment {
                                index: tc.index,
                                call_id: tc.id.clone(),
                                item_id: tc.item_id.clone(),
                                fragment: args.to_string(),
                            });
                        }
                    }
                    continue;
                };
                remember_tool_call_ids(tc, idx, &mut id_to_idx, &mut output_to_idx);
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
                if let Some(sig) = &tc.thought_signature {
                    if !sig.is_empty() {
                        entry.4 = Some(sig.clone());
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
                    if entry.3 == "server" && started.insert(call_id.clone()) {
                        // grok-build `BackendToolCallStarted` carries
                        // name+id only. Hosted search query/sources land
                        // later on OutputItemDone (ToolCallResult).
                        let arguments_json = "{}".to_string();
                        if entry.1 == "x_thread_fetch" || entry.1.starts_with("x_") || entry.1 == "x_search" {
                            tracing::info!(
                                tool = %entry.1,
                                call_id = %call_id,
                                input = %arguments_json,
                                "[{}] Invoked: call_id='{}', input={}",
                                entry.1,
                                call_id,
                                arguments_json
                            );
                        }
                        observer.on_event(AgentEvent::ToolCallStart {
                            call_id,
                            name: entry.1.clone(),
                            arguments_json,
                        });
                    }
                }
                if !parked.is_empty() {
                    graft_parked_fragments(idx, tc, &mut tool_call_acc, &mut parked);
                }
            }
        }
    }

    salvage_parked_fragments(&mut tool_call_acc, &mut parked);
    finalize_turn(&mut turn, tool_call_acc);
    Ok(turn)
}

/// A tool-argument fragment that resolved to no known call on arrival.
#[derive(Debug)]
struct ParkedFragment {
    index: u32,
    call_id: Option<String>,
    item_id: Option<String>,
    fragment: String,
}

/// Graft parked fragments whose identity (call_id / item_id) matches a
/// just-registered call. This handles fragments that arrive *before*
/// their `output_item.added` (delta-before-added reordering).
fn graft_parked_fragments(
    idx: u32,
    tc: &crate::llm::types::ToolCallDelta,
    acc: &mut BTreeMap<u32, (String, String, String, String, Option<String>)>,
    parked: &mut Vec<ParkedFragment>,
) {
    let Some(entry) = acc.get_mut(&idx) else {
        return;
    };
    let identities: Vec<String> = [tc.id.clone(), tc.item_id.clone()]
        .into_iter()
        .flatten()
        .filter(|s| !s.is_empty())
        .collect();
    if identities.is_empty() {
        return;
    }
    let mut still_parked = Vec::new();
    for p in parked.drain(..) {
        let matches = p
            .call_id
            .as_ref()
            .or(p.item_id.as_ref())
            .is_some_and(|id| identities.contains(id));
        if matches && entry.3 != "server" {
            tracing::debug!(
                index = p.index,
                target = %idx,
                "grafted parked tool-argument fragment after call registration"
            );
            append_tool_argument_fragment(&mut entry.2, &p.fragment);
        } else {
            still_parked.push(p);
        }
    }
    *parked = still_parked;
}

/// Last-resort salvage for parked fragments before `finalize_turn`.
///
/// gro-build never needs this: its `ResponseCompleted` output is the
/// argument source of truth. Hermes-style gateways send no usable
/// canonical snapshot, so when exactly one still-empty client call was
/// observed, fragments parked by the resolver must belong to it. With
/// more than one open call the attribution is ambiguous and the
/// fragments stay dropped (gro-build parity).
fn salvage_parked_fragments(
    acc: &mut BTreeMap<u32, (String, String, String, String, Option<String>)>,
    parked: &mut Vec<ParkedFragment>,
) {
    if parked.is_empty() {
        return;
    }
    let candidates: Vec<u32> = acc
        .iter()
        .filter(|(_, entry)| entry.3 != "server" && is_placeholder_tool_args(&entry.2))
        .map(|(idx, _)| *idx)
        .collect();
    if candidates.len() != 1 {
        tracing::debug!(
            fragments = parked.len(),
            open_calls = candidates.len(),
            "dropping unmatched tool-argument fragments (ambiguous or no open call)"
        );
        parked.clear();
        return;
    }
    let entry = acc.get_mut(&candidates[0]).expect("candidate exists");
    let mut buffer = String::new();
    for p in parked.iter() {
        append_tool_argument_fragment(&mut buffer, &p.fragment);
    }
    if is_complete_json(&buffer) {
        tracing::info!(
            call_id = %entry.0,
            tool = %entry.1,
            args = %buffer,
            "salvaged parked tool-argument fragments onto the only open call"
        );
        entry.2 = buffer;
    } else {
        tracing::warn!(
            fragments = parked.len(),
            buffer = %buffer,
            "parked tool-argument fragments never formed complete JSON; dropping"
        );
    }
    parked.clear();
}

fn idle_timeout_of(err: &EngineError) -> Option<Duration> {
    match err {
        EngineError::StreamIdleTimeout(d) => Some(*d),
        EngineError::Stream(msg) if msg.contains("idle timeout") => Some(Duration::from_secs(120)),
        _ => None,
    }
}

fn finalize_turn(
    turn: &mut CollectedTurn,
    tool_call_acc: BTreeMap<u32, (String, String, String, String, Option<String>)>,
) {
    // Assemble streamed tool calls in index order.
    let mut streamed_calls = Vec::new();
    for (id, name, arguments, kind, thought_signature) in tool_call_acc.into_values() {
        if name.is_empty() {
            continue;
        }
        streamed_calls.push(ToolCall {
            id: if id.is_empty() {
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
            thought_signature,
        });
    }

    if let Some(output) = turn.final_output.take() {
        turn.items = items_from_canonical_output(&output, &streamed_calls);
    } else {
        turn.tool_calls = streamed_calls.clone();
        crate::llm::types::inject_streaming_reasoning_fallback(
            &mut turn.reasoning_items,
            &turn.reasoning,
        );
        if turn.reasoning_items.is_empty() {
            if let Some(item) = crate::llm::types::build_synthetic_reasoning(
                String::new(),
                None,
                turn.encrypted_reasoning.as_deref(),
            ) {
                turn.reasoning_items.push(item);
            }
        }
        turn.items = synthesize_items_from_accumulated(turn, streamed_calls);
    }

    // Apply streamed reasoning text to empty-summary items (canonical and
    // fallback). Encrypted-only tco_* blobs keep summary: [].
    let mut reasoning_view = turn.reasoning_from_items();
    crate::llm::types::inject_streaming_reasoning_fallback(&mut reasoning_view, &turn.reasoning);
    let mut ri = 0;
    for item in &mut turn.items {
        if let crate::conversation::ConversationItem::Reasoning(r) = item {
            if ri < reasoning_view.len() {
                *r = reasoning_view[ri].clone();
                ri += 1;
            }
        }
    }
    while ri < reasoning_view.len() {
        turn.items
            .insert(0, crate::conversation::ConversationItem::Reasoning(reasoning_view[ri].clone()));
        ri += 1;
    }

    turn.sync_derived_views();
}

fn items_from_canonical_output(
    output: &[crate::llm::types::OutputItemWire],
    streamed_calls: &[ToolCall],
) -> Vec<crate::conversation::ConversationItem> {
    use crate::conversation::{AssistantItem, BackendToolCallItem, ConversationItem};
    use crate::llm::types::OutputItemWire;

    let mut items = Vec::new();
    let mut text = String::new();
    let mut client_calls = Vec::new();

    for item in output {
        match item {
            OutputItemWire::Message { text: t, .. } => {
                if !text.is_empty() && !t.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            OutputItemWire::Reasoning(r) => {
                items.push(ConversationItem::Reasoning(r.clone()));
            }
            OutputItemWire::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                let streamed = streamed_calls
                    .iter()
                    .find(|c| c.id == *call_id)
                    .map(|c| c.function.arguments.as_str());
                let args = prefer_tool_arguments(arguments, streamed);
                let thought_signature = streamed_calls
                    .iter()
                    .find(|c| c.id == *call_id)
                    .and_then(|c| c.thought_signature.clone());
                client_calls.push(ToolCall {
                    id: call_id.clone(),
                    kind: "function".into(),
                    function: ToolCallFunction {
                        name: name.clone(),
                        arguments: args,
                    },
                    thought_signature,
                });
            }
            OutputItemWire::LocalShellCall {
                id,
                call_id,
                action,
                ..
            } => {
                let cid = call_id
                    .clone()
                    .or(id.clone())
                    .unwrap_or_else(|| "local_shell_1".into());
                let args = serde_json::to_string(&action).unwrap_or_default();
                client_calls.push(ToolCall {
                    id: cid,
                    kind: "local_shell".into(),
                    function: ToolCallFunction {
                        name: "local_shell".into(),
                        arguments: args,
                    },
                    thought_signature: None,
                });
            }
            OutputItemWire::CustomToolCall {
                id,
                call_id,
                name,
                input,
                output,
                raw,
                ..
            } => {
                let cid = if !call_id.is_empty() {
                    call_id.clone()
                } else {
                    id.clone().unwrap_or_else(|| "custom_tool_1".into())
                };
                if name == "x_search" || name.starts_with("x_") {
                    let mut payload_obj = if let Some(serde_json::Value::Object(m)) = raw {
                        m.clone()
                    } else {
                        let mut m = serde_json::Map::new();
                        m.insert("type".into(), serde_json::Value::String("custom_tool_call".into()));
                        m.insert("id".into(), serde_json::Value::String(cid.clone()));
                        m.insert("call_id".into(), serde_json::Value::String(cid.clone()));
                        m.insert("name".into(), serde_json::Value::String(name.clone()));
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&input) {
                            m.insert("input".into(), v);
                        } else {
                            m.insert("input".into(), serde_json::Value::String(input.clone()));
                        }
                        if let Some(ref out) = output {
                            m.insert("output".into(), out.clone());
                        }
                        m
                    };
                    if let Some(ref out) = output {
                        if !payload_obj.contains_key("output") {
                            payload_obj.insert("output".into(), out.clone());
                        }
                    }
                    let out_display = payload_obj
                        .get("output")
                        .map(|o: &serde_json::Value| o.to_string())
                        .unwrap_or_else(|| "{}".to_string());
                    if name == "x_thread_fetch" || name.starts_with("x_") || name == "x_search" {
                        tracing::info!(
                            tool = %name,
                            call_id = %cid,
                            input = %input,
                            output = %out_display,
                            "[{}] Fetched content: call_id='{}', input={}, output={}",
                            name,
                            cid,
                            input,
                            out_display
                        );
                    }
                    items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                        item_type: "custom_tool_call".into(),
                        id: cid,
                        payload: serde_json::Value::Object(payload_obj),
                    }));
                } else {
                    client_calls.push(ToolCall {
                        id: cid,
                        kind: "custom_tool".into(),
                        function: ToolCallFunction {
                            name: name.clone(),
                            arguments: if input.trim().is_empty() { "{}".into() } else { input.clone() },
                        },
                        thought_signature: None,
                    });
                }
            }
            OutputItemWire::Backend {
                item_type,
                id,
                payload,
            } => {
                items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                    item_type: item_type.clone(),
                    id: id.clone(),
                    payload: payload.clone(),
                }));
            }
        }
    }

    items.push(ConversationItem::Assistant(AssistantItem {
        content: text,
        tool_calls: client_calls,
        reasoning_content: None,
        encrypted_reasoning: None,
        origin: None,
    }));
    items
}

fn synthesize_items_from_accumulated(
    turn: &CollectedTurn,
    streamed_calls: Vec<ToolCall>,
) -> Vec<crate::conversation::ConversationItem> {
    use crate::conversation::{AssistantItem, BackendToolCallItem, ConversationItem};

    let mut items = Vec::new();
    for r in &turn.reasoning_items {
        items.push(ConversationItem::Reasoning(r.clone()));
    }

    let mut client_calls = Vec::new();
    for tc in streamed_calls {
        if tc.kind == "server" {
            let item_type = crate::conversation::server_tool_item_type(&tc.function.name);
            let payload = crate::conversation::reconstruct_backend_payload(
                &item_type,
                &tc.id,
                &tc.function.arguments,
            );
            items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                item_type,
                id: tc.id,
                payload,
            }));
        } else {
            client_calls.push(tc);
        }
    }

    items.push(ConversationItem::Assistant(AssistantItem {
        content: turn.text.clone(),
        tool_calls: client_calls,
        reasoning_content: if turn.reasoning.is_empty() {
            None
        } else {
            Some(turn.reasoning.clone())
        },
        encrypted_reasoning: turn.encrypted_reasoning.clone(),
        origin: None,
    }));
    items
}

fn is_complete_json(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    serde_json::from_str::<serde::de::IgnoredAny>(s).is_ok()
}

/// Empty / `{}` is the Responses `output_item.added` and Anthropic
/// `tool_use` start placeholder — not a finished argument object.
pub(crate) fn is_placeholder_tool_args(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    matches!(
        serde_json::from_str::<serde_json::Value>(t),
        Ok(serde_json::Value::Object(m)) if m.is_empty()
    )
}

fn prefer_tool_arguments(canonical: &str, streamed: Option<&str>) -> String {
    if !is_placeholder_tool_args(canonical) {
        return canonical.to_string();
    }
    if let Some(s) = streamed.filter(|s| !is_placeholder_tool_args(s)) {
        return s.to_string();
    }
    if canonical.trim().is_empty() {
        "{}".into()
    } else {
        canonical.to_string()
    }
}

fn tool_delta_has_name(tc: &crate::llm::types::ToolCallDelta) -> bool {
    tc.function
        .as_ref()
        .and_then(|f| f.name.as_deref())
        .is_some_and(|n| !n.is_empty())
}

fn resolve_tool_call_index(
    tc: &crate::llm::types::ToolCallDelta,
    id_to_idx: &std::collections::HashMap<String, u32>,
    output_to_idx: &std::collections::HashMap<u32, u32>,
) -> Option<u32> {
    if let Some(id) = tc.id.as_deref().filter(|s| !s.is_empty()) {
        if let Some(&idx) = id_to_idx.get(id) {
            return Some(idx);
        }
    }
    if let Some(id) = tc.item_id.as_deref().filter(|s| !s.is_empty()) {
        if let Some(&idx) = id_to_idx.get(id) {
            return Some(idx);
        }
    }
    if let Some(&idx) = output_to_idx.get(&tc.index) {
        return Some(idx);
    }
    // New call: Chat Completions first chunk, or Responses output_item.added.
    if tool_delta_has_name(tc) {
        return Some(tc.index);
    }
    None
}

fn remember_tool_call_ids(
    tc: &crate::llm::types::ToolCallDelta,
    idx: u32,
    id_to_idx: &mut std::collections::HashMap<String, u32>,
    output_to_idx: &mut std::collections::HashMap<u32, u32>,
) {
    if let Some(id) = tc.id.as_deref().filter(|s| !s.is_empty()) {
        id_to_idx.insert(id.to_string(), idx);
    }
    if let Some(id) = tc.item_id.as_deref().filter(|s| !s.is_empty()) {
        id_to_idx.insert(id.to_string(), idx);
    }
    output_to_idx.insert(tc.index, idx);
}

/// Accumulate a tool-call argument fragment.
///
/// grok-build streams `function_call_arguments.delta` fragments and
/// ignores later full snapshots. Codex never streams function-call
/// arguments: it takes the complete `output_item.done` item. PhoneBuddy
/// has to do both (xAI-style deltas *and* proxy snapshots).
///
/// `{}` is a start placeholder, not a finished object. A later complete
/// snapshot must replace it; two complete non-placeholder objects must
/// not concatenate into `{...}{...}`.
fn append_tool_argument_fragment(buffer: &mut String, incoming: &str) {
    if incoming.is_empty() {
        return;
    }
    if is_placeholder_tool_args(buffer) {
        buffer.clear();
        buffer.push_str(incoming);
        return;
    }
    if is_complete_json(buffer) {
        // The buffer already holds a closed argument object (streamed to
        // completion, or seeded by a Codex-style added snapshot). A later
        // complete snapshot is a repeat (grok-build ignores it); a later
        // partial fragment cannot continue a closed JSON object and must
        // not concatenate onto it.
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
    let chunk: ChatCompletionChunk = serde_json::from_str(data)
        .map_err(|e| EngineError::Stream(format!("failed to parse SSE chunk: {e}: {data:.120}")))?;
    if chunk.choices.is_empty() && chunk.usage.is_none() {
        return Ok(None);
    }
    Ok(Some(chunk))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{AgentEvent, NullObserver, RecordingObserver};
    use crate::llm::types::{
        ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ToolCallDelta, ToolCallFunctionDelta,
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
        tc_kind(index, id, name, arguments, "function")
    }

    fn tc_kind(
        index: u32,
        id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
        kind: &str,
    ) -> ToolCallDelta {
        ToolCallDelta {
            index,
            id: id.map(str::to_string),
            kind: Some(kind.into()),
            function: Some(ToolCallFunctionDelta {
                name: name.map(str::to_string),
                arguments: arguments.map(str::to_string),
            }),
            ..Default::default()
        }
    }

    fn tc_with_item_id(
        index: u32,
        id: Option<&str>,
        item_id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> ToolCallDelta {
        let mut d = tc(index, id, name, arguments);
        d.item_id = item_id.map(str::to_string);
        d
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
            Ok(chunk_with_tool(tc(
                0,
                None,
                None,
                Some("\"https://news.cctv.com/\"}"),
            ))),
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
            Ok(chunk_with_tool(tc(
                0,
                Some("nav"),
                Some("browser_navigate"),
                Some(json),
            ))),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.arguments, json);
    }

    #[tokio::test]
    async fn placeholder_object_is_replaced_by_later_snapshot() {
        // Anthropic-translated Responses: output_item.added carries "{}"
        // then output_item.done / arguments.done carries the real JSON.
        let json = r#"{"query":"today news"}"#;
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc(
                1,
                Some("toolu_1"),
                Some("web_search"),
                Some("{}"),
            ))),
            Ok(chunk_with_tool(tc(
                1,
                Some("toolu_1"),
                Some("web_search"),
                Some(json),
            ))),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.arguments, json);
    }

    #[tokio::test]
    async fn item_id_binds_argument_delta_when_output_index_is_wrong() {
        // Official deltas use item_id; proxies often omit output_index
        // (defaults to 0) while the function_call sits at output_index 1
        // after a reasoning item.
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc_with_item_id(
                1,
                Some("toolu_1"),
                Some("fc_1"),
                Some("web_search"),
                None,
            ))),
            Ok(chunk_with_tool(tc_with_item_id(
                0,
                None,
                Some("fc_1"),
                None,
                Some("{\"query\":\"today news\"}"),
            ))),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].id, "toolu_1");
        assert_eq!(
            turn.tool_calls[0].function.arguments,
            "{\"query\":\"today news\"}"
        );
    }

    #[tokio::test]
    async fn orphan_argument_delta_at_index_zero_is_dropped() {
        let json = r#"{"query":"from done"}"#;
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc(
                1,
                Some("toolu_1"),
                Some("web_search"),
                None,
            ))),
            Ok(chunk_with_tool(tc(0, None, None, Some("{\"query\":\"orphan\"}")))),
            Ok(chunk_with_tool(tc(
                1,
                Some("toolu_1"),
                Some("web_search"),
                Some(json),
            ))),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.arguments, json);
    }

    #[tokio::test]
    async fn canonical_placeholder_yields_to_streamed_args() {
        use crate::conversation::ConversationItem;
        use crate::llm::types::OutputItemWire;
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc(
                0,
                Some("c1"),
                Some("web_search"),
                Some(r#"{"query":"streamed"}"#),
            ))),
            Ok(chunk_final_output(vec![OutputItemWire::FunctionCall {
                call_id: "c1".into(),
                name: "web_search".into(),
                arguments: "{}".into(),
            }])),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        match &turn.items[0] {
            ConversationItem::Assistant(a) => {
                assert_eq!(a.tool_calls[0].function.arguments, r#"{"query":"streamed"}"#);
            }
            other => panic!("expected assistant, got {other:?}"),
        }
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
    async fn server_web_search_start_omits_placeholder_action() {
        // grok-build BackendToolCallStarted carries name+id only. The
        // placeholder action on output_item.added must not leak into
        // ToolCallStart; the assembled turn (and later ToolCallResult)
        // picks up query/sources from the done snapshot.
        let observer = RecordingObserver::new();
        let placeholder = r#"{"type":"search","query":"","sources":[]}"#;
        let done = r#"{"type":"search","query":"rust async runtime","sources":[{"type":"url","url":"https://example.com"}]}"#;
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc_kind(
                0,
                Some("ws_1"),
                Some("web_search"),
                Some(placeholder),
                "server",
            ))),
            Ok(chunk_with_tool(tc_kind(
                0,
                Some("ws_1"),
                Some("web_search"),
                Some(done),
                "server",
            ))),
        ]);
        let turn = collect_stream(Box::pin(stream), &observer).await.unwrap();

        let starts: Vec<_> = observer
            .snapshot()
            .into_iter()
            .filter_map(|e| match e {
                AgentEvent::ToolCallStart {
                    call_id,
                    name,
                    arguments_json,
                } => Some((call_id, name, arguments_json)),
                _ => None,
            })
            .collect();
        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0].0, "ws_1");
        assert_eq!(starts[0].1, "web_search");
        assert_eq!(starts[0].2, "{}");
        assert_eq!(turn.tool_calls.len(), 1);
        assert!(
            turn.tool_calls[0]
                .function
                .arguments
                .contains("rust async runtime"),
            "assembled args: {}",
            turn.tool_calls[0].function.arguments
        );
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

    fn reasoning_item(
        id: &str,
        summary: Vec<crate::llm::types::SummaryPart>,
        enc: Option<&str>,
    ) -> crate::llm::types::ReasoningItem {
        crate::llm::types::ReasoningItem {
            id: id.into(),
            summary,
            content: None,
            encrypted_content: enc.map(str::to_string),
            status: None,
        }
    }

    fn chunk_with_reasoning(
        item: Option<crate::llm::types::ReasoningItem>,
        reasoning_text: Option<&str>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "c".into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: "m".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatChunkDelta {
                    reasoning_content: reasoning_text.map(str::to_string),
                    reasoning_items: item.into_iter().collect(),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    #[tokio::test]
    async fn merges_reasoning_items_with_the_same_id() {
        // output_item.added is an empty stub; output_item.done carries
        // encrypted_content. Pushing both as siblings made the next
        // turn's input[1] a reasoning item with no summary.
        let stream = stream::iter(vec![
            Ok(chunk_with_reasoning(
                Some(reasoning_item("rs_1", Vec::new(), None)),
                None,
            )),
            Ok(chunk_with_reasoning(
                Some(reasoning_item("rs_1", Vec::new(), Some("enc"))),
                None,
            )),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.reasoning_items.len(), 1);
        assert_eq!(turn.reasoning_items[0].id, "rs_1");
        assert_eq!(
            turn.reasoning_items[0].encrypted_content.as_deref(),
            Some("enc")
        );
    }

    #[tokio::test]
    async fn fills_empty_reasoning_summary_from_streamed_text() {
        // grok-build inject_streaming_reasoning_fallback: if a typed
        // Reasoning sibling arrived with no text, splice streamed
        // reasoning deltas into its summary so the next-turn payload
        // is valid.
        let stream = stream::iter(vec![
            Ok(chunk_with_reasoning(
                Some(reasoning_item("rs_1", Vec::new(), None)),
                None,
            )),
            Ok(chunk_with_reasoning(None, Some("thinking about APUS"))),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.reasoning_items.len(), 1);
        assert_eq!(
            crate::llm::types::reasoning_item_text(&turn.reasoning_items[0]),
            "thinking about APUS"
        );
    }

    fn chunk_final_output(output: Vec<crate::llm::types::OutputItemWire>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "resp_1".into(),
            object: "response.chunk".into(),
            created: 0,
            model: "m".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatChunkDelta {
                    final_output: Some(output),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    #[tokio::test]
    async fn response_completed_is_canonical() {
        use crate::conversation::ConversationItem;
        use crate::llm::types::OutputItemWire;
        let stream = stream::iter(vec![
            Ok(chunk_with_tool(tc(
                0,
                Some("c1"),
                Some("read_file"),
                Some("{\"path\":"),
            ))),
            Ok(chunk_with_tool(tc(0, None, None, Some("\"a\"}")))),
            Ok(chunk_final_output(vec![
                OutputItemWire::Backend {
                    item_type: "web_search_call".into(),
                    id: "ws_1".into(),
                    payload: serde_json::json!({"type":"web_search_call","id":"ws_1"}),
                },
                OutputItemWire::FunctionCall {
                    call_id: "c1".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"canonical"}"#.into(),
                },
                OutputItemWire::Message {
                    id: "msg_1".into(),
                    text: "hello".into(),
                },
            ])),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert!(matches!(
            turn.items[0],
            ConversationItem::BackendToolCall(_)
        ));
        match &turn.items[1] {
            ConversationItem::Assistant(a) => {
                assert_eq!(a.content, "hello");
                assert_eq!(a.tool_calls.len(), 1);
                assert_eq!(a.tool_calls[0].function.arguments, r#"{"path":"canonical"}"#);
            }
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn canonical_output_replaces_merge_heuristics() {
        use crate::conversation::ConversationItem;
        use crate::llm::types::OutputItemWire;
        let stream = stream::iter(vec![
            Ok(chunk_with_reasoning(
                Some(reasoning_item("rs_1", Vec::new(), Some("enc-delta"))),
                None,
            )),
            Ok(chunk_final_output(vec![
                OutputItemWire::Reasoning(reasoning_item("rs_1", Vec::new(), Some("enc-final"))),
                OutputItemWire::Message {
                    id: "m".into(),
                    text: "ok".into(),
                },
            ])),
        ]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        let reasoning: Vec<_> = turn
            .items
            .iter()
            .filter(|i| matches!(i, ConversationItem::Reasoning(_)))
            .collect();
        assert_eq!(reasoning.len(), 1);
        match &turn.items[0] {
            ConversationItem::Reasoning(r) => {
                assert_eq!(r.encrypted_content.as_deref(), Some("enc-final"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn derived_views_match_items() {
        use crate::conversation::ConversationItem;
        use crate::llm::types::OutputItemWire;
        let stream = stream::iter(vec![Ok(chunk_final_output(vec![
            OutputItemWire::Reasoning(reasoning_item("rs_1", Vec::new(), None)),
            OutputItemWire::FunctionCall {
                call_id: "c1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            },
            OutputItemWire::Message {
                id: "m".into(),
                text: "hi".into(),
            },
        ]))]);
        let turn = collect_stream(Box::pin(stream), &NullObserver)
            .await
            .unwrap();
        assert_eq!(turn.text, turn.text_from_items());
        assert_eq!(turn.client_tool_calls().len(), 1);
        assert_eq!(
            turn.reasoning_items.len(),
            turn.items
                .iter()
                .filter(|i| matches!(i, ConversationItem::Reasoning(_)))
                .count()
        );
        assert_eq!(turn.text, "hi");
        assert_eq!(turn.tool_calls[0].id, "c1");
    }
}

#[allow(dead_code)]
fn _assert_usage_send() {
    fn _take(_: Usage) {}
}
