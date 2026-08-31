//! Hermes-style Responses gateway robustness: tool arguments must survive
//! translation quirks (full args on `output_item.added`, misnumbered
//! argument deltas) instead of reaching client tools as `{}`.
//!
//! gro-build tolerates lossy deltas because its `ResponseCompleted` output
//! is the argument source of truth. PhoneBuddy's fallback path runs without
//! that guarantee, so these tests pin the salvage contract.

use phone_buddy::events::NullObserver;
use phone_buddy::llm::stream::collect_stream;
use phone_buddy::llm::types::{ChatCompletionChunk, CollectedTurn};
use phone_buddy::llm::wire::responses::parse_responses_chunk;

/// Run already-parsed chunks through the collector.
async fn collect_chunks(chunks: Vec<ChatCompletionChunk>) -> CollectedTurn {
    let stream = futures_util::stream::iter(chunks.into_iter().map(Ok));
    collect_stream(Box::pin(stream), &NullObserver)
        .await
        .unwrap()
}

/// Parse a Responses SSE event and keep only chunks that carry deltas.
fn responses_event(event: &str, data: serde_json::Value) -> Option<ChatCompletionChunk> {
    parse_responses_chunk(event, &data.to_string()).expect("event parses")
}

#[tokio::test]
async fn added_snapshot_feeds_args_when_no_delta_follows() {
    // Codex/Hermes-style proxies inline the complete argument object on
    // output_item.added and stream no delta/done afterwards. gro-build can
    // ignore added args (canonical output owns them); without the canonical
    // snapshot the arguments must survive here.
    let added = responses_event(
        "response.output_item.added",
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "toolu_1",
                "name": "web_search",
                "arguments": {"query": "today news"}
            }
        }),
    )
    .expect("added carries a delta");
    let turn = collect_chunks(vec![added]).await;
    assert_eq!(turn.tool_calls.len(), 1);
    assert_eq!(turn.tool_calls[0].function.name, "web_search");
    let args = &turn.tool_calls[0].function.arguments;
    assert!(
        args.contains("today news"),
        "added args must survive: {args}"
    );
}

#[tokio::test]
async fn misnumbered_deltas_are_salvaged_when_one_call_is_open() {
    // Hermes translation quirk: added sits at output_index 1 (after a
    // reasoning item) but argument deltas keep index 0 and carry no
    // call_id/item_id. gro-build drops them; the salvage path must recover
    // them onto the only open call instead of executing with `{}`.
    let added = responses_event(
        "response.output_item.added",
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "toolu_1",
                "name": "web_search",
                "arguments": ""
            }
        }),
    )
    .expect("added carries a delta");
    let d1 = responses_event(
        "response.function_call_arguments.delta",
        serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "delta": "{\"query\":"
        }),
    )
    .expect("delta carries a delta");
    let d2 = responses_event(
        "response.function_call_arguments.delta",
        serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "delta": "\"today news news\"}"
        }),
    )
    .expect("delta carries a delta");
    let turn = collect_chunks(vec![added, d1, d2]).await;
    assert_eq!(turn.tool_calls.len(), 1);
    let args = &turn.tool_calls[0].function.arguments;
    assert!(args.contains("today news"), "salvaged args: {args}");
}

#[tokio::test]
async fn ambiguous_open_calls_keep_dropping_unmatched_fragments() {
    // Two open calls and an unidentifiable fragment: attribution is
    // ambiguous, so the fragments stay dropped (gro-build parity) rather
    // than being attached to a random call.
    let json = r#"{"query":"orphan"}"#;
    let added = serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "type": "function_call",
            "id": "fc_a",
            "call_id": "toolu_a",
            "name": "web_search",
            "arguments": ""
        }
    });
    let added_b = serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 2,
        "item": {
            "type": "function_call",
            "id": "fc_b",
            "call_id": "toolu_b",
            "name": "web_search",
            "arguments": ""
        }
    });
    let orphan = responses_event(
        "response.function_call_arguments.delta",
        serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "delta": json
        }),
    )
    .expect("delta carries a delta");
    let chunks = vec![
        responses_event("response.output_item.added", added).unwrap(),
        responses_event("response.output_item.added", added_b).unwrap(),
        orphan,
    ];
    let turn = collect_chunks(chunks).await;
    assert_eq!(turn.tool_calls.len(), 2, "both calls must still surface");
    for call in &turn.tool_calls {
        assert_eq!(
            call.function.arguments, "{}",
            "ambiguous fragments must not attach"
        );
    }
}
