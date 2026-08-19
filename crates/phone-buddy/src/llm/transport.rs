//! LLM transports.
//!
//! `HttpTransport` is the production path: reqwest against an
//! OpenAI-compatible `/chat/completions` endpoint with SSE streaming.
//! `MockTransport` is a deterministic, offline scripted provider used by the
//! demo and tests so the whole agent loop can run without an API key.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures_util::Stream;

use crate::error::{EngineError, EngineResult};
use crate::llm::retry::{classify_status, is_retry_vetoed_message, RetryClass};
use crate::llm::types::{
    ApiBackend, ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ChatCompletionRequest,
    Role, ToolCallDelta, ToolCallFunctionDelta, Usage,
};

pub type ChunkStream =
    Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, EngineError>> + Send>>;

/// A transport turns a chat-completion request into a stream of chunks.
///
/// Implementations must surface HTTP status information through
/// [`EngineError::Llm`] with a `status=<code>` prefix so the retry layer can
/// classify it (see [`status_from_error`]).
pub trait LlmTransport: Send + Sync {
    fn request_stream(&self, req: &ChatCompletionRequest)
        -> impl std::future::Future<Output = EngineResult<ChunkStream>> + Send;

    /// Transport name for diagnostics.
    fn name(&self) -> &str;
}

/// Extract an HTTP status code from an [`EngineError::Llm`] message that was
/// produced by [`HttpTransport`] (`status=<code>` prefix convention).
pub fn status_from_error(err: &EngineError) -> Option<u16> {
    let EngineError::Llm(msg) = err else {
        return None;
    };
    let rest = msg.strip_prefix("status=")?;
    let code: u16 = rest.split_whitespace().next()?.parse().ok()?;
    Some(code)
}

/// Whether a transport error is worth retrying.
pub fn retry_class_for_error(err: &EngineError) -> RetryClass {
    match err {
        EngineError::Llm(msg) => {
            // Upstream vetoes first: x-should-retry=false / context overflow.
            if is_retry_vetoed_message(msg) {
                return RetryClass::Fatal;
            }
            if let Some(code) = status_from_error(err) {
                return classify_status(code);
            }
            // Connection-level failures (no status): retry.
            if msg.contains("connection")
                || msg.contains("timeout")
                || msg.contains("reset")
                || msg.contains("dns")
                || msg.contains("error sending request")
            {
                RetryClass::Retry
            } else {
                RetryClass::Fatal
            }
        }
        EngineError::Stream(_) | EngineError::EmptyResponse => RetryClass::Retry,
        _ => RetryClass::Fatal,
    }
}

// ── HTTP transport ───────────────────────────────────────────────────────

pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    idle_timeout: Duration,
    api_backend: ApiBackend,
    extra_headers: std::collections::HashMap<String, String>,
    extra_body: std::collections::HashMap<String, serde_json::Value>,
    /// Opt into server doom-loop recovery (Responses API).
    doom_loop_enabled: bool,
}

impl HttpTransport {
    pub fn new(
        base_url: &str,
        api_key: &str,
        idle_timeout: Duration,
        api_backend: ApiBackend,
        extra_headers: std::collections::HashMap<String, String>,
    ) -> EngineResult<Self> {
        Self::new_with_doom_loop_and_extra_body(
            base_url,
            api_key,
            idle_timeout,
            api_backend,
            extra_headers,
            std::collections::HashMap::new(),
            false,
        )
    }

    pub fn new_with_doom_loop(
        base_url: &str,
        api_key: &str,
        idle_timeout: Duration,
        api_backend: ApiBackend,
        extra_headers: std::collections::HashMap<String, String>,
        doom_loop_enabled: bool,
    ) -> EngineResult<Self> {
        Self::new_with_doom_loop_and_extra_body(
            base_url,
            api_key,
            idle_timeout,
            api_backend,
            extra_headers,
            std::collections::HashMap::new(),
            doom_loop_enabled,
        )
    }

    pub fn new_with_doom_loop_and_extra_body(
        base_url: &str,
        api_key: &str,
        idle_timeout: Duration,
        api_backend: ApiBackend,
        extra_headers: std::collections::HashMap<String, String>,
        extra_body: std::collections::HashMap<String, serde_json::Value>,
        doom_loop_enabled: bool,
    ) -> EngineResult<Self> {
        // The ring crypto provider is installed by PhoneBuddyEngine::new.
        let client = reqwest::Client::builder()
            .user_agent(format!("PhoneBuddy/{} (Mobile SDK; LLM Client)", crate::VERSION))
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(60 * 10))
            .build()
            .map_err(|e| EngineError::Config(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            idle_timeout,
            api_backend,
            extra_headers,
            extra_body,
            doom_loop_enabled: doom_loop_enabled
                && matches!(api_backend, ApiBackend::Responses),
        })
    }

    fn endpoint(&self) -> String {
        let trimmed = self.base_url.trim_end_matches('/');
        let root = if let Some(stripped) = trimmed.strip_suffix("/chat/completions") {
            stripped
        } else if let Some(stripped) = trimmed.strip_suffix("/responses") {
            stripped
        } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
            stripped
        } else {
            trimmed
        };
        match self.api_backend {
            ApiBackend::ChatCompletions => format!("{root}/chat/completions"),
            ApiBackend::Responses => format!("{root}/responses"),
            ApiBackend::Messages => format!("{root}/messages"),
        }
    }
}

/// Merges custom key-value pairs from `extra_body` into the target JSON object.
pub fn merge_extra_body(
    body: &mut serde_json::Value,
    extra_body: &std::collections::HashMap<String, serde_json::Value>,
) {
    if let Some(obj) = body.as_object_mut() {
        for (k, v) in extra_body {
            obj.insert(k.clone(), v.clone());
        }
    }
}

impl LlmTransport for HttpTransport {
    async fn request_stream(
        &self,
        req: &ChatCompletionRequest,
    ) -> EngineResult<ChunkStream> {
        let endpoint = self.endpoint();
        let mut builder = self.client.post(&endpoint).header("Accept", "text/event-stream");

        for (k, v) in &self.extra_headers {
            builder = builder.header(k, v);
        }

        // Opt-in server doom-loop check (Responses API only).
        if self.doom_loop_enabled {
            builder = builder.header(
                crate::llm::doom_loop_wire::DOOM_LOOP_CHECK_HEADER,
                "1",
            );
        }

        let mut body = match self.api_backend {
            ApiBackend::ChatCompletions => {
                if !self.api_key.is_empty() {
                    builder = builder.bearer_auth(&self.api_key);
                }
                let mut val = serde_json::to_value(req)?;
                val["stream"] = serde_json::Value::Bool(true);
                val["stream_options"] = serde_json::json!({ "include_usage": true });
                val
            }
            ApiBackend::Responses => {
                if !self.api_key.is_empty() {
                    builder = builder.bearer_auth(&self.api_key);
                }
                build_responses_payload(req)
            }
            ApiBackend::Messages => {
                if !self.api_key.is_empty() {
                    builder = builder
                        .header("x-api-key", &self.api_key)
                        .bearer_auth(&self.api_key);
                }
                builder = builder.header("anthropic-version", "2023-06-01");
                build_messages_payload(req)
            }
        };

        merge_extra_body(&mut body, &self.extra_body);

        let resp = builder
            .json(&body)
            .send()
            .await
            .map_err(|e| EngineError::Llm(format!("error sending request: {e}")))?;

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            // Capture retry-related headers before consuming the body
            // (ported from grok sampler: Retry-After + x-should-retry).
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let should_retry = resp
                .headers()
                .get("x-should-retry")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let text = resp.text().await.unwrap_or_default();
            let detail = truncate_for_error(&text);
            let mut msg = format!("status={status}");
            if let Some(ra) = retry_after {
                msg.push_str(&format!(" retry-after={ra}"));
            }
            if let Some(sr) = should_retry {
                msg.push_str(&format!(" x-should-retry={sr}"));
            }
            msg.push(' ');
            msg.push_str(&detail);
            return Err(EngineError::Llm(msg));
        }

        use eventsource_stream::Eventsource as _;
        use futures_util::StreamExt as _;
        let byte_stream = resp.bytes_stream();
        let sse = byte_stream.eventsource();

        let idle_timeout = self.idle_timeout;
        let backend = self.api_backend;
        let doom_enabled = self.doom_loop_enabled;
        let collector = if doom_enabled {
            Some(crate::llm::doom_loop_collector::DoomLoopSignalCollector::new(
                crate::llm::doom_loop_wire::DoomLoopRecoveryPolicy::default(),
            ))
        } else {
            None
        };
        let chunk_stream = async_stream::stream! {
            futures_util::pin_mut!(sse);
            loop {
                let next = tokio::time::timeout(idle_timeout, sse.next()).await;
                let event = match next {
                    Ok(Some(ev)) => ev,
                    Ok(None) => {
                        // Terminal: act on confident doom-loop signals.
                        if let Some(ref c) = collector {
                            if let Some(triggers) = c.abort_triggers() {
                                yield Err(EngineError::DoomLoopServer(triggers.join(", ")));
                            }
                        }
                        break;
                    }
                    Err(_) => {
                        yield Err(EngineError::Stream(format!(
                            "idle timeout after {idle_timeout:?} with no SSE events"
                        )));
                        return;
                    }
                };
                let event = match event {
                    Ok(ev) => ev,
                    Err(e) => {
                        yield Err(EngineError::Stream(format!("SSE error: {e}")));
                        return;
                    }
                };

                // Swallow / record server doom-loop check events (Responses).
                // `absorb` also records terminal `response.doom_loop_check` fields.
                if let Some(ref c) = collector {
                    let swallow = c.absorb(&event.event, &event.data);
                    if let Some(triggers) = c.abort_triggers() {
                        yield Err(EngineError::DoomLoopServer(triggers.join(", ")));
                        return;
                    }
                    if swallow {
                        continue;
                    }
                }

                let res = match backend {
                    ApiBackend::ChatCompletions => crate::llm::stream::parse_chunk(&event.data),
                    ApiBackend::Responses => parse_responses_chunk(&event.event, &event.data),
                    ApiBackend::Messages => parse_messages_chunk(&event.event, &event.data),
                };
                match res {
                    Ok(Some(chunk)) => yield Ok(chunk),
                    Ok(None) => {}
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
        };

        Ok(Box::pin(chunk_stream))
    }

    fn name(&self) -> &str {
        match self.api_backend {
            ApiBackend::ChatCompletions => "http (chat/completions)",
            ApiBackend::Responses => "http (responses)",
            ApiBackend::Messages => "http (messages)",
        }
    }
}

/// Inject the `type: "reasoning_text"` discriminator the API requires.
/// Ported verbatim from grok-build `conversation/responses.rs::patch_reasoning_text_types`.
pub fn patch_reasoning_text_types(body: &mut serde_json::Value) {
    let Some(input) = body.get_mut("input").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for item in input.iter_mut() {
        if item.get("type").and_then(|t| t.as_str()) != Some("reasoning") {
            continue;
        }
        let Some(content) = item.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for c in content.iter_mut() {
            if let Some(obj) = c.as_object_mut() {
                obj.entry("type")
                    .or_insert_with(|| serde_json::Value::String("reasoning_text".into()));
            }
        }
    }
}

fn build_responses_payload(req: &ChatCompletionRequest) -> serde_json::Value {
    let mut instructions = String::new();
    let mut input = Vec::new();

    for msg in &req.messages {
        match msg.role {
            Role::System => {
                if let Some(ref content) = msg.content {
                    if !instructions.is_empty() {
                        instructions.push_str("\n\n");
                    }
                    instructions.push_str(content);
                }
            }
            Role::User => {
                if let Some(ref content) = msg.content {
                    input.push(serde_json::json!({
                        "role": "user",
                        "content": content
                    }));
                }
            }
            Role::Assistant => {
                // Sibling Reasoning items (preserved exactly like grok-build conversation/responses.rs)
                for r in &msg.reasoning_items {
                    let mut r_val = serde_json::to_value(r).unwrap_or_default();
                    if let Some(obj) = r_val.as_object_mut() {
                        obj.remove("status");
                    }
                    let mut item_obj = serde_json::json!({
                        "type": "reasoning",
                    });
                    if let Some(summary) = r_val.get("summary") {
                        item_obj["summary"] = summary.clone();
                    }
                    if let Some(content) = r_val.get("content") {
                        if !content.is_null() {
                            item_obj["content"] = content.clone();
                        }
                    }
                    if let Some(enc) = r_val.get("encrypted_content") {
                        if !enc.is_null() {
                            item_obj["encrypted_content"] = enc.clone();
                        }
                    }
                    if let Some(id) = r_val.get("id").and_then(|s| s.as_str()) {
                        if !id.is_empty() {
                            item_obj["id"] = serde_json::Value::String(id.to_string());
                        }
                    }
                    input.push(item_obj);
                }

                if let Some(ref content) = msg.content {
                    if !content.is_empty() {
                        input.push(serde_json::json!({
                            "role": "assistant",
                            "content": content
                        }));
                    }
                }
                for tc in &msg.tool_calls {
                    input.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }));
                }
            }
            Role::Tool => {
                if let Some(ref call_id) = msg.tool_call_id {
                    input.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": msg.content.as_deref().unwrap_or("")
                    }));
                }
            }
        }
    }

    let mut payload = serde_json::json!({
        "model": req.model,
        "input": input,
        "stream": true,
    });

    patch_reasoning_text_types(&mut payload);

    if !instructions.is_empty() {
        payload["instructions"] = serde_json::Value::String(instructions);
    }
    if let Some(temp) = req.temperature {
        payload["temperature"] = serde_json::json!(temp);
    }
    if let Some(max_tokens) = req.max_tokens {
        payload["max_output_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(ref tools) = req.tools {
        let tools_val: Vec<_> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters
                })
            })
            .collect();
        payload["tools"] = serde_json::Value::Array(tools_val);
    }
    if let Some(ref search) = req.search_parameters {
        payload["search_parameters"] = serde_json::to_value(search).unwrap_or_default();
    }

    payload
}

fn build_messages_payload(req: &ChatCompletionRequest) -> serde_json::Value {
    let mut system_text = String::new();
    let mut messages: Vec<serde_json::Value> = Vec::new();

    for msg in &req.messages {
        match msg.role {
            Role::System => {
                if let Some(ref content) = msg.content {
                    if !system_text.is_empty() {
                        system_text.push_str("\n\n");
                    }
                    system_text.push_str(content);
                }
            }
            Role::User => {
                if let Some(ref content) = msg.content {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": content
                    }));
                }
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                if let Some(ref reasoning) = msg.reasoning_content {
                    if !reasoning.is_empty() || msg.encrypted_reasoning.is_some() {
                        blocks.push(serde_json::json!({
                            "type": "thinking",
                            "thinking": reasoning,
                            "signature": msg.encrypted_reasoning.as_deref().unwrap_or("")
                        }));
                    }
                }
                if let Some(ref text) = msg.content {
                    if !text.is_empty() {
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                }
                for tc in &msg.tool_calls {
                    let input_val: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::json!({}));
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": input_val
                    }));
                }
                if !blocks.is_empty() {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": blocks
                    }));
                }
            }
            Role::Tool => {
                let call_id = msg.tool_call_id.as_deref().unwrap_or("");
                let content = msg.content.as_deref().unwrap_or("");
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": content
                        }
                    ]
                }));
            }
        }
    }

    let mut payload = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens.unwrap_or(8192),
        "stream": true,
    });

    if !system_text.is_empty() {
        payload["system"] = serde_json::Value::String(system_text);
    }
    if let Some(temp) = req.temperature {
        payload["temperature"] = serde_json::json!(temp);
    }
    if let Some(ref tools) = req.tools {
        let tools_val: Vec<_> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.function.name,
                    "description": t.function.description,
                    "input_schema": t.function.parameters
                })
            })
            .collect();
        payload["tools"] = serde_json::Value::Array(tools_val);
    }

    payload
}

fn parse_responses_chunk(event_name: &str, data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
    let raw = data.trim();
    if raw.is_empty() || raw == "[DONE]" {
        return Ok(None);
    }

    if let Ok(Some(chunk)) = crate::llm::stream::parse_chunk(raw) {
        if !chunk.choices.is_empty() {
            return Ok(Some(chunk));
        }
    }

    let v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(val) => val,
        Err(_) => return Ok(None),
    };

    let mut delta = ChatChunkDelta::default();

    let type_str = event_name.to_lowercase();
    let json_type = v.get("type").and_then(|s| s.as_str()).unwrap_or("");

    if type_str.contains("reasoning_summary_text.delta") || json_type.contains("reasoning_summary_text.delta") {
        if let Some(text) = v.get("delta").and_then(|s| s.as_str()) {
            delta.reasoning_content = Some(text.to_string());
        }
    } else if type_str.contains("reasoning_text.delta") || json_type.contains("reasoning_text.delta") {
        if let Some(text) = v.get("delta").and_then(|s| s.as_str()) {
            delta.reasoning_content = Some(text.to_string());
        }
    } else if type_str.contains("output_text.delta") || json_type.contains("output_text.delta") || type_str.contains("text.delta") || json_type.contains("text.delta") {
        if let Some(text) = v.get("delta").and_then(|s| s.as_str()) {
            delta.content = Some(text.to_string());
        }
    } else if type_str.contains("function_call_arguments.delta")
        || json_type.contains("function_call_arguments.delta")
    {
        let id = v.get("call_id").or_else(|| v.get("item_id")).and_then(|s| s.as_str()).map(|s| s.to_string());
        let name = v.get("name").and_then(|s| s.as_str()).map(|s| s.to_string());
        let args = v.get("delta").or_else(|| v.get("arguments")).and_then(|s| s.as_str()).map(|s| s.to_string());
        delta.tool_calls.push(ToolCallDelta {
            index: 0,
            id,
            kind: Some("function".to_string()),
            function: Some(ToolCallFunctionDelta { name, arguments: args }),
        });
    } else if let Some(item) = v.get("item").or_else(|| v.get("reasoning")) {
        if item.get("type").and_then(|s| s.as_str()) == Some("reasoning") {
            if let Ok(ri) = serde_json::from_value::<crate::llm::types::ReasoningItem>(item.clone()) {
                delta.encrypted_reasoning = ri.encrypted_content.clone();
                delta.reasoning_items.push(ri);
            }
        }
    } else if let Some(output) = v.get("output").or_else(|| v.get("response").and_then(|r| r.get("output"))).and_then(|o| o.as_array()) {
        for item in output {
            if item.get("type").and_then(|s| s.as_str()) == Some("reasoning") {
                if let Ok(ri) = serde_json::from_value::<crate::llm::types::ReasoningItem>(item.clone()) {
                    delta.encrypted_reasoning = ri.encrypted_content.clone();
                    delta.reasoning_items.push(ri);
                }
            } else if let Some(text) = item.get("content").and_then(|c| c.as_str()) {
                delta.content = Some(text.to_string());
            }
        }
    } else if let Some(d) = v.get("delta").and_then(|s| s.as_str()) {
        delta.content = Some(d.to_string());
    }

    let usage = v.get("usage").map(|u| Usage {
        prompt_tokens: u.get("input_tokens").or_else(|| u.get("prompt_tokens")).and_then(|n| n.as_u64()).unwrap_or(0) as u32,
        completion_tokens: u.get("output_tokens").or_else(|| u.get("completion_tokens")).and_then(|n| n.as_u64()).unwrap_or(0) as u32,
        total_tokens: u.get("total_tokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32,
    });

    if delta.content.is_none()
        && delta.reasoning_content.is_none()
        && delta.reasoning_items.is_empty()
        && delta.encrypted_reasoning.is_none()
        && delta.tool_calls.is_empty()
        && usage.is_none()
    {
        return Ok(None);
    }

    Ok(Some(ChatCompletionChunk {
        id: v.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        object: "response.chunk".to_string(),
        created: 0,
        model: v.get("model").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta,
            finish_reason: None,
        }],
        usage,
    }))
}

fn parse_messages_chunk(event_name: &str, data: &str) -> EngineResult<Option<ChatCompletionChunk>> {
    let raw = data.trim();
    if raw.is_empty() || raw == "[DONE]" {
        return Ok(None);
    }

    let v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(val) => val,
        Err(_) => return Ok(None),
    };

    let event_type = if !event_name.is_empty() {
        event_name
    } else {
        v.get("type").and_then(|s| s.as_str()).unwrap_or("")
    };

    let mut delta = ChatChunkDelta::default();
    let mut usage = None;
    let mut finish_reason = None;

    match event_type {
        "message_start" => {
            if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
                usage = Some(Usage {
                    prompt_tokens: u.get("input_tokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32,
                    completion_tokens: u.get("output_tokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32,
                    total_tokens: 0,
                });
            }
        }
        "content_block_start" => {
            if let Some(block) = v.get("content_block") {
                let btype = block.get("type").and_then(|s| s.as_str()).unwrap_or("");
                let index = v.get("index").and_then(|n| n.as_u64()).unwrap_or(0) as u32;
                if btype == "tool_use" {
                    let id = block.get("id").and_then(|s| s.as_str()).map(|s| s.to_string());
                    let name = block.get("name").and_then(|s| s.as_str()).map(|s| s.to_string());
                    delta.tool_calls.push(ToolCallDelta {
                        index,
                        id,
                        kind: Some("function".to_string()),
                        function: Some(ToolCallFunctionDelta {
                            name,
                            arguments: Some(String::new()),
                        }),
                    });
                }
            }
        }
        "content_block_delta" => {
            let index = v.get("index").and_then(|n| n.as_u64()).unwrap_or(0) as u32;
            if let Some(d) = v.get("delta") {
                let dtype = d.get("type").and_then(|s| s.as_str()).unwrap_or("");
                match dtype {
                    "text_delta" => {
                        if let Some(text) = d.get("text").and_then(|s| s.as_str()) {
                            delta.content = Some(text.to_string());
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial) = d.get("partial_json").and_then(|s| s.as_str()) {
                            delta.tool_calls.push(ToolCallDelta {
                                index,
                                id: None,
                                kind: None,
                                function: Some(ToolCallFunctionDelta {
                                    name: None,
                                    arguments: Some(partial.to_string()),
                                }),
                            });
                        }
                    }
                    "thinking_delta" => {
                        if let Some(thinking) = d.get("thinking").and_then(|s| s.as_str()) {
                            delta.reasoning_content = Some(thinking.to_string());
                        }
                    }
                    "signature_delta" => {
                        if let Some(sig) = d.get("signature").and_then(|s| s.as_str()) {
                            delta.encrypted_reasoning = Some(sig.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        "message_delta" => {
            if let Some(u) = v.get("usage") {
                usage = Some(Usage {
                    prompt_tokens: 0,
                    completion_tokens: u.get("output_tokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32,
                    total_tokens: 0,
                });
            }
            if let Some(d) = v.get("delta") {
                if let Some(sr) = d.get("stop_reason").and_then(|s| s.as_str()) {
                    finish_reason = Some(sr.to_string());
                }
            }
        }
        _ => {}
    }

    if delta.content.is_none()
        && delta.reasoning_content.is_none()
        && delta.encrypted_reasoning.is_none()
        && delta.tool_calls.is_empty()
        && usage.is_none()
        && finish_reason.is_none()
    {
        return Ok(None);
    }

    Ok(Some(ChatCompletionChunk {
        id: v.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 0,
        model: String::new(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
    }))
}

fn truncate_for_error(text: &str) -> String {
    let t = text.trim();
    let lower = t.to_ascii_lowercase();
    if lower.starts_with("<!doctype") || lower.starts_with("<html") || lower.contains("<title>") {
        if let Some(start_title) = lower.find("<title>") {
            let after_tag = &t[start_title + 7..];
            if let Some(end_title) = after_tag.to_ascii_lowercase().find("</title>") {
                let title = after_tag[..end_title].trim();
                return format!("(HTML page: \"{title}\")");
            }
        }
        return "(HTML error response from server/WAF)".to_string();
    }
    if t.chars().count() <= 500 {
        t.to_string()
    } else {
        t.chars().take(500).collect::<String>() + "…"
    }
}





// ── Mock transport ───────────────────────────────────────────────────────

/// A scripted, deterministic LLM used by the offline demo and tests.
///
/// Each entry is one assistant turn: either plain text or a list of tool
/// calls (+ optional text). Turns are returned in order; the last entry is
/// repeated if the agent keeps looping.
#[derive(Clone)]
pub struct MockTurn {
    pub text: String,
    pub tool_calls: Vec<(String, String, serde_json::Value)>, // (id, name, args)
}

impl MockTurn {
    pub fn text(t: impl Into<String>) -> Self {
        Self {
            text: t.into(),
            tool_calls: Vec::new(),
        }
    }
    pub fn calls(calls: Vec<(String, String, serde_json::Value)>) -> Self {
        Self {
            text: String::new(),
            tool_calls: calls,
        }
    }
}

pub struct MockTransport {
    turns: std::sync::Mutex<Vec<MockTurn>>,
}

impl MockTransport {
    pub fn new(turns: Vec<MockTurn>) -> Arc<Self> {
        Arc::new(Self {
            turns: std::sync::Mutex::new(turns),
        })
    }
}

impl LlmTransport for MockTransport {
    async fn request_stream(
        &self,
        _req: &ChatCompletionRequest,
    ) -> EngineResult<ChunkStream> {
        let turn = {
            let mut q = self.turns.lock().unwrap();
            if q.len() > 1 {
                q.remove(0)
            } else {
                q.first().cloned().unwrap_or_else(|| MockTurn::text("(mock: no more turns)"))
            }
        };

        // Emit the scripted turn as a sequence of realistic SSE chunks.
        let stream = async_stream::stream! {
            let mut choice_delta = ChatChunkDelta::default();
            // role first
            choice_delta.role = Some(crate::llm::types::Role::Assistant);

            // reasoning fragment
            yield Ok(make_chunk(&choice_delta));

            // text in small pieces (char-boundary safe)
            let text = turn.text.clone();
            let mut start = 0usize;
            while start < text.len() {
                let mut end = (start + 16).min(text.len());
                while end > start && !text.is_char_boundary(end) {
                    end -= 1;
                }
                if end == start {
                    end = text.len();
                }
                let mut d = ChatChunkDelta::default();
                d.content = Some(text[start..end].to_string());
                yield Ok(make_chunk(&d));
                start = end;
                tokio::time::sleep(Duration::from_millis(2)).await;
            }

            // tool calls
            for (idx, (id, name, args)) in turn.tool_calls.iter().enumerate() {
                let args_str = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
                // first chunk: id + name + first half of arguments
                let (a, b) = split_half(&args_str);
                let mut d = ChatChunkDelta::default();
                d.tool_calls.push(ToolCallDelta {
                    index: idx as u32,
                    id: Some(id.clone()),
                    kind: Some("function".into()),
                    function: Some(ToolCallFunctionDelta {
                        name: Some(name.clone()),
                        arguments: Some(a.to_string()),
                    }),
                });
                yield Ok(make_chunk(&d));
                // second chunk: rest of arguments
                let mut d2 = ChatChunkDelta::default();
                d2.tool_calls.push(ToolCallDelta {
                    index: idx as u32,
                    id: None,
                    kind: None,
                    function: Some(ToolCallFunctionDelta {
                        name: None,
                        arguments: Some(b.to_string()),
                    }),
                });
                yield Ok(make_chunk(&d2));
            }

            // finish chunk
            let mut finish = ChatChunkChoice {
                index: 0,
                delta: ChatChunkDelta::default(),
                finish_reason: Some(
                    if turn.tool_calls.is_empty() { "stop".into() } else { "tool_calls".into() },
                ),
            };
            let _ = &mut finish;
            yield Ok(ChatCompletionChunk {
                id: "mock".into(),
                object: "chat.completion.chunk".into(),
                created: 0,
                model: "mock-model".into(),
                choices: vec![finish],
                usage: Some(Usage { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 }),
            });
        };

        Ok(Box::pin(stream))
    }

    fn name(&self) -> &str {
        "mock"
    }
}



fn make_chunk(delta: &ChatChunkDelta) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: "mock".into(),
        object: "chat.completion.chunk".into(),
        created: 0,
        model: "mock-model".into(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: delta.clone(),
            finish_reason: None,
        }],
        usage: None,
    }
}

fn split_half(s: &str) -> (&str, &str) {
    let mid = s.len() / 2;
    // keep char boundary
    let mut m = mid;
    while m > 0 && !s.is_char_boundary(m) {
        m -= 1;
    }
    s.split_at(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatMessage, FunctionDefinitionWire, ToolDefinitionWire};

    #[test]
    fn test_api_backend_endpoint_resolution() {
        let base_url = "https://api.x.ai/v1";

        let t_chat = HttpTransport::new(base_url, "key", Duration::from_secs(10), ApiBackend::ChatCompletions, std::collections::HashMap::new()).unwrap();
        assert_eq!(t_chat.endpoint(), "https://api.x.ai/v1/chat/completions");

        let t_resp = HttpTransport::new(base_url, "key", Duration::from_secs(10), ApiBackend::Responses, std::collections::HashMap::new()).unwrap();
        assert_eq!(t_resp.endpoint(), "https://api.x.ai/v1/responses");

        let t_msg = HttpTransport::new(base_url, "key", Duration::from_secs(10), ApiBackend::Messages, std::collections::HashMap::new()).unwrap();
        assert_eq!(t_msg.endpoint(), "https://api.x.ai/v1/messages");
    }

    #[test]
    fn test_responses_payload_building() {
        let req = ChatCompletionRequest {
            model: "grok-3".into(),
            messages: vec![
                ChatMessage::system("You are helpful."),
                ChatMessage::user("Hello!"),
            ],
            stream: Some(true),
            tools: Some(vec![ToolDefinitionWire {
                kind: "function".into(),
                function: FunctionDefinitionWire {
                    name: "test_tool".into(),
                    description: Some("a test tool".into()),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]),
            tool_choice: None,
            temperature: Some(0.7),
            max_tokens: Some(1024),
            search_parameters: None,
        };

        let payload = build_responses_payload(&req);
        assert_eq!(payload["model"], "grok-3");
        assert_eq!(payload["instructions"], "You are helpful.");
        assert_eq!(payload["input"][0]["role"], "user");
        assert_eq!(payload["input"][0]["content"], "Hello!");
        assert_eq!(payload["tools"][0]["name"], "test_tool");
    }

    #[test]
    fn test_messages_payload_building() {
        let req = ChatCompletionRequest {
            model: "claude-3-5-sonnet".into(),
            messages: vec![
                ChatMessage::system("System prompt"),
                ChatMessage::user("Hi"),
            ],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: Some(0.5),
            max_tokens: Some(2048),
            search_parameters: None,
        };

        let payload = build_messages_payload(&req);
        assert_eq!(payload["model"], "claude-3-5-sonnet");
        assert_eq!(payload["system"], "System prompt");
        assert_eq!(payload["messages"][0]["role"], "user");
        assert_eq!(payload["messages"][0]["content"], "Hi");
        assert_eq!(payload["max_tokens"], 2048);
    }

    #[test]
    fn test_messages_chunk_parsing() {
        let text_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}"#;
        let chunk = parse_messages_chunk("content_block_delta", text_delta).unwrap().unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello world"));

        let thinking_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Thinking deeply..."}}"#;
        let chunk = parse_messages_chunk("content_block_delta", thinking_delta).unwrap().unwrap();
        assert_eq!(chunk.choices[0].delta.reasoning_content.as_deref(), Some("Thinking deeply..."));
    }

    #[test]
    fn test_responses_payload_with_reasoning_and_encrypted_content() {
        let req = ChatCompletionRequest {
            model: "grok-3".into(),
            messages: vec![
                ChatMessage::system("System instructions"),
                ChatMessage::user("Solve math problem"),
                ChatMessage::assistant_with_reasoning(
                    "Here is the solution: 42",
                    Some("Step 1: calculate...".into()),
                    vec![crate::llm::types::ReasoningItem {
                        id: "rs_1".into(),
                        summary: vec![crate::llm::types::SummaryPart::SummaryText(
                            crate::llm::types::SummaryTextContent {
                                text: "Step 1: calculate...".into(),
                            },
                        )],
                        content: Some(vec![crate::llm::types::ReasoningTextContent {
                            r#type: "reasoning_text".into(),
                            text: "Detailed reasoning".into(),
                        }]),
                        encrypted_content: Some("enc_token_xyz".into()),
                        status: None,
                    }],
                    Some("enc_token_xyz".into()),
                ),
            ],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: Some(0.7),
            max_tokens: Some(1024),
            search_parameters: None,
        };

        let payload = build_responses_payload(&req);
        let input = payload["input"].as_array().unwrap();
        // Item 0: user message
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "Solve math problem");
        // Item 1: reasoning sibling item (type: reasoning, with reasoning_text discriminator)
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["id"], "rs_1");
        assert_eq!(input[1]["summary"][0]["text"], "Step 1: calculate...");
        assert_eq!(input[1]["content"][0]["type"], "reasoning_text");
        assert_eq!(input[1]["encrypted_content"], "enc_token_xyz");
        // Item 2: assistant message
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[2]["content"], "Here is the solution: 42");
    }

    #[test]
    fn test_responses_chunk_parsing_reasoning() {
        let reasoning_delta = r#"{"type":"response.reasoning_text.delta","delta":"Analyzing..."}"#;
        let chunk = parse_responses_chunk("response.reasoning_text.delta", reasoning_delta).unwrap().unwrap();
        assert_eq!(chunk.choices[0].delta.reasoning_content.as_deref(), Some("Analyzing..."));

        let output_done = r#"{
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "r_123",
                "summary": [{"type": "summary_text", "text": "Summary text"}],
                "encrypted_content": "enc_secret"
            }
        }"#;
        let chunk = parse_responses_chunk("response.output_item.done", output_done).unwrap().unwrap();
        assert_eq!(chunk.choices[0].delta.encrypted_reasoning.as_deref(), Some("enc_secret"));
        assert_eq!(chunk.choices[0].delta.reasoning_items.len(), 1);
        assert_eq!(chunk.choices[0].delta.reasoning_items[0].id, "r_123");
    }

    #[test]
    fn test_extra_body_merging_into_payloads() {
        let req = ChatCompletionRequest {
            model: "claude-3-5-sonnet".into(),
            messages: vec![
                ChatMessage::user("Hello!"),
            ],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: Some(0.7),
            max_tokens: Some(1024),
            search_parameters: None,
        };

        let mut extra = std::collections::HashMap::new();
        extra.insert("custom_app_id".to_string(), serde_json::json!("org.example.app"));
        extra.insert("client_version".to_string(), serde_json::json!("1.0.0"));
        extra.insert("user_tier".to_string(), serde_json::json!("premium"));

        // Test Responses API payload
        let mut resp_payload = build_responses_payload(&req);
        merge_extra_body(&mut resp_payload, &extra);
        assert_eq!(resp_payload["custom_app_id"], "org.example.app");
        assert_eq!(resp_payload["client_version"], "1.0.0");
        assert_eq!(resp_payload["user_tier"], "premium");
        assert_eq!(resp_payload["model"], "claude-3-5-sonnet");

        // Test Messages API payload
        let mut msg_payload = build_messages_payload(&req);
        merge_extra_body(&mut msg_payload, &extra);
        assert_eq!(msg_payload["custom_app_id"], "org.example.app");
        assert_eq!(msg_payload["client_version"], "1.0.0");
        assert_eq!(msg_payload["user_tier"], "premium");
        assert_eq!(msg_payload["model"], "claude-3-5-sonnet");
    }
}
