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
    ApiBackend, ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ConversationRequest,
    ToolCallDelta, ToolCallFunctionDelta, Usage,
};
use crate::llm::wire::adapter_for;

pub type ChunkStream =
    Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, EngineError>> + Send>>;

/// A transport turns a chat-completion request into a stream of chunks.
///
/// Implementations must surface HTTP status information through
/// [`EngineError::Llm`] with a `status=<code>` prefix so the retry layer can
/// classify it (see [`status_from_error`]).
pub trait LlmTransport: Send + Sync {
    fn request_stream(&self, req: &ConversationRequest)
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
        EngineError::Stream(_)
        | EngineError::StreamIdleTimeout(_)
        | EngineError::EmptyResponse => RetryClass::Retry,
        _ => RetryClass::Fatal,
    }
}

// ── HTTP transport ───────────────────────────────────────────────────────

use crate::llm::profiles::ClientProfile;

pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    idle_timeout: Duration,
    api_backend: ApiBackend,
    client_profile: ClientProfile,
    client_version: Option<String>,
    client_session_id: Option<String>,
    extra_headers: std::collections::HashMap<String, String>,
    extra_body: std::collections::HashMap<String, serde_json::Value>,
    /// Opt into server doom-loop recovery (Responses API).
    doom_loop_enabled: bool,
    /// HTTP Traffic Dumper for diagnostics
    dumper: crate::llm::dumper::HttpDumper,
}

impl HttpTransport {
    pub fn new(
        base_url: &str,
        api_key: &str,
        idle_timeout: Duration,
        api_backend: ApiBackend,
        extra_headers: std::collections::HashMap<String, String>,
    ) -> EngineResult<Self> {
        Self::new_with_all_options(
            base_url,
            api_key,
            idle_timeout,
            api_backend,
            ClientProfile::Default,
            None,
            None,
            extra_headers,
            std::collections::HashMap::new(),
            false,
            crate::llm::dumper::HttpDumper::new(
                crate::llm::dumper::HttpDumpConfig::default(),
                std::env::temp_dir().join("phone-buddy").join("http_dumps"),
            ),
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
        Self::new_with_all_options(
            base_url,
            api_key,
            idle_timeout,
            api_backend,
            ClientProfile::Default,
            None,
            None,
            extra_headers,
            std::collections::HashMap::new(),
            doom_loop_enabled,
            crate::llm::dumper::HttpDumper::new(
                crate::llm::dumper::HttpDumpConfig::default(),
                std::env::temp_dir().join("phone-buddy").join("http_dumps"),
            ),
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
        Self::new_with_all_options(
            base_url,
            api_key,
            idle_timeout,
            api_backend,
            ClientProfile::Default,
            None,
            None,
            extra_headers,
            extra_body,
            doom_loop_enabled,
            crate::llm::dumper::HttpDumper::new(
                crate::llm::dumper::HttpDumpConfig::default(),
                std::env::temp_dir().join("phone-buddy").join("http_dumps"),
            ),
        )
    }

    pub fn new_with_all_options(
        base_url: &str,
        api_key: &str,
        idle_timeout: Duration,
        api_backend: ApiBackend,
        client_profile: ClientProfile,
        client_version: Option<String>,
        client_session_id: Option<String>,
        extra_headers: std::collections::HashMap<String, String>,
        extra_body: std::collections::HashMap<String, serde_json::Value>,
        doom_loop_enabled: bool,
        dumper: crate::llm::dumper::HttpDumper,
    ) -> EngineResult<Self> {
        // The ring crypto provider is installed by PhoneBuddyEngine::new.
        let client = reqwest::Client::builder()
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
            client_profile,
            client_version,
            client_session_id,
            extra_headers,
            extra_body,
            doom_loop_enabled: doom_loop_enabled
                && matches!(api_backend, ApiBackend::Responses),
            dumper,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn endpoint(&self) -> String {
        self.endpoint_for_model("")
    }

    fn endpoint_for_model(&self, model: &str) -> String {
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
        adapter_for(self.api_backend).endpoint(root, model, true)
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
        req: &ConversationRequest,
    ) -> EngineResult<ChunkStream> {
        let adapter = adapter_for(self.api_backend);
        let endpoint = self.endpoint_for_model(&req.model);
        let mut builder = self.client.post(&endpoint);

        let mut req_headers_map = std::collections::BTreeMap::new();

        // 1. Profile default headers (UA, vendor headers, auth)
        let profile_headers = crate::llm::profiles::build_profile_headers(
            self.client_profile,
            &self.api_key,
            self.client_session_id.as_deref(),
            self.client_version.as_deref(),
            self.doom_loop_enabled,
        );

        for (k, v) in profile_headers {
            builder = builder.header(&k, &v);
            req_headers_map.insert(k.to_ascii_lowercase(), self.dumper.mask_header_value(&k, &v));
        }

        // Gemini uses x-goog-api-key rather than Bearer.
        if matches!(self.api_backend, ApiBackend::Gemini) && !self.api_key.is_empty() {
            builder = builder.header("x-goog-api-key", &self.api_key);
            req_headers_map.insert(
                "x-goog-api-key".into(),
                self.dumper.mask_header_value("x-goog-api-key", &self.api_key),
            );
        }

        // 2. Extra headers override anything from the profile
        for (k, v) in &self.extra_headers {
            builder = builder.header(k, v);
            req_headers_map.insert(k.to_ascii_lowercase(), self.dumper.mask_header_value(k, v));
        }

        let mut body = adapter.build_payload(req)?;

        if matches!(self.api_backend, ApiBackend::Messages) {
            if body
                .get("output_config")
                .and_then(|o| o.get("effort"))
                .is_some()
            {
                let beta_key = "anthropic-beta";
                let effort_beta = "effort-2025-11-24";
                let current = req_headers_map.get(beta_key).cloned();
                let new_val = match current {
                    Some(ref v) if !v.contains(effort_beta) => format!("{v},{effort_beta}"),
                    Some(v) => v,
                    None => effort_beta.to_string(),
                };
                builder = builder.header(beta_key, &new_val);
                req_headers_map.insert(
                    beta_key.to_string(),
                    self.dumper.mask_header_value(beta_key, &new_val),
                );
            }
        }

        merge_extra_body(&mut body, &self.extra_body);
        req_headers_map.insert("content-type".to_string(), "application/json".to_string());

        let req_id = format!("req_{}", uuid::Uuid::new_v4().simple());
        let start_time = std::time::Instant::now();
        let timestamp_str = chrono::Utc::now().to_rfc3339();

        let req_dump = crate::llm::dumper::HttpRequestDump {
            method: "POST".into(),
            url: endpoint.clone(),
            headers: req_headers_map,
            body: body.clone(),
        };

        let resp_result = builder.json(&body).send().await;

        let resp = match resp_result {
            Ok(r) => r,
            Err(e) => {
                let err_msg = format!("error sending request: {e}");
                let mut dump_path_opt = None;
                if self.dumper.should_dump_error() {
                    let dump_rec = crate::llm::dumper::HttpDumpRecord {
                        schema_version: "1.0".into(),
                        request_id: req_id,
                        timestamp: timestamp_str,
                        duration_ms: start_time.elapsed().as_millis() as u64,
                        request: req_dump,
                        response: None,
                        error: Some(err_msg.clone()),
                    };
                    dump_path_opt = self.dumper.dump(&dump_rec);
                }
                let mut full_err = err_msg;
                if let Some(path) = dump_path_opt {
                    full_err.push_str(&format!(" [HTTP dump: {}]", path.display()));
                }
                return Err(EngineError::Llm(full_err));
            }
        };

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
            let resp_headers_dump = self.dumper.extract_headers(resp.headers());
            let status_text = resp.status().canonical_reason().unwrap_or("Unknown").to_string();
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

            if self.dumper.should_dump_error() {
                let dump_rec = crate::llm::dumper::HttpDumpRecord {
                    schema_version: "1.0".into(),
                    request_id: req_id,
                    timestamp: timestamp_str,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                    request: req_dump,
                    response: Some(crate::llm::dumper::HttpResponseDump {
                        status,
                        status_text,
                        headers: resp_headers_dump,
                        body_text: text,
                    }),
                    error: Some(msg.clone()),
                };
                if let Some(path) = self.dumper.dump(&dump_rec) {
                    msg.push_str(&format!(" [HTTP dump: {}]", path.display()));
                }
            }

            return Err(EngineError::Llm(msg));
        }

        if self.dumper.should_dump_success() {
            let resp_headers_dump = self.dumper.extract_headers(resp.headers());
            let status_text = resp.status().canonical_reason().unwrap_or("OK").to_string();
            let dump_rec = crate::llm::dumper::HttpDumpRecord {
                schema_version: "1.0".into(),
                request_id: req_id,
                timestamp: timestamp_str,
                duration_ms: start_time.elapsed().as_millis() as u64,
                request: req_dump,
                response: Some(crate::llm::dumper::HttpResponseDump {
                    status,
                    status_text,
                    headers: resp_headers_dump,
                    body_text: "<streaming SSE response started>".into(),
                }),
                error: None,
            };
            self.dumper.dump(&dump_rec);
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
                        yield Err(EngineError::StreamIdleTimeout(idle_timeout));
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

                let res = adapter_for(backend).parse_event(&event.event, &event.data);
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
            ApiBackend::Gemini => "http (gemini)",
        }
    }
}

/// Inject the `type: "reasoning_text"` discriminator the API requires.

pub use crate::llm::wire::responses::{build_responses_payload, parse_responses_chunk, patch_reasoning_text_types};
pub use crate::llm::wire::messages::{build_messages_payload, parse_messages_chunk};

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
    pub tool_calls: Vec<(String, String, serde_json::Value, String)>, // (id, name, args, kind)
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
            tool_calls: calls
                .into_iter()
                .map(|(id, name, args)| (id, name, args, "function".into()))
                .collect(),
        }
    }
    pub fn server_calls_and_text(
        text: impl Into<String>,
        calls: Vec<(String, String, serde_json::Value)>,
    ) -> Self {
        Self {
            text: text.into(),
            tool_calls: calls
                .into_iter()
                .map(|(id, name, args)| (id, name, args, "server".into()))
                .collect(),
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
        _req: &ConversationRequest,
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
            for (idx, (id, name, args, kind)) in turn.tool_calls.iter().enumerate() {
                let args_str = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
                // first chunk: id + name + first half of arguments
                let (a, b) = split_half(&args_str);
                let mut d = ChatChunkDelta::default();
                d.tool_calls.push(ToolCallDelta {
                    index: idx as u32,
                    id: Some(id.clone()),
                    kind: Some(kind.clone()),
                    function: Some(ToolCallFunctionDelta {
                        name: Some(name.clone()),
                        arguments: Some(a.to_string()),
                    }),
                    thought_signature: None,
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
                    thought_signature: None,
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
    use crate::llm::types::{
        ChatCompletionRequest, ChatMessage, ConversationRequest, FunctionDefinitionWire,
        HostedTool, SearchParameters, ToolDefinitionWire,
    };

    fn conv(req: ChatCompletionRequest) -> ConversationRequest {
        ConversationRequest::from_chat(req)
    }

    #[test]
    fn test_api_backend_endpoint_resolution() {
        let base_url = "https://api.x.ai/v1";

        let t_chat = HttpTransport::new(base_url, "key", Duration::from_secs(10), ApiBackend::ChatCompletions, std::collections::HashMap::new()).unwrap();
        assert_eq!(t_chat.endpoint(), "https://api.x.ai/v1/chat/completions");

        let t_resp = HttpTransport::new(base_url, "key", Duration::from_secs(10), ApiBackend::Responses, std::collections::HashMap::new()).unwrap();
        assert_eq!(t_resp.endpoint(), "https://api.x.ai/v1/responses");

        let t_msg = HttpTransport::new(base_url, "key", Duration::from_secs(10), ApiBackend::Messages, std::collections::HashMap::new()).unwrap();
        assert_eq!(t_msg.endpoint(), "https://api.x.ai/v1/messages");

        let t_gem = HttpTransport::new(
            "https://generativelanguage.googleapis.com/v1beta",
            "key",
            Duration::from_secs(10),
            ApiBackend::Gemini,
            std::collections::HashMap::new(),
        )
        .unwrap();
        assert_eq!(
            t_gem.endpoint_for_model("gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
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
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };

        let payload = build_responses_payload(&conv(req)).unwrap();
        assert_eq!(payload["model"], "grok-3");
        assert_eq!(payload["instructions"], "You are helpful.");
        assert_eq!(payload["input"][0]["role"], "user");
        assert_eq!(payload["input"][0]["content"], "Hello!");
        assert_eq!(payload["tools"][0]["name"], "test_tool");
        assert!(payload.get("search_parameters").is_none());
        assert!(payload.get("previous_response_id").is_none());
    }

    #[test]
    fn responses_payload_includes_previous_response_id() {
        let mut req = ChatCompletionRequest {
            model: "grok-4.6".into(),
            messages: vec![ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: Some("resp_abc".into()),
        };
        let payload = build_responses_payload(&conv(req.clone())).unwrap();
        assert_eq!(payload["previous_response_id"], "resp_abc");
        req.previous_response_id = Some(String::new());
        let payload = build_responses_payload(&conv(req)).unwrap();
        assert!(payload.get("previous_response_id").is_none());
    }

    #[test]
    fn parse_responses_captures_resp_id_from_response_created() {
        let raw = r#"{"type":"response.created","response":{"id":"resp_xyz","status":"in_progress"}}"#;
        let chunk = parse_responses_chunk("response.created", raw)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.id, "resp_xyz");
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn responses_payload_never_sends_search_parameters() {
        let req = ChatCompletionRequest {
            model: "grok-4.6".into(),
            messages: vec![ChatMessage::user("你好")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: Some(SearchParameters {
                mode: Some("auto".into()),
                ..Default::default()
            }),
            hosted_tools: vec![],
            previous_response_id: None,
        };
        let payload = build_responses_payload(&conv(req)).unwrap();
        assert!(payload.get("search_parameters").is_none());
        assert!(payload.get("tools").is_none());
    }

    #[test]
    fn responses_payload_splices_hosted_web_search_and_drops_function_collision() {
        let req = ChatCompletionRequest {
            model: "grok-4.6".into(),
            messages: vec![ChatMessage::user("你好")],
            stream: Some(true),
            tools: Some(vec![
                ToolDefinitionWire {
                    kind: "function".into(),
                    function: FunctionDefinitionWire {
                        name: "web_search".into(),
                        description: Some("client search".into()),
                        parameters: serde_json::json!({"type": "object"}),
                    },
                },
                ToolDefinitionWire {
                    kind: "function".into(),
                    function: FunctionDefinitionWire {
                        name: "read_file".into(),
                        description: Some("read".into()),
                        parameters: serde_json::json!({"type": "object"}),
                    },
                },
            ]),
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![HostedTool::WebSearch],
            previous_response_id: None,
        };
        let payload = build_responses_payload(&conv(req)).unwrap();
        let tools = payload["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "web_search");
        assert!(tools[0].get("name").is_none());
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[1]["type"], "function");
        assert_eq!(tools[1]["name"], "read_file");
        assert!(payload.get("search_parameters").is_none());
    }

    #[test]
    fn hosted_search_tools_only_on_responses() {
        assert!(HostedTool::for_request(true, ApiBackend::Responses) == vec![HostedTool::WebSearch]);
        assert!(HostedTool::for_request(true, ApiBackend::ChatCompletions).is_empty());
        assert!(HostedTool::for_request(true, ApiBackend::Messages).is_empty());
        assert!(HostedTool::for_request(true, ApiBackend::Gemini).is_empty());
        assert!(HostedTool::for_request(false, ApiBackend::Responses).is_empty());
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
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };

        let payload = build_messages_payload(&conv(req)).unwrap();
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
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };

        let payload = build_responses_payload(&conv(req)).unwrap();
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
    fn responses_payload_requests_concise_reasoning_summary() {
        let req = ChatCompletionRequest {
            model: "grok-4.6".into(),
            messages: vec![ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };
        let payload = build_responses_payload(&conv(req)).unwrap();
        assert_eq!(payload["reasoning"]["summary"], "concise");
        assert!(payload["reasoning"].get("effort").is_none());
    }

    #[test]
    fn responses_payload_includes_reasoning_effort_when_configured() {
        let req = ChatCompletionRequest {
            model: "grok-4.6".into(),
            messages: vec![ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: Some(crate::llm::types::ReasoningEffort::High),
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };
        let payload = build_responses_payload(&conv(req)).unwrap();
        assert_eq!(payload["reasoning"]["summary"], "concise");
        assert_eq!(payload["reasoning"]["effort"], "high");
    }

    #[test]
    fn chat_completions_and_messages_payload_carries_reasoning_effort() {
        let req = ChatCompletionRequest {
            model: "o3-mini".into(),
            messages: vec![ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: Some(crate::llm::types::ReasoningEffort::Medium),
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };
        let cc_payload = crate::llm::wire::chat_completions::build_chat_completions_payload(&conv(req.clone())).unwrap();
        assert_eq!(cc_payload["reasoning_effort"], "medium");

        let msg_payload = crate::llm::wire::messages::build_messages_payload(&conv(req)).unwrap();
        assert_eq!(msg_payload["output_config"]["effort"], "medium");
        assert_eq!(msg_payload["thinking"]["type"], "adaptive");
    }

    #[test]
    fn messages_payload_model_aware_effort_opus_and_sonnet() {
        // Opus 4.6 allows 'max' effort
        let req_opus = ChatCompletionRequest {
            model: "claude-opus-4-6".into(),
            messages: vec![ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: Some(crate::llm::types::ReasoningEffort::Max),
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };
        let opus_payload = crate::llm::wire::messages::build_messages_payload(&conv(req_opus)).unwrap();
        assert_eq!(opus_payload["output_config"]["effort"], "max");
        assert_eq!(opus_payload["thinking"]["type"], "adaptive");

        // Sonnet 4.6 clamps 'max' and 'xhigh' to 'high' for Claude Code / API compatibility
        let req_sonnet = ChatCompletionRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: Some(crate::llm::types::ReasoningEffort::Max),
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };
        let sonnet_payload = crate::llm::wire::messages::build_messages_payload(&conv(req_sonnet)).unwrap();
        assert_eq!(sonnet_payload["output_config"]["effort"], "high");
        assert_eq!(sonnet_payload["thinking"]["type"], "adaptive");

        // None reasoning_effort omits output_config and thinking
        let req_none = ChatCompletionRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: Some(crate::llm::types::ReasoningEffort::None),
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };
        let none_payload = crate::llm::wire::messages::build_messages_payload(&conv(req_none)).unwrap();
        assert!(none_payload.get("output_config").is_none());
        assert!(none_payload.get("thinking").is_none());
    }

    #[test]
    fn responses_payload_always_includes_reasoning_summary_field() {
        // grok-build `test_responses_api_with_only_encrypted_reasoning`:
        // empty summary is valid and must serialize as `[]`, never omitted.
        let req = ChatCompletionRequest {
            model: "grok-4.6".into(),
            messages: vec![
                ChatMessage::user("你会做什么"),
                ChatMessage::assistant_with_reasoning(
                    "我是 PhoneBuddy",
                    None,
                    vec![crate::llm::types::ReasoningItem {
                        id: "rs_added_stub".into(),
                        summary: Vec::new(),
                        content: None,
                        encrypted_content: Some("enc".into()),
                        status: None,
                    }],
                    Some("enc".into()),
                ),
                ChatMessage::user("帮我查一下apus这家公司"),
            ],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };

        let payload = build_responses_payload(&conv(req)).unwrap();
        let input = payload["input"].as_array().unwrap();
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["id"], "rs_added_stub");
        assert!(
            input[1].get("summary").is_some(),
            "summary must be present, got {}",
            input[1]
        );
        assert_eq!(input[1]["summary"], serde_json::json!([]));
        assert_eq!(input[1]["encrypted_content"], "enc");
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[3]["role"], "user");
    }

    #[test]
    fn responses_payload_does_not_invent_summary_from_reasoning_content() {
        // grok-build fills summary at collect time via
        // `inject_streaming_reasoning_fallback`, not at wire build.
        // An empty-summary sibling stays `summary: []`.
        let req = ChatCompletionRequest {
            model: "grok-4.6".into(),
            messages: vec![
                ChatMessage::user("hi"),
                ChatMessage::assistant_with_reasoning(
                    "hello",
                    Some("I should greet the user".into()),
                    vec![crate::llm::types::ReasoningItem {
                        id: "rs_1".into(),
                        summary: Vec::new(),
                        content: None,
                        encrypted_content: None,
                        status: None,
                    }],
                    None,
                ),
            ],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };

        let payload = build_responses_payload(&conv(req)).unwrap();
        let input = payload["input"].as_array().unwrap();
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["summary"], serde_json::json!([]));
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

        // grok-build ignores Reasoning on OutputItemAdded.
        let added = r#"{
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_stub",
                "summary": []
            }
        }"#;
        let added_chunk = parse_responses_chunk("response.output_item.added", added).unwrap();
        assert!(
            added_chunk.is_none()
                || added_chunk
                    .as_ref()
                    .is_some_and(|c| c.choices[0].delta.reasoning_items.is_empty()),
            "output_item.added reasoning stub must not be persisted"
        );
    }

    #[test]
    fn test_responses_chunk_parses_function_call_output_item() {
        let added = r#"{
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": "c0",
                "name": "web_fetch",
                "arguments": ""
            }
        }"#;
        let chunk = parse_responses_chunk("response.output_item.added", added)
            .unwrap()
            .unwrap();
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(tc.index, 1);
        assert_eq!(tc.id.as_deref(), Some("c0"));
        assert_eq!(
            tc.function.as_ref().and_then(|f| f.name.as_deref()),
            Some("web_fetch")
        );

        let done = r#"{
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": "c0",
                "name": "web_fetch",
                "arguments": "{\"url\":\"https://news.sina.com.cn\"}"
            }
        }"#;
        let chunk = parse_responses_chunk("response.output_item.done", done)
            .unwrap()
            .unwrap();
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(
            tc.function.as_ref().and_then(|f| f.arguments.as_deref()),
            Some("{\"url\":\"https://news.sina.com.cn\"}")
        );
    }

    #[test]
    fn test_responses_web_search_added_is_id_and_name_only() {
        // grok-build: OutputItemAdded(WebSearchCall) is not the payload.
        // Grok often puts a placeholder action
        // `{"type":"search","query":"","sources":[]}` on added; query and
        // sources arrive on OutputItemDone.
        let added = r#"{
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "action": {"type":"search","query":"","sources":[]}
            }
        }"#;
        let chunk = parse_responses_chunk("response.output_item.added", added)
            .unwrap()
            .unwrap();
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(tc.index, 2);
        assert_eq!(tc.id.as_deref(), Some("ws_1"));
        assert_eq!(tc.kind.as_deref(), Some("server"));
        assert_eq!(
            tc.function.as_ref().and_then(|f| f.name.as_deref()),
            Some("web_search")
        );
        assert!(
            tc.function
                .as_ref()
                .and_then(|f| f.arguments.as_deref())
                .is_none_or(|a| a.is_empty()),
            "added must not carry the placeholder action as arguments: {:?}",
            tc.function.as_ref().and_then(|f| f.arguments.as_ref())
        );
    }

    #[test]
    fn test_responses_web_search_done_carries_action_query() {
        let done = r#"{
            "type": "response.output_item.done",
            "output_index": 2,
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "today news",
                    "sources": [{"type":"url","url":"https://example.com"}]
                }
            }
        }"#;
        let chunk = parse_responses_chunk("response.output_item.done", done)
            .unwrap()
            .unwrap();
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(tc.id.as_deref(), Some("ws_1"));
        assert_eq!(
            tc.function.as_ref().and_then(|f| f.name.as_deref()),
            Some("web_search")
        );
        let args = tc
            .function
            .as_ref()
            .and_then(|f| f.arguments.as_deref())
            .unwrap_or("");
        assert!(args.contains("today news"), "done args: {args}");
        assert!(args.contains("https://example.com"), "done args: {args}");
    }

    #[test]
    fn test_responses_function_call_arguments_done_is_ignored() {
        // grok-build drops ResponseFunctionCallArgumentsDone: the full
        // JSON is a snapshot of already-streamed deltas. Treating it as
        // another fragment concatenates `{...}{...}`.
        let done = r#"{
            "type": "response.function_call_arguments.done",
            "output_index": 1,
            "call_id": "nav",
            "arguments": "{\"url\": \"https://news.cctv.com/\"}"
        }"#;
        let chunk = parse_responses_chunk("response.function_call_arguments.done", done).unwrap();
        assert!(
            chunk.is_none()
                || chunk
                    .as_ref()
                    .map(|c| c.choices[0].delta.tool_calls.is_empty())
                    .unwrap_or(true),
            "function_call_arguments.done must not be appended as a delta"
        );
    }

    #[test]
    fn test_responses_function_call_arguments_delta_uses_output_index() {
        let delta = r#"{
            "type": "response.function_call_arguments.delta",
            "output_index": 3,
            "call_id": "c2",
            "delta": "{\"q\":"
        }"#;
        let chunk = parse_responses_chunk("response.function_call_arguments.delta", delta)
            .unwrap()
            .unwrap();
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(tc.index, 3);
        assert_eq!(tc.id.as_deref(), Some("c2"));
        assert_eq!(
            tc.function.as_ref().and_then(|f| f.arguments.as_deref()),
            Some("{\"q\":")
        );
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
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };

        let mut extra = std::collections::HashMap::new();
        extra.insert("custom_app_id".to_string(), serde_json::json!("org.example.app"));
        extra.insert("client_version".to_string(), serde_json::json!("1.0.0"));
        extra.insert("user_tier".to_string(), serde_json::json!("premium"));

        // Test Responses API payload
        let mut resp_payload = build_responses_payload(&conv(req.clone())).unwrap();
        merge_extra_body(&mut resp_payload, &extra);
        assert_eq!(resp_payload["custom_app_id"], "org.example.app");
        assert_eq!(resp_payload["client_version"], "1.0.0");
        assert_eq!(resp_payload["user_tier"], "premium");
        assert_eq!(resp_payload["model"], "claude-3-5-sonnet");

        // Test Messages API payload
        let mut msg_payload = build_messages_payload(&conv(req)).unwrap();
        merge_extra_body(&mut msg_payload, &extra);
        assert_eq!(msg_payload["custom_app_id"], "org.example.app");
        assert_eq!(msg_payload["client_version"], "1.0.0");
        assert_eq!(msg_payload["user_tier"], "premium");
        assert_eq!(msg_payload["model"], "claude-3-5-sonnet");
    }

    #[tokio::test]
    async fn test_http_transport_dump_on_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dump_dir = tmp.path().join("dumps");

        let dumper = crate::llm::dumper::HttpDumper::new(
            crate::llm::dumper::HttpDumpConfig {
                mode: crate::llm::dumper::HttpDumpMode::OnError,
                dump_dir: Some(dump_dir.clone()),
                mask_sensitive: true,
                max_files: 10,
            },
            dump_dir.clone(),
        );

        // Point to an invalid local port that will fail connection
        let transport = HttpTransport::new_with_all_options(
            "http://127.0.0.1:59999/v1",
            "test-secret-key-123456",
            Duration::from_secs(2),
            ApiBackend::ChatCompletions,
            ClientProfile::Default,
            None,
            None,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            false,
            dumper,
        )
        .unwrap();


        let req = ChatCompletionRequest {
            model: "test-model".into(),
            messages: vec![ChatMessage::user("Hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };

        let res = transport.request_stream(&conv(req)).await;
        assert!(res.is_err());
        let err = res.err().unwrap();
        let err_str = err.to_string();
        assert!(err_str.contains("HTTP dump:"));

        // Verify that dump file was created and contains masked auth header
        let entries: Vec<_> = std::fs::read_dir(&dump_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        let dump_content = std::fs::read_to_string(entries[0].path()).unwrap();
        let json: serde_json::Value = serde_json::from_str(&dump_content).unwrap();
        assert_eq!(json["request"]["method"], "POST");
        assert_eq!(json["request"]["headers"]["authorization"], "Bearer test***3456");
        assert!(json["error"].is_string());
    }

    #[tokio::test]
    async fn test_http_transport_claude_code_profile_headers() {
        let tmp = tempfile::tempdir().unwrap();
        let dump_dir = tmp.path().join("dumps");

        let dumper = crate::llm::dumper::HttpDumper::new(
            crate::llm::dumper::HttpDumpConfig {
                mode: crate::llm::dumper::HttpDumpMode::OnError,
                dump_dir: Some(dump_dir.clone()),
                mask_sensitive: false,
                max_files: 10,
            },
            dump_dir.clone(),
        );

        let transport = HttpTransport::new_with_all_options(
            "http://127.0.0.1:59999/v1",
            "sk-ant-test-key",
            Duration::from_secs(2),
            ApiBackend::Messages,
            ClientProfile::ClaudeCode,
            Some("2.1.238".into()),
            Some("sess-claude-999".into()),
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            false,
            dumper,
        )
        .unwrap();

        let req = ChatCompletionRequest {
            model: "claude-opus-5".into(),
            messages: vec![ChatMessage::user("Hi")],

            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
        };

        let res = transport.request_stream(&conv(req)).await;
        assert!(res.is_err());

        let entries: Vec<_> = std::fs::read_dir(&dump_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        let dump_content = std::fs::read_to_string(entries[0].path()).unwrap();
        let json: serde_json::Value = serde_json::from_str(&dump_content).unwrap();

        assert_eq!(json["request"]["headers"]["x-api-key"], "sk-ant-test-key");
        assert_eq!(json["request"]["headers"]["anthropic-version"], "2023-06-01");
        assert_eq!(json["request"]["headers"]["x-app"], "cli");
        assert_eq!(
            json["request"]["headers"]["x-claude-code-session-id"],
            "sess-claude-999"
        );
        assert_eq!(
            json["request"]["headers"]["user-agent"],
            "claude-cli/2.1.238 (external, cli)"
        );
    }
}

