//! Host-provided LLM transport.
//!
//! The engine does not call HTTP. Instead it notifies the host (via a
//! callback registered through FFI) with a `request_id` + serialized
//! [`ChatCompletionRequest`]. The host streams OpenAI-compatible chunks
//! back with [`HostLlmHub::push_chunk`] and closes the stream with
//! [`HostLlmHub::finish`] or [`HostLlmHub::fail`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use uuid::Uuid;

use crate::error::{EngineError, EngineResult};
use crate::llm::transport::{ChunkStream, LlmTransport};
use crate::llm::types::{ChatCompletionChunk, ConversationRequest};

/// Notifies the host that a new LLM request is ready.
/// Arguments: `(request_id, envelope_or_request_json)`.
pub type HostLlmNotify = Arc<dyn Fn(String, String) + Send + Sync>;

/// Notifies the host that an in-flight request was cancelled or dropped.
pub type HostLlmCancelNotify = Arc<dyn Fn(String) + Send + Sync>;

/// Metadata attached to a host LLM request envelope.
#[derive(Debug, Clone, Default)]
pub struct HostRequestMeta {
    pub operation_id: String,
    pub pool_id: String,
    pub provider_id: String,
    pub deadline_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HostLlmUnavailableBody {
    pub code: String,
    #[serde(default = "default_provider_scope")]
    pub scope: String,
    #[serde(default = "default_true")]
    pub before_output: bool,
    #[serde(default)]
    pub message: String,
}

fn default_provider_scope() -> String {
    "provider".into()
}

fn default_true() -> bool {
    true
}

struct PendingHostStream {
    tx: mpsc::UnboundedSender<Result<ChatCompletionChunk, EngineError>>,
    operation_id: String,
}

/// Shared hub between the transport and the FFI push APIs.
pub struct HostLlmHub {
    notify: Mutex<Option<HostLlmNotify>>,
    cancel_notify: Mutex<Option<HostLlmCancelNotify>>,
    pending: Mutex<HashMap<String, PendingHostStream>>,
}

impl HostLlmHub {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            notify: Mutex::new(None),
            cancel_notify: Mutex::new(None),
            pending: Mutex::new(HashMap::new()),
        })
    }

    /// Register (or replace) the host notify callback.
    pub fn set_notify(&self, cb: HostLlmNotify) {
        *self.notify.lock().unwrap() = Some(cb);
    }

    pub fn set_cancel_notify(&self, cb: HostLlmCancelNotify) {
        *self.cancel_notify.lock().unwrap() = Some(cb);
    }

    /// Clear the notify callback (engine teardown).
    pub fn clear_notify(&self) {
        *self.notify.lock().unwrap() = None;
        *self.cancel_notify.lock().unwrap() = None;
    }

    /// Begin a host LLM request: register a channel, fire notify, return the
    /// receiver the transport will stream from.
    #[cfg(test)]
    fn begin(
        &self,
        req: &ConversationRequest,
    ) -> EngineResult<mpsc::UnboundedReceiver<Result<ChatCompletionChunk, EngineError>>> {
        self.begin_with_meta(req, HostRequestMeta::default())
            .map(|(_, rx)| rx)
    }

    fn begin_with_meta(
        &self,
        req: &ConversationRequest,
        meta: HostRequestMeta,
    ) -> EngineResult<(
        String,
        mpsc::UnboundedReceiver<Result<ChatCompletionChunk, EngineError>>,
    )> {
        let notify = self
            .notify
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| EngineError::Llm("host LLM notify callback is not set".into()))?;

        let request_id = Uuid::new_v4().to_string();
        let request_json = crate::llm::wire::chat_completions::host_request_json(req)
            .map_err(|e| EngineError::Llm(format!("serialize request: {e}")))?;
        let inner: serde_json::Value = serde_json::from_str(&request_json)
            .unwrap_or_else(|_| serde_json::Value::String(request_json.clone()));
        let envelope = serde_json::json!({
            "request_id": request_id,
            "operation_id": meta.operation_id,
            "pool_id": meta.pool_id,
            "provider_id": meta.provider_id,
            "deadline_unix_ms": meta.deadline_unix_ms,
            "request_json": inner,
        });
        let envelope_json = serde_json::to_string(&envelope)
            .map_err(|e| EngineError::Llm(format!("serialize host envelope: {e}")))?;

        let (tx, rx) = mpsc::unbounded_channel();
        self.pending.lock().unwrap().insert(
            request_id.clone(),
            PendingHostStream {
                tx,
                operation_id: meta.operation_id,
            },
        );

        // Host may call push_chunk from another thread immediately.
        notify(request_id.clone(), envelope_json);
        Ok((request_id, rx))
    }

    /// Push one streaming chunk for an in-flight request.
    pub fn push_chunk(&self, request_id: &str, chunk: ChatCompletionChunk) -> Result<(), String> {
        let pending = self.pending.lock().unwrap();
        let Some(pending) = pending.get(request_id) else {
            return Err(format!("unknown LLM request_id: {request_id}"));
        };
        pending
            .tx
            .send(Ok(chunk))
            .map_err(|_| format!("LLM request {request_id} receiver dropped"))?;
        Ok(())
    }

    /// Signal successful end-of-stream (drop the sender).
    pub fn finish(&self, request_id: &str) -> Result<(), String> {
        let mut pending = self.pending.lock().unwrap();
        if pending.remove(request_id).is_none() {
            return Err(format!("unknown LLM request_id: {request_id}"));
        }
        Ok(())
    }

    /// Fail an in-flight request and close the stream.
    pub fn fail(&self, request_id: &str, message: impl Into<String>) -> Result<(), String> {
        let mut pending = self.pending.lock().unwrap();
        let Some(pending) = pending.remove(request_id) else {
            return Err(format!("unknown LLM request_id: {request_id}"));
        };
        let _ = pending.tx.send(Err(host_fail_error(&message.into())));
        Ok(())
    }

    /// Drop all pending streams (e.g. on cancel/teardown).
    pub fn abort_all(&self, message: &str) {
        let mut pending = self.pending.lock().unwrap();
        let ids: Vec<String> = pending.keys().cloned().collect();
        for (_id, stream) in pending.drain() {
            let _ = stream
                .tx
                .send(Err(EngineError::Llm(message.to_string())));
        }
        drop(pending);
        let cancel = self.cancel_notify.lock().unwrap().clone();
        if let Some(cancel) = cancel {
            for id in ids {
                cancel(id);
            }
        }
    }

    pub fn abort_operation(&self, operation_id: &str, message: &str) {
        let mut pending = self.pending.lock().unwrap();
        let ids: Vec<String> = pending
            .iter()
            .filter(|(_, p)| p.operation_id == operation_id)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &ids {
            if let Some(stream) = pending.remove(id) {
                let _ = stream
                    .tx
                    .send(Err(EngineError::Llm(message.to_string())));
            }
        }
        drop(pending);
        let cancel = self.cancel_notify.lock().unwrap().clone();
        if let Some(cancel) = cancel {
            for id in ids {
                cancel(id);
            }
        }
    }

    fn on_receiver_dropped(&self, request_id: &str) {
        let removed = self.pending.lock().unwrap().remove(request_id).is_some();
        if removed {
            if let Some(cancel) = self.cancel_notify.lock().unwrap().clone() {
                cancel(request_id.to_string());
            }
        }
    }
}

fn host_fail_error(message: &str) -> EngineError {
    if let Ok(body) = serde_json::from_str::<HostLlmUnavailableBody>(message) {
        if !body.code.is_empty() {
            return EngineError::HostUnavailable {
                code: body.code,
                scope: body.scope,
                before_output: body.before_output,
                message: body.message,
            };
        }
    }
    EngineError::Llm(message.to_string())
}

/// Transport that delegates every completion to the host via [`HostLlmHub`].
pub struct HostLlmTransport {
    hub: Arc<HostLlmHub>,
    provider_id: String,
}

impl HostLlmTransport {
    pub fn new(hub: Arc<HostLlmHub>) -> Self {
        Self {
            hub,
            provider_id: String::new(),
        }
    }

    pub fn with_provider_id(hub: Arc<HostLlmHub>, provider_id: impl Into<String>) -> Self {
        Self {
            hub,
            provider_id: provider_id.into(),
        }
    }

    pub fn hub(&self) -> &Arc<HostLlmHub> {
        &self.hub
    }
}

struct HostStreamGuard {
    hub: Arc<HostLlmHub>,
    request_id: String,
}

impl Drop for HostStreamGuard {
    fn drop(&mut self) {
        self.hub.on_receiver_dropped(&self.request_id);
    }
}

impl LlmTransport for HostLlmTransport {
    async fn request_stream(&self, req: &ConversationRequest) -> EngineResult<ChunkStream> {
        self.request_stream_with_meta(req, HostRequestMeta {
            provider_id: self.provider_id.clone(),
            ..Default::default()
        })
        .await
    }

    fn request_stream_in_context<'a>(
        &'a self,
        req: &'a ConversationRequest,
        context: &'a crate::llm::transport::LlmTurnContext,
    ) -> futures_util::future::BoxFuture<'a, EngineResult<ChunkStream>> {
        let meta = HostRequestMeta {
            operation_id: context.operation_id().unwrap_or_default(),
            pool_id: context.pool_id().unwrap_or_default(),
            provider_id: if self.provider_id.is_empty() {
                context.provider_id().unwrap_or_default()
            } else {
                self.provider_id.clone()
            },
            deadline_unix_ms: context.deadline_unix_ms(),
        };
        Box::pin(self.request_stream_with_meta(req, meta))
    }

    fn name(&self) -> &str {
        "host"
    }
}

impl HostLlmTransport {
    async fn request_stream_with_meta(
        &self,
        req: &ConversationRequest,
        meta: HostRequestMeta,
    ) -> EngineResult<ChunkStream> {
        let (request_id, mut rx) = self.hub.begin_with_meta(req, meta)?;
        let guard = HostStreamGuard {
            hub: self.hub.clone(),
            request_id,
        };
        let stream = async_stream::stream! {
            let _guard = guard;
            while let Some(item) = rx.recv().await {
                match item {
                    Ok(chunk) => yield Ok(chunk),
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatChunkChoice, ChatChunkDelta, ConversationRequest, Role};
    use futures_util::StreamExt;

    fn sample_req() -> ConversationRequest {
        ConversationRequest {
            model: "local".into(),
            items: vec![crate::conversation::ConversationItem::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: Some(0.2),
            max_tokens: Some(128),
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
            response_format: None,
            image_bytes: crate::llm::image::ImageBytesStore::default(),
            audio_bytes: crate::llm::image::AudioBytesStore::default(),
        }
    }

    fn text_chunk(s: &str) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "c1".into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: "local".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatChunkDelta {
                    role: Some(Role::Assistant),
                    content: Some(s.into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    #[tokio::test]
    async fn host_transport_streams_chunks_from_hub() {
        let hub = HostLlmHub::new();
        let hub_push = hub.clone();
        hub.set_notify(Arc::new(move |req_id, _json| {
            let hub = hub_push.clone();
            let id = req_id.clone();
            std::thread::spawn(move || {
                let _ = hub.push_chunk(&id, text_chunk("hel"));
                let _ = hub.push_chunk(&id, text_chunk("lo"));
                let _ = hub.finish(&id);
            });
        }));

        let transport = HostLlmTransport::new(hub);
        let mut stream = transport.request_stream(&sample_req()).await.unwrap();
        let mut text = String::new();
        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let Some(c) = chunk.choices.first().and_then(|c| c.delta.content.clone()) {
                text.push_str(&c);
            }
        }
        assert_eq!(text, "hello");
    }

    #[tokio::test]
    async fn host_contract_unchanged() {
        use std::sync::Mutex;
        let hub = HostLlmHub::new();
        let captured = Arc::new(Mutex::new(String::new()));
        let cap = captured.clone();
        hub.set_notify(Arc::new(move |_id, json| {
            *cap.lock().unwrap() = json;
        }));
        let req = ConversationRequest {
            model: "local".into(),
            items: vec![
                crate::conversation::ConversationItem::user("hi"),
                crate::conversation::ConversationItem::Reasoning(
                    crate::llm::types::ReasoningItem {
                        id: "rs_1".into(),
                        summary: Vec::new(),
                        content: None,
                        encrypted_content: Some("enc".into()),
                        status: None,
                    },
                ),
                crate::conversation::ConversationItem::assistant("hello"),
            ],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: Some(0.2),
            max_tokens: Some(128),
            reasoning_effort: None,
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
            response_format: None,
            image_bytes: crate::llm::image::ImageBytesStore::default(),
            audio_bytes: crate::llm::image::AudioBytesStore::default(),
        };
        let _ = hub.begin(&req).unwrap();
        let json = captured.lock().unwrap().clone();
        let envelope: serde_json::Value = serde_json::from_str(&json).unwrap();
        let parsed: crate::llm::types::ChatCompletionRequest =
            serde_json::from_value(envelope["request_json"].clone()).unwrap();
        assert_eq!(parsed.model, "local");
        let inner = envelope["request_json"].to_string();
        assert!(!inner.contains("\"type\":\"reasoning\""));
        assert!(!inner.contains("backend_tool_call"));
    }

    #[tokio::test]
    async fn host_transport_errors_without_notify() {
        let hub = HostLlmHub::new();
        let transport = HostLlmTransport::new(hub);
        match transport.request_stream(&sample_req()).await {
            Ok(_) => panic!("expected error without notify"),
            Err(e) => assert!(e.to_string().contains("notify"), "{e}"),
        }
    }
}
