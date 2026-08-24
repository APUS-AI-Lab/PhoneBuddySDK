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
use crate::llm::types::{ChatCompletionChunk, ChatCompletionRequest};

/// Notifies the host that a new LLM request is ready.
/// Arguments: `(request_id, request_json)`.
pub type HostLlmNotify = Arc<dyn Fn(String, String) + Send + Sync>;

/// Shared hub between the transport and the FFI push APIs.
pub struct HostLlmHub {
    notify: Mutex<Option<HostLlmNotify>>,
    pending:
        Mutex<HashMap<String, mpsc::UnboundedSender<Result<ChatCompletionChunk, EngineError>>>>,
}

impl HostLlmHub {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            notify: Mutex::new(None),
            pending: Mutex::new(HashMap::new()),
        })
    }

    /// Register (or replace) the host notify callback.
    pub fn set_notify(&self, cb: HostLlmNotify) {
        *self.notify.lock().unwrap() = Some(cb);
    }

    /// Clear the notify callback (engine teardown).
    pub fn clear_notify(&self) {
        *self.notify.lock().unwrap() = None;
    }

    /// Begin a host LLM request: register a channel, fire notify, return the
    /// receiver the transport will stream from.
    fn begin(
        &self,
        req: &ChatCompletionRequest,
    ) -> EngineResult<mpsc::UnboundedReceiver<Result<ChatCompletionChunk, EngineError>>> {
        let notify = self
            .notify
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| EngineError::Llm("host LLM notify callback is not set".into()))?;

        let request_id = Uuid::new_v4().to_string();
        let request_json = serde_json::to_string(req)
            .map_err(|e| EngineError::Llm(format!("serialize request: {e}")))?;

        let (tx, rx) = mpsc::unbounded_channel();
        self.pending
            .lock()
            .unwrap()
            .insert(request_id.clone(), tx);

        // Host may call push_chunk from another thread immediately.
        notify(request_id, request_json);
        Ok(rx)
    }

    /// Push one streaming chunk for an in-flight request.
    pub fn push_chunk(&self, request_id: &str, chunk: ChatCompletionChunk) -> Result<(), String> {
        let pending = self.pending.lock().unwrap();
        let Some(tx) = pending.get(request_id) else {
            return Err(format!("unknown LLM request_id: {request_id}"));
        };
        tx.send(Ok(chunk))
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
        let Some(tx) = pending.remove(request_id) else {
            return Err(format!("unknown LLM request_id: {request_id}"));
        };
        let _ = tx.send(Err(EngineError::Llm(message.into())));
        Ok(())
    }

    /// Drop all pending streams (e.g. on cancel/teardown).
    pub fn abort_all(&self, message: &str) {
        let mut pending = self.pending.lock().unwrap();
        for (_id, tx) in pending.drain() {
            let _ = tx.send(Err(EngineError::Llm(message.to_string())));
        }
    }
}

/// Transport that delegates every completion to the host via [`HostLlmHub`].
pub struct HostLlmTransport {
    hub: Arc<HostLlmHub>,
}

impl HostLlmTransport {
    pub fn new(hub: Arc<HostLlmHub>) -> Self {
        Self { hub }
    }

    pub fn hub(&self) -> &Arc<HostLlmHub> {
        &self.hub
    }
}

impl LlmTransport for HostLlmTransport {
    async fn request_stream(&self, req: &ChatCompletionRequest) -> EngineResult<ChunkStream> {
        let mut rx = self.hub.begin(req)?;
        let stream = async_stream::stream! {
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

    fn name(&self) -> &str {
        "host"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatChunkChoice, ChatChunkDelta, Role};
    use futures_util::StreamExt;

    fn sample_req() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "local".into(),
            messages: vec![crate::llm::types::ChatMessage::user("hi")],
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: Some(0.2),
            max_tokens: Some(128),
            search_parameters: None,
            hosted_tools: vec![],
            previous_response_id: None,
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
    async fn host_transport_errors_without_notify() {
        let hub = HostLlmHub::new();
        let transport = HostLlmTransport::new(hub);
        match transport.request_stream(&sample_req()).await {
            Ok(_) => panic!("expected error without notify"),
            Err(e) => assert!(e.to_string().contains("notify"), "{e}"),
        }
    }
}
