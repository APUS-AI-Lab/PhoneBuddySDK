//! Host-provided system WebView fetch.
//!
//! Mobile wrappers (WKWebView / Android WebView) register a notify callback
//! and complete each request with HTML. Desktop / C hosts leave this unset
//! so `web_search` skips scraping and uses the LLM search API instead.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Notifies the host that a WebView fetch is needed.
/// Arguments: `(call_id, request_json)`.
pub type WebViewFetchNotify = Arc<dyn Fn(String, String) + Send + Sync>;

/// Request payload sent to the host WebView.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebViewFetchRequest {
    pub url: String,
    pub method: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub body: String,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    20_000
}

impl WebViewFetchRequest {
    pub fn post_form(url: impl Into<String>, body: impl Into<String>, timeout_ms: u64) -> Self {
        let mut headers = HashMap::new();
        headers.insert(
            "Content-Type".into(),
            "application/x-www-form-urlencoded".into(),
        );
        Self {
            url: url.into(),
            method: "POST".into(),
            headers,
            body: body.into(),
            timeout_ms,
        }
    }
}

/// Pending host WebView fetches.
pub struct WebViewHost {
    notify: Mutex<Option<WebViewFetchNotify>>,
    pending: Mutex<HashMap<String, tokio::sync::oneshot::Sender<Result<String, String>>>>,
}

impl WebViewHost {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            notify: Mutex::new(None),
            pending: Mutex::new(HashMap::new()),
        })
    }

    pub fn set_notify(&self, cb: WebViewFetchNotify) {
        *self.notify.lock().unwrap() = Some(cb);
    }

    pub fn clear_notify(&self) {
        *self.notify.lock().unwrap() = None;
    }

    /// True when a mobile host has registered a system WebView.
    pub fn is_available(&self) -> bool {
        self.notify.lock().unwrap().is_some()
    }

    /// Ask the host WebView to load `request` and return the document HTML.
    pub async fn fetch(
        &self,
        request: WebViewFetchRequest,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<String, String> {
        let notify = self
            .notify
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "host WebView is not available".to_string())?;

        let call_id = Uuid::new_v4().to_string();
        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("failed to serialize WebView request: {e}"))?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.lock().unwrap().insert(call_id.clone(), tx);

        notify(call_id.clone(), request_json);

        let timeout = Duration::from_millis(request.timeout_ms.saturating_add(2_000));
        tokio::select! {
            _ = cancel.cancelled() => {
                self.pending.lock().unwrap().remove(&call_id);
                Err("WebView fetch cancelled".into())
            }
            _ = tokio::time::sleep(timeout) => {
                self.pending.lock().unwrap().remove(&call_id);
                Err(format!(
                    "WebView fetch timed out after {} ms",
                    timeout.as_millis()
                ))
            }
            res = rx => {
                match res {
                    Ok(Ok(html)) => Ok(html),
                    Ok(Err(msg)) => Err(msg),
                    Err(_) => Err("WebView result channel closed".into()),
                }
            }
        }
    }

    /// Host finished a WebView fetch. `ok` is false when the host failed.
    pub fn complete(&self, call_id: &str, ok: bool, output: impl Into<String>) -> Result<(), String> {
        let mut pending = self.pending.lock().unwrap();
        let Some(tx) = pending.remove(call_id) else {
            return Err(format!("unknown WebView call_id: {call_id}"));
        };
        let output = output.into();
        let payload = if ok { Ok(output) } else { Err(output) };
        tx.send(payload)
            .map_err(|_| format!("WebView call {call_id} receiver dropped"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn webview_round_trip() {
        let host = WebViewHost::new();
        assert!(!host.is_available());

        let host_c = host.clone();
        host.set_notify(Arc::new(move |call_id, request_json| {
            assert!(request_json.contains("lite.duckduckgo.com"));
            let host = host_c.clone();
            std::thread::spawn(move || {
                let _ = host.complete(&call_id, true, "<html>ok</html>");
            });
        }));
        assert!(host.is_available());

        let cancel = tokio_util::sync::CancellationToken::new();
        let html = host
            .fetch(
                WebViewFetchRequest::post_form(
                    "https://lite.duckduckgo.com/lite/",
                    "q=test",
                    5_000,
                ),
                &cancel,
            )
            .await
            .unwrap();
        assert_eq!(html, "<html>ok</html>");
    }

    #[tokio::test]
    async fn webview_unavailable() {
        let host = WebViewHost::new();
        let cancel = tokio_util::sync::CancellationToken::new();
        let err = host
            .fetch(
                WebViewFetchRequest::post_form("https://example.com", "", 1_000),
                &cancel,
            )
            .await
            .unwrap_err();
        assert!(err.contains("not available"));
    }
}
