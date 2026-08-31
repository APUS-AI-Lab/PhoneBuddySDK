//! `web_search` tool.
//!
//! grok-build inlines hosted `{type: web_search}` on the agent Responses
//! SSE when the model exposes backend search. This client function tool
//! is the fallback used when:
//! - hosted search is off for the provider, or
//! - the stream closed while a `web_search_call` was still `in_progress`
//!   (buffering proxies) and the engine salvaged it into a client call.
//!
//! Order:
//! 1. iOS/Android: DuckDuckGo Lite in the host WebView, unless a recent
//!    failure put DDG in cooldown.
//! 2. If DDG fails and the current pool member has `enable_web_search`,
//!    a *separate* Responses request with `{type: web_search}` — the same
//!    shape as grok-build's `WebSearchClient`, not the agent stream.
//! 3. If that member has no hosted search, try later pool members that do.
//! 4. Domain filtering (`allowed_domains`, `blocked_domains`).
//! 5. Markdown hyperlink formatting `[Title](URL)` for source citations.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use serde_json::Value;

use crate::error::EngineResult;
use crate::llm::endpoint::SharedLlmEndpointProvider;
use crate::llm::types::ApiBackend;
use crate::tools::webview::{WebViewFetchRequest, WebViewHost};
use crate::tools::{
    arg_opt_str_list, arg_str, s_string, s_string_array, schema_object, truncate_chars, Tool,
    ToolCtx, ToolOutput, ToolSpec,
};

pub const DDG_LITE_URL: &str = "https://lite.duckduckgo.com/lite/";
pub const MAX_SEARCH_RESULTS: usize = 8;
pub const MAX_SNIPPET_CHARS: usize = 350;
pub const MAX_TOTAL_OUTPUT_CHARS: usize = 15_000;
pub const DDG_COOLDOWN_DURATION: Duration = Duration::from_secs(300);
pub const DDG_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Default)]
pub struct WebSearchConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub api_backend: Option<ApiBackend>,
    pub extra_headers: std::collections::HashMap<String, String>,
    pub extra_body: std::collections::HashMap<String, serde_json::Value>,
    /// This credential set may call hosted `{type: web_search}` on Responses.
    pub enable_web_search: bool,
}

pub struct WebSearchTool {
    client: reqwest::Client,
    config: WebSearchConfig,
    /// Live credentials from the currently selected pool member.
    /// Analog of grok-build `SharedApiKeyProvider`.
    endpoint_provider: Option<SharedLlmEndpointProvider>,
    webview: Arc<WebViewHost>,
    cooldown_until: Arc<Mutex<Option<Instant>>>,
    probe_enabled: bool,
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self::with_config(WebSearchConfig::default())
    }

    pub fn with_config(config: WebSearchConfig) -> Self {
        Self::with_config_and_webview(config, WebViewHost::new())
    }

    pub fn with_config_and_webview(config: WebSearchConfig, webview: Arc<WebViewHost>) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(90))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            config,
            endpoint_provider: None,
            webview,
            cooldown_until: Arc::new(Mutex::new(None)),
            probe_enabled: true,
        }
    }

    pub fn with_probe_enabled(mut self, enabled: bool) -> Self {
        self.probe_enabled = enabled;
        self
    }

    /// Attach a live credential source. Each fallback call re-reads the
    /// currently selected pool member (key, URL, backend, model).
    pub fn with_endpoint_provider(mut self, provider: SharedLlmEndpointProvider) -> Self {
        self.endpoint_provider = Some(provider);
        self
    }

    /// Static EngineConfig snapshot, overlaid by the selected pool member
    /// when an endpoint provider is attached.
    fn resolved_fallback_config(&self) -> WebSearchConfig {
        if let Some(provider) = &self.endpoint_provider {
            if let Some(ep) = provider.current_endpoint() {
                if !ep.api_key.trim().is_empty() && !ep.base_url.trim().is_empty() {
                    return Self::config_from_endpoint(&ep, &self.config);
                }
            }
        }
        self.config.clone()
    }

    fn config_from_endpoint(ep: &crate::llm::endpoint::LlmEndpoint, engine: &WebSearchConfig) -> WebSearchConfig {
        WebSearchConfig {
            api_key: Some(ep.api_key.clone()),
            base_url: Some(ep.base_url.clone()),
            model: if ep.model.trim().is_empty() {
                engine.model.clone()
            } else {
                Some(ep.model.clone())
            },
            api_backend: Some(ep.api_backend),
            extra_headers: ep.extra_headers.clone(),
            extra_body: ep.extra_body.clone(),
            enable_web_search: ep.enable_web_search,
        }
    }

    /// Pool members that may run hosted `{type: web_search}` as a
    /// separate request. Current member first, then the rest of the pool.
    fn hosted_search_candidates(&self) -> Vec<(String, WebSearchConfig)> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let push = |out: &mut Vec<(String, WebSearchConfig)>,
                    seen: &mut std::collections::HashSet<String>,
                    id: String,
                    cfg: WebSearchConfig| {
            if !cfg.enable_web_search {
                return;
            }
            if !matches!(cfg.api_backend, Some(ApiBackend::Responses)) {
                return;
            }
            if !seen.insert(id.clone()) {
                return;
            }
            out.push((id, cfg));
        };

        if let Some(provider) = &self.endpoint_provider {
            if let Some(ep) = provider.current_endpoint() {
                if !ep.api_key.trim().is_empty() && !ep.base_url.trim().is_empty() {
                    let id = ep.provider_id.clone();
                    push(&mut out, &mut seen, id, Self::config_from_endpoint(&ep, &self.config));
                }
            }
            for ep in provider.fallback_endpoints() {
                if ep.api_key.trim().is_empty() || ep.base_url.trim().is_empty() {
                    continue;
                }
                let id = ep.provider_id.clone();
                push(&mut out, &mut seen, id, Self::config_from_endpoint(&ep, &self.config));
            }
        } else {
            push(
                &mut out,
                &mut seen,
                "engine".to_string(),
                self.resolved_fallback_config(),
            );
        }
        out
    }

    pub fn from_engine_config(cfg: &crate::config::EngineConfig) -> Self {
        Self::from_engine_config_with_webview(cfg, WebViewHost::new())
    }

    pub fn from_engine_config_with_webview(
        cfg: &crate::config::EngineConfig,
        webview: Arc<WebViewHost>,
    ) -> Self {
        Self::with_config_and_webview(
            WebSearchConfig {
                api_key: if cfg.api_key.trim().is_empty() {
                    None
                } else {
                    Some(cfg.api_key.clone())
                },
                base_url: if cfg.base_url.trim().is_empty() {
                    None
                } else {
                    Some(cfg.base_url.clone())
                },
                model: if cfg.model.trim().is_empty() {
                    None
                } else {
                    Some(cfg.model.clone())
                },
                api_backend: Some(cfg.api_backend),
                extra_headers: cfg.extra_headers.clone(),
                extra_body: cfg.extra_body.clone(),
                enable_web_search: cfg.enable_web_search,
            },
            webview,
        )
    }

    pub fn is_ddg_in_cooldown(&self) -> bool {
        if let Ok(guard) = self.cooldown_until.lock() {
            if let Some(until) = *guard {
                if Instant::now() < until {
                    return true;
                }
            }
        }
        false
    }

    pub fn trigger_ddg_cooldown(&self) {
        if let Ok(mut guard) = self.cooldown_until.lock() {
            *guard = Some(Instant::now() + DDG_COOLDOWN_DURATION);
            tracing::info!(
                "[web_search] DuckDuckGo marked unreachable. Cooldown activated for {} seconds.",
                DDG_COOLDOWN_DURATION.as_secs()
            );
        }
    }

    pub fn clear_ddg_cooldown(&self) {
        if let Ok(mut guard) = self.cooldown_until.lock() {
            if guard.is_some() {
                *guard = None;
                tracing::info!("[web_search] DuckDuckGo cooldown cleared.");
            }
        }
    }

    /// Fast probe to test if DuckDuckGo server is reachable at network/TLS level.
    async fn probe_ddg_connectivity(&self, cancel: &tokio_util::sync::CancellationToken) -> bool {
        let probe_fut = self
            .client
            .get(DDG_LITE_URL)
            .timeout(DDG_PROBE_TIMEOUT)
            .header(
                reqwest::header::USER_AGENT,
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
            )
            .send();

        tokio::select! {
            _ = cancel.cancelled() => false,
            res = probe_fut => {
                match res {
                    Ok(_) => {
                        tracing::debug!("[web_search] DuckDuckGo connectivity probe succeeded");
                        true
                    }
                    Err(err) => {
                        tracing::warn!("[web_search] DuckDuckGo connectivity probe failed: {err}");
                        false
                    }
                }
            }
        }
    }

    /// Primary search on mobile: DuckDuckGo Lite via the host system WebView.
    async fn search_via_webview(
        &self,
        query: &str,
        allowed_domains: &[String],
        blocked_domains: &[String],
        ctx: &ToolCtx,
    ) -> Result<String, String> {
        let effective_query = build_ddg_query(query, allowed_domains, blocked_domains);
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("q", &effective_query)
            .finish();
        let request = WebViewFetchRequest::post_form(DDG_LITE_URL, body, 20_000);

        let html = self.webview.fetch(request, &ctx.cancel).await?;

        if html.contains("anomaly.js")
            || html.contains("challenge-form")
            || html.contains("anomaly-modal")
            || html.contains("Unfortunately, bots use DuckDuckGo too")
        {
            return Err("DuckDuckGo returned Anti-bot Challenge (anomaly detection)".to_string());
        }

        let items = parse_ddg_lite_html(&html);
        if items.is_empty() {
            return Err(format!(
                "No search results found on DuckDuckGo Lite for query: '{query}'"
            ));
        }

        let mut output = format!("Search Results for: \"{query}\" (via DuckDuckGo WebView)\n");
        if !allowed_domains.is_empty() {
            output.push_str(&format!("Allowed Domains: {}\n", allowed_domains.join(", ")));
        } else if !blocked_domains.is_empty() {
            output.push_str(&format!("Blocked Domains: {}\n", blocked_domains.join(", ")));
        }
        output.push('\n');

        for (idx, item) in items.iter().take(MAX_SEARCH_RESULTS).enumerate() {
            let snippet = truncate_chars(item.snippet.trim(), MAX_SNIPPET_CHARS);
            if snippet.is_empty() {
                output.push_str(&format!(
                    "{}. [{}]({})\n",
                    idx + 1,
                    item.title,
                    item.url,
                ));
            } else {
                output.push_str(&format!(
                    "{}. [{}]({})\n   {}\n",
                    idx + 1,
                    item.title,
                    item.url,
                    snippet
                ));
            }
        }

        Ok(truncate_chars(output.trim(), MAX_TOTAL_OUTPUT_CHARS))
    }

    /// Separate (non-agent-stream) LLM search. Responses backends send
    /// hosted `{type: web_search}` the way grok-build's `WebSearchClient` does.
    async fn search_llm_fallback(
        &self,
        config: &WebSearchConfig,
        query: &str,
        allowed_domains: &[String],
        blocked_domains: &[String],
    ) -> Result<String, String> {
        let backend = config.api_backend.unwrap_or_else(|| {
            match std::env::var("PHONEBUDDY_API_BACKEND").as_deref() {
                Ok("messages") => ApiBackend::Messages,
                Ok("responses") => ApiBackend::Responses,
                _ => ApiBackend::ChatCompletions,
            }
        });

        let api_key = config
            .api_key
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(String::from)
            .or_else(|| std::env::var("PHONEBUDDY_API_KEY").ok())
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .or_else(|| std::env::var("XAI_API_KEY").ok());

        let api_key = match api_key {
            Some(k) if !k.trim().is_empty() => k,
            _ => {
                return Err(
                    "API key missing (set on the selected pool member, in EngineConfig, or PHONEBUDDY_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY environment variable)".into(),
                );
            }
        };

        let base_url = config
            .base_url
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(String::from)
            .or_else(|| std::env::var("PHONEBUDDY_BASE_URL").ok())
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .or_else(|| std::env::var("ANTHROPIC_BASE_URL").ok())
            .unwrap_or_else(|| match backend {
                ApiBackend::Messages => "https://api.anthropic.com/v1".to_string(),
                ApiBackend::Responses => "https://api.openai.com/v1".to_string(),
                ApiBackend::ChatCompletions => "https://api.openai.com/v1".to_string(),
                ApiBackend::Gemini => {
                    "https://generativelanguage.googleapis.com/v1beta".to_string()
                }
            });

        let trimmed = base_url.trim_end_matches('/');
        let root = if let Some(stripped) = trimmed.strip_suffix("/chat/completions") {
            stripped
        } else if let Some(stripped) = trimmed.strip_suffix("/responses") {
            stripped
        } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
            stripped
        } else {
            trimmed
        };

        let mut endpoint = match backend {
            ApiBackend::ChatCompletions => format!("{root}/chat/completions"),
            ApiBackend::Responses => format!("{root}/responses"),
            ApiBackend::Messages => format!("{root}/messages"),
            ApiBackend::Gemini => format!("{root}/models/placeholder:generateContent"),
        };

        let mut model = config
            .model
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(String::from)
            .or_else(|| std::env::var("PHONEBUDDY_MODEL").ok())
            .or_else(|| std::env::var("OPENAI_MODEL").ok())
            .or_else(|| std::env::var("ANTHROPIC_MODEL").ok())
            .unwrap_or_else(|| match backend {
                ApiBackend::Messages => "claude-3-5-sonnet-20241022".to_string(),
                _ => "gpt-4o-mini".to_string(),
            });

        // Default to gpt-4o-mini if the configured model is a third-party non-OpenAI model hitting api.openai.com
        if (model.starts_with("grok") || model.starts_with("claude"))
            && endpoint.contains("api.openai.com")
        {
            model = "gpt-4o-mini".to_string();
        }
        if matches!(backend, ApiBackend::Gemini) {
            endpoint = format!("{root}/models/{model}:generateContent");
        }

        let mut prompt_text = format!(
            "Search the web and provide detailed, up-to-date factual results with sources and URLs for: {query}"
        );
        if !allowed_domains.is_empty() {
            prompt_text.push_str(&format!(
                "\nOnly include search results from these domains: {}",
                allowed_domains.join(", ")
            ));
        } else if !blocked_domains.is_empty() {
            prompt_text.push_str(&format!(
                "\nDo NOT include search results from these domains: {}",
                blocked_domains.join(", ")
            ));
        }

        match backend {
            ApiBackend::Messages => {
                let mut payload = serde_json::json!({
                    "model": model,
                    "max_tokens": 1500,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt_text
                        }
                    ]
                });
                if let Some(obj) = payload.as_object_mut() {
                    for (k, v) in &config.extra_body {
                        obj.insert(k.clone(), v.clone());
                    }
                }

                let mut req = self
                    .client
                    .post(&endpoint)
                    .timeout(std::time::Duration::from_secs(90))
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", "2023-06-01")
                    .bearer_auth(&api_key)
                    .header("Content-Type", "application/json")
                    .json(&payload);

                for (k, v) in &config.extra_headers {
                    req = req.header(k, v);
                }

                let resp = req.send().await.map_err(|e| {
                    if e.is_timeout() {
                        format!("Messages API request timed out waiting for LLM response ({endpoint}): {e}")
                    } else {
                        format!("Failed to connect to Messages API ({endpoint}): {e}")
                    }
                })?;

                let status = resp.status();
                if !status.is_success() {
                    let err_body = resp.text().await.unwrap_or_default();
                    return Err(format!("Messages API returned HTTP {status}: {err_body}"));
                }

                let json_val: Value = resp
                    .json()
                    .await
                    .map_err(|e| format!("Failed to parse Messages API response JSON: {e}"))?;

                parse_messages_api_json(&json_val, query)
            }
            ApiBackend::ChatCompletions => {
                let mut payload = serde_json::json!({
                    "model": model,
                    "max_tokens": 1500,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt_text
                        }
                    ]
                });
                if let Some(obj) = payload.as_object_mut() {
                    for (k, v) in &config.extra_body {
                        obj.insert(k.clone(), v.clone());
                    }
                }

                let mut req = self
                    .client
                    .post(&endpoint)
                    .timeout(std::time::Duration::from_secs(90))
                    .bearer_auth(&api_key)
                    .header("Content-Type", "application/json")
                    .json(&payload);

                for (k, v) in &config.extra_headers {
                    req = req.header(k, v);
                }

                let resp = req.send().await.map_err(|e| {
                    if e.is_timeout() {
                        format!("ChatCompletions API request timed out waiting for LLM response ({endpoint}): {e}")
                    } else {
                        format!("Failed to connect to ChatCompletions API ({endpoint}): {e}")
                    }
                })?;

                let status = resp.status();
                if !status.is_success() {
                    let err_body = resp.text().await.unwrap_or_default();
                    return Err(format!("ChatCompletions API returned HTTP {status}: {err_body}"));
                }

                let json_val: Value = resp
                    .json()
                    .await
                    .map_err(|e| format!("Failed to parse ChatCompletions API response JSON: {e}"))?;

                parse_chat_completions_api_json(&json_val, query)
            }
            ApiBackend::Responses => {
                let mut payload = serde_json::json!({
                    "model": model,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                     "type": "input_text",
                                     "text": prompt_text
                                }
                            ]
                        }
                    ],
                    "tools": [
                        {
                            "type": "web_search"
                        }
                    ]
                });
                if let Some(obj) = payload.as_object_mut() {
                    for (k, v) in &config.extra_body {
                        obj.insert(k.clone(), v.clone());
                    }
                }

                let mut req = self
                    .client
                    .post(&endpoint)
                    .timeout(std::time::Duration::from_secs(90))
                    .bearer_auth(&api_key)
                    .header("Content-Type", "application/json")
                    .json(&payload);

                for (k, v) in &config.extra_headers {
                    req = req.header(k, v);
                }

                let resp = req.send().await.map_err(|e| {
                    if e.is_timeout() {
                        format!("Responses API request timed out waiting for LLM response ({endpoint}): {e}")
                    } else {
                        format!("Failed to connect to Responses API ({endpoint}): {e}")
                    }
                })?;

                let status = resp.status();
                if !status.is_success() {
                    let err_body = resp.text().await.unwrap_or_default();
                    return Err(format!("Responses API returned HTTP {status}: {err_body}"));
                }

                let json_val: Value = resp
                    .json()
                    .await
                    .map_err(|e| format!("Failed to parse Responses API response JSON: {e}"))?;

                parse_responses_api_json(&json_val, query)
            }
            ApiBackend::Gemini => {
                let mut payload = serde_json::json!({
                    "contents": [{
                        "role": "user",
                        "parts": [{ "text": prompt_text }]
                    }]
                });
                if let Some(obj) = payload.as_object_mut() {
                    for (k, v) in &config.extra_body {
                        obj.insert(k.clone(), v.clone());
                    }
                }

                let mut req = self
                    .client
                    .post(&endpoint)
                    .timeout(std::time::Duration::from_secs(90))
                    .header("x-goog-api-key", &api_key)
                    .header("Content-Type", "application/json")
                    .json(&payload);

                for (k, v) in &config.extra_headers {
                    req = req.header(k, v);
                }

                let resp = req.send().await.map_err(|e| {
                    if e.is_timeout() {
                        format!("Gemini API request timed out waiting for LLM response ({endpoint}): {e}")
                    } else {
                        format!("Failed to connect to Gemini API ({endpoint}): {e}")
                    }
                })?;

                let status = resp.status();
                if !status.is_success() {
                    let err_body = resp.text().await.unwrap_or_default();
                    return Err(format!("Gemini API returned HTTP {status}: {err_body}"));
                }

                let json_val: Value = resp
                    .json()
                    .await
                    .map_err(|e| format!("Failed to parse Gemini API response JSON: {e}"))?;

                let text = json_val
                    .pointer("/candidates/0/content/parts")
                    .and_then(|p| p.as_array())
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| json_val.to_string());
                parse_chat_completions_api_json(
                    &serde_json::json!({
                        "choices": [{ "message": { "content": text } }]
                    }),
                    query,
                )
            }
        }
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SearchResultItem {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_search".into(),
            description: concat!(
                "Search the web for real-time information, documentation, news, and current events ",
                "beyond training knowledge. ",
                "Prefer `allowed_domains` (e.g. [\"docs.rs\", \"github.com\"]) when looking up a specific library. ",
                "After using results, cite relevant URLs as markdown links [Title](URL). ",
                "On iOS/Android this first searches DuckDuckGo Lite through the system WebView, ",
                "then falls back to the model's hosted web_search when the provider supports it."
            )
            .into(),
            parameters: schema_object(
                vec![
                    ("query", s_string(), "The search query string to look up."),
                    (
                        "allowed_domains",
                        s_string_array(),
                        "Optional list of domains to restrict search results to (e.g. [\"docs.rs\", \"github.com\"]).",
                    ),
                    (
                        "blocked_domains",
                        s_string_array(),
                        "Optional list of domains to exclude from search results.",
                    ),
                ],
                &["query"],
            ),
        }
    }

    async fn execute(&self, args: Value, _ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let query = arg_str(&args, "query")?.trim().to_string();
        if query.chars().count() < 2 {
            return Ok(ToolOutput::new(
                "Error: Search query must be at least 2 characters long.",
            ));
        }

        let allowed_domains = arg_opt_str_list(&args, "allowed_domains");
        let blocked_domains = arg_opt_str_list(&args, "blocked_domains");

        if !allowed_domains.is_empty() && !blocked_domains.is_empty() {
            return Ok(ToolOutput::new(
                "Error: Cannot specify both allowed_domains and blocked_domains in the same request.",
            ));
        }

        // 1. Mobile WebView + DuckDuckGo Lite (skipped while in cooldown).
        // 2. Hosted `{type: web_search}` on a *separate* Responses request,
        //    walking the pool until a member with enable_web_search succeeds.
        let webview_err = if self.webview.is_available() {
            if self.is_ddg_in_cooldown() {
                tracing::info!(
                    "[web_search] DuckDuckGo is in cooldown (recently unreachable); skipping WebView for query '{query}'"
                );
                Some("DuckDuckGo search connection failed".to_string())
            } else {
                let reachable = if self.probe_enabled {
                    tracing::info!(
                        "[web_search] Probing DuckDuckGo connectivity before WebView search: query='{query}'"
                    );
                    self.probe_ddg_connectivity(&_ctx.cancel).await
                } else {
                    true
                };

                if !reachable {
                    self.trigger_ddg_cooldown();
                    Some("DuckDuckGo search connection failed".to_string())
                } else {
                    tracing::info!(
                        "[web_search] Attempting search via host Headless WebView (DuckDuckGo Lite): query='{query}'"
                    );
                    match self
                        .search_via_webview(&query, &allowed_domains, &blocked_domains, _ctx)
                        .await
                    {
                        Ok(results) => {
                            self.clear_ddg_cooldown();
                            tracing::info!(
                                "[web_search] Host Headless WebView DuckDuckGo search succeeded for query='{query}', result_len={}",
                                results.len()
                            );
                            return Ok(ToolOutput::new(results));
                        }
                        Err(err) => {
                            self.trigger_ddg_cooldown();
                            tracing::warn!(
                                "[web_search] Host WebView search failed for query '{query}': {err}; attempting LLM API fallback"
                            );
                            Some("DuckDuckGo search connection failed".to_string())
                        }
                    }
                }
            }
        } else {
            tracing::info!(
                "[web_search] No host WebView registered; skipping DuckDuckGo scrape for query '{query}'"
            );
            None
        };

        let ddg_line = match &webview_err {
            Some(err) => format!("1. DuckDuckGo: {err}\n"),
            None => "1. System WebView: not available on this host (skipped)\n".to_string(),
        };
        let candidates = self.hosted_search_candidates();
        if candidates.is_empty() {
            tracing::warn!(
                "[web_search] DuckDuckGo unavailable and no pool member exposes hosted web_search for query='{query}'"
            );
            return Ok(ToolOutput::new(format!(
                "[WebSearch Error]: Search failed for query '{query}':\n{ddg_line}2. LLM hosted web_search: not available on this model (enableWebSearch=false); tried no further providers"
            )));
        }

        let mut last_err = String::new();
        for (provider_id, cfg) in &candidates {
            tracing::info!(
                "[web_search] Trying hosted {{type: web_search}} on '{provider_id}' (model={}) for query='{query}'",
                cfg.model.as_deref().unwrap_or("default")
            );
            match self
                .search_llm_fallback(cfg, &query, &allowed_domains, &blocked_domains)
                .await
            {
                Ok(resp_results) => {
                    tracing::info!(
                        "[web_search] Hosted web_search on '{provider_id}' succeeded for query='{query}', result_len={}",
                        resp_results.len()
                    );
                    if webview_err.is_some() {
                        return Ok(ToolOutput::new(format!(
                            "[Notice: DuckDuckGo search connection failed; used hosted web_search on {provider_id}]\n\n{resp_results}"
                        )));
                    }
                    return Ok(ToolOutput::new(resp_results));
                }
                Err(resp_err) => {
                    tracing::warn!(
                        "[web_search] Hosted web_search on '{provider_id}' failed for query='{query}': {resp_err}"
                    );
                    last_err = format!("{provider_id}: {resp_err}");
                }
            }
        }

        Ok(ToolOutput::new(format!(
            "[WebSearch Error]: Search failed for query '{query}':\n{ddg_line}2. LLM hosted web_search: {last_err}"
        )))
    }
}

/// Helper function to clean domain string (strips protocol and trailing slashes).
fn clean_domain(domain: &str) -> &str {
    domain
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
}

/// Builds a DuckDuckGo search query string including domain filters if present.
pub fn build_ddg_query(
    query: &str,
    allowed_domains: &[String],
    blocked_domains: &[String],
) -> String {
    let mut q = query.trim().to_string();
    if !allowed_domains.is_empty() {
        let sites = allowed_domains
            .iter()
            .map(|d| format!("site:{}", clean_domain(d)))
            .collect::<Vec<_>>();
        if sites.len() == 1 {
            q.push_str(&format!(" {}", sites[0]));
        } else {
            q.push_str(&format!(" ({})", sites.join(" OR ")));
        }
    } else if !blocked_domains.is_empty() {
        for d in blocked_domains {
            q.push_str(&format!(" -site:{}", clean_domain(d)));
        }
    }
    q
}

/// Parses HTML output from DuckDuckGo Lite (`lite.duckduckgo.com/lite/`).
/// Uses Regex to extract titles, URLs, and snippets safely without heavy DOM dependencies.
pub fn parse_ddg_lite_html(html: &str) -> Vec<SearchResultItem> {
    let mut items = Vec::new();

    // Matching result links: <a ... class='result-link' ...>Title</a> or <a ... href='...' ...>
    let link_re = regex::Regex::new(
        r#"(?is)<a[^>]*href=["']([^"']+)["'][^>]*class=["'][^"']*result-link[^"']*["'][^>]*>(.*?)</a>|(?is)<a[^>]*class=["'][^"']*result-link[^"']*["'][^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>"#,
    )
    .ok();

    // Snippet class regex (multi-line supported)
    let snippet_re = regex::Regex::new(
        r#"(?is)<td[^>]*class=["'][^"']*result-snippet[^"']*["'][^>]*>(.*?)</td>"#,
    )
    .ok();

    let tag_re = regex::Regex::new(r"<[^>]*>").ok();

    if let (Some(l_re), Some(s_re), Some(t_re)) = (link_re, snippet_re, tag_re) {
        let snippets: Vec<String> = s_re
            .captures_iter(html)
            .map(|cap| {
                let raw = cap.get(1).map_or("", |m| m.as_str());
                let clean = t_re.replace_all(raw, "").to_string();
                let single_spaced = clean.split_whitespace().collect::<Vec<_>>().join(" ");
                single_spaced.replace("&nbsp;", " ").trim().to_string()
            })
            .collect();

        for (i, cap) in l_re.captures_iter(html).enumerate() {
            let url = cap
                .get(1)
                .or_else(|| cap.get(3))
                .map_or("", |m| m.as_str())
                .to_string();
            let raw_title = cap
                .get(2)
                .or_else(|| cap.get(4))
                .map_or("", |m| m.as_str());
            let title = t_re
                .replace_all(raw_title, "")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");

            // Skip internal DuckDuckGo help links or ads if wanted, or retain organic results
            if url.contains("duckduckgo.com/duckduckgo-help-pages") {
                continue;
            }

            let snippet = snippets.get(i).cloned().unwrap_or_default();
            items.push(SearchResultItem {
                title,
                url,
                snippet,
            });
        }
    }

    items
}

/// Parses the JSON response returned by the Anthropic Messages API (`/v1/messages`).
pub fn parse_messages_api_json(resp_json: &Value, query: &str) -> Result<String, String> {
    let mut text_parts = Vec::new();

    if let Some(content_arr) = resp_json.get("content").and_then(Value::as_array) {
        for block in content_arr {
            if let Some(t) = block.get("text").and_then(Value::as_str) {
                if !t.trim().is_empty() {
                    text_parts.push(t.to_string());
                }
            }
        }
    } else if let Some(t) = resp_json.get("content").and_then(Value::as_str) {
        if !t.trim().is_empty() {
            text_parts.push(t.to_string());
        }
    }

    if text_parts.is_empty() {
        return Err("Messages API response did not contain message text".to_string());
    }

    let mut formatted = format!(
        "Search Results for: \"{query}\" (via Anthropic Messages API fallback)\n\n"
    );
    formatted.push_str(&text_parts.join("\n\n"));
    Ok(truncate_chars(formatted.trim(), MAX_TOTAL_OUTPUT_CHARS))
}

/// Parses the JSON response returned by the OpenAI Chat Completions API (`/v1/chat/completions`).
pub fn parse_chat_completions_api_json(resp_json: &Value, query: &str) -> Result<String, String> {
    if let Some(content) = resp_json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("content"))
        .and_then(Value::as_str)
    {
        if !content.trim().is_empty() {
            let mut formatted = format!(
                "Search Results for: \"{query}\" (via ChatCompletions API fallback)\n\n"
            );
            formatted.push_str(content.trim());
            return Ok(truncate_chars(formatted.trim(), MAX_TOTAL_OUTPUT_CHARS));
        }
    }

    Err("ChatCompletions API response did not contain message content".to_string())
}

/// Parses the JSON response returned by the OpenAI Responses API (`/v1/responses`).
pub fn parse_responses_api_json(resp_json: &Value, query: &str) -> Result<String, String> {
    let mut text_parts = Vec::new();
    let mut search_actions = Vec::new();

    if let Some(output_arr) = resp_json.get("output").and_then(Value::as_array) {
        for item in output_arr {
            let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
            if item_type == "web_search_call" {
                if let Some(action) = item.get("action") {
                    if let Some(q) = action.get("query").and_then(Value::as_str) {
                        search_actions.push(format!("Searched: \"{q}\""));
                    }
                }
            } else if item_type == "message" {
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for c in content {
                        if let Some(t) = c.get("text").and_then(Value::as_str) {
                            text_parts.push(t.to_string());
                        }
                    }
                } else if let Some(t) = item.get("content").and_then(Value::as_str) {
                    text_parts.push(t.to_string());
                }
            }
        }
    }

    // Fallbacks for alternative/compact Responses or completions output shapes
    if text_parts.is_empty() {
        if let Some(output_text) = resp_json.get("output_text").and_then(Value::as_str) {
            text_parts.push(output_text.to_string());
        } else if let Some(choices) = resp_json.get("choices").and_then(Value::as_array) {
            for choice in choices {
                if let Some(content) = choice.pointer("/message/content").and_then(Value::as_str) {
                    text_parts.push(content.to_string());
                }
            }
        }
    }

    if text_parts.is_empty() {
        return Err("Responses API response did not contain message output".to_string());
    }

    let mut formatted = format!(
        "Search Results for: \"{query}\" (via OpenAI Responses API web_search)\n\n"
    );
    if !search_actions.is_empty() {
        formatted.push_str(&format!("Actions: {}\n\n", search_actions.join(", ")));
    }
    formatted.push_str(&text_parts.join("\n\n"));
    Ok(truncate_chars(formatted.trim(), MAX_TOTAL_OUTPUT_CHARS))
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(WebSearchTool::new())
}

pub fn arc_with_config(config: WebSearchConfig) -> Arc<dyn Tool> {
    Arc::new(WebSearchTool::with_config(config))
}

pub fn arc_from_engine_config(cfg: &crate::config::EngineConfig) -> Arc<dyn Tool> {
    Arc::new(WebSearchTool::from_engine_config(cfg))
}

pub fn arc_from_engine_config_with_webview(
    cfg: &crate::config::EngineConfig,
    webview: Arc<WebViewHost>,
) -> Arc<dyn Tool> {
    Arc::new(WebSearchTool::from_engine_config_with_webview(cfg, webview))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ddg_lite_html() {
        let sample_html = r#"
        <table>
            <tr>
                <td>1.</td>
                <td><a href="https://example.com" class="result-link">Example Domain</a></td>
            </tr>
            <tr>
                <td></td>
                <td class="result-snippet">
                    This is a <b>multiline</b>
                    example snippet text.
                </td>
            </tr>
            <tr>
                <td>2.</td>
                <td><a class="result-link" href="https://rust-lang.org">Rust Language</a></td>
            </tr>
            <tr>
                <td></td>
                <td class="result-snippet">A language empowering everyone.</td>
            </tr>
        </table>
        "#;

        let results = parse_ddg_lite_html(sample_html);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Example Domain");
        assert_eq!(results[0].url, "https://example.com");
        assert_eq!(results[0].snippet, "This is a multiline example snippet text.");

        assert_eq!(results[1].title, "Rust Language");
        assert_eq!(results[1].url, "https://rust-lang.org");
        assert_eq!(results[1].snippet, "A language empowering everyone.");
    }

    #[test]
    fn test_build_ddg_query() {
        let q1 = build_ddg_query(
            "rust async",
            &["docs.rs".into()],
            &[],
        );
        assert_eq!(q1, "rust async site:docs.rs");

        let q2 = build_ddg_query(
            "tokio stream",
            &["https://docs.rs".into(), "github.com/tokio-rs/".into()],
            &[],
        );
        assert_eq!(q2, "tokio stream (site:docs.rs OR site:github.com/tokio-rs)");

        let q3 = build_ddg_query(
            "python typing",
            &[],
            &["w3schools.com".into(), "geeksforgeeks.org".into()],
        );
        assert_eq!(q3, "python typing -site:w3schools.com -site:geeksforgeeks.org");
    }

    #[test]
    fn test_parse_messages_api_json() {
        let mock_json = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Latest News: Major advancements announced today."
                }
            ]
        });

        let res = parse_messages_api_json(&mock_json, "latest news").unwrap();
        assert!(res.contains("Latest News: Major advancements announced today."));
        assert!(res.contains("Anthropic Messages API fallback"));
    }

    #[test]
    fn test_parse_chat_completions_api_json() {
        let mock_json = serde_json::json!({
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Here are the top news highlights for today."
                    },
                    "finish_reason": "stop"
                }
            ]
        });

        let res = parse_chat_completions_api_json(&mock_json, "top news").unwrap();
        assert!(res.contains("Here are the top news highlights for today."));
        assert!(res.contains("ChatCompletions API fallback"));
    }

    #[test]
    fn test_parse_responses_api_json() {
        let mock_json = serde_json::json!({
            "id": "resp_12345",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "Rust 1.85 release notes"
                    }
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Rust 1.85.0 is released featuring edition 2024."
                        }
                    ]
                }
            ]
        });

        let res = parse_responses_api_json(&mock_json, "Rust 1.85").unwrap();
        assert!(res.contains("Rust 1.85.0 is released featuring edition 2024."));
        assert!(res.contains("Actions: Searched: \"Rust 1.85 release notes\""));
        assert!(res.contains("OpenAI Responses API web_search"));
    }

    #[test]
    fn test_parse_responses_api_json_empty_error() {
        let mock_json = serde_json::json!({
            "id": "resp_empty",
            "output": []
        });

        let res = parse_responses_api_json(&mock_json, "test query");
        assert!(res.is_err());
    }

    struct FixedEndpoint(crate::llm::endpoint::LlmEndpoint);

    impl crate::llm::endpoint::LlmEndpointProvider for FixedEndpoint {
        fn current_endpoint(&self) -> Option<crate::llm::endpoint::LlmEndpoint> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn fallback_config_prefers_selected_pool_member_over_empty_engine_config() {
        let mut cfg = crate::config::EngineConfig::default();
        cfg.api_key.clear();
        cfg.base_url.clear();
        cfg.api_backend = ApiBackend::ChatCompletions;
        cfg.model = "engine-model".into();

        let tool = WebSearchTool::from_engine_config(&cfg).with_endpoint_provider(Arc::new(
            FixedEndpoint(crate::llm::endpoint::LlmEndpoint {
                provider_id: "builtin-wududu-grok".into(),
                api_key: "sk-pool".into(),
                base_url: "https://sub.wududu.com/v1".into(),
                api_backend: ApiBackend::Responses,
                model: "grok-4.6".into(),
                extra_headers: Default::default(),
                extra_body: Default::default(),
                enable_web_search: true,
            }),
        ));
        let resolved = tool.resolved_fallback_config();
        assert_eq!(resolved.api_key.as_deref(), Some("sk-pool"));
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://sub.wududu.com/v1")
        );
        assert_eq!(resolved.api_backend, Some(ApiBackend::Responses));
        assert_eq!(resolved.model.as_deref(), Some("grok-4.6"));
        assert!(resolved.enable_web_search);
    }

    #[test]
    fn fallback_config_keeps_engine_snapshot_without_endpoint_provider() {
        let mut cfg = crate::config::EngineConfig::default();
        cfg.api_key = "sk-engine".into();
        cfg.base_url = "https://api.x.ai/v1".into();
        cfg.api_backend = ApiBackend::Responses;
        cfg.model = "grok-4.6".into();
        let tool = WebSearchTool::from_engine_config(&cfg);
        let resolved = tool.resolved_fallback_config();
        assert_eq!(resolved.api_key.as_deref(), Some("sk-engine"));
        assert_eq!(resolved.base_url.as_deref(), Some("https://api.x.ai/v1"));
        assert_eq!(resolved.api_backend, Some(ApiBackend::Responses));
        assert!(!resolved.enable_web_search);
    }

    struct ChainEndpoints {
        current: crate::llm::endpoint::LlmEndpoint,
        rest: Vec<crate::llm::endpoint::LlmEndpoint>,
    }

    impl crate::llm::endpoint::LlmEndpointProvider for ChainEndpoints {
        fn current_endpoint(&self) -> Option<crate::llm::endpoint::LlmEndpoint> {
            Some(self.current.clone())
        }
        fn fallback_endpoints(&self) -> Vec<crate::llm::endpoint::LlmEndpoint> {
            self.rest.clone()
        }
    }

    fn ep(id: &str, backend: ApiBackend, hosted: bool) -> crate::llm::endpoint::LlmEndpoint {
        crate::llm::endpoint::LlmEndpoint {
            provider_id: id.into(),
            api_key: format!("sk-{id}"),
            base_url: format!("https://{id}.example/v1"),
            api_backend: backend,
            model: "grok-4.6".into(),
            extra_headers: Default::default(),
            extra_body: Default::default(),
            enable_web_search: hosted,
        }
    }

    #[test]
    fn hosted_search_candidates_skip_current_without_flag_and_walk_pool() {
        let tool = WebSearchTool::new().with_endpoint_provider(Arc::new(ChainEndpoints {
            current: ep("packy", ApiBackend::ChatCompletions, false),
            rest: vec![
                ep("openlux-gpt", ApiBackend::Responses, false),
                ep("wududu-grok", ApiBackend::Responses, true),
            ],
        }));
        let ids: Vec<_> = tool
            .hosted_search_candidates()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, vec!["wududu-grok".to_string()]);
    }

    #[test]
    fn hosted_search_candidates_prefer_current_when_it_has_hosted_search() {
        let tool = WebSearchTool::new().with_endpoint_provider(Arc::new(ChainEndpoints {
            current: ep("wududu-grok", ApiBackend::Responses, true),
            rest: vec![ep("openlux-grok", ApiBackend::Responses, true)],
        }));
        let ids: Vec<_> = tool
            .hosted_search_candidates()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            ids,
            vec!["wududu-grok".to_string(), "openlux-grok".to_string()]
        );
    }

    #[tokio::test]
    async fn test_web_search_validation() {
        let tool = WebSearchTool::new();
        let ctx = ToolCtx {
            sandbox: Arc::new(crate::tools::fs::Sandbox::new(std::path::Path::new("/tmp")).unwrap()),
            cancel: tokio_util::sync::CancellationToken::new(),
        };

        // Query too short
        let res_short = tool
            .execute(serde_json::json!({ "query": "a" }), &ctx)
            .await
            .unwrap();
        assert!(res_short.text.contains("at least 2 characters"));

        // Mutex conflict: both allowed_domains and blocked_domains
        let res_conflict = tool
            .execute(
                serde_json::json!({
                    "query": "rust docs",
                    "allowed_domains": ["docs.rs"],
                    "blocked_domains": ["example.com"]
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(res_conflict.text.contains("Cannot specify both allowed_domains and blocked_domains"));
    }

    #[tokio::test]
    async fn test_web_search_skips_ddg_without_webview() {
        let tool = WebSearchTool::new();
        let ctx = ToolCtx {
            sandbox: Arc::new(crate::tools::fs::Sandbox::new(std::path::Path::new("/tmp")).unwrap()),
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        let res = tool
            .execute(serde_json::json!({ "query": "rust async" }), &ctx)
            .await
            .unwrap();
        assert!(
            res.text.contains("System WebView: not available"),
            "expected skip-to-API path, got: {}",
            res.text
        );
        assert!(
            res.text.contains("enableWebSearch=false"),
            "expected no hosted fallback when the flag is off, got: {}",
            res.text
        );
        assert!(!res.text.contains("DuckDuckGo returned"));
    }

    #[tokio::test]
    async fn test_web_search_uses_host_webview_html() {
        let webview = WebViewHost::new();
        let webview_c = webview.clone();
        webview.set_notify(Arc::new(move |call_id, _req| {
            let host = webview_c.clone();
            std::thread::spawn(move || {
                let html = r#"
                <table>
                    <tr>
                        <td>1.</td>
                        <td><a href="https://doc.rust-lang.org" class="result-link">The Rust Book</a></td>
                    </tr>
                    <tr>
                        <td></td>
                        <td class="result-snippet">Official Rust documentation.</td>
                    </tr>
                </table>
                "#;
                let _ = host.complete(&call_id, true, html);
            });
        }));

        let tool = WebSearchTool::with_config_and_webview(WebSearchConfig::default(), webview)
            .with_probe_enabled(false);
        let ctx = ToolCtx {
            sandbox: Arc::new(crate::tools::fs::Sandbox::new(std::path::Path::new("/tmp")).unwrap()),
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        let res = tool
            .execute(serde_json::json!({ "query": "rust book" }), &ctx)
            .await
            .unwrap();
        assert!(res.text.contains("via DuckDuckGo WebView"));
        assert!(res.text.contains("The Rust Book"));
        assert!(res.text.contains("https://doc.rust-lang.org"));
        assert!(!tool.is_ddg_in_cooldown());
    }

    #[tokio::test]
    async fn test_web_search_cooldown_skips_webview_and_formats_notice() {
        let webview = WebViewHost::new();
        let webview_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let webview_called_c = webview_called.clone();
        webview.set_notify(Arc::new(move |_call_id, _req| {
            webview_called_c.store(true, std::sync::atomic::Ordering::SeqCst);
        }));

        let tool = WebSearchTool::with_config_and_webview(WebSearchConfig::default(), webview)
            .with_probe_enabled(false);

        // Manually trigger cooldown (e.g. following previous connectivity failure)
        tool.trigger_ddg_cooldown();
        assert!(tool.is_ddg_in_cooldown());

        let ctx = ToolCtx {
            sandbox: Arc::new(crate::tools::fs::Sandbox::new(std::path::Path::new("/tmp")).unwrap()),
            cancel: tokio_util::sync::CancellationToken::new(),
        };

        let res = tool
            .execute(serde_json::json!({ "query": "latest news" }), &ctx)
            .await
            .unwrap();

        // Host WebView should NOT be called during active cooldown
        assert!(
            !webview_called.load(std::sync::atomic::Ordering::SeqCst),
            "WebView should have been skipped during active cooldown"
        );

        // Fallback error should report DuckDuckGo failure and skip hosted search
        assert!(res.text.contains("DuckDuckGo: DuckDuckGo search connection failed"));
        assert!(res.text.contains("enableWebSearch=false"));

        // Clear cooldown
        tool.clear_ddg_cooldown();
        assert!(!tool.is_ddg_in_cooldown());
    }

    #[tokio::test]
    async fn test_web_search_cooldown_lifecycle() {
        let tool = WebSearchTool::new();
        assert!(!tool.is_ddg_in_cooldown());

        tool.trigger_ddg_cooldown();
        assert!(tool.is_ddg_in_cooldown());

        tool.clear_ddg_cooldown();
        assert!(!tool.is_ddg_in_cooldown());
    }
}
