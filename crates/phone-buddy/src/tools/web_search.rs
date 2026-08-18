//! `web_search` tool.
//!
//! Provides high-reliability web search with:
//! 1. Primary on iOS/Android: DuckDuckGo Lite loaded in the host system WebView
//!    (WKWebView / Android WebView) so TLS and cookies match a real browser.
//! 2. Desktop / C hosts (`c_demo`, CLI): skip scraping and go straight to the
//!    configured LLM search API.
//! 3. Automatic resilience fallback: if the host WebView is blocked or fails,
//!    retry via the LLM Responses / Messages / ChatCompletions API.
//! 4. Domain filtering (`allowed_domains`, `blocked_domains`).
//! 5. Markdown hyperlink formatting `[Title](URL)` for easy source citations.

use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value;

use crate::error::EngineResult;
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

#[derive(Debug, Clone, Default)]
pub struct WebSearchConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub api_backend: Option<ApiBackend>,
    pub extra_headers: std::collections::HashMap<String, String>,
    pub extra_body: std::collections::HashMap<String, serde_json::Value>,
}

pub struct WebSearchTool {
    client: reqwest::Client,
    config: WebSearchConfig,
    webview: Arc<WebViewHost>,
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
            webview,
        }
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
            },
            webview,
        )
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

    /// Fallback search provider: Dispatches to LLM backend (Messages, Responses, or ChatCompletions) based on configuration.
    async fn search_llm_fallback(
        &self,
        query: &str,
        allowed_domains: &[String],
        blocked_domains: &[String],
    ) -> Result<String, String> {
        let backend = self.config.api_backend.unwrap_or_else(|| {
            match std::env::var("PHONEBUDDY_API_BACKEND").as_deref() {
                Ok("messages") => ApiBackend::Messages,
                Ok("responses") => ApiBackend::Responses,
                _ => ApiBackend::ChatCompletions,
            }
        });

        let api_key = self
            .config
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
                    "API key missing (set in EngineConfig or PHONEBUDDY_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY environment variable)".into(),
                );
            }
        };

        let base_url = self
            .config
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

        let endpoint = match backend {
            ApiBackend::ChatCompletions => format!("{root}/chat/completions"),
            ApiBackend::Responses => format!("{root}/responses"),
            ApiBackend::Messages => format!("{root}/messages"),
        };

        let mut model = self
            .config
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
                    for (k, v) in &self.config.extra_body {
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

                for (k, v) in &self.config.extra_headers {
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
                    for (k, v) in &self.config.extra_body {
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

                for (k, v) in &self.config.extra_headers {
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
                    for (k, v) in &self.config.extra_body {
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

                for (k, v) in &self.config.extra_headers {
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
                "On iOS/Android, searches DuckDuckGo Lite through the system WebView; ",
                "otherwise uses the configured LLM API."
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

        // Unified search pipeline:
        // 1. Mobile: host system WebView (WKWebView / Android WebView).
        // 2. Desktop / C hosts: skip scraping.
        // 3. Fallback: configured LLM search API.
        let webview_err = if self.webview.is_available() {
            tracing::info!(
                "[web_search] Attempting search via host Headless WebView (DuckDuckGo Lite): query='{query}'"
            );
            match self
                .search_via_webview(&query, &allowed_domains, &blocked_domains, _ctx)
                .await
            {
                Ok(results) => {
                    tracing::info!(
                        "[web_search] Host Headless WebView DuckDuckGo search succeeded for query='{query}', result_len={}",
                        results.len()
                    );
                    return Ok(ToolOutput::new(results));
                }
                Err(err) => {
                    tracing::warn!(
                        "[web_search] Host WebView search failed for query '{query}': {err}; attempting LLM API fallback"
                    );
                    Some(err)
                }
            }
        } else {
            tracing::info!(
                "[web_search] No host WebView registered; skipping DuckDuckGo scrape for query '{query}'"
            );
            None
        };

        let backend = self.config.api_backend.unwrap_or(ApiBackend::ChatCompletions);
        let backend_name = match backend {
            ApiBackend::Messages => "Messages API",
            ApiBackend::Responses => "Responses API",
            ApiBackend::ChatCompletions => "ChatCompletions API",
        };
        tracing::info!(
            "[web_search] Executing LLM search API fallback ({backend_name}) for query='{query}'"
        );
        match self
            .search_llm_fallback(&query, &allowed_domains, &blocked_domains)
            .await
        {
            Ok(resp_results) => {
                tracing::info!(
                    "[web_search] LLM search fallback ({backend_name}) succeeded for query='{query}', result_len={}",
                    resp_results.len()
                );
                if let Some(err) = webview_err {
                    let notice = format!(
                        "[Notice: system WebView search failed ({err}). Retrieved results via LLM {backend_name} fallback]\n\n{resp_results}"
                    );
                    Ok(ToolOutput::new(notice))
                } else {
                    Ok(ToolOutput::new(resp_results))
                }
            }
            Err(resp_err) => {
                tracing::warn!(
                    "[web_search] LLM search fallback ({backend_name}) failed for query='{query}': {resp_err}"
                );
                let ddg_line = match webview_err {
                    Some(err) => format!("1. System WebView: {err}\n"),
                    None => {
                        "1. System WebView: not available on this host (skipped)\n".to_string()
                    }
                };
                Ok(ToolOutput::new(format!(
                    "[WebSearch Error]: Search failed for query '{query}':\n{ddg_line}2. LLM {backend_name} fallback: {resp_err}"
                )))
            }
        }
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

        let tool = WebSearchTool::with_config_and_webview(WebSearchConfig::default(), webview);
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
    }
}
