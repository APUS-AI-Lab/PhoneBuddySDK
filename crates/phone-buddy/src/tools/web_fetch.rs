//! `web_fetch` tool.
//!
//! Fetches a webpage and converts its HTML into LLM-optimized representations
//! (Markdown via `htmd`, Interactive Element Tree with refs, Compact Text).
//!
//! Ported safety path from grok-build: URL validation, HTTPS upgrade, SSRF
//! checks (`tools/ssrf.rs`), body size limits. Markdown conversion uses the
//! same `htmd` crate as desktop (skip script/style/noscript/svg/iframe/…).

use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::error::{EngineError, EngineResult};
use crate::tools::browser::{
    apply_browser_headers, decode_bytes_to_string, get_fallback_fingerprint,
    get_platform_fingerprint, BrowserFingerprint, MAX_WEB_BODY_LENGTH,
};
use crate::tools::ssrf::{check_ssrf, validate_and_normalize_url};
use crate::tools::webview::{WebViewFetchRequest, WebViewHost};
use crate::tools::{
    arg_opt_str, arg_opt_usize, arg_str, s_enum, s_integer, s_string, schema_object, truncate_chars,
    Tool, ToolCtx, ToolOutput, ToolSpec,
};

/// Default maximum returned characters for web_fetch output.
pub const DEFAULT_MAX_CHARS: usize = 8000;
/// Max response body (10 MB).
pub const MAX_CONTENT_LENGTH: usize = MAX_WEB_BODY_LENGTH;
/// Max redirects (10).
pub const MAX_REDIRECTS: usize = 10;

/// Generates platform-native User-Agent strings matching system browser behavior.
pub fn get_native_user_agent() -> &'static str {
    get_platform_fingerprint().user_agent
}

pub struct WebFetchTool {
    client: reqwest::Client,
    allow_local: bool,
    webview: Arc<WebViewHost>,
    /// Shared HTML→Markdown converter (htmd), matching grok-build web_fetch.
    converter: htmd::HtmlToMarkdown,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self::with_allow_local_and_webview(false, WebViewHost::new())
    }

    pub fn with_allow_local(allow_local: bool) -> Self {
        Self::with_allow_local_and_webview(allow_local, WebViewHost::new())
    }

    pub fn with_allow_local_and_webview(allow_local: bool, webview: Arc<WebViewHost>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let converter = htmd::HtmlToMarkdown::builder()
            .skip_tags(vec![
                "script", "style", "noscript", "svg", "iframe", "object", "embed",
            ])
            .build();
        Self {
            client,
            allow_local,
            webview,
            converter,
        }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_fetch".into(),
            description: concat!(
                "Fetch a web page and convert its content into a token-efficient representation ",
                "(Markdown, Interactive Element Tree with refs, or Compact Text). ",
                "Uses native browser User-Agent headers to avoid anti-bot blocks."
            )
            .into(),
            parameters: schema_object(
                vec![
                    ("url", s_string(), "The HTTP/HTTPS URL of the web page to fetch."),
                    (
                        "mode",
                        s_enum(&["markdown", "interactive", "compact", "text"]),
                        "Format of content returned: 'markdown' (clean main article), 'interactive' (buttons/links/inputs with [ref=eN]), 'compact' (headings + content + refs), 'text' (plain text). Default: 'markdown'.",
                    ),
                    (
                        "user_agent",
                        s_string(),
                        "Optional custom User-Agent string. If omitted, uses default platform-native browser User-Agent.",
                    ),
                    (
                        "max_chars",
                        s_integer(),
                        "Maximum character limit of output (default: 8000).",
                    ),
                ],
                &["url"],
            ),
        }
    }

    async fn execute(&self, args: Value, ctx: &ToolCtx) -> EngineResult<ToolOutput> {
        let raw_url = arg_str(&args, "url")?;
        let url = validate_and_normalize_url(&raw_url)?;
        // SSRF: block private/loopback unless allow_local + explicit loopback host.
        check_ssrf(&url, self.allow_local).await?;

        let mode = arg_opt_str(&args, "mode").unwrap_or_else(|| "markdown".to_string());
        let custom_ua = arg_opt_str(&args, "user_agent");
        let max_chars = arg_opt_usize(&args, "max_chars", DEFAULT_MAX_CHARS);

        let base_fp = get_platform_fingerprint();
        let (active_ua, is_custom_ua) = match &custom_ua {
            Some(ua) => (ua.clone(), true),
            None => (base_fp.user_agent.to_string(), false),
        };
        let url_str = url.as_str().to_string();

        // 1. If host system WebView is available, attempt to fetch page via headless WebView
        // (handles JavaScript rendering, cookies, Cloudflare/anti-bot protection).
        let mut html_content_opt: Option<String> = None;
        if self.webview.is_available() {
            tracing::info!("[web_fetch] Attempting fetch via host Headless WebView: url='{url_str}'");
            let mut headers = std::collections::HashMap::new();
            if is_custom_ua {
                headers.insert("User-Agent".to_string(), active_ua.clone());
            }
            let webview_req = WebViewFetchRequest {
                url: url_str.clone(),
                method: "GET".to_string(),
                headers,
                body: String::new(),
                timeout_ms: 20_000,
            };
            match self.webview.fetch(webview_req, &ctx.cancel).await {
                Ok(html) => {
                    if !html.trim().is_empty() {
                        tracing::info!(
                            "[web_fetch] Host Headless WebView fetch succeeded: url='{url_str}', html_len={}",
                            html.len()
                        );
                        html_content_opt = Some(html);
                    } else {
                        tracing::warn!(
                            "[web_fetch] Host Headless WebView returned empty HTML for url='{url_str}'; falling back to HTTP client"
                        );
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "[web_fetch] Host Headless WebView fetch failed for url='{url_str}': {err}; falling back to HTTP client"
                    );
                }
            }
        } else {
            tracing::info!(
                "[web_fetch] No host WebView registered; fetching url='{url_str}' directly via HTTP client (reqwest/curl-equivalent)"
            );
        }

        // 2. If WebView fetch wasn't available or returned empty, fetch via reqwest HTTP client
        let html_content = match html_content_opt {
            Some(html) => html,
            None => {
                tracing::info!(
                    "[web_fetch] Sending HTTP GET request via reqwest client (curl-equivalent): url='{url_str}', user_agent='{active_ua}'"
                );
                let primary_fp = BrowserFingerprint {
                    user_agent: if is_custom_ua {
                        base_fp.user_agent
                    } else {
                        base_fp.user_agent
                    },
                    sec_ch_ua: if is_custom_ua { None } else { base_fp.sec_ch_ua },
                    sec_ch_ua_platform: if is_custom_ua { None } else { base_fp.sec_ch_ua_platform },
                    sec_ch_ua_mobile: if is_custom_ua { None } else { base_fp.sec_ch_ua_mobile },
                    accept: base_fp.accept,
                    accept_language: base_fp.accept_language,
                };

                let req = apply_browser_headers(
                    self.client.get(url.clone()),
                    &primary_fp,
                    "none",
                    None,
                );
                let req = if is_custom_ua {
                    req.header("User-Agent", &active_ua)
                } else {
                    req
                };

                let mut response = match req.send().await {
                    Ok(res) => res,
                    Err(err) => {
                        return Ok(ToolOutput::new(format!(
                            "[WebFetch Error]: Failed to connect to URL '{url_str}': {err}"
                        )));
                    }
                };

                // Anti-bot retry: If primary platform fingerprint encounters 403 / 429 / 503 challenge,
                // attempt one retry using the secondary desktop Chrome fingerprint with Cache-Control headers.
                let mut status = response.status();
                if !is_custom_ua
                    && (status == reqwest::StatusCode::FORBIDDEN
                        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE)
                {
                    let fallback_fp = get_fallback_fingerprint();
                    let retry_req = apply_browser_headers(
                        self.client.get(url.clone()),
                        &fallback_fp,
                        "none",
                        Some("max-age=0"),
                    );
                    if let Ok(retry_res) = retry_req.send().await {
                        if retry_res.status().is_success() {
                            response = retry_res;
                            status = response.status();
                        }
                    }
                }

                // Re-check final URL after redirects (same SSRF policy).
                if let Some(final_url) = response.url().host_str().map(|_| response.url().clone()) {
                    if let Err(e) = check_ssrf(&final_url, self.allow_local).await {
                        return Ok(ToolOutput::new(format!("[WebFetch Error]: {e}")));
                    }
                }

                if status == reqwest::StatusCode::FORBIDDEN || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    return Ok(ToolOutput::new(format!(
                        "[WebFetch Anti-Bot Block]: Target URL '{url_str}' returned HTTP {status} (Anti-bot / Cloudflare challenge). Consider using `web_search` or searching for alternative mirror URLs."
                    )));
                }

                if !status.is_success() {
                    return Ok(ToolOutput::new(format!(
                        "[WebFetch Error]: Server returned HTTP status {status} for URL '{url_str}'."
                    )));
                }

                let raw_bytes = response.bytes().await.map_err(|e| EngineError::Tool {
                    name: "web_fetch".into(),
                    message: format!("Failed to read response body bytes: {e}"),
                })?;
                tracing::info!(
                    "[web_fetch] HTTP client response received: url='{url_str}', status={status}, body_bytes={}",
                    raw_bytes.len()
                );
                if raw_bytes.len() > MAX_CONTENT_LENGTH {
                    return Ok(ToolOutput::new(format!(
                        "[WebFetch Error]: Response body {} bytes exceeds max {MAX_CONTENT_LENGTH}",
                        raw_bytes.len()
                    )));
                }

                decode_bytes_to_string(&raw_bytes)
            }
        };

        let result_text = match mode.to_lowercase().as_str() {
            "interactive" => convert_html_to_interactive_tree(&html_content),
            "compact" => convert_html_to_compact_tree(&html_content),
            "text" => convert_html_to_clean_text(&html_content),
            _ => html_to_markdown_htmd(&self.converter, &html_content),
        };

        let formatted = format!(
            "URL: {}\nMode: {}\nUser-Agent: {}\n\n{}",
            url_str, mode, active_ua, result_text
        );

        let output = truncate_chars(&formatted, max_chars);
        Ok(ToolOutput::new(output))
    }
}

/// Port of grok `html_to_markdown` via htmd.
fn html_to_markdown_htmd(converter: &htmd::HtmlToMarkdown, html: &str) -> String {
    match converter.convert(html) {
        Ok(md) => normalize_whitespace(&md),
        Err(_) => {
            // Fallback to regex path if htmd fails on pathological HTML.
            convert_html_to_markdown(html, "")
        }
    }
}

/// Strips noise tags (<script>, <style>, <noscript>, <svg>, comments) from HTML.
fn strip_html_noise(html: &str) -> String {
    let re_script = Regex::new(r"(?is)<script[^>]*?>.*?</script>").unwrap();
    let re_style = Regex::new(r"(?is)<style[^>]*?>.*?</style>").unwrap();
    let re_noscript = Regex::new(r"(?is)<noscript[^>]*?>.*?</noscript>").unwrap();
    let re_svg = Regex::new(r"(?is)<svg[^>]*?>.*?</svg>").unwrap();
    let re_comments = Regex::new(r"(?is)<!--.*?-->").unwrap();

    let cleaned = re_script.replace_all(html, "");
    let cleaned = re_style.replace_all(&cleaned, "");
    let cleaned = re_noscript.replace_all(&cleaned, "");
    let cleaned = re_svg.replace_all(&cleaned, "");
    let cleaned = re_comments.replace_all(&cleaned, "");

    cleaned.to_string()
}

/// Converts HTML into clean, readable Markdown (stripping nav, footer, header noise).
pub fn convert_html_to_markdown(html: &str, base_url: &str) -> String {
    let clean = strip_html_noise(html);

    // Strip header, footer, nav wrappers if possible to isolate main content
    let re_nav = Regex::new(r"(?is)<nav[^>]*?>.*?</nav>|<footer[^>]*?>.*?</footer>|<header[^>]*?>.*?</header>|<aside[^>]*?>.*?</aside>").unwrap();
    let main_content = re_nav.replace_all(&clean, "");

    let tag_re = Regex::new(r"(?is)<([^>]+)>").unwrap();
    let h_re = Regex::new(r"(?is)<h([1-6])[^>]*?>(.*?)</h[1-6]>").unwrap();
    let p_re = Regex::new(r"(?is)<p[^>]*?>(.*?)</p>").unwrap();
    let a_re = Regex::new(r#"(?is)<a[^>]*?href=["']([^"']+)["'][^>]*?>(.*?)</a>"#).unwrap();
    let li_re = Regex::new(r"(?is)<li[^>]*?>(.*?)</li>").unwrap();
    let br_re = Regex::new(r"(?is)<br\s*/?>").unwrap();

    // 1. Process Headings
    let res = h_re.replace_all(&main_content, |caps: &regex::Captures| {
        let level: usize = caps[1].parse().unwrap_or(1);
        let text = tag_re.replace_all(&caps[2], "").trim().to_string();
        if text.is_empty() {
            "".to_string()
        } else {
            format!("\n\n{} {}\n", "#".repeat(level), text)
        }
    });

    // 2. Process Links
    let res = a_re.replace_all(&res, |caps: &regex::Captures| {
        let href = &caps[1];
        let text = tag_re.replace_all(&caps[2], "").trim().to_string();
        if text.is_empty() {
            "".to_string()
        } else {
            format!(" [{}]({}) ", text, resolve_url(base_url, href))
        }
    });

    // 3. Process Paragraphs
    let res = p_re.replace_all(&res, |caps: &regex::Captures| {
        let text = tag_re.replace_all(&caps[1], "").trim().to_string();
        if text.is_empty() {
            "".to_string()
        } else {
            format!("\n\n{}\n", text)
        }
    });

    // 4. Process List items
    let res = li_re.replace_all(&res, |caps: &regex::Captures| {
        let text = tag_re.replace_all(&caps[1], "").trim().to_string();
        if text.is_empty() {
            "".to_string()
        } else {
            format!("\n* {}", text)
        }
    });

    // 5. Process Linebreaks and remaining HTML tags
    let res = br_re.replace_all(&res, "\n");
    let res = tag_re.replace_all(&res, " ");

    // Normalize multiple spaces and blank lines
    normalize_whitespace(&res)
}

/// Extracts interactive ARIA-like elements (`button`, `link`, `textbox`) with `[ref=eN]` tags.
pub fn convert_html_to_interactive_tree(html: &str) -> String {
    let clean = strip_html_noise(html);
    let mut refs = Vec::new();
    let mut ref_counter = 0;

    let tag_re = Regex::new(r"(?is)<([^>]+)>").unwrap();

    // Match links <a href="...">text</a>
    let a_re = Regex::new(r#"(?is)<a[^>]*?href=["']([^"']+)["'][^>]*?>(.*?)</a>"#).unwrap();
    for caps in a_re.captures_iter(&clean) {
        let href = caps[1].to_string();
        let text = tag_re.replace_all(&caps[2], "").trim().to_string();
        if !text.is_empty() {
            ref_counter += 1;
            refs.push(format!(
                "- link \"{}\" [ref=e{}] (href: {})",
                text, ref_counter, href
            ));
        }
    }

    // Match buttons <button...>text</button>
    let btn_re = Regex::new(r"(?is)<button[^>]*?>(.*?)</button>").unwrap();
    for caps in btn_re.captures_iter(&clean) {
        let text = tag_re.replace_all(&caps[1], "").trim().to_string();
        if !text.is_empty() {
            ref_counter += 1;
            refs.push(format!("- button \"{}\" [ref=e{}]", text, ref_counter));
        }
    }

    // Match input fields <input ...>
    let input_re = Regex::new(r#"(?is)<input[^>]*?>"#).unwrap();
    let type_re = Regex::new(r#"(?is)type=["']([^"']+)["']"#).unwrap();
    let name_re = Regex::new(r#"(?is)name=["']([^"']+)["']"#).unwrap();
    let holder_re = Regex::new(r#"(?is)placeholder=["']([^"']+)["']"#).unwrap();

    for caps in input_re.captures_iter(&clean) {
        let tag_str = &caps[0];
        let input_type = type_re
            .captures(tag_str)
            .map_or("text", |c| c.get(1).unwrap().as_str());
        let input_name = name_re
            .captures(tag_str)
            .map_or("", |c| c.get(1).unwrap().as_str());
        let placeholder = holder_re
            .captures(tag_str)
            .map_or("", |c| c.get(1).unwrap().as_str());

        if input_type == "hidden" {
            continue;
        }

        ref_counter += 1;
        let mut info = format!("- input[{}]", input_type);
        if !placeholder.is_empty() {
            info.push_str(&format!(" \"{}\"", placeholder));
        } else if !input_name.is_empty() {
            info.push_str(&format!(" \"{}\"", input_name));
        }
        info.push_str(&format!(" [ref=e{}]", ref_counter));
        refs.push(info);
    }

    if refs.is_empty() {
        "(No interactive elements found on page)".to_string()
    } else {
        refs.join("\n")
    }
}

/// Converts HTML into compact tree with headings, content paragraphs, and interactive refs.
pub fn convert_html_to_compact_tree(html: &str) -> String {
    let interactive = convert_html_to_interactive_tree(html);
    let clean_text = convert_html_to_clean_text(html);
    let truncated_text = truncate_chars(&clean_text, 1500);

    format!(
        "=== Interactive Elements ===\n{}\n\n=== Content Summary ===\n{}",
        interactive, truncated_text
    )
}

/// Strips all HTML tags and normalizes whitespace into clean plain text.
pub fn convert_html_to_clean_text(html: &str) -> String {
    let clean = strip_html_noise(html);
    let tag_re = Regex::new(r"(?is)<[^>]+>").unwrap();
    let text = tag_re.replace_all(&clean, " ");
    normalize_whitespace(&text)
}

/// Resolves relative URLs to absolute URLs.
fn resolve_url(base_url: &str, href: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") || href.starts_with("//") {
        return href.to_string();
    }
    if let Ok(base) = url::Url::parse(base_url) {
        if let Ok(joined) = base.join(href) {
            return joined.to_string();
        }
    }
    href.to_string()
}

/// Replaces multiple spaces, tabs, and duplicate blank lines with clean single spacing.
fn normalize_whitespace(text: &str) -> String {
    let mut lines = Vec::new();
    for line in text.lines() {
        let trimmed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !trimmed.is_empty() {
            lines.push(trimmed);
        }
    }
    lines.join("\n\n")
}

pub fn arc() -> Arc<dyn Tool> {
    Arc::new(WebFetchTool::new())
}

pub fn arc_with_allow_local(allow_local: bool) -> Arc<dyn Tool> {
    Arc::new(WebFetchTool::with_allow_local(allow_local))
}

pub fn arc_with_allow_local_and_webview(allow_local: bool, webview: Arc<WebViewHost>) -> Arc<dyn Tool> {
    Arc::new(WebFetchTool::with_allow_local_and_webview(allow_local, webview))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_user_agent() {
        let ua = get_native_user_agent();
        assert!(!ua.is_empty());
        assert!(ua.contains("Mozilla/5.0"));
    }

    #[test]
    fn test_convert_html_to_markdown_htmd() {
        let html = r#"
        <html>
            <head><title>Test</title><script>console.log('secret');</script></head>
            <body>
                <h1>Title Heading</h1>
                <p>Hello <b>world</b>!</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        "#;

        let conv = htmd::HtmlToMarkdown::builder()
            .skip_tags(vec!["script", "style", "noscript", "svg", "iframe"])
            .build();
        let md = html_to_markdown_htmd(&conv, html);
        assert!(md.contains("Title Heading"), "md={md}");
        assert!(md.contains("Hello") || md.contains("world"), "md={md}");
        assert!(!md.contains("console.log"), "md={md}");
    }

    #[test]
    fn test_convert_html_to_interactive_tree() {
        let html = r#"
        <div>
            <a href="https://example.com">Home</a>
            <input type="text" name="username" placeholder="Enter username" />
            <button>Submit</button>
        </div>
        "#;

        let tree = convert_html_to_interactive_tree(html);
        assert!(tree.contains("- link \"Home\" [ref=e1]"));
        assert!(tree.contains("- button \"Submit\""));
        assert!(tree.contains("[ref=e2]") || tree.contains("[ref=e3]"));
    }
}
