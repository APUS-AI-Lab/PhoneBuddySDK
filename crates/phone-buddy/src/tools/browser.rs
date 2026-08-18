//! Browser fingerprinting and shared HTTP utilities for `web_fetch` and `web_search`.
//!
//! Provides platform-native User-Agent strings, modern browser Client Hints (sec-ch-ua),
//! standard navigation headers, and multi-encoding decoding helpers.

use reqwest::RequestBuilder;

/// Default maximum allowed response body size (10 MB).
pub const MAX_WEB_BODY_LENGTH: usize = 10 * 1024 * 1024;

/// Structured browser fingerprint matching platform and client capabilities.
#[derive(Debug, Clone)]
pub struct BrowserFingerprint {
    pub user_agent: &'static str,
    pub sec_ch_ua: Option<&'static str>,
    pub sec_ch_ua_platform: Option<&'static str>,
    pub sec_ch_ua_mobile: Option<&'static str>,
    pub accept: &'static str,
    pub accept_language: &'static str,
}

/// Returns the platform-native primary browser fingerprint.
pub fn get_platform_fingerprint() -> BrowserFingerprint {
    #[cfg(target_os = "ios")]
    {
        BrowserFingerprint {
            user_agent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/605.1.15",
            sec_ch_ua: None,
            sec_ch_ua_platform: None,
            sec_ch_ua_mobile: None,
            accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            accept_language: "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    }

    #[cfg(target_os = "android")]
    {
        BrowserFingerprint {
            user_agent: "Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
            sec_ch_ua: Some(r#""Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99""#),
            sec_ch_ua_platform: Some(r#""Android""#),
            sec_ch_ua_mobile: Some("?1"),
            accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            accept_language: "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    }

    #[cfg(target_os = "macos")]
    {
        BrowserFingerprint {
            user_agent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
            sec_ch_ua: None,
            sec_ch_ua_platform: None,
            sec_ch_ua_mobile: None,
            accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            accept_language: "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    }

    #[cfg(not(any(target_os = "ios", target_os = "android", target_os = "macos")))]
    {
        BrowserFingerprint {
            user_agent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            sec_ch_ua: Some(r#""Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99""#),
            sec_ch_ua_platform: Some(r#""Linux""#),
            sec_ch_ua_mobile: Some("?0"),
            accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            accept_language: "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        }
    }
}

/// Returns a secondary fallback browser fingerprint (Chrome Desktop) for anti-bot retry.
pub fn get_fallback_fingerprint() -> BrowserFingerprint {
    BrowserFingerprint {
        user_agent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        sec_ch_ua: Some(r#""Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99""#),
        sec_ch_ua_platform: Some(r#""macOS""#),
        sec_ch_ua_mobile: Some("?0"),
        accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        accept_language: "en-US,en;q=0.9",
    }
}

/// Applies browser navigation headers to a `reqwest::RequestBuilder`.
pub fn apply_browser_headers(
    mut req: RequestBuilder,
    fp: &BrowserFingerprint,
    site_mode: &str,
    cache_control: Option<&str>,
) -> RequestBuilder {
    req = req
        .header("User-Agent", fp.user_agent)
        .header("Accept", fp.accept)
        .header("Accept-Language", fp.accept_language)
        .header("Upgrade-Insecure-Requests", "1")
        .header("Sec-Fetch-Dest", "document")
        .header("Sec-Fetch-Mode", "navigate")
        .header("Sec-Fetch-Site", site_mode)
        .header("Sec-Fetch-User", "?1");

    if let Some(ch_ua) = fp.sec_ch_ua {
        req = req.header("sec-ch-ua", ch_ua);
    }
    if let Some(ch_platform) = fp.sec_ch_ua_platform {
        req = req.header("sec-ch-ua-platform", ch_platform);
    }
    if let Some(ch_mobile) = fp.sec_ch_ua_mobile {
        req = req.header("sec-ch-ua-mobile", ch_mobile);
    }
    if let Some(cc) = cache_control {
        req = req.header("Cache-Control", cc);
    }

    req
}

/// Decodes raw HTTP response bytes to `String` using UTF-8 with `encoding_rs` fallback.
pub fn decode_bytes_to_string(bytes: &[u8]) -> String {
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_string();
    }
    let (cow, _, _) = encoding_rs::UTF_8.decode(bytes);
    cow.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_fingerprint() {
        let fp = get_platform_fingerprint();
        assert!(!fp.user_agent.is_empty());
        assert!(!fp.accept.is_empty());
        assert!(!fp.accept_language.is_empty());
    }

    #[test]
    fn test_fallback_fingerprint() {
        let fp = get_fallback_fingerprint();
        assert!(fp.user_agent.contains("Chrome"));
        assert!(fp.sec_ch_ua.is_some());
    }

    #[test]
    fn test_decode_bytes_to_string() {
        let valid_utf8 = b"Hello, world!";
        assert_eq!(decode_bytes_to_string(valid_utf8), "Hello, world!");

        // Malformed UTF-8 byte
        let invalid = b"Hello \xFF world!";
        let decoded = decode_bytes_to_string(invalid);
        assert!(decoded.contains("Hello"));
        assert!(decoded.contains("world!"));
    }
}
