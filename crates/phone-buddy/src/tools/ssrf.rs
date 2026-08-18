//! SSRF (Server-Side Request Forgery) protection for `web_fetch`.
//!
//! Ported from grok-build `implementations/grok_build/web_fetch/ssrf.rs`.
//!
//! Policy:
//! - Non-public addresses (loopback, RFC 1918, link-local, CGNAT, TEST-NET,
//!   multicast, etc.) are blocked by default.
//! - Local access is opt-in via `allow_local`. Even when enabled, only
//!   **explicit** loopback hosts are allowed (`localhost`, `127.0.0.0/8`
//!   literals, `::1`). A public hostname that resolves to loopback stays blocked.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use url::Url;

use crate::error::{EngineError, EngineResult};

/// Hostnames/IP literals that may reach loopback when local binding is enabled.
pub fn is_explicit_local_host(host: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(&host);
    let host = host.split('%').next().unwrap_or(host);

    if host == "localhost" {
        return true;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return ip.is_loopback();
    }
    false
}

/// Returns `true` if an IP is not globally routable.
pub fn is_non_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_non_public_ipv4(v4),
        IpAddr::V6(v6) => is_non_public_ipv6(v6),
    }
}

fn is_non_public_ipv4(ip: Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_multicast()
        || ip.is_broadcast()
        || ipv4_in_cidr(ip, [0, 0, 0, 0], 8)
        || ipv4_in_cidr(ip, [100, 64, 0, 0], 10)
        || ipv4_in_cidr(ip, [192, 0, 0, 0], 24)
        || ipv4_in_cidr(ip, [192, 0, 2, 0], 24)
        || ipv4_in_cidr(ip, [198, 18, 0, 0], 15)
        || ipv4_in_cidr(ip, [198, 51, 100, 0], 24)
        || ipv4_in_cidr(ip, [203, 0, 113, 0], 24)
        || ipv4_in_cidr(ip, [240, 0, 0, 0], 4)
}

fn ipv4_in_cidr(ip: Ipv4Addr, base: [u8; 4], prefix: u8) -> bool {
    let ip = u32::from(ip);
    let base = u32::from(Ipv4Addr::from(base));
    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - prefix)
    };
    (ip & mask) == (base & mask)
}

fn is_non_public_ipv6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_non_public_ipv4(v4);
    }
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
}

fn is_loopback_addr(ip: IpAddr) -> bool {
    if ip.is_loopback() {
        return true;
    }
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback()),
        IpAddr::V4(_) => false,
    }
}

/// Whether a resolved address is blocked for this request host.
pub fn is_blocked_for_host(ip: IpAddr, host: &str, allow_local: bool) -> bool {
    if !is_non_public_ip(ip) {
        return false;
    }
    if allow_local && is_loopback_addr(ip) && is_explicit_local_host(host) {
        return false;
    }
    true
}

/// Validate URL shape, upgrade http→https, reject overly long URLs.
pub fn validate_and_normalize_url(raw: &str) -> EngineResult<Url> {
    const MAX_URL_LENGTH: usize = 2_000;
    if raw.len() > MAX_URL_LENGTH {
        return Err(EngineError::Tool {
            name: "web_fetch".into(),
            message: format!("URL exceeds max length ({MAX_URL_LENGTH})"),
        });
    }
    let upgraded = if raw.starts_with("http://") {
        format!("https://{}", raw.trim_start_matches("http://"))
    } else if raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("https://{raw}")
    };
    let url = Url::parse(&upgraded).map_err(|e| EngineError::Tool {
        name: "web_fetch".into(),
        message: format!("invalid URL: {e}"),
    })?;
    if url.scheme() != "https" && url.scheme() != "http" {
        return Err(EngineError::Tool {
            name: "web_fetch".into(),
            message: format!("unsupported URL scheme: {}", url.scheme()),
        });
    }
    if url.host_str().is_none() {
        return Err(EngineError::Tool {
            name: "web_fetch".into(),
            message: "URL has no host".into(),
        });
    }
    Ok(url)
}

/// Resolve hostname via DNS and verify none of the resolved addresses are blocked.
pub async fn check_ssrf(url: &Url, allow_local: bool) -> EngineResult<()> {
    let host = url.host_str().ok_or_else(|| EngineError::Tool {
        name: "web_fetch".into(),
        message: "URL has no host".into(),
    })?;

    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_blocked_for_host(ip, host, allow_local) {
            return Err(EngineError::Tool {
                name: "web_fetch".into(),
                message: format!("SSRF blocked: host {host} resolves to non-public address {ip}"),
            });
        }
        return Ok(());
    }

    let port = url.port_or_known_default().unwrap_or(443);
    let addr_str = format!("{host}:{port}");
    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(&addr_str)
        .await
        .map_err(|e| EngineError::Tool {
            name: "web_fetch".into(),
            message: format!("DNS resolution failed for {host}: {e}"),
        })?
        .collect();

    if addrs.is_empty() {
        return Err(EngineError::Tool {
            name: "web_fetch".into(),
            message: format!("DNS returned no addresses for {host}"),
        });
    }

    if let Some(addr) = addrs
        .iter()
        .find(|addr| is_blocked_for_host(addr.ip(), host, allow_local))
    {
        return Err(EngineError::Tool {
            name: "web_fetch".into(),
            message: format!(
                "SSRF blocked: host {host} resolves to non-public address {}",
                addr.ip()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_rfc1918() {
        assert!(is_non_public_ip("10.0.0.1".parse().unwrap()));
        assert!(is_non_public_ip("192.168.1.1".parse().unwrap()));
        assert!(is_non_public_ip("127.0.0.1".parse().unwrap()));
        assert!(!is_non_public_ip("8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn allow_local_only_explicit_loopback() {
        let loopback: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(!is_blocked_for_host(loopback, "localhost", true));
        assert!(!is_blocked_for_host(loopback, "127.0.0.1", true));
        // Public name resolving to loopback still blocked.
        assert!(is_blocked_for_host(loopback, "evil.example", true));
        // Private never allowed even with allow_local.
        let private: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(is_blocked_for_host(private, "10.0.0.1", true));
    }

    #[test]
    fn https_upgrade() {
        let u = validate_and_normalize_url("http://example.com/a").unwrap();
        assert_eq!(u.scheme(), "https");
        assert_eq!(u.host_str(), Some("example.com"));
    }
}
