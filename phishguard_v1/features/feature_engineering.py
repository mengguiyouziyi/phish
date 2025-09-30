from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import urlparse

SECURITY_HEADERS = {
    "strict-transport-security",
    "content-security-policy",
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
    "permissions-policy",
    "x-xss-protection",
    "cross-origin-opener-policy",
    "cross-origin-resource-policy",
}

SUSPICIOUS_META_KEYWORDS = {
    "login",
    "verify",
    "security",
    "update",
    "password",
    "account",
    "confirm",
}

URL_METRIC_COLS = [
    "url_len",
    "host_len",
    "path_len",
    "num_digits",
    "num_letters",
    "num_specials",
    "num_dots",
    "num_hyphen",
    "num_slash",
    "num_qm",
    "num_at",
    "num_pct",
    "num_equal",
    "num_amp",
    "num_plus",
    "num_hash",
    "subdomain_depth",
    "query_len",
    "fragment_len",
    "domain_len",
    "digit_ratio",
    "special_ratio",
    "letter_ratio",
    "path_depth",
    "num_params",
]

URL_FLAG_COLS = [
    "has_ip",
    "tld_suspicious",
    "has_punycode",
    "scheme_https",
    "has_params",
    "has_file_ext",
    "is_suspicious_file",
    "has_www",
    "is_long_domain",
]

HTML_BASE_COLS = [
    "has_html",
    "title_len",
    "num_meta",
    "num_links",
    "num_stylesheets",
    "num_scripts",
    "num_script_ext",
    "num_script_inline",
    "num_iframes",
    "num_forms",
    "has_password_input",
    "has_email_input",
    "suspicious_js_inline",
    "external_form_actions",
    "num_hidden_inputs",
    "external_links",
    "internal_links",
    "external_images",
]

HTML_DERIVED_COLS = [
    "external_form_ratio",
    "hidden_input_ratio",
    "external_link_ratio",
    "internal_link_ratio",
    "kw_hit_count",
    "lib_hit_count",
    "meta_refresh_count",
    "meta_og_count",
    "meta_charset_count",
    "meta_viewport_flag",
    "meta_robots_block",
    "meta_description_len",
    "meta_keywords_len",
    "meta_sensitive_kw_flag",
    "title_entropy",
    "external_script_host_count",
    "external_stylesheet_host_count",
    "external_resource_host_diversity",
]

HTTP_FEATURE_COLS = [
    "status_code",
    "bytes",
    "body_len",
    "content_type_len",
    "content_type_is_html",
    "content_type_is_octet",
    "http_header_count",
    "http_security_header_count",
    "http_server_len",
    "http_powered_by_len",
    "http_cache_control_len",
    "http_location_len",
    "http_redirect_count",
    "http_tls_retry_flag",
    "http_ok_flag",
]

COOKIE_FEATURE_COLS = [
    "cookie_count",
    "cookie_secure_ratio",
    "cookie_httponly_ratio",
    "cookie_samesite_ratio",
    "cookie_max_age_ratio",
]

FINGERPRINT_FEATURE_COLS = [
    "fingerprint_hash_len",
    "fingerprint_entropy",
]

FEATURE_COLUMNS: List[str] = [
    *URL_METRIC_COLS,
    *URL_FLAG_COLS,
    *HTML_BASE_COLS,
    *HTML_DERIVED_COLS,
    *HTTP_FEATURE_COLS,
    *COOKIE_FEATURE_COLS,
    *FINGERPRINT_FEATURE_COLS,
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0):
        return 0.0
    return numerator / denominator


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _normalize_headers(headers: Mapping[str, Any] | None) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    if not headers:
        return normalized
    for key, value in headers.items():
        if key is None:
            continue
        normalized[key.lower()] = str(value)
    return normalized


def _count_cookie_flag(set_cookie: str, flag: str) -> int:
    if not set_cookie:
        return 0
    lowered = set_cookie.lower()
    return lowered.count(flag.lower())


def _unique_hosts(urls: Iterable[str]) -> List[str]:
    hosts: List[str] = []
    for item in urls:
        if not item:
            continue
        parsed = urlparse(item)
        host = parsed.hostname
        if host:
            hosts.append(host.lower())
    return list({host for host in hosts})


def compute_feature_dict(raw: Dict[str, Any], html_features: Dict[str, Any] | None = None) -> Dict[str, float]:
    url_feats = raw.get("url_feats") or {}
    html_feats = html_features or raw.get("html_feats") or {}
    headers = _normalize_headers(raw.get("headers"))
    cookies = raw.get("cookies") or {}
    set_cookie = raw.get("set_cookie") or ""
    html = raw.get("html") or ""
    meta = raw.get("meta") or {}

    features: Dict[str, float] = {}

    # URL metrics and flags
    for key in URL_METRIC_COLS:
        features[key] = _to_float(url_feats.get(key, 0.0))
    for key in URL_FLAG_COLS:
        features[key] = 1.0 if url_feats.get(key) else 0.0

    # HTML base metrics
    for key in HTML_BASE_COLS:
        if key == "has_html":
            features[key] = 1.0 if html else 0.0
        else:
            features[key] = _to_float(html_feats.get(key, 0.0))

    num_forms = max(features.get("num_forms", 0.0), 0.0)
    num_links = max(features.get("num_links", 0.0), 0.0)

    external_links = features.get("external_links", 0.0)
    internal_links = features.get("internal_links", 0.0)
    external_forms = features.get("external_form_actions", 0.0)
    hidden_inputs = features.get("num_hidden_inputs", 0.0)

    features["external_form_ratio"] = _safe_ratio(external_forms, num_forms) if num_forms else 0.0
    features["hidden_input_ratio"] = _safe_ratio(hidden_inputs, num_forms) if num_forms else 0.0
    link_total = external_links + internal_links
    features["external_link_ratio"] = _safe_ratio(external_links, link_total)
    features["internal_link_ratio"] = _safe_ratio(internal_links, link_total)

    kw_hits = html_feats.get("kw_hits") or {}
    lib_hits = html_feats.get("lib_hits") or {}

    features["kw_hit_count"] = float(sum(1 for v in kw_hits.values() if v))
    features["lib_hit_count"] = float(sum(1 for v in lib_hits.values() if v))

    meta_kv = html_feats.get("meta_kv") or {}
    features["meta_refresh_count"] = float(
        sum(
            1
            for key, value in meta_kv.items()
            if (key and "refresh" in key.lower()) or (value and "refresh" in value.lower())
        )
    )
    features["meta_og_count"] = float(sum(1 for key in meta_kv if key.lower().startswith("og:")))
    features["meta_charset_count"] = float(
        sum(1 for key in meta_kv if "charset" in key.lower())
    )
    features["meta_viewport_flag"] = 1.0 if meta_kv.get("viewport") else 0.0

    robots_value = meta_kv.get("robots", "") or ""
    features["meta_robots_block"] = (
        1.0 if any(token in robots_value.lower() for token in ["noindex", "nofollow"]) else 0.0
    )

    description_val = meta_kv.get("description", "") or ""
    keywords_val = meta_kv.get("keywords", "") or ""
    features["meta_description_len"] = float(len(description_val))
    features["meta_keywords_len"] = float(len(keywords_val))
    features["meta_sensitive_kw_flag"] = (
        1.0 if any(word in (description_val + " " + keywords_val).lower() for word in SUSPICIOUS_META_KEYWORDS) else 0.0
    )

    title = html_feats.get("title") or ""
    features["title_entropy"] = _shannon_entropy(title)

    script_srcs = html_feats.get("script_srcs") or []
    stylesheet_hrefs = html_feats.get("stylesheets") or []

    script_hosts = _unique_hosts(script_srcs)
    style_hosts = _unique_hosts(stylesheet_hrefs)

    features["external_script_host_count"] = float(len(script_hosts))
    features["external_stylesheet_host_count"] = float(len(style_hosts))

    total_external_resources = len(script_srcs) + len(stylesheet_hrefs)
    unique_hosts = len({*script_hosts, *style_hosts})
    features["external_resource_host_diversity"] = _safe_ratio(
        unique_hosts, total_external_resources
    ) if total_external_resources else 0.0

    # HTTP level features
    status_code = raw.get("status_code") or 0
    features["status_code"] = _to_float(status_code, 0.0)
    features["bytes"] = _to_float(raw.get("bytes", 0.0), 0.0)
    features["body_len"] = float(len(html))

    content_type = raw.get("content_type") or ""
    lowered_ct = content_type.lower()
    features["content_type_len"] = float(len(content_type))
    features["content_type_is_html"] = 1.0 if "text/html" in lowered_ct else 0.0
    features["content_type_is_octet"] = 1.0 if "application/octet" in lowered_ct else 0.0

    features["http_header_count"] = float(len(headers))
    features["http_security_header_count"] = float(
        sum(1 for key in headers if key in SECURITY_HEADERS)
    )
    features["http_server_len"] = float(len(headers.get("server", "")))
    features["http_powered_by_len"] = float(len(headers.get("x-powered-by", "")))
    features["http_cache_control_len"] = float(len(headers.get("cache-control", "")))
    features["http_location_len"] = float(len(headers.get("location", "")))

    redirects = meta.get("redirects") or []
    features["http_redirect_count"] = float(len(redirects))
    features["http_tls_retry_flag"] = 1.0 if meta.get("tls_retry") else 0.0
    features["http_ok_flag"] = 1.0 if raw.get("ok") else 0.0

    # Cookie features
    cookie_count = len(cookies) if isinstance(cookies, Mapping) else 0
    features["cookie_count"] = float(cookie_count)
    secure_flags = _count_cookie_flag(set_cookie, "secure")
    httponly_flags = _count_cookie_flag(set_cookie, "httponly")
    samesite_flags = _count_cookie_flag(set_cookie, "samesite")
    max_age_flags = _count_cookie_flag(set_cookie, "max-age")

    denom = cookie_count if cookie_count else (secure_flags or httponly_flags or samesite_flags or max_age_flags or 1)
    features["cookie_secure_ratio"] = _safe_ratio(secure_flags, denom)
    features["cookie_httponly_ratio"] = _safe_ratio(httponly_flags, denom)
    features["cookie_samesite_ratio"] = _safe_ratio(samesite_flags, denom)
    features["cookie_max_age_ratio"] = _safe_ratio(max_age_flags, denom)

    # Fingerprint features
    fingerprint_hash = html_feats.get("fingerprint_hash", "") or ""
    features["fingerprint_hash_len"] = float(len(fingerprint_hash))
    features["fingerprint_entropy"] = _shannon_entropy(fingerprint_hash)

    for column in FEATURE_COLUMNS:
        features[column] = _to_float(features.get(column, 0.0), 0.0)

    return features


def ensure_feature_columns(record: Dict[str, Any]) -> None:
    for column in FEATURE_COLUMNS:
        record.setdefault(column, 0.0)
        record[column] = _to_float(record[column], 0.0)


def prepare_features_dataframe(df):
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    for column in missing:
        df[column] = 0.0
    for column in FEATURE_COLUMNS:
        df[column] = df[column].astype(float, copy=False)
    return df
