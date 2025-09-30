from __future__ import annotations
import re
import tldextract
from pathlib import Path
from urllib.parse import urlparse

SUS_TLDS = {"zip", "mov", "gq", "men", "work", "link", "country", "stream"}
SUS_FILE_EXTS = {
    "exe",
    "bat",
    "cmd",
    "scr",
    "pif",
    "apk",
    "jar",
    "msi",
    "hta",
    "js",
    "vbs",
    "ps1",
}

_TLD_CACHE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "tldextract"
_TLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=str(_TLD_CACHE_DIR))

def url_stats(u: str) -> dict:
    parsed = urlparse(u if re.match(r'^https?://', u) else f"http://{u}")
    host = parsed.hostname or ""
    ext = _TLD_EXTRACTOR(parsed.hostname or "")
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    subdomain = ext.subdomain
    path = parsed.path or "/"
    q = parsed.query or ""
    s = u

    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    specials = len(s) - digits - letters

    num_equal = s.count("=")
    num_amp = s.count("&")
    num_plus = s.count("+")
    num_hash = s.count("#")
    path_segments = [seg for seg in path.split("/") if seg]
    path_depth = len(path_segments)
    query_params = [param for param in (q.split("&") if q else []) if param]
    num_params = len(query_params)
    has_params = 1 if q else 0

    file_part = path_segments[-1] if path_segments else ""
    has_file_ext = 1 if "." in file_part else 0
    file_ext = file_part.split(".")[-1].lower() if has_file_ext else ""
    is_suspicious_file = 1 if file_ext in SUS_FILE_EXTS else 0

    feats = {
        "url_len": len(s),
        "host_len": len(host),
        "path_len": len(path),
        "num_digits": digits,
        "num_letters": letters,
        "num_specials": specials,
        "num_dots": s.count("."),
        "num_hyphen": s.count("-"),
        "num_slash": s.count("/"),
        "num_qm": s.count("?"),
        "num_at": s.count("@"),
        "num_pct": s.count("%"),
        "num_equal": num_equal,
        "num_amp": num_amp,
        "num_plus": num_plus,
        "num_hash": num_hash,
        "has_ip": bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', host or "")),
        "subdomain_depth": len(subdomain.split(".")) if subdomain else 0,
        "tld": ext.suffix or "",
        "tld_suspicious": 1 if (ext.suffix or "") in SUS_TLDS else 0,
        "has_punycode": 1 if "xn--" in host else 0,
        "scheme_https": 1 if parsed.scheme == "https" else 0,
        "query_len": len(q),
        "fragment_len": len(parsed.fragment or ""),
        "domain_len": len(ext.domain) if ext.domain else len(host),
        "digit_ratio": digits / len(s) if len(s) else 0.0,
        "special_ratio": specials / len(s) if len(s) else 0.0,
        "letter_ratio": letters / len(s) if len(s) else 0.0,
        "path_depth": path_depth,
        "num_params": num_params,
        "has_params": has_params,
        "has_file_ext": has_file_ext,
        "is_suspicious_file": is_suspicious_file,
        "has_www": 1 if host.startswith("www.") else 0,
        "is_long_domain": 1 if len(host) > 30 else 0,
        "domain": domain,
        "hostname": host,
    }
    return feats
