from __future__ import annotations
import re
import tldextract
from urllib.parse import urlparse

SUS_TLDS = set(["zip","mov","gq","men","work","link","country","stream"])

def url_stats(u: str) -> dict:
    parsed = urlparse(u if re.match(r'^https?://', u) else f"http://{u}")
    host = parsed.hostname or ""
    ext = tldextract.extract(parsed.hostname or "")
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    subdomain = ext.subdomain
    path = parsed.path or "/"
    q = parsed.query or ""
    s = u

    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    specials = len(s) - digits - letters

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
        "has_ip": bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', host or "")),
        "subdomain_depth": len(subdomain.split(".")) if subdomain else 0,
        "tld": ext.suffix or "",
        "tld_suspicious": 1 if (ext.suffix or "") in SUS_TLDS else 0,
        "has_punycode": 1 if "xn--" in host else 0,
        "scheme_https": 1 if parsed.scheme == "https" else 0,
        "query_len": len(q),
        "fragment_len": len(parsed.fragment or ""),
        "domain": domain,
        "hostname": host,
    }
    return feats
