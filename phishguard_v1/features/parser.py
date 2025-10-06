from __future__ import annotations
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse

from .url_features import url_stats


def _asset_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath("assets/rules", filename)


KEYWORDS = {
    line.strip()
    for line in _asset_path("keywords.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
}
LIBS = json.loads(_asset_path("libraries.json").read_text(encoding="utf-8"))
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "mil", "int"}


def _bool(flag: bool) -> int:
    return 1 if flag else 0


class FastHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: List[str] = []
        self.meta_kv: Dict[str, str] = {}
        self.stylesheets: List[str] = []
        self.script_srcs: List[str] = []
        self.inline_scripts: List[str] = []
        self.forms: List[Dict[str, str]] = []
        self.inputs: List[Dict[str, str]] = []
        self.anchors: List[Dict[str, str]] = []
        self.images: List[str] = []
        self.num_iframes = 0
        self.num_links = 0
        self.num_anchors = 0
        self.num_scripts = 0
        self._in_title = False
        self._collect_script = False
        self._current_script: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        attr = {k.lower(): v for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = attr.get("name") or attr.get("property")
            content = attr.get("content")
            if name and content and name not in self.meta_kv:
                self.meta_kv[name] = content
        elif tag == "link":
            self.num_links += 1
            rel = attr.get("rel", "")
            if "stylesheet" in rel.lower():
                href = attr.get("href")
                if href:
                    self.stylesheets.append(href)
        elif tag == "script":
            self.num_scripts += 1
            src = attr.get("src")
            if src:
                self.script_srcs.append(src)
                self._collect_script = False
            else:
                self._collect_script = True
                self._current_script = []
        elif tag == "form":
            self.forms.append(attr)
        elif tag == "input":
            self.inputs.append(attr)
        elif tag == "iframe":
            self.num_iframes += 1
        elif tag == "img":
            src = attr.get("src")
            if src:
                self.images.append(src)
        elif tag == "a":
            self.num_anchors += 1
            href = attr.get("href")
            self.anchors.append(attr)

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._collect_script:
            script_text = "".join(self._current_script)
            self.inline_scripts.append(script_text)
            self._collect_script = False
            self._current_script = []

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._in_title:
            self.title_parts.append(data)
        if self._collect_script:
            self._current_script.append(data)


def extract_from_html(html: str, base_url: str) -> Dict[str, Any]:
    if not html:
        return {"has_html": 0}

    parser = FastHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # 碰到非标准或二进制内容时直接回退为空特征
        return {"has_html": 1, "title": "", "title_len": 0, "num_meta": 0, "num_links": 0, "num_anchors": 0,
                "num_stylesheets": 0, "num_scripts": 0, "num_script_ext": 0, "num_script_inline": 0,
                "num_iframes": 0, "num_forms": 0, "has_password_input": 0, "has_email_input": 0,
                "suspicious_js_inline": 0, "external_form_actions": 0, "num_hidden_inputs": 0,
                "external_links": 0, "internal_links": 0, "external_images": 0, "is_subdomain": 0,
                "has_www": 0, "is_common_tld": 0, "meta_kv": {}, "script_srcs": [], "stylesheets": [],
                "kw_hits": {}, "lib_hits": {}, "fingerprint_hash": "", "anchors": []}

    title = "".join(parser.title_parts).strip()
    meta_kv = parser.meta_kv
    stylesheets = parser.stylesheets
    script_srcs = parser.script_srcs
    inline_scripts = [s for s in parser.inline_scripts if s.strip()]

    parsed_url = urlparse(base_url)
    domain = parsed_url.netloc or ""
    domain_parts = domain.split(".") if domain else []
    is_subdomain = len(domain_parts) > 2
    has_www = domain.startswith("www.")
    tld = domain_parts[-1] if domain_parts else ""
    is_common_tld = tld in COMMON_TLDS

    form_actions = [f.get("action", "") for f in parser.forms if f.get("action")]
    external_form_actions = sum(
        1
        for action in form_actions
        if action and not action.startswith("/") and base_url not in action
    )

    hidden_inputs = [i for i in parser.inputs if i.get("type", "").lower() == "hidden"]
    num_hidden_inputs = len(hidden_inputs)

    external_links = sum(
        1
        for link in parser.anchors
        if link.get("href")
        and not link["href"].startswith("#")
        and not link["href"].startswith("/")
        and base_url not in link["href"]
    )
    internal_links = sum(
        1
        for link in parser.anchors
        if link.get("href")
        and (link["href"].startswith("/") or base_url in link["href"])
    )

    external_images = sum(
        1
        for src in parser.images
        if not src.startswith("/") and base_url not in src
    )

    has_pwd = any((i.get("type", "").lower() == "password") for i in parser.inputs)
    has_email = any(
        ("email" in (i.get("type", "").lower() or ""))
        or ("email" in (i.get("name", "").lower() or ""))
        for i in parser.inputs
    )

    text_for_kw = " ".join([title] + list(meta_kv.values()))[:5000].lower()
    kw_hits = {k: _bool(k in text_for_kw) for k in list(KEYWORDS)[:64]}

    lib_hits: Dict[str, int] = {}
    joined = " ".join(script_srcs + stylesheets).lower()
    for lib, patterns in LIBS.items():
        lib_hits[f"lib_{lib}"] = _bool(any(p in joined for p in patterns))

    high_risk = ["atob(", "fromcharcode", "unescape(", "eval(", "location.replace("]
    medium_risk = ["document.write("]
    whitelist = [
        "document.write('<a href=",
        "encodeuricomponent",
        "window.location.href",
    ]

    suspicious_js = 0
    for script in inline_scripts:
        content = script.lower()
        if any(pat in content for pat in whitelist):
            continue
        risk = sum(2 for pat in high_risk if pat in content) + sum(
            1 for pat in medium_risk if pat in content
        )
        if risk >= 2:
            suspicious_js += 1

    external_hosts = []
    for url in script_srcs + stylesheets:
        host = urlparse(url).hostname
        if host:
            external_hosts.append(host)
    fingerprint_basis = "|".join(
        sorted(set(external_hosts))
        + [f"{k}:{(meta_kv.get(k) or '')[:64]}" for k in sorted(meta_kv.keys()) if k]
    )
    fingerprint_hash = (
        hashlib.sha256(fingerprint_basis.encode("utf-8")).hexdigest()
        if fingerprint_basis
        else ""
    )

    return {
        "has_html": 1,
        "title": title,
        "title_len": len(title),
        "num_meta": len(meta_kv),
        "num_links": parser.num_anchors,
        "num_anchors": parser.num_anchors,
        "num_stylesheets": len(stylesheets),
        "num_scripts": parser.num_scripts,
        "num_script_ext": len(script_srcs),
        "num_script_inline": len(inline_scripts),
        "num_iframes": parser.num_iframes,
        "num_forms": len(parser.forms),
        "has_password_input": _bool(has_pwd),
        "has_email_input": _bool(has_email),
        "suspicious_js_inline": suspicious_js,
        "external_form_actions": external_form_actions,
        "num_hidden_inputs": num_hidden_inputs,
        "external_links": external_links,
        "internal_links": internal_links,
        "external_images": external_images,
        "is_subdomain": _bool(is_subdomain),
        "has_www": _bool(has_www),
        "is_common_tld": _bool(is_common_tld),
        "meta_kv": meta_kv,
        "script_srcs": script_srcs,
        "stylesheets": stylesheets,
        "kw_hits": kw_hits,
        "lib_hits": lib_hits,
        "fingerprint_hash": fingerprint_hash,
        "anchors": parser.anchors,
    }
