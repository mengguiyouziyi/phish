from __future__ import annotations
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re, json, hashlib
from typing import Dict, Any, List
from .url_features import url_stats
from pathlib import Path

KEYWORDS = {k.strip() for k in Path(__file__).resolve().parents[1].joinpath("assets/rules/keywords.txt").read_text(encoding="utf-8").splitlines() if k.strip()}
LIBS = json.loads(Path(__file__).resolve().parents[1].joinpath("assets/rules/libraries.json").read_text(encoding="utf-8"))

def _bool(b): return 1 if b else 0

def extract_from_html(html: str, base_url: str) -> Dict[str, Any]:
    if not html:
        return {"has_html": 0}

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""
    metas = soup.find_all("meta")
    meta_kv = {m.get("name") or m.get("property") or "": m.get("content") or "" for m in metas if (m.get("name") or m.get("property"))}
    links = soup.find_all("link")
    stylesheets = [l.get("href","") for l in links if (l.get("rel") and "stylesheet" in " ".join(l.get("rel")).lower())]
    scripts = soup.find_all("script")
    script_srcs = [s.get("src","") for s in scripts if s.get("src")]
    inline_scripts = [s.get_text() or "" for s in scripts if not s.get("src")]
    iframes = soup.find_all("iframe")
    forms = soup.find_all("form")

    # 表单 & 密码框
    inputs = soup.find_all("input")
    has_pwd = any((i.get("type","").lower() == "password") for i in inputs)
    has_email = any(("email" in (i.get("type","").lower() or "") or "email" in (i.get("name","").lower() or "")) for i in inputs)

    # 增强特征：域名年龄和信誉特征（从URL推导）
    from urllib.parse import urlparse
    parsed_url = urlparse(base_url)
    domain = parsed_url.netloc

    # 域名特征
    domain_parts = domain.split('.')
    is_subdomain = len(domain_parts) > 2
    has_www = domain.startswith('www.')

    # TLD特征
    tld = domain_parts[-1] if domain_parts else ""
    common_tlds = {'com', 'org', 'net', 'edu', 'gov', 'mil', 'int'}
    is_common_tld = tld in common_tlds

    # 增强特征：表单分析
    form_actions = [f.get("action", "") for f in forms if f.get("action")]
    external_form_actions = sum(1 for action in form_actions if action and not action.startswith("/") and base_url not in action)

    # 增强特征：隐藏字段检测
    hidden_inputs = [i for i in inputs if i.get("type", "").lower() == "hidden"]
    num_hidden_inputs = len(hidden_inputs)

    # 增强特征：外部链接统计
    all_links = soup.find_all("a", href=True)
    external_links = sum(1 for link in all_links if link.get("href") and not link["href"].startswith("#") and not link["href"].startswith("/") and base_url not in link["href"])
    internal_links = len([link for link in all_links if link.get("href") and (link["href"].startswith("/") or base_url in link["href"])])

    # 增强特征：图像分析
    images = soup.find_all("img")
    external_images = sum(1 for img in images if img.get("src") and not img["src"].startswith("/") and base_url not in img["src"])

    # 关键词命中
    text_for_kw = " ".join([title] + list(meta_kv.values()))[:5000].lower()
    kw_hits = {k: _bool(k in text_for_kw) for k in list(KEYWORDS)[:64]}  # 限制维度

    # 外部库识别
    lib_hits = {}
    joined = " ".join(script_srcs + stylesheets).lower()
    for lib, patterns in LIBS.items():
        lib_hits[f"lib_{lib}"] = _bool(any(p in joined for p in patterns))

    # 改进的可疑JS特征检测
    suspicious_js = 0
    for sc in inline_scripts:
        if not sc:
            continue

        sc_lower = sc.lower()
        script_content = sc_lower

        # 白名单检查 - 排除知名网站的常见模式
        whitelist_patterns = [
            "document.write('<a href=",  # 百度登录链接生成
            "encodeuricomponent",        # URL编码，常见于正常功能
            "window.location.href",      # 正常的页面跳转
        ]

        # 如果匹配白名单模式，跳过可疑检测
        if any(pattern in script_content for pattern in whitelist_patterns):
            continue

        # 高危可疑模式
        high_risk_patterns = [
            "atob(",
            "fromcharcode",
            "unescape(",
            "eval(",
            "location.replace("
        ]

        # 中危可疑模式 - 需要上下文判断
        medium_risk_patterns = [
            "document.write(",
        ]

        # 计算可疑分数
        risk_score = 0
        for pattern in high_risk_patterns:
            if pattern in script_content:
                risk_score += 2

        for pattern in medium_risk_patterns:
            if pattern in script_content:
                risk_score += 1

        # 只有当风险分数>=2时才计为可疑
        if risk_score >= 2:
            suspicious_js += 1

    # 指纹摘要（外链域名 + meta片段）
    ex_hosts = []
    for u in script_srcs + stylesheets:
        h = urlparse(u).hostname
        if h: ex_hosts.append(h)
    fp_basis = "|".join(sorted(set(ex_hosts)) + sorted([f"{k}:{(meta_kv.get(k) or '')[:64]}" for k in sorted(meta_kv.keys()) if k]))
    fp_hash = hashlib.sha256(fp_basis.encode("utf-8")).hexdigest() if fp_basis else ""

    return {
        "has_html": 1,
        "title": title,
        "title_len": len(title),
        "num_meta": len(metas),
        "num_links": len(links),
        "num_stylesheets": len(stylesheets),
        "num_scripts": len(scripts),
        "num_script_ext": len(script_srcs),
        "num_script_inline": len(inline_scripts),
        "num_iframes": len(iframes),
        "num_forms": len(forms),
        "has_password_input": _bool(has_pwd),
        "has_email_input": _bool(has_email),
        "suspicious_js_inline": suspicious_js,

        # 新增强特征
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
        "fingerprint_hash": fp_hash,
    }
