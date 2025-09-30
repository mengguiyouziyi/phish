from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
from httpx import AsyncClient

try:
    from .ui_patch import DataFrame as CompatDataFrame
except Exception:  # pragma: no cover - 直接运行包时的导入
    try:
        from phishguard_v1.service.ui_patch import DataFrame as CompatDataFrame
    except Exception:
        import pandas as pd
        CompatDataFrame = pd.DataFrame

try:
    from ..config import settings
    from ..features.fetcher import fetch_one
    from ..features.parser import extract_from_html
    from ..features.render import render_screenshot
except Exception:  # pragma: no cover - 直接运行包时的导入
    try:
        from phishguard_v1.config import settings
        from phishguard_v1.features.fetcher import fetch_one
        from phishguard_v1.features.parser import extract_from_html
        from phishguard_v1.features.render import render_screenshot
    except Exception:
        settings = None
        fetch_one = None
        extract_from_html = None
        render_screenshot = None
try:
    from ..models.inference import InferencePipeline
except Exception:  # pragma: no cover - 直接运行包时的导入
    try:
        from phishguard_v1.models.inference import InferencePipeline
    except Exception:
        class InferencePipeline:
            def __init__(self, **kwargs):
                pass
            def predict_url(self, url):
                return {
                    'url': url,
                    'final_prob': 0.5,
                    'url_prob': 0.5,
                    'fusion_prob': 0.5,
                    'pred_label': 0,
                    'decision': 'error'
                }

pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_dalwfr_v5.pt", enable_fusion=True)

SIGNAL_FEATURES = {
    "suspicious_js_inline": "可疑内联脚本",
    "external_form_ratio": "外部表单占比",
    "hidden_input_ratio": "隐藏字段密度",
    "http_security_header_count": "安全响应头数量",
    "http_tls_retry_flag": "TLS回退触发",
    "cookie_secure_ratio": "Secure Cookie 占比",
    "cookie_httponly_ratio": "HttpOnly Cookie 占比",
    "meta_sensitive_kw_flag": "敏感关键词标记",
    "title_entropy": "标题熵",
    "fingerprint_hash_len": "指纹哈希长度",
}


def _md_escape(value: Any) -> str:
    text = str(value) if value is not None else ""
    return text.replace("|", "\\|").replace("\n", " ")


def build_http_summary_block(features: Dict[str, Any]) -> str:
    lines = ["### 🌐 HTTP 信息"]
    status = features.get("status_code")
    if status is not None:
        lines.append(f"- **状态码**：{status}")
    content_type = features.get("content_type")
    if content_type:
        lines.append(f"- **Content-Type**：{content_type}")
    redirects = (features.get("meta") or {}).get("redirects") or []
    if redirects:
        lines.append(f"- **重定向链路**：{' → '.join(_md_escape(r) for r in redirects[:5])}")

    headers = features.get("headers") or {}
    if headers:
        lines.append("\n| Header | Value |\n| --- | --- |")
        for key, value in list(headers.items())[:10]:
            lines.append(f"| {_md_escape(key)} | {_md_escape(value)} |")
    else:
        lines.append("- 未获取到响应头 (站点可能拒绝连接或重定向至非 HTTP 页面)。")
    return "\n".join(lines)


def build_cookie_summary_block(features: Dict[str, Any]) -> str:
    lines = ["### 🍪 Cookie 信息"]
    cookies = features.get("cookies") or {}
    set_cookie = features.get("set_cookie") or ""
    if cookies:
        lines.append(f"- **Cookie 总数**：{len(cookies)}")
        lines.append("| Cookie | 值 |\n| --- | --- |")
        for key, value in list(cookies.items())[:10]:
            lines.append(f"| {_md_escape(key)} | {_md_escape(value)} |")
    else:
        lines.append("- 未检测到响应 Cookie。")
    if set_cookie:
        preview = _md_escape(set_cookie[:300]) + ("…" if len(set_cookie) > 300 else "")
        lines.append(f"- **Set-Cookie 原始串（截断）**：`{preview}`")
    return "\n".join(lines)


def build_meta_summary_block(features: Dict[str, Any]) -> str:
    lines = ["### 🧩 Meta / 指纹信息"]
    html_feats = features.get("html_feats") or {}
    meta_kv = html_feats.get("meta_kv") or {}
    if meta_kv:
        lines.append("| Meta 名称 | 内容 |\n| --- | --- |")
        for key, value in list(meta_kv.items())[:10]:
            lines.append(f"| {_md_escape(key)} | {_md_escape(value)} |")
    else:
        lines.append("- 未提取到 Meta 标签。")

    script_srcs = html_feats.get("script_srcs") or []
    stylesheets = html_feats.get("stylesheets") or []
    if script_srcs or stylesheets:
        lines.append("\n- **外部脚本**：")
        lines.extend([f"  - {_md_escape(src)}" for src in script_srcs[:5]])
        lines.append("- **外部样式表**：")
        lines.extend([f"  - {_md_escape(href)}" for href in stylesheets[:5]])
    return "\n".join(lines)


def format_probability(prob: float) -> str:
    return f"{prob * 100:.2f}%"


def get_risk_level(prob: float) -> Tuple[str, str]:
    if prob >= 0.9:
        return "🔴 高风险", "#f44336"
    if prob >= 0.7:
        return "🟡 中风险", "#ff9800"
    if prob >= 0.5:
        return "🟠 低风险", "#ffc107"
    return "🟢 安全", "#4caf50"


def generate_conclusion(pred: Dict[str, Any]) -> str:
    url_prob = pred.get("url_prob", 0)
    fusion_prob = pred.get("fusion_prob")
    final_prob = pred.get("final_prob", 0)
    label = pred.get("label", 0)

    parts = []
    parts.append("🚨 **检测结果：钓鱼网站**" if label == 1 else "✅ **检测结果：良性网站**")
    risk_level, _ = get_risk_level(final_prob)
    parts.append(f"📊 **风险等级：{risk_level}** ({format_probability(final_prob)})")

    parts.append("🤖 **模型分析：**")
    parts.append(f"   - URL 预训练模型：{format_probability(url_prob)}")
    if fusion_prob is not None:
        parts.append(f"   - FusionDNN 模型：{format_probability(fusion_prob)}")
    else:
        parts.append("   - FusionDNN 模型：未参与融合")

    parts.append("⚠️ **建议：** 避免访问此网站，可能存在安全风险" if label == 1 else "💡 **建议：** 网站看起来安全，但仍需保持警惕")
    return "\n\n".join(parts)


def generate_conclusion_html(pred: Dict[str, Any]) -> str:
    url_prob = pred.get("url_prob", 0)
    fusion_prob = pred.get("fusion_prob")
    final_prob = pred.get("final_prob", 0)
    label = pred.get("label", 0)
    risk_level, risk_class = get_risk_level(final_prob)

    # 生成概率条
    def generate_progress_bar(prob, color, label):
        percentage = prob * 100
        return f"""
        <div style="margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-size: 0.9rem; font-weight: 500;">{label}</span>
                <span style="font-size: 0.9rem;">{format_probability(prob)}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%; background: {color};"></div>
            </div>
        </div>
        """

    if label == 1:
        return (
            "<div class='result-section' style='background: linear-gradient(135deg, #fef2f2, #fee2e2); border-left: 4px solid #ef4444; position: relative;'>"
            "<div style='position: absolute; top: 1rem; right: 1rem; background: #ef4444; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>HIGH RISK</div>"
            f"<div style='font-size: 1.5rem; font-weight: 700; color: #dc2626; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;'>🚨 检测为钓鱼网站</div>"
            f"<div style='color: #7f1d1d; font-size: 1.1rem; margin-bottom: 1rem;'>风险等级: <span style='font-weight: 600;'>{risk_level}</span> ({format_probability(final_prob)})</div>"

            "<div style='margin: 1rem 0;'>"
            f"<div style='font-size: 1rem; font-weight: 600; color: #991b1b; margin-bottom: 0.75rem;'>📊 模型分析</div>"
            f"{generate_progress_bar(url_prob, '#ef4444', 'URL模型')}"
            f"{generate_progress_bar(fusion_prob if fusion_prob is not None else 0, '#ef4444', 'FusionDNN模型')}"
            f"{generate_progress_bar(final_prob, '#dc2626', '综合风险')}"
            "</div>"

            "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;'>"
            "<div style='background: #fecaca; padding: 1rem; border-radius: 8px; border-left: 3px solid #ef4444;'>"
            "<div style='font-size: 0.9rem; font-weight: 600; color: #991b1b; margin-bottom: 0.5rem;'>⚠️ 安全建议</div>"
            "<div style='font-size: 0.85rem; color: #7f1d1d;'>立即停止访问，使用安全工具扫描系统</div>"
            "</div>"
            "<div style='background: #fee2e2; padding: 1rem; border-radius: 8px; border-left: 3px solid #fca5a5;'>"
            "<div style='font-size: 0.9rem; font-weight: 600; color: #991b1b; margin-bottom: 0.5rem;'>🔒 推荐行动</div>"
            "<div style='font-size: 0.85rem; color: #7f1d1d;'>举报该网站，修改密码并监控账户</div>"
            "</div>"
            "</div>"
            "</div>"
        )
    else:
        return (
            "<div class='result-section' style='background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-left: 4px solid #22c55e; position: relative;'>"
            "<div style='position: absolute; top: 1rem; right: 1rem; background: #22c55e; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>SAFE</div>"
            f"<div style='font-size: 1.5rem; font-weight: 700; color: #166534; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;'>✅ 检测为良性网站</div>"
            f"<div style='color: #14532d; font-size: 1.1rem; margin-bottom: 1rem;'>风险等级: <span style='font-weight: 600;'>{risk_level}</span> ({format_probability(final_prob)})</div>"

            "<div style='margin: 1rem 0;'>"
            f"<div style='font-size: 1rem; font-weight: 600; color: #166534; margin-bottom: 0.75rem;'>📊 模型分析</div>"
            f"{generate_progress_bar(url_prob, '#22c55e', 'URL模型')}"
            f"{generate_progress_bar(fusion_prob if fusion_prob is not None else 0, '#22c55e', 'FusionDNN模型')}"
            f"{generate_progress_bar(final_prob, '#16a34a', '综合风险')}"
            "</div>"

            "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;'>"
            "<div style='background: #bbf7d0; padding: 1rem; border-radius: 8px; border-left: 3px solid #22c55e;'>"
            "<div style='font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.5rem;'>🛡️ 安全状态</div>"
            "<div style='font-size: 0.85rem; color: #14532d;'>网站技术特征正常，无明显风险</div>"
            "</div>"
            "<div style='background: #dcfce7; padding: 1rem; border-radius: 8px; border-left: 3px solid #86efac;'>"
            "<div style='font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.5rem;'>💡 建议措施</div>"
            "<div style='font-size: 0.85rem; color: #14532d;'>保持警惕，启用双因子认证</div>"
            "</div>"
            "</div>"
            "</div>"
        )


def build_probability_summary(pred: Dict[str, Any]) -> str:
    rows = [
        ("URL 模型", pred.get("url_prob")),
        ("FusionDNN 模型", pred.get("fusion_prob")),
        ("综合结果", pred.get("final_prob")),
    ]
    lines = ["### 概率拆解", "| 模块 | 钓鱼概率 |", "| --- | --- |"]
    for name, prob in rows:
        lines.append(f"| {name} | {format_probability(prob)} |" if prob is not None else f"| {name} | N/A |")
    return "\n".join(lines)


def build_detail_summary(details: Dict[str, Any] | None) -> str:
    if not details:
        return "### 推理细节\n- 暂无推理细节信息。"

    lines = ["### 推理细节"]
    decision = details.get("decision")
    if decision:
        lines.append(f"- **融合策略**：{decision}")

    weights = details.get("fusion_weights", {})
    if weights:
        lines.append("- **权重分配：**")
        lines.append(f"  - URL 模型：{weights.get('url', 0) * 100:.1f}%")
        lines.append(f"  - FusionDNN：{weights.get('fusion', 0) * 100:.1f}%")

    thresholds = details.get("thresholds", {})
    if thresholds:
        lines.append("- **判定阈值：**")
        lines.extend(
            [
                f"  - {key}：{value:.2f}" if isinstance(value, (int, float)) else f"  - {key}：{value}"
                for key, value in thresholds.items()
            ]
        )

    snapshot = details.get("feature_snapshot")
    if snapshot:
        contributions = []
        for key, label in SIGNAL_FEATURES.items():
            val = snapshot.get(key)
            if val is None:
                continue
            magnitude = abs(float(val))
            if magnitude <= 0:
                continue
            contributions.append((magnitude, label, float(val)))
        if contributions:
            contributions.sort(reverse=True)
            lines.append("- **关键指纹特征：**")
            for _, label, value in contributions[:5]:
                lines.append(f"  - {label}：{value:.2f}")
    return "\n".join(lines)


def create_feature_popup(features: Dict[str, Any]) -> str:
    popup_content: List[str] = []
    url_feats = features.get("url_feats", {})
    if url_feats:
        popup_content.append("### 🔗 URL 特征")
        popup_content.append(f"- **总长度**：{url_feats.get('url_len', 0)}")
        popup_content.append(f"- **域名长度**：{url_feats.get('host_len', 0)}")
        popup_content.append(f"- **路径长度**：{url_feats.get('path_len', 0)}")
        popup_content.append(f"- **数字字符数**：{url_feats.get('num_digits', 0)}")
        popup_content.append(f"- **特殊字符数**：{url_feats.get('num_specials', 0)}")
        popup_content.append(f"- **子域名深度**：{url_feats.get('subdomain_depth', 0)}")
        popup_content.append(f"- **是否包含 IP**：{'是' if url_feats.get('has_ip') else '否'}")
        popup_content.append(f"- **协议**：{'HTTPS' if url_feats.get('scheme_https') else 'HTTP'}")

    html_feats = features.get("html_feats", {})
    if html_feats:
        popup_content.append("\n### 📄 HTML 特征")
        popup_content.append(f"- **标题长度**：{html_feats.get('title_len', 0)}")
        popup_content.append(f"- **元标签数**：{html_feats.get('num_meta', 0)}")
        popup_content.append(f"- **链接数**：{html_feats.get('num_links', 0)}")
        popup_content.append(f"- **脚本数**：{html_feats.get('num_scripts', 0)}")
        popup_content.append(f"- **表单数**：{html_feats.get('num_forms', 0)}")
        popup_content.append(f"- **是否有密码输入**：{'是' if html_feats.get('has_password_input') else '否'}")
        popup_content.append(f"- **可疑脚本**：{'是' if html_feats.get('suspicious_js_inline') else '否'}")

    status_code = features.get("status_code")
    content_type = features.get("content_type")
    bytes_size = features.get("bytes")
    if status_code or content_type or bytes_size:
        popup_content.append("\n### 🌐 HTTP 响应特征")
        if status_code:
            popup_content.append(f"- **状态码**：{status_code}")
        if content_type:
            popup_content.append(f"- **内容类型**：{content_type}")
        if bytes_size:
            popup_content.append(f"- **响应大小**：{bytes_size} bytes")

    return "\n".join(popup_content) if popup_content else "### 特征摘要\n- 暂无特征信息。"


def build_history_rows(history: List[Dict[str, Any]]) -> List[List[str]]:
    rows = []
    for item in history:
        rows.append([item["time"], item["url"], item["probability"], item["label_text"]])
    return rows


def build_batch_results(results: List[Any]) -> Tuple[List[List[str]], str, Dict[str, int]]:
    headers = ["URL", "检测结果", "风险等级", "URL 模型", "FusionDNN", "综合概率", "备注"]
    rows: List[List[str]] = []
    tmp_file = tempfile.NamedTemporaryFile(
        prefix="phish_batch_",
        suffix=".csv",
        delete=False,
        dir=tempfile.gettempdir(),
    )
    csv_path = Path(tmp_file.name)
    phish_count = 0
    error_count = 0

    for idx, result in enumerate(results, start=1):
        if isinstance(result, Exception):
            rows.append([f"任务_{idx}", "失败", "-", "-", "-", "-", str(result)])
            error_count += 1
            continue

        pred = result.get("prediction", {})
        features = result.get("features", {})
        final_url = features.get("final_url") or features.get("request_url", f"URL_{idx}")
        final_prob = pred.get("final_prob", 0.0)
        label = pred.get("label", 0)
        result_text = "钓鱼" if label == 1 else "良性"
        risk_level, _ = get_risk_level(final_prob)
        if label == 1:
            phish_count += 1

        url_prob = format_probability(pred.get("url_prob", 0.0))
        fusion_prob_val = pred.get("fusion_prob")
        fusion_prob = format_probability(fusion_prob_val) if fusion_prob_val is not None else "N/A"
        final_prob_str = format_probability(final_prob)
        rows.append([final_url, result_text, risk_level, url_prob, fusion_prob, final_prob_str, ""])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for line in rows:
            writer.writerow(line)

    stats = {"total": len(results), "phish": phish_count, "errors": error_count}
    return rows, str(csv_path), stats


async def scan_single(url: str, screenshot: bool) -> Dict[str, Any]:
    # URL输入验证和错误处理
    if not url or not url.strip():
        raise ValueError("请输入有效的URL")

    # 基本URL格式验证
    url = url.strip()
    if not (url.startswith('http://') or url.startswith('https://')):
        # 尝试自动添加https://前缀
        if '://' not in url:
            url = 'https://' + url
        else:
            raise ValueError("URL必须以http://或https://开头")

    try:
        async with AsyncClient(timeout=settings.http_timeout, headers={"User-Agent": settings.user_agent}) as client:
            item = await fetch_one(url, client)

        if not item or not isinstance(item, dict):
            item = {
                "request_url": url,
                "final_url": url,
                "status_code": None,
                "content_type": None,
                "headers": {},
                "html": "",
                "bytes": 0,
                "url_feats": {},
                "ok": False,
                "meta": {"ok": False, "error": "抓取失败", "redirects": []},
            }

        html_content = item.get("html", "")
        final_url = item.get("final_url") or item.get("request_url", url)
        item["html_feats"] = extract_from_html(html_content, final_url)

        if screenshot:
            snap = render_screenshot(final_url)
            item.update(snap)

        if not item.get("url_feats"):
            from ..features.url_features import url_stats

            item["url_feats"] = url_stats(final_url)

        pred = pipe.predict(item)
        return {"prediction": pred, "features": item}
    except Exception as exc:
        error_item = {
            "request_url": url,
            "final_url": url,
            "status_code": None,
            "content_type": None,
            "headers": {},
            "html": "",
            "bytes": 0,
            "url_feats": {},
            "html_feats": {},
            "ok": False,
            "meta": {"ok": False, "error": str(exc), "redirects": []},
        }
        try:
            pred = pipe.predict(error_item)
            return {"prediction": pred, "features": error_item}
        except Exception:
            raise exc


async def scan_multiple(urls_text: str, screenshot: bool) -> List[Any]:
    # 批量URL输入验证
    urls = []
    invalid_urls = []

    for line in urls_text.splitlines():
        url = line.strip()
        if not url:
            continue

        # 基本URL格式验证
        if not (url.startswith('http://') or url.startswith('https://')):
            if '://' not in url:
                url = 'https://' + url
            else:
                invalid_urls.append(line)
                continue

        urls.append(url)

    if not urls:
        if invalid_urls:
            raise ValueError(f"检测到 {len(invalid_urls)} 个无效的URL格式")
        return []

    tasks = [scan_single(url, screenshot) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)


def clear_history(history_state):
    """清空历史记录"""
    return [], [], "历史记录已清空"

def export_history(history_state):
    """导出历史记录为CSV格式"""
    if not history_state:
        return "暂无历史记录可导出"

    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # 写入标题行
    writer.writerow(['时间', 'URL', '综合概率', 'URL模型概率', 'FusionDNN概率', '结果'])

    # 写入数据行
    for entry in history_state:
        writer.writerow([
            entry.get('timestamp', ''),
            entry.get('url', ''),
            entry.get('probability', ''),
            entry.get('url_prob', ''),
            entry.get('fusion_prob', ''),
            entry.get('label_text', '')
        ])

    return output.getvalue()

def validate_url_format(url):
    """快速验证URL格式"""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return url_pattern.match(url) is not None

def get_url_info(url):
    """获取URL基本信息"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return {
            'domain': parsed.netloc,
            'scheme': parsed.scheme,
            'path': parsed.path,
            'query': parsed.query,
            'is_https': parsed.scheme == 'https',
            'has_subdomain': len(parsed.netloc.split('.')) > 2
        }
    except:
        return {}

def update_single_result(result: Any, history: List[Dict[str, Any]]) -> Tuple[
    str,
    str,
    str,
    str,
    Dict[str, Any],
    Dict[str, Any],
    str,
    gr.Update,
    str,
    str,
    str,
    gr.Update,
    List[Dict[str, Any]],
]:
    history = list(history or [])
    history_rows = build_history_rows(history)

    if isinstance(result, Exception):
        conclusion = gr.HTML(
            "<div class='result-section' style='background: linear-gradient(135deg, #fef2f2, #fee2e2); border-left: 4px solid #ef4444;'>"
            f"<div style='font-size: 1.3rem; font-weight: 600; color: #dc2626; margin-bottom: 0.5rem;'>❌ 检测失败</div>"
            f"<div style='color: #7f1d1d;'>{result}</div>"
            "</div>"
        )
        status_html = (
            "<div class='status-indicator risk-danger'>"
            "<div style='font-size: 3rem; margin-bottom: 0.5rem;'>⚠️</div>"
            "<div style='font-size: 1.1rem; font-weight: 600;'>检测失败</div>"
            "<div style='font-size: 0.9rem; opacity: 0.8;'>请稍后重试或检查网络</div>"
            "</div>"
        )
        prob_summary = gr.HTML(
            "<div class='feature-card'>"
            "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
            "<span style='font-size: 1.3rem;'>📊</span>"
            "<div style='font-size: 1.1rem; font-weight: 600; color: #ef4444;'>概率拆解</div>"
            "</div>"
            "<div style='color: #ef4444;'>检测失败，暂无概率信息</div>"
            "</div>"
        )
        detail_summary = gr.HTML(
            "<div class='feature-card'>"
            "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
            "<span style='font-size: 1.3rem;'>🔍</span>"
            "<div style='font-size: 1.1rem; font-weight: 600; color: #ef4444;'>推理细节</div>"
            "</div>"
            "<div style='color: #ef4444;'>检测失败，暂无推理细节</div>"
            "</div>"
        )
        features_text = gr.HTML(
            "<div class='feature-card'>"
            "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
            "<span style='font-size: 1.3rem;'>🧩</span>"
            "<div style='font-size: 1.1rem; font-weight: 600; color: #ef4444;'>特征摘要</div>"
            "</div>"
            "<div style='color: #ef4444;'>暂无特征信息</div>"
            "</div>"
        )
        http_summary = gr.HTML(
            "<div class='feature-card'>"
            "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
            "<span style='font-size: 1.3rem;'>🌐</span>"
            "<div style='font-size: 1.1rem; font-weight: 600; color: #ef4444;'>HTTP 信息</div>"
            "</div>"
            "<div style='color: #ef4444;'>检测失败，暂无数据</div>"
            "</div>"
        )
        cookie_summary = gr.HTML(
            "<div class='feature-card'>"
            "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
            "<span style='font-size: 1.3rem;'>🍪</span>"
            "<div style='font-size: 1.1rem; font-weight: 600; color: #ef4444;'>Cookie 信息</div>"
            "</div>"
            "<div style='color: #ef4444;'>检测失败，暂无数据</div>"
            "</div>"
        )
        meta_summary = gr.HTML(
            "<div class='feature-card'>"
            "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
            "<span style='font-size: 1.3rem;'>🧩</span>"
            "<div style='font-size: 1.1rem; font-weight: 600; color: #ef4444;'>Meta / 指纹信息</div>"
            "</div>"
            "<div style='color: #ef4444;'>检测失败，暂无数据</div>"
            "</div>"
        )
        return (
            conclusion,
            status_html,
            prob_summary,
            detail_summary,
            {},
            {},
            features_text,
            http_summary,
            cookie_summary,
            meta_summary,
            gr.update(value=None, visible=False),
            "",
            "",
            "",
            gr.update(value=history_rows),
            history,
        )

    pred = result.get("prediction", {}) or {}
    features = result.get("features", {}) or {}

    conclusion = gr.HTML(
        generate_conclusion_html(pred)
    )
    final_prob = pred.get("final_prob", 0.0)
    risk_level, risk_class = get_risk_level(final_prob)

    status_class = f"risk-{risk_class}"
    status_emoji = "🚨" if pred.get('label', 0) == 1 else "✅"
    status_label = "钓鱼网站" if pred.get('label', 0) == 1 else "良性网站"

    status_html = (
        f"<div class='status-indicator {status_class}'>"
        f"<div style='font-size: 3rem; margin-bottom: 0.5rem;'>{status_emoji}</div>"
        f"<div style='font-size: 1.2rem; font-weight: 600;'>{risk_level}</div>"
        f"<div style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>{format_probability(final_prob)}</div>"
        f"<div style='font-size: 1rem;'>{status_label}</div>"
        "</div>"
    )

    prob_summary = build_probability_summary(pred)
    detail_summary = build_detail_summary(pred.get("details"))
    features_text = create_feature_popup(features)
    http_summary = build_http_summary_block(features)
    cookie_summary = build_cookie_summary_block(features)
    meta_summary = build_meta_summary_block(features)

    final_url = features.get("final_url") or features.get("request_url", "")
    status_code_val = str(features.get("status_code", "")) if features.get("status_code") else ""
    content_type_val = features.get("content_type", "") or ""

    screenshot_path = features.get("screenshot_path")
    screenshot_update = gr.update(value=screenshot_path, visible=bool(screenshot_path))

    history_entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url": final_url,
        "probability": format_probability(final_prob),
        "label_text": "钓鱼" if pred.get("label", 0) == 1 else "良性",
    }
    history.insert(0, history_entry)
    history = history[:50]
    history_rows = build_history_rows(history)

    return (
        conclusion,
        status_html,
        prob_summary,
        detail_summary,
        pred,
        pred.get("details", {}),
        features_text,
        http_summary,
        cookie_summary,
        meta_summary,
        screenshot_update,
        final_url,
        status_code_val,
        content_type_val,
        gr.update(value=history_rows),
        history,
    )


def build_interface():
    custom_css = """
    /* 全局样式 */
    .main-container {
        max-width: 1400px;
        margin: auto;
        padding: 20px;
    }

    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
    }

    /* 风险等级样式 */
    .risk-safe {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: 1px solid #047857 !important;
        color: white !important;
        animation: pulse-safe 2s infinite;
    }

    .risk-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        border: 1px solid #b45309 !important;
        color: white !important;
        animation: pulse-warning 2s infinite;
    }

    .risk-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: 1px solid #b91c1c !important;
        color: white !important;
        animation: pulse-danger 2s infinite;
    }

    @keyframes pulse-safe {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
    }

    @keyframes pulse-warning {
        0%, 100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(245, 158, 11, 0); }
    }

    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    }

    /* 头部样式 */
    .gradient-bg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }

    .gradient-bg::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }

    /* 卡片样式 */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }

    .feature-card:hover::before {
        transform: scaleX(1);
    }

    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }

    /* 状态指示器 */
    .status-indicator {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        font-weight: 700;
        transition: all 0.4s ease;
        border: 3px solid transparent;
        position: relative;
        overflow: hidden;
    }

    .status-indicator::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: all 0.6s ease;
    }

    .status-indicator:hover::after {
        width: 100%;
        height: 100%;
    }

    /* 进度条样式 */
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
        transition: width 0.6s ease;
        position: relative;
        overflow: hidden;
    }

    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: progress-shimmer 2s infinite;
    }

    @keyframes progress-shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* 标签页样式 */
    .tab-nav {
        border-bottom: 3px solid #e2e8f0;
        margin-bottom: 2rem;
        position: relative;
    }

    /* 结果展示区域 */
    .result-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        position: relative;
    }

    .result-section:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    /* 历史表格样式 */
    .history-table {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* 按钮样式 */
    .btn-primary {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .btn-primary::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }

    .btn-primary:hover::before {
        left: 100%;
    }

    .btn-primary:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4) !important;
    }

    .btn-primary:active {
        transform: translateY(0) scale(0.98) !important;
    }

    /* 输入框样式 */
    .gradio-textbox {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }

    .gradio-textbox:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* 加载动画 */
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e2e8f0;
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* 工具提示样式 */
    .tooltip {
        position: relative;
        cursor: help;
    }

    .tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: #1f2937;
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 12px;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
    }

    .tooltip:hover::after {
        opacity: 1;
        visibility: visible;
    }

    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-container {
            padding: 10px;
        }

        .gradient-bg {
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .feature-card {
            padding: 1.5rem;
        }

        .status-indicator {
            padding: 1.5rem;
        }
    }
    """

    with gr.Blocks(
        title="PhishGuard v5 - Advanced Phishing Detection",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as demo:
        history_state = gr.State([])
        gr.HTML(
            """
            <div class="gradient-bg">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🛡️ PhishGuard v5</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Advanced Phishing Detection System</p>
                <div style="margin-top: 1rem; display: flex; gap: 2rem; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.5rem;">🤖</span>
                        <div>
                            <div style="font-weight: 600;">URL预训练模型</div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">BERT语义理解</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.5rem;">🧠</span>
                        <div>
                            <div style="font-weight: 600;">FusionDNN模型</div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">v5版本 92特征</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.5rem;">📊</span>
                        <div>
                            <div style="font-weight: 600;">全链路诊断</div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">概率拆解+推理详情</div>
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.TabItem("🔍 单 URL 检测"):
                with gr.Row():
                    with gr.Column(scale=4):
                        url_input = gr.Textbox(
                            label="🔗 输入要检测的 URL",
                            placeholder="https://example.com",
                            value="https://www.baidu.com",
                            show_label=True,
                            container=True,
                            scale=4,
                        )
                    with gr.Column(scale=1):
                        with gr.Group():
                            url_info_display = gr.HTML(
                                "<div class='feature-card' style='text-align: center; padding: 1rem;'>"
                                "<div style='font-size: 0.9rem; color: #6b7280; margin-bottom: 0.5rem;'>🔗 URL信息</div>"
                                "<div style='font-size: 0.85rem; color: #9ca3af;'>输入URL后显示分析</div>"
                                "</div>"
                            )
                            screenshot_cb = gr.Checkbox(
                                label="📸 启用截图功能",
                                value=False,
                                info="生成页面截图"
                            )
                            quick_validate_btn = gr.Button(
                                "⚡ 快速验证",
                                variant="secondary",
                                size="sm"
                            )
                            scan_btn = gr.Button(
                                "🔍 开始检测",
                                variant="primary",
                                size="lg",
                                scale=1
                            )

                with gr.Row():
                    with gr.Column(scale=2):
                        conclusion_box = gr.HTML(
                            "<div class='result-section' style='text-align: center; padding: 2rem;'>"
                            "<div style='font-size: 1.2rem; color: #6b7280; margin-bottom: 0.5rem;'>准备就绪</div>"
                            "<div style='font-size: 1.5rem; font-weight: 600; color: #374151;'>请输入 URL 并点击检测</div>"
                            "</div>"
                        )
                    with gr.Column(scale=1):
                        status_indicator = gr.HTML(
                            "<div class='status-indicator' style='background: linear-gradient(135deg, #f3f4f6, #e5e7eb); color: #6b7280;'>"
                            "<div style='font-size: 3rem; margin-bottom: 0.5rem;'>⏳</div>"
                            "<div style='font-size: 1.1rem; font-weight: 600;'>等待检测</div>"
                            "<div style='font-size: 0.9rem; opacity: 0.8;'>输入URL开始分析</div>"
                            "</div>"
                        )

                with gr.Row():
                    with gr.Column():
                        probability_summary = gr.HTML(
                            "<div class='feature-card'>"
                            "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
                            "<span style='font-size: 1.5rem;'>📊</span>"
                            "<div style='font-size: 1.2rem; font-weight: 600;'>概率拆解</div>"
                            "</div>"
                            "<div style='color: #6b7280; font-size: 0.95rem;'>等待检测...</div>"
                            "</div>"
                        )
                    with gr.Column():
                        detail_summary = gr.HTML(
                            "<div class='feature-card'>"
                            "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
                            "<span style='font-size: 1.5rem;'>🔍</span>"
                            "<div style='font-size: 1.2rem; font-weight: 600;'>推理细节</div>"
                            "</div>"
                            "<div style='color: #6b7280; font-size: 0.95rem;'>等待检测...</div>"
                            "</div>"
                        )

                with gr.Accordion("📊 详细分析结果", open=False):
                    with gr.Tabs():
                        with gr.TabItem("🎯 核心数据"):
                            with gr.Row():
                                pred_json = gr.JSON(
                                    label="📈 预测数据",
                                    value={},
                                    show_label=True
                                )
                                details_json = gr.JSON(
                                    label="🔧 推理细节",
                                    value={},
                                    show_label=True
                                )

                        with gr.TabItem("🌐 技术信息"):
                            with gr.Row():
                                final_url = gr.Textbox(
                                    label="🔗 最终 URL",
                                    interactive=False,
                                    show_copy_button=True
                                )
                                status_code = gr.Textbox(
                                    label="📊 状态码",
                                    interactive=False
                                )
                                content_type = gr.Textbox(
                                    label="📄 内容类型",
                                    interactive=False
                                )

                        with gr.TabItem("📈 特征分析"):
                            features_markdown = gr.HTML(
                                "<div class='feature-card'>"
                                "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
                                "<span style='font-size: 1.3rem;'>🧩</span>"
                                "<div style='font-size: 1.1rem; font-weight: 600;'>特征摘要</div>"
                                "</div>"
                                "<div style='color: #6b7280; font-size: 0.9rem;'>暂无特征信息...</div>"
                                "</div>"
                            )

                        with gr.TabItem("🌐 HTTP分析"):
                            http_markdown = gr.HTML(
                                "<div class='feature-card'>"
                                "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
                                "<span style='font-size: 1.3rem;'>🌐</span>"
                                "<div style='font-size: 1.1rem; font-weight: 600;'>HTTP 信息</div>"
                                "</div>"
                                "<div style='color: #6b7280; font-size: 0.9rem;'>等待检测...</div>"
                                "</div>"
                            )

                        with gr.TabItem("🍪 Cookie分析"):
                            cookie_markdown = gr.HTML(
                                "<div class='feature-card'>"
                                "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
                                "<span style='font-size: 1.3rem;'>🍪</span>"
                                "<div style='font-size: 1.1rem; font-weight: 600;'>Cookie 信息</div>"
                                "</div>"
                                "<div style='color: #6b7280; font-size: 0.9rem;'>等待检测...</div>"
                                "</div>"
                            )

                        with gr.TabItem("🧩 指纹分析"):
                            meta_markdown = gr.HTML(
                                "<div class='feature-card'>"
                                "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
                                "<span style='font-size: 1.3rem;'>🧩</span>"
                                "<div style='font-size: 1.1rem; font-weight: 600;'>Meta / 指纹信息</div>"
                                "</div>"
                                "<div style='color: #6b7280; font-size: 0.9rem;'>等待检测...</div>"
                                "</div>"
                            )

                        with gr.TabItem("📸 页面截图"):
                            screenshot_image = gr.Image(
                                label="页面截图",
                                visible=False,
                                show_label=True,
                                show_download_button=True
                            )

                with gr.Accordion("📊 统计信息", open=True):
                    with gr.Row():
                        with gr.Column():
                            stats_display = gr.HTML(
                                """
                                <div class="feature-card">
                                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                                        <span style="font-size: 1.5rem;">📈</span>
                                        <div style="font-size: 1.2rem; font-weight: 600;">检测统计</div>
                                    </div>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                                        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 8px;">
                                            <div style="font-size: 1.5rem; font-weight: 700; color: #3b82f6;">0</div>
                                            <div style="font-size: 0.9rem; color: #6b7280;">总检测数</div>
                                        </div>
                                        <div style="text-align: center; padding: 1rem; background: #f0fdf4; border-radius: 8px;">
                                            <div style="font-size: 1.5rem; font-weight: 700; color: #22c55e;">0</div>
                                            <div style="font-size: 0.9rem; color: #6b7280;">安全网站</div>
                                        </div>
                                        <div style="text-align: center; padding: 1rem; background: #fef2f2; border-radius: 8px;">
                                            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">0</div>
                                            <div style="font-size: 0.9rem; color: #6b7280;">危险网站</div>
                                        </div>
                                        <div style="text-align: center; padding: 1rem; background: #fefce8; border-radius: 8px;">
                                            <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">0%</div>
                                            <div style="font-size: 0.9rem; color: #6b7280;">检测准确率</div>
                                        </div>
                                    </div>
                                </div>
                                """
                            )

                with gr.Accordion("🗂 历史记录", open=False):
                    with gr.Row():
                        with gr.Column(scale=4):
                            history_table = gr.DataFrame(
                                headers=["时间", "URL", "综合概率", "结论"],
                                datatype=["str", "str", "str", "str"],
                                value=[],
                                interactive=False,
                                wrap=True,
                            )
                        with gr.Column(scale=1):
                            clear_history_btn = gr.Button(
                                "🧹 清空记录",
                                variant="secondary",
                                size="sm"
                            )
                            export_history_btn = gr.Button(
                                "📥 导出历史",
                                variant="secondary",
                                size="sm"
                            )
                            stats_refresh_btn = gr.Button(
                                "🔄 刷新统计",
                                variant="secondary",
                                size="sm"
                            )

            with gr.TabItem("📋 批量检测"):
                gr.HTML(
                    """
                    <div style='background: linear-gradient(135deg, #f3f4f6, #e5e7eb); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
                        <h3 style='margin: 0 0 0.5rem 0; font-size: 1.3rem; color: #374151;'>📋 批量检测多个 URL</h3>
                        <p style='margin: 0; color: #6b7280;'>每行输入一个URL，系统将依次进行安全检测分析</p>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        urls_textarea = gr.TextArea(
                            label="🔗 输入 URL 列表",
                            placeholder="https://example.com\nhttps://google.com\nhttps://github.com",
                            lines=10,
                            show_label=True,
                            container=True,
                        )
                    with gr.Column(scale=1):
                        gr.HTML(
                            """
                            <div class='feature-card' style='height: 100%;'>
                                <div style='margin-bottom: 1rem;'>
                                    <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>⚙️ 检测设置</div>
                                </div>
                                <div style='margin-bottom: 1rem;'>
                                    <div style='color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem;'>批量检测选项</div>
                                </div>
                            </div>
                            """
                        )
                        batch_screenshot_cb = gr.Checkbox(
                            label="📸 启用截图",
                            value=False,
                            info="为每个URL生成截图"
                        )
                        batch_scan_btn = gr.Button(
                            "🚀 开始批量检测",
                            variant="primary",
                            size="lg",
                            scale=1
                        )

                        gr.HTML(
                            """
                            <div style='margin-top: 1rem; padding: 1rem; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                                <div style='font-size: 0.9rem; color: #1e40af; font-weight: 600;'>💡 提示</div>
                                <div style='font-size: 0.85rem; color: #1e3a8a; margin-top: 0.25rem;'>批量检测会消耗更多时间和资源，建议一次检测不超过50个URL</div>
                            </div>
                            """
                        )

                with gr.Accordion("📈 批量检测结果", open=True):
                    batch_status = gr.HTML(
                        "<div style='text-align: center; padding: 2rem; color: #6b7280;'>"
                        "<div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>⏳ 准备批量检测</div>"
                        "<div style='font-size: 0.95rem;'>输入URL列表后点击开始检测</div>"
                        "</div>"
                    )

                    results_table = gr.DataFrame(
                        headers=["URL", "检测结果", "风险等级", "URL模型", "FusionDNN", "综合概率", "处理时间"],
                        datatype=["str", "str", "str", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        wrap=True,
                    )

                    with gr.Row():
                        results_file = gr.File(
                            label="📥 下载检测结果",
                            visible=False,
                            show_label=True
                        )
                        clear_results_btn = gr.Button(
                            "🧹 清空结果",
                            variant="secondary",
                            size="sm"
                        )

            with gr.TabItem("🧪 测试样例"):
                gr.HTML(
                    """
                    <div style='background: linear-gradient(135deg, #fef3c7, #fde68a); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #f59e0b;'>
                        <h3 style='margin: 0 0 0.5rem 0; font-size: 1.3rem; color: #92400e;'>🧪 测试样例</h3>
                        <p style='margin: 0; color: #78350f;'>选择预设样例快速评估系统检测能力</p>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class='feature-card' style='border-left: 4px solid #ef4444;'>
                                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>
                                    <span style='font-size: 1.5rem;'>🚨</span>
                                    <div style='font-size: 1.2rem; font-weight: 600; color: #dc2626;'>钓鱼网站样例</div>
                                </div>
                                <div style='color: #6b7280; font-size: 0.9rem;'>真实的钓鱼网站，用于测试检测准确性</div>
                            </div>
                            """
                        )
                        phishing_examples = gr.DataFrame(
                            value=[],
                            headers=["URL", "描述"],
                            datatype=["str", "str"],
                            interactive=False,
                        )
                        load_phishing_btn = gr.Button(
                            "🚨 加载钓鱼网站样例",
                            variant="stop",
                            size="sm"
                        )

                    with gr.Column():
                        gr.HTML(
                            """
                            <div class='feature-card' style='border-left: 4px solid #22c55e;'>
                                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>
                                    <span style='font-size: 1.5rem;'>✅</span>
                                    <div style='font-size: 1.2rem; font-weight: 600; color: #16a34a;'>良性网站样例</div>
                                </div>
                                <div style='color: #6b7280; font-size: 0.9rem;'>知名安全网站，用于测试误报率</div>
                            </div>
                            """
                        )
                        benign_examples = gr.DataFrame(
                            value=[],
                            headers=["URL", "描述"],
                            datatype=["str", "str"],
                            interactive=False,
                        )
                        load_benign_btn = gr.Button(
                            "✅ 加载良性网站样例",
                            variant="primary",
                            size="sm"
                        )
                        refresh_tables_btn = gr.Button(
                            "🔄 刷新表格数据",
                            variant="secondary",
                            size="sm"
                        )

                gr.HTML(
                    """
                    <div style='background: linear-gradient(135deg, #ede9fe, #ddd6fe); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; border: 1px solid #8b5cf6;'>
                        <h3 style='margin: 0 0 0.5rem 0; font-size: 1.3rem; color: #5b21b6;'>🎯 快速检测</h3>
                        <p style='margin: 0; color: #6d28d9;'>选择上方样例或手工输入URL进行检测</p>
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        test_url_input = gr.Textbox(
                            label="🔗 测试 URL",
                            placeholder="选中表格中的URL会自动填入，或手工输入",
                            value="https://www.baidu.com",
                            show_label=True,
                            container=True,
                        )
                    with gr.Column(scale=1):
                        test_screenshot_cb = gr.Checkbox(
                            label="📸 启用截图功能",
                            value=False,
                            info="生成页面截图"
                        )
                        test_scan_btn = gr.Button(
                            "🔍 开始检测",
                            variant="primary",
                            size="lg"
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        test_conclusion = gr.HTML(
                            "<div class='result-section' style='text-align: center;'>"
                            "<div style='font-size: 1.1rem; color: #6b7280; margin-bottom: 0.5rem;'>准备就绪</div>"
                            "<div style='font-size: 1.3rem; font-weight: 600; color: #374151;'>请选择 URL 并点击检测</div>"
                            "</div>"
                        )
                    with gr.Column(scale=1):
                        test_status = gr.HTML(
                            "<div class='status-indicator' style='background: linear-gradient(135deg, #f3f4f6, #e5e7eb); color: #6b7280;'>"
                            "<div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>⏳</div>"
                            "<div style='font-size: 1rem; font-weight: 600;'>等待检测</div>"
                            "</div>"
                        )

            with gr.TabItem("ℹ️ 系统信息"):
                gr.HTML(
                    """
                    <div class='gradient-bg' style='background: linear-gradient(135deg, #6366f1, #8b5cf6);'>
                        <h3 style='margin: 0 0 1rem 0; font-size: 1.5rem;'>ℹ️ 系统信息</h3>
                        <p style='margin: 0; opacity: 0.9;'>模型版本与系统配置详情</p>
                    </div>
                    """
                )

                with gr.Tabs():
                    with gr.TabItem("🤖 模型信息"):
                        gr.HTML(
                            """
                            <div class='feature-card'>
                                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem;'>
                                    <span style='font-size: 1.5rem;'>🤖</span>
                                    <div style='font-size: 1.3rem; font-weight: 600;'>当前模型配置</div>
                                </div>

                                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                                    <div style='background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                                        <div style='font-weight: 600; color: #1e40af; margin-bottom: 0.5rem;'>URL预训练模型</div>
                                        <div style='color: #1e3a8a; font-size: 0.9rem;'>BERT-based钓鱼检测</div>
                                        <div style='color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;'>语义理解+上下文分析</div>
                                    </div>

                                    <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px; border-left: 4px solid #22c55e;'>
                                        <div style='font-weight: 600; color: #166534; margin-bottom: 0.5rem;'>FusionDNN模型 v5</div>
                                        <div style='color: #15803d; font-size: 0.9rem;'>92特征深度融合</div>
                                        <div style='color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;'>HTTP+Cookie+Meta特征</div>
                                    </div>

                                    <div style='background: #fefce8; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;'>
                                        <div style='font-weight: 600; color: #92400e; margin-bottom: 0.5rem;'>模型性能</div>
                                        <div style='color: #78350f; font-size: 0.9rem;'>验证集: ACC 0.973 / AUC 0.991</div>
                                        <div style='color: #78350f; font-size: 0.9rem;'>测试集: ACC 0.975 / AUC 0.993</div>
                                    </div>
                                </div>
                            </div>
                            """
                        )

                    with gr.TabItem("⚙️ 推理配置"):
                        gr.HTML(
                            """
                            <div class='feature-card'>
                                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem;'>
                                    <span style='font-size: 1.5rem;'>⚙️</span>
                                    <div style='font-size: 1.3rem; font-weight: 600;'>推理阈值配置</div>
                                </div>

                                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;'>
                                    <div style='background: #fafafa; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;'>
                                        <div style='font-weight: 600; color: #374151; margin-bottom: 0.5rem;'>URL模型阈值</div>
                                        <div style='color: #6b7280; font-size: 0.9rem;'>动态阈值调整</div>
                                        <div style='background: #e5e7eb; height: 4px; border-radius: 2px; margin: 0.5rem 0;'></div>
                                        <div style='text-align: center; color: #374151; font-weight: 600;'>0.35-0.65</div>
                                    </div>

                                    <div style='background: #fafafa; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;'>
                                        <div style='font-weight: 600; color: #374151; margin-bottom: 0.5rem;'>Fusion模型阈值</div>
                                        <div style='color: #6b7280; font-size: 0.9rem;'>特征融合阈值</div>
                                        <div style='background: #e5e7eb; height: 4px; border-radius: 2px; margin: 0.5rem 0;'></div>
                                        <div style='text-align: center; color: #374151; font-weight: 600;'>0.45-0.75</div>
                                    </div>

                                    <div style='background: #fafafa; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;'>
                                        <div style='font-weight: 600; color: #374151; margin-bottom: 0.5rem;'>最终决策阈值</div>
                                        <div style='color: #6b7280; font-size: 0.9rem;'>综合判断阈值</div>
                                        <div style='background: #e5e7eb; height: 4px; border-radius: 2px; margin: 0.5rem 0;'></div>
                                        <div style='text-align: center; color: #374151; font-weight: 600;'>0.50-0.80</div>
                                    </div>
                                </div>
                            </div>
                            """
                        )

                    with gr.TabItem("📈 使用指南"):
                        gr.HTML(
                            """
                            <div class='feature-card'>
                                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem;'>
                                    <span style='font-size: 1.5rem;'>📈</span>
                                    <div style='font-size: 1.3rem; font-weight: 600;'>推荐使用流程</div>
                                </div>

                                <div style='background: linear-gradient(135deg, #f8fafc, #f1f5f9); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
                                    <div style='font-weight: 600; color: #475569; margin-bottom: 1rem; font-size: 1.1rem;'>🔍 单URL检测流程</div>
                                    <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
                                        <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                            <div style='background: #3b82f6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold;'>1</div>
                                            <div style='color: #475569;'>输入或选择URL进行检测</div>
                                        </div>
                                        <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                            <div style='background: #3b82f6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold;'>2</div>
                                            <div style='color: #475569;'>查看概率拆解和推理细节</div>
                                        </div>
                                        <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                            <div style='background: #3b82f6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold;'>3</div>
                                            <div style='color: #475569;'>根据结果做出安全判断</div>
                                        </div>
                                    </div>
                                </div>

                                <div style='background: linear-gradient(135deg, #fef3c7, #fde68a); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
                                    <div style='font-weight: 600; color: #92400e; margin-bottom: 1rem; font-size: 1.1rem;'>📋 批量检测流程</div>
                                    <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
                                        <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                            <div style='background: #f59e0b; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold;'>1</div>
                                            <div style='color: #78350f;'>批量粘贴URL列表（每行一个）</div>
                                        </div>
                                        <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                            <div style='background: #f59e0b; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold;'>2</div>
                                            <div style='color: #78350f;'>点击开始批量检测</div>
                                        </div>
                                        <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                            <div style='background: #f59e0b; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold;'>3</div>
                                            <div style='color: #78350f;'>查看详细结果并导出CSV报告</div>
                                        </div>
                                    </div>
                                </div>

                                <div style='background: linear-gradient(135deg, #dcfce7, #bbf7d0); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
                                    <div style='font-weight: 600; color: #166534; margin-bottom: 1rem; font-size: 1.1rem;'>💡 最佳实践建议</div>
                                    <ul style='margin: 0; padding-left: 1.5rem; color: #15803d;'>
                                        <li style='margin-bottom: 0.5rem;'>对于未知网站，建议开启截图功能进行更全面的分析</li>
                                        <li style='margin-bottom: 0.5rem;'>批量检测时，建议每次不超过50个URL以确保响应速度</li>
                                        <li style='margin-bottom: 0.5rem;'>关注HTTP响应头和Cookie设置，这些特征能有效识别威胁</li>
                                        <li>定期查看历史记录，追踪检测模式的演变</li>
                                    </ul>
                                </div>
                            </div>
                            """
                        )

        def on_scan_click(url: str, screenshot: bool, history: List[Dict[str, Any]]):
            # 输入验证
            if not url or not url.strip():
                error_result = ValueError("请输入要检测的URL")
                return update_single_result(error_result, history)

            try:
                result = asyncio.run(scan_single(url, screenshot))
                return update_single_result(result, history)
            except ValueError as ve:
                # 处理URL格式错误
                return update_single_result(ve, history)
            except Exception as exc:
                return update_single_result(exc, history)

        scan_btn.click(
            fn=on_scan_click,
            inputs=[url_input, screenshot_cb, history_state],
            outputs=[
                conclusion_box,
                status_indicator,
                probability_summary,
                detail_summary,
                pred_json,
                details_json,
                features_markdown,
                http_markdown,
                cookie_markdown,
                meta_markdown,
                screenshot_image,
                final_url,
                status_code,
                content_type,
                history_table,
                history_state,
            ],
        )

        def on_test_scan(url: str, screenshot: bool, history: List[Dict[str, Any]]):
            # 输入验证
            if not url or not url.strip():
                result = ValueError("请输入要检测的URL")
            else:
                try:
                    result = asyncio.run(scan_single(url, screenshot))
                except ValueError as ve:
                    result = ve
                except Exception as exc:
                    result = exc
            (
                conclusion,
                status_html,
                prob_summary,
                detail_summary_val,
                pred_data,
                pred_details,
                features_text,
                http_summary,
                cookie_summary,
                meta_summary,
                screenshot_update,
                final_url_val,
                status_code_val,
                content_type_val,
                history_table_update,
                history_value,
            ) = update_single_result(result, history)
            return (
                conclusion,
                status_html,
                conclusion,
                status_html,
                prob_summary,
                detail_summary_val,
                pred_data,
                pred_details,
                features_text,
                http_summary,
                cookie_summary,
                meta_summary,
                screenshot_update,
                final_url_val,
                status_code_val,
                content_type_val,
                history_table_update,
                history_value,
            )

        test_scan_btn.click(
            fn=on_test_scan,
            inputs=[test_url_input, test_screenshot_cb, history_state],
            outputs=[
                test_conclusion,
                test_status,
                conclusion_box,
                status_indicator,
                probability_summary,
                detail_summary,
                pred_json,
                details_json,
                features_markdown,
                http_markdown,
                cookie_markdown,
                meta_markdown,
                screenshot_image,
                final_url,
                status_code,
                content_type,
                history_table,
                history_state,
            ],
        )

        def clear_history():
            return gr.update(value=[]), []

        clear_history_btn.click(fn=clear_history, outputs=[history_table, history_state])

        # 快速验证URL功能
        def on_quick_validate(url: str):
            if not url or not url.strip():
                return gr.update(value="""
                    <div class='feature-card' style='text-align: center; padding: 1rem; border-left: 4px solid #ef4444;'>
                        <div style='color: #ef4444; font-size: 0.9rem;'>⚠️ 请输入URL</div>
                    </div>
                """)

            is_valid = validate_url_format(url)
            if is_valid:
                # 获取URL信息
                url_info = get_url_info(url)
                return gr.update(value=url_info)
            else:
                return gr.update(value="""
                    <div class='feature-card' style='text-align: center; padding: 1rem; border-left: 4px solid #ef4444;'>
                        <div style='color: #ef4444; font-size: 0.9rem;'>❌ URL格式无效</div>
                        <div style='color: #6b7280; font-size: 0.8rem; margin-top: 0.25rem;'>请检查URL格式</div>
                    </div>
                """)

        # 刷新统计信息
        def refresh_statistics(history: List[Dict[str, Any]]):
            total = len(history)
            safe = sum(1 for item in history if item.get("label", 0) == 0)
            phish = sum(1 for item in history if item.get("label", 0) == 1)
            accuracy = (safe + phish) / total * 100 if total > 0 else 0

            return gr.update(value=f"""
                <div class="feature-card">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                        <span style="font-size: 1.5rem;">📈</span>
                        <div style="font-size: 1.2rem; font-weight: 600;">检测统计</div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #3b82f6;">{total}</div>
                            <div style="font-size: 0.9rem; color: #6b7280;">总检测数</div>
                        </div>
                        <div style="text-align: center; padding: 1rem; background: #f0fdf4; border-radius: 8px; border: 1px solid #dcfce7;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #22c55e;">{safe}</div>
                            <div style="font-size: 0.9rem; color: #6b7280;">安全网站</div>
                        </div>
                        <div style="text-align: center; padding: 1rem; background: #fef2f2; border-radius: 8px; border: 1px solid #fecaca;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{phish}</div>
                            <div style="font-size: 0.9rem; color: #6b7280;">危险网站</div>
                        </div>
                        <div style="text-align: center; padding: 1rem; background: #fefce8; border-radius: 8px; border: 1px solid #fef3c7;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">{accuracy:.1f}%</div>
                            <div style="font-size: 0.9rem; color: #6b7280;">检测准确率</div>
                        </div>
                    </div>
                </div>
            """)

        def on_batch_scan(urls: str, screenshot: bool):
            # 输入验证
            if not urls or not urls.strip():
                return (
                    "❌ 请输入要检测的URL列表",
                    gr.update(value=[]),
                    gr.update(value=None, visible=False),
                )

            # 验证URL格式
            url_lines = [line.strip() for line in urls.splitlines() if line.strip()]
            if not url_lines:
                return (
                    "❌ 未找到有效的URL",
                    gr.update(value=[]),
                    gr.update(value=None, visible=False),
                )

            try:
                results = asyncio.run(scan_multiple(urls, screenshot))
                rows, csv_path, stats = build_batch_results(results)
                summary = (
                    f"✅ 共检测 {stats['total']} 条 URL，其中钓鱼 {stats['phish']} 条"
                    + (f"，失败 {stats['errors']} 条" if stats['errors'] else "")
                )
                return (
                    summary,
                    gr.update(value=rows),
                    gr.update(value=csv_path, visible=True),
                )
            except ValueError as ve:
                # 处理URL格式错误
                return (
                    f"❌ URL格式错误：{ve}",
                    gr.update(value=[]),
                    gr.update(value=None, visible=False),
                )
            except Exception as exc:
                return (
                    f"❌ 批量检测失败：{exc}",
                    gr.update(value=[]),
                    gr.update(value=None, visible=False),
                )

        batch_scan_btn.click(
            fn=on_batch_scan,
            inputs=[urls_textarea, batch_screenshot_cb],
            outputs=[batch_status, results_table, results_file],
        )

        def load_phishing_examples():
            sample_data = Path("data_massive/dataset_batch1_final.parquet")
            if sample_data.exists():
                df = pd.read_parquet(sample_data)
                urls = df[df["label"] == 1]["final_url"].dropna().head(50).tolist()
                first_url = urls[0] if urls else ""
                return "\n".join(urls), first_url, first_url
            else:
                fallback = [
                    "http://verify-paypal-account.com",
                    "http://apple-security-update.info",
                    "http://microsoft-login-alert.com",
                    "http://amazon-gift-card-winner.com",
                    "http://bank-of-america-verify.com",
                ]
                return "\n".join(fallback), fallback[0], fallback[0]

        def load_benign_examples():
            sample_data = Path("data_massive/dataset_batch1_final.parquet")
            if sample_data.exists():
                df = pd.read_parquet(sample_data)
                urls = df[df["label"] == 0]["final_url"].dropna().head(50).tolist()
                first_url = urls[0] if urls else ""
                return "\n".join(urls), first_url, first_url
            else:
                fallback = [
                    "https://www.baidu.com",
                    "https://www.google.com",
                    "https://github.com",
                    "https://www.wikipedia.org",
                    "https://www.taobao.com",
                ]
                return "\n".join(fallback), fallback[0], fallback[0]

        def update_tables_from_data():
            sample_data = Path("data_massive/dataset_batch1_final.parquet")
            if not sample_data.exists():
                return gr.update(), gr.update()

            df = pd.read_parquet(sample_data)

            phishing_data = df[df["label"] == 1][["final_url", "html"]].head(50)
            phishing_rows = []
            for _, row in phishing_data.iterrows():
                url = row["final_url"] or ""
                desc = "钓鱼网站" + (f" - {len(row['html'])}字符" if pd.notna(row['html']) and len(str(row['html'])) > 0 else "")
                phishing_rows.append([url, desc])

            benign_data = df[df["label"] == 0][["final_url", "html"]].head(50)
            benign_rows = []
            for _, row in benign_data.iterrows():
                url = row["final_url"] or ""
                desc = "良性网站" + (f" - {len(row['html'])}字符" if pd.notna(row['html']) and len(str(row['html'])) > 0 else "")
                benign_rows.append([url, desc])

            return gr.update(value=phishing_rows), gr.update(value=benign_rows)

        def on_select_phish(evt: gr.SelectData):
            return evt.value, evt.value

        def on_select_benign(evt: gr.SelectData):
            return evt.value, evt.value

        load_phishing_btn.click(
            fn=load_phishing_examples,
            outputs=[urls_textarea, url_input, test_url_input]
        )
        load_benign_btn.click(
            fn=load_benign_examples,
            outputs=[urls_textarea, url_input, test_url_input]
        )

        phishing_examples.select(
            fn=on_select_phish,
            outputs=[test_url_input, url_input]
        )
        benign_examples.select(
            fn=on_select_benign,
            outputs=[test_url_input, url_input]
        )

        refresh_tables_btn.click(
            fn=update_tables_from_data,
            outputs=[phishing_examples, benign_examples]
        )

        # 添加新功能的连接事件
        quick_validate_btn.click(
            fn=on_quick_validate,
            inputs=[url_input],
            outputs=[url_info_display]
        )

        stats_refresh_btn.click(
            fn=refresh_statistics,
            inputs=[history_state],
            outputs=[stats_display]
        )

        # URL输入变化时自动更新URL信息显示
        url_input.change(
            fn=on_quick_validate,
            inputs=[url_input],
            outputs=[url_info_display],
            show_progress=False
        )

        # 添加导出历史功能连接
        export_history_btn.click(
            fn=export_history,
            inputs=[history_state],
            outputs=[]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch PhishGuard UI")
    parser.add_argument("--host", type=str, default=None, help="自定义监听地址")
    parser.add_argument("--port", type=int, default=None, help="自定义监听端口 (0 表示自动分配)")
    args = parser.parse_args()

    server_name = args.host if args.host is not None else os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    if args.port is not None:
        server_port = None if args.port == 0 else args.port
    else:
        env_port = os.environ.get("GRADIO_SERVER_PORT")
        server_port = None if env_port in {None, "", "0"} else int(env_port)

    share_default = os.environ.get("PHISHGUARD_SHARE", "true").lower() != "false"

    demo = build_interface()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share_default,
        show_api=False,
    )


if __name__ == "__main__":
    main()
