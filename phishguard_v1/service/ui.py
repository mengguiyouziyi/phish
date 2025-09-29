from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import gradio as gr
from httpx import AsyncClient

from ..config import settings
from ..features.fetcher import fetch_one
from ..features.parser import extract_from_html
from ..features.render import render_screenshot
from ..models.inference import InferencePipeline

pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_advanced_v3.pt", enable_fusion=True)


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
    csv_path = f"/tmp/phish_batch_{uuid.uuid4().hex}.csv"
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
    return rows, csv_path, stats


async def scan_single(url: str, screenshot: bool) -> Dict[str, Any]:
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
    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]
    if not urls:
        return []
    tasks = [scan_single(url, screenshot) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)


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
        conclusion = f"❌ 检测失败：{result}"
        status_html = (
            "<div style=\"text-align:center;padding:20px;background-color:#f4433620;border-radius:8px;\">"
            "<div style=\"font-size:24px;margin-bottom:8px;\">检测失败</div>"
            "<div style=\"font-size:16px;\">请稍后重试或检查网络</div>"
            "</div>"
        )
        prob_summary = "### 概率拆解\n- 检测失败，暂无概率信息。"
        detail_summary = "### 推理细节\n- 检测失败，暂无推理细节。"
        features_text = "### 特征摘要\n- 暂无特征信息。"
        return (
            conclusion,
            status_html,
            prob_summary,
            detail_summary,
            {},
            {},
            features_text,
            gr.update(value=None, visible=False),
            "",
            "",
            "",
            gr.update(value=history_rows),
            history,
        )

    pred = result.get("prediction", {}) or {}
    features = result.get("features", {}) or {}

    conclusion = generate_conclusion(pred)
    final_prob = pred.get("final_prob", 0.0)
    risk_level, color = get_risk_level(final_prob)
    status_html = (
        "<div style=\"text-align:center;padding:20px;border-radius:8px;\" "
        f"style=\"background-color:{color}20\">"
        f"<div style=\"font-size:24px;margin-bottom:8px;\">{risk_level}</div>"
        f"<div style=\"font-size:20px;font-weight:bold;\">{format_probability(final_prob)}</div>"
        f"<div style=\"margin-top:8px;\">{'🚨 钓鱼网站' if pred.get('label', 0) == 1 else '✅ 良性网站'}</div>"
        "</div>"
    )

    prob_summary = build_probability_summary(pred)
    detail_summary = build_detail_summary(pred.get("details"))
    features_text = create_feature_popup(features)

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
        screenshot_update,
        final_url,
        status_code_val,
        content_type_val,
        gr.update(value=history_rows),
        history,
    )


def build_interface():
    history_state = gr.State([])

    with gr.Blocks(
        title="PhishGuard v1",
        theme=gr.themes.Soft(),
        css="""
        .main-container {max-width: 1180px; margin: auto;}
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🛡️ PhishGuard v1 - 高级钓鱼网站检测系统

            - 🤖 **URL 预训练模型**：即时语义理解
            - 🧠 **FusionDNN 模型**：多特征深度融合
            - 📊 **全链路诊断**：概率拆解 + 决策细节 + 历史追踪
            """
        )

        with gr.Tabs():
            with gr.TabItem("🔍 单 URL 检测"):
                with gr.Row():
                    with gr.Column(scale=3):
                        url_input = gr.Textbox(
                            label="输入要检测的 URL",
                            placeholder="https://example.com",
                            value="https://example.com",
                        )
                    with gr.Column(scale=1):
                        screenshot_cb = gr.Checkbox(label="启用截图功能", value=False)
                        scan_btn = gr.Button("🔍 开始检测", variant="primary")

                with gr.Row():
                    with gr.Column(scale=2):
                        conclusion_box = gr.Markdown(value="请输入 URL 并点击检测")
                    with gr.Column(scale=1):
                        status_indicator = gr.HTML("<div style='text-align:center;padding:20px;'>⏳ 等待检测...</div>")

                with gr.Row():
                    probability_summary = gr.Markdown("### 概率拆解\n- 等待检测")
                    detail_summary = gr.Markdown("### 推理细节\n- 等待检测")

                with gr.Accordion("📊 详细分析结果", open=False):
                    with gr.Row():
                        pred_json = gr.JSON(label="预测数据", value={})
                        details_json = gr.JSON(label="推理细节", value={})
                    features_markdown = gr.Markdown("### 特征摘要\n- 暂无特征信息。")
                    screenshot_image = gr.Image(label="页面截图", visible=False)

                with gr.Row():
                    final_url = gr.Textbox(label="最终 URL", interactive=False)
                    status_code = gr.Textbox(label="状态码", interactive=False)
                    content_type = gr.Textbox(label="内容类型", interactive=False)

                with gr.Accordion("🗂 历史记录", open=False):
                    with gr.Row():
                        history_table = gr.Dataframe(
                            headers=["时间", "URL", "综合概率", "结论"],
                            datatype=["str", "str", "str", "str"],
                            value=[],
                            interactive=False,
                        )
                        clear_history_btn = gr.Button("🧹 清空记录", variant="secondary")

            with gr.TabItem("📋 批量检测"):
                gr.Markdown("### 批量检测多个 URL（每行一个）")
                urls_textarea = gr.TextArea(
                    label="输入 URL 列表",
                    placeholder="https://example.com\nhttps://google.com",
                    lines=8,
                )
                with gr.Row():
                    batch_screenshot_cb = gr.Checkbox(label="启用截图", value=False)
                    batch_scan_btn = gr.Button("🚀 开始批量检测", variant="primary")

                with gr.Accordion("📈 批量检测结果", open=True):
                    batch_status = gr.Markdown("等待批量检测...")
                    results_table = gr.Dataframe(
                        headers=["URL", "检测结果", "风险等级", "URL 模型", "FusionDNN", "综合概率", "备注"],
                        datatype=["str", "str", "str", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                    )
                    results_file = gr.File(label="下载结果", visible=False)

            with gr.TabItem("🧪 测试样例"):
                gr.Markdown("### 选择预设样例快速评估")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🚨 钓鱼网站样例")
                        phishing_examples = gr.DataFrame(
                            value=[
                                ["http://verify-paypal-account.com", "假冒 PayPal 验证网站"],
                                ["http://apple-security-update.info", "假冒 Apple 安全更新"],
                                ["http://microsoft-login-alert.com", "假冒 Microsoft 登录警告"],
                                ["http://amazon-gift-card-winner.com", "假冒 Amazon 中奖通知"],
                                ["http://bank-of-america-verify.com", "假冒银行验证网站"],
                            ],
                            headers=["URL", "描述"],
                            datatype=["str", "str"],
                            interactive=False,
                            height=200,
                        )
                        load_phishing_btn = gr.Button("🚨 加载钓鱼网站样例", variant="stop")
                    with gr.Column():
                        gr.Markdown("#### ✅ 良性网站样例")
                        benign_examples = gr.DataFrame(
                            value=[
                                ["https://www.baidu.com", "百度搜索"],
                                ["https://www.google.com", "Google 搜索"],
                                ["https://github.com", "GitHub 代码托管"],
                                ["https://www.wikipedia.org", "维基百科"],
                                ["https://www.taobao.com", "淘宝网"],
                            ],
                            headers=["URL", "描述"],
                            datatype=["str", "str"],
                            interactive=False,
                            height=200,
                        )
                        load_benign_btn = gr.Button("✅ 加载良性网站样例", variant="primary")

                gr.Markdown("### 🎯 快速检测")
                with gr.Row():
                    test_url_input = gr.Textbox(
                        label="选中后自动填入（亦可手工输入）",
                        value="https://www.baidu.com",
                    )
                    test_screenshot_cb = gr.Checkbox(label="启用截图功能", value=False)
                    test_scan_btn = gr.Button("🔍 开始检测", variant="primary")

                with gr.Row():
                    test_conclusion = gr.Markdown("请选择 URL 并点击检测")
                    test_status = gr.HTML("<div style='text-align:center;padding:20px;'>⏳ 等待检测...</div>")

            with gr.TabItem("ℹ️ 系统信息"):
                gr.Markdown(
                    """
                    ### 🤖 模型信息
                    - **URL 预训练模型**：`imanoop7/bert-phishing-detector`
                    - **FusionDNN 模型**：42 特征增强版本

                    ### ⚙️ 推理配置
                    - URL 阈值：`settings.url_phish_threshold`
                    - Fusion 阈值：`settings.fusion_phish_threshold`
                    - 最终阈值：`settings.final_phish_threshold`

                    ### 📈 推荐流程
                    1. 输入或批量粘贴 URL
                    2. 查看概率拆解 & 推理细节
                    3. 下载批量 CSV 复核/留档
                    """
                )

        def on_scan_click(url: str, screenshot: bool, history: List[Dict[str, Any]]):
            try:
                result = asyncio.run(scan_single(url, screenshot))
                return update_single_result(result, history)
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
                screenshot_image,
                final_url,
                status_code,
                content_type,
                history_table,
                history_state,
            ],
        )

        def on_test_scan(url: str, screenshot: bool, history: List[Dict[str, Any]]):
            try:
                result = asyncio.run(scan_single(url, screenshot))
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

        def on_batch_scan(urls: str, screenshot: bool):
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
            examples = [
                "http://verify-paypal-account.com",
                "http://apple-security-update.info",
                "http://microsoft-login-alert.com",
                "http://amazon-gift-card-winner.com",
                "http://bank-of-america-verify.com",
            ]
            return "\n".join(examples)

        def load_benign_examples():
            examples = [
                "https://www.baidu.com",
                "https://www.google.com",
                "https://github.com",
                "https://www.wikipedia.org",
                "https://www.taobao.com",
            ]
            return "\n".join(examples)

        load_phishing_btn.click(fn=load_phishing_examples, outputs=[urls_textarea])
        load_benign_btn.click(fn=load_benign_examples, outputs=[urls_textarea])

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
