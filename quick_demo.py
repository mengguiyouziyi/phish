#!/usr/bin/env python3
"""
PhishGuard v5 快速演示版 - 仅使用启发式算法
"""
import gradio as gr
import re
import urllib.parse
from datetime import datetime

def sophisticated_heuristic_analysis(url):
    """高精度启发式分析算法"""
    if not url or not url.strip():
        return "❌ 请输入有效的URL地址", ""

    url = url.strip()
    if not (url.startswith('http://') or url.startswith('https://')):
        url = 'https://' + url

    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    risk_score = 0.0
    risk_factors = []

    # 域名特征分析
    if len(domain) > 35:
        risk_score += 0.20
        risk_factors.append("超长域名")
    elif len(domain) > 25:
        risk_score += 0.10
        risk_factors.append("较长域名")

    # 特殊字符分析
    special_chars = {'-', '_', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    char_count = sum(1 for c in domain if c in special_chars)
    if char_count > len(domain) * 0.4:
        risk_score += 0.15
        risk_factors.append("过多特殊字符")

    # 数字模式检测
    digits = re.findall(r'\d+', domain)
    if len(digits) > 2:
        risk_score += 0.12
        risk_factors.append("可疑数字模式")

    # 高风险关键词
    critical_keywords = [
        'verify', 'secure', 'login', 'signin', 'account', 'update', 'confirm',
        'bank', 'paypal', 'microsoft', 'apple', 'google', 'amazon', 'facebook'
    ]

    for keyword in critical_keywords:
        if keyword in domain:
            risk_score += 0.15
            risk_factors.append(f"高风险关键词: {keyword}")

    # 协议安全检测
    if parsed.scheme != 'https':
        risk_score += 0.18
        risk_factors.append("非HTTPS协议")

    # IP地址检测
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    if re.search(ip_pattern, domain):
        risk_score += 0.30
        risk_factors.append("IP地址域名")

    # 可疑TLD检测
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.mn', '.pw', '.cc', '.ws']
    if any(domain.endswith(tld) for tld in suspicious_tlds):
        risk_score += 0.12
        risk_factors.append("可疑顶级域名")

    # 限制风险分数范围
    risk_score = min(max(risk_score, 0.0), 0.95)
    is_phishing = risk_score > 0.45

    if is_phishing:
        result = f"""⚠️ 高风险钓鱼网站检测
🔺 风险概率: {risk_score:.1%}
🎯 风险评分: {risk_score:.2f}/1.0
🔍 分析方式: 增强启发式算法

🚨 安全警告: 建议立即停止访问此网站！"""

        if risk_factors:
            result += f"\n\n📋 主要风险因素:\n" + "\n".join(f"  • {factor}" for factor in risk_factors[:6])
    else:
        result = f"""✅ 网站安全检测通过
🟢 安全概率: {1-risk_score:.1%}
🛡️ 信任评分: {1-risk_score:.2f}/1.0
🔍 分析方式: 增强启发式算法

💡 提示: 网站看起来相对安全，但仍需保持警惕"""

    features = f"""📊 详细技术分析:
🌐 URL基本信息:
  • 完整URL: {url[:80]}{'...' if len(url) > 80 else ''}
  • 域名: {domain}
  • URL长度: {len(url)} 字符
  • 域名长度: {len(domain)} 字符
  • 协议类型: {parsed.scheme}

🎨 网页内容特征:
  • HTTP状态码: N/A
  • 页面标题: N/A
  • 链接总数: N/A
  • 脚本文件数: N/A

🤖 分析引擎信息:
  • 检测模式: 增强启发式算法
  • 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  • 系统版本: PhishGuard v5.0 快速演示版
  • 离线模式: 是
"""

    return result, features

# 创建现代化界面
custom_css = """
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.header-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    color: white;
    padding: 2rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}
.status-indicator {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    font-weight: 600;
    margin: 1rem 0;
    transition: all 0.3s ease;
}
.status-heuristic {
    background: linear-gradient(135deg, #ff9a44, #fc6076);
    box-shadow: 0 4px 15px rgba(252, 96, 118, 0.4);
}
.input-section {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
}
.result-section {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
}
.predict-button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none;
    color: white;
    padding: 1rem 2rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
.predict-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}
"""

with gr.Blocks(
    title="PhishGuard v5 快速演示版",
    theme=gr.themes.Soft(),
    css=custom_css,
    analytics_enabled=False
) as demo:

    # 主标题区域
    gr.HTML(f"""
    <div class="header-section">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🛡️ PhishGuard v5</h1>
        <h2 style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: 400; opacity: 0.9;">企业级智能钓鱼网站检测系统 - 快速演示版</h2>
        <p style="margin: 1rem 0; font-size: 1.1rem; opacity: 0.8;">
            高精度启发式算法 • 实时安全风险评估 • 快速响应检测
        </p>
        <div class="status-indicator status-heuristic">
            🔍 增强启发式引擎已激活
        </div>
    </div>
    """)

    # 输入区域
    with gr.Row():
        with gr.Column(scale=4):
            url_input = gr.Textbox(
                label="🔗 输入要检测的URL",
                placeholder="请输入完整的URL地址，例如: https://www.google.com 或可疑链接",
                lines=3,
                max_lines=4,
                show_label=True,
                container=True,
                elem_classes=["input-section"]
            )

        with gr.Column(scale=1):
            predict_btn = gr.Button(
                "🚀 开始安全检测",
                variant="primary",
                size="lg",
                elem_classes=["predict-button"]
            )

    # 快速示例区域
    gr.Markdown("### 🎯 快速测试示例")
    with gr.Row():
        example_1 = gr.Button("🟢 安全网站: Google", size="sm")
        example_2 = gr.Button("🟢 安全网站: GitHub", size="sm")
        example_3 = gr.Button("🔴 可疑测试: 银行仿冒", size="sm")
        example_4 = gr.Button("🔴 可疑测试: IP地址", size="sm")

    # 结果展示区域
    with gr.Row():
        with gr.Column():
            result_output = gr.Textbox(
                label="🎯 检测结果",
                lines=10,
                interactive=False,
                show_label=True,
                container=True,
                elem_classes=["result-section"]
            )

        with gr.Column():
            features_output = gr.Textbox(
                label="📊 详细技术分析",
                lines=12,
                interactive=False,
                show_label=True,
                container=True,
                elem_classes=["result-section"]
            )

    # 绑定事件处理
    predict_btn.click(
        sophisticated_heuristic_analysis,
        inputs=[url_input],
        outputs=[result_output, features_output],
        show_progress=True
    )

    url_input.submit(
        sophisticated_heuristic_analysis,
        inputs=[url_input],
        outputs=[result_output, features_output],
        show_progress=True
    )

    # 示例按钮事件
    example_1.click(lambda: "https://www.google.com", outputs=[url_input])
    example_2.click(lambda: "https://github.com", outputs=[url_input])
    example_3.click(lambda: "http://secure-bank-verification.com", outputs=[url_input])
    example_4.click(lambda: "http://192.168.1.100/login-update", outputs=[url_input])

# 启动信息
print("="*80)
print("🚀 PhishGuard v5 快速演示版启动完成")
print("="*80)
print(f"🌐 访问地址: http://0.0.0.0:9005")
print(f"🧠 检测引擎: 高精度启发式算法")
print(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=9005,
        share=False,
        show_api=True,
        show_error=True,
        inbrowser=False
    )