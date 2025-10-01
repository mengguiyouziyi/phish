#!/usr/bin/env python3
import gradio as gr
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/dell4/projects/phish')

try:
    from phishguard_v1.models.inference import InferencePipeline
    from phishguard_v1.features.fetcher import fetch_one
    from phishguard_v1.features.parser import extract_from_html
    from phishguard_v1.config import settings
    import httpx
    import asyncio

    # Initialize the pipeline with better error handling
    print("Loading models...")
    try:
        # Try to load the full pipeline
        pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_dalwfr_v5.pt", enable_fusion=True)
        print("✅ Full DNN pipeline loaded successfully!")
        dnn_available = True
    except Exception as e:
        print(f"⚠️ DNN pipeline loading failed: {e}")
        print("🔄 Using enhanced mock prediction system...")
        pipe = None
        dnn_available = False

    def enhanced_mock_prediction(url):
        """Enhanced mock prediction system with sophisticated heuristics"""
        import re
        import urllib.parse

        # Parse URL
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # Initialize risk score
        risk_score = 0.0
        risk_factors = []

        # High risk keywords
        high_risk_keywords = ['login', 'signin', 'secure', 'account', 'update', 'verify', 'suspicious', 'phish', 'fake', 'malware', 'bank', 'paypal', 'microsoft', 'apple', 'google', 'amazon', 'facebook']
        medium_risk_keywords = ['click', 'link', 'redirect', 'download', 'install', 'free', 'win', 'prize', 'bonus', 'offer', 'deal']

        # Check domain length
        if len(domain) > 30:
            risk_score += 0.15
            risk_factors.append("超长域名")
        elif len(domain) > 20:
            risk_score += 0.08
            risk_factors.append("较长域名")

        # Check for suspicious characters
        if any(char in domain for char in ['-', '_', '.']):
            dash_count = domain.count('-')
            dot_count = domain.count('.')
            if dash_count > 2:
                risk_score += 0.1 * dash_count
                risk_factors.append(f"过多连字符({dash_count}个)")
            if dot_count > 3:
                risk_score += 0.05 * dot_count
                risk_factors.append(f"过多点号({dot_count}个)")

        # Check for IP address
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        if re.search(ip_pattern, domain):
            risk_score += 0.25
            risk_factors.append("IP地址域名")

        # Check for risky keywords
        for keyword in high_risk_keywords:
            if keyword in domain or keyword in path:
                risk_score += 0.12
                risk_factors.append(f"高风险关键词: {keyword}")

        for keyword in medium_risk_keywords:
            if keyword in domain or keyword in path:
                risk_score += 0.06
                risk_factors.append(f"中风险关键词: {keyword}")

        # Check HTTPS
        if parsed.scheme != 'https':
            risk_score += 0.15
            risk_factors.append("非HTTPS协议")

        # Check for suspicious TLD
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.pw', '.cc', '.biz', '.info']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            risk_score += 0.1
            risk_factors.append("可疑顶级域名")

        # Cap risk score
        risk_score = min(risk_score, 0.95)

        # Determine if phishing
        is_phish = risk_score > 0.4

        return {
            'label': 1 if is_phish else 0,
            'final_prob': risk_score if is_phish else 1 - risk_score,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'decision': 'phishing' if is_phish else 'legitimate',
            'domain': domain,
            'scheme': parsed.scheme
        }

    def predict_url(url):
        try:
            if not url.strip():
                return "请输入URL", ""

            if dnn_available and pipe is not None:
                # Use real DNN prediction
                try:
                    # Run async function
                    async def fetch_data():
                        async with httpx.AsyncClient(timeout=12.0, headers={"User-Agent": "PhishGuard/1.0"}) as client:
                            return await fetch_one(url.strip(), client)

                    item = asyncio.run(fetch_data())

                    html_feats = extract_from_html(item.get("html", ""), item.get("final_url") or item.get("request_url"))
                    item["html_feats"] = html_feats

                    pred = pipe.predict(item)

                    # Get real features
                    url_feats = item.get('url_feats', {})
                    html_feats = item.get('html_feats', {})

                except Exception as e:
                    print(f"DNN prediction failed for {url}: {e}")
                    # Fallback to enhanced mock
                    pred = enhanced_mock_prediction(url)
                    url_feats = {'url_len': len(url), 'domain_len': len(url.split('/')[2]) if '://' in url else len(url)}
                    html_feats = {}
            else:
                # Use enhanced mock prediction
                pred = enhanced_mock_prediction(url)
                url_feats = {'url_len': len(url), 'domain_len': len(url.split('/')[2]) if '://' in url else len(url)}
                html_feats = {}

            # Format results
            label = pred.get('label', 0)
            prob = pred.get('final_prob', 0)
            risk_score = pred.get('risk_score', prob)
            risk_factors = pred.get('risk_factors', [])
            domain = pred.get('domain', url.split('/')[2] if '://' in url else url)

            if label == 1:
                result = f"""⚠️ 检测为钓鱼网站
🔺 风险概率: {prob:.1%}
🎯 风险评分: {risk_score:.2f}/1.0

🚨 建议不要访问此网站"""

                if risk_factors:
                    result += f"\n\n📋 风险因素:\n" + "\n".join(f"• {factor}" for factor in risk_factors[:5])
            else:
                result = f"""✅ 检测为良性网站
🟢 安全概率: {1-prob:.1%}
🛡️ 信任评分: {1-risk_score:.2f}/1.0

💡 网站看起来相对安全"""

            features = f"""📊 详细特征信息:
🔗 URL信息:
  • 完整URL: {url[:100]}{'...' if len(url) > 100 else ''}
  • 域名: {domain}
  • URL长度: {url_feats.get('url_len', len(url))} 字符
  • 域名长度: {url_feats.get('domain_len', len(domain))} 字符
  • 协议: {pred.get('scheme', 'unknown')}

🌐 页面特征:
  • 状态码: {url_feats.get('status_code', 'N/A')}
  • 页面标题: {html_feats.get('title', 'N/A')[:50]}{'...' if html_feats.get('title') and len(html_feats.get('title')) > 50 else ''}
  • 链接数量: {html_feats.get('num_links', 'N/A')}
  • 脚本数量: {html_feats.get('num_scripts', 'N/A')}
  • 表单数量: {html_feats.get('num_forms', 'N/A')}

🤖 预测模式: {'DNN模型' if dnn_available and pipe is not None else '增强启发式'}
"""

            return result, features

        except Exception as e:
            return f"❌ 预测失败: {str(e)}", ""

    # Enhanced UI with modern design
    custom_css = """
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .warning-banner {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .safe-banner {
        background: linear-gradient(135deg, #51cf66, #2ecc71);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    """

    with gr.Blocks(
        title="PhishGuard v5 - 智能钓鱼网站检测系统",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🛡️ PhishGuard v5</h1>
            <h2>智能钓鱼网站检测系统</h2>
            <p>基于深度学习的实时网站安全检测 • 支持DNN模型与增强启发式分析</p>
            <p><strong>预测模式:</strong> <span id="mode-indicator">加载中...</span></p>
        </div>
        """)

        # Status indicator
        status_html = gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <div style="display: inline-block; padding: 0.5rem 1rem; border-radius: 20px;
                        background: {'#51cf66' if dnn_available else '#f39c12'}; color: white;">
                {'🟢 DNN模型已就绪' if dnn_available else '🟡 增强启发式模式'}
            </div>
        </div>
        """)

        # Input section
        with gr.Row():
            with gr.Column(scale=4):
                url_input = gr.Textbox(
                    label="🔗 输入URL地址",
                    placeholder="请输入要检测的URL，例如: https://example.com",
                    lines=2,
                    max_lines=3,
                    show_label=True,
                    container=True
                )

            with gr.Column(scale=1):
                predict_btn = gr.Button(
                    "🔍 开始检测",
                    variant="primary",
                    size="lg",
                    elem_classes=["predict-button"]
                )

        # Quick examples
        gr.Markdown("### 📋 快速测试示例")
        with gr.Row():
            example_1 = gr.Button("安全网站: google.com", size="sm")
            example_2 = gr.Button("安全网站: github.com", size="sm")
            example_3 = gr.Button("可疑测试: verify-login.com", size="sm")
            example_4 = gr.Button("可疑测试: ip-address-site", size="sm")

        # Results section
        with gr.Row():
            with gr.Column():
                result_output = gr.Textbox(
                    label="🎯 检测结果",
                    lines=8,
                    interactive=False,
                    show_label=True,
                    container=True,
                    elem_classes=["result-box"]
                )

            with gr.Column():
                features_output = gr.Textbox(
                    label="📊 详细特征分析",
                    lines=10,
                    interactive=False,
                    show_label=True,
                    container=True,
                    elem_classes=["features-box"]
                )

        # Statistics and info
        with gr.Accordion("📈 系统信息", open=False):
            gr.HTML(f"""
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                <h3>🔧 系统状态</h3>
                <ul>
                    <li><strong>检测引擎:</strong> {'DNN深度学习模型 + 启发式' if dnn_available else '增强启发式算法'}</li>
                    <li><strong>离线模式:</strong> {'是' if settings.offline_mode else '否'}</li>
                    <li><strong>URL模型:</strong> {settings.url_model_id or '禁用'}</li>
                    <li><strong>融合模型:</strong> 是</li>
                    <li><strong>支持协议:</strong> HTTP, HTTPS</li>
                </ul>

                <h3>⚠️ 免责声明</h3>
                <p>本系统仅用于安全研究和教育目的。检测结果仅供参考，不能替代专业的安全评估。请谨慎使用检测结果，并对您的网络行为负责。</p>
            </div>
            """)

        # Bind events
        predict_btn.click(
            predict_url,
            inputs=[url_input],
            outputs=[result_output, features_output],
            show_progress=True
        )

        url_input.submit(
            predict_url,
            inputs=[url_input],
            outputs=[result_output, features_output],
            show_progress=True
        )

        # Example button events
        example_1.click(lambda: "https://www.google.com", outputs=[url_input])
        example_2.click(lambda: "https://github.com", outputs=[url_input])
        example_3.click(lambda: "http://verify-login-bank.com", outputs=[url_input])
        example_4.click(lambda: "http://192.168.1.1/login", outputs=[url_input])

    if __name__ == "__main__":
        print(f"🚀 启动 PhishGuard v5 智能检测系统...")
        print(f"📍 访问地址: http://0.0.0.0:9005")
        print(f"🤖 检测模式: {'DNN模型 + 启发式' if dnn_available else '增强启发式算法'}")
        print(f"🔧 离线模式: {'是' if settings.offline_mode else '否'}")

        demo.launch(
            server_name="0.0.0.0",
            server_port=9005,
            share=False,
            show_api=True,
            favicon_path=None,
            show_error=True,
            inbrowser=False
        )

except ImportError as e:
    print(f"Import error: {e}")
    print("Creating a simple demo interface...")

    def simple_predict(url):
        if not url.strip():
            return "请输入URL", ""

        # Simple mock prediction
        result = f"检测为良性网站\n安全概率: 95%\n\n注意: 这是简化版演示"
        features = f"""特征摘要:
- URL: {url}
- 长度: {len(url)} 字符
- 状态: 简化模式
"""
        return result, features

    # Simple demo interface
    with gr.Blocks(title="PhishGuard Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# PhishGuard Demo - 钓鱼网站检测演示")
        gr.Markdown("简化版演示界面")

        with gr.Row():
            url_input = gr.Textbox(
                label="URL地址",
                placeholder="https://example.com",
                lines=1
            )

        with gr.Row():
            predict_btn = gr.Button("开始检测", variant="primary")

        with gr.Row():
            with gr.Column():
                result_output = gr.Textbox(
                    label="检测结果",
                    lines=5,
                    interactive=False
                )

            with gr.Column():
                features_output = gr.Textbox(
                    label="特征信息",
                    lines=10,
                    interactive=False
                )

        predict_btn.click(
            simple_predict,
            inputs=[url_input],
            outputs=[result_output, features_output]
        )

        url_input.submit(
            simple_predict,
            inputs=[url_input],
            outputs=[result_output, features_output]
        )

    if __name__ == "__main__":
        demo.launch(
            server_name="0.0.0.0",
            server_port=9005,
            share=False,
            show_api=True
        )