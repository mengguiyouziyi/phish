#!/usr/bin/env python3
"""
PhishGuard v5 最终版本 - 智能钓鱼网站检测系统
整合了DNN模型和增强启发式算法的高性能检测引擎
"""

import gradio as gr
import sys
import os
import re
import urllib.parse
import httpx
import asyncio
from datetime import datetime
import json

# Add the project root to Python path
sys.path.insert(0, '/home/dell4/projects/phish')

try:
    from phishguard_v1.models.inference import InferencePipeline
    from phishguard_v1.features.fetcher import fetch_one
    from phishguard_v1.features.parser import extract_from_html
    from phishguard_v1.config import settings

    class EnhancedPhishGuard:
        def __init__(self):
            self.dnn_available = False
            self.pipeline = None
            self.load_models()

        def load_models(self):
            """智能模型加载系统"""
            print("🔄 正在初始化PhishGuard v5检测引擎...")
            try:
                self.pipeline = InferencePipeline(
                    fusion_ckpt_path="artifacts/fusion_dalwfr_v5.pt",
                    enable_fusion=True
                )
                self.dnn_available = True
                print("✅ DNN模型加载成功 - 深度学习引擎已就绪")
            except Exception as e:
                print(f"⚠️ DNN模型加载失败: {str(e)}")
                print("🔄 启用增强启发式检测引擎...")
                self.pipeline = None
                self.dnn_available = False
                print("✅ 增强启发式引擎已就绪")

        def sophisticated_heuristic_analysis(self, url):
            """高精度启发式分析算法"""
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

            # 子域名深度
            subdomain_count = domain.split('.')
            if len(subdomain_count) > 4:
                risk_score += 0.08
                risk_factors.append("过深子域名")

            # 高风险关键词（扩展版）
            critical_keywords = [
                'verify', 'secure', 'login', 'signin', 'account', 'update', 'confirm',
                'bank', 'paypal', 'microsoft', 'apple', 'google', 'amazon', 'facebook',
                'instagram', 'twitter', 'linkedin', 'telegram', 'whatsapp', 'gmail',
                'yahoo', 'hotmail', 'outlook', 'office', 'adobe', 'netflix', 'spotify'
            ]

            medium_keywords = [
                'click', 'link', 'redirect', 'download', 'install', 'free', 'win',
                'prize', 'bonus', 'offer', 'deal', 'discount', 'sale', 'promo'
            ]

            # 关键词检测（上下文感知）
            for keyword in critical_keywords:
                if keyword in domain:
                    context_score = 0.15
                    # 检查是否在可疑上下文中
                    suspicious_contexts = ['verify-', '-secure', 'login-', '-account', 'confirm-']
                    for ctx in suspicious_contexts:
                        if ctx in domain or ctx.replace('-', '') in domain:
                            context_score += 0.10
                            break
                    risk_score += context_score
                    risk_factors.append(f"高风险关键词: {keyword}")

            for keyword in medium_keywords:
                if keyword in domain or keyword in path:
                    risk_score += 0.08
                    risk_factors.append(f"中风险关键词: {keyword}")

            # 协议安全检测
            if parsed.scheme != 'https':
                risk_score += 0.18
                risk_factors.append("非HTTPS协议")

            # 端口检测
            if parsed.port and parsed.port not in [80, 443]:
                risk_score += 0.25
                risk_factors.append(f"非标准端口: {parsed.port}")

            # IP地址检测
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            if re.search(ip_pattern, domain):
                risk_score += 0.30
                risk_factors.append("IP地址域名")

            # 可疑TLD检测
            suspicious_tlds = [
                '.tk', '.ml', '.ga', '.cf', '.gq', '.mn', '.pw', '.cc', '.ws', '.biz',
                '.info', '.work', '.click', '.download', '.racing', '.science', '.top'
            ]
            if any(domain.endswith(tld) for tld in suspicious_tlds):
                risk_score += 0.12
                risk_factors.append("可疑顶级域名")

            # URL相似性检测
            legitimate_domains = ['google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
                                 'apple.com', 'netflix.com', 'instagram.com', 'twitter.com']
            for legit in legitimate_domains:
                if legit in domain and domain != legit:
                    risk_score += 0.20
                    risk_factors.append(f"仿冒域名: 类似{legit}")
                    break

            # 限制风险分数范围
            risk_score = min(max(risk_score, 0.0), 0.95)

            # 判断是否为钓鱼网站
            is_phishing = risk_score > 0.45

            return {
                'label': 1 if is_phishing else 0,
                'final_prob': risk_score if is_phishing else 1 - risk_score,
                'risk_score': risk_score,
                'risk_factors': sorted(risk_factors, key=len, reverse=True),
                'decision': 'phishing' if is_phishing else 'legitimate',
                'domain': domain,
                'scheme': parsed.scheme,
                'analysis_type': 'enhanced_heuristic'
            }

        async def comprehensive_analysis(self, url):
            """综合分析系统"""
            if self.dnn_available and self.pipeline:
                try:
                    # 使用真实的网络数据进行分析
                    async with httpx.AsyncClient(
                        timeout=15.0,
                        headers={"User-Agent": "PhishGuard/5.0 (Security Research Bot)"}
                    ) as client:
                        item = await fetch_one(url.strip(), client)

                    # 提取HTML特征
                    html_feats = extract_from_html(
                        item.get("html", ""),
                        item.get("final_url") or item.get("request_url")
                    )
                    item["html_feats"] = html_feats

                    # DNN模型预测
                    pred = self.pipeline.predict(item)

                    # 获取真实特征
                    url_feats = item.get('url_feats', {})
                    html_feats = item.get('html_feats', {})

                    return pred, url_feats, html_feats, 'dnn_model'

                except Exception as e:
                    print(f"DNN分析失败，回退到启发式: {e}")
                    # 回退到启发式分析
                    heuristic_pred = self.sophisticated_heuristic_analysis(url)
                    return heuristic_pred, {}, {}, 'enhanced_heuristic'
            else:
                # 使用增强启发式分析
                heuristic_pred = self.sophisticated_heuristic_analysis(url)
                return heuristic_pred, {}, {}, 'enhanced_heuristic'

        def format_results(self, url, pred, url_feats, html_feats, analysis_type):
            """格式化结果输出"""
            label = pred.get('label', 0)
            prob = pred.get('final_prob', 0)
            risk_score = pred.get('risk_score', prob)
            risk_factors = pred.get('risk_factors', [])
            domain = pred.get('domain', urllib.parse.urlparse(url).netloc)

            if label == 1:
                result = f"""⚠️ 高风险钓鱼网站检测
🔺 风险概率: {prob:.1%}
🎯 风险评分: {risk_score:.2f}/1.0
🔍 分析方式: {'DNN深度学习' if analysis_type == 'dnn_model' else '增强启发式算法'}

🚨 安全警告: 建议立即停止访问此网站！"""

                if risk_factors:
                    result += f"\n\n📋 主要风险因素:\n" + "\n".join(f"  • {factor}" for factor in risk_factors[:6])
            else:
                result = f"""✅ 网站安全检测通过
🟢 安全概率: {1-prob:.1%}
🛡️ 信任评分: {1-risk_score:.2f}/1.0
🔍 分析方式: {'DNN深度学习' if analysis_type == 'dnn_model' else '增强启发式算法'}

💡 提示: 网站看起来相对安全，但仍需保持警惕"""

            features = f"""📊 详细技术分析:
🌐 URL基本信息:
  • 完整URL: {url[:80]}{'...' if len(url) > 80 else ''}
  • 域名: {domain}
  • URL长度: {url_feats.get('url_len', len(url))} 字符
  • 域名长度: {url_feats.get('domain_len', len(domain))} 字符
  • 协议类型: {pred.get('scheme', 'unknown')}

🎨 网页内容特征:
  • HTTP状态码: {url_feats.get('status_code', 'N/A')}
  • 页面标题: {html_feats.get('title', 'N/A')[:40]}{'...' if html_feats.get('title') and len(html_feats.get('title')) > 40 else ''}
  • 链接总数: {html_feats.get('num_links', 'N/A')}
  • 脚本文件数: {html_feats.get('num_scripts', 'N/A')}
  • 表单数量: {html_feats.get('num_forms', 'N/A')}
  • 外部资源: {html_feats.get('external_resources', 'N/A')}

🤖 分析引擎信息:
  • 检测模式: {'DNN + 启发式融合' if analysis_type == 'dnn_model' else '高精度启发式算法'}
  • 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  • 系统版本: PhishGuard v5.0
  • 离线模式: {'是' if settings.offline_mode else '否'}
"""

            return result, features

    # 初始化检测系统
    detector = EnhancedPhishGuard()

    async def predict_url(url):
        """主要预测函数"""
        try:
            if not url or not url.strip():
                return "❌ 请输入有效的URL地址", ""

            url = url.strip()
            if not (url.startswith('http://') or url.startswith('https://')):
                url = 'https://' + url

            print(f"🔍 开始分析: {url}")

            # 执行综合分析
            pred, url_feats, html_feats, analysis_type = await detector.comprehensive_analysis(url)

            # 格式化结果
            result, features = detector.format_results(url, pred, url_feats, html_feats, analysis_type)

            print(f"✅ 分析完成: {pred.get('decision', 'unknown')} (置信度: {pred.get('final_prob', 0):.2f})")

            return result, features

        except Exception as e:
            error_msg = f"❌ 分析过程中发生错误: {str(e)}"
            print(f"❌ 预测失败: {e}")
            return error_msg, f"错误详情: {str(e)}\n\n请检查URL格式是否正确，或稍后重试。"

    # 创建现代化界面
    custom_css = """
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    .header-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    .status-indicator {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .status-dnn {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.4);
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
    .example-button {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        border: none;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    .example-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.4);
    }
    """

    with gr.Blocks(
        title="PhishGuard v5 - 企业级钓鱼网站检测系统",
        theme=gr.themes.Soft(),
        css=custom_css,
        analytics_enabled=False
    ) as demo:

        # 主标题区域
        gr.HTML(f"""
        <div class="header-section">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🛡️ PhishGuard v5</h1>
            <h2 style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: 400; opacity: 0.9;">企业级智能钓鱼网站检测系统</h2>
            <p style="margin: 1rem 0; font-size: 1.1rem; opacity: 0.8;">
                融合深度学习与增强启发式算法 • 实时安全风险评估 • 企业级防护能力
            </p>
            <div class="status-indicator {'status-dnn' if detector.dnn_available else 'status-heuristic'}">
                {'🧠 DNN深度学习引擎已激活' if detector.dnn_available else '🔍 增强启发式引擎已激活'}
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
            example_1 = gr.Button("🟢 安全网站: Google", size="sm", elem_classes=["example-button"])
            example_2 = gr.Button("🟢 安全网站: GitHub", size="sm", elem_classes=["example-button"])
            example_3 = gr.Button("🔴 可疑测试: 银行仿冒", size="sm", elem_classes=["example-button"])
            example_4 = gr.Button("🔴 可疑测试: IP地址", size="sm", elem_classes=["example-button"])

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

        # 系统信息区域
        with gr.Accordion("📋 系统信息 & 技术规格", open=False):
            gr.HTML(f"""
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px;">
                <h3 style="color: #495057; margin-top: 0;">🔧 系统技术规格</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
                    <div>
                        <h4 style="color: #6c757d;">🧠 检测引擎</h4>
                        <ul>
                            <li><strong>主要引擎:</strong> {'DNN深度学习模型' if detector.dnn_available else '增强启发式算法'}</li>
                            <li><strong>备用引擎:</strong> 高精度启发式分析</li>
                            <li><strong>融合模型:</strong> BERT + Fusion DNN</li>
                            <li><strong>分析维度:</strong> URL特征 + 内容分析 + 行为模式</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #6c757d;">⚙️ 配置信息</h4>
                        <ul>
                            <li><strong>离线模式:</strong> {'是' if settings.offline_mode else '否'}</li>
                            <li><strong>URL模型:</strong> {settings.url_model_id}</li>
                            <li><strong>融合阈值:</strong> {settings.fusion_phish_threshold}</li>
                            <li><strong>并发处理:</strong> {settings.concurrency} 线程</li>
                        </ul>
                    </div>
                </div>

                <h3 style="color: #495057;">🛡️ 安全特性</h3>
                <ul>
                    <li>✅ 实时URL特征分析</li>
                    <li>✅ 网页内容深度检测</li>
                    <li>✅ 机器学习模型预测</li>
                    <li>✅ 多维度风险评估</li>
                    <li>✅ 智能降级机制</li>
                </ul>

                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; margin-top: 1rem; border-radius: 5px;">
                    <h4 style="color: #856404; margin-top: 0;">⚠️ 免责声明</h4>
                    <p style="margin-bottom: 0; color: #856404;">
                        本系统仅供安全研究和教育目的使用。检测结果仅供参考，不能替代专业的安全评估服务。
                        用户应当谨慎使用检测结果，并对自己的网络行为承担相应责任。
                    </p>
                </div>
            </div>
            """)

        # 绑定事件处理
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

        # 示例按钮事件
        example_1.click(lambda: "https://www.google.com", outputs=[url_input])
        example_2.click(lambda: "https://github.com", outputs=[url_input])
        example_3.click(lambda: "http://secure-bank-verification.com", outputs=[url_input])
        example_4.click(lambda: "http://192.168.1.100/login-update", outputs=[url_input])

    # 启动信息
    print("="*80)
    print("🚀 PhishGuard v5 企业级检测系统启动完成")
    print("="*80)
    print(f"🌐 访问地址: http://0.0.0.0:9005")
    print(f"🧠 检测引擎: {'DNN深度学习 + 启发式融合' if detector.dnn_available else '高精度启发式算法'}")
    print(f"🔧 离线模式: {'启用' if settings.offline_mode else '禁用'}")
    print(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    if __name__ == "__main__":
        demo.launch(
            server_name="0.0.0.0",
            server_port=9005,
            share=False,
            show_api=True,
            show_error=True,
            inbrowser=False,
            favicon_path=None
        )

except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("正在创建基础演示版本...")

    def basic_prediction(url):
        if not url or not url.strip():
            return "❌ 请输入URL", ""

        # 简单的本地检测逻辑
        risk_indicators = ['login', 'verify', 'secure', 'bank', 'account', 'update']
        risk_score = sum(1 for indicator in risk_indicators if indicator in url.lower())

        if risk_score >= 2:
            return "⚠️ 可疑网站\n建议谨慎访问", f"URL: {url}\n风险指标: {risk_score}"
        else:
            return "✅ 相对安全\n仍需保持警惕", f"URL: {url}\n风险指标: {risk_score}"

    # 基础界面
    with gr.Blocks(title="PhishGuard Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# PhishGuard v5 - 演示版本")
        gr.Markdown("⚠️ 基础演示模式，功能有限")

        with gr.Row():
            url_input = gr.Textbox(label="输入URL", placeholder="https://example.com")

        with gr.Row():
            predict_btn = gr.Button("检测", variant="primary")

        with gr.Row():
            with gr.Column():
                result_output = gr.Textbox(label="结果", lines=5, interactive=False)
            with gr.Column():
                features_output = gr.Textbox(label="详情", lines=5, interactive=False)

        predict_btn.click(basic_prediction, inputs=[url_input], outputs=[result_output, features_output])
        url_input.submit(basic_prediction, inputs=[url_input], outputs=[result_output, features_output])

    if __name__ == "__main__":
        demo.launch(server_name="0.0.0.0", server_port=9005, share=False)