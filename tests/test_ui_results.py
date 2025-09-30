#!/usr/bin/env python3
"""
测试UI显示结果的一致性
"""

import sys
sys.path.append('.')

from phishguard_v1.models.inference import InferencePipeline

def test_ui_display():
    """测试UI显示结果"""
    print("🔍 测试UI显示结果一致性...")

    # 创建管道
    pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_balanced_v2.pt", enable_fusion=True)

    # 测试URLs
    test_urls = [
        "https://www.baidu.com",
        "https://www.google.com",
        "http://verify-paypal-account.com"
    ]

    for url in test_urls:
        print(f"\n📊 测试URL: {url}")

        # 模拟特征数据
        features = {
            "request_url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html",
            "bytes": 1024,  # 使用固定值
            "url_feats": {
                "url_len": len(url),
                "host_len": len(url.split('//')[-1].split('/')[0]),
                "path_len": 1,
                "num_digits": 0,
                "num_letters": 16,
                "num_specials": 5,
                "num_dots": 2,
                "num_hyphen": 0,
                "num_slash": 2,
                "num_qm": 0,
                "num_at": 0,
                "num_pct": 0,
                "has_ip": False,
                "subdomain_depth": 1,
                "tld_suspicious": 0,
                "has_punycode": 0,
                "scheme_https": 1 if url.startswith('https') else 0,
                "query_len": 0,
                "fragment_len": 0
            },
            "html_feats": {
                "has_html": 1,
                "title": "",
                "title_len": 0,
                "num_meta": 1,
                "num_links": 0,
                "num_stylesheets": 0,
                "num_scripts": 1,
                "num_script_ext": 0,
                "num_script_inline": 1,
                "num_iframes": 0,
                "num_forms": 0,
                "has_password_input": 0,
                "has_email_input": 0,
                "suspicious_js_inline": 1,
                "external_form_actions": 0,
                "num_hidden_inputs": 0,
                "external_links": 0,
                "internal_links": 0,
                "external_images": 0,
                "is_subdomain": 1,
                "has_www": 1,
                "is_common_tld": 1
            }
        }

        # 预测
        result = pipe.predict(features)

        # 格式化显示（模拟UI显示）
        url_prob = result.get("url_prob", 0)
        fusion_prob = result.get("fusion_prob")
        final_prob = result.get("final_prob", 0)
        label = result.get("label", 0)

        def format_probability(prob: float) -> str:
            return f"{prob * 100:.2f}%"

        def get_risk_level(prob: float) -> str:
            if prob >= 0.9:
                return "🔴 高风险"
            elif prob >= 0.7:
                return "🟡 中风险"
            elif prob >= 0.5:
                return "🟠 低风险"
            else:
                return "🟢 安全"

        # 生成结论
        conclusion_parts = []
        if label == 1:
            conclusion_parts.append("✅ **检测结果：良性网站**")
        else:
            conclusion_parts.append("🚨 **检测结果：钓鱼网站**")

        risk_level = get_risk_level(final_prob)
        conclusion_parts.append(f"📊 **风险等级：{risk_level}** ({format_probability(final_prob)})")
        conclusion_parts.append(f"🤖 **模型分析：**")
        conclusion_parts.append(f"   - URL预训练模型: {format_probability(url_prob)}")
        if fusion_prob is not None:
            conclusion_parts.append(f"   - FusionDNN模型: {format_probability(fusion_prob)}")

        print("\n".join(conclusion_parts))

if __name__ == "__main__":
    test_ui_display()