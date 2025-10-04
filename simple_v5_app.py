#!/usr/bin/env python3
"""
简化的基于 DALWFR-Fusion v5 模型的钓鱼网站检测服务
部署在 9006 端口
"""

import sys
import os
sys.path.append('/home/dell4/projects/phishing-detector')

import gradio as gr
import json
import torch
import numpy as np
import pandas as pd
import re
import urllib.parse
from phishguard_v1.models.inference import InferencePipeline

# 加载模型
print("正在加载 DALWFR-Fusion v5 模型...")
try:
    pipe = InferencePipeline(
        fusion_ckpt_path="/home/dell4/projects/phishing-detector/phishguard_v1/artifacts/fusion_dalwfr_v5.pt",
        enable_fusion=True
    )
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("将使用基于规则的备用检测方法")
    pipe = None

def create_simple_features(url):
    """创建简单的特征用于备用检测"""
    features = {}

    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc

        # URL长度特征
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)

        # 特殊字符数量
        features['special_chars'] = len(re.findall(r'[@%_\-+=]', domain))

        # 数字数量
        features['digit_count'] = len(re.findall(r'\d', domain))

        # 子域名数量
        features['subdomain_count'] = domain.count('.') - 1 if '.' in domain else 0

        # 是否HTTPS
        features['is_https'] = 1 if url.startswith('https://') else 0

        # 是否包含IP地址
        features['has_ip'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0

        # 可疑TLD
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.click', '.download']
        features['has_suspicious_tld'] = 1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0

        # 钓鱼关键词
        phishing_keywords = ['login', 'secure', 'verify', 'account', 'password', 'signin', 'banking']
        features['phishing_keywords'] = sum(1 for keyword in phishing_keywords if keyword in url.lower())

    except Exception as e:
        print(f"特征提取错误: {e}")
        # 返回默认特征
        features = {
            'url_length': 50, 'domain_length': 20, 'special_chars': 0, 'digit_count': 0,
            'subdomain_count': 1, 'is_https': 1, 'has_ip': 0, 'has_suspicious_tld': 0, 'phishing_keywords': 0
        }

    return features

def rule_based_prediction(features):
    """基于规则的预测方法"""
    score = 0.1  # 基础分数

    # URL长度风险
    if features['url_length'] > 100:
        score += 0.2
    elif features['url_length'] > 50:
        score += 0.1

    # HTTPS检查
    if features['is_https'] == 0:
        score += 0.2

    # 可疑TLD
    if features['has_suspicious_tld'] == 1:
        score += 0.3

    # IP地址检查
    if features['has_ip'] == 1:
        score += 0.4

    # 特殊字符检查
    if features['special_chars'] > 0:
        score += 0.2

    # 钓鱼关键词检查
    score += min(features['phishing_keywords'] * 0.1, 0.3)

    return max(0.0, min(1.0, score))

def predict_url(url):
    """预测URL是否为钓鱼网站"""
    try:
        if not url or not url.strip():
            return "请输入有效的URL", 0.0, {}

        url = url.strip()

        if pipe is not None:
            # 使用v5模型进行预测
            try:
                result = pipe.predict(url)

                if isinstance(result, dict):
                    probability = result.get('phishing_prob', 0.0)
                    is_phishing = result.get('is_phishing', False)
                    features = result.get('features', {})

                    if is_phishing:
                        status = f"🚨 钓鱼网站 (概率: {probability:.2%}) - v5模型检测"
                    else:
                        status = f"✅ 安全网站 (概率: {probability:.2%}) - v5模型检测"

                    return status, probability, features
                else:
                    # 如果返回的是简单概率值
                    probability = float(result) if isinstance(result, (int, float)) else 0.5
                    is_phishing = probability > 0.5
                    status = f"🚨 钓鱼网站 (概率: {probability:.2%}) - v5模型检测" if is_phishing else f"✅ 安全网站 (概率: {probability:.2%}) - v5模型检测"
                    return status, probability, {}

            except Exception as model_error:
                print(f"v5模型预测失败，使用备用方法: {model_error}")
                # 降级到规则方法

        # 备用规则检测
        features = create_simple_features(url)
        probability = rule_based_prediction(features)
        is_phishing = probability > 0.5

        if is_phishing:
            status = f"⚠️ 钓鱼网站 (概率: {probability:.2%}) - 规则检测"
        else:
            status = f"✅ 安全网站 (概率: {probability:.2%}) - 规则检测"

        return status, probability, features

    except Exception as e:
        error_msg = f"预测失败: {str(e)}"
        print(f"Error: {error_msg}")
        return error_msg, 0.0, {}

# 创建Gradio界面
with gr.Blocks(title="钓鱼网站检测系统 v5", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎣 钓鱼网站检测系统 v5")
    gr.Markdown("基于 DALWFR-Fusion v5 模型的高精度钓鱼网站检测")

    if pipe is None:
        gr.Markdown("⚠️ **注意**: v5模型加载失败，当前使用规则检测作为备用方案")

    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(
                label="输入URL",
                placeholder="请输入要检测的URL，例如: https://example.com",
                lines=2
            )
            predict_btn = gr.Button("检测", variant="primary")

        with gr.Column():
            status_output = gr.Textbox(label="检测结果", interactive=False)
            prob_output = gr.Number(label="钓鱼概率", minimum=0.0, maximum=1.0, interactive=False)

    with gr.Row():
        features_output = gr.JSON(label="详细特征", visible=False)

    # 绑定事件
    predict_btn.click(
        predict_url,
        inputs=[url_input],
        outputs=[status_output, prob_output, features_output]
    )

    # 示例
    gr.Examples(
        examples=[
            ["https://www.google.com"],
            ["https://github.com"],
            ["http://paypal.com.secure-update.com/login"],
            ["https://microsoft.com"],
            ["https://example.tk/login"],
        ],
        inputs=[url_input]
    )

if __name__ == "__main__":
    print("启动钓鱼网站检测系统 v5...")
    print(f"模型状态: {'✅ v5模型已加载' if pipe else '⚠️ 使用规则检测'}")
    print("端口: 9005")

    demo.launch(
        server_name="0.0.0.0",
        server_port=9005,
        share=False,
        show_error=True
    )