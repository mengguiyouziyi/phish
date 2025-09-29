#!/usr/bin/env python3
"""
直接测试FusionDNN模型，验证其预测能力
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from train_advanced_v3 import extract_enhanced_features, create_dataset
from phishguard_v1.models.fusion_model import AdvancedFusionDNN, predict_proba

def test_model_directly():
    """直接测试模型预测"""
    print("🔍 直接测试FusionDNN模型...")

    # 加载模型
    try:
        ckpt = torch.load("artifacts/fusion_advanced_v3.pt", map_location="cpu", weights_only=False)
        model = AdvancedFusionDNN(num_features=ckpt["input_features"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"✅ 模型加载成功，特征数: {ckpt['input_features']}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 测试钓鱼网站
    phishing_urls = [
        "http://secure-login.apple.com.verify-login.com",
        "http://www.amazon.update.account.secure-login.net",
        "http://paypal.com.secure.transaction.update.com",
        "http://verify-paypal-account.com",
        "http://microsoft-login-alert.com"
    ]

    # 测试良性网站
    benign_urls = [
        "https://www.baidu.com",
        "https://www.google.com",
        "https://github.com"
    ]

    print("\n📊 直接测试结果:")
    print("-" * 80)

    # 获取标准化参数
    scaler_mean = torch.tensor(ckpt.get("scaler_mean", [0.0] * ckpt["input_features"]))
    scaler_scale = torch.tensor(ckpt.get("scaler_scale", [1.0] * ckpt["input_features"]))

    for url in phishing_urls + benign_urls:
        # 提取特征
        features = extract_enhanced_features(url)

        # 标准化
        x_array = np.array(features).reshape(1, -1)
        x_array = (x_array - scaler_mean.numpy()) / scaler_scale.numpy()
        x = torch.tensor(x_array, dtype=torch.float32)

        # 预测
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            benign_prob = probs[0, 0].item()
            phishing_prob = probs[0, 1].item()

        expected = "钓鱼" if url in phishing_urls else "良性"
        predicted = "良性" if benign_prob > 0.5 else "钓鱼"
        correct = "✅" if predicted == expected else "❌"

        print(f"URL: {url}")
        print(f"  期望: {expected}")
        print(f"  预测: {predicted} {correct}")
        print(f"  良性概率: {benign_prob:.4f}")
        print(f"  钓鱼概率: {phishing_prob:.4f}")
        print()

if __name__ == "__main__":
    test_model_directly()