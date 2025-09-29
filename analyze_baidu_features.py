#!/usr/bin/env python3
"""
分析百度URL特征，找出误分类原因
"""

import sys
sys.path.append('.')

from train_advanced_v3 import extract_enhanced_features
import torch
import numpy as np

def analyze_baidu_features():
    """分析百度URL特征"""
    print("🔍 分析百度URL特征...")

    # 测试URLs
    test_urls = [
        "https://www.baidu.com",  # 基础URL - 应该正确分类
        "https://www.baidu.com/index.php",  # 带路径 - 被误分类
        "https://www.baidu.com/s?wd=test",  # 带查询参数 - 被误分类
        "https://github.com/user/repo",  # GitHub路径 - 被误分类
    ]

    for url in test_urls:
        print(f"\n📊 URL: {url}")
        features = extract_enhanced_features(url)

        # 关键特征分析
        print(f"  URL长度: {features[0]}")
        print(f"  主机长度: {features[1]}")
        print(f"  路径长度: {features[2]}")
        print(f"  特殊字符数: {features[5]}")
        print(f"  路径深度: {features[22]}")
        print(f"  参数数量: {features[23]}")
        print(f"  子域名深度: {features[33]}")
        print(f"  是否HTTPS: {features[27]}")
        print(f"  是否有参数: {features[28]}")
        print(f"  是否有文件扩展名: {features[29]}")

        # 检查是否有可疑特征
        suspicious_indicators = []
        if features[2] > 5:  # 路径长度
            suspicious_indicators.append(f"路径过长({features[2]})")
        if features[23] > 0:  # 参数数量
            suspicious_indicators.append(f"有参数({features[23]})")
        if features[29] == 1:  # 文件扩展名
            suspicious_indicators.append("有文件扩展名")

        if suspicious_indicators:
            print(f"  ⚠️ 可疑指标: {', '.join(suspicious_indicators)}")
        else:
            print(f"  ✅ 无明显可疑指标")

    # 加载模型看看这些URL的实际预测
    print(f"\n🧠 模型预测分析:")
    try:
        ckpt = torch.load("artifacts/fusion_advanced_v3.pt", map_location="cpu", weights_only=False)
        from phishguard_v1.models.fusion_model import AdvancedFusionDNN

        model = AdvancedFusionDNN(num_features=ckpt["input_features"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        scaler_mean = torch.tensor(ckpt.get("scaler_mean", [0.0] * ckpt["input_features"]))
        scaler_scale = torch.tensor(ckpt.get("scaler_scale", [1.0] * ckpt["input_features"]))

        for url in test_urls:
            features = extract_enhanced_features(url)
            x_array = np.array(features).reshape(1, -1)
            x_array = (x_array - scaler_mean.numpy()) / scaler_scale.numpy()
            x = torch.tensor(x_array, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                benign_prob = probs[0, 0].item()
                phishing_prob = probs[0, 1].item()

            print(f"  {url}")
            print(f"    良性概率: {benign_prob:.4f}")
            print(f"    钓鱼概率: {phishing_prob:.4f}")
            print(f"    预测: {'良性' if benign_prob > 0.5 else '钓鱼'}")

    except Exception as e:
        print(f"  ❌ 模型预测失败: {e}")

if __name__ == "__main__":
    analyze_baidu_features()