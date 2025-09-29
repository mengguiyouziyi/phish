#!/usr/bin/env python3
"""
检查标准化器参数问题
"""

import torch
import numpy as np
import sys
sys.path.append('.')

def check_model_scaler():
    """检查模型的标准化器参数"""
    print("🔍 检查模型标准化器参数...")

    # 加载模型检查点
    ckpt = torch.load('artifacts/fusion_balanced_v2.pt', map_location='cpu', weights_only=False)

    print(f"📊 模型信息:")
    print(f"  特征数量: {ckpt.get('input_features', 'N/A')}")
    print(f"  特征名称: {ckpt.get('feature_names', [])}")
    print(f"  标准化器均值: {ckpt.get('scaler_mean', [])}")
    print(f"  标准化器标准差: {ckpt.get('scaler_scale', [])}")

    # 检查百度特征的标准化
    baidu_features = [21.0, 13.0, 1.0, 0.0, 16.0, 5.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 200.0, 227.0]

    print(f"\n📊 百度特征标准化检查:")
    print(f"  原始特征: {baidu_features}")

    scaler_mean = np.array(ckpt.get('scaler_mean', [0.0] * len(baidu_features)))
    scaler_scale = np.array(ckpt.get('scaler_scale', [1.0] * len(baidu_features)))

    print(f"  标准化器均值: {scaler_mean}")
    print(f"  标准化器标准差: {scaler_scale}")

    # 逐个特征标准化
    normalized_features = []
    for i, (feat, mean, scale) in enumerate(zip(baidu_features, scaler_mean, scaler_scale)):
        normalized = (feat - mean) / scale
        normalized_features.append(normalized)
        if abs(normalized) > 10:  # 检查极端值
            print(f"  ⚠️  特征 {i} ({ckpt.get('feature_names', [f'feat_{i}'])[i]}): {feat} -> {normalized:.2f}")

    print(f"  标准化后特征: {normalized_features}")

if __name__ == "__main__":
    check_model_scaler()