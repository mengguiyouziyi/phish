#!/usr/bin/env python3
"""
检查高级模型详细信息
"""

import torch
import sys
sys.path.append('.')

def check_model_info():
    """检查模型信息"""
    print("🔍 检查高级模型信息...")

    # 加载模型
    checkpoint = torch.load('artifacts/fusion_advanced.pt', map_location='cpu', weights_only=False)

    print(f"📊 模型信息:")
    print(f"  特征数量: {checkpoint.get('input_features', 'Unknown')}")
    print(f"  模型类型: {checkpoint.get('model_type', 'Unknown')}")
    print(f"  训练准确率: {checkpoint.get('training_accuracy', 'Unknown'):.4f}")
    print(f"  测试准确率: {checkpoint.get('test_accuracy', 'Unknown'):.4f}")
    print(f"  精确率: {checkpoint.get('precision', 'Unknown'):.4f}")
    print(f"  召回率: {checkpoint.get('recall', 'Unknown'):.4f}")
    print(f"  F1分数: {checkpoint.get('f1_score', 'Unknown'):.4f}")
    print(f"  数据集大小: {checkpoint.get('training_data_size', 'Unknown')}")
    print(f"  良性样本数: {checkpoint.get('benign_count', 'Unknown')}")
    print(f"  钓鱼样本数: {checkpoint.get('phishing_count', 'Unknown')}")

    feature_names = checkpoint.get('feature_names', [])
    print(f"📋 特征列表 ({len(feature_names)} 个):")
    for i, feature in enumerate(feature_names):
        print(f"  {i+1:2d}. {feature}")

    # 检查混淆矩阵
    cm = checkpoint.get('confusion_matrix', [])
    if cm:
        print(f"📊 混淆矩阵:")
        print(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")

if __name__ == "__main__":
    check_model_info()