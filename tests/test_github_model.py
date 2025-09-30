#!/usr/bin/env python3
"""
修复版本：使用与GitHub增强模型匹配的特征提取
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import time
import sys
sys.path.append('.')

from phishguard_v1.models.fusion_model import FusionDNN

# GitHub增强模型使用的21个特征
GITHUB_FEATURES = [
    'url_len', 'host_len', 'path_len', 'num_digits', 'num_letters', 'num_specials',
    'num_dots', 'num_hyphen', 'num_slash', 'num_qm', 'num_at', 'num_pct',
    'has_ip', 'subdomain_depth', 'tld_suspicious', 'has_punycode', 'scheme_https',
    'query_len', 'fragment_len', 'status_code', 'bytes'
]

def create_github_compatible_features(item):
    """创建与GitHub模型兼容的特征向量"""
    features = {}

    # URL特征
    url_feats = item.get("url_feats", {})
    for feat in GITHUB_FEATURES[:18]:  # 前18个是URL特征
        if feat in url_feats:
            features[feat] = url_feats[feat]
        else:
            # 默认值
            features[feat] = 0 if feat.startswith('has_') else 0

    # HTTP响应特征
    features['status_code'] = item.get('status_code', 200)
    features['bytes'] = item.get('bytes', 0)

    return features

def test_github_model():
    """测试GitHub增强模型"""
    print("🧪 测试GitHub增强模型...")

    # 模拟百度数据
    baidu_features = {
        'url_len': 31,
        'host_len': 13,
        'path_len': 10,
        'num_digits': 0,
        'num_letters': 24,
        'num_specials': 7,
        'num_dots': 3,
        'num_hyphen': 0,
        'num_slash': 3,
        'num_qm': 0,
        'num_at': 0,
        'num_pct': 0,
        'has_ip': 0,
        'subdomain_depth': 1,
        'tld_suspicious': 0,
        'has_punycode': 0,
        'scheme_https': 1,
        'query_len': 0,
        'fragment_len': 0,
        'status_code': 200,
        'bytes': 227
    }

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('artifacts/fusion_github_simple.pt', map_location='cpu', weights_only=False)

    model = FusionDNN(num_features=ckpt['input_features']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 准备数据
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt['scaler_mean'])
    scaler.scale_ = np.array(ckpt['scaler_scale'])

    # 转换特征
    X = np.array([[baidu_features[feat] for feat in GITHUB_FEATURES]])
    X_scaled = scaler.transform(X)

    # 预测
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)

        phishing_prob = probs[0][1].item()
        benign_prob = probs[0][0].item()

        print(f"📊 百度检测结果:")
        print(f"   良性概率: {benign_prob:.4f} ({benign_prob*100:.2f}%)")
        print(f"   钓鱼概率: {phishing_prob:.4f} ({phishing_prob*100:.2f}%)")
        print(f"   预测标签: {'良性' if benign_prob > phishing_prob else '钓鱼'}")

        return phishing_prob

if __name__ == "__main__":
    test_github_model()