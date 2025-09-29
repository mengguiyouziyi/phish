#!/usr/bin/env python3
"""
测试平衡模型对百度等中国网站的表现
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')

from phishguard_v1.models.fusion_model import FusionDNN

def test_balanced_model():
    """测试平衡模型"""
    print("🧪 测试平衡融合模型 v2...")

    # 测试URL列表
    test_urls = [
        # 中国良性网站
        'https://www.baidu.com',
        'https://www.taobao.com',
        'https://www.qq.com',
        'https://www.jd.com',
        'https://www.zhihu.com',
        # 国际良性网站
        'https://www.google.com',
        'https://www.github.com',
        'https://www.wikipedia.org',
        # 钓鱼网站样例
        'http://verify-paypal-account.com',
        'http://apple-security-update.info',
        'http://microsoft-login-alert.com',
    ]

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('artifacts/fusion_balanced_v2.pt', map_location='cpu', weights_only=False)

    model = FusionDNN(num_features=ckpt['input_features']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 准备标准化器
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt['scaler_mean'])
    scaler.scale_ = np.array(ckpt['scaler_scale'])

    feature_names = ckpt['feature_names']

    def extract_url_features(url):
        """提取URL特征"""
        from urllib.parse import urlparse
        import re

        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""

        features = {}

        features['url_len'] = len(url)
        features['host_len'] = len(hostname)
        features['path_len'] = len(path)
        features['query_len'] = len(query)
        features['fragment_len'] = len(fragment)

        features['num_digits'] = len(re.findall(r'\d', url))
        features['num_letters'] = len(re.findall(r'[a-zA-Z]', url))
        features['num_specials'] = len(re.findall(r'[^\w\d]', url))
        features['num_dots'] = url.count('.')
        features['num_hyphen'] = url.count('-')
        features['num_slash'] = url.count('/')
        features['num_qm'] = url.count('?')
        features['num_at'] = url.count('@')
        features['num_pct'] = url.count('%')

        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', hostname) else 0
        features['subdomain_depth'] = len(hostname.split('.')) - 2 if hostname.count('.') > 1 else 0
        features['tld_suspicious'] = 1 if hostname.split('.')[-1] in ['tk', 'ml', 'ga', 'cf', 'top'] else 0
        features['has_punycode'] = 1 if 'xn--' in hostname else 0
        features['scheme_https'] = 1 if parsed.scheme == 'https' else 0

        features['status_code'] = 200
        features['bytes'] = 1024

        return features

    print(f"\n📊 测试结果:")
    print("-" * 80)

    for url in test_urls:
        # 提取特征
        features = extract_url_features(url)
        X = np.array([[features[feat] for feat in feature_names]])
        X_scaled = scaler.transform(X)

        # 预测
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

            benign_prob = probs[0][0].item()
            phishing_prob = probs[0][1].item()

            prediction = "良性" if benign_prob > phishing_prob else "钓鱼"
            confidence = max(benign_prob, phishing_prob)

            print(f"{url}")
            print(f"  预测: {prediction} (置信度: {confidence:.2%})")
            print(f"  良性概率: {benign_prob:.4f} ({benign_prob*100:.2f}%)")
            print(f"  钓鱼概率: {phishing_prob:.4f} ({phishing_prob*100:.2f}%)")
            print()

if __name__ == "__main__":
    test_balanced_model()