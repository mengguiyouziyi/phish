#!/usr/bin/env python3
"""
重新训练高级模型v3 - 增强钓鱼网站特征识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('.')

from phishguard_v1.models.fusion_model import AdvancedFusionDNN

def create_dataset():
    """创建增强数据集"""
    np.random.seed(42)

    # 良性网站特征 (36维)
    benign_features = []
    benign_urls = [
        "https://www.baidu.com", "https://www.qq.com", "https://www.taobao.com",
        "https://www.tmall.com", "https://www.jd.com", "https://www.sohu.com",
        "https://www.163.com", "https://www.sina.com.cn", "https://www.ifeng.com",
        "https://www.cctv.com", "https://www.xinhuanet.com", "https://www.people.com.cn",
        "https://www.google.com", "https://www.microsoft.com", "https://www.amazon.com",
        "https://www.facebook.com", "https://www.twitter.com", "https://www.linkedin.com",
        "https://www.apple.com", "https://www.github.com", "https://www.wikipedia.org"
    ]

    for url in benign_urls:
        features = extract_enhanced_features(url)
        benign_features.append(features)

    # 钓鱼网站特征 - 更典型的钓鱼模式
    phishing_features = []
    phishing_urls = [
        # 高度可疑的钓鱼网站
        "http://secure-login.apple.com.verify-login.com",
        "http://www.amazon.update.account.secure-login.net",
        "http://paypal.com.secure.transaction.update.com",
        "http://www.microsoft.account.verify.urgent-action.com",
        "http://google.com.login.verify.account-update.com",
        "http://facebook.com.security.check.account-confirm.com",
        "http://linkedin.com.account.suspended.verify-now.com",
        "http://twitter.com.password.reset.urgent-action.net",
        "http://bankofamerica.com.account.verify.secure-login.com",
        "http://chase.com.online.banking.verify.account.com",
        "http://icbc.com.cn.online.banking.secure-login.net",
        "http://alipay.com.account.verify.urgent-action.com",
        "http://taobao.com.login.verify.account-update.com",
        "http://qq.com.security.check.account-confirm.com",
        "http://baidu.com.account.verify.secure-login.com",
        "http://weibo.com.account.suspended.verify-now.com",
        "http://jd.com.password.reset.urgent-action.net",
        "http://tmall.com.account.verify.urgent-action.com",
        "http://sohu.com.login.verify.account-update.com",
        "http://163.com.security.check.account-confirm.com",
        # 更多钓鱼模式
        "http://apple.com.cn.verify-login.cn.com",
        "http://update.microsoft.com.security-alert.org",
        "http://amazon-gift-card-winner.com",
        "http://verify-paypal-account.com",
        "http://microsoft-login-alert.com",
        "http://security-check.facebook.com.net",
        "http://twitter.com.account.locked.warning.com",
        "http://github.com.account.verify.urgent.net"
    ]

    for url in phishing_urls:
        features = extract_enhanced_features(url)
        phishing_features.append(features)

    X = np.array(benign_features + phishing_features)
    y = np.array([0] * len(benign_features) + [1] * len(phishing_features))  # 0=良性，1=钓鱼

    print(f"数据集大小: {len(X)} (良性: {len(benign_features)}, 钓鱼: {len(phishing_features)})")
    print(f"特征维度: {X.shape[1]}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def extract_enhanced_features(url):
    """提取增强的36个特征"""
    import re

    host = url.split('//')[-1].split('/')[0] if '//' in url else url.split('/')[0]
    path = '/' + '/'.join(url.split('/')[3:]) if len(url.split('/')) > 3 else '/'
    query = url.split('?')[-1] if '?' in url else ''
    fragment = url.split('#')[-1] if '#' in url else ''

    features = []

    # URL numeric features (24个)
    features.extend([
        float(len(url)),  # url_len
        float(len(host)),  # host_len
        float(len(path)),  # path_len
        float(sum(c.isdigit() for c in url)),  # num_digits
        float(sum(c.isalpha() for c in url)),  # num_letters
        float(sum(not c.isalnum() for c in url)),  # num_specials
        float(url.count('.')),  # num_dots
        float(url.count('-')),  # num_hyphen
        float(url.count('/')),  # num_slash
        float(url.count('?')),  # num_qm
        float(url.count('@')),  # num_at
        float(url.count('%')),  # num_pct
        float(url.count('=')),  # num_equal
        float(url.count('&')),  # num_amp
        float(url.count('+')),  # num_plus
        float(url.count('#')),  # num_hash
        float(len(query)),  # query_len
        float(len(fragment)),  # fragment_len
        float(len(host.split('.')[0]) if '.' in host else len(host)),  # domain_len
        float(sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0),  # digit_ratio
        float(sum(not c.isalnum() for c in url) / len(url) if len(url) > 0 else 0),  # special_ratio
        float(sum(c.isalpha() for c in url) / len(url) if len(url) > 0 else 0),  # letter_ratio
        float(len([p for p in path.split('/') if p]) if path != '/' else 0),  # path_depth
        float(len([p for p in query.split('&') if p]) if query else 0),  # num_params
    ])

    # 增强的布尔特征 (12个)
    features.extend([
        float(1 if any(part.isdigit() for part in host.split('.')) else 0),  # has_ip
        float(1 if any(tld in host.lower() for tld in ['.tk', '.ml', '.ga', '.cf', '.top', '.click', '.xyz', '.info']) else 0),  # tld_suspicious
        float(1 if 'xn--' in host.lower() else 0),  # has_punycode
        float(1 if url.startswith('https') else 0),  # scheme_https - 重要特征！
        float(1 if '?' in url else 0),  # has_params
        float(1 if any(ext in path.lower() for ext in ['.php', '.html', '.htm', '.asp', '.aspx', '.jsp', '.cgi', '.pl']) else 0),  # has_file_ext
        float(1 if any(ext in path.lower() for ext in ['.exe', '.bat', '.cmd', '.scr', '.pif']) else 0),  # is_suspicious_file
        float(1 if host.startswith('www.') else 0),  # has_www
        float(1 if len(host) > 30 else 0),  # is_long_domain
        float(host.count('.') if host != 'localhost' else 0),  # subdomain_depth - 重要特征！
        float(200),  # status_code (固定值)
        float(1024),  # bytes (固定值)
    ])

    return features

def train_model():
    """训练模型"""
    print("🔍 创建增强数据集...")
    X, y, scaler = create_dataset()

    print("🧠 创建模型...")
    model = AdvancedFusionDNN(num_features=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🏋️ 训练模型...")
    model.train()

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 增加训练轮数
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/500], Loss: {loss.item():.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean()
        print(f"✅ 训练准确率: {accuracy:.4f}")

        # 计算概率
        probs = F.softmax(outputs, dim=1)
        benign_probs = probs[y == 0, 0].mean().item()
        phishing_probs = probs[y == 1, 1].mean().item()
        print(f"良性样本平均良性概率: {benign_probs:.4f}")
        print(f"钓鱼样本平均钓鱼概率: {phishing_probs:.4f}")

        # 检查个别样本
        print("\n📊 样本检查:")
        for i in range(min(5, len(y))):
            if y[i] == 1:  # 钓鱼样本
                print(f"钓鱼样本 {i}: P(良性)={probs[i,0]:.4f}, P(钓鱼)={probs[i,1]:.4f}")
            else:  # 良性样本
                print(f"良性样本 {i}: P(良性)={probs[i,0]:.4f}, P(钓鱼)={probs[i,1]:.4f}")

    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_features': X.shape[1],
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'training_accuracy': accuracy.item(),
        'model_type': 'AdvancedFusionDNN-v3',
        'training_data_size': len(X),
        'benign_count': len(y[y == 0]),
        'phishing_count': len(y[y == 1]),
        'label_mapping': {0: 'benign', 1: 'phishing'},
        'description': 'Enhanced phishing detection model v3 with better features',
        'training_epochs': 500
    }

    torch.save(checkpoint, 'artifacts/fusion_advanced_v3.pt')
    print("✅ 模型已保存到 artifacts/fusion_advanced_v3.pt")

if __name__ == "__main__":
    train_model()