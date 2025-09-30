#!/usr/bin/env python3
"""
训练平衡的融合模型，包含更多样化的良性网站
"""

import pandas as pd
import numpy as np
from pathlib import Path
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
from phishguard_v1.models.dataset import NUMERIC_COLS

def create_balanced_training_data():
    """创建平衡的训练数据"""
    print("📦 创建平衡训练数据集...")

    # 加载现有数据
    try:
        df_old = pd.read_parquet('data/dataset.parquet')
        print(f"  现有数据: {len(df_old)} 条记录")
    except Exception as e:
        print(f"❌ 无法加载现有数据: {e}")
        return None

    # 加载GitHub数据
    try:
        df_github = pd.read_parquet('validated_github_data/training_data.parquet')
        print(f"  GitHub数据: {len(df_github)} 条记录")
    except Exception as e:
        print(f"❌ 无法加载GitHub数据: {e}")
        return None

    # 添加更多良性网站样例
    additional_benign = [
        # 中国知名网站
        {'url': 'https://www.baidu.com', 'label': 0},
        {'url': 'https://www.taobao.com', 'label': 0},
        {'url': 'https://www.qq.com', 'label': 0},
        {'url': 'https://www.weibo.com', 'label': 0},
        {'url': 'https://www.zhihu.com', 'label': 0},
        {'url': 'https://www.douban.com', 'label': 0},
        {'url': 'https://www.tmall.com', 'label': 0},
        {'url': 'https://www.jd.com', 'label': 0},
        {'url': 'https://www.163.com', 'label': 0},
        {'url': 'https://www.sohu.com', 'label': 0},
        # 国际知名网站
        {'url': 'https://www.instagram.com', 'label': 0},
        {'url': 'https://www.linkedin.com', 'label': 0},
        {'url': 'https://www.github.com', 'label': 0},
        {'url': 'https://www.stackoverflow.com', 'label': 0},
        {'url': 'https://www.wikipedia.org', 'label': 0},
        # 中等规模良性网站
        {'url': 'https://medium.com', 'label': 0},
        {'url': 'https://www.reddit.com', 'label': 0},
        {'url': 'https://www.quora.com', 'label': 0},
        {'url': 'https://www.bilibili.com', 'label': 0},
        {'url': 'https://www.douyin.com', 'label': 0},
    ]

    # 创建额外良性网站DataFrame
    df_additional = pd.DataFrame(additional_benign)
    print(f"  额外良性网站: {len(df_additional)} 条记录")

    # 合并所有数据
    all_data = [df_old, df_github, df_additional]
    merged_df = pd.concat(all_data, ignore_index=True)

    # 去重
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['url'], keep='first')
    final_count = len(merged_df)

    print(f"  合并后: {final_count} 条记录")
    print(f"  去重减少: {initial_count - final_count} 条")

    # 统计标签分布
    phishing_count = len(merged_df[merged_df['label'] == 1])
    benign_count = len(merged_df[merged_df['label'] == 0])
    print(f"  钓鱼网站: {phishing_count} ({phishing_count/final_count*100:.1f}%)")
    print(f"  良性网站: {benign_count} ({benign_count/final_count*100:.1f}%)")

    return merged_df

def extract_features_for_url(url):
    """为URL提取特征"""
    from urllib.parse import urlparse
    import re

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    # URL基础特征
    features = {}

    features['url_len'] = len(url)
    features['host_len'] = len(hostname)
    features['path_len'] = len(path)
    features['query_len'] = len(query)
    features['fragment_len'] = len(fragment)

    # 字符统计
    features['num_digits'] = len(re.findall(r'\d', url))
    features['num_letters'] = len(re.findall(r'[a-zA-Z]', url))
    features['num_specials'] = len(re.findall(r'[^\w\d]', url))
    features['num_dots'] = url.count('.')
    features['num_hyphen'] = url.count('-')
    features['num_slash'] = url.count('/')
    features['num_qm'] = url.count('?')
    features['num_at'] = url.count('@')
    features['num_pct'] = url.count('%')

    # 域名特征
    features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', hostname) else 0
    features['subdomain_depth'] = len(hostname.split('.')) - 2 if hostname.count('.') > 1 else 0
    features['tld_suspicious'] = 1 if hostname.split('.')[-1] in ['tk', 'ml', 'ga', 'cf', 'top'] else 0
    features['has_punycode'] = 1 if 'xn--' in hostname else 0
    features['scheme_https'] = 1 if parsed.scheme == 'https' else 0

    # HTTP响应特征（默认值）
    features['status_code'] = 200
    features['bytes'] = 1024  # 假设的平均大小

    return features

def prepare_balanced_data(df):
    """准备平衡的训练数据"""
    print("🔧 准备平衡训练数据...")

    # 为每个URL提取特征
    feature_data = []
    for _, row in df.iterrows():
        url = row['url']
        label = row['label']
        features = extract_features_for_url(url)
        features['label'] = label
        feature_data.append(features)

    df_features = pd.DataFrame(feature_data)

    # 确保所有特征列都存在
    required_features = [
        'url_len', 'host_len', 'path_len', 'num_digits', 'num_letters', 'num_specials',
        'num_dots', 'num_hyphen', 'num_slash', 'num_qm', 'num_at', 'num_pct',
        'has_ip', 'subdomain_depth', 'tld_suspicious', 'has_punycode', 'scheme_https',
        'query_len', 'fragment_len', 'status_code', 'bytes'
    ]

    for feat in required_features:
        if feat not in df_features.columns:
            df_features[feat] = 0

    # 转换boolean类型
    for col in ['has_ip', 'tld_suspicious', 'has_punycode', 'scheme_https']:
        df_features[col] = df_features[col].astype(int)

    # 提取特征和标签
    X = df_features[required_features].values.astype(float)
    y = df_features['label'].values

    # 处理缺失值
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"  有效样本: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_balanced_model(X_train, X_test, y_train, y_test):
    """训练平衡模型"""
    print("🧠 训练平衡融合模型...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")

    # 创建模型
    input_dim = X_train.shape[1]
    model = FusionDNN(num_features=input_dim).to(device)

    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # 训练参数
    num_epochs = 200
    batch_size = 32
    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size].long()

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 40 == 0:
            print(f"    轮次 {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()

        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions, zero_division=0)
        recall = recall_score(y_test, test_predictions, zero_division=0)
        f1 = f1_score(y_test, test_predictions, zero_division=0)

        print(f"  测试性能:")
        print(f"    准确率: {accuracy:.4f}")
        print(f"    精确率: {precision:.4f}")
        print(f"    召回率: {recall:.4f}")
        print(f"    F1分数: {f1:.4f}")

    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_balanced_model(model, scaler, metrics):
    """保存平衡模型"""
    print("💾 保存平衡模型...")

    save_dir = Path("artifacts")
    save_dir.mkdir(exist_ok=True)

    feature_names = [
        'url_len', 'host_len', 'path_len', 'num_digits', 'num_letters', 'num_specials',
        'num_dots', 'num_hyphen', 'num_slash', 'num_qm', 'num_at', 'num_pct',
        'has_ip', 'subdomain_depth', 'tld_suspicious', 'has_punycode', 'scheme_https',
        'query_len', 'fragment_len', 'status_code', 'bytes'
    ]

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_features': len(feature_names),
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'metrics': metrics,
        'training_time': time.time(),
        'model_type': 'balanced_fusion_v2'
    }

    model_path = save_dir / "fusion_balanced_v2.pt"
    torch.save(checkpoint, model_path)
    print(f"  模型已保存: {model_path}")

    return model_path

def main():
    """主函数"""
    print("🚀 训练平衡融合模型 v2")
    print("=" * 50)

    # 1. 创建平衡数据
    df = create_balanced_training_data()
    if df is None:
        return

    # 2. 准备数据
    X_train, X_test, y_train, y_test, scaler = prepare_balanced_data(df)

    # 3. 训练模型
    model, metrics = train_balanced_model(X_train, X_test, y_train, y_test)

    # 4. 保存模型
    model_path = save_balanced_model(model, scaler, metrics)

    # 5. 输出总结
    print(f"\n🎉 训练完成!")
    print(f"📊 模型信息:")
    print(f"  训练样本: {len(df)}")
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  模型性能: 准确率 {metrics['accuracy']:.2%}, 召回率 {metrics['recall']:.2%}")

    print(f"\n🔧 使用方法:")
    print(f"  1. 更新API配置使用模型: {model_path}")
    print(f"  2. 测试新模型性能")
    print(f"  3. 特别测试百度等中国网站")

if __name__ == "__main__":
    main()