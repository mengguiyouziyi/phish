#!/usr/bin/env python3
"""
使用GitHub数据和现有数据训练简化版融合模型
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

# 共同特征列表（两个数据集都有的特征）
COMMON_FEATURES = [
    'url_len', 'host_len', 'path_len', 'num_digits', 'num_letters', 'num_specials',
    'num_dots', 'num_hyphen', 'num_slash', 'num_qm', 'num_at', 'num_pct',
    'has_ip', 'subdomain_depth', 'tld_suspicious', 'has_punycode', 'scheme_https',
    'query_len', 'fragment_len', 'status_code', 'bytes'
]

def load_and_merge_data():
    """加载并合并数据"""
    print("📦 加载数据集...")

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

    # 提取共同特征
    old_features = df_old[COMMON_FEATURES + ['label', 'url']].copy()
    github_features = df_github[COMMON_FEATURES + ['label', 'url']].copy()

    # 合并数据
    merged_df = pd.concat([old_features, github_features], ignore_index=True)

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

def prepare_data(df):
    """准备训练数据"""
    print("🔧 准备训练数据...")

    # 复制数据框避免修改原数据
    df_processed = df.copy()

    # 转换boolean类型为int
    for col in COMMON_FEATURES:
        if df_processed[col].dtype == 'bool':
            df_processed[col] = df_processed[col].astype(int)

    # 提取特征和标签
    X = df_processed[COMMON_FEATURES].values.astype(float)
    y = df['label'].values

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

def train_model(X_train, X_test, y_train, y_test):
    """训练模型"""
    print("🧠 训练融合模型...")

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
    num_epochs = 150
    batch_size = 32
    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)  # 使用交叉熵损失

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size].long()  # CrossEntropy需要long类型

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 30 == 0:
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

def save_model(model, scaler, metrics):
    """保存模型"""
    print("💾 保存模型...")

    save_dir = Path("artifacts")
    save_dir.mkdir(exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_features': len(COMMON_FEATURES),
        'feature_names': COMMON_FEATURES,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'metrics': metrics,
        'training_time': time.time(),
        'model_type': 'github_enhanced_fusion'
    }

    model_path = save_dir / "fusion_github_simple.pt"
    torch.save(checkpoint, model_path)
    print(f"  模型已保存: {model_path}")

    return model_path

def main():
    """主函数"""
    print("🚀 训练GitHub增强融合模型")
    print("=" * 50)

    # 1. 加载和合并数据
    df = load_and_merge_data()
    if df is None:
        return

    # 2. 准备数据
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # 3. 训练模型
    model, metrics = train_model(X_train, X_test, y_train, y_test)

    # 4. 保存模型
    model_path = save_model(model, scaler, metrics)

    # 5. 输出总结
    print(f"\n🎉 训练完成!")
    print(f"📊 模型信息:")
    print(f"  训练样本: {len(df)}")
    print(f"  特征维度: {len(COMMON_FEATURES)}")
    print(f"  模型性能: 准确率 {metrics['accuracy']:.2%}, 召回率 {metrics['recall']:.2%}")

    print(f"\n🔧 使用方法:")
    print(f"  1. 更新API配置使用模型: {model_path}")
    print(f"  2. 测试新模型性能")
    print(f"  3. 对比与原模型的改进")

if __name__ == "__main__":
    main()