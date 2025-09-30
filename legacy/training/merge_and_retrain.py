#!/usr/bin/env python3
"""
合并GitHub验证数据与现有数据，重新训练融合模型
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import time
import sys
sys.path.append('.')

from phishguard_v1.models.fusion_model import FusionDNN
from phishguard_v1.models.dataset import NUMERIC_COLS

def load_existing_data():
    """加载现有数据"""
    data_files = [
        "data/enhanced_dataset.parquet",
        "artifacts/fusion_training_data.parquet",
        "data/training_data.parquet",
        "data/dataset.parquet"
    ]

    for file_path in data_files:
        if Path(file_path).exists():
            try:
                print(f"📦 尝试加载现有数据: {file_path}")
                df = pd.read_parquet(file_path)
                print(f"  成功加载: {len(df)} 条记录")
                return df
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
                continue

    print("❌ 未找到有效的现有训练数据")
    return None

def load_github_data():
    """加载GitHub验证数据"""
    github_file = "validated_github_data/training_data.parquet"
    if Path(github_file).exists():
        print(f"📦 加载GitHub数据: {github_file}")
        return pd.read_parquet(github_file)
    else:
        print("❌ 未找到GitHub验证数据")
        return None

def merge_datasets(existing_df, github_df):
    """合并数据集"""
    print(f"🔄 合并数据集...")
    print(f"  现有数据: {len(existing_df)} 条记录")
    print(f"  GitHub数据: {len(github_df)} 条记录")

    # 确保列名一致
    existing_df = existing_df.copy()
    github_df = github_df.copy()

    # 重命名GitHub数据的列以匹配现有格式
    column_mapping = {
        'url': 'url',
        'final_url': 'final_url',
        'label': 'label',
        'timestamp': 'timestamp'
    }

    # 只保留共有的特征列
    common_features = set(existing_df.columns) & set(github_df.columns)
    common_features = [col for col in common_features if col in NUMERIC_COLS or col in column_mapping.values()]

    print(f"  共同特征数: {len([f for f in common_features if f in NUMERIC_COLS])}")

    # 提取共同特征
    existing_features = existing_df[common_features].copy()
    github_features = github_df[common_features].copy()

    # 合并数据
    merged_df = pd.concat([existing_features, github_features], ignore_index=True)

    # 去重（基于URL）
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['url'], keep='first')
    final_count = len(merged_df)

    print(f"  合并后记录: {final_count} 条")
    print(f"  去重后减少: {initial_count - final_count} 条")

    return merged_df

def prepare_training_data(df):
    """准备训练数据"""
    print("🔧 准备训练数据...")

    # 确保所有必需的特征列都存在
    missing_features = [col for col in NUMERIC_COLS if col not in df.columns]
    if missing_features:
        print(f"⚠️  缺失特征: {missing_features}")
        # 为缺失特征添加默认值
        for col in missing_features:
            df[col] = 0.0

    # 选择特征列和标签
    feature_cols = [col for col in NUMERIC_COLS if col in df.columns]
    X = df[feature_cols].values
    y = df['label'].values

    # 移除包含NaN的行
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"  有效样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  钓鱼网站: {np.sum(y == 1)} 个")
    print(f"  良性网站: {np.sum(y == 0)} 个")

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def train_fusion_model(X_train, X_test, y_train, y_test, feature_cols):
    """训练融合模型"""
    print("🧠 训练融合模型...")

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")

    # 创建模型
    input_dim = X_train.shape[1]
    model = FusionDNN(num_features=input_dim).to(device)

    # 转换数据为张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # 训练参数
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss().to(device)  # 将损失函数移动到设备

    # 训练循环
    model.train()
    print(f"  开始训练，共 {num_epochs} 轮...")

    for epoch in range(num_epochs):
        # 小批量训练
        total_loss = 0
        num_batches = 0

        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 20 == 0:
            print(f"    轮次 {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")

    # 评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        test_predictions = (test_outputs >= 0.5).float()

        test_accuracy = accuracy_score(y_test, test_predictions.cpu().numpy())
        test_precision = precision_score(y_test, test_predictions.cpu().numpy(), zero_division=0)
        test_recall = recall_score(y_test, test_predictions.cpu().numpy(), zero_division=0)
        test_f1 = f1_score(y_test, test_predictions.cpu().numpy(), zero_division=0)

        print(f"  测试集性能:")
        print(f"    准确率: {test_accuracy:.4f}")
        print(f"    精确率: {test_precision:.4f}")
        print(f"    召回率: {test_recall:.4f}")
        print(f"    F1分数: {test_f1:.4f}")

    return model, scaler, {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }

def save_model(model, scaler, feature_cols, metrics):
    """保存模型"""
    print("💾 保存模型...")

    # 创建保存目录
    save_dir = Path("artifacts")
    save_dir.mkdir(exist_ok=True)

    # 保存模型检查点
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_features': len(feature_cols),
        'feature_names': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'metrics': metrics,
        'training_time': time.time()
    }

    model_path = save_dir / "fusion_github_enhanced.pt"
    torch.save(checkpoint, model_path)
    print(f"  模型已保存: {model_path}")

    # 保存训练数据
    training_data_path = save_dir / "fusion_github_training_data.parquet"
    return model_path, training_data_path

def main():
    """主函数"""
    print("🚀 开始合并数据并重新训练融合模型")
    print("=" * 60)

    # 1. 加载数据
    existing_df = load_existing_data()
    github_df = load_github_data()

    if existing_df is None and github_df is None:
        print("❌ 没有找到任何训练数据，退出")
        return

    if existing_df is None:
        print("⚠️  仅使用GitHub数据训练新模型")
        merged_df = github_df
    elif github_df is None:
        print("⚠️  未找到GitHub数据，仅使用现有数据训练")
        merged_df = existing_df
    else:
        # 2. 合并数据
        merged_df = merge_datasets(existing_df, github_df)

    # 3. 准备训练数据
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_training_data(merged_df)

    # 4. 训练模型
    model, scaler, metrics = train_fusion_model(X_train, X_test, y_train, y_test, feature_cols)

    # 5. 保存模型
    model_path, data_path = save_model(model, scaler, feature_cols, metrics)

    # 6. 保存合并后的数据
    merged_data_path = "artifacts/fusion_github_merged_data.parquet"
    merged_df.to_parquet(merged_data_path)
    print(f"  合并数据已保存: {merged_data_path}")

    # 7. 输出总结
    print(f"\n🎉 训练完成!")
    print(f"📊 最终统计:")
    print(f"  总训练样本: {len(merged_df)}")
    print(f"  特征维度: {len(feature_cols)}")
    print(f"  模型性能: 准确率 {metrics['accuracy']:.2%}, 召回率 {metrics['recall']:.2%}")
    print(f"  模型文件: {model_path}")
    print(f"  数据文件: {merged_data_path}")

    print(f"\n🔧 下一步操作:")
    print(f"  1. 更新API配置使用新模型: fusion_github_enhanced.pt")
    print(f"  2. 测试模型性能")
    print(f"  3. 评估改进效果")

if __name__ == "__main__":
    main()