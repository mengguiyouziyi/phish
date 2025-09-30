#!/usr/bin/env python3
"""
重新训练增强的DNN模型
使用改进的特征和更多的数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from typing import Dict, Any, Tuple

# 导入本地模块
from phishguard_v1.models.dataset import NUMERIC_COLS, build_feature_matrix, ParquetDataset
from phishguard_v1.models.fusion_model import FusionDNN, predict_proba

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载并预处理数据"""
    print(f"📂 加载数据从: {data_path}")

    # 读取所有数据文件
    data_files = ["train.parquet", "val.parquet", "test.parquet"]
    all_data = []

    for file in data_files:
        file_path = Path(data_path) / file
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_data.append(df)
            print(f"  ✅ {file}: {len(df)} 条记录")

    if not all_data:
        raise FileNotFoundError(f"在 {data_path} 中未找到数据文件")

    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 总数据量: {len(combined_df)} 条记录")

    # 数据统计
    benign_count = len(combined_df[combined_df["label"] == 0])
    phishing_count = len(combined_df[combined_df["label"] == 1])

    print(f"  良性网站: {benign_count} ({benign_count/len(combined_df)*100:.1f}%)")
    print(f"  钓鱼网站: {phishing_count} ({phishing_count/len(combined_df)*100:.1f}%)")

    # 重新分割数据 (80/10/10)
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.2,
        random_state=42,
        stratify=combined_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"]
    )

    print(f"📈 数据分割:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")

    return train_df, val_df, test_df

def create_model(input_dim: int, hidden_dims: list = None) -> FusionDNN:
    """创建增强的DNN模型"""
    if hidden_dims is None:
        hidden_dims = [512, 256, 128, 64]  # 更深的网络

    print(f"🧠 创建增强DNN模型:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层: {hidden_dims}")
    print(f"  输出维度: 2")

    model = FusionDNN(num_features=input_dim)

    # 如果需要不同的架构，可以在这里修改
    # 当前使用默认的 3层架构

    return model

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 50 == 0:
            print(f"    批次 {batch_idx}/{len(dataloader)} - 损失: {loss.item():.4f}")

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy

def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets)

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[FusionDNN, Dict[str, Any]]:
    """训练模型"""
    print("🚀 开始训练增强DNN模型...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")

    # 构建特征矩阵
    print("🔧 构建特征矩阵...")
    X_train, _ = build_feature_matrix(train_df)
    X_val, _ = build_feature_matrix(val_df)
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    print(f"  训练特征形状: {X_train.shape}")
    print(f"  验证特征形状: {X_val.shape}")

    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_dim = X_train.shape[1]
    model = create_model(input_dim)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练循环
    best_val_accuracy = 0
    best_model_state = None
    training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    print(f"📊 开始训练 ({num_epochs} epochs)...")
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{num_epochs}")

        # 训练
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_accuracy, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_accuracy)
        training_history["val_loss"].append(val_loss)
        training_history["val_accuracy"].append(val_accuracy)

        print(f"  训练 - 损失: {train_loss:.4f}, 准确率: {train_accuracy:.2f}%")
        print(f"  验证 - 损失: {val_loss:.4f}, 准确率: {val_accuracy:.2f}%")

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  🎯 新最佳模型! 验证准确率: {val_accuracy:.2f}%")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_history["best_val_accuracy"] = best_val_accuracy

    return model, training_history

def evaluate_model(model: nn.Module, test_df: pd.DataFrame) -> Dict[str, Any]:
    """评估模型"""
    print("🧪 评估模型性能...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 构建测试数据
    X_test, _ = build_feature_matrix(test_df)
    y_test = test_df["label"].values

    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 预测
    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = output.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(target.numpy())

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)

    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    phishing_recall = recall_score(all_targets, all_predictions, pos_label=1)

    print(f"📊 测试结果:")
    print(f"  总体准确率: {accuracy*100:.2f}%")
    print(f"  钓鱼网站召回率: {phishing_recall*100:.2f}%")

    # 详细报告
    print(f"\n📋 分类报告:")
    print(classification_report(all_targets, all_predictions, target_names=["良性", "钓鱼"]))

    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    print(f"🔄 混淆矩阵:")
    print(f"  {cm[0][0]} 真阴性 | {cm[0][1]} 假阳性")
    print(f"  {cm[1][0]} 假阴性 | {cm[1][1]} 真阳性")

    return {
        "accuracy": accuracy,
        "phishing_recall": phishing_recall,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "targets": all_targets,
        "confusion_matrix": cm
    }

def save_model(model: nn.Module, output_path: str, training_history: Dict[str, Any]):
    """保存模型"""
    print(f"💾 保存模型到: {output_path}")

    # 创建模型目录
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True)

    # 准备保存内容
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_features": len(NUMERIC_COLS),
        "model_architecture": "FusionDNN",
        "training_history": training_history,
        "feature_columns": NUMERIC_COLS,
        "best_val_accuracy": training_history.get("best_val_accuracy", 0)
    }

    torch.save(checkpoint, output_path)
    print(f"  ✅ 模型已保存")

def main():
    """主函数"""
    print("🎯 开始重新训练增强DNN模型")
    print("=" * 60)

    # 数据路径
    data_path = "data_enhanced"  # 使用增强数据
    if not Path(data_path).exists():
        print(f"⚠️  {data_path} 不存在，使用原始数据")
        data_path = "data"

    # 加载数据
    train_df, val_df, test_df = load_and_preprocess_data(data_path)

    # 训练模型
    model, training_history = train_model(
        train_df, val_df,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )

    # 评估模型
    evaluation_results = evaluate_model(model, test_df)

    # 保存模型
    model_path = "artifacts/fusion_enhanced.pt"
    save_model(model, model_path, training_history)

    print("\n🎉 训练完成!")
    print(f"📈 最佳验证准确率: {training_history['best_val_accuracy']:.2f}%")
    print(f"📊 测试准确率: {evaluation_results['accuracy']*100:.2f}%")
    print(f"🎯 钓鱼网站召回率: {evaluation_results['phishing_recall']*100:.2f}%")
    print(f"💾 模型保存至: {model_path}")

if __name__ == "__main__":
    main()