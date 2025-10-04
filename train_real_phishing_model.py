#!/usr/bin/env python3
"""
基于真实钓鱼网站数据训练超强模型
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from loguru import logger
import matplotlib.pyplot as plt

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from phishguard_v1.models.fusion_model import FusionDNN
from phishguard_v1.models.training_utils import TrainingArtifacts

class RealPhishingTrainer:
    """真实钓鱼网站模型训练器"""

    def __init__(self,
                 data_dir: str = "data_real_phishing_v2",
                 model_dir: str = "artifacts",
                 device: str = "auto"):
        """
        初始化训练器

        Args:
            data_dir: 数据集目录
            model_dir: 模型保存目录
            device: 设备 ("auto", "cuda", "cpu")
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"🚀 初始化真实钓鱼网站模型训练器")
        logger.info(f"📁 数据目录: {self.data_dir}")
        logger.info(f"💾 模型目录: {self.model_dir}")
        logger.info(f"💻 计算设备: {self.device}")

        # 创建模型目录
        self.model_dir.mkdir(exist_ok=True)

        # 训练历史
        self.history = []

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, Tuple[str, ...]]:
        """加载和预处理数据"""
        logger.info("📖 加载真实钓鱼网站数据集...")

        # 加载数据
        train_df = pd.read_parquet(self.data_dir / "train.parquet")
        val_df = pd.read_parquet(self.data_dir / "val.parquet")
        test_df = pd.read_parquet(self.data_dir / "test.parquet")

        logger.info(f"✅ 训练集: {len(train_df)} 样本")
        logger.info(f"✅ 验证集: {len(val_df)} 样本")
        logger.info(f"✅ 测试集: {len(test_df)} 样本")

        # 检查数据平衡
        logger.info(f"📊 训练集标签分布: {train_df['label'].value_counts().to_dict()}")
        logger.info(f"📊 验证集标签分布: {val_df['label'].value_counts().to_dict()}")

        # 提取特征和标签
        feature_columns = [col for col in train_df.columns if col != 'label']
        logger.info(f"🔧 特征数量: {len(feature_columns)}")

        X_train = train_df[feature_columns].values
        y_train = train_df["label"].values.astype(int)

        X_val = val_df[feature_columns].values
        y_val = val_df["label"].values.astype(int)

        X_test = test_df[feature_columns].values
        y_test = test_df["label"].values.astype(int)

        # 数据标准化
        logger.info("🔄 数据标准化...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # 创建数据加载器
        batch_size = 512 if self.device.type == "cuda" else 256

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader, scaler, tuple(feature_columns)

    def create_model(self, num_features: int) -> nn.Module:
        """创建增强的FusionDNN模型"""
        logger.info(f"🧠 创建增强FusionDNN模型，特征数: {num_features}")

        # 创建自定义2分类模型
        class EnhancedPhishingDetector(nn.Module):
            def __init__(self, input_dim):
                super().__init__()

                # 输入层
                self.fc1 = nn.Linear(input_dim, 1024)
                self.bn1 = nn.BatchNorm1d(1024)
                self.dropout1 = nn.Dropout(0.4)

                # 隐藏层
                self.fc2 = nn.Linear(1024, 512)
                self.bn2 = nn.BatchNorm1d(512)
                self.dropout2 = nn.Dropout(0.3)

                self.fc3 = nn.Linear(512, 256)
                self.bn3 = nn.BatchNorm1d(256)
                self.dropout3 = nn.Dropout(0.3)

                self.fc4 = nn.Linear(256, 128)
                self.bn4 = nn.BatchNorm1d(128)
                self.dropout4 = nn.Dropout(0.2)

                self.fc5 = nn.Linear(128, 64)
                self.bn5 = nn.BatchNorm1d(64)
                self.dropout5 = nn.Dropout(0.2)

                self.fc6 = nn.Linear(64, 32)
                self.bn6 = nn.BatchNorm1d(32)
                self.dropout6 = nn.Dropout(0.1)

                # 输出层
                self.fc7 = nn.Linear(32, 2)  # 2分类：合法vs钓鱼

                self._advanced = True

            def forward(self, x):
                x = torch.relu(self.bn1(self.fc1(x)))
                x = self.dropout1(x)

                x = torch.relu(self.bn2(self.fc2(x)))
                x = self.dropout2(x)

                x = torch.relu(self.bn3(self.fc3(x)))
                x = self.dropout3(x)

                x = torch.relu(self.bn4(self.fc4(x)))
                x = self.dropout4(x)

                x = torch.relu(self.bn5(self.fc5(x)))
                x = self.dropout5(x)

                x = torch.relu(self.bn6(self.fc6(x)))
                x = self.dropout6(x)

                x = self.fc7(x)
                return x

        model = EnhancedPhishingDetector(num_features)
        model = model.to(self.device)
        logger.info(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def compute_class_weights(self, y_train: np.ndarray) -> torch.Tensor:
        """计算类别权重"""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        logger.info(f"⚖️ 类别权重: {weights}")
        return class_weights

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   class_weights: torch.Tensor) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)

            # 使用加权损失
            if len(class_weights) == 2:
                loss = criterion(output, target)
                # 手动应用类别权重
                weights = class_weights[target]
                loss = (loss * weights).mean()

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 20 == 0:
                logger.info(f"📈 Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def evaluate(self, model: nn.Module, data_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float, Dict[str, Any]]:
        """评估模型"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                # 收集预测结果
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total

        # 计算详细指标
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        auc_score = roc_auc_score(all_targets, all_probs)
        report = classification_report(all_targets, all_preds, target_names=['benign', 'phish'], output_dict=True)

        return avg_loss, accuracy, {
            'auc': auc_score,
            'classification_report': report,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }

    def train(self, epochs: int = 100, lr: float = 0.001, patience: int = 15) -> TrainingArtifacts:
        """训练模型"""
        logger.info(f"🚀 开始训练真实钓鱼网站检测模型 - Epochs: {epochs}, LR: {lr}")

        # 加载数据
        train_loader, val_loader, test_loader, scaler, feature_names = self.load_data()

        # 创建模型
        num_features = len(feature_names)
        model = self.create_model(num_features)

        # 计算类别权重
        train_labels = []
        for _, target in train_loader:
            train_labels.extend(target.numpy())
        class_weights = self.compute_class_weights(np.array(train_labels))

        # 优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=patience//3
        )
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"📅 Epoch {epoch+1}/{epochs}")
            logger.info(f"{'='*50}")

            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, class_weights
            )

            # 验证
            val_loss, val_acc, val_metrics = self.evaluate(model, val_loader, criterion)
            val_auc = val_metrics['auc']

            # 学习率调度
            scheduler.step(val_auc)

            # 记录历史
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'lr': optimizer.param_groups[0]['lr']
            }
            self.history.append(epoch_record)

            # 日志
            logger.info(f"📊 训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            logger.info(f"📊 验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
            logger.info(f"🎯 学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                logger.info(f"🏆 新的最佳模型! AUC: {val_auc:.4f}")
            else:
                patience_counter += 1
                logger.info(f"⏳ 等待改进: {patience_counter}/{patience}")

            # 早停
            if patience_counter >= patience:
                logger.info(f"⏹️ 早停触发! 最佳AUC: {best_val_auc:.4f}")
                break

        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # 最终评估
        logger.info("\n" + "="*50)
        logger.info("🧪 最终测试评估")
        logger.info("="*50)

        test_loss, test_acc, test_metrics = self.evaluate(model, test_loader, criterion)

        logger.info(f"📊 测试损失: {test_loss:.4f}")
        logger.info(f"📊 测试准确率: {test_acc:.2f}%")
        logger.info(f"📊 测试AUC: {test_metrics['auc']:.4f}")

        # 详细分类报告
        report = test_metrics['classification_report']
        logger.info(f"🎯 Phishing F1: {report['phish']['f1-score']:.4f}")
        logger.info(f"🎯 Benign F1: {report['benign']['f1-score']:.4f}")

        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"real_phishing_advanced_{timestamp}"

        model_path = self.model_dir / f"{model_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'feature_names': feature_names,
            'model_config': {
                'num_features': num_features,
                'model_type': 'real_phishing_advanced'
            },
            'class_weights': class_weights.cpu().numpy(),
            'history': self.history,
            'metrics': {
                'test_acc': test_acc,
                'test_auc': test_metrics['auc'],
                'test_report': report,
                'best_epoch': epoch + 1 - patience_counter
            }
        }, model_path)

        logger.info(f"💾 真实钓鱼网站模型已保存: {model_path}")

        # 保存训练历史
        history_path = self.model_dir / f"training_history_{model_name}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        # 绘制训练曲线
        self.plot_training_history(model_name)

        logger.info(f"🎉 真实钓鱼网站模型训练完成! 最佳验证AUC: {best_val_auc:.4f}")

        return TrainingArtifacts(
            model_state=model.state_dict(),
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            feature_names=tuple(feature_names),
            class_weights=class_weights.cpu().numpy(),
            metrics={
                'test_acc': test_acc,
                'test_auc': test_metrics['auc'],
                'test_report': report
            },
            history=self.history
        )

    def plot_training_history(self, model_name: str):
        """绘制训练历史"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            epochs = [h['epoch'] for h in self.history]

            # 损失曲线
            ax1.plot(epochs, [h['train_loss'] for h in self.history], 'b-', label='训练损失')
            ax1.plot(epochs, [h['val_loss'] for h in self.history], 'r-', label='验证损失')
            ax1.set_title('Loss Curve')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # 准确率曲线
            ax2.plot(epochs, [h['train_acc'] for h in self.history], 'b-', label='训练准确率')
            ax2.plot(epochs, [h['val_acc'] for h in self.history], 'r-', label='验证准确率')
            ax2.set_title('Accuracy Curve')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)

            # AUC曲线
            ax3.plot(epochs, [h['val_auc'] for h in self.history], 'g-', label='验证AUC')
            ax3.set_title('AUC Curve')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('AUC')
            ax3.legend()
            ax3.grid(True)

            # 学习率曲线
            ax4.plot(epochs, [h['lr'] for h in self.history], 'm-', label='学习率')
            ax4.set_title('Learning Rate')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()
            plot_path = self.model_dir / f"training_curves_{model_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"📈 训练曲线已保存: {plot_path}")

        except Exception as e:
            logger.warning(f"绘制训练曲线失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练真实钓鱼网站检测模型")
    parser.add_argument("--data-dir", type=str, default="data_real_phishing_v2",
                       help="数据集目录")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="学习率")
    parser.add_argument("--patience", type=int, default=15,
                       help="早停耐心值")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="计算设备")

    args = parser.parse_args()

    # 设置日志
    logger.add("logs/real_phishing_training_{time}.log", rotation="10 MB", level="INFO")

    # 开始训练
    trainer = RealPhishingTrainer(
        data_dir=args.data_dir,
        device=args.device
    )

    start_time = time.time()
    artifacts = trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )
    end_time = time.time()

    training_time = end_time - start_time
    logger.info(f"\n🎉 真实钓鱼网站模型训练完成!")
    logger.info(f"⏱️ 总训练时间: {training_time/60:.1f} 分钟")
    logger.info(f"📊 最终测试AUC: {artifacts.metrics['test_auc']:.4f}")
    logger.info(f"📊 最终测试准确率: {artifacts.metrics['test_acc']:.2f}%")

if __name__ == "__main__":
    main()