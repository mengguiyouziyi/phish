from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
import torch

from phishguard_v1.models.training_utils import TrainingArtifacts, train_fusion_dnn


def train(
    train_path: Path,
    val_path: Path,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    ckpt_path: Path = Path("artifacts/fusion_custom.pt"),
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
) -> None:
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    artifacts: TrainingArtifacts = train_fusion_dnn(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": artifacts.model_state,
        "input_features": len(artifacts.feature_names),
        "feature_names": list(artifacts.feature_names),
        "scaler_mean": artifacts.scaler_mean.tolist(),
        "scaler_scale": artifacts.scaler_scale.tolist(),
        "class_weights": artifacts.class_weights.tolist() if artifacts.class_weights is not None else None,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "val_metrics": artifacts.metrics,
    }
    torch.save(payload, ckpt_path)
    logger.info(
        "✅ 训练完成并保存模型 -> %s (val_acc=%.4f, val_auc=%.4f)",
        ckpt_path,
        artifacts.metrics.get("acc", 0.0),
        artifacts.metrics.get("auc", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 FusionDNN 模型")
    parser.add_argument("--train", type=Path, required=True, help="训练集 parquet 文件")
    parser.add_argument("--val", type=Path, required=True, help="验证集 parquet 文件")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ckpt", type=Path, default=Path("artifacts/fusion_custom.pt"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ckpt_path=args.ckpt,
        device=args.device,
    )


if __name__ == "__main__":
    main()
