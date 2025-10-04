from __future__ import annotations
import argparse
import json
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
    patience: int = 20,
    monitor: str = "auc",
    min_delta: float = 1e-4,
    clip_norm: Optional[float] = 5.0,
    use_amp: bool = False,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    seed: Optional[int] = 42,
    history_path: Optional[Path] = None,
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
        patience=patience,
        monitor=monitor,
        min_delta=min_delta,
        clip_grad_norm=clip_norm,
        use_amp=use_amp,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        seed=seed,
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
        "training_history": artifacts.history,
    }
    torch.save(payload, ckpt_path)
    logger.info(
        "✅ 训练完成并保存模型 -> {} (val_acc={:.4f}, val_auc={:.4f})",
        ckpt_path,
        artifacts.metrics.get("acc", 0.0),
        artifacts.metrics.get("auc", 0.0),
    )

    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", encoding="utf-8") as fp:
            json.dump(artifacts.history, fp, ensure_ascii=False, indent=2)
        logger.info("📈 训练曲线已写入 -> {}", history_path)


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
    parser.add_argument("--patience", type=int, default=20, help="早停耐心周期数")
    parser.add_argument("--monitor", type=str, choices=["auc", "acc", "f1"], default="auc", help="用于早停的监控指标")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="指标最小提升阈值")
    parser.add_argument("--clip-norm", type=float, default=5.0, help="梯度裁剪阈值，设置为<=0表示禁用")
    parser.add_argument("--use-amp", action="store_true", help="启用自动混合精度训练")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Plateau调度器的学习率缩放因子")
    parser.add_argument("--scheduler-patience", type=int, default=5, help="Plateau调度器耐心周期")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，设置为负值以禁用固定种子")
    parser.add_argument("--history-json", type=Path, default=None, help="可选: 将训练历程保存为 JSON")
    args = parser.parse_args()

    clip_norm = args.clip_norm if args.clip_norm is None or args.clip_norm > 0 else None
    seed = args.seed if args.seed is None or args.seed >= 0 else None

    train(
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ckpt_path=args.ckpt,
        device=args.device,
        patience=args.patience,
        monitor=args.monitor,
        min_delta=args.min_delta,
        clip_norm=clip_norm,
        use_amp=args.use_amp,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        seed=seed,
        history_path=args.history_json,
    )


if __name__ == "__main__":
    main()
