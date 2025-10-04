from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import torch

from phishguard_v1.models.dataset import build_feature_matrix
from phishguard_v1.models.fusion_model import FusionDNN
from phishguard_v1.models.training_utils import TrainingArtifacts, train_fusion_dnn


@dataclass
class ActiveLearningStats:
    round_idx: int
    labeled_size: int
    val_acc: float
    val_auc: float
    newly_labeled: int
    pool_remaining: int


@dataclass
class DALWFROptions:
    rounds: int = 5
    query_size: int = 128
    initial_labeled: int = 512
    val_fraction: float = 0.2
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: Optional[str] = None
    patience: int = 10
    monitor: str = "auc"
    min_delta: float = 1e-4
    clip_norm: Optional[float] = 5.0
    use_amp: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    seed: Optional[int] = 42


def _predict_uncertainty(model: FusionDNN, scaler_mean: np.ndarray, scaler_scale: np.ndarray, feature_names: List[str], pool_df: pd.DataFrame, device: str) -> np.ndarray:
    if pool_df.empty:
        return np.array([])
    X_pool, _ = build_feature_matrix(pool_df)
    safe_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    X_pool = (X_pool - scaler_mean) / safe_scale
    tensor = torch.tensor(X_pool, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    entropy = -probs * np.log(probs) - (1 - probs) * np.log(1 - probs)
    return entropy


def run_active_learning(
    labeled_path: Path,
    unlabeled_path: Path,
    output_dir: Path,
    options: DALWFROptions,
) -> List[ActiveLearningStats]:
    labeled_df = pd.read_parquet(labeled_path)
    unlabeled_df = pd.read_parquet(unlabeled_path)

    if len(labeled_df) < options.initial_labeled:
        raise ValueError("初始标注数据不足，请增大 labeled 集合或调小 initial_labeled")

    # 随机下采样初始标注集
    initial_df, remainder_df = train_test_split(
        labeled_df,
        train_size=options.initial_labeled,
        stratify=labeled_df["label"],
        random_state=42,
    )
    unlabeled_df = pd.concat([unlabeled_df, remainder_df], ignore_index=True)
    labeled_df = initial_df.reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: List[ActiveLearningStats] = []

    for round_idx in range(1, options.rounds + 1):
        logger.info(
            "[Round {}/{}] 标注样本数={} , 未标注样本={}",
            round_idx,
            options.rounds,
            len(labeled_df),
            len(unlabeled_df),
        )
        train_df, val_df = train_test_split(
            labeled_df,
            test_size=options.val_fraction,
            stratify=labeled_df["label"],
            random_state=round_idx * 17,
        )

        artifacts: TrainingArtifacts = train_fusion_dnn(
            train_df=train_df,
            val_df=val_df,
            epochs=options.epochs,
            batch_size=options.batch_size,
            lr=options.lr,
            weight_decay=options.weight_decay,
            device=options.device,
            patience=options.patience,
            monitor=options.monitor,
            min_delta=options.min_delta,
            clip_grad_norm=options.clip_norm,
            use_amp=options.use_amp,
            scheduler_factor=options.scheduler_factor,
            scheduler_patience=options.scheduler_patience,
            seed=options.seed,
        )

        stats.append(
            ActiveLearningStats(
                round_idx=round_idx,
                labeled_size=len(labeled_df),
                val_acc=float(artifacts.metrics.get("acc", 0.0)),
                val_auc=float(artifacts.metrics.get("auc", 0.0)),
                newly_labeled=0,
                pool_remaining=len(unlabeled_df),
            )
        )

        if unlabeled_df.empty:
            logger.info("未标注样本耗尽，提前结束主动学习")
            break

        model = FusionDNN(num_features=len(artifacts.feature_names))
        model.load_state_dict(artifacts.model_state)
        model.to(options.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        uncertainties = _predict_uncertainty(
            model,
            artifacts.scaler_mean,
            artifacts.scaler_scale,
            list(artifacts.feature_names),
            unlabeled_df,
            options.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        if uncertainties.size == 0:
            logger.warning("未能计算不确定性，跳过采样")
            break

        query_size = min(options.query_size, len(unlabeled_df))
        query_indices = np.argsort(-uncertainties)[:query_size]
        queried = unlabeled_df.iloc[query_indices]

        labeled_df = pd.concat([labeled_df, queried], ignore_index=True)
        unlabeled_df = unlabeled_df.drop(unlabeled_df.index[query_indices]).reset_index(drop=True)

        stats[-1].newly_labeled = len(queried)
        stats[-1].pool_remaining = len(unlabeled_df)

        ckpt_path = output_dir / f"dalwfr_round_{round_idx}.pt"
        torch.save(
            {
                "model_state_dict": artifacts.model_state,
                "feature_names": list(artifacts.feature_names),
                "scaler_mean": artifacts.scaler_mean.tolist(),
                "scaler_scale": artifacts.scaler_scale.tolist(),
                "metrics": artifacts.metrics,
                "labeled_size": len(labeled_df),
            },
            ckpt_path,
        )
        logger.info("已保存轮次模型 -> {}", ckpt_path)

        labeled_df.to_parquet(output_dir / f"labeled_round_{round_idx}.parquet", index=False)
        unlabeled_df.to_parquet(output_dir / f"pool_round_{round_idx}.parquet", index=False)

    stats_path = output_dir / "dalwfr_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in stats], f, ensure_ascii=False, indent=2)
    logger.info("主动学习完成，统计信息写入 {}", stats_path)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 DALWFR 主动学习循环")
    parser.add_argument("--labeled", type=Path, required=True, help="已标注数据 parquet")
    parser.add_argument("--unlabeled", type=Path, required=True, help="未标注数据 parquet")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dalwfr"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--query-size", type=int, default=256)
    parser.add_argument("--initial", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--monitor", type=str, choices=["auc", "acc", "f1"], default="auc")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    options = DALWFROptions(
        rounds=args.rounds,
        query_size=args.query_size,
        initial_labeled=args.initial,
        val_fraction=args.val_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        monitor=args.monitor,
        min_delta=args.min_delta,
        clip_norm=args.clip_norm if args.clip_norm > 0 else None,
        use_amp=args.use_amp,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        seed=args.seed if args.seed >= 0 else None,
    )

    run_active_learning(
        labeled_path=args.labeled,
        unlabeled_path=args.unlabeled,
        output_dir=args.output,
        options=options,
    )


if __name__ == "__main__":
    main()
