from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Tuple
import contextlib

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from phishguard_v1.models.dataset import build_feature_matrix
from phishguard_v1.models.fusion_model import FusionDNN


@dataclass
class TrainingArtifacts:
    model_state: Dict[str, Any]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    feature_names: Tuple[str, ...]
    class_weights: Optional[np.ndarray]
    metrics: Dict[str, Any]
    history: List[Dict[str, float]]


def _build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, StandardScaler, Tuple[str, ...], np.ndarray, np.ndarray]:
    X_train, feature_names = build_feature_matrix(train_df)
    X_val, _ = build_feature_matrix(val_df)
    y_train = train_df["label"].astype(int).values
    y_val = val_df["label"].astype(int).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, tuple(feature_names), y_train, y_val


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())

    acc = correct / max(total, 1)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["benign", "phish"],
        zero_division=0,
        output_dict=True,
    )
    return {"acc": acc, "auc": auc, "report": report}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fusion_dnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    patience: int = 20,
    monitor: str = "auc",
    min_delta: float = 1e-4,
    clip_grad_norm: Optional[float] = 5.0,
    use_amp: bool = False,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    seed: Optional[int] = 42,
) -> TrainingArtifacts:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("FusionDNN 训练使用设备: {}", device)

    if seed is not None:
        _set_seed(seed)

    train_loader, val_loader, scaler, feature_names, y_train, _ = _build_loaders(
        train_df, val_df, batch_size
    )

    model = FusionDNN(num_features=len(feature_names)).to(device)

    class_weights_tensor: Optional[torch.Tensor] = None
    unique, counts = np.unique(y_train, return_counts=True)
    if len(unique) == 2 and np.all(counts):
        total = counts.sum()
        weights = [total / (2 * c) for c in counts]
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=max(min(scheduler_factor, 0.99), 1e-3),
        patience=max(int(scheduler_patience), 1),
        min_lr=1e-7,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_state = None
    best_metrics = {"acc": 0.0, "auc": 0.0, "report": {}}
    best_score = float("-inf")
    epochs_without_improve = 0
    monitor = monitor.lower()
    history: List[Dict[str, float]] = []

    use_cuda_amp = (
        use_amp
        and isinstance(device, str)
        and device.startswith("cuda")
        and torch.cuda.is_available()
    )
    if use_cuda_amp:
        from torch.cuda import amp as cuda_amp

        amp_scaler = cuda_amp.GradScaler(enabled=True)
        autocast_ctx = cuda_amp.autocast
    else:
        amp_scaler = None
        autocast_ctx = contextlib.nullcontext

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            with autocast_ctx():
                logits = model(xb)
                loss = criterion(logits, yb)

            if amp_scaler is None:
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            else:
                amp_scaler.scale(loss).backward()
                if clip_grad_norm is not None:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics = _evaluate(model, val_loader, device)
        phish_report = val_metrics.get("report", {}).get("phish", {})
        val_f1 = float(phish_report.get("f1-score", 0.0))
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(avg_loss),
                "val_acc": float(val_metrics["acc"]),
                "val_auc": float(val_metrics["auc"]),
                "val_f1": val_f1,
                "lr": float(current_lr),
            }
        )

        logger.info(
            "Epoch {}/{} - loss {:.4f} - val_acc {:.4f} - val_auc {:.4f} - val_f1 {:.4f} - lr {:.2e}",
            epoch,
            epochs,
            avg_loss,
            val_metrics["acc"],
            val_metrics["auc"],
            val_f1,
            current_lr,
        )

        monitor_value = {
            "acc": val_metrics["acc"],
            "auc": val_metrics["auc"],
            "f1": val_f1,
        }.get(monitor, val_metrics["auc"])
        if not np.isfinite(monitor_value):
            monitor_value = float(val_metrics["acc"])

        scheduler.step(monitor_value)

        if monitor_value > best_score + min_delta:
            best_metrics = val_metrics
            best_state = model.state_dict()
            best_score = monitor_value
            epochs_without_improve = 0
            logger.info(
                "✅ 指标 {} 达到新高: {:.4f} (epoch {})",
                monitor,
                monitor_value,
                epoch,
            )
        else:
            epochs_without_improve += 1
            if patience > 0 and epochs_without_improve >= patience:
                logger.info(
                    "⏹️ 早停触发: 指标 {} 连续 {} 个周期未提升",
                    monitor,
                    patience,
                )
                break

    assert best_state is not None, "Training failed to produce a model state."

    artifacts = TrainingArtifacts(
        model_state=best_state,
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        feature_names=feature_names,
        class_weights=class_weights_tensor.detach().cpu().numpy() if class_weights_tensor is not None else None,
        metrics=best_metrics,
        history=history,
    )
    return artifacts
