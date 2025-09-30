from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


def train_fusion_dnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
) -> TrainingArtifacts:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("FusionDNN 训练使用设备: %s", device)

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
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_state = None
    best_metrics = {"acc": 0.0, "auc": 0.0, "report": {}}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics = _evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d/%d - loss %.4f - val_acc %.4f - val_auc %.4f",
            epoch,
            epochs,
            avg_loss,
            val_metrics["acc"],
            val_metrics["auc"],
        )
        if val_metrics["acc"] >= best_metrics["acc"]:
            best_metrics = val_metrics
            best_state = model.state_dict()

    assert best_state is not None, "Training failed to produce a model state."

    artifacts = TrainingArtifacts(
        model_state=best_state,
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        feature_names=feature_names,
        class_weights=class_weights_tensor.detach().cpu().numpy() if class_weights_tensor is not None else None,
        metrics=best_metrics,
    )
    return artifacts
