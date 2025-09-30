from __future__ import annotations
import argparse, os, pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader
from .dataset import ParquetDataset
from .fusion_model import FusionDNN
from loguru import logger

def train(args):
    train_ds = ParquetDataset(args.train)
    val_ds = ParquetDataset(args.val)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu"
    model = FusionDNN(num_features=train_ds.X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / max(1,total)
        logger.info(f"epoch {epoch+1}/{args.epochs} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            os.makedirs("artifacts", exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "num_features": train_ds.X.shape[1]}, "artifacts/fusion.pt")
            logger.info("saved artifacts/fusion.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()
    train(args)
