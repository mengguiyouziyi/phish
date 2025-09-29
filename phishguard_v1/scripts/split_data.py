from __future__ import annotations
import argparse, pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    # 按域名去重后再分层（简化处理：按URL去重）
    df = df.drop_duplicates(subset=["url"])
    train_df, test_df = train_test_split(df, test_size=args.test_size, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=args.val_size, stratify=train_df["label"], random_state=42)

    train_df.to_parquet(args.train, index=False)
    val_df.to_parquet(args.val, index=False)
    test_df.to_parquet(args.test, index=False)
    print(f"saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
