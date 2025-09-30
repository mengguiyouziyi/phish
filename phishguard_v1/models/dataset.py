from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from phishguard_v1.features.feature_engineering import (
    FEATURE_COLUMNS,
    prepare_features_dataframe,
)

NUMERIC_COLS = FEATURE_COLUMNS


def build_feature_matrix(df: pd.DataFrame, extra_binary_cols=None):
    prepared = prepare_features_dataframe(df.copy())
    cols = FEATURE_COLUMNS.copy()
    if extra_binary_cols:
        for name in extra_binary_cols:
            if name not in cols:
                cols.append(name)
            if name not in prepared.columns:
                prepared[name] = 0.0
    X = prepared[cols].astype(float).values
    return X, cols


class ParquetDataset(Dataset):
    def __init__(self, path, label_col="label", extra_binary_cols=None):
        df = pd.read_parquet(path)
        self.labels = df[label_col].astype(int).values
        self.X, self.cols = build_feature_matrix(df, extra_binary_cols=extra_binary_cols)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y
