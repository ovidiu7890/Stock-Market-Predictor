

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .label_generator import generate_labels
from .preprocessing import (
    align_news_to_dates,
    apply_scaler,
    create_sliding_windows,
    fit_scaler,
)


class StockDataset(Dataset):


    def __init__(
        self,
        tech_df: pd.DataFrame,
        news_df: pd.DataFrame,
        feature_columns: list[str],
        seq_len: int = 30,
        scaler=None,
        label_cfg: dict | None = None,
        stock_symbol: str | None = None,
    ):
        label_cfg = label_cfg or {}

        windows, targets, dates, companies = create_sliding_windows(
            tech_df, feature_columns, seq_len=seq_len
        )

        if scaler is None:
            self.scaler = fit_scaler(tech_df, feature_columns)
        else:
            self.scaler = scaler
        windows = apply_scaler(windows, self.scaler)

        if targets is not None:
            target_series = pd.Series(targets, index=dates)
            labels = generate_labels(target_series, **label_cfg)
            valid_mask = labels.index.isin(dates)
            labels = labels[valid_mask]
        else:
            labels = pd.Series(dtype=int)

        valid_positions = np.array(
            [i for i, d in enumerate(dates) if d in labels.index]
        )

        if len(valid_positions) > 0:
            self.windows = windows[valid_positions]
            self.labels = labels.values
            valid_dates = dates[valid_positions]
            valid_companies = companies[valid_positions]
        else:
            self.windows = windows
            self.labels = np.full(len(windows), 1, dtype=int)
            valid_dates = dates
            valid_companies = companies

        last_steps_scaled = self.windows[:, -1, :]
        self.snapshots = self.scaler.inverse_transform(last_steps_scaled)

        self.texts = align_news_to_dates(
            news_df, valid_dates, target_companies=valid_companies
        )

        assert len(self.windows) == len(self.labels) == len(self.texts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = torch.tensor(self.windows[idx], dtype=torch.float32)
        snapshot = torch.tensor(self.snapshots[idx], dtype=torch.float32)
        text = self.texts[idx]
        label = int(self.labels[idx])
        return window, snapshot, text, label


def collate_fn(batch):

    windows, snapshots, texts, labels = zip(*batch)
    windows = torch.stack(windows)
    snapshots = torch.stack(snapshots)
    labels = torch.tensor(labels, dtype=torch.long)
    return windows, snapshots, list(texts), labels
