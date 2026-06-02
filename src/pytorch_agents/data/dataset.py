"""
PyTorch Dataset that produces aligned (window, text, label) triplets
for the multi-agent ensemble.
"""

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
    """
    Yields tuples of:
        - ``window``  : ``Tensor[seq_len, n_features]``  — scaled OHLCV window
        - ``snapshot`` : ``Tensor[n_features]``           — last row of window (for RF)
        - ``text``    : ``str``                           — aligned news article text
        - ``label``   : ``int``                           — 0/1/2 (Sell/Hold/Buy)

    Parameters
    ----------
    tech_df : pd.DataFrame
        Technical data (date-indexed, ascending).
    news_df : pd.DataFrame
        Labeled news data.
    feature_columns : list[str]
        Feature columns to extract from ``tech_df``.
    seq_len : int
        Sliding window length.
    scaler : StandardScaler or None
        If ``None``, a new scaler is fitted on ``tech_df``.
    label_cfg : dict
        Keyword arguments forwarded to ``generate_labels``.
    stock_symbol : str or None
        Filter news by ticker.  ``None`` uses all articles.
    """

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

        # --- Build sliding windows ---
        windows, targets, dates = create_sliding_windows(
            tech_df, feature_columns, seq_len=seq_len
        )

        # --- Scale features ---
        if scaler is None:
            self.scaler = fit_scaler(tech_df, feature_columns)
        else:
            self.scaler = scaler
        windows = apply_scaler(windows, self.scaler)

        # --- Generate labels from raw returns ---
        if targets is not None:
            target_series = pd.Series(targets, index=dates)
            labels = generate_labels(target_series, **label_cfg)
            # Keep only rows where we have a valid label
            valid_mask = labels.index.isin(dates)
            labels = labels[valid_mask]
        else:
            labels = pd.Series(dtype=int)

        # Align arrays to valid label indices
        valid_positions = np.array(
            [i for i, d in enumerate(dates) if d in labels.index]
        )

        if len(valid_positions) > 0:
            self.windows = windows[valid_positions]
            self.labels = labels.values
            valid_dates = dates[valid_positions]
        else:
            self.windows = windows
            self.labels = np.full(len(windows), 1, dtype=int)  # default Hold
            valid_dates = dates

        # --- Snapshots: last timestep of each window (for RF wrapper) ---
        # Apply inverse scaling so the RF gets raw features
        last_steps_scaled = self.windows[:, -1, :]  # [N, n_features]
        # Inverse transform to get original scale for sklearn models
        self.snapshots = self.scaler.inverse_transform(last_steps_scaled)

        # --- Align news articles ---
        self.texts = align_news_to_dates(
            news_df, valid_dates, stock_symbol=stock_symbol
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
    """
    Custom collate that handles the mixed types (tensors + strings).

    Returns
    -------
    windows   : Tensor[batch, seq_len, n_features]
    snapshots : Tensor[batch, n_features]
    texts     : list[str] of length batch
    labels    : Tensor[batch] (long)
    """
    windows, snapshots, texts, labels = zip(*batch)
    windows = torch.stack(windows)
    snapshots = torch.stack(snapshots)
    labels = torch.tensor(labels, dtype=torch.long)
    return windows, snapshots, list(texts), labels
