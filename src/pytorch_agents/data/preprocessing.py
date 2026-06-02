"""
Preprocessing utilities — sliding-window construction, feature scaling,
and temporal alignment between technical and news data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_sliding_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int = 30,
    target_column: str | None = "Target_5d_Return",
) -> tuple[np.ndarray, np.ndarray | None, pd.DatetimeIndex]:
    """
    Build overlapping sliding windows from a time-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date (ascending) with a DatetimeIndex.
    feature_columns : list[str]
        Column names to include in each window.
    seq_len : int
        Number of timesteps per window.
    target_column : str or None
        If provided, extract the target value at the *last* timestep of each
        window.  Pass ``None`` to skip (inference mode).

    Returns
    -------
    windows : np.ndarray, shape ``[N, seq_len, n_features]``
    targets : np.ndarray, shape ``[N]``  (or ``None``)
    dates   : pd.DatetimeIndex of length ``N`` — the date of the last
              timestep in each window (used for alignment).
    """
    data = df[feature_columns].values
    n_samples = len(data) - seq_len + 1

    if n_samples <= 0:
        raise ValueError(
            f"DataFrame has {len(data)} rows but seq_len={seq_len}; "
            "need at least seq_len rows."
        )

    windows = np.lib.stride_tricks.sliding_window_view(data, (seq_len, len(feature_columns)))
    windows = windows.squeeze(axis=1)           # [N, seq_len, n_features]

    dates = df.index[seq_len - 1:]              # last date of each window

    targets = None
    if target_column is not None and target_column in df.columns:
        targets = df[target_column].values[seq_len - 1:]

    return windows, targets, dates


def fit_scaler(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:
    """Fit a StandardScaler on the training portion of the data."""
    scaler = StandardScaler()
    scaler.fit(df[feature_columns].values)
    return scaler


def apply_scaler(
    windows: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply a fitted StandardScaler to sliding windows.

    Parameters
    ----------
    windows : np.ndarray, shape ``[N, seq_len, n_features]``
    scaler  : fitted StandardScaler

    Returns
    -------
    np.ndarray of same shape, scaled.
    """
    N, T, F = windows.shape
    flat = windows.reshape(-1, F)
    scaled = scaler.transform(flat)
    return scaled.reshape(N, T, F)


def align_news_to_dates(
    news_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    stock_symbol: str | None = None,
    text_column: str = "Full_Text",
    max_articles: int = 1,
) -> list[str]:
    """
    For each target date, find the nearest preceding news article text.

    This implements the plan's recommendation:
      - Training:  exact date match (Option A)
      - Inference:  latest available article (Option B)

    Since we call this once per date, both modes reduce to the same logic:
    "find the most recent article on or before this date".

    Parameters
    ----------
    news_df : pd.DataFrame
        Must contain a parsed datetime column ``'Parsed_Date'``,
        ``'Stock_symbol'``, and the ``text_column``.
    target_dates : pd.DatetimeIndex
        Dates to align articles to.
    stock_symbol : str or None
        If provided, filter articles for this ticker only.
    text_column : str
        Column containing the text to extract.
    max_articles : int
        Number of articles to concatenate per date (currently 1).

    Returns
    -------
    list[str] of length ``len(target_dates)``
    """
    df = news_df.copy()

    # Ensure datetime
    if "Parsed_Date" not in df.columns:
        df["Parsed_Date"] = pd.to_datetime(
            df["Date"], format="mixed", utc=True
        ).dt.tz_localize(None)

    # Build full text if not present
    if text_column not in df.columns and "Article_title" in df.columns:
        df["Full_Text"] = (
            df["Article_title"].fillna("") + " " + df["Article"].fillna("")
        )

    if stock_symbol is not None:
        df = df[df["Stock_symbol"].str.upper() == stock_symbol.upper()]

    df = df.dropna(subset=[text_column])
    df = df.sort_values("Parsed_Date")

    aligned_texts: list[str] = []
    for dt in target_dates:
        # Articles on or before this date
        candidates = df[df["Parsed_Date"] <= pd.Timestamp(dt)]
        if candidates.empty:
            aligned_texts.append("")
        else:
            # Take the most recent article(s)
            recent = candidates.tail(max_articles)
            aligned_texts.append(" ".join(recent[text_column].tolist()))

    return aligned_texts
