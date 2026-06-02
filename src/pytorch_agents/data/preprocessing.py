

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_sliding_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int = 30,
    target_column: str | None = "Target_5d_Return",
) -> tuple[np.ndarray, np.ndarray | None, pd.DatetimeIndex, np.ndarray]:

    all_windows = []
    all_targets = []
    all_dates = []
    all_companies = []

    groups = df.groupby("Company") if "Company" in df.columns else [(None, df)]

    for company, group in groups:
        data = group[feature_columns].values
        n_samples = len(data) - seq_len + 1

        if n_samples <= 0:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(data, (seq_len, len(feature_columns)))
        windows = windows.squeeze(axis=1)

        dates = group.index[seq_len - 1:]

        targets = None
        if target_column is not None and target_column in group.columns:
            targets = group[target_column].values[seq_len - 1:]

        all_windows.append(windows)
        all_dates.append(dates)

        comps = np.full(len(dates), company) if company is not None else np.full(len(dates), "UNKNOWN")
        all_companies.append(comps)

        if targets is not None:
            all_targets.append(targets)

    if not all_windows:
        raise ValueError(f"No valid windows created. Ensure data length >= seq_len ({seq_len})")

    windows_out = np.concatenate(all_windows, axis=0)
    dates_out = pd.DatetimeIndex(np.concatenate(all_dates))
    targets_out = np.concatenate(all_targets, axis=0) if all_targets else None
    companies_out = np.concatenate(all_companies, axis=0)

    return windows_out, targets_out, dates_out, companies_out


def fit_scaler(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_columns].values)
    return scaler


def apply_scaler(
    windows: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    N, T, F = windows.shape
    flat = windows.reshape(-1, F)
    scaled = scaler.transform(flat)
    return scaled.reshape(N, T, F)


def align_news_to_dates(
    news_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    target_companies: np.ndarray | None = None,
    text_column: str = "Full_Text",
    max_articles: int = 1,
) -> list[str]:

    df = news_df.copy()

    if "Parsed_Date" not in df.columns:
        df["Parsed_Date"] = pd.to_datetime(
            df["Date"], format="mixed", utc=True
        ).dt.tz_localize(None)

    if text_column not in df.columns and "Article_title" in df.columns:
        df["Full_Text"] = (
            df["Article_title"].fillna("") + " " + df["Article"].fillna("")
        )

    df = df.dropna(subset=[text_column])
    df = df.sort_values("Parsed_Date")

    aligned_texts: list[str] = []

    news_by_company = {
        symbol.upper(): group
        for symbol, group in df.groupby(df["Stock_symbol"].astype(str).str.upper())
    } if "Stock_symbol" in df.columns else {}

    for i, dt in enumerate(target_dates):
        comp = str(target_companies[i]).upper() if target_companies is not None else None

        candidates = df
        if comp and comp in news_by_company:
            candidates = news_by_company[comp]
        elif comp:
            candidates = pd.DataFrame(columns=df.columns)

        candidates = candidates[candidates["Parsed_Date"] <= pd.Timestamp(dt)]

        if candidates.empty:
            aligned_texts.append("")
        else:
            recent = candidates.tail(max_articles)
            aligned_texts.append(" ".join(recent[text_column].tolist()))

    return aligned_texts
