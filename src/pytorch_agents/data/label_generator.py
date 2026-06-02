

import numpy as np
import pandas as pd


CLASS_MAP = {"Sell": 0, "Hold": 1, "Buy": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def generate_labels(
    returns: pd.Series,
    strategy: str = "percentile",
    buy_percentile: float = 67,
    sell_percentile: float = 33,
    buy_threshold: float = 0.01,
    sell_threshold: float = -0.01,
) -> pd.Series:
    returns = returns.dropna()

    if strategy == "percentile":
        buy_thresh = np.percentile(returns, buy_percentile)
        sell_thresh = np.percentile(returns, sell_percentile)
    elif strategy == "fixed":
        buy_thresh = buy_threshold
        sell_thresh = sell_threshold
    else:
        raise ValueError(f"Unknown label strategy: {strategy!r}")

    labels = pd.Series(CLASS_MAP["Hold"], index=returns.index, dtype=int)
    labels[returns > buy_thresh] = CLASS_MAP["Buy"]
    labels[returns < sell_thresh] = CLASS_MAP["Sell"]

    return labels
