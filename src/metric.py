from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def daily_spearman_sharpe(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """
    Compute the IC‑Sharpe style metric used in the competition.

    - For each date_id (row index), compute Spearman rank correlation across columns.
    - Return mean of daily correlations divided by their sample standard deviation.

    Args:
        true_df: DataFrame with index = date_id, columns = target_0..target_423, values = true labels
        pred_df: DataFrame with same shape/index/columns, values = predictions

    Returns:
        float IC‑Sharpe value.
    """
    # Align indices/columns
    common_index = true_df.index.intersection(pred_df.index)
    common_columns = true_df.columns.intersection(pred_df.columns)
    if len(common_index) == 0 or len(common_columns) == 0:
        return float("nan")

    daily_values: list[float] = []
    for date_id in common_index:
        t = true_df.loc[date_id, common_columns].to_numpy()
        p = pred_df.loc[date_id, common_columns].to_numpy()
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() >= 3:
            r, _ = spearmanr(t[mask], p[mask])
            daily_values.append(r)

    if len(daily_values) < 2:
        return float("nan")

    daily_arr = np.array(daily_values, dtype=float)
    mean_ic = float(np.nanmean(daily_arr))
    std_ic = float(np.nanstd(daily_arr, ddof=1))
    if std_ic == 0.0 or not np.isfinite(std_ic):
        return float("nan")
    return mean_ic / std_ic


__all__ = ["daily_spearman_sharpe"]


