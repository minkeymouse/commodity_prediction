from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class RidgeAutoRegModel:
    target_name: str
    lags: List[int]
    model: Ridge

    def predict_row(self, lag_values_by_lag: Dict[int, float]) -> float:
        features = [lag_values_by_lag.get(l, 0.0) for l in self.lags]
        return float(self.model.predict(np.asarray(features, dtype=float).reshape(1, -1))[0])


def _build_lag_feature_frame(series: pd.Series, lags: List[int]) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.concat({f"lag_{l}": series.shift(l) for l in lags}, axis=1)
    y = series
    df = pd.concat([X, y.rename("y")], axis=1).dropna()
    return df.drop(columns=["y"]), df["y"]


def train_per_target_ridge(
    labels_wide: pd.DataFrame,
    lags: List[int] | None = None,
    alpha: float = 1.0,
) -> Dict[str, RidgeAutoRegModel]:
    """
    Train one Ridge autoregressive model per target using past labels as features.

    Args:
        labels_wide: index=date_id ascending, columns=target_0..target_423
        lags: list of integer lags to use as features (e.g., [1,2,3,4])
        alpha: ridge regularization strength

    Returns:
        Mapping of target name to RidgeAutoRegModel.
    """
    if lags is None:
        lags = [1, 2, 3, 4]

    models: Dict[str, RidgeAutoRegModel] = {}
    for target in labels_wide.columns:
        ser = labels_wide[target].astype(float)
        X, y = _build_lag_feature_frame(ser, lags)
        if len(X) < 20:
            # Not enough data; skip
            continue
        mdl = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
        mdl.fit(X.values, y.values)
        models[target] = RidgeAutoRegModel(target, lags, mdl)

    return models


def predict_wide(models: Dict[str, RidgeAutoRegModel], lag_frames_by_lag: Dict[int, pd.Series]) -> pd.Series:
    """
    Generate predictions for all targets for a single date using per-lag label rows.

    Args:
        models: mapping target->trained autoreg model
        lag_frames_by_lag: mapping lag k -> pd.Series of target values available for that lag on the current date

    Returns:
        pd.Series of predictions indexed by target names.
    """
    # Prepare a lookup: for each target, map lag->value
    predictions = {}
    for target, mdl in models.items():
        vals: Dict[int, float] = {}
        for l in mdl.lags:
            ser = lag_frames_by_lag.get(l)
            if ser is not None and target in ser.index and pd.notna(ser[target]):
                vals[l] = float(ser[target])
        predictions[target] = mdl.predict_row(vals)
    return pd.Series(predictions)


