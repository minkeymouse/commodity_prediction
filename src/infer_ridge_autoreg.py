from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import polars as pl

from models.ridge_autoreg import predict_wide


def _load_lag_series_for_date(lag_dir: Path, date_id: int) -> Dict[int, pd.Series]:
    by_lag: Dict[int, pd.Series] = {}
    for lag in [1, 2, 3, 4]:
        f = lag_dir / f"test_labels_lag_{lag}.csv"
        if not f.exists():
            continue
        df = pl.read_csv(f)
        row = df.filter(pl.col("date_id") == date_id)
        if row.height == 0:
            continue
        row = row.drop([c for c in ["date_id", "label_date_id"] if c in row.columns])
        by_lag[lag] = row.to_pandas().iloc[0]
    return by_lag


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer per-day predictions using trained Ridge autoreg models")
    parser.add_argument("--models", type=str, required=True, help="Path to pickle with trained models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test.csv and lagged_test_labels")
    parser.add_argument("--out", type=str, required=True, help="Output submission.parquet path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    lag_dir = data_dir / "lagged_test_labels"
    if not lag_dir.exists():
        raise FileNotFoundError(f"Missing {lag_dir}")

    models = pd.read_pickle(args.models)
    test = pl.read_csv(data_dir / "test.csv")
    date_ids = test["date_id"].unique(maintain_order=True).to_list()

    rows = []
    for d in date_ids:
        by_lag = _load_lag_series_for_date(lag_dir, d)
        preds = predict_wide(models, by_lag)
        row = preds.to_frame().T
        row.insert(0, "date_id", d)
        rows.append(row)

    submission = pd.concat(rows, ignore_index=True)
    ordered_cols = ["date_id"] + [c for c in submission.columns if c != "date_id"]
    submission = submission[ordered_cols]
    submission.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with shape {submission.shape}")


if __name__ == "__main__":
    main()


