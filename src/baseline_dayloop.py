from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import polars as pl


def _read_lagged_labels_columns(label_dir: Path) -> List[str]:
    cols: List[str] = []
    for lag in [1, 2, 3, 4]:
        f = label_dir / f"test_labels_lag_{lag}.csv"
        if not f.exists():
            continue
        df = pl.read_csv(f, n_rows=1)
        # Drop metadata columns to isolate target names
        keep = [c for c in df.columns if c not in ("date_id", "label_date_id")]
        cols.extend(keep)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for c in cols:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _predict_from_lagged_series(lagged_values_by_target: Dict[str, float], columns: List[str]) -> pd.DataFrame:
    # Simple momentum baseline: predict today's targets using the most recently released label per target
    values = [lagged_values_by_target.get(c, 0.0) for c in columns]
    # Add tiny jitter to reduce ties
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 1e-6, size=len(values))
    arr = np.asarray(values, dtype=float) + jitter
    return pd.DataFrame([arr], columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal per-day loop baseline that writes submission.parquet")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to competition data directory (containing test.csv and lagged_test_labels)")
    parser.add_argument("--out", type=str, default="submission.parquet", help="Output parquet path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_path = data_dir / "test.csv"
    lag_dir = data_dir / "lagged_test_labels"
    # Fallback: if labels live next to `data_dir` (e.g., test at data/test.csv and labels at ./lagged_test_labels)
    if not lag_dir.exists():
        parent_candidate = data_dir.parent / "lagged_test_labels"
        if parent_candidate.exists():
            lag_dir = parent_candidate

    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")
    if not lag_dir.exists():
        raise FileNotFoundError(f"Missing {lag_dir}")

    test = pl.read_csv(test_path)
    date_ids = test["date_id"].unique(maintain_order=True).to_list()

    target_columns = _read_lagged_labels_columns(lag_dir)
    if not target_columns:
        raise RuntimeError("Could not infer target columns from lagged_test_labels")

    # Collect rows to mirror gateway behavior: row id first, then predictions
    rows = []
    # Preload lag files for per-day access
    lag_frames = {}
    for lag in [1, 2, 3, 4]:
        f = lag_dir / f"test_labels_lag_{lag}.csv"
        if f.exists():
            lag_frames[lag] = pl.read_csv(f)

    for d in date_ids:
        # Build per-day union of available lagged labels (drop metadata cols)
        per_day_cols: List[str] = []
        per_day_values: Dict[str, float] = {}
        for lag, df in lag_frames.items():
            row = df.filter(pl.col("date_id") == d)
            if row.height == 0:
                continue
            row = row.drop([c for c in ["date_id", "label_date_id"] if c in row.columns])
            # Convert single-row polars frame to dict
            pdf = row.to_pandas()
            for c in pdf.columns:
                val = float(pdf.iloc[0][c]) if pd.notna(pdf.iloc[0][c]) else 0.0
                per_day_values[c] = val
                per_day_cols.append(c)

        # If no per-day values (shouldn't happen), fall back to global columns
        cols_for_pred = list(dict.fromkeys(per_day_cols)) or target_columns
        pred_row = _predict_from_lagged_series(per_day_values, cols_for_pred)
        pred_row.insert(0, "date_id", d)
        rows.append(pred_row)

    submission = pd.concat(rows, ignore_index=True)
    # Column order: date_id then targets
    ordered_cols = ["date_id"] + [c for c in submission.columns if c != "date_id"]
    submission = submission[ordered_cols]
    submission.to_parquet(args.out, index=False)

    print(f"Wrote {args.out} with shape {submission.shape}")


if __name__ == "__main__":
    main()


