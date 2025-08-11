from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.ridge_autoreg import train_per_target_ridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-target Ridge autoregressive models from train_labels.csv")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train_labels.csv")
    parser.add_argument("--out", type=str, required=True, help="Path to save models as pickle")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    parser.add_argument("--lags", type=int, nargs="*", default=[1, 2, 3, 4], help="Lags to use as features")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_labels_path = data_dir / "train_labels.csv"
    if not train_labels_path.exists():
        raise FileNotFoundError(f"Missing {train_labels_path}")

    labels = pd.read_csv(train_labels_path)
    if "date_id" not in labels.columns:
        raise ValueError("train_labels.csv missing date_id column")
    labels = labels.set_index("date_id").sort_index()

    models = train_per_target_ridge(labels, lags=args.lags, alpha=args.alpha)
    pd.to_pickle(models, args.out)
    print(f"Saved {len(models)} models to {args.out}")


if __name__ == "__main__":
    main()


