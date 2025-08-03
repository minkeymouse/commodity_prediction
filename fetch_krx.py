"""Fetch daily KRX index data and store as CSV files.

This script is intentionally kept separate from the preprocessing pipeline so
that raw index data can be gathered once and the heavy lifting is not exposed
when sharing research code.  The downloaded files follow the same column schema
as the bitcoin and foreign‐exchange data used elsewhere in the project:

``datetime,currency,market,index_name,platform,open,high,low,close,volume,change_pct``

The script relies on :mod:`pykrx` which must be installed in the running
environment.  The data are fetched using the convenience wrappers defined in
``krx.indices``.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Iterable

import pandas as pd

from krx.indices import fetch_index_ohlcv


# Default indices to download.  Additional tickers/names understood by
# ``pykrx`` may be supplied if desired.
DEFAULT_INDICES = ["KOSPI", "KOSDAQ", "KOSPI200", "KOSDAQ150"]


def _prepare_dataframe(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Normalise raw ``pykrx`` output to the project's CSV schema."""

    # ``pykrx`` may return either Korean or English column names depending on
    # the version installed.  Make the renaming resilient to both by mapping any
    # known variants to our canonical schema.
    df = df.rename(
        columns={
            "시가": "open",
            "Open": "open",
            "고가": "high",
            "High": "high",
            "저가": "low",
            "Low": "low",
            "종가": "close",
            "Close": "close",
            "거래량": "volume",
            "Volume": "volume",
        }
    )

    # ``pykrx`` does not provide day‑to‑day change percentages for indices, so
    # compute it from the closing prices.
    df["change_pct"] = df["close"].pct_change() * 100

    # The index column name varies (e.g. "날짜" in some locales); capture it
    # before resetting so we can rename it consistently.
    date_col = df.index.name or "index"
    df = df.reset_index().rename(columns={date_col: "datetime"})
    df["currency"] = "KRW"
    df["market"] = "KRX"
    df["index_name"] = index_name
    df["platform"] = "KRX"

    return df[
        [
            "datetime",
            "currency",
            "market",
            "index_name",
            "platform",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "change_pct",
        ]
    ]


def fetch_and_save(indices: Iterable[str], start: str, end: str, out_dir: str) -> None:
    """Fetch ``indices`` between ``start`` and ``end`` and write to ``out_dir``."""

    os.makedirs(out_dir, exist_ok=True)
    for idx in indices:
        df = fetch_index_ohlcv(idx, start, end)
        df = _prepare_dataframe(df, idx)
        out_path = os.path.join(out_dir, f"{idx}.csv")
        df.to_csv(out_path, index=False)
        print(f"✔ Saved {idx} ({len(df)} rows) to {out_path}")


def main() -> None:
    start = "2010-01-01"
    end = date.today().strftime("%Y-%m-%d")
    out_dir = os.path.join("data", "krx_data")
    fetch_and_save(DEFAULT_INDICES, start, end, out_dir)


if __name__ == "__main__":
    main()

