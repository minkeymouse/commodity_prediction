from __future__ import annotations

import glob
import os
from datetime import date
from typing import Iterable, List

import pandas as pd

from krx.indices import fetch_index_ohlcv

# Default KRX indices to include in the final dataset
DEFAULT_INDICES = ["KOSPI", "KOSDAQ", "KOSPI200", "KOSDAQ150"]


def _prepare_krx_dataframe(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Normalise raw ``pykrx`` output to the project's CSV schema."""
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
    df["change_pct"] = df["close"].pct_change() * 100
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


def fetch_krx(indices: Iterable[str], start: str, end: str) -> List[pd.DataFrame]:
    """Fetch OHLCV data for ``indices`` between ``start`` and ``end``."""
    dfs: List[pd.DataFrame] = []
    for idx in indices:
        df = fetch_index_ohlcv(idx, start, end)
        dfs.append(_prepare_krx_dataframe(df, idx))
    return dfs


# Loaders for existing Bitcoin and FX datasets

def load_aggregated(btc_data_dir: str):
    agg_dfs = []
    for path in glob.glob(os.path.join(btc_data_dir, "bitcoinity_data_*.csv")):
        currency = (
            os.path.basename(path)
            .replace("bitcoinity_data_", "")
            .replace(".csv", "")
        )
        df = pd.read_csv(path, parse_dates=["Time"])
        df = df.melt(id_vars=["Time"], var_name="market", value_name="close").rename(
            columns={"Time": "datetime"}
        )
        df["platform"] = "bitcoinity"
        for col in ["open", "high", "low", "volume", "change_pct"]:
            df[col] = pd.NA
        df["currency"] = currency
        df["index_name"] = pd.NA
        agg_dfs.append(df)
    return agg_dfs

def load_exchange_specific(btc_data_dir: str):
    exch_dfs = []
    for path in glob.glob(os.path.join(btc_data_dir, "BTC_* 과거 데이터.csv")):
        fname = os.path.basename(path)
        name = fname.replace("BTC_", "").replace(" 과거 데이터.csv", "")
        parts = name.split(" ", 1)
        currency = parts[0]
        market = parts[1] if len(parts) > 1 else parts[0]
        df = pd.read_csv(path, parse_dates=["날짜"], thousands=",")
        df = df.rename(
            columns={
                "날짜": "datetime",
                "시가": "open",
                "고가": "high",
                "저가": "low",
                "종가": "close",
                "거래량": "volume",
                "변동 %": "change_pct",
            }
        )
        df["currency"] = currency
        df["platform"] = currency  # e.g. "BRL"
        df["market"] = market      # English platform name
        df["index_name"] = pd.NA
        exch_dfs.append(df)
    return exch_dfs

def load_fx(currency_data_dir: str):
    fx_dfs = []
    for path in glob.glob(os.path.join(currency_data_dir, "USD_* 과거 데이터.csv")):
        fname = os.path.basename(path)
        currency = fname.replace("USD_", "").replace(" 과거 데이터.csv", "")
        df = pd.read_csv(path, parse_dates=["날짜"])
        df = df.rename(
            columns={
                "날짜": "datetime",
                "시가": "open",
                "고가": "high",
                "저가": "low",
                "종가": "close",
                "거래량": "volume",
                "변동 %": "change_pct",
            }
        )
        df["currency"] = currency
        df["platform"] = currency
        df["market"] = f"USD_{currency}"
        df["index_name"] = pd.NA
        fx_dfs.append(df)
    return fx_dfs


def main() -> None:
    btc_data_dir = os.path.join("data", "btc_data")
    currency_data_dir = os.path.join("data", "currency_data")
    start = "2017-01-01"
    end = date.today().strftime("%Y-%m-%d")
    print(f"Fetching KRX indices from {start} to {end}...")

    agg_dfs = load_aggregated(btc_data_dir)
    exch_dfs = load_exchange_specific(btc_data_dir)
    fx_dfs = load_fx(currency_data_dir)
    krx_dfs = fetch_krx(DEFAULT_INDICES, start, end)
    print(f"Fetched {len(krx_dfs)} KRX indices.")

    full = pd.concat(agg_dfs + exch_dfs + fx_dfs + krx_dfs, ignore_index=True)
    full["datetime"] = pd.to_datetime(full["datetime"], utc=True).dt.tz_convert(None)
    full = full[
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
    full = full.sort_values("datetime").reset_index(drop=True)
    print(f"Combined data contains {len(full)} rows.")

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "datafile.csv")
    full.to_csv(output_path, index=False)
    print(f"✔ Saved combined data ({len(full)} rows) to {output_path}")


if __name__ == "__main__":
    main()