#!/usr/bin/env python3
import os
import glob
import pandas as pd

def load_aggregated(btc_data_dir: str):
    agg_dfs = []
    for path in glob.glob(os.path.join(btc_data_dir, "bitcoinity_data_*.csv")):
        currency = os.path.basename(path).replace("bitcoinity_data_", "").replace(".csv", "")
        df = pd.read_csv(path, parse_dates=["Time"])
        df = df.melt(
            id_vars=["Time"],
            var_name="market",
            value_name="close"
        ).rename(columns={"Time": "datetime"})
        df["platform"] = "bitcoinity"
        for col in ["open", "high", "low", "volume", "change_pct"]:
            df[col] = pd.NA
        df["currency"] = currency
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
        df = df.rename(columns={
            "날짜":    "datetime",
            "시가":     "open",
            "고가":     "high",
            "저가":     "low",
            "종가":     "close",
            "거래량":   "volume",
            "변동 %":   "change_pct"
        })
        df["currency"] = currency
        df["platform"] = currency  # e.g. "BRL"
        df["market"] = market      # English platform name
        exch_dfs.append(df)
    return exch_dfs

def load_fx(currency_data_dir: str):
    fx_dfs = []
    for path in glob.glob(os.path.join(currency_data_dir, "USD_* 과거 데이터.csv")):
        fname = os.path.basename(path)
        currency = fname.replace("USD_", "").replace(" 과거 데이터.csv", "")
        df = pd.read_csv(path, parse_dates=["날짜"])
        df = df.rename(columns={
            "날짜":    "datetime",
            "시가":     "open",
            "고가":     "high",
            "저가":     "low",
            "종가":     "close",
            "거래량":   "volume",
            "변동 %":   "change_pct"
        })
        df["currency"] = currency
        df["platform"] = currency
        df["market"] = f"USD_{currency}"
        fx_dfs.append(df)
    return fx_dfs

def main():
    btc_data_dir = os.path.join("data", "btc_data")
    currency_data_dir = os.path.join("data", "currency_data")

    agg_dfs = load_aggregated(btc_data_dir)
    exch_dfs = load_exchange_specific(btc_data_dir)
    fx_dfs = load_fx(currency_data_dir)

    full = pd.concat(agg_dfs + exch_dfs + fx_dfs, ignore_index=True)

    full["datetime"] = (
        pd.to_datetime(full["datetime"], utc=True)
          .dt.tz_convert(None)
    )

    full = full[[
        "datetime",
        "currency",
        "market",
        "platform",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change_pct"
    ]]

    full = full.sort_values("datetime").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "BTC.csv")
    full.to_csv(output_path, index=False)
    print(f"✔ Saved combined data ({len(full)} rows) to {output_path}")

if __name__ == "__main__":
    main()
