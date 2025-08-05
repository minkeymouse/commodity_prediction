from __future__ import annotations

import glob
import os
from datetime import date
from typing import Dict, List

import pandas as pd

from krx.indices import fetch_index_ohlcv

# Mapping of KRX index codes to human readable names to include in the dataset
# The codes are those recognised by the pykrx package.  Additional indices can
# be appended here if required.
KRX_INDICES: Dict[str, str] = {
    "1001": "KOSPI",
    "1028": "KOSPI 200",
    "1034": "KOSPI 100",
    "1035": "KOSPI 50",
    "1167": "KOSPI 200 MidSmall",
    "1182": "KOSPI 200 ex-Large",
    "1244": "KOSPI ex KOSPI200",
    "1150": "KOSPI 200 Communication Services",
    "1151": "KOSPI 200 Construction",
    "1152": "KOSPI 200 Heavy Industry",
    "1153": "KOSPI 200 Steel & Materials",
    "1154": "KOSPI 200 Energy & Chemicals",
    "1155": "KOSPI 200 Information Technology",
    "1156": "KOSPI 200 Finance",
    "1157": "KOSPI 200 Consumer Staples",
    "1158": "KOSPI 200 Consumer Discretionary",
    "1159": "KOSPI 200 Industrials",
    "1160": "KOSPI 200 Healthcare",
    "1005": "Food & Beverage",
    "1006": "Textiles & Clothing",
    "1007": "Paper & Wood",
    "1008": "Chemicals",
    "1009": "Pharmaceuticals",
    "1010": "Non-metal Minerals",
    "1011": "Steel & Metals",
    "1012": "Machinery",
    "1013": "Electrical & Electronics",
    "1014": "Medical Precision",
    "1015": "Transportation Equipment",
    "1016": "Distribution",
    "1017": "Utilities",
    "1018": "Construction",
    "1019": "Transport & Storage",
    "1020": "Telecommunications",
    "1021": "Finance",
    "1022": "Banking",
    "1024": "Securities",
    "1025": "Insurance",
    "1026": "Services",
    "1027": "Manufacturing",
    "1002": "Large Cap",
    "1003": "Mid Cap",
    "1004": "Small Cap",
    "1224": "KOSPI 200 Cap 30%",
    "1227": "KOSPI 200 Cap 25%",
    "1232": "KOSPI 200 Cap 20%",
    "2001": "KOSDAQ",
    "2203": "KOSDAQ 150",
}


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


def fetch_krx(indices: Dict[str, str], start: str, end: str) -> List[pd.DataFrame]:
    """Fetch OHLCV data for ``indices`` between ``start`` and ``end``."""
    dfs: List[pd.DataFrame] = []
    for code, name in indices.items():
        try:
            df = fetch_index_ohlcv(code, start, end)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"⚠️  Skipping {code} ({name}): {exc}")
            continue
        dfs.append(_prepare_krx_dataframe(df, name))
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


def load_msci(msci_data_dir: str):
    msci_dfs = []
    # Known region to currency mapping where not explicitly stated
    region_currency = {
        "Australia": "AUD",
        "Brazil": "BRL",
        "Canada": "CAD",
        "Europe": "EUR",
        "India": "INR",
        "Indonesia": "IDR",
        "Japan Net": "JPY",
        "Korea": "KRW",
        "Malaysia": "MYR",
        "Mexico": "MXN",
        "New Zealand": "NZD",
        "Philippines": "PHP",
        "Poland": "PLN",
        "South Africa": "ZAR",
        "Switzerland": "CHF",
        "Thailand": "THB",
        "Turkey": "TRY",
        "United Kingdom": "GBP",
    }

    for path in glob.glob(os.path.join(msci_data_dir, "MSCI*과거 데이터.csv")):
        fname = os.path.basename(path)
        name_part = fname.replace("MSCI ", "").replace(" 과거 데이터.csv", "").strip()
        parts = name_part.split()
        if parts[-1].isupper() and len(parts[-1]) == 3:
            currency = parts[-1]
            region = " ".join(parts[:-1])
        else:
            region = name_part
            currency = region_currency.get(region, "USD")
        df = pd.read_csv(path, thousands=",")
        df["날짜"] = pd.to_datetime(df["날짜"].str.replace(" ", ""))
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
        df["platform"] = "MSCI"
        df["market"] = "MSCI"
        df["index_name"] = f"MSCI {region}"
        msci_dfs.append(df)
    return msci_dfs


def main() -> None:
    btc_data_dir = os.path.join("data", "btc_data")
    currency_data_dir = os.path.join("data", "currency_data")
    msci_data_dir = os.path.join("data", "MSCI_data")
    start = "2017-01-01"
    end = date.today().strftime("%Y-%m-%d")
    print(f"Fetching KRX indices from {start} to {end}...")

    agg_dfs = load_aggregated(btc_data_dir)
    exch_dfs = load_exchange_specific(btc_data_dir)
    fx_dfs = load_fx(currency_data_dir)
    msci_dfs = load_msci(msci_data_dir)
    krx_dfs = fetch_krx(KRX_INDICES, start, end)
    print(f"Fetched {len(krx_dfs)} KRX indices.")

    full = pd.concat(agg_dfs + exch_dfs + fx_dfs + msci_dfs + krx_dfs, ignore_index=True)
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
    output_path = os.path.join("data", "final_datafile.csv")
    full.to_csv(output_path, index=False)
    print(f"✔ Saved combined data ({len(full)} rows) to {output_path}")


if __name__ == "__main__":
    main()
