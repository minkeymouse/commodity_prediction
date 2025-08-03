"""Utilities for fetching daily KRX index data.

This module leverages the `pykrx` package to download daily OHLCV data for
Korean stock indices such as KOSPI and KOSDAQ.  The functions return pandas
``DataFrame`` objects with the standard open/high/low/close/volume columns so
that they can be used for further analysis, e.g. examining the relationship
between the so‑called *kimchi premium* and index returns.

Example
-------
>>> from fetch_krx import fetch_index_ohlcv
>>> df = fetch_index_ohlcv("KOSPI", "2023-01-01", "2023-01-10")
>>> print(df.head())

The ``pykrx`` package must be installed in the environment for these
functions to operate.  If it is missing the import below will raise a
``ModuleNotFoundError``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable

import pandas as pd
from pykrx import stock

# Commonly used index names mapped to their official codes recognised by pykrx.
# Users can pass either the code (e.g. "1001") or the human readable name
# (e.g. "KOSPI").  The mapping below is not exhaustive but covers the major
# benchmarks.
INDEX_CODES: Dict[str, str] = {
    "KOSPI": "1001",    # 코스피
    "KOSDAQ": "2001",   # 코스닥
    "KOSPI200": "1028",  # 코스피200
    "KOSDAQ150": "2203",  # 코스닥150
}


def _format_date(date_str: str) -> str:
    """Return a ``YYYYMMDD`` formatted string.

    ``pykrx`` accepts dates without hyphens; this helper makes the function
    tolerant of both ``YYYYMMDD`` and ``YYYY-MM-DD`` inputs.
    """
    if "-" in date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    return date_str


def fetch_index_ohlcv(index: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a single index.

    Parameters
    ----------
    index:
        The name or code of the index.  Common names such as ``"KOSPI"`` or
        ``"KOSDAQ"`` are mapped to their codes automatically.
    start, end:
        Date range (inclusive) specified either as ``YYYYMMDD`` or
        ``YYYY-MM-DD``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date containing columns ``Open``, ``High``,
        ``Low``, ``Close``, ``Volume`` and ``Change``.
    """
    start = _format_date(start)
    end = _format_date(end)

    # Translate human readable index names to codes if necessary.
    code = INDEX_CODES.get(index.upper(), index)

    df = stock.get_index_ohlcv_by_date(start, end, code)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_indices_ohlcv(indices: Iterable[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple indices.

    Parameters
    ----------
    indices:
        Iterable of index names or codes.
    start, end:
        Date range for which to download data.

    Returns
    -------
    dict
        A mapping from the supplied index identifier to its corresponding
        DataFrame of OHLCV data.
    """
    return {idx: fetch_index_ohlcv(idx, start, end) for idx in indices}


def fetch_index_volume(index: str, start: str, end: str) -> pd.Series:
    """Convenience wrapper returning only the volume series for an index."""
    df = fetch_index_ohlcv(index, start, end)
    return df["Volume"].rename(index)


if __name__ == "__main__":  # pragma: no cover - simple usage demonstration
    # Example usage: fetch KOSPI and KOSDAQ data for the first week of 2023.
    indices = ["KOSPI", "KOSDAQ"]
    data = fetch_indices_ohlcv(indices, "2023-01-01", "2023-01-07")
    for name, df in data.items():
        print(f"{name}:")
        print(df.head(), end="\n\n")
