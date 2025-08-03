"""Trading statistics utilities for KRX markets.

This module provides convenience wrappers around the ``pykrx`` APIs to
retrieve trading volume and value broken down by investor type (individuals,
foreigners, institutions, etc.).  The returned objects are ``pandas``
``DataFrame`` instances indexed by date.

Example
-------
>>> from krx.trading import fetch_trading_volume_by_investor
>>> df = fetch_trading_volume_by_investor("KOSPI", "2023-01-01", "2023-01-10")
>>> print(df.head())
"""

from __future__ import annotations

import pandas as pd
from pykrx import stock

from .indices import _format_date


def fetch_trading_volume_by_investor(
    market: str, start: str, end: str, *, etf: bool = False, etn: bool = False, elw: bool = False
) -> pd.DataFrame:
    """Return trading volume by investor type for a market or ticker.

    Parameters
    ----------
    market:
        Ticker or market code (e.g. ``"KOSPI"``).
    start, end:
        Date range specified as ``YYYYMMDD`` or ``YYYY-MM-DD``.
    etf, etn, elw:
        Include ETFs, ETNs or ELWs in the aggregation.
    """
    start = _format_date(start)
    end = _format_date(end)

    df = stock.get_market_trading_volume_by_investor(
        start, end, market, etf=etf, etn=etn, elw=elw
    )
    df.index = pd.to_datetime(df.index)
    return df


def fetch_trading_value_by_investor(
    market: str, start: str, end: str, *, on: str | None = None, etf: bool = False, etn: bool = False, elw: bool = False
) -> pd.DataFrame:
    """Return trading value by investor type.

    ``pykrx`` does not expose a dedicated function for value-by-investor,
    but ``get_market_trading_value_by_date`` with ``detail=True`` provides the
    necessary breakdown.
    """
    start = _format_date(start)
    end = _format_date(end)

    df = stock.get_market_trading_value_by_date(
        start, end, market, on=on, etf=etf, etn=etn, elw=elw, detail=True
    )
    df.index = pd.to_datetime(df.index)
    return df
