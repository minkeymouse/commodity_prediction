"""Convenient accessors for Korean stock market data via :mod:`pykrx`.

This package provides a small collection of wrappers around the `pykrx`
project.  It focuses on index level OHLCV data and trading statistics broken
down by investor type, which are useful for analysing phenomena such as the
*Korean cryptocurrency premium*.
"""

from .indices import fetch_index_ohlcv, fetch_indices_ohlcv, fetch_index_volume
from .trading import (
    fetch_trading_value_by_investor,
    fetch_trading_volume_by_investor,
)

__all__ = [
    "fetch_index_ohlcv",
    "fetch_indices_ohlcv",
    "fetch_index_volume",
    "fetch_trading_volume_by_investor",
    "fetch_trading_value_by_investor",
]
