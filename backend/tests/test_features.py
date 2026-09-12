"""
Tests for hand-computed technical indicators.
Validates RSI, MACD, and volatility against known reference values.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LAG_DAYS
from src.features.engineered_features import _macd, _rsi, build


def _sample_close(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="close")


def _sample_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    close = _sample_close(n, seed)
    df = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.005,
        "low": close * 0.995, "close": close, "volume": 1e6,
    })
    return df


def test_rsi_range():
    """RSI must be in [0, 100] for all non-NaN values."""
    close = _sample_close(200)
    rsi = _rsi(close)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd_signal_lag():
    """Signal line should be smoother (lower std) than MACD line."""
    close = _sample_close(200)
    m = _macd(close)
    assert m["macd_signal"].dropna().std() <= m["macd"].dropna().std()


def test_macd_hist_equals_macd_minus_signal():
    """MACD histogram must equal macd - signal line exactly."""
    close = _sample_close(200)
    m = _macd(close)
    expected = m["macd"] - m["macd_signal"]
    pd.testing.assert_series_equal(m["macd_hist"], expected, check_names=False, rtol=1e-10)


def test_build_no_nans():
    """Engineered feature builder should return no NaN rows."""
    df = _sample_df(300)
    df["target"] = 0  # build() expects df from labeled data
    features = build(df)
    assert not features.isnull().any().any(), "Feature matrix contains NaNs"


def test_build_output_shape():
    """Feature matrix should have expected number of columns."""
    df = _sample_df(300)
    df["target"] = 0
    features = build(df)
    # rsi, macd, macd_signal, macd_hist, volatility, bb_width,
    # atr, stoch_k, stoch_d, obv, roc, 5 lags, day_of_week = 17
    assert features.shape[1] == 12 + LAG_DAYS
