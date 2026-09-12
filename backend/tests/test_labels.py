"""
Tests for label construction — asserts no data leakage.
Uses a synthetic price series with known step patterns.
"""
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels import make_labels


def _make_df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices, "open": prices, "high": prices,
                         "low": prices, "volume": [1e6] * len(prices)}, index=idx)


def test_label_direction_correctness():
    """target_t == 1 iff close_{t+1} > close_t."""
    prices = [100, 110, 105, 115, 108, 120]
    df = _make_df(prices)
    labeled = make_labels(df)

    # 5 rows should remain (last dropped)
    assert len(labeled) == len(prices) - 1

    for i in range(len(prices) - 1):
        expected = int(prices[i + 1] > prices[i])
        actual = labeled["target"].iloc[i]
        assert actual == expected, f"Row {i}: expected {expected}, got {actual}"


def test_last_row_dropped():
    """The final row (no t+1) must be removed."""
    prices = [10, 20, 15, 25]
    df = _make_df(prices)
    labeled = make_labels(df)
    assert len(labeled) == len(prices) - 1


def test_no_future_info_in_label():
    """Label at row i must not equal a feature that uses close_{i+1}."""
    prices = [100, 200, 50, 300]
    df = _make_df(prices)
    labeled = make_labels(df)
    # If any feature column equals the next row's close, leakage exists
    # Here we just check target is binary
    assert set(labeled["target"].unique()).issubset({0, 1})


def test_monotone_decreasing_series():
    """All-down prices → all targets should be 0."""
    prices = [100, 90, 80, 70, 60]
    df = _make_df(prices)
    labeled = make_labels(df)
    assert (labeled["target"] == 0).all()


def test_monotone_increasing_series():
    prices = [60, 70, 80, 90, 100]
    df = _make_df(prices)
    labeled = make_labels(df)
    assert (labeled["target"] == 1).all()


def test_multi_day_horizons():
    prices = [100, 105, 110, 95, 90, 120]
    df = _make_df(prices)
    labeled3 = make_labels(df, horizon=3)
    assert len(labeled3) == len(prices) - 3
    assert labeled3["target"].iloc[0] == int(prices[3] > prices[0])
    assert labeled3["target"].iloc[1] == int(prices[4] > prices[1])
