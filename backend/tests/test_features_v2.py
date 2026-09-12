"""
Tests for engineered_features_v2 — verifies new signals produce valid values.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineered_features_v2 import build


def _sample_df(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 400 + np.cumsum(rng.normal(0, 1.5, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(50_000_000, 200_000_000, n).astype(float),
        "target": rng.integers(0, 2, n),
    }, index=idx)


def test_v2_no_nans():
    """v2 feature matrix must have zero NaN rows after dropna."""
    df = _sample_df()
    X = build(df)
    assert not X.isnull().any().any(), "NaNs found in v2 features"


def test_v2_more_features_than_v1():
    """v2 must have more columns than v1 (we added at minimum 10 new signals)."""
    from src.features.engineered_features import build as build_v1
    df = _sample_df()
    df_v1 = df.copy()
    X_v1 = build_v1(df_v1)
    X_v2 = build(df)
    assert X_v2.shape[1] > X_v1.shape[1], "v2 should have more features than v1"


def test_overnight_gap_uses_prev_close():
    """Overnight gap on day t should use close of t-1, not t."""
    df = _sample_df()
    X = build(df)
    # Manually compute for a known row
    i = 50
    date = X.index[i]
    prev_date = df.index[df.index.get_loc(date) - 1]
    expected = (df.loc[date, "open"] - df.loc[prev_date, "close"]) / df.loc[prev_date, "close"]
    assert abs(X.loc[date, "overnight_gap"] - expected) < 1e-10


def test_rsi_regime_dummies_binary():
    """rsi_ob and rsi_os must be 0 or 1 only."""
    df = _sample_df()
    X = build(df)
    assert set(X["rsi_ob"].unique()).issubset({0, 1})
    assert set(X["rsi_os"].unique()).issubset({0, 1})


def test_ma_flags_binary():
    """above_50ma, above_200ma, golden_cross must be binary."""
    df = _sample_df()
    X = build(df)
    for col in ("above_50ma", "above_200ma", "golden_cross"):
        assert set(X[col].unique()).issubset({0, 1}), f"{col} is not binary"
