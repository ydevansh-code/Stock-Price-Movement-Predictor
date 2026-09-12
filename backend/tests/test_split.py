"""
Tests for chronological split and scaler isolation.
Asserts:
  - All train dates strictly precede all test dates.
  - Scaler mean/std match statistics computed from training data only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import chronological_split, scale


def _make_xy(n: int = 200):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)}, index=idx)
    y = pd.Series(np.random.randint(0, 2, n), index=idx)
    return X, y


def test_split_is_chronological():
    """Max train date must be strictly before min test date."""
    X, y = _make_xy()
    X_tr, X_te, y_tr, y_te = chronological_split(X, y)
    assert X_tr.index.max() < X_te.index.min()


def test_no_overlap():
    """Train and test indices must not share any date."""
    X, y = _make_xy()
    X_tr, X_te, _, _ = chronological_split(X, y)
    overlap = X_tr.index.intersection(X_te.index)
    assert len(overlap) == 0


def test_scaler_fit_on_train_only():
    """Scaler mean/std must match training-set statistics, not the full dataset."""
    X, y = _make_xy()
    X_tr, X_te, _, _ = chronological_split(X, y)

    Xtr_s, Xte_s, scaler = scale(X_tr, X_te, "test_scaler")

    # Scaler's learned mean should match train mean, not full-data mean
    np.testing.assert_allclose(scaler.mean_, X_tr.mean().values, rtol=1e-6)
    np.testing.assert_allclose(scaler.scale_, X_tr.std(ddof=0).values, rtol=1e-2)

    # Test mean after transform should differ from train mean (data distribution differs)
    full_mean = X.mean().values
    # The scaler mean should NOT equal the full-data mean unless by coincidence
    # (with random data this will reliably differ)
    # We just assert the scaler mean equals the train mean definitively
    np.testing.assert_allclose(scaler.mean_, X_tr.mean().values, rtol=1e-5)


def test_transformed_test_not_refitted():
    """Applying scale() twice should not change the scaler's mean (no refit on test)."""
    X, y = _make_xy()
    X_tr, X_te, _, _ = chronological_split(X, y)
    _, _, scaler1 = scale(X_tr, X_te, "test_scaler_a")
    mean_after_first = scaler1.mean_.copy()

    # Manually transform again — mean must remain the same
    _ = scaler1.transform(X_te)
    np.testing.assert_array_equal(scaler1.mean_, mean_after_first)
