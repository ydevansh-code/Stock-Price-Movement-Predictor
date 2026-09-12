"""Sanity-check OHLCV DataFrame before any modelling."""
import pandas as pd


_REQUIRED = {"open", "high", "low", "close", "volume"}


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if df fails basic OHLCV sanity checks."""
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not sorted ascending.")

    if df.index.duplicated().any():
        raise ValueError("Duplicate dates found.")

    if len(df) < 252:
        raise ValueError(f"Too few rows ({len(df)}); need at least 1 year of data.")

    if (df["close"] <= 0).any():
        raise ValueError("Non-positive close prices detected.")
