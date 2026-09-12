"""
Raw-price feature set — ablation study control.

Uses only lagged OHLCV values and daily returns.
No derived indicators, no technical analysis.
All operations are vectorized pandas; no row-level Python loops.
"""
import pandas as pd

from src.config import LAG_DAYS


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of raw lagged features aligned with df's index."""
    out = pd.DataFrame(index=df.index)

    for lag in range(1, LAG_DAYS + 1):
        out[f"close_lag{lag}"] = df["close"].shift(lag)
        out[f"open_lag{lag}"] = df["open"].shift(lag)
        out[f"high_lag{lag}"] = df["high"].shift(lag)
        out[f"low_lag{lag}"] = df["low"].shift(lag)
        out[f"volume_lag{lag}"] = df["volume"].shift(lag)
        out[f"return_lag{lag}"] = df["close"].pct_change(lag)

    return out.dropna()
