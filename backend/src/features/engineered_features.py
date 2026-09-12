"""
Engineered feature set — hand-computed indicators + ta library extras.

Three indicators computed from scratch in pandas (vectorized, no loops):
  1. RSI-14   — Wilder's relative strength index
  2. MACD     — 12/26 EMA difference + 9-period signal line + histogram
  3. Vol-10   — rolling 10-day return std (volatility proxy) + BB width

Plus extras via ta library: ATR-14, Stoch %K/%D, OBV, ROC-10.
Calendar feature and lagged returns round out the set.
"""
import pandas as pd
import ta

from src.config import (
    LAG_DAYS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    VOL_PERIOD,
)


# ── Hand-computed indicators ─────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> pd.DataFrame:
    fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal, "macd_hist": macd_line - signal},
        index=close.index,
    )


def _bollinger_width(close: pd.Series, period: int = VOL_PERIOD) -> pd.Series:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std) / mid.replace(0, float("nan"))


# ── Full feature builder ──────────────────────────────────────────────────────

def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return engineered feature DataFrame aligned with df's index."""
    out = pd.DataFrame(index=df.index)

    # Hand-computed
    out["rsi"] = _rsi(df["close"])
    macd_df = _macd(df["close"])
    out = pd.concat([out, macd_df], axis=1)
    out["volatility"] = df["close"].pct_change().rolling(VOL_PERIOD).std()
    out["bb_width"] = _bollinger_width(df["close"])

    # ta library extras (all vectorized internally)
    out["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    out["roc"] = ta.momentum.roc(df["close"], window=10)

    # Lagged returns
    for lag in range(1, LAG_DAYS + 1):
        out[f"ret_lag{lag}"] = df["close"].pct_change().shift(lag)

    # Calendar
    out["day_of_week"] = df.index.dayofweek

    return out.dropna()
