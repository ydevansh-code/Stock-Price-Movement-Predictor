"""
Enhanced feature set v2 — cross-asset macro context + regime awareness.

Layers:
  1. Base indicators (RSI, MACD, vol, BB, stoch, OBV, ROC, lags, calendar)
  2. Price structure (overnight gap, intraday range, close position)
  3. Trend regime (ADX, RSI zones, MACD crossover, trend agreement)
  4. Multi-timeframe (weekly/monthly returns, MA crossovers)
  5. Volume-volatility (volume deviation, ATR-normalized return)
  6. Cross-asset context (VIX level/change, bond/gold/oil momentum, inter-market divergence)
  7. Volatility regime (vol regime bucket, vol-of-vol, range compression)

All operations vectorized. No row loops.
"""
import numpy as np
import pandas as pd
import ta

from src.config import (
    CROSS_ASSET_TICKERS,
    LAG_DAYS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    VOL_PERIOD,
)


def _get_weights(d: float, size: int) -> np.ndarray:
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w[::-1])

def _frac_diff(series: pd.Series, d: float = 0.4, window: int = 40) -> pd.Series:
    weights = _get_weights(d, window)
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> pd.DataFrame:
    fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal, "macd_hist": line - signal},
        index=close.index,
    )


def _bollinger_width(close: pd.Series, period: int = VOL_PERIOD) -> pd.Series:
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    return (2 * std) / mid.replace(0, float("nan"))


def _add_base_indicators(df: pd.DataFrame, out: pd.DataFrame) -> None:
    out["rsi"] = _rsi(df["close"])
    macd_df = _macd(df["close"])
    for col in macd_df.columns:
        out[col] = macd_df[col]
    out["volatility"] = df["close"].pct_change().rolling(VOL_PERIOD, min_periods=VOL_PERIOD).std()
    out["bb_width"] = _bollinger_width(df["close"])
    out["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    out["roc"] = ta.momentum.roc(df["close"], window=10)
    pct = df["close"].pct_change()
    for lag in range(1, LAG_DAYS + 1):
        out[f"ret_lag{lag}"] = pct.shift(lag)
    out["day_of_week"] = df.index.dayofweek
    out["frac_diff_04"] = _frac_diff(np.log(df["close"]).replace([np.inf, -np.inf], np.nan).dropna())


def _add_price_structure(df: pd.DataFrame, out: pd.DataFrame) -> None:
    prev_close = df["close"].shift(1)
    out["overnight_gap"]  = (df["open"] - prev_close) / prev_close
    out["intraday_range"] = (df["high"] - df["low"]) / df["close"]
    out["close_to_high"]  = (df["high"] - df["close"]) / df["close"]
    out["close_to_low"]   = (df["close"] - df["low"]) / df["close"]


def _add_trend_regime(df: pd.DataFrame, out: pd.DataFrame) -> None:
    out["adx"]         = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    out["adx_trending"] = (out["adx"] > 25).astype(int)
    out["rsi_ob"] = (out["rsi"] > 70).astype(int)
    out["rsi_os"] = (out["rsi"] < 30).astype(int)
    out["macd_cross"] = (
        (out["macd"] > 0) & (out["macd"].shift(1) <= 0) |
        (out["macd"] < 0) & (out["macd"].shift(1) >= 0)
    ).astype(int)
    macd_sign = np.sign(out["macd"])
    rsi_sign  = np.sign(out["rsi"] - 50)
    out["trend_agree"] = (macd_sign == rsi_sign).astype(int)


def _add_multi_timeframe(df: pd.DataFrame, out: pd.DataFrame) -> None:
    out["weekly_ret"]  = df["close"].pct_change(5)
    out["monthly_ret"] = df["close"].pct_change(21)
    ma50  = df["close"].rolling(50, min_periods=50).mean()
    ma200 = df["close"].rolling(200, min_periods=200).mean()
    out["above_50ma"]   = (df["close"] > ma50).astype(int)
    out["above_200ma"]  = (df["close"] > ma200).astype(int)
    out["golden_cross"] = (ma50 > ma200).astype(int)


def _add_volume_volatility(df: pd.DataFrame, out: pd.DataFrame) -> None:
    vol_ma = df["volume"].rolling(20, min_periods=20).mean()
    out["volume_dev"]   = (df["volume"] - vol_ma) / vol_ma.replace(0, float("nan"))
    atr_norm = out["atr"] / df["close"]
    out["atr_norm_ret"] = df["close"].pct_change() / atr_norm.replace(0, float("nan"))


def _add_cross_asset(df: pd.DataFrame, out: pd.DataFrame) -> None:
    """Cross-asset features: VIX level, bond/gold/oil momentum, inter-market signals."""
    available = [prefix for _, prefix in CROSS_ASSET_TICKERS.items() if f"{prefix}_close" in df.columns]
    if not available:
        return

    if "vix_close" in df.columns:
        vix = df["vix_close"]
        out["vix_level"] = vix
        out["vix_change"] = vix.pct_change()
        out["vix_5d_change"] = vix.pct_change(5)
        vix_ma20 = vix.rolling(20, min_periods=20).mean()
        out["vix_above_ma20"] = (vix > vix_ma20).astype(int)
        vix_percentile = vix.rolling(60, min_periods=60).rank(pct=True)
        out["vix_percentile"] = vix_percentile
        out["vix_term_regime"] = pd.cut(
            vix, bins=[0, 15, 20, 30, 100], labels=[0, 1, 2, 3]
        ).astype(float)

    for prefix in ["tlt", "gld", "uso"]:
        col = f"{prefix}_close"
        if col not in df.columns:
            continue
        price = df[col]
        out[f"{prefix}_ret1"]  = price.pct_change()
        out[f"{prefix}_ret5"]  = price.pct_change(5)
        out[f"{prefix}_ret21"] = price.pct_change(21)
        ma20 = price.rolling(20, min_periods=20).mean()
        out[f"{prefix}_above_ma20"] = (price > ma20).astype(int)

    if "tlt_close" in df.columns:
        spy_ret5 = df["close"].pct_change(5)
        tlt_ret5 = df["tlt_close"].pct_change(5)
        out["stock_bond_diverge"] = spy_ret5 - tlt_ret5
        corr_win = 20
        out["stock_bond_corr"] = spy_ret5.rolling(corr_win, min_periods=corr_win).corr(tlt_ret5)

    if "gld_close" in df.columns:
        spy_ret5 = df["close"].pct_change(5)
        gld_ret5 = df["gld_close"].pct_change(5)
        out["stock_gold_diverge"] = spy_ret5 - gld_ret5

    if "uso_close" in df.columns:
        out["oil_spy_ratio"] = df["uso_close"] / df["close"]


def _add_volatility_regime(df: pd.DataFrame, out: pd.DataFrame) -> None:
    """Volatility regime features: vol regime bucket, vol-of-vol, range compression."""
    realized_vol = df["close"].pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
    vol_median = realized_vol.rolling(252, min_periods=60).median()
    out["vol_regime"] = (realized_vol > vol_median).astype(int)

    vol_of_vol = realized_vol.rolling(20, min_periods=20).std()
    out["vol_of_vol"] = vol_of_vol

    atr_short = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=5)
    atr_long  = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=20)
    out["range_compression"] = atr_short / atr_long.replace(0, float("nan"))

    out["realized_vol_ann"] = realized_vol


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full v2 enhanced feature matrix. df must have OHLCV + target + cross-asset columns."""
    out = pd.DataFrame(index=df.index)
    _add_base_indicators(df, out)
    _add_price_structure(df, out)
    _add_trend_regime(df, out)
    _add_multi_timeframe(df, out)
    _add_volume_volatility(df, out)
    _add_cross_asset(df, out)
    _add_volatility_regime(df, out)
    return out.dropna()
