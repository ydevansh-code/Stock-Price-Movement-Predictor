"""Fetch and cache OHLCV data via yfinance, with cross-asset enrichment."""
import pandas as pd
import yfinance as yf

from src.config import CROSS_ASSET_TICKERS, DATA_CACHE, END_DATE, START_DATE, TICKER


def _download_single(
    ticker: str, start: str, end: str, cache_tag: str | None = None
) -> pd.DataFrame:
    tag = cache_tag or ticker
    cache_file = DATA_CACHE / f"{tag}_{start}_{end}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"])

    df.to_parquet(cache_file)
    return df


def load(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Return daily OHLCV DataFrame enriched with cross-asset closes."""
    df = _download_single(ticker, start, end)

    for xticker, prefix in CROSS_ASSET_TICKERS.items():
        try:
            xdf = _download_single(xticker, start, end, cache_tag=prefix)
            df[f"{prefix}_close"] = xdf["close"]
        except (ValueError, KeyError):
            pass

    df = df.ffill().dropna()
    return df
