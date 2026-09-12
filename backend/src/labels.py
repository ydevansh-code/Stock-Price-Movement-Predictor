import pandas as pd

def make_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["target"] = (out["close"].shift(-horizon) > out["close"]).astype(int)
    for h in [1, 3, 5]:
        out[f"target_{h}d"] = (out["close"].shift(-h) > out["close"]).astype(int)
    out = out.iloc[:-max(1, horizon)].copy()
    return out

def make_multi_horizon_labels(
    df: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5)
) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"target_{h}d"] = (out["close"].shift(-h) > out["close"]).astype(int)
    out["target"] = out["target_1d"]
    out = out.iloc[:-max(horizons)].copy()
    return out
