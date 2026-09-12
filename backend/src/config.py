"""Central configuration — all magic numbers live here."""
from pathlib import Path

import os

MARKET = os.environ.get("MARKET", "us").lower()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / MARKET
DATA_CACHE = ROOT / "artifacts" / "cache"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_CACHE.mkdir(parents=True, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
START_DATE: str = "2000-01-01"
END_DATE: str = "2026-12-31"

if MARKET == "ind":
    TICKER: str = "^NSEI"
    CROSS_ASSET_TICKERS: dict[str, str] = {
        "^INDIAVIX": "vix",
        "GLD": "gld",
        "USO": "uso",
    }
else:
    TICKER: str = "SPY"
    CROSS_ASSET_TICKERS: dict[str, str] = {
        "^VIX": "vix",
        "TLT": "tlt",
        "GLD": "gld",
        "USO": "uso",
    }

# ── Model ─────────────────────────────────────────────────────────────────────
TEST_SPLIT: float = 0.20   # last 20 % of dates → test
RANDOM_SEED: int = 42
N_WALK_FOLDS: int = 5

# ── Indicators ────────────────────────────────────────────────────────────────
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
VOL_PERIOD: int = 10
LAG_DAYS: int = 10         # number of raw-feature lag days

# ── API ───────────────────────────────────────────────────────────────────────
CORS_ORIGINS: list[str] = ["http://localhost:3000"]
RATE_LIMIT: str = "100/minute"
