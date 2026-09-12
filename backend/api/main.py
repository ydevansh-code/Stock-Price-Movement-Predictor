"""
FastAPI application serving pre-computed pipeline artifacts.
The API is read-only — no retraining on request.
Security: CORS restricted, rate limited, errors sanitized.
"""
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Add parent to path so src.config is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ROOT, CORS_ORIGINS, RATE_LIMIT

logger = logging.getLogger("uvicorn.error")

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
app = FastAPI(title="Stock Direction Predictor API", docs_url="/docs")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _load(name: str, market: str):
    path = ROOT / "artifacts" / market / f"{name}.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Artifact '{name}' for market '{market}' not found. Run the pipeline first.",
        )
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load artifact: %s", name)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/api/ohlcv")
@limiter.limit(RATE_LIMIT)
def ohlcv(request: Request, market: str = "us"):
    return _load("ohlcv", market)


@app.get("/api/class-balance")
@limiter.limit(RATE_LIMIT)
def class_balance(request: Request, market: str = "us"):
    return _load("class_balance", market)


@app.get("/api/comparison-table")
@limiter.limit(RATE_LIMIT)
def comparison_table(request: Request, market: str = "us"):
    return _load("comparison_table", market)


@app.get("/api/predictions")
@limiter.limit(RATE_LIMIT)
def predictions(request: Request, market: str = "us"):
    return _load("predictions", market)


@app.get("/api/feature-importance")
@limiter.limit(RATE_LIMIT)
def feature_importance(request: Request, market: str = "us"):
    return _load("feature_importance", market)


@app.get("/api/backtest")
@limiter.limit(RATE_LIMIT)
def backtest(request: Request, market: str = "us"):
    return _load("backtest", market)


@app.get("/api/walk-forward")
@limiter.limit(RATE_LIMIT)
def walk_forward(request: Request, market: str = "us"):
    return _load("walk_forward", market)


@app.get("/api/calibration")
@limiter.limit(RATE_LIMIT)
def calibration(request: Request, market: str = "us"):
    return _load("calibration", market)


@app.get("/health")
def health():
    return {"status": "ok"}
