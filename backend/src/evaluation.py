"""
Evaluation helpers: metrics, walk-forward CV, backtest, calibration.
All return plain dicts/lists suitable for JSON serialization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import N_WALK_FOLDS, RANDOM_SEED, TEST_SPLIT


# ── Core metrics ─────────────────────────────────────────────────────────────

def metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": cm,
    }


def class_balance(y: pd.Series, label: str) -> dict:
    vc = y.value_counts(normalize=True)
    return {
        "split": label,
        "up_pct": round(float(vc.get(1, 0.0)), 4),
        "down_pct": round(float(vc.get(0, 0.0)), 4),
        "n": int(len(y)),
    }


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: "Callable[[np.ndarray, np.ndarray], Any]",
    gap: int = 0,
) -> list[dict]:
    """Expanding-window walk-forward across N_WALK_FOLDS folds.

    Args:
        X: Scaled feature matrix, shape (n_samples, n_features).
        y: Binary target array, shape (n_samples,).
        model_fn: Callable(X_train, y_train) -> fitted model.
                  Model must expose .predict(X) and .predict_proba(X).
        gap: Number of samples to drop from the end of each train fold to prevent overlap.

    Returns:
        List of per-fold dicts with keys: fold, train_size, accuracy, roc_auc.
    """
    n = len(X)
    base = int(n * (1 - TEST_SPLIT))
    fold_size = (n - base) // N_WALK_FOLDS
    results = []

    for i in range(N_WALK_FOLDS):
        test_start = base + i * fold_size
        test_end = test_start + fold_size if i < N_WALK_FOLDS - 1 else n
        if test_end <= test_start or test_start - gap <= 0:
            break
        X_tr, y_tr = X[:test_start - gap], y[:test_start - gap]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        model = model_fn(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        acc = float(accuracy_score(y_te, model.predict(X_te)))
        auc = float(roc_auc_score(y_te, prob))
        results.append({"fold": i + 1, "train_size": int(len(X_tr)), "accuracy": round(acc, 4), "roc_auc": round(auc, 4)})

    return results


# ── Trading backtest ──────────────────────────────────────────────────────────

def backtest(
    close: pd.Series, y_pred: np.ndarray, slippage_bps: float = 5.0
) -> dict:
    """
    Simplified long-only backtest: go long when model predicts 'up', else flat.
    Applies one-way slippage on each trade entry/exit.
    Returns equity curve data and summary statistics.
    DISCLAIMER: For illustration only. Does not account for taxes, market impact,
    borrowing costs, or real execution. Not investment advice.
    """
    slip = slippage_bps / 10_000
    price = close.values
    returns = np.diff(price) / price[:-1]  # daily returns aligned with predictions

    # Trim to same length as predictions (preds may be shorter due to feature NaNs)
    n = min(len(returns), len(y_pred) - 1)
    returns = returns[-n:]
    positions = y_pred[-n - 1 : -1]  # position held during that day

    # Slippage on transitions (entry/exit)
    transitions = np.abs(np.diff(np.concatenate([[0], positions])))
    strat_returns = positions * returns - transitions * slip

    bh_eq = np.cumprod(1 + returns)
    strat_eq = np.cumprod(1 + strat_returns)

    def max_drawdown(eq: np.ndarray) -> float:
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return round(float(dd.min()), 4)

    def sharpe(r: np.ndarray) -> float:
        if r.std() == 0:
            return 0.0
        return round(float(r.mean() / r.std() * np.sqrt(252)), 4)

    return {
        "equity_curve": [
            {"t": i, "strategy": round(float(strat_eq[i]), 6), "buy_hold": round(float(bh_eq[i]), 6)}
            for i in range(len(strat_eq))
        ],
        "strategy_total_return": round(float(strat_eq[-1] - 1), 4),
        "buy_hold_total_return": round(float(bh_eq[-1] - 1), 4),
        "strategy_sharpe": sharpe(strat_returns),
        "strategy_max_dd": max_drawdown(strat_eq),
        "disclaimer": (
            "Illustrative only. Ignores taxes, real slippage, market impact, "
            "borrowing costs. NOT investment advice."
        ),
    }


# ── Probability calibration ───────────────────────────────────────────────────

def calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Return binned calibration curve points."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return [
        {"mean_predicted": round(float(p), 4), "fraction_positive": round(float(t), 4)}
        for p, t in zip(prob_pred, prob_true)
    ]
