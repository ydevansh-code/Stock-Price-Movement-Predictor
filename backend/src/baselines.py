"""
Naive baselines evaluated on the same test set as real models.

Persistence  — predict that tomorrow repeats today's realized direction.
Majority     — always predict the most common class in the training split.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _scores(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(auc), 4),
    }


def persistence_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
    X_test_raw_direction: pd.Series,
) -> dict:
    """Predict tomorrow = today's actual direction."""
    y_pred = X_test_raw_direction.values
    return {"model": "Persistence", **_scores(y_test.values, y_pred)}


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Always predict the majority class from the training set."""
    majority = int(y_train.mode()[0])
    y_pred = np.full(len(y_test), majority)
    y_prob = np.full(len(y_test), float(majority))
    return {"model": "Majority Class", **_scores(y_test.values, y_pred, y_prob)}
