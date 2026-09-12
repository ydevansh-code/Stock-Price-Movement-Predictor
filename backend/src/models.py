"""
Model training with strict chronological split and train-only scaler fitting.

Split rule: last TEST_SPLIT fraction of dates → test, rest → train.
No shuffling. Scaler fitted on train, .transform()-only on test.
"""
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS, RANDOM_SEED, TEST_SPLIT


def chronological_split(
    X: pd.DataFrame, y: pd.Series, gap: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X/y at chronological cutoff. Returns X_train, X_test, y_train, y_test.
    
    If gap > 0, the last `gap` samples are dropped from the training set to prevent
    overlap leakage (embargo) between train and test.
    """
    n = len(X)
    split = int(n * (1 - TEST_SPLIT))
    train_end = split - gap
    return X.iloc[:train_end], X.iloc[split:], y.iloc[:train_end], y.iloc[split:]


def scale(
    X_train: pd.DataFrame, X_test: pd.DataFrame, name: str = "scaler"
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train, transform both. Returns DataFrames and fitted scaler."""
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_te = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    joblib.dump(scaler, ARTIFACTS / f"{name}.joblib")
    return X_tr, X_te, scaler


def train_logistic(
    X_train: np.ndarray, y_train: pd.Series, tag: str = "lr"
) -> LogisticRegression:
    model = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    joblib.dump(model, ARTIFACTS / f"{tag}.joblib")
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    tag: str = "rf",
    max_depth: int = 5,
    min_samples_leaf: int = 10,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, ARTIFACTS / f"{tag}.joblib")
    return model


def train_lightgbm(
    X_train: np.ndarray, y_train: pd.Series, n_trials: int = 20, gap: int = 0
) -> lgb.LGBMClassifier:
    """Train LightGBM classifier with Optuna hyperparameter tuning using TimeSeriesSplit."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=3, gap=gap)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    model = lgb.LGBMClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, ARTIFACTS / "lgbm.joblib")
    return model
