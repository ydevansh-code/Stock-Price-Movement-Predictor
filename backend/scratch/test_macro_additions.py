import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from src.data import loader
from src.labels import make_labels
from src.features import engineered_features_v2 as eng_v2
from src.models import chronological_split, scale

raw_df = loader.load()
labeled = make_labels(raw_df)

tickers = ["HYG", "LQD", "^TNX", "UUP"]
for t in tickers:
    raw = yf.download(t, start="2018-01-01", end="2024-12-31", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    tag = t.lower().replace("^", "")
    labeled[f"{tag}_close"] = raw["Close"]

labeled = labeled.ffill().dropna()

X = eng_v2.build(labeled)

if "hyg_close" in labeled.columns and "lqd_close" in labeled.columns:
    X["credit_ratio"] = labeled["hyg_close"] / labeled["lqd_close"]
    X["credit_ret5"] = X["credit_ratio"].pct_change(5)
    X["credit_ret21"] = X["credit_ratio"].pct_change(21)

if "tnx_close" in labeled.columns:
    X["tnx_ret1"] = labeled["tnx_close"].pct_change()
    X["tnx_ret5"] = labeled["tnx_close"].pct_change(5)

if "uup_close" in labeled.columns:
    X["uup_ret1"] = labeled["uup_close"].pct_change()
    X["uup_ret5"] = labeled["uup_close"].pct_change(5)

ma20 = labeled["close"].rolling(20).mean()
std20 = labeled["close"].rolling(20).std()
X["zscore_20"] = (labeled["close"] - ma20) / std20

X = X.dropna()
y = labeled["target"].loc[X.index]

X_tr, X_te, y_tr, y_te = chronological_split(X, y)
Xs_tr, Xs_te, _ = scale(X_tr, X_te)

rf_full = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=20, random_state=42)
rf_full.fit(Xs_tr, y_tr)
importances = pd.Series(rf_full.feature_importances_, index=X.columns).sort_values(ascending=False)

for top_k in [15, 20, 25, 30]:
    top_cols = list(importances.index[:top_k])
    X_tr_k = Xs_tr[top_cols]
    X_te_k = Xs_te[top_cols]
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=20, random_state=42)
    rf.fit(X_tr_k, y_tr)
    prob_rf = rf.predict_proba(X_te_k)[:, 1]
    auc_rf = roc_auc_score(y_te, prob_rf)
    acc_rf = accuracy_score(y_te, (prob_rf > 0.5).astype(int))
    
    lgb = LGBMClassifier(n_estimators=60, max_depth=2, learning_rate=0.03, num_leaves=4, min_child_samples=30, random_state=42, verbose=-1)
    lgb.fit(X_tr_k, y_tr)
    prob_lgb = lgb.predict_proba(X_te_k)[:, 1]
    auc_lgb = roc_auc_score(y_te, prob_lgb)
    acc_lgb = accuracy_score(y_te, (prob_lgb > 0.5).astype(int))
    
    ens_prob = 0.5 * prob_rf + 0.5 * prob_lgb
    auc_ens = roc_auc_score(y_te, ens_prob)
    acc_ens = accuracy_score(y_te, (ens_prob > 0.5).astype(int))
    
    print(f"Top {top_k:2d} | RF: AUC={auc_rf:.4f} Acc={acc_rf:.4f} | LGB: AUC={auc_lgb:.4f} Acc={acc_lgb:.4f} | Ens: AUC={auc_ens:.4f} Acc={acc_ens:.4f}")
