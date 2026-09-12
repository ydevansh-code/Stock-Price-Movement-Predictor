import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from src.data import loader
from src.features import engineered_features_v2 as eng_v2
from src.models import chronological_split, scale

raw_df = loader.load()

for h in [1, 3, 5]:
    df = raw_df.copy()
    df["target"] = (df["close"].shift(-h) > df["close"]).astype(int)
    df = df.iloc[:-h]
    
    X = eng_v2.build(df)
    y = df["target"].loc[X.index]
    
    X_tr, X_te, y_tr, y_te = chronological_split(X, y)
    Xs_tr, Xs_te, _ = scale(X_tr, X_te)
    
    rf_selector = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=15, random_state=42)
    rf_selector.fit(Xs_tr, y_tr)
    top25 = list(pd.Series(rf_selector.feature_importances_, index=X.columns).nlargest(25).index)
    
    Xtr25, Xte25 = Xs_tr[top25], Xs_te[top25]
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42)
    rf.fit(Xtr25, y_tr)
    p_rf = rf.predict_proba(Xte25)[:, 1]
    auc_rf = roc_auc_score(y_te, p_rf)
    acc_rf = accuracy_score(y_te, (p_rf > 0.5).astype(int))
    prec_rf = precision_score(y_te, (p_rf > 0.5).astype(int), zero_division=0)
    
    lgb = LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.03, num_leaves=6, min_child_samples=25, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    lgb.fit(Xtr25, y_tr)
    p_lgb = lgb.predict_proba(Xte25)[:, 1]
    auc_lgb = roc_auc_score(y_te, p_lgb)
    acc_lgb = accuracy_score(y_te, (p_lgb > 0.5).astype(int))
    prec_lgb = precision_score(y_te, (p_lgb > 0.5).astype(int), zero_division=0)
    
    p_ens = 0.5 * p_rf + 0.5 * p_lgb
    auc_ens = roc_auc_score(y_te, p_ens)
    acc_ens = accuracy_score(y_te, (p_ens > 0.5).astype(int))
    prec_ens = precision_score(y_te, (p_ens > 0.5).astype(int), zero_division=0)
    
    print(f"Horizon {h}d: RF AUC={auc_rf:.4f} Acc={acc_rf:.4f} | LGB AUC={auc_lgb:.4f} Acc={acc_lgb:.4f} | Ens AUC={auc_ens:.4f} Acc={acc_ens:.4f} Prec={prec_ens:.4f}")
