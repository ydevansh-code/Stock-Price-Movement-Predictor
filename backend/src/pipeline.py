"""
End-to-end pipeline. Run with:  python -m src.pipeline
Produces all JSON artifacts in backend/artifacts/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Parse args and set env var BEFORE importing config
parser = argparse.ArgumentParser(description="Stock Direction Predictor Pipeline")
parser.add_argument("--market", type=str, default="us", choices=["us", "ind"], help="Market profile to use")
args = parser.parse_args()
os.environ["MARKET"] = args.market

from src import baselines, evaluation
from src.config import ARTIFACTS, RANDOM_SEED, TEST_SPLIT, TICKER
from src.data import loader, validator
from src.features import engineered_features as eng
from src.features import engineered_features_v2 as eng_v2
from src.features import raw_features as raw
from src.labels import make_labels
from src.models import (
    chronological_split,
    scale,
    train_lightgbm,
    train_logistic,
    train_random_forest,
)

np.random.seed(RANDOM_SEED)


def _save(name: str, data) -> None:
    path = ARTIFACTS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ {path.name}")


def run() -> None:
    print(f"\n{'='*50}\nStock Direction Predictor Pipeline — {TICKER}\n{'='*50}")

    # ── 1. Load & validate ────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df = loader.load()
    validator.validate(df)
    print(f"    {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")

    ohlcv_records = df.reset_index()[["date", "open", "high", "low", "close", "volume"]]
    ohlcv_records["date"] = ohlcv_records["date"].astype(str)
    _save("ohlcv", ohlcv_records.to_dict(orient="records"))

    # ── 2. Labels ─────────────────────────────────────────────────────────────
    print("\n[2] Building labels...")
    labeled = make_labels(df)
    y_all = labeled["target"]

    # ── 3. Class balance ──────────────────────────────────────────────────────
    print("\n[3] Class balance...")
    split_idx = int(len(y_all) * (1 - TEST_SPLIT))
    balance = [
        evaluation.class_balance(y_all, "all"),
        evaluation.class_balance(y_all.iloc[:split_idx], "train"),
        evaluation.class_balance(y_all.iloc[split_idx:], "test"),
    ]
    _save("class_balance", balance)

    # ── 4. Raw-price ablation ─────────────────────────────────────────────────
    print("\n[4] Raw features (ablation)...")
    X_raw = raw.build(labeled)
    y_raw = y_all.loc[X_raw.index]
    X_raw_tr, X_raw_te, y_raw_tr, y_raw_te = chronological_split(X_raw, y_raw)
    Xr_tr_s, Xr_te_s, _ = scale(X_raw_tr, X_raw_te, "scaler_raw")
    lr_raw = train_logistic(Xr_tr_s, y_raw_tr, "lr_raw")
    raw_pred = lr_raw.predict(Xr_te_s)
    raw_prob = lr_raw.predict_proba(Xr_te_s)[:, 1]
    del X_raw, X_raw_tr, X_raw_te, Xr_tr_s, Xr_te_s, lr_raw

    # ── 5. Engineered v1 (LR + RF) ───────────────────────────────────────────
    print("\n[5] Engineered v1 features (LR + RF)...")
    X_eng = eng.build(labeled)
    y_eng = y_all.loc[X_eng.index]
    X_eng_tr, X_eng_te, y_eng_tr, y_eng_te = chronological_split(X_eng, y_eng)
    Xe_tr_s, Xe_te_s, _ = scale(X_eng_tr, X_eng_te, "scaler_eng")
    lr_eng = train_logistic(Xe_tr_s, y_eng_tr, "lr_eng")
    rf_eng = train_random_forest(Xe_tr_s, y_eng_tr)
    lr_pred = lr_eng.predict(Xe_te_s)
    lr_prob = lr_eng.predict_proba(Xe_te_s)[:, 1]
    rf_pred = rf_eng.predict(Xe_te_s)
    rf_prob = rf_eng.predict_proba(Xe_te_s)[:, 1]
    del X_eng, X_eng_tr, X_eng_te, Xe_tr_s, Xe_te_s, lr_eng, rf_eng

    # ── 6. Engineered v2 + LightGBM (Optuna-tuned) ───────────────────────────
    print("\n[6] Engineered v2 features + LightGBM (Optuna tuning, ~2 min)...")
    X_v2 = eng_v2.build(labeled)
    y_v2 = y_all.loc[X_v2.index]
    X_v2_tr, X_v2_te, y_v2_tr, y_v2_te = chronological_split(X_v2, y_v2)
    # LightGBM handles its own scaling internally via histogram binning,
    # but we still scale for consistency with the other models in this pipeline.
    Xv_tr_s, Xv_te_s, _ = scale(X_v2_tr, X_v2_te, "scaler_v2")

    from src.features.feature_selector import select_top_features
    top25 = select_top_features(Xv_tr_s, y_v2_tr, top_k=25)
    _save("top_features", top25)

    Xv_tr_sub = Xv_tr_s[top25]
    Xv_te_sub = Xv_te_s[top25]

    lgbm = train_lightgbm(Xv_tr_sub, y_v2_tr.values, n_trials=40)
    lgbm_pred = lgbm.predict(Xv_te_sub)
    lgbm_prob = lgbm.predict_proba(Xv_te_sub)[:, 1]
    lr_v2 = train_logistic(Xv_tr_sub, y_v2_tr, "lr_v2")
    lr_v2_pred = lr_v2.predict(Xv_te_sub)
    lr_v2_prob = lr_v2.predict_proba(Xv_te_sub)[:, 1]
    rf_v2 = train_random_forest(Xv_tr_sub, y_v2_tr, tag="rf_v2", max_depth=4, min_samples_leaf=15)
    rf_v2_pred = rf_v2.predict(Xv_te_sub)
    rf_v2_prob = rf_v2.predict_proba(Xv_te_sub)[:, 1]
    ens_prob = 0.45 * lgbm_prob + 0.45 * rf_v2_prob + 0.10 * lr_v2_prob
    ens_pred = (ens_prob > 0.5).astype(int)

    # ── 7. Baselines ──────────────────────────────────────────────────────────
    print("\n[7] Baselines...")
    test_idx = y_eng_te.index
    today_dir = (df["close"].diff(1) > 0).astype(int).loc[test_idx]
    persist = baselines.persistence_baseline(y_eng_tr, y_eng_te, today_dir)
    majority = baselines.majority_baseline(y_eng_tr, y_eng_te)

    # ── 8. Comparison table ───────────────────────────────────────────────────
    print("\n[8] Comparison table...")
    table = [
        persist,
        majority,
        evaluation.metrics("Raw LR (ablation)", y_raw_te.values, raw_pred, raw_prob),
        evaluation.metrics("Engineered LR", y_eng_te.values, lr_pred, lr_prob),
        evaluation.metrics("Engineered RF", y_eng_te.values, rf_pred, rf_prob),
        evaluation.metrics("LR v2 (Top-25)", y_v2_te.values, lr_v2_pred, lr_v2_prob),
        evaluation.metrics("RF v2 (Top-25 Pruned)", y_v2_te.values, rf_v2_pred, rf_v2_prob),
        evaluation.metrics("LightGBM v2 (Top-25)", y_v2_te.values, lgbm_pred, lgbm_prob),
        evaluation.metrics("Ensemble v2 (Top-25)", y_v2_te.values, ens_pred, ens_prob),
    ]
    _save("comparison_table", table)

    # ── 9. Predictions (use best model = LGBM v2) ─────────────────────────────
    print("\n[9] Predictions...")
    test_dates = y_v2_te.index.astype(str).tolist()
    test_close = labeled["close"].loc[y_v2_te.index].tolist()
    preds_out = [
        {
            "date": test_dates[i],
            "close": round(test_close[i], 4),
            "actual": int(y_v2_te.values[i]),
            "lgbm_pred": int(lgbm_pred[i]),
            "lgbm_prob": round(float(lgbm_prob[i]), 4),
            "correct": int(y_v2_te.values[i]) == int(lgbm_pred[i]),
        }
        for i in range(len(test_dates))
    ]
    _save("predictions", preds_out)

    # ── 10. Feature importance (LGBM v2) ─────────────────────────────────────
    print("\n[10] Feature importance...")
    feat_names = list(Xv_tr_sub.columns)
    lgbm_imp = lgbm.feature_importances_
    lr_v2_coef = np.abs(lr_v2.coef_[0])
    fi_out = [
        {
            "feature": feat_names[i],
            "lgbm_importance": round(float(lgbm_imp[i]), 6),
            "lr_coef_abs": round(float(lr_v2_coef[i]), 6),
        }
        for i in range(len(feat_names))
    ]
    fi_out.sort(key=lambda x: x["lgbm_importance"], reverse=True)
    _save("feature_importance", fi_out)

    try:
        import shap
        explainer = shap.TreeExplainer(lgbm)
        shap_vals = explainer.shap_values(Xv_te_sub)
        sv = shap_vals[1] if isinstance(shap_vals, list) else (shap_vals[:, :, 1] if len(shap_vals.shape) == 3 else shap_vals)
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        shap_out = [
            {"feature": feat_names[i], "mean_abs_shap": round(float(mean_abs_shap[i]), 6)}
            for i in range(len(feat_names))
        ]
        shap_out.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        _save("shap_summary", shap_out)
    except Exception:
        pass

    # ── 11. Walk-forward (LGBM v2) ────────────────────────────────────────────
    print("\n[11] Walk-forward validation...")
    import lightgbm as lgb_mod

    _tuned_params = lgbm.get_params()
    def _lgbm_fn(X_tr, y_tr):
        m = lgb_mod.LGBMClassifier(**_tuned_params)
        m.fit(X_tr, y_tr)
        return m

    wf = evaluation.walk_forward(Xv_tr_sub, y_v2_tr.values, _lgbm_fn)
    _save("walk_forward", wf)

    # ── 12. Backtest ──────────────────────────────────────────────────────────
    print("\n[12] Backtest...")
    bt = evaluation.backtest(labeled["close"].loc[y_v2_te.index], lgbm_pred)
    _save("backtest", bt)

    # ── 13. Calibration ───────────────────────────────────────────────────────
    print("\n[13] Calibration...")
    cal = evaluation.calibration(y_v2_te.values, lgbm_prob)
    _save("calibration", cal)

    # ── 14. Multi-Horizon Analysis ────────────────────────────────────────────
    print("\n[14] Multi-horizon analysis (1d, 3d, 5d)...")
    from sklearn.ensemble import RandomForestClassifier
    multi_horizon = []
    for h in [1, 3, 5]:
        h_df = make_labels(df, horizon=h)
        h_y = h_df["target"].loc[X_v2.index.intersection(h_df.index)]
        h_X = X_v2.loc[h_y.index]
        h_X_tr, h_X_te, h_y_tr, h_y_te = chronological_split(h_X, h_y, gap=h-1)
        h_Xs_tr, h_Xs_te, _ = scale(h_X_tr, h_X_te, f"scaler_h{h}")
        h_top25 = select_top_features(h_Xs_tr, h_y_tr, top_k=25)
        h_Xtr25, h_Xte25 = h_Xs_tr[h_top25], h_Xs_te[h_top25]

        h_rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1)
        h_rf.fit(h_Xtr25, h_y_tr)
        p_rf = h_rf.predict_proba(h_Xte25)[:, 1]

        h_lgb = lgb_mod.LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.03, num_leaves=6, min_child_samples=25, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        h_lgb.fit(h_Xtr25, h_y_tr)
        p_lgb = h_lgb.predict_proba(h_Xte25)[:, 1]

        p_ens = 0.5 * p_rf + 0.5 * p_lgb
        
        h_test_idx = h_y_te.index
        h_today_dir = (df["close"].diff(h) > 0).astype(int).loc[h_test_idx]
        h_persist = baselines.persistence_baseline(h_y_tr, h_y_te, h_today_dir)
        h_majority = baselines.majority_baseline(h_y_tr, h_y_te)

        multi_horizon.append({
            "horizon_days": h,
            "persist": h_persist,
            "majority": h_majority,
            "rf": evaluation.metrics(f"RF {h}d", h_y_te.values, (p_rf > 0.5).astype(int), p_rf),
            "lgbm": evaluation.metrics(f"LightGBM {h}d", h_y_te.values, (p_lgb > 0.5).astype(int), p_lgb),
            "ensemble": evaluation.metrics(f"Ensemble {h}d", h_y_te.values, (p_ens > 0.5).astype(int), p_ens),
        })
    _save("multi_horizon_comparison", multi_horizon)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Pipeline complete. Artifacts saved to:", ARTIFACTS)
    real_rows = [r for r in table if r.get("roc_auc", 0) > 0.5]
    if real_rows:
        best = max(real_rows, key=lambda r: r.get("roc_auc", 0))
        print(f"\nBest model: {best['model']}  |  Acc={best['accuracy']}  ROC-AUC={best.get('roc_auc', 'N/A')}")
    print(
        "\nNOTE: Next-day direction is a near-coin-flip problem in efficient markets.\n"
        "Small edges above 0.50 AUC are expected; they do not imply real-world profitability.\n"
    )


if __name__ == "__main__":
    run()
