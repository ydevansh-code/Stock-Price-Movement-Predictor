# Stock Price Movement Predictor — Full Breakdown

## What the Task Asked For

The club task is a **binary classification problem** on financial time-series data:

> Given today's OHLCV data + technical indicators, predict whether tomorrow's closing price will be **up (1)** or **down (0)** compared to today.

### Required Deliverables (from the task sheet)
| Requirement | Status |
|---|---|
| Both naive baselines (persistence + majority) | ✅ |
| Raw-price model (ablation) | ✅ |
| Engineered-feature model | ✅ |
| Two models on final feature set | ✅ (RF + LightGBM) |
| Class balance reported | ✅ |
| Label shift shown explicitly in code | ✅ |
| Time-based split, scaler fit on training only | ✅ |
| Predicted vs actual direction plot | ✅ (frontend) |

---

## What We Actually Built

### Architecture Overview

```
backend/
├── src/
│   ├── config.py              ← all magic numbers (ticker, dates, split %, indicator periods)
│   ├── labels.py              ← leakage-safe target construction
│   ├── baselines.py           ← 2 naive baselines
│   ├── models.py              ← LR + RF + LightGBM (Optuna-tuned)
│   ├── evaluation.py          ← metrics, walk-forward, backtest, calibration
│   ├── pipeline.py            ← orchestrator — runs everything, writes JSON artifacts
│   ├── data/
│   │   ├── loader.py          ← downloads SPY OHLCV via yfinance, caches locally
│   │   └── validator.py       ← asserts column names, no NaNs, date ordering
│   └── features/
│       ├── raw_features.py         ← just OHLCV prices, no indicators (ablation baseline)
│       ├── engineered_features.py  ← v1: RSI, MACD, Bollinger, lags, day-of-week
│       └── engineered_features_v2.py ← v2: everything in v1 + ADX, overnight gap, multi-timeframe MAs, volume deviation, ATR-normalized return
├── artifacts/                 ← JSON outputs consumed by the frontend
└── tests/                     ← pytest unit tests for labels, features, split logic

frontend/
└── Next.js app that reads the JSON artifacts and renders charts
    ├── comparison table (all models side by side)
    ├── predicted vs actual direction chart
    ├── feature importance chart
    ├── walk-forward chart
    ├── backtest equity curve
    └── calibration curve
```

---

## The Data Flow (Step by Step)

```
1. loader.py          → downloads SPY daily OHLCV from 2018–2024 via yfinance
        ↓
2. validator.py       → checks shape, column names, no NaNs, sorted dates
        ↓
3. labels.py          → adds target = (close[t+1] > close[t])
                        drops the last row (no t+1 exists for it)
        ↓
4. features/*.py      → builds 3 different X matrices from the same labeled df
    - raw_features:       just OHLCV columns (no indicators)
    - engineered_v1:      RSI, MACD, BB-width, lags, day-of-week
    - engineered_v2:      v1 + ADX, overnight gap, stoch, OBV, ROC,
                          weekly/monthly returns, MA crossovers, volume dev, ATR-norm
        ↓
5. chronological_split()  → splits each X/y at the 80th percentile date
                            NO shuffling. Train = rows 0..80%, Test = rows 80..100%
        ↓
6. scale()            → StandardScaler fitted ONLY on train, .transform() on test
                        (prevents data leakage from test distribution)
        ↓
7. Models trained:
    - LogisticRegression   on raw features     (ablation — shows raw prices alone are useless)
    - LogisticRegression   on v1 features      (simple linear baseline with engineered features)
    - RandomForest         on v1 features      (non-linear, handles feature interactions)
    - LightGBM (Optuna)    on v2 features      (best model — gradient boosted trees, hyperparameter searched)
        ↓
8. Baselines computed:
    - Majority class baseline  → always predicts the more common direction
    - Persistence baseline     → predicts "tomorrow = today's direction"
        ↓
9. All results saved as JSON → artifacts/ folder
        ↓
10. Frontend reads artifacts → renders charts
```

---

## How the Label Is Built (Critical — No Leakage)

```python
# labels.py
out["target"] = (out["close"].shift(-1) > out["close"]).astype(int)
out = out.iloc[:-1]
```

- `shift(-1)` means: "align tomorrow's close with today's row"
- The final row is **dropped** because there's no t+1 for the last date
- Features built from row-t only use close/high/low/open/volume at time t
- **No future data is used in any feature** — this is what "label shift shown explicitly in code" means

---

## What Can Be Improved (and Why It's Safe)

### ✅ Already Fixed Today
- `class_weight="balanced"` on LR → stops model from being biased toward majority direction
- `subsample` + `colsample_bytree` in Optuna → better regularization for LightGBM, reduces overfitting
- Walk-forward now reports `roc_auc` per fold → accuracy alone is misleading on near-50/50 data
- `lgbm_pred/lgbm_prob` key names → was mislabeled as `rf_pred`, frontend was reading wrong data
- `LAG_DAYS` 5 → 10 → more autocorrelation context for the model

### ✅ Still Could Be Done (If Time Allows)
| Idea | Why It Helps | Risk |
|---|---|---|
| Increase `n_trials` in Optuna from 40 → 100 | Finds better hyperparams | Just slower (~4 min extra) |
| Add `min_periods` to all rolling windows | Makes warmup behavior explicit | Cosmetic only |
| Add `reg_alpha` / `reg_lambda` to Optuna search | L1/L2 regularization for LGBM | No risk, pure improvement |
| Try a wider RSI period (e.g., 21) alongside default 14 | Multi-period RSI as features | Safe, add as extra columns |
| Add `log_volume` feature (log of raw volume) | Compresses outliers, helps tree models | Easy 1-liner |

---

## What CANNOT Be Changed (Hard Constraints)

| Thing | Why It's Off-Limits |
|---|---|
| **`shift(-1)` in labels.py** | This IS the leakage-safe label. Change it and you have data leakage or wrong targets. |
| **Scaler fitted on train only** | Requirement. Fitting on full data leaks test distribution into training. |
| **Chronological split (no shuffle)** | Required by task. Shuffling destroys time ordering — model sees future data during training. |
| **Both baselines must be in comparison** | Required. They're the sanity check. |
| **No `pandas-ta`** | Explicitly banned in the task guidelines (fails on NumPy 2.x). |
| **Raw-price model must exist** | It's the ablation — shows that raw OHLCV alone is near-random. |
| **Two models on final feature set** | Currently LightGBM + RF. Both must be reported. |

---

## Why the Model Can't Get Much Better (Fundamental Ceiling)

The problem is predicting **next-day stock direction** in an efficient market (SPY = S&P 500 ETF).

- **Efficient market hypothesis**: all public information is already priced in
- **Near coin-flip**: SPY goes up ~53% of trading days historically
- **Expected AUC ceiling**: ~0.53–0.57 with technical indicators alone
- **Noise dominates**: daily returns are dominated by macro news, Fed announcements, geopolitical events — none of which are in OHLCV data

Any model claiming >0.60 AUC on daily SPY direction using only OHLCV features is likely **overfitting** or has a **data leakage bug**.

> The point of the task is NOT to build a profitable trading bot.  
> It's to demonstrate: correct label construction, leakage-free splits, honest baseline comparison, and feature engineering methodology.
