/** Shared TypeScript types for API responses */

export interface ClassBalance {
  split: string;
  up_pct: number;
  down_pct: number;
  n: number;
}

export interface ModelMetrics {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  confusion_matrix?: number[][];
}

export interface Prediction {
  date: string;
  close: number;
  actual: number;
  lgbm_pred: number;
  lgbm_prob: number;
  correct: boolean;
}

export interface FeatureImportance {
  feature: string;
  lgbm_importance: number;
  lr_coef_abs: number;
}

export interface BacktestPoint {
  t: number;
  strategy: number;
  buy_hold: number;
}

export interface BacktestResult {
  equity_curve: BacktestPoint[];
  strategy_total_return: number;
  buy_hold_total_return: number;
  strategy_sharpe: number;
  strategy_max_dd: number;
  disclaimer: string;
}

export interface WalkForwardFold {
  fold: number;
  train_size: number;
  accuracy: number;
  roc_auc: number;
}

export interface CalibrationPoint {
  mean_predicted: number;
  fraction_positive: number;
}
