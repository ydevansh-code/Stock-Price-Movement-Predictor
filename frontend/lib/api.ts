/** Type-safe fetch helpers for the FastAPI backend. */

const BASE = "/api";

async function get<T>(path: string, market: string = "us"): Promise<T> {
  const res = await fetch(`${BASE}${path}?market=${market}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  classBalance: (market = "us") => get<import("./types").ClassBalance[]>("/class-balance", market),
  comparisonTable: (market = "us") => get<import("./types").ModelMetrics[]>("/comparison-table", market),
  predictions: (market = "us") => get<import("./types").Prediction[]>("/predictions", market),
  featureImportance: (market = "us") => get<import("./types").FeatureImportance[]>("/feature-importance", market),
  backtest: (market = "us") => get<import("./types").BacktestResult>("/backtest", market),
  walkForward: (market = "us") => get<import("./types").WalkForwardFold[]>("/walk-forward", market),
  calibration: (market = "us") => get<import("./types").CalibrationPoint[]>("/calibration", market),
  nextDay: (market = "us") => get<import("./types").NextDaySignal>("/next-day", market),
};
