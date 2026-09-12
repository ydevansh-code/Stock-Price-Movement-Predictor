"use client";
import type { NextDaySignal } from "../lib/types";

interface NextDayCardProps {
  signal: NextDaySignal | null;
  onRefresh?: () => void;
}

export default function NextDayCard({ signal, onRefresh }: NextDayCardProps) {
  if (!signal) return null;

  const isUp = signal.predicted_direction === "UP";
  const probPct = (signal.probability_up * 100).toFixed(1);

  return (
    <div
      className="card"
      style={{
        background: isUp
          ? "linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(15, 23, 42, 0.9) 60%)"
          : "linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(15, 23, 42, 0.9) 60%)",
        border: isUp ? "1px solid rgba(16, 185, 129, 0.35)" : "1px solid rgba(239, 68, 68, 0.35)",
        padding: "1.5rem",
        display: "flex",
        flexDirection: "column",
        gap: "1.25rem",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.3rem" }}>
            <span className={isUp ? "terminal-badge-green" : "terminal-badge"}>
              LIVE FORWARD INFERENCE
            </span>
            <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
              AS OF {signal.as_of_date} CLOSE (${signal.latest_close})
            </span>
          </div>
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>
            Next Session Forecast: {signal.next_trading_day} ({signal.ticker})
          </h2>
        </div>

        {onRefresh && (
          <button
            onClick={onRefresh}
            className="btn-terminal"
            style={{ fontSize: 11, padding: "0.35rem 0.75rem" }}
          >
            ⚡ Re-Query Live
          </button>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <div
            style={{
              width: 52,
              height: 52,
              borderRadius: "50%",
              background: isUp ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.2)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 24,
              color: isUp ? "var(--bull)" : "var(--bear)",
              boxShadow: isUp ? "0 0 16px rgba(16, 185, 129, 0.3)" : "0 0 16px rgba(239, 68, 68, 0.3)",
            }}
          >
            {isUp ? "▲" : "▼"}
          </div>
          <div>
            <div className="mono" style={{ fontSize: 24, fontWeight: 800, color: isUp ? "var(--bull)" : "var(--bear)" }}>
              {isUp ? "PREDICTED UP" : "PREDICTED DOWN"}
            </div>
            <div style={{ fontSize: 12.5, color: "var(--muted)" }}>
              {probPct}% Bullish Probability · {signal.confidence_level}
            </div>
          </div>
        </div>

        <div style={{ background: "rgba(0,0,0,0.25)", padding: "0.85rem 1rem", borderRadius: 6, border: "1px solid var(--border)" }}>
          <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
            Multi-Model Consensus
          </div>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", fontSize: 12 }}>
            <div>
              <span style={{ color: "var(--muted)" }}>LGBM v2: </span>
              <span className="mono" style={{ fontWeight: 600, color: signal.model_signals.lightgbm_v2.dir === "UP" ? "var(--bull)" : "var(--bear)" }}>
                {signal.model_signals.lightgbm_v2.dir} ({(signal.model_signals.lightgbm_v2.prob * 100).toFixed(1)}%)
              </span>
            </div>
            <div>
              <span style={{ color: "var(--muted)" }}>RF v2: </span>
              <span className="mono" style={{ fontWeight: 600, color: signal.model_signals.random_forest_v2.dir === "UP" ? "var(--bull)" : "var(--bear)" }}>
                {signal.model_signals.random_forest_v2.dir} ({(signal.model_signals.random_forest_v2.prob * 100).toFixed(1)}%)
              </span>
            </div>
            <div>
              <span style={{ color: "var(--muted)" }}>LR v2: </span>
              <span className="mono" style={{ fontWeight: 600, color: signal.model_signals.logreg_v2.dir === "UP" ? "var(--bull)" : "var(--bear)" }}>
                {signal.model_signals.logreg_v2.dir} ({(signal.model_signals.logreg_v2.prob * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>

        <div style={{ background: "rgba(0,0,0,0.25)", padding: "0.85rem 1rem", borderRadius: 6, border: "1px solid var(--border)" }}>
          <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
            Latest Market Inputs
          </div>
          <div style={{ display: "flex", gap: "1.25rem", fontSize: 12 }}>
            <div>
              <span style={{ color: "var(--muted)" }}>RSI (14): </span>
              <span className="mono" style={{ fontWeight: 600, color: "var(--text)" }}>{signal.key_features.rsi_14}</span>
            </div>
            <div>
              <span style={{ color: "var(--muted)" }}>5d Return: </span>
              <span className="mono" style={{ fontWeight: 600, color: signal.key_features.ret_5d >= 0 ? "var(--bull)" : "var(--bear)" }}>
                {signal.key_features.ret_5d > 0 ? "+" : ""}{signal.key_features.ret_5d}%
              </span>
            </div>
            <div>
              <span style={{ color: "var(--muted)" }}>Macro VIX: </span>
              <span className="mono" style={{ fontWeight: 600, color: "var(--text)" }}>{signal.key_features.vix}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
