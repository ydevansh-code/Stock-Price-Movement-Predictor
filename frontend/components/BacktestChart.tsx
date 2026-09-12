"use client";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import type { BacktestResult } from "../lib/types";

export default function BacktestChart({ result }: { result: BacktestResult }) {
  const data = result.equity_curve.map((p) => ({
    ...p,
    strategy: (p.strategy - 1) * 100,
    buy_hold: (p.buy_hold - 1) * 100,
  }));

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "0.5rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Cumulative Return Simulation (Test Split)</h2>
          <p className="muted" style={{ margin: "0.25rem 0 0", fontSize: 13 }}>
            Theoretical equity curve on out-of-sample data (2022 – 2026)
          </p>
        </div>
        <span className="terminal-badge">Execution Cost: 5 bps</span>
      </div>

      <div style={{ background: "rgba(0,0,0,0.25)", border: "1px solid var(--border)", borderRadius: 6, padding: "0.75rem 1rem" }}>
        <p style={{ margin: 0, fontSize: 12.5, color: "var(--muted)", lineHeight: 1.5 }}>
          <strong style={{ color: "var(--terminal-amber)" }}>Empirical Finding:</strong> Strategy return (+{(result.strategy_total_return * 100).toFixed(1)}%) underperforms unhedged Buy & Hold (+{(result.buy_hold_total_return * 100).toFixed(1)}%) during sustained secular expansion. The meta-labeling filter prioritizes downside tail-risk containment over unrestricted bull-market beta exposure.
        </p>
      </div>

      <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap" }}>
        {[
          ["Strategy Return", `${(result.strategy_total_return * 100).toFixed(1)}%`, "var(--terminal-emerald)"],
          ["Buy & Hold Beta", `${(result.buy_hold_total_return * 100).toFixed(1)}%`, "var(--terminal-amber)"],
          ["Strategy Sharpe", result.strategy_sharpe.toFixed(2), "var(--text)"],
          ["Max Drawdown", `${(result.strategy_max_dd * 100).toFixed(1)}%`, "var(--bear)"],
        ].map(([k, v, color]) => (
          <div key={k} style={{ minWidth: 120 }}>
            <p className="muted" style={{ fontSize: 11, margin: 0, textTransform: "uppercase" }}>{k}</p>
            <p className="mono" style={{ fontSize: 18, fontWeight: 700, margin: "2px 0 0", color }}>{v}</p>
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="t" tick={{ fill: "var(--muted)", fontSize: 11 }} interval={Math.floor(data.length / 6)} />
          <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(v) => `${v.toFixed(0)}%`} />
          <Tooltip
            contentStyle={{ background: "var(--surface-solid)", border: "1px solid var(--border)", borderRadius: 6 }}
            formatter={(v) => [`${Number(v).toFixed(2)}%`]}
          />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          <Line type="monotone" dataKey="strategy" stroke="var(--terminal-emerald)" dot={false} strokeWidth={2} name="Meta-Labeled Strategy" />
          <Line type="monotone" dataKey="buy_hold" stroke="var(--terminal-amber)" dot={false} strokeWidth={1.75} name="Unhedged Buy & Hold" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
