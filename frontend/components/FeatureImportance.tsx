"use client";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import type { FeatureImportance } from "../lib/types";

export default function FeatureImportanceChart({ data }: { data: FeatureImportance[] }) {
  const top = data.slice(0, 12);
  const maxVal = Math.max(...top.map((d) => d.lgbm_importance), 1);
  const xDomainMax = Math.ceil(maxVal * 1.22);

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Feature Importance (LightGBM v2)</h2>
          <p className="muted" style={{ margin: "0.25rem 0 0", fontSize: 13 }}>
            Top non-leaking predictors ranked by information gain across trees
          </p>
        </div>
        <span className="terminal-badge">Top 12 Signals</span>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={top}
          layout="vertical"
          margin={{ top: 5, right: 35, left: 15, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, xDomainMax]}
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            label={{
              value: "Relative Importance (Split Gain)",
              position: "insideBottom",
              offset: -12,
              fill: "var(--muted)",
              fontSize: 11,
              fontFamily: "monospace",
            }}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fill: "var(--text)", fontSize: 11, fontFamily: "monospace" }}
            width={125}
          />
          <Tooltip
            contentStyle={{ background: "var(--surface-solid)", border: "1px solid var(--border)", borderRadius: 6 }}
            formatter={(v) => [Number(v).toFixed(4), "Split Gain"]}
          />
          <Bar
            dataKey="lgbm_importance"
            fill="var(--terminal-amber)"
            radius={[0, 4, 4, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
