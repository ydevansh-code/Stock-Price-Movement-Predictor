"use client";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { WalkForwardFold } from "../lib/types";

export default function WalkForwardChart({ folds }: { folds: WalkForwardFold[] }) {
  return (
    <div className="card">
      <h2 style={{ margin: "0 0 0.25rem", fontSize: 16, fontWeight: 600 }}>Walk-Forward Validation</h2>
      <p className="muted" style={{ margin: "0 0 1rem", fontSize: 13 }}>
        Accuracy per expanding fold — shows stability (or instability) over time.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={folds}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="fold" tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(v) => `Fold ${v}`} />
          <YAxis
            domain={[0.45, 0.65]}
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
            formatter={(v) => [`${(Number(v) * 100).toFixed(1)}%`, "Accuracy"]}
          />
          <Bar dataKey="accuracy" fill="var(--bull)" radius={[4, 4, 0, 0]} name="Accuracy" />
          <Bar dataKey="roc_auc" fill="#6366f1" radius={[4, 4, 0, 0]} name="ROC-AUC" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
