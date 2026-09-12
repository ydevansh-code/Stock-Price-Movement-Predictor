"use client";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { CalibrationPoint } from "../lib/types";

export default function CalibrationChart({ data }: { data: CalibrationPoint[] }) {
  const chartData = data.map((p) => ({
    predicted: p.mean_predicted,
    actual: p.fraction_positive,
  }));

  const perfect = [
    { predicted: 0, actual: 0 },
    { predicted: 1, actual: 1 },
  ];

  return (
    <div className="card">
      <h2 style={{ margin: "0 0 0.25rem", fontSize: 16, fontWeight: 600 }}>Probability Calibration</h2>
      <p className="muted" style={{ margin: "0 0 1rem", fontSize: 13 }}>
        Perfect calibration = diagonal. Deviation shows over/under-confidence.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="predicted"
            type="number"
            domain={[0, 1]}
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: "Mean Predicted Prob", position: "insideBottom", offset: -2, fill: "var(--muted)", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
            formatter={(v) => [Number(v).toFixed(3)]}
          />
          <Line
            data={perfect}
            type="linear"
            dataKey="actual"
            stroke="var(--muted)"
            strokeDasharray="4 4"
            dot={false}
            name="Perfect"
          />
          <Line
            data={chartData}
            type="monotone"
            dataKey="actual"
            stroke="var(--bull)"
            strokeWidth={2}
            dot={{ fill: "var(--bull)", r: 4 }}
            name="RF Model"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
