"use client";
import { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { Prediction } from "../lib/types";

interface DirectionChartProps {
  data: Prediction[];
}

export default function DirectionChart({ data }: DirectionChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const chartData = data.map((d) => ({
    date: d.date.slice(5),
    fullDate: d.date,
    price: d.close,
    correct: d.correct,
  }));

  const correctCount = data.filter((d) => d.correct).length;
  const accuracyPct = ((correctCount / data.length) * 100).toFixed(1);

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>
            Price Trajectory & Directional Accuracy Strip
          </h2>
          <p className="muted" style={{ margin: "0.25rem 0 0", fontSize: 13 }}>
            Continuous market asset price (top) paired with decoupled directional hit strip (bottom)
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div style={{ textAlign: "right" }}>
            <span className="mono" style={{ fontSize: 14, fontWeight: 700, color: "var(--terminal-emerald)" }}>
              {accuracyPct}%
            </span>
            <span className="muted" style={{ fontSize: 11, marginLeft: 4 }}>
              ({correctCount}/{data.length} hits)
            </span>
          </div>
        </div>
      </div>

      <div>
        <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
          Asset Close Price
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="date"
              tick={{ fill: "var(--muted)", fontSize: 11 }}
              interval={Math.floor(chartData.length / 8)}
            />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fill: "var(--muted)", fontSize: 11 }}
              tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{ background: "var(--surface-solid)", border: "1px solid var(--border)", borderRadius: 6 }}
              labelStyle={{ color: "var(--muted)", fontSize: 12 }}
              formatter={(v) => [`$${Number(v).toFixed(2)}`, "Close Price"]}
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke="var(--terminal-amber)"
              fillOpacity={1}
              fill="url(#priceGradient)"
              strokeWidth={1.75}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Model Precision Strip (Green = Correct Signal · Dark Slate = Miss)
          </div>
          {hoveredIndex !== null && (
            <div className="mono" style={{ fontSize: 11, color: data[hoveredIndex].correct ? "var(--bull)" : "var(--bear)" }}>
              {data[hoveredIndex].date}: {data[hoveredIndex].correct ? "CORRECT" : "MISS"} (Pred: {data[hoveredIndex].lgbm_pred})
            </div>
          )}
        </div>
        <div className="heartbeat-strip">
          {data.map((d, i) => (
            <div
              key={i}
              className="heartbeat-bar"
              onMouseEnter={() => setHoveredIndex(i)}
              onMouseLeave={() => setHoveredIndex(null)}
              style={{
                background: d.correct ? "var(--terminal-emerald)" : "rgba(100, 116, 139, 0.28)",
              }}
              title={`${d.date}: ${d.correct ? "Hit" : "Miss"}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
