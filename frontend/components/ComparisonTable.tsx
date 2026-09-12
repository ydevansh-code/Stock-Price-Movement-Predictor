"use client";
import type { ModelMetrics } from "../lib/types";

function Badge({ value, isBest }: { value: number; isBest: boolean }) {
  return (
    <span
      className="mono"
      style={{
        background: isBest ? "rgba(16, 185, 129, 0.15)" : "transparent",
        color: isBest ? "var(--terminal-emerald)" : "inherit",
        borderRadius: 4,
        padding: "2px 6px",
        fontWeight: isBest ? 700 : 400,
        border: isBest ? "1px solid rgba(16, 185, 129, 0.3)" : "1px solid transparent",
      }}
    >
      {value.toFixed(4)}
    </span>
  );
}

export default function ComparisonTable({ rows }: { rows: ModelMetrics[] }) {
  const cols: (keyof ModelMetrics)[] = ["accuracy", "precision", "recall", "f1", "roc_auc"];

  const candidateRows = rows.filter((r) => !["Majority Class", "Persistence"].includes(r.model));

  const bests: Record<string, number> = {};
  for (const c of cols) {
    const vals = candidateRows.map((r) => Number(r[c] ?? 0));
    bests[c] = vals.length > 0 ? Math.max(...vals) : 0;
  }

  return (
    <div className="card" style={{ padding: "1.25rem", overflow: "hidden" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Model Benchmark & Out-of-Sample Metrics</h2>
          <p className="muted" style={{ margin: "0.25rem 0 0", fontSize: 13 }}>
            Evaluated on strictly unseen chronological holdout · Green highlight denotes best performance
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <span className="terminal-badge-green">Strict OOS</span>
          <span className="terminal-badge">Zero Leakage</span>
        </div>
      </div>

      <div style={{ width: "100%", overflowX: "auto", paddingBottom: "0.5rem" }}>
        <table style={{ width: "100%", minWidth: 640, borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", background: "rgba(0,0,0,0.2)" }}>
              <th style={{ textAlign: "left", padding: "0.75rem 0.5rem", color: "var(--muted)", fontWeight: 600, fontSize: 12 }}>
                ARCHITECTURE
              </th>
              {cols.map((c) => (
                <th key={c} style={{ textAlign: "right", padding: "0.75rem 0.5rem", color: "var(--muted)", fontWeight: 600, fontSize: 12 }}>
                  {c === "roc_auc" ? "ROC-AUC" : c.toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const isBaseline = ["Majority Class", "Persistence"].includes(row.model);
              const isLightGBM = row.model.includes("LightGBM");

              return (
                <tr
                  key={row.model}
                  style={{
                    borderBottom: "1px solid var(--border)",
                    background: isLightGBM
                      ? "rgba(16, 185, 129, 0.04)"
                      : isBaseline
                      ? "rgba(0, 0, 0, 0.25)"
                      : i % 2 === 0
                      ? "rgba(255, 255, 255, 0.015)"
                      : "transparent",
                    opacity: isBaseline ? 0.75 : 1,
                  }}
                >
                  <td style={{ padding: "0.65rem 0.5rem", fontWeight: 500 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <span>{row.model}</span>
                      {isLightGBM && (
                        <span className="terminal-badge-green" style={{ fontSize: 9, padding: "1px 4px" }}>
                          LEADER
                        </span>
                      )}
                      {isBaseline && (
                        <span style={{ fontSize: 10, color: "var(--muted)", border: "1px solid var(--border)", borderRadius: 3, padding: "0px 4px" }}>
                          NAIVE
                        </span>
                      )}
                    </div>
                  </td>
                  {cols.map((c) => {
                    const val = Number(row[c] ?? 0);
                    const isBest = !isBaseline && Math.abs(val - bests[c]) < 0.0001;
                    return (
                      <td key={c} style={{ textAlign: "right", padding: "0.65rem 0.5rem" }}>
                        <Badge value={val} isBest={isBest} />
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
