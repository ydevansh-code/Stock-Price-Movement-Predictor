"use client";
import { useState } from "react";

const links = [
  { id: "thesis", label: "Research Thesis", icon: "📑" },
  { id: "overview", label: "Live Terminal", icon: "📊" },
  { id: "comparison", label: "Model Benchmark", icon: "⚡" },
  { id: "backtest", label: "Backtest Simulation", icon: "📈" },
  { id: "features", label: "Features & Calibration", icon: "🎯" },
];

export default function Sidebar({ active, setActive }: {
  active: string;
  setActive: (id: string) => void;
}) {
  const [open, setOpen] = useState(true);

  return (
    <aside
      style={{
        width: open ? 240 : 64,
        minHeight: "100vh",
        background: "var(--surface-solid)",
        borderRight: "1px solid var(--border)",
        transition: "width 0.2s ease",
        display: "flex",
        flexDirection: "column",
        padding: "1.25rem 0",
        flexShrink: 0,
        zIndex: 20,
      }}
    >
      <div style={{ padding: "0 1rem", display: "flex", alignItems: "center", justifyContent: open ? "space-between" : "center" }}>
        {open && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--terminal-amber)", boxShadow: "0 0 8px var(--terminal-amber)" }} />
            <span className="mono" style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.08em", color: "var(--text)" }}>ALPHA TERM</span>
          </div>
        )}
        <button
          onClick={() => setOpen(!open)}
          style={{
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: 4,
            color: "var(--muted)",
            cursor: "pointer",
            width: 26,
            height: 26,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 11,
          }}
          aria-label="Toggle sidebar"
        >
          {open ? "◀" : "▶"}
        </button>
      </div>

      <div style={{ marginTop: "1.75rem", display: "flex", flexDirection: "column", gap: "0.3rem", padding: "0 0.5rem" }}>
        {links.map((l) => {
          const isCurrent = active === l.id;
          return (
            <button
              key={l.id}
              onClick={() => setActive(l.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                width: "100%",
                background: isCurrent ? "rgba(245, 158, 11, 0.12)" : "transparent",
                border: isCurrent ? "1px solid rgba(245, 158, 11, 0.3)" : "1px solid transparent",
                borderRadius: 6,
                color: isCurrent ? "var(--terminal-amber)" : "var(--muted)",
                cursor: "pointer",
                padding: "0.6rem 0.8rem",
                textAlign: "left",
                fontSize: 13,
                fontWeight: isCurrent ? 600 : 400,
                whiteSpace: "nowrap",
                overflow: "hidden",
                transition: "all 0.15s ease",
              }}
            >
              <span style={{ fontSize: 14 }}>{l.icon}</span>
              {open && <span>{l.label}</span>}
            </button>
          );
        })}
      </div>

      {open && (
        <div style={{ marginTop: "auto", padding: "1rem", borderTop: "1px solid var(--border)", color: "var(--muted)", fontSize: 11, display: "flex", flexDirection: "column", gap: "0.3rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--terminal-emerald)" }} />
            <span className="mono" style={{ fontWeight: 600, color: "var(--text)" }}>QUANT ENGINE V2</span>
          </div>
          <span style={{ color: "var(--muted)" }}>Purged Chrono Holdout</span>
        </div>
      )}
    </aside>
  );
}
