"use client";

interface KpiCardProps {
  label: string;
  value: string;
  sub?: string;
  badge?: string;
  sparkline?: number[];
  color?: string;
}

function MiniSparkline({ data, strokeColor }: { data: number[]; strokeColor: string }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const width = 64;
  const height = 18;
  const step = width / (data.length - 1);

  const points = data
    .map((v, i) => {
      const x = i * step;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <polyline
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

function KpiCard({ label, value, sub, badge, sparkline, color }: KpiCardProps) {
  return (
    <div
      className="card"
      style={{
        flex: 1,
        minWidth: 160,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        gap: "0.5rem",
        padding: "1rem 1.15rem",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>
          {label}
        </span>
        {badge && (
          <span style={{ fontSize: 10, padding: "1px 5px", borderRadius: 3, background: "rgba(245,158,11,0.1)", color: "var(--terminal-amber)", border: "1px solid rgba(245,158,11,0.25)", fontFamily: "monospace" }}>
            {badge}
          </span>
        )}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span className="mono" style={{ fontSize: 24, fontWeight: 700, color: color ?? "var(--text)" }}>
          {value}
        </span>
        {sparkline && (
          <MiniSparkline
            data={sparkline}
            strokeColor={color ?? "var(--terminal-amber)"}
          />
        )}
      </div>

      {sub && (
        <span style={{ fontSize: 11.5, color: "var(--muted)" }}>
          {sub}
        </span>
      )}
    </div>
  );
}

interface KpiRowProps {
  bestAcc: number;
  bestModel: string;
  baselineFloor: number;
  persistAcc: number;
  upPct: number;
  testRange: string;
}

export default function KpiRow({ bestAcc, bestModel, baselineFloor, persistAcc, upPct, testRange }: KpiRowProps) {
  const edgeOverBaseline = ((bestAcc - baselineFloor) * 100).toFixed(1);

  return (
    <div style={{ display: "flex", gap: "0.85rem", flexWrap: "wrap" }}>
      <KpiCard
        label="Best Model Acc"
        value={`${(bestAcc * 100).toFixed(1)}%`}
        sub={`${bestModel} · +${edgeOverBaseline}% edge`}
        badge="LEADER"
        color="var(--terminal-emerald)"
        sparkline={[50.2, 51.1, 51.8, 52.4, 52.1, 53.1]}
      />
      <KpiCard
        label="Majority Baseline"
        value={`${(baselineFloor * 100).toFixed(1)}%`}
        sub="Naive static floor"
        badge="FLOOR"
        color="var(--muted)"
        sparkline={[49.2, 49.2, 49.2, 49.2, 49.2, 49.2]}
      />
      <KpiCard
        label="Persistence Acc"
        value={`${(persistAcc * 100).toFixed(1)}%`}
        sub="1-Day momentum drift"
        badge="NAIVE"
        sparkline={[49.1, 49.8, 48.9, 50.1, 49.4]}
      />
      <KpiCard
        label="Class Balance"
        value={`${(upPct * 100).toFixed(1)}%`}
        sub="Positive labels (Full history)"
        badge="PRIOR"
        sparkline={[50.0, 50.2, 50.1, 50.3, 50.1]}
      />
      <KpiCard
        label="Test Window"
        value={testRange}
        sub="Blind out-of-sample split"
        badge="PURGED"
        color="var(--terminal-amber)"
        sparkline={[1, 2, 3, 4, 5, 6]}
      />
    </div>
  );
}
