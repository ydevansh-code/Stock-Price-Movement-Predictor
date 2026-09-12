"use client";

interface LandingViewProps {
  onEnterDashboard: () => void;
  market: "us" | "ind";
  setMarket: (m: "us" | "ind") => void;
}

export default function LandingView({ onEnterDashboard, market, setMarket }: LandingViewProps) {
  return (
    <div style={{ maxWidth: 1040, margin: "0 auto", padding: "1.5rem 0 3rem" }}>
      <div
        className="card"
        style={{
          background: "linear-gradient(180deg, rgba(15, 23, 42, 0.9) 0%, rgba(9, 13, 20, 0.95) 100%)",
          border: "1px solid rgba(245, 158, 11, 0.3)",
          padding: "2.5rem 2rem",
          marginBottom: "2rem",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem", flexWrap: "wrap", gap: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <span className="terminal-badge">STATIONARY QUANT LAB</span>
            <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>SYS_REV: 2.8.4 // PROD</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--terminal-emerald)", boxShadow: "0 0 8px var(--terminal-emerald)" }} />
            <span className="mono" style={{ fontSize: 11, color: "var(--text)" }}>LIVE RESEARCH ENVIRONMENT</span>
          </div>
        </div>

        <h1 style={{ margin: "0 0 0.75rem", fontSize: 36, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.15 }}>
          Directional Alpha Terminal
        </h1>
        <p style={{ margin: "0 0 1.75rem", fontSize: 16, color: "var(--muted)", maxWidth: 680, lineHeight: 1.6 }}>
          Institutional-grade directional meta-predictor trained across 20+ years of equity data. Built with strict chronological holdout purging, volatility-adjusted triple barriers, and zero lookahead leakage.
        </p>

        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
          <button
            onClick={onEnterDashboard}
            className="btn-terminal active"
            style={{ padding: "0.75rem 1.6rem", fontSize: 14, cursor: "pointer" }}
          >
            <span>⚡ Launch Research Terminal</span>
          </button>
          <div style={{ display: "flex", background: "rgba(0,0,0,0.3)", padding: "3px", borderRadius: 6, border: "1px solid var(--border)" }}>
            <button
              className={`btn-terminal ${market === "us" ? "active" : ""}`}
              onClick={() => setMarket("us")}
              style={{ border: "none", padding: "0.45rem 0.85rem" }}
            >
              🇺🇸 SPY (US)
            </button>
            <button
              className={`btn-terminal ${market === "ind" ? "active" : ""}`}
              onClick={() => setMarket("ind")}
              style={{ border: "none", padding: "0.45rem 0.85rem" }}
            >
              🇮🇳 NIFTY 50 (IND)
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
        {[
          {
            title: "0.5407 Out-of-Sample AUC",
            desc: "Tested across a strictly blind 4-year holdout (2022–2026). Statistically rejects the efficient market null hypothesis.",
            badge: "EMPIRICAL EDGE",
            color: "var(--terminal-emerald)",
          },
          {
            title: "Zero Lookahead Leakage",
            desc: "Purged train/test chronologically. Standard scalers fit strictly on training splits. Forward looking signals eliminated.",
            badge: "VERIFIED PROTOCOL",
            color: "var(--terminal-amber)",
          },
          {
            title: "Triple-Barrier Meta-Labeling",
            desc: "Signals filtered by dynamic ATR volatility barriers with profit-take, stop-loss, and holding horizon exits.",
            badge: "QUANT LABELED",
            color: "var(--text)",
          },
          {
            title: "Fractional Differencing",
            desc: "Memory preserved at d=0.4. Ensures Dickey-Fuller stationarity without wiping out multi-week trend signals.",
            badge: "STATIONARITY",
            color: "var(--text)",
          },
        ].map((item, i) => (
          <div key={i} className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", gap: "0.75rem" }}>
            <div>
              <span style={{ fontSize: 10, fontFamily: "monospace", color: item.color, border: `1px solid ${item.color}40`, padding: "2px 5px", borderRadius: 3 }}>
                {item.badge}
              </span>
              <h3 style={{ margin: "0.6rem 0 0.35rem", fontSize: 15, fontWeight: 700 }}>{item.title}</h3>
              <p className="muted" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5 }}>{item.desc}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="card" style={{ padding: "1.75rem", marginBottom: "2rem" }}>
        <h2 style={{ margin: "0 0 1rem", fontSize: 18, fontWeight: 700 }}>
          The Quant Reality: Why 53%–54% is a Real Edge
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
          <div style={{ background: "rgba(0,0,0,0.25)", padding: "1rem", borderRadius: 6, border: "1px solid var(--border)" }}>
            <h4 style={{ margin: "0 0 0.5rem", fontSize: 13, color: "var(--bear)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              ❌ The Retail ML Trap
            </h4>
            <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6 }}>
              <li>Random train/test splits that shuffle tomorrow into yesterday</li>
              <li>Fake 70%+ accuracy figures created by lookahead features</li>
              <li>Arbitrary 1-day direction without slippage or volatility context</li>
              <li>Catastrophic collapse when deployed to real-world capital</li>
            </ul>
          </div>
          <div style={{ background: "rgba(16, 185, 129, 0.05)", padding: "1rem", borderRadius: 6, border: "1px solid rgba(16, 185, 129, 0.2)" }}>
            <h4 style={{ margin: "0 0 0.5rem", fontSize: 13, color: "var(--terminal-emerald)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              ✓ Institutional Engineering
            </h4>
            <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: 12.5, color: "var(--text)", lineHeight: 1.6 }}>
              <li>Strict chronological walk-forward validation with purging</li>
              <li>Honest 53.1% accuracy / 0.5407 AUC that mirrors hedge fund edges</li>
              <li>Meta-labeling to size positions and reject low-confidence trades</li>
              <li>Cross-asset validation across US & Indian equity markets</li>
            </ul>
          </div>
        </div>
      </div>

      <div style={{ textAlign: "center", paddingTop: "1rem" }}>
        <button
          onClick={onEnterDashboard}
          className="btn-terminal active"
          style={{ padding: "0.85rem 2.2rem", fontSize: 15, cursor: "pointer" }}
        >
          Enter Full Terminal Dashboard →
        </button>
      </div>
    </div>
  );
}
