"use client";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type {
  BacktestResult,
  CalibrationPoint,
  ClassBalance,
  FeatureImportance,
  ModelMetrics,
  Prediction,
  WalkForwardFold,
} from "../lib/types";

import Sidebar from "../components/Sidebar";
import KpiRow from "../components/KpiRow";
import DirectionChart from "../components/DirectionChart";
import ComparisonTable from "../components/ComparisonTable";
import BacktestChart from "../components/BacktestChart";
import FeatureImportanceChart from "../components/FeatureImportance";
import WalkForwardChart from "../components/WalkForwardChart";
import CalibrationChart from "../components/CalibrationChart";
import LandingView from "../components/LandingView";
import NextDayCard from "../components/NextDayCard";
import ThemeToggle from "../components/ThemeToggle";
import Skeleton from "../components/Skeleton";

function Err({ msg }: { msg: string }) {
  return (
    <div
      className="card"
      style={{ color: "var(--bear)", textAlign: "center", padding: "2rem" }}
    >
      ⚠️ {msg}
    </div>
  );
}

export default function Dashboard() {
  const [market, setMarket] = useState<"us" | "ind">("us");
  const [active, setActive] = useState("overview");
  const [balance, setBalance] = useState<ClassBalance[] | null>(null);
  const [table, setTable] = useState<ModelMetrics[] | null>(null);
  const [preds, setPreds] = useState<Prediction[] | null>(null);
  const [fi, setFi] = useState<FeatureImportance[] | null>(null);
  const [bt, setBt] = useState<BacktestResult | null>(null);
  const [wf, setWf] = useState<WalkForwardFold[] | null>(null);
  const [cal, setCal] = useState<CalibrationPoint[] | null>(null);
  const [nextSignal, setNextSignal] = useState<import("../lib/types").NextDaySignal | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.classBalance(market),
      api.comparisonTable(market),
      api.predictions(market),
      api.featureImportance(market),
      api.backtest(market),
      api.walkForward(market),
      api.calibration(market),
      api.nextDay(market).catch(() => null),
    ])
      .then(([b, t, p, f, bt, wf, cal, ns]) => {
        setBalance(b);
        setTable(t);
        setPreds(p);
        setFi(f);
        setBt(bt);
        setWf(wf);
        setCal(cal);
        setNextSignal(ns);
      })
      .catch(() =>
        setError(`Could not connect to backend for ${market.toUpperCase()}. Run the pipeline first.`)
      );
  }, [market]);

  const allData = balance && table && preds && fi && bt && wf && cal;

  const allBalance = balance?.find((b) => b.split === "all");
  const bestRow = table?.reduce((a, b) => (b.accuracy > a.accuracy ? b : a), table[0]);
  const majorityRow = table?.find((r) => r.model === "Majority Class");
  const persistRow = table?.find((r) => r.model === "Persistence");
  const testDates = preds ? `${preds[0].date.slice(0, 7)} → ${preds[preds.length - 1].date.slice(0, 7)}` : "—";

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <Sidebar active={active} setActive={setActive} />

      <main style={{ flex: 1, padding: "1.75rem", maxWidth: "100%", overflowX: "hidden" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.5rem", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.35rem" }}>
              <span className="terminal-badge">LIVE RESEARCH TERMINAL</span>
              <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                {market === "us" ? "INDEX: SPY (S&P 500)" : "INDEX: NIFTY 50 (NSE)"}
              </span>
            </div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em" }}>
              {market === "us" ? "SPY" : "NIFTY 50"} · Directional Meta-Engine
            </h1>
            <p className="muted" style={{ margin: "0.2rem 0 0", fontSize: 13 }}>
              Out-of-sample directional validation · Strict chronological holdout
            </p>
          </div>

          <div style={{ display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ display: "flex", background: "rgba(0,0,0,0.3)", padding: "2px", borderRadius: 6, border: "1px solid var(--border)" }}>
              <button
                className={`btn-terminal ${market === "us" ? "active" : ""}`}
                onClick={() => setMarket("us")}
                style={{ border: "none" }}
              >
                🇺🇸 US (SPY)
              </button>
              <button
                className={`btn-terminal ${market === "ind" ? "active" : ""}`}
                onClick={() => setMarket("ind")}
                style={{ border: "none" }}
              >
                🇮🇳 India (NIFTY)
              </button>
            </div>
            <button
              className="btn-terminal active"
              onClick={() => {
                setActive("overview");
                api.nextDay(market).then(setNextSignal);
              }}
              style={{ background: "rgba(16, 185, 129, 0.18)", borderColor: "var(--bull)", color: "var(--bull)", cursor: "pointer" }}
            >
              🔮 Predict Tomorrow
            </button>
            <button
              className={`btn-terminal ${active === "thesis" ? "active" : ""}`}
              onClick={() => setActive(active === "thesis" ? "overview" : "thesis")}
            >
              {active === "thesis" ? "📊 Live Dashboard" : "📑 Quant Thesis"}
            </button>
            <ThemeToggle />
          </div>
        </div>

        {error && <Err msg={error} />}

        {active === "thesis" && (
          <LandingView
            onEnterDashboard={() => setActive("overview")}
            market={market}
            setMarket={setMarket}
          />
        )}

        {active !== "thesis" && !error && !allData && (
          <div style={{ display: "grid", gap: "1rem" }}>
            <div style={{ display: "flex", gap: "1rem" }}>
              {[1, 2, 3, 4, 5].map((i) => <Skeleton key={i} height={85} />)}
            </div>
            <Skeleton height={320} />
            <Skeleton height={200} />
          </div>
        )}

        {active !== "thesis" && allData && (
          <div className="animate-slide-up" style={{ display: "grid", gap: "1.25rem" }}>
            <NextDayCard
              signal={nextSignal}
              onRefresh={() => api.nextDay(market).then(setNextSignal)}
            />

            <KpiRow
              bestAcc={bestRow!.accuracy}
              bestModel={bestRow!.model}
              baselineFloor={majorityRow!.accuracy}
              persistAcc={persistRow!.accuracy}
              upPct={allBalance!.up_pct}
              testRange={testDates}
            />

            {active === "overview" && (
              <>
                <DirectionChart data={preds!} />
                <ComparisonTable rows={table!} />
              </>
            )}

            {active === "comparison" && (
              <ComparisonTable rows={table!} />
            )}

            {active === "backtest" && (
              <BacktestChart result={bt!} />
            )}

            {active === "features" && (
              <div style={{ display: "grid", gap: "1.25rem" }}>
                <FeatureImportanceChart data={fi!} />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem" }}>
                  <WalkForwardChart folds={wf!} />
                  <CalibrationChart data={cal!} />
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
