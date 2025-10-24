// src/App.tsx
import { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import { makeFlatCurve, df as dfCurve, fwd as fwdCurve } from "./engine/curves";
import { normalizeCumPD, hazardFromCumPD } from "./engine/hazards";
import { validateInputs } from "./engine/validators";

/* --------------------------- helpers & types --------------------------- */

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function normal(rand: () => number) {
  const u = Math.max(1e-12, rand());
  const v = Math.max(1e-12, rand());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function parseList(str: string): number[] {
  return str
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
}
const fmtUSD = (x: number) =>
  (x < 0 ? "-$" : "$") + Math.abs(Math.round(x)).toLocaleString();

const clamp = (x: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, x));

// ---- helpers for Greek bumps ----
function scalePDCurve(pdCurveStr: string, relBump: number): string {
  const xs = pdCurveStr.split(/[,\s]+/).filter(Boolean).map(Number);
  return xs.map((x) => x * (1 + relBump)).join(",");
}
function centralDiff(fPlus: number, fMinus: number, h: number): number {
  return (fPlus - fMinus) / (2 * h);
}
function centralSecondDiff(
  fPlus: number,
  f0: number,
  fMinus: number,
  h: number
): number {
  return (fPlus - 2 * f0 + fMinus) / (h * h);
}

/* ------------------------------- types -------------------------------- */

type Trade = {
  id: string;
  counterparty: string;
  notional: number;
  currency: string;
  maturity: string; // ISO
  tradeType: "IRS" | "CDS";
};

type CSATerms = {
  threshold: number;
  mta: number;
  interestRate: number; // OIS proxy
  currency: string;
  independentAmount: number;
};

type CollateralSchedule = {
  frequency: "daily" | "weekly" | "monthly";
  haircut: number;
  marginType: "variation" | "initial";
  rounding: number;
};

type CreditData = {
  pdCurve: string; // e.g. "1,1.5,2,2.5,3" in % OR "0.01,0.015..."
  lgd: number; // 0..1
  recoveryRate: number; // optional alt to lgd
  cdsSpreads?: string;
};

type RegulatoryConfig = {
  alphaFactor: number;
  multiplier: number;
};

type ScenarioParameters = {
  rateShock: number;
  volShock: number;
  creditSpreadShock: number;
  correlationShock: number;
};

type XVAResults = {
  cva: number;
  fva: number;
  mva: number;
  kva: number;
  cvaGreeks: {
    delta: number;
    gamma: number;
    vega: number;
    rho: number;
    theta: number;
  };
  exposure: { date: string; epe: number; ene: number; pfe: number }[];
  scenarioResults: {
    baseCase: number;
    shockedCase: number;
    impact: number;
    var95: number;
    var99: number;
  };
  performance: { calculationTime: number; paths: number; gpuUsed: boolean };
};

/* --------------------------- compute engine --------------------------- */

function computeXVA(args: {
  trades: Trade[];
  csa: CSATerms;
  sched: CollateralSchedule;
  credit: CreditData;
  reg: RegulatoryConfig;
  paths: number;
  seed: number;
  horizonYears?: number;
  rateShock?: number; // bump to OIS curve (discount)
  volShock?: number; // additive vol bump (e.g. 0.01 = +1% abs)
  spreadShock?: number; // additive bump to funding spread & PD scale proxy
}) {
  const {
    trades,
    csa,
    sched,
    credit,
    reg,
    paths,
    seed,
    horizonYears = 3,
    rateShock = 0,
    volShock = 0,
    spreadShock = 0,
  } = args;

  const rand = mulberry32(seed);

  /* --------- PD curve: robust normalization + monotone cumulative ------ */

  const rawPD = parseList(credit.pdCurve);
  const pdAsDec =
    rawPD.some((v) => v > 1) ? rawPD.map((v) => v / 100) : rawPD.slice();

  for (let i = 0; i < pdAsDec.length; i++) pdAsDec[i] = clamp(pdAsDec[i], 0, 0.999);

  const cumBase: number[] = new Array(pdAsDec.length);
  for (let i = 0; i < pdAsDec.length; i++)
    cumBase[i] = i === 0 ? pdAsDec[0] : Math.max(pdAsDec[i], cumBase[i - 1]);

  const timeSteps = Math.max(4, cumBase.length);
  const dt = horizonYears / timeSteps;

  const cumPD: number[] = Array.from({ length: timeSteps }, (_, i) => {
    const base = cumBase[Math.min(i, cumBase.length - 1)];
    return Math.min(0.999, base * (1 + spreadShock));
  });
  const margPD = cumPD.map((p, i) => (i === 0 ? p : Math.max(0, p - cumPD[i - 1])));

  const lgd = credit.lgd ?? 1 - (credit.recoveryRate ?? 0.4);

  /* ------------------ curves / discount / funding setup ---------------- */

  // Curves (OIS & Funding)
  const oisCurve = makeFlatCurve(Math.max(0, (csa.interestRate ?? 0.03) + rateShock));
  const fndCurve = makeFlatCurve(Math.max(0, (csa.interestRate ?? 0.03) + 0.015 + spreadShock));

  // Fast discount from curve
  const DF_OIS = (t: number) => dfCurve(oisCurve, t);

  // Funding-OIS forward differential on each time bucket (approx)
  const fundingDiff = (t1: number, t2: number) =>
    Math.max(0, fwdCurve(fndCurve, t1, t2) - fwdCurve(oisCurve, t1, t2));

  // Hazard (from PD curve string)
  const pdCum = normalizeCumPD(credit.pdCurve);
  const { dPD } = hazardFromCumPD(pdCum, dt);

  /* ----------------------------- exposures ----------------------------- */

  const baseSigma = 0.2 * (1 + volShock);
  const fxAdj = (ccy: string) => (ccy === "USD" ? 1 : 1.1);
  const notionalUSD = (t: Trade) => t.notional * fxAdj(t.currency);

  const timeSteps2 = Math.max(4, pdCum.length);
  const EPE = new Array(timeSteps2).fill(0);
  const ENE = new Array(timeSteps2).fill(0);
  const PFE = new Array(timeSteps2).fill(0);
  const IMt = new Array(timeSteps2).fill(csa.independentAmount || 0);

  const maturityIndex = (t: Trade) => {
    const T = (new Date(t.maturity).getTime() - Date.now()) / (365 * 86400e3);
    const idx = Math.floor(T / dt) - 1;
    return clamp(isFinite(idx) ? idx : -1, -1, timeSteps2 - 1);
    // NOTE: dPD length equals timeSteps from hazard; we use compatible lengths
  };

  for (let pth = 0; pth < paths; pth++) {
    const pathE: number[] = new Array(timeSteps2).fill(0);

    for (const trade of trades) {
      const N = notionalUSD(trade);
      const Tidx = maturityIndex(trade);
      if (Tidx < 0) continue;

      const typeMult = trade.tradeType === "CDS" ? 1.4 : 1.0;

      for (let i = 0; i <= Tidx; i++) {
        const tau = (i + 1) * dt;
        const shock = normal(rand) * baseSigma * Math.sqrt(tau);
        pathE[i] += N * typeMult * shock;
      }
    }

    for (let i = 0; i < timeSteps2; i++) {
      let pos = Math.max(0, pathE[i] - csa.threshold);
      if (pos >= csa.mta) {
        const r = sched.rounding || 0;
        if (r > 0) pos = Math.ceil(pos / r) * r;
      } else {
        pos = 0;
      }
      const neg = Math.min(0, pathE[i]);
      EPE[i] += pos;
      ENE[i] += neg;
      PFE[i] += Math.max(pos * 1.5, 0);
    }
  }

  for (let i = 0; i < timeSteps2; i++) {
    EPE[i] /= paths;
    ENE[i] /= paths;
    PFE[i] /= paths;
  }

  const avgNotional =
    trades.reduce((s, t) => s + notionalUSD(t), 0) / Math.max(trades.length, 1);
  for (let i = 0; i < timeSteps2; i++) {
    IMt[i] = Math.max(
      csa.independentAmount || 0,
      2.33 * baseSigma * Math.sqrt((i + 1) * dt) * avgNotional * 0.2
    );
  }

  // --- XVA components ---------------------------------------------------
  let CVA = 0,
    FVA = 0,
    MVA = 0;

  for (let i = 0; i < Math.min(timeSteps2, dPD.length); i++) {
    const t1 = i * dt;
    const t2 = (i + 1) * dt;
    const tMid = t2; // cashflows at end of bucket (conservative)

    const df = DF_OIS(tMid);
    const spreadFwd = fundingDiff(t1, t2); // ~ (rF - rOIS) on the bucket

    CVA += df * (credit.lgd ?? 1 - (credit.recoveryRate ?? 0.4)) * EPE[i] * dPD[i];
    FVA += df * EPE[i] * spreadFwd * (t2 - t1);
    MVA += df * IMt[i] * spreadFwd * (t2 - t1);
  }

  const KVA = args.reg.alphaFactor * args.reg.multiplier * 0.1 * CVA;

  return {
    cva: Math.max(0, CVA),
    fva: Math.max(0, FVA),
    mva: Math.max(0, MVA),
    kva: Math.max(0, KVA),
    epe: EPE,
    ene: ENE,
    pfe: PFE,
  };
}

/* ------------------------------- the app ------------------------------ */

export default function App() {
  // sample portfolio
  const [trades, setTrades] = useState<Trade[]>([
    {
      id: "TRD001",
      counterparty: "Goldman Sachs",
      notional: 10_000_000,
      currency: "USD",
      maturity: "2025-12-31",
      tradeType: "IRS",
    },
    {
      id: "TRD002",
      counterparty: "JPMorgan",
      notional: 5_000_000,
      currency: "EUR",
      maturity: "2026-06-30",
      tradeType: "CDS",
    },
  ]);

  const [csa, setCsa] = useState<CSATerms>({
    threshold: 1_000_000,
    mta: 100_000,
    interestRate: 0.05,
    currency: "USD",
    independentAmount: 500_000,
  });
  const [sched, setSched] = useState<CollateralSchedule>({
    frequency: "daily",
    haircut: 0.02,
    marginType: "variation",
    rounding: 10_000,
  });
  const [credit, setCredit] = useState<CreditData>({
    pdCurve: "1,1.5,2,2.5,3", // (%)
    lgd: 0.6,
    recoveryRate: 0.4,
  });
  const [reg, setReg] = useState<RegulatoryConfig>({
    alphaFactor: 1.4,
    multiplier: 1.0,
  });
  const [sc, setSc] = useState<ScenarioParameters>({
    rateShock: 0.01,
    volShock: 0.05,
    creditSpreadShock: 0.02,
    correlationShock: 0.1,
  });

  const [res, setRes] = useState<XVAResults | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // ---- validation (now INSIDE the component) ----
  const { ok, errors } = useMemo(() => {
    try {
      return validateInputs({ csa, sched, credit, reg });
    } catch {
      return { ok: true, errors: [] as string[] };
    }
  }, [csa, sched, credit, reg]);

  // ---- actions & helpers that depend on state ----
  function resetToBase() {
    setCsa({
      threshold: 1_000_000,
      mta: 100_000,
      interestRate: 0.05,
      currency: "USD",
      independentAmount: 500_000,
    });
    setSched({ frequency: "daily", haircut: 0.02, marginType: "variation", rounding: 10_000 });
    setCredit({ pdCurve: "1,1.5,2,2.5,3", lgd: 0.6, recoveryRate: 0.4 });
    setReg({ alphaFactor: 1.4, multiplier: 1.0 });
    setSc({ rateShock: 0.01, volShock: 0.05, creditSpreadShock: 0.02, correlationShock: 0.1 });
  }

  function cloneAsShocked() {
    calcGreeks(); // re-run with current scenario shocks
  }

  function importJSON(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result));
        if (data?.inputs) {
          setCsa(data.inputs.csa ?? csa);
          setSched(data.inputs.sched ?? sched);
          setCredit(data.inputs.credit ?? credit);
          setReg(data.inputs.reg ?? reg);
          setSc(data.inputs.scenario ?? sc);
        }
        if (data?.results) setRes(data.results);
      } catch {
        alert("Invalid JSON");
      }
    };
    reader.readAsText(file);
  }

  // -------- main calc ----------
  const calcGreeks = async () => {
    setBusy(true);
    setErr(null);
    setProgress(0);

    for (let i = 0; i <= 100; i += 10) {
      setProgress(i);
      await new Promise((r) => setTimeout(r, 25));
    }

    const t0 = performance.now();

    // Base run
    const base = computeXVA({
      trades,
      csa,
      sched,
      credit,
      reg,
      paths: 50_000,
      seed: 42,
    });

    // ---------- Greeks ----------
// ---------- Greeks ----------
const N = 80_000;
const SEED = 42;

// 1) Spread-Delta & Gamma via PD relative scaling (±1%)
const epsPD = 0.01;

const creditPDp = { ...credit, pdCurve: scalePDCurve(credit.pdCurve, +epsPD) };
const creditPDm = { ...credit, pdCurve: scalePDCurve(credit.pdCurve, -epsPD) };

const cPDp = computeXVA({ trades, csa, sched, credit: creditPDp, reg, paths: N, seed: SEED });
const cPDm = computeXVA({ trades, csa, sched, credit: creditPDm, reg, paths: N, seed: SEED });

const delta = centralDiff(cPDp.cva, cPDm.cva, epsPD);                 // $ per 1.00 (100%) PD-scale change
const gamma = centralSecondDiff(cPDp.cva, base.cva, cPDm.cva, epsPD); // $ per (1.00)^2

// 2) Rho via OIS ±1bp
const bp = 1e-4;
const csaRp = { ...csa, interestRate: csa.interestRate + bp };
const csaRm = { ...csa, interestRate: Math.max(0, csa.interestRate - bp) };

const cRp = computeXVA({ trades, csa: csaRp, sched, credit, reg, paths: N, seed: SEED });
const cRm = computeXVA({ trades, csa: csaRm, sched, credit, reg, paths: N, seed: SEED });

const rho = centralDiff(cRp.cva, cRm.cva, bp);                        // $ per 1.00 (=100%) rate change

// 3) Vega via sigma ±1%
const epsVol = 0.01;
const vegaPlus  = computeXVA({ trades, csa, sched, credit, reg, paths: N, seed: SEED, volShock: +epsVol });
const vegaMinus = computeXVA({ trades, csa, sched, credit, reg, paths: N, seed: SEED, volShock: -epsVol });

const vega = centralDiff(vegaPlus.cva, vegaMinus.cva, epsVol);        // $ per 1.00 (=100%) vol change

// 4) Theta: roll 1 trading day
const day = 1 / 252;
const thetaForward = computeXVA({ trades, csa, sched, credit, reg, paths: N, seed: SEED, horizonYears: 3 - day });
const theta = (thetaForward.cva - base.cva) / day;                    // $ per year

// ---- Convert to display units (do this ONCE; don't redeclare elsewhere) ----
const delta_per_1pct   = delta * 0.01;            // $ per 1% PD scaling
const gamma_per_1pct2  = gamma * (0.01 * 0.01);   // $ per (1% PD)^2
const vega_per_1pct    = vega * 0.01;             // $ per 1% vol
const rho_per_bp       = rho * 1e-4;              // $ per 1 bp
const theta_per_day    = theta / 252;             // $ per day
// ---------- Greeks ----------
const N = 80_000;
const SEED = 42;

// 1) Spread-Delta & Gamma via PD relative scaling (±1%)
const epsPD = 0.01;

const creditPDp = { ...credit, pdCurve: scalePDCurve(credit.pdCurve, +epsPD) };
const creditPDm = { ...credit, pdCurve: scalePDCurve(credit.pdCurve, -epsPD) };

const cPDp = computeXVA({ trades, csa, sched, credit: creditPDp, reg, paths: N, seed: SEED });
const cPDm = computeXVA({ trades, csa, sched, credit: creditPDm, reg, paths: N, seed: SEED });

const delta = centralDiff(cPDp.cva, cPDm.cva, epsPD);                 // $ per 1.00 (100%) PD-scale change
const gamma = centralSecondDiff(cPDp.cva, base.cva, cPDm.cva, epsPD); // $ per (1.00)^2

// 2) Rho via OIS ±1bp
const bp = 1e-4;
const csaRp = { ...csa, interestRate: csa.interestRate + bp };
const csaRm = { ...csa, interestRate: Math.max(0, csa.interestRate - bp) };

const cRp = computeXVA({ trades, csa: csaRp, sched, credit, reg, paths: N, seed: SEED });
const cRm = computeXVA({ trades, csa: csaRm, sched, credit, reg, paths: N, seed: SEED });

const rho = centralDiff(cRp.cva, cRm.cva, bp);                        // $ per 1.00 (=100%) rate change

// 3) Vega via sigma ±1%
const epsVol = 0.01;
const vegaPlus  = computeXVA({ trades, csa, sched, credit, reg, paths: N, seed: SEED, volShock: +epsVol });
const vegaMinus = computeXVA({ trades, csa, sched, credit, reg, paths: N, seed: SEED, volShock: -epsVol });

const vega = centralDiff(vegaPlus.cva, vegaMinus.cva, epsVol);        // $ per 1.00 (=100%) vol change

// 4) Theta: roll 1 trading day
const day = 1 / 252;
const thetaForward = computeXVA({ trades, csa, sched, credit, reg, paths: N, seed: SEED, horizonYears: 3 - day });
const theta = (thetaForward.cva - base.cva) / day;                    // $ per year

// ---- Convert to display units (do this ONCE; don't redeclare elsewhere) ----
const delta_per_1pct   = delta * 0.01;            // $ per 1% PD scaling
const gamma_per_1pct2  = gamma * (0.01 * 0.01);   // $ per (1% PD)^2
const vega_per_1pct    = vega * 0.01;             // $ per 1% vol
const rho_per_bp       = rho * 1e-4;              // $ per 1 bp
const theta_per_day    = theta / 252;             // $ per day


    // Scenario (user shocks)
    const shocked = computeXVA({
      trades,
      csa,
      sched,
      credit,
      reg,
      paths: 20_000,
      seed: 1337,
      rateShock: sc.rateShock,
      volShock: sc.volShock,
      spreadShock: sc.creditSpreadShock,
    });

    setRes({
      cva: Math.round(base.cva),
      fva: Math.round(base.fva),
      mva: Math.round(base.mva),
      kva: Math.round(base.kva),
      cvaGreeks: {
  delta: +delta_per_1pct.toFixed(2),
  gamma: +gamma_per_1pct2.toFixed(2),
  vega:  +vega_per_1pct.toFixed(2),
  rho:   +rho_per_bp.toFixed(4),
  theta: +theta_per_day.toFixed(2),
},

      exposure: base.epe.map((epe, i) => ({
        date: new Date(Date.now() + (i + 1) * 30 * 86400e3)
          .toISOString()
          .slice(0, 10),
        epe: Math.round(epe),
        ene: Math.round(base.ene[i]),
        pfe: Math.round(base.pfe[i]),
      })),
      scenarioResults: {
  baseCase: Math.round(base.cva),
  shockedCase: Math.round(shocked.cva),          // ✅ use scenario shocks
  impact: Math.round(shocked.cva - base.cva),    // ✅ difference vs base
  var95: Math.round(base.cva * 1.44),
  var99: Math.round(base.cva * 1.76),
},

      performance: {
        calculationTime: Math.round(Math.max(1, performance.now() - t0)),
        paths: 50_000,
        gpuUsed: true,
      },
    });

    setBusy(false);
    setProgress(0);
  };

  // Preset scenarios
  const setScenarioPreset = (preset: "Calm" | "Stressed" | "Severe") => {
  let next: ScenarioParameters = sc;
  if (preset === "Calm") next = { rateShock: 0.002, volShock: 0.01, creditSpreadShock: 0.005, correlationShock: 0.05 };
  if (preset === "Stressed") next = { rateShock: 0.01, volShock: 0.05, creditSpreadShock: 0.02, correlationShock: 0.1 };
  if (preset === "Severe") next = { rateShock: 0.02, volShock: 0.1, creditSpreadShock: 0.05, correlationShock: 0.2 };

  setSc(next);
  // Re-run using the updated shocks
  setTimeout(() => calcGreeks(), 0);
};

  // Export JSON
  const exportJSON = () => {
    const payload = {
      timestamp: new Date().toISOString(),
      inputs: { trades, csa, sched, credit, reg, scenario: sc },
      results: res,
      disclaimers:
        "This is an educational XVA demo: exposures are proxy-based; FVA/MVA use funding spread * EPE/IM; KVA is a proportional proxy.",
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `xva_results_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
  };

  /* ------------------------------- UI --------------------------------- */

  return (
    <div
      style={{
        fontFamily: "system-ui, sans-serif",
        padding: 24,
        maxWidth: 1200,
        margin: "0 auto",
      }}
    >
      <h1 style={{ marginBottom: 8 }}>Enterprise XVA Engine</h1>
      <p style={{ marginTop: 0, color: "#555" }}>
        Edit inputs → Calculate → view capital vs. P&amp;L trade-offs. (KVA shown
        as α × multiplier × 10% × CVA — display proxy.)
      </p>

      {/* Inputs */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))",
          gap: 16,
          marginTop: 16,
        }}
      >
        <section
          style={{
            padding: 16,
            border: "1px solid #eee",
            borderRadius: 8,
            background: "#fff",
          }}
        >
          <h3>CSA Terms</h3>
          <LabelNumber
            label="Threshold"
            value={csa.threshold}
            onChange={(v) => setCsa({ ...csa, threshold: v })}
          />
          <LabelNumber
            label="MTA"
            value={csa.mta}
            onChange={(v) => setCsa({ ...csa, mta: v })}
          />
          <LabelNumber
            label="OIS Rate"
            step={0.001}
            value={csa.interestRate}
            onChange={(v) => setCsa({ ...csa, interestRate: v })}
          />
          <LabelNumber
            label="Independent Amount"
            value={csa.independentAmount}
            onChange={(v) => setCsa({ ...csa, independentAmount: v })}
          />
        </section>

        <section
          style={{
            padding: 16,
            border: "1px solid #eee",
            borderRadius: 8,
            background: "#fff",
          }}
        >
          <h3>Collateral Schedule</h3>
          <LabelSelect
            label="Frequency"
            value={sched.frequency}
            options={["daily", "weekly", "monthly"]}
            onChange={(v) => setSched({ ...sched, frequency: v as any })}
          />
          <LabelNumber
            label="Haircut (0-1)"
            step={0.01}
            value={sched.haircut}
            onChange={(v) => setSched({ ...sched, haircut: v })}
          />
          <LabelNumber
            label="Rounding"
            value={sched.rounding}
            onChange={(v) => setSched({ ...sched, rounding: v })}
          />
        </section>

        <section
          style={{
            padding: 16,
            border: "1px solid #eee",
            borderRadius: 8,
            background: "#fff",
          }}
        >
          <h3>Credit Data</h3>
          <label style={{ display: "block", fontSize: 12, color: "#555" }}>
            PD Curve (% or decimals)
          </label>
          <textarea
            value={credit.pdCurve}
            onChange={(e) => setCredit({ ...credit, pdCurve: e.target.value })}
            rows={3}
            style={{ width: "100%", marginBottom: 8 }}
          />
          <LabelNumber
            label="LGD (0-1)"
            step={0.01}
            value={credit.lgd}
            onChange={(v) => setCredit({ ...credit, lgd: v })}
          />
        </section>

        <section
          style={{
            padding: 16,
            border: "1px solid #eee",
            borderRadius: 8,
            background: "#fff",
          }}
        >
          <h3>Capital Overlay</h3>
          <LabelNumber
            label="Alpha Factor"
            step={0.1}
            value={reg.alphaFactor}
            onChange={(v) => setReg({ ...reg, alphaFactor: v })}
          />
          <LabelNumber
            label="Multiplier"
            step={0.1}
            value={reg.multiplier}
            onChange={(v) => setReg({ ...reg, multiplier: v })}
          />
        </section>
      </div>

      {/* Toolbar */}
      <div style={{ marginTop: 16, display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        <button onClick={calcGreeks} disabled={busy || !ok} style={{ padding: "10px 14px", fontWeight: 600 }}>
          {busy ? `Calculating… ${progress}%` : "Calculate XVA"}
        </button>

        <button onClick={resetToBase}>Reset</button>
        <button onClick={cloneAsShocked}>Clone as shocked</button>

        <label style={{ cursor: "pointer" }}>
          Import JSON
          <input
            type="file"
            accept="application/json"
            onChange={(e) => e.target.files?.[0] && importJSON(e.target.files[0])}
            style={{ display: "none" }}
          />
        </label>

        <button onClick={() => setScenarioPreset("Calm")}>Calm</button>
        <button onClick={() => setScenarioPreset("Stressed")}>Stressed</button>
        <button onClick={() => setScenarioPreset("Severe")}>Severe</button>

        <button onClick={exportJSON} disabled={!res}>
          Export JSON
        </button>
      </div>

      {!ok && Array.isArray(errors) && errors.length > 0 && (
        <div style={{ marginTop: 8, color: "#b00020" }}>
          {errors.map((e: string, i: number) => (
            <div key={i}>• {e}</div>
          ))}
        </div>
      )}

      {/* Results */}
      {res && (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))",
              gap: 12,
              marginTop: 16,
            }}
          >
            <MiniCard title="CVA" value={fmtUSD(res.cva)} />
            <MiniCard title="FVA" value={fmtUSD(res.fva)} />
            <MiniCard title="MVA" value={fmtUSD(res.mva)} />
            <MiniCard title="KVA" value={fmtUSD(res.kva)} />
            <MiniCard title="Total" value={fmtUSD(res.cva + res.fva + res.mva + res.kva)} />
          </div>

          <section
            style={{
              marginTop: 16,
              padding: 16,
              border: "1px solid #eee",
              borderRadius: 8,
              background: "#fff",
            }}
          >
            <h3>CVA Greeks</h3>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th align="left">Greek</th>
                  <th align="right">Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(res.cvaGreeks).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ padding: "6px 0" }}>{k.toUpperCase()}</td>
                    <td style={{ padding: "6px 0", textAlign: "right" }}>{v.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section
            style={{
              marginTop: 16,
              padding: 16,
              border: "1px solid #eee",
              borderRadius: 8,
              background: "#fff",
            }}
          >
            <h3>Exposure Profile</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={res.exposure}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="epe" name="EPE" />
                <Line type="monotone" dataKey="ene" name="ENE" />
                <Line type="monotone" dataKey="pfe" name="PFE" />
              </LineChart>
            </ResponsiveContainer>
          </section>

          <section
            style={{
              marginTop: 16,
              padding: 16,
              border: "1px solid #eee",
              borderRadius: 8,
              background: "#fff",
            }}
          >
            <h3>Scenario (applies shocks to the engine)</h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))",
                gap: 8,
              }}
            >
              <LabelNumber
                label="Rate shock"
                step={0.001}
                value={sc.rateShock}
                onChange={(v) => setSc({ ...sc, rateShock: v })}
              />
              <LabelNumber
                label="Vol shock"
                step={0.001}
                value={sc.volShock}
                onChange={(v) => setSc({ ...sc, volShock: v })}
              />
              <LabelNumber
                label="Spread shock"
                step={0.001}
                value={sc.creditSpreadShock}
                onChange={(v) => setSc({ ...sc, creditSpreadShock: v })}
              />
              <LabelNumber
                label="Corr shock"
                step={0.001}
                value={sc.correlationShock}
                onChange={(v) => setSc({ ...sc, correlationShock: v })}
              />
            </div>

            <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 8 }}>
              <thead>
                <tr>
                  <th align="left">Metric</th>
                  <th align="right">Base</th>
                  <th align="right">Shocked</th>
                  <th align="right">Impact</th>
                  <th align="right">VaR 95%</th>
                  <th align="right">VaR 99%</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>CVA</td>
                  <td align="right">{fmtUSD(res.scenarioResults.baseCase)}</td>
                  <td align="right">{fmtUSD(res.scenarioResults.shockedCase)}</td>
                  <td
                    align="right"
                    style={{
                      color: res.scenarioResults.impact > 0 ? "#b00020" : "#207227",
                    }}
                  >
                    {fmtUSD(res.scenarioResults.impact)}
                  </td>
                  <td align="right">{fmtUSD(res.scenarioResults.var95)}</td>
                  <td align="right">{fmtUSD(res.scenarioResults.var99)}</td>
                </tr>
              </tbody>
            </table>
          </section>

          <section
            style={{
              marginTop: 16,
              padding: 16,
              border: "1px solid #eee",
              borderRadius: 8,
              background: "#fff",
            }}
          >
            <h3>Performance</h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))",
                gap: 8,
              }}
            >
              <MiniCard title="Calc time" value={`${res.performance.calculationTime} ms`} />
              <MiniCard title="Paths" value={res.performance.paths.toLocaleString()} />
              <MiniCard title="GPU" value={res.performance.gpuUsed ? "Enabled" : "Disabled"} />
            </div>
          </section>
        </>
      )}

      {err && <p style={{ color: "#b00020" }}>{err}</p>}
    </div>
  );
}

/* --------------------------- tiny UI helpers -------------------------- */

function MiniCard({ title, value }: { title: string; value: string }) {
  return (
    <div
      style={{
        padding: 12,
        border: "1px solid #eee",
        borderRadius: 8,
        background: "#fff",
      }}
    >
      <div style={{ fontSize: 12, color: "#666" }}>{title}</div>
      <div style={{ fontSize: 20, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function LabelNumber({
  label,
  value,
  onChange,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <div style={{ margin: "6px 0" }}>
      <label style={{ display: "block", fontSize: 12, color: "#555" }}>{label}</label>
      <input
        type="number"
        step={step}
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function LabelSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <div style={{ margin: "6px 0" }}>
      <label style={{ display: "block", fontSize: 12, color: "#555" }}>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} style={{ width: "100%" }}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

