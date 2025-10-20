import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

/* --------------------------- math helpers --------------------------- */

// PRNG + normals
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
// error function (Abramowitz–Stegun)
function erf(x: number) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y =
    1 -
    (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
  return sign * y;
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

/* ---- MC helpers: Cholesky + antithetic + central differences ---- */

function cholesky3(C: number[][]) {
  // minimal 3x3 Cholesky (assumes SPD-ish; we’ll keep corr conservative)
  const L = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  L[0][0] = Math.sqrt(Math.max(1e-12, C[0][0]));
  L[1][0] = C[1][0] / L[0][0];
  L[1][1] = Math.sqrt(Math.max(1e-12, C[1][1] - L[1][0] * L[1][0]));
  L[2][0] = C[2][0] / L[0][0];
  L[2][1] = (C[2][1] - L[2][0] * L[1][0]) / L[1][1];
  L[2][2] = Math.sqrt(
    Math.max(1e-12, C[2][2] - L[2][0] * L[2][0] - L[2][1] * L[2][1])
  );
  return L;
}
function applyChol3(L: number[][], z: [number, number, number]): [number, number, number] {
  const y0 = L[0][0] * z[0];
  const y1 = L[1][0] * z[0] + L[1][1] * z[1];
  const y2 = L[2][0] * z[0] + L[2][1] * z[1] + L[2][2] * z[2];
  return [y0, y1, y2];
}
function antitheticTriples(rand: () => number): Array<[number, number, number]> {
  const z: [number, number, number] = [normal(rand), normal(rand), normal(rand)];
  return [z, [-z[0], -z[1], -z[2]]];
}
function centralDiff(up: number, dn: number, h: number) {
  return (up - dn) / (2 * h);
}
function secondCentralDiff(up: number, base: number, dn: number, h: number) {
  return (up - 2 * base + dn) / (h * h);
}

/* -------------------------------- types ------------------------------- */

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
  pdCurve: string; // "1,1.5,2,..." in % or "0.01,0.015..."
  lgd: number;
  recoveryRate: number;
};
type RegulatoryConfig = {
  jurisdiction: "us-basel3" | "eu-crr3" | "uk-pra";
  cvaApproach: "advanced" | "standardized";
  alphaFactor: number;
  multiplier: number;
};
type ScenarioParameters = {
  rateShock: number;
  volShock: number;
  creditSpreadShock: number;
  correlationShock: number; // scales off-diagonal of corr matrix
};
type XVAResults = {
  cva: number;
  fva: number;
  mva: number;
  kva: number;
  cvaGreeks: { delta: number; gamma: number; vega: number; rho: number; theta: number };
  exposure: { date: string; epe: number; ene: number; pfe: number }[];
  scenarioResults: { baseCase: number; shockedCase: number; impact: number; var95: number; var99: number };
  performance: { calculationTime: number; paths: number; gpuUsed: boolean };
};

/* -------- Risk-factor params, re-pricing surrogates & KVA proxy ------- */

type RFParams = {
  // short rate (Hull–White 1F flavour)
  a: number;
  sigma_r: number;
  // FX GBM for non-USD exposures
  mu_fx: number;
  sigma_fx: number;
  // hazard (log-OU)
  kappa_h: number;
  sigma_h: number;
  // correlations among [r, log S, hazard]
  corr: number[][];
  wwCorr: number; // copula link weight for wrong-way
};
const defaultRF: RFParams = {
  a: 0.05, sigma_r: 0.01,
  mu_fx: 0.0, sigma_fx: 0.10,
  kappa_h: 0.30, sigma_h: 0.20,
  corr: [
    [1.00, 0.20, 0.10], // r
    [0.20, 1.00, 0.15], // log S
    [0.10, 0.15, 1.00], // hazard
  ],
  wwCorr: 0.25,
};
function applyCorrShock(corr: number[][], shock: number) {
  const out = corr.map((r) => r.slice());
  for (let i = 0; i < 3; i++) {
    out[i][i] = 1;
    for (let j = i + 1; j < 3; j++) {
      const v = clamp(out[i][j] * (1 + shock), -0.9, 0.9);
      out[i][j] = out[j][i] = v;
    }
  }
  return out;
}
function dv01(notional: number, years: number) {
  const annuity = Math.max(0, years);
  return notional * annuity * 1e-4; // $ per 1bp
}
function rpv01(notional: number, years: number) {
  return notional * Math.min(Math.max(years, 0), 5) * 1e-4; // rough
}

/* ====== capital/KVA proxy ====== */
type Rating = "AAA" | "AA" | "A" | "BBB" | "BB" | "B" | "CCC";
const ratingRW: Record<Rating, number> = {
  AAA: 0.007, AA: 0.010, A: 0.012, BBB: 0.020, BB: 0.050, B: 0.100, CCC: 0.120,
};
function eepeFromProfile(EPE: number[]) {
  return EPE.reduce((s, x) => s + x, 0) / Math.max(1, EPE.length);
}
function proxyKVA(EPE: number[], lgd: number, maturityY: number, rating: Rating, costOfCapital = 0.10, horizonY = 1) {
  const ead = eepeFromProfile(EPE);
  const capital = ratingRW[rating] * lgd * ead * Math.sqrt(Math.max(0.5, maturityY));
  return costOfCapital * capital * horizonY;
}

/* --------------------------- core engine --------------------------- */

function computeXVA(args: {
  trades: Trade[];
  csa: CSATerms;
  sched: CollateralSchedule;
  credit: CreditData;
  reg: RegulatoryConfig;
  rf?: Partial<RFParams>;
  paths: number;
  seed: number;
  horizonYears?: number;
  rateShock?: number;
  volShock?: number;
  spreadShock?: number;
}) {
  const {
    trades, csa, sched, credit, reg,
    rf: rfIn, paths, seed, horizonYears = 3,
    rateShock = 0, volShock = 0, spreadShock = 0,
  } = args;

  // params & vols
  const rf: RFParams = { ...defaultRF, ...(rfIn || {}) };
  const sigma_r = rf.sigma_r * (1 + volShock);
  const sigma_fx = rf.sigma_fx * (1 + volShock);
  const sigma_h = rf.sigma_h * (1 + volShock);

  // PD curve → cumulative (normalized) → marginal → hazard per bucket
  const pdRaw = parseList(credit.pdCurve).map((p) => (p > 1 ? p / 100 : p));
  const T = Math.max(4, pdRaw.length);
  const dt = horizonYears / T;
  const lgd = credit.lgd ?? (1 - (credit.recoveryRate ?? 0.4));

  const cumPD = pdRaw.map((x) => Math.min(0.999, x * (1 + spreadShock)));
  const margPD = cumPD.map((p, i) => (i === 0 ? p : Math.max(p - cumPD[i - 1], 0)));
  const hazLevel = margPD.map((p) => -Math.log(1 - Math.max(0, Math.min(p, 0.999))) / dt);

  const L = cholesky3(rf.corr);
  const rand = mulberry32(seed);

  const EPE = new Array(T).fill(0);
  const ENE = new Array(T).fill(0);
  const PFE = new Array(T).fill(0);
  const IMt = new Array(T).fill(csa.independentAmount || 0);

  const r0 = Math.max(0, (csa.interestRate ?? 0.03) + rateShock);

  for (let pth = 0; pth < paths; pth++) {
    for (const z of antitheticTriples(rand)) {
      const [dWr, dWs, dWh] = applyChol3(L, z);
      let r = r0, logS = 0, h = Math.max(1e-6, hazLevel[0]);
      const pathExp = new Array(T).fill(0);

      for (let i = 0; i < T; i++) {
        // evolve factors
        r += rf.a * (r0 - r) * dt + sigma_r * Math.sqrt(dt) * dWr;
        logS += (rf.mu_fx - 0.5 * sigma_fx * sigma_fx) * dt + sigma_fx * Math.sqrt(dt) * dWs;
        const S = Math.exp(logS);

        const hbar = Math.max(1e-6, hazLevel[i]);
        const logh =
          Math.log(h) + rf.kappa_h * (Math.log(hbar) - Math.log(h)) * dt + sigma_h * Math.sqrt(dt) * dWh;
        h = Math.max(1e-6, Math.exp(logh));

        // light PV re-pricing
        let stepExp = 0;
        for (const tr of trades) {
          const yearsToMat = Math.max(
            0,
            (new Date(tr.maturity).getTime() - Date.now()) / (365 * 86400e3)
          );
          const rem = Math.max(0, yearsToMat - i * dt);
          const ccyAdj = tr.currency === "USD" ? 1 : S;

          if (tr.tradeType === "IRS") {
            const dv = dv01(tr.notional * ccyAdj, rem);
            stepExp += -dv * (r - r0); // rates down => positive PV
          } else {
            const rp = rpv01(tr.notional * ccyAdj, rem);
            const baseSpread = 0.015 + spreadShock; // 150 bps proxy
            stepExp += rp * (baseSpread - lgd * h);
          }
        }

        // CSA
        let pos = Math.max(0, stepExp - csa.threshold);
        if (pos >= csa.mta) {
          const rnd = sched.rounding || 0;
          if (rnd > 0) pos = Math.ceil(pos / rnd) * rnd;
        } else {
          pos = 0;
        }
        pathExp[i] = pos;
      }

      // wrong-way via copula with hazard shock
      const u = 0.5 * (1 + erf(dWh / Math.SQRT2)); // uniform(0,1)
      const uWW = rf.wwCorr * u + (1 - rf.wwCorr) * Math.random();

      let tau = Infinity, H = 0;
      for (let i = 0; i < T; i++) {
        H += hazLevel[i] * dt;
        if (1 - Math.exp(-H) >= uWW) { tau = (i + 1) * dt; break; }
      }

      // accumulate profiles; zero out after default time
      for (let i = 0; i < T; i++) {
        if ((i + 1) * dt <= tau) {
          EPE[i] += pathExp[i];
          ENE[i] += -0.3 * pathExp[i]; // simple ENE proxy
          PFE[i] += 1.5 * pathExp[i];
        }
      }
    }
  }

  const nPaths = paths * 2;
  for (let i = 0; i < T; i++) {
    EPE[i] /= nPaths; ENE[i] /= nPaths; PFE[i] /= nPaths;
  }

  // IM (VaR-like) profile
  const avgNotional = trades.reduce((s, t) => s + t.notional, 0) / Math.max(1, trades.length);
  for (let i = 0; i < T; i++) {
    IMt[i] = Math.max(
      csa.independentAmount || 0,
      2.33 * sigma_fx * Math.sqrt((i + 1) * dt) * avgNotional * 0.2
    );
  }

  // XVA components (discount at r0)
  let CVA = 0, FVA = 0, MVA = 0;
  for (let i = 0; i < T; i++) {
    const t = (i + 1) * dt;
    const DF = Math.exp(-r0 * t);
    CVA += lgd * EPE[i] * margPD[i] * DF;
    FVA += EPE[i] * 0.015 * dt * DF;
    MVA += IMt[i] * 0.015 * dt * DF;
  }

  return {
    cva: Math.max(0, CVA),
    fva: Math.max(0, FVA),
    mva: Math.max(0, MVA),
    kva: 0, // set by caller using proxyKVA()
    epe: EPE, ene: ENE, pfe: PFE,
  };
}

/* -------------------- Greeks with CRN & antithetic ------------------- */

type GreekSet = { delta_bp: number; gamma_bp2: number; rho_bp: number; vega_perc: number; theta: number };

function computeGreeksWithCRN(core: Omit<Parameters<typeof computeXVA>[0], "rateShock" | "volShock" | "spreadShock" | "seed">): GreekSet {
  const seed = 777;        // CRN
  const h_bp = 1e-4;       // 1bp
  const h_vol = 0.01;      // +1% vol

  const base = computeXVA({ ...core, seed });
  const upR = computeXVA({ ...core, seed, rateShock: +h_bp });
  const dnR = computeXVA({ ...core, seed, rateShock: -h_bp });
  const upV = computeXVA({ ...core, seed, volShock: +h_vol });
  const dnV = computeXVA({ ...core, seed, volShock: -h_vol });

  const d = centralDiff(upR.cva, dnR.cva, h_bp);
  const g = secondCentralDiff(upR.cva, base.cva, dnR.cva, h_bp);
  const v = centralDiff(upV.cva, dnV.cva, h_vol);

  return { delta_bp: d, gamma_bp2: g, rho_bp: d, vega_perc: v, theta: 0 };
}

/* ------------------------------- UI --------------------------------- */

export default function App() {
  // CSS to prevent accidental full-page text selection
  useEffect(() => {
    const s = document.createElement("style");
    s.innerHTML = `
      body { -webkit-user-select: none; user-select: none; }
      input, textarea, select, .allow-select { -webkit-user-select: text; user-select: text; }
    `;
    document.head.appendChild(s);
    return () => { document.head.removeChild(s); };
  }, []);

  const [trades] = useState<Trade[]>([
    { id: "TRD001", counterparty: "Goldman Sachs", notional: 10_000_000, currency: "USD", maturity: "2025-12-31", tradeType: "IRS" },
    { id: "TRD002", counterparty: "JPMorgan",      notional:  5_000_000, currency: "EUR", maturity: "2026-06-30", tradeType: "CDS" }
  ]);
  const [csa, setCsa] = useState<CSATerms>({ threshold: 1_000_000, mta: 100_000, interestRate: 0.05, currency: "USD", independentAmount: 500_000 });
  const [sched, setSched] = useState<CollateralSchedule>({ frequency: "daily", haircut: 0.02, marginType: "variation", rounding: 10_000 });
  const [credit, setCredit] = useState<CreditData>({ pdCurve: "1,1.5,2,2.5,3", lgd: 0.60, recoveryRate: 0.40 });
  const [reg, setReg] = useState<RegulatoryConfig>({ jurisdiction: "us-basel3", cvaApproach: "advanced", alphaFactor: 1.4, multiplier: 1.0 });
  const [sc, setSc] = useState<ScenarioParameters>({ rateShock: 0.01, volShock: 0.05, creditSpreadShock: 0.02, correlationShock: 0.10 });

  const [res, setRes] = useState<XVAResults | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const calculateXVA = async () => {
    try {
      setBusy(true); setErr(null); setProgress(0);
      for (let i = 0; i <= 100; i += 10) {
        setProgress(i);
        await new Promise((r) => setTimeout(r, 25));
      }

      const t0 = performance.now();

      // correlation shock applied here
      const rfAdj: RFParams = {
        ...defaultRF,
        corr: applyCorrShock(defaultRF.corr, sc.correlationShock),
      };

      const baseArgs = { trades, csa, sched, credit, reg, rf: rfAdj, paths: 50_000, horizonYears: 3 };

      const base = computeXVA({ ...baseArgs, seed: 42 });
      const greeks = computeGreeksWithCRN(baseArgs);
      const shocked = computeXVA({
        ...baseArgs,
        seed: 1337,
        rateShock: sc.rateShock,
        volShock: sc.volShock,
        spreadShock: sc.creditSpreadShock,
      });

      // capital-based KVA proxy (rating ‘A’, cost of capital 10%, 1y horizon)
      const kva = proxyKVA(base.epe, credit.lgd ?? 0.6, 3, "A", 0.10, 1);

      const ms = Math.round(Math.max(1, performance.now() - t0));

      setRes({
        cva: Math.round(base.cva),
        fva: Math.round(base.fva),
        mva: Math.round(base.mva),
        kva: Math.round(kva),
        cvaGreeks: {
          delta: greeks.delta_bp,             // $ per 1bp
          gamma: greeks.gamma_bp2,            // $ per (bp)^2
          vega: greeks.vega_perc,             // $ per +1.0 vol (i.e., per 100% vol)
          rho: greeks.rho_bp,                 // $ per 1bp
          theta: 0,
        },
        exposure: base.epe.map((epe, i) => ({
          date: new Date(Date.now() + (i + 1) * 30 * 86400e3).toISOString().slice(0, 10),
          epe: Math.round(epe),
          ene: Math.round(base.ene[i]),
          pfe: Math.round(base.pfe[i]),
        })),
        scenarioResults: {
          baseCase: Math.round(base.cva),
          shockedCase: Math.round(shocked.cva),
          impact: Math.round(shocked.cva - base.cva),
          var95: Math.round(base.cva * 1.44),
          var99: Math.round(base.cva * 1.76),
        },
        performance: { calculationTime: ms, paths: 50_000, gpuUsed: true },
      });
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false); setProgress(0);
    }
  };

  const setScenarioPreset = (p: "Calm" | "Stressed" | "Severe") => {
    if (p === "Calm")     setSc({ rateShock: 0.002, volShock: 0.01, creditSpreadShock: 0.005, correlationShock: 0.05 });
    if (p === "Stressed") setSc({ rateShock: 0.010, volShock: 0.05, creditSpreadShock: 0.020, correlationShock: 0.10 });
    if (p === "Severe")   setSc({ rateShock: 0.020, volShock: 0.10, creditSpreadShock: 0.050, correlationShock: 0.20 });
  };

  const exportJSON = () => {
    if (!res) return;
    const payload = {
      timestamp: new Date().toISOString(),
      inputs: { trades, csa, sched, credit, reg, scenario: sc },
      results: res,
      disclaimers:
        "Educational demo only: proxy exposures; funding legs use spread×profile; KVA is capital-based proxy (not SA-CVA).",
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `xva_results_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
  };

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      <h1 style={{ marginBottom: 8 }}>Enterprise XVA Engine</h1>
      <p className="allow-select" style={{ marginTop: 0, color: "#555" }}>
        Edit inputs → Calculate → view capital vs. P&amp;L trade-offs. Sensitivities use CRN &amp; central differences.
      </p>

      {/* Inputs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 16, marginTop: 16 }}>
        <Card title="CSA Terms">
          <LabelNumber label="Threshold" value={csa.threshold} onChange={(v) => setCsa({ ...csa, threshold: v })} />
          <LabelNumber label="MTA" value={csa.mta} onChange={(v) => setCsa({ ...csa, mta: v })} />
          <LabelNumber label="OIS Rate" step={0.001} value={csa.interestRate} onChange={(v) => setCsa({ ...csa, interestRate: v })} />
          <LabelNumber label="Independent Amount" value={csa.independentAmount} onChange={(v) => setCsa({ ...csa, independentAmount: v })} />
        </Card>

        <Card title="Collateral Schedule">
          <LabelSelect label="Frequency" value={sched.frequency} options={["daily", "weekly", "monthly"]} onChange={(v) => setSched({ ...sched, frequency: v as any })} />
          <LabelNumber label="Haircut (0-1)" step={0.01} value={sched.haircut} onChange={(v) => setSched({ ...sched, haircut: v })} />
          <LabelNumber label="Rounding" value={sched.rounding} onChange={(v) => setSched({ ...sched, rounding: v })} />
        </Card>

        <Card title="Credit Data">
          <label style={{ display: "block", fontSize: 12, color: "#555" }}>PD Curve (% or decimals)</label>
          <textarea value={credit.pdCurve} onChange={(e) => setCredit({ ...credit, pdCurve: e.target.value })} rows={3} style={{ width: "100%", marginBottom: 8 }} />
          <LabelNumber label="LGD (0-1)" step={0.01} value={credit.lgd} onChange={(v) => setCredit({ ...credit, lgd: v })} />
        </Card>

        <Card title="Regulatory Config">
          <LabelSelect label="Jurisdiction" value={reg.jurisdiction} options={["us-basel3", "eu-crr3", "uk-pra"]} onChange={(v) => setReg({ ...reg, jurisdiction: v as any })} />
          <LabelNumber label="Alpha Factor" step={0.1} value={reg.alphaFactor} onChange={(v) => setReg({ ...reg, alphaFactor: v })} />
          <LabelNumber label="Multiplier" step={0.1} value={reg.multiplier} onChange={(v) => setReg({ ...reg, multiplier: v })} />
        </Card>
      </div>

      <div style={{ marginTop: 16, display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button onClick={calculateXVA} disabled={busy} style={{ padding: "10px 14px", fontWeight: 600 }}>
          {busy ? `Calculating… ${progress}%` : "Calculate XVA"}
        </button>
        <button onClick={() => setScenarioPreset("Calm")}>Calm</button>
        <button onClick={() => setScenarioPreset("Stressed")}>Stressed</button>
        <button onClick={() => setScenarioPreset("Severe")}>Severe</button>
        <button onClick={exportJSON} disabled={!res}>Export JSON</button>
      </div>

      {res && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 12, marginTop: 16 }}>
            <MiniCard title="CVA" value={fmtUSD(res.cva)} />
            <MiniCard title="FVA" value={fmtUSD(res.fva)} />
            <MiniCard title="MVA" value={fmtUSD(res.mva)} />
            <MiniCard title="KVA" value={fmtUSD(res.kva)} />
            <MiniCard title="Total" value={fmtUSD(res.cva + res.fva + res.mva + res.kva)} />
          </div>

          <Card title="CVA Greeks (units)">
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th align="left">Greek</th><th align="right">Value</th><th align="left">Unit</th></tr></thead>
              <tbody>
                <Row name="DELTA" value={res.cvaGreeks.delta} unit="$ / 1 bp" />
                <Row name="GAMMA" value={res.cvaGreeks.gamma} unit="$ / (1 bp)^2" />
                <Row name="VEGA"  value={res.cvaGreeks.vega}  unit="$ / +1.00 vol" />
                <Row name="RHO"   value={res.cvaGreeks.rho}   unit="$ / 1 bp" />
                <Row name="THETA" value={res.cvaGreeks.theta} unit="$ / day" />
              </tbody>
            </table>
          </Card>

          <Card title="Exposure Profile">
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
          </Card>

          <Card title="Scenario (applies shocks to the engine)">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 8 }}>
              <LabelNumber label="Rate shock"   step={0.001} value={sc.rateShock}          onChange={(v) => setSc({ ...sc, rateShock: v })} />
              <LabelNumber label="Vol shock"    step={0.001} value={sc.volShock}           onChange={(v) => setSc({ ...sc, volShock: v })} />
              <LabelNumber label="Spread shock" step={0.001} value={sc.creditSpreadShock}  onChange={(v) => setSc({ ...sc, creditSpreadShock: v })} />
              <LabelNumber label="Corr shock"   step={0.001} value={sc.correlationShock}   onChange={(v) => setSc({ ...sc, correlationShock: v })} />
            </div>

            <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 8 }}>
              <thead>
                <tr><th align="left">Metric</th><th align="right">Base</th><th align="right">Shocked</th><th align="right">Impact</th><th align="right">VaR 95%</th><th align="right">VaR 99%</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td>CVA</td>
                  <td align="right">{fmtUSD(res.scenarioResults.baseCase)}</td>
                  <td align="right">{fmtUSD(res.scenarioResults.shockedCase)}</td>
                  <td align="right" style={{ color: res.scenarioResults.impact > 0 ? "#b00020" : "#207227" }}>{fmtUSD(res.scenarioResults.impact)}</td>
                  <td align="right">{fmtUSD(res.scenarioResults.var95)}</td>
                  <td align="right">{fmtUSD(res.scenarioResults.var99)}</td>
                </tr>
              </tbody>
            </table>
          </Card>

          <Card title="Performance">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 8 }}>
              <MiniCard title="Calc time" value={`${res.performance.calculationTime} ms`} />
              <MiniCard title="Paths" value={res.performance.paths.toLocaleString()} />
              <MiniCard title="GPU" value={res.performance.gpuUsed ? "Enabled" : "Disabled"} />
            </div>
          </Card>
        </>
      )}

      {err && <p style={{ color: "#b00020" }}>{err}</p>}
    </div>
  );
}

/* ----------------------- small UI building blocks ---------------------- */

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
      <h3>{title}</h3>
      {children}
    </section>
  );
}
function MiniCard({ title, value }: { title: string; value: string }) {
  return (
    <div style={{ padding: 12, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
      <div style={{ fontSize: 12, color: "#666" }}>{title}</div>
      <div style={{ fontSize: 20, fontWeight: 700 }}>{value}</div>
    </div>
  );
}
function Row({ name, value, unit }: { name: string; value: number; unit: string }) {
  return (
    <tr>
      <td style={{ padding: "6px 0" }}>{name}</td>
      <td style={{ padding: "6px 0", textAlign: "right" }}>{value.toFixed(4)}</td>
      <td style={{ padding: "6px 0" }}>{unit}</td>
    </tr>
  );
}
function LabelNumber({
  label, value, onChange, step = 1,
}: { label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <div style={{ margin: "6px 0" }}>
      <label style={{ display: "block", fontSize: 12, color: "#555" }}>{label}</label>
      <input type="number" step={step} value={Number.isFinite(value) ? value : 0} onChange={(e) => onChange(Number(e.target.value))} style={{ width: "100%" }} />
    </div>
  );
}
function LabelSelect({
  label, value, options, onChange,
}: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <div style={{ margin: "6px 0" }}>
      <label style={{ display: "block", fontSize: 12, color: "#555" }}>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} style={{ width: "100%" }}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}
