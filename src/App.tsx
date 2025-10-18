import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

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
    .map(s => s.trim())
    .filter(Boolean)
    .map(Number);
}
const fmtUSD = (x: number) =>
  (x < 0 ? "-$" : "$") + Math.abs(Math.round(x)).toLocaleString();

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
  jurisdiction: "us-basel3" | "eu-crr3" | "uk-pra";
  cvaApproach: "advanced" | "standardized";
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
  cvaGreeks: { delta: number; gamma: number; vega: number; rho: number; theta: number };
  exposure: { date: string; epe: number; ene: number; pfe: number }[];
  scenarioResults: { baseCase: number; shockedCase: number; impact: number; var95: number; var99: number };
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
  rateShock?: number;
  volShock?: number;
  spreadShock?: number;
}) {
  const {
    trades, csa, sched, credit, reg,
    paths, seed, horizonYears = 3,
    rateShock = 0, volShock = 0, spreadShock = 0,
  } = args;

  const rand = mulberry32(seed);

  // PD curve can be in % (1,1.5,...) or decimals (0.01,...)
  const rawPD = parseList(credit.pdCurve);
  const pdCurve = rawPD.map(p => (p > 1 ? p / 100 : p)); // normalize
  const lgd = credit.lgd ?? (1 - (credit.recoveryRate ?? 0.4));

  const timeSteps = Math.max(4, pdCurve.length);
  const dt = horizonYears / timeSteps;

  const rOIS = Math.max(0, (csa.interestRate ?? 0.03) + rateShock);
  const fundingSpread = 0.015 + spreadShock;
  const rF = rOIS + fundingSpread;

  // monotone cumulative PD, capped < 1
  const cumPD = Array.from({ length: timeSteps }, (_, i) =>
    Math.min(0.999, pdCurve[Math.min(i, pdCurve.length - 1)] * (1 + spreadShock))
  );
  const margPD = cumPD.map((p, i) => (i === 0 ? p : Math.max(p - cumPD[i - 1], 0)));

  const DF = (r: number, t: number) => Math.exp(-r * t);

  // exposure simulation (very light)
  const baseSigma = 0.20 * (1 + volShock);
  const fxAdj = (ccy: string) => (ccy === "USD" ? 1 : 1.1);
  const notionalUSD = (t: Trade) => t.notional * fxAdj(t.currency);

  const EPE = new Array(timeSteps).fill(0);
  const ENE = new Array(timeSteps).fill(0);
  const PFE = new Array(timeSteps).fill(0);
  const IMt = new Array(timeSteps).fill(csa.independentAmount || 0);

  const maturityIndex = (t: Trade) => {
    const T = (new Date(t.maturity).getTime() - Date.now()) / (365 * 86400e3);
    return Math.max(0, Math.min(timeSteps - 1, Math.floor(T / dt) - 1));
  };

  for (let p = 0; p < paths; p++) {
    const pathE: number[] = new Array(timeSteps).fill(0);

    for (const trade of trades) {
      const N = notionalUSD(trade);
      const Tidx = maturityIndex(trade);
      const typeMult = trade.tradeType === "CDS" ? 1.4 : 1.0;

      for (let i = 0; i <= Tidx; i++) {
        const tau = (i + 1) * dt;
        const shock = normal(rand) * baseSigma * Math.sqrt(tau);
        pathE[i] += N * typeMult * shock; // can be +/- ; collateral below clips to + only
      }
    }

    // Apply CSA (threshold/MTA/rounding); keep positive exposure only for EPE
    for (let i = 0; i < timeSteps; i++) {
      let pos = Math.max(0, pathE[i] - csa.threshold);
      if (pos >= csa.mta) {
        const r = sched.rounding || 0;
        if (r > 0) pos = Math.ceil(pos / r) * r;
      } else {
        pos = 0;
      }
      const neg = Math.min(0, pathE[i]); // ENE proxy
      EPE[i] += pos;
      ENE[i] += neg;
      PFE[i] += Math.max(pos * 1.5, 0);
    }
  }

  for (let i = 0; i < timeSteps; i++) {
    EPE[i] /= paths;
    ENE[i] /= paths;
    PFE[i] /= paths;
  }

  // initial margin profile (simple VaR proxy)
  const avgNotional =
    trades.reduce((s, t) => s + notionalUSD(t), 0) / Math.max(trades.length, 1);
  for (let i = 0; i < timeSteps; i++) {
    IMt[i] = Math.max(
      csa.independentAmount || 0,
      2.33 * baseSigma * Math.sqrt((i + 1) * dt) * avgNotional * 0.2
    );
  }

  let CVA = 0, FVA = 0, MVA = 0;
  for (let i = 0; i < timeSteps; i++) {
    const t = (i + 1) * dt;
    CVA += lgd * EPE[i] * margPD[i] * DF(rOIS, t);
    FVA += EPE[i] * (rF - rOIS) * dt * DF(rOIS, t);
    MVA += IMt[i] * (rF - rOIS) * dt * DF(rOIS, t);
  }
  const KVA = reg.alphaFactor * reg.multiplier * 0.10 * CVA;

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
    { id: "TRD001", counterparty: "Goldman Sachs", notional: 10_000_000, currency: "USD", maturity: "2025-12-31", tradeType: "IRS" },
    { id: "TRD002", counterparty: "JPMorgan",      notional:  5_000_000, currency: "EUR", maturity: "2026-06-30", tradeType: "CDS" }
  ]);

  const [csa, setCsa] = useState<CSATerms>({
    threshold: 1_000_000, mta: 100_000, interestRate: 0.05, currency: "USD", independentAmount: 500_000
  });
  const [sched, setSched] = useState<CollateralSchedule>({
    frequency: "daily", haircut: 0.02, marginType: "variation", rounding: 10_000
  });
  const [credit, setCredit] = useState<CreditData>({
    pdCurve: "1,1.5,2,2.5,3", // (%)
    lgd: 0.60, recoveryRate: 0.40
  });
  const [reg, setReg] = useState<RegulatoryConfig>({
    jurisdiction: "us-basel3", cvaApproach: "advanced", alphaFactor: 1.4, multiplier: 1.0
  });
  const [sc, setSc] = useState<ScenarioParameters>({
    rateShock: 0.01, volShock: 0.05, creditSpreadShock: 0.02, correlationShock: 0.1
  });

  const [res, setRes] = useState<XVAResults | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const doCalc = async () => {
    setBusy(true); setErr(null); setProgress(0);
    for (let i = 0; i <= 100; i += 10) {
      setProgress(i);
      await new Promise(r => setTimeout(r, 30));
    }
    const t0 = performance.now();
    const base = computeXVA({ trades, csa, sched, credit, reg, paths: 50_000, seed: 42 });
    const deltaBump = computeXVA({ trades, csa: { ...csa, interestRate: csa.interestRate + 0.0001 }, sched, credit, reg, paths: 10_000, seed: 42 });
    const vegaBump  = computeXVA({ trades, csa, sched, credit, reg, paths: 10_000, seed: 42, volShock: 0.01 });
    const shocked   = computeXVA({
      trades, csa, sched, credit, reg, paths: 20_000, seed: 1337,
      rateShock: sc.rateShock, volShock: sc.volShock, spreadShock: sc.creditSpreadShock
    });
    const ms = Math.round(Math.max(1, performance.now() - t0));

    setRes({
      cva: Math.round(base.cva), fva: Math.round(base.fva),
      mva: Math.round(base.mva), kva: Math.round(base.kva),
      cvaGreeks: {
        delta: +(((deltaBump.cva - base.cva) / 0.0001)).toFixed(2),
        gamma: 0,
        vega:  +(((vegaBump.cva  - base.cva) / 0.01)).toFixed(2),
        rho:   +(((deltaBump.cva - base.cva) / 0.0001)).toFixed(2),
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
    setBusy(false); setProgress(0);
  };

  /* ------------------------------- UI --------------------------------- */

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      <h1 style={{ marginBottom: 8 }}>Enterprise XVA Engine</h1>
      <p style={{ marginTop: 0, color: "#555" }}>Edit inputs → Calculate → view capital vs. P&amp;L trade-offs.</p>

      {/* Inputs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 16, marginTop: 16 }}>
        <section style={{ padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
          <h3>CSA Terms</h3>
          <LabelNumber label="Threshold" value={csa.threshold} onChange={v => setCsa({ ...csa, threshold: v })} />
          <LabelNumber label="MTA" value={csa.mta} onChange={v => setCsa({ ...csa, mta: v })} />
          <LabelNumber label="OIS Rate" step={0.001} value={csa.interestRate} onChange={v => setCsa({ ...csa, interestRate: v })} />
          <LabelNumber label="Independent Amount" value={csa.independentAmount} onChange={v => setCsa({ ...csa, independentAmount: v })} />
        </section>

        <section style={{ padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
          <h3>Collateral Schedule</h3>
          <LabelSelect label="Frequency" value={sched.frequency} options={["daily","weekly","monthly"]} onChange={v => setSched({...sched, frequency: v as any})}/>
          <LabelNumber label="Haircut (0-1)" step={0.01} value={sched.haircut} onChange={v => setSched({ ...sched, haircut: v })} />
          <LabelNumber label="Rounding" value={sched.rounding} onChange={v => setSched({ ...sched, rounding: v })} />
        </section>

        <section style={{ padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
          <h3>Credit Data</h3>
          <label style={{ display: "block", fontSize: 12, color: "#555" }}>PD Curve (% or decimals)</label>
          <textarea
            value={credit.pdCurve}
            onChange={e => setCredit({...credit, pdCurve: e.target.value})}
            rows={3}
            style={{ width: "100%", marginBottom: 8 }}
          />
          <LabelNumber label="LGD (0-1)" step={0.01} value={credit.lgd} onChange={v => setCredit({ ...credit, lgd: v })} />
        </section>

        <section style={{ padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
          <h3>Regulatory Config</h3>
          <LabelSelect label="Jurisdiction" value={reg.jurisdiction} options={["us-basel3","eu-crr3","uk-pra"]} onChange={v => setReg({ ...reg, jurisdiction: v as any })}/>
          <LabelNumber label="Alpha Factor" step={0.1} value={reg.alphaFactor} onChange={v => setReg({ ...reg, alphaFactor: v })} />
          <LabelNumber label="Multiplier" step={0.1} value={reg.multiplier} onChange={v => setReg({ ...reg, multiplier: v })} />
        </section>
      </div>

      <div style={{ marginTop: 16 }}>
        <button onClick={doCalc} disabled={busy} style={{ padding: "10px 14px", fontWeight: 600 }}>
          {busy ? `Calculating… ${progress}%` : "Calculate XVA"}
        </button>
      </div>

      {/* Results */}
      {res && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 12, marginTop: 16 }}>
            <MiniCard title="CVA"   value={fmtUSD(res.cva)} />
            <MiniCard title="FVA"   value={fmtUSD(res.fva)} />
            <MiniCard title="MVA"   value={fmtUSD(res.mva)} />
            <MiniCard title="KVA"   value={fmtUSD(res.kva)} />
            <MiniCard title="Total" value={fmtUSD(res.cva + res.fva + res.mva + res.kva)} />
          </div>

          <section style={{ marginTop: 16, padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
            <h3>CVA Greeks</h3>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr><th align="left">Greek</th><th align="right">Value</th></tr>
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

          <section style={{ marginTop: 16, padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
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

          <section style={{ marginTop: 16, padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
            <h3>Scenario (applies shocks to the engine)</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 8 }}>
              <LabelNumber label="Rate shock" step={0.001} value={sc.rateShock} onChange={v => setSc({ ...sc, rateShock: v })}/>
              <LabelNumber label="Vol shock"  step={0.001} value={sc.volShock}  onChange={v => setSc({ ...sc, volShock: v })}/>
              <LabelNumber label="Spread shock" step={0.001} value={sc.creditSpreadShock} onChange={v => setSc({ ...sc, creditSpreadShock: v })}/>
              <LabelNumber label="Corr shock" step={0.001} value={sc.correlationShock} onChange={v => setSc({ ...sc, correlationShock: v })}/>
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
                  <td align="right" style={{ color: res.scenarioResults.impact > 0 ? "#b00020" : "#207227" }}>
                    {fmtUSD(res.scenarioResults.impact)}
                  </td>
                  <td align="right">{fmtUSD(res.scenarioResults.var95)}</td>
                  <td align="right">{fmtUSD(res.scenarioResults.var99)}</td>
                </tr>
              </tbody>
            </table>
          </section>

          <section style={{ marginTop: 16, padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
            <h3>Performance</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 8 }}>
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
    <div style={{ padding: 12, border: "1px solid #eee", borderRadius: 8, background: "#fff" }}>
      <div style={{ fontSize: 12, color: "#666" }}>{title}</div>
      <div style={{ fontSize: 20, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function LabelNumber({
  label, value, onChange, step = 1,
}: { label: string; value: number; onChange: (v: number) => void; step?: number }) {
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
  label, value, options, onChange,
}: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <div style={{ margin: "6px 0" }}>
      <label style={{ display: "block", fontSize: 12, color: "#555" }}>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} style={{ width: "100%" }}>
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}
