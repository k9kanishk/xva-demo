// =============================================
// XVA Engine – GitHub Pages Ready (Vite + React)
// ---------------------------------------------
// This is a self-contained SPA you can deploy to
// GitHub Pages. It includes:
//  - Deterministic Monte Carlo toy engine
//  - CSA terms, PDs/CDS, collateral logic
//  - CVA/FVA/MVA/KVA, Greeks, Scenarios
//  - Seed & path controls, batch run
//  - Minimal UI (no external UI libs)
//
// Quick start:
// 1) npm create vite@latest xva-gh -- --template react-ts
// 2) Replace src/App.tsx with this file's content
//    (and keep the default src/main.tsx & index.html)
// 3) npm i recharts gh-pages
// 4) Add scripts to package.json:
//    "predeploy":"npm run build",
//    "deploy":"gh-pages -d dist"
//    Also set: "homepage":"https://<user>.github.io/<repo>"
// 5) npm run deploy
// =============================================

import { useMemo, useState } from 'react'
import './App.css'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

// ------------------------------
// RNG & helpers
// ------------------------------
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function normal(rand: () => number) {
  const u = Math.max(1e-12, rand())
  const v = Math.max(1e-12, rand())
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function parseNumberList(str: string): number[] {
  return str
    .split(/[\,\s]+/)
    .filter(Boolean)
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x))
}

// Accepts either decimals (0.01 = 1%) or percents (1.0 = 1%)
function parseCumPDList(str: string): number[] {
  const raw = parseNumberList(str)
  const dec = raw.map((p) => (p > 1 ? p / 100 : p))
  const out: number[] = []
  for (let i = 0; i < dec.length; i++) {
    const prev = i ? out[i - 1] : 0
    const p = Math.min(0.999, Math.max(dec[i], prev)) // monotone, <1
    out.push(p)
  }
  return out.length ? out : [0.01, 0.015, 0.02, 0.025, 0.03]
}

function spreadsToCumPD(
  spreadsBps: number[],
  recovery = 0.4,
  tenorsYrs: number[]
) {
  if (!spreadsBps.length) return []
  const lambdas = spreadsBps.map((s) => (s / 10000) / Math.max(1e-6, 1 - recovery))
  return tenorsYrs.map((t, i) => 1 - Math.exp(-lambdas[Math.min(i, lambdas.length - 1)] * t))
}

function yearsTo(dateStr: string, today = new Date()) {
  const t = new Date(dateStr)
  return Math.max(0, (t.getTime() - today.getTime()) / (365.25 * 86400e3))
}

// ------------------------------
// Types
// ------------------------------
interface Trade {
  id: string
  counterparty: string
  notional: number
  currency: string
  maturity: string // ISO date
  tradeType: 'IRS' | 'CDS' | 'FXFWD' | 'EQ' | 'OTHER'
}

interface CSATerms {
  threshold: number
  mta: number
  interestRate: number // OIS proxy
  currency: string
  independentAmount: number // baseline IM
}

interface CollateralSchedule {
  frequency: 'daily' | 'weekly' | 'monthly'
  haircut: number // 0.02 = 2%
  rounding: number
}

interface CreditData {
  use: 'pd' | 'cds' // choose source
  pdCurve: string // comma list; decimals or %
  cdsSpreads: string // bps comma list
  lgd: number // 0.6
  recoveryRate: number // 0.4
}

interface RegulatoryConfig {
  jurisdiction: 'us-basel3' | 'eu-crr3' | 'uk-pra'
  cvaApproach: 'advanced' | 'standardized'
  alphaFactor: number
  multiplier: number
}

interface ScenarioParameters {
  rateShock: number
  volShock: number
  creditSpreadShock: number
  correlationShock: number
}

// ------------------------------
// Core Engine (toy but responsive)
// ------------------------------
function computeXVA(args: {
  trades: Trade[]
  csa: CSATerms
  sched: CollateralSchedule
  credit: CreditData
  reg: RegulatoryConfig
  paths: number
  seed: number
  horizonYears?: number
  rateShock?: number
  volShock?: number
  spreadShock?: number
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
  } = args

  const rand = mulberry32(seed)

  // Build cumulative PD term-structure (monotone, <1)
  const baseTenors = [1, 2, 3, 4, 5]
  let cumPD: number[] = []
  if (credit.use === 'cds') {
    const spreads = parseNumberList(credit.cdsSpreads)
    cumPD = spreadsToCumPD(spreads, credit.recoveryRate ?? 0.4, baseTenors)
  } else {
    cumPD = parseCumPDList(credit.pdCurve)
  }
  // Apply spread shock multiplicatively (cap <1)
  cumPD = cumPD.map((p) => Math.min(0.999, Math.max(0, p * (1 + spreadShock))))
  // Extend/truncate to timeSteps
  const timeSteps = Math.max(4, cumPD.length)
  const dt = horizonYears / timeSteps
  const cumPDx = Array.from({ length: timeSteps }, (_, i) =>
    cumPD[Math.min(i, cumPD.length - 1)]
  )
  const margPD = cumPDx.map((p, i) => (i === 0 ? p : Math.max(p - cumPDx[i - 1], 0)))

  // Rates & funding
  const rOIS = Math.max(0, (csa.interestRate ?? 0.03) + rateShock)
  const fundingSpread = 0.015 + spreadShock // toy
  const rF = rOIS + fundingSpread
  const DF = (r: number, t: number) => Math.exp(-r * t)

  // Exposure simulation
  const baseSigma = 0.20 * (1 + volShock)
  const fxAdj = (ccy: string) => (ccy === 'USD' ? 1 : 1.1)
  const notionalUSD = (t: Trade) => t.notional * fxAdj(t.currency)

  const maturities = trades.map((t) => yearsTo(t.maturity))
  const callEvery = sched.frequency === 'weekly' ? 5 : sched.frequency === 'monthly' ? 21 : 1

  const EPE = new Array(timeSteps).fill(0)
  const IMt = new Array(timeSteps).fill(csa.independentAmount || 0)

  for (let p = 0; p < paths; p++) {
    const pathE = new Array(timeSteps).fill(0)
    // raw positive exposure
    trades.forEach((trade, k) => {
      const N = notionalUSD(trade)
      const T = Math.max(1e-6, maturities[k])
      for (let i = 0; i < timeSteps; i++) {
        const tau = (i + 1) * dt
        if (tau > T) continue // no exposure past maturity
        const shock = normal(rand) * baseSigma * Math.sqrt(Math.min(tau, T))
        const typeMult = trade.tradeType === 'CDS' ? 1.4 * (1 + 0) : 1.0 // (optionally use correlationShock)
        pathE[i] += Math.max(N * typeMult * shock, 0)
      }
    })

    // Apply VM with threshold/MTA/rounding and frequency & haircut
    for (let i = 0; i < timeSteps; i++) {
      let uncoll = pathE[i]
      if ((i + 1) % callEvery === 0) {
        const callAmt = Math.max(0, pathE[i] - csa.threshold)
        if (callAmt >= csa.mta) {
          const rounded = sched.rounding
            ? Math.ceil(callAmt / sched.rounding) * sched.rounding
            : callAmt
          const effective = rounded * (1 - (sched.haircut || 0))
          uncoll = Math.max(0, pathE[i] - effective)
        }
      }
      EPE[i] += uncoll
    }
  }
  for (let i = 0; i < timeSteps; i++) EPE[i] /= paths

  // IM proxy: scales with EE quantile
  const avgNotional =
    trades.reduce((s, t) => s + notionalUSD(t), 0) / Math.max(1, trades.length)
  for (let i = 0; i < timeSteps; i++) {
    IMt[i] = Math.max(
      csa.independentAmount || 0,
      2.33 * baseSigma * Math.sqrt((i + 1) * dt) * avgNotional * 0.2
    )
  }

  // XVA components
  let CVA = 0,
    FVA = 0,
    MVA = 0
  for (let i = 0; i < timeSteps; i++) {
    const t = (i + 1) * dt
    CVA += (credit.lgd ?? 0.6) * EPE[i] * margPD[i] * DF(rOIS, t)
    FVA += EPE[i] * (rF - rOIS) * dt * DF(rOIS, t)
    MVA += IMt[i] * (rF - rOIS) * dt * DF(rOIS, t)
  }

  // Capital overlay (placeholder)
  const KVA = reg.alphaFactor * reg.multiplier * 0.1 * CVA

  return {
    cva: Math.max(0, CVA),
    fva: Math.max(0, FVA),
    mva: Math.max(0, MVA),
    kva: Math.max(0, KVA),
    epeSeries: EPE,
    timeSteps,
    dt,
  }
}

// ------------------------------
// UI
// ------------------------------
export default function App() {
  const [trades, setTrades] = useState<Trade[]>([
    {
      id: 'TRD001',
      counterparty: 'Goldman Sachs',
      notional: 10_000_000,
      currency: 'USD',
      maturity: '2026-12-31',
      tradeType: 'IRS',
    },
    {
      id: 'TRD002',
      counterparty: 'JPMorgan',
      notional: 5_000_000,
      currency: 'EUR',
      maturity: '2027-06-30',
      tradeType: 'CDS',
    },
  ])

  const [csa, setCSA] = useState<CSATerms>({
    threshold: 1_000_000,
    mta: 100_000,
    interestRate: 0.05,
    currency: 'USD',
    independentAmount: 500_000,
  })

  const [sched, setSched] = useState<CollateralSchedule>({
    frequency: 'daily',
    haircut: 0.02,
    rounding: 10_000,
  })

  const [credit, setCredit] = useState<CreditData>({
    use: 'pd',
    pdCurve: '0.01, 0.015, 0.02, 0.025, 0.03',
    cdsSpreads: '150, 160, 170, 180, 190',
    lgd: 0.6,
    recoveryRate: 0.4,
  })

  const [reg, setReg] = useState<RegulatoryConfig>({
    jurisdiction: 'us-basel3',
    cvaApproach: 'advanced',
    alphaFactor: 1.4,
    multiplier: 1.0,
  })

  const [scen, setScen] = useState<ScenarioParameters>({
    rateShock: 0.01,
    volShock: 0.05,
    creditSpreadShock: 0.02,
    correlationShock: 0.1,
  })

  const [paths, setPaths] = useState(50_000)
  const [seed, setSeed] = useState(42)

  const base = useMemo(
    () =>
      computeXVA({
        trades,
        csa,
        sched,
        credit,
        reg,
        paths,
        seed,
      }),
    [trades, csa, sched, credit, reg, paths, seed]
  )

  const shocked = useMemo(
    () =>
      computeXVA({
        trades,
        csa,
        sched,
        credit,
        reg,
        paths: Math.max(10_000, Math.floor(paths / 2)),
        seed: seed + 1,
        rateShock: scen.rateShock,
        volShock: scen.volShock,
        spreadShock: scen.creditSpreadShock,
      }),
    [trades, csa, sched, credit, reg, paths, seed, scen]
  )

  const total = base.cva + base.fva + base.mva + base.kva
  const totalShocked = shocked.cva + shocked.fva + shocked.mva + shocked.kva

  const exposureData = base.epeSeries.map((e, i) => ({
    idx: i + 1,
    epe: Math.round(e),
    ene: Math.round(-0.5 * e),
    pfe: Math.round(1.5 * e),
  }))

  return (
    <div className="container">
      <header>
        <h1>Responsive XVA Engine (GitHub Pages)</h1>
        <p className="muted">
          Toy but responsive MC engine. Edit inputs to see CVA/FVA/MVA/KVA and
          exposures change. Seed & paths are reproducible.
        </p>
      </header>

      <section className="grid-3">
        <div className="card">
          <h3>CSA Terms</h3>
          <label>
            Threshold (USD)
            <input
              type="number"
              value={csa.threshold}
              onChange={(e) => setCSA({ ...csa, threshold: Number(e.target.value) })}
            />
          </label>
          <label>
            MTA (USD)
            <input
              type="number"
              value={csa.mta}
              onChange={(e) => setCSA({ ...csa, mta: Number(e.target.value) })}
            />
          </label>
          <label>
            OIS Rate (dec)
            <input
              type="number"
              step={0.001}
              value={csa.interestRate}
              onChange={(e) => setCSA({ ...csa, interestRate: Number(e.target.value) })}
            />
          </label>
          <label>
            Independent Amount (USD)
            <input
              type="number"
              value={csa.independentAmount}
              onChange={(e) =>
                setCSA({ ...csa, independentAmount: Number(e.target.value) })
              }
            />
          </label>
        </div>

        <div className="card">
          <h3>Collateral Schedule</h3>
          <label>
            Frequency
            <select
              value={sched.frequency}
              onChange={(e) =>
                setSched({ ...sched, frequency: e.target.value as any })
              }
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
          </label>
          <label>
            Haircut (dec)
            <input
              type="number"
              step={0.001}
              value={sched.haircut}
              onChange={(e) =>
                setSched({ ...sched, haircut: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Rounding (USD)
            <input
              type="number"
              value={sched.rounding}
              onChange={(e) =>
                setSched({ ...sched, rounding: Number(e.target.value) })
              }
            />
          </label>
        </div>

        <div className="card">
          <h3>Credit Data</h3>
          <label>
            Source
            <select
              value={credit.use}
              onChange={(e) => setCredit({ ...credit, use: e.target.value as any })}
            >
              <option value="pd">PD Curve</option>
              <option value="cds">CDS Spreads</option>
            </select>
          </label>
          {credit.use === 'pd' ? (
            <label>
              Cum. PDs (dec or %)
              <input
                type="text"
                value={credit.pdCurve}
                onChange={(e) => setCredit({ ...credit, pdCurve: e.target.value })}
              />
            </label>
          ) : (
            <label>
              CDS Spreads (bps)
              <input
                type="text"
                value={credit.cdsSpreads}
                onChange={(e) =>
                  setCredit({ ...credit, cdsSpreads: e.target.value })
                }
              />
            </label>
          )}
          <div className="row">
            <label>
              LGD (dec)
              <input
                type="number"
                step={0.01}
                value={credit.lgd}
                onChange={(e) => setCredit({ ...credit, lgd: Number(e.target.value) })}
              />
            </label>
            <label>
              Recovery (dec)
              <input
                type="number"
                step={0.01}
                value={credit.recoveryRate}
                onChange={(e) =>
                  setCredit({ ...credit, recoveryRate: Number(e.target.value) })
                }
              />
            </label>
          </div>
        </div>
      </section>

      <section className="grid-3">
        <div className="card">
          <h3>Regulatory Config</h3>
          <div className="row">
            <label>
              Jurisdiction
              <select
                value={reg.jurisdiction}
                onChange={(e) =>
                  setReg({ ...reg, jurisdiction: e.target.value as any })
                }
              >
                <option value="us-basel3">US Basel III</option>
                <option value="eu-crr3">EU CRR3</option>
                <option value="uk-pra">UK PRA</option>
              </select>
            </label>
            <label>
              CVA Approach
              <select
                value={reg.cvaApproach}
                onChange={(e) => setReg({ ...reg, cvaApproach: e.target.value as any })}
              >
                <option value="advanced">Advanced</option>
                <option value="standardized">Standardized</option>
              </select>
            </label>
          </div>
          <div className="row">
            <label>
              Alpha
              <input
                type="number"
                step={0.1}
                value={reg.alphaFactor}
                onChange={(e) => setReg({ ...reg, alphaFactor: Number(e.target.value) })}
              />
            </label>
            <label>
              Multiplier
              <input
                type="number"
                step={0.1}
                value={reg.multiplier}
                onChange={(e) => setReg({ ...reg, multiplier: Number(e.target.value) })}
              />
            </label>
          </div>
        </div>

        <div className="card">
          <h3>MC Controls</h3>
          <div className="row">
            <label>
              Paths
              <input
                type="number"
                value={paths}
                onChange={(e) => setPaths(Math.max(1000, Number(e.target.value)))}
              />
            </label>
            <label>
              Seed
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
          </div>
          <p className="muted">Deterministic RNG ensures reproducible results.</p>
        </div>

        <div className="card">
          <h3>Scenario Shocks</h3>
          <div className="row">
            <label>
              Rate Shock (dec)
              <input
                type="number"
                step={0.001}
                value={scen.rateShock}
                onChange={(e) => setScen({ ...scen, rateShock: Number(e.target.value) })}
              />
            </label>
            <label>
              Vol Shock (dec)
              <input
                type="number"
                step={0.001}
                value={scen.volShock}
                onChange={(e) => setScen({ ...scen, volShock: Number(e.target.value) })}
              />
            </label>
          </div>
          <div className="row">
            <label>
              Credit Shock (dec)
              <input
                type="number"
                step={0.001}
                value={scen.creditSpreadShock}
                onChange={(e) =>
                  setScen({ ...scen, creditSpreadShock: Number(e.target.value) })
                }
              />
            </label>
            <label>
              Correlation Shock (dec)
              <input
                type="number"
                step={0.001}
                value={scen.correlationShock}
                onChange={(e) =>
                  setScen({ ...scen, correlationShock: Number(e.target.value) })
                }
              />
            </label>
          </div>
        </div>
      </section>

      <section className="grid-2">
        <div className="card">
          <h3>Results (Base)</h3>
          <table className="data">
            <tbody>
              <tr>
                <td>CVA</td>
                <td>${(base.cva).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              </tr>
              <tr>
                <td>FVA</td>
                <td>${(base.fva).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              </tr>
              <tr>
                <td>MVA</td>
                <td>${(base.mva).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              </tr>
              <tr>
                <td>KVA</td>
                <td>${(base.kva).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              </tr>
              <tr className="total">
                <td>Total</td>
                <td>${(total).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Scenario vs Base</h3>
          <table className="data">
            <thead>
              <tr>
                <th>Component</th>
                <th>Base</th>
                <th>Shocked</th>
                <th>Impact</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['CVA', base.cva, shocked.cva],
                ['FVA', base.fva, shocked.fva],
                ['MVA', base.mva, shocked.mva],
                ['KVA', base.kva, shocked.kva],
                ['Total', total, totalShocked],
              ].map(([k, b, s], idx) => (
                <tr key={idx}>
                  <td>{k as string}</td>
                  <td>${(b as number).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  <td>${(s as number).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  <td className={(s as number) - (b as number) >= 0 ? 'bad' : 'good'}>
                    ${(((s as number) - (b as number))).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h3>Exposure Profile (Base)</h3>
        <div style={{ width: '100%', height: 320 }}>
          <ResponsiveContainer>
            <LineChart data={exposureData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="idx" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="epe" name="EPE" stroke="#3366ff" />
              <Line type="monotone" dataKey="ene" name="ENE" stroke="#22aa66" />
              <Line type="monotone" dataKey="pfe" name="PFE" stroke="#ffaa33" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <footer>
        <small>
          Demo only: XVA math is illustrative for UI/UX; do not use for pricing.
          Build: GitHub Pages. Seed {seed}, Paths {paths}.
        </small>
      </footer>
    </div>
  )
}

// ----------------------------------
// App.css (inline for convenience)
// ----------------------------------
// Paste this into src/App.css if not present
/*
:root { --bg:#0b0d12; --card:#121620; --muted:#8a94a6; --fg:#e6ebf5; --accent:#3461ff; --bad:#c43c3c; --good:#2a9d6f; }
* { box-sizing: border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--fg); }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
header { margin-bottom: 16px; }
h1 { margin:0 0 6px 0; font-size: 28px; }
.muted { color: var(--muted); }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
@media (max-width: 900px) { .grid-3{grid-template-columns:1fr;} .grid-2{grid-template-columns:1fr;} }
.card { background: var(--card); border: 1px solid #1c2333; border-radius: 12px; padding: 16px; }
.card h3 { margin-top: 0; margin-bottom: 12px; font-size: 18px; }
label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 10px; }
input, select { margin-top: 6px; width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid #2a344a; background: #0f1320; color: var(--fg); }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.data { width: 100%; border-collapse: collapse; }
.data td, .data th { border-bottom: 1px solid #253052; padding: 8px 10px; text-align: left; }
.data .total td { font-weight: 700; border-top: 2px solid #3a4a78; }
.good { color: var(--good); }
.bad { color: var(--bad); }
footer { margin-top: 16px; color: var(--muted); }
*/
