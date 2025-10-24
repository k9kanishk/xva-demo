// src/engine/curves.ts
// Minimal curve utilities for discounting & forward rates.

export type CurvePt = { t: number; r: number }; // time in years, cont. rate

function sorted(curve: CurvePt[]): CurvePt[] {
  return [...curve].sort((a, b) => a.t - b.t);
}

/** Flat curve from a single continuous compounding rate. */
export function makeFlatCurve(rate: number, horizonYears = 50): CurvePt[] {
  return sorted([
    { t: 0, r: rate },
    { t: horizonYears, r: rate },
  ]);
}

/** Piecewise-constant-on-rate integral to get DF(t) = exp(-∫ r du). */
export function df(curveIn: CurvePt[], t: number): number {
  if (t <= 0) return 1;
  const curve = sorted(curveIn);
  let a = 0;
  let acc = 0;
  let r = curve[0].r;

  for (let i = 1; i < curve.length; i++) {
    const b = Math.min(t, curve[i].t);
    const dt = b - a;
    if (dt > 0) acc += r * dt;
    if (t <= curve[i].t) return Math.exp(-acc);
    r = curve[i].r;
    a = curve[i].t;
  }
  // beyond last knot, use last rate
  acc += r * (t - a);
  return Math.exp(-acc);
}

/** Simple forward rate on [t1,t2] from DFs (continuous comp.). */
export function fwd(curve: CurvePt[], t1: number, t2: number): number {
  if (t2 <= t1) return 0;
  const df1 = df(curve, t1);
  const df2 = df(curve, t2);
  return -Math.log(df2 / df1) / (t2 - t1);
}

/** Bump all points by X bps (1bp = 0.0001). */
export function bumpCurveBp(curve: CurvePt[], bp: number): CurvePt[] {
  const add = bp / 10000;
  return sorted(curve.map((p) => ({ t: p.t, r: p.r + add })));
}
