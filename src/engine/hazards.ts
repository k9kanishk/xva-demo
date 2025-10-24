// src/engine/hazards.ts
// Build a hazard λ(t), survival S(t), and default density dPD(t).

export function clamp(x: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, x));
}

/** Parse PD list that may be in % or decimals; enforce monotone cumulative < 1. */
export function normalizeCumPD(pdStr: string): number[] {
  const raw = pdStr
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);

  const asDec = raw.some((v) => v > 1) ? raw.map((v) => v / 100) : raw.slice();
  for (let i = 0; i < asDec.length; i++) asDec[i] = clamp(asDec[i], 0, 0.999);

  const cum: number[] = new Array(asDec.length);
  for (let i = 0; i < asDec.length; i++)
    cum[i] = i === 0 ? asDec[0] : Math.max(asDec[i], asDec[i - 1]);

  return cum;
}

/**
 * From cumulative PD at equally spaced time points (Δt = horizon/steps),
 * recover a piecewise-homogeneous hazard λ[i] on each interval.
 *
 * dPD_i = cum[i]-cum[i-1] = S_{i-1} * (1 - exp(-λ_i Δt))  ⇒  λ_i ≈ -ln(1 - dPD_i / S_{i-1})/Δt
 */
export function hazardFromCumPD(
  cumPD: number[],
  dt: number,
  floor = 1e-8
): { lambda: number[]; survival: number[]; dPD: number[] } {
  const n = cumPD.length;
  const lambda = new Array(n).fill(0);
  const S = new Array(n + 1).fill(0);
  const dPD = new Array(n).fill(0);

  S[0] = 1.0;
  for (let i = 0; i < n; i++) {
    const d = clamp(cumPD[i] - (i > 0 ? cumPD[i - 1] : 0), 0, 1);
    const Si = S[i];
    const q = d / Math.max(1e-12, Si); // conditional default prob on [i-1,i]
    const lam = -Math.log(clamp(1 - q, 1e-12, 1)) / Math.max(dt, 1e-9);
    lambda[i] = Math.max(floor, lam);
    dPD[i] = Si * (1 - Math.exp(-lambda[i] * dt));
    S[i + 1] = Si - dPD[i];
  }
  return { lambda, survival: S.slice(0, n), dPD };
}
