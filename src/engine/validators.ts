// --- validators.ts (or top of App.tsx) ---
export function validateInputs({
  csa, sched, credit, reg,
}: {
  csa: { threshold: number; mta: number; interestRate: number; independentAmount: number; };
  sched: { haircut: number; rounding: number; };
  credit: { pdCurve: string; lgd: number; };
  reg: { alphaFactor: number; multiplier: number; };
}) {
  const errors: string[] = [];

  if (csa.threshold < 0) errors.push("Threshold cannot be negative.");
  if (csa.mta < 0) errors.push("MTA cannot be negative.");
  if (csa.interestRate < 0 || csa.interestRate > 0.25) errors.push("OIS must be in [0, 25%].");
  if (csa.independentAmount < 0) errors.push("Independent amount cannot be negative.");

  if (sched.haircut < 0 || sched.haircut > 0.5) errors.push("Haircut must be in [0, 50%].");
  if (sched.rounding < 0) errors.push("Rounding cannot be negative.");

  if (credit.lgd < 0 || credit.lgd > 1) errors.push("LGD must be in [0,1].");

  // PD parse check
  const pd = credit.pdCurve.split(/[,\s]+/).filter(Boolean).map(Number);
  if (!pd.length || pd.some(x => !Number.isFinite(x) || x < 0)) {
    errors.push("PD curve must be a comma-separated list of non-negative numbers (%, or decimals).");
  }

  if (reg.alphaFactor <= 0 || reg.alphaFactor > 3) errors.push("Alpha factor must be in (0, 3].");
  if (reg.multiplier <= 0 || reg.multiplier > 3) errors.push("Multiplier must be in (0, 3].");

  return { ok: errors.length === 0, errors };
}
