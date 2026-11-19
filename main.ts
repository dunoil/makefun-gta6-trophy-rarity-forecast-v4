// gta6-trophy-rarity-forecast-01 entry: dilution-aware refresh model.
//
// Exophase refreshes a game's trophy counts in batches. Each refresh adds
// tracked players who are, on average, much less progressed than the players
// already tracked, so every earn rate is diluted. A rate trend cannot see
// this: before the first refresh the page is perfectly flat for hours, then
// every rate drops at once. This program models the refresh process itself:
//
//   * Earners are recovered from the printed 2-decimal rates (integer
//     earners = round(rate * players / 100)) and re-anchored by projecting
//     the modelled path back from the last 15 page states, which kills the
//     rounding-bin bias at large player counts.
//   * At a refresh from P players with batch B, a trophy with share s keeps
//     E' = E + alpha * s * B earners, where alpha is the new players' earn
//     rate as a fraction of the current rate. alpha is fitted per tier by
//     grid search against the exact discrete prediction (rates print to 2
//     decimals, so a candidate alpha is scored on how many historical
//     refreshes it predicts exactly, to the printed digit).
//   * Between refreshes, shares drift on the logit scale at a per-tier rate
//     fitted from stretches of history where the player count is constant
//     (pure earning, no dilution). With no such stretch the drift is zero.
//   * Future tracked players follow a saturating adoption curve
//     P(t) = Pinf - (Pinf - P0) * exp(-lambda * t), fitted by a Pinf grid
//     search plus OLS on ln(Pinf - P), scored on the LOG-players scale.
//     With no growth in the history (the practice cases are perfectly flat)
//     the program falls back to the prior refresh process documented in
//     RULES.txt: first refresh 638 -> 3,868 players at 2026-11-19T04:30Z,
//     batch 3,230, alpha 0.10, then one refresh every 30 minutes.
//   * Future refreshes are simulated on that schedule to the target: grow
//     the shares, dilute at each refresh, grow again to the target. A trophy
//     printed at 100% stays at 100%: every tracked player has it, new
//     players included, as the documented refresh shows.
//   * Spreads come from the delta method: alpha sensitivity times alpha
//     uncertainty, players-path sensitivity times log-players uncertainty,
//     the rounding bin width, and a floor, combined in quadrature.
//
// Node 22, standard library only, deterministic: no randomness, no clock,
// fixed iteration orders, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DEBUG = process.env.MAKEFUN_DEBUG === '1';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Prior dilution alpha (RULES.txt documents the first refresh; 0.10 is the
 *  central value the refresh is consistent with). */
const PRIOR_ALPHA = 0.10;
/** Uncertainty on the prior alpha: the refresh is consistent with roughly
 *  0.08-0.12, so the prior spans about that range. */
const PRIOR_ALPHA_SIGMA = 0.02;
/** Prior refresh batch: 3,868 - 638 players at the documented first refresh. */
const PRIOR_BATCH = 3230;
/** The documented first refresh, 2026-11-19T04:30:00Z. */
const PRIOR_FIRST_REFRESH_MS = Date.parse('2026-11-19T04:30:00Z');
/** Prior refresh cadence, minutes. */
const PRIOR_CADENCE_MIN = 30;
/** Uncertainty on log future players when the prior process is used. */
const PRIOR_LOGP_SIGMA = 0.05;
/** Floor on the log-players uncertainty of a fitted adoption curve. */
const LOGP_FLOOR = 0.03;
/** Floor on every trophy's total logit spread: unmodelled refresh timing. */
const SPREAD_FLOOR = 0.03;
/** A rate printed as 100.00 stays 100: every tracked player has it. */
const FULL_RATE = 99.995;

interface Trophy {
  id: number;
  name: string;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

interface Refresh {
  tMs: number;
  batch: number;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampRate = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** x rounded to 2 decimals, as Exophase prints a rate. */
const round2 = (x: number): number => Math.round(x * 100) / 100;
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

/** Grow a share s on the logit scale by g per hour for dtH hours. */
const growShare = (s: number, g: number, dtH: number): number => {
  if (g === 0 || dtH <= 0 || s <= 0 || s >= 1) return s;
  const p = clamp(s, 1e-9, 1 - 1e-9);
  const l = Math.log(p / (1 - p)) + g * dtH;
  return 1 / (1 + Math.exp(-l));
};

/** Dilute a share s at a refresh that adds batch B to P players. */
const diluteShare = (s: number, p: number, b: number, alpha: number): number => {
  if (b <= 0) return s;
  return (s * (p + alpha * b)) / (p + b);
};

/**
 * Step shares forward from (p0, t0Ms) to targetMs through the given
 * refreshes: grow between refreshes, dilute at each refresh, grow to the
 * target. growthByIdx / alphaByIdx are per-trophy values.
 */
function simulate(
  shares: number[],
  p0: number,
  t0Ms: number,
  refreshes: Refresh[],
  targetMs: number,
  growthByIdx: number[],
  alphaByIdx: number[],
): { shares: number[]; players: number } {
  const s = shares.slice();
  let p = p0;
  let t = t0Ms;
  const ordered = refreshes
    .filter((r) => r.tMs > t0Ms && r.tMs <= targetMs)
    .sort((a, b) => a.tMs - b.tMs);
  for (const r of ordered) {
    const dtH = (r.tMs - t) / 3600000;
    for (let i = 0; i < s.length; i++) s[i] = growShare(s[i], growthByIdx[i] ?? 0, dtH);
    for (let i = 0; i < s.length; i++) s[i] = diluteShare(s[i], p, r.batch, alphaByIdx[i] ?? PRIOR_ALPHA);
    p += r.batch;
    t = r.tMs;
  }
  const dtH = (targetMs - t) / 3600000;
  if (dtH > 0) {
    for (let i = 0; i < s.length; i++) s[i] = growShare(s[i], growthByIdx[i] ?? 0, dtH);
  }
  return { shares: s, players: p };
}

/** Forward-fill null rates per trophy (a read that lacked the trophy). */
function filledRates(c: CaseFile): number[][] {
  const n = c.trophies.length;
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    const col: number[] = [];
    let last: number | null = null;
    for (const h of c.history) {
      const r = h.rates[i];
      if (r !== null && r !== undefined) last = r;
      col.push(last);
    }
    // Back-fill a leading run of nulls from the first real read.
    let first: number | null = null;
    for (let j = c.history.length - 1; j >= 0; j--) {
      const r = c.history[j].rates[i];
      if (r !== null && r !== undefined) first = r;
    }
    for (let j = 0; j < col.length; j++) {
      if (col[j] === null) col[j] = first ?? 0.01;
    }
    out.push(col);
  }
  return out;
}

/**
 * Per-tier logit drift per hour, fitted from read pairs where the player
 * count is constant, so any rate move is pure earning. Falls back to zero
 * when too few visible moves exist (the practice histories are perfectly
 * flat, so the drift is exactly zero there).
 */
function fitGrowth(c: CaseFile, timesMs: number[], fwd: number[][]): Record<string, number> {
  const out: Record<string, number> = {};
  const tiers = [...new Set(c.trophies.map((t) => t.tier))];
  for (const tier of tiers) {
    let num = 0;
    let den = 0;
    let nz = 0;
    for (let j = 1; j < c.history.length; j++) {
      if (c.history[j - 1].players_tracked !== c.history[j].players_tracked) continue;
      const dtH = (timesMs[j] - timesMs[j - 1]) / 3600000;
      if (dtH <= 0) continue;
      for (let i = 0; i < c.trophies.length; i++) {
        if (c.trophies[i].tier !== tier) continue;
        const r0 = fwd[i][j - 1];
        const r1 = fwd[i][j];
        if (r0 >= 99.99 || r1 >= 99.99 || r0 <= 0.01 || r1 <= 0.01) continue;
        const d = logit(r1) - logit(r0);
        num += d;
        den += dtH;
        if (d !== 0) nz++;
      }
    }
    let g = den > 0 ? num / den : 0;
    if (nz < 5) g = 0;
    out[tier] = clamp(g, -0.05, 0.05);
  }
  return out;
}

interface AlphaFit {
  alpha: Record<string, number>;
  sigma: Record<string, number>;
  global: number;
}

/**
 * Per-tier dilution alpha, by grid search against the exact discrete
 * prediction: for each historical refresh, predict the printed after-rate
 * and count exact matches to the printed digit. Falls back to the prior
 * (0.10) when the history holds no refresh, and to the pooled global fit
 * when a tier's objective is flat.
 */
function fitAlpha(
  c: CaseFile,
  timesMs: number[],
  growth: Record<string, number>,
): AlphaFit {
  const events: Array<{ j: number; p0: number; p1: number; batch: number; dtH: number }> = [];
  for (let j = 1; j < c.history.length; j++) {
    const p0 = c.history[j - 1].players_tracked;
    const p1 = c.history[j].players_tracked;
    if (p1 > p0) {
      events.push({ j, p0, p1, batch: p1 - p0, dtH: (timesMs[j] - timesMs[j - 1]) / 3600000 });
    }
  }
  const tiers = [...new Set(c.trophies.map((t) => t.tier))];
  const alpha: Record<string, number> = {};
  const sigma: Record<string, number> = {};
  if (events.length === 0) {
    for (const tier of tiers) {
      alpha[tier] = PRIOR_ALPHA;
      sigma[tier] = PRIOR_ALPHA_SIGMA;
    }
    return { alpha, sigma, global: PRIOR_ALPHA };
  }
  const grid: number[] = [];
  for (let a = 0; a <= 0.3001; a += 0.005) grid.push(Math.round(a * 1000) / 1000);

  // Score one candidate alpha on one set of trophy indices.
  const score = (a: number, idxs: number[]): { matches: number; abserr: number; n: number } => {
    let matches = 0;
    let abserr = 0;
    let n = 0;
    for (const e of events) {
      for (const i of idxs) {
        const r0 = c.history[e.j - 1].rates[i];
        const r1 = c.history[e.j].rates[i];
        if (r0 === null || r0 === undefined || r1 === null || r1 === undefined) continue;
        if (r0 >= 99.99) continue; // the 100% trophy stays 100 whatever alpha is
        const g = growth[c.trophies[i].tier] ?? 0;
        const s0 = Math.round((r0 / 100) * e.p0) / e.p0;
        const sGrown = growShare(s0, g, e.dtH);
        const eAfter = sGrown * e.p0 + a * sGrown * e.batch;
        const pred = round2((100 * eAfter) / e.p1);
        n++;
        if (pred === r1) matches++;
        abserr += Math.abs(pred - r1);
      }
    }
    return { matches, abserr, n };
  };

  const allIdx = c.trophies.map((_, i) => i);
  const pick = (idxs: number[], tiebreak: number): { a: number; s: number } => {
    let bestA = tiebreak;
    let bestMatches = -1;
    let bestAbserr = Infinity;
    let bestN = 0;
    const rows: Array<{ a: number; matches: number; abserr: number }> = [];
    for (const a of grid) {
      const r = score(a, idxs);
      rows.push({ a, matches: r.matches, abserr: r.abserr });
      if (
        r.matches > bestMatches ||
        (r.matches === bestMatches && r.abserr < bestAbserr - 1e-12) ||
        (r.matches === bestMatches && Math.abs(r.abserr - bestAbserr) <= 1e-12 && Math.abs(a - tiebreak) < Math.abs(bestA - tiebreak))
      ) {
        bestA = a;
        bestMatches = r.matches;
        bestAbserr = r.abserr;
        bestN = r.n;
      }
    }
    // Uncertainty: half-width of the alphas within one exact match of the best.
    const good = rows.filter((r) => r.matches >= bestMatches - 1).map((r) => r.a);
    const width = good.length > 0 ? (Math.max(...good) - Math.min(...good)) / 2 : 0.01;
    const s = bestN === 0 ? PRIOR_ALPHA_SIGMA : clamp(width, 0.01, 0.04);
    return { a: bestA, s };
  };

  const globalPick = pick(allIdx, PRIOR_ALPHA);
  for (const tier of tiers) {
    const idxs = c.trophies.map((t, i) => (t.tier === tier ? i : -1)).filter((i) => i >= 0);
    const p = pick(idxs, globalPick.a);
    alpha[tier] = p.a;
    sigma[tier] = p.s;
  }
  return { alpha, sigma, global: globalPick.a };
}

interface CurveFit {
  valid: boolean;
  pinf: number;
  lambda: number;
  a: number;
  b: number;
  rmse: number;
  t0Ms: number;
}

/**
 * Saturating adoption curve P(t) = Pinf - exp(a + b*t), t in hours since the
 * first read. For each candidate Pinf, lambda comes from OLS of
 * ln(Pinf - P) on t; candidates are scored by mean squared error on the
 * LOG-players scale. The fit uses one point per player-count level (the
 * first read showing each new count): between refreshes the tracked count
 * is a flat step, so scoring every read would measure the sawtooth between
 * the steps, not the trend, and would inflate the uncertainty. A flat
 * history (one distinct player count) is degenerate and returns
 * valid: false, so the caller uses the prior refresh process instead.
 */
function fitPlayers(timesMs: number[], players: number[]): CurveFit {
  const bad: CurveFit = { valid: false, pinf: 0, lambda: 0, a: 0, b: 0, rmse: 0, t0Ms: timesMs[0] ?? 0 };
  const pMax = Math.max(...players);
  const pMin = Math.min(...players);
  if (players.length < 2 || pMax === pMin) return bad;
  const t0Ms = timesMs[0];
  // One point per player-count level: the first read showing each new count.
  const pts: Array<{ t: number; p: number }> = [];
  for (let j = 0; j < timesMs.length; j++) {
    if (j === 0 || players[j] !== players[j - 1]) {
      pts.push({ t: (timesMs[j] - t0Ms) / 3600000, p: players[j] });
    }
  }
  if (pts.length < 3) return bad;
  const lo = Math.max(pMax * 1.02, pMax + 200);
  const hi = pMax * 200;
  // Evaluate one candidate Pinf: lambda by OLS, scored on the log scale.
  const evalPinf = (pinf: number): { pinf: number; lambda: number; a: number; b: number; mse: number } | null => {
    let n = 0;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (const pt of pts) {
      const y = Math.log(pinf - pt.p);
      n++;
      sx += pt.t;
      sy += y;
      sxx += pt.t * pt.t;
      sxy += pt.t * y;
    }
    const den = n * sxx - sx * sx;
    if (den === 0) return null;
    const b = (n * sxy - sx * sy) / den;
    const a = (sy - b * sx) / n;
    const lambda = -b;
    if (lambda <= 1e-9) return null;
    let se = 0;
    for (const pt of pts) {
      const pred = pinf - Math.exp(a + b * pt.t);
      if (pred <= 0) return null;
      const d = Math.log(pred) - Math.log(pt.p);
      se += d * d;
    }
    return { pinf, lambda, a, b, mse: se / n };
  };
  // Coarse log-spaced scan, then two refinement passes around the best
  // point. Refinement matters: the log-scale score is hypersensitive to
  // Pinf near the optimum (the early, small-P levels dominate it), so a
  // coarse grid alone can land several percent off with a large rmse.
  const scan = (centre: number, halfWidth: number, count: number): { pinf: number; lambda: number; a: number; b: number; mse: number } | null => {
    let bestLocal: { pinf: number; lambda: number; a: number; b: number; mse: number } | null = null;
    for (let k = 0; k < count; k++) {
      const pinf = centre * Math.exp(-halfWidth + (2 * halfWidth * k) / (count - 1));
      const r = evalPinf(pinf);
      if (r !== null && (bestLocal === null || r.mse < bestLocal.mse)) bestLocal = r;
    }
    return bestLocal;
  };
  let best = scan(Math.sqrt(lo * hi), Math.log(hi / lo) / 2, 160);
  if (best !== null) {
    const refined = scan(best.pinf, Math.log(1.04), 61);
    if (refined !== null) best = refined;
    const refined2 = scan(best.pinf, Math.log(1.008), 61);
    if (refined2 !== null) best = refined2;
  }
  if (best === null) return bad;
  return { valid: true, pinf: best.pinf, lambda: best.lambda, a: best.a, b: best.b, rmse: Math.sqrt(best.mse), t0Ms };
}

const curveValue = (curve: CurveFit, tMs: number): number => {
  const t = (tMs - curve.t0Ms) / 3600000;
  return curve.pinf - Math.exp(curve.a + curve.b * t);
};

/** Next 30-minute clock slot strictly after tMs. */
const nextSlot30 = (tMs: number): number => {
  const slot = 30 * 60000;
  return (Math.floor(tMs / slot) + 1) * slot;
};

/**
 * Future refreshes from the last read to the target. Cadence comes from the
 * gaps between observed refreshes when at least two were seen; otherwise
 * the prior process (first refresh at the documented 04:30Z, then every
 * 30 minutes). Batches come from the adoption curve's increments when it
 * fitted, else the prior batch.
 */
function futureRefreshes(
  timesMs: number[],
  players: number[],
  curve: CurveFit,
  targetMs: number,
): Refresh[] {
  const lastT = timesMs[timesMs.length - 1];
  const eventTimes: number[] = [];
  for (let j = 1; j < players.length; j++) {
    if (players[j] > players[j - 1]) eventTimes.push(timesMs[j]);
  }
  let cadenceMin: number;
  let firstMs: number;
  if (eventTimes.length >= 2) {
    const gaps: number[] = [];
    for (let k = 1; k < eventTimes.length; k++) gaps.push((eventTimes[k] - eventTimes[k - 1]) / 60000);
    cadenceMin = clamp(median(gaps), 10, 180);
    const lastEv = eventTimes[eventTimes.length - 1];
    firstMs = lastEv + Math.ceil((lastT - lastEv) / (cadenceMin * 60000)) * cadenceMin * 60000;
    if (firstMs <= lastT) firstMs += cadenceMin * 60000;
  } else if (eventTimes.length === 1) {
    cadenceMin = PRIOR_CADENCE_MIN;
    firstMs = eventTimes[0] + cadenceMin * 60000;
    if (firstMs <= lastT) firstMs = nextSlot30(lastT);
  } else {
    cadenceMin = PRIOR_CADENCE_MIN;
    firstMs = PRIOR_FIRST_REFRESH_MS > lastT ? PRIOR_FIRST_REFRESH_MS : nextSlot30(lastT);
  }
  const list: Refresh[] = [];
  // If the adoption curve did not fit but refreshes were seen, the observed
  // batch is a better prior than the documented 3,230 (which is launch
  // night's batch, not necessarily the final's).
  const observedBatches: number[] = [];
  for (let j = 1; j < players.length; j++) {
    if (players[j] > players[j - 1]) observedBatches.push(players[j] - players[j - 1]);
  }
  const fallbackBatch = observedBatches.length > 0 ? Math.round(median(observedBatches)) : PRIOR_BATCH;
  let t = firstMs;
  let prevT = lastT;
  let guard = 0;
  while (t <= targetMs && guard++ < 600) {
    let batch: number;
    if (curve.valid) {
      batch = Math.max(0, Math.round(curveValue(curve, t) - curveValue(curve, prevT)));
    } else {
      batch = fallbackBatch;
    }
    if (batch > 0) list.push({ tMs: t, batch });
    prevT = t;
    t += cadenceMin * 60000;
  }
  return list;
}

/**
 * Recover current earners by re-anchoring the modelled log-rate path to the
 * last 15 page states: for each of the last 15 reads, take its integer
 * earners, project them forward to the last read through the fitted
 * refresh/dilution steps, and take the median implied current earners.
 * Returns current shares (earners / last players).
 */
function anchorEarners(
  c: CaseFile,
  timesMs: number[],
  players: number[],
  fwd: number[][],
  growth: Record<string, number>,
  alpha: Record<string, number>,
): number[] {
  const n = c.history.length;
  const m = Math.min(15, n);
  const lastP = players[n - 1];
  const lastT = timesMs[n - 1];
  const out: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    if (fwd[i][n - 1] >= FULL_RATE) {
      out.push(1);
      continue;
    }
    const tier = c.trophies[i].tier;
    const g = growth[tier] ?? 0;
    const a = alpha[tier] ?? PRIOR_ALPHA;
    const implied: number[] = [];
    for (let j = n - m; j < n; j++) {
      const e0 = Math.round((fwd[i][j] / 100) * players[j]);
      const refs: Refresh[] = [];
      for (let k = j + 1; k < n; k++) {
        if (players[k] > players[k - 1]) refs.push({ tMs: timesMs[k], batch: players[k] - players[k - 1] });
      }
      const res = simulate([e0 / players[j]], players[j], timesMs[j], refs, lastT, [g], [a]);
      implied.push(res.shares[0] * res.players);
    }
    out.push(clamp(Math.round(median(implied)) / lastP, 0, 1));
  }
  return out;
}

interface ForecastRow {
  id: number;
  q: number[];
}

function forecastCase(c: CaseFile): ForecastRow[] {
  const timesMs = c.history.map((h) => Date.parse(h.at));
  const players = c.history.map((h) => h.players_tracked);
  const targetMs = Date.parse(c.target);
  const lastT = timesMs[timesMs.length - 1];
  const lastP = players[players.length - 1];
  const horizonHours = Math.max(0, (targetMs - lastT) / 3600000);
  const fwd = filledRates(c);
  const growth = fitGrowth(c, timesMs, fwd);
  const alphaFit = fitAlpha(c, timesMs, growth);
  const curve = fitPlayers(timesMs, players);
  const shares0 = anchorEarners(c, timesMs, players, fwd, growth, alphaFit.alpha);
  const refs = futureRefreshes(timesMs, players, curve, targetMs);
  const growthByIdx = c.trophies.map((t) => growth[t.tier] ?? 0);
  const alphaByIdx = c.trophies.map((t) => alphaFit.alpha[t.tier] ?? PRIOR_ALPHA);
  const base = simulate(shares0, lastP, lastT, refs, targetMs, growthByIdx, alphaByIdx);

  // Delta-method sensitivities, by re-running the simulation perturbed.
  // Alpha: +0.01 for one tier at a time.
  const sensAlpha: number[] = c.trophies.map(() => 0);
  for (const tier of [...new Set(c.trophies.map((t) => t.tier))]) {
    const pert = alphaByIdx.map((a, i) => (c.trophies[i].tier === tier ? a + 0.01 : a));
    const run = simulate(shares0, lastP, lastT, refs, targetMs, growthByIdx, pert);
    for (let i = 0; i < c.trophies.length; i++) {
      if (c.trophies[i].tier !== tier) continue;
      const d = Math.abs(logit(run.shares[i] * 100) - logit(base.shares[i] * 100)) / 0.01;
      sensAlpha[i] = d;
    }
  }
  // Players path: batches scaled 1.1 and 0.9; sensitivity per unit log players.
  let sensPlayers: number[] = c.trophies.map(() => 0);
  let sigmaLogP: number;
  if (refs.length > 0) {
    const hi = refs.map((r) => ({ tMs: r.tMs, batch: Math.round(r.batch * 1.1) }));
    const lo = refs.map((r) => ({ tMs: r.tMs, batch: Math.round(r.batch * 0.9) }));
    const runHi = simulate(shares0, lastP, lastT, hi, targetMs, growthByIdx, alphaByIdx);
    const runLo = simulate(shares0, lastP, lastT, lo, targetMs, growthByIdx, alphaByIdx);
    const dlnP = Math.log(runHi.players) - Math.log(runLo.players);
    if (dlnP > 1e-9) {
      sensPlayers = c.trophies.map((_, i) =>
        Math.abs(logit(runHi.shares[i] * 100) - logit(runLo.shares[i] * 100)) / dlnP,
      );
    }
    sigmaLogP = curve.valid
      ? Math.max(curve.rmse, LOGP_FLOOR) * (1 + horizonHours / 96)
      : PRIOR_LOGP_SIGMA;
  } else {
    sigmaLogP = 0;
  }

  if (DEBUG) {
    console.error(
      `[debug ${c.id}] growth=${JSON.stringify(growth)} alpha=${JSON.stringify(alphaFit.alpha)} ` +
        `alphaSigma=${JSON.stringify(alphaFit.sigma)} globalAlpha=${alphaFit.global} ` +
        `curve valid=${curve.valid} pinf=${Math.round(curve.pinf)} lambda=${curve.lambda.toFixed(5)} ` +
        `rmse=${curve.rmse.toFixed(4)} refreshes=${refs.length} players ${lastP} -> ${base.players}`,
    );
  }

  const rows: ForecastRow[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const id = c.trophies[i].id;
    if (fwd[i][timesMs.length - 1] >= FULL_RATE) {
      rows.push({ id, q: [100, 100, 100, 100, 100] });
      continue;
    }
    const medRate = clamp(base.shares[i] * 100, 0.01, 99.99);
    const p = medRate / 100;
    const sigmaRound = 0.00005 / (p * (1 - p)) / Math.sqrt(3);
    const tier = c.trophies[i].tier;
    const sigma = Math.sqrt(
      (sensAlpha[i] * (alphaFit.sigma[tier] ?? PRIOR_ALPHA_SIGMA)) ** 2 +
        (sensPlayers[i] * sigmaLogP) ** 2 +
        sigmaRound ** 2 +
        SPREAD_FLOOR ** 2,
    );
    const l = logit(medRate);
    const q: number[] = [];
    let prev = -Infinity;
    for (const z of Z) {
      const v = round4(clamp(sigmoidPct(l + z * sigma), 0, 100));
      const w = Math.max(v, prev);
      q.push(w);
      prev = w;
    }
    rows.push({ id, q });
  }
  rows.sort((a, b) => a.id - b.id);
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
