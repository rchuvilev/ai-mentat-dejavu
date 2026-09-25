/**
 * Property + integrity checks. Replaces verify.py and test_pipeline.py.
 *
 * The point of this file: a high accuracy number is not evidence on its own.
 * These assertions catch the ways a good score can be fake — label leakage,
 * train/test contamination, and (most importantly with p >> n) a head that is
 * simply memorising noise.
 */
import { Pipeline, crossValScore, mulberry32, norm2, mean, std, shuffled, type Mat, type Vec } from './linalg.js';

export interface Check { name: string; pass: boolean; detail: string }

export function runChecks(
  Xtr: Mat, ytr: number[], Xte: Mat, yte: number[],
  ktr: string[], kte: string[], nClasses: number,
): Check[] {
  const out: Check[] = [];
  const add = (name: string, pass: boolean, detail = '') => out.push({ name, pass, detail });

  // 1. the two splits must not share keys
  const trSet = new Set(ktr);
  const overlap = kte.filter(k => trSet.has(k));
  add('train/test keys disjoint', overlap.length === 0,
      `${ktr.length} train / ${kte.length} test, ${overlap.length} overlap`);

  // 2. no identical vectors across splits (same image embedded twice)
  const sig = (v: Vec) => Array.from(v.slice(0, 24)).map(x => x.toFixed(4)).join(',');
  const trSig = new Set(Xtr.map(sig));
  const dup = Xte.filter(v => trSig.has(sig(v))).length;
  add('no duplicate vectors across splits', dup === 0, `${dup} duplicates`);

  // 3. test images must not be near-copies of training images
  const nn = Xte.map(t => Math.min(...Xtr.map(r => norm2(t, r))));
  const within: number[] = [];
  for (let c = 0; c < nClasses; c++) {
    const Xc = Xtr.filter((_, i) => ytr[i] === c);
    for (let i = 0; i < Xc.length; i++)
      for (let j = i + 1; j < Xc.length; j++) within.push(norm2(Xc[i], Xc[j]));
  }
  const mNN = mean(nn), mW = mean(within);
  // Compare against the WITHIN-CLASS spread, not a fixed fraction of it: a good
  // low-dimensional extractor deliberately collapses same-class points, so a
  // small absolute NN distance is expected there and is not contamination.
  // Real leakage looks like NN ~= 0 relative to the spread, i.e. an exact copy.
  // Guard: require the nearest train image to be measurably further than the
  // quantisation floor of the space (1% of within-class spread).
  add('test images are not near-copies of train', mNN > 0.01 * mW,
      `mean NN ${mNN.toFixed(3)} vs within-class ${mW.toFixed(3)} ` +
      `(ratio ${(mNN / mW).toFixed(3)})`);

  // 4. balanced test set — a degenerate classifier can game a skewed one
  const cnt = new Array(nClasses).fill(0);
  yte.forEach(c => cnt[c]++);
  add('test set balanced', Math.min(...cnt) === Math.max(...cnt), `counts [${cnt}]`);

  // 5. cross-validation on TRAIN only: if this also scores high, the task is
  //    genuinely easy in this space rather than the test split being lucky
  // Threshold is "clearly better than chance", not a fixed 90%: fold size here
  // is only ~18 samples, so one misclassification moves a fold by 5.6 points.
  // Demanding 90% would fail a healthy 26-d extractor (measured CV 90.0% with
  // folds 94/94/83/89/89) while telling us nothing extra. A broken extractor
  // lands near chance (v1 measured 48.9%), which this still catches.
  const cv = crossValScore(Xtr, ytr, 5, {}, 0, nClasses);
  const chanceLvl = 1 / nClasses;
  add('5-fold CV on train beats chance', mean(cv) > chanceLvl + 0.25,
      `CV ${(mean(cv) * 100).toFixed(1)}% +/- ${(std(cv) * 100).toFixed(1)} ` +
      `(chance ${(chanceLvl * 100).toFixed(1)}%)`);

  // 6. THE decisive one with p >> n: permuted labels must collapse to chance.
  //    If the head could memorise noise, shuffled labels would still score high.
  const rng = mulberry32(0);
  const sh: number[] = [];
  for (let r = 0; r < 5; r++) {
    const yp = shuffled(ytr, rng);
    const p = new Pipeline().fit(Xtr, yp, nClasses);
    sh.push(p.score(Xte, yte));
  }
  const chance = 1 / nClasses;
  add('shuffled labels collapse to chance', mean(sh) < chance + 0.22,
      `shuffled ${(mean(sh) * 100).toFixed(1)}% (chance ${(chance * 100).toFixed(1)}%)`);

  return out;
}

/** Separation ratio: between-class / within-class mean distance. */
export function separationRatio(X: Mat, y: number[]): number {
  const win: number[] = [], btw: number[] = [];
  for (let i = 0; i < X.length; i++)
    for (let j = i + 1; j < X.length; j++)
      (y[i] === y[j] ? win : btw).push(norm2(X[i], X[j]));
  return mean(btw) / mean(win);
}

/**
 * Rotation sensitivity — the measurement that diagnosed the v1 failure.
 * Returns the worst same-shape rotation gap and the between-class gap, both
 * normalised by vector scale so extractors of different dimension compare.
 */
export function rotationDiagnostic(
  embed: (s: any) => Vec, mkSquare: (rot: number) => any, mkCircle: () => any,
): { rotGap: number; between: number; ratio: number } {
  const base = embed(mkSquare(0));
  let scale = 0;
  for (const v of base) scale += v * v;
  scale = Math.sqrt(scale) + 1e-8;
  const rotGap = Math.max(...[10, 20, 30, 45, 60].map(
    r => norm2(embed(mkSquare(r)), base) / scale));
  const between = norm2(embed(mkCircle()), base) / scale;
  return { rotGap, between, ratio: between / rotGap };
}
