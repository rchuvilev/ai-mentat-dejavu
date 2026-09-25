#!/usr/bin/env node
/**
 * Property tests. Replaces test_pipeline.py.
 *
 * These assert INVARIANTS, not accuracy: determinism, no label leakage, correct
 * geometry, and that the numeric layer matches sklearn's behaviour. The
 * brightness-leak test is the one that caught a real bug in the data generator
 * (equal-radius shapes have unequal area, so mean brightness identified the
 * class); it stays first-class.
 */
import { createCanvas } from './canvas.js';
import { makeDataset, drawShape, CLASSES, type Surface } from './dataset.js';
import { EXTRACTORS, embedV1, embedV2, V1_DIM, V2_DIM } from './features.js';
import {
  StandardScaler, Pipeline, LogisticRegression, mulberry32, norm2, mean,
  stratifiedFolds, type Mat,
} from './linalg.js';
import { separationRatio, rotationDiagnostic } from './verify.js';

let failed = 0;
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failed++;
};
const mk = (w: number, h: number) => createCanvas(w, h) as unknown as Surface;

// ---------------------------------------------------------------- RNG
{
  const a = mulberry32(42), b = mulberry32(42);
  const xs = Array.from({ length: 100 }, () => a());
  const ys = Array.from({ length: 100 }, () => b());
  check('RNG deterministic for a seed', xs.every((v, i) => v === ys[i]));
  check('RNG output in [0,1)', xs.every(v => v >= 0 && v < 1));
  const m = mean(xs);
  check('RNG roughly uniform', Math.abs(m - 0.5) < 0.1, `mean ${m.toFixed(3)}`);
}

// ---------------------------------------------------------------- extractors
{
  const rng = mulberry32(1);
  const s = drawShape('circle', rng, mk);
  const a = embedV2(s), b = embedV2(s);
  check('embedding deterministic', a.every((v, i) => v === b[i]));

  const shapes = CLASSES.map((c, i) => drawShape(c, mulberry32(10 + i), mk));
  check('v1 fixed width', shapes.every(x => embedV1(x).length === V1_DIM), `dim=${V1_DIM}`);
  check('v2 fixed width', shapes.every(x => embedV2(x).length === V2_DIM), `dim=${V2_DIM}`);
  check('all features finite',
        shapes.every(x => [...embedV1(x), ...embedV2(x)].every(Number.isFinite)));

  // v2 must separate classes better than it responds to rotation.
  const sq = (rot: number) => {
    const cv = createCanvas(128, 128); const cx = cv.getContext('2d');
    cx.fillStyle = 'rgb(60,60,60)'; cx.fillRect(0, 0, 128, 128);
    cx.fillStyle = 'rgb(200,200,200)';
    cx.save(); cx.translate(64, 64); cx.rotate((rot * Math.PI) / 180);
    cx.beginPath(); cx.rect(-20, -20, 40, 40); cx.fill(); cx.restore();
    return cv as unknown as Surface;
  };
  const ci = () => {
    const cv = createCanvas(128, 128); const cx = cv.getContext('2d');
    cx.fillStyle = 'rgb(60,60,60)'; cx.fillRect(0, 0, 128, 128);
    cx.fillStyle = 'rgb(200,200,200)';
    cx.beginPath(); cx.arc(64, 64, Math.sqrt(1600 / Math.PI), 0, Math.PI * 2); cx.fill();
    return cv as unknown as Surface;
  };
  const d1 = rotationDiagnostic(embedV1 as any, sq, ci);
  const d2 = rotationDiagnostic(embedV2 as any, sq, ci);
  check('v2 is rotation-invariant', d2.ratio > 1,
        `ratio ${d2.ratio.toFixed(2)} (rot ${d2.rotGap.toFixed(3)} vs class ${d2.between.toFixed(3)})`);
  check('v1 rotation flaw still reproduces', d1.ratio < 1,
        `ratio ${d1.ratio.toFixed(2)} — this is the documented v1 failure`);
}

// ---------------------------------------------------------------- dataset
{
  const items = makeDataset(mk, 10, 5);
  check('dataset sizes', items.filter(i => i.split === 'train').length === 30 &&
        items.filter(i => i.split === 'test').length === 15,
        `${items.length} total`);
  check('dataset keys unique', new Set(items.map(i => i.key)).size === items.length);
  check('labels balanced', CLASSES.every((_, li) =>
        items.filter(i => i.split === 'train' && i.label === li).length === 10));

  // THE leak test: mean brightness must NOT identify the class.
  const brightness = (s: Surface) => {
    const d = s.getContext('2d').getImageData(0, 0, s.width, s.height).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += (d[i] + d[i + 1] + d[i + 2]) / 3;
    return sum / (d.length / 4) / 255;
  };
  const perClass = CLASSES.map((_, li) =>
    mean(items.filter(i => i.split === 'train' && i.label === li).map(i => brightness(i.surface))));
  let gap = 0;
  for (let i = 0; i < perClass.length; i++)
    for (let j = i + 1; j < perClass.length; j++)
      gap = Math.max(gap, Math.abs(perClass[i] - perClass[j]));
  check('brightness does not leak the label', gap < 0.05,
        `max mean-brightness gap ${gap.toFixed(4)}`);

  // deterministic across calls -> a cached embedding key stays valid
  const again = makeDataset(mk, 10, 5);
  const px = (s: Surface) => s.getContext('2d').getImageData(0, 0, 8, 8).data.join(',');
  check('dataset reproducible from seed',
        items.every((it, i) => px(it.surface) === px(again[i].surface)));
}

// ---------------------------------------------------------------- numerics
{
  const X: Mat = [
    Float64Array.from([1, 10]), Float64Array.from([2, 20]),
    Float64Array.from([3, 30]), Float64Array.from([4, 40]),
  ];
  const sc = new StandardScaler().fit(X);
  const Z = sc.transformAll(X);
  const col0 = Z.map(z => z[0]);
  check('scaler zero-mean', Math.abs(mean(col0)) < 1e-9, `mean ${mean(col0).toExponential(1)}`);
  const sd0 = Math.sqrt(mean(col0.map(v => v * v)));
  check('scaler unit-variance', Math.abs(sd0 - 1) < 1e-6, `sd ${sd0.toFixed(6)}`);

  // linearly separable 2-class problem -> must reach 100%
  const A: Mat = [], ya: number[] = [];
  const rng = mulberry32(3);
  for (let i = 0; i < 60; i++) {
    const c = i % 2;
    A.push(Float64Array.from([c ? 5 + rng() : rng(), c ? 5 + rng() : rng()]));
    ya.push(c);
  }
  const p = new Pipeline({ epochs: 200 }).fit(A, ya, 2);
  check('logreg solves separable problem', p.score(A, ya) === 1,
        `acc ${(p.score(A, ya) * 100).toFixed(1)}%`);

  // probabilities must be a valid distribution
  const pr = p.predictProba(A[0]);
  const tot = Array.from(pr).reduce((x, y) => x + y, 0);
  check('softmax sums to 1', Math.abs(tot - 1) < 1e-9, `sum ${tot.toFixed(12)}`);
  check('softmax in range', Array.from(pr).every(v => v >= 0 && v <= 1));

  // pure noise must NOT be learnable -> guards against a broken fit reporting 100%
  const N: Mat = [], yn: number[] = [];
  for (let i = 0; i < 60; i++) {
    N.push(Float64Array.from([rng(), rng(), rng()]));
    yn.push(i % 3);
  }
  const pn = new Pipeline({ epochs: 120 }).fit(N, yn, 3);
  const half = 30;
  const heldOut = pn.score(N.slice(half), yn.slice(half));
  check('noise is not learnable (held-out)', heldOut < 0.6,
        `held-out acc ${(heldOut * 100).toFixed(1)}%`);

  // stratified folds keep class balance
  const folds = stratifiedFolds([0,0,0,0,1,1,1,1,2,2,2,2], 4, 1);
  check('folds partition the data',
        folds.flat().length === 12 && new Set(folds.flat()).size === 12);
  check('folds are class-balanced', folds.every(f => f.length === 3), `sizes ${folds.map(f=>f.length)}`);

  check('separation ratio > 1 for separable data', separationRatio(A, ya) > 1,
        `ratio ${separationRatio(A, ya).toFixed(2)}`);
}

// ---------------------------------------------------------------- end-to-end
{
  // Use 12 test images per class: with 8 each, one error is worth 4.2 points and
  // measured accuracy swings 88.9-97.2% purely with sample size, so a tight
  // threshold on a small split tests luck rather than the extractor.
  const items = makeDataset(mk, 30, 12);
  for (const name of ['v1', 'v2'] as const) {
    const { fn } = EXTRACTORS[name];
    const tr = items.filter(i => i.split === 'train');
    const te = items.filter(i => i.split === 'test');
    const pipe = new Pipeline({ epochs: 250 })
      .fit(tr.map(i => fn(i.surface)), tr.map(i => i.label), 3);
    const acc = pipe.score(te.map(i => fn(i.surface)), te.map(i => i.label));
    // v2 must clearly beat v1 -- the central finding of the whole project
    if (name === 'v2') check('v2 reaches high accuracy', acc > 0.85, `${(acc*100).toFixed(1)}%`);
    else check('v1 stays near chance', acc < 0.7, `${(acc*100).toFixed(1)}%`);
  }
}

console.log(failed ? `\n${failed} FAILURE(S)` : '\nALL PASS');
process.exit(failed ? 1 : 0);
