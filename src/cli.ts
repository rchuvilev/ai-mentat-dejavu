#!/usr/bin/env node
/**
 * Full pipeline, headless, pure TypeScript. Replaces train_head.py + verify.py.
 *
 *   node dist/cli.js                    # v1 vs v2 comparison + checks
 *   node dist/cli.js --extractor v2     # one extractor
 *   node dist/cli.js --cache emb_cache.jsonl   # train on cached DINOv2 vectors
 *
 * The hand-built extractors need no model download, so this runs anywhere Node
 * runs — phone or desktop — with no ONNX, no WASM and no network.
 */
import { readFileSync, existsSync } from 'node:fs';
import { createCanvas } from './canvas.js';
import { makeDataset, CLASSES, drawShape, type Surface } from './dataset.js';
import { EXTRACTORS, type ExtractorName } from './features.js';
import { Pipeline, confusionMatrix, mulberry32, type Mat } from './linalg.js';
import { runChecks, separationRatio, rotationDiagnostic } from './verify.js';

const argv = process.argv.slice(2);
const arg = (k: string, d?: string) => {
  const i = argv.indexOf(`--${k}`);
  return i > -1 ? argv[i + 1] : d;
};
const has = (k: string) => argv.includes(`--${k}`);

const nTrain = +(arg('train', '30')!);
const nTest = +(arg('test', '12')!);
const cachePath = arg('cache');

const C = (s: string) => s;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function report(label: string, dim: number, Xtr: Mat, ytr: number[],
                Xte: Mat, yte: number[], ktr: string[], kte: string[],
                embMs?: number) {
  const t0 = Date.now();
  const pipe = new Pipeline({ epochs: 300 }).fit(Xtr, ytr, CLASSES.length);
  const fitMs = Date.now() - t0;
  const tr = pipe.score(Xtr, ytr), te = pipe.score(Xte, yte);
  const pred = Xte.map(x => pipe.predict(x));
  const cm = confusionMatrix(yte, pred, CLASSES.length);

  console.log(`\n${'='.repeat(58)}`);
  console.log(`${label}  (${dim}-d)`);
  console.log('='.repeat(58));
  if (embMs !== undefined) console.log(`  embed      ${embMs.toFixed(0)} ms/image`);
  console.log(`  head fit   ${fitMs} ms · ${dim * CLASSES.length + CLASSES.length} params`);
  console.log(`  train ${pct(tr)}   TEST ${pct(te)}   chance ${pct(1 / CLASSES.length)}`);
  console.log(`  separation ratio (test) ${separationRatio(Xte, yte).toFixed(2)}`);
  console.log('  confusion (rows=true):');
  console.log('       ' + CLASSES.map(c => c.padStart(10)).join(''));
  cm.forEach((row, i) =>
    console.log('  ' + CLASSES[i].padEnd(5) + row.map(v => String(v).padStart(10)).join('')));

  const checks = runChecks(Xtr, ytr, Xte, yte, ktr, kte, CLASSES.length);
  console.log('  integrity checks:');
  for (const c of checks) {
    console.log(`    [${c.pass ? 'PASS' : 'FAIL'}] ${c.name}${c.detail ? '  -- ' + c.detail : ''}`);
  }
  const bad = checks.filter(c => !c.pass);
  if (bad.length) console.log(`    ${bad.length} CHECK(S) FAILED`);
  return { test: te, failed: bad.length };
}

async function main() {
  console.log(`platform: ${process.platform}/${process.arch}  node ${process.version}`);

  // ---- cached-embedding mode: train on vectors produced by the browser run
  if (cachePath) {
    if (!existsSync(cachePath)) { console.error(`no cache at ${cachePath}`); process.exit(1); }
    const rows = new Map<string, any>();
    for (const line of readFileSync(cachePath, 'utf8').split('\n')) {
      if (line.trim()) { const r = JSON.parse(line); rows.set(r.key, r); }
    }
    console.log(`loaded ${rows.size} unique cached embeddings from ${cachePath}`);
    const pick = (split: string) => {
      const sel = [...rows.values()].filter(r => r.split === split)
        .sort((a, b) => a.key.localeCompare(b.key));
      return {
        X: sel.map(r => Float64Array.from(r.vec)) as Mat,
        y: sel.map(r => r.label as number),
        k: sel.map(r => r.key as string),
      };
    };
    const A = pick('train'), B = pick('test');
    const r = report('DINOv2-small CLS (cached)', A.X[0].length, A.X, A.y, B.X, B.y, A.k, B.k);
    process.exit(r.failed ? 1 : 0);
  }

  // ---- hand-built extractors: no download, no ONNX
  const mk = (w: number, h: number) => createCanvas(w, h) as unknown as Surface;
  const items = makeDataset(mk, nTrain, nTest);
  console.log(`dataset : ${items.filter(i => i.split === 'train').length} train / ` +
              `${items.filter(i => i.split === 'test').length} test`);

  const only = arg('extractor') as ExtractorName | undefined;
  const names: ExtractorName[] = only ? [only] : ['v1', 'v2'];
  const results: Record<string, number> = {};
  let anyFailed = 0;

  for (const name of names) {
    const { fn, dim, label } = EXTRACTORS[name];
    const t0 = Date.now();
    const Xtr: Mat = [], Xte: Mat = [];
    const ytr: number[] = [], yte: number[] = [];
    const ktr: string[] = [], kte: string[] = [];
    for (const it of items) {
      const v = fn(it.surface);
      if (it.split === 'train') { Xtr.push(v); ytr.push(it.label); ktr.push(it.key); }
      else { Xte.push(v); yte.push(it.label); kte.push(it.key); }
    }
    const embMs = (Date.now() - t0) / items.length;
    const r = report(label, dim, Xtr, ytr, Xte, yte, ktr, kte, embMs);
    results[name] = r.test;
    anyFailed += r.failed;
  }

  // ---- the diagnostic that explains WHY v1 fails
  if (names.length > 1) {
    console.log(`\n${'='.repeat(58)}`);
    console.log('ROTATION DIAGNOSTIC — why v1 cannot work');
    console.log('='.repeat(58));
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
    for (const name of ['v1', 'v2'] as const) {
      const d = rotationDiagnostic(EXTRACTORS[name].fn as any, sq, ci);
      const verdict = d.ratio > 1 ? 'OK' : 'BROKEN — rotation dominates class';
      console.log(`  ${name}: rotation gap ${d.rotGap.toFixed(3)} · between-class ` +
                  `${d.between.toFixed(3)} · ratio ${d.ratio.toFixed(2)}  ${verdict}`);
    }
    console.log(`\n  v1 ${pct(results.v1)} -> v2 ${pct(results.v2)} ` +
                `(${((results.v2 - results.v1) * 100).toFixed(1)} points, ` +
                `${EXTRACTORS.v1.dim}-d -> ${EXTRACTORS.v2.dim}-d)`);
  }
  process.exit(anyFailed ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
