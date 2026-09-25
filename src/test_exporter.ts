#!/usr/bin/env node
/**
 * Tests for the model exporter.
 *
 * exportScript() writes a file a user downloads and runs somewhere this app
 * will never see. Every way it can be wrong is therefore SILENT here and loud
 * there: a syntax error in the generated source, weights mangled on the way
 * out, a class name that breaks out of its string literal. So the central test
 * is not "does the text look right" — it is to write the output to disk and
 * actually `import()` it, which is the only thing that proves the file a user
 * receives is loadable at all.
 *
 * Node has no canvas, so the inference path (classify -> extractor) stays in
 * the browser suite. What is asserted here is the part that is pure: the
 * generated module's shape, its weights, and its escaping.
 */
import { exportScript } from './exporter.js';
import type { StoredModel } from './store.js';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

let failed = 0;
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failed++;
};

const dir = mkdtempSync(join(tmpdir(), 'dejavu-export-'));
let seq = 0;

/** Write the generated source out and import it, exactly as a user would. */
async function loadExport(rec: StoredModel, opts = {}): Promise<any> {
  const src = exportScript(rec, opts);
  const file = join(dir, `m${seq++}.mjs`);
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const model = (over: Partial<StoredModel> = {}): StoredModel => ({
  id: 'm1',
  name: 'shapes',
  createdAt: 1_700_000_000_000,
  extractor: 'photo',
  dim: 3,
  classes: ['circle', 'square'],
  mu: Float64Array.from([0.5, -0.25, 1]),
  sd: Float64Array.from([1, 2, 0.5]),
  W: Float64Array.from([1, 0, -1, -1, 0, 1]),   // k * dim = 2 * 3
  b: Float64Array.from([0.1, -0.1]),
  metrics: { train: 0.98, test: 0.91, ratio: 0.8, nTrain: 60, nTest: 24 },
  ...over,
}) as StoredModel;

async function main() {
  /* ── the file a user receives actually loads ──────────────────────────── */

  const mod = await loadExport(model());
  check('the generated file is valid ESM and imports',
    !!mod && typeof mod === 'object');
  check('it exports MODEL and createDetector',
    !!mod.MODEL && typeof mod.createDetector === 'function');

  /* ── the model survives the trip ──────────────────────────────────────── */

  const rec = model();
  check('classes round-trip', JSON.stringify(mod.MODEL.classes) === JSON.stringify(rec.classes),
    mod.MODEL.classes.join(','));
  check('extractor and dim round-trip',
    mod.MODEL.extractor === rec.extractor && mod.MODEL.dim === rec.dim);
  check('metrics round-trip', JSON.stringify(mod.MODEL.metrics) === JSON.stringify(rec.metrics));

  const near = (a: number[], b: ArrayLike<number>) =>
    a.length === b.length && a.every((v, i) => Math.abs(v - b[i]!) < 1e-6);
  check('scaler weights survive serialisation', near(mod.MODEL.mu, rec.mu) && near(mod.MODEL.sd, rec.sd));
  check('head weights survive serialisation', near(mod.MODEL.W, rec.W) && near(mod.MODEL.b, rec.b));
  check('W keeps its k*dim length, so predict() can index it',
    mod.MODEL.W.length === rec.classes.length * rec.dim,
    `${mod.MODEL.W.length} === ${rec.classes.length} * ${rec.dim}`);

  /* ── rounding is lossy on purpose, but bounded ────────────────────────── */

  const precise = await loadExport(model({
    mu: Float64Array.from([0.123456789, -0.987654321, 2.000000499]),
  }));
  check('weights are rounded to 6dp, not truncated or dropped',
    Math.abs(precise.MODEL.mu[0] - 0.123457) < 1e-9
    && Math.abs(precise.MODEL.mu[1] + 0.987654) < 1e-9,
    `${precise.MODEL.mu[0]}, ${precise.MODEL.mu[1]}`);

  /* ── escaping: a class name is user-supplied text ─────────────────────── */

  const nasty = ["it's", 'say "hi"', 'back\\slash', 'new\nline', '</script>', '`tick`', '${x}'];
  const esc = await loadExport(model({ classes: nasty, dim: 2, W: Float64Array.from(new Array(nasty.length * 2).fill(0.5)), b: Float64Array.from(new Array(nasty.length).fill(0)), mu: Float64Array.from([0, 0]), sd: Float64Array.from([1, 1]) }));
  check('class names that could break out of a string literal do not',
    JSON.stringify(esc.MODEL.classes) === JSON.stringify(nasty),
    `${nasty.length} awkward names survived verbatim`);

  /* ── the self-contained claim is honest ───────────────────────────────── */

  const pureSrc = 'export function features(){ return [1,2,3]; }';
  const pure = exportScript(model({ extractor: 'photo' }), { extractorSource: pureSrc });
  check('a pure extractor is inlined, so the file really is self-contained',
    pure.includes(pureSrc));
  check('a pure extractor says it needs nothing',
    pure.includes('Self-contained'));

  const heavy = exportScript(model({ extractor: 'dinov2' }), { extractorSource: pureSrc });
  check('a backbone extractor does NOT pretend the source is inlined',
    !heavy.includes(pureSrc));
  check('a backbone extractor documents its download instead',
    /@huggingface\/transformers/.test(heavy) && /23 MB/.test(heavy));

  /* ── the header describes the real model ──────────────────────────────── */

  const hdr = exportScript(model(), { name: 'my-detector' });
  check('the name option is honoured', hdr.includes('my-detector'));
  check('the header reports the real accuracy, not a placeholder',
    hdr.includes('98.0%') && hdr.includes('91.0%'));
  check('the header lists the real classes', /circle, square/.test(hdr));

  /* ── CONTROL: the loader can actually fail ────────────────────────────── */

  let threw = false;
  try {
    const bad = join(dir, 'broken.mjs');
    writeFileSync(bad, 'export const MODEL = {;');
    await import(pathToFileURL(bad).href);
  } catch { threw = true; }
  check('CONTROL: a syntactically broken module fails to import',
    threw, 'so the import assertions above are not vacuous');

  console.log(failed ? `\n${failed} FAILED` : '\nALL PASS');
  process.exit(failed ? 1 : 0);
}

main();
