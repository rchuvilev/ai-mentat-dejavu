#!/usr/bin/env node
/**
 * Tests for the action layer and model serialisation.
 *
 * These run in Node (no IndexedDB), so storage itself is exercised in the
 * browser suite. What matters here is the pure logic: the match rule, sink
 * routing, cooldown, payload shape, webhook failure handling, and that a
 * serialised model predicts identically to the one it came from.
 */
import { dispatch, defaultConfig, resetCooldowns, type ActionConfig, type Prediction } from './actions.js';
import { Pipeline, mulberry32, type Mat } from './linalg.js';
import { toRecord, fromRecord, toJSON, fromJSON, classify } from './serialize.js';

let failed = 0;
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failed++;
};

const CLASSES = ['circle', 'square', 'triangle'];
const pred = (c: string, conf: number): Prediction => ({
  predicted: c, confidence: conf,
  probs: CLASSES.map(x => (x === c ? conf : (1 - conf) / 2)),
  classes: CLASSES, modelId: 'm1', modelName: 'test',
});

// silence console during dispatch tests, but count the calls
const realInfo = console.info, realWarn = console.warn;
let infoCount = 0, warnCount = 0;
console.info = (...a: any[]) => { infoCount++; };
console.warn = (...a: any[]) => { warnCount++; };

await (async () => {
  // ---------------------------------------------------------- match rule
  const cfg: ActionConfig = {
    ...defaultConfig(CLASSES), targets: ['circle'], threshold: 0.7,
    onMatched: 'console', onNotMatched: 'console',
  };
  resetCooldowns();

  let r = await dispatch(pred('circle', 0.9), cfg);
  check('target + high conf => matched', r.matched && r.delivered === 'console');

  r = await dispatch(pred('circle', 0.5), cfg);
  check('target + low conf => NOT matched', !r.matched, `conf 0.5 < thr 0.7`);

  r = await dispatch(pred('square', 0.99), cfg);
  check('non-target => NOT matched even at 0.99', !r.matched);

  r = await dispatch(pred('circle', 0.7), cfg);
  check('threshold is inclusive (>=)', r.matched, 'conf == threshold');

  // multiple targets
  r = await dispatch(pred('square', 0.8), { ...cfg, targets: ['circle', 'square'] });
  check('multiple targets honoured', r.matched);

  // ---------------------------------------------------------- sink routing
  resetCooldowns();
  r = await dispatch(pred('circle', 0.9), { ...cfg, onMatched: 'off' });
  check('sink off => nothing delivered', r.delivered === 'none' && r.matched);

  const before = infoCount;
  await dispatch(pred('circle', 0.9), { ...cfg, onMatched: 'console' });
  check('console sink calls console.info on match', infoCount === before + 1);

  const wbefore = warnCount;
  await dispatch(pred('square', 0.9), { ...cfg, onNotMatched: 'console' });
  check('console sink calls console.warn on no-match', warnCount === wbefore + 1);

  // independent branches: match->off, no-match->console
  resetCooldowns();
  const split: ActionConfig = { ...cfg, onMatched: 'off', onNotMatched: 'console' };
  const a = await dispatch(pred('circle', 0.9), split);
  const b = await dispatch(pred('square', 0.9), split);
  check('branches configured independently',
        a.delivered === 'none' && b.delivered === 'console');

  // ---------------------------------------------------------- payload
  r = await dispatch(pred('circle', 0.876), { ...cfg, extra: { deviceId: 'x1' } });
  const p: any = r.payload;
  check('payload event name', p.event === 'onMatched', p.event);
  check('payload has rounded confidence', p.confidence === 0.876, String(p.confidence));
  check('payload probabilities keyed by class',
        Object.keys(p.probabilities).join(',') === CLASSES.join(','));
  check('payload merges extra fields', p.deviceId === 'x1');
  check('payload has ISO timestamp', typeof p.ts === 'string' && p.ts.includes('T'));
  const notMatched: any = (await dispatch(pred('square', 0.9), cfg)).payload;
  check('no-match payload names onNotMatched', notMatched.event === 'onNotMatched');

  // ---------------------------------------------------------- cooldown
  resetCooldowns();
  const cd: ActionConfig = { ...cfg, cooldownMs: 5000 };
  const first = await dispatch(pred('circle', 0.9), cd);
  const second = await dispatch(pred('circle', 0.9), cd);
  check('cooldown suppresses the second fire',
        first.delivered === 'console' && second.skipped === 'cooldown');
  // the OTHER branch must not be blocked by the first branch's cooldown
  const other = await dispatch(pred('square', 0.9), { ...cd, onNotMatched: 'console' });
  check('cooldown is per-branch', other.delivered === 'console');

  // ---------------------------------------------------------- webhook
  resetCooldowns();
  const noUrl = await dispatch(pred('circle', 0.9), { ...cfg, onMatched: 'webhook', webhookUrl: '' });
  check('webhook with no URL degrades safely', noUrl.delivered === 'none');

  // unreachable host must be captured, not thrown
  resetCooldowns();
  const bad = await dispatch(pred('circle', 0.9), {
    ...cfg, onMatched: 'webhook', webhookUrl: 'http://127.0.0.1:9/nope',
  });
  check('webhook failure is captured not thrown',
        bad.delivered.startsWith('webhook:err'), bad.delivered.slice(0, 40));

  // real local endpoint
  const { createServer } = await import('node:http');
  const received: any[] = [];
  const srv = createServer((req, res) => {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      received.push({ method: req.method, url: req.url, body });
      res.writeHead(204); res.end();
    });
  });
  await new Promise<void>(r => srv.listen(0, '127.0.0.1', () => r()));
  const port = (srv.address() as any).port;

  resetCooldowns();
  const ok = await dispatch(pred('circle', 0.9), {
    ...cfg, onMatched: 'webhook', webhookUrl: `http://127.0.0.1:${port}/hook`,
  });
  check('webhook POST delivers', ok.delivered === 'webhook:204', ok.delivered);
  check('webhook body is the payload JSON',
        received.length === 1 && JSON.parse(received[0].body).event === 'onMatched');
  check('webhook uses POST by default', received[0].method === 'POST');

  resetCooldowns();
  await dispatch(pred('circle', 0.9), {
    ...cfg, onMatched: 'webhook', webhookMethod: 'GET',
    webhookUrl: `http://127.0.0.1:${port}/hook`,
  });
  check('GET mode puts payload in query string',
        received[1].method === 'GET' && received[1].url.includes('payload='));

  resetCooldowns();
  const both = await dispatch(pred('circle', 0.9), {
    ...cfg, onMatched: 'both', webhookUrl: `http://127.0.0.1:${port}/hook`,
  });
  check("'both' sink hits console AND webhook", both.delivered === 'webhook:204' && received.length === 3);
  srv.close();

  // ---------------------------------------------------------- serialisation
  const rng = mulberry32(5);
  const X: Mat = [], y: number[] = [];
  for (let i = 0; i < 60; i++) {
    const c = i % 3;
    X.push(Float64Array.from([c * 5 + rng(), c * 5 + rng(), rng()]));
    y.push(c);
  }
  const pipe = new Pipeline({ epochs: 200 }).fit(X, y, 3);
  const rec = toRecord(pipe, {
    name: 'unit', extractor: 'v2', classes: CLASSES,
    metrics: { train: 1, test: 1, ratio: 2, nTrain: 60, nTest: 0 },
  });
  check('record captures dim', rec.dim === 3, `dim ${rec.dim}`);
  check('record has an id', typeof rec.id === 'string' && rec.id.length > 4);

  const revived = fromRecord(rec);
  const sameAll = X.every(x => revived.predict(x) === pipe.predict(x));
  check('revived model predicts identically', sameAll);
  const p0 = pipe.predictProba(X[0]), p1 = revived.predictProba(X[0]);
  check('revived probabilities identical',
        Array.from(p0).every((v, i) => Math.abs(v - p1[i]) < 1e-12));

  const round = fromRecord(fromJSON(toJSON(rec)));
  check('JSON round-trip preserves predictions',
        X.every(x => round.predict(x) === pipe.predict(x)));

  const c = classify(revived, CLASSES, X[0]);
  check('classify returns a class name', CLASSES.includes(c.predicted), c.predicted);
  check('classify probs sum to 1',
        Math.abs(c.probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);
  check('classify confidence is the max prob',
        Math.abs(c.confidence - Math.max(...c.probs)) < 1e-12);

  // model size sanity: a head must be small enough to store comfortably
  const bytes = (rec.mu.length + rec.sd.length + rec.W.length + rec.b.length) * 8;
  check('head is small', bytes < 50_000, `${bytes} bytes for ${rec.dim}-d`);
})();

console.info = realInfo; console.warn = realWarn;
// ---------------------------------------------------------------- export
await (async () => {
  const { exportScript } = await import('./exporter.js');
  const { RUNTIME_MEL, RUNTIME_PHOTO } = await import('./runtime-src.js');
  const { writeFileSync, mkdtempSync } = await import('node:fs');
  const { join } = await import('node:path');
  const { tmpdir } = await import('node:os');

  const rng2 = mulberry32(9);
  const Xa: Mat = [], ya: number[] = [];
  for (let i = 0; i < 40; i++) {
    const c = i % 2;
    const v = new Float64Array(200);
    for (let j = 0; j < 200; j++) v[j] = rng2() + (j % 2 === c ? 2 : 0);
    Xa.push(v); ya.push(c);
  }
  const p2 = new Pipeline({ epochs: 200 }).fit(Xa, ya, 2);
  const rec2 = toRecord(p2, {
    name: 'exp', extractor: 'mel', classes: ['low', 'high'],
    metrics: { train: 1, test: 1, ratio: 5, nTrain: 40, nTest: 0 },
  });

  const js = exportScript(rec2, { extractorSource: RUNTIME_MEL, name: 'd' });
  check('export is self-contained for pure-JS extractors',
        !/\bimport\s|\brequire\(/.test(js.replace(/\*[\s\S]*?\*\//g, '')),
        'a dependency would break the Node and Bun targets');
  check('export bundles the weights',
        js.includes('mu:') && js.includes('W:') && js.includes('classes:'));
  check('export documents accuracy and provenance',
        js.includes('trained on') && js.includes('accuracy'));
  check('export is ESM with a default',
        js.includes('export function createDetector') && js.includes('export default'));

  // Actually RUN the generated file — the only proof that matters.
  const dir = mkdtempSync(join(tmpdir(), 'aidejavu-'));
  const file = join(dir, 'd.mjs');
  writeFileSync(file, js);
  const mod = await import(file);
  const fired: string[] = [];
  const det = mod.createDetector({
    targets: ['high'], threshold: 0.7,
    onMatched: (r: any) => fired.push('M:' + r.predicted),
    onNotMatched: (r: any) => fired.push('N:' + r.predicted),
  });
  const hi = det.fromVector(Xa[1]);   // a class-1 vector
  const lo = det.fromVector(Xa[0]);   // a class-0 vector
  check('generated script predicts correctly',
        hi.predicted === 'high' && lo.predicted === 'low',
        `${hi.predicted} / ${lo.predicted}`);
  check('generated script fires the right branch',
        fired.join(',') === 'M:high,N:low', fired.join(','));
  check('generated script exposes keyed probabilities',
        Object.keys(hi.probabilities).join(',') === 'low,high');
  check('generated script rejects a wrong-width vector',
        (() => { try { det.fromVector(new Float64Array(5)); return false; }
                 catch { return true; } })());
  check('generated script rejects an unknown target',
        (() => { try { mod.createDetector({ targets: ['nope'] }); return false; }
                 catch { return true; } })(),
        'fail fast beats never matching');
  const thrower = mod.createDetector({
    targets: ['high'], onMatched: () => { throw new Error('boom'); },
  });
  check('a throwing callback does not break detection',
        thrower.fromVector(Xa[1]).predicted === 'high');
  const cd = mod.createDetector({ targets: ['high'], cooldownMs: 5000, onMatched: () => {} });
  check('cooldown is honoured in the export',
        cd.fromVector(Xa[1]).skipped === undefined &&
        cd.fromVector(Xa[1]).skipped === 'cooldown');

  // a backbone extractor cannot be inlined and must say so
  const rec3 = toRecord(p2, {
    name: 'exp', extractor: 'tmnet', classes: ['a', 'b'],
    metrics: { train: 1, test: 1, ratio: 5, nTrain: 40, nTest: 0 },
  });
  const js3 = exportScript(rec3, {});
  // It names the npm package, not the prose name — assert what it actually says.
  check('backbone export documents its requirement',
        js3.includes('@tensorflow/tfjs') && js3.includes('cannot be inlined') &&
        js3.includes('fromVector'),
        'must not pretend to be self-contained');
})();

console.log(failed ? `\n${failed} FAILURE(S)` : '\nALL PASS');
process.exit(failed ? 1 : 0);
