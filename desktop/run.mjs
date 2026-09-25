#!/usr/bin/env node
/**
 * Desktop runner — no browser, no cache server, no patches.
 *
 * On a glibc desktop (Linux/macOS/Windows) `onnxruntime-node` loads its native
 * binary, so transformers.js runs headless and MUCH faster than WASM. The whole
 * pipeline is the same as the phone version; only the plumbing differs:
 *
 *   phone   : browser + WASM + /cache/put across turns (native ORT is glibc-linked)
 *   desktop : plain node, native ORT, one process, no cache needed
 *
 * Usage:  node desktop/run.mjs [--model dinov2-small] [--train 30] [--test 12]
 */
import { AutoModel, AutoProcessor, RawImage, env } from '@huggingface/transformers';
import { createCanvas } from './canvas-shim.mjs';

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > -1 ? process.argv[i + 1] : d;
};
const MODEL_ARG = arg('model', 'Xenova/dinov2-small');
const N_TRAIN = +arg('train', 30);
const N_TEST = +arg('test', 12);
const CLASSES = ['circle', 'square', 'triangle'];

// Desktop can fetch straight from the HF hub — the CDN stall was a WebView issue.
env.allowRemoteModels = true;

// ---- seeded RNG, identical to the browser build so datasets match exactly
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ---- dataset: equal filled AREA per shape, else brightness leaks the label
function drawShape(kind, rng, S = 128) {
  const cv = createCanvas(S, S);
  const cx = cv.getContext('2d');
  const base = [30 + rng() * 60, 30 + rng() * 60, 30 + rng() * 60];
  const id = cx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    for (let c = 0; c < 3; c++)
      id.data[i * 4 + c] = Math.max(0, Math.min(255, base[c] + (rng() * 50 - 25)));
    id.data[i * 4 + 3] = 255;
  }
  cx.putImageData(id, 0, 0);

  const col = `rgb(${90 + rng() * 165 | 0},${90 + rng() * 165 | 0},${90 + rng() * 165 | 0})`;
  const area = Math.pow(20 + rng() * 10, 2) * Math.PI;
  const r = Math.sqrt(area / Math.PI);
  const s = Math.sqrt(area) / 2;
  const t = Math.sqrt(area * 4 / (3 * Math.sqrt(3)));
  const bound = Math.ceil(Math.max(r, s * Math.SQRT2, t));
  const px = bound + 4 + rng() * (S - 2 * (bound + 4));
  const py = bound + 4 + rng() * (S - 2 * (bound + 4));
  const rot = rng() * Math.PI * 2;

  cx.fillStyle = col;
  cx.save();
  cx.translate(px, py); cx.rotate(rot);
  cx.beginPath();
  if (kind === 'circle') cx.arc(0, 0, r, 0, Math.PI * 2);
  else if (kind === 'square') cx.rect(-s, -s, 2 * s, 2 * s);
  else {
    for (let i = 0; i < 3; i++) {
      const a = i * 2 * Math.PI / 3;
      i ? cx.lineTo(t * Math.cos(a), t * Math.sin(a))
        : cx.moveTo(t * Math.cos(a), t * Math.sin(a));
    }
    cx.closePath();
  }
  cx.fill();
  cx.restore();
  return cv;
}

// ---- multinomial logistic regression = TM's dense softmax head
class Head {
  constructor(d, k) {
    this.d = d; this.k = k;
    this.W = new Float64Array(d * k); this.b = new Float64Array(k);
    this.mu = new Float64Array(d); this.sd = new Float64Array(d).fill(1);
  }
  scale(X) {
    for (let j = 0; j < this.d; j++) {
      let m = 0; for (const x of X) m += x[j]; m /= X.length;
      let v = 0; for (const x of X) v += (x[j] - m) ** 2;
      this.mu[j] = m; this.sd[j] = Math.sqrt(v / X.length) + 1e-8;
    }
  }
  z(x) { const o = new Float64Array(this.d);
    for (let j = 0; j < this.d; j++) o[j] = (x[j] - this.mu[j]) / this.sd[j]; return o; }
  soft(z) {
    const l = new Float64Array(this.k);
    for (let c = 0; c < this.k; c++) {
      let s = this.b[c];
      for (let j = 0; j < this.d; j++) s += this.W[c * this.d + j] * z[j];
      l[c] = s;
    }
    const m = Math.max(...l); const e = l.map(v => Math.exp(v - m));
    const t = e.reduce((a, b) => a + b, 0); return e.map(v => v / t);
  }
  fit(X, y, ep = 300, lr = 0.1, l2 = 1e-4) {
    this.scale(X);
    const Z = X.map(x => this.z(x)); const idx = [...Z.keys()];
    const rng = mulberry32(7);
    for (let e = 0; e < ep; e++) {
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      for (const i of idx) {
        const p = this.soft(Z[i]);
        for (let c = 0; c < this.k; c++) {
          const g = p[c] - (y[i] === c ? 1 : 0);
          for (let j = 0; j < this.d; j++) {
            const w = c * this.d + j;
            this.W[w] -= lr * (g * Z[i][j] + l2 * this.W[w]);
          }
          this.b[c] -= lr * g;
        }
      }
    }
  }
  pred(x) { const p = this.soft(this.z(x));
    let b = 0; for (let c = 1; c < p.length; c++) if (p[c] > p[b]) b = c;
    return { label: b, conf: p[b] }; }
}

(async () => {
  console.log(`platform : ${process.platform}/${process.arch}  node ${process.version}`);
  console.log(`model    : ${MODEL_ARG}`);

  let t0 = Date.now();
  const model = await AutoModel.from_pretrained(MODEL_ARG, { dtype: 'q8' });
  const proc = await AutoProcessor.from_pretrained(MODEL_ARG);
  console.log(`backend  : ${model.sessions?.model?.handler?.constructor?.name ?? 'n/a'}`);
  console.log(`outputs  : ${JSON.stringify(model.sessions?.model?.outputNames)}`);
  console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

  const rng = mulberry32(1234);
  const items = [];
  for (const [split, n] of [['train', N_TRAIN], ['test', N_TEST]])
    for (let li = 0; li < CLASSES.length; li++)
      for (let i = 0; i < n; i++)
        items.push({ split, label: li, canvas: drawShape(CLASSES[li], rng) });
  console.log(`dataset  : ${items.filter(i => i.split === 'train').length} train / ` +
              `${items.filter(i => i.split === 'test').length} test`);

  const embed = async (cv) => {
    const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height);
    const img = new RawImage(new Uint8ClampedArray(d.data), cv.width, cv.height, 4);
    const out = await model(await proc(img));
    const t = out.last_hidden_state ?? out.pooler_output ?? out[Object.keys(out)[0]];
    if (t.dims.length === 3) {                    // [1, tokens, D] -> CLS token
      const D = t.dims[2];
      return Float64Array.from({ length: D }, (_, j) => t.data[j]);
    }
    if (t.dims.length === 4) {                    // [1,C,H,W] -> global avg pool
      const [, C, H, W] = t.dims, hw = H * W;
      return Float64Array.from({ length: C }, (_, c) => {
        let s = 0; for (let i = 0; i < hw; i++) s += t.data[c * hw + i]; return s / hw;
      });
    }
    return Float64Array.from(t.data);
  };

  t0 = Date.now();
  const Xtr = [], ytr = [], Xte = [], yte = [];
  let n = 0;
  for (const it of items) {
    const v = await embed(it.canvas);
    if (it.split === 'train') { Xtr.push(v); ytr.push(it.label); }
    else { Xte.push(v); yte.push(it.label); }
    if (++n % 20 === 0) process.stdout.write(`\r  embedded ${n}/${items.length}`);
  }
  const emS = (Date.now() - t0) / 1000;
  console.log(`\r  embedded ${items.length} in ${emS.toFixed(1)}s ` +
              `(${(emS * 1000 / items.length).toFixed(0)} ms each)\n`);

  const head = new Head(Xtr[0].length, CLASSES.length);
  t0 = Date.now();
  head.fit(Xtr, ytr);
  console.log(`head     : ${Xtr[0].length}-d, trained in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const acc = (X, y) => X.filter((x, i) => head.pred(x).label === y[i]).length / X.length;
  const cm = CLASSES.map(() => CLASSES.map(() => 0));
  Xte.forEach((x, i) => cm[yte[i]][head.pred(x).label]++);

  console.log(`\nTRAIN ${(acc(Xtr, ytr) * 100).toFixed(1)}%   ` +
              `TEST ${(acc(Xte, yte) * 100).toFixed(1)}%   chance 33.3%`);
  console.log('confusion (rows=true):');
  console.log('            ' + CLASSES.map(c => c.padStart(9)).join(''));
  cm.forEach((row, i) =>
    console.log('  ' + CLASSES[i].padEnd(10) + row.map(v => String(v).padStart(9)).join('')));
})();
