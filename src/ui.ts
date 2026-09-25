/**
 * Browser UI — pure TypeScript, shares every module with the CLI.
 *
 * Same code paths as `node dist/cli.js`: same dataset generator, same
 * extractors, same Pipeline, same integrity checks. The only browser-specific
 * part is the DOM and the optional DINOv2 backbone (ONNX runs only here).
 */
import { makeDataset, CLASSES, drawShape, type Surface } from './dataset.js';
import { EXTRACTORS, type ExtractorName } from './features.js';
import { Pipeline, confusionMatrix, mulberry32, type Mat, type Vec } from './linalg.js';
import { runChecks, separationRatio, rotationDiagnostic } from './verify.js';

const $ = (id: string) => document.getElementById(id)!;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

/** Real browser canvas — the CLI swaps in the pure-JS shim here. */
const mk = (w: number, h: number): Surface => {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  return cv as unknown as Surface;
};

let logEl: HTMLElement;
const log = (m: string, cls = '') => {
  logEl.innerHTML += `\n${cls ? `<span class="${cls}">${m}</span>` : m}`;
  logEl.scrollTop = logEl.scrollHeight;
};

interface RunResult {
  name: string; dim: number; test: number; train: number;
  cm: number[][]; ratio: number; ms: number;
  preds: { label: number; conf: number }[];
  items: { label: number; surface: Surface }[];
  checks: { name: string; pass: boolean; detail: string }[];
}

function renderResult(r: RunResult) {
  const host = $('results');
  const card = document.createElement('div');
  card.className = 'card';
  const cmRows = r.cm.map((row, i) =>
    `<tr><td>${CLASSES[i]}</td>${row.map((v, j) =>
      `<td class="${v && i === j ? 'g' : v ? 'r' : 'dim'}">${v}</td>`).join('')}</tr>`).join('');
  const checkRows = r.checks.map(c =>
    `<div class="chk"><span class="${c.pass ? 'g' : 'r'}">[${c.pass ? 'PASS' : 'FAIL'}]</span> ${c.name}` +
    `${c.detail ? `<span class="dim"> — ${c.detail}</span>` : ''}</div>`).join('');

  card.innerHTML = `
    <div class="rowh"><b>${r.name}</b><span class="tag">${r.dim}-d</span></div>
    <div class="kpis">
      <div class="kpi"><div class="n ${r.test > 0.9 ? 'g' : r.test > 0.6 ? 'o' : 'r'}">${pct(r.test)}</div><div class="l">TEST</div></div>
      <div class="kpi"><div class="n dim">${pct(r.train)}</div><div class="l">TRAIN</div></div>
      <div class="kpi"><div class="n b">${r.ratio.toFixed(2)}</div><div class="l">SEPARATION</div></div>
      <div class="kpi"><div class="n o">${r.ms.toFixed(0)}</div><div class="l">MS/IMG</div></div>
    </div>
    <table class="cm"><tr><th>true ↓ pred →</th>${CLASSES.map(c => `<th>${c}</th>`).join('')}</tr>${cmRows}</table>
    <div class="checks">${checkRows}</div>
    <div class="thumbs"></div>`;
  host.appendChild(card);

  const th = card.querySelector('.thumbs') as HTMLElement;
  r.items.forEach((it, i) => {
    const p = r.preds[i];
    const bad = p.label !== it.label;
    const d = document.createElement('div');
    d.className = 'th' + (bad ? ' err' : '');
    d.appendChild(it.surface as unknown as HTMLCanvasElement);
    const s = document.createElement('span');
    s.textContent = `${CLASSES[p.label].slice(0, 3)} ${Math.round(p.conf * 100)}%`;
    d.appendChild(s);
    d.title = `true ${CLASSES[it.label]} → pred ${CLASSES[p.label]} (${p.conf.toFixed(3)})`;
    th.appendChild(d);
  });
}

function runExtractor(name: ExtractorName, nTrain: number, nTest: number): RunResult {
  const { fn, dim, label } = EXTRACTORS[name];
  const items = makeDataset(mk, nTrain, nTest);
  const t0 = performance.now();
  const Xtr: Mat = [], Xte: Mat = [];
  const ytr: number[] = [], yte: number[] = [];
  const ktr: string[] = [], kte: string[] = [];
  const testItems: { label: number; surface: Surface }[] = [];
  for (const it of items) {
    const v = fn(it.surface);
    if (it.split === 'train') { Xtr.push(v); ytr.push(it.label); ktr.push(it.key); }
    else { Xte.push(v); yte.push(it.label); kte.push(it.key); testItems.push(it); }
  }
  const ms = (performance.now() - t0) / items.length;
  const pipe = new Pipeline({ epochs: 300 }).fit(Xtr, ytr, CLASSES.length);
  const preds = Xte.map(x => {
    const probs = pipe.predictProba(x);
    let b = 0;
    for (let c = 1; c < probs.length; c++) if (probs[c] > probs[b]) b = c;
    return { label: b, conf: probs[b] };
  });
  return {
    name: label, dim, ms,
    train: pipe.score(Xtr, ytr), test: pipe.score(Xte, yte),
    cm: confusionMatrix(yte, preds.map(p => p.label), CLASSES.length),
    ratio: separationRatio(Xte, yte),
    preds, items: testItems,
    checks: runChecks(Xtr, ytr, Xte, yte, ktr, kte, CLASSES.length),
  };
}

/** Optional: real pretrained DINOv2 features. Only available in the browser. */
async function runBackbone(nTrain: number, nTest: number): Promise<RunResult> {
  log('importing transformers.js…');
  const T: any = await import(/* @vite-ignore */ '../transformers.web.js');
  const { env, AutoModel, AutoProcessor, RawImage } = T;
  env.allowLocalModels = true;
  env.localModelPath = '/models/';
  env.allowRemoteModels = false;
  env.backends.onnx.wasm.numThreads = 1;
  log('loading dinov2-small (local weights)…');
  const model = await AutoModel.from_pretrained('dinov2-small', { dtype: 'q8', device: 'wasm' });
  const proc = await AutoProcessor.from_pretrained('dinov2-small');
  log('backbone loaded (frozen, 0 params trained)', 'g');

  const items = makeDataset(mk, nTrain, nTest);
  const embed = async (s: Surface): Promise<Vec> => {
    const d = s.getContext('2d').getImageData(0, 0, s.width, s.height);
    const img = new RawImage(new Uint8ClampedArray(d.data), s.width, s.height, 4);
    const out = await model(await proc(img));
    const t = out.last_hidden_state;
    const D = t.dims[2];
    return Float64Array.from({ length: D }, (_, j) => t.data[j]);   // CLS token
  };

  const t0 = performance.now();
  const Xtr: Mat = [], Xte: Mat = [];
  const ytr: number[] = [], yte: number[] = [];
  const ktr: string[] = [], kte: string[] = [];
  const testItems: { label: number; surface: Surface }[] = [];
  let n = 0;
  for (const it of items) {
    const v = await embed(it.surface);
    if (it.split === 'train') { Xtr.push(v); ytr.push(it.label); ktr.push(it.key); }
    else { Xte.push(v); yte.push(it.label); kte.push(it.key); testItems.push(it); }
    if (++n % 10 === 0) log(`  embedded ${n}/${items.length}`);
    await new Promise(r => setTimeout(r, 0));      // yield: keeps the page responsive
  }
  const ms = (performance.now() - t0) / items.length;
  const pipe = new Pipeline({ epochs: 300 }).fit(Xtr, ytr, CLASSES.length);
  const preds = Xte.map(x => {
    const probs = pipe.predictProba(x);
    let b = 0;
    for (let c = 1; c < probs.length; c++) if (probs[c] > probs[b]) b = c;
    return { label: b, conf: probs[b] };
  });
  return {
    name: 'DINOv2-small CLS (pretrained, frozen)', dim: Xtr[0].length, ms,
    train: pipe.score(Xtr, ytr), test: pipe.score(Xte, yte),
    cm: confusionMatrix(yte, preds.map(p => p.label), CLASSES.length),
    ratio: separationRatio(Xte, yte), preds, items: testItems,
    checks: runChecks(Xtr, ytr, Xte, yte, ktr, kte, CLASSES.length),
  };
}

function rotationPanel() {
  const sq = (rot: number) => {
    const cv = mk(128, 128); const cx = cv.getContext('2d');
    cx.fillStyle = 'rgb(60,60,60)'; cx.fillRect(0, 0, 128, 128);
    cx.fillStyle = 'rgb(200,200,200)';
    cx.save(); cx.translate(64, 64); cx.rotate((rot * Math.PI) / 180);
    cx.beginPath(); cx.rect(-20, -20, 40, 40); cx.fill(); cx.restore();
    return cv;
  };
  const ci = () => {
    const cv = mk(128, 128); const cx = cv.getContext('2d');
    cx.fillStyle = 'rgb(60,60,60)'; cx.fillRect(0, 0, 128, 128);
    cx.fillStyle = 'rgb(200,200,200)';
    cx.beginPath(); cx.arc(64, 64, Math.sqrt(1600 / Math.PI), 0, Math.PI * 2); cx.fill();
    return cv;
  };
  const rows = (['v1', 'v2'] as const).map(n => {
    const d = rotationDiagnostic(EXTRACTORS[n].fn as any, sq, ci);
    return `<tr><td>${n}</td><td>${d.rotGap.toFixed(3)}</td><td>${d.between.toFixed(3)}</td>` +
      `<td class="${d.ratio > 1 ? 'g' : 'r'}">${d.ratio.toFixed(2)}</td>` +
      `<td class="${d.ratio > 1 ? 'g' : 'r'}">${d.ratio > 1 ? 'OK' : 'rotation dominates'}</td></tr>`;
  }).join('');
  $('rot').innerHTML =
    `<table class="cm"><tr><th>extractor</th><th>rotation gap</th><th>between-class</th>` +
    `<th>ratio</th><th>verdict</th></tr>${rows}</table>` +
    `<div class="dim" style="margin-top:8px;font-size:12px">Rotating the same square vs ` +
    `changing its class. Ratio below 1 means the representation cannot work — no amount ` +
    `of training fixes it.</div>`;
}

export function boot() {
  logEl = $('log');
  log('ready. hand-built extractors need no download.', 'dim');
  rotationPanel();

  $('runFast').addEventListener('click', () => {
    ($('runFast') as HTMLButtonElement).disabled = true;
    $('results').innerHTML = '';
    const nTr = +( $('nTrain') as HTMLInputElement).value;
    const nTe = +( $('nTest') as HTMLInputElement).value;
    log(`running v1 + v2 on ${nTr * 3} train / ${nTe * 3} test…`);
    setTimeout(() => {
      for (const n of ['v1', 'v2'] as const) {
        const r = runExtractor(n, nTr, nTe);
        renderResult(r);
        log(`${n}: test ${pct(r.test)} · ratio ${r.ratio.toFixed(2)} · ` +
            `${r.checks.filter(c => !c.pass).length} check(s) failed`,
            r.test > 0.9 ? 'g' : 'o');
      }
      ($('runFast') as HTMLButtonElement).disabled = false;
    }, 20);
  });

  $('runBb').addEventListener('click', async () => {
    ($('runBb') as HTMLButtonElement).disabled = true;
    try {
      const nTr = +( $('nTrain') as HTMLInputElement).value;
      const nTe = +( $('nTest') as HTMLInputElement).value;
      const r = await runBackbone(nTr, nTe);
      renderResult(r);
      log(`dinov2: test ${pct(r.test)} · ${r.ms.toFixed(0)} ms/img`, 'g');
    } catch (e: any) {
      log('backbone failed: ' + (e?.message ?? e), 'r');
    }
    ($('runBb') as HTMLButtonElement).disabled = false;
  });

  // ---- draw-your-own pad, classified with v2 (instant, no model needed)
  const pad = $('pad') as HTMLCanvasElement;
  const pcx = pad.getContext('2d')!;
  const clear = () => { pcx.fillStyle = '#3c3c3c'; pcx.fillRect(0, 0, 128, 128); };
  clear();
  let drawing = false, trained: Pipeline | null = null;
  const at = (e: PointerEvent) => {
    const r = pad.getBoundingClientRect();
    return [(e.clientX - r.left) * 128 / r.width, (e.clientY - r.top) * 128 / r.height];
  };
  const dot = (x: number, y: number) => {
    pcx.fillStyle = '#c8c8c8'; pcx.beginPath(); pcx.arc(x, y, 9, 0, 7); pcx.fill();
  };
  pad.addEventListener('pointerdown', e => { drawing = true; const [x, y] = at(e); dot(x, y); });
  pad.addEventListener('pointermove', e => { if (drawing) { const [x, y] = at(e); dot(x, y); } });
  addEventListener('pointerup', () => { drawing = false; });
  $('clr').addEventListener('click', () => { clear(); $('myres').textContent = ''; });
  $('cls').addEventListener('click', () => {
    if (!trained) {
      const items = makeDataset(mk, 30, 1);
      const tr = items.filter(i => i.split === 'train');
      trained = new Pipeline({ epochs: 300 })
        .fit(tr.map(i => EXTRACTORS.v2.fn(i.surface)), tr.map(i => i.label), CLASSES.length);
    }
    const probs = trained.predictProba(EXTRACTORS.v2.fn(pad as unknown as Surface));
    let b = 0;
    for (let c = 1; c < probs.length; c++) if (probs[c] > probs[b]) b = c;
    $('myres').innerHTML = `<b class="b">${CLASSES[b]}</b> <span class="dim">` +
      CLASSES.map((c, i) => `${c.slice(0, 3)} ${(probs[i] * 100).toFixed(0)}%`).join(' / ') + '</span>';
  });
}
