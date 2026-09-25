/**
 * AI Déjà Vu — MVP app logic.
 *
 * Train a classifier in the browser, persist it to IndexedDB, then run live
 * inference where every prediction fires a configurable onMatched / onNotMatched
 * action (console, webhook, both, or off).
 *
 * Everything is shared with the CLI: same dataset generator, same extractors,
 * same Pipeline, same integrity checks.
 */
import { makeDataset, CLASSES, type Surface } from './dataset.js';
import { EXTRACTORS, type ExtractorName } from './features.js';
import { Pipeline, confusionMatrix, type Mat } from './linalg.js';
import { runChecks, separationRatio } from './verify.js';
import { store, hasIDB, type StoredModel, type MatchEvent } from './store.js';
import { toRecord, fromRecord, toJSON, classify } from './serialize.js';
import { dispatch, defaultConfig, type ActionConfig, type Sink } from './actions.js';
import { renderVerdict } from './meters.js';
import { quiet } from './failsafe.js';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

const mk = (w: number, h: number): Surface => {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  return cv as unknown as Surface;
};

let logEl: HTMLElement;
const log = (m: string, cls = '') => {
  const t = new Date().toLocaleTimeString();
  logEl.innerHTML += `\n<span class="dim">${t}</span> ${cls ? `<span class="${cls}">${m}</span>` : m}`;
  logEl.scrollTop = logEl.scrollHeight;
};

let activeModel: { rec: StoredModel; pipe: Pipeline } | null = null;
let config: ActionConfig = defaultConfig([...CLASSES]);

// ---------------------------------------------------------------- config UI
function readConfig(): ActionConfig {
  const targets = [...document.querySelectorAll<HTMLInputElement>('.tgt:checked')]
    .map(i => i.value);
  return {
    targets,
    threshold: +$<HTMLInputElement>('thr').value / 100,
    onMatched: $<HTMLSelectElement>('sinkM').value as Sink,
    onNotMatched: $<HTMLSelectElement>('sinkN').value as Sink,
    webhookUrl: $<HTMLInputElement>('hookUrl').value.trim(),
    webhookMethod: $<HTMLSelectElement>('hookMethod').value as 'POST' | 'GET',
    cooldownMs: +$<HTMLInputElement>('cool').value,
  };
}

function syncConfig() {
  config = readConfig();
  $('thrLbl').textContent = `${$<HTMLInputElement>('thr').value}%`;
  localStorage.setItem('aidejavu.cfg', JSON.stringify(config));
}

function restoreConfig() {
  try {
    const raw = localStorage.getItem('aidejavu.cfg');
    if (!raw) return;
    const c = JSON.parse(raw) as ActionConfig;
    $<HTMLInputElement>('thr').value = String(Math.round(c.threshold * 100));
    $<HTMLSelectElement>('sinkM').value = c.onMatched;
    $<HTMLSelectElement>('sinkN').value = c.onNotMatched;
    $<HTMLInputElement>('hookUrl').value = c.webhookUrl ?? '';
    $<HTMLSelectElement>('hookMethod').value = c.webhookMethod ?? 'POST';
    $<HTMLInputElement>('cool').value = String(c.cooldownMs ?? 0);
    document.querySelectorAll<HTMLInputElement>('.tgt').forEach(i => {
      i.checked = (c.targets ?? []).includes(i.value);
    });
  } catch (e) {
    // Same as studio: the page must still load, but the user losing their
    // saved targets with no message is a silent data loss.
    quiet('mvp.loadConfig', () => { throw e; }, null);
  }
}

// ---------------------------------------------------------------- training
async function train(extractor: ExtractorName, nTrain: number, nTest: number) {
  const { fn, dim, label } = EXTRACTORS[extractor];
  log(`training on ${nTrain * 3} / ${nTest * 3} images with ${label}…`);
  const items = makeDataset(mk, nTrain, nTest);

  const t0 = performance.now();
  const Xtr: Mat = [], Xte: Mat = [];
  const ytr: number[] = [], yte: number[] = [];
  const ktr: string[] = [], kte: string[] = [];
  for (const it of items) {
    const v = fn(it.surface);
    if (it.split === 'train') { Xtr.push(v); ytr.push(it.label); ktr.push(it.key); }
    else { Xte.push(v); yte.push(it.label); kte.push(it.key); }
  }
  const embMs = (performance.now() - t0) / items.length;

  const pipe = new Pipeline({ epochs: 300 }).fit(Xtr, ytr, CLASSES.length);
  const train = pipe.score(Xtr, ytr), test = pipe.score(Xte, yte);
  const ratio = separationRatio(Xte, yte);
  const preds = Xte.map(x => pipe.predict(x));
  const cm = confusionMatrix(yte, preds, CLASSES.length);
  const checks = runChecks(Xtr, ytr, Xte, yte, ktr, kte, CLASSES.length);

  const rec = toRecord(pipe, {
    name: `${extractor} · ${new Date().toLocaleString()}`,
    extractor, classes: [...CLASSES],
    metrics: { train, test, ratio, nTrain: Xtr.length, nTest: Xte.length },
  });

  if (hasIDB) { await store.saveModel(rec); log(`saved to IndexedDB (id ${rec.id})`, 'g'); }
  else log('IndexedDB unavailable — model kept in memory only', 'o');

  activeModel = { rec, pipe };
  log(`trained: test ${pct(test)} · ratio ${ratio.toFixed(2)} · ${embMs.toFixed(1)} ms/img`,
      test > 0.9 ? 'g' : 'o');

  renderMetrics(rec, cm, checks, embMs);
  await refreshModels();
  $<HTMLButtonElement>('classify').disabled = false;
}

function renderMetrics(rec: StoredModel, cm: number[][],
                       checks: { name: string; pass: boolean; detail: string }[], embMs: number) {
  const cmRows = cm.map((row, i) =>
    `<tr><td>${CLASSES[i]}</td>${row.map((v, j) =>
      `<td class="${v && i === j ? 'g' : v ? 'r' : 'dim'}">${v}</td>`).join('')}</tr>`).join('');
  const failedN = checks.filter(c => !c.pass).length;
  $('metrics').innerHTML = `
    <div class="kpis">
      <div class="kpi"><div class="n ${rec.metrics.test > 0.9 ? 'g' : 'o'}">${pct(rec.metrics.test)}</div><div class="l">TEST</div></div>
      <div class="kpi"><div class="n dim">${pct(rec.metrics.train)}</div><div class="l">TRAIN</div></div>
      <div class="kpi"><div class="n b">${rec.dim}</div><div class="l">DIMS</div></div>
      <div class="kpi"><div class="n o">${embMs.toFixed(1)}</div><div class="l">MS/IMG</div></div>
    </div>
    <table class="cm"><tr><th>true ↓ pred →</th>${CLASSES.map(c => `<th>${c}</th>`).join('')}</tr>${cmRows}</table>
    <div class="checks">${checks.map(c =>
      `<div><span class="${c.pass ? 'g' : 'r'}">[${c.pass ? 'PASS' : 'FAIL'}]</span> ${c.name}` +
      `<span class="dim"> ${c.detail}</span></div>`).join('')}</div>
    <div class="dim" style="font-size:11.5px;margin-top:6px">
      ${failedN ? `${failedN} check(s) failed` : 'all integrity checks passed'} ·
      head ${(rec.W.length + rec.b.length + rec.mu.length + rec.sd.length) * 8} bytes
    </div>`;
}

// ---------------------------------------------------------------- models list
async function refreshModels() {
  if (!hasIDB) return;
  const models = await store.listModels();
  const sel = $<HTMLSelectElement>('modelSel');
  sel.innerHTML = models.map(m =>
    `<option value="${m.id}"${activeModel?.rec.id === m.id ? ' selected' : ''}>` +
    `${m.extractor} · ${pct(m.metrics.test)} · ${m.dim}-d · ${new Date(m.createdAt).toLocaleTimeString()}</option>`).join('');
  // With nothing stored, the select renders as an empty void and Load / Export
  // JSON / Delete stay enabled — three primary-looking actions that can only
  // fail, next to a line that already says "0 model(s)". Reflect the real state
  // in the controls instead of only in the caption.
  const empty = models.length === 0;
  if (empty) sel.innerHTML = '<option value="" disabled selected>no models stored yet</option>';
  sel.disabled = empty;
  for (const id of ['load', 'export', 'del']) {
    const b = $<HTMLButtonElement>(id);
    b.disabled = empty;
    b.title = empty ? 'Train and save a model first' : '';
  }

  const u = await store.usage();
  $('usage').textContent =
    `${u.models} model(s), ${u.events} event(s), ~${(u.bytes / 1024).toFixed(1)} KB of weights`;
}

// ---------------------------------------------------------------- inference
async function runPrediction(surface: Surface, source: string) {
  if (!activeModel) { log('no model — train or load one first', 'r'); return; }
  const { rec, pipe } = activeModel;
  const fn = EXTRACTORS[rec.extractor as ExtractorName]?.fn;
  if (!fn) { log(`extractor ${rec.extractor} unavailable in this build`, 'r'); return; }

  const res = classify(pipe, rec.classes, fn(surface));
  const out = await dispatch({
    predicted: res.predicted, confidence: res.confidence, probs: res.probs,
    classes: rec.classes, modelId: rec.id, modelName: rec.name,
  }, config, log);

  const ev: MatchEvent = {
    modelId: rec.id, ts: Date.now(), predicted: res.predicted,
    confidence: res.confidence, matched: out.matched, probs: res.probs,
    delivered: out.delivered,
  };
  if (hasIDB) await store.addEvent(ev);

  // Same animated meters as the Studio — the demo previously showed only a
  // verdict line, so the two pages looked inconsistent.
  renderVerdict($('verdict'), rec.classes, res, out, source,
                config.threshold, config.targets);
  $('payload').textContent = JSON.stringify(out.payload, null, 2);
  await renderEvents();
}

async function renderEvents() {
  if (!hasIDB) return;
  const evs = await store.listEvents(25);
  $('events').innerHTML = evs.length
    ? evs.map(e =>
        `<tr><td class="dim">${new Date(e.ts).toLocaleTimeString()}</td>` +
        `<td class="${e.matched ? 'g' : 'dim'}">${e.matched ? 'MATCH' : '—'}</td>` +
        `<td>${e.predicted}</td><td>${pct(e.confidence)}</td>` +
        `<td class="dim">${e.delivered}</td></tr>`).join('')
    : '<tr><td colspan="5" class="dim">no events yet</td></tr>';
}

// ---------------------------------------------------------------- boot
export async function boot() {
  logEl = $('log');
  log(`AI Déjà Vu ready · IndexedDB ${hasIDB ? 'available' : 'UNAVAILABLE'}`,
      hasIDB ? 'g' : 'r');

  // target checkboxes
  $('targets').innerHTML = CLASSES.map((c, i) =>
    `<label><input type="checkbox" class="tgt" value="${c}"${i === 0 ? ' checked' : ''}> ${c}</label>`).join('');
  restoreConfig();
  syncConfig();

  document.querySelectorAll('input,select').forEach(el =>
    el.addEventListener('change', syncConfig));
  $('thr').addEventListener('input', syncConfig);

  $('train').addEventListener('click', async () => {
    const btn = $<HTMLButtonElement>('train');
    btn.disabled = true;
    try {
      await train(
        $<HTMLSelectElement>('extractor').value as ExtractorName,
        +$<HTMLInputElement>('nTrain').value,
        +$<HTMLInputElement>('nTest').value,
      );
    } catch (e: any) { log('training failed: ' + (e?.message ?? e), 'r'); }
    btn.disabled = false;
  });

  $('load').addEventListener('click', async () => {
    const id = $<HTMLSelectElement>('modelSel').value;
    if (!id) return;
    const rec = await store.getModel(id);
    if (!rec) { log('model not found', 'r'); return; }
    activeModel = { rec, pipe: fromRecord(rec) };
    log(`loaded ${rec.extractor} · ${rec.dim}-d · test ${pct(rec.metrics.test)}`, 'g');
    $<HTMLButtonElement>('classify').disabled = false;
  });

  $('del').addEventListener('click', async () => {
    const id = $<HTMLSelectElement>('modelSel').value;
    if (!id) return;
    await store.deleteModel(id);
    if (activeModel?.rec.id === id) { activeModel = null; $<HTMLButtonElement>('classify').disabled = true; }
    log('model deleted', 'o');
    await refreshModels();
  });

  $('export').addEventListener('click', async () => {
    const id = $<HTMLSelectElement>('modelSel').value;
    const rec = id ? await store.getModel(id) : activeModel?.rec;
    if (!rec) return;
    const blob = new Blob([toJSON(rec)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `aidejavu-${rec.id}.json`;
    a.click();
    log('exported model JSON', 'g');
  });

  $('clearEv').addEventListener('click', async () => {
    await store.clearEvents(); await renderEvents(); await refreshModels();
    log('event log cleared', 'o');
  });

  // ---- drawing pad
  const pad = $<HTMLCanvasElement>('pad');
  const pcx = pad.getContext('2d')!;
  const clear = () => { pcx.fillStyle = '#3c3c3c'; pcx.fillRect(0, 0, 128, 128); };
  clear();
  let drawing = false;
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
  $('clrPad').addEventListener('click', clear);
  $('classify').addEventListener('click', () =>
    runPrediction(pad as unknown as Surface, 'drawing'));

  // ---- random sample from the generator, useful for testing the actions
  $('sample').addEventListener('click', () => {
    const items = makeDataset(mk, 1, 1, Date.now() & 0xffff);
    const it = items[Math.floor(Math.random() * items.length)];
    const pv = $<HTMLCanvasElement>('preview');
    pv.getContext('2d')!.drawImage(it.surface as unknown as HTMLCanvasElement, 0, 0);
    log(`sampled a ${CLASSES[it.label]}`);
    runPrediction(it.surface, `sample(${CLASSES[it.label]})`);
  });

  await refreshModels();
  await renderEvents();
}
