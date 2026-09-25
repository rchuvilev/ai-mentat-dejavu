/**
 * Studio: record/upload your own samples, label them, train, then run live
 * inference where every prediction fires the configured action.
 *
 * Two modalities share one pipeline — only the capture widget and the extractor
 * differ:
 *   image: camera frame or file  -> embedPhoto (116-d)
 *   audio: mic recording or file -> embedAudio (200-d)
 */
import { Pipeline, confusionMatrix, type Mat, type Vec } from './linalg.js';
import { separationRatio, runChecks } from './verify.js';
import { store, hasIDB, type StoredModel } from './store.js';
import { samples, trainability, makeThumb, findDuplicates, dedupe,
         type Sample, type Modality } from './samples.js';
import { toRecord, fromRecord, toJSON, classify } from './serialize.js';
import { dispatch, defaultConfig, type ActionConfig, type Sink } from './actions.js';
import { embedPhoto, PHOTO_DIM, fileToSurface, videoToSurface } from './photo.js';
import { embedAudio, AUDIO_DIM, decodeAudioFile, rmsDb, resample, toMono } from './audio.js';
import type { Surface } from './dataset.js';
import { renderVerdict } from './meters.js';
import { loadBackbone, backboneReady } from './backbone.js';
import { loadTmNet, tmReady, TM_DIM } from './tmnet.js';
import { loadScNet, scReady, SC_DIM } from './scnet.js';
import { startListening, isSilent, type Listener } from './listen.js';
import { runSelfTest } from './selftest.js';
import { exportScript } from './exporter.js';
import { RUNTIME_PHOTO, RUNTIME_MEL } from './runtime-src.js';
import { quiet } from './failsafe.js';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

let logEl: HTMLElement;
const log = (m: string, cls = '') => {
  const t = new Date().toLocaleTimeString();
  logEl.innerHTML += `\n<span class="dim">${t}</span> ${cls ? `<span class="${cls}">${m}</span>` : m}`;
  logEl.scrollTop = logEl.scrollHeight;
};

/**
 * Available extractors per modality. 'dinov2' is a pretrained backbone: far
 * better generalisation from few examples, ~400x slower per image, and needs a
 * one-time ~23 MB download. The UI states that trade rather than hiding it.
 */
/**
 * Extractor catalogue.
 *
 * Every entry carries the DEVICE class it is sensible on and the TASK it suits,
 * because the numbers alone mislead: DINOv3 has the best features here but is
 * ~400x slower than the hand-built one, so "best" depends entirely on what you
 * are running it on and what you are classifying.
 *
 * Device tiers, from measurements on a Redmi Note 12 Pro (mid-range 2022):
 *   any     — runs everywhere including old tablets, no download
 *   gpu     — needs working WebGL; falls back to CPU at ~17x the cost
 *   desktop — 20+ MB download and seconds per image; fine on a laptop
 *
 * Everything listed exports real FEATURES. Two candidates were removed after
 * measurement: MobileNetV4 and ResNet-18 ONNX exports expose only `logits`,
 * which scored separation 1.40 / 1.53 and 51.7% accuracy — worse than the
 * 5 ms hand-built extractor while costing a download.
 */
interface ExtractorSpec {
  name: string;
  dim: number;
  /** Short name shown in the dropdown. */
  title: string;
  device: 'any' | 'gpu' | 'desktop';
  /** What this is actually good at. */
  task: string;
  cost: string;
}

const EXTRACTORS_BY_MODALITY: Record<Modality, ExtractorSpec[]> = {
  image: [
    { name: 'photo', dim: PHOTO_DIM, title: 'Colour & edges (built-in)',
      device: 'any', task: 'scenes differing in colour, layout or texture',
      cost: '~5 ms · no download' },
    { name: 'tmnet', dim: TM_DIM, title: 'MobileNetV2 (Teachable Machine)',
      device: 'gpu', task: 'everyday objects — the best all-rounder',
      cost: '~70 ms · 1.6 MB' },
    { name: 'dinov3', dim: 384, title: 'DINOv3 small (newest)',
      device: 'desktop', task: 'fine distinctions, few examples',
      cost: '~2 s · 21 MB' },
    { name: 'dinov2', dim: 384, title: 'DINOv2 small (older)',
      device: 'desktop', task: 'same as DINOv3; keep for existing models',
      cost: '~2 s · 23 MB' },
  ],
  audio: [
    { name: 'mel', dim: AUDIO_DIM, title: 'Log-mel bands (built-in)',
      device: 'any', task: 'claps, whistles, alarms, machine hum',
      cost: '~5 ms · no download' },
    { name: 'scnet', dim: SC_DIM, title: 'Speech-commands (Teachable Machine)',
      device: 'gpu', task: 'spoken words and syllables — try this for speech',
      cost: '~40 ms · 2 MB' },
  ],
};

const DEVICE_TAG: Record<ExtractorSpec['device'], string> = {
  any: 'any device',
  gpu: 'needs GPU',
  desktop: 'desktop/laptop',
};

/** Dropdown text: what it is, where it runs, what it costs. */
function extractorLabel(e: ExtractorSpec): string {
  return `${e.title} — ${DEVICE_TAG[e.device]} · ${e.cost} · ${e.dim}-d`;
}

/** Currently selected extractor name, per modality. */
const chosen: Record<Modality, string> = { image: 'photo', audio: 'mel' };

function currentExtractor(): ExtractorSpec {
  const list = EXTRACTORS_BY_MODALITY[modality];
  return list.find(e => e.name === chosen[modality]) ?? list[0];
}

/** Back-compat shim: code below used EXTRACTOR[modality].name */
const EXTRACTOR = new Proxy({} as Record<Modality, { name: string; dim: number }>, {
  get: (_t, k: string) => {
    const m = k as Modality;
    const list = EXTRACTORS_BY_MODALITY[m];
    const e = list.find(x => x.name === chosen[m]) ?? list[0];
    return { name: e.name, dim: e.dim };
  },
});

let modality: Modality = 'image';
let project = 'default';
let active: { rec: StoredModel; pipe: Pipeline } | null = null;
let config: ActionConfig = defaultConfig([]);
let stream: MediaStream | null = null;
/** Which camera to request next; flipped by the switch button. */
let facing: 'environment' | 'user' = 'environment';
/** Cameras the device reports, once permission has been granted. */
let videoInputs: MediaDeviceInfo[] = [];
let recorder: MediaRecorder | null = null;
let meterTimer: number | null = null;

// ---------------------------------------------------------------- features
async function featurise(s: Sample): Promise<Vec> {
  const ex = EXTRACTOR[s.modality].name;
  if (s.vecs?.[ex]) return Float64Array.from(s.vecs[ex]);
  let v: Vec;
  if (s.modality === 'image') {
    const surf = await fileToSurface(s.blob);
    v = await embedFor(ex, surf);
  } else {
    const { samples: pcm, sampleRate } = await decodeAudioFile(s.blob);
    v = await embedAudioFor(ex, pcm, sampleRate);
  }
  s.vecs = { ...(s.vecs ?? {}), [ex]: Array.from(v).map(x => +x.toFixed(6)) };
  if (s.id !== undefined) await samples.update(s);   // cache so retrain is fast
  return v;
}

/** Embed one image surface with whichever extractor the model was trained on. */
async function embedFor(extractor: string, surf: Surface): Promise<Vec> {
  if (extractor === 'dinov2' || extractor === 'dinov3') {
    const bb = await loadBackbone(extractor, m => log(m));
    return bb.embed(surf);
  }
  if (extractor === 'tmnet') {
    const bb = await loadTmNet(m => log(m));
    return bb.embed(surf);
  }
  if (extractor === 'photo') return embedPhoto(surf);
  // Never fall through to a default: the shapes demo stores v1/v2 models in the
  // SAME database, and feeding photo features to a v2 head (26-d vs 116-d) would
  // either throw or, worse, produce a confident wrong answer.
  throw new Error(`this page cannot compute "${extractor}" features`);
}

/** Embed one clip with whichever audio extractor the model was trained on. */
async function embedAudioFor(
  extractor: string, pcm: Float32Array, sampleRate: number,
): Promise<Vec> {
  if (extractor === 'scnet') {
    const bb = await loadScNet(m => log(m));
    return bb.embed(pcm, sampleRate);
  }
  if (extractor === 'mel') return embedAudio(pcm, sampleRate);
  throw new Error(`this page cannot compute "${extractor}" features`);
}

/** Extractors this page can actually run, per modality. */
const RUNNABLE: Record<string, Modality> =
  { photo: 'image', tmnet: 'image', dinov3: 'image', dinov2: 'image',
    mel: 'audio', scnet: 'audio' };

/** Is a stored model usable here, and if not, why? */
function compatibility(rec: StoredModel): { ok: boolean; reason: string } {
  const m = RUNNABLE[rec.extractor];
  if (!m) {
    return {
      ok: false,
      reason: `trained with the "${rec.extractor}" extractor, which belongs to the ` +
              `shapes demo — this page cannot compute those features`,
    };
  }
  return { ok: true, reason: m };
}

/**
 * Turn a getUserMedia rejection into something actionable.
 *
 * NotAllowedError is by far the most common on mobile and in embedded WebViews,
 * where the host app may not hold the camera permission at all — so the useful
 * advice is "use the upload button", not the raw message.
 */
function deviceError(e: any): string {
  const n = e?.name ?? '';
  if (n === 'NotAllowedError' || n === 'SecurityError') {
    return 'permission denied — use the upload button instead ' +
           '(on a phone it offers the camera too)';
  }
  if (n === 'NotFoundError' || n === 'OverconstrainedError') {
    return 'no matching device found — try the upload button';
  }
  if (n === 'NotReadableError') return 'device busy in another app';
  return String(e?.message ?? e).slice(0, 90);
}

/**
 * One line saying what unblocks the next action.
 *
 * ponytail rung 2: steps 1 and 2 would have duplicated #captureHint and
 * #trainability, which already sit beside the controls they describe. The only
 * state NOT visible anywhere was "is a model loaded, and what will it do" — so
 * that is all this renders, and it reads `active`/`config` directly instead of
 * re-querying IndexedDB on every keystroke.
 */
function syncSteps() {
  const el = document.getElementById('steps');
  if (!el) return;
  if (!active) {
    el.className = 'steps';
    el.innerHTML = '<span class="dim">No model yet — collect samples below, then Train. ' +
                   'Classifying and actions unlock after that.</span>';
    return;
  }
  const t = config.targets.length ? config.targets.join(', ') : 'none picked';
  el.className = 'steps live';
  el.innerHTML =
    `<b>${active.rec.classes.join(' / ')}</b>` +
    `<span class="dim"> · ${active.rec.extractor} · match on <b>${t}</b> ` +
    `at ${Math.round(config.threshold * 100)}% · ` +
    `${config.onMatched}/${config.onNotMatched}</span>`;
}

/**
 * Enable only what can actually work right now.
 *
 * The first version left Capture/Record/continuous enabled with no camera, no
 * mic and no model, so the UI invited an action and then reported an error. A
 * disabled control with a hint is clearer than a live one that fails.
 */
function syncControls() {
  const hasLabel = !!$<HTMLInputElement>('label').value.trim();
  const hasStream = !!stream;
  const hasModel = !!active;
  $<HTMLButtonElement>('shoot').disabled = !(hasStream && hasLabel);
  $<HTMLButtonElement>('rec').disabled = !(hasStream && hasLabel);
  $<HTMLButtonElement>('camOff').disabled = !hasStream;
  $<HTMLButtonElement>('micOff').disabled = !hasStream;
  // The flip button lives over the video, so it must never be usable (or even
  // present) without a running camera.
  const flip = $<HTMLButtonElement>('flip');
  flip.disabled = !hasStream;
  if (!hasStream) flip.style.display = 'none';
  $<HTMLButtonElement>('runBtn').disabled = !hasModel;
  $<HTMLButtonElement>('loop').disabled = !hasModel;
  $<HTMLButtonElement>('exportBtn').disabled = !$<HTMLSelectElement>('modelSel').value;
  $<HTMLButtonElement>('loadBtn').disabled = !$<HTMLSelectElement>('modelSel').value;

  const hint = !hasStream
    ? (modality === 'image' ? 'turn the camera on, or upload images'
                            : 'turn the microphone on, or upload audio')
    : !hasLabel ? 'type a label to start collecting'
    : 'ready to capture';
  $('captureHint').textContent = hint;
  syncSteps();
}

// ---------------------------------------------------------------- capture
async function startCamera() {
  await stopCapture();
  // Ask for the preferred camera, but do NOT use `exact`: on a single-camera
  // device an exact facingMode fails outright, whereas `ideal` degrades to
  // whatever exists. Fall back to bare video for the same reason.
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 640 } }, audio: false,
    });
  } catch (e: any) {
    if (e?.name === 'OverconstrainedError' || e?.name === 'NotFoundError') {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } else { throw e; }
  }

  const v = $<HTMLVideoElement>('cam');
  v.srcObject = stream;
  // Mirror the selfie view: users expect a mirror, and an un-mirrored front
  // camera feels broken. The CAPTURED frame is un-mirrored on purpose so the
  // stored sample matches what the lens actually saw.
  v.style.transform = facing === 'user' ? 'scaleX(-1)' : '';
  await v.play();

  // Labels are only populated after permission is granted, so enumerate now.
  try {
    videoInputs = (await navigator.mediaDevices.enumerateDevices())
      .filter(d => d.kind === 'videoinput');
  } catch (e) {
    // Recorded, not swallowed: an empty list hides the flip button, so the
    // user sees a missing control with no explanation. enumerateDevices can
    // fail on permission revocation or in an insecure context.
    videoInputs = [];
    quiet('studio.enumerateDevices', () => { throw e; }, null);
  }
  $('flip').style.display = videoInputs.length > 1 ? '' : 'none';

  const track = stream.getVideoTracks()[0];
  const actual = track?.getSettings?.().facingMode ?? facing;
  $('camWrap').style.display = '';
  syncControls();
  log(`camera on (${actual === 'user' ? 'front' : 'back'}${
    videoInputs.length > 1 ? `, ${videoInputs.length} available` : ''})`, 'g');
}

/** Flip between front and back, restarting the stream. */
async function flipCamera() {
  facing = facing === 'environment' ? 'user' : 'environment';
  try {
    await startCamera();
  } catch (e: any) {
    log(`could not switch camera: ${deviceError(e)}`, 'r');
    facing = facing === 'environment' ? 'user' : 'environment';   // revert
    syncControls();
  }
}

async function startMic() {
  await stopCapture();
  stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  $('micWrap').style.display = '';
  // live level meter, so the user can see the mic is actually working
  const AC: any = (window as any).AudioContext || (window as any).webkitAudioContext;
  const ctx = new AC();
  const src = ctx.createMediaStreamSource(stream);
  const an = ctx.createAnalyser();
  an.fftSize = 1024;
  src.connect(an);
  const buf = new Float32Array(an.fftSize);
  meterTimer = window.setInterval(() => {
    an.getFloatTimeDomainData(buf);
    const db = rmsDb(buf);
    const pctv = Math.max(0, Math.min(100, (db + 60) / 60 * 100));
    $('meterBar').style.width = pctv + '%';
    $('meterVal').textContent = `${db.toFixed(0)} dB`;
  }, 100);
  syncControls();
  log('microphone on', 'g');
}

async function stopCapture() {
  if (meterTimer) { clearInterval(meterTimer); meterTimer = null; }
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  recorder = null;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $('camWrap').style.display = 'none';
  $('micWrap').style.display = 'none';
  $('flip').style.display = 'none';
  syncControls();
}

/** Record a fixed-length clip from the live mic stream. */
function recordClip(ms: number): Promise<Blob> {
  return new Promise((res, rej) => {
    if (!stream) return rej(new Error('microphone not started'));
    const chunks: Blob[] = [];
    let mr: MediaRecorder;
    try { mr = new MediaRecorder(stream); } catch (e) { return rej(e as Error); }
    recorder = mr;
    mr.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    mr.onstop = () => res(new Blob(chunks, { type: mr.mimeType || 'audio/webm' }));
    mr.onerror = (e: any) => rej(e?.error ?? new Error('recorder error'));
    mr.start();
    setTimeout(() => { if (mr.state !== 'inactive') mr.stop(); }, ms);
  });
}

/** Horizontally flip a canvas (front camera preview is mirrored, the file is not). */
function unmirror(src: HTMLCanvasElement): HTMLCanvasElement {
  const cv = document.createElement('canvas');
  cv.width = src.width; cv.height = src.height;
  const cx = cv.getContext('2d')!;
  cx.translate(cv.width, 0);
  cx.scale(-1, 1);
  cx.drawImage(src, 0, 0);
  return cv;
}

function canvasToBlob(cv: HTMLCanvasElement): Promise<Blob> {
  return new Promise(res => cv.toBlob(b => res(b!), 'image/jpeg', 0.85));
}

// ---------------------------------------------------------------- add samples
async function addSample(blob: Blob, source: string, label: string, durationMs?: number) {
  if (!label) { log('pick or type a label first', 'r'); return; }
  const s: Sample = {
    project, modality, label, createdAt: Date.now(), source, blob, durationMs,
  };
  if (modality === 'image') {
    try { s.thumb = await makeThumb(blob); } catch { /* preview is optional */ }
  }
  await samples.add(s);
  log(`+1 "${label}" (${source})`, 'g');
  await refreshSamples();
  syncControls();
}

// ---------------------------------------------------------------- sample list
async function refreshSamples() {
  const counts = await samples.labels(project, modality);
  const t = trainability(counts);
  const rows = await samples.list(project, modality);

  $('labelChips').innerHTML = Object.keys(counts).sort().map(l =>
    `<span class="chip" data-label="${l}">${l} <b>${counts[l]}</b></span>`).join('')
    || '<span class="dim">no samples yet</span>';

  const dup = findDuplicates(rows, EXTRACTOR[modality].name);
  const dupNote = dup.duplicateCount
    ? ` · <span class="r">${dup.duplicateCount} duplicate sample(s)</span>` +
      ` <button id="dedupe" class="sec" style="padding:3px 9px;font-size:11px">remove</button>`
    : '';
  $('trainability').innerHTML = (t.ok
    ? `<span class="g">ready</span> <span class="dim">${t.reason}</span>`
    : `<span class="o">not ready</span> <span class="dim">${t.reason}</span>`) + dupNote;
  const ddBtn = document.getElementById('dedupe');
  if (ddBtn) ddBtn.addEventListener('click', async () => {
    const n = await dedupe(rows, EXTRACTOR[modality].name);
    log(`removed ${n} duplicate sample(s)`, 'o');
    await refreshSamples();
  });
  $<HTMLButtonElement>('trainBtn').disabled = !t.ok;

  $('grid').innerHTML = rows.slice(-40).map(r => {
    const inner = r.modality === 'image' && r.thumb
      ? `<img src="${r.thumb}" alt="">`
      : `<div class="aud">♪ ${r.durationMs ? (r.durationMs / 1000).toFixed(1) + 's' : 'clip'}</div>`;
    return `<div class="cell" data-id="${r.id}">${inner}<span>${r.label}</span>
      <button class="rm" data-id="${r.id}" title="delete">×</button></div>`;
  }).join('');

  $('grid').querySelectorAll<HTMLButtonElement>('.rm').forEach(b =>
    b.addEventListener('click', async e => {
      e.stopPropagation();
      await samples.remove(+b.dataset.id!);
      log('sample deleted', 'o');
      await refreshSamples();
    }));

  // clicking a chip fills the label box — faster than retyping
  $('labelChips').querySelectorAll<HTMLElement>('.chip').forEach(c =>
    c.addEventListener('click', () => {
      $<HTMLInputElement>('label').value = c.dataset.label!;
    }));
}

// ---------------------------------------------------------------- train
async function train() {
  const rows = await samples.list(project, modality);
  const counts = await samples.labels(project, modality);
  const t = trainability(counts);
  if (!t.ok) { log(t.reason, 'r'); return; }
  if (!t.reason.startsWith(`${t.labels.length}`)) log(t.reason, 'o');

  log(`featurising ${rows.length} samples…`);
  const labels = t.labels;
  const X: Mat = [], y: number[] = [], keys: string[] = [];
  for (const r of rows) {
    X.push(await featurise(r));
    y.push(labels.indexOf(r.label));
    keys.push(String(r.id));
    await new Promise(res => setTimeout(res, 0));      // keep the page responsive
  }

  // Hold out ~25% per label, stratified, so the score means something.
  const byLabel = new Map<number, number[]>();
  y.forEach((c, i) => { if (!byLabel.has(c)) byLabel.set(c, []); byLabel.get(c)!.push(i); });
  const teIdx = new Set<number>();
  for (const idxs of byLabel.values()) {
    const nTest = Math.max(1, Math.floor(idxs.length * 0.25));
    // take every k-th so the split is deterministic, not random per run
    const step = Math.max(1, Math.floor(idxs.length / nTest));
    for (let i = 0, taken = 0; i < idxs.length && taken < nTest; i += step, taken++) {
      teIdx.add(idxs[i]);
    }
  }
  const tr = [...X.keys()].filter(i => !teIdx.has(i));
  const te = [...teIdx];

  const Xtr = tr.map(i => X[i]), ytr = tr.map(i => y[i]);
  const Xte = te.map(i => X[i]), yte = te.map(i => y[i]);

  const pipe = new Pipeline({ epochs: 400 }).fit(Xtr, ytr, labels.length);
  const trainAcc = pipe.score(Xtr, ytr);
  const testAcc = Xte.length ? pipe.score(Xte, yte) : NaN;
  const ratio = Xte.length > 2 ? separationRatio(Xte, yte) : NaN;
  const cm = Xte.length
    ? confusionMatrix(yte, Xte.map(x => pipe.predict(x)), labels.length)
    : [];
  const checks = Xte.length >= labels.length * 2
    ? runChecks(Xtr, ytr, Xte, yte, tr.map(i => keys[i]), te.map(i => keys[i]), labels.length)
    : [];

  const rec = toRecord(pipe, {
    name: `${project}/${modality} · ${labels.join('|')}`,
    extractor: EXTRACTOR[modality].name,
    classes: labels,
    metrics: {
      train: trainAcc, test: Number.isNaN(testAcc) ? trainAcc : testAcc,
      ratio: Number.isNaN(ratio) ? 0 : ratio,
      nTrain: Xtr.length, nTest: Xte.length,
    },
  });
  if (hasIDB) await store.saveModel(rec);
  active = { rec, pipe };
  config = { ...defaultConfig(labels), ...config, targets: config.targets.filter(x => labels.includes(x)) };
  if (!config.targets.length) config.targets = [labels[0]];

  log(`trained ${labels.length} classes · train ${pct(trainAcc)} · ` +
      `test ${Number.isNaN(testAcc) ? 'n/a' : pct(testAcc)}`, 'g');
  renderMetrics(rec, cm, checks, labels);
  renderTargets(labels);
  // On a phone the metrics render far below the fold; without this the button
  // appears to do nothing.
  $('metrics').scrollIntoView({ behavior: 'smooth', block: 'start' });
  await refreshModels();
  syncControls();
}

function renderMetrics(rec: StoredModel, cm: number[][],
                       checks: { name: string; pass: boolean; detail: string }[],
                       labels: string[]) {
  const cmHtml = cm.length
    ? `<table class="cm"><tr><th>true ↓ pred →</th>${labels.map(l => `<th>${l}</th>`).join('')}</tr>` +
      cm.map((row, i) => `<tr><td>${labels[i]}</td>${row.map((v, j) =>
        `<td class="${v && i === j ? 'g' : v ? 'r' : 'dim'}">${v}</td>`).join('')}</tr>`).join('') +
      '</table>'
    : '<div class="dim" style="font-size:11.5px">too few samples for a held-out split — score is on training data</div>';
  $('metrics').innerHTML = `
    <div class="kpis">
      <div class="kpi"><div class="n ${rec.metrics.test > 0.8 ? 'g' : 'o'}">${pct(rec.metrics.test)}</div><div class="l">TEST</div></div>
      <div class="kpi"><div class="n dim">${pct(rec.metrics.train)}</div><div class="l">TRAIN</div></div>
      <div class="kpi"><div class="n b">${rec.dim}</div><div class="l">DIMS</div></div>
      <div class="kpi"><div class="n o">${rec.metrics.nTrain}/${rec.metrics.nTest}</div><div class="l">TRAIN/TEST</div></div>
    </div>${cmHtml}
    ${checks.length ? `<div class="checks">${checks.map(c =>
      `<div><span class="${c.pass ? 'g' : 'r'}">[${c.pass ? 'PASS' : 'FAIL'}]</span> ${c.name}` +
      `<span class="dim"> ${c.detail}</span></div>`).join('')}</div>` : ''}`;
}

function renderTargets(labels: string[]) {
  $('targets').innerHTML = labels.map(l =>
    `<label><input type="checkbox" class="tgt" value="${l}"` +
    `${config.targets.includes(l) ? ' checked' : ''}> ${l}</label>`).join('');
  $('targets').querySelectorAll('input').forEach(i =>
    i.addEventListener('change', syncConfig));
}

/** Silence gate in dBFS; below this a window is not classified. */
function gateDb(): number {
  const el = document.getElementById('gate') as HTMLInputElement | null;
  return el ? +el.value : -45;
}

// ---------------------------------------------------------------- config
function syncConfig() {
  config = {
    targets: [...document.querySelectorAll<HTMLInputElement>('.tgt:checked')].map(i => i.value),
    threshold: +$<HTMLInputElement>('thr').value / 100,
    onMatched: $<HTMLSelectElement>('sinkM').value as Sink,
    onNotMatched: $<HTMLSelectElement>('sinkN').value as Sink,
    webhookUrl: $<HTMLInputElement>('hookUrl').value.trim(),
    webhookMethod: $<HTMLSelectElement>('hookMethod').value as 'POST' | 'GET',
    cooldownMs: +$<HTMLInputElement>('cool').value,
  };
  $('thrLbl').textContent = `${$<HTMLInputElement>('thr').value}%`;
  localStorage.setItem('aidejavu.studio.cfg', JSON.stringify(config));
}

function restoreConfig() {
  try {
    const c = JSON.parse(localStorage.getItem('aidejavu.studio.cfg') || 'null');
    if (!c) return;
    $<HTMLInputElement>('thr').value = String(Math.round((c.threshold ?? 0.7) * 100));
    $<HTMLSelectElement>('sinkM').value = c.onMatched ?? 'console';
    $<HTMLSelectElement>('sinkN').value = c.onNotMatched ?? 'off';
    $<HTMLInputElement>('hookUrl').value = c.webhookUrl ?? '';
    $<HTMLSelectElement>('hookMethod').value = c.webhookMethod ?? 'POST';
    $<HTMLInputElement>('cool').value = String(c.cooldownMs ?? 0);
    config = { ...config, ...c };
  } catch (e) {
    // A malformed saved config must not stop the page loading, but silently
    // discarding it means the user's settings vanish with no clue why.
    quiet('studio.loadConfig', () => { throw e; }, null);
  }
}

// ---------------------------------------------------------------- inference
async function runInference(v: Vec, source: string, staleMs?: number) {
  if (!active) { log('train or load a model first', 'r'); return; }
  const { rec, pipe } = active;
  const res = classify(pipe, rec.classes, v);
  const out = await dispatch({
    predicted: res.predicted, confidence: res.confidence, probs: res.probs,
    classes: rec.classes, modelId: rec.id, modelName: rec.name,
  }, config, log);
  if (hasIDB) {
    await store.addEvent({
      modelId: rec.id, ts: Date.now(), predicted: res.predicted,
      confidence: res.confidence, matched: out.matched, probs: res.probs,
      delivered: out.delivered,
    });
  }
  renderVerdict($('verdict'), rec.classes, res, out, source,
                config.threshold, config.targets, staleMs);
  $('payload').textContent = JSON.stringify(out.payload, null, 2);
  await renderEvents();
}

async function renderEvents() {
  if (!hasIDB) return;
  const evs = await store.listEvents(25);
  $('events').innerHTML = evs.length
    ? evs.map(e => `<tr><td class="dim">${new Date(e.ts).toLocaleTimeString()}</td>` +
        `<td class="${e.matched ? 'g' : 'dim'}">${e.matched ? 'MATCH' : '—'}</td>` +
        `<td>${e.predicted}</td><td>${pct(e.confidence)}</td>` +
        `<td class="dim">${e.delivered}</td></tr>`).join('')
    : '<tr><td colspan="5" class="dim">no events yet — classify something in step 5</td></tr>';
}

async function refreshModels() {
  if (!hasIDB) return;
  const ms = await store.listModels();
  $<HTMLSelectElement>('modelSel').innerHTML = ms.map(m => {
    const bad = !compatibility(m).ok;
    return `<option value="${m.id}"${active?.rec.id === m.id ? ' selected' : ''}>` +
      `${bad ? '⚠ ' : ''}${m.name} · ${pct(m.metrics.test)}` +
      `${bad ? ' (shapes demo — not usable here)' : ''}</option>`;
  }).join('');
  const u = await store.usage();
  $('usage').textContent = `${u.models} model(s), ${u.events} event(s), ~${(u.bytes / 1024).toFixed(1)} KB`;
}

// ---------------------------------------------------------------- boot
export async function boot() {
  logEl = $('log');
  if (!hasIDB) log('IndexedDB unavailable — nothing will persist', 'r');
  restoreConfig();
  syncConfig();

  const renderExtractorChoices = () => {
    const list = EXTRACTORS_BY_MODALITY[modality];
    $<HTMLSelectElement>('extractor').innerHTML = list.map(e =>
      `<option value="${e.name}" title="Good for: ${e.task}"` +
      `${e.name === chosen[modality] ? ' selected' : ''}>${extractorLabel(e)}</option>`).join('');
    $<HTMLSelectElement>('extractor').disabled = list.length < 2;
  };
  $('extractor').addEventListener('change', async () => {
    chosen[modality] = $<HTMLSelectElement>('extractor').value;
    const e = currentExtractor();
    log(`extractor: ${e.title} (${DEVICE_TAG[e.device]}) — good for ${e.task}`);
    if ((e.name === 'dinov2' || e.name === 'dinov3') && !backboneReady(e.name)) {
      log(`first training run downloads ${e.cost.split('· ')[1]} (then cached); ` +
          `on a phone prefer MobileNetV2`, 'o');
    }
    if (e.name === 'scnet' && !scReady()) {
      log('first run downloads the speech-commands model (~2 MB, then cached)', 'o');
    }
    if (e.name === 'tmnet' && !tmReady()) {
      log("first run downloads Teachable Machine's MobileNetV2 (~1.6 MB, then cached)", 'o');
    }
    await refreshSamples();   // duplicate/vector state is per-extractor
  });
  renderExtractorChoices();

  $('modality').addEventListener('change', async () => {
    modality = $<HTMLSelectElement>('modality').value as Modality;
    renderExtractorChoices();
    await stopCapture();
    $('imgTools').style.display = modality === 'image' ? '' : 'none';
    $('audTools').style.display = modality === 'audio' ? '' : 'none';
    log(`modality: ${modality} (extractor ${EXTRACTOR[modality].name}, ${EXTRACTOR[modality].dim}-d)`);
    await refreshSamples();
  });

  $('project').addEventListener('change', async () => {
    project = $<HTMLInputElement>('project').value.trim() || 'default';
    await refreshSamples();
  });

  ['thr', 'sinkM', 'sinkN', 'hookUrl', 'hookMethod', 'cool'].forEach(id =>
    $(id).addEventListener('change', syncConfig));
  $('thr').addEventListener('input', syncConfig);

  // ---- image capture
  $('camOn').addEventListener('click', () => startCamera().catch(e => {
    log(`camera unavailable: ${deviceError(e)}`, 'r');
    syncControls();                      // was left enabled after a failure
  }));
  $('camOff').addEventListener('click', () => { stopCapture(); log('capture stopped'); });
  $('shoot').addEventListener('click', async () => {
    const v = $<HTMLVideoElement>('cam');
    if (!stream) { log('turn the camera on first', 'r'); return; }
    if (!$<HTMLInputElement>('label').value.trim()) {
      log('pick or type a label first', 'r'); return;
    }
    // readyState >= 2 (HAVE_CURRENT_DATA) means there is a decodable frame.
    // videoWidth alone can be non-zero before the first frame arrives, which
    // produced blank captures.
    if (!v.videoWidth || v.readyState < 2) {
      log('camera still starting — try again in a moment', 'o'); return;
    }
    try {
      // Un-mirror the front camera so the SAVED frame matches the real scene,
      // even though the preview is mirrored for the user's benefit.
      const surf = videoToSurface(v);
      const cv = surf as unknown as HTMLCanvasElement;
      const out = facing === 'user' ? unmirror(cv) : cv;
      const blob = await canvasToBlob(out);
      if (!blob || blob.size < 256) { log('captured an empty frame — try again', 'r'); return; }
      await addSample(blob, facing === 'user' ? 'camera-front' : 'camera',
                      $<HTMLInputElement>('label').value.trim());
    } catch (e: any) {
      log('capture failed: ' + (e?.message ?? e), 'r');
    }
  });

  $('flip').addEventListener('click', () => { flipCamera(); });
  $('imgFile').addEventListener('change', async e => {
    const files = (e.target as HTMLInputElement).files;
    if (!files) return;
    const label = $<HTMLInputElement>('label').value.trim();
    for (const f of Array.from(files)) await addSample(f, 'upload', label);
    (e.target as HTMLInputElement).value = '';
  });

  // ---- audio capture
  $('micOn').addEventListener('click', () => startMic().catch(e => {
    log(`microphone unavailable: ${deviceError(e)}`, 'r');
    syncControls();
  }));
  $('micOff').addEventListener('click', () => { stopCapture(); log('capture stopped'); });
  $('rec').addEventListener('click', async () => {
    const ms = +$<HTMLInputElement>('clipMs').value;
    const label = $<HTMLInputElement>('label').value.trim();
    // Device first: "pick a label" is confusing when the real blocker is no mic.
    if (!stream) { log('turn the microphone on first', 'r'); return; }
    if (!label) { log('pick or type a label first', 'r'); return; }
    log(`recording ${ms} ms…`);
    try {
      const blob = await recordClip(ms);
      await addSample(blob, 'mic', label, ms);
    } catch (e: any) { log('record failed: ' + (e?.message ?? e), 'r'); }
  });
  $('audFile').addEventListener('change', async e => {
    const files = (e.target as HTMLInputElement).files;
    if (!files) return;
    const label = $<HTMLInputElement>('label').value.trim();
    for (const f of Array.from(files)) await addSample(f, 'upload', label);
    (e.target as HTMLInputElement).value = '';
  });

  // ---- train / models
  $('trainBtn').addEventListener('click', async () => {
    const b = $<HTMLButtonElement>('trainBtn');
    b.disabled = true;
    try { await train(); } catch (e: any) { log('training failed: ' + (e?.message ?? e), 'r'); }
    b.disabled = false;
  });
  $('loadBtn').addEventListener('click', async () => {
    const id = $<HTMLSelectElement>('modelSel').value;
    const rec = id ? await store.getModel(id) : undefined;
    if (!rec) { log('no model selected', 'r'); return; }
    const compat = compatibility(rec);
    if (!compat.ok) {
      log(`cannot load: ${compat.reason}`, 'r');
      $<HTMLButtonElement>('runBtn').disabled = true;
      return;
    }
    // Align the UI with the model so "Classify now" uses the right capture widget.
    if (modality !== compat.reason) {
      modality = compat.reason as Modality;
      $<HTMLSelectElement>('modality').value = modality;
      $<HTMLSelectElement>('modality').dispatchEvent(new Event('change', { bubbles: true }));
      await new Promise(r => setTimeout(r, 0));
    }
    chosen[modality] = rec.extractor;
    $<HTMLSelectElement>('extractor').value = rec.extractor;
    active = { rec, pipe: fromRecord(rec) };
    config.targets = config.targets.filter(t => rec.classes.includes(t));
    if (!config.targets.length) config.targets = [rec.classes[0]];
    renderTargets(rec.classes);
    log(`loaded "${rec.name}" (${rec.extractor}, ${rec.dim}-d)`, 'g');
    syncControls();
  });
  $('exportBtn').addEventListener('click', async () => {
    const id = $<HTMLSelectElement>('modelSel').value;
    const rec = id ? await store.getModel(id) : active?.rec;
    if (!rec) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([toJSON(rec)], { type: 'application/json' }));
    a.download = `aidejavu-${rec.id}.json`;
    a.click();
  });
  $('exportJs').addEventListener('click', async () => {
    const id = $<HTMLSelectElement>('modelSel').value;
    const rec = id ? await store.getModel(id) : active?.rec;
    if (!rec) { log('no model selected', 'r'); return; }
    const srcMap: Record<string, string | undefined> =
      { photo: RUNTIME_PHOTO, mel: RUNTIME_MEL };
    const js = exportScript(rec, {
      extractorSource: srcMap[rec.extractor],
      name: `detector-${rec.extractor}`,
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([js], { type: 'text/javascript' }));
    a.download = `detector-${rec.extractor}-${rec.id}.js`;
    a.click();
    const selfContained = rec.extractor === 'photo' || rec.extractor === 'mel';
    log(`exported ${(js.length / 1024).toFixed(1)} KB script` +
        (selfContained ? ' — self-contained, runs in browser/node/bun'
                       : ` — needs the ${rec.extractor} backbone at runtime`), 'g');
  });

  $('clearSamples').addEventListener('click', async () => {
    const counts = await samples.labels(project);
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (total > 0 && !confirm(
        `Delete all ${total} sample(s) in "${project}"? This cannot be undone.`)) {
      return;
    }
    const n = await samples.clearProject(project);
    log(`cleared ${n} sample(s) from "${project}"`, 'o');
    await refreshSamples();
  });
  $('clearEv').addEventListener('click', async () => {
    await store.clearEvents(); await renderEvents(); await refreshModels();
  });

  // ---- run inference on a NEW input
  $('runBtn').addEventListener('click', async () => {
    if (!active) return;
    try {
      if (RUNNABLE[active.rec.extractor] === 'image') {
        const v = $<HTMLVideoElement>('cam');
        if (stream && v.videoWidth) {
          await runInference(await embedFor(active.rec.extractor, videoToSurface(v)), 'camera');
        } else { log('start the camera, or use "test a file" below', 'o'); }
      } else {
        if (!stream) { log('start the microphone first', 'o'); return; }
        const ms = +$<HTMLInputElement>('clipMs').value;
        log(`listening ${ms} ms…`);
        const blob = await recordClip(ms);
        const { samples: pcm, sampleRate } = await decodeAudioFile(blob);
        await runInference(await embedAudioFor(active.rec.extractor, pcm, sampleRate), 'mic');
      }
    } catch (e: any) { log('inference failed: ' + (e?.message ?? e), 'r'); }
  });

  $('testFile').addEventListener('change', async e => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f || !active) return;
    try {
      if (RUNNABLE[active.rec.extractor] === 'image') {
        await runInference(await embedFor(active.rec.extractor, await fileToSurface(f)), 'file');
      } else {
        const { samples: pcm, sampleRate } = await decodeAudioFile(f);
        await runInference(await embedAudioFor(active.rec.extractor, pcm, sampleRate), 'file');
      }
    } catch (err: any) { log('could not read file: ' + (err?.message ?? err), 'r'); }
    (e.target as HTMLInputElement).value = '';
  });

  // ---- continuous mode: classify every N ms and let the actions fire
  let loopStop: (() => void) | null = null;
  $('loop').addEventListener('click', async () => {
    const btn = $<HTMLButtonElement>('loop');
    if (loopStop) {
      loopStop(); loopStop = null;
      btn.textContent = 'Start continuous'; btn.classList.remove('on');
      log('continuous mode off');
      return;
    }
    if (!active) { log('train or load a model first', 'r'); return; }
    const isAudio = RUNNABLE[active.rec.extractor] === 'audio';
    if (active.rec.extractor === 'dinov2' || active.rec.extractor === 'dinov3') {
      log(`note: ${active.rec.extractor} takes ~2 s per frame — raise the interval`, 'o');
    }
    if (!stream) {
      log(isAudio ? 'start the microphone first' : 'start the camera first', 'o');
      return;
    }
    btn.textContent = 'Stop continuous'; btn.classList.add('on');
    log('continuous mode on — actions will fire on every frame that matches', 'g');

    // Self-scheduling, NOT setInterval: an async callback on a fixed interval
    // queues another run before the previous one finishes, so with a slow
    // extractor the backlog grows without bound and every result describes a
    // frame further in the past. Waiting for each pass, then sleeping the
    // remainder of the interval, keeps the newest frame the one being judged.
    let stop = false;
    const period = Math.max(300, +$<HTMLInputElement>('loopMs').value);

    // Audio taps the live stream ONCE into a rolling buffer, so consecutive
    // windows can overlap and a sound on a boundary is never lost between clips.
    let listener: Listener | null = null;
    if (isAudio) {
      const windowMs = Math.max(400, +$<HTMLInputElement>('clipMs').value);
      listener = startListening(stream, windowMs);
      log(`listening continuously (${windowMs} ms window, gate ${gateDb()} dB)`, 'g');
    }
    loopStop = () => { stop = true; listener?.stop(); listener = null; };

    (async () => {
      while (!stop) {
        const started = performance.now();
        try {
          if (isAudio && listener) {
            const pcm = listener.snapshot();
            if (pcm.length < 1024) {
              // buffer still filling on the first pass
            } else if (isSilent(pcm, gateDb())) {
              // Classifying silence would force it into the nearest label, so
              // report it and skip inference entirely.
              $('verdict').innerHTML =
                `<span class="dim">listening… ${listener.level().toFixed(0)} dB ` +
                `(below the ${gateDb()} dB gate)</span>`;
            } else {
              const grabbed = Date.now();
              const vec = await embedAudioFor(active!.rec.extractor, pcm, listener.sampleRate);
              await runInference(vec, 'live-audio', Date.now() - grabbed);
            }
          } else {
            const v = $<HTMLVideoElement>('cam');
            if (v.videoWidth && v.readyState >= 2) {
              const grabbed = Date.now();
              const vec = await embedFor(active!.rec.extractor, videoToSurface(v));
              // Staleness = how old the input was by the time we had an answer.
              // Surfaced so a slow extractor is visible rather than silently
              // reporting the past as the present.
              await runInference(vec, 'live', Date.now() - grabbed);
            }
          }
        } catch (e: any) {
          log('live inference failed: ' + (e?.message ?? e), 'r');
          stop = true;
        }
        const elapsed = performance.now() - started;
        await new Promise(r => setTimeout(r, Math.max(0, period - elapsed)));
      }
      listener?.stop();
    })();
  });

  $('selftest').addEventListener('click', () => {
    log('running label-mapping self test…');
    const { lines, verdict } = runSelfTest();
    for (const l of lines) {
      if (l.name.startsWith('—')) { log(l.name + ' ' + l.detail, 'b'); continue; }
      log(`  [${l.pass ? 'PASS' : 'FAIL'}] ${l.name} — ${l.detail}`, l.pass ? 'g' : 'r');
    }
    log(verdict, lines.every(l => l.pass) ? 'o' : 'r');
  });

  $('label').addEventListener('input', syncControls);
  $('modelSel').addEventListener('change', syncControls);

  await refreshSamples();
  await refreshModels();
  await renderEvents();
  syncControls();
  log('studio ready — pick a modality, add samples, train', 'g');
}
