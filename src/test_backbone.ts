#!/usr/bin/env node
/**
 * Static checks on the backbone integration.
 *
 * The backbone itself needs a browser (ONNX runs via WASM there; the native
 * onnxruntime-node is glibc-linked and fails on musl), so its accuracy is
 * verified live in the browser. What CAN be checked headlessly is that the
 * integration is wired correctly and cannot silently degrade:
 *   - the registry offers dinov2 for images and not for audio
 *   - a model records WHICH extractor trained it, so inference cannot mismatch
 *   - the loader asserts the export gives features, not logits
 */
import { readFileSync } from 'node:fs';
import { Pipeline, mulberry32, type Mat } from './linalg.js';
import { toRecord, fromRecord } from './serialize.js';

let failed = 0;
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failed++;
};

const src = (f: string) => readFileSync(new URL(`../src/${f}`, import.meta.url), 'utf8');

// ---------------------------------------------------------------- wiring
{
  const studio = src('studio.ts');
  check('registry offers dinov2 for images',
        /image:\s*\[[\s\S]{0,1400}?dinov2/.test(studio),
        'entries are multi-line now, so the window must be wider');
  check('audio does not offer dinov2',
        !/audio:\s*\[[\s\S]{0,200}?dinov2/.test(studio));
  check('inference dispatches on the model\'s own extractor',
        studio.includes("active.rec.extractor === 'dinov2'"),
        'a photo-trained model must not be run through dinov2 or vice versa');
  check('vectors cached per extractor name',
        studio.includes('s.vecs?.[ex]') && studio.includes('[ex]:'),
        'switching extractor must recompute, not reuse the wrong features');
}

// ---------------------------------------------------------------- loader guard
{
  const bb = src('backbone.ts');
  check('loader rejects a logits-only export',
        bb.includes('last_hidden_state') && bb.includes('throw new Error'),
        'classifier-head exports measured 1.40 separation vs 2.26 for features');
  check('concurrent loads share one promise',
        bb.includes('const inFlight = loading[id]') && bb.includes('if (inFlight) return inFlight'),
        'keyed per backbone now: dinov2 and dinov3 load independently');
  check('failed load does not poison retries',
        /loading\[id\]!\.catch/.test(bb));
  check('uses CDN so the deploy stays small',
        bb.includes('cdn.jsdelivr.net') && bb.includes('allowRemoteModels = true'));
}

// ---------------------------------------------------------------- round trip
{
  // A 384-d head must serialise and predict identically — same as the 116-d one.
  const rng = mulberry32(11);
  const X: Mat = [], y: number[] = [];
  for (let i = 0; i < 30; i++) {
    const c = i % 3;
    const v = new Float64Array(384);
    for (let j = 0; j < 384; j++) v[j] = rng() + (j % 3 === c ? 1.5 : 0);
    X.push(v); y.push(c);
  }
  const pipe = new Pipeline({ epochs: 150 }).fit(X, y, 3);
  const rec = toRecord(pipe, {
    name: 'dinov2 test', extractor: 'dinov2', classes: ['a', 'b', 'c'],
    metrics: { train: 1, test: 1, ratio: 2, nTrain: 30, nTest: 0 },
  });
  check('384-d model records its extractor', rec.extractor === 'dinov2');
  check('384-d model round-trips', X.every(x =>
    fromRecord(rec).predict(x) === pipe.predict(x)));
  const bytes = (rec.W.length + rec.b.length + rec.mu.length + rec.sd.length) * 8;
  check('384-d head still small', bytes < 30_000, `${(bytes / 1024).toFixed(1)} KB`);
}

// ---------------------------------------------------------------- TM backbone
{
  const tm = src('tmnet.ts');
  const studio = src('studio.ts');
  check('TM MobileNetV2 is offered for images',
        /image:\s*\[[\s\S]{0,500}?tmnet/.test(studio));
  check('tmnet is runnable and routed as image',
        /RUNNABLE[\s\S]{0,120}tmnet: 'image'/.test(studio));
  check('loader rejects a classifier head',
        tm.includes('looks like a classifier head') && tm.includes('TM_DIM'),
        'a logits export measured 1.40 separation vs 2.26 for real features');
  check('prefers WebGL, falls back to CPU',
        tm.includes("tf.setBackend('webgl')") && tm.includes("tf.setBackend('cpu')"),
        'measured 70 ms GPU vs 1188 ms CPU');
  check('frees GPU tensors',
        tm.includes('tf.tidy') && tm.includes('t.dispose()'),
        'WebGL textures leak in continuous mode without this');
  check('concurrent loads share one promise',
        tm.includes('if (loading) return loading'));
  check('a failed load does not poison retries',
        tm.includes('loading.catch'));
  check('resizes to the 224 input the model expects',
        tm.includes('INPUT = 224') && tm.includes('drawImage'));
  check('inputs normalised to [-1,1]',
        tm.includes('127.5') && tm.includes('tf.sub'));
  check('weights come from Google\'s public bucket',
        tm.includes('storage.googleapis.com/teachable-machine-models'),
        'Apache-2.0, CORS verified for this origin');
}

// ---------------------------------------------------------------- catalogue
{
  const studio = src('studio.ts');
  check('DINOv3 is offered', /name: 'dinov3'/.test(studio));
  check('one loader serves both dino generations',
        /dinov2' \|\| extractor === 'dinov3'/.test(studio) &&
        src('backbone.ts').includes('MODELS'),
        'same 384-d shape and 224 input — no second code path');
  check('per-backbone load state',
        /loaded: Record<string/.test(src('backbone.ts')) &&
        src('backbone.ts').includes('backboneReady(id'),
        'a shared flag would report dinov3 ready after loading dinov2');
  // They may be NAMED in the comment that records why they were dropped; what
  // must not exist is a registry entry or a dispatch branch.
  check('logits-only models are gone',
        !/name: '(mobilenetv4|resnet18)'/.test(studio) &&
        !/extractor === '(mobilenetv4|resnet18)'/.test(studio),
        'measured separation 1.40 / 1.53 and 51.7% accuracy');
  // device + task labelling
  check('every extractor states a device tier',
        (studio.match(/device: '(any|gpu|desktop)'/g) ?? []).length >= 5);
  check('every extractor states what it is good for',
        (studio.match(/task: '/g) ?? []).length >= 5);
  check('dropdown shows device and cost',
        studio.includes('DEVICE_TAG[e.device]') && studio.includes('e.cost'),
        'the numbers alone mislead without knowing where it runs');
  check('task guidance reaches the user as a tooltip',
        /title="Good for: \$\{e\.task\}"/.test(studio));
  check('phone users are steered away from the heavy backbones',
        studio.includes('on a phone prefer MobileNetV2'));
}

console.log(failed ? `\n${failed} FAILURE(S)` : '\nALL PASS');
process.exit(failed ? 1 : 0);
