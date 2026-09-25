/**
 * In-page self test for the "labels come back swapped" report.
 *
 * Reported symptom: saying "tu" matches "la" and vice versa — a clean inversion
 * in BOTH image and audio mode. A weak model produces noise, not a consistent
 * swap, so the cause is a mapping error somewhere, not accuracy.
 *
 * Every layer was verified correct in isolation (numerics, label->index, the
 * stratified split, save/load round trip, the capture path). So this runs the
 * SAME code against synthetic data with known-correct answers, on the user's
 * device, and reports which layer inverts — if any. If all layers pass, the
 * swap is in the recorded data rather than the code, and it says so.
 */
import { Pipeline, confusionMatrix, mulberry32, type Mat } from './linalg.js';
import { classify, toRecord, fromRecord } from './serialize.js';
import { embedAudio } from './audio.js';
import { embedPhoto } from './photo.js';
import type { Surface } from './dataset.js';

export interface Line { name: string; pass: boolean; detail: string }

const SR = 16000;
function tone(freq: number, sec = 1, amp = 0.3): Float32Array {
  const n = Math.floor(SR * sec);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR);
  return x;
}
function patch(kind: 'left' | 'right', seed: number): Surface {
  const S = 96;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const cx = cv.getContext('2d')!;
  const rng = mulberry32(seed);
  const id = cx.createImageData(S, S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = (y * S + x) * 4;
      let r = 110, g = 110, b = 110;
      if (kind === 'left' ? x < S / 2 : x >= S / 2) { r = 215; g = 65; b = 50; }
      const n = (rng() - 0.5) * 22;
      id.data[i] = Math.max(0, Math.min(255, r + n));
      id.data[i + 1] = Math.max(0, Math.min(255, g + n));
      id.data[i + 2] = Math.max(0, Math.min(255, b + n));
      id.data[i + 3] = 255;
    }
  }
  cx.putImageData(id, 0, 0);
  return cv as unknown as Surface;
}

/** Train on two known-distinct groups and check the names come back right. */
function roundTrip(
  labelA: string, labelB: string, aVecs: Mat, bVecs: Mat, probeA: Float64Array,
  probeB: Float64Array,
): Line[] {
  const out: Line[] = [];
  // labels are sorted, exactly as trainability() produces them
  const labels = [labelA, labelB].sort();
  const X: Mat = [], y: number[] = [];
  aVecs.forEach(v => { X.push(v); y.push(labels.indexOf(labelA)); });
  bVecs.forEach(v => { X.push(v); y.push(labels.indexOf(labelB)); });

  const pipe = new Pipeline({ epochs: 400 }).fit(X, y, labels.length);
  const rA = classify(pipe, labels, probeA);
  const rB = classify(pipe, labels, probeB);
  out.push({
    name: `"${labelA}" input predicts "${labelA}"`,
    pass: rA.predicted === labelA,
    detail: `got "${rA.predicted}" ${(rA.confidence * 100).toFixed(0)}%`,
  });
  out.push({
    name: `"${labelB}" input predicts "${labelB}"`,
    pass: rB.predicted === labelB,
    detail: `got "${rB.predicted}" ${(rB.confidence * 100).toFixed(0)}%`,
  });
  const inverted = rA.predicted === labelB && rB.predicted === labelA;
  out.push({
    name: 'not inverted',
    pass: !inverted,
    detail: inverted ? 'BOTH swapped — this is the reported bug' : 'mapping is correct',
  });

  // and after a save/load cycle, which is what "load model" does
  const rec = toRecord(pipe, {
    name: 'selftest', extractor: 'selftest', classes: labels,
    metrics: { train: 1, test: 1, ratio: 0, nTrain: X.length, nTest: 0 },
  });
  const back = fromRecord(rec);
  const sA = classify(back, rec.classes, probeA);
  out.push({
    name: 'survives save + load',
    pass: sA.predicted === labelA,
    detail: `got "${sA.predicted}"`,
  });
  const cm = confusionMatrix(y, X.map(v => pipe.predict(v)), labels.length);
  out.push({
    name: 'diagonal confusion on training data',
    pass: cm.every((row, i) => row[i] === Math.max(...row)),
    detail: JSON.stringify(cm),
  });
  return out;
}

export function runSelfTest(): { lines: Line[]; verdict: string } {
  const lines: Line[] = [];

  // ---- AUDIO: 300 Hz labelled "la", 1500 Hz labelled "tu"
  const la: Mat = [], tu: Mat = [];
  for (let i = 0; i < 6; i++) {
    la.push(embedAudio(tone(300 + i * 7), SR));
    tu.push(embedAudio(tone(1500 + i * 7), SR));
  }
  lines.push({ name: '— audio —', pass: true, detail: '300 Hz = "la", 1500 Hz = "tu"' });
  lines.push(...roundTrip('la', 'tu', la, tu,
    embedAudio(tone(305), SR), embedAudio(tone(1510), SR)));

  // ---- IMAGE: red-left labelled "left", red-right labelled "right"
  const L: Mat = [], R: Mat = [];
  for (let i = 0; i < 6; i++) {
    L.push(embedPhoto(patch('left', 100 + i)));
    R.push(embedPhoto(patch('right', 200 + i)));
  }
  lines.push({ name: '— image —', pass: true, detail: 'red-left = "left", red-right = "right"' });
  lines.push(...roundTrip('left', 'right', L, R,
    embedPhoto(patch('left', 999)), embedPhoto(patch('right', 888))));

  const bad = lines.filter(l => !l.pass);
  const verdict = bad.length === 0
    ? 'All layers map labels correctly on this device. If your own model still ' +
      'swaps, the labels attached to the RECORDED samples are crossed — check the ' +
      'sample grid: each thumbnail shows the label it was saved with.'
    : `${bad.length} layer(s) inverted: ${bad.map(b => b.name).join('; ')}`;
  return { lines, verdict };
}
