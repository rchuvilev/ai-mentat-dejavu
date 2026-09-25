/**
 * Hand-designed frozen feature extractors — ported from features.py / features2.py.
 *
 * These need no model download and run in ~13 ms, so they are the fast path and
 * the reference the pretrained backbones are measured against.
 *
 * v1 (absolute orientation, 224-d) scored 43.3% and is kept deliberately: it is
 * the negative control. Rotating a square moved its v1 embedding further than
 * changing the shape did, which is why it could never work.
 * v2 (rotation-invariant, 26-d) scored 95.0%.
 */
import type { Surface } from './dataset.js';

const SIZE = 96;

function toGrayAndRGB(s: Surface): { gray: Float64Array; rgb: Float64Array } {
  const src = s.getContext('2d').getImageData(0, 0, s.width, s.height);
  const W = s.width, H = s.height;
  const gray = new Float64Array(SIZE * SIZE);
  const rgb = new Float64Array(SIZE * SIZE * 3);
  // nearest-neighbour resize to SIZE x SIZE (bilinear is not needed: the shapes
  // are large flat regions, and both extractors pool over tiles/rings anyway)
  for (let y = 0; y < SIZE; y++) {
    const sy = Math.min(H - 1, Math.floor((y * H) / SIZE));
    for (let x = 0; x < SIZE; x++) {
      const sx = Math.min(W - 1, Math.floor((x * W) / SIZE));
      const si = (sy * W + sx) * 4;
      const r = src.data[si] / 255, g = src.data[si + 1] / 255, b = src.data[si + 2] / 255;
      const di = y * SIZE + x;
      gray[di] = 0.299 * r + 0.587 * g + 0.114 * b;
      rgb[di * 3] = r; rgb[di * 3 + 1] = g; rgb[di * 3 + 2] = b;
    }
  }
  return { gray, rgb };
}

/** Central-difference gradient, matching numpy.gradient. */
function gradient(gray: Float64Array): { gx: Float64Array; gy: Float64Array } {
  const gx = new Float64Array(SIZE * SIZE);
  const gy = new Float64Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x;
      const xm = x > 0 ? gray[i - 1] : gray[i];
      const xp = x < SIZE - 1 ? gray[i + 1] : gray[i];
      const ym = y > 0 ? gray[i - SIZE] : gray[i];
      const yp = y < SIZE - 1 ? gray[i + SIZE] : gray[i];
      gx[i] = (xp - xm) / (x > 0 && x < SIZE - 1 ? 2 : 1);
      gy[i] = (yp - ym) / (y > 0 && y < SIZE - 1 ? 2 : 1);
    }
  }
  return { gx, gy };
}

// ---------------------------------------------------------------- v1
const TILE = 4, ORIENT = 8;
export const V1_DIM = TILE * TILE * ORIENT + TILE * TILE * 6;

export function embedV1(s: Surface): Float64Array {
  const { gray, rgb } = toGrayAndRGB(s);
  const { gx, gy } = gradient(gray);
  const out = new Float64Array(V1_DIM);
  const step = SIZE / TILE;
  let o = 0;

  // histogram of oriented gradients per tile (ABSOLUTE orientation -> the flaw)
  for (let ty = 0; ty < TILE; ty++) {
    for (let tx = 0; tx < TILE; tx++) {
      const h = new Float64Array(ORIENT);
      for (let y = ty * step; y < (ty + 1) * step; y++) {
        for (let x = tx * step; x < (tx + 1) * step; x++) {
          const i = y * SIZE + x;
          const mag = Math.hypot(gx[i], gy[i]);
          let ang = Math.atan2(gy[i], gx[i]) % Math.PI;
          if (ang < 0) ang += Math.PI;
          const b = Math.min(ORIENT - 1, Math.floor((ang / Math.PI) * ORIENT));
          h[b] += mag;
        }
      }
      let sum = 0;
      for (const v of h) sum += v;
      for (let b = 0; b < ORIENT; b++) out[o++] = h[b] / (sum + 1e-8);
    }
  }

  // per-tile colour mean + std
  for (let ty = 0; ty < TILE; ty++) {
    for (let tx = 0; tx < TILE; tx++) {
      const m = [0, 0, 0], v = [0, 0, 0];
      let n = 0;
      for (let y = ty * step; y < (ty + 1) * step; y++) {
        for (let x = tx * step; x < (tx + 1) * step; x++) {
          const i = (y * SIZE + x) * 3;
          for (let c = 0; c < 3; c++) m[c] += rgb[i + c];
          n++;
        }
      }
      for (let c = 0; c < 3; c++) m[c] /= n;
      for (let y = ty * step; y < (ty + 1) * step; y++) {
        for (let x = tx * step; x < (tx + 1) * step; x++) {
          const i = (y * SIZE + x) * 3;
          for (let c = 0; c < 3; c++) v[c] += (rgb[i + c] - m[c]) ** 2;
        }
      }
      for (let c = 0; c < 3; c++) out[o++] = m[c];
      for (let c = 0; c < 3; c++) out[o++] = Math.sqrt(v[c] / n);
    }
  }
  return out;
}

// ---------------------------------------------------------------- v2
const RINGS = 8, REL_BINS = 12;
export const V2_DIM = RINGS + REL_BINS + 6;

export function embedV2(s: Surface): Float64Array {
  const { gray } = toGrayAndRGB(s);
  const { gx, gy } = gradient(gray);

  // foreground = brighter than the midpoint of the range (the shape is bright)
  let lo = Infinity, hi = -Infinity;
  for (const v of gray) { if (v < lo) lo = v; if (v > hi) hi = v; }
  let thr = lo + 0.5 * (hi - lo);
  let mask = new Uint8Array(SIZE * SIZE);
  let cnt = 0;
  for (let i = 0; i < gray.length; i++) if (gray[i] > thr) { mask[i] = 1; cnt++; }
  if (cnt < 8) {                                  // degenerate: fall back to mean
    let m = 0; for (const v of gray) m += v; m /= gray.length;
    mask = new Uint8Array(SIZE * SIZE); cnt = 0;
    for (let i = 0; i < gray.length; i++) if (gray[i] > m) { mask[i] = 1; cnt++; }
  }

  let cx = 0, cy = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) if (mask[y * SIZE + x]) { cx += x; cy += y; }
  }
  cx /= cnt; cy /= cnt;

  // radial distances, normalised by the shape's own extent -> scale invariant
  const rad = new Float64Array(SIZE * SIZE);
  const angC = new Float64Array(SIZE * SIZE);
  let rmax = 1e-8;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x;
      const dy = y - cy, dx = x - cx;
      rad[i] = Math.hypot(dx, dy);
      angC[i] = Math.atan2(dy, dx);
      if (mask[i] && rad[i] > rmax) rmax = rad[i];
    }
  }

  const out = new Float64Array(V2_DIM);
  let o = 0;

  // 1. radial occupancy profile — rotation cannot change this
  for (let k = 0; k < RINGS; k++) {
    const a = k / RINGS, b = (k + 1) / RINGS;
    let tot = 0, fg = 0;
    for (let i = 0; i < mask.length; i++) {
      const rn = rad[i] / rmax;
      if (rn >= a && rn < b) { tot++; if (mask[i]) fg++; }
    }
    out[o++] = tot ? fg / tot : 0;
  }

  // 2. gradient orientation RELATIVE to the centroid direction -> invariant
  const h = new Float64Array(REL_BINS);
  let hsum = 0;
  for (let i = 0; i < mask.length; i++) {
    const mag = Math.hypot(gx[i], gy[i]);
    let rel = (Math.atan2(gy[i], gx[i]) - angC[i]) % Math.PI;
    if (rel < 0) rel += Math.PI;
    const b = Math.min(REL_BINS - 1, Math.floor((rel / Math.PI) * REL_BINS));
    h[b] += mag; hsum += mag;
  }
  for (let b = 0; b < REL_BINS; b++) out[o++] = h[b] / (hsum + 1e-8);

  // 3. pose-free scalar descriptors
  let per = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x;
      if (!mask[i]) continue;
      if (x === 0 || x === SIZE - 1 || y === 0 || y === SIZE - 1 ||
          !mask[i - 1] || !mask[i + 1] || !mask[i - SIZE] || !mask[i + SIZE]) per++;
    }
  }
  per += 1e-8;
  const rFg: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) rFg.push(rad[i]);
  rFg.sort((a, b) => a - b);
  const pct = (p: number) => rFg[Math.min(rFg.length - 1, Math.floor(p * rFg.length))];
  const rMean = rFg.reduce((a, b) => a + b, 0) / rFg.length;
  const rStd = Math.sqrt(rFg.reduce((a, b) => a + (b - rMean) ** 2, 0) / rFg.length);

  out[o++] = (4 * Math.PI * cnt) / (per * per);      // compactness: 1.0 = circle
  out[o++] = rMean / rmax;
  out[o++] = rStd / rmax;
  out[o++] = cnt / (Math.PI * rmax * rmax);          // fill vs enclosing circle
  out[o++] = pct(0.9) / rmax;
  out[o++] = pct(0.5) / rmax;
  return out;
}

export const EXTRACTORS = {
  v1: { fn: embedV1, dim: V1_DIM, label: 'v1 absolute orientation' },
  v2: { fn: embedV2, dim: V2_DIM, label: 'v2 rotation-invariant' },
} as const;
export type ExtractorName = keyof typeof EXTRACTORS;
