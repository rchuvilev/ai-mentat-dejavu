/**
 * Synthetic shape dataset — the "point a webcam at 3 things" stand-in.
 *
 * Ported from make_data.py. Two properties are load-bearing and are asserted by
 * the test suite rather than assumed:
 *   1. colour is random, so it cannot identify the class
 *   2. filled AREA is equalised across shapes, so mean brightness cannot either
 *      (circle pi*r^2 vs triangle ~1.3*r^2 -- an earlier version leaked the
 *      label this way and the property test caught it)
 */
import { mulberry32 } from './linalg.js';

export const CLASSES = ['circle', 'square', 'triangle'] as const;
export type ClassName = typeof CLASSES[number];

/** Minimal 2D surface — implemented by the browser canvas and the Node shim. */
export interface Surface {
  width: number;
  height: number;
  getContext(t: '2d'): SurfaceCtx;
}
export interface SurfaceCtx {
  fillStyle: string;
  createImageData(w: number, h: number): { width: number; height: number; data: Uint8ClampedArray };
  putImageData(d: { data: Uint8ClampedArray; width: number; height: number }, x: number, y: number): void;
  getImageData(x: number, y: number, w: number, h: number): { data: Uint8ClampedArray; width: number; height: number };
  fillRect(x: number, y: number, w: number, h: number): void;
  save(): void;
  restore(): void;
  translate(x: number, y: number): void;
  rotate(r: number): void;
  beginPath(): void;
  closePath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  rect(x: number, y: number, w: number, h: number): void;
  arc(cx: number, cy: number, r: number, s: number, e: number): void;
  fill(): void;
}

export type MakeSurface = (w: number, h: number) => Surface;

export interface Item {
  key: string;
  split: 'train' | 'test';
  label: number;
  surface: Surface;
}

export function drawShape(
  kind: ClassName, rng: () => number, mk: MakeSurface, S = 128,
): Surface {
  const cv = mk(S, S);
  const cx = cv.getContext('2d');

  // noisy background
  const base = [30 + rng() * 60, 30 + rng() * 60, 30 + rng() * 60];
  const id = cx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    for (let c = 0; c < 3; c++) {
      id.data[i * 4 + c] = Math.max(0, Math.min(255, base[c] + (rng() * 50 - 25)));
    }
    id.data[i * 4 + 3] = 255;
  }
  cx.putImageData(id, 0, 0);

  const col = `rgb(${(90 + rng() * 165) | 0},${(90 + rng() * 165) | 0},${(90 + rng() * 165) | 0})`;

  // Equal-area sizing. Reference radius stays small enough that the equal-area
  // TRIANGLE (largest circumradius of the three) still fits with margin.
  const area = Math.pow(20 + rng() * 10, 2) * Math.PI;
  const r = Math.sqrt(area / Math.PI);
  const s = Math.sqrt(area) / 2;
  const t = Math.sqrt((area * 4) / (3 * Math.sqrt(3)));
  const bound = Math.ceil(Math.max(r, s * Math.SQRT2, t));
  const px = bound + 4 + rng() * (S - 2 * (bound + 4));
  const py = bound + 4 + rng() * (S - 2 * (bound + 4));
  const rot = rng() * Math.PI * 2;

  cx.fillStyle = col;
  cx.save();
  cx.translate(px, py);
  cx.rotate(rot);
  cx.beginPath();
  if (kind === 'circle') {
    cx.arc(0, 0, r, 0, Math.PI * 2);
  } else if (kind === 'square') {
    cx.rect(-s, -s, 2 * s, 2 * s);
  } else {
    for (let i = 0; i < 3; i++) {
      const a = (i * 2 * Math.PI) / 3;
      const X = t * Math.cos(a), Y = t * Math.sin(a);
      if (i) cx.lineTo(X, Y); else cx.moveTo(X, Y);
    }
    cx.closePath();
  }
  cx.fill();
  cx.restore();
  return cv;
}

/**
 * Deterministic dataset. Keys are `split/class/index` so a cached embedding can
 * be matched back to its image across processes and turns.
 */
export function makeDataset(
  mk: MakeSurface, nTrain = 30, nTest = 12, seed = 1234,
): Item[] {
  const rng = mulberry32(seed);
  const out: Item[] = [];
  for (const [split, n] of [['train', nTrain], ['test', nTest]] as const) {
    for (let li = 0; li < CLASSES.length; li++) {
      for (let i = 0; i < n; i++) {
        out.push({
          key: `${split}/${CLASSES[li]}/${i}`,
          split,
          label: li,
          surface: drawShape(CLASSES[li], rng, mk),
        });
      }
    }
  }
  return out;
}

export function pixels(s: Surface): Uint8ClampedArray {
  return s.getContext('2d').getImageData(0, 0, s.width, s.height).data;
}
