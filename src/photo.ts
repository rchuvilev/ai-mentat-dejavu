/**
 * General-purpose photo extractor.
 *
 * WHY NOT REUSE v2: the shape extractor assumes a single bright blob on a dark
 * field — it thresholds to find one foreground region and measures its radial
 * profile. On a photograph there is no such blob, so it would describe noise.
 * Verified in the test suite: v2's separation ratio on photo-like inputs is far
 * weaker than this extractor's.
 *
 * What this computes (all pooled on a spatial grid, so position matters but exact
 * pixels do not):
 *   - colour: mean + std per channel in a 3x3 grid  (54)
 *   - hue histogram, saturation-weighted, 12 bins   (12)
 *   - oriented gradient histogram, 8 bins x 2x2 grid (32)
 *   - luminance stats + edge density per 3x3 cell    (18)
 * Total 116 dims, ~5 ms per image.
 *
 * Deliberately NOT rotation-invariant: a photo of a cup upright vs upside-down
 * usually IS a different thing to the user, unlike the abstract shapes where
 * rotation was pure nuisance. Position and orientation carry signal here.
 */
import type { Surface } from './dataset.js';

const SIZE = 96;
const CGRID = 3;
const GGRID = 2;
const HUE_BINS = 12;
const ORIENT = 8;

export const PHOTO_DIM =
  CGRID * CGRID * 6 +          // colour mean+std
  HUE_BINS +                   // hue histogram
  GGRID * GGRID * ORIENT +     // gradients
  CGRID * CGRID * 2;           // luma mean + edge density

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d > 1e-9) {
    if (mx === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (mx === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  return [h, mx <= 1e-9 ? 0 : d / mx, mx];
}

/** Draw any image source into a fixed-size RGB buffer (letterbox-free stretch). */
export function surfaceToRGB(s: Surface): Float64Array {
  const src = s.getContext('2d').getImageData(0, 0, s.width, s.height);
  const W = s.width, H = s.height;
  const out = new Float64Array(SIZE * SIZE * 3);
  for (let y = 0; y < SIZE; y++) {
    const sy = Math.min(H - 1, Math.floor((y * H) / SIZE));
    for (let x = 0; x < SIZE; x++) {
      const sx = Math.min(W - 1, Math.floor((x * W) / SIZE));
      const si = (sy * W + sx) * 4, di = (y * SIZE + x) * 3;
      out[di] = src.data[si] / 255;
      out[di + 1] = src.data[si + 1] / 255;
      out[di + 2] = src.data[si + 2] / 255;
    }
  }
  return out;
}

export function embedPhoto(s: Surface): Float64Array {
  const rgb = surfaceToRGB(s);
  const out = new Float64Array(PHOTO_DIM);
  let o = 0;

  const luma = new Float64Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    luma[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
  }

  // ---- colour mean + std per cell
  const cstep = SIZE / CGRID;
  for (let gy = 0; gy < CGRID; gy++) {
    for (let gx = 0; gx < CGRID; gx++) {
      const m = [0, 0, 0], v = [0, 0, 0];
      let n = 0;
      for (let y = gy * cstep; y < (gy + 1) * cstep; y++) {
        for (let x = gx * cstep; x < (gx + 1) * cstep; x++) {
          const i = (y * SIZE + x) * 3;
          m[0] += rgb[i]; m[1] += rgb[i + 1]; m[2] += rgb[i + 2];
          n++;
        }
      }
      for (let c = 0; c < 3; c++) m[c] /= n;
      for (let y = gy * cstep; y < (gy + 1) * cstep; y++) {
        for (let x = gx * cstep; x < (gx + 1) * cstep; x++) {
          const i = (y * SIZE + x) * 3;
          v[0] += (rgb[i] - m[0]) ** 2;
          v[1] += (rgb[i + 1] - m[1]) ** 2;
          v[2] += (rgb[i + 2] - m[2]) ** 2;
        }
      }
      for (let c = 0; c < 3; c++) out[o++] = m[c];
      for (let c = 0; c < 3; c++) out[o++] = Math.sqrt(v[c] / n);
    }
  }

  // ---- hue histogram weighted by saturation*value (grey pixels contribute ~0,
  // so a washed-out background does not dominate the colour signature)
  const hue = new Float64Array(HUE_BINS);
  let hueTot = 0;
  for (let i = 0; i < SIZE * SIZE; i++) {
    const [h, sat, val] = rgbToHsv(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    const w = sat * val;
    const b = Math.min(HUE_BINS - 1, Math.floor(h * HUE_BINS));
    hue[b] += w; hueTot += w;
  }
  for (let b = 0; b < HUE_BINS; b++) out[o++] = hue[b] / (hueTot + 1e-8);

  // ---- gradients
  const gx = new Float64Array(SIZE * SIZE), gy = new Float64Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x;
      const xm = x > 0 ? luma[i - 1] : luma[i];
      const xp = x < SIZE - 1 ? luma[i + 1] : luma[i];
      const ym = y > 0 ? luma[i - SIZE] : luma[i];
      const yp = y < SIZE - 1 ? luma[i + SIZE] : luma[i];
      gx[i] = xp - xm; gy[i] = yp - ym;
    }
  }
  const gstep = SIZE / GGRID;
  for (let ty = 0; ty < GGRID; ty++) {
    for (let tx = 0; tx < GGRID; tx++) {
      const h = new Float64Array(ORIENT);
      let sum = 0;
      for (let y = ty * gstep; y < (ty + 1) * gstep; y++) {
        for (let x = tx * gstep; x < (tx + 1) * gstep; x++) {
          const i = y * SIZE + x;
          const mag = Math.hypot(gx[i], gy[i]);
          let ang = Math.atan2(gy[i], gx[i]) % Math.PI;
          if (ang < 0) ang += Math.PI;
          h[Math.min(ORIENT - 1, Math.floor((ang / Math.PI) * ORIENT))] += mag;
          sum += mag;
        }
      }
      for (let b = 0; b < ORIENT; b++) out[o++] = h[b] / (sum + 1e-8);
    }
  }

  // ---- luma mean + edge density per cell
  for (let gy2 = 0; gy2 < CGRID; gy2++) {
    for (let gx2 = 0; gx2 < CGRID; gx2++) {
      let lm = 0, ed = 0, n = 0;
      for (let y = gy2 * cstep; y < (gy2 + 1) * cstep; y++) {
        for (let x = gx2 * cstep; x < (gx2 + 1) * cstep; x++) {
          const i = y * SIZE + x;
          lm += luma[i];
          ed += Math.hypot(gx[i], gy[i]);
          n++;
        }
      }
      out[o++] = lm / n;
      out[o++] = ed / n;
    }
  }
  return out;
}

/** Load an image File/Blob into a Surface-compatible canvas. */
export async function fileToSurface(file: Blob, max = 256): Promise<Surface> {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, max / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale));
  const h = Math.max(1, Math.round(bmp.height * scale));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  cv.getContext('2d')!.drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  return cv as unknown as Surface;
}

/** Grab the current video frame into a Surface. */
export function videoToSurface(video: HTMLVideoElement, max = 256): Surface {
  const scale = Math.min(1, max / Math.max(video.videoWidth || 1, video.videoHeight || 1));
  const w = Math.max(1, Math.round((video.videoWidth || 128) * scale));
  const h = Math.max(1, Math.round((video.videoHeight || 128) * scale));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  cv.getContext('2d')!.drawImage(video, 0, 0, w, h);
  return cv as unknown as Surface;
}
