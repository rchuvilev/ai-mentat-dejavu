// Inlined into exported scripts for the `photo` extractor.
// Accepts a canvas, an ImageData, an HTMLImageElement/ImageBitmap, or
// { data, width, height } — so the same file works in a browser and in Node
// with any decoder that yields raw RGBA.
const SIZE = 96, CGRID = 3, GGRID = 2, HUE_BINS = 12, ORIENT = 8;
const PHOTO_DIM = CGRID * CGRID * 6 + HUE_BINS + GGRID * GGRID * ORIENT + CGRID * CGRID * 2;

function toRGBA(input) {
  // { data, width, height } — the universal shape
  if (input && input.data && input.width && input.height) return input;
  // a canvas (browser or node-canvas)
  if (input && typeof input.getContext === 'function') {
    return input.getContext('2d').getImageData(0, 0, input.width, input.height);
  }
  // an <img> / ImageBitmap: needs a canvas to read pixels
  if (typeof document !== 'undefined' &&
      (input instanceof HTMLImageElement ||
       (typeof ImageBitmap !== 'undefined' && input instanceof ImageBitmap))) {
    const w = input.naturalWidth || input.width;
    const h = input.naturalHeight || input.height;
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    cv.getContext('2d').drawImage(input, 0, 0);
    return cv.getContext('2d').getImageData(0, 0, w, h);
  }
  throw new Error('unsupported image input: pass a canvas, ImageData, <img>, ' +
                  'or { data, width, height } with RGBA bytes');
}

function rgbToHsv(r, g, b) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d > 1e-9) {
    if (mx === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (mx === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  return [h, mx <= 1e-9 ? 0 : d / mx, mx];
}

async function embed(input) {
  const img = toRGBA(input);
  const W = img.width, H = img.height, src = img.data;
  const rgb = new Float64Array(SIZE * SIZE * 3);
  for (let y = 0; y < SIZE; y++) {
    const sy = Math.min(H - 1, Math.floor((y * H) / SIZE));
    for (let x = 0; x < SIZE; x++) {
      const sx = Math.min(W - 1, Math.floor((x * W) / SIZE));
      const si = (sy * W + sx) * 4, di = (y * SIZE + x) * 3;
      rgb[di] = src[si] / 255; rgb[di + 1] = src[si + 1] / 255; rgb[di + 2] = src[si + 2] / 255;
    }
  }
  const out = new Float64Array(PHOTO_DIM);
  let o = 0;
  const luma = new Float64Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    luma[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
  }
  const cstep = SIZE / CGRID;
  for (let gy = 0; gy < CGRID; gy++) {
    for (let gx = 0; gx < CGRID; gx++) {
      const m = [0, 0, 0], v = [0, 0, 0];
      let n = 0;
      for (let y = gy * cstep; y < (gy + 1) * cstep; y++) {
        for (let x = gx * cstep; x < (gx + 1) * cstep; x++) {
          const i = (y * SIZE + x) * 3;
          m[0] += rgb[i]; m[1] += rgb[i + 1]; m[2] += rgb[i + 2]; n++;
        }
      }
      for (let c = 0; c < 3; c++) m[c] /= n;
      for (let y = gy * cstep; y < (gy + 1) * cstep; y++) {
        for (let x = gx * cstep; x < (gx + 1) * cstep; x++) {
          const i = (y * SIZE + x) * 3;
          v[0] += (rgb[i] - m[0]) ** 2; v[1] += (rgb[i + 1] - m[1]) ** 2;
          v[2] += (rgb[i + 2] - m[2]) ** 2;
        }
      }
      for (let c = 0; c < 3; c++) out[o++] = m[c];
      for (let c = 0; c < 3; c++) out[o++] = Math.sqrt(v[c] / n);
    }
  }
  const hue = new Float64Array(HUE_BINS);
  let hueTot = 0;
  for (let i = 0; i < SIZE * SIZE; i++) {
    const [h, sat, val] = rgbToHsv(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    const w = sat * val;
    hue[Math.min(HUE_BINS - 1, Math.floor(h * HUE_BINS))] += w;
    hueTot += w;
  }
  for (let b = 0; b < HUE_BINS; b++) out[o++] = hue[b] / (hueTot + 1e-8);
  const gx2 = new Float64Array(SIZE * SIZE), gy2 = new Float64Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x;
      gx2[i] = (x < SIZE - 1 ? luma[i + 1] : luma[i]) - (x > 0 ? luma[i - 1] : luma[i]);
      gy2[i] = (y < SIZE - 1 ? luma[i + SIZE] : luma[i]) - (y > 0 ? luma[i - SIZE] : luma[i]);
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
          const mag = Math.hypot(gx2[i], gy2[i]);
          let ang = Math.atan2(gy2[i], gx2[i]) % Math.PI;
          if (ang < 0) ang += Math.PI;
          h[Math.min(ORIENT - 1, Math.floor((ang / Math.PI) * ORIENT))] += mag;
          sum += mag;
        }
      }
      for (let b = 0; b < ORIENT; b++) out[o++] = h[b] / (sum + 1e-8);
    }
  }
  for (let gy = 0; gy < CGRID; gy++) {
    for (let gx = 0; gx < CGRID; gx++) {
      let lm = 0, ed = 0, n = 0;
      for (let y = gy * cstep; y < (gy + 1) * cstep; y++) {
        for (let x = gx * cstep; x < (gx + 1) * cstep; x++) {
          const i = y * SIZE + x;
          lm += luma[i]; ed += Math.hypot(gx2[i], gy2[i]); n++;
        }
      }
      out[o++] = lm / n; out[o++] = ed / n;
    }
  }
  return out;
}
