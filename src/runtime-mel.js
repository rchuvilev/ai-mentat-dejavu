// Inlined into exported scripts for the `mel` extractor.
// Accepts { pcm: Float32Array, sampleRate } or a bare Float32Array (assumed
// 16 kHz). Pure JS FFT, so it runs identically in a browser, Node and Bun.
const N_MELS = 40, N_SEGMENTS = 3;
const AUDIO_DIM = N_MELS * 2 + N_MELS * N_SEGMENTS;
const FFT_SIZE = 1024, HOP = 512, F_MIN = 20, F_MAX = 8000;

const hzToMel = f => 2595 * Math.log10(1 + f / 700);
const melToHz = m => 700 * (10 ** (m / 2595) - 1);

let _sr = 0, _filters = null;
function melFilters(sr, nBins) {
  const lo = hzToMel(F_MIN), hi = hzToMel(Math.min(F_MAX, sr / 2));
  const pts = [];
  for (let i = 0; i < N_MELS + 2; i++) pts.push(melToHz(lo + ((hi - lo) * i) / (N_MELS + 1)));
  const binHz = sr / 2 / (nBins - 1);
  const out = [];
  for (let m = 0; m < N_MELS; m++) {
    const f = new Float64Array(nBins);
    const a = pts[m], b = pts[m + 1], c = pts[m + 2];
    for (let k = 0; k < nBins; k++) {
      const hz = k * binHz;
      if (hz >= a && hz <= b) f[k] = b === a ? 1 : (hz - a) / (b - a);
      else if (hz > b && hz <= c) f[k] = c === b ? 1 : (c - hz) / (c - b);
    }
    out.push(f);
  }
  return out;
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const ur = re[i + k], ui = im[i + k];
        const xr = re[i + k + half], xi = im[i + k + half];
        const vr = xr * cr - xi * ci, vi = xr * ci + xi * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

const hann = (() => {
  const w = new Float64Array(FFT_SIZE);
  for (let i = 0; i < FFT_SIZE; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
  }
  return w;
})();

async function embed(input) {
  let samples, sampleRate;
  if (input instanceof Float32Array) { samples = input; sampleRate = 16000; }
  else if (input && input.pcm) { samples = input.pcm; sampleRate = input.sampleRate || 16000; }
  else throw new Error('pass { pcm: Float32Array, sampleRate } or a Float32Array');

  const nBins = FFT_SIZE / 2 + 1;
  if (sampleRate !== _sr || !_filters) { _filters = melFilters(sampleRate, nBins); _sr = sampleRate; }

  const frames = [];
  for (let start = 0; start + FFT_SIZE <= samples.length; start += HOP) {
    const re = new Float64Array(FFT_SIZE), im = new Float64Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) re[i] = samples[start + i] * hann[i];
    fft(re, im);
    const band = new Float64Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      const f = _filters[m];
      for (let k = 0; k < nBins; k++) if (f[k] !== 0) s += f[k] * (re[k] * re[k] + im[k] * im[k]);
      band[m] = Math.log(s + 1e-10);
    }
    frames.push(band);
  }
  const out = new Float64Array(AUDIO_DIM);
  if (!frames.length) return out;

  // ONE global offset, not per band: subtracting each band's own mean forces its
  // time-average to zero and destroys the spectral shape (measured separation
  // 1.27 -> 26.24 after this fix).
  let grand = 0;
  for (const f of frames) for (let m = 0; m < N_MELS; m++) grand += f[m];
  grand /= frames.length * N_MELS;
  const centred = frames.map(f => {
    const c = new Float64Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) c[m] = f[m] - grand;
    return c;
  });

  let o = 0;
  for (let m = 0; m < N_MELS; m++) {
    let mu = 0;
    for (const f of centred) mu += f[m];
    mu /= centred.length;
    let v = 0;
    for (const f of centred) v += (f[m] - mu) ** 2;
    out[o++] = mu;
    out[o++] = Math.sqrt(v / centred.length);
  }
  for (let s = 0; s < N_SEGMENTS; s++) {
    const a = Math.floor((s * centred.length) / N_SEGMENTS);
    const b = Math.max(a + 1, Math.floor(((s + 1) * centred.length) / N_SEGMENTS));
    for (let m = 0; m < N_MELS; m++) {
      let mu = 0, n = 0;
      for (let i = a; i < b && i < centred.length; i++) { mu += centred[i][m]; n++; }
      out[o++] = n ? mu / n : 0;
    }
  }
  return out;
}
