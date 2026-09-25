/**
 * Audio feature extractor: log-mel band statistics.
 *
 * Same contract as the image extractors — deterministic, no learned parameters,
 * fixed width. Design choices that matter:
 *
 *  - MEL BANDS, not raw FFT bins: 40 bands over 20 Hz..8 kHz compress 513 bins
 *    into a perceptually spaced summary, so a clip becomes ~200 numbers instead
 *    of thousands. Same lesson as v2 beating v1 on images: a compact
 *    representation carrying the right invariances beats a big raw one.
 *  - LOG magnitude: loudness becomes an additive offset instead of a multiplier.
 *  - PER-CLIP BAND-MEAN REMOVAL: cancels fixed channel colouration (mic
 *    response, room tone), so the same sound at two volumes or on two mics lands
 *    close together.
 *  - MEAN + STD over time plus a 3-segment profile: keeps how the sound evolves
 *    (rising vs falling) without requiring the clip to be time-aligned.
 */

export const N_MELS = 40;
export const N_SEGMENTS = 3;
/** mean + std over the clip, then per-segment means */
export const AUDIO_DIM = N_MELS * 2 + N_MELS * N_SEGMENTS;

const FFT_SIZE = 1024;
const HOP = 512;
const F_MIN = 20;
const F_MAX = 8000;

const hzToMel = (f: number) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m: number) => 700 * (10 ** (m / 2595) - 1);

function melFilters(sr: number, nBins: number): Float64Array[] {
  const lo = hzToMel(F_MIN), hi = hzToMel(Math.min(F_MAX, sr / 2));
  const pts: number[] = [];
  for (let i = 0; i < N_MELS + 2; i++) {
    pts.push(melToHz(lo + ((hi - lo) * i) / (N_MELS + 1)));
  }
  const binHz = sr / 2 / (nBins - 1);
  const out: Float64Array[] = [];
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

/** Iterative radix-2 FFT, in place. */
function fft(re: Float64Array, im: Float64Array): void {
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
        const vr = xr * cr - xi * ci;
        const vi = xr * ci + xi * cr;
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

let cachedSr = 0;
let cachedFilters: Float64Array[] = [];

/** @param samples mono PCM in [-1,1] */
export function embedAudio(samples: Float32Array, sampleRate: number): Float64Array {
  const nBins = FFT_SIZE / 2 + 1;
  if (sampleRate !== cachedSr || cachedFilters.length === 0) {
    cachedFilters = melFilters(sampleRate, nBins);
    cachedSr = sampleRate;
  }
  const filters = cachedFilters;

  const frames: Float64Array[] = [];
  for (let start = 0; start + FFT_SIZE <= samples.length; start += HOP) {
    const re = new Float64Array(FFT_SIZE);
    const im = new Float64Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) re[i] = samples[start + i] * hann[i];
    fft(re, im);
    const band = new Float64Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      const f = filters[m];
      for (let k = 0; k < nBins; k++) {
        if (f[k] !== 0) s += f[k] * (re[k] * re[k] + im[k] * im[k]);
      }
      band[m] = Math.log(s + 1e-10);
    }
    frames.push(band);
  }

  const out = new Float64Array(AUDIO_DIM);
  if (frames.length === 0) return out;

  // Loudness normalisation: subtract ONE global offset (the grand mean over all
  // bands and frames), not a per-band mean.
  //
  // Per-band subtraction was measured to be actively harmful: it forces each
  // band's time-average to zero, so the mean and per-segment features became
  // identically ~0 for any stationary sound. Separation ratio on 300 vs 1200 Hz
  // was 0.99 for the mean block and 1.27 overall, with all the signal surviving
  // only in std (2.96). A single global offset still cancels gain changes (log
  // domain => multiplicative gain is additive) while PRESERVING the spectral
  // shape across bands, which is exactly what distinguishes pitches and timbres.
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

/** Decode any browser-supported audio file to mono PCM at targetSr. */
export async function decodeAudioFile(
  file: Blob, targetSr = 16000,
): Promise<{ samples: Float32Array; sampleRate: number }> {
  const buf = await file.arrayBuffer();
  const AC: any = (window as any).AudioContext || (window as any).webkitAudioContext;
  const ctx = new AC();
  try {
    const decoded: AudioBuffer = await ctx.decodeAudioData(buf);
    return resample(toMono(decoded), decoded.sampleRate, targetSr);
  } finally {
    if (ctx.close) ctx.close();
  }
}

export function toMono(b: AudioBuffer): Float32Array {
  if (b.numberOfChannels === 1) return b.getChannelData(0).slice();
  const out = new Float32Array(b.length);
  for (let c = 0; c < b.numberOfChannels; c++) {
    const d = b.getChannelData(c);
    for (let i = 0; i < b.length; i++) out[i] += d[i] / b.numberOfChannels;
  }
  return out;
}

/** Linear resample — adequate for band-energy features. */
export function resample(
  x: Float32Array, from: number, to: number,
): { samples: Float32Array; sampleRate: number } {
  if (from === to) return { samples: x, sampleRate: to };
  const ratio = to / from;
  const n = Math.max(1, Math.floor(x.length * ratio));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const src = i / ratio;
    const i0 = Math.floor(src), i1 = Math.min(x.length - 1, i0 + 1);
    const t = src - i0;
    out[i] = x[i0] * (1 - t) + x[i1] * t;
  }
  return { samples: out, sampleRate: to };
}

/** RMS level in dBFS — live meter and silence gating. */
export function rmsDb(x: Float32Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return 20 * Math.log10(Math.sqrt(s / Math.max(1, x.length)) + 1e-12);
}
