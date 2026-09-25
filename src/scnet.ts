/**
 * Teachable Machine's audio backbone: the speech-commands browser_fft model.
 *
 * TM trains on top of @tensorflow-models/speech-commands, which is a small CNN
 * pretrained on spoken words. That is the gap versus the built-in log-mel
 * extractor: log-mel band statistics separate claps and whistles fine, but two
 * spoken syllables ("la" vs "tu") are a PHONETIC distinction, and a model
 * trained on speech carries learned representations for exactly that.
 *
 * The library itself is not used — it pins tfjs ^1.3 as a peer dependency and we
 * already load 4.22 for the image backbone. The model is a plain layers-model,
 * so it loads directly and the preprocessing is reimplemented here.
 *
 * PREPROCESSING MUST MATCH EXACTLY, read out of the library source rather than
 * guessed, because a mismatch degrades features silently — the same failure mode
 * as the per-band-mean bug in audio.ts:
 *   input shape       [43, 232, 1]      43 frames x 232 bins
 *   sample rate       44100 Hz          (SAMPLE_RATE_HZ)
 *   fftSize           1024              -> 512 bins, TRUNCATED to 232 (~10 kHz)
 *   frame duration    1024/44100        ~= 23.2 ms, so 43 frames ~= 1 s
 *   magnitude         getFloatFrequencyData -> DECIBELS, not power
 *   normalisation     (x - mean) / (sqrt(var) + eps) over the whole tensor
 *
 * The feature layer is the penultimate Dense (2000 units); the final 20-way
 * Dense is the word classifier and is discarded.
 */

const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.22.0/dist/tf-core.min.js';
const TFJS_WEBGL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.22.0/dist/tf-backend-webgl.min.js';
const TFJS_CPU = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@4.22.0/dist/tf-backend-cpu.min.js';
const TFJS_LAYERS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-layers@4.22.0/dist/tf-layers.min.js';
const MODEL =
  'https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.4/browser_fft/18w/model.json';

export const SC_SR = 44100;
export const SC_FFT = 1024;
export const SC_FRAMES = 43;
export const SC_BINS = 232;
/** Penultimate Dense width — the embedding TM trains on. */
export const SC_DIM = 2000;

export interface ScBackbone {
  dim: number;
  backend: string;
  /** @param pcm mono PCM; resampled internally to 44.1 kHz if needed */
  embed(pcm: Float32Array, sampleRate: number): Promise<Float64Array>;
}

let loading: Promise<ScBackbone> | null = null;
let loaded: ScBackbone | null = null;

export function scReady(): boolean { return loaded !== null; }

function loadScript(src: string): Promise<void> {
  return new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) return res();
    const el = document.createElement('script');
    el.src = src;
    el.onload = () => res();
    el.onerror = () => rej(new Error(`failed to load ${src.split('/').pop()}`));
    document.head.appendChild(el);
  });
}

/** Linear resample to 44.1 kHz — the rate the model was trained at. */
function toScRate(pcm: Float32Array, from: number): Float32Array {
  if (from === SC_SR) return pcm;
  const ratio = SC_SR / from;
  const n = Math.max(1, Math.floor(pcm.length * ratio));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = i / ratio;
    const i0 = Math.floor(s), i1 = Math.min(pcm.length - 1, i0 + 1);
    const t = s - i0;
    out[i] = pcm[i0] * (1 - t) + pcm[i1] * t;
  }
  return out;
}

const hann = (() => {
  const w = new Float32Array(SC_FFT);
  for (let i = 0; i < SC_FFT; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (SC_FFT - 1)));
  }
  return w;
})();

/** Radix-2 FFT, in place. */
function fft(re: Float32Array, im: Float32Array): void {
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

/**
 * Build the [43, 232] dB spectrogram the model expects.
 *
 * Frames do NOT overlap: the library's default overlapFactor for this model
 * gives contiguous 1024-sample frames, so 43 frames span 43*1024/44100 ~= 1.0 s.
 * The newest audio is kept when the input is longer.
 */
export function scSpectrogram(pcm44: Float32Array): Float32Array {
  const need = SC_FRAMES * SC_FFT;
  const src = pcm44.length >= need
    ? pcm44.subarray(pcm44.length - need)          // newest 1 s
    : (() => { const p = new Float32Array(need); p.set(pcm44, need - pcm44.length); return p; })();

  const out = new Float32Array(SC_FRAMES * SC_BINS);
  const re = new Float32Array(SC_FFT), im = new Float32Array(SC_FFT);
  for (let f = 0; f < SC_FRAMES; f++) {
    const off = f * SC_FFT;
    for (let i = 0; i < SC_FFT; i++) { re[i] = src[off + i] * hann[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < SC_BINS; k++) {
      const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]) / SC_FFT;
      // getFloatFrequencyData reports dB, floored the way WebAudio does
      out[f * SC_BINS + k] = Math.max(-100, 20 * Math.log10(mag + 1e-10));
    }
  }

  // Per-spectrogram z-normalisation, matching the library's normalize():
  // (x - mean) / (sqrt(variance) + epsilon)
  let mean = 0;
  for (let i = 0; i < out.length; i++) mean += out[i];
  mean /= out.length;
  let varsum = 0;
  for (let i = 0; i < out.length; i++) varsum += (out[i] - mean) ** 2;
  const sd = Math.sqrt(varsum / out.length) + 1e-7;
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - mean) / sd;
  return out;
}

export function loadScNet(onProgress?: (m: string) => void): Promise<ScBackbone> {
  if (loaded) return Promise.resolve(loaded);
  if (loading) return loading;

  loading = (async () => {
    onProgress?.('loading TensorFlow.js…');
    await loadScript(TFJS);
    await Promise.all([loadScript(TFJS_WEBGL), loadScript(TFJS_CPU)]);
    await loadScript(TFJS_LAYERS);
    const tf: any = (window as any).tf;
    if (!tf) throw new Error('TensorFlow.js did not initialise');

    let backend = 'cpu';
    try { await tf.setBackend('webgl'); await tf.ready(); backend = tf.getBackend(); }
    catch { await tf.setBackend('cpu'); await tf.ready(); backend = 'cpu'; }

    onProgress?.('downloading speech-commands (~2 MB, then cached)…');
    const full = await tf.loadLayersModel(MODEL);

    // Sanity-check the architecture before relying on it.
    const inShape: number[] = full.inputs[0].shape;
    if (inShape[1] !== SC_FRAMES || inShape[2] !== SC_BINS) {
      throw new Error(
        `expected input [_,${SC_FRAMES},${SC_BINS},1], got ${JSON.stringify(inShape)}`);
    }

    // Take the penultimate Dense as the embedding; the last Dense is the
    // 20-word classifier, whose logits would be a far worse representation
    // (the same trap that made MobileNetV4 logits score 51.7%).
    const dense = full.layers.filter((l: any) => l.getClassName() === 'Dense');
    if (dense.length < 2) throw new Error('expected two Dense layers');
    const featLayer = dense[dense.length - 2];
    if (featLayer.units !== SC_DIM) {
      throw new Error(`expected a ${SC_DIM}-unit feature layer, got ${featLayer.units}`);
    }
    const model = tf.model({ inputs: full.inputs, outputs: featLayer.output });

    const bb: ScBackbone = {
      dim: SC_DIM,
      backend,
      async embed(pcm: Float32Array, sampleRate: number): Promise<Float64Array> {
        const spec = scSpectrogram(toScRate(pcm, sampleRate));
        // tf.tidy frees intermediates; WebGL textures otherwise leak in
        // continuous mode.
        const t = tf.tidy(() => {
          const x = tf.tensor4d(spec, [1, SC_FRAMES, SC_BINS, 1]);
          return tf.squeeze(model.predict(x));
        });
        const data = await t.data();
        t.dispose();
        return Float64Array.from(data);
      },
    };
    loaded = bb;
    onProgress?.(`speech-commands ready (${SC_DIM}-d, ${backend})`);
    return bb;
  })();

  loading.catch(() => { loading = null; });
  return loading;
}
