/**
 * Teachable Machine's own backbone: MobileNetV2 alpha 0.35, no classifier head.
 *
 * This is literally the model TM trains on top of — Apache-2.0, served from
 * Google's public bucket with `access-control-allow-origin: *` (verified).
 *
 * Why it earns its place next to DINOv2, measured on this device:
 *
 *   backbone            dim    per image   download
 *   TM MobileNetV2      1280   70 ms (WebGL) / 1188 ms (CPU)   1.6 MB
 *   DINOv2 (ONNX/WASM)   384   ~2000 ms                        23 MB
 *
 * 28x faster and 14x smaller. The reason it works at all is the backend:
 * TensorFlow.js uses WebGL shaders, which this Adreno GPU runs fine, whereas
 * ONNX Runtime's WebGPU backend HUNG during shader compilation on the same
 * device. Different runtime, different GPU path, different outcome.
 *
 * The runtime is loaded from a CDN on first use so the deployed site stays small
 * and users who never pick this extractor pay nothing.
 */

const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.22.0/dist/tf-core.min.js';
const TFJS_WEBGL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.22.0/dist/tf-backend-webgl.min.js';
const TFJS_CPU = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@4.22.0/dist/tf-backend-cpu.min.js';
const TFJS_LAYERS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-layers@4.22.0/dist/tf-layers.min.js';
const MODEL =
  'https://storage.googleapis.com/teachable-machine-models/' +
  'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top/model.json';

/** MobileNetV2 alpha 0.35 global-average-pooled feature width. */
export const TM_DIM = 1280;
const INPUT = 224;

export interface TmBackbone {
  dim: number;
  backend: string;
  embed(surface: unknown): Promise<Float64Array>;
}

let loading: Promise<TmBackbone> | null = null;
let loaded: TmBackbone | null = null;

export function tmReady(): boolean { return loaded !== null; }

/** Load a classic <script> once; tfjs UMD bundles are not ES modules. */
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

export function loadTmNet(onProgress?: (m: string) => void): Promise<TmBackbone> {
  if (loaded) return Promise.resolve(loaded);
  if (loading) return loading;        // a double tap must not fetch twice

  loading = (async () => {
    onProgress?.('loading TensorFlow.js…');
    // Order matters: core first, then a backend, then layers.
    await loadScript(TFJS);
    await Promise.all([loadScript(TFJS_WEBGL), loadScript(TFJS_CPU)]);
    await loadScript(TFJS_LAYERS);
    const tf: any = (window as any).tf;
    if (!tf) throw new Error('TensorFlow.js did not initialise');

    // Prefer the GPU: 70 ms vs 1188 ms measured. Fall back rather than fail.
    let backend = 'cpu';
    try {
      await tf.setBackend('webgl');
      await tf.ready();
      backend = tf.getBackend();
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
      backend = 'cpu';
    }
    if (backend !== 'webgl') {
      onProgress?.('WebGL unavailable — using CPU, expect ~1 s per image');
    }

    onProgress?.('downloading MobileNetV2 (~1.6 MB, then cached)…');
    const model = await tf.loadLayersModel(MODEL);

    // This must be the headless feature extractor, not a classifier. A
    // classifier export would emit [1,1000] logits, which measured a separation
    // ratio of 1.40 versus 2.26 for real features — fail loudly instead.
    const shape: number[] = model.outputs[0].shape;
    if (shape.length !== 4 || shape[3] !== TM_DIM) {
      throw new Error(
        `expected a [_,h,w,${TM_DIM}] feature map, got ${JSON.stringify(shape)} — ` +
        `this looks like a classifier head, not a backbone`);
    }

    const bb: TmBackbone = {
      dim: TM_DIM,
      backend,
      async embed(surface: any): Promise<Float64Array> {
        // MobileNetV2 expects 224x224 in [-1,1].
        const src = surface as HTMLCanvasElement;
        let input = src;
        if (src.width !== INPUT || src.height !== INPUT) {
          const cv = document.createElement('canvas');
          cv.width = cv.height = INPUT;
          cv.getContext('2d')!.drawImage(src, 0, 0, INPUT, INPUT);
          input = cv;
        }
        // tf.tidy frees the intermediates; without it WebGL textures leak and
        // the page dies after a few hundred frames in continuous mode.
        const t = tf.tidy(() => {
          const px = tf.expandDims(
            tf.sub(tf.div(tf.cast(tf.browser.fromPixels(input), 'float32'), 127.5), 1), 0);
          // [1,7,7,1280] -> global average pool -> [1280]
          return tf.squeeze(tf.mean(model.predict(px), [1, 2]));
        });
        const data = await t.data();
        t.dispose();
        return Float64Array.from(data);
      },
    };
    loaded = bb;
    onProgress?.(`TM MobileNetV2 ready (${TM_DIM}-d, ${backend})`);
    return bb;
  })();

  loading.catch(() => { loading = null; });   // a failure must not block retries
  return loading;
}


