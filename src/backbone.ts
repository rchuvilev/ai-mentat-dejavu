/**
 * Optional pretrained backbone (DINOv2-small) via transformers.js.
 *
 * This is the accuracy gap versus Teachable Machine. TM freezes MobileNet v1
 * (alpha 0.25, 224x224, ImageNet) and trains a softmax head on its embeddings.
 * The hand-built photo extractor is fast and needs no download, but it cannot
 * generalise from a handful of examples the way a pretrained backbone does.
 *
 * DINOv2-small rather than MobileNet, for a measured reason: most ImageNet
 * *classifier* ONNX exports only expose `logits`, which are a poor
 * representation for out-of-distribution input (MobileNetV4 logits scored a
 * separation ratio of 1.40 and 51.7% on the shapes task). Models with NO
 * classifier head export `last_hidden_state` — real features. DINOv2 measured
 * 2.26 / 100% on the same task.
 *
 * Cost: ~2 s/image under WASM on a phone, versus ~5 ms for the hand-built
 * extractor. That is the trade the UI makes explicit.
 *
 * Everything loads from CDNs so the deployed site stays small: the transformers
 * bundle and ORT runtime from jsDelivr, the weights from the HuggingFace CDN
 * (verified to send access-control-allow-origin for this site).
 */

const BUNDLE = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';
/** Self-supervised ViT backbones. Both are 384-d and 224x224, so one loader
 *  serves both — only the repo id differs. */
const MODELS: Record<string, { repo: string; mb: number }> = {
  dinov2: { repo: 'onnx-community/dinov2-small', mb: 23 },
  // DINOv3 trained on LVD-1689M vs DINOv2's LVD-142M (~10x the corpus) and is
  // SMALLER on disk. Same output shape, so it needs no separate code path.
  dinov3: { repo: 'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX', mb: 21 },
};

export interface Backbone {
  id: string;
  dim: number;
  embed(surface: unknown): Promise<Float64Array>;
}

const loading: Record<string, Promise<Backbone> | undefined> = {};
const loaded: Record<string, Backbone | undefined> = {};

export function backboneReady(id = 'dinov2'): boolean { return !!loaded[id]; }

/**
 * Load the backbone once. Concurrent callers share the same promise, so tapping
 * "train" twice cannot start two 23 MB downloads.
 */
export function loadBackbone(
  id = 'dinov2',
  onProgress?: (msg: string) => void,
): Promise<Backbone> {
  const spec = MODELS[id];
  if (!spec) return Promise.reject(new Error(`unknown backbone "${id}"`));
  const hit = loaded[id];
  if (hit) return Promise.resolve(hit);
  const inFlight = loading[id];
  if (inFlight) return inFlight;

  loading[id] = (async () => {
    onProgress?.('loading transformers.js…');
    const T: any = await import(/* @vite-ignore */ BUNDLE);
    const { env, AutoModel, AutoProcessor, RawImage } = T;

    // Weights come from the HF hub; only the runtime is pinned locally-agnostic.
    env.allowRemoteModels = true;
    env.allowLocalModels = false;
    env.backends.onnx.wasm.numThreads = 1;   // PRoot/mobile dislikes thread pools

    onProgress?.(`downloading ${id} (~${spec.mb} MB, cached after first run)…`);
    let lastPct = -10;
    const model = await AutoModel.from_pretrained(spec.repo, {
      dtype: 'q8', device: 'wasm',
      progress_callback: (p: any) => {
        if (p?.status === 'progress' && typeof p.progress === 'number') {
          const v = Math.round(p.progress);
          if (v >= lastPct + 10) { lastPct = v; onProgress?.(`weights ${v}%`); }
        }
      },
    });
    const proc = await AutoProcessor.from_pretrained(spec.repo);

    // Confirm this export really gives features, not logits. If a future model
    // swap regresses to logits, fail loudly instead of silently degrading.
    const outNames: string[] | undefined = model.sessions?.model?.outputNames;
    if (outNames && !outNames.includes('last_hidden_state')) {
      throw new Error(
        `backbone exports ${JSON.stringify(outNames)} — needs last_hidden_state ` +
        `(a classifier-head export gives logits, which measure far worse)`);
    }

    const bb: Backbone = {
      id,
      dim: 384,
      async embed(surface: any): Promise<Float64Array> {
        const cx = surface.getContext('2d');
        const d = cx.getImageData(0, 0, surface.width, surface.height);
        const img = new RawImage(
          new Uint8ClampedArray(d.data), surface.width, surface.height, 4);
        const out = await model(await proc(img));
        const t = out.last_hidden_state;
        // CLS token (index 0) is the standard image-level embedding for a ViT
        const D = t.dims[2];
        const v = new Float64Array(D);
        for (let j = 0; j < D; j++) v[j] = t.data[j];
        return v;
      },
    };
    loaded[id] = bb;
    onProgress?.(`${id} ready (384-d features, frozen)`);
    return bb;
  })();

  // A failed load must not poison later attempts.
  loading[id]!.catch(() => { loading[id] = undefined; });
  return loading[id]!;
}
