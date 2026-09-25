# AI Déjà Vu

Train an image classifier **in the browser**, persist it to **IndexedDB**, then run
live inference where every prediction fires a switchable `onMatched` /
`onNotMatched` action — `console.log`, webhook, both, or off.

No server, no upload, no API key. Training and inference both run locally.

**Live:** https://hexstack.app/ai-dejavu/ · **Studio:** https://hexstack.app/ai-dejavu/studio

Two pages:
- **`/`** — the built-in shapes demo, showing why representation beats classifier.
- **`/studio`** — train on **your own** examples: camera or image files for photos,
  microphone or audio files for sounds, your own labels, then run new inputs
  through the model with the same webhook/console actions.

## Architecture

A **frozen feature extractor** turns each image into a vector; a small **softmax
head** is the only thing trained. That's the Teachable Machine architecture, and
it's why a trained model is ~1 KB and fits in a database row.

```
image → extractor (frozen, 0 params) → vector → softmax head (81 params) → class
                                                          ↓
                                            onMatched / onNotMatched
                                            console · webhook · both · off
```

## Quick start

```sh
npm install
npm run build          # tsc -> dist/
npm test               # 24 core + 33 action tests
npm run site           # serve the MVP at :8773
npm run cli            # headless v1 vs v2 comparison
```

## What it measures

| representation | dim | infer | separation | test acc |
|---|---|---|---|---|
| v1 absolute orientation | 224 | ~3 ms | 0.99 | **38.9%** |
| **v2 rotation-invariant** | **26** | ~5 ms | 1.99 | **94.4%** |
| DINOv2-small CLS (pretrained) | 384 | ~2.1 s | 2.26 | **100.0%** |

**Representation beats classifier.** v2 has 8.6× *fewer* dimensions than v1 and
scores ~56 points higher. The classifier is byte-identical.

### v1 is a deliberate broken control

```
        rotation gap   between-class   ratio
v1         0.713          0.414        0.58   rotation dominates
v2         0.196          0.395        2.01   OK
```

Rotating the *same square* moves its v1 embedding **further** than changing the
shape does. Within-class variation exceeds between-class separation, so no amount
of training can succeed. v2 measures gradient orientation *relative to the shape's
centroid*, adds radial ring occupancy and pose-free scalars (compactness, extent).

## Studio: your own examples

| modality | capture | extractor | dims | per image |
|---|---|---|---|---|
| image | camera frame, image files | hand-built: colour grid + hue histogram + oriented gradients + edge density | 116 | ~5 ms |
| image | same | **DINOv2-small pretrained** (optional) | 384 | ~2 s |
| audio | mic clip (with dB meter), audio files | log-mel band mean/std + 3-segment time profile | 200 | ~20 ms |

Pick the extractor from the *features* dropdown before training. DINOv2 is the
same idea as Teachable Machine's frozen MobileNet — much better at generalising
from a handful of real-world examples, at ~400x the cost per image. The runtime
and weights load from CDNs (one-time ~23 MB, then browser-cached), so the
deployed site stays ~174 KB.

Inference always uses the extractor recorded **in the model**, never the current
dropdown: the two vector spaces are incomparable, so a mismatch would produce
confident nonsense.

Labels are free-form strings, so a binary **yes / no** task is just two labels and
multi-class needs no different UI. Guardrails:

- **trainability** blocks <2 labels or <3 examples of any label, and warns above
  5:1 imbalance instead of training a model that predicts the majority class.
- **duplicate detection** flags samples with identical feature vectors and offers
  one-click removal. Duplicates inflate the score silently — one in both splits
  means testing on memorised data.
- **25% per label held out** with a deterministic stratified split. Integrity
  checks run only when the split is big enough to be meaningful; below that the
  UI says the score is on training data rather than implying otherwise.

Raw captures are stored alongside the cached vectors, so samples survive an
extractor change and you can review what you actually recorded.

### Why two image extractors

The shapes demo's 26-d v2 extractor thresholds for **one bright blob on a dark
field** and measures its radial profile — on a photograph there is no such blob,
so it describes noise. Measured on photo-like input: the photo extractor scores
**46.20** separation vs v2's **31.05**.

### Audio normalisation, and a bug worth recording

The first version subtracted each mel band's own mean over time. That forces
every band's time-average to **zero**, so the mean and per-segment features were
identically ~0 for any stationary sound — separation collapsed to **1.27**
(300 vs 1200 Hz) and **1.14** (tone vs noise), with all signal surviving only in
`std`. Subtracting a single **global** offset instead still cancels gain (log
domain ⇒ multiplicative gain is additive) while preserving spectral shape across
bands: **26.24** and **16.81**. Loudness invariance held — a 10× amplitude change
moves the vector 15.47 while a pitch change moves it 42.91.

## Versus Teachable Machine

Both are *frozen extractor + trained softmax head* — TM's head is
`dense → softmax` with categorical cross-entropy and Adam (checked in their
source); this uses multinomial logistic regression, the same model class.

**Still missing:** pose projects (TM uses PoseNet for 17 body keypoints);
TFLite / Keras / Coral export; cloud hosting with a shareable model URL;
tunable epochs / batch size / learning rate in the UI; and a pretrained audio
backbone (TM builds on `speech-commands`, this uses raw log-mel DSP).

**Added beyond TM:** actions on prediction — `onMatched` / `onNotMatched` with
independent console/webhook sinks, threshold and multi-target rules, per-branch
cooldown, and a persisted event log. TM stops at "here is the class". Plus the
integrity checks, duplicate detection, the trainability guard, raw-capture
storage, a headless CLI on the identical code path, and a ~1–15 KB model.

## Match rule

A prediction is a **match** when the predicted class is in `targets` **and**
confidence ≥ `threshold`. The two branches are configured independently, because
you usually want a webhook on match and console on no-match.

Webhook payload:

```json
{
  "event": "onMatched",
  "matched": true,
  "predicted": "triangle",
  "confidence": 0.9993,
  "threshold": 0.5,
  "targets": ["circle", "square", "triangle"],
  "probabilities": { "circle": 0, "square": 0.0007, "triangle": 0.9993 },
  "model": { "id": "mmt2mj9wbo1r3", "name": "v2 · …" },
  "ts": "2026-08-21T07:25:18.140Z"
}
```

`POST` sends JSON; `GET` puts the payload in a `?payload=` query param.
Cross-origin endpoints need CORS. **Webhook failures are logged, never thrown** —
a dead endpoint must not break the inference loop. An optional per-branch cooldown
suppresses rapid re-fires.

## Storage

IndexedDB (`tmjs` v1), two stores:

- `models` — scaler + head weights + metrics recorded at training time, so a
  loaded model is self-describing
- `events` — the match/no-match audit log with what was delivered

Models survive reloads and export to portable JSON. Action config persists in
`localStorage`.

## Integrity checks

Accuracy alone is not evidence, so every training run asserts:

- train/test keys disjoint; no duplicate vectors across splits
- test images are not near-copies of training images
- balanced test set
- 5-fold stratified CV on train agrees
- **shuffled labels collapse to chance** — the decisive one. With more features
  than samples, a head that memorised noise would still score high on permuted
  labels. Measured: 31.1% vs 33.3% chance.

The property suite also guards the dataset: **mean brightness must not identify
the class.** An earlier generator sized shapes by equal *radius*, but a circle
covers πr² and a triangle only ~1.3r², so brightness leaked the label. Shapes are
now sized for equal filled **area** — the test caught that bug.

## Layout

```
src/
  linalg.ts     StandardScaler, softmax regression (SGD), Pipeline,
                stratified k-fold, cross-val — replaces numpy + sklearn
  dataset.ts    seeded synthetic shapes; deterministic split/class/index keys
  features.ts   v1 (224-d) and v2 (26-d) frozen extractors
  verify.ts     integrity checks + rotation diagnostic
  store.ts      IndexedDB models + event log
  actions.ts    match rule and onMatched/onNotMatched dispatch
  serialize.ts  Pipeline <-> stored record <-> JSON
  canvas.ts     pure-JS raster shim so the dataset works headless in Node
  mvp.ts        the app
  cli.ts        headless pipeline
  server.ts     static files + resumable embedding cache
site/index.html the deployable static page
```

Pure TypeScript, `tsc --strict` clean. The same modules drive the CLI, the server
and the browser — no duplicated logic, so the CLI and the page produce identical
numbers.

## Optional: pretrained backbone

DINOv2-small via `transformers.js` reaches 100%, but costs ~2.1 s/image under WASM
and only runs in the browser here (`onnxruntime-node` is glibc-linked and fails on
musl with `__getauxval`). `web/cache_run.html` embeds resumably — each vector is
POSTed to `/cache/put` immediately, so an interruption costs at most one image —
then `node dist/cli.js --cache emb_cache.jsonl` trains the head in ~350 ms.

Note: most ImageNet *classifier* exports only expose `logits`, which are a poor
representation for out-of-distribution shapes (MobileNetV4 scored 51.7%). Only
models with **no classifier head** export usable features. Check
`session.outputNames` before assuming.
