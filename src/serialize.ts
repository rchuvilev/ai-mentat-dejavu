/**
 * Convert a trained Pipeline to/from a plain record.
 *
 * Kept separate from linalg.ts so the numeric core has no storage concerns, and
 * separate from store.ts so this is testable in Node where IndexedDB is absent.
 */
import { Pipeline, type Vec } from './linalg.js';
import type { StoredModel } from './store.js';

export function toRecord(
  pipe: Pipeline, opts: {
    id?: string; name: string; extractor: string; classes: string[];
    metrics: StoredModel['metrics'];
  },
): StoredModel {
  return {
    id: opts.id ?? `m${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
    name: opts.name,
    createdAt: Date.now(),
    extractor: opts.extractor,
    dim: pipe.clf.dim,
    classes: opts.classes,
    mu: pipe.scaler.mu,
    sd: pipe.scaler.sd,
    W: pipe.clf.W,
    b: pipe.clf.b,
    metrics: opts.metrics,
  };
}

/** Rebuild a usable Pipeline without retraining. */
export function fromRecord(m: StoredModel): Pipeline {
  const p = new Pipeline();
  p.scaler.mu = Float64Array.from(m.mu);
  p.scaler.sd = Float64Array.from(m.sd);
  p.clf.W = Float64Array.from(m.W);
  p.clf.b = Float64Array.from(m.b);
  p.clf.dim = m.dim;
  p.clf.k = m.classes.length;
  return p;
}

/** Portable JSON export (arrays instead of typed arrays). */
export function toJSON(m: StoredModel): string {
  return JSON.stringify({
    ...m,
    mu: Array.from(m.mu), sd: Array.from(m.sd),
    W: Array.from(m.W), b: Array.from(m.b),
  }, null, 2);
}

export function fromJSON(s: string): StoredModel {
  const o = JSON.parse(s);
  return {
    ...o,
    mu: Float64Array.from(o.mu), sd: Float64Array.from(o.sd),
    W: Float64Array.from(o.W), b: Float64Array.from(o.b),
  };
}

/** Predict with class names and probabilities attached. */
export function classify(pipe: Pipeline, classes: string[], v: Vec) {
  const probs = Array.from(pipe.predictProba(v));
  let b = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[b]) b = i;
  return { predicted: classes[b], confidence: probs[b], probs, index: b };
}
