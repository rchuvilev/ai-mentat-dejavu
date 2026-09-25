/**
 * Numeric primitives. Replaces the parts of numpy/sklearn this project used.
 *
 * Written from scratch because the Python side depended on sklearn's
 * StandardScaler + LogisticRegression + cross_val_score, none of which have a
 * JS equivalent worth pulling in for ~200 lines of maths.
 */

export type Vec = Float64Array;
export type Mat = Vec[];

export function norm2(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

export function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

export function std(xs: number[]): number {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
}

/** Deterministic RNG — identical stream in browser, CLI and tests. */
export function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffled<T>(xs: T[], rng: () => number): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Zero-mean unit-variance per feature. Mirrors sklearn StandardScaler. */
export class StandardScaler {
  mu: Vec = new Float64Array(0);
  sd: Vec = new Float64Array(0);

  fit(X: Mat): this {
    const d = X[0].length;
    this.mu = new Float64Array(d);
    this.sd = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let m = 0;
      for (const x of X) m += x[j];
      m /= X.length;
      let v = 0;
      for (const x of X) v += (x[j] - m) ** 2;
      this.mu[j] = m;
      // guard against constant features (sd 0 -> division by zero)
      this.sd[j] = Math.sqrt(v / X.length) + 1e-12;
    }
    return this;
  }

  transform(x: Vec): Vec {
    const o = new Float64Array(x.length);
    for (let j = 0; j < x.length; j++) o[j] = (x[j] - this.mu[j]) / this.sd[j];
    return o;
  }

  transformAll(X: Mat): Mat { return X.map(x => this.transform(x)); }
}

export interface LogRegOpts {
  epochs?: number;
  lr?: number;
  l2?: number;
  seed?: number;
}

/**
 * Multinomial logistic regression (softmax) trained by SGD.
 *
 * This is the same model class as sklearn's LogisticRegression with
 * multi_class='multinomial', and the same as Teachable Machine's trained dense
 * softmax head. sklearn uses lbfgs; SGD reaches the same optimum on these
 * well-separated problems, verified against the Python results.
 */
export class LogisticRegression {
  W: Vec = new Float64Array(0);
  b: Vec = new Float64Array(0);
  dim = 0;
  k = 0;
  private opts: Required<LogRegOpts>;

  constructor(opts: LogRegOpts = {}) {
    this.opts = {
      epochs: opts.epochs ?? 300,
      lr: opts.lr ?? 0.1,
      l2: opts.l2 ?? 1e-4,
      seed: opts.seed ?? 7,
    };
  }

  private softmax(z: Vec): Vec {
    const l = new Float64Array(this.k);
    for (let c = 0; c < this.k; c++) {
      let s = this.b[c];
      const off = c * this.dim;
      for (let j = 0; j < this.dim; j++) s += this.W[off + j] * z[j];
      l[c] = s;
    }
    let m = -Infinity;
    for (let c = 0; c < this.k; c++) if (l[c] > m) m = l[c];
    let tot = 0;
    for (let c = 0; c < this.k; c++) { l[c] = Math.exp(l[c] - m); tot += l[c]; }
    for (let c = 0; c < this.k; c++) l[c] /= tot;
    return l;
  }

  fit(Z: Mat, y: number[], nClasses?: number): this {
    this.dim = Z[0].length;
    this.k = nClasses ?? (Math.max(...y) + 1);
    this.W = new Float64Array(this.dim * this.k);
    this.b = new Float64Array(this.k);
    const rng = mulberry32(this.opts.seed);
    const idx = [...Z.keys()];
    for (let ep = 0; ep < this.opts.epochs; ep++) {
      for (const i of shuffled(idx, rng)) {
        const p = this.softmax(Z[i]);
        for (let c = 0; c < this.k; c++) {
          const g = p[c] - (y[i] === c ? 1 : 0);
          const off = c * this.dim;
          for (let j = 0; j < this.dim; j++) {
            this.W[off + j] -= this.opts.lr * (g * Z[i][j] + this.opts.l2 * this.W[off + j]);
          }
          this.b[c] -= this.opts.lr * g;
        }
      }
    }
    return this;
  }

  predictProba(z: Vec): Vec { return this.softmax(z); }

  predict(z: Vec): number {
    const p = this.softmax(z);
    let b = 0;
    for (let c = 1; c < p.length; c++) if (p[c] > p[b]) b = c;
    return b;
  }
}

/** Scaler + classifier, so callers cannot forget to scale. */
export class Pipeline {
  scaler = new StandardScaler();
  clf: LogisticRegression;

  constructor(opts: LogRegOpts = {}) { this.clf = new LogisticRegression(opts); }

  fit(X: Mat, y: number[], nClasses?: number): this {
    this.scaler.fit(X);
    this.clf.fit(this.scaler.transformAll(X), y, nClasses);
    return this;
  }

  predict(x: Vec): number { return this.clf.predict(this.scaler.transform(x)); }

  predictProba(x: Vec): Vec { return this.clf.predictProba(this.scaler.transform(x)); }

  score(X: Mat, y: number[]): number {
    let ok = 0;
    for (let i = 0; i < X.length; i++) if (this.predict(X[i]) === y[i]) ok++;
    return ok / X.length;
  }
}

export function confusionMatrix(yTrue: number[], yPred: number[], k: number): number[][] {
  const cm = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < yTrue.length; i++) cm[yTrue[i]][yPred[i]]++;
  return cm;
}

/** Stratified k-fold indices — keeps class balance in every fold. */
export function stratifiedFolds(y: number[], k: number, seed = 0): number[][] {
  const rng = mulberry32(seed);
  const byClass = new Map<number, number[]>();
  y.forEach((c, i) => {
    if (!byClass.has(c)) byClass.set(c, []);
    byClass.get(c)!.push(i);
  });
  const folds: number[][] = Array.from({ length: k }, () => []);
  for (const idxs of byClass.values()) {
    shuffled(idxs, rng).forEach((idx, n) => folds[n % k].push(idx));
  }
  return folds;
}

export function crossValScore(
  X: Mat, y: number[], k = 5, opts: LogRegOpts = {}, seed = 0, nClasses?: number,
): number[] {
  const folds = stratifiedFolds(y, k, seed);
  const kk = nClasses ?? (Math.max(...y) + 1);
  return folds.map(test => {
    const testSet = new Set(test);
    const trIdx = [...X.keys()].filter(i => !testSet.has(i));
    const p = new Pipeline(opts).fit(trIdx.map(i => X[i]), trIdx.map(i => y[i]), kk);
    return p.score(test.map(i => X[i]), test.map(i => y[i]));
  });
}
