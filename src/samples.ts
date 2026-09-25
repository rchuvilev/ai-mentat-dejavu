/**
 * User-recorded training samples, persisted in IndexedDB.
 *
 * Design notes:
 *  - The raw media Blob is kept alongside the feature vector. Storing only the
 *    vector would make the dataset unusable the moment an extractor changes, and
 *    the user could never review what they actually captured.
 *  - Vectors are cached so retraining does not re-decode every clip, but they
 *    are keyed by extractor name so switching extractor recomputes correctly.
 *  - Labels are free-form strings, not a fixed enum: the whole point is the user
 *    defines their own classes. A binary yes/no task is just two labels.
 */

export type Modality = 'image' | 'audio';

export interface Sample {
  id?: number;
  project: string;          // groups samples into a dataset
  modality: Modality;
  label: string;
  createdAt: number;
  source: string;           // 'camera' | 'upload' | 'mic'
  blob: Blob;               // the raw capture, so it can be re-extracted/reviewed
  thumb?: string;           // small dataURL preview (images only)
  durationMs?: number;      // audio only
  /** cached feature vectors, keyed by extractor name */
  vecs?: Record<string, number[]>;
}

const DB = 'tmjs';
/**
 * Kept only so the test suite can assert both modules agree. The runtime does
 * NOT pin it — see openAt() below. A stale cached module requesting an older
 * version than the live DB throws and kills the page, which happened live.
 */
const VERSION = 2;

function openAt(version?: number): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const req = version === undefined ? indexedDB.open(DB) : indexedDB.open(DB, version);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('models')) db.createObjectStore('models', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('events')) {
        const s = db.createObjectStore('events', { keyPath: 'id', autoIncrement: true });
        s.createIndex('byModel', 'modelId');
        s.createIndex('byTs', 'ts');
      }
      if (!db.objectStoreNames.contains('samples')) {
        const s = db.createObjectStore('samples', { keyPath: 'id', autoIncrement: true });
        s.createIndex('byProject', 'project');
        s.createIndex('byLabel', 'label');
      }
    };
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}

const NEEDED = ['models', 'events', 'samples'] as const;

async function open(): Promise<IDBDatabase> {
  let db = await openAt();
  const missing = NEEDED.filter(n => !db.objectStoreNames.contains(n));
  if (missing.length) {
    const next = db.version + 1;
    db.close();
    db = await openAt(next);
  }
  return db;
}

function tx<T>(store: string, mode: IDBTransactionMode,
               fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return open().then(db => new Promise<T>((res, rej) => {
    const t = db.transaction(store, mode);
    const r = fn(t.objectStore(store));
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
    t.oncomplete = () => db.close();
  }));
}

export const samples = {
  add(s: Sample): Promise<IDBValidKey> {
    return tx('samples', 'readwrite', st => st.add(s));
  },
  update(s: Sample): Promise<IDBValidKey> {
    return tx('samples', 'readwrite', st => st.put(s));
  },
  remove(id: number): Promise<undefined> {
    return tx('samples', 'readwrite', st => st.delete(id)) as Promise<undefined>;
  },
  async list(project: string, modality?: Modality): Promise<Sample[]> {
    const all = await tx<Sample[]>('samples', 'readonly', st => st.getAll());
    return all
      .filter(s => s.project === project && (!modality || s.modality === modality))
      .sort((a, b) => a.createdAt - b.createdAt);
  },
  async labels(project: string, modality?: Modality): Promise<Record<string, number>> {
    const rows = await this.list(project, modality);
    const out: Record<string, number> = {};
    for (const r of rows) out[r.label] = (out[r.label] ?? 0) + 1;
    return out;
  },
  async clearProject(project: string): Promise<number> {
    const rows = await this.list(project);
    for (const r of rows) if (r.id !== undefined) await this.remove(r.id);
    return rows.length;
  },
  async projects(): Promise<string[]> {
    const all = await tx<Sample[]>('samples', 'readonly', st => st.getAll());
    return [...new Set(all.map(s => s.project))].sort();
  },
};

/**
 * Is this label set trainable? Returns a human-readable reason if not.
 * Guards the common beginner failure: one class, or one example per class.
 */
export function trainability(counts: Record<string, number>): {
  ok: boolean; reason: string; labels: string[];
} {
  const labels = Object.keys(counts).sort();
  if (labels.length < 2) {
    return { ok: false, labels, reason: 'need at least 2 labels — record examples of each' };
  }
  const thin = labels.filter(l => counts[l] < 3);
  if (thin.length) {
    return {
      ok: false, labels,
      reason: `need >=3 examples per label; thin: ${thin.map(l => `${l}(${counts[l]})`).join(', ')}`,
    };
  }
  const n = labels.map(l => counts[l]);
  const imbalance = Math.max(...n) / Math.min(...n);
  if (imbalance > 5) {
    return {
      ok: true, labels,
      reason: `warning: imbalanced ${Math.max(...n)}:${Math.min(...n)} — the head will favour the larger class`,
    };
  }
  return { ok: true, labels, reason: `${labels.length} labels, ${n.reduce((a, b) => a + b, 0)} samples` };
}

/**
 * Find samples whose cached feature vectors are identical.
 *
 * Duplicates are a realistic user mistake — uploading the same file twice, or
 * holding the camera still while tapping capture repeatedly. They inflate the
 * score silently: if a duplicate lands in both the train and test split, the
 * model is being tested on data it memorised. The training integrity check
 * catches it, but surfacing it in the sample list lets the user fix the cause.
 */
export function findDuplicates(rows: Sample[], extractor: string): {
  groups: number[][]; duplicateCount: number;
} {
  const bySig = new Map<string, number[]>();
  for (const r of rows) {
    const v = r.vecs?.[extractor];
    if (!v || r.id === undefined) continue;
    const sig = v.slice(0, 16).map(x => x.toFixed(5)).join(',');
    if (!bySig.has(sig)) bySig.set(sig, []);
    bySig.get(sig)!.push(r.id);
  }
  const groups = [...bySig.values()].filter(g => g.length > 1);
  return { groups, duplicateCount: groups.reduce((n, g) => n + g.length - 1, 0) };
}

/** Remove all but the first of each duplicate group. Returns how many went. */
export async function dedupe(rows: Sample[], extractor: string): Promise<number> {
  const { groups } = findDuplicates(rows, extractor);
  let removed = 0;
  for (const g of groups) {
    for (const id of g.slice(1)) { await samples.remove(id); removed++; }
  }
  return removed;
}

/** Small dataURL preview so the sample grid does not hold full images in memory. */
export async function makeThumb(blob: Blob, size = 64): Promise<string> {
  const bmp = await createImageBitmap(blob);
  const cv = document.createElement('canvas');
  cv.width = size; cv.height = size;
  const cx = cv.getContext('2d')!;
  const scale = Math.max(size / bmp.width, size / bmp.height);
  const w = bmp.width * scale, h = bmp.height * scale;
  cx.drawImage(bmp, (size - w) / 2, (size - h) / 2, w, h);
  bmp.close?.();
  return cv.toDataURL('image/jpeg', 0.6);
}
