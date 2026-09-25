/**
 * IndexedDB persistence for trained models.
 *
 * Why IndexedDB and not localStorage: the head is Float64Array weights plus a
 * scaler, and localStorage is a ~5 MB string-only store. A 384-d 3-class head is
 * only ~9 KB, but the extractor config, thresholds and action settings belong
 * with it, and IndexedDB stores structured values without JSON round-tripping
 * every Float64Array.
 *
 * Schema (db `tmjs`, v1):
 *   models  keyPath 'id'   -- one record per saved model
 *   events  keyPath 'id' autoIncrement -- match/no-match audit log
 */

export interface StoredModel {
  id: string;
  name: string;
  createdAt: number;
  extractor: string;          // 'v1' | 'v2' | 'dinov2'
  dim: number;
  classes: string[];
  /** scaler */
  mu: Float64Array;
  sd: Float64Array;
  /** head */
  W: Float64Array;
  b: Float64Array;
  /** metrics recorded at training time, so a loaded model is self-describing */
  metrics: { train: number; test: number; ratio: number; nTrain: number; nTest: number };
}

export interface MatchEvent {
  id?: number;
  modelId: string;
  ts: number;
  predicted: string;
  confidence: number;
  matched: boolean;           // did it satisfy the target + threshold rule
  probs: number[];
  delivered: string;          // 'console' | 'webhook:200' | 'webhook:err' | 'none'
}

const DB = 'tmjs';
/**
 * MUST match samples.ts. Both modules open the same database, and IndexedDB
 * throws "the requested version (1) is less than the existing version (2)" if
 * one of them asks for an older version — which broke the MVP page for anyone
 * who had visited the studio first. Bump both together, and create every store
 * in both upgrade handlers so whichever module opens first leaves a complete DB.
 */
const VERSION = 2;

/**
 * Open WITHOUT pinning a version number.
 *
 * Pinning was fragile in practice: two modules share this database, and a
 * browser holding a stale cached copy of one module requests an older version
 * than the DB already has, which throws "the requested version (1) is less than
 * the existing version (2)" and kills the page. Observed live — a fresh fetch of
 * store.js reported VERSION 2 while the imported (cached) copy reported 1.
 *
 * Calling open() with no version opens whatever exists, so a stale module can
 * never downgrade-request. If a required store is missing we reopen at
 * version+1 and create it. Self-healing, and immune to module cache skew.
 */
function openAt(version?: number): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const req = version === undefined
      ? indexedDB.open(DB)
      : indexedDB.open(DB, version);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('models')) {
        db.createObjectStore('models', { keyPath: 'id' });
      }
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

/** Open, and if a store is missing, bump the version once to create it. */
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

export const store = {
  saveModel(m: StoredModel): Promise<IDBValidKey> {
    return tx('models', 'readwrite', s => s.put(m));
  },
  getModel(id: string): Promise<StoredModel | undefined> {
    return tx('models', 'readonly', s => s.get(id));
  },
  listModels(): Promise<StoredModel[]> {
    return tx<StoredModel[]>('models', 'readonly', s => s.getAll())
      .then(rows => rows.sort((a, b) => b.createdAt - a.createdAt));
  },
  deleteModel(id: string): Promise<undefined> {
    return tx('models', 'readwrite', s => s.delete(id)) as Promise<undefined>;
  },
  addEvent(e: MatchEvent): Promise<IDBValidKey> {
    return tx('events', 'readwrite', s => s.add(e));
  },
  listEvents(limit = 50): Promise<MatchEvent[]> {
    return tx<MatchEvent[]>('events', 'readonly', s => s.getAll())
      .then(rows => rows.sort((a, b) => b.ts - a.ts).slice(0, limit));
  },
  clearEvents(): Promise<undefined> {
    return tx('events', 'readwrite', s => s.clear()) as Promise<undefined>;
  },
  /** Rough byte estimate — useful to show the head really is small. */
  async usage(): Promise<{ models: number; events: number; bytes: number }> {
    const [ms, es] = await Promise.all([this.listModels(), this.listEvents(10000)]);
    const bytes = ms.reduce((n, m) =>
      n + (m.mu.length + m.sd.length + m.W.length + m.b.length) * 8, 0);
    return { models: ms.length, events: es.length, bytes };
  },
};

/** IndexedDB is unavailable in Node; callers can degrade gracefully. */
export const hasIDB = typeof indexedDB !== 'undefined';
