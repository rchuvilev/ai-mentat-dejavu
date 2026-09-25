/**
 * Fail-safe execution helpers.
 *
 * WHY
 * ---
 * This app runs entirely in the browser, so a thrown error in a UI handler
 * silently stops that handler and leaves the page half-updated. Swallowing the
 * error keeps the app usable — which is the right call for an optional camera
 * preview or a malformed saved config — but a swallow with no record makes the
 * symptom ("the flip button vanished", "my settings did not load") impossible
 * to trace.
 *
 * These keep the resilience and add the record. Deliberately dependency-free
 * and side-effect-free at import time so they can be used from the browser
 * bundle, the CLI and the tests alike.
 */

export interface Failure {
  at: number;
  op: string;
  message: string;
  context?: unknown;
}

const recent: Failure[] = [];
const MAX_RECENT = 100;

// Injectable so tests can capture without spamming stderr, and so a future
// build can route this to a UI panel without touching call sites.
let sink: (f: Failure) => void = (f) => {
  console.warn(`[failsafe] ${f.op}: ${f.message}`, f.context ?? '');
};

export function setFailureSink(fn: (f: Failure) => void): (f: Failure) => void {
  const prev = sink;
  sink = fn;
  return prev;
}

export function recentFailures(): Failure[] {
  return recent.slice();
}

export function clearFailures(): void {
  recent.length = 0;
}

function record(op: string, err: unknown, context?: unknown): void {
  const message = err instanceof Error ? err.message : String(err);
  const f: Failure = { at: Date.now(), op, message, context };
  recent.push(f);
  // Trim the OLDEST: dropping the newest would discard the failure currently
  // being investigated.
  if (recent.length > MAX_RECENT) recent.splice(0, recent.length - MAX_RECENT);
  sink(f);
}

/**
 * Run a sync operation, returning `fallback` if it throws.
 * `op` must be a stable literal — it is the grep key in the logs.
 */
export function quiet<T>(op: string, fn: () => T, fallback: T, context?: unknown): T {
  try {
    return fn();
  } catch (err) {
    record(op, err, context);
    return fallback;
  }
}

/** Async form. Never rejects — safe to leave unawaited in an event handler. */
export async function quietAsync<T>(
  op: string,
  fn: () => Promise<T>,
  fallback: T,
  context?: unknown
): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    record(op, err, context);
    return fallback;
  }
}

/** Fire-and-forget side effect. Returns whether it succeeded. */
export function attempt(op: string, fn: () => void, context?: unknown): boolean {
  try {
    fn();
    return true;
  } catch (err) {
    record(op, err, context);
    return false;
  }
}
