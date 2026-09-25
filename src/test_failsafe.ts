// Tests for failsafe.ts — errors must be survivable AND traceable.
// Follows the existing suites' plain-node style: no framework, [PASS] lines.

import { quiet, quietAsync, attempt, recentFailures, clearFailures, setFailureSink, Failure } from './failsafe.js';

let failures = 0;
function ok(cond: boolean, name: string): void {
  if (cond) { console.log(`[PASS] ${name}`); }
  else { console.log(`[FAIL] ${name}`); failures++; }
}

// Capture instead of printing, so the suite output stays readable.
const captured: Failure[] = [];
setFailureSink((f) => { captured.push(f); });

function reset(): void { clearFailures(); captured.length = 0; }

// ── quiet ──────────────────────────────────────────────────────────────────
reset();
ok(quiet('t.ok', () => 42, -1) === 42, 'returns the value when nothing throws');
ok(recentFailures().length === 0, 'a success records nothing');

reset();
ok(quiet('t.bad', () => { throw new Error('boom'); }, -1) === -1,
   'returns the fallback when the operation throws');
ok(recentFailures().length === 1, 'the failure IS recorded, not swallowed');
ok(recentFailures()[0].op === 't.bad', 'the operation label is kept for grepping');
ok(recentFailures()[0].message === 'boom', 'the error message is kept');
ok(captured.length === 1, 'the failure reaches the sink, not just the buffer');

reset();
quiet('t.str', () => { throw 'plain string'; }, null);
// `throw 'string'` is legal JS; this used to log "undefined".
ok(recentFailures()[0].message === 'plain string', 'non-Error throws are handled');

reset();
quiet('t.ctx', () => { throw new Error('e'); }, null, { deviceId: 'cam0' });
ok(JSON.stringify(recentFailures()[0].context) === '{"deviceId":"cam0"}',
   'caller context survives for debugging');

// ── quietAsync ─────────────────────────────────────────────────────────────
async function asyncTests(): Promise<void> {
  reset();
  ok(await quietAsync('t.aok', async () => 'v', 'fb') === 'v',
     'async: resolves normally');

  reset();
  ok(await quietAsync('t.arej', async () => { throw new Error('nope'); }, 'fb') === 'fb',
     'async: returns the fallback on rejection');
  ok(recentFailures().length === 1, 'async: rejection is recorded');

  // An unhandled rejection in a browser event handler is invisible to the user
  // and kills the rest of that handler; this must never reject.
  reset();
  let threw = false;
  try { await quietAsync('t.safe', async () => { throw new Error('x'); }, 0); }
  catch { threw = true; }
  ok(!threw, 'async: never rejects, so an unawaited call is safe');
}

// ── attempt ────────────────────────────────────────────────────────────────
reset();
ok(attempt('t.side', () => { /* fine */ }) === true, 'attempt reports success');
ok(attempt('t.sidefail', () => { throw new Error('e'); }) === false, 'attempt reports failure');

// ── buffer discipline ──────────────────────────────────────────────────────
reset();
for (let i = 0; i < 130; i++) quiet('t.n', () => { throw new Error(String(i)); }, 0);
const buf = recentFailures();
ok(buf.length <= 100, 'buffer is bounded — this runs in a long-lived page');
ok(buf[buf.length - 1].message === '129',
   'buffer keeps the NEWEST entries, not the oldest');

reset();
quiet('t.copy', () => { throw new Error('e'); }, 0);
recentFailures().length = 0;
ok(recentFailures().length === 1, 'recentFailures returns a copy, not the live buffer');

asyncTests().then(() => {
  if (failures) { console.log(`\n${failures} FAILED`); process.exit(1); }
  console.log('\nALL PASS');
});
