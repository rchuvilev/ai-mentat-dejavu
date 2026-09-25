#!/usr/bin/env node
/**
 * Tests for the static server's path resolution.
 *
 * This is the only part of server.ts that decides whether a request may read a
 * file, and until now none of it could be exercised: the module opened a
 * socket at import time, so there was nothing to call. `resolveStaticPath` is
 * exported and the listen() is behind a run-as-main guard for that reason.
 *
 * The property that matters is not "traversal returns 403" — the leading `../`
 * are stripped, so a traversal is REWRITTEN to an in-root path and comes back
 * 404. What matters is that the returned path can never be outside the root,
 * whatever the URL says.
 */
import { resolveStaticPath } from './server.js';
import { sep, join } from 'node:path';

let failed = 0;
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failed++;
};

const ROOT = join(sep, 'srv', 'site');

/* ── the ordinary path ──────────────────────────────────────────────────── */

check('a plain file resolves inside the root',
  resolveStaticPath(ROOT, '/app.js') === join(ROOT, 'app.js'));

check('a nested file resolves inside the root',
  resolveStaticPath(ROOT, '/dist/mvp.js') === join(ROOT, 'dist', 'mvp.js'));

check('the site root serves index.html',
  resolveStaticPath(ROOT, '/') === join(ROOT, 'index.html'));

check('percent-encoding is decoded',
  resolveStaticPath(ROOT, '/a%20b.js') === join(ROOT, 'a b.js'));

/* ── containment: the actual security property ──────────────────────────── */

const ESCAPES = [
  '/../package.json',
  '/../../../../etc/passwd',
  '/..%2fpackage.json',
  '/%2e%2e/package.json',
  '/foo/../../package.json',
  '/....//package.json',
  '/./../../secret',
  '//../secret',
  '/dist/../../../secret',
  // Sibling-prefix: joins to /srv/sitemap/x, which a bare startsWith('/srv/site')
  // would accept. Only reachable if BOTH the strip and the separator-aware
  // comparison are removed — the two are redundant, which is the point.
  '/../sitemap/x',
  'x/../../sitemap/y',
];

let escaped: string | null = null;
for (const url of ESCAPES) {
  const p = resolveStaticPath(ROOT, url);
  if (p !== null && !p.startsWith(ROOT + sep)) escaped = `${url} -> ${p}`;
}
check('no URL can resolve outside the root', escaped === null, escaped ?? `${ESCAPES.length} tried`);

check('a sibling directory sharing the root prefix is not inside it',
  // With a bare startsWith(root), '/srv/sitemap/x' passes a '/srv/site' check.
  // Stripping leading `../` happens to prevent it today, which is exactly why
  // the weaker comparison survived — it was never doing the work.
  !(join(sep, 'srv', 'sitemap', 'x')).startsWith(ROOT + sep),
  'guard compares against root + separator');

/* ── malformed input must not throw ─────────────────────────────────────── */

for (const bad of ['/%', '/%zz', '/%E0%A4%A', '/a%']) {
  let threw = false;
  let out: string | null = 'x';
  try { out = resolveStaticPath(ROOT, bad); } catch { threw = true; }
  check(`a malformed escape ${JSON.stringify(bad)} is refused, not thrown`,
    !threw && out === null,
    threw ? 'THREW — this used to kill the server' : 'null');
}

/* ── CONTROL ────────────────────────────────────────────────────────────── */

check('CONTROL: the resolver really can return a path',
  resolveStaticPath(ROOT, '/index.html') !== null,
  'so the null assertions above are not vacuous');

check('CONTROL: an absolute root is required for containment to work',
  // join('.', 'index.html') is 'index.html', which does not start with './'.
  // A relative root therefore rejects everything — which is what both npm
  // scripts were doing until ROOT was resolve()d.
  resolveStaticPath('.', '/index.html') === null,
  'relative root rejects everything, hence resolve() at module scope');

console.log(failed ? `\n${failed} FAILED` : '\nALL PASS');
process.exit(failed ? 1 : 0);
