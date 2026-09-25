#!/usr/bin/env node
/**
 * Static file server + resumable embedding cache. Replaces cache_server.py.
 *
 * Why the cache exists: DINOv2 under WASM is slow enough that a full run may not
 * fit in one session. Each embedding is POSTed the moment it is computed and
 * appended to a JSONL file, so a reload or interruption costs at most one image.
 *
 *   GET  /cache/status  -> { count, keys }
 *   POST /cache/put     -> { key, label, split, vec }
 *   GET  /cache/dump    -> all records
 *   GET  /*             -> static files
 */
import { createServer } from 'node:http';
import { pathToFileURL } from 'node:url';
import { readFile, appendFile, open } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { join, extname, normalize, sep, resolve } from 'node:path';

// Resolved, not taken as given. The containment check compares the joined
// path against ROOT, and join('.', 'index.html') is 'index.html', which does
// not start with '.' — so a RELATIVE root rejected every request with 403.
// Both npm scripts pass one ('serve' passes '.', 'site' passes 'site'), so the
// static server has never served a file from either.
const ROOT = resolve(process.argv[3] ?? process.cwd());
const PORT = +(process.argv[2] ?? 8772);
const CACHE = join(ROOT, 'emb_cache.jsonl');

const MIME: Record<string, string> = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.css': 'text/css',
  '.png': 'image/png', '.onnx': 'application/octet-stream', '.map': 'application/json',
};

/** Keys currently stored. The log is append-only, so dedup happens on read. */
function loadKeys(): string[] {
  if (!existsSync(CACHE)) return [];
  const keys = new Set<string>();
  for (const line of readFileSync(CACHE, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { keys.add(JSON.parse(line).key); } catch { /* skip partial line */ }
  }
  return [...keys].sort();
}

// Serialise appends: concurrent POSTs could otherwise interleave partial lines.
let writeChain: Promise<unknown> = Promise.resolve();

/**
 * Resolve a request URL to a file inside `root`, or null when it escapes.
 *
 * Exported so the traversal defence is testable: the server itself listens at
 * module scope, so nothing in this file could be exercised without opening a
 * socket.
 *
 * Two things here are load-bearing:
 *
 *  1. decodeURIComponent THROWS on a malformed escape — `GET /%` is enough.
 *     Inside the request handler that was an uncaught exception, so any client
 *     could stop the server with one request. A bad escape is now just a 403.
 *
 *  2. The containment check compares against `root + sep`, not `root`. A bare
 *     startsWith(root) also accepts a SIBLING whose name merely begins with it
 *     — with root `/srv/site`, the path `/srv/sitemap/secret` passes. Stripping
 *     leading `../` happens to prevent that today, which is exactly why the
 *     weaker check survived: it was never the thing doing the work.
 */
export function resolveStaticPath(root: string, url: string): string | null {
  let decoded: string;
  try { decoded = decodeURIComponent(url); } catch { return null; }
  const rel = normalize(decoded).replace(/^(\.\.[/\\])+/, '');
  const path = join(root, rel === '/' ? 'index.html' : rel);
  const base = root.endsWith(sep) ? root : root + sep;
  return path.startsWith(base) ? path : null;
}

const server = createServer((req, res) => {
  const url = (req.url ?? '/').split('?')[0];
  const json = (obj: unknown, code = 200) => {
    const b = Buffer.from(JSON.stringify(obj));
    res.writeHead(code, {
      'Content-Type': 'application/json',
      'Content-Length': b.length,
      'Access-Control-Allow-Origin': '*',
    });
    res.end(b);
  };

  if (req.method === 'GET' && url === '/cache/status') {
    const keys = loadKeys();
    return json({ count: keys.length, keys });
  }

  if (req.method === 'GET' && url === '/cache/dump') {
    if (!existsSync(CACHE)) return json([]);
    const rows = readFileSync(CACHE, 'utf8').split('\n')
      .filter(l => l.trim()).map(l => JSON.parse(l));
    return json(rows);
  }

  if (req.method === 'POST' && url === '/cache/put') {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      let rec: any;
      try { rec = JSON.parse(body); } catch (e) { return json({ error: 'bad json' }, 400); }
      for (const k of ['key', 'label', 'vec']) {
        if (!(k in rec)) return json({ error: `missing ${k}` }, 400);
      }
      const line = JSON.stringify({
        key: rec.key, label: rec.label, split: rec.split ?? 'train', vec: rec.vec,
      }) + '\n';
      // fsync each record: an interrupted process must not lose completed work
      writeChain = writeChain.then(async () => {
        const fh = await open(CACHE, 'a');
        try { await fh.writeFile(line); await fh.sync(); } finally { await fh.close(); }
      }).then(() => json({ ok: true, count: loadKeys().length }),
              (e) => json({ error: String(e) }, 500));
    });
    return;
  }

  // ---- static files, with traversal guard
  const path = resolveStaticPath(ROOT, url);
  if (path === null) { res.writeHead(403); return res.end('forbidden'); }
  readFile(path).then(buf => {
    res.writeHead(200, {
      'Content-Type': MIME[extname(path)] ?? 'application/octet-stream',
      'Content-Length': buf.length,
    });
    res.end(buf);
  }).catch(() => { res.writeHead(404); res.end('not found'); });
});

// Only listen when run as a script, so the helpers above are importable — and
// therefore testable — without opening a socket. Same rule as the SDK's
// bundle-electron.js.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  server.listen(PORT, '127.0.0.1', () => {
    console.log(`serving ${ROOT} on http://127.0.0.1:${PORT}`);
    console.log(`cache   ${CACHE} (${loadKeys().length} records)`);
  });
}
