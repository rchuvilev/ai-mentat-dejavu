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
import { readFile, appendFile, open } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { join, extname, normalize } from 'node:path';

const ROOT = process.argv[3] ?? process.cwd();
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
  const rel = normalize(decodeURIComponent(url)).replace(/^(\.\.[/\\])+/, '');
  const path = join(ROOT, rel === '/' ? 'index.html' : rel);
  if (!path.startsWith(ROOT)) { res.writeHead(403); return res.end('forbidden'); }
  readFile(path).then(buf => {
    res.writeHead(200, {
      'Content-Type': MIME[extname(path)] ?? 'application/octet-stream',
      'Content-Length': buf.length,
    });
    res.end(buf);
  }).catch(() => { res.writeHead(404); res.end('not found'); });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`serving ${ROOT} on http://127.0.0.1:${PORT}`);
  console.log(`cache   ${CACHE} (${loadKeys().length} records)`);
});
