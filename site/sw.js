/**
 * AI Déjà Vu service worker.
 *
 * THE PROBLEM THIS SOLVES: Cloudflare Pages serves /dist/*.js with
 * `cache-control: public, max-age=14400`, so a browser holds compiled modules
 * for four hours. Repeatedly during development a deploy went live while the
 * page still imported the previous build — an extractor option was missing, and
 * once an IndexedDB version constant was stale enough to kill boot entirely.
 * A `?v=` on the HTML does not help: module URLs are unchanged, so the import
 * comes from cache regardless.
 *
 * DELIBERATELY DIFFERENT FROM THE GAME SW: dune2026 is cache-first, because a
 * game should load instantly and work offline. This app is NETWORK-FIRST for
 * its own code, because being one build behind is the failure mode we are
 * fixing. The cache is a fallback for offline, not the primary source.
 *
 * BUILD is stamped at deploy time. A new BUILD means a new cache name, and the
 * activate handler deletes every cache that is not the current one — so a stale
 * bundle cannot survive an update.
 */
const BUILD = '__BUILD__';
const CACHE = 'aidejavu-' + BUILD;

/** Shell files worth having offline. Model weights are NOT cached: they are
 *  1.6–23 MB and come from third-party CDNs with their own caching. */
const SHELL = [
  './',
  './index.html',
  './studio.html',
  './manifest.webmanifest',
];

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(CACHE);
    // Individual failures must not abort the install — a missing optional file
    // should not leave the app with no service worker at all.
    await Promise.all(SHELL.map(u => c.add(u).catch(() => {})));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter(n => n !== CACHE).map(n => caches.delete(n)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  // Only handle our own origin. Model weights from googleapis/jsdelivr/HF must
  // go straight to the network — intercepting them would mean caching tens of
  // megabytes we do not control.
  if (url.origin !== self.location.origin) return;

  e.respondWith((async () => {
    try {
      // NETWORK FIRST: always prefer the freshly deployed file.
      const fresh = await fetch(req, { cache: 'no-cache' });
      if (fresh && fresh.ok) {
        const c = await caches.open(CACHE);
        c.put(req, fresh.clone()).catch(() => {});
      }
      return fresh;
    } catch {
      // Offline (or the network failed): fall back to whatever we stored.
      const hit = await caches.match(req);
      if (hit) return hit;
      // Navigations should still land somewhere useful.
      if (req.mode === 'navigate') {
        const shell = await caches.match('./studio.html') || await caches.match('./index.html');
        if (shell) return shell;
      }
      throw new Error('offline and not cached');
    }
  })());
});

/** Let the page trigger an immediate takeover after an update. */
self.addEventListener('message', (e) => {
  if (e.data === 'skipWaiting') self.skipWaiting();
});
