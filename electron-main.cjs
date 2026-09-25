/**
 * Electron wrapper for AI Déjà Vu.  (CommonJS — see the note below.)
 *
 * The UI in site/ is a plain static page, but it CANNOT be loaded over
 * file://: index.html uses `<script type="module">` importing ./dist/mvp.js,
 * and ES module imports are blocked by CORS on file:// origins. It also POSTs
 * embeddings to /cache/put while training.
 *
 * So this wrapper does the two things a fresh clone needs and nothing more:
 *
 *   1. `dist/` is NOT tracked in git, so a fresh clone has no compiled JS.
 *      Build it with the local tsc on first run (skipped when already built).
 *   2. Start src/server.ts's compiled server — the same static + cache server
 *      the browser flow already uses — on an ephemeral port, then point a
 *      BrowserWindow at it.
 *
 * CommonJS in an ESM package. package.json keeps `"type": "module"` because
 * src/ compiles to real ES modules, so this file carries the `.cjs` extension
 * — which is what lets it `require()` the CommonJS SDK. Nothing is imported
 * from src/ here: the compiled server is spawned as a child process, so the
 * conversion costs nothing.
 *
 * Still deliberately NO IPC surface and NO auto-update. This app has no
 * privileged host operations to expose, and the renderer is the same sandboxed
 * page that runs on the web build, so contextIsolation stays on and
 * nodeIntegration stays off. It takes the family's window factory and
 * packaging, and nothing else.
 */
const { app, BrowserWindow, shell } = require('electron');
const { spawn, spawnSync } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const net = require('node:net');
const { createWindow: createWindow_ } = require('./sdk/ui/window');

const ROOT = __dirname;
const SITE = path.join(ROOT, 'site');
const SERVER = path.join(ROOT, 'dist', 'server.js');

let mainWindow = null;
let serverProcess = null;
let serverUrl = null;

/** An OS-assigned free port, so two instances cannot collide on a fixed one. */
function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

/**
 * Compile TS -> dist/ when the pieces the page needs are missing.
 * Both files matter: server.js is what we spawn, site/dist/mvp.js is what
 * index.html imports. Checking only one would leave a half-built tree looking
 * ready.
 */
function ensureBuild() {
  const needed = [SERVER, path.join(SITE, 'dist', 'mvp.js')];
  if (needed.every((f) => fs.existsSync(f))) return true;

  const tsc = path.join(
    ROOT,
    'node_modules',
    '.bin',
    process.platform === 'win32' ? 'tsc.cmd' : 'tsc',
  );
  if (!fs.existsSync(tsc)) {
    console.error('[dejavu] tsc not found — run `npm install` in the app directory first.');
    return false;
  }

  console.log('[dejavu] building dist/ (first run)…');
  const built = spawnSync(tsc, [], { cwd: ROOT, stdio: 'inherit' });
  if (built.status !== 0) {
    console.error('[dejavu] tsc failed with status', built.status);
    return false;
  }

  // index.html imports ./dist/mvp.js RELATIVE TO site/, so the compiled
  // browser modules must also exist under site/dist/ (build-site.sh does this
  // copy for web deploys; replicate the copy only, not the deploy steps).
  const siteDist = path.join(SITE, 'dist');
  fs.mkdirSync(siteDist, { recursive: true });
  for (const f of fs.readdirSync(path.join(ROOT, 'dist'))) {
    if (f.endsWith('.js')) {
      fs.copyFileSync(path.join(ROOT, 'dist', f), path.join(siteDist, f));
    }
  }
  return needed.every((f) => fs.existsSync(f));
}

/** Spawn the static+cache server and resolve once it actually answers. */
async function startServer() {
  const port = await freePort();
  serverProcess = spawn(process.execPath, [SERVER, String(port), SITE], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    // ELECTRON_RUN_AS_NODE makes the Electron binary behave as plain node, so
    // the server child does not try to open its own window.
    env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
  });
  serverProcess.stdout.on('data', (d) => process.stdout.write(`[server] ${d}`));
  serverProcess.stderr.on('data', (d) => process.stderr.write(`[server] ${d}`));

  // Poll until it serves, rather than sleeping a hopeful fixed delay.
  const url = `http://127.0.0.1:${port}/`;
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(url, { method: 'HEAD' });
      if (res.ok || res.status === 404) return url;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`server did not start on ${url}`);
}

function createWindow(url) {
  // `load: { url }` rather than a file: this app points at the local static
  // server it just spawned, which is the reason the factory takes a union.
  // No preload — there is no IPC surface to expose.
  mainWindow = createWindow_({
    BrowserWindow,
    width: 1100,
    height: 860,
    backgroundColor: '#0d1117',
    title: 'AI Déjà Vu',
    load: { url },
    onReady: (win) => win.setMenuBarVisibility(false),
  });

  // External links open in the real browser, never inside the app window.
  mainWindow.webContents.setWindowOpenHandler(({ url: target }) => {
    shell.openExternal(target);
    return { action: 'deny' };
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function stopServer() {
  if (serverProcess && !serverProcess.killed) {
    serverProcess.kill();
    serverProcess = null;
  }
}

app.whenReady().then(async () => {
  if (!ensureBuild()) {
    app.quit();
    return;
  }
  try {
    serverUrl = await startServer();
    createWindow(serverUrl);
  } catch (err) {
    console.error('[dejavu]', err.message);
    app.quit();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0 && serverUrl) createWindow(serverUrl);
  });
});

app.on('window-all-closed', () => {
  stopServer();
  if (process.platform !== 'darwin') app.quit();
});
app.on('before-quit', stopServer);
process.on('exit', stopServer);
