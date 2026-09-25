// REPL driver for the ai-mentat Electron apps (macOS).
//
// Adapted from the `run` skill's Electron skeleton. Differences that matter
// here: macOS has a real window server, so no xvfb and no --no-sandbox; and
// the Electron binary lives inside Electron.app/Contents/MacOS/.
//
// Usage: node /tmp/mentat-driver.mjs <app-dir>
// Designed for tmux: send-keys commands, capture-pane output.

import { _electron as electron } from 'playwright-core';
import * as readline from 'node:readline';
import * as fs from 'node:fs';
import * as path from 'node:path';

const APP_DIR = path.resolve(process.argv[2] || process.cwd());
const SHOT_DIR = process.env.SCREENSHOT_DIR || '/tmp/shots';
fs.mkdirSync(SHOT_DIR, { recursive: true });

let app = null;
let page = null;
const mainLog = [];

const electronBin = process.platform === 'darwin'
  ? path.join(APP_DIR, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(APP_DIR, 'node_modules/electron/dist/electron');

const COMMANDS = {
  async launch() {
    if (app) return console.log('already launched');
    if (!fs.existsSync(electronBin)) return console.log('ERROR: no electron binary at', electronBin);
    // The app's `main` is the esbuild bundle, so it must exist before launch.
    const bundle = ['electron-main.bundle.js','electron-main.bundle.cjs']
      .map((n) => path.join(APP_DIR, n)).find((p) => fs.existsSync(p)) || path.join(APP_DIR, 'electron-main.bundle.js');
    if (!fs.existsSync(bundle)) return console.log('ERROR: run `npm run bundle` first — no', bundle);

    app = await electron.launch({
      executablePath: electronBin,
      args: [APP_DIR],
      cwd: APP_DIR,
      timeout: 60_000,
    });
    // Capture main-process stdout/stderr: for these apps that is where the
    // real diagnostics go (n8n boot, Lima probes, failsafe records).
    app.process().stdout.on('data', (d) => mainLog.push(String(d)));
    app.process().stderr.on('data', (d) => mainLog.push(String(d)));

    page = await app.firstWindow({ timeout: 60_000 });
    await page.waitForLoadState('domcontentloaded').catch(() => {});
    console.log('launched.', app.windows().length, 'window(s):');
    for (const w of app.windows()) console.log('  ', w.url());
  },

  async ss(name) {
    if (!page) return console.log('ERROR: launch first');
    const f = path.join(SHOT_DIR, (name || `ss-${Date.now()}`) + '.png');
    await page.screenshot({ path: f });
    console.log('screenshot:', f);
  },

  // DOM click, not locator.click(): coordinate math is unreliable when
  // content sits under an overlay, and these apps use loading overlays.
  async click(sel) {
    if (!page) return console.log('ERROR: launch first');
    const r = await page.evaluate((s) => {
      const el = document.querySelector(s);
      if (!el) return 'NOT_FOUND';
      el.click();
      return 'OK';
    }, sel);
    console.log('click', sel, '->', r);
  },

  async tab(name) {
    if (!page) return console.log('ERROR: launch first');
    const r = await page.evaluate((t) => {
      const btn = [...document.querySelectorAll('.tab-bar button')].find((b) => b.dataset.tab === t);
      if (!btn) return 'NOT_FOUND';
      btn.click();
      const panel = document.getElementById('tab-' + t);
      return `OK active=${!!panel?.classList.contains('active')}`;
    }, name);
    console.log('tab', name, '->', r);
  },

  async fill(arg) {
    if (!page) return console.log('ERROR: launch first');
    const [sel, ...rest] = arg.split(/\s+/);
    const value = rest.join(' ');
    const r = await page.evaluate(([s, v]) => {
      const el = document.querySelector(s);
      if (!el) return 'NOT_FOUND';
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      return 'OK';
    }, [sel, value]);
    console.log('fill', sel, '->', r);
  },

  async text(sel) {
    if (!page) return console.log('ERROR: launch first');
    const out = await page.evaluate(
      (s) => (s ? document.querySelector(s) : document.body)?.innerText ?? '(null)',
      sel || null);
    console.log(out);
  },

  async eval(expr) {
    if (!page) return console.log('ERROR: launch first');
    try { console.log(JSON.stringify(await page.evaluate(expr))); }
    catch (e) { console.log('ERROR:', e.message); }
  },

  /** Console messages and page errors — a broken renderer shows up here. */
  async errors() {
    console.log(JSON.stringify(pageErrors, null, 1));
  },

  /** Main-process output. */
  async mainlog() {
    console.log(mainLog.join('') || '(empty)');
  },

  async windows() {
    if (!app) return console.log('ERROR: launch first');
    for (const w of app.windows()) console.log('  ', w.url());
  },

  async quit() {
    if (app) await app.close().catch(() => {});
    app = null; page = null;
  },
  help() { console.log('commands:', Object.keys(COMMANDS).join(', ')); },
};

const pageErrors = [];

// Attach renderer diagnostics as soon as a page exists.
const origLaunch = COMMANDS.launch;
COMMANDS.launch = async function launch() {
  const r = await origLaunch.call(COMMANDS);
  if (page) {
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning') pageErrors.push(`[${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => pageErrors.push(`[pageerror] ${e.message}`));
  }
  return r;
};

// Electron steals stdin; use the raw fd so the REPL keeps its input.
const stdin = fs.createReadStream(null, { fd: fs.openSync('/dev/stdin', 'r') });
const rl = readline.createInterface({ input: stdin, output: process.stdout, prompt: 'driver> ' });

rl.on('line', async (line) => {
  const [cmd, ...rest] = line.trim().split(/\s+/);
  if (!cmd) return rl.prompt();
  const fn = COMMANDS[cmd];
  if (!fn) { console.log('unknown:', cmd, '- try: help'); return rl.prompt(); }
  try { await fn(rest.join(' ')); } catch (e) { console.log('ERROR:', e.message); }
  if (cmd === 'quit') { rl.close(); process.exit(0); }
  rl.prompt();
});
rl.on('close', async () => { await COMMANDS.quit(); process.exit(0); });

console.log(`mentat driver [${path.basename(APP_DIR)}] - "help" for commands, "launch" to start`);
rl.prompt();
