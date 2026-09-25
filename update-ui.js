/**
 * Update notification bar.
 *
 * Extracted verbatim (behaviour-wise) from the licensing overlay that used to
 * own it. Licensing was removed from these apps, and the update bar happened to
 * live inside `licensing-ui.js` — deleting that file without this extraction
 * would have silently dropped the "Restart to update" prompt, which has nothing
 * to do with licensing.
 *
 * Loaded as a plain <script> from app.html, same as its predecessor.
 */
(function () {
  'use strict';

  function setupUpdateBar() {
    // The bridge is optional: a build without auto-update wiring simply has no
    // bar rather than throwing at load.
    if (!window.electronAPI || !window.electronAPI.onUpdateDownloaded) return;

    window.electronAPI.onUpdateDownloaded((info) => {
      const existing = document.getElementById('update-bar');
      if (existing) existing.remove();

      const version = info && info.version ? ' v' + info.version : '';
      const bar = document.createElement('div');
      bar.id = 'update-bar';
      bar.style.cssText =
        'position:fixed;bottom:0;left:0;right:0;z-index:99998;background:#2d5a3d;' +
        'padding:8px 16px;text-align:center;font-size:13px;color:#fff;' +
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;';
      bar.innerHTML =
        'Update' + version + ' ready! <button id="update-bar-btn" style="margin-left:12px;' +
        'padding:4px 16px;border:none;border-radius:4px;background:#4a6cf7;color:#fff;' +
        'cursor:pointer;font-size:13px">Restart to update</button>';
      document.body.appendChild(bar);

      // Listener rather than an inline onclick handler: inline handlers need a
      // global, and this file deliberately exports nothing.
      const btn = document.getElementById('update-bar-btn');
      if (btn) btn.addEventListener('click', () => window.electronAPI.installUpdate());
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupUpdateBar);
  } else {
    setupUpdateBar();
  }
})();
