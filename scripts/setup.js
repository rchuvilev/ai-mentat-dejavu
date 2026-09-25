#!/usr/bin/env node
/** `npm run setup` — install every npm and non-npm dependency needed to run. */
'use strict';
const path = require('path');
const { setup } = require('../sdk/logic/app-scripts');

setup({
  appName: 'ai-mentat-dejavu',
  root: path.resolve(__dirname, '..'),
  system: [],
  extra: () => {
    // The renderer imports compiled ES modules, so a fresh clone needs tsc.
    try { require('child_process').execSync('npm run build', { stdio: 'inherit' }); }
    catch { console.warn('    (tsc build skipped)'); }
  },
});
