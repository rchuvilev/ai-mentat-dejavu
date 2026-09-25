#!/usr/bin/env node
/** `npm run check` (first half) — build for the current system. */
'use strict';
const path = require('path');
const { build } = require('../sdk/logic/app-scripts');

build({ appName: 'ai-mentat-dejavu', root: path.resolve(__dirname, '..') });
