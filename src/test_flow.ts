#!/usr/bin/env node
/**
 * User-flow regression tests.
 *
 * These are static/source checks, because the DOM behaviour is verified live in
 * the browser. What they pin down are the flow bugs actually found by clicking
 * through the Studio:
 *
 *  1. "Capture frame", "Record sample" and continuous mode were ENABLED with no
 *     camera / no mic / no model, so the UI invited an action then reported an
 *     error instead of preventing it.
 *  2. Loading a shapes-demo model (v1/v2) into the Studio "succeeded", enabled
 *     Classify, then failed with a misleading message — and embedFor() would
 *     have silently returned photo features for a 26-d head.
 *  3. Record reported "pick a label first" when the real blocker was no mic.
 *  4. Every id the script touches must exist in the page it belongs to,
 *     otherwise boot() dies on a null dereference.
 */
import { readFileSync } from 'node:fs';

let failed = 0;
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failed++;
};
const read = (rel: string) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const studioTs = read('../src/studio.ts');
const studioHtml = read('../site/studio.html');
const mvpTs = read('../src/mvp.ts');
const mvpHtml = read('../site/index.html');
const metersTs = read('../src/meters.ts');
const swJs = read('../site/sw.js');
const manifest = read('../site/manifest.webmanifest');

// ---------------------------------------------------------------- 4. no null $()
function idsUsed(src: string): string[] {
  const out = new Set<string>();
  for (const m of src.matchAll(/\$(?:<[^>]+>)?\('([A-Za-z0-9_]+)'\)/g)) out.add(m[1]);
  for (const m of src.matchAll(/getElementById\('([A-Za-z0-9_]+)'\)/g)) out.add(m[1]);
  return [...out];
}
function idsDefined(html: string): Set<string> {
  const out = new Set<string>();
  for (const m of html.matchAll(/id="([A-Za-z0-9_]+)"/g)) out.add(m[1]);
  return out;
}
{
  // Ids the script INJECTS at runtime are legitimate even though the static HTML
  // has no such element (e.g. the dedupe button inside the trainability line).
  const injected = idsDefined(studioTs);
  const have = new Set([...idsDefined(studioHtml), ...injected]);
  const missing = idsUsed(studioTs).filter(i => !have.has(i));
  check('studio.ts touches only ids present in studio.html or injected by it',
        missing.length === 0, missing.length ? `missing: ${missing.join(', ')}` : 'all resolve');

  const haveMvp = new Set([...idsDefined(mvpHtml), ...idsDefined(mvpTs)]);
  const missingMvp = idsUsed(mvpTs).filter(i => !haveMvp.has(i));
  check('mvp.ts touches only ids present in index.html',
        missingMvp.length === 0, missingMvp.length ? `missing: ${missingMvp.join(', ')}` : 'all resolve');
}

// ---------------------------------------------------------------- 1. affordance
{
  check('a central syncControls() governs button state',
        studioTs.includes('function syncControls()'));
  for (const id of ['shoot', 'rec', 'runBtn', 'loop']) {
    check(`${id} is disabled until usable`,
          new RegExp(`\\$<HTMLButtonElement>\\('${id}'\\)\\.disabled\\s*=`).test(studioTs),
          'was enabled with no device/model');
  }
  check('syncControls runs on label input',
        studioTs.includes("$('label').addEventListener('input', syncControls)"),
        'typing a label must enable capture without a reload');
  check('syncControls runs after capture start/stop',
        (studioTs.match(/syncControls\(\);/g) ?? []).length >= 5);
  check('a hint explains why capture is blocked',
        studioTs.includes("$('captureHint')") && studioHtml.includes('id="captureHint"'));
  check('hint lives outside the modality panels',
        studioHtml.indexOf('id="captureHint"') > studioHtml.indexOf('id="audTools"'),
        'otherwise it disappears when audio is selected');
}

// ---------------------------------------------------------------- 2. compatibility
{
  check('a compatibility() gate exists', studioTs.includes('function compatibility('));
  check('embedFor refuses unknown extractors',
        /cannot compute .*features/.test(studioTs) && studioTs.includes('throw new Error'),
        'must not fall through to embedPhoto for a v1/v2 head');
  check('load refuses an incompatible model',
        studioTs.includes('cannot load:'),
        'shapes-demo models share the same IndexedDB');
  check('incompatible models are flagged in the dropdown',
        studioTs.includes('not usable here'));
  check('loading aligns modality with the model',
        studioTs.includes('modality !== compat.reason'),
        'so Classify uses the right capture widget');
}

// ---------------------------------------------------------------- 3. error order
{
  const recBlock = studioTs.slice(studioTs.indexOf("$('rec').addEventListener"));
  const micIdx = recBlock.indexOf('microphone on first');
  const labelIdx = recBlock.indexOf('pick or type a label first');
  check('record reports the missing device before the missing label',
        micIdx > -1 && micIdx < labelIdx,
        'the real blocker must be named first');

  const shootBlock = studioTs.slice(studioTs.indexOf("$('shoot').addEventListener"));
  check('capture reports no-camera before no-label',
        shootBlock.indexOf('camera on first') > -1 &&
        shootBlock.indexOf('camera on first') < shootBlock.indexOf('label first'));
  check('capture distinguishes "not started" from "still starting"',
        shootBlock.includes('still starting'),
        'a warming-up camera is not the same error as no camera');
}

// ---------------------------------------------------------------- mobile
{
  // Camera and mic are commonly DENIED on mobile / in embedded WebViews
  // (measured NotAllowedError for both), so upload is the essential path and
  // the failure must be actionable rather than a raw DOMException message.
  check('device errors are translated for humans',
        studioTs.includes('function deviceError(') &&
        studioTs.includes('NotAllowedError') &&
        studioTs.includes('use the upload button'),
        'raw "Permission denied" gives the user nothing to do');
  check('a failed camera start re-syncs controls',
        /camera unavailable[\s\S]{0,120}syncControls\(\)/.test(studioTs),
        'Capture stayed enabled after the camera failed');
  check('a failed mic start re-syncs controls',
        /microphone unavailable[\s\S]{0,120}syncControls\(\)/.test(studioTs));
  check('adding a sample re-syncs controls',
        /await refreshSamples\(\);\s*\n\s*syncControls\(\);/.test(studioTs),
        'the hint went stale/empty after an upload');
  check('image input offers the phone camera',
        /id="imgFile"[^>]*capture="environment"/.test(studioHtml),
        'the upload fallback should reach the camera when getUserMedia cannot');
  check('capture button sits under the video frame',
        studioHtml.indexOf('id="shoot"') > studioHtml.indexOf('id="cam"') &&
        studioHtml.indexOf('id="shoot"') < studioHtml.indexOf('</div>', studioHtml.indexOf('id="cam"')) + 400,
        'aim then tap, instead of a button above the preview');
  check('real controls meet a mobile tap target',
        /\.row button[^{]*\{min-height:44px\}/.test(studioHtml),
        'all controls measured 27-35px tall');
  check('tap-target rule is scoped, not every button',
        !/(^|\n)button\{min-height:44px\}/.test(studioHtml),
        'min-height always beats height, so a blanket rule inflated the ' +
        '22px thumbnail delete badge into a 44px oval over the preview');
}

// ---------------------------------------------------------------- meters
{
  check('meters live in ONE shared module',
        metersTs.includes('export function renderVerdict('),
        'both pages must use the same implementation');
  check('studio imports the shared meters',
        studioTs.includes("from './meters.js'") &&
        !studioTs.includes('function renderVerdict('),
        'no local copy left behind');
  check('the shapes demo also renders meters',
        mvpTs.includes("from './meters.js'") && mvpTs.includes('renderVerdict('),
        'the demo previously showed only a verdict line');
  check('meter nodes are REUSED so widths animate',
        metersTs.includes("bars.dataset.classes !== classes.join('|')") &&
        metersTs.includes('fill.style.width'),
        'rebuilding innerHTML restarts the element and kills the transition');
  for (const [name, html] of [['studio', studioHtml], ['demo', mvpHtml]] as const) {
    check(`${name} page has the meter CSS`,
          /\.bfill\{[^}]*transition:width/.test(html) &&
          html.includes('.bfill.win') && html.includes('.bfill.hit'),
          'markup without the styles renders as unstyled rows');
  }
  check('threshold marker drawn on target classes only',
        metersTs.includes('thr.style.left') && metersTs.includes("isTarget ? '' : 'none'"));
  check('percentages shown to one decimal',
        metersTs.includes('(p * 100).toFixed(1)'));
  check('studio-link CTA is readable, not a bare dark-on-dark link',
        mvpHtml.includes('class="cta"') && /\.cta\{[^}]*background:var\(--blue\)/.test(mvpHtml),
        'measured unreadable: default link blue on a dark card');
  check('demo controls meet the tap target too',
        /#classify[^{]*\{min-height:44px\}|\.row button[^{]*\{min-height:44px\}/.test(mvpHtml));
}

// ---------------------------------------------------------------- camera + ux
{
  check('front/back camera switch exists',
        studioTs.includes('function flipCamera(') && studioHtml.includes('id="flip"'));
  check('flip is hidden/disabled without a stream',
        studioTs.includes("flip.disabled = !hasStream") &&
        studioTs.includes("if (!hasStream) flip.style.display = 'none'"),
        'it overlays the video, so it must not be usable with no camera');
  check('facingMode is ideal, not exact, with a bare-video fallback',
        !studioTs.includes("facingMode: { exact:") &&
        studioTs.includes("{ video: true, audio: false }"),
        'exact facingMode fails outright on a single-camera device');
  check('front preview is mirrored but the saved frame is not',
        studioTs.includes("scaleX(-1)") && studioTs.includes('function unmirror('),
        'users expect a mirror; the stored sample should match the real scene');
  check('capture waits for a decodable frame',
        studioTs.includes('v.readyState < 2'),
        'videoWidth can be set before the first frame, giving blank captures');
  check('capture rejects an empty blob',
        studioTs.includes('blob.size < 256'));
  check('clearing a project asks for confirmation',
        studioTs.includes('cannot be undone'),
        'samples are unrecoverable');
  check('training scrolls its result into view',
        studioTs.includes("scrollIntoView"),
        'metrics render ~800px below the button on a phone');
  check('status line does not duplicate existing state UI',
        studioTs.includes('function syncSteps()') &&
        !/n: 1, ok: nSamples/.test(studioTs),
        'captureHint and trainability already cover steps 1-2');
  check('empty event log says what to do',
        studioTs.includes('classify something in step 5'));
  for (const id of ['modelSel', 'thr', 'hookUrl']) {
    check(`${id} has an accessible name`,
          new RegExp(`id="${id}"[^>]*aria-label=`).test(studioHtml));
  }
}

// ---------------------------------------------------------------- live loop
{
  // setInterval with an async callback queues another run before the previous
  // finishes. With a slow extractor (TM CPU ~1188 ms, DINOv2 ~2000 ms) the
  // backlog grows without bound and every action describes an older frame.
  // The dB meter legitimately uses setInterval — it is a synchronous 100 ms UI
  // tick that cannot overlap. What must never use it is an ASYNC callback.
  check('no async callback on a fixed interval',
        !/setInterval\(\s*async/.test(studioTs),
        'async work on a fixed interval queues up and drifts behind reality');
  check('the loop is self-scheduling',
        /while \(!stop\)/.test(studioTs) &&
        /period - elapsed/.test(studioTs),
        'wait for each pass, then sleep the remainder of the period');
  check('the loop can be stopped',
        studioTs.includes('loopStop') && studioTs.includes('stop = true'));
  check('a failed live pass stops the loop',
        /live inference failed[\s\S]{0,80}stop = true/.test(studioTs),
        'otherwise it spins on a broken extractor');
  check('live frames wait for decodable data',
        /v\.videoWidth && v\.readyState >= 2/.test(studioTs));
  check('staleness is measured and surfaced',
        studioTs.includes('Date.now() - grabbed') &&
        metersTs.includes('ms old'),
        'a slow extractor must not silently report the past as the present');
  check('staleness hidden when negligible',
        metersTs.includes('staleMs > 250'),
        'no need to clutter the verdict at 5 ms');
}

// ---------------------------------------------------------------- PWA
{
  // Cloudflare Pages serves /dist/*.js with max-age=14400, and the Direct Upload
  // API ignores _headers (verified: it is served as a static file, 200). So the
  // service worker is the ONLY thing preventing a four-hour-stale bundle.
  check('sw uses a build-stamped cache name',
        swJs.includes("const BUILD = '__BUILD__'") && swJs.includes("'aidejavu-' + BUILD"),
        'a fixed cache name never busts');
  check('activate deletes every non-current cache',
        /caches\.keys\(\)[\s\S]{0,200}caches\.delete/.test(swJs));
  check('app code is fetched NETWORK-first',
        /fetch\(req, \{ cache: 'no-cache' \}\)/.test(swJs),
        'cache-first would reintroduce the stale-module bug this fixes');
  check('offline falls back to cache',
        swJs.includes('caches.match(req)') && swJs.includes("req.mode === 'navigate'"));
  check('third-party model weights are not intercepted',
        swJs.includes('url.origin !== self.location.origin'),
        'never cache 1.6-23 MB of CDN weights we do not control');
  check('registration disables the sw.js http cache',
        studioHtml.includes("updateViaCache: 'none'") &&
        mvpHtml.includes("updateViaCache: 'none'"),
        'a cached sw.js freezes the BUILD id and nothing ever busts');
  check('a new worker takes over without waiting for tabs to close',
        studioHtml.includes("postMessage('skipWaiting')") &&
        swJs.includes("self.skipWaiting()"));
  check('controllerchange reloads exactly once',
        /reloaded = true;\s*location\.reload\(\)/.test(studioHtml),
        'without the guard it loops');
  // Window is 800 chars: the registration block carries an updatefound handler
  // between register() and .catch(), measured at 555.
  check('sw failure does not break the app',
        /\.register\([\s\S]{0,800}?\.catch\(/.test(studioHtml),
        'the SW is an enhancement, not a dependency');
  const mf = JSON.parse(manifest);
  check('manifest is installable',
        !!mf.name && !!mf.start_url && mf.display === 'standalone' && mf.icons.length >= 2,
        `${mf.icons.length} icons, display ${mf.display}`);
  check('manifest has a maskable icon',
        mf.icons.some((i: any) => i.purpose === 'maskable'),
        'Android crops non-maskable icons badly');
  for (const page of [['studio', studioHtml], ['demo', mvpHtml]] as const) {
    check(`${page[0]} links the manifest`,
          page[1].includes('rel="manifest"') && page[1].includes('theme-color'));
  }
}

// ---------------------------------------------------------------- live audio
{
  const listen = read('../src/listen.ts');
  check('continuous mode is no longer image-only',
        !studioTs.includes('image-only'),
        'audio was blocked with a log message');
  // Check for CALLS, not mentions: the module explains in prose why it does not
  // use MediaRecorder, so a bare substring test fails on its own comment.
  check('audio uses a rolling buffer, not repeated MediaRecorder clips',
        listen.includes('startListening') && listen.includes('ring[') &&
        !/new MediaRecorder/.test(listen),
        'restarting the recorder per window drops sounds on the boundary');
  check('snapshot returns samples oldest-first',
        /\(start \+ i\) % size/.test(listen),
        'a raw ring read would scramble the time axis');
  check('partial buffer is handled before it fills',
        listen.includes('filled >= size') && listen.includes('subarray'));
  check('silence is gated, not classified',
        studioTs.includes('isSilent(pcm, gateDb())') &&
        studioTs.includes('below the'),
        'classifying silence forces it into the nearest label');
  check('the gate is user-adjustable',
        studioHtml.includes('id="gate"') && studioTs.includes('function gateDb('));
  check('the listener is stopped on exit and on loop end',
        (studioTs.match(/listener\?\.stop\(\)/g) ?? []).length >= 2,
        'an open AudioContext keeps the mic hot');
  check('audio staleness is measured too',
        studioTs.includes("'live-audio'") && studioTs.includes('Date.now() - grabbed'));
  check('the right device is named when missing',
        /isAudio \? 'start the microphone first' : 'start the camera first'/.test(studioTs));
}

console.log(failed ? `\n${failed} FAILURE(S)` : '\nALL PASS');
process.exit(failed ? 1 : 0);
