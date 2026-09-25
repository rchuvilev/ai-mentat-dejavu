# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Train an image classifier **in the browser**, persist it to **IndexedDB**, then run

## Repository location

- Local: `/var/minis/repos/ai-mentat-dejavu` — **all repos live under `/var/minis/repos/`**
- Remote: `hexstack-apps/ai-mentat-dejavu` (private)

## Stack

JavaScript, Python, TypeScript

## Commands

```sh
npm run build        # tsc
npm run test         # node dist/test.js && node dist/test_actions.js
npm run cli          # node dist/cli.js
npm run serve        # node dist/server.js 8772 .
npm run site         # tsc && mkdir -p site/dist && cp dist/*.js site/dist/ && node
```

## Tests

```sh
node desktop/test_shim.mjs
```

## Conventions

- One logical change = one commit, with the measurements behind it.
- Add a `Requested: "..."` trailer citing the originating request.
- Push to the private `hexstack-apps` remote — that is the backup.
- Never `mv` a git repo inside `/var/minis` (it corrupts the object
  store on this Android FS); re-clone from GitHub instead.
- Run tests AND build before deploying; smoke-test the bundle.

## 🔴 Test wiring — three suites were compiled but never run

`npm test` ran only `test.js` and `test_actions.js`. `test_backbone`,
`test_flow` and `test_media` were compiled by `tsc` into `dist/` and then
never executed — **140 of 210 assertions, two thirds of the coverage, silently
dead.** All three passed when run by hand, so nothing was broken; they simply
could not catch a future regression.

Fixed 2026-08-30: `test` now chains all five plus `test_failsafe`.
**228 assertions.**

⚠️ A test file that is not named in the `test` script is invisible. When adding
a suite, add it to `package.json` — `tsc` compiling it is not enough.

`npm run test:build` rebuilds first; the tests run against `dist/`, so an
un-rebuilt change is tested as its old self.

## Error handling — fail-safe, never silent

`src/failsafe.ts`. This app runs in the browser, so a throw in a UI handler
stops that handler and leaves the page half-updated. Swallowing keeps the app
usable — right for an optional camera preview — but a swallow with no record
makes "the flip button vanished" untraceable.

```ts
import { quiet, quietAsync, attempt } from './failsafe.js';
const devices = await quietAsync('studio.enumerateDevices', () => nav.enumerateDevices(), []);
```

Stable `op` label + message + optional context, bounded 100-entry buffer via
`recentFailures()`, injectable sink. `quietAsync` never rejects — an unhandled
rejection in an event handler is invisible and kills the rest of the handler.

Applied to the two that hid user-visible failures:
- `studio.enumerateDevices` — an empty list hides the flip button, so the user
  sees a missing control with no explanation.
- `studio.loadConfig` / `mvp.loadConfig` — a malformed saved config was
  discarded silently, so settings vanished with no clue why.

The remaining bare catches are deliberate and commented (`/* already gone */`,
`/* skip partial line */`, tf.js CPU-backend fallbacks) — failure there is the
expected path, not a hidden problem.
