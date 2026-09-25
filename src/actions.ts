/**
 * onMatched / onNotMatched dispatch.
 *
 * A prediction becomes a MATCH when the predicted class is in `targets` AND its
 * confidence >= `threshold`. Everything else is a NO-MATCH. Both branches are
 * independently switchable between console, webhook, both, or off — you usually
 * want a webhook on match and console on no-match, so they are separate configs
 * rather than one global mode.
 */

export type Sink = 'off' | 'console' | 'webhook' | 'both';

export interface ActionConfig {
  targets: string[];        // class names that count as a match
  threshold: number;        // 0..1 minimum confidence
  onMatched: Sink;
  onNotMatched: Sink;
  webhookUrl: string;
  webhookMethod: 'POST' | 'GET';
  /** extra static fields merged into the payload (e.g. a device id) */
  extra?: Record<string, unknown>;
  /** don't fire the same branch more often than this (ms); 0 = no limit */
  cooldownMs: number;
}

export const defaultConfig = (classes: string[]): ActionConfig => ({
  targets: classes.slice(0, 1),
  threshold: 0.7,
  onMatched: 'console',
  onNotMatched: 'off',
  webhookUrl: '',
  webhookMethod: 'POST',
  cooldownMs: 0,
});

export interface Prediction {
  predicted: string;
  confidence: number;
  probs: number[];
  classes: string[];
  modelId: string;
  modelName: string;
}

export interface DispatchResult {
  matched: boolean;
  sink: Sink;
  delivered: string;        // 'none' | 'console' | 'webhook:<status>' | 'webhook:err:<msg>'
  skipped?: 'cooldown';
  payload: Record<string, unknown>;
}

function buildPayload(p: Prediction, matched: boolean, cfg: ActionConfig) {
  return {
    event: matched ? 'onMatched' : 'onNotMatched',
    matched,
    predicted: p.predicted,
    confidence: +p.confidence.toFixed(4),
    threshold: cfg.threshold,
    targets: cfg.targets,
    probabilities: Object.fromEntries(
      p.classes.map((c, i) => [c, +p.probs[i].toFixed(4)])),
    model: { id: p.modelId, name: p.modelName },
    ts: new Date().toISOString(),
    ...(cfg.extra ?? {}),
  };
}

const lastFired: Record<string, number> = {};

/**
 * Evaluate a prediction against the rule and fire the configured sink.
 * Never throws: a failing webhook must not break the inference loop, so the
 * error is captured in `delivered` instead.
 */
export async function dispatch(
  p: Prediction, cfg: ActionConfig,
  logger: (line: string, cls?: string) => void = () => {},
): Promise<DispatchResult> {
  const matched = cfg.targets.includes(p.predicted) && p.confidence >= cfg.threshold;
  const sink = matched ? cfg.onMatched : cfg.onNotMatched;
  const payload = buildPayload(p, matched, cfg);
  const branch = matched ? 'matched' : 'notMatched';

  if (sink === 'off') return { matched, sink, delivered: 'none', payload };

  if (cfg.cooldownMs > 0) {
    const now = Date.now();
    if (now - (lastFired[branch] ?? 0) < cfg.cooldownMs) {
      return { matched, sink, delivered: 'none', skipped: 'cooldown', payload };
    }
    lastFired[branch] = now;
  }

  let delivered = 'none';

  if (sink === 'console' || sink === 'both') {
    // Use the real console so it shows in devtools, and mirror to the in-page log.
    const fn = matched ? console.info : console.warn;
    fn(`[${payload.event}]`, payload);
    logger(`${payload.event}: ${p.predicted} @ ${(p.confidence * 100).toFixed(1)}%`,
           matched ? 'g' : 'o');
    delivered = 'console';
  }

  if (sink === 'webhook' || sink === 'both') {
    if (!cfg.webhookUrl) {
      delivered = delivered === 'console' ? 'console' : 'none';
      logger('webhook skipped: no URL configured', 'r');
    } else {
      try {
        let res: Response;
        if (cfg.webhookMethod === 'GET') {
          const u = new URL(cfg.webhookUrl);
          u.searchParams.set('payload', JSON.stringify(payload));
          res = await fetch(u.toString(), { method: 'GET' });
        } else {
          res = await fetch(cfg.webhookUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
        }
        delivered = `webhook:${res.status}`;
        logger(`webhook ${cfg.webhookMethod} -> ${res.status}`, res.ok ? 'g' : 'r');
      } catch (e: any) {
        // Most commonly CORS or an unreachable host. Report, never throw.
        const msg = String(e?.message ?? e).slice(0, 80);
        delivered = `webhook:err:${msg}`;
        logger(`webhook failed: ${msg}`, 'r');
      }
    }
  }

  return { matched, sink, delivered, payload };
}

/** Reset cooldown state — used by tests. */
export function resetCooldowns(): void {
  for (const k of Object.keys(lastFired)) delete lastFired[k];
}
