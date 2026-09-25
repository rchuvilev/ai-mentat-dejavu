/**
 * Continuous audio listening via a rolling PCM buffer.
 *
 * WHY NOT REUSE recordClip(): that path is MediaRecorder -> encoded blob ->
 * decodeAudioData. Looping it means stopping and restarting the recorder for
 * every window, which leaves a gap between clips — a sound landing on the
 * boundary is split across two windows and recognised in neither. It also pays
 * an encode + decode round trip per window for data we already have as PCM.
 *
 * Instead: tap the live stream once with an AnalyserNode, keep the newest
 * `windowMs` of samples in a ring buffer, and read a snapshot whenever the
 * classifier wants one. Windows can then OVERLAP, so a sound is always fully
 * inside at least one of them.
 *
 * Energy gating matters here in a way it does not for images: a camera pointed
 * at nothing still shows something, but a microphone in a quiet room produces
 * silence, and classifying silence forces it into whichever label happens to be
 * closest. So a window below the threshold is reported as silence and the
 * classifier is not run at all.
 */
import { rmsDb } from './audio.js';

export interface Listener {
  /** Newest `windowMs` of mono PCM, oldest sample first. */
  snapshot(): Float32Array;
  /** dBFS of the current window — drives the meter and the gate. */
  level(): number;
  sampleRate: number;
  stop(): void;
}

/**
 * @param stream a live mic MediaStream
 * @param windowMs how much audio a classification looks at
 */
export function startListening(stream: MediaStream, windowMs = 1200): Listener {
  const AC: any = (window as any).AudioContext || (window as any).webkitAudioContext;
  const ctx: AudioContext = new AC();
  const src = ctx.createMediaStreamSource(stream);
  const an = ctx.createAnalyser();
  an.fftSize = 2048;
  src.connect(an);

  const sr = ctx.sampleRate;
  const size = Math.max(an.fftSize, Math.ceil((windowMs / 1000) * sr));
  const ring = new Float32Array(size);
  let write = 0;
  let filled = 0;

  const chunk = new Float32Array(an.fftSize);
  // Poll at roughly half the analyser window so no samples are missed. This is
  // a synchronous copy of already-captured data, so a fixed interval is safe
  // here (unlike an async inference loop, which must self-schedule).
  const periodMs = Math.max(20, Math.floor((an.fftSize / sr) * 1000 / 2));
  const timer = window.setInterval(() => {
    an.getFloatTimeDomainData(chunk);
    for (let i = 0; i < chunk.length; i++) {
      ring[write] = chunk[i];
      write = (write + 1) % size;
    }
    filled = Math.min(size, filled + chunk.length);
  }, periodMs);

  return {
    sampleRate: sr,
    snapshot(): Float32Array {
      // Unwrap the ring into chronological order, oldest first.
      const out = new Float32Array(size);
      const start = write;
      for (let i = 0; i < size; i++) out[i] = ring[(start + i) % size];
      // Before the buffer has filled, only the tail holds real audio.
      return filled >= size ? out : out.subarray(size - filled);
    },
    level(): number {
      an.getFloatTimeDomainData(chunk);
      return rmsDb(chunk);
    },
    stop() {
      clearInterval(timer);
      try { src.disconnect(); } catch { /* already gone */ }
      if (ctx.state !== 'closed') ctx.close().catch(() => {});
    },
  };
}

/** Is this window loud enough to be worth classifying? */
export function isSilent(pcm: Float32Array, gateDb: number): boolean {
  return rmsDb(pcm) < gateDb;
}
