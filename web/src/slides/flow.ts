// Getting the deck: read the slides' state; when nothing is written yet, ask the server to write them (once); while
// Muse Spark writes, read the state again every 1.5 s; stop when the deck is ready or the writing failed.
// No DOM, so `npm test` can run it with plain Node: the page passes in the real calls.
import type { SlidesStatus } from '../types';

export const POLL_MS = 1500;
/** Answers of "nothing written" after asking for the slides before the page gives up (the server should say
 *  "writing" at once). */
export const NONE_LIMIT = 6;
/** Failed reads in a row while polling before the page shows the error (one is a blip: a server restart). */
export const MISS_LIMIT = 3;

export const NOT_STARTED = 'Muse Spark did not start writing the slides. Try again.';

export interface FlowCalls {
  /** GET slides.json. */
  get(): Promise<SlidesStatus>;
  /** POST slides/write (joins a run already going). */
  write(): Promise<SlidesStatus>;
  /** Waits; rejects with an AbortError when the page goes away. */
  wait(ms: number): Promise<void>;
  /** Every state read, for the page to show. */
  show(s: SlidesStatus): void;
  /** False once the page has gone away or started over. */
  alive(): boolean;
}

const aborted = (e: unknown) => (e as Error | null)?.name === 'AbortError';

/**
 * Runs until the deck is ready or the writing failed, and returns that last state (or the last one seen when the
 * page went away). `write` starts by asking for the slides (Try again after a failure); otherwise it reads first and
 * asks only when nothing is written. It never asks twice in one run. Throws one plain line.
 */
export async function runFlow(calls: FlowCalls, opts: { write?: boolean } = {}): Promise<SlidesStatus> {
  let wrote = !!opts.write;
  let s = wrote ? await calls.write() : await calls.get();
  let none = 0;
  let misses = 0;
  for (;;) {
    if (!calls.alive()) return s;
    calls.show(s);
    if (s.state === 'ready' || s.state === 'failed') return s;
    if (s.state === 'none') {
      if (!wrote) {
        wrote = true;
        s = await calls.write();
        continue;
      }
      if (++none > NONE_LIMIT) throw new Error(NOT_STARTED);
    } else none = 0;
    await calls.wait(POLL_MS);
    if (!calls.alive()) return s;
    try {
      s = await calls.get();
      misses = 0;
    } catch (e) {
      if (aborted(e) || ++misses >= MISS_LIMIT) throw e;
      // Keep showing the last state and read again after the next wait.
    }
  }
}

/** One call per key at a time: a second caller gets the call already going (a page mounted twice, two buttons). */
export function joinCall<T>(calls: Map<string, Promise<T>>, key: string, start: () => Promise<T>): Promise<T> {
  let call = calls.get(key);
  if (!call) {
    call = start();
    calls.set(key, call);
    const forget = () => {
      if (calls.get(key) === call) calls.delete(key);
    };
    call.then(forget, forget);
  }
  return call;
}

/** A wait that ends early with an AbortError when the signal fires. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      const e = new Error('The page went away.');
      e.name = 'AbortError';
      reject(e);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', abort);
      resolve();
    }, ms);
    if (signal?.aborted) abort();
    else signal?.addEventListener('abort', abort, { once: true });
  });
}
