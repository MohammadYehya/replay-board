// The Ask Muse exchange with the server (app/ask.py): reading the answer's event stream, checking what comes back,
// the history sent with a question, and the cost line under an answer.
// No DOM, so `npm test` can run it with plain Node.
import type { AskDone, AskLabelledItem, AskRef, AskTurn } from '../types';
import { SseParser, type SseEvent } from './sse.ts';
import { cleanItem, shortLabel, type TrayEntry } from './tray.ts';

/** The server keeps this many earlier turns as history, each answer cut to this many characters. */
export const HISTORY_TURNS = 6;
export const HISTORY_CHARS = 2000;
export const QUESTION_MAX = 2000;

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const str = (v: unknown, fallback = '') => (typeof v === 'string' ? v : fallback);
const KINDS = new Set(['piece', 'line', 'element']);

export function cleanRef(x: unknown): AskRef | null {
  if (!x || typeof x !== 'object') return null;
  const o = x as Record<string, unknown>;
  if (typeof o.ref !== 'string' || !KINDS.has(o.kind as string)) return null;
  if (typeof o.id !== 'number' && typeof o.id !== 'string') return null;
  return {
    ref: o.ref,
    kind: o.kind as AskRef['kind'],
    id: o.id,
    t: finite(o.t) ? o.t : 0,
    label: str(o.label, o.ref).slice(0, 120),
  };
}

const refs = (x: unknown) => (Array.isArray(x) ? x.map(cleanRef).filter((r): r is AskRef => !!r) : []);
const times = (x: unknown) => (Array.isArray(x) ? x.filter(finite) : []);

export function cleanDone(x: unknown): AskDone {
  const o = (x && typeof x === 'object' ? x : {}) as Record<string, unknown>;
  return {
    id: str(o.id, o.id == null ? '' : String(o.id)),
    answer: str(o.answer),
    items: refs(o.items),
    times: times(o.times),
    cost_usd: finite(o.cost_usd) ? o.cost_usd : null,
    cached: o.cached === true,
    saved: o.saved !== false,
  };
}

/** A saved attached item, with its label when it has one. */
export function cleanLabelledItem(x: unknown): AskLabelledItem | null {
  const item = cleanItem(x);
  if (!item) return null;
  const label = (x as { label?: unknown }).label;
  return typeof label === 'string' && label.trim() ? { ...item, label: shortLabel(label) } : item;
}

/** The selection sent with a question. A span carries its chip's label (the words the student selected), so the
 *  saved conversation shows them after a reload; other items are named by the server from the lecture. */
export function requestSelection(entries: TrayEntry[]): AskLabelledItem[] {
  return entries.map((e) => (e.item.kind === 'span' && e.label.trim() ? { ...e.item, label: e.label } : e.item));
}

export function cleanTurn(x: unknown): AskTurn | null {
  if (!x || typeof x !== 'object') return null;
  const o = x as Record<string, unknown>;
  if (typeof o.q !== 'string' || typeof o.answer !== 'string') return null;
  const selection = Array.isArray(o.selection)
    ? o.selection.map(cleanLabelledItem).filter((i): i is AskLabelledItem => !!i)
    : [];
  return {
    id: str(o.id, o.id == null ? '' : String(o.id)),
    q: o.q,
    selection,
    answer: o.answer,
    items: refs(o.items),
    times: times(o.times),
    t: finite(o.t) ? o.t : null,
    at: typeof o.at === 'string' || finite(o.at) ? o.at : undefined,
    cost_usd: finite(o.cost_usd) ? o.cost_usd : null,
    cached: o.cached === true,
  };
}

export function cleanChat(x: unknown): AskTurn[] {
  const turns = x && typeof x === 'object' ? (x as { turns?: unknown }).turns : null;
  return Array.isArray(turns) ? turns.map(cleanTurn).filter((t): t is AskTurn => !!t) : [];
}

/** One plain line from whatever the server put in an error event. */
export function plainLine(x: unknown, fallback = 'Muse Spark could not answer. Try again.'): string {
  const s = typeof x === 'string' ? x.replace(/\s+/g, ' ').trim() : '';
  return s ? s.slice(0, 300) : fallback;
}

export class StreamError extends Error {
  name = 'StreamError';
}

/**
 * Read the answer's event stream: "delta" events go to onDelta as they arrive; resolves with the "done" event.
 * An "error" event, or a stream that ends without "done", rejects with one plain line (StreamError).
 */
export async function readAnswer(
  body: ReadableStream<Uint8Array>,
  onDelta: (text: string) => void,
): Promise<AskDone> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  const got: { done: AskDone | null } = { done: null };
  const handle = (events: SseEvent[]) => {
    for (const ev of events) {
      if (got.done) return; // the answer is whole: nothing after "done" can take it away
      let data: unknown;
      try {
        data = JSON.parse(ev.data);
      } catch {
        continue; // not ours (e.g. a keep-alive)
      }
      if (ev.event === 'delta') {
        const text = (data as { text?: unknown } | null)?.text;
        if (typeof text === 'string' && text) onDelta(text);
      } else if (ev.event === 'done') got.done = cleanDone(data);
      else if (ev.event === 'error') throw new StreamError(plainLine((data as { error?: unknown } | null)?.error));
    }
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      handle(parser.push(decoder.decode(value, { stream: true })));
      if (got.done) break;
    }
    handle(parser.push(decoder.decode()));
    handle(parser.end());
  } catch (e) {
    reader.cancel().catch(() => {});
    if (e instanceof StreamError || (e as Error)?.name === 'AbortError') throw e;
    throw new StreamError('The connection to the server was lost while Muse Spark was answering.');
  }
  if (got.done) {
    reader.cancel().catch(() => {});
    return got.done;
  }
  throw new StreamError('The answer stopped before it finished. Try again.');
}

/** What is sent as history: the last finished turns, each answer cut short. */
export function historyFor(turns: { q: string; answer: string; ok: boolean }[]): { q: string; a: string }[] {
  return turns
    .filter((t) => t.ok && t.answer.trim())
    .slice(-HISTORY_TURNS)
    .map((t) => ({ q: t.q, a: t.answer.slice(0, HISTORY_CHARS) }));
}

/** "$0.004", "<$0.001", "cached", or '' when the cost is not known. */
export function costLabel(cost: number | null | undefined, cached?: boolean): string {
  if (cached) return 'cached';
  if (cost == null || !Number.isFinite(cost)) return '';
  if (cost <= 0) return '$0';
  if (cost < 0.001) return '<$0.001';
  return `$${cost < 0.1 ? cost.toFixed(3) : cost.toFixed(2)}`;
}
