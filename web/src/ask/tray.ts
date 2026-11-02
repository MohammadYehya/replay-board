// The tray of things attached to the next question: board writing, slide text, a stretch of speech, a moment.
// It survives seeking and reloads: it is kept in sessionStorage, one tray per lecture.
// No DOM and no runtime imports, so `npm test` can run it with plain Node.
import type { AskItem } from '../types';

/** The server takes at most this many attached items per question. */
export const TRAY_MAX = 12;

export interface TrayEntry {
  item: AskItem;
  /** Short plain text for the chip (at most about 60 characters). */
  label: string;
  /** Where to seek to see it (video clock). */
  t: number;
}

export type AddStatus = 'added' | 'duplicate' | 'full';

export const trayKey = (jobId: string) => `replay-board:ask-tray:${jobId}`;
export const openKey = (jobId: string) => `replay-board:ask-open:${jobId}`;

const r2 = (x: number) => Math.round(x * 100) / 100;

/** One key per distinct item: the same writing, or the same moment to the second, is attached once. */
export function itemKey(item: AskItem): string {
  switch (item.kind) {
    case 'piece':
    case 'line':
    case 'element':
      return `${item.kind}:${item.id}`;
    case 'span':
      return `span:${r2(item.t0)}-${r2(item.t1)}`;
    case 'moment':
      return `moment:${Math.round(item.t)}`;
  }
}

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** An item as the server expects it, or null when it is not one (e.g. read back from an old tray). */
export function cleanItem(x: unknown): AskItem | null {
  if (!x || typeof x !== 'object') return null;
  const o = x as Record<string, unknown>;
  switch (o.kind) {
    case 'piece':
      return finite(o.id) && Number.isInteger(o.id) ? { kind: 'piece', id: o.id } : null;
    case 'line':
      return typeof o.id === 'string' && /^L\d+$/.test(o.id) ? { kind: 'line', id: o.id } : null;
    case 'element':
      return typeof o.id === 'string' && /^s\d+e\d+$/.test(o.id) ? { kind: 'element', id: o.id } : null;
    case 'span':
      return finite(o.t0) && finite(o.t1) && o.t1 >= o.t0 && o.t0 >= 0 ? { kind: 'span', t0: r2(o.t0), t1: r2(o.t1) } : null;
    case 'moment':
      return finite(o.t) && o.t >= 0 ? { kind: 'moment', t: Math.round(o.t * 10) / 10 } : null;
    default:
      return null;
  }
}

/** Cut a label to `max` characters on a word boundary when one is near, with an ellipsis. */
export function shortLabel(s: string, max = 60): string {
  const t = s.replace(/\s+/g, ' ').trim();
  if (t.length <= max) return t;
  const cut = t.slice(0, max - 1);
  const sp = cut.lastIndexOf(' ');
  return `${(sp > max * 0.6 ? cut.slice(0, sp) : cut).replace(/[\s,;:.]+$/, '')}…`;
}

/** Add an entry unless it is already there or the tray is full. Never changes the tray it is given. */
export function addEntry(tray: TrayEntry[], entry: TrayEntry): { tray: TrayEntry[]; status: AddStatus } {
  const key = itemKey(entry.item);
  if (tray.some((e) => itemKey(e.item) === key)) return { tray, status: 'duplicate' };
  if (tray.length >= TRAY_MAX) return { tray, status: 'full' };
  return { tray: [...tray, entry], status: 'added' };
}

export function removeEntry(tray: TrayEntry[], key: string): TrayEntry[] {
  return tray.filter((e) => itemKey(e.item) !== key);
}

export const hasEntry = (tray: TrayEntry[], item: AskItem) => {
  const key = itemKey(item);
  return tray.some((e) => itemKey(e.item) === key);
};

type Reader = { getItem(key: string): string | null };
type Writer = { setItem(key: string, value: string): void; removeItem(key: string): void };

/** The saved tray for a lecture; anything unreadable is dropped (storage can be blocked or hold an old shape). */
export function loadTray(storage: Reader | null | undefined, jobId: string): TrayEntry[] {
  let raw: string | null = null;
  try {
    raw = storage?.getItem(trayKey(jobId)) ?? null;
  } catch {
    return [];
  }
  if (!raw) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }
  const list = Array.isArray(parsed) ? parsed : [];
  let tray: TrayEntry[] = [];
  for (const x of list) {
    const o = (x ?? {}) as Record<string, unknown>;
    const item = cleanItem(o.item);
    if (!item) continue;
    const label = typeof o.label === 'string' ? shortLabel(o.label) : '';
    const t = finite(o.t) ? o.t : 0;
    tray = addEntry(tray, { item, label, t }).tray;
  }
  return tray;
}

export function saveTray(storage: Writer | null | undefined, jobId: string, tray: TrayEntry[]): void {
  try {
    if (tray.length) storage?.setItem(trayKey(jobId), JSON.stringify(tray));
    else storage?.removeItem(trayKey(jobId));
  } catch {
    /* storage full or blocked: the tray still works for this page */
  }
}

/** sessionStorage, or null where the browser blocks it. */
export function session(): (Reader & Writer) | null {
  try {
    return typeof sessionStorage === 'undefined' ? null : sessionStorage;
  } catch {
    return null;
  }
}
