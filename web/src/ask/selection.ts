// Words selected in the transcript become a stretch of speech to ask about: {kind: "span", t0, t1}, labelled with
// the selected text. No DOM here (the transcript finds which words a selection covers), so `npm test` can run it.
import type { Word } from '../types';
import type { TrayEntry } from './tray';

export interface SpanPick {
  t0: number;
  t1: number;
  label: string;
}

const r2 = (x: number) => Math.round(x * 100) / 100;

/** Clean selected text into a chip label: one line, no stray punctuation at the ends, at most `max` characters. */
export function selectionLabel(text: string, max = 60): string {
  const t = text
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^[\s,;:.!?\-–—"“”']+|[\s,;:.\-–—"“”']+$/g, '');
  if (t.length <= max) return t;
  const cut = t.slice(0, max - 1);
  const sp = cut.lastIndexOf(' ');
  return `${(sp > max * 0.6 ? cut.slice(0, sp) : cut).replace(/[\s,;:.]+$/, '')}…`;
}

/** The span covered by the selected words (indices into the sorted word list, in any order), or null. */
export function spanFromWords(words: Word[], indices: number[], text = ''): SpanPick | null {
  const ok = indices.filter((i) => Number.isInteger(i) && i >= 0 && i < words.length && Number.isFinite(words[i]?.s));
  if (!ok.length) return null;
  const first = Math.min(...ok);
  const last = Math.max(...ok);
  const t0 = words[first].s;
  const t1 = Math.max(t0, Number.isFinite(words[last].e) ? words[last].e : words[last].s);
  const label = selectionLabel(text) || selectionLabel(words.slice(first, last + 1).map((w) => w.w).join(' '));
  return { t0: r2(t0), t1: r2(t1), label };
}

/** A selection inside paragraphs that have no word times: the whole of those paragraphs. */
export function spanFromTurns(turns: { start: number; end: number; text: string }[], first: number, last: number, text = '') {
  const a = turns[Math.min(first, last)];
  const b = turns[Math.max(first, last)];
  if (!a || !b || !Number.isFinite(a.start)) return null;
  const t1 = Number.isFinite(b.end) ? Math.max(a.start, b.end) : a.start;
  return { t0: r2(a.start), t1: r2(t1), label: selectionLabel(text) || selectionLabel(a.text) };
}

/** The tray entry for a picked span. */
export const spanEntry = (s: SpanPick): TrayEntry => ({
  item: { kind: 'span', t0: s.t0, t1: s.t1 },
  label: s.label || 'Speech',
  t: s.t0,
});
