// Every board item (lines, and pieces not in a line) or slide element of a lecture, for Ask Muse: the "Browse
// lecture" list by chapter, labels and seek times for the tray's chips, and the refs ([#12], [#L3], [#s0e1]) an
// answer can cite. No DOM, so `npm test` can run it with plain Node.
import type { AskItem, AskLabelledItem, AskRef, BoardLine, Piece, Replay, SlideElement } from '../types';
import { schedule } from '../player/timing.ts';
import { shortLabel, type TrayEntry } from './tray.ts';

export type ItemKind = 'piece' | 'line' | 'element';

export interface LectureItem {
  ref: string; // "#12", "#L3", "#s0e1"
  item: { kind: 'piece'; id: number } | { kind: 'line'; id: string } | { kind: 'element'; id: string };
  /** The reading's kind: "equation", "graph", "bullet"... */
  kind: string;
  text: string;
  latex: string;
  label: string;
  /** Where to seek to see it explained: first spoken (but not before the board starts drawing it), else redrawn
   *  or written, else its slide's start. */
  t: number;
  erasedAt: number | null;
  preExisting: boolean;
}

export interface ChapterGroup {
  title: string;
  t: number;
  items: LectureItem[];
}

export interface LectureIndex {
  /** Every piece, line and slide element by ref. */
  byRef: Map<string, LectureItem>;
  /** What "Browse lecture" lists: lines and the pieces not in one (or each distinct slide element), by chapter. */
  groups: ChapterGroup[];
  count: number;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export const kindLabel = (k: string | undefined, fallback = 'Writing') =>
  k ? k.charAt(0).toUpperCase() + k.slice(1) : fallback;

/** Short plain text for an item: what it says, else its LaTeX, else its kind. */
export function itemLabel(read: { kind?: string; text?: string; latex?: string } | undefined, fallback = 'Writing') {
  const text = (read?.text ?? '').trim();
  if (text && text.toLowerCase() !== 'unreadable') return shortLabel(text);
  const latex = (read?.latex ?? '').trim().replace(/^\$+|\$+$/g, '');
  if (latex) return shortLabel(latex);
  return kindLabel(read?.kind, fallback);
}

const firstSpoken = (spans: { start: number }[] | undefined): number | null => {
  let best: number | null = null;
  for (const s of spans ?? []) {
    const v = num(s?.start);
    if (v != null && (best == null || v < best)) best = v;
  }
  return best;
};

const pieceWhen = (p: Piece) => num(p.redraw_at) ?? num(p.t_seen) ?? 0;

/** Seek time of a piece (the same rule the server uses for refs). */
// A teacher often starts talking about something a few seconds before writing it: a chip that jumped there would
// outline an empty patch of board. So it jumps to the later of the two (the board's own drawing schedule).
const drawStart = (p: Piece) => schedule(p).start;

export function pieceTime(p: Piece): number {
  const spoken = firstSpoken(p.spoken);
  return spoken != null ? Math.max(spoken, drawStart(p)) : pieceWhen(p);
}

export function lineTime(ln: BoardLine, members: Piece[]): number {
  const spoken = firstSpoken(members.flatMap((p) => p.spoken ?? []));
  if (spoken != null) return members.length ? Math.max(spoken, Math.min(...members.map(drawStart))) : spoken;
  return members.length ? Math.min(...members.map(pieceWhen)) : num(ln.t_complete) ?? 0;
}

/** "#12" -> piece 12, "#L3" -> line L3, "#s0e1" -> slide element s0e1; anything else -> null. */
export function parseRef(ref: string): LectureItem['item'] | null {
  const m = /^#(?:(\d{1,6})|(L\d{1,6})|(s\d{1,5}e\d{1,5}))$/.exec(ref.trim());
  if (!m) return null;
  if (m[1]) return { kind: 'piece', id: Number(m[1]) };
  if (m[2]) return { kind: 'line', id: m[2] };
  return { kind: 'element', id: m[3] };
}

export const refOf = (item: LectureItem['item']) => `#${item.id}`;

export interface IndexOptions {
  /** Lines the board shows as one line (others are listed as their pieces). Default: lines of two or more pieces. */
  isLine?(ln: BoardLine): boolean;
  /** The processed part starts here: seek times are never before it. */
  clipStart?: number;
}

export function lectureIndex(replay: Replay, opts: IndexOptions = {}): LectureIndex {
  const clipStart = opts.clipStart ?? replay.video?.start ?? 0;
  const at = (t: number) => Math.max(clipStart, t);
  const byRef = new Map<string, LectureItem>();
  const listed: LectureItem[] = [];
  const board = replay.board;
  if (board) {
    const pieces = new Map<number, Piece>();
    for (const p of board.pieces ?? []) if (p && Number.isFinite(p.id)) pieces.set(p.id, p);
    const inLine = new Set<number>();
    for (const ln of board.lines ?? []) {
      if (!ln || typeof ln.id !== 'string' || !Array.isArray(ln.pieces)) continue;
      const members = ln.pieces.map((id) => pieces.get(id)).filter((p): p is Piece => !!p);
      const it: LectureItem = {
        ref: `#${ln.id}`,
        item: { kind: 'line', id: ln.id },
        kind: ln.read?.kind ?? '',
        text: (ln.read?.text ?? '').trim(),
        latex: (ln.read?.latex ?? '').trim(),
        label: itemLabel(ln.read),
        t: at(lineTime(ln, members)),
        erasedAt: num(ln.removed_at),
        preExisting: !!ln.pre_existing,
      };
      byRef.set(it.ref, it);
      const shown = ln.pieces.length >= 2 && !!(it.text || it.latex) && (opts.isLine ? opts.isLine(ln) : true);
      if (shown && members.length) {
        listed.push(it);
        ln.pieces.forEach((id) => inLine.add(id));
      }
    }
    for (const p of pieces.values()) {
      const it: LectureItem = {
        ref: `#${p.id}`,
        item: { kind: 'piece', id: p.id },
        kind: p.read?.kind ?? '',
        text: (p.read?.text ?? '').trim(),
        latex: (p.read?.latex ?? '').trim(),
        label: itemLabel(p.read),
        t: at(pieceTime(p)),
        erasedAt: num(p.removed_at),
        preExisting: !!p.pre_existing,
      };
      byRef.set(it.ref, it);
      if (!inLine.has(p.id)) listed.push(it);
    }
  } else if (replay.slides) {
    const seen = new Set<string>(); // a build step repeats the slide's earlier elements under new ids
    for (const st of replay.slides.states ?? []) {
      for (const el of (st?.elements ?? []) as SlideElement[]) {
        if (!el || typeof el.id !== 'string') continue;
        const it: LectureItem = {
          ref: `#${el.id}`,
          item: { kind: 'element', id: el.id },
          kind: el.kind ?? '',
          text: (el.text ?? '').trim(),
          latex: (el.latex ?? '').trim(),
          label: itemLabel(el, 'Text'),
          t: at(firstSpoken(el.spoken) ?? num(st.t_start) ?? 0),
          erasedAt: null,
          preExisting: false,
        };
        byRef.set(it.ref, it);
        const key = `${it.kind}|${(it.latex || it.text).replace(/\s+/g, ' ').toLowerCase()}`;
        if (!seen.has(key)) {
          seen.add(key);
          listed.push(it);
        }
      }
    }
  }
  listed.sort((a, b) => a.t - b.t || a.ref.localeCompare(b.ref));

  const chapters = (replay.chapters ?? [])
    .filter((c) => c && Number.isFinite(c.t))
    .slice()
    .sort((a, b) => a.t - b.t);
  const groups: ChapterGroup[] = chapters.length
    ? chapters.map((c) => ({ title: c.title || 'Chapter', t: c.t, items: [] }))
    : [{ title: 'Lecture', t: clipStart, items: [] }];
  for (const it of listed) {
    let g = 0;
    for (let i = 0; i < groups.length; i++) if (groups[i].t <= it.t + 0.05) g = i;
    groups[g].items.push(it);
  }
  return { byRef, groups, count: listed.length };
}

/** Whether an item matches a search: every word of the query is in its text, LaTeX, kind or ref. */
export function matches(it: LectureItem, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const hay = `${it.label} ${it.text} ${it.latex} ${it.kind} ${it.ref}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}

export function filterGroups(groups: ChapterGroup[], query: string): ChapterGroup[] {
  if (!query.trim()) return groups;
  return groups.map((g) => ({ ...g, items: g.items.filter((it) => matches(it, query)) })).filter((g) => g.items.length);
}

export const askRefOf = (it: LectureItem): AskRef => ({
  ref: it.ref,
  kind: it.item.kind,
  id: it.item.id,
  t: it.t,
  label: it.label,
});

/** The chip for an attached item (label and seek time), from the lecture when it is a board or slide item. */
export function entryFor(item: AskItem, index: LectureIndex | null): TrayEntry {
  switch (item.kind) {
    case 'piece':
    case 'line':
    case 'element': {
      const it = index?.byRef.get(`#${item.id}`);
      return { item, label: it?.label ?? `#${item.id}`, t: it?.t ?? 0 };
    }
    case 'span':
      return { item, label: 'Speech', t: item.t0 };
    case 'moment':
      return { item, label: 'Moment', t: item.t };
  }
}

/** The chip for an item attached to a saved question: board and slide items are named from the lecture (like the
 *  chips of a new question); a span or a moment keeps the label saved with it (the selected words, "The board at
 *  1:23"), and so does an item the lecture no longer has. */
export function savedEntry(saved: AskLabelledItem, index: LectureIndex | null): TrayEntry {
  const { label, ...rest } = saved;
  const item = rest as AskItem;
  const entry = entryFor(item, index);
  if (!label) return entry;
  const known = item.kind !== 'span' && item.kind !== 'moment' && !!index?.byRef.has(`#${item.id}`);
  return known ? entry : { ...entry, label };
}

/** The lecture item a line of the study notes stands for: same reading, closest in time. */
export function matchNote(
  index: LectureIndex,
  note: { t: number; latex?: string; text?: string },
): LectureItem | null {
  const tex = (s: string) => s.trim().replace(/^\$+|\$+$/g, '').replace(/\s+/g, '');
  const txt = (s: string) => s.replace(/\s+/g, ' ').trim().toLowerCase();
  const nl = tex(note.latex ?? '');
  const nt = txt(note.text ?? '');
  let best: LectureItem | null = null;
  for (const it of index.byRef.values()) {
    const same = nl ? tex(it.latex) === nl : !!nt && txt(it.text) === nt;
    if (!same) continue;
    if (!best || Math.abs(it.t - note.t) < Math.abs(best.t - note.t)) best = it;
  }
  return best;
}
