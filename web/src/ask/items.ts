// Every board item (lines, and pieces not in a line) or slide element of a lecture, for Ask Muse: the "Browse
// lecture" list by chapter, labels and seek times for the tray's chips, and the refs ([#12], [#L3], [#s0e1]) an
// answer can cite. No DOM, so `npm test` can run it with plain Node.
import type { AskItem, AskLabelledItem, AskRef, BoardLine, Piece, Replay, SlideElement } from '../types';
import { progressAt, schedule } from '../player/timing.ts';
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
  /** Where to seek to see it explained, with the board showing it: for board writing, from when it is explained on
   *  to when the board shows nearly all of it (shownFrom), but still in the chapter it is explained in; for a slide
   *  element, when it is explained. */
  t: number;
  /** Where it is explained: for board writing, when it is first spoken about (never before the board starts drawing
   *  it; else redrawn or written); for a slide element, first spoken, else its slide's start. */
  explained: number;
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

// A teacher often starts talking about something a few seconds before writing it, so a chip jumps no earlier than
// the board starts drawing it (the board's own drawing schedule).
const drawStart = (p: Piece) => schedule(p).start;

/** A chip shows its writing outlined on the board, so it jumps on to where the board has drawn a piece whole, or this
 *  share of a line (by area: a line often gets a dot or a sign added minutes later)... */
export const SHOWN_SHARE = 0.9;
/** ...but never more than this many seconds after it is first explained (a line can take a minute to write)... */
export const SHOWN_WAIT = 30;
/** ...never closer than this to when one of its pieces is erased (the board fades it out from then)... */
export const ERASE_MARGIN = 1;
/** ...and never into the next chapter (Browse lists it under the chapter it is explained in, and a chip that jumped
 *  past that showed a later part of the lecture: synth3d-vector3d's axes, explained at 0:13 under "Coordinate axes",
 *  jumped to 0:37, two chapters on). A jump this close to the next chapter is still in its own (chapterAt). Not more:
 *  the generated lectures draw right up to their chapter starts, and 1 s before them the saddle's arrow (#28) was 7%
 *  drawn, against 75% 0.1 s before. */
export const CHAPTER_MARGIN = 0.1;

/** The first time from t on when the board shows `share` of these pieces (by area, as the board draws them), but
 *  at most `wait` seconds later, never after `before`, and never within ERASE_MARGIN of one of them being erased.
 *  Jumping to where an item is first talked about outlined an empty patch: the board had drawn none of 25 of the 31
 *  items Browse lists in the three real lectures there (test-full #34 jumped to 0:24, the first instant of its
 *  redraw, and the outline was gone before the formula came). */
export function shownFrom(members: Piece[], t: number, share = SHOWN_SHARE, wait = SHOWN_WAIT, before = Infinity): number {
  const parts = members.map((p) => ({
    s: schedule(p),
    area: Math.max(1, (num(p.box?.[2]) ?? 0) * (num(p.box?.[3]) ?? 0)),
  }));
  const total = parts.reduce((sum, q) => sum + q.area, 0);
  const shown = (u: number) => parts.reduce((sum, q) => sum + q.area * progressAt(q.s, u), 0) / total;
  if (!parts.length || shown(t) >= share) return t;
  let limit = Math.min(t + wait, before);
  for (const q of parts) if (q.s.removedAt != null && q.s.removedAt > t) limit = Math.min(limit, q.s.removedAt - ERASE_MARGIN);
  // the last hundredth of a second it may go to
  limit = Math.floor(limit * 100 + 1e-6) / 100;
  if (limit <= t) return t;
  if (shown(limit) < share) return limit;
  // The board only ever adds to a piece (progressAt never goes back), so the share grows with time: the first time
  // it reaches `share` is found by halving [t, limit], whatever the shape of the board's drawing schedule.
  let lo = t;
  let hi = limit;
  for (let i = 0; i < 24 && hi - lo > 0.002; i++) {
    const mid = (lo + hi) / 2;
    if (shown(mid) >= share) hi = mid;
    else lo = mid;
  }
  // the first hundredth of a second from there on where the share is reached (never just under it)
  let at = Math.ceil(lo * 100 - 1e-6) / 100;
  while (at < hi && shown(at) < share) at = Math.round(at * 100 + 1) / 100;
  return shown(at) >= share ? at : hi;
}

/** The start of the chapter after the one t is in (chapterAt's rule), from the chapters' sorted starts. */
const nextChapter = (starts: readonly number[], t: number) => starts.find((s) => s > t + 0.05) ?? Infinity;

/** Where a piece is explained: first spoken about (but not before the board starts drawing it), else when it is
 *  redrawn or written. */
export function pieceExplained(p: Piece): number {
  const spoken = firstSpoken(p.spoken);
  return spoken != null ? Math.max(spoken, drawStart(p)) : pieceWhen(p);
}

/** Seek time of a piece: from where it is explained (not before `from`, where the processed part starts) on to where
 *  the board shows it whole (shownFrom), in the same chapter (`chapters`: the chapters' sorted starts). */
export function pieceTime(p: Piece, chapters: readonly number[] = [], from = -Infinity): number {
  const e = Math.max(from, pieceExplained(p));
  return shownFrom([p], e, 1, SHOWN_WAIT, nextChapter(chapters, e) - CHAPTER_MARGIN);
}

/** Where a line is explained: its pieces' first mention (but not before the board starts drawing the line), else
 *  when its first piece is redrawn or written. */
export function lineExplained(ln: BoardLine, members: Piece[]): number {
  if (!members.length) return num(ln.t_complete) ?? 0;
  const spoken = firstSpoken(members.flatMap((p) => p.spoken ?? []));
  return spoken != null ? Math.max(spoken, Math.min(...members.map(drawStart))) : Math.min(...members.map(pieceWhen));
}

/** Seek time of a line: from where it is explained (not before `from`) on to where the board shows SHOWN_SHARE of it,
 *  in the same chapter. */
export function lineTime(ln: BoardLine, members: Piece[], chapters: readonly number[] = [], from = -Infinity): number {
  const e = Math.max(from, lineExplained(ln, members));
  if (!members.length) return e;
  return shownFrom(members, e, SHOWN_SHARE, SHOWN_WAIT, nextChapter(chapters, e) - CHAPTER_MARGIN);
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
  const chapters = (replay.chapters ?? [])
    .filter((c) => c && Number.isFinite(c.t))
    .slice()
    .sort((a, b) => a.t - b.t);
  const starts = chapters.map((c) => c.t);
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
        t: at(lineTime(ln, members, starts, clipStart)),
        explained: at(lineExplained(ln, members)),
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
        t: at(pieceTime(p, starts, clipStart)),
        explained: at(pieceExplained(p)),
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
        const t = at(firstSpoken(el.spoken) ?? num(st.t_start) ?? 0);
        const it: LectureItem = {
          ref: `#${el.id}`,
          item: { kind: 'element', id: el.id },
          kind: el.kind ?? '',
          text: (el.text ?? '').trim(),
          latex: (el.latex ?? '').trim(),
          label: itemLabel(el, 'Text'),
          t,
          explained: t,
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
  // By the time each row shows (where it jumps), under the chapter it is explained in: its jump never leaves that
  // chapter (CHAPTER_MARGIN), so the rows' times run in order through the chapters.
  listed.sort((a, b) => a.t - b.t || a.explained - b.explained || a.ref.localeCompare(b.ref));

  const groups: ChapterGroup[] = chapters.length
    ? chapters.map((c) => ({ title: c.title || 'Chapter', t: c.t, items: [] }))
    : [{ title: 'Lecture', t: clipStart, items: [] }];
  for (const it of listed) {
    let g = 0;
    for (let i = 0; i < groups.length; i++) if (groups[i].t <= it.explained + 0.05) g = i;
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

/** The formula of an attached board or slide item, for its chip to typeset as Browse does ('' for speech, a moment,
 *  plain text, or an item the lecture no longer has: the chip shows its label). */
export function entryLatex(entry: TrayEntry, index: LectureIndex | null): string {
  const it = entry.item;
  if (it.kind !== 'piece' && it.kind !== 'line' && it.kind !== 'element') return '';
  return (index?.byRef.get(`#${it.id}`)?.latex ?? '').trim().replace(/^\$+|\$+$/g, '').trim();
}

/** Where an attached item's chip jumps, and so the time it shows: the lecture's own time for a board or slide item it
 *  has (a chip added from the Notes page, or saved before the jump rule changed, carries an older time), else the
 *  time saved with it. */
export function entryTime(entry: TrayEntry, index: LectureIndex | null): number {
  const it = entry.item;
  if (it.kind !== 'piece' && it.kind !== 'line' && it.kind !== 'element') return entry.t;
  return index?.byRef.get(`#${it.id}`)?.t ?? entry.t;
}

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

/** The lecture item a line of the study notes stands for: same reading, explained closest in time (a note's time is
 *  where its item is first talked about, app/notes.py). */
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
    if (!best || Math.abs(it.explained - note.t) < Math.abs(best.explained - note.t)) best = it;
  }
  return best;
}
