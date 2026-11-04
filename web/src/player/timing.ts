// Pure timing rules for the player: when each piece of writing is drawn, faded, tagged and talked about,
// plus transcript/chapter lookups. No DOM here, so `npm test` can run it with plain Node.
import type { BoardLine, Chapter, Piece, Span, Word } from '../types';

/** Kinds that are drawings: they always stay as handwriting, even in typeset mode. */
export const DRAWING_KINDS = new Set([
  'diagram', 'graph', 'arrow', 'scribble', 'underline', 'box',
  'drawing', 'figure', 'sketch', 'plot', 'chart', 'circle', 'line', 'shape', 'picture', 'bracket',
]);

// Reveal progress is quantised to one step per level of the piece's drawing-order map (1..255, linear in time over
// its draw window, board.py piece_image): a level is drawn once the latest pixel it can hold is due, so no pixel is
// drawn before its time in the order map (which spreads each sample's new ink over the sample interval before it, so at
// most one capture step, 1/3 s, before the replay log shows it). 64 steps drew a 3-minute piece up to 2.6 s early.
export const STEPS = 255;
const FADE_OUT = 0.4; // seconds for an erased piece to fade away
const TAG_FADE = 0.25;
const REDRAW_MAX = 6; // longest redraw of writing that was hidden, seconds

export interface Schedule {
  /** Reveal runs from start (progress 0) to end (progress 1). Pre-existing pieces use -Infinity. */
  start: number;
  end: number;
  removedAt: number | null;
  /** "was hidden" tag window, or null when the piece was never hidden. */
  tagStart: number | null;
  tagEnd: number | null;
  /** When each level of the order map is due, linear over this window (absent: over start..end). */
  due?: readonly [number, number];
  /** A hidden piece that kept growing, or gained ink seen as it was written: levels 1..to are redrawn from start to
   *  end, the rest are drawn when due. */
  redraw?: { start: number; end: number; to: number };
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export function schedule(p: Piece): Schedule {
  const removedAt = num(p.removed_at);
  const tSeen = num(p.t_seen) ?? 0;
  const tShown = num(p.t_shown) ?? tSeen;
  // The real write window, when the backend gives one (the order map is linear in time over it).
  const d0 = Array.isArray(p.draw) ? num(p.draw[0]) : null;
  const d1 = Array.isArray(p.draw) ? num(p.draw[1]) : null;
  const draw = d0 != null && d1 != null && d1 >= d0 ? ([d0, d1] as const) : null;
  if (p.pre_existing) {
    return { start: -Infinity, end: -Infinity, removedAt, tagStart: null, tagEnd: null };
  }
  if (p.revealed) {
    // Redrawn when it was talked about, but never before it can have been written (written_after) and never
    // after the camera already shows it (t_seen): the board must not contradict the video beside it.
    const redraw = num(p.redraw_at);
    const after = num(p.written_after) ?? -Infinity;
    const start = redraw == null ? tSeen : Math.min(tSeen, Math.max(redraw, after));
    // Redrawn early: what the camera showed in the first REDRAW_MAX s after it came into view, and of that only the
    // levels the backend allows (redraw_to: ink whose spot was last seen blank before the redraw can start). Ink the
    // piece gained later (it kept growing), or saw written in view, is drawn when the replay shows it, never before.
    const cut = tSeen + REDRAW_MAX;
    let to = draw && draw[1] > cut ? levelIn(draw, cut) : STEPS;
    const allowed = num(p.redraw_to);
    if (draw && allowed != null) to = Math.min(to, Math.max(0, Math.floor(allowed)));
    const upto = draw ? (to >= STEPS ? draw[1] : Math.min(draw[1], draw[0] + ((draw[1] - draw[0]) * to) / (STEPS - 1))) : tShown;
    const span = draw ? Math.min(upto, cut) - draw[0] : tShown - tSeen;
    // As long as writing it takes (redraw_secs: its ink at a typical writing speed; the camera showed it at once), but
    // done soon after the camera shows it all.
    const write = num(p.redraw_secs);
    const asWritten = write == null ? 0 : Math.min(write, Math.min(upto, cut) + 0.6 - start);
    const dur = Math.min(REDRAW_MAX, Math.max(1.2, span + 0.6, asWritten));
    const s: Schedule = { start, end: start + dur, removedAt, tagStart: start, tagEnd: start + Math.max(3, dur + 1) };
    if (draw && to < STEPS) {
      s.due = draw;
      s.redraw = { start, end: start + dur, to };
      s.end = Math.max(start + dur, draw[1]);
    }
    return s;
  }
  if (draw) return { start: draw[0], end: Math.max(draw[1], draw[0] + 0.3), removedAt, tagStart: null, tagEnd: null };
  const start = tSeen - 0.7;
  const end = Math.max(tShown, tSeen - 0.4);
  return { start, end: Math.max(end, start + 0.05), removedAt, tagStart: null, tagEnd: null };
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** The highest order-map level (0..STEPS) all of whose pixels are due by t, for a map linear in time over [a, b]:
 *  board.py gives a pixel at fraction f the level 1 + round(254 f), so level v holds f up to (v - 0.5) / 254. */
function levelIn([a, b]: readonly [number, number], t: number): number {
  if (t <= a) return 0;
  if (t >= b) return STEPS;
  return Math.min(STEPS, Math.floor(((STEPS - 1) * (t - a)) / (b - a) + 0.5));
}

/** New partial frames a second, at most, for a piece being drawn (each is built from its pixels, sprites.ts): one per
 *  display frame at 60 Hz, so a short piece draws as smoothly as with continuous progress; a long one is limited by its
 *  255 levels instead. */
export const FRAME_HZ = 60;

/** Drawing progress 0..1 at time t: the share of the order map's levels drawn (a whole number of levels, see STEPS),
 *  as of the last 1/FRAME_HZ tick (never ahead of t). */
export function progressAt(s: Schedule, t: number): number {
  if (s.end === -Infinity) return 1;
  if (t <= s.start) return 0;
  if (t >= s.end) return 1;
  const tick = Math.floor(t * FRAME_HZ + 1e-9) / FRAME_HZ;
  let level = levelIn(s.due ?? [s.start, s.end], tick);
  if (s.redraw) {
    const r = s.redraw;
    level = Math.max(level, Math.floor(r.to * clamp01((tick - r.start) / (r.end - r.start)) + 1e-9));
  }
  return level >= STEPS ? 1 : level / STEPS;
}

/** Opacity 0..1 at time t (fades out over 0.4 s from removed_at). */
export function alphaAt(s: Schedule, t: number): number {
  if (s.removedAt == null || t < s.removedAt) return 1;
  if (t >= s.removedAt + FADE_OUT) return 0;
  return clamp01(1 - (t - s.removedAt) / FADE_OUT);
}

/** Half-erased writing: the backend sets removed_at once half of a piece's shown pixels are erased, but the teacher can
 *  leave the rest (test.mp4 #39: a "$1" he circles at about 300 s; 73% of the piece's drawn pixels are ever erased). A
 *  piece with a gone map erased less than this share of its drawn pixels keeps drawing what is left of it after
 *  removed_at; one erased this much or more fades whole at removed_at (never fading those leaves more residue than ink
 *  on test4). */
export const REMNANT_MAX = 0.9;

/** Whether a piece keeps what is left of it after removed_at. `erased`: the share of its drawn pixels ever erased
 *  (sprites.ts goneOf), null without a gone map (or before its sprite is loaded): then it fades whole, as before. */
export function keepsRemnant(erased: number | null | undefined): boolean {
  return typeof erased === 'number' && Number.isFinite(erased) && erased < REMNANT_MAX;
}

/** Opacity of a piece's handwriting at t. A remnant (keepsRemnant) stays: its erased pixels leave the frame as each
 *  is erased (sprites.ts FrameCache.frame), the rest stay drawn. Any other piece fades whole from removed_at (alphaAt).
 *  removed_at keeps its meaning everywhere else: the piece as a whole (its typeset reading, its finished line, its
 *  "was hidden" tag, a 3D view's badge) still goes by alphaAt. */
export function inkAlphaAt(s: Schedule, t: number, remnant: boolean): number {
  return remnant ? 1 : alphaAt(s, t);
}

/** How strongly a piece's handwriting is drawn: `ink` (inkAlphaAt) and `whole` (alphaAt) at t; `reading`: its typeset
 *  reading stands in for it (typeset mode); `hide`: how far a bigger reading that repeats it has replaced it; `lineHide`:
 *  how far its finished line has. A piece shown typeset draws no handwriting while its reading is up; a remnant's comes
 *  in as its reading fades at removed_at (the two never show the piece twice over). */
export function handwritingAlpha(ink: number, whole: number, reading: boolean, hide: number, lineHide: number): number {
  if (lineHide > 0.98) return 0;
  const a = reading ? Math.max(0, ink - whole) * (1 - hide) : ink;
  return a * (1 - lineHide);
}

/** Whether anything of a piece is on the board at t (drawn, and its handwriting not faded): what can be hovered.
 *  With remnant false, whether the piece as a whole is (the old rule: a 3D view's badge goes by it). */
export function visibleAt(s: Schedule, t: number, remnant: boolean): boolean {
  return progressAt(s, t) > 0 && inkAlphaAt(s, t, remnant) >= 0.05;
}

/** What a hover or click finds a piece by at t, so that only what is drawn is found: 'box' (a typeset reading, its own
 *  or its finished line's, anywhere in its box), 'ink' (its handwriting: the board looks for unerased ink near the
 *  point), or null (nothing of it is shown). `reading`, `hide`, `lineHide` as for handwritingAlpha (typeset mode). */
export function hitBy(
  s: Schedule, t: number, o: { remnant: boolean; reading: boolean; hide: number; lineHide: number },
): 'box' | 'ink' | null {
  if (!visibleAt(s, t, o.remnant)) return null;
  if (o.reading && o.hide > 0.95) return null; // a bigger reading that repeats it has replaced it (its remnant too)
  if ((o.reading && alphaAt(s, t) >= 0.05) || o.lineHide > 0.5) return 'box';
  return 'ink';
}

/** Opacity of the "was hidden" tag at time t. */
export function tagAt(s: Schedule, t: number): number {
  if (s.tagStart == null || s.tagEnd == null) return 0;
  if (t < s.tagStart || t > s.tagEnd) return 0;
  return clamp01(Math.min((t - s.tagStart) / TAG_FADE, (s.tagEnd - t) / TAG_FADE));
}

/** Quantised reveal step 0..STEPS (0 = nothing drawn, STEPS = the whole piece): the levels progressAt says are drawn,
 *  rounded down (a level is never drawn before it is due). */
export function stepFor(progress: number): number {
  if (progress <= 0) return 0;
  if (progress >= 1) return STEPS;
  return Math.min(STEPS - 1, Math.floor(progress * STEPS + 1e-6));
}

/** Highest order value (1..255) visible at a step. */
export function thresholdFor(step: number): number {
  return Math.round((step / STEPS) * 255);
}

/** Where a click on a piece seeks to: its first spoken time, else when it was redrawn/written. */
export function seekTarget(p: Piece, clipStart = 0): number {
  const starts = (p.spoken ?? []).map((sp) => sp.start).filter((v) => Number.isFinite(v));
  if (starts.length) return Math.max(clipStart, Math.min(...starts));
  const redraw = num(p.redraw_at);
  if (redraw != null) return Math.max(clipStart, redraw);
  return Math.max(clipStart, (num(p.t_seen) ?? 0) - 1);
}

/** 0..1: how strongly a piece (or slide element) is "being talked about" at t (soft 0.3 s edges). */
export function talkingAt(p: { spoken?: Span[] }, t: number): number {
  let best = 0;
  for (const sp of p.spoken ?? []) {
    if (t < sp.start - 0.3 || t > sp.end + 0.3) continue;
    best = Math.max(best, clamp01(Math.min((t - sp.start + 0.3) / 0.3, (sp.end + 0.3 - t) / 0.3)));
  }
  return best;
}

/** Kinds whose plain-text reading is usually a description ("small red and black marks"), not the writing. */
const VAGUE_KINDS = new Set(['other', 'unknown', 'mark', 'marks']);

/** Whether a piece is shown as typeset text/maths in typeset mode. */
export function isTypesettable(p: Piece): boolean {
  const r = p.read;
  if (!r) return false;
  const kind = (r.kind || '').toLowerCase();
  if (DRAWING_KINDS.has(kind)) return false;
  const hasLatex = Boolean(r.latex && r.latex.trim());
  if (VAGUE_KINDS.has(kind) && !hasLatex) return false;
  return hasLatex || Boolean(r.text && r.text.trim());
}

/** Index of the last word that has started by t, or -1. Words must be sorted by start. */
export function lastStartedWord(words: Word[], t: number): number {
  let lo = 0;
  let hi = words.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid].s <= t) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

/** The word being spoken at t (held for 0.35 s after it ends to bridge short gaps), or -1. */
export function activeWord(words: Word[], t: number): number {
  const i = lastStartedWord(words, t);
  if (i < 0) return -1;
  return t <= words[i].e + 0.35 ? i : -1;
}

/** Index of the current chapter at t, or -1 before the first one. Chapters must be sorted. */
export function chapterAt(chapters: Chapter[], t: number): number {
  let idx = -1;
  for (let i = 0; i < chapters.length; i++) if (chapters[i].t <= t + 0.05) idx = i;
  return idx;
}

/**
 * Assign each word (sorted) to a turn (sorted). The boundary between two turns is the middle of the
 * silence between them (aligned words can start a little before their turn's start), at most 0.5 s early.
 */
export function groupWords(turns: { start: number; end: number }[], words: Word[]): number[][] {
  const groups: number[][] = turns.map(() => []);
  if (!turns.length) return groups;
  const bound = turns.map((tu, j) => {
    if (j === 0) return -Infinity;
    const prevEnd = turns[j - 1].end;
    const mid = prevEnd < tu.start ? (prevEnd + tu.start) / 2 : tu.start - 0.15;
    return Math.max(mid, tu.start - 0.5);
  });
  let j = 0;
  for (let i = 0; i < words.length; i++) {
    while (j + 1 < turns.length && words[i].s >= bound[j + 1]) j++;
    groups[j].push(i);
  }
  return groups;
}

/** Paragraphs for a transcript that has words but no turns: break at pauses (or every ~60 words). */
export function turnsFromWords(words: Word[], pause = 1.2): { start: number; end: number; text: string }[] {
  const out: { start: number; end: number; text: string }[] = [];
  let cur: Word[] = [];
  const flush = () => {
    if (cur.length) out.push({ start: cur[0].s, end: cur[cur.length - 1].e, text: cur.map((w) => w.w).join(' ') });
    cur = [];
  };
  for (const w of words) {
    const prev = cur[cur.length - 1];
    if (prev && (w.s - prev.e > pause || cur.length >= 60)) flush();
    cur.push(w);
  }
  flush();
  return out;
}

/** Whether a piece's reading should be typeset: no for drawings, empty readings, and big merged blocks
 *  (a large box with a long reading is several things read as one; a paragraph of maths can't stand in for it). */
export function typesetFits(p: Piece, boardW: number, boardH: number): boolean {
  if (!isTypesettable(p)) return false;
  const r = p.read!;
  const len = (r.latex?.trim() || r.text?.trim() || '').length;
  const area = p.box[2] * p.box[3];
  return !(area > 0.1 * boardW * boardH && len > 60);
}

/** Share of box a's area that lies inside box b. */
function shareInside(a: Piece['box'], b: Piece['box']): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? (w * h) / Math.max(1, a[2] * a[3]) : 0;
}

/** Ids of the pieces drawn as typeset in typeset mode: typesetFits, minus merged blocks (a long reading whose box
 *  holds another piece is several things written at different times and read as one). */
export function typesetIds(pieces: Piece[], boardW: number, boardH: number): Set<number> {
  const out = new Set<number>();
  for (const p of pieces) {
    if (!typesetFits(p, boardW, boardH)) continue;
    const len = (p.read?.latex?.trim() || p.read?.text?.trim() || '').length;
    if (len > 40 && pieces.some((o) => o !== p && shareInside(o.box, p.box) >= 0.5)) continue;
    out.add(p.id);
  }
  return out;
}

/** Taller than this (pixels of a 960-wide board, about four lines of handwriting) a "line" is not one line. */
const LINE_MAX_H = 100;

/**
 * Whether a group of pieces from the backend is one written line, to be typeset (and read in the tooltip) whole.
 * Pieces chained together through a big block of writing can come back as a "line" covering most of the board,
 * with one short reading: typeset, it would replace all their writing. Its pieces are then shown on their own.
 */
export function isWrittenLine(line: BoardLine, boardW: number): boolean {
  const b = line?.box;
  if (!Array.isArray(b) || b.length < 4 || !Array.isArray(line.pieces)) return false;
  const u = Math.max(0.25, (boardW || 960) / 960);
  return b[3] <= LINE_MAX_H * u;
}

export function formatTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}
