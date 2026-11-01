// Pure timing rules for the player: when each piece of writing is drawn, faded, tagged and talked about,
// plus transcript/chapter lookups. No DOM here, so `npm test` can run it with plain Node.
import type { Chapter, Piece, Span, Word } from '../types';

/** Kinds that are drawings: they always stay as handwriting, even in typeset mode. */
export const DRAWING_KINDS = new Set([
  'diagram', 'graph', 'arrow', 'scribble', 'underline', 'box',
  'drawing', 'figure', 'sketch', 'plot', 'chart', 'circle', 'line', 'shape', 'picture', 'bracket',
]);

export const STEPS = 64; // reveal progress is quantised to this many frames per piece
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
    // Capped: a hidden piece that kept growing for minutes (t_shown far after t_seen) is still redrawn briskly.
    const span = draw ? draw[1] - draw[0] : tShown - tSeen;
    const dur = Math.min(REDRAW_MAX, Math.max(1.2, span + 0.6));
    return { start, end: start + dur, removedAt, tagStart: start, tagEnd: start + Math.max(3, dur + 1) };
  }
  if (draw) return { start: draw[0], end: Math.max(draw[1], draw[0] + 0.3), removedAt, tagStart: null, tagEnd: null };
  const start = tSeen - 0.7;
  const end = Math.max(tShown, tSeen - 0.4);
  return { start, end: Math.max(end, start + 0.05), removedAt, tagStart: null, tagEnd: null };
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** Drawing progress 0..1 at time t. */
export function progressAt(s: Schedule, t: number): number {
  if (s.end === -Infinity) return 1;
  if (t <= s.start) return 0;
  if (t >= s.end) return 1;
  return clamp01((t - s.start) / (s.end - s.start));
}

/** Opacity 0..1 at time t (fades out over 0.4 s from removed_at). */
export function alphaAt(s: Schedule, t: number): number {
  if (s.removedAt == null || t < s.removedAt) return 1;
  if (t >= s.removedAt + FADE_OUT) return 0;
  return clamp01(1 - (t - s.removedAt) / FADE_OUT);
}

/** Opacity of the "was hidden" tag at time t. */
export function tagAt(s: Schedule, t: number): number {
  if (s.tagStart == null || s.tagEnd == null) return 0;
  if (t < s.tagStart || t > s.tagEnd) return 0;
  return clamp01(Math.min((t - s.tagStart) / TAG_FADE, (s.tagEnd - t) / TAG_FADE));
}

/** Quantised reveal step 0..STEPS (0 = nothing drawn, STEPS = the whole piece). */
export function stepFor(progress: number): number {
  if (progress <= 0) return 0;
  if (progress >= 1) return STEPS;
  return Math.max(1, Math.min(STEPS, Math.round(progress * STEPS)));
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

export function formatTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}
