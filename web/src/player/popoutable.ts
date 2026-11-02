// Which board pieces get a "View in 3D" button: drawings (graphs, diagrams, sketches...) and equations that
// define a graph, y = f(x) or z = f(x, y). Slides never do. And which drawings already have a 3D view made (the
// server's GET /popouts), for the board's "3D" badges and its "Watch in 3D" pill.
import type { Box, Piece } from '../types';

const DRAWING = /\b(graph|diagram|figure|plot|sketch|drawing|chart|curve)s?\b/i;
const EQUATION = /\b(equation|formula|function)s?\b/i;

/** "y = |ln x|", "z = x^2 + y^2", "f(x) = e^{-x}", "f(x, y) = ..." (LaTeX or plain), with x on the right. */
export function looksLikeGraphEquation(s: string): boolean {
  const flat = s
    .replace(/\$/g, '')
    .replace(/\\(left|right|displaystyle|,|;|!|quad)/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  const m = /^(?:y|z|[fgh]\s*\(\s*x\s*(?:,\s*y\s*)?\))\s*=\s*(.+)$/i.exec(flat);
  if (!m) return false;
  // the right-hand side (up to a second "=", as in y = |ln x| = |log_e x|) must use x
  const rhs = m[1].split('=')[0];
  return /x/i.test(rhs.replace(/\\[a-z]+/gi, ' '));
}

export function canPopout(p: Piece | null | undefined): boolean {
  const read = p?.read;
  if (!read) return false;
  const kind = read.kind ?? '';
  if (DRAWING.test(kind)) return true;
  if (!EQUATION.test(kind)) return false;
  return looksLikeGraphEquation(read.latex ?? '') || looksLikeGraphEquation(read.text ?? '');
}

/** A drawing whose 3D view is already made (GET /api/jobs/{id}/popouts). */
export interface MadeScene {
  /** The board piece the view was made for. */
  drawing: number;
  pieces: number[];
  title: string;
  steps: number;
  synced: boolean;
  /** The steps' lecture times when synced (sorted), else []. */
  times: number[];
}

/** Seconds before a made scene's first step and after its last one that the board offers "Watch in 3D". */
export const PILL_BEFORE = 1;
export const PILL_AFTER = 5;

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Reads the server's list of made scenes defensively: anything malformed is left out; an answer that is not the
 *  list (an older server's error page) is no list. Never throws. */
export function cleanMadeScenes(raw: unknown): MadeScene[] {
  const list = raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as { drawings?: unknown }).drawings : null;
  if (!Array.isArray(list)) return [];
  const out: MadeScene[] = [];
  for (const d of list) {
    if (!d || typeof d !== 'object') continue;
    const r = d as Record<string, unknown>;
    const drawing = num(r.drawing);
    if (drawing === null || out.some((m) => m.drawing === Math.round(drawing))) continue;
    const synced = r.synced === true;
    const times = synced && Array.isArray(r.times) ? r.times.map(num).filter((t): t is number => t !== null).sort((a, b) => a - b) : [];
    out.push({
      drawing: Math.round(drawing),
      pieces: Array.isArray(r.pieces) ? r.pieces.map(num).filter((p): p is number => p !== null).map(Math.round) : [],
      title: typeof r.title === 'string' ? r.title.replace(/\s+/g, ' ').trim().slice(0, 120) : '',
      steps: Math.max(0, Math.round(num(r.steps) ?? 0)),
      synced: synced && times.length > 0,
      times,
    });
  }
  return out;
}

/** The made scenes the lecture is explaining at this clock (from PILL_BEFORE s before the first step to PILL_AFTER s
 *  after the last): the board offers "Watch in 3D" by them. */
export function explainedAt(made: readonly MadeScene[], clock: number): MadeScene[] {
  if (!Number.isFinite(clock)) return [];
  return made.filter((m) => m.synced && m.times.length > 0 && clock >= m.times[0] - PILL_BEFORE && clock <= m.times[m.times.length - 1] + PILL_AFTER);
}

/** A made drawing is on the board at a moment when any of its pieces is: the saddle's drawing (piece 23, its label)
 *  is written at 40.7 s, after three of its five steps, while its curves (pieces 8, 12, 17) are there from 13 s. */
export function drawingOnBoard(m: Pick<MadeScene, 'drawing' | 'pieces'>, isVisible: (id: number) => boolean): boolean {
  return isVisible(m.drawing) || m.pieces.some((id) => id !== m.drawing && isVisible(id));
}

/** The area two boxes share (0 when they do not meet). */
export function boxOverlap(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
}

/** Where a made view's badge goes: the box it takes (board pixels), and the edge it is placed by ("right": its right
 *  edge, so text a little wider than measured grows to the left; "left": beside the drawing's right side, away from
 *  it). */
export interface BadgeSpot {
  box: Box;
  align: 'left' | 'right';
}

/**
 * Where a made view's badge goes on the board (board pixels): beside the drawing where it covers the least writing,
 * never off the board. `badge` is its size (the board places the "3D" badge and the wider "Watch in 3D" pill each);
 * `ink(box)` says how much writing a box covers. Tried in this order, the first of equals kept: just above the drawing's top-right corner, beside that
 * corner, under the drawing, beside its top-left corner, and inside its top-right corner (where it used to be: on
 * test2 #4 it covered the edge label "b").
 */
export function badgeSpot(box: Box, board: [number, number], badge: [number, number], gap: number, ink: (b: Box) => number): BadgeSpot {
  const [x, y, w, h] = box;
  const [W, H] = board;
  const [bw, bh] = badge;
  const inside: BadgeSpot = {
    box: [Math.max(0, Math.min(x + w - gap - bw, W - bw)), Math.max(0, Math.min(y + gap, H - bh)), bw, bh],
    align: 'right',
  };
  const tries: BadgeSpot[] = (
    [
      { box: [x + w - bw, y - gap - bh, bw, bh], align: 'right' },
      { box: [x + w + gap, y, bw, bh], align: 'left' },
      { box: [x + w - bw, y + h + gap, bw, bh], align: 'right' },
      { box: [x - gap - bw, y, bw, bh], align: 'right' },
    ] as BadgeSpot[]
  ).filter(({ box: b }) => b[0] >= 0 && b[1] >= 0 && b[0] + bw <= W && b[1] + bh <= H);
  tries.push(inside);
  let best = inside;
  let least = Infinity;
  for (const t of tries) {
    const n = ink(t.box);
    if (n < least) {
      least = n;
      best = t;
    }
  }
  return best;
}

/** Where a made view's "3D" badge goes, and its "Watch in 3D" pill (shown while the lecture explains the drawing):
 *  the pill is written out (`long`) only where it covers no more writing than the badge; else it keeps the badge's
 *  size and place, lit up (test2 #4: the pill above the monoclinic cell covered "SIMPLE CUBIC" for three minutes). */
export function badgePlaces(
  box: Box,
  board: [number, number],
  badge: [number, number],
  pill: [number, number],
  gap: number,
  ink: (b: Box) => number,
): { badge: BadgeSpot; pill: BadgeSpot; long: boolean } {
  const b = badgeSpot(box, board, badge, gap, ink);
  const p = badgeSpot(box, board, pill, gap, ink);
  const long = ink(p.box) <= ink(b.box);
  return { badge: b, pill: long ? p : b, long };
}
