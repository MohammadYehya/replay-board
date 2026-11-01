// Pure timing and geometry for the slide view: which slide state shows at t, the crossfade between states,
// when each pointer annotation is drawn, held and faded, the pulse rings, the pointer dot, and the SVG shapes.
// The board view draws the teacher's hand gestures with the same annotations (see "board annotations").
// Every animation is a function of the shared clock, so scrubbing shows exactly the frame for that time.
// No DOM here, so `npm test` can run it with plain Node.
import type {
  BoardData, BoardLine, Box, Piece, PointerSample, SlideAnnotation, SlideElement, SlideState,
} from '../types';

/** An annotation stays this long after the gesture ends, then fades out over FADE_OUT. */
export const HOLD = 1.5;
export const FADE_OUT = 0.4;
/** Fade-in of the spotlight's dimming; strokes and rings come in faster (they are drawn progressively). */
export const SPOT_FADE_IN = 0.25;
const STROKE_FADE_IN = 0.12;
const PULSE_FADE_IN = 0.2;
/** Shortest drawing time per type (a quick flick of the pointer still draws calmly). */
const DRAW_MIN: Record<string, number> = { circle: 0.6, underline: 0.35, arrow: 0.45 };
export const PULSE_PERIOD = 1.5;
export const PULSE_RINGS = 3;
/** The pointer dot shows only when a tracked sample is this close in time. */
export const POINTER_NEAR = 0.2;

export const ANNOTATION_TYPES = ['spotlight', 'circle', 'underline', 'arrow', 'pulse'] as const;
export type Kind = (typeof ANNOTATION_TYPES)[number];

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
export const easeOutCubic = (x: number) => 1 - (1 - clamp01(x)) ** 3;
export const easeInOutSine = (x: number) => -(Math.cos(Math.PI * clamp01(x)) - 1) / 2;
export const smoothstep = (x: number) => {
  const k = clamp01(x);
  return k * k * (3 - 2 * k);
};
const fin = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** Annotation types the player draws; anything else (a future type) is shown as a pulse. */
export function kindOf(a: SlideAnnotation): Kind {
  return (ANNOTATION_TYPES as readonly string[]).includes(a.type) ? (a.type as Kind) : 'pulse';
}

/** Annotations the player can draw: a box and a start time. */
export function drawable(a: SlideAnnotation | null | undefined): a is SlideAnnotation {
  return !!a && Array.isArray(a.box) && a.box.length >= 4 && fin(a.t_start);
}

export const toolOf = (tool: string | undefined): 'cursor' | 'laser' =>
  tool === 'laser' || tool === 'l' ? 'laser' : 'cursor';

/** The teacher's hand on a filmed board: a calm teal, apart from the cursor's amber and the laser's red. Deeper on a
 *  light board (a brighter teal fades into a grey-white board), brighter on dark chalk. */
export const HAND_ON_LIGHT = '#0d9488';
export const HAND_ON_DARK = '#2dd4bf';

/** Warm amber for the mouse cursor, red for a laser pointer, teal for the teacher's hand (light: the background). */
export const toolColour = (tool: string | undefined, light = true) =>
  tool === 'hand' ? (light ? HAND_ON_LIGHT : HAND_ON_DARK) : toolOf(tool) === 'laser' ? '#ef4444' : '#f59e0b';

// ---------------------------------------------------------------- slide states

/** When the change to this state began (its crossfade runs from here to t_start). */
export function changeOf(s: SlideState): number {
  return fin(s.t_change) ? Math.min(s.t_change, s.t_start) : s.t_start;
}

/** States in the order they appear. */
export function sortStates(states: SlideState[]): SlideState[] {
  return states.filter((s) => s && fin(s.t_start)).sort((a, b) => changeOf(a) - changeOf(b) || a.t_start - b.t_start);
}

/** Index of the state shown at t: the last one whose change has begun (the first one before any). -1 if none. */
export function stateIndexAt(states: SlideState[], t: number): number {
  if (!states.length) return -1;
  let lo = 0;
  let hi = states.length - 1;
  let ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (changeOf(states[mid]) <= t) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

/** During state i's change window [t_change, t_start) the previous state's image fades into the new one:
 *  prev is the index fading out (-1 when none) and mix the new image's opacity. */
export function crossfadeAt(states: SlideState[], i: number, t: number): { prev: number; mix: number } {
  if (i <= 0 || i >= states.length) return { prev: -1, mix: 1 };
  const s = states[i];
  const c = changeOf(s);
  if (t < c || t >= s.t_start || s.t_start - c < 1e-6) return { prev: -1, mix: 1 };
  return { prev: i - 1, mix: smoothstep((t - c) / (s.t_start - c)) };
}

/**
 * When the slide that state i shows goes away: its t_end, carried on through any following "addition" states
 * (a bullet appearing or pen ink keeps everything already on the slide in place), up to the next new slide.
 * States must be sorted. NaN for a bad index.
 */
export function slideEndAt(states: SlideState[], i: number): number {
  if (!(i >= 0 && i < states.length)) return NaN;
  let j = i;
  while (j + 1 < states.length && states[j + 1].kind === 'addition') j++;
  return states[j].t_end;
}

// ---------------------------------------------------------------- annotations

export interface AnnotationTiming {
  start: number;
  /** When the pointer gesture ended. */
  gestureEnd: number;
  /** When the progressive drawing completes. */
  drawEnd: number;
  /** When the fade-out begins. */
  holdEnd: number;
  /** Fully gone. */
  end: number;
  fadeIn: number;
}

/**
 * Visible from t_start until t_end + HOLD, then fading out over FADE_OUT. An annotation never outlives its
 * slide: if the slide goes earlier (stateEnd, from slideEndAt: when the change to the next slide begins) it
 * starts fading there instead.
 */
export function annotationTiming(a: SlideAnnotation, stateEnd?: number | null): AnnotationTiming {
  const kind = kindOf(a);
  const start = fin(a.t_start) ? a.t_start : 0;
  const gestureEnd = Math.max(fin(a.t_end) ? a.t_end : start, start);
  const drawEnd = Math.max(gestureEnd, start + (DRAW_MIN[kind] ?? 0));
  let holdEnd = gestureEnd + HOLD;
  if (fin(stateEnd) && stateEnd > start) holdEnd = Math.min(holdEnd, stateEnd);
  holdEnd = Math.max(holdEnd, start);
  const fadeIn = kind === 'spotlight' ? SPOT_FADE_IN : kind === 'pulse' ? PULSE_FADE_IN : STROKE_FADE_IN;
  return { start, gestureEnd, drawEnd, holdEnd, end: holdEnd + FADE_OUT, fadeIn };
}

/** Opacity 0..1 of the whole annotation at t (eased fade in, hold, eased fade out). */
export function annotationAlpha(tm: AnnotationTiming, t: number): number {
  if (!(t >= tm.start) || t >= tm.end) return 0;
  const fadeIn = tm.fadeIn > 0 ? easeOutCubic((t - tm.start) / tm.fadeIn) : 1;
  const fadeOut = t <= tm.holdEnd ? 1 : 1 - smoothstep((t - tm.holdEnd) / FADE_OUT);
  return Math.min(fadeIn, fadeOut);
}

/** How much of a circle / underline / arrow is drawn at t, 0..1 (eased like a hand: slow start and finish). */
export function drawProgress(tm: AnnotationTiming, t: number): number {
  if (!(t > tm.start)) return 0;
  if (t >= tm.drawEnd) return 1;
  return easeInOutSine((t - tm.start) / (tm.drawEnd - tm.start));
}

/** The arrowhead pops in (0..1) as the shaft reaches the target. */
export function arrowHeadAt(tm: AnnotationTiming, t: number): number {
  return easeOutCubic((t - (tm.drawEnd - 0.06)) / 0.22);
}

export interface Ring {
  /** 0 = hugging the box, 1 = fully expanded. */
  k: number;
  alpha: number;
}

/**
 * PULSE_RINGS rings, staggered by a third of a period, repeat while the gesture lasts (at least one round of
 * all rings); after that the rings in flight finish expanding. Always returns PULSE_RINGS entries
 * (alpha 0 = not showing).
 */
export function pulseRings(tm: AnnotationTiming, t: number): Ring[] {
  const out: Ring[] = [];
  const spawnEnd = Math.max(tm.gestureEnd, tm.start + (PULSE_PERIOD * (PULSE_RINGS - 1)) / PULSE_RINGS);
  for (let j = 0; j < PULSE_RINGS; j++) {
    const first = tm.start + (j * PULSE_PERIOD) / PULSE_RINGS;
    if (!(t >= first) || t >= tm.end) {
      out.push({ k: 0, alpha: 0 });
      continue;
    }
    const n = Math.floor((Math.min(t, spawnEnd) - first) / PULSE_PERIOD);
    const phase = (t - (first + Math.max(0, n) * PULSE_PERIOD)) / PULSE_PERIOD;
    if (phase >= 1) {
      out.push({ k: 1, alpha: 0 });
      continue;
    }
    out.push({ k: easeOutCubic(phase), alpha: Math.min(1, phase / 0.12) * (1 - phase) ** 1.6 });
  }
  return out;
}

// ---------------------------------------------------------------- pointer

export interface PointerDot {
  x: number;
  y: number;
  tool: 'cursor' | 'laser';
  alpha: number;
}

/** The teacher's pointer at t, interpolated between tracked samples; null when no sample is within
 *  POINTER_NEAR seconds (the pointer was not seen then). Samples must be sorted by time. */
export function pointerAt(samples: PointerSample[], t: number): PointerDot | null {
  if (!samples.length || !fin(t)) return null;
  let lo = 0;
  let hi = samples.length - 1;
  let i = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (samples[mid][0] <= t) {
      i = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  const a = i >= 0 ? samples[i] : null;
  const b = i + 1 < samples.length ? samples[i + 1] : null;
  const da = a ? t - a[0] : Infinity;
  const db = b ? b[0] - t : Infinity;
  const near = Math.min(da, db);
  if (near > POINTER_NEAR) return null;
  const alpha = 1 - smoothstep((near - POINTER_NEAR / 2) / (POINTER_NEAR / 2));
  if (a && b && a[1] === b[1] && b[0] > a[0] && b[0] - a[0] <= 2 * POINTER_NEAR) {
    const f = (t - a[0]) / (b[0] - a[0]);
    return { x: a[2] + (b[2] - a[2]) * f, y: a[3] + (b[3] - a[3]) * f, tool: toolOf(a[1]), alpha };
  }
  const s = da <= db ? a! : b!;
  return { x: s[2], y: s[3], tool: toolOf(s[1]), alpha };
}

// ---------------------------------------------------------------- elements

/** Where a click on a slide element seeks to: its first spoken time, else when its slide state starts. */
export function elementSeekTime(el: SlideElement, state: SlideState | undefined, clipStart = 0): number {
  const starts = (el.spoken ?? []).map((s) => s.start).filter(fin);
  const t = starts.length ? Math.min(...starts) : state ? state.t_start : clipStart;
  return Math.max(clipStart, t);
}

/** The element under slide point (x, y): the smallest box that holds it (a label inside a chart wins). */
export function pickElement(elements: SlideElement[], x: number, y: number, pad = 3): SlideElement | null {
  let best: SlideElement | null = null;
  let bestArea = Infinity;
  for (const el of elements) {
    const [bx, by, bw, bh] = el.box;
    if (x < bx - pad || x > bx + bw + pad || y < by - pad || y > by + bh + pad) continue;
    const area = Math.max(1, bw * bh);
    if (area < bestArea) {
      best = el;
      bestArea = area;
    }
  }
  return best;
}

export function isLightColour(c: string | undefined): boolean {
  const m = (c ?? '').trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (!m) return true; // slides are usually light
  const h = m[1].length === 3 ? m[1].replace(/./g, (x) => x + x) : m[1];
  const [r, g, b] = [0, 2, 4].map((k) => parseInt(h.slice(k, k + 2), 16));
  return 0.299 * r + 0.587 * g + 0.114 * b > 120;
}

// ---------------------------------------------------------------- geometry (slide pixels)

export type Pt = [number, number];
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
  r: number;
}

/** Slide pixels per pixel of a 960-wide slide: strokes and paddings scale with the slide. */
export const unitOf = (size: [number, number]) => Math.max(0.25, (size[0] || 960) / 960);
/** Stroke width of the drawn annotations, in slide pixels. */
export const strokeOf = (size: [number, number]) => 3.4 * unitOf(size);

const f1 = (v: number) => (Math.round(v * 10) / 10).toString();
const pt = (p: Pt) => `${f1(p[0])} ${f1(p[1])}`;

/**
 * Where a drawn circle starts (angle on the ellipse, screen coordinates: y down) and which way it turns
 * (1 = clockwise on screen), taken from the teacher's own loop so the drawing follows the pointer in the video.
 * Without a usable path: upper left, clockwise, like most hands.
 */
export function circleStart(path: Pt[] | undefined, cx: number, cy: number, rx: number, ry: number): { a0: number; dir: 1 | -1 } {
  const fallback = { a0: (-140 * Math.PI) / 180, dir: 1 as const };
  const pts = (path ?? []).filter((p) => Array.isArray(p) && fin(p[0]) && fin(p[1]));
  if (pts.length < 4 || !(rx > 0) || !(ry > 0)) return fallback;
  let turn = 0; // swept angle around the centre, summed
  let prev = Math.atan2((pts[0][1] - cy) / ry, (pts[0][0] - cx) / rx);
  for (let i = 1; i < pts.length; i++) {
    const a = Math.atan2((pts[i][1] - cy) / ry, (pts[i][0] - cx) / rx);
    let d = a - prev;
    if (d > Math.PI) d -= 2 * Math.PI;
    if (d < -Math.PI) d += 2 * Math.PI;
    turn += d;
    prev = a;
  }
  if (Math.abs(turn) < Math.PI / 2) return fallback; // not a loop around this box
  return { a0: Math.atan2((pts[0][1] - cy) / ry, (pts[0][0] - cx) / rx), dir: turn > 0 ? 1 : -1 };
}

/** A hand-drawn ellipse around box (~10 px clear of it): one slightly tilted loop that overshoots its start a
 *  little, as a pen does, starting where and turning the way the teacher's gesture did. Drawn with
 *  pathLength=1, so stroke-dashoffset 1 → 0 draws it. */
export function circlePath(box: Box, size: [number, number], gesture?: Pt[]): string {
  const u = unitOf(size);
  const [x, y, w, h] = box;
  const pad = 10 * u;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const rx = (w / 2) * 1.08 + pad;
  const ry = (h / 2) * 1.25 + pad;
  const tilt = ((-2.2 * Math.PI) / 180) * Math.min(1, (2 * ry) / rx); // flat ellipses barely tilt
  const { a0, dir } = circleStart(gesture, cx, cy, rx, ry);
  const turns = 1.07;
  const N = 120;
  const parts: string[] = [];
  for (let i = 0; i <= N; i++) {
    const f = i / N;
    const a = a0 + dir * f * turns * 2 * Math.PI;
    const g = 0.985 + 0.045 * f; // starts a hair inside, ends a hair outside: the ends overlap cleanly
    const ex = rx * g * Math.cos(a);
    const ey = ry * g * Math.sin(a);
    const p: Pt = [cx + ex * Math.cos(tilt) - ey * Math.sin(tilt), cy + ex * Math.sin(tilt) + ey * Math.cos(tilt)];
    parts.push(`${i ? 'L' : 'M'}${pt(p)}`);
  }
  return parts.join(' ');
}

/** A slightly hand-drawn line just under box (or at underlineY), left to right. */
export function underlinePath(box: Box, size: [number, number], underlineY?: number | null): string {
  const u = unitOf(size);
  const [x, y, w, h] = box;
  const y0 = fin(underlineY) ? underlineY : Math.min(size[1] - 3 * u, y + h + 5 * u);
  const x0 = x - 3 * u;
  const x1 = x + w + 3 * u;
  const rise = Math.min(4 * u, (x1 - x0) * 0.008);
  const sag = 2.2 * u;
  return `M${pt([x0, y0])} Q${pt([(x0 + x1) / 2, y0 + sag])} ${pt([x1, y0 - rise])}`;
}

/**
 * Height of an underline: underline_y when the backend gives it; else just under the teacher's measured sweep,
 * when that ran along the lower half of the box or a little below it (a box read off the slide can sit a line
 * gap above the ink, while the pointer track is measured); else just under the box.
 */
export function underlineY(a: SlideAnnotation, size: [number, number]): number {
  if (fin(a.underline_y)) return a.underline_y;
  const u = unitOf(size);
  const [, y, , h] = a.box;
  const ys = (a.path ?? [])
    .filter((p) => Array.isArray(p) && fin(p[1]))
    .map((p) => p[1])
    .sort((p, q) => p - q);
  if (ys.length >= 2) {
    const mid = ys[ys.length >> 1];
    if (mid >= y + h * 0.5 && mid <= y + h + Math.max(30 * u, 1.5 * h)) return Math.min(size[1] - 3 * u, mid + 2 * u);
  }
  return Math.min(size[1] - 3 * u, y + h + 5 * u);
}

const inside = (p: Pt, b: Box) => p[0] >= b[0] && p[0] <= b[0] + b[2] && p[1] >= b[1] && p[1] <= b[1] + b[3];

/** Where the segment p → q leaves box (p inside it), moved `gap` further along; null if q is inside too. */
export function exitPoint(p: Pt, q: Pt, box: Box, gap: number): Pt | null {
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return null;
  const [x, y, w, h] = box;
  const tx = dx > 0 ? (x + w - p[0]) / dx : dx < 0 ? (x - p[0]) / dx : Infinity;
  const ty = dy > 0 ? (y + h - p[1]) / dy : dy < 0 ? (y - p[1]) / dy : Infinity;
  const tExit = Math.min(tx, ty) + gap / len;
  if (!(tExit < 1)) return null;
  return [p[0] + dx * tExit, p[1] + dy * tExit];
}

export interface ArrowGeom {
  d: string; // the shaft, a gentle curve (pathLength=1)
  head: string; // the two wings of the arrowhead
  tip: Pt;
  from: Pt;
  ctrl: Pt; // the shaft's control point
  wings: [Pt, Pt]; // ends of the head's wings
}

/** The boxes an arrow points from and to, and (board) the other writing it should keep off. */
export interface ArrowBoxes {
  from: Box | null;
  to: Box | null;
  obstacles?: readonly Box[];
}

/**
 * A curved arrow from `from` to `to`. When an end sits inside its element's box (the backend points at element
 * centres), it is pulled back to just outside that box, so the arrow points at the text instead of covering it.
 * With `obstacles` (the board's other writing) an arrow that would run over them goes round instead (keepOff).
 */
export function arrowGeometry(
  from: Pt,
  to: Pt,
  size: [number, number],
  fromBox?: Box | null,
  toBox?: Box | null,
  obstacles?: readonly Box[] | null,
): ArrowGeom {
  const g = directArrow(from, to, size, fromBox, toBox);
  return obstacles?.length ? keepOff(g, from, size, fromBox, toBox, obstacles) : g;
}

function directArrow(from: Pt, to: Pt, size: [number, number], fromBox?: Box | null, toBox?: Box | null): ArrowGeom {
  const u = unitOf(size);
  let a: Pt = from;
  let b: Pt = to;
  if (fromBox && inside(a, fromBox)) a = exitPoint(a, b, fromBox, 5 * u) ?? a;
  if (toBox && inside(b, toBox)) b = exitPoint(b, a, toBox, 6 * u) ?? b;
  if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 36 * u) {
    // Too short to point from one box to the other: between two lines stacked close, curve round their ends
    // (a straight arrow would sit on their writing); else point from / to as given.
    if (fromBox && toBox && stacked(fromBox, toBox)) return sideArrow(fromBox, toBox, size);
    a = from;
    b = to;
  }
  return bowArrow(a, b, u);
}

/** A quadratic arrow a → b bowed upwards (downwards with `down`) by k of its length, at most cap (960-wide px). */
function bowArrow(a: Pt, b: Pt, u: number, k = 0.14, cap = 50, down = false): ArrowGeom {
  return arrowFrom(a, bowCtrl(a, b, u, k, cap, down), b, Math.max(1e-6, Math.hypot(b[0] - a[0], b[1] - a[1])), u);
}

/** The control point of bowArrow's curve. */
function bowCtrl(a: Pt, b: Pt, u: number, k: number, cap: number, down: boolean): Pt {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.max(1e-6, Math.hypot(dx, dy));
  let nx = dy / len;
  let ny = -dx / len;
  if (ny > 0 || (ny === 0 && nx < 0)) {
    nx = -nx;
    ny = -ny; // always bow upwards
  }
  const bend = Math.min(len * k, cap * u) * (down ? -1 : 1);
  return [(a[0] + b[0]) / 2 + nx * bend, (a[1] + b[1]) / 2 + ny * bend];
}

/** Where an arrow runs, without its SVG: tail, control point, tip and the ends of its head's wings. */
type ArrowShape = Pick<ArrowGeom, 'from' | 'ctrl' | 'tip' | 'wings'>;

/**
 * Visits the points along an arrow as drawn, until `stop` returns true: its shaft, about every 3 px, and the wings of
 * its head. In no particular order: the head first, then the shaft from the tip back, every 8th point before the ones
 * between (writing in the way shows up soonest).
 */
function eachArrowPoint(g: ArrowShape, u: number, stop: (x: number, y: number) => boolean): void {
  const [a, c, b] = [g.from, g.ctrl, g.tip];
  const len = Math.hypot(c[0] - a[0], c[1] - a[1]) + Math.hypot(b[0] - c[0], b[1] - c[1]);
  const n = Math.min(200, Math.max(16, Math.ceil(len / (3 * u))));
  for (const w of g.wings) {
    for (let i = 0; i < 4; i++) if (stop(w[0] + ((b[0] - w[0]) * i) / 4, w[1] + ((b[1] - w[1]) * i) / 4)) return;
  }
  for (const k of [0, 4, 2, 6, 1, 5, 3, 7]) {
    for (let i = n - k; i >= 0; i -= 8) {
      const s = i / n;
      const p = (1 - s) ** 2;
      const q = 2 * s * (1 - s);
      const r = s * s;
      if (stop(p * a[0] + q * c[0] + r * b[0], p * a[1] + q * c[1] + r * b[1])) return;
    }
  }
}

const offBoard = (p: Pt, size: [number, number]) => p[0] < 0 || p[0] > size[0] || p[1] < 0 || p[1] > size[1];

/** Whether part of an arrow leaves the board (the layer would cut its head off at the edge). */
function leavesBoard(c: ArrowShape, size: [number, number]): boolean {
  // The shaft stays inside the triangle of its ends and control point, the head inside its wings.
  if ([c.from, c.ctrl, c.tip, c.wings[0], c.wings[1]].every((p) => !offBoard(p, size))) return false;
  let off = false;
  eachArrowPoint(c, unitOf(size), (x, y) => (off = offBoard([x, y], size)));
  return off;
}

/**
 * How many of an arrow's points (eachArrowPoint) are off the board or within 2 units of one of `boxes`. Counting
 * stops at `limit` (1: is it clear at all?).
 */
function hitCount(c: ArrowShape, size: [number, number], boxes: readonly Box[], limit = Infinity): number {
  const u = unitOf(size);
  const s = 2 * u;
  const [W, H] = size;
  // The shaft stays inside the triangle of its ends and control point, the head inside its wings: only the boxes
  // near those can hold one of its points (as [left, right, top, bottom] with the slack).
  const xs = [c.from[0], c.ctrl[0], c.tip[0], c.wings[0][0], c.wings[1][0]];
  const ys = [c.from[1], c.ctrl[1], c.tip[1], c.wings[0][1], c.wings[1][1]];
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const near: [number, number, number, number][] = [];
  for (const b of boxes) {
    const l = b[0] - s;
    const r = b[0] + b[2] + s;
    const t = b[1] - s;
    const d = b[1] + b[3] + s;
    if (l <= x1 && r >= x0 && t <= y1 && d >= y0) near.push([l, r, t, d]);
  }
  if (!near.length && x0 >= 0 && y0 >= 0 && x1 <= W && y1 <= H) return 0;
  let n = 0;
  eachArrowPoint(c, u, (x, y) => {
    let hit = x < 0 || x > W || y < 0 || y > H;
    for (let i = 0; !hit && i < near.length; i++) {
      const e = near[i];
      hit = x >= e[0] && x <= e[1] && y >= e[2] && y <= e[3];
    }
    return hit && ++n >= limit;
  });
  return n;
}

/**
 * An arrow on the board that runs over other writing (`obstacles`), or lands its tip on it, goes round instead:
 * between two lines one above the other, round their ends like sideArrow (its usual side first, bowing out less and
 * less, then the other side), with the tip beside the target's end; else bowed further up or down between its own
 * ends, and when none of those bows is clear, between other ends round its boxes (otherEnds). The first route clear
 * of the writing (and on the board) wins. If none is: round the ends of stacked lines, the route over the least
 * writing; an arrow whose tip is free and that only skims the edge of some writing, the arrow as it was; else
 * otherEnds' choice.
 */
function keepOff(
  g: ArrowGeom,
  from: Pt,
  size: [number, number],
  fromBox: Box | null | undefined,
  toBox: Box | null | undefined,
  obstacles: readonly Box[],
): ArrowGeom {
  const u = unitOf(size);
  let least = hitCount(g, size, obstacles);
  if (!least) return g;
  let best = g;
  if (fromBox && toBox && stacked(fromBox, toBox)) {
    const first = sideOf(fromBox, toBox, size);
    for (const side of [first, first === 'right' ? 'left' : 'right'] as const) {
      for (const k of [1, 0.7, 0.45, 0.25]) {
        const c = sideArrow(fromBox, toBox, size, side, k);
        const n = hitCount(c, size, obstacles);
        if (!n) return c;
        if (n < least) {
          best = c;
          least = n;
        }
      }
    }
    return best;
  }
  for (const k of [0.3, 0.5]) {
    for (const down of [false, true]) {
      const c = bowArrow(g.from, g.tip, u, k, 120, down);
      if (!hitCount(c, size, obstacles, 1)) return c;
    }
  }
  // Its tip on the writing, or running through it: other ends. Only skimming the edge of some (as the teacher's own
  // gesture may), it keeps its ends.
  const tipOn = obstacles.some((b) => within(g.tip, b, 2 * u));
  if (!tipOn && deepestIn(g, u, obstacles) <= 2 * u) return g;
  return otherEnds(g, from, size, fromBox, toBox, obstacles, least);
}

/** How far inside one of `boxes` the arrow's deepest point lies (0 or less when none is inside one). */
function deepestIn(c: ArrowShape, u: number, boxes: readonly Box[]): number {
  let deepest = -Infinity;
  eachArrowPoint(c, u, (x, y) => {
    for (const b of boxes) deepest = Math.max(deepest, Math.min(x - b[0], b[0] + b[2] - x, y - b[1], b[1] + b[3] - y));
    return false;
  });
  return deepest;
}

/** Shaft shapes tried between other ends: [bow k, cap, down, penalty in units]. The usual gentle bow (bowArrow's
 *  default) is the one to beat; a bigger bow also pays for its extra length. */
const END_SHAPES: readonly [number, number, boolean, number][] = [
  [0.14, 50, false, 0],
  [0, 50, false, 3], // straight
  [0.14, 50, true, 4],
  [0.3, 120, false, 6],
  [0.3, 120, true, 8],
];

/** How many routes otherEnds tries, closest first (on test-full, 99 arrows in 100 that have a clear route find it
 *  among the first 360), and when none is clear, how many of the closest it weighs. */
const MAX_ROUTES = 360;
const FALLBACK_ROUTES = 60;

/** A point an arrow could start or end at, and what it costs (in units) for not facing the other end. */
interface EndPt {
  p: Pt;
  pen: number;
}

/**
 * Points just outside each side of box (`gap` out): the side's middle and its two ends (a little in from the
 * corners), each with how badly that side faces away from `toward` (0 when it faces it squarely, up to 24 units).
 */
function sideEnds(box: Box, gap: number, toward: Pt, u: number): EndPt[] {
  const [x, y, w, h] = box;
  const dx = toward[0] - (x + w / 2);
  const dy = toward[1] - (y + h / 2);
  const d = Math.max(1e-6, Math.hypot(dx, dy));
  const pen = (nx: number, ny: number) => 24 * (1 - Math.max(0, (nx * dx + ny * dy) / d));
  const along = (len: number) => {
    const inset = Math.min(len / 4, 10 * u);
    return [inset, len / 2, len - inset];
  };
  const out: EndPt[] = [];
  for (const s of along(w)) {
    out.push({ p: [x + s, y - gap], pen: pen(0, -1) }, { p: [x + s, y + h + gap], pen: pen(0, 1) });
  }
  for (const s of along(h)) {
    out.push({ p: [x - gap, y + s], pen: pen(-1, 0) }, { p: [x + w + gap, y + s], pen: pen(1, 0) });
  }
  return out;
}

const dist = (p: Pt, q: Pt) => Math.hypot(q[0] - p[0], q[1] - p[1]);

/**
 * keepOff for an arrow whose boxes are not one above the other, when no bow between its own ends is clear (its tip
 * may even sit on someone else's writing): other ends, just outside the sides of the target's box (each side's middle
 * and ends) and of the box it comes from (likewise, or where the line from `from` to the new tip leaves it), joined
 * by a straight shaft or a bow either way. Each route must also keep off the two boxes themselves. The clear route
 * closest to the arrow as it was wins: short, its ends near the old ones, on the sides that face each other. If none
 * is clear, the one over the least writing, with its tip off the writing whenever some route allows it; the arrow as
 * it was stays unless its tip is on the writing or another route touches clearly less (a quarter less).
 */
function otherEnds(
  g: ArrowGeom,
  from: Pt,
  size: [number, number],
  fromBox: Box | null | undefined,
  toBox: Box | null | undefined,
  obstacles: readonly Box[],
  gHits: number,
): ArrowGeom {
  const u = unitOf(size);
  const free = (p: Pt) => !offBoard(p, size) && !obstacles.some((b) => within(p, b, 2 * u));
  const tips: EndPt[] = [{ p: g.tip, pen: 0 }];
  if (toBox) for (const e of sideEnds(toBox, 6 * u, g.from, u)) if (free(e.p)) tips.push(e);
  const tails: EndPt[] = [{ p: g.from, pen: 0 }];
  if (fromBox) for (const e of sideEnds(fromBox, 5 * u, g.tip, u)) if (free(e.p)) tails.push(e);
  const aim = fromBox && inside(from, fromBox) ? fromBox : null; // the box to aim out of

  const chord0 = dist(g.from, g.tip);
  const routes: { a: Pt; c: Pt; b: Pt; cost: number }[] = [];
  for (const t of tips) {
    const starts = tails.slice();
    if (aim && t !== tips[0]) {
      const e = exitPoint(from, t.p, aim, 5 * u); // from the writing straight towards the new tip
      if (e && free(e)) starts.push({ p: e, pen: 0 });
    }
    for (const s of starts) {
      const chord = dist(s.p, t.p);
      if (chord < 24 * u) continue; // shorter than a head and a bit of shaft
      // longer, moved, facing away, shaped otherwise: each costs
      const base = chord - chord0 + 0.5 * (dist(s.p, g.from) + dist(t.p, g.tip)) + (s.pen + t.pen) * u;
      let usual = NaN; // the length of the usual gentle bow between these ends (the first shape)
      for (const [k, cap, down, pen] of END_SHAPES) {
        const c = bowCtrl(s.p, t.p, u, k, cap, down);
        const len = dist(s.p, c) + dist(c, t.p);
        if (Number.isNaN(usual)) usual = len;
        if (s === tails[0] && t === tips[0] && k === 0.14 && !down) continue; // the arrow as it was
        routes.push({ a: s.p, c, b: t.p, cost: base + Math.max(0, len - usual) + pen * u });
      }
    }
  }
  routes.sort((p, q) => p.cost - q.cost);
  routes.length = Math.min(routes.length, MAX_ROUTES);
  const boxes = [...obstacles, ...[fromBox, toBox].filter((b): b is Box => !!b)];
  const shape = (r: (typeof routes)[number]) => arrowShape(r.a, r.c, r.b, dist(r.a, r.b), u);
  for (const r of routes) {
    const c = shape(r);
    if (!hitCount(c, size, boxes, 1)) return arrowGeom(c);
  }

  // Nothing clear: the least writing, the tip off it if possible, the arrow as it was unless clearly worse. Only the
  // routes closest to the arrow are weighed (and then the first one with its tip off the writing, if none had it).
  const tipOn = (c: ArrowShape) => obstacles.some((b) => within(c.tip, b, 2 * u));
  let best: ArrowShape = g;
  let bestOn = tipOn(g);
  let bestHits = gHits;
  for (const [i, r] of routes.entries()) {
    if (i >= FALLBACK_ROUTES && !bestOn) break;
    const c = shape(r);
    if (leavesBoard(c, size)) continue; // its head would be cut off at the edge
    const on = tipOn(c);
    if (on && !bestOn) continue;
    const limit = !on && bestOn ? Infinity : best === g ? Math.ceil(0.75 * gHits) : bestHits;
    const n = hitCount(c, size, boxes, limit);
    if (n < limit) {
      best = c;
      bestOn = on;
      bestHits = n;
    }
  }
  return best === g ? g : arrowGeom(best);
}

/** Two boxes one above the other: side by side across, apart (or just touching) down. */
export function stacked(p: Box, q: Box): boolean {
  const across = Math.min(p[0] + p[2], q[0] + q[2]) - Math.max(p[0], q[0]);
  const down = Math.min(p[1] + p[3], q[1] + q[3]) - Math.max(p[1], q[1]);
  return across > 0.3 * Math.min(p[2], q[2]) && down <= 0.15 * Math.min(p[3], q[3]);
}

/**
 * An arrow between two lines stacked close (the result under the formula it came from): from the end of one line
 * it bows out beside them and comes back into the end of the other, like a hand-drawn ⤴. On the right, or on the
 * left when the right has no room for it (or on `side`, bowing out k of the usual amount).
 */
export function sideArrow(fromBox: Box, toBox: Box, size: [number, number], side?: 'right' | 'left', k = 1): ArrowGeom {
  const u = unitOf(size);
  const gap = 5 * u;
  const yA = fromBox[1] + fromBox[3] / 2;
  const yB = toBox[1] + toBox[3] / 2;
  const bulge = sideBulge(fromBox, toBox, u) * k;
  const onRight = (side ?? sideOf(fromBox, toBox, size)) === 'right';
  const a: Pt = onRight ? [fromBox[0] + fromBox[2] + gap, yA] : [fromBox[0] - gap, yA];
  const b: Pt = onRight ? [toBox[0] + toBox[2] + gap, yB] : [toBox[0] - gap, yB];
  const c: Pt = [onRight ? Math.max(a[0], b[0]) + bulge : Math.min(a[0], b[0]) - bulge, (yA + yB) / 2];
  return arrowFrom(a, c, b, Math.hypot(b[0] - a[0], b[1] - a[1]), u);
}

/** How far a sideArrow bows out beside the lines. */
function sideBulge(fromBox: Box, toBox: Box, u: number): number {
  const dy = toBox[1] + toBox[3] / 2 - (fromBox[1] + fromBox[3] / 2);
  return Math.min(40 * u, Math.max(16 * u, 0.6 * Math.abs(dy)));
}

/** The side a sideArrow goes round: the right, else the side with more room. */
function sideOf(fromBox: Box, toBox: Box, size: [number, number]): 'right' | 'left' {
  const u = unitOf(size);
  const gap = 5 * u;
  const bulge = sideBulge(fromBox, toBox, u);
  const right = Math.max(fromBox[0] + fromBox[2], toBox[0] + toBox[2]) + gap + bulge;
  const left = Math.min(fromBox[0], toBox[0]) - gap - bulge;
  return right <= size[0] - 4 * u || size[0] - right > left ? 'right' : 'left';
}

/** A quadratic arrow a → b through control c, with its head along the curve's end tangent. */
function arrowFrom(a: Pt, c: Pt, b: Pt, len: number, u: number): ArrowGeom {
  return arrowGeom(arrowShape(a, c, b, len, u));
}

/** The SVG of an arrow's shape. */
function arrowGeom(s: ArrowShape): ArrowGeom {
  const { from: a, ctrl: c, tip: b, wings } = s;
  return {
    d: `M${pt(a)} Q${pt(c)} ${pt(b)}`,
    head: `M${pt(wings[0])} L${pt(b)} L${pt(wings[1])}`,
    tip: b,
    from: a,
    ctrl: c,
    wings,
  };
}

/** arrowFrom without the SVG (the head's length grows with len, the distance a → b). */
function arrowShape(a: Pt, c: Pt, b: Pt, len: number, u: number): ArrowShape {
  const tx = b[0] - c[0];
  const ty = b[1] - c[1];
  const tl = Math.max(1e-6, Math.hypot(tx, ty));
  const ux = tx / tl;
  const uy = ty / tl;
  const hl = Math.min(17 * u, Math.max(11 * u, len * 0.1));
  const ang = (27 * Math.PI) / 180;
  const wing = (s: number): Pt => {
    const cs = Math.cos(s * ang);
    const sn = Math.sin(s * ang);
    return [b[0] - hl * (ux * cs - uy * sn), b[1] - hl * (ux * sn + uy * cs)];
  };
  return { from: a, ctrl: c, tip: b, wings: [wing(1), wing(-1)] };
}

/** The spotlight's hole: box plus ~12 px, rounded. */
export function spotlightHole(box: Box, size: [number, number]): Rect {
  const u = unitOf(size);
  const pad = 12 * u;
  const [x, y, w, h] = box;
  const W = w + 2 * pad;
  const H = h + 2 * pad;
  return { x: x - pad, y: y - pad, w: W, h: H, r: Math.min(18 * u, W / 2, H / 2) };
}

/** A pulse ring at expansion k (0..1): a pill around the box that grows outwards. */
export function pulseRect(box: Box, size: [number, number], k: number): Rect {
  const u = unitOf(size);
  const g = 4 * u + 26 * u * clamp01(k);
  const [x, y, w, h] = box;
  const W = w + 2 * g;
  const H = h + 2 * g;
  return { x: x - g, y: y - g, w: W, h: H, r: Math.min(W, H) / 2 };
}

const within = (p: Pt, b: Box, slack: number) =>
  p[0] >= b[0] - slack && p[0] <= b[0] + b[2] + slack && p[1] >= b[1] - slack && p[1] <= b[1] + b[3] + slack;

/**
 * End points of an arrow: from / to, else the gesture path's ends, else the box diagonal.
 * The backend points from / to at element centres, and the centre of a wide line of text can be far from where
 * the teacher's pointer actually started or stopped (an arrow drawn from there can cut through the line above).
 * So when the measured path starts on the "from" element (ends on the "to" element), within `slack`, its own
 * end is used instead.
 */
export function arrowEnds(a: SlideAnnotation, fromBox?: Box | null, toBox?: Box | null, slack = 8): [Pt, Pt] {
  const ok = (p: unknown): p is Pt => Array.isArray(p) && p.length >= 2 && fin(p[0]) && fin(p[1]);
  const path = (a.path ?? []).filter(ok).map((p): Pt => [p[0], p[1]]);
  if (ok(a.from) && ok(a.to)) {
    let from: Pt = [a.from[0], a.from[1]];
    let to: Pt = [a.to[0], a.to[1]];
    if (path.length >= 2) {
      const p0 = path[0];
      const p1 = path[path.length - 1];
      if (fromBox && within(p0, fromBox, slack) && Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 1) from = p0;
      if (toBox && within(p1, toBox, slack) && Math.hypot(p1[0] - from[0], p1[1] - from[1]) > 1) to = p1;
    }
    return [from, to];
  }
  if (path.length >= 2) return [path[0], path[path.length - 1]];
  const [x, y, w, h] = a.box;
  return [
    [x, y + h],
    [x + w, y],
  ];
}

// ---------------------------------------------------------------- board annotations

/** A target id as a board piece id (targets are piece ids as strings); null when it is not one. */
export function pieceIdOf(id: unknown): number | null {
  const n = typeof id === 'number' ? id : typeof id === 'string' && /^\s*-?\d+\s*$/.test(id) ? Number(id) : NaN;
  return Number.isInteger(n) ? n : null;
}

/** The ids of the board pieces an annotation points at. */
export function targetPieceIds(a: SlideAnnotation): number[] {
  return ((a.target ?? []) as unknown[]).map(pieceIdOf).filter((n): n is number => n != null);
}

/** When the writing an annotation points at starts leaving the board: the earliest removed_at among its target
 *  pieces, null when none of them is ever erased. */
export function targetGoneAt(a: SlideAnnotation, pieces: readonly Piece[]): number | null {
  const ids = new Set(targetPieceIds(a));
  let gone: number | null = null;
  for (const p of pieces) {
    if (!ids.has(p.id) || !fin(p.removed_at)) continue;
    if (gone == null || p.removed_at < gone) gone = p.removed_at;
  }
  return gone;
}

/** When the hand moves: t_start to t_end (t_start alone when t_end is missing or earlier). */
function gestureSpan(a: SlideAnnotation): [number, number] {
  return [a.t_start, fin(a.t_end) ? Math.max(a.t_end, a.t_start) : a.t_start];
}

/**
 * The box an arrow on the board points from / to for one of its targets: the piece's whole written line (`lines`:
 * the board's lines that are one written line, see isWrittenLine), so the arrow never runs over the rest of that
 * line, and one between two lines stacked close curves round their ends; the piece itself when it has no line, when
 * the arrow goes between two pieces of the same line, or when the line is not finished by the end of the gesture or
 * was erased before it.
 */
export function boardTargetBox(
  a: SlideAnnotation,
  id: string,
  pieces: readonly Piece[],
  lines: readonly BoardLine[] = [],
): Box | null {
  const n = pieceIdOf(id);
  if (n == null) return null;
  const [t0, t1] = gestureSpan(a);
  const others = targetPieceIds(a).filter((m) => m !== n);
  const line = lines.find(
    (ln) =>
      ln.pieces?.includes(n) &&
      !others.some((m) => ln.pieces.includes(m)) &&
      !(fin(ln.t_complete) && ln.t_complete > t1) &&
      !(fin(ln.removed_at) && ln.removed_at <= t0),
  );
  return line?.box ?? pieces.find((p) => p.id === n)?.box ?? null;
}

/** Share of box p's area that lies inside box q. */
function shareIn(p: Box, q: Box): number {
  const w = Math.min(p[0] + p[2], q[0] + q[2]) - Math.max(p[0], q[0]);
  const h = Math.min(p[1] + p[3], q[1] + q[3]) - Math.max(p[1], q[1]);
  return w > 0 && h > 0 ? (w * h) / Math.max(1e-6, p[2] * p[3]) : 0;
}

/**
 * The other writing on the board while the hand moves, which an arrow should keep off: pieces shown by the end of
 * the gesture (startOf: when a piece starts showing, timing's schedule) and not erased before it, and the written
 * lines complete by then, without the targets and their lines. Left out: anything mostly inside an end box or
 * holding most of one, and regions (a graph, a block of working: an arrow may cross their blank space).
 */
export function boardObstacles(
  a: SlideAnnotation,
  pieces: readonly Piece[],
  lines: readonly BoardLine[],
  size: [number, number],
  ends: readonly (Box | null)[],
  startOf: (p: Piece) => number,
): Box[] {
  const u = unitOf(size);
  const [t0, t1] = gestureSpan(a);
  const ids = new Set(targetPieceIds(a));
  const there = (start: unknown, removed: unknown) => !(fin(start) && start > t1) && !(fin(removed) && removed <= t0);
  const boxes: Box[] = [];
  for (const p of pieces) if (!ids.has(p.id) && there(startOf(p), p.removed_at)) boxes.push(p.box);
  for (const ln of lines) {
    if (!ln.pieces?.some((n) => ids.has(n)) && there(ln.t_complete, ln.removed_at)) boxes.push(ln.box);
  }
  return boxes.filter(
    (b) =>
      Array.isArray(b) &&
      b.length >= 4 &&
      b.every(fin) &&
      b[3] <= REGION_MIN * u &&
      !ends.some((e) => e && (shareIn(b, e) >= 0.5 || shareIn(e, b) >= 0.5)),
  );
}

/**
 * An arrow on the board: from its first target to its second (the backend's from / to; with one target, to it), each
 * as boardTargetBox gives it, and the writing it should keep off (boardObstacles). `lines` are the written lines.
 */
export function boardArrowBoxes(
  a: SlideAnnotation,
  pieces: readonly Piece[],
  lines: readonly BoardLine[],
  size: [number, number],
  startOf: (p: Piece) => number,
): ArrowBoxes {
  const tg = a.target ?? [];
  const from = tg.length >= 2 ? boardTargetBox(a, tg[0], pieces, lines) : null;
  const to = tg.length ? boardTargetBox(a, tg[tg.length >= 2 ? 1 : 0], pieces, lines) : null;
  return { from, to, obstacles: boardObstacles(a, pieces, lines, size, [from, to], startOf) };
}

/** A hand gesture on the board is timed like one on a slide (drawn over the gesture, held HOLD s, faded over
 *  FADE_OUT), but it fades together with its writing when that is erased before then. */
export function boardAnnotationTiming(a: SlideAnnotation, pieces: readonly Piece[]): AnnotationTiming {
  return annotationTiming(a, targetGoneAt(a, pieces));
}

/** Taller than this (pixels of a 960-wide image, about four lines of handwriting) a target is a region of the
 *  board, not a line of writing. */
const REGION_MIN = 100;

/**
 * An underline whose target is a whole region of the board (a graph with its labels, a block of working) would run
 * across the middle of it, through the writing: it goes under the stretch of the region the hand actually swept
 * instead, as if the box were the line just above the sweep. When the sweep is not over the region (the hand came
 * from the side) or there is none, it goes under the whole region (at the sweep's height it would cross the middle).
 * Anything else is returned as it is.
 */
export function fitUnderline(a: SlideAnnotation, size: [number, number]): SlideAnnotation {
  if (!drawable(a) || kindOf(a) !== 'underline' || fin(a.underline_y)) return a;
  const u = unitOf(size);
  const [bx, by, bw, bh] = a.box;
  if (!(bh > REGION_MIN * u)) return a;
  const under = { ...a, underline_y: Math.min(size[1] - 3 * u, by + bh + 5 * u) };
  const pts = (a.path ?? []).filter((p) => Array.isArray(p) && fin(p[0]) && fin(p[1]));
  if (pts.length < 2) return under;
  const ys = pts.map((p) => p[1]).sort((p, q) => p - q);
  const mid = ys[ys.length >> 1];
  if (mid < by || mid > by + bh) return under;
  const xs = pts.map((p) => p[0]);
  let x0 = Math.max(bx, Math.min(...xs));
  let x1 = Math.min(bx + bw, Math.max(...xs));
  const minW = 24 * u;
  if (x1 - x0 < minW) {
    if (x1 - x0 < minW / 2) return under; // the sweep barely touches the region
    const c = (x0 + x1) / 2;
    x0 = Math.max(bx, c - minW / 2);
    x1 = Math.min(bx + bw, x0 + minW);
  }
  const lh = 20 * u;
  return { ...a, box: [x0, mid - lh, x1 - x0, lh] };
}

/** The board's annotations as the player draws them. */
export function boardAnnotations(board: BoardData): SlideAnnotation[] {
  return (board.annotations ?? []).filter(drawable).map((a) => fitUnderline(a, board.size));
}
