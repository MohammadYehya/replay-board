// Pure timing and geometry for the slide view: which slide state shows at t, the crossfade between states,
// when each pointer annotation is drawn, held and faded, the pulse rings, the pointer dot, and the SVG shapes.
// Every animation is a function of the shared clock, so scrubbing shows exactly the frame for that time.
// No DOM here, so `npm test` can run it with plain Node.
import type { Box, PointerSample, SlideAnnotation, SlideElement, SlideState } from '../types';

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

export const toolOf = (tool: string | undefined): 'cursor' | 'laser' =>
  tool === 'laser' || tool === 'l' ? 'laser' : 'cursor';

/** Warm amber for the mouse cursor, red for a laser pointer. */
export const toolColour = (tool: string | undefined) => (toolOf(tool) === 'laser' ? '#ef4444' : '#f59e0b');

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
}

/**
 * A curved arrow from `from` to `to`. When an end sits inside its element's box (the backend points at element
 * centres), it is pulled back to just outside that box, so the arrow points at the text instead of covering it.
 */
export function arrowGeometry(from: Pt, to: Pt, size: [number, number], fromBox?: Box | null, toBox?: Box | null): ArrowGeom {
  const u = unitOf(size);
  let a: Pt = from;
  let b: Pt = to;
  if (fromBox && inside(a, fromBox)) a = exitPoint(a, b, fromBox, 5 * u) ?? a;
  if (toBox && inside(b, toBox)) b = exitPoint(b, a, toBox, 6 * u) ?? b;
  if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 36 * u) {
    a = from;
    b = to;
  }
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.max(1e-6, Math.hypot(dx, dy));
  let nx = dy / len;
  let ny = -dx / len;
  if (ny > 0 || (ny === 0 && nx < 0)) {
    nx = -nx;
    ny = -ny; // always bow upwards
  }
  const bend = Math.min(len * 0.14, 50 * u);
  const c: Pt = [(a[0] + b[0]) / 2 + nx * bend, (a[1] + b[1]) / 2 + ny * bend];
  // Arrowhead along the curve's end tangent.
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
  return {
    d: `M${pt(a)} Q${pt(c)} ${pt(b)}`,
    head: `M${pt(wing(1))} L${pt(b)} L${pt(wing(-1))}`,
    tip: b,
    from: a,
  };
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
