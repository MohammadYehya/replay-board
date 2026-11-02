// Where the 3D view's labels go, with no three.js and no DOM (so it runs in Node tests):
// - placeLabels(): labels on screen never sit on each other. The most important keep their place; each one after
//   moves the least it can (up, down or sideways) to a free spot, or is dimmed when there is none near.
// - arcLabelAt(): an angle's arc names its angle out along the angle's bisector, past the arc, rather than on it,
//   so three arcs from one corner (alpha, beta, gamma) put their names apart.

export interface LabelBox {
  /** Where the label is on the stage now (CSS pixels, before any nudge). */
  x: number;
  y: number;
  w: number;
  h: number;
  /** False for a label whose place is its meaning (an axis tick's number): it is dimmed, never moved. */
  movable: boolean;
}

export interface LabelPlace {
  dx: number;
  dy: number;
  /** No free spot near: shown faint. */
  dim: boolean;
}

export interface PlaceOptions {
  /** The stage's size: a label inside it stays inside it. */
  width: number;
  height: number;
  /** Last frame's places (same order), tried first when their spot is still free, so labels do not hop around. */
  prev?: readonly (LabelPlace | undefined)[];
  /** Pixels kept between two labels. */
  pad?: number;
  /** The farthest a label moves from its place, in pixels. */
  reach?: number;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The most labels a layout looks for a free spot for (the rest that collide are dimmed). */
export const MAX_SEARCHES = 32;

const hits = (a: Rect, b: Rect, pad: number) =>
  a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad;

/**
 * Places labels given in priority order (the first keeps its place whatever happens). Returns one place per label,
 * in the same order: the nudge to apply, or dim when no spot within `reach` is free.
 */
export function placeLabels(boxes: readonly LabelBox[], opts: PlaceOptions): LabelPlace[] {
  const pad = opts.pad ?? 2;
  const reach = opts.reach ?? 44;
  const placed: Rect[] = [];
  const out: LabelPlace[] = [];
  const inside = (r: Rect) => r.x >= 0 && r.y >= 0 && r.x + r.w <= opts.width && r.y + r.h <= opts.height;
  // It runs on every frame drawn: past this many searches for a free spot (a pile of labels in one corner), the rest
  // that collide are dimmed straight away.
  let searches = 0;
  boxes.forEach((b, i) => {
    const at = (dx: number, dy: number): Rect => ({ x: b.x + dx, y: b.y + dy, w: b.w, h: b.h });
    const wasInside = inside(b);
    const free = (dx: number, dy: number) => {
      const r = at(dx, dy);
      return (!wasInside || inside(r)) && !placed.some((q) => hits(r, q, pad));
    };
    const take = (dx: number, dy: number): LabelPlace => {
      placed.push(at(dx, dy));
      return { dx, dy, dim: false };
    };
    if (free(0, 0)) return void out.push(take(0, 0));
    if (!b.movable) return void out.push({ dx: 0, dy: 0, dim: true });
    const p = opts.prev?.[i];
    if (p && !p.dim && (p.dx || p.dy) && Math.hypot(p.dx, p.dy) <= reach && free(p.dx, p.dy))
      return void out.push(take(p.dx, p.dy));
    if (++searches > MAX_SEARCHES) return void out.push({ dx: 0, dy: 0, dim: true });
    // The moves that clear one label in the way (each side of it), and two of them at once (round a corner): the
    // shortest that is free wins. A move within reach can only meet the labels within reach.
    const local = placed.filter((q) => hits(at(0, 0), q, reach + pad));
    const freeHere = (dx: number, dy: number) => {
      const r = at(dx, dy);
      return (!wasInside || inside(r)) && !local.some((q) => hits(r, q, pad));
    };
    const cx = b.x + b.w / 2;
    const cy = b.y + b.h / 2;
    const dist = (q: Rect) => Math.hypot(q.x + q.w / 2 - cx, q.y + q.h / 2 - cy);
    const near = local.slice().sort((q, r) => dist(q) - dist(r)).slice(0, 6);
    const xs = [0];
    const ys = [0];
    for (const q of near) {
      ys.push(q.y - pad - (b.y + b.h), q.y + q.h + pad - b.y); // just above it, just below it
      xs.push(q.x - pad - (b.x + b.w), q.x + q.w + pad - b.x); // just left of it, just right of it
    }
    const moves: [number, number, number][] = [];
    for (const dx of xs)
      for (const dy of ys) {
        const d = Math.hypot(dx, dy);
        if (d > 0 && d <= reach) moves.push([dx, dy, d]);
      }
    moves.sort((m, n) => m[2] - n[2] || m[1] - n[1] || m[0] - n[0]);
    for (const [dx, dy] of moves) if (freeHere(dx, dy)) return void out.push(take(dx, dy));
    out.push({ dx: 0, dy: 0, dim: true });
  });
  return out;
}

type P3 = readonly [number, number, number];

const sub = (a: P3, b: P3): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: P3, b: P3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: P3, b: P3): [number, number, number] => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const len = (a: P3) => Math.sqrt(dot(a, a));

/** The centre of the circle through three points, or null when they are (nearly) on one line. */
export function circumcentre(a: P3, b: P3, c: P3): [number, number, number] | null {
  const ab = sub(b, a);
  const ac = sub(c, a);
  const n = cross(ab, ac);
  const nn = dot(n, n);
  const scale = Math.max(dot(ab, ab), dot(ac, ac));
  if (!(nn > 1e-12 * scale * scale)) return null;
  const t1 = cross(n, ab).map((v) => v * dot(ac, ac));
  const t2 = cross(ac, n).map((v) => v * dot(ab, ab));
  return [a[0] + (t1[0] + t2[0]) / (2 * nn), a[1] + (t1[1] + t2[1]) / (2 * nn), a[2] + (t1[2] + t2[2]) / (2 * nn)];
}

/**
 * Where an angle's name goes: when the points (in order along a curve) lie on one circular arc short of a full turn
 * (an angle's arc drawn round its corner), `factor` times the radius out from the centre along the arc's middle,
 * which is the angle's bisector; else null (the curve is not an arc: its label stays at its middle).
 */
export function arcLabelAt(points: readonly P3[], factor = 1.35): [number, number, number] | null {
  if (points.length < 5) return null;
  const a = points[0];
  const m = points[Math.floor((points.length - 1) / 2)];
  const b = points[points.length - 1];
  const c = circumcentre(a, m, b);
  if (!c) return null;
  const r = len(sub(a, c));
  if (!(r > 1e-9)) return null;
  for (const p of points) if (Math.abs(len(sub(p, c)) - r) > 0.04 * r) return null;
  // a full (or nearly full) turn is a circle, not an angle
  if (len(sub(a, b)) < 0.05 * r) return null;
  const d = sub(m, c);
  return [c[0] + d[0] * factor, c[1] + d[1] * factor, c[2] + d[2] * factor];
}
