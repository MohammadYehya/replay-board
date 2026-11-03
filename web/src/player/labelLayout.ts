// Where the 3D view's labels go, with no three.js and no DOM (so it runs in Node tests):
// - placeLabels(): labels on screen never sit on each other. The most important keep their place; each one after
//   moves the least it can (up, down or sideways) to a free spot, or is dimmed when there is none near.
// - arcLabelAt(): an angle's arc names its angle out along the angle's bisector, past the arc, rather than on it,
//   so three arcs from one corner (alpha, beta, gamma) put their names apart.
// - glide(): a label never jumps to a new place: it slides there in a few frames.
// - tickCenter(): an axis' numbers sit beside the axis as seen on screen, never on it; originCenter(): the origin's
//   number in a gap between all the axes, clear of each.

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
  /** Last frame's places (same order), tried first when their spot is still free, so labels do not hop around; when
   *  it is not, the free spot nearest it of those about as near the label's own place as the nearest (KEEP_SLACK). */
  prev?: readonly (LabelPlace | undefined)[];
  /** Pixels kept between two labels. */
  pad?: number;
  /** The farthest a label moves from its place, in pixels. */
  reach?: number;
  /** Places a label that has to move never moves onto (the angles' arcs and wedges, so a name never covers the mark
   *  it or another angle is about); a label in its own place may sit on one. */
  avoid?: readonly Rect[];
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The most labels a layout looks for a free spot for (the rest that collide are dimmed). */
export const MAX_SEARCHES = 32;
/** Pixels farther from its own place than the nearest free spot that a label may go to stay near last frame's spot. */
export const KEEP_SLACK = 6;

const hits = (a: Rect, b: Rect, pad: number) =>
  a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad;

/**
 * Places labels given in priority order (the first keeps its place whatever happens). Returns one place per label,
 * in the same order: the nudge to apply, or dim when no spot within `reach` is free.
 */
export function placeLabels(boxes: readonly LabelBox[], opts: PlaceOptions): LabelPlace[] {
  const pad = opts.pad ?? 2;
  const reach = opts.reach ?? 44;
  const avoid = opts.avoid ?? [];
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
      return (!wasInside || inside(r)) && !placed.some((q) => hits(r, q, pad)) && (!(dx || dy) || !avoid.some((q) => hits(r, q, 0)));
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
    const marks = avoid.filter((q) => hits(at(0, 0), q, reach + pad));
    const freeHere = (dx: number, dy: number) => {
      const r = at(dx, dy);
      return (!wasInside || inside(r)) && !local.some((q) => hits(r, q, pad)) && !marks.some((q) => hits(r, q, 0));
    };
    const cx = b.x + b.w / 2;
    const cy = b.y + b.h / 2;
    const dist = (q: Rect) => Math.hypot(q.x + q.w / 2 - cx, q.y + q.h / 2 - cy);
    const near = [...local.slice().sort((q, r) => dist(q) - dist(r)).slice(0, 6), ...marks.slice().sort((q, r) => dist(q) - dist(r)).slice(0, 4)];
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
    // A label nudged last frame whose spot is now taken goes to the free spot nearest that one, of those about as
    // near its own place as the nearest free spot (KEEP_SLACK): as test2 #4's cell came to rest, beta's name, kept
    // just right of "b", touched it by a fraction of a pixel and went 31 px up to the spot nearest home (0.5 px
    // nearer), then came back once all was still. Never farther: the saddle's "y", following the spot beside "z = x²"
    // as the view turned, drifted 39 px right of a free spot 9 px above its place, and slid back there at rest.
    const kept = p && !p.dim && (p.dx || p.dy) ? p : null;
    let first = -1;
    let best: [number, number] | null = null;
    let bestOff = Infinity;
    for (const [dx, dy, d] of moves) {
      if (first >= 0 && (!kept || d > first + KEEP_SLACK)) break;
      if (!freeHere(dx, dy)) continue;
      if (first < 0) first = d;
      const off = kept ? Math.hypot(dx - kept.dx, dy - kept.dy) : 0;
      if (off < bestOff) {
        bestOff = off;
        best = [dx, dy];
      }
    }
    out.push(best ? take(best[0], best[1]) : { dx: 0, dy: 0, dim: true });
  });
  return out;
}

/** How briskly a label glides to a new place (per second): a critically damped spring that starts and ends gently,
 *  95% of the way in 4.7 / GLIDE_RATE s (0.26 s). */
export const GLIDE_RATE = 18;

/** A label's nudge as shown, and how fast it is moving (pixels, pixels per second). */
export interface Glide {
  dx: number;
  dy: number;
  vx: number;
  vy: number;
}

/**
 * Where a label's nudge is on the next frame, `dt` seconds after this one: from where it is shown now (and as fast as
 * it moves) towards its place, as a critically damped spring (worked out exactly, so a long frame never overshoots
 * from rest), and on it once within half a pixel and nearly still. Labels are placed afresh once the drawing is still
 * (so a moment looks the same however the view got there), and while it moves a label keeps its spot while it is
 * free; the two differ, and a label hopped up to 32 px in one frame on a still drawing when a change ended (test2 #4).
 * `from` undefined (a label that was not on screen) or no rate (reduced motion): straight to its place.
 */
export function glide(
  from: Glide | undefined,
  to: { dx: number; dy: number },
  dt: number,
  rate = GLIDE_RATE,
): Glide & { done: boolean } {
  const there = { dx: to.dx, dy: to.dy, vx: 0, vy: 0, done: true };
  if (!from || !(rate > 0)) return there;
  const t = Math.max(0, dt);
  const k = Math.exp(-rate * t);
  const axis = (x: number, v: number, goal: number): [number, number] => {
    const e = x - goal;
    const c = v + rate * e;
    return [goal + (e + c * t) * k, (v - rate * c * t) * k];
  };
  const [dx, vx] = axis(from.dx, from.vx, to.dx);
  const [dy, vy] = axis(from.dy, from.vy, to.dy);
  if (Math.hypot(to.dx - dx, to.dy - dy) <= 0.5 && Math.hypot(vx, vy) <= 20) return there;
  return { dx, dy, vx, vy, done: false };
}

/** Pixels between an axis and the near edge of its numbers. */
export const TICK_GAP = 6;

/**
 * The label renderer's centre (CSS2DObject.center) for an axis number whose anchor is its tick, on the axis: the
 * label sits on the side square to the axis as seen on screen, its near edge `gap` pixels off it. Below an axis that
 * runs across the screen and left of one that runs up it, as on graph paper; between, the side towards the bottom
 * left. (dx, dy): the axis' direction on screen, in pixels, y down; w x h: the label's size. Seen from the top, the y
 * axis runs up the screen, and its numbers, centred on it, sat under the dashed asymptote x = 0 drawn along it
 * (test-full #3, gest-test #10).
 */
export function tickCenter(dx: number, dy: number, w: number, h: number, gap = TICK_GAP): [number, number] {
  const n = tickSide(dx, dy);
  return n ? besideCenter(n[0], n[1], w, h, gap) : [0.5, 0.5];
}

/** The side of an axis its numbers go on, as a unit vector on screen (y down), square to the axis: see tickCenter.
 *  Null for an axis with no direction on screen (seen end on). */
export function tickSide(dx: number, dy: number): [number, number] | null {
  const len = Math.hypot(dx, dy);
  if (!(len > 1e-9)) return null;
  const nx = -dy / len;
  const ny = dx / len;
  return ny - nx >= -ny + nx ? [nx, ny] : [-nx, -ny];
}

/** The label renderer's centre for a label w x h that sits out from its anchor along the unit vector (nx, ny), its
 *  nearest corner `gap` pixels off the line through the anchor square to that vector. The origin's number goes out
 *  between the sides of the two axes that meet there, so it sits under neither (an asymptote x = 0 runs down the y
 *  axis past it). */
export function besideCenter(nx: number, ny: number, w: number, h: number, gap = TICK_GAP): [number, number] {
  const len = Math.hypot(nx, ny);
  if (!(len > 1e-9) || !(w > 0) || !(h > 0)) return [0.5, 0.5];
  const ux = nx / len;
  const uy = ny / len;
  // the label's middle goes out by the gap and half its box's reach along u
  const t = gap + (Math.abs(ux) * w) / 2 + (Math.abs(uy) * h) / 2;
  return [0.5 - (ux * t) / w, 0.5 - (uy * t) / h];
}

/** How far from where the axes cross the origin's number may go to find room, in label sizes (w + h). */
const ORIGIN_REACH = 3;
const DOWN_LEFT: readonly [number, number] = [-Math.SQRT1_2, Math.SQRT1_2];

/** Whether the ray from (0, 0) along (rx, ry) passes within `gap` of the box centred at (cx, cy), w x h. */
function rayHits(rx: number, ry: number, cx: number, cy: number, w: number, h: number, gap: number): boolean {
  let lo = 0;
  let hi = Infinity;
  for (const [r, c, half] of [[rx, cx, w / 2 + gap], [ry, cy, h / 2 + gap]]) {
    if (Math.abs(r) < 1e-12) {
      if (Math.abs(c) > half) return false;
      continue;
    }
    const t1 = (c - half) / r;
    const t2 = (c + half) / r;
    lo = Math.max(lo, Math.min(t1, t2));
    hi = Math.min(hi, Math.max(t1, t2));
    if (lo > hi) return false;
  }
  return true;
}

/**
 * The label renderer's centre for the origin's number (w x h), anchored where the axes cross: in a gap between the
 * axes as seen, its box `gap` pixels clear of every one, as near the crossing as that allows; of places about as near,
 * the one most towards the bottom left, as on graph paper. `rays`: each axis' direction on screen from the crossing
 * (y down), both ways where it runs on past it. With x and y alone that is below-left of the crossing; with z as well
 * the "0" was put between the sides of x and y only, and sat on the x axis (the saddle from the front) or on z (iso).
 */
export function originCenter(rays: readonly (readonly [number, number])[], w: number, h: number, gap = TICK_GAP): [number, number] {
  if (!(w > 0) || !(h > 0)) return [0.5, 0.5];
  const dirs = rays
    .map(([x, y]) => [x, y, Math.hypot(x, y)])
    .filter(([, , n]) => n > 1e-9)
    .map(([x, y, n]) => [x / n, y / n] as const);
  if (!dirs.length) return besideCenter(DOWN_LEFT[0], DOWN_LEFT[1], w, h, gap);
  const clear = (cx: number, cy: number) => dirs.every(([rx, ry]) => !rayHits(rx, ry, cx, cy, w, h, gap));
  const angles = dirs.map(([x, y]) => Math.atan2(y, x)).sort((a, b) => a - b);
  const far = ORIGIN_REACH * (w + h);
  const found: { t: number; u: [number, number]; lean: number }[] = [];
  angles.forEach((a0, i) => {
    const a1 = i + 1 < angles.length ? angles[i + 1] : angles[0] + 2 * Math.PI;
    if (a1 - a0 < 1e-6) return;
    const N = 24; // directions tried across the gap
    for (let k = 0; k < N; k++) {
      const a = a0 + ((k + 0.5) / N) * (a1 - a0);
      const u: [number, number] = [Math.cos(a), Math.sin(a)];
      if (!clear(u[0] * far, u[1] * far)) continue;
      let lo = 0;
      let hi = far;
      for (let j = 0; j < 20; j++) {
        const m = (lo + hi) / 2;
        if (clear(u[0] * m, u[1] * m)) hi = m;
        else lo = m;
      }
      found.push({ t: hi, u, lean: u[0] * DOWN_LEFT[0] + u[1] * DOWN_LEFT[1] });
    }
  });
  if (!found.length) return besideCenter(DOWN_LEFT[0], DOWN_LEFT[1], w, h, gap);
  const near = Math.min(...found.map((f) => f.t)) + 1.5;
  const { t, u } = found.filter((f) => f.t <= near).reduce((a, b) => (b.lean > a.lean ? b : a));
  return [0.5 - (u[0] * t) / w, 0.5 - (u[1] * t) / h];
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
