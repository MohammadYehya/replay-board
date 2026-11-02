// An "angle" object of a 3D scene, worked out with no three.js (so it runs in Node tests): the angle at a vertex
// between the rays vertex -> p1 and vertex -> p2 (0..180 degrees), drawn in the plane of the two rays as an arc (and a
// wedge), or as a right-angle mark within RIGHT_TOLERANCE of 90 degrees; its label sits on the bisector.
// Everything is in maths coordinates; the stage maps the points to the world like any other object's.
import type { Vec3 } from '../types';

/** Shorter arms than this (or anything not finite): the angle is not drawn in that state. */
export const DEGENERATE_ARM = 1e-9;
/** Within this many degrees of 90 the angle is drawn as a right-angle mark instead of an arc. */
export const RIGHT_TOLERANCE = 0.5;
/** The auto radius, as a share of the shorter arm. */
export const AUTO_RADIUS = 0.3;
/** The right-angle mark's side, as a share of the radius. */
export const RIGHT_SIDE = 0.7;
/** The label's distance from the vertex along the bisector, as a share of the radius. */
export const LABEL_OUT = 1.35;

export interface AngleShape {
  /** The angle, 0..180. */
  degrees: number;
  radius: number;
  /** Drawn as a right-angle mark. */
  right: boolean;
  vertex: Vec3;
  /** The line drawn: the arc from the first arm to the second, or the mark's two outer sides (3 points). */
  line: Vec3[];
  /** The filled area as a fan around the vertex: the vertex, then the outline's points ([] when there is none). */
  fan: Vec3[];
  /** Where the label goes: on the bisector, LABEL_OUT x radius from the vertex. */
  label: Vec3;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const scale = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const along = (o: Vec3, a: Vec3, ka: number, b: Vec3 = [0, 0, 0], kb = 0): Vec3 => [
  o[0] + a[0] * ka + b[0] * kb,
  o[1] + a[1] * ka + b[1] * kb,
  o[2] + a[2] * ka + b[2] * kb,
];
const finite = (...vs: Vec3[]) => vs.every((v) => v.length === 3 && v.every(Number.isFinite));

/** The angle at `vertex` between the rays to p1 and p2, in degrees; null when degenerate. */
export function measureAngle(vertex: Vec3, p1: Vec3, p2: Vec3): number | null {
  if (!finite(vertex, p1, p2)) return null;
  const a = sub(p1, vertex);
  const b = sub(p2, vertex);
  if (!(len(a) >= DEGENERATE_ARM) || !(len(b) >= DEGENERATE_ARM)) return null;
  // atan2 of |a x b| and a.b: accurate near 0 and 180 too (acos is not)
  const d = (Math.atan2(len(cross(a, b)), dot(a, b)) * 180) / Math.PI;
  return Number.isFinite(d) ? d : null;
}

/** The radius an angle is drawn with: the one it gives (> 0), or AUTO_RADIUS x its shorter arm; null when it gives a
 *  radius that is not a positive number. */
export function angleRadius(vertex: Vec3, p1: Vec3, p2: Vec3, given: number | null): number | null {
  if (given !== null) return Number.isFinite(given) && given > 0 ? given : null;
  const r = AUTO_RADIUS * Math.min(len(sub(p1, vertex)), len(sub(p2, vertex)));
  return Number.isFinite(r) && r > 0 ? r : null;
}

/**
 * The shape of an angle in one state, or null when it is degenerate there (an arm shorter than DEGENERATE_ARM, a
 * radius that is not positive, anything not finite): it is then not drawn. `given`: the radius from its size, or null
 * for the auto radius.
 */
export function angleShape(vertex: Vec3, p1: Vec3, p2: Vec3, given: number | null = null): AngleShape | null {
  const degrees = measureAngle(vertex, p1, p2);
  if (degrees === null) return null;
  const radius = angleRadius(vertex, p1, p2, given);
  if (radius === null) return null;
  const a = sub(p1, vertex);
  const b = sub(p2, vertex);
  const u1 = scale(a, 1 / len(a));
  const u2 = scale(b, 1 / len(b));
  // the unit vector in the plane of the rays, square to the first arm, on the side of the second
  let w = along(u2, u1, -dot(u1, u2));
  if (len(w) < 1e-9) {
    // the arms are in one line (0 or 180 degrees): any plane through them will do
    const other: Vec3 = Math.abs(u1[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    w = cross(other, u1);
  }
  w = scale(w, 1 / len(w));
  const theta = (degrees * Math.PI) / 180;
  const half = theta / 2;
  const label = along(vertex, u1, LABEL_OUT * radius * Math.cos(half), w, LABEL_OUT * radius * Math.sin(half));
  const right = Math.abs(degrees - 90) <= RIGHT_TOLERANCE;
  if (right) {
    const s = RIGHT_SIDE * radius;
    const line: Vec3[] = [along(vertex, u1, s), along(vertex, u1, s, u2, s), along(vertex, u2, s)];
    return { degrees, radius, right, vertex, line, fan: [vertex, ...line], label };
  }
  const n = Math.max(2, Math.ceil(degrees / 4));
  const line: Vec3[] = [];
  for (let i = 0; i <= n; i++) {
    const t = (theta * i) / n;
    line.push(along(vertex, u1, radius * Math.cos(t), w, radius * Math.sin(t)));
  }
  return { degrees, radius, right, vertex, line, fan: degrees > 1e-6 ? [vertex, ...line] : [], label };
}

/** The points an angle reaches (for bounds and framing): its vertex +/- its radius along each axis. */
export function angleReach(shape: AngleShape): Vec3[] {
  const [x, y, z] = shape.vertex;
  const r = shape.radius;
  return [
    [x - r, y, z],
    [x + r, y, z],
    [x, y - r, z],
    [x, y + r, z],
    [x, y, z - r],
    [x, y, z + r],
  ];
}
