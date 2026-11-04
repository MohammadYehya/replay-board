// The three.js stage of the 3D view: draws a format-2 scene (types.ts Scene3D) and shows, every frame, the state the
// view asks for (scene3dState.ts sceneState). Loaded only with the 3D view's chunk, so three.js never weighs on the
// player's main bundle.
//
// Maths coordinates have z up; three.js has y up: world = (x, z, -y), scaled so the scene fits a box about L wide.
// Everything is built straight in world coordinates, so a stretched axis (a graph much taller than wide) stretches
// shapes honestly while tubes, arrow heads and dots keep the same thickness on screen.
//
// Every object is built when the scene loads (hidden until a step shows it) and every shader is compiled before the
// first frame, so a step that shows many parts at once only uncovers them. After that an object's geometry is rebuilt
// only when a parameter it reads changes (and only while it is visible); fades, growth and the highlight's glow only
// touch materials and transforms.
//
// Lines (the edges of solids, wire shapes) are drawn a few pixels wide (WebGL's own lines are always one pixel), and
// tubes and rods are made thicker when the drawing is spread out, so both read at the stage's size.
import {
  AmbientLight, BoxGeometry, BufferGeometry, CanvasTexture, Color, ConeGeometry, Curve, CylinderGeometry,
  DirectionalLight, DoubleSide, EdgesGeometry, Euler, Float32BufferAttribute, Fog, FrontSide, Group, HemisphereLight,
  LineBasicMaterial, LineSegments, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial, PerspectiveCamera,
  PlaneGeometry, Quaternion, Scene, ShapeUtils, SphereGeometry, Spherical, SRGBColorSpace, TorusGeometry, TubeGeometry,
  Vector2, Vector3, WebGLRenderer, type Material, type Side,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { Scene3D, Scene3DObject, Scene3DParam, Vec3 } from '../types';
import { angleReach, angleShape } from './angle3d.ts';
import { tryCompileVars, type VarsFn } from './expr.ts';
import {
  arcLabelAt, besideCenter, glide, originCenter, placeLabels, tickSide, type Glide, type LabelPlace, type Rect,
} from './labelLayout.ts';
import { distinctCurveColors, fitRange, formatTick, niceTicks, sampleSegments, stageColors } from './plotMath.ts';
import { liveNumberLabels, stepFraming, type Camera, type SceneState } from './scene3dState.ts';
import { angleLabel, displayName, formatValue, labelText } from './sceneText.ts';

export interface StageOptions {
  /** The theme's accent: the first colour for objects that bring none. */
  accent: string;
  /** prefers-reduced-motion: the camera jumps and nothing pulses. */
  reducedMotion: boolean;
  /** Called on every animation frame: what to show now. */
  frame(now: number): SceneState;
  /** The viewer turned or zoomed the view (true; steps stop moving the camera), or reset it (false). */
  onTurn?(turned: boolean): void;
  /** Pixels at each edge of the stage covered by what is laid over it (the caption bar at the bottom, the values at
   *  the top, the "Try it" sliders at the left): the drawing is framed in the rest. */
  insets?(): Insets;
  /** A box laid over the stage's top right corner (Reset view): how far it reaches in from the right edge and down
   *  from the top. The drawing is framed below it or to the left of it, whichever shows the drawing larger (keeping
   *  the whole top of the stage clear made "Try it" drawings up to 22% smaller). */
  corner?(): { width: number; height: number } | null;
  /** What is laid over the stage: when one changes size (a longer caption, the sliders opening), the drawing is
   *  framed again. */
  insetElements?(): (Element | null)[];
}

export interface Insets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface Stage3D {
  /** Back to the current step's view, and let the steps move the camera again. */
  resetView(): void;
  dispose(): void;
}

// ---------------------------------------------------------------- sizes (world units; the scene is about L wide)
const L = 2.4;
const BG = '#161512';
const AXIS = '#b3ab9a';
const GRID = '#7f786b';
const TUBE = 0.018;
const ROD = 0.017;
const HEAD_LEN = 0.11;
const HEAD_R = 0.042;
const DOT = 0.042;
export const DASH = 0.075;
export const GAP = 0.05;
const DEG = Math.PI / 180;
const TWEEN_MS = 900;
/** Tubes and rods get thicker, up to this many times, when the drawing is spread out (its framing radius, world units,
 *  over THICK_AT), so they keep about the same width on screen. */
const THICK_MAX = 1.9;
const THICK_AT = 1.5;
/** Lines' width in pixels: the stage's height over this, kept between 1.5 and 3. */
const LINE_PX_PER = 170;
/** How far back the fog (a gentle depth cue) reaches: at the back of the drawing it has faded about a fifth into the
 *  background. */
const FOG_FAR = 6;
/** A highlight's pulse (per second) and how much brighter it makes lines. */
const PULSE_HZ = 0.9;
/** Lines the current step shows or highlights are this much thicker and drawn last, so they read over a line they lie
 *  on (synth3d-vector3d: the x component lies on the x axis). */
export const EMPH = 1.3;
/** Where along a curve its name goes (a share of its samples): the middle of a curve symmetric about the origin is the
 *  origin, where the saddle's two parabolas and its saddle point put three names on each other. */
export const CURVE_LABEL_AT = 0.82;
/** The camera frames each step's drawing; never closer than this share of the whole animation's reach. */
const FRAME_MIN = 0.3;
/** ms after a slider stops moving before the view is framed again on the student's values. */
const REFIT_MS = 250;
/** An angle's label dims while its plane is seen this close to edge-on (|normal . line of sight|). */
const ANGLE_EDGE_ON = 0.25;

/** Where each camera preset looks from, in maths coordinates (z up). Front and side are a little off their axis, so
 *  an angle in a plane that meets the view edge-on (alpha and gamma on test2 #4, seen from the front) is still seen
 *  as an angle; top stays exact, for graphs. */
const CAMERA_DIRS: Record<Camera, Vec3> = {
  front: [0.2, -1, 0.25],
  top: [0, -0.02, 1],
  side: [1, 0.2, 0.25],
  iso: [0.7, -1.2, 0.62],
};
/** "iso" from the front-left corner, for a scene whose main vector points into the front-right one (its iso_from
 *  "left", app/popout.py pick_cameras). */
const ISO_LEFT: Vec3 = [-0.7, -1.2, 0.62];
/** An axis whose screen length is under this share of its true length points at the viewer: its numbers would pile
 *  up on the origin, so they are hidden (sin of the angle between the axis and the line of sight). */
const AXIS_EDGE_ON = 0.25;
/** Zoom the viewer chose, as a share of the fitted distance. */
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 4;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
/** The objects drawn as lines, which a step's emphasis thickens. */
const LINES = new Set(['segment', 'vector', 'curve', 'graph']);

/** Maths to world: world = (kx (x - cx), kz (z - cz), -ky (y - cy)). */
class Frame {
  readonly c: Vec3;
  readonly k: Vec3;
  constructor(c: Vec3, k: Vec3) {
    this.c = c;
    this.k = k;
  }
  w(x: number, y: number, z: number, out = new Vector3()): Vector3 {
    return out.set((x - this.c[0]) * this.k[0], (z - this.c[2]) * this.k[2], -(y - this.c[1]) * this.k[1]);
  }
  /** A maths direction as a world direction (unscaled: for camera presets). */
  dir([x, y, z]: Vec3): Vector3 {
    return new Vector3(x, z, -y).normalize();
  }
  matrix(): Matrix4 {
    const [kx, ky, kz] = this.k;
    const [cx, cy, cz] = this.c;
    return new Matrix4().set(kx, 0, 0, -kx * cx, 0, 0, kz, -kz * cz, 0, -ky, 0, ky * cy, 0, 0, 0, 1);
  }
}

/** A polyline parameterised by arc length, for TubeGeometry. */
class Polyline extends Curve<Vector3> {
  private cum: number[] = [0];
  private pts: Vector3[];
  readonly length: number;
  constructor(pts: Vector3[]) {
    super();
    this.pts = pts;
    for (let i = 1; i < pts.length; i++) this.cum.push(this.cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
    this.length = this.cum[this.cum.length - 1];
  }
  getPoint(u: number, target = new Vector3()): Vector3 {
    const d = clamp(u, 0, 1) * this.length;
    let lo = 0;
    let hi = this.cum.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.cum[mid] <= d) lo = mid;
      else hi = mid;
    }
    const span = this.cum[hi] - this.cum[lo] || 1;
    return target.lerpVectors(this.pts[lo], this.pts[hi], (d - this.cum[lo]) / span);
  }
  getUtoTmapping(u: number): number {
    return u; // already by arc length
  }
}

// ---------------------------------------------------------------- geometry helpers (world space)

const UP = new Vector3(0, 1, 0);

/** A cylinder from a to b. */
function rodGeom(a: Vector3, b: Vector3, r: number): BufferGeometry | null {
  const d = b.clone().sub(a);
  const len = d.length();
  if (!(len > 1e-7)) return null;
  const g = new CylinderGeometry(r, r, len, 10, 1, false);
  g.applyQuaternion(new Quaternion().setFromUnitVectors(UP, d.divideScalar(len)));
  g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
  return g;
}

const _rq = new Quaternion();
const _rs = new Vector3();
const _rm = new Vector3();
const _rd = new Vector3();
/**
 * Places a rod (a mesh of the unit cylinder: radius 1, height 1, along y, centred) from a to b with radius r, by its
 * matrix: moving it later is a new matrix, not a new geometry. False when a and b are too close to show it.
 */
export function placeRod(o: Mesh, a: Vector3, b: Vector3, r: number): boolean {
  _rd.subVectors(b, a);
  const len = _rd.length();
  if (!(len > 1e-7) || !Number.isFinite(len)) return false;
  _rq.setFromUnitVectors(UP, _rd.divideScalar(len));
  o.matrix.compose(_rm.addVectors(a, b).multiplyScalar(0.5), _rq, _rs.set(r, len, r));
  o.matrixWorldNeedsUpdate = true;
  return true;
}

/** An arrow head whose tip is at `tip`, pointing along dir (unit). */
function headGeom(tip: Vector3, dir: Vector3, len: number, r: number): BufferGeometry {
  const g = new ConeGeometry(r, len, 18);
  g.applyQuaternion(new Quaternion().setFromUnitVectors(UP, dir));
  const c = tip.clone().addScaledVector(dir, -len / 2);
  g.translate(c.x, c.y, c.z);
  return g;
}

/** Most tube rings along one dashed line: a long, wiggly line gets coarser dashes rather than a huge geometry. */
const DASH_RINGS = 6000;

/**
 * A dashed tube along a polyline: one tube along the whole path (rings evenly spaced by arc length) that keeps only
 * the faces inside the dashes, in path order, so a draw range draws it from the start. It is rebuilt on every frame
 * a parameter moves it, so it must stay cheap: a rod per dash, merged, took 0.2 s for an ordinary curve and over 20 s
 * for a wiggly graph (thousands of rods).
 */
export function dashesGeom(pts: Vector3[], r: number): BufferGeometry | null {
  if (pts.length < 2) return null;
  const path = new Polyline(pts);
  if (!(path.length > 1e-6)) return null;
  const n = Math.round(clamp(path.length / (DASH / 5), 2, DASH_RINGS));
  const radial = 8;
  const g = new TubeGeometry(path, n, r, radial, false);
  const all = g.index;
  if (!all) return g;
  const per = 6 * radial; // indices per ring-to-ring band (TubeGeometry's order)
  const period = DASH + GAP;
  const keep: number[] = [];
  for (let j = 0; j < n; j++) {
    const s = ((j + 0.5) / n) * path.length;
    if (s % period >= DASH) continue;
    for (let k = j * per; k < (j + 1) * per; k++) keep.push(all.getX(k));
  }
  g.setIndex(keep);
  return g;
}

function tubeGeom(pts: Vector3[], r: number): BufferGeometry | null {
  if (pts.length < 2) return null;
  if (pts.length === 2) return rodGeom(pts[0], pts[1], r);
  const path = new Polyline(pts);
  if (!(path.length > 1e-6)) return null;
  const segments = Math.round(clamp(path.length / 0.01, 24, 1600));
  return new TubeGeometry(path, segments, r, 10, false);
}

/** Pairs of points for LineSegments along a polyline. */
function polylinePairs(pts: Vector3[]): Vector3[] {
  const out: Vector3[] = [];
  for (let i = 1; i < pts.length; i++) out.push(pts[i - 1], pts[i]);
  return out;
}

/** A flat polygon (corners in world space), triangulated in its own plane. */
function polygonGeom(pts: Vector3[]): BufferGeometry | null {
  const n = new Vector3();
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    n.x += (a.y - b.y) * (a.z + b.z);
    n.y += (a.z - b.z) * (a.x + b.x);
    n.z += (a.x - b.x) * (a.y + b.y);
  }
  if (n.lengthSq() < 1e-14) return null;
  n.normalize();
  const u = (Math.abs(n.x) < 0.9 ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0)).cross(n).normalize();
  const v = n.clone().cross(u);
  const flat = pts.map((p) => new Vector2(p.dot(u), p.dot(v)));
  let faces: number[][] = [];
  try {
    faces = ShapeUtils.triangulateShape(flat, []);
  } catch {
    faces = [];
  }
  if (!faces.length) for (let i = 1; i < pts.length - 1; i++) faces.push([0, i, i + 1]); // a fan
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(pts.flatMap((p) => [p.x, p.y, p.z]), 3));
  g.setIndex(faces.flat());
  g.computeVertexNormals();
  return g;
}

/** Lines of latitude and longitude on a unit sphere (a smooth sphere has no edges to draw). */
function sphereRings(): BufferGeometry {
  const pairs: Vector3[] = [];
  const N = 64;
  for (const lat of [-60, -30, 0, 30, 60]) {
    const z = Math.sin(lat * DEG);
    const r = Math.cos(lat * DEG);
    for (let i = 0; i < N; i++) {
      const a = (i / N) * 2 * Math.PI;
      const b = ((i + 1) / N) * 2 * Math.PI;
      pairs.push(new Vector3(r * Math.cos(a), r * Math.sin(a), z), new Vector3(r * Math.cos(b), r * Math.sin(b), z));
    }
  }
  for (let m = 0; m < 6; m++) {
    const lon = (m / 6) * Math.PI;
    for (let i = 0; i < N; i++) {
      const a = (i / N) * 2 * Math.PI;
      const b = ((i + 1) / N) * 2 * Math.PI;
      const p = (t: number) => new Vector3(Math.cos(t) * Math.cos(lon), Math.cos(t) * Math.sin(lon), Math.sin(t));
      pairs.push(p(a), p(b));
    }
  }
  return new BufferGeometry().setFromPoints(pairs);
}

/** Up-and-down lines on a cylinder or cone around maths z (r1 at the bottom, r2 at the top, height h). */
function generatorLines(r1: number, r2: number, h: number, n = 8): BufferGeometry {
  const pairs: Vector3[] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * 2 * Math.PI;
    pairs.push(
      new Vector3(r1 * Math.cos(a), r1 * Math.sin(a), -h / 2),
      new Vector3(r2 * Math.cos(a), r2 * Math.sin(a), h / 2),
    );
  }
  return new BufferGeometry().setFromPoints(pairs);
}

// ---------------------------------------------------------------- compiled objects

interface Fns {
  pts: VarsFn[][];
  center: VarsFn[];
  size: VarsFn[];
  rot: VarsFn[];
  expr: VarsFn | null;
  xyz: VarsFn[];
  range: VarsFn[];
  range2: VarsFn[];
  /** Indices (in scene.params) of the parameters the object reads. */
  deps: number[];
}

function compileObject(o: Scene3DObject, names: string[]): Fns | null {
  const deps = new Set<number>();
  let ok = true;
  const c = (s: string, vars: string[] = []): VarsFn => {
    const r = tryCompileVars(s, [...vars, ...names]);
    if (!r) {
      ok = false;
      return () => NaN;
    }
    for (const u of r.uses) {
      const i = names.indexOf(u);
      if (i >= 0) deps.add(i);
    }
    return r.fn;
  };
  const fieldVars = o.type === 'graph' ? ['x'] : o.type === 'surface' ? ['x', 'y'] : o.type === 'curve' ? ['t'] : ['u', 'v'];
  const fns: Fns = {
    pts: o.points.map((p) => p.map((s) => c(s))),
    center: o.center.map((s) => c(s)),
    size: o.size.map((s) => c(s)),
    rot: o.rotation.map((s) => c(s)),
    expr: o.expr ? c(o.expr, fieldVars) : null,
    xyz: o.x ? [c(o.x, fieldVars), c(o.y, fieldVars), c(o.z, fieldVars)] : [],
    range: o.range.map((s) => c(s)),
    range2: o.range2.map((s) => c(s)),
    deps: [],
  };
  fns.deps = [...deps].sort((a, b) => a - b);
  return ok ? fns : null;
}

/** Evaluates a field's expressions with its own variables in front of the parameters. */
class Ev {
  readonly buf: Float64Array;
  constructor(P: Float64Array, nField: number) {
    this.buf = new Float64Array(nField + P.length);
    this.buf.set(P, nField);
  }
  f1(fn: VarsFn, a: number): number {
    this.buf[0] = a;
    return fn(this.buf);
  }
  f2(fn: VarsFn, a: number, b: number): number {
    this.buf[0] = a;
    this.buf[1] = b;
    return fn(this.buf);
  }
}

type Grow = 'scale' | 'length' | 'draw' | 'none';

interface Mat {
  m: MeshStandardMaterial | LineMaterial;
  base: number; // opacity when fully shown
  color: Color;
  emissive: number; // base emissive intensity (mesh materials)
  width: number; // lines: width as a share of the stage's line width
}

interface Item {
  def: Scene3DObject;
  fns: Fns;
  color: string;
  root: Group; // at the anchor; scaled to grow in
  inner: Group; // undoes the anchor: its children are in world coordinates
  mats: Mat[];
  geoms: BufferGeometry[];
  draws: { geom: BufferGeometry; count: number }[];
  label: CSS2DObject | null;
  /** A solid segment's rod (placed by its matrix: see placeRod), moved without being rebuilt. */
  rod: Mesh | null;
  /** An angle's vertex and its label's anchor (world): the label is turned to stand out from the vertex. */
  bisector: [Vector3, Vector3] | null;
  /** An angle's vertex and the points of its arc or mark (world): a label that has to move keeps off them. */
  marks: Vector3[] | null;
  /** The label's pill inside the element the CSS2DRenderer places: nudged off other labels, dimmed, lit. */
  labelInner: HTMLElement | null;
  built: Float64Array | null; // the values of deps when built
  /** A solid placed by its matrix: its shape (type and size) when built; '' for anything else. */
  shape: string;
  grow: Grow;
  vis: number;
  glow: number;
  /** EMPH while the current step shows or highlights this line (it is drawn over a line it lies on), else 1. */
  emph: number;
  /** An angle's plane normal (world, unit) when built: its label dims while the plane is edge-on. */
  normal: Vector3 | null;
  /** A label that is a number the lecture keeps equal to one parameter: that parameter (shown live). */
  live: Scene3DParam | null;
}

// ---------------------------------------------------------------- the stage

/** World units per maths unit along x, y and z for a scene whose box spans \`spans\`: one scale for every axis (a cube stays a
 *  cube, the longest about L), unless the axes are shown and one is far shorter than the longest (a graph of e^x, a
 *  tall surface): that one is stretched to at least 35% of the longest. A surface over x and y whose z span is the
 *  longer (the saddle z = x^2 - y^2: x, y in +-2, z in +-4, a narrow shape twice as tall as wide) gets its z axis drawn
 *  as long as the longer of x and y; the numbers on the axes keep their values. */
export function axisScales(spans: Vec3, axesShown: boolean, surface: boolean): Vec3 {
  const maxSpan = Math.max(...spans, 1e-9);
  const xy = Math.max(spans[0], spans[1]);
  if (axesShown && surface && xy > 1e-9 && spans[2] > xy)
    return spans.map((s, a) => (a === 2 ? L / s : s > 1e-9 && s < 0.35 * xy ? (L * 0.35) / s : L / xy)) as Vec3;
  return spans.map((s) => (axesShown && s > 1e-9 && s < 0.35 * maxSpan ? (L * 0.35) / s : L / maxSpan)) as Vec3;
}

// ---------------------------------------------------------------- framing in the part of the stage nothing covers

/** The full view the camera renders a window of (setViewOffset): a w x h stage, widened and heightened so that the
 *  centre of the view is the centre of the part `ins` leaves uncovered. */
export function fullViewOf(w: number, h: number, ins: Insets) {
  const dx = ins.left - ins.right;
  const dy = ins.top - ins.bottom;
  return { w: w + Math.abs(dx), h: h + Math.abs(dy), x: dx < 0 ? -dx : 0, y: dy < 0 ? -dy : 0 };
}

/** Distance from `centre` along `dir` at which every point fits the part of a w x h stage that `ins` leaves uncovered
 *  (with a 10% margin), for a camera of vertical field `fov` degrees rendering the full view (fullViewOf). */
export function fitDistanceOf(points: Vector3[], centre: Vector3, dir: Vector3, fov: number, w: number, h: number, ins: Insets) {
  const full = Math.tan((fov * DEG) / 2);
  const fullH = Math.max(1, fullViewOf(w, h, ins).h);
  const tv = (full * Math.max(1, h - ins.top - ins.bottom)) / fullH;
  const th = (full * Math.max(1, w - ins.left - ins.right)) / fullH;
  const right = UP.clone().cross(dir);
  if (right.lengthSq() < 1e-8) right.set(1, 0, 0);
  right.normalize();
  const up = dir.clone().cross(right).normalize();
  const margin = 1.1;
  let d = 0.5;
  for (const c of points) {
    const p = c.clone().sub(centre);
    const depth = p.dot(dir);
    d = Math.max(d, depth + (Math.abs(p.dot(right)) * margin) / th, depth + (Math.abs(p.dot(up)) * margin) / tv);
  }
  return d + 0.05;
}

export type CornerSide = 'top' | 'right';

/** The insets with a box over the top right corner (`corner`: how far it reaches in from the right and down from
 *  the top) kept clear from `side`: below it (a band across the top) or left of it (a band down the right). */
export const cornerInsets = (edges: Insets, corner: { width: number; height: number }, side: CornerSide): Insets =>
  side === 'top'
    ? { ...edges, top: Math.max(edges.top, corner.height) }
    : { ...edges, right: Math.max(edges.right, corner.width) };

/** Which side the drawing passes the corner box on, seen from `dir`: the one that shows it larger (pixels per world
 *  unit go with the full view's height over the distance). From `current`, it changes only for 3% or more, so the
 *  drawing does not hop from one side to the other as the student moves a slider. */
export function cornerSide(
  points: Vector3[], centre: Vector3, dir: Vector3, fov: number, w: number, h: number,
  edges: Insets, corner: { width: number; height: number }, current: CornerSide = 'top',
): CornerSide {
  if (!(corner.width > 0 && corner.height > 0)) return current;
  const scale = (side: CornerSide) => {
    const ins = cornerInsets(edges, corner, side);
    return fullViewOf(w, h, ins).h / fitDistanceOf(points, centre, dir, fov, w, h, ins);
  };
  const other: CornerSide = current === 'top' ? 'right' : 'top';
  return scale(other) > scale(current) * 1.03 ? other : current;
}

/** Builds the stage in `host`. Throws when WebGL is not available (the caller shows a plain line). */
export function mountScene(host: HTMLElement, scene: Scene3D, opts: StageOptions): Stage3D {
  // The values every expression reads: the parameters, then the defs worked out from them (withDefs).
  const defs = scene.defs ?? [];
  const np = scene.params.length;
  const names = [...scene.params.map((p) => p.name), ...defs.map((d) => d.name)];
  const defFns = defs.map((d) => tryCompileVars(d.expr, names)?.fn ?? (() => NaN));
  const withDefs = (vals: Float64Array): Float64Array => {
    for (let k = 0; k < defFns.length; k++) vals[np + k] = defFns[k](vals);
    return vals;
  };
  const P = withDefs(new Float64Array([...scene.params.map((p) => p.value), ...defs.map(() => NaN)]));
  const accent = opts.accent || '#7aa7ff';
  const palette = [accent, '#ff9f6b', '#6fdc9a', '#f2c94c', '#c792ea', '#5fd4e6', '#ff7aa8', '#b8c4ff'];
  const ax = scene.axes;

  const colors = distinctCurveColors(scene.objects, stageColors(scene.objects.map((o) => o.color), palette), palette);
  const items: Item[] = [];
  scene.objects.forEach((def, i) => {
    const fns = compileObject(def, names);
    if (!fns) return;
    const root = new Group();
    const inner = new Group();
    root.add(inner);
    items.push({
      def,
      fns,
      color: colors[i],
      root,
      inner,
      mats: [],
      geoms: [],
      draws: [],
      label: null,
      labelInner: null,
      rod: null,
      bisector: null,
      marks: null,
      built: null,
      shape: '',
      grow:
        def.type === 'segment' || def.type === 'vector'
          ? 'length'
          : def.type === 'graph' || def.type === 'curve'
            ? 'draw'
            : def.type === 'label'
              ? 'none'
              : 'scale', // an angle grows from its vertex
      vis: -1,
      glow: -1,
      emph: 1,
      normal: null,
      live: null,
    });
  });
  // A label that is a number the lecture keeps equal to one parameter (vector3d: '3' by the x component, vx = 3) shows
  // that parameter's value live, so it stays true while a student moves the sliders.
  const live = liveNumberLabels(scene);
  for (const it of items) {
    const pi = scene.params.findIndex((p) => p.name === live[it.def.id]);
    if (pi < 0) continue;
    it.live = scene.params[pi];
    if (!it.fns.deps.includes(pi)) it.fns.deps = [...it.fns.deps, pi].sort((a, b) => a - b);
  }

  // ---------- where things are, in maths coordinates (for framing) ----------
  const probe = (it: Item, params: Float64Array): Vec3[] => {
    const f = it.fns;
    const out: Vec3[] = [];
    const v3 = (fs: VarsFn[]) => fs.map((fn) => fn(params)) as Vec3;
    const push = (p: Vec3) => p.every(Number.isFinite) && out.push(p);
    const ev1 = new Ev(params, 1);
    const ev2 = new Ev(params, 2);
    const range = (fs: VarsFn[], lo: number, hi: number): [number, number] => {
      const a = fs.length === 2 ? fs[0](params) : lo;
      const b = fs.length === 2 ? fs[1](params) : hi;
      return Number.isFinite(a) && Number.isFinite(b) && a !== b ? [Math.min(a, b), Math.max(a, b)] : [lo, hi];
    };
    switch (it.def.type) {
      case 'point':
      case 'label':
      case 'segment':
      case 'vector':
      case 'polygon':
        f.pts.forEach((p) => push(v3(p)));
        break;
      case 'angle': {
        // its vertex +/- its radius (a degenerate state reaches nowhere)
        const sh = angleShape(v3(f.pts[0]), v3(f.pts[1]), v3(f.pts[2]), f.size.length ? f.size[0](params) : null);
        if (sh) [sh.vertex, ...angleReach(sh)].forEach(push);
        break;
      }
      case 'box':
      case 'sphere':
      case 'cylinder':
      case 'cone':
      case 'torus': {
        const c = v3(f.center);
        const s = f.size.map((fn) => Math.abs(fn(params)));
        const r =
          it.def.type === 'box'
            ? Math.hypot(s[0], s[1], s[2]) / 2
            : it.def.type === 'sphere'
              ? s[0]
              : it.def.type === 'torus'
                ? s[0] + s[1]
                : Math.hypot(s[0], s[1] / 2);
        if (!Number.isFinite(r)) break;
        const turned = f.rot.length === 3 && f.rot.some((fn) => Math.abs(fn(params)) > 1e-9);
        if (turned) {
          for (const dx of [-r, r]) for (const dy of [-r, r]) for (const dz of [-r, r]) push([c[0] + dx, c[1] + dy, c[2] + dz]);
          break;
        }
        // Not turned: points on the shape itself, so the camera frames the solid, not the cube round its bounding
        // sphere (the cylinder filled a fraction of the stage).
        const ring = (rad: number, z: number) => {
          for (let k = 0; k < 12; k++) push([c[0] + rad * Math.cos((k * Math.PI) / 6), c[1] + rad * Math.sin((k * Math.PI) / 6), z]);
        };
        if (it.def.type === 'box')
          for (const dx of [-s[0] / 2, s[0] / 2]) for (const dy of [-s[1] / 2, s[1] / 2]) for (const dz of [-s[2] / 2, s[2] / 2]) push([c[0] + dx, c[1] + dy, c[2] + dz]);
        else if (it.def.type === 'sphere') {
          const q = s[0] / Math.sqrt(3);
          for (const a of [0, 1, 2]) for (const sg of [-1, 1]) push(c.map((v, i) => v + (i === a ? sg * s[0] : 0)) as Vec3);
          for (const dx of [-q, q]) for (const dy of [-q, q]) for (const dz of [-q, q]) push([c[0] + dx, c[1] + dy, c[2] + dz]);
        } else if (it.def.type === 'torus') {
          const t = Math.min(s[1], s[0]);
          ring(s[0] + t, c[2] - t);
          ring(s[0] + t, c[2] + t);
        } else {
          ring(s[0], c[2] - s[1] / 2);
          if (it.def.type === 'cone') push([c[0], c[1], c[2] + s[1] / 2]);
          else ring(s[0], c[2] + s[1] / 2);
        }
        break;
      }
      case 'graph': {
        const [x0, x1] = range(f.range, ax.x.min, ax.x.max);
        for (let i = 0; i <= 48; i++) {
          const x = x0 + ((x1 - x0) * i) / 48;
          const y = ev1.f1(f.expr!, x);
          if (Number.isFinite(y)) push([x, clamp(y, ax.y.min, ax.y.max), 0]);
        }
        break;
      }
      case 'surface':
        for (let j = 0; j <= 16; j++)
          for (let i = 0; i <= 16; i++) {
            const x = ax.x.min + ((ax.x.max - ax.x.min) * i) / 16;
            const y = ax.y.min + ((ax.y.max - ax.y.min) * j) / 16;
            const z = ev2.f2(f.expr!, x, y);
            if (Number.isFinite(z)) push([x, y, clamp(z, ax.z.min, ax.z.max)]);
          }
        break;
      case 'curve': {
        const [t0, t1] = range(f.range, 0, 1);
        for (let i = 0; i <= 64; i++) {
          const t = t0 + ((t1 - t0) * i) / 64;
          push([ev1.f1(f.xyz[0], t), ev1.f1(f.xyz[1], t), ev1.f1(f.xyz[2], t)]);
        }
        break;
      }
      case 'psurface': {
        const [u0, u1] = range(f.range, 0, 1);
        const [w0, w1] = range(f.range2, 0, 1);
        for (let j = 0; j <= 16; j++)
          for (let i = 0; i <= 16; i++) {
            const u = u0 + ((u1 - u0) * i) / 16;
            const v = w0 + ((w1 - w0) * j) / 16;
            push([ev2.f2(f.xyz[0], u, v), ev2.f2(f.xyz[1], u, v), ev2.f2(f.xyz[2], u, v)]);
          }
        break;
      }
    }
    return out;
  };

  // Parameter values over the whole animation: the start, and the middle and end of every step's change.
  const snaps: Float64Array[] = [new Float64Array(P)];
  {
    const snap = new Float64Array(P);
    for (const s of scene.steps) {
      const mid = new Float64Array(snap);
      for (const { param, to } of s.set) {
        const i = names.indexOf(param);
        if (i < 0 || i >= np) continue;
        mid[i] = (snap[i] + to) / 2;
        snap[i] = to;
      }
      if (s.set.length) snaps.push(withDefs(mid), new Float64Array(withDefs(snap)));
    }
  }
  // Where each object reaches over the animation (a box per object and moment, in maths coordinates).
  const reach: [Vec3, Vec3][] = [];
  /** The points themselves (not their boxes' corners, mostly empty space round a leaning cell): what the camera
   *  frames. */
  const framePts: Vec3[] = [];
  for (const it of items)
    for (const ps of snaps) {
      const pts = probe(it, ps);
      if (!pts.length) continue;
      for (const q of pts) framePts.push(q);
      const a: Vec3 = [Infinity, Infinity, Infinity];
      const b: Vec3 = [-Infinity, -Infinity, -Infinity];
      for (const q of pts)
        for (let i = 0; i < 3; i++) {
          a[i] = Math.min(a[i], q[i]);
          b[i] = Math.max(b[i], q[i]);
        }
      reach.push([a, b]);
    }

  /** The box that sets the scale: the server's bounds (every state of every step), or our own from `reach`; with
   *  the axes when they are shown. */
  const lo: Vec3 = [Infinity, Infinity, Infinity];
  const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  const grow = (p: Vec3) =>
    p.forEach((v, a) => {
      lo[a] = Math.min(lo[a], v);
      hi[a] = Math.max(hi[a], v);
    });
  if (scene.bounds) {
    grow(scene.bounds[0]);
    grow(scene.bounds[1]);
  } else
    for (const [a, b] of reach) {
      grow(a);
      grow(b);
    }
  if (ax.show || !Number.isFinite(lo[0])) {
    grow([ax.x.min, ax.y.min, ax.z.min]);
    grow([ax.x.max, ax.y.max, ax.z.max]);
  }
  // A drawing that lies in the plane z = 0 (graphs) gets no z axis: seen from the top it points at the viewer, and
  // its numbers piled up on the origin.
  const flat =
    ax.show &&
    reach.length > 0 &&
    items.every((it) => it.def.type !== 'surface' && it.def.type !== 'psurface') &&
    reach.every(([a, b]) => Math.abs(a[2]) < 1e-9 && Math.abs(b[2]) < 1e-9);
  if (flat) {
    lo[2] = Math.min(0, Math.max(lo[2], 0));
    hi[2] = Math.max(0, Math.min(hi[2], 0));
  }
  const spans = [0, 1, 2].map((a) => hi[a] - lo[a]);
  // the scale of each axis: axisScales
  const k = axisScales(spans as Vec3, ax.show, items.some((it) => it.def.type === 'surface' || it.def.type === 'psurface'));
  const frame = new Frame([0, 1, 2].map((a) => (lo[a] + hi[a]) / 2) as Vec3, k);
  const w = (x: number, y: number, z: number) => frame.w(x, y, z);
  const W = frame.matrix();

  // ---------- three.js basics ----------
  const renderer = new WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(BG, 1);
  renderer.domElement.className = 'p3-canvas';
  host.appendChild(renderer.domElement);
  const labels = new CSS2DRenderer();
  labels.domElement.className = 'p3-labels';
  host.appendChild(labels.domElement);

  const world = new Scene();
  const camera = new PerspectiveCamera(36, 1, 0.01, 200);
  // Lit from above and in front, with a cool light from behind for the rims: solids read as solids and spheres as
  // atoms (a dim ambient light, so their shading shows).
  world.add(new AmbientLight('#ffffff', 0.32));
  world.add(new HemisphereLight('#dfe8ff', '#1c2028', 0.85));
  const sun = new DirectionalLight('#ffffff', 2.1);
  sun.position.set(3, 5, 4);
  world.add(sun);
  const fill = new DirectionalLight('#9fb8ff', 0.55);
  fill.position.set(-4, 1, -3);
  world.add(fill);
  const rim = new DirectionalLight('#b9ccff', 0.9);
  rim.position.set(-2, 3, -6);
  world.add(rim);
  // a gentle depth cue: the back of the drawing fades a little into the background (near and far follow the camera)
  const fog = new Fog(BG, 1, 100);
  world.fog = fog;

  const guides = new Group();
  world.add(guides);
  const content = new Group();
  world.add(content);
  for (const it of items) content.add(it.root);
  const guideGeoms: BufferGeometry[] = [];
  const guideMats: Material[] = [];
  const guideTextures: CanvasTexture[] = [];

  // ---------- materials ----------
  const meshMat = (it: Item, opacity: number, emissive = 0.14, side: Side = DoubleSide, roughness = 0.42): Mat => {
    const m = new MeshStandardMaterial({
      color: it.color, roughness, metalness: 0.05, emissive: it.color, emissiveIntensity: emissive,
      transparent: true, opacity, depthWrite: opacity >= 0.95, side,
    });
    const mat = { m, base: opacity, color: new Color(it.color), emissive, width: 0 };
    it.mats.push(mat);
    return mat;
  };
  /** Lines a few pixels wide (their width follows the stage's size: see resize). */
  let linePx = 2;
  const lineMat = (it: Item, opacity = 1, dashed = false, color = it.color, width = 1): Mat => {
    const m = new LineMaterial({
      color, linewidth: linePx * width, worldUnits: false, transparent: true, opacity, dashed, dashSize: DASH,
      gapSize: GAP, fog: true,
    });
    m.resolution.set(Math.max(1, host.clientWidth), Math.max(1, host.clientHeight));
    const mat = { m, base: opacity, color: new Color(color), emissive: 0, width };
    it.mats.push(mat);
    return mat;
  };
  /** Materials are made once per object (on its first build) and kept; geometry is rebuilt. */
  const matsOf = new Map<Item, Record<string, Mat>>();
  const mat = (it: Item, key: string, make: () => Mat): Mat['m'] => {
    let rec = matsOf.get(it);
    if (!rec) matsOf.set(it, (rec = {}));
    return (rec[key] ??= make()).m;
  };

  // ---------- building one object ----------
  const add = (it: Item, g: BufferGeometry | null, m: Mat['m'], kind: 'mesh' | 'lines' = 'mesh', draw = false) => {
    if (!g) return;
    if (kind === 'lines') {
      // pairs of points (every line geometry here is unindexed pairs) as wide lines
      const flat = g.index ? g.toNonIndexed() : g;
      const pos = flat.getAttribute('position');
      const n = pos ? pos.count - (pos.count % 2) : 0;
      const arr = n ? new Float32Array((pos.array as Float32Array).subarray(0, n * 3)) : null;
      if (flat !== g) flat.dispose();
      g.dispose();
      if (!arr) return;
      const lg = new LineSegmentsGeometry().setPositions(arr);
      it.geoms.push(lg);
      const o = new LineSegments2(lg, m as LineMaterial);
      if (it.emph > 1) o.renderOrder = 2;
      if ((m as LineMaterial).dashed) o.computeLineDistances();
      it.inner.add(o);
      if (draw) it.draws.push({ geom: lg, count: n });
      return;
    }
    it.geoms.push(g);
    const o = new Mesh(g, m);
    if ((m as MeshStandardMaterial).opacity < 0.95) o.renderOrder = 1;
    if (it.emph > 1) o.renderOrder = 2;
    it.inner.add(o);
    if (draw) it.draws.push({ geom: g, count: g.index ? g.index.count : g.getAttribute('position').count });
  };
  /** A line drawn as the style asks: a tube, dashes, or a thin line. */
  const stroke = (it: Item, pts: Vector3[], r: number, draw = false) => {
    if (pts.length < 2) return;
    const style = it.def.style;
    if (style === 'wire') {
      add(it, new BufferGeometry().setFromPoints(polylinePairs(pts)), mat(it, 'line', () => lineMat(it)), 'lines', draw);
      return;
    }
    const m = mat(it, 'tube', () => meshMat(it, it.def.opacity, 0.16));
    add(it, style === 'dashed' ? dashesGeom(pts, r) : tubeGeom(pts, r), m, 'mesh', draw);
  };
  const setLabel = (it: Item, at: Vector3, cx: number, cy: number, cls: string, shown?: string) => {
    const pi = it.live ? scene.params.indexOf(it.live) : -1;
    const text = shown ?? (it.live && pi >= 0 ? formatValue(P[pi], it.live.unit) : labelText(it.def.label));
    if (!text) return;
    if (it.labelInner && it.labelInner.textContent !== text) it.labelInner.textContent = text; // an angle's value
    if (!it.label) {
      const { outer, inner } = labelElements(text, cls);
      // the label's frame is its object's colour, so a name is matched with its arc or edge at a glance
      if (/^#[0-9a-f]{6}$/i.test(it.color)) inner.style.borderColor = `${it.color}b3`;
      it.label = new CSS2DObject(outer);
      it.labelInner = inner;
      it.root.add(it.label);
    }
    it.label.visible = true;
    it.label.center.set(cx, cy);
    it.label.position.copy(at).sub(it.root.position);
  };

  /** Where a solid is, its size, and the matrix that places its shape (made around the origin) in the world. */
  const solidAt = (it: Item) => {
    const f = it.fns;
    const c = f.center.map((fn) => fn(P)) as Vec3;
    const s = f.size.map((fn) => Math.max(1e-4, Math.abs(fn(P))));
    const rot = f.rot.length === 3 ? f.rot.map((fn) => fn(P) * DEG) : [0, 0, 0];
    if (![...c, ...s, ...rot].every(Number.isFinite)) return null;
    const M = W.clone()
      .multiply(new Matrix4().makeTranslation(c[0], c[1], c[2]))
      .multiply(new Matrix4().makeRotationFromEuler(new Euler(rot[0], rot[1], rot[2], 'XYZ')));
    const type = it.def.type;
    const top = type === 'box' ? s[2] / 2 : type === 'sphere' ? s[0] : type === 'torus' ? Math.min(s[1], s[0]) : s[1] / 2;
    return { c, s, M, top, key: `${type}|${it.def.style}|${s.join(',')}` };
  };
  /** Latitude and longitude lines of a unit sphere, made once per stage (never shared between stages: three.js
   *  keeps a listener of every renderer on a shared geometry). */
  let rings: BufferGeometry | null = null;
  /** The unit cylinder every rod (solid segments, polygon edges) is drawn with, placed by its matrix. */
  const unitRod = new CylinderGeometry(1, 1, 1, 12, 1, false);
  const addRod = (it: Item, a: Vector3, b: Vector3, r: number, m: Mat['m']): Mesh | null => {
    const o = new Mesh(unitRod, m);
    o.matrixAutoUpdate = false;
    if (it.emph > 1) o.renderOrder = 2;
    if (!placeRod(o, a, b, r)) return null;
    it.inner.add(o);
    return o;
  };
  /** A solid segment whose ends moved: its rod gets a new matrix, its label a new place (test2 #7: twelve edges of a
   *  cell rebuilt on every frame of a change). False when it must be built again. */
  const moveRod = (it: Item): boolean => {
    const o = it.rod;
    if (!o) return false;
    const f = it.fns;
    const a = w(f.pts[0][0](P), f.pts[0][1](P), f.pts[0][2](P));
    const b = w(f.pts[1][0](P), f.pts[1][1](P), f.pts[1][2](P));
    if (![a.x, a.y, a.z, b.x, b.y, b.z].every(Number.isFinite)) return false;
    it.root.position.copy(a);
    it.inner.position.copy(a).negate();
    if (!placeRod(o, a, b, ROD * 0.9 * thick * it.emph)) return false;
    if (it.label) {
      it.label.visible = true;
      it.label.position.copy(a.clone().lerp(b, 0.5)).sub(it.root.position);
    }
    it.built = new Float64Array(it.fns.deps.map((i) => P[i]));
    return true;
  };
  /**
   * A solid whose shape is unchanged only moves or turns: its matrix changes and nothing is rebuilt. (On test2 #7,
   * thirteen atoms riding on a stretching prism were rebuilt every frame, up to 0.3 s a frame in headless Chrome.)
   * False when it must be built again.
   */
  const move = (it: Item): boolean => {
    if (!it.shape || !it.geoms.length) return false;
    const sol = solidAt(it);
    if (!sol || sol.key !== it.shape) return false;
    const { c, M } = sol;
    const at = w(c[0], c[1], c[2]);
    it.root.position.copy(at);
    it.inner.position.copy(at).negate();
    for (const o of it.inner.children) {
      o.matrix.copy(M);
      o.matrixWorldNeedsUpdate = true;
    }
    if (it.label) {
      it.label.visible = true;
      it.label.position.copy(w(c[0], c[1], c[2] + sol.top)).sub(it.root.position);
    }
    it.built = new Float64Array(it.fns.deps.map((i) => P[i]));
    return true;
  };

  const build = (it: Item) => {
    const f = it.fns;
    for (const g of it.geoms) g.dispose();
    it.geoms = [];
    it.draws = [];
    it.shape = '';
    it.inner.clear();
    it.rod = null;
    it.bisector = null;
    it.marks = null;
    if (it.label) it.label.visible = false; // shown again where the object is drawn; never left where it was
    const def = it.def;
    const v3 = (fs: VarsFn[]) => w(fs[0](P), fs[1](P), fs[2](P));
    const ok = (p: Vector3) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z);
    const anchorAt = (p: Vector3) => {
      it.root.position.copy(p);
      it.inner.position.copy(p).negate();
    };
    const range = (fs: VarsFn[], lo: number, hi: number): [number, number] | null => {
      if (fs.length !== 2) return [lo, hi];
      const a = fs[0](P);
      const b = fs[1](P);
      return Number.isFinite(a) && Number.isFinite(b) && a !== b ? [a, b] : null;
    };

    switch (def.type) {
      case 'point': {
        const p = v3(f.pts[0]);
        if (!ok(p)) break;
        anchorAt(p);
        const g = new SphereGeometry(DOT * Math.sqrt(thick), 20, 14);
        g.translate(p.x, p.y, p.z);
        add(it, g, mat(it, 'dot', () => meshMat(it, 1, 0.25)));
        setLabel(it, p, 0, 1, 'p3-label p3-point');
        break;
      }
      case 'label': {
        const p = v3(f.pts[0]);
        if (!ok(p)) break;
        anchorAt(p);
        setLabel(it, p, 0.5, 0.5, 'p3-label p3-text');
        if (it.labelInner) it.labelInner.style.color = it.color;
        break;
      }
      case 'segment':
      case 'vector': {
        const a = v3(f.pts[0]);
        const b = v3(f.pts[1]);
        if (!ok(a) || !ok(b)) break;
        anchorAt(a);
        const d = b.clone().sub(a);
        const len = d.length();
        const tk = thick * it.emph;
        if (def.type === 'vector' && len > 1e-6) {
          const dir = d.clone().divideScalar(len);
          const hl = Math.min(HEAD_LEN * tk, len * 0.45);
          stroke(it, [a, b.clone().addScaledVector(dir, -hl * 0.9)], ROD * tk);
          add(it, headGeom(b, dir, hl, HEAD_R * tk * (hl / (HEAD_LEN * tk)) ** 0.5), mat(it, 'head', () => meshMat(it, Math.max(0.6, def.opacity), 0.2)));
          setLabel(it, b.clone().addScaledVector(dir, 0.06), 0, 1, 'p3-label');
        } else {
          if (def.type === 'segment' && def.style === 'solid')
            it.rod = addRod(it, a, b, ROD * 0.9 * tk, mat(it, 'tube', () => meshMat(it, def.opacity, 0.16)));
          else stroke(it, [a, b], ROD * 0.9 * tk);
          setLabel(it, a.clone().lerp(b, 0.5), 0, 1, 'p3-label');
        }
        break;
      }
      case 'polygon': {
        const pts = f.pts.map(v3);
        if (!pts.every(ok)) break;
        const centroid = pts.reduce((s, p) => s.add(p), new Vector3()).divideScalar(pts.length);
        anchorAt(centroid);
        if (def.style !== 'wire') {
          const fillOpacity = Math.min(0.6, def.opacity) * (def.style === 'dashed' ? 0.5 : 1);
          add(it, polygonGeom(pts), mat(it, 'fill', () => meshMat(it, fillOpacity, 0.1)));
        }
        const ring = [...pts, pts[0]];
        if (def.style === 'solid') {
          const m = mat(it, 'edge', () => meshMat(it, 1, 0.2));
          for (let i = 1; i < ring.length; i++) addRod(it, ring[i - 1], ring[i], ROD * 0.7 * thick, m);
        } else {
          add(it, new BufferGeometry().setFromPoints(polylinePairs(ring)), mat(it, 'line', () => lineMat(it, 1, def.style === 'dashed')), 'lines');
        }
        setLabel(it, centroid, 0.5, 0.5, 'p3-label');
        break;
      }
      case 'box':
      case 'sphere':
      case 'cylinder':
      case 'cone':
      case 'torus': {
        const sol = solidAt(it);
        if (!sol) break;
        const { c, s, M } = sol;
        anchorAt(w(c[0], c[1], c[2]));
        let geom: BufferGeometry;
        let edges: BufferGeometry;
        switch (def.type) {
          case 'box':
            geom = new BoxGeometry(s[0], s[1], s[2]);
            edges = new EdgesGeometry(geom);
            break;
          case 'sphere':
            geom = new SphereGeometry(s[0], 40, 28);
            // a solid sphere is shaded like an atom, with no lines on it; a wire or dashed one shows its rings
            edges = def.style === 'solid' ? new BufferGeometry() : (rings ??= sphereRings()).clone().scale(s[0], s[0], s[0]);
            break;
          case 'torus':
            geom = new TorusGeometry(s[0], Math.min(s[1], s[0]), 24, 72);
            edges = new EdgesGeometry(new TorusGeometry(s[0], Math.min(s[1], s[0]), 6, 24), 1);
            break;
          default: {
            const cone = def.type === 'cone';
            geom = cone ? new ConeGeometry(s[0], s[1], 56) : new CylinderGeometry(s[0], s[0], s[1], 56);
            geom.rotateX(Math.PI / 2); // axis along maths z, apex up
            const caps = new EdgesGeometry(geom, 30);
            const sides = generatorLines(s[0], cone ? 0 : s[0], s[1]);
            edges = mergeGeometries([caps, sides]) ?? caps;
            if (edges !== caps) caps.dispose();
            sides.dispose();
          }
        }
        // Dashes are measured along the geometry itself, so a dashed solid is built in place; any other keeps its
        // shape and is placed by its matrix, so moving or turning it later does not rebuild it (see `move`).
        const byMatrix = def.style !== 'dashed';
        if (!byMatrix) {
          geom.applyMatrix4(M);
          edges.applyMatrix4(M);
        }
        if (def.style !== 'wire') {
          const opacity = def.style === 'dashed' ? Math.min(0.25, def.opacity) : def.opacity;
          const atom = def.type === 'sphere' && opacity >= 0.95;
          add(it, geom, mat(it, 'fill', () => (atom ? meshMat(it, opacity, 0.03, FrontSide, 0.3) : meshMat(it, opacity, 0.08))));
        } else geom.dispose();
        const edgeColor = '#' + new Color(it.color).offsetHSL(0, 0, 0.12).getHexString();
        add(it, edges, mat(it, 'edge', () => lineMat(it, 0.95, def.style === 'dashed', edgeColor)), 'lines');
        if (byMatrix) {
          for (const o of it.inner.children) {
            o.matrixAutoUpdate = false;
            o.matrix.copy(M);
            o.matrixWorldNeedsUpdate = true;
          }
          it.shape = sol.key;
        }
        setLabel(it, w(c[0], c[1], c[2] + sol.top), 0.5, 1, 'p3-label');
        break;
      }
      case 'angle': {
        // worked out in maths coordinates and mapped like everything else (a stretched axis stretches it honestly);
        // degenerate in this state: nothing is drawn, and its label is hidden
        const m3 = (fs: VarsFn[]) => fs.map((fn) => fn(P)) as Vec3;
        const sh = angleShape(m3(f.pts[0]), m3(f.pts[1]), m3(f.pts[2]), f.size.length ? f.size[0](P) : null);
        if (!sh) break;
        anchorAt(w(...sh.vertex));
        const line = sh.line.map((p) => w(...p));
        if (def.style === 'solid' && sh.fan.length >= 3) {
          const fan = sh.fan.map((p) => w(...p));
          const g = new BufferGeometry();
          g.setAttribute('position', new Float32BufferAttribute(fan.flatMap((p) => [p.x, p.y, p.z]), 3));
          const idx: number[] = [];
          for (let i = 1; i < fan.length - 1; i++) idx.push(0, i, i + 1);
          g.setIndex(idx);
          g.computeVertexNormals();
          add(it, g, mat(it, 'fill', () => meshMat(it, Math.min(def.opacity, 0.35), 0.12)));
        }
        const r = ROD * 0.85 * thick;
        add(it, def.style === 'dashed' ? dashesGeom(line, r) : tubeGeom(line, r), mat(it, 'tube', () => meshMat(it, def.opacity, 0.18)));
        const at = w(...sh.label);
        setLabel(it, at, 0.5, 0.5, 'p3-label p3-angle', angleLabel(def.label, sh.degrees));
        it.bisector = [w(...sh.vertex), at];
        const v0 = w(...sh.vertex);
        const n = line[0].clone().sub(v0).cross(line[line.length - 1].clone().sub(v0));
        it.normal = n.lengthSq() > 1e-16 ? n.normalize() : null;
        it.marks = [it.bisector[0], ...line];
        break;
      }
      case 'graph': {
        const xr = range(f.range, ax.x.min, ax.x.max);
        if (!xr) break;
        const ev = new Ev(P, 1);
        const fx = (x: number) => ev.f1(f.expr!, x);
        let segs = sampleSegments(fx, Math.min(...xr), Math.max(...xr), ax.y.min, ax.y.max, 600);
        if (!segs.length) {
          // the axes do not show it: fit the y window to the values
          const ys: number[] = [];
          for (let i = 0; i <= 200; i++) ys.push(fx(xr[0] + ((xr[1] - xr[0]) * i) / 200));
          const fit = fitRange(ys, ax.y.min, ax.y.max);
          segs = sampleSegments(fx, Math.min(...xr), Math.max(...xr), fit.min, fit.max, 600);
        }
        let longest: Vector3[] = [];
        let first: Vector3 | null = null;
        for (const s of segs) {
          const pts = s.map(([x, y]) => w(x, y, 0));
          first ??= pts[0];
          if (pts.length > longest.length) longest = pts;
          stroke(it, pts, TUBE * thick * it.emph, true);
        }
        anchorAt(first ?? w(0, 0, 0));
        if (longest.length) setLabel(it, longest[longest.length - 1], 0, 1, 'p3-label');
        break;
      }
      case 'curve': {
        const tr = range(f.range, 0, 1);
        if (!tr) break;
        const ev = new Ev(P, 1);
        const n = 480;
        let run: Vector3[] = [];
        const runs: Vector3[][] = [];
        for (let i = 0; i <= n; i++) {
          const t = tr[0] + ((tr[1] - tr[0]) * i) / n;
          const p = w(ev.f1(f.xyz[0], t), ev.f1(f.xyz[1], t), ev.f1(f.xyz[2], t));
          if (ok(p)) run.push(p);
          else if (run.length) {
            runs.push(run);
            run = [];
          }
        }
        if (run.length) runs.push(run);
        for (const r of runs) stroke(it, r, TUBE * thick * it.emph, true);
        const all = runs.flat();
        anchorAt(all[0] ?? w(0, 0, 0));
        // An angle's arc names its angle out along the bisector, past the arc (arcs from one corner, alpha, beta and
        // gamma, put their names apart); any other curve at its middle.
        if (all.length && it.def.label) {
          const at = runs.length === 1 ? arcLabelAt(all.map((p) => [p.x, p.y, p.z] as const)) : null;
          if (at) setLabel(it, new Vector3(...at), 0.5, 0.5, 'p3-label');
          else setLabel(it, all[Math.round((all.length - 1) * CURVE_LABEL_AT)], 0, 1, 'p3-label');
        }
        break;
      }
      case 'surface':
      case 'psurface': {
        const ps = def.type === 'psurface';
        const ur = ps ? range(f.range, 0, 1) : ([ax.x.min, ax.x.max] as [number, number]);
        const vr = ps ? range(f.range2, 0, 1) : ([ax.y.min, ax.y.max] as [number, number]);
        if (!ur || !vr) break;
        const nu = ps ? 64 : 60;
        const nv = ps ? 48 : 60;
        const ev = new Ev(P, 2);
        const zSpan = ax.z.max - ax.z.min;
        const zlo = ax.z.min - 0.05 * zSpan;
        const zhi = ax.z.max + 0.05 * zSpan;
        const pos = new Float32Array((nu + 1) * (nv + 1) * 3);
        const height = new Float32Array((nu + 1) * (nv + 1));
        const valid = new Uint8Array((nu + 1) * (nv + 1));
        const p = new Vector3();
        let zmin = Infinity;
        let zmax = -Infinity;
        for (let j = 0; j <= nv; j++)
          for (let i = 0; i <= nu; i++) {
            const u = ur[0] + ((ur[1] - ur[0]) * i) / nu;
            const v = vr[0] + ((vr[1] - vr[0]) * j) / nv;
            const x = ps ? ev.f2(f.xyz[0], u, v) : u;
            const y = ps ? ev.f2(f.xyz[1], u, v) : v;
            const z = ps ? ev.f2(f.xyz[2], u, v) : ev.f2(f.expr!, u, v);
            const kk = j * (nu + 1) + i;
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
            if (!ps && (z < zlo || z > zhi)) continue;
            frame.w(x, y, z, p);
            pos.set([p.x, p.y, p.z], kk * 3);
            height[kk] = z;
            valid[kk] = 1;
            zmin = Math.min(zmin, z);
            zmax = Math.max(zmax, z);
          }
        const idx: number[] = [];
        for (let j = 0; j < nv; j++)
          for (let i = 0; i < nu; i++) {
            const a = j * (nu + 1) + i;
            const b = a + 1;
            const d = a + nu + 1;
            const e = d + 1;
            if (valid[a] && valid[b] && valid[d]) idx.push(a, b, d);
            if (valid[b] && valid[e] && valid[d]) idx.push(b, e, d);
          }
        if (!idx.length) break;
        // centre of the drawn part, for growing in and the label
        let n = 0;
        const centre = new Vector3();
        for (let kk = 0; kk < valid.length; kk++)
          if (valid[kk]) {
            centre.x += pos[kk * 3];
            centre.y += pos[kk * 3 + 1];
            centre.z += pos[kk * 3 + 2];
            n++;
          }
        centre.divideScalar(n);
        anchorAt(centre);
        if (def.style === 'solid') {
          const col = new Float32Array(pos.length);
          const base = new Color(it.color);
          const c = new Color();
          const span = zmax - zmin || 1;
          for (let kk = 0; kk < valid.length; kk++) {
            c.copy(base).offsetHSL(0, 0, ((height[kk] - zmin) / span - 0.5) * 0.3);
            col.set([c.r, c.g, c.b], kk * 3);
          }
          const g = new BufferGeometry();
          g.setAttribute('position', new Float32BufferAttribute(pos, 3));
          g.setAttribute('color', new Float32BufferAttribute(col, 3));
          g.setIndex(idx);
          g.computeVertexNormals();
          add(
            it,
            g,
            mat(it, 'fill', () => {
              const m = meshMat(it, def.opacity, 0.06);
              (m.m as MeshStandardMaterial).vertexColors = true;
              (m.m as MeshStandardMaterial).color.set('#ffffff');
              (m.m as MeshStandardMaterial).polygonOffset = true;
              (m.m as MeshStandardMaterial).polygonOffsetFactor = 1;
              (m.m as MeshStandardMaterial).polygonOffsetUnits = 1;
              return m;
            }),
          );
        }
        // grid lines: faint over a solid surface, the surface itself when it is a wire
        const every = def.style === 'solid' ? 6 : 3;
        const pairs: number[] = [];
        const at = (kk: number) => pairs.push(pos[kk * 3], pos[kk * 3 + 1], pos[kk * 3 + 2]);
        for (let j = 0; j <= nv; j++)
          for (let i = 0; i <= nu; i++) {
            const kk = j * (nu + 1) + i;
            if (!valid[kk]) continue;
            if (j % every === 0 && i < nu && valid[kk + 1]) {
              at(kk);
              at(kk + 1);
            }
            if (i % every === 0 && j < nv && valid[kk + nu + 1]) {
              at(kk);
              at(kk + nu + 1);
            }
          }
        if (pairs.length) {
          const g = new BufferGeometry();
          g.setAttribute('position', new Float32BufferAttribute(pairs, 3));
          const m =
            def.style === 'solid'
              ? mat(it, 'grid', () => lineMat(it, 0.16, false, '#ffffff', 0.5))
              : mat(it, 'line', () => lineMat(it, 0.85, def.style === 'dashed'));
          add(it, g, m, 'lines');
        }
        setLabel(it, centre, 0.5, 1, 'p3-label');
        break;
      }
    }
    it.built = new Float64Array(it.fns.deps.map((i) => P[i]));
    it.vis = -1; // materials and growth are set again for the new geometry
  };

  const stale = (it: Item) => !it.built || it.fns.deps.some((pi, j) => Math.abs(P[pi] - it.built![j]) > 1e-9 * (1 + Math.abs(P[pi])));

  // ---------- axes ----------
  /** The axes' labels (name and numbers), with the axis' world direction: hidden while it points at the viewer. Its
   *  ends (world), and its numbers, each anchored on its tick: placeTicks puts them beside the axis as seen (the
   *  origin's number, kept on x, between the sides of x and y). */
  const axisLabels: {
    axis: number;
    dir: Vector3;
    objs: CSS2DObject[];
    from: Vector3;
    to: Vector3;
    ticks: CSS2DObject[];
    origin: CSS2DObject | null;
  }[] = [];
  /** Every label on the stage besides the objects' own: laid out with them, at the lowest priority. */
  const guideLabels: { obj: CSS2DObject; inner: HTMLElement; rank: number; movable: boolean }[] = [];
  if (ax.show) {
    const o: Vec3 = [clamp(0, ax.x.min, ax.x.max), clamp(0, ax.y.min, ax.y.max), clamp(0, ax.z.min, ax.z.max)];
    const lineM = new LineBasicMaterial({ color: AXIS, transparent: true, opacity: 0.75 });
    const gridM = new LineBasicMaterial({ color: GRID, transparent: true, opacity: 0.13, depthWrite: false });
    const headM = new MeshStandardMaterial({ color: AXIS, roughness: 0.5 });
    guideMats.push(lineM, gridM, headM);
    const axisPairs: Vector3[] = [];
    const tickPairs: Vector3[] = [];
    const gridPairs: Vector3[] = [];
    const axes = [ax.x, ax.y, ax.z];
    const at = (a: number, v: number): Vec3 => o.map((c, i) => (i === a ? v : c)) as Vec3;
    const ticks = axes.map((r) => niceTicks(r.min, r.max, 5));
    axes.forEach((r, a) => {
      if (a === 2 && flat) return;
      const from = w(...at(a, r.min));
      const to = w(...at(a, r.max));
      const dir = to.clone().sub(from);
      if (dir.length() < 1e-6) return;
      dir.normalize();
      const tip = to.clone().addScaledVector(dir, 0.12);
      axisPairs.push(from, tip);
      const g = headGeom(tip, dir, 0.07, 0.022);
      guideGeoms.push(g);
      guides.add(new Mesh(g, headM));
      const objs: CSS2DObject[] = [];
      const name = displayName(r.label) || ['x', 'y', 'z'][a];
      const nameLabel = cssLabel(name, 'p3-axis', tip.clone().addScaledVector(dir, 0.1));
      guides.add(nameLabel.obj);
      objs.push(nameLabel.obj);
      guideLabels.push({ ...nameLabel, rank: 6, movable: true });
      // ticks: a short mark across the axis, and its number (anchored on the tick; placeTicks sets its side)
      const side = a === 2 ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0);
      const tickObjs: CSS2DObject[] = [];
      let origin: CSS2DObject | null = null;
      for (const t of ticks[a].ticks) {
        const atOrigin = Math.abs(t - o[a]) < 1e-9 * (1 + Math.abs(t));
        if (atOrigin && a > 0) continue; // the origin's number once, on x
        const p = w(...at(a, t));
        tickPairs.push(p.clone().addScaledVector(side, -0.025), p.clone().addScaledVector(side, 0.025));
        const tick = cssLabel(formatTick(t, ticks[a].step), 'p3-tick', p.clone(), a === 2 ? 1 : 0.5, a === 2 ? 0.5 : 0);
        guides.add(tick.obj);
        objs.push(tick.obj);
        tickObjs.push(tick.obj);
        if (atOrigin) origin = tick.obj;
        guideLabels.push({ ...tick, rank: 7, movable: false });
      }
      axisLabels.push({ axis: a, dir, objs, from, to, ticks: tickObjs, origin });
    });
    // graph paper on the x-y plane through the origin
    for (const x of ticks[0].ticks) gridPairs.push(w(x, ax.y.min, o[2]), w(x, ax.y.max, o[2]));
    for (const y of ticks[1].ticks) gridPairs.push(w(ax.x.min, y, o[2]), w(ax.x.max, y, o[2]));
    for (const [pairs, m] of [
      [axisPairs, lineM],
      [tickPairs, lineM],
      [gridPairs, gridM],
    ] as const) {
      if (!pairs.length) continue;
      const g = new BufferGeometry().setFromPoints(pairs);
      guideGeoms.push(g);
      guides.add(new LineSegments(g, m));
    }
  }

  // ---------- camera ----------
  // The camera frames what is drawn at some moment (every object's reach, and the axes), not the corners of the
  // whole box, which are mostly empty: the drawing then fills the stage from every preset.
  const corners: Vector3[] = [];
  const boxCorners = (a: Vec3, b: Vec3) => {
    for (const x of [a[0], b[0]]) for (const y of [a[1], b[1]]) for (const z of [a[2], b[2]]) corners.push(w(x, y, z));
  };
  for (const q of framePts) corners.push(w(q[0], q[1], q[2]));
  const nObjCorners = corners.length;
  if (ax.show) {
    const o: Vec3 = [clamp(0, ax.x.min, ax.x.max), clamp(0, ax.y.min, ax.y.max), clamp(0, ax.z.min, ax.z.max)];
    boxCorners([ax.x.min, ax.y.min, o[2]], [ax.x.max, ax.y.max, o[2]]); // the graph paper
    const tips = [w(ax.x.max, o[1], o[2]), w(o[0], ax.y.max, o[2]), w(o[0], o[1], ax.z.max)];
    const from = [w(ax.x.min, o[1], o[2]), w(o[0], ax.y.min, o[2]), w(o[0], o[1], ax.z.min)];
    tips.forEach((t, i) => {
      if (i === 2 && flat) return;
      const d = t.clone().sub(from[i]);
      corners.push(from[i], d.lengthSq() > 1e-12 ? t.clone().addScaledVector(d.normalize(), 0.25) : t);
    });
  }
  if (corners.length < 2) boxCorners(lo, hi);
  const cmin = new Vector3(Infinity, Infinity, Infinity);
  const cmax = new Vector3(-Infinity, -Infinity, -Infinity);
  for (const c of corners) {
    cmin.min(c);
    cmax.max(c);
  }
  const floorY = cmin.y; // (cmin becomes the target below)
  const target = cmin.add(cmax).multiplyScalar(0.5);
  /** How far the framing points reach from the centre (world units): tubes and rods are thicker when it is large. */
  const reachR = Math.max(1e-6, ...corners.map((c) => c.distanceTo(target)));
  const thick = clamp(reachR / THICK_AT, 1, THICK_MAX);
  /** The axes' framing points (the graph paper, the arrows' tips): framed at every step. */
  const axisCorners = corners.slice(nObjCorners);

  // Each step is framed on what is on screen then (the objects visible after it, where its change starts and where it
  // ends), not on every part of the whole animation at once: the cylinder sat in a fifth of the stage until its
  // ghosts came, test2 #7's three arrows in a corner for four minutes.
  const stepFrames = new Map<number, Vector3[]>();
  stepFraming(scene).forEach(({ visible, from, to }, i) => {
    const on = new Set(visible);
    const at = (vals: Record<string, number>) =>
      withDefs(new Float64Array([...scene.params.map((p) => vals[p.name] ?? p.value), ...defs.map(() => NaN)]));
    const states = from === to ? [at(to)] : [at(from), at(to)];
    const out: Vector3[] = [];
    for (const it of items) if (on.has(it.def.id)) for (const q of states) for (const v of probe(it, q)) out.push(w(v[0], v[1], v[2]));
    stepFrames.set(i - 1, out);
  });
  /** The points the camera frames now, and their centre (it looks at it). */
  let fc: Vector3[] = corners;
  const ft = target.clone();
  /** Frames step `index` (-1: before the first), with the `extra` points (the student's own values: never smaller
   *  than the lecture's framing of that step). Never closer than FRAME_MIN of the whole animation's reach. */
  const setFraming = (index: number, extra: Vector3[] | null) => {
    const own = stepFrames.get(index) ?? [];
    const pts = own.length || extra?.length ? [...own, ...(extra ?? []), ...axisCorners] : [...corners];
    const a = new Vector3(Infinity, Infinity, Infinity);
    const b = new Vector3(-Infinity, -Infinity, -Infinity);
    for (const q of pts) {
      a.min(q);
      b.max(q);
    }
    const c = a.add(b).multiplyScalar(0.5);
    let r = 0;
    for (const q of pts) r = Math.max(r, q.distanceTo(c));
    const least = FRAME_MIN * reachR;
    if (r < least)
      for (const d of [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)])
        pts.push(c.clone().addScaledVector(d, least), c.clone().addScaledVector(d, -least));
    fc = pts;
    ft.copy(c);
  };
  /** Where the visible objects are now (the student's values, while they explore). */
  const visiblePoints = (st: SceneState) => {
    const out: Vector3[] = [];
    for (const it of items)
      if ((st.vis[it.def.id] ?? 1) > 0.002) for (const q of probe(it, P)) out.push(w(q[0], q[1], q[2]));
    return out;
  };

  // A soft contact shadow with a faint grid under a drawing that stands in space (no axes, not flat): it sits on
  // something, and turning it reads as turning.
  if (!ax.show && cmax.y - floorY > 0.05 * L && typeof document !== 'undefined') {
    let foot = 0;
    for (const c of corners) foot = Math.max(foot, Math.hypot(c.x - target.x, c.z - target.z));
    const size = Math.max(0.6, foot * 2.5);
    const tex = groundTexture();
    const g = new PlaneGeometry(size, size);
    g.rotateX(-Math.PI / 2);
    const m = new MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, fog: true });
    const ground = new Mesh(g, m);
    ground.position.set(target.x, floorY - 0.025 * L, target.z);
    ground.renderOrder = -1;
    guides.add(ground);
    guideGeoms.push(g);
    guideMats.push(m);
    guideTextures.push(tex);
  }

  let width = 0;
  let height = 0;
  /** What the drawing is framed clear of: what covers the stage's edges (opts.insets), and the corner box on the
   *  side chosen for it (clearCorner). */
  const inset: Insets = { top: 0, right: 0, bottom: 0, left: 0 };
  const edges: Insets = { top: 0, right: 0, bottom: 0, left: 0 };
  const corner = { width: 0, height: 0 };
  let side: CornerSide = 'top';
  const fullView = () => fullViewOf(width, height, inset);
  /** Distance at which every framing point fits the part of the stage nothing covers (the caption, the values, the
   *  sliders, Reset view). */
  const fitDistance = (dir: Vector3) => fitDistanceOf(fc, ft, dir, camera.fov, width, height, inset);
  /** The viewer's zoom (Ctrl + wheel, a pinch): a share of the fitted distance, kept when the steps turn the camera. */
  let zoom = 1;
  const dirFor = (preset: Camera) => frame.dir(preset === 'iso' && scene.isoFrom === 'left' ? ISO_LEFT : CAMERA_DIRS[preset]);
  const viewFor = (preset: Camera) => {
    const dir = dirFor(preset);
    return { pos: ft.clone().addScaledVector(dir, fitDistance(dir) * zoom), target: ft.clone() };
  };
  /** Frames the drawing past the corner box on the side that shows it larger from `dir` (cornerSide). Returns whether
   *  the insets changed (applyView then). */
  const clearCorner = (dir: Vector3) => {
    side = cornerSide(fc, ft, dir, camera.fov, width, height, edges, corner, side);
    const next = cornerInsets(edges, corner, side);
    const changed = (Object.keys(next) as (keyof Insets)[]).some((k) => next[k] !== inset[k]);
    Object.assign(inset, next);
    return changed;
  };
  /** The camera renders the window of the full view that is the stage (fullViewOf). */
  const applyView = () => {
    const fv = fullView();
    camera.aspect = fv.w / fv.h;
    camera.setViewOffset(fv.w, fv.h, fv.x, fv.y, width, height);
    camera.updateProjectionMatrix();
  };

  const controls = new OrbitControls(camera, renderer.domElement);
  // OrbitControls listens for the Ctrl key on the canvas's root node (the document), and on dispose removes those
  // listeners from whatever root the canvas has then. React takes the view out of the page before its effects clean
  // up, so by then the root is the detached view itself: the document kept the listeners, and with them the whole
  // stage (one WebGL canvas and ~0.8 MB leaked per opening). dispose() removes them from this root.
  const controlsRoot = renderer.domElement.getRootNode();
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.7;
  controls.zoomSpeed = 0.8;
  controls.target.copy(target);
  controls.minDistance = 0.3;
  controls.maxDistance = 40;
  let dirty = true;
  let turned = false;
  let tween: { from: { pos: Vector3; target: Vector3 }; to: { pos: Vector3; target: Vector3 }; t0: number } | null = null;
  /** True while this stage moves the camera itself (a preset, its glide, the controls' damping), not the viewer. */
  let ours = false;
  const update = (dt?: number) => {
    ours = true;
    try {
      return controls.update(dt);
    } finally {
      ours = false;
    }
  };
  // The viewer turned the view when the camera moves inside the controls' own handlers (a drag, a pinch that pans).
  // A click that does not move it is not a turn: the steps keep moving the camera. Nor is a zoom (the direction and
  // the target stay): the steps keep turning the camera, at the distance the viewer chose.
  const lastDir = new Vector3();
  const lastTarget = new Vector3();
  let known = false;
  const remember = (dir: Vector3) => {
    lastDir.copy(dir);
    lastTarget.copy(controls.target);
    known = true;
  };
  controls.addEventListener('change', () => {
    dirty = true;
    const off = camera.position.clone().sub(controls.target);
    const dist = off.length();
    const dir = off.divideScalar(dist || 1);
    if (ours) return remember(dir);
    const zoomed = known && dir.distanceTo(lastDir) < 1e-4 && controls.target.distanceTo(lastTarget) < 1e-6;
    remember(dir);
    if (zoomed) {
      zoom = clamp(dist / Math.max(1e-6, fitDistance(dir)), ZOOM_MIN, ZOOM_MAX);
      if (tween) tween.to = viewFor(preset);
      return;
    }
    tween = null;
    if (!turned) {
      turned = true;
      opts.onTurn?.(true);
    }
  });
  // The wheel scrolls the page, as it does everywhere else on it (the transcript is below the view); Ctrl or Cmd +
  // wheel zooms, and so does a trackpad pinch (the browser sends it as Ctrl + wheel). A listener on the canvas in the
  // capture phase runs before the controls' own and keeps the plain wheel from them.
  const onWheel = (e: WheelEvent) => {
    if (!e.ctrlKey && !e.metaKey) e.stopImmediatePropagation();
  };
  renderer.domElement.addEventListener('wheel', onWheel, { capture: true, passive: true });

  let preset: Camera = 'iso';
  let presetStep = -2;
  /** The step the camera is framed on, the student's values it was framed with ('' for none), and when to frame it
   *  again on them (0: not waiting). */
  let frameStep = -2;
  let lastOwn = '';
  let refitAt = 0;
  const moveTo = (p: Camera, animate: boolean) => {
    if (clearCorner(dirFor(p))) applyView(); // this step's drawing may pass Reset view better on the other side
    const to = viewFor(p);
    if (!animate) {
      tween = null;
      camera.position.copy(to.pos);
      controls.target.copy(to.target);
      update();
    } else tween = { from: { pos: camera.position.clone(), target: controls.target.clone() }, to, t0: performance.now() };
    dirty = true;
  };
  const sa = new Spherical();
  const sb = new Spherical();
  const stepTween = (now: number) => {
    if (!tween) return;
    const p = clamp((now - tween.t0) / TWEEN_MS, 0, 1);
    const e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
    sa.setFromVector3(tween.from.pos.clone().sub(tween.from.target));
    sb.setFromVector3(tween.to.pos.clone().sub(tween.to.target));
    let dTheta = sb.theta - sa.theta;
    if (dTheta > Math.PI) dTheta -= 2 * Math.PI;
    if (dTheta < -Math.PI) dTheta += 2 * Math.PI;
    const s = new Spherical(
      sa.radius * Math.pow(sb.radius / sa.radius, e),
      sa.phi + (sb.phi - sa.phi) * e,
      sa.theta + dTheta * e,
    );
    const t = tween.from.target.clone().lerp(tween.to.target, e);
    controls.target.copy(t);
    camera.position.copy(t).add(new Vector3().setFromSpherical(s));
    if (p >= 1) tween = null;
    dirty = true;
  };

  let placed = false;
  const resize = () => {
    const wpx = host.clientWidth;
    const hpx = host.clientHeight;
    if (!wpx || !hpx) return;
    const want = opts.insets?.() ?? { top: 0, right: 0, bottom: 0, left: 0 };
    // what covers the stage never takes more than a quarter of its height, or 40% of its width
    const ins: Insets = {
      top: clamp(Math.round(want.top), 0, Math.round(hpx * 0.25)),
      bottom: clamp(Math.round(want.bottom), 0, Math.round(hpx * 0.25)),
      left: clamp(Math.round(want.left), 0, Math.round(wpx * 0.4)),
      right: clamp(Math.round(want.right), 0, Math.round(wpx * 0.4)),
    };
    const box = opts.corner?.();
    const cw = box ? clamp(Math.round(box.width), 0, Math.round(wpx * 0.4)) : 0;
    const ch = box ? clamp(Math.round(box.height), 0, Math.round(hpx * 0.25)) : 0;
    const same =
      (Object.keys(ins) as (keyof Insets)[]).every((k) => ins[k] === edges[k]) && cw === corner.width && ch === corner.height;
    if (wpx === width && hpx === height && same) return;
    const resized = wpx !== width || hpx !== height;
    width = wpx;
    height = hpx;
    Object.assign(edges, ins);
    corner.width = cw;
    corner.height = ch;
    Object.assign(inset, cornerInsets(edges, corner, side)); // moveTo picks the side again, unless the viewer turned it
    if (resized) {
      renderer.setSize(wpx, hpx);
      labels.setSize(wpx, hpx);
      linePx = clamp(hpx / LINE_PX_PER, 1.5, 3);
      sizes.clear(); // labels are measured again (fonts may have arrived)
      for (const it of items)
        for (const m of it.mats)
          if (m.m instanceof LineMaterial) {
            m.m.resolution.set(wpx, hpx);
            m.m.linewidth = linePx * m.width * it.emph * (1 + 1.2 * Math.max(0, it.glow));
          }
    }
    applyView();
    // Refit the view to the new shape (not one the viewer chose). The caption grows and shrinks from step to step,
    // the sliders open and close: the drawing then glides to stay clear of them.
    if (!placed || !turned) moveTo(preset, placed && !resized && !opts.reducedMotion);
    placed = true;
    dirty = true;
  };
  const ro = new ResizeObserver(() => resize());
  ro.observe(host);
  for (const el of opts.insetElements?.() ?? []) if (el) ro.observe(el);

  // ---------- applying a state ----------
  const white = new Color('#ffffff');
  let lastKey = '';
  let pulsing = false;
  const apply = (st: SceneState, now: number) => {
    const key = `${st.index}|${Object.values(st.params).join(',')}|${Object.values(st.vis).join(',')}|${st.glow}|${st.highlight.join(',')}`;
    const changed = key !== lastKey;
    lastKey = key;
    if (changed) {
      scene.params.forEach((p, i) => (P[i] = st.params[p.name] ?? p.value));
      withDefs(P);
    }
    const lit = new Set(st.highlight);
    const stepNow = st.index >= 0 ? scene.steps[st.index] : null;
    const emphIds = new Set(stepNow ? [...stepNow.show, ...stepNow.highlight] : []);
    pulsing = !opts.reducedMotion && st.glow > 0 && st.highlight.length > 0;
    if (!changed && !pulsing) return;
    const pulse = pulsing ? 0.55 + 0.45 * Math.sin((now / 1000) * 2 * Math.PI * PULSE_HZ) : 1;
    for (const it of items) {
      const v = clamp(st.vis[it.def.id] ?? 1, 0, 1);
      const visible = v > 0.002;
      it.root.visible = visible;
      if (!visible) {
        it.vis = 0;
        continue;
      }
      const e = emphIds.has(it.def.id) && LINES.has(it.def.type) ? EMPH : 1;
      if (e !== it.emph) {
        it.emph = e;
        build(it);
        it.vis = it.glow = -1; // its new parts take the fade and glow again
      } else if (stale(it) && !move(it) && !moveRod(it)) build(it);
      const g = lit.has(it.def.id) ? st.glow * pulse : 0;
      if (v === it.vis && g === it.glow) continue;
      it.vis = v;
      it.glow = g;
      for (const m of it.mats) {
        m.m.opacity = Math.min(1, m.base * v + (g > 0 ? 0.25 * g * v : 0));
        if (m.m instanceof MeshStandardMaterial) m.m.emissiveIntensity = m.emissive + 1.1 * g;
        else {
          m.m.color.copy(m.color).lerp(white, 0.55 * g);
          m.m.linewidth = linePx * m.width * it.emph * (1 + 1.2 * g);
        }
      }
      if (it.grow === 'scale') it.root.scale.setScalar(0.6 + 0.4 * v);
      else if (it.grow === 'length') it.root.scale.setScalar(Math.max(0.001, v));
      else if (it.grow === 'draw') {
        const total = it.draws.reduce((s, d) => s + d.count, 0);
        let left = total * v;
        for (const d of it.draws) {
          const n = clamp(left, 0, d.count);
          if (d.geom instanceof LineSegmentsGeometry) d.geom.instanceCount = v >= 1 ? d.count / 2 : Math.floor(n / 2);
          else {
            const indexed = !!d.geom.index;
            d.geom.setDrawRange(0, v >= 1 ? Infinity : indexed ? Math.floor(n / 3) * 3 : Math.floor(n / 2) * 2);
          }
          left -= d.count;
        }
      }
      if (it.label) it.label.element.style.opacity = String(v);
      it.labelInner?.classList.toggle('hl', g > 0.05);
    }
    dirty = true;
  };

  // ---------- labels: off each other ----------
  /** An axis' name and numbers are hidden while it points at the viewer (they would pile up where it starts). */
  const axesFacing = () => {
    if (!axisLabels.length) return;
    const view = controls.target.clone().sub(camera.position).normalize();
    for (const a of axisLabels) {
      const on = a.dir.clone().cross(view).length() >= AXIS_EDGE_ON;
      for (const o of a.objs) o.visible = on;
    }
  };
  const lastPlace = new Map<HTMLElement, LabelPlace>();
  /**
   * After the labels are placed by the renderer: the ones that would sit on others are nudged (through their inner
   * pill: the renderer owns the outer element's transform) or dimmed. Most important first: what the current step
   * highlights, then what it shows, then points and text, other objects, curves, segments, axis names, numbers.
   */
  // Where each label is, worked out as the label renderer places it (its anchor projected, less its centre times its
  // size) rather than read back from the page: reading it forced the whole page's style and layout in the middle of
  // every frame (up to 245 ms when a step chip jumped across the lecture). A label's size is measured once for each
  // length its text takes (tabular figures: one length, one width): an angle's value moving from 80° to 75° goes
  // "β 80°", "β 79.9°", "β 79°"..., and measuring it again at each change of length forced 34-38 page layouts during
  // one 2 s change of test2 #4 (3 now).
  // The renderer places the outer element (its centre times its size off the anchor); what is seen, and laid out, is
  // the pill inside it, which may sit off it: a point's name has a margin (-5 px up, 7 px right), so its pill stood
  // 5 px above the box laid out for it and overlapped the label above by 3 (test-full #3's "x=0" and "-1").
  const sizes = new Map<HTMLElement, Map<number, { w: number; h: number; px: number; py: number; pw: number; ph: number }>>();
  const viewProj = new Matrix4();
  const anchor = new Vector3();
  const sizeOf = (outer: HTMLElement, inner: HTMLElement) => {
    const len = inner.textContent?.length ?? 0;
    let byLen = sizes.get(outer);
    if (!byLen) sizes.set(outer, (byLen = new Map()));
    let sz = byLen.get(len);
    if (!sz) {
      const mine = inner.offsetParent === outer;
      sz = {
        w: outer.offsetWidth,
        h: outer.offsetHeight,
        px: mine ? inner.offsetLeft : 0,
        py: mine ? inner.offsetTop : 0,
        pw: inner.offsetWidth || outer.offsetWidth,
        ph: inner.offsetHeight || outer.offsetHeight,
      };
      if (!sz.w || !sz.h || !sz.pw || !sz.ph) return null;
      byLen.set(len, sz);
    }
    return sz;
  };
  const boxOf = (obj: CSS2DObject, inner: HTMLElement) => {
    const outer = obj.element;
    if (outer.style.display === 'none') return null; // behind the camera, or hidden by the renderer
    const sz = sizeOf(outer, inner);
    if (!sz) return null;
    anchor.setFromMatrixPosition(obj.matrixWorld).applyMatrix4(viewProj);
    const x = (anchor.x * 0.5 + 0.5) * width - obj.center.x * sz.w + sz.px;
    const y = (-anchor.y * 0.5 + 0.5) * height - obj.center.y * sz.h + sz.py;
    return { x, y, w: sz.pw, h: sz.ph };
  };
  /** An axis' numbers beside it as it is seen now (tickCenter): below one across the screen, left of one up it, never
   *  on it (seen from the top, the y axis' numbers sat under a dashed asymptote drawn along it). */
  const ta = new Vector3();
  const tb = new Vector3();
  const scr = new Vector3();
  const onScreen = (v: Vector3): [number, number] => {
    scr.copy(v).project(camera);
    return [(scr.x * 0.5 + 0.5) * width, (-scr.y * 0.5 + 0.5) * height];
  };
  const placeTicks = () => {
    const sides = new Map<number, [number, number]>();
    for (const a of axisLabels) {
      if (!a.objs[0]?.visible) continue;
      ta.copy(a.from).project(camera);
      tb.copy(a.to).project(camera);
      const n = tickSide((tb.x - ta.x) * width, -(tb.y - ta.y) * height);
      if (n) sides.set(a.axis, n);
    }
    for (const a of axisLabels) {
      const n = sides.get(a.axis);
      if (!n) continue;
      for (const t of a.ticks) {
        const sz = t.element.isConnected ? sizeOf(t.element, t.element.firstElementChild as HTMLElement) : null;
        if (!sz) {
          dirty = true; // not measured yet: again next frame
          continue;
        }
        if (t === a.origin) {
          // clear of every axis drawn, as seen: x and y on graph paper, and z as well in space (originCenter)
          const [ox, oy] = onScreen(t.position);
          const rays: [number, number][] = [];
          for (const b of axisLabels)
            for (const end of [b.from, b.to]) {
              const [ex, ey] = onScreen(end);
              if (Math.hypot(ex - ox, ey - oy) > 2) rays.push([ex - ox, ey - oy]);
            }
          t.center.set(...originCenter(rays, sz.w, sz.h));
        } else t.center.set(...besideCenter(n[0], n[1], sz.w, sz.h));
      }
    }
  };
  /** Each label's nudge as shown on screen now, and its speed: it glides to its place (glide), never jumps there. */
  let shownAt = new Map<HTMLElement, Glide>();
  const AT_HOME: Glide = { dx: 0, dy: 0, vx: 0, vy: 0 };
  /** Some label is still on its way to its place: keep drawing frames. */
  let gliding = false;
  const layoutLabels = (st: SceneState, settled: boolean, dt: number) => {
    gliding = false;
    const lit = new Set(st.glow > 0 ? st.highlight : []);
    const shown = new Set(st.index >= 0 ? (scene.steps[st.index]?.show ?? []) : []);
    const entries: { obj: CSS2DObject; inner: HTMLElement; rank: number; movable: boolean; faint: boolean }[] = [];
    for (const it of items) {
      if (!it.label || !it.labelInner || !it.root.visible || !it.label.visible || it.vis < 0.2) continue;
      const id = it.def.id;
      const t = it.def.type;
      const rank = lit.has(id) ? 0 : shown.has(id) ? 1 : t === 'point' || t === 'label' || t === 'angle' ? 2 : t === 'curve' ? 4 : t === 'segment' ? 5 : 3;
      entries.push({ obj: it.label, inner: it.labelInner, rank, movable: true, faint: it.vis < 0.5 });
    }
    for (const g of guideLabels)
      if (g.obj.visible) entries.push({ obj: g.obj, inner: g.inner, rank: g.rank, movable: g.movable, faint: false });
    if (!entries.length) return;
    entries.sort((a, b) => a.rank - b.rank);
    viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    // Each angle's arc or mark on screen: a label nudged off another never lands on one (test2 #7: beta's name was
    // moved onto the corner, over gamma's arc and its own mark).
    const avoid: Rect[] = [];
    for (const it of items) {
      if (!it.marks || !it.root.visible || it.vis < 0.2) continue;
      let x0 = Infinity;
      let y0 = Infinity;
      let x1 = -Infinity;
      let y1 = -Infinity;
      for (const m of it.marks) {
        anchor.copy(m).applyMatrix4(viewProj);
        if (!(anchor.z <= 1)) continue; // behind the camera
        const sx = (anchor.x * 0.5 + 0.5) * width;
        const sy = (-anchor.y * 0.5 + 0.5) * height;
        x0 = Math.min(x0, sx);
        y0 = Math.min(y0, sy);
        x1 = Math.max(x1, sx);
        y1 = Math.max(y1, sy);
      }
      if (x1 >= x0 && y1 >= y0) avoid.push({ x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
    }
    const keep: typeof entries = [];
    const boxes = [];
    for (const e of entries) {
      const b = boxOf(e.obj, e.inner);
      if (!b) continue;
      boxes.push({ ...b, movable: e.movable });
      keep.push(e);
    }
    const places = placeLabels(boxes, {
      width,
      height,
      // While something moves, a label keeps last frame's place when it is still free (no hopping); once all is still,
      // the places are worked out afresh, so a moment looks the same however the view got there (a seek back to
      // test2 #4's step 2 put "α 90°" 22 px from where playing there had).
      prev: settled ? undefined : keep.map((e) => lastPlace.get(e.inner)),
      avoid,
      // Reset view: the drawing is framed past it (clearCorner), and a label sticking out of the framing is moved off it
      block: corner.width && corner.height ? [{ x: width - corner.width, y: 0, w: corner.width, h: corner.height }] : undefined,
    });
    // A label slides to its place rather than jumping there: while things move it keeps its spot when it can, and
    // once all is still it is placed afresh, which hopped labels up to 32 px on a still drawing (test2 #4). A label
    // that was not laid out last frame starts at its place; one still fading in (shown faint, in its own place
    // meanwhile) glides out from there.
    const next = new Map<HTMLElement, Glide>();
    places.forEach((p, i) => {
      const inner = keep[i].inner;
      const from = shownAt.get(inner) ?? (keep[i].faint && !inner.style.transform ? AT_HOME : undefined);
      const g = glide(opts.reducedMotion ? undefined : from, p, dt);
      if (!g.done) gliding = true;
      next.set(inner, g);
      const tf = Math.round(g.dx) || Math.round(g.dy) ? `translate(${Math.round(g.dx)}px, ${Math.round(g.dy)}px)` : '';
      if (inner.style.transform !== tf) inner.style.transform = tf;
      inner.classList.toggle('dim', p.dim);
      lastPlace.set(inner, p);
    });
    // a label no longer laid out (fading, hidden, behind the camera) goes back to its own place, plain
    for (const inner of shownAt.keys()) {
      if (next.has(inner)) continue;
      inner.style.transform = '';
      inner.classList.remove('dim');
    }
    shownAt = next;
  };

  /** An angle's label stands out from its vertex, along the bisector as seen now: its near edge at its anchor, so it
   *  never covers the arc it names (three angles at one corner of a cell kept covering theirs). */
  const sv = new Vector3();
  const sl = new Vector3();
  const turnAngleLabels = () => {
    const view = controls.target.clone().sub(camera.position).normalize();
    for (const it of items) {
      if (!it.bisector || !it.label || !it.root.visible) continue;
      // seen edge-on, its mark is a line: "α 90°" floated with nothing to show where (test2 #7 from the top)
      it.labelInner?.classList.toggle('edgeon', !!it.normal && Math.abs(it.normal.dot(view)) < ANGLE_EDGE_ON);
      sv.copy(it.bisector[0]).project(camera);
      sl.copy(it.bisector[1]).project(camera);
      const dx = (sl.x - sv.x) * width;
      const dy = (sl.y - sv.y) * height;
      const len = Math.hypot(dx, dy);
      if (len < 1e-6) it.label.center.set(0.5, 0.5);
      else it.label.center.set(0.5 - (0.5 * dx) / len, 0.5 + (0.5 * dy) / len);
    }
  };

  // ---------- the loop ----------
  let raf = 0;
  let last = performance.now();
  const loop = (now: number) => {
    raf = requestAnimationFrame(loop);
    const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
    last = now;
    if (!placed) resize();
    if (!placed) return;
    const st = opts.frame(now);
    apply(st, now);
    // "Try it": once the sliders rest for REFIT_MS, the view is framed on the student's values as well (a = 6 ran
    // off the stage); back to the lecture's framing when the values are the lecture's again.
    const own = st.overridden.length ? st.overridden.map((n) => st.params[n]).join(',') : '';
    if (own !== lastOwn) {
      lastOwn = own;
      refitAt = now + REFIT_MS;
    }
    const refit = refitAt > 0 && now >= refitAt;
    if (st.cameraStep !== presetStep || st.camera !== preset || st.index !== frameStep || refit) {
      const first = presetStep === -2;
      preset = st.camera;
      presetStep = st.cameraStep;
      frameStep = st.index;
      if (refit) refitAt = 0;
      setFraming(frameStep, own ? visiblePoints(st) : null);
      if (!turned) moveTo(preset, !first && !opts.reducedMotion);
    }
    stepTween(now);
    const turning = update(dt);
    if (turning) dirty = true;
    const settled = !st.moving && !tween && !turning;
    if (dirty || pulsing) {
      dirty = false;
      const dist = camera.position.distanceTo(controls.target);
      fog.near = Math.max(0.01, dist - reachR * 0.4);
      fog.far = dist + reachR * FOG_FAR;
      renderer.render(world, camera);
      axesFacing();
      turnAngleLabels();
      placeTicks();
      labels.render(world, camera);
      layoutLabels(st, settled, dt);
    } else if (gliding) layoutLabels(st, settled, dt); // the drawing is still: only labels on their way move
  };
  // Every object built now (each stays hidden until a step shows it) and every shader compiled before the first
  // frame: a step that shows many parts at once then only uncovers them.
  for (const it of items) build(it);
  renderer.compile(world, camera);
  resize();
  // One frame with everything shown: every program is set up, every buffer uploaded and every vertex array made now.
  // The first real frame, drawn at once in the same task (so the warm-up is never on screen), shows the state.
  renderer.render(world, camera);
  loop(performance.now());

  return {
    resetView() {
      if (turned) {
        turned = false;
        opts.onTurn?.(false);
      }
      zoom = 1;
      moveTo(preset, !opts.reducedMotion);
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener('wheel', onWheel, { capture: true });
      const keys = controls as unknown as { _interceptControlDown?: EventListener; _interceptControlUp?: EventListener };
      if (keys._interceptControlDown) controlsRoot.removeEventListener('keydown', keys._interceptControlDown, { capture: true });
      if (keys._interceptControlUp) controlsRoot.removeEventListener('keyup', keys._interceptControlUp, { capture: true });
      for (const it of items) {
        it.geoms.forEach((g) => g.dispose());
        it.label?.element.remove();
      }
      const mats = [...[...matsOf.values()].flatMap((rec) => Object.values(rec).map((m) => m.m)), ...guideMats];
      // three.js shares one lookup texture (the DFG LUT) between the MeshStandardMaterials of every renderer, and
      // each renderer that uploads it adds a 'dispose' listener to it that holds the renderer, its WebGL context and
      // its canvas: every view opened stayed in memory. Disposing that texture makes each renderer let go of it (a
      // renderer still in use would upload it again).
      const shared = new Set<{ dispose(): void }>();
      for (const m of mats) {
        const u = (renderer.properties.get(m) as { uniforms?: Record<string, { value?: unknown } | undefined> }).uniforms;
        const lut = u?.dfgLUT?.value as { isTexture?: boolean; dispose(): void } | null | undefined;
        if (lut?.isTexture) shared.add(lut);
      }
      mats.forEach((m) => m.dispose());
      guideGeoms.forEach((g) => g.dispose());
      guideTextures.forEach((t) => t.dispose());
      rings?.dispose();
      unitRod.dispose();
      renderer.dispose();
      shared.forEach((t) => t.dispose());
      renderer.forceContextLoss();
      renderer.domElement.remove();
      labels.domElement.remove();
    },
  };
}

/** A label: the element the CSS2DRenderer places (its transform is the renderer's), and the pill inside it that the
 *  layout nudges off other labels. */
function labelElements(text: string, cls: string): { outer: HTMLDivElement; inner: HTMLDivElement } {
  const outer = document.createElement('div');
  outer.className = 'p3-anchor';
  const inner = document.createElement('div');
  inner.className = cls;
  inner.textContent = text;
  outer.appendChild(inner);
  return { outer, inner };
}

function cssLabel(text: string, cls: string, at: Vector3, cx = 0.5, cy = 0.5): { obj: CSS2DObject; inner: HTMLElement } {
  const { outer, inner } = labelElements(text, cls);
  const o = new CSS2DObject(outer);
  o.position.copy(at);
  o.center.set(cx, cy);
  return { obj: o, inner };
}

/** The ground under a drawing that stands in space: a soft shadow in the middle and a faint grid, both fading out. */
function groundTexture(): CanvasTexture {
  const n = 256;
  const make = () => {
    const c = document.createElement('canvas');
    c.width = c.height = n;
    return c;
  };
  const out = make();
  const g = out.getContext('2d')!;
  const shadow = g.createRadialGradient(n / 2, n / 2, 0, n / 2, n / 2, n / 2);
  shadow.addColorStop(0, 'rgba(0, 0, 0, 0.5)');
  shadow.addColorStop(0.4, 'rgba(0, 0, 0, 0.24)');
  shadow.addColorStop(1, 'rgba(0, 0, 0, 0)');
  g.fillStyle = shadow;
  g.fillRect(0, 0, n, n);
  const grid = make();
  const k = grid.getContext('2d')!;
  k.strokeStyle = 'rgba(205, 195, 175, 0.22)';
  k.lineWidth = 1.2;
  k.beginPath();
  for (let i = 1; i < 14; i++) {
    const x = Math.round((i / 14) * n) + 0.5;
    k.moveTo(x, 0);
    k.lineTo(x, n);
    k.moveTo(0, x);
    k.lineTo(n, x);
  }
  k.stroke();
  // the grid fades out towards the edge
  k.globalCompositeOperation = 'destination-in';
  const fade = k.createRadialGradient(n / 2, n / 2, 0, n / 2, n / 2, n / 2);
  fade.addColorStop(0, 'rgba(0, 0, 0, 1)');
  fade.addColorStop(0.55, 'rgba(0, 0, 0, 0.6)');
  fade.addColorStop(1, 'rgba(0, 0, 0, 0)');
  k.fillStyle = fade;
  k.fillRect(0, 0, n, n);
  g.drawImage(grid, 0, 0);
  const tex = new CanvasTexture(out);
  tex.colorSpace = SRGBColorSpace;
  return tex;
}
