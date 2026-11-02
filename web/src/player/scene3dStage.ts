// The three.js stage of the 3D view: draws a format-2 scene (types.ts Scene3D) and shows, every frame, the state the
// view asks for (scene3dState.ts sceneState). Loaded only with the 3D view's chunk, so three.js never weighs on the
// player's main bundle.
//
// Maths coordinates have z up; three.js has y up: world = (x, z, -y), scaled so the scene fits a box about L wide.
// Everything is built straight in world coordinates, so a stretched axis (a graph much taller than wide) stretches
// shapes honestly while tubes, arrow heads and dots keep the same thickness on screen.
//
// An object's geometry is rebuilt only when a parameter it reads changes (and only while it is visible); fades,
// growth and the highlight's glow only touch materials and transforms.
import {
  AmbientLight, BoxGeometry, BufferGeometry, Color, ConeGeometry, Curve, CylinderGeometry, DirectionalLight,
  DoubleSide, EdgesGeometry, Euler, Float32BufferAttribute, Group, HemisphereLight, LineBasicMaterial,
  LineDashedMaterial, LineSegments, Matrix4, Mesh, MeshStandardMaterial, PerspectiveCamera, Quaternion,
  Scene, ShapeUtils, SphereGeometry, Spherical, TorusGeometry, TubeGeometry, Vector2, Vector3, WebGLRenderer,
  type Material,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { Scene3D, Scene3DObject, Vec3 } from '../types';
import { tryCompileVars, type VarsFn } from './expr.ts';
import { arcLabelAt, placeLabels, type LabelPlace } from './labelLayout.ts';
import { distinctCurveColors, fitRange, formatTick, niceTicks, sampleSegments, stageColors } from './plotMath.ts';
import type { Camera, SceneState } from './scene3dState.ts';

export interface StageOptions {
  /** The theme's accent: the first colour for objects that bring none. */
  accent: string;
  /** prefers-reduced-motion: the camera jumps and nothing pulses. */
  reducedMotion: boolean;
  /** Called on every animation frame: what to show now. */
  frame(now: number): SceneState;
  /** The viewer turned or zoomed the view (true; steps stop moving the camera), or reset it (false). */
  onTurn?(turned: boolean): void;
  /** Pixels at the bottom of the stage covered by the caption bar: the drawing is framed above them. */
  insetBottom?(): number;
  /** The caption bar itself: when its height changes (a longer caption), the drawing is framed again. */
  insetElement?(): Element | null;
}

export interface Stage3D {
  /** Back to the current step's view, and let the steps move the camera again. */
  resetView(): void;
  dispose(): void;
}

// ---------------------------------------------------------------- sizes (world units; the scene is about L wide)
const L = 2.4;
const BG = '#101319';
const AXIS = '#aab4c6';
const GRID = '#8d98ad';
const TUBE = 0.016;
const ROD = 0.015;
const HEAD_LEN = 0.11;
const HEAD_R = 0.042;
const DOT = 0.042;
export const DASH = 0.075;
export const GAP = 0.05;
const DEG = Math.PI / 180;
const TWEEN_MS = 900;

/** Where each camera preset looks from, in maths coordinates (z up). Front and side are a little off their axis, so
 *  an angle in a plane that meets the view edge-on (alpha and gamma on test2 #4, seen from the front) is still seen
 *  as an angle; top stays exact, for graphs. */
const CAMERA_DIRS: Record<Camera, Vec3> = {
  front: [0.2, -1, 0.25],
  top: [0, -0.02, 1],
  side: [1, 0.2, 0.25],
  iso: [0.7, -1.2, 0.62],
};
/** An axis whose screen length is under this share of its true length points at the viewer: its numbers would pile
 *  up on the origin, so they are hidden (sin of the angle between the axis and the line of sight). */
const AXIS_EDGE_ON = 0.25;
/** Zoom the viewer chose, as a share of the fitted distance. */
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 4;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

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
  m: MeshStandardMaterial | LineBasicMaterial | LineDashedMaterial;
  base: number; // opacity when fully shown
  color: Color;
  emissive: number; // base emissive intensity (mesh materials)
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
  /** The label's pill inside the element the CSS2DRenderer places: nudged off other labels, dimmed, lit. */
  labelInner: HTMLElement | null;
  built: Float64Array | null; // the values of deps when built
  /** A solid placed by its matrix: its shape (type and size) when built; '' for anything else. */
  shape: string;
  grow: Grow;
  vis: number;
  glow: number;
}

// ---------------------------------------------------------------- the stage

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
      built: null,
      shape: '',
      grow:
        def.type === 'segment' || def.type === 'vector'
          ? 'length'
          : def.type === 'graph' || def.type === 'curve'
            ? 'draw'
            : def.type === 'label'
              ? 'none'
              : 'scale',
      vis: -1,
      glow: -1,
    });
  });

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
        for (const dx of [-r, r]) for (const dy of [-r, r]) for (const dz of [-r, r]) push([c[0] + dx, c[1] + dy, c[2] + dz]);
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
  const maxSpan = Math.max(...spans, 1e-9);
  // One scale for every axis (a cube stays a cube), unless the axes are shown and one is far shorter than the
  // longest (a graph of e^x, a tall surface): that one is stretched to at least 35% of the longest.
  const k = spans.map((s) => (ax.show && s > 1e-9 && s < 0.35 * maxSpan ? (L * 0.35) / s : L / maxSpan)) as Vec3;
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
  world.add(new AmbientLight('#ffffff', 0.55));
  world.add(new HemisphereLight('#dfe8ff', '#1c2028', 0.9));
  const sun = new DirectionalLight('#ffffff', 1.5);
  sun.position.set(3, 5, 4);
  world.add(sun);
  const fill = new DirectionalLight('#9fb8ff', 0.5);
  fill.position.set(-4, 1, -3);
  world.add(fill);

  const guides = new Group();
  world.add(guides);
  const content = new Group();
  world.add(content);
  for (const it of items) content.add(it.root);
  const guideGeoms: BufferGeometry[] = [];
  const guideMats: Material[] = [];

  // ---------- materials ----------
  const meshMat = (it: Item, opacity: number, emissive = 0.14, side = DoubleSide): Mat => {
    const m = new MeshStandardMaterial({
      color: it.color, roughness: 0.42, metalness: 0.05, emissive: it.color, emissiveIntensity: emissive,
      transparent: true, opacity, depthWrite: opacity >= 0.95, side,
    });
    const mat = { m, base: opacity, color: new Color(it.color), emissive };
    it.mats.push(mat);
    return mat;
  };
  const lineMat = (it: Item, opacity = 1, dashed = false, color = it.color): Mat => {
    const m = dashed
      ? new LineDashedMaterial({ color, dashSize: DASH, gapSize: GAP, transparent: true, opacity })
      : new LineBasicMaterial({ color, transparent: true, opacity });
    const mat = { m, base: opacity, color: new Color(color), emissive: 0 };
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
    it.geoms.push(g);
    const o = kind === 'mesh' ? new Mesh(g, m) : new LineSegments(g, m);
    if (kind === 'lines' && m instanceof LineDashedMaterial) (o as LineSegments).computeLineDistances();
    if (kind === 'mesh' && (m as MeshStandardMaterial).opacity < 0.95) o.renderOrder = 1;
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
  const setLabel = (it: Item, at: Vector3, cx: number, cy: number, cls: string) => {
    const text = it.def.label;
    if (!text) return;
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
        const g = new SphereGeometry(DOT, 20, 14);
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
        if (def.type === 'vector' && len > 1e-6) {
          const dir = d.clone().divideScalar(len);
          const hl = Math.min(HEAD_LEN, len * 0.45);
          stroke(it, [a, b.clone().addScaledVector(dir, -hl * 0.9)], ROD);
          add(it, headGeom(b, dir, hl, HEAD_R * (hl / HEAD_LEN) ** 0.5), mat(it, 'head', () => meshMat(it, Math.max(0.6, def.opacity), 0.2)));
          setLabel(it, b.clone().addScaledVector(dir, 0.06), 0, 1, 'p3-label');
        } else {
          stroke(it, [a, b], ROD * 0.9);
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
          for (let i = 1; i < ring.length; i++) add(it, rodGeom(ring[i - 1], ring[i], ROD * 0.55), m);
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
            edges = (rings ??= sphereRings()).clone().scale(s[0], s[0], s[0]);
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
          add(it, geom, mat(it, 'fill', () => meshMat(it, opacity, 0.08)));
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
          stroke(it, pts, TUBE, true);
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
        for (const r of runs) stroke(it, r, TUBE, true);
        const all = runs.flat();
        anchorAt(all[0] ?? w(0, 0, 0));
        // An angle's arc names its angle out along the bisector, past the arc (arcs from one corner, alpha, beta and
        // gamma, put their names apart); any other curve at its middle.
        if (all.length && it.def.label) {
          const at = runs.length === 1 ? arcLabelAt(all.map((p) => [p.x, p.y, p.z] as const)) : null;
          if (at) setLabel(it, new Vector3(...at), 0.5, 0.5, 'p3-label');
          else setLabel(it, all[Math.floor((all.length - 1) / 2)], 0, 1, 'p3-label');
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
              ? mat(it, 'grid', () => lineMat(it, 0.16, false, '#ffffff'))
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
  /** The axes' labels (name and numbers), with the axis' world direction: hidden while it points at the viewer. */
  const axisLabels: { dir: Vector3; objs: CSS2DObject[] }[] = [];
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
      const name = r.label || ['x', 'y', 'z'][a];
      const nameLabel = cssLabel(name, 'p3-axis', tip.clone().addScaledVector(dir, 0.1));
      guides.add(nameLabel.obj);
      objs.push(nameLabel.obj);
      guideLabels.push({ ...nameLabel, rank: 6, movable: true });
      // ticks: a short mark across the axis, and its number
      const side = a === 2 ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0);
      for (const t of ticks[a].ticks) {
        if (Math.abs(t - o[a]) < 1e-9 * (1 + Math.abs(t)) && a > 0) continue; // the origin's number once, on x
        const p = w(...at(a, t));
        tickPairs.push(p.clone().addScaledVector(side, -0.025), p.clone().addScaledVector(side, 0.025));
        const off = a === 2 ? new Vector3(-0.07, 0, 0) : new Vector3(0, -0.07, 0);
        const tick = cssLabel(formatTick(t, ticks[a].step), 'p3-tick', p.clone().add(off), a === 2 ? 1 : 0.5, a === 2 ? 0.5 : 0);
        guides.add(tick.obj);
        objs.push(tick.obj);
        guideLabels.push({ ...tick, rank: 7, movable: false });
      }
      axisLabels.push({ dir, objs });
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
  const target = cmin.add(cmax).multiplyScalar(0.5);
  let width = 0;
  let height = 0;
  let inset = 0;
  /** Distance at which every framing point fits the part of the stage above the caption. The camera renders the
   *  lower part of a taller view (setViewOffset), so the centre of the view sits above the caption. */
  const fitDistance = (dir: Vector3) => {
    const full = Math.tan((camera.fov * DEG) / 2);
    const fullH = Math.max(1, height + inset);
    const tv = (full * Math.max(1, height - inset)) / fullH;
    const th = (full * Math.max(1, width)) / fullH;
    const right = UP.clone().cross(dir);
    if (right.lengthSq() < 1e-8) right.set(1, 0, 0);
    right.normalize();
    const up = dir.clone().cross(right).normalize();
    const margin = 1.1;
    let d = 0.5;
    for (const c of corners) {
      const p = c.clone().sub(target);
      const depth = p.dot(dir);
      d = Math.max(d, depth + (Math.abs(p.dot(right)) * margin) / th, depth + (Math.abs(p.dot(up)) * margin) / tv);
    }
    return d + 0.05;
  };
  /** The viewer's zoom (Ctrl + wheel, a pinch): a share of the fitted distance, kept when the steps turn the camera. */
  let zoom = 1;
  const viewFor = (preset: Camera) => {
    const dir = frame.dir(CAMERA_DIRS[preset]);
    return { pos: target.clone().addScaledVector(dir, fitDistance(dir) * zoom), target: target.clone() };
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
  const moveTo = (p: Camera, animate: boolean) => {
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
    const ins = clamp(Math.round(opts.insetBottom?.() ?? 0), 0, Math.round(hpx * 0.25));
    if (wpx === width && hpx === height && ins === inset) return;
    const resized = wpx !== width || hpx !== height;
    width = wpx;
    height = hpx;
    inset = ins;
    if (resized) {
      renderer.setSize(wpx, hpx);
      labels.setSize(wpx, hpx);
    }
    camera.aspect = wpx / (hpx + inset);
    camera.setViewOffset(wpx, hpx + inset, 0, inset, wpx, hpx);
    camera.updateProjectionMatrix();
    // Refit the view to the new shape (not one the viewer chose). The caption grows and shrinks from step to step:
    // the drawing then glides to stay above it.
    if (!placed || !turned) moveTo(preset, placed && !resized && !opts.reducedMotion);
    placed = true;
    dirty = true;
  };
  const ro = new ResizeObserver(() => resize());
  ro.observe(host);
  const inBottom = opts.insetElement?.();
  if (inBottom) ro.observe(inBottom);

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
    pulsing = !opts.reducedMotion && st.glow > 0 && st.highlight.length > 0;
    if (!changed && !pulsing) return;
    const pulse = pulsing ? 0.72 + 0.28 * Math.sin((now / 1000) * 2 * Math.PI * 0.8) : 1;
    for (const it of items) {
      const v = clamp(st.vis[it.def.id] ?? 1, 0, 1);
      const visible = v > 0.002;
      it.root.visible = visible;
      if (!visible) {
        it.vis = 0;
        continue;
      }
      if (stale(it) && !move(it)) build(it);
      const g = lit.has(it.def.id) ? st.glow * pulse : 0;
      if (v === it.vis && g === it.glow) continue;
      it.vis = v;
      it.glow = g;
      for (const m of it.mats) {
        m.m.opacity = Math.min(1, m.base * v + (g > 0 ? 0.18 * g * v : 0));
        if (m.m instanceof MeshStandardMaterial) m.m.emissiveIntensity = m.emissive + 0.85 * g;
        else m.m.color.copy(m.color).lerp(white, 0.5 * g);
      }
      if (it.grow === 'scale') it.root.scale.setScalar(0.6 + 0.4 * v);
      else if (it.grow === 'length') it.root.scale.setScalar(Math.max(0.001, v));
      else if (it.grow === 'draw') {
        const total = it.draws.reduce((s, d) => s + d.count, 0);
        let left = total * v;
        for (const d of it.draws) {
          const n = clamp(left, 0, d.count);
          const indexed = !!d.geom.index;
          d.geom.setDrawRange(0, v >= 1 ? Infinity : indexed ? Math.floor(n / 3) * 3 : Math.floor(n / 2) * 2);
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
  const layoutLabels = (st: SceneState) => {
    const lit = new Set(st.glow > 0 ? st.highlight : []);
    const shown = new Set(st.index >= 0 ? (scene.steps[st.index]?.show ?? []) : []);
    const entries: { outer: HTMLElement; inner: HTMLElement; rank: number; movable: boolean }[] = [];
    for (const it of items) {
      if (!it.label || !it.labelInner || !it.root.visible || !it.label.visible || it.vis < 0.2) continue;
      const id = it.def.id;
      const t = it.def.type;
      const rank = lit.has(id) ? 0 : shown.has(id) ? 1 : t === 'point' || t === 'label' ? 2 : t === 'curve' ? 4 : t === 'segment' ? 5 : 3;
      entries.push({ outer: it.label.element, inner: it.labelInner, rank, movable: true });
    }
    for (const g of guideLabels) if (g.obj.visible) entries.push({ outer: g.obj.element, inner: g.inner, rank: g.rank, movable: g.movable });
    if (!entries.length) return;
    entries.sort((a, b) => a.rank - b.rank);
    const base = host.getBoundingClientRect();
    const keep: typeof entries = [];
    const boxes = [];
    for (const e of entries) {
      if (e.outer.style.display === 'none') continue;
      const r = e.outer.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      boxes.push({ x: r.left - base.left, y: r.top - base.top, w: r.width, h: r.height, movable: e.movable });
      keep.push(e);
    }
    const places = placeLabels(boxes, {
      width: base.width,
      height: base.height,
      prev: keep.map((e) => lastPlace.get(e.inner)),
    });
    places.forEach((p, i) => {
      const inner = keep[i].inner;
      const tf = p.dx || p.dy ? `translate(${Math.round(p.dx)}px, ${Math.round(p.dy)}px)` : '';
      if (inner.style.transform !== tf) inner.style.transform = tf;
      inner.classList.toggle('dim', p.dim);
      lastPlace.set(inner, p);
    });
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
    if (st.cameraStep !== presetStep || st.camera !== preset) {
      const first = presetStep === -2;
      preset = st.camera;
      presetStep = st.cameraStep;
      if (!turned) moveTo(preset, !first && !opts.reducedMotion);
    }
    apply(st, now);
    stepTween(now);
    if (update(dt)) dirty = true;
    if (dirty || pulsing) {
      dirty = false;
      renderer.render(world, camera);
      axesFacing();
      labels.render(world, camera);
      layoutLabels(st);
    }
  };
  raf = requestAnimationFrame(loop);

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
      rings?.dispose();
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
