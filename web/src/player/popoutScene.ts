// Builds and runs the three.js stage for a 3D pop-out scene (types.ts PopoutScene). Loaded only with the pop-out
// chunk, so three.js never weighs on the main player bundle.
//
// Everything is drawn in a unit-ish box whatever the scene's ranges are: each axis is scaled to a fixed length
// (solids share one scale so a cube stays a cube). plot2d stands upright like the board (x right, y up) and the
// curve is lifted towards the viewer on a ribbon; the 3D kinds use maths axes (z up).
import {
  AmbientLight, BoxGeometry, BufferGeometry, Color, ConeGeometry, Curve, CylinderGeometry, DirectionalLight,
  DoubleSide, EdgesGeometry, Euler, Float32BufferAttribute, Group, HemisphereLight, LineBasicMaterial, LineLoop,
  LineSegments, Mesh, MeshBasicMaterial, MeshStandardMaterial, Object3D, PerspectiveCamera, PlaneGeometry,
  Quaternion, Raycaster, Scene, SphereGeometry, TorusGeometry, TubeGeometry, Vector3, WebGLRenderer, type Material,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { PopoutScene } from '../types';
import { tryCompile, type Fn } from './expr';
import { fitRange, formatTick, niceTicks, sampleSegments, stageInk } from './plotMath';

export interface StageOptions {
  accent: string;
  reducedMotion?: boolean;
}

export interface Stage {
  /** Colour and label of each drawn function or curve, for the legend. */
  legend: { label: string; color: string }[];
  /** How many things were actually drawn (0: every expression failed, show a message instead). */
  drawn: number;
  resetView(): void;
  dispose(): void;
}

type Kind = 'plot2d' | 'surface' | 'curve3d' | 'solids';
interface Range {
  min: number;
  max: number;
  label: string;
}

const BG = '#101319';
const AXIS = '#aab4c6';
const GRID = '#8d98ad';
const POINT = '#f3f5fa';
const LIFT = 0.22; // plot2d: how far the curve stands out of the board, in world units
const deg = Math.PI / 180;

const num = (v: unknown, fallback: number) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);

function range(a: PopoutScene['x'], name: string): Range {
  let min = num(a?.min, -5);
  let max = num(a?.max, 5);
  if (!(max > min)) [min, max] = [-5, 5];
  return { min, max, label: (a?.label ?? '').trim() || name };
}

/** Maths (x, y, z) to world. plot2d: world = k(p - c) with x right, y up and z out of the board. 3D kinds: maths
 *  z is world up and maths y runs away from the viewer (world -z), a proper rotation. */
class Frame {
  readonly c: [number, number, number];
  readonly space: boolean;
  readonly k: [number, number, number];
  constructor(space: boolean, r: [Range, Range, Range], k: [number, number, number], centreZ?: number) {
    this.space = space;
    this.k = k;
    this.c = [(r[0].min + r[0].max) / 2, (r[1].min + r[1].max) / 2, centreZ ?? (r[2].min + r[2].max) / 2];
  }
  w(x: number, y: number, z: number): Vector3 {
    const a = (x - this.c[0]) * this.k[0];
    const b = (y - this.c[1]) * this.k[1];
    const h = (z - this.c[2]) * this.k[2];
    return this.space ? new Vector3(a, h, -b) : new Vector3(a, b, h);
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
    const d = Math.min(1, Math.max(0, u)) * this.length;
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

function tube(pts: Vector3[], radius: number, color: string): Object3D | null {
  if (pts.length < 2) return null;
  const path = new Polyline(pts);
  if (!(path.length > 1e-6)) return null;
  const segments = Math.round(Math.min(3000, Math.max(24, path.length / 0.005)));
  const mat = new MeshStandardMaterial({
    color, roughness: 0.35, metalness: 0.08, emissive: color, emissiveIntensity: 0.18,
  });
  const g = new Group();
  g.add(new Mesh(new TubeGeometry(path, segments, radius, 12, false), mat));
  // round caps
  const cap = new SphereGeometry(radius, 12, 8);
  for (const p of [pts[0], pts[pts.length - 1]]) {
    const m = new Mesh(cap, mat);
    m.position.copy(p);
    g.add(m);
  }
  return g;
}

/** A translucent wall from the board (z = 0) up to the lifted curve. */
function ribbon(pts: Vector3[], color: string): Mesh {
  const pos: number[] = [];
  const idx: number[] = [];
  pts.forEach((p, i) => {
    pos.push(p.x, p.y, 0, p.x, p.y, p.z);
    if (i) {
      const a = 2 * (i - 1);
      idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  });
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  const m = new MeshBasicMaterial({ color, transparent: true, opacity: 0.24, side: DoubleSide, depthWrite: false });
  return new Mesh(g, m);
}

/** A low-poly stand-in for a tube, only for checking which labels it hides. */
function occluderTube(pts: Vector3[], radius: number, mat: Material): Mesh | null {
  if (pts.length < 2) return null;
  const path = new Polyline(pts);
  if (!(path.length > 1e-6)) return null;
  const segments = Math.min(400, Math.max(8, Math.round(path.length / 0.02)));
  const m = new Mesh(new TubeGeometry(path, segments, radius, 6), mat);
  m.updateMatrixWorld(true);
  return m;
}

function lines(pairs: Vector3[], color: string, opacity: number, alpha?: (p: Vector3) => number): LineSegments {
  const g = new BufferGeometry();
  g.setFromPoints(pairs);
  let mat: LineBasicMaterial;
  if (alpha) {
    const c = new Color(color);
    const col: number[] = [];
    for (const p of pairs) col.push(c.r, c.g, c.b, opacity * alpha(p));
    g.setAttribute('color', new Float32BufferAttribute(col, 4));
    mat = new LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false });
  } else mat = new LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
  return new LineSegments(g, mat);
}

/** A rod from a to b (thin cylinder), optionally with an arrowhead at b. */
function rod(a: Vector3, b: Vector3, r: number, mat: Material, head = false): Object3D {
  const g = new Group();
  const dir = b.clone().sub(a);
  const len = dir.length();
  if (len < 1e-9) return g;
  dir.normalize();
  const q = new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), dir);
  const headLen = head ? Math.min(len * 0.4, r * 9) : 0;
  const shaft = new Mesh(new CylinderGeometry(r, r, len - headLen, 8), mat);
  shaft.quaternion.copy(q);
  shaft.position.copy(a).addScaledVector(dir, (len - headLen) / 2);
  g.add(shaft);
  if (head) {
    const cone = new Mesh(new ConeGeometry(r * 3, headLen, 16), mat);
    cone.quaternion.copy(q);
    cone.position.copy(b).addScaledVector(dir, -headLen / 2);
    g.add(cone);
  }
  return g;
}

function label(text: string, cls: string, at: Vector3, cx = 0.5, cy = 0.5): CSS2DObject {
  const el = document.createElement('div');
  el.className = cls;
  el.textContent = text;
  const o = new CSS2DObject(el);
  o.position.copy(at);
  o.center.set(cx, cy);
  return o;
}

// Height colours for a single surface: indigo -> blue -> teal -> green -> yellow (bright enough on the dark stage).
const RAMP = ['#4338a8', '#2f78c4', '#25a8a8', '#7ccf6b', '#f4dc4e'].map((c) => new Color(c));
function ramp(t: number, out: Color): Color {
  const x = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  return out.lerpColors(RAMP[i], RAMP[i + 1], x - i);
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function mountStage(host: HTMLElement, raw: PopoutScene, opts: StageOptions): Stage {
  const kind = (['plot2d', 'surface', 'curve3d', 'solids'].includes(raw.kind) ? raw.kind : 'plot2d') as Kind;
  const accent = opts.accent || '#7aa7ff';
  const rx = range(raw.x, 'x');
  let ry = range(raw.y, 'y');
  let rz = range(raw.z, 'z');
  const legend: Stage['legend'] = [];

  // ---------- compile and sample (before the frame: a hopeless range is refitted to the values) ----------
  const fns = (raw.functions ?? [])
    .slice(0, 6)
    .map((f, i) => ({
      f: tryCompile(String(f?.expr ?? '')),
      color: stageInk(f?.color, '') || palette(i, accent),
      label: String(f?.label || f?.expr || ''),
    }))
    .filter((f): f is { f: Fn; color: string; label: string } => !!f.f);

  let surfaceGrids: { values: Float64Array; color: string }[] = [];
  const N = 100; // surface grid
  if (kind === 'plot2d' && fns.length) {
    const all: number[] = [];
    for (const { f } of fns) for (let i = 0; i <= 400; i++) all.push(f(rx.min + ((rx.max - rx.min) * i) / 400, 0, 0));
    const fit = fitRange(all, ry.min, ry.max);
    ry = { ...ry, ...fit };
  }
  if (kind === 'surface' && fns.length) {
    const all: number[] = [];
    surfaceGrids = fns.map(({ f, color }) => {
      const values = new Float64Array((N + 1) * (N + 1));
      for (let j = 0; j <= N; j++)
        for (let i = 0; i <= N; i++) {
          const v = f(rx.min + ((rx.max - rx.min) * i) / N, ry.min + ((ry.max - ry.min) * j) / N, 0);
          values[j * (N + 1) + i] = v;
          all.push(v);
        }
      return { values, color };
    });
    rz = { ...rz, ...fitRange(all, rz.min, rz.max) };
  }
  type CurvePts = { pts: [number, number, number][]; color: string; label: string };
  const curves: CurvePts[] = [];
  if (kind === 'curve3d') {
    (raw.curves ?? []).slice(0, 4).forEach((c, i) => {
      const fx = tryCompile(String(c?.x ?? ''));
      const fy = tryCompile(String(c?.y ?? ''));
      const fz = tryCompile(String(c?.z ?? ''));
      if (!fx || !fy || !fz) return;
      let t0 = num(c.t_min, 0);
      let t1 = num(c.t_max, 2 * Math.PI);
      if (!(t1 > t0)) [t0, t1] = [0, 2 * Math.PI];
      const pts: [number, number, number][] = [];
      for (let k = 0; k <= 900; k++) {
        const t = t0 + ((t1 - t0) * k) / 900;
        pts.push([fx(0, 0, t), fy(0, 0, t), fz(0, 0, t)]);
      }
      curves.push({ pts, color: stageInk(c.color, '') || palette(i, accent), label: String(c.label ?? '') });
    });
    const axes = [rx, ry, rz].map((r, a) => {
      const values = curves.flatMap((c) => c.pts.map((p) => p[a]));
      return { ...r, ...fitRange(values, r.min, r.max) };
    });
    ry = axes[1];
    rz = axes[2];
    rx.min = axes[0].min;
    rx.max = axes[0].max;
  }

  // ---------- frame ----------
  let frame: Frame;
  const spanX = rx.max - rx.min;
  const spanY = ry.max - ry.min;
  const spanZ = rz.max - rz.min;
  if (kind === 'plot2d') {
    const ratio = clamp(spanY / spanX, 0.55, 1.1);
    const W = 2.6 / Math.max(1, ratio);
    const H = W * ratio;
    const kx = W / spanX;
    const ky = H / spanY;
    frame = new Frame(false, [rx, ry, rz], [kx, ky, Math.min(kx, ky)], 0);
  } else if (kind === 'solids') {
    const k = 2.4 / Math.max(spanX, spanY, spanZ);
    frame = new Frame(true, [rx, ry, rz], [k, k, k]);
  } else {
    const big = Math.max(spanX, spanY);
    const L = 2.4;
    const Lx = L * clamp(spanX / big, 0.6, 1);
    const Ly = L * clamp(spanY / big, 0.6, 1);
    const Lz = L * clamp(spanZ / big, kind === 'surface' ? 0.4 : 0.45, kind === 'surface' ? 0.75 : 1.2);
    frame = new Frame(true, [rx, ry, rz], [Lx / spanX, Ly / spanY, Lz / spanZ]);
  }
  const w = (x: number, y: number, z: number) => frame.w(x, y, z);

  // ---------- three.js basics ----------
  const renderer = new WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(BG, 1);
  renderer.domElement.className = 'p3-canvas';
  host.appendChild(renderer.domElement);
  const labels = new CSS2DRenderer();
  labels.domElement.className = 'p3-labels';
  host.appendChild(labels.domElement);

  const scene = new Scene();
  const camera = new PerspectiveCamera(36, 1, 0.01, 200);
  scene.add(new AmbientLight('#ffffff', 0.55));
  scene.add(new HemisphereLight('#dfe8ff', '#1c2028', 0.9));
  const sun = new DirectionalLight('#ffffff', 1.6);
  sun.position.set(3, 5, 4);
  scene.add(sun);
  const fill = new DirectionalLight('#9fb8ff', 0.5);
  fill.position.set(-4, 1, -3);
  scene.add(fill);

  const content = new Group(); // what the camera frames
  scene.add(content);
  let drawn = 0;
  // Coarse copies of the surface and the curves: labels behind them (seen from the camera) are faded, since HTML
  // labels are always drawn on top of the canvas.
  const occluders: Mesh[] = [];
  const occluderMat = new MeshBasicMaterial({ side: DoubleSide });
  const addOccluder = (m: Mesh | null) => m && occluders.push(m);

  // ---------- content ----------
  if (kind === 'plot2d') {
    for (const fn of fns) {
      const segs = sampleSegments((x) => fn.f(x, 0, 0), rx.min, rx.max, ry.min, ry.max);
      for (const s of segs) {
        const lifted = s.map(([x, y]) => w(x, y, 0).setZ(LIFT));
        const t = tube(lifted, 0.022, fn.color);
        if (!t) continue;
        addOccluder(occluderTube(lifted, 0.026, occluderMat));
        content.add(t, ribbon(lifted, fn.color));
        const flat = lifted.map((p) => new Vector3(p.x, p.y, 0.002));
        const pairs: Vector3[] = [];
        for (let i = 1; i < flat.length; i++) pairs.push(flat[i - 1], flat[i]);
        content.add(lines(pairs, fn.color, 0.55));
        drawn++;
      }
      if (fn.label) legend.push({ label: fn.label, color: fn.color });
    }
  } else if (kind === 'surface') {
    const zlo = rz.min - 0.02 * spanZ;
    const zhi = rz.max + 0.02 * spanZ;
    surfaceGrids.forEach(({ values, color }, si) => {
      const pos = new Float32Array((N + 1) * (N + 1) * 3);
      const col = new Float32Array((N + 1) * (N + 1) * 3);
      const ok = new Uint8Array((N + 1) * (N + 1));
      const c = new Color();
      const tint = new Color(color);
      for (let j = 0; j <= N; j++)
        for (let i = 0; i <= N; i++) {
          const k = j * (N + 1) + i;
          const v = values[k];
          if (!(Number.isFinite(v) && v >= zlo && v <= zhi)) continue;
          ok[k] = 1;
          const p = w(rx.min + (spanX * i) / N, ry.min + (spanY * j) / N, v);
          pos.set([p.x, p.y, p.z], k * 3);
          const t = (v - rz.min) / spanZ;
          if (surfaceGrids.length === 1) ramp(t, c);
          else c.copy(tint).offsetHSL(0, 0, (t - 0.5) * 0.3);
          col.set([c.r, c.g, c.b], k * 3);
        }
      const idx: number[] = [];
      for (let j = 0; j < N; j++)
        for (let i = 0; i < N; i++) {
          const a = j * (N + 1) + i;
          const b = a + 1;
          const d = a + N + 1;
          const e = d + 1;
          if (ok[a] && ok[b] && ok[d]) idx.push(a, b, d);
          if (ok[b] && ok[e] && ok[d]) idx.push(b, e, d);
        }
      if (!idx.length) return;
      const g = new BufferGeometry();
      g.setAttribute('position', new Float32BufferAttribute(pos, 3));
      g.setAttribute('color', new Float32BufferAttribute(col, 3));
      g.setIndex(idx);
      g.computeVertexNormals();
      const mat = new MeshStandardMaterial({
        vertexColors: true, side: DoubleSide, roughness: 0.55, metalness: 0.02,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
        transparent: surfaceGrids.length > 1, opacity: surfaceGrids.length > 1 ? 0.85 : 1,
      });
      content.add(new Mesh(g, mat));
      if (surfaceGrids.length === 1) {
        const S = 4; // every 4th grid point: 26 x 26, cheap to raycast against
        const n = N / S;
        const cidx: number[] = [];
        const corner = (i: number, j: number) => j * S * (N + 1) + i * S;
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) {
            const a = corner(i, j);
            const b = corner(i + 1, j);
            const d = corner(i, j + 1);
            const e = corner(i + 1, j + 1);
            if (ok[a] && ok[b] && ok[d] && ok[e]) cidx.push(a, b, d, b, e, d);
          }
        const cg = new BufferGeometry();
        cg.setAttribute('position', new Float32BufferAttribute(pos, 3));
        cg.setIndex(cidx);
        const occ = new Mesh(cg, occluderMat);
        occ.updateMatrixWorld(true);
        occluders.push(occ);
      }
      // a faint wireframe every 5th grid line
      const wire: Vector3[] = [];
      const at = (k: number) => new Vector3(pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2]);
      for (let j = 0; j <= N; j++)
        for (let i = 0; i <= N; i++) {
          const k = j * (N + 1) + i;
          if (!ok[k]) continue;
          if (j % 5 === 0 && i < N && ok[k + 1]) wire.push(at(k), at(k + 1));
          if (i % 5 === 0 && j < N && ok[k + N + 1]) wire.push(at(k), at(k + N + 1));
        }
      content.add(lines(wire, '#ffffff', 0.14));
      drawn++;
      const f = fns[si];
      if (f.label) legend.push({ label: f.label, color: surfaceGrids.length === 1 ? RAMP[3].getStyle() : f.color });
    });
  } else if (kind === 'curve3d') {
    const m = 0.04;
    const inBox = (p: [number, number, number]) =>
      p.every((v, a) => {
        const r = [rx, ry, rz][a];
        const pad = m * (r.max - r.min);
        return Number.isFinite(v) && v >= r.min - pad && v <= r.max + pad;
      });
    for (const c of curves) {
      let run: Vector3[] = [];
      const flush = () => {
        const t = tube(run, 0.024, c.color);
        if (t) {
          addOccluder(occluderTube(run, 0.028, occluderMat));
          content.add(t);
          drawn++;
        }
        run = [];
      };
      for (const p of c.pts) {
        if (inBox(p)) run.push(w(p[0], p[1], p[2]));
        else flush();
      }
      flush();
      if (c.label) legend.push({ label: c.label, color: c.color });
    }
  } else {
    const mathRoot = new Group(); // maths axes, uniform scale: children use the scene's own coordinates
    if (frame.space) mathRoot.rotation.x = -Math.PI / 2;
    mathRoot.scale.setScalar(frame.k[0]);
    const inner = new Group();
    inner.position.set(-frame.c[0], -frame.c[1], -frame.c[2]);
    mathRoot.add(inner);
    content.add(mathRoot);
    (raw.solids ?? []).slice(0, 24).forEach((s, i) => {
      const obj = solid(s, stageInk(s?.color, '') || palette(i, accent), frame.k[0]);
      if (!obj) return;
      inner.add(obj.object);
      drawn++;
      if (s.label) {
        const c = s.center.map((v) => num(v, 0));
        const top = w(c[0], c[1], c[2] + obj.top);
        content.add(label(String(s.label), 'p3-solid', top.add(new Vector3(0, 0.08, 0)), 0.5, 1));
      }
    });
  }

  // points
  for (const p of (raw.points ?? []).slice(0, 24)) {
    const at = (p?.at ?? []).map((v) => num(v, NaN));
    if (at.length !== 3 || at.some((v) => !Number.isFinite(v))) continue;
    const pos = kind === 'plot2d' ? w(at[0], at[1], 0).setZ(LIFT) : w(at[0], at[1], at[2]);
    if (Math.abs(pos.x) > 4 || Math.abs(pos.y) > 4 || Math.abs(pos.z) > 4) continue; // far outside the view
    const dot = new Mesh(new SphereGeometry(0.036, 20, 14), new MeshStandardMaterial({ color: POINT, roughness: 0.3 }));
    dot.position.copy(pos);
    content.add(dot);
    if (p.label) content.add(label(String(p.label), 'p3-point', pos, 0, 1));
  }

  // ---------- axes, ticks and grid ----------
  const axisMat = new MeshBasicMaterial({ color: AXIS, transparent: true, opacity: 0.85 });
  const guides = new Group();
  scene.add(guides);
  const tx = niceTicks(rx.min, rx.max, 6);
  const ty = niceTicks(ry.min, ry.max, kind === 'plot2d' ? 5 : 6);
  const tz = niceTicks(rz.min, rz.max, 4);
  if (kind === 'plot2d') {
    const x0 = clamp(0, rx.min, rx.max);
    const y0 = clamp(0, ry.min, ry.max);
    const ext = 0.16;
    const xa = w(rx.min, y0, 0);
    const xb = w(rx.max, y0, 0).add(new Vector3(ext, 0, 0));
    const ya = w(x0, ry.min, 0);
    const yb = w(x0, ry.max, 0).add(new Vector3(0, ext, 0));
    guides.add(rod(xa, xb, 0.0065, axisMat, true), rod(ya, yb, 0.0065, axisMat, true));
    guides.add(label(rx.label, 'p3-axis', xb.clone().add(new Vector3(0.07, 0, 0)), 0, 0.5));
    guides.add(label(ry.label, 'p3-axis', yb.clone().add(new Vector3(0, 0.07, 0)), 0.5, 1));
    // graph paper on the board
    const paper: Vector3[] = [];
    for (const x of tx.ticks) paper.push(w(x, ry.min, -0.001), w(x, ry.max, -0.001));
    for (const y of ty.ticks) paper.push(w(rx.min, y, -0.001), w(rx.max, y, -0.001));
    guides.add(lines(paper, GRID, 0.1));
    // ticks and their labels
    const marks: Vector3[] = [];
    for (const x of tx.ticks) {
      const p = w(x, y0, 0);
      marks.push(p.clone().setY(p.y - 0.03), p.clone().setY(p.y + 0.03));
      guides.add(label(formatTick(x, tx.step), 'p3-tick', p.clone().setY(p.y - 0.05), 0.5, 0));
    }
    const originShown = x0 === 0 && y0 === 0 && tx.ticks.includes(0); // "0" once at the origin, from the x axis
    for (const y of ty.ticks) {
      const p = w(x0, y, 0);
      marks.push(p.clone().setX(p.x - 0.03), p.clone().setX(p.x + 0.03));
      if (y === 0 && originShown) continue;
      guides.add(label(formatTick(y, ty.step), 'p3-tick', p.clone().setX(p.x - 0.06), 1, 0.5));
    }
    guides.add(lines(marks, AXIS, 0.9));
    // a soft floor under the board, fading out
    const bottom = w(0, ry.min, 0).y - 0.14;
    const halfW = (rx.max - rx.min) * frame.k[0] * 0.5 + 0.5;
    const floor: Vector3[] = [];
    const step = 0.2;
    const fade = (p: Vector3) => {
      const r = Math.hypot(p.x / halfW, (p.z - 0.1) / 1.0);
      return clamp(1 - (r - 0.55) / 0.45, 0, 1);
    };
    const piece = (a: Vector3, b: Vector3) => {
      for (let i = 0; i < 16; i++) floor.push(a.clone().lerp(b, i / 16), a.clone().lerp(b, (i + 1) / 16));
    };
    for (let x = -Math.floor(halfW / step) * step; x <= halfW + 1e-9; x += step)
      piece(new Vector3(x, bottom, -0.9), new Vector3(x, bottom, 1.1));
    for (let z = -0.9; z <= 1.1 + 1e-9; z += step) piece(new Vector3(-halfW, bottom, z), new Vector3(halfW, bottom, z));
    guides.add(lines(floor, GRID, 0.22, fade));
  } else {
    // floor grid on z = zmin over the x-y range, ticks along its front and right edges, a z ruler at the back left
    const grid: Vector3[] = [];
    for (const x of tx.ticks) grid.push(w(x, ry.min, rz.min), w(x, ry.max, rz.min));
    for (const y of ty.ticks) grid.push(w(rx.min, y, rz.min), w(rx.max, y, rz.min));
    guides.add(lines(grid, GRID, 0.16));
    const rim = [
      w(rx.min, ry.min, rz.min), w(rx.max, ry.min, rz.min), w(rx.max, ry.min, rz.min), w(rx.max, ry.max, rz.min),
      w(rx.max, ry.max, rz.min), w(rx.min, ry.max, rz.min), w(rx.min, ry.max, rz.min), w(rx.min, ry.min, rz.min),
      w(rx.min, ry.max, rz.min), w(rx.min, ry.max, rz.max),
    ];
    guides.add(lines(rim, AXIS, 0.5));
    const out = 0.13;
    for (const x of tx.ticks)
      guides.add(label(formatTick(x, tx.step), 'p3-tick', w(x, ry.min, rz.min).add(new Vector3(0, 0, out)), 0.5, 0.5));
    for (const y of ty.ticks)
      guides.add(label(formatTick(y, ty.step), 'p3-tick', w(rx.max, y, rz.min).add(new Vector3(out, 0, 0)), 0.5, 0.5));
    const zm: Vector3[] = [];
    for (const z of tz.ticks) {
      const p = w(rx.min, ry.max, z);
      zm.push(p, p.clone().add(new Vector3(-0.04, 0, 0)));
      guides.add(label(formatTick(z, tz.step), 'p3-tick', p.clone().add(new Vector3(-0.08, 0, 0)), 1, 0.5));
    }
    guides.add(lines(zm, AXIS, 0.8));
    const xm = (rx.min + rx.max) / 2;
    const ym = (ry.min + ry.max) / 2;
    guides.add(label(rx.label, 'p3-axis', w(xm, ry.min, rz.min).add(new Vector3(0, 0, out + 0.22)), 0.5, 0.5));
    guides.add(label(ry.label, 'p3-axis', w(rx.max, ym, rz.min).add(new Vector3(out + 0.24, 0, 0)), 0.5, 0.5));
    guides.add(label(rz.label, 'p3-axis', w(rx.min, ry.max, rz.max).add(new Vector3(0, 0.1, 0)), 0.5, 1));
  }

  // ---------- camera ----------
  // Frame what is actually drawn (a sample of every mesh's vertices and every label's anchor), not its bounding
  // box: a curve that fills only part of its box (|ln x|) then still fills the view.
  const pts: Vector3[] = [];
  const v = new Vector3();
  scene.updateMatrixWorld(true);
  scene.traverse((o) => {
    if (o instanceof CSS2DObject) {
      pts.push(o.getWorldPosition(new Vector3()));
      return;
    }
    const m = o as Mesh;
    if (!m.isMesh) return;
    const pos = (m.geometry as BufferGeometry).getAttribute('position');
    if (!pos) return;
    const stride = Math.max(1, Math.floor(pos.count / 300));
    for (let i = 0; i < pos.count; i += stride)
      pts.push(v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld).clone());
  });
  // the data box too, so an empty or tiny drawing is still framed sensibly
  pts.push(w(rx.min, ry.min, kind === 'plot2d' ? 0 : rz.min), w(rx.max, ry.max, kind === 'plot2d' ? 0 : rz.max));
  const bounds = { min: new Vector3(Infinity, Infinity, Infinity), max: new Vector3(-Infinity, -Infinity, -Infinity) };
  for (const p of pts) {
    bounds.min.min(p);
    bounds.max.max(p);
  }
  const radius = Math.max(0.8, bounds.max.distanceTo(bounds.min) / 2);
  const az = (kind === 'plot2d' ? 32 : 38) * deg;
  const el = (kind === 'plot2d' ? 17 : 26) * deg;
  const sway = kind === 'plot2d' ? 24 * deg : 0; // plot2d rocks gently instead of spinning round to its back
  const toward = (a: number) => new Vector3(Math.sin(a) * Math.cos(el), Math.sin(el), Math.cos(a) * Math.cos(el));
  const basis = (a: number) => {
    const f = toward(a);
    const right = new Vector3(0, 1, 0).cross(f).normalize();
    return { f, right, up: f.clone().cross(right).normalize() };
  };
  const dir = toward(az);
  const target = bounds.min.clone().add(bounds.max).multiplyScalar(0.5);
  if (sway) {
    // centre the drawing as seen from the starting angle
    const { right, up } = basis(az);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of pts) {
      const d = p.clone().sub(target);
      x0 = Math.min(x0, d.dot(right));
      x1 = Math.max(x1, d.dot(right));
      y0 = Math.min(y0, d.dot(up));
      y1 = Math.max(y1, d.dot(up));
    }
    target.addScaledVector(right, (x0 + x1) / 2).addScaledVector(up, (y0 + y1) / 2);
  }
  const home = { pos: new Vector3(), target: target.clone() };
  const rel = pts.map((p) => p.clone().sub(target));
  /** Distance at which the drawing fills the view (with room for label text) from every angle the auto-rotation
   *  visits: the starting angle and both ends of the sway for plot2d, all round for the 3D kinds. */
  const fitDistance = (aspect: number) => {
    const tv = Math.tan((camera.fov * deg) / 2);
    const th = tv * aspect;
    const margin = 1.1;
    const angles = sway ? [az - sway, az, az + sway] : Array.from({ length: 12 }, (_, i) => (i * Math.PI) / 6);
    let d = 0;
    for (const a of angles) {
      const { f, right, up } = basis(a);
      for (const p of rel) {
        const depth = p.dot(f);
        d = Math.max(d, depth + (Math.abs(p.dot(right)) * margin) / th, depth + (Math.abs(p.dot(up)) * margin) / tv);
      }
    }
    return d;
  };

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.7;
  controls.zoomSpeed = 0.8;
  controls.target.copy(target);
  controls.autoRotate = !opts.reducedMotion;
  controls.autoRotateSpeed = kind === 'plot2d' ? 0.9 : 0.6;
  let dirty = true;
  controls.addEventListener('start', () => {
    controls.autoRotate = false;
  });
  controls.addEventListener('change', () => {
    dirty = true;
  });

  let width = 0;
  let height = 0;
  let placed = false;
  const resize = () => {
    const wpx = host.clientWidth;
    const hpx = host.clientHeight;
    if (!wpx || !hpx || (wpx === width && hpx === height)) return;
    width = wpx;
    height = hpx;
    renderer.setSize(wpx, hpx);
    labels.setSize(wpx, hpx);
    camera.aspect = wpx / hpx;
    camera.updateProjectionMatrix();
    if (!placed) {
      placed = true;
      home.pos.copy(target).addScaledVector(dir, fitDistance(camera.aspect));
      camera.position.copy(home.pos);
      controls.minDistance = radius * 0.4;
      controls.maxDistance = home.pos.distanceTo(target) * 3;
      controls.update();
    }
    dirty = true;
  };
  const ro = new ResizeObserver(resize);
  ro.observe(host);
  resize();

  const tagged: CSS2DObject[] = [];
  // Point labels sit on their curve, so a curve never hides them; a surface can.
  const curvy = kind === 'plot2d' || kind === 'curve3d';
  if (occluders.length)
    scene.traverse((o) => {
      if (o instanceof CSS2DObject && !(curvy && o.element.classList.contains('p3-point'))) tagged.push(o);
    });
  const ray = new Raycaster();
  const at = new Vector3();
  let occlusionDue = false;
  let lastOcclusion = 0;
  const updateOcclusion = () => {
    for (const o of tagged) {
      o.getWorldPosition(at);
      const d = at.distanceTo(camera.position);
      ray.set(camera.position, at.sub(camera.position).divideScalar(d));
      ray.far = d - 0.06;
      o.element.classList.toggle('behind', ray.intersectObjects(occluders, false).length > 0);
    }
  };

  let raf = 0;
  let last = performance.now();
  const frameLoop = (now: number) => {
    raf = requestAnimationFrame(frameLoop);
    const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
    last = now;
    if (!placed) return;
    if (sway && controls.autoRotate) {
      const a = controls.getAzimuthalAngle();
      if (a > az + sway) controls.autoRotateSpeed = Math.abs(controls.autoRotateSpeed);
      else if (a < az - sway) controls.autoRotateSpeed = -Math.abs(controls.autoRotateSpeed);
    }
    const moved = controls.update(dt);
    if (moved || dirty) {
      dirty = false;
      renderer.render(scene, camera);
      labels.render(scene, camera);
      occlusionDue = true;
    }
    if (occlusionDue && now - lastOcclusion > 120) {
      occlusionDue = false;
      lastOcclusion = now;
      updateOcclusion();
    }
  };
  raf = requestAnimationFrame(frameLoop);

  return {
    legend,
    drawn,
    resetView() {
      camera.position.copy(home.pos);
      controls.target.copy(home.target);
      controls.update();
      dirty = true;
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      const geoms = new Set<BufferGeometry>();
      const mats = new Set<Material>();
      scene.traverse((o) => {
        const m = o as Mesh;
        if (m.geometry) geoms.add(m.geometry as BufferGeometry);
        const mat = m.material as Material | Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => mats.add(x));
        else if (mat) mats.add(mat);
      });
      for (const o of occluders) {
        geoms.add(o.geometry);
        mats.add(o.material as Material);
      }
      geoms.forEach((g) => g.dispose());
      mats.forEach((m) => m.dispose());
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      labels.domElement.remove();
    },
  };
}

/** A distinct colour for the i-th item when the scene gives none (the first is the theme accent). */
function palette(i: number, accent: string): string {
  return [accent, '#ff9f6b', '#6fdc9a', '#f2c94c', '#c792ea', '#5fd4e6'][i % 6];
}

/** One solid in maths coordinates (the caller scales and orients its parent). `top` is how far it reaches above
 *  its centre, for the label. Sizes: box [w, d, h]; sphere [r] or diameters [dx, dy, dz]; cylinder and cone
 *  [r, h]; torus [R, r]; plane [w, d]; arrow: a vector [dx, dy, dz] or a length along x. */
function solid(
  s: NonNullable<PopoutScene['solids']>[number],
  color: string,
  scale: number,
): { object: Object3D; top: number } | null {
  const c = (s?.center ?? []).map((v) => num(v, NaN));
  if (c.length !== 3 || c.some((v) => !Number.isFinite(v))) return null;
  const size = (s.size ?? []).map((v) => Math.min(1e4, Math.max(1e-3, Math.abs(num(v, 1)))));
  if (!size.length) return null;
  const [a, b = a, d = b] = size;
  const shape = String(s.shape ?? 'box').toLowerCase();
  const fillMat = new MeshStandardMaterial({
    color, transparent: true, opacity: 0.32, roughness: 0.5, metalness: 0.05, side: DoubleSide, depthWrite: false,
  });
  const edgeMat = new LineBasicMaterial({ color: new Color(color).offsetHSL(0, 0, 0.12) });
  const holder = new Group();
  holder.position.set(c[0], c[1], c[2]);
  const rot = (s.rotation ?? []).map((v) => num(v, 0));
  holder.rotation.copy(new Euler((rot[0] ?? 0) * deg, (rot[1] ?? 0) * deg, (rot[2] ?? 0) * deg, 'XYZ'));
  let geom: BufferGeometry;
  let top: number;
  let rings = false;
  switch (shape) {
    case 'sphere': {
      const r = size.length === 3 ? [a / 2, b / 2, d / 2] : [a, a, a];
      geom = new SphereGeometry(1, 48, 32);
      geom.scale(r[0], r[1], r[2]);
      top = r[2];
      rings = true;
      break;
    }
    case 'cylinder':
    case 'cone': {
      const [rx, ry, h] = size.length === 3 ? [a / 2, b / 2, d] : [a, a, size.length === 2 ? b : 2 * a];
      geom = shape === 'cone' ? new ConeGeometry(1, h, 64) : new CylinderGeometry(1, 1, h, 64);
      geom.scale(rx, 1, ry);
      geom.rotateX(Math.PI / 2); // axis along maths z
      top = h / 2;
      break;
    }
    case 'torus': {
      const R = a;
      const r = size.length >= 2 ? Math.min(b, R) : R / 4;
      geom = new TorusGeometry(R, r, 32, 96);
      top = r;
      break;
    }
    case 'plane':
      geom = new PlaneGeometry(a, size.length >= 2 ? b : a);
      top = 0;
      break;
    case 'arrow': {
      const raw = (s.size ?? []).map((v) => num(v, 0));
      const v = raw.length === 3 ? new Vector3(raw[0], raw[1], raw[2]) : new Vector3(Math.abs(raw[0] || 1), 0, 0);
      if (v.length() < 1e-6) return null;
      const r = 0.012 / scale; // a constant thickness on screen whatever the scene's units
      const mat = new MeshStandardMaterial({ color, roughness: 0.4, emissive: color, emissiveIntensity: 0.15 });
      holder.add(rod(new Vector3(), v, r, mat, true));
      return { object: holder, top: Math.max(0, v.z) };
    }
    default:
      geom = new BoxGeometry(a, b, d);
      top = d / 2;
  }
  const mesh = new Mesh(geom, fillMat);
  mesh.renderOrder = 1;
  holder.add(mesh);
  holder.add(new LineSegments(new EdgesGeometry(geom, 25), edgeMat));
  if (rings) {
    // a smooth sphere has no edges: an equator and a meridian keep it crisp
    const circle = (rotate: (g: BufferGeometry) => void) => {
      const pts: Vector3[] = [];
      for (let i = 0; i < 96; i++) {
        const a = (i / 96) * 2 * Math.PI;
        pts.push(new Vector3(Math.cos(a), Math.sin(a), 0));
      }
      const g = new BufferGeometry().setFromPoints(pts);
      rotate(g);
      const bb = new Vector3();
      geom.computeBoundingBox();
      geom.boundingBox!.getSize(bb);
      g.scale(bb.x / 2, bb.y / 2, bb.z / 2);
      return new LineLoop(g, edgeMat);
    };
    holder.add(circle(() => {}), circle((g) => g.rotateX(Math.PI / 2)));
  }
  return { object: holder, top };
}
