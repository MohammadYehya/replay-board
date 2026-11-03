// The 3D view's pure logic, with no three.js and no DOM, so it runs in Node tests:
// - cleanScene() reads a format-2 scene defensively (the server may be older or newer than this player): anything
//   malformed is dropped, never trusted;
// - sceneState() says what the view shows at a moment: a pure function of (scene, clock), so seeking anywhere, even
//   backwards, gives the same picture as playing up to there.
//
// STATE RULES (shared with the server's bounds check): before the first step every parameter is at its value, only
// the objects that no step shows are visible (the base drawing), and the camera is at the preset of the first step
// that names one ("iso" when none does). Step i applies when the clock reaches its time (synced) or when the viewer
// steps to it (not synced): parameters move, eased, from their value at that moment to "to" over "over" seconds;
// "show" objects fade in and "hide" objects fade out over the same time; "highlight" glows while step i is the current
// step only; "camera" moves to its preset when the step starts. Defs are worked out from the parameters at every
// moment (the stage does that), so they are not part of the state.
import type {
  CameraPreset, Scene3D, Scene3DAxis, Scene3DDef, Scene3DObject, Scene3DObjectType, Scene3DParam, Scene3DStep, Vec3,
  Vec3E,
} from '../types';
import { measureAngle } from './angle3d.ts';
import { GRAMMAR_NAMES, tryCompileVars, type VarsFn } from './expr.ts';
import { formatTime } from './timing.ts';

export const OBJECT_TYPES: readonly Scene3DObjectType[] = [
  'point', 'segment', 'vector', 'polygon', 'box', 'sphere', 'cylinder', 'cone', 'torus', 'graph', 'surface', 'curve',
  'psurface', 'label', 'angle',
];
export const CAMERAS: readonly CameraPreset[] = ['keep', 'front', 'top', 'side', 'iso'];
export const LIMITS = {
  params: 8,
  defs: 16,
  objects: 40,
  steps: 12,
  corners: 12,
  title: 80,
  explain: 300,
  caption: 140,
  label: 30,
  paramLabel: 32,
  axisLabel: 20,
  overMin: 0.6,
  overMax: 8,
} as const;
/** The variables of each kind of expression field, besides the scene's parameters. */
export const FIELD_VARS = {
  graph: ['x'],
  surface: ['x', 'y'],
  curve: ['t'],
  psurface: ['u', 'v'],
} as const;
/** Parameter names that would clash with a field's variable. */
const RESERVED = new Set(['x', 'y', 't', 'u', 'v']);
const PARAM_NAME = /^[a-z][a-z0-9_]{0,15}$/;
const OBJECT_ID = /^[a-z][a-z0-9_]{0,23}$/;
const COLOR = /^#[0-9a-f]{6}$/i;

export type CleanResult =
  | { kind: 'scene'; scene: Scene3D }
  /** A scene from a server older than steps (format 1): the owner must restart the server. */
  | { kind: 'old' }
  | { kind: 'bad'; why: string };

type Raw = Record<string, unknown>;
const isObj = (v: unknown): v is Raw => !!v && typeof v === 'object' && !Array.isArray(v);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const fin = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Plain text, whitespace collapsed, cut at a word with "…" to at most max characters. */
export function text(v: unknown, max: number): string {
  if (typeof v !== 'string') return '';
  const s = v.replace(/\s+/g, ' ').trim();
  if (s.length <= max) return s;
  const cut = s.slice(0, max - 1);
  const sp = cut.lastIndexOf(' ');
  return (sp > max * 0.6 ? cut.slice(0, sp) : cut).replace(/[\s,;:.-]+$/, '') + '…';
}

/** An expression string (a plain number is one too), or null. */
function exprOf(v: unknown): string | null {
  if (typeof v === 'number' && Number.isFinite(v)) return String(v);
  if (typeof v !== 'string') return null;
  const s = v.trim();
  return s && s.length <= 500 ? s : null;
}

function cleanParams(raw: unknown): Scene3DParam[] {
  const out: Scene3DParam[] = [];
  for (const p of arr(raw)) {
    if (out.length >= LIMITS.params) break;
    if (!isObj(p) || typeof p.name !== 'string') continue;
    const name = p.name.trim();
    if (!PARAM_NAME.test(name) || GRAMMAR_NAMES.has(name) || RESERVED.has(name)) continue;
    if (out.some((q) => q.name === name)) continue;
    const value = fin(p.value);
    if (value === null) continue;
    let min = fin(p.min) ?? value;
    let max = fin(p.max) ?? value;
    if (min > max) [min, max] = [max, min];
    out.push({
      name,
      value,
      min: Math.min(min, value),
      max: Math.max(max, value),
      label: text(p.label, LIMITS.paramLabel) || name,
      unit: p.unit === 'deg' ? 'deg' : '',
    });
  }
  return out;
}

/** Defs in order: each may use the parameters and the defs before it (no field variable); a malformed one is left
 *  out, and so is every object that uses it. */
function cleanDefs(raw: unknown, params: string[]): Scene3DDef[] {
  const out: Scene3DDef[] = [];
  const names = [...params];
  for (const d of arr(raw)) {
    if (out.length >= LIMITS.defs) break;
    if (!isObj(d) || typeof d.name !== 'string') continue;
    const name = d.name.trim();
    if (!PARAM_NAME.test(name) || GRAMMAR_NAMES.has(name) || RESERVED.has(name) || names.includes(name)) continue;
    const expr = exprOf(d.expr);
    if (expr === null || !tryCompileVars(expr, names)) continue;
    out.push({ name, expr });
    names.push(name);
  }
  return out;
}

/** Reads one object; null when it is malformed (it is then left out, with every step's mention of it). */
function cleanObject(o: unknown, params: string[]): Scene3DObject | null {
  if (!isObj(o)) return null;
  const id = typeof o.id === 'string' ? o.id.trim() : '';
  const type = o.type as Scene3DObjectType;
  if (!OBJECT_ID.test(id) || !OBJECT_TYPES.includes(type)) return null;
  const e = (v: unknown, vars: readonly string[] = []): string | null => {
    const s = exprOf(v);
    return s !== null && tryCompileVars(s, [...vars, ...params]) ? s : null;
  };
  const vec = (v: unknown): Vec3E | null => {
    const a = arr(v);
    if (a.length < 3) return null;
    const out = a.slice(0, 3).map((x) => e(x));
    return out.every((x) => x !== null) ? (out as Vec3E) : null;
  };
  const list = (v: unknown, n: number): string[] | null => {
    const a = arr(v);
    if (a.length < n) return null;
    const out = a.slice(0, n).map((x) => e(x));
    return out.every((x) => x !== null) ? (out as string[]) : null;
  };
  const pair = (v: unknown) => list(v, 2) as [string, string] | null;
  const opacity = fin(o.opacity);
  const obj: Scene3DObject = {
    id,
    type,
    points: [],
    center: [],
    size: [],
    rotation: [],
    expr: '',
    x: '',
    y: '',
    z: '',
    range: [],
    range2: [],
    color: typeof o.color === 'string' && COLOR.test(o.color.trim()) ? o.color.trim().toLowerCase() : '',
    opacity: opacity === null ? 1 : clamp(opacity, 0.15, 1),
    style: o.style === 'wire' || o.style === 'dashed' ? o.style : 'solid',
    label: text(o.label, LIMITS.label),
  };
  const points = arr(o.points);
  switch (type) {
    case 'angle': {
      // exactly three points (the vertex, then a point on each arm), and no size or one radius
      if (points.length !== 3) return null;
      const ps = points.map(vec);
      if (ps.some((p) => !p)) return null;
      const size = arr(o.size);
      if (size.length > 1) return null;
      const r = size.length ? e(size[0]) : null;
      if (size.length && r === null) return null;
      obj.points = ps as Vec3E[];
      obj.size = r === null ? [] : [r];
      break;
    }
    case 'point':
    case 'label': {
      const p = vec(points[0]);
      if (!p || (type === 'label' && !obj.label)) return null;
      obj.points = [p];
      break;
    }
    case 'segment':
    case 'vector': {
      const a = vec(points[0]);
      const b = vec(points[1]);
      if (!a || !b) return null;
      obj.points = [a, b];
      break;
    }
    case 'polygon': {
      const ps = points.slice(0, LIMITS.corners).map(vec);
      if (ps.length < 3 || ps.some((p) => !p)) return null;
      obj.points = ps as Vec3E[];
      break;
    }
    case 'box':
    case 'sphere':
    case 'cylinder':
    case 'cone':
    case 'torus': {
      const center = vec(o.center);
      const size = list(o.size, type === 'box' ? 3 : type === 'sphere' ? 1 : 2);
      if (!center || !size) return null;
      obj.center = center;
      obj.size = size;
      obj.rotation = vec(o.rotation) ?? [];
      break;
    }
    case 'graph':
    case 'surface': {
      const expr = e(o.expr, FIELD_VARS[type]);
      if (!expr) return null;
      obj.expr = expr;
      if (type === 'graph') obj.range = pair(o.range) ?? [];
      break;
    }
    case 'curve':
    case 'psurface': {
      const vars = FIELD_VARS[type];
      const [x, y, z] = [e(o.x, vars), e(o.y, vars), e(o.z, vars)];
      const range = pair(o.range);
      const range2 = type === 'psurface' ? pair(o.range2) : null;
      if (!x || !y || !z || !range || (type === 'psurface' && !range2)) return null;
      Object.assign(obj, { x, y, z, range, range2: range2 ?? [] });
      break;
    }
  }
  return obj;
}

function cleanAxis(a: unknown): Scene3DAxis {
  const o = isObj(a) ? a : {};
  const min = fin(o.min);
  const max = fin(o.max);
  const ok = min !== null && max !== null && max > min;
  return { min: ok ? min : -5, max: ok ? max : 5, label: text(o.label, LIMITS.axisLabel) };
}

function cleanBounds(b: unknown): [Vec3, Vec3] | null {
  const [lo, hi] = arr(b).map((c) => arr(c).slice(0, 3).map(fin));
  if (!lo || !hi || lo.length !== 3 || hi.length !== 3) return null;
  if (lo.some((v) => v === null) || hi.some((v) => v === null)) return null;
  if (lo.some((v, i) => (v as number) > (hi[i] as number))) return null;
  return [lo as Vec3, hi as Vec3];
}

function cleanSteps(raw: unknown, ids: Set<string>, params: Set<string>): Scene3DStep[] {
  const out: Scene3DStep[] = [];
  const idList = (v: unknown) => [...new Set(arr(v).filter((x): x is string => typeof x === 'string' && ids.has(x)))];
  for (const s of arr(raw)) {
    if (!isObj(s)) continue;
    const show = idList(s.show);
    const hide = idList(s.hide).filter((id) => !show.includes(id)); // shown and hidden at once: shown
    const highlight = idList(s.highlight);
    const set: Scene3DStep['set'] = [];
    for (const x of arr(s.set)) {
      if (!isObj(x) || typeof x.param !== 'string' || !params.has(x.param)) continue;
      const to = fin(x.to);
      if (to === null) continue;
      const k = set.findIndex((y) => y.param === x.param);
      if (k >= 0) set[k] = { param: x.param, to }; // the last word on a parameter wins
      else set.push({ param: x.param, to });
    }
    const camera = CAMERAS.includes(s.camera as CameraPreset) ? (s.camera as CameraPreset) : 'keep';
    const caption = text(s.caption, LIMITS.caption);
    if (!caption && !show.length && !hide.length && !highlight.length && !set.length && camera === 'keep') continue;
    out.push({
      at: typeof s.at === 'string' ? s.at.trim().slice(0, 10) : '',
      t: fin(s.t),
      over: clamp(fin(s.over) ?? 1.5, LIMITS.overMin, LIMITS.overMax),
      caption,
      show,
      hide,
      highlight,
      set,
      camera,
    });
  }
  return out;
}

/** Reads a scene from the server. Never throws. */
export function cleanScene(raw: unknown): CleanResult {
  if (!isObj(raw)) return { kind: 'bad', why: 'The 3D view came back empty.' };
  const format = raw.format;
  if (format === undefined || (typeof format === 'number' && format < 2)) {
    return 'kind' in raw || format !== undefined ? { kind: 'old' } : { kind: 'bad', why: 'The 3D view came back in a form this player does not know.' };
  }
  if (typeof format !== 'number') return { kind: 'bad', why: 'The 3D view came back in a form this player does not know.' };
  const params = cleanParams(raw.params);
  const names = params.map((p) => p.name);
  const defs = cleanDefs(raw.defs, names);
  const usable = [...names, ...defs.map((d) => d.name)];
  const objects: Scene3DObject[] = [];
  for (const o of arr(raw.objects)) {
    if (objects.length >= LIMITS.objects) break;
    const c = cleanObject(o, usable);
    if (c && !objects.some((x) => x.id === c.id)) objects.push(c);
  }
  let steps = cleanSteps(raw.steps, new Set(objects.map((o) => o.id)), new Set(names));
  let synced = raw.synced === true && steps.some((s) => s.t !== null);
  if (synced) {
    steps = steps.filter((s) => s.t !== null);
    steps = steps.map((s, i) => ({ s, i })).sort((a, b) => a.s.t! - b.s.t! || a.i - b.i).map(({ s }) => s);
  } else steps = steps.map((s) => ({ ...s, t: null }));
  steps = steps.slice(0, LIMITS.steps);
  synced = synced && steps.length > 0;
  const axes = isObj(raw.axes) ? raw.axes : {};
  const piece = fin(raw.piece);
  return {
    kind: 'scene',
    scene: {
      format: 2,
      version: typeof raw.version === 'string' ? raw.version : '',
      piece: piece !== null ? Math.round(piece) : -1,
      pieces: arr(raw.pieces).filter((p): p is number => fin(p) !== null).map(Math.round),
      title: text(raw.title, LIMITS.title),
      explain: text(raw.explain, LIMITS.explain),
      synced,
      axes: { show: axes.show === true, x: cleanAxis(axes.x), y: cleanAxis(axes.y), z: cleanAxis(axes.z) },
      params,
      defs,
      objects,
      steps,
      bounds: cleanBounds(raw.bounds),
      isoFrom: raw.iso_from === 'left' ? 'left' : 'right',
    },
  };
}

/**
 * One request per drawing, shared: going back to the board while Muse Spark is still thinking and opening the view
 * again (or React running an effect twice in development) must not start a second model call. Only a usable scene is
 * kept for the session: a failure, an older server's answer or a malformed one is forgotten, so "Try again" (or
 * opening the view again once the server is restarted) asks again.
 */
export function sceneLoader(
  fetchScene: (jobId: string, pieceId: number) => Promise<unknown>,
): (jobId: string, pieceId: number) => Promise<CleanResult> {
  const kept = new Map<string, Promise<CleanResult>>();
  return (jobId, pieceId) => {
    const key = `${jobId}/${pieceId}`;
    let p = kept.get(key);
    if (!p) {
      const mine = fetchScene(jobId, pieceId).then(cleanScene);
      kept.set(key, mine);
      const forget = () => {
        if (kept.get(key) === mine) kept.delete(key);
      };
      // registered before any caller's own handlers, so the entry is gone by the time a caller sees the outcome
      mine.then((r) => r.kind !== 'scene' && forget(), forget);
      p = mine;
    }
    return p;
  };
}

// ---------------------------------------------------------------------------------------------------------------
// The state at a moment

/** Virtual seconds between the steps of a scene that is not synced: more than the longest change. */
export const STEP_GAP = 10;
/** Seconds a highlight takes to come up. */
export const GLOW_IN = 0.35;

/** Where the view is: a moment on the video's clock (synced), or a step the viewer went to and how long ago. */
export type Moment = { clock: number } | { step: number; elapsed: number };

export interface StateOptions {
  /** prefers-reduced-motion: every change jumps instead of animating. */
  reducedMotion?: boolean;
  /** "Try it": values the viewer set with the sliders, on top of the lecture's (each kept inside its parameter's
   *  range). A parameter in here does not move with the steps until the overrides end. */
  overrides?: Readonly<Record<string, number>>;
  /** The scene's check that its angles make a shape that exists (shapeGuard): the viewer's values are applied only
   *  as far as they do with the lecture's values of the moment (fitToShape). */
  guard?: ShapeGuard | null;
}

export type Camera = Exclude<CameraPreset, 'keep'>;

export interface SceneState {
  /** The current step (-1: before the first; the base drawing). */
  index: number;
  /** How far the current step's change has gone, 0..1 (1 when it has none or it is done). */
  progress: number;
  /** Every parameter's value. */
  params: Record<string, number>;
  /** The parameters a step is moving at this moment (not the viewer's own values), in the scene's order. */
  changing: string[];
  /** The parameters at the viewer's own value ("Try it"), in the scene's order. */
  overridden: string[];
  /** Every object's visibility, 0 (hidden) .. 1 (fully shown). */
  vis: Record<string, number>;
  /** Objects that glow now (the current step's). */
  highlight: string[];
  /** How strongly they glow, 0..1. */
  glow: number;
  /** The camera preset asked for by now, and the step that asked (-1: the opening view). */
  camera: Camera;
  cameraStep: number;
  /** Something is still changing at this moment (keep drawing frames). */
  moving: boolean;
}

/** easeInOutCubic. */
export function ease(p: number): number {
  if (!(p > 0)) return 0;
  if (p >= 1) return 1;
  return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
}

/** When each step starts: its lecture time when synced, else STEP_GAP apart on a virtual clock. */
export function stepTimes(scene: Scene3D): number[] {
  return scene.steps.map((s, i) => (scene.synced && s.t !== null ? s.t : i * STEP_GAP));
}

/** The clock of a moment (-Infinity before the first step). */
export function momentClock(scene: Scene3D, at: Moment): number {
  if ('clock' in at) return Number.isFinite(at.clock) ? at.clock : -Infinity;
  const times = stepTimes(scene);
  if (!(at.step >= 0) || !times.length) return -Infinity;
  const i = Math.min(Math.floor(at.step), times.length - 1);
  const next = times[i + 1] ?? Infinity;
  return Math.min(times[i] + Math.max(0, at.elapsed || 0), next - 1e-6);
}

/** The current step at a clock: the last one that has started (-1 before the first). */
export function stepAt(times: number[], clock: number): number {
  let index = -1;
  for (let i = 0; i < times.length && times[i] <= clock; i++) index = i;
  return index;
}

interface Seg {
  from: number;
  to: number;
  t0: number;
  d: number;
}
const segAt = (s: Seg, t: number) =>
  s.d <= 0 || t >= s.t0 + s.d ? s.to : t <= s.t0 ? s.from : s.from + (s.to - s.from) * ease((t - s.t0) / s.d);
const segMoving = (s: Seg, t: number) => s.d > 0 && t >= s.t0 && t < s.t0 + s.d && s.from !== s.to;

/** The camera before the first step: the first preset a step names, so a graph seen from the top opens from the top
 *  instead of on empty 3D axes; "iso" when no step names one. */
export function openingCamera(scene: Scene3D): Camera {
  const s = scene.steps.find((x) => x.camera !== 'keep');
  return s ? (s.camera as Camera) : 'iso';
}

/** What the view shows at a moment. Pure: the same scene and moment always give the same state. */
export function sceneState(scene: Scene3D, at: Moment, opts: StateOptions = {}): SceneState {
  const times = stepTimes(scene);
  const clock = momentClock(scene, at);
  const index = stepAt(times, clock);
  const rm = !!opts.reducedMotion;

  const shownLater = new Set(scene.steps.flatMap((s) => s.show));
  const pseg = new Map<string, Seg>(scene.params.map((p) => [p.name, { from: p.value, to: p.value, t0: -Infinity, d: 0 }]));
  const vseg = new Map<string, Seg>(
    scene.objects.map((o) => {
      const v = shownLater.has(o.id) ? 0 : 1;
      return [o.id, { from: v, to: v, t0: -Infinity, d: 0 }];
    }),
  );
  let camera: Camera = openingCamera(scene);
  let cameraStep = -1;
  for (let i = 0; i <= index; i++) {
    const s = scene.steps[i];
    const t = times[i];
    const d = rm ? 0 : s.over;
    for (const { param, to } of s.set) {
      const seg = pseg.get(param);
      if (seg) pseg.set(param, { from: segAt(seg, t), to, t0: t, d });
    }
    for (const [ids, to] of [
      [s.show, 1],
      [s.hide, 0],
    ] as const) {
      for (const id of ids) {
        const seg = vseg.get(id);
        if (seg) vseg.set(id, { from: segAt(seg, t), to, t0: t, d });
      }
    }
    if (s.camera !== 'keep') {
      camera = s.camera;
      cameraStep = i;
    }
  }

  let moving = false;
  let params: Record<string, number> = {};
  const changing: string[] = [];
  const overridden: string[] = [];
  const own = opts.overrides ?? {};
  const lecture: Record<string, number> = {};
  const mineNow: Record<string, number> = {};
  for (const p of scene.params) {
    const seg = pseg.get(p.name)!;
    lecture[p.name] = segAt(seg, clock);
    const mine = Object.prototype.hasOwnProperty.call(own, p.name) ? own[p.name] : undefined;
    if (typeof mine === 'number' && Number.isFinite(mine)) {
      params[p.name] = mineNow[p.name] = clamp(mine, Math.min(p.min, p.max), Math.max(p.min, p.max));
      overridden.push(p.name);
      continue;
    }
    params[p.name] = lecture[p.name];
    if (segMoving(seg, clock)) {
      moving = true;
      changing.push(p.name);
    }
  }
  // The lecture may move an angle the viewer left alone into one that, with theirs, makes no shape (they set beta
  // with gamma at 90, then the hexagon step takes gamma to 120): theirs then give way, as little as it takes.
  if (opts.guard && overridden.length) params = fitToShape(opts.guard, lecture, mineNow);
  const vis: Record<string, number> = {};
  for (const [id, seg] of vseg) {
    vis[id] = segAt(seg, clock);
    moving ||= segMoving(seg, clock);
  }
  const step = index >= 0 ? scene.steps[index] : null;
  const since = step ? clock - times[index] : 0;
  const glow = !step || !step.highlight.length ? 0 : rm ? 1 : ease(Math.min(1, since / GLOW_IN));
  if (step && step.highlight.length && glow < 1) moving = true;
  const d = step && !rm ? step.over : 0;
  return {
    index,
    progress: !step ? 0 : d > 0 ? clamp(since / d, 0, 1) : 1,
    params,
    changing,
    overridden,
    vis,
    highlight: step ? step.highlight : [],
    glow,
    camera,
    cameraStep,
    moving,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Words and targets for the view's controls

/** Seconds of lecture to replay before the first step when "Play the explanation" starts. */
export const LEAD_IN = 2;

/**
 * Where the video seeks to show step i. Playing: just past its start, so its change plays. Paused (the video stays
 * paused: one clock, so a change would freeze at its first frame): where its change is over, before the next step.
 */
export function seekTimeFor(scene: Scene3D, i: number, playing = true): number | null {
  const s = scene.synced ? scene.steps[i] : undefined;
  if (!s || typeof s.t !== 'number') return null;
  if (playing) return s.t + 0.02;
  const next = scene.steps[i + 1]?.t;
  const end = Math.min(s.t + Math.max(s.over, GLOW_IN) + 0.02, typeof next === 'number' ? next - 0.05 : Infinity);
  return Math.max(s.t + 0.02, end);
}

/** Where "Play the explanation" starts: a moment before the first step, not before the analysed clip. */
export function playFrom(scene: Scene3D, clipStart: number): number | null {
  const t = scene.synced ? scene.steps[0]?.t : null;
  return typeof t === 'number' ? Math.max(clipStart, t - LEAD_IN) : null;
}

/** Seconds between the start of "Play the explanation" and the drawing's own explanation that make a second
 *  button worth it (test2 #7: the series starts at 0:13, the hexagon itself at 3:52). */
export const DRAWING_LATER = 20;

/**
 * Where "This drawing from m:ss" starts: a moment before the teacher first talks about the drawing itself (`spoken`:
 * the starts of its spoken ranges), when that is well after the first step and steps follow it; else null (one
 * button is enough).
 */
export function drawingFrom(scene: Scene3D, spoken: readonly number[], clipStart: number, clipEnd = Infinity): number | null {
  const first = scene.synced ? scene.steps[0]?.t : null;
  if (typeof first !== 'number') return null;
  const talked = spoken.filter((t) => Number.isFinite(t) && t >= clipStart - 1 && t <= clipEnd);
  if (!talked.length) return null;
  const t = Math.min(...talked);
  if (t < first + DRAWING_LATER) return null;
  if (!scene.steps.some((s) => s.t !== null && s.t >= t - LEAD_IN - 3)) return null;
  return Math.max(clipStart, t - LEAD_IN);
}

/** Seconds a step stays on screen when the viewer plays the steps of a scene that is not synced: its change, then
 *  time to read its caption. */
export function dwell(step: Scene3DStep): number {
  return step.over + clamp(step.caption.length / 15, 2.5, 7);
}

/** "Step 2 of 6 · 3:07" (the time only when synced). */
export function stepHeading(scene: Scene3D, i: number): string {
  const s = scene.steps[i];
  if (!s) return '';
  const head = `Step ${i + 1} of ${scene.steps.length}`;
  return scene.synced && s.t !== null ? `${head} · ${formatTime(s.t)}` : head;
}

/** A chip's short caption: the first words, at most max characters. */
export function chipCaption(caption: string, max = 34): string {
  return text(caption, max);
}

/** The caption bar before the first step. */
export function beforeLine(scene: Scene3D): string {
  if (!scene.steps.length) return 'Drag to turn the drawing.';
  if (scene.synced && scene.steps[0].t !== null) return `The explanation starts at ${formatTime(scene.steps[0].t)}.`;
  return 'This drawing is not talked about in the lecture: go through it with Next, or play the steps.';
}

/** Labels that are a plain number the lecture keeps equal to one parameter in every resting state where the label
 *  shows (the start and the end of every step): synth3d-vector3d's "3", "2" and "4" beside the components while vx,
 *  vy and vz are 3, 2 and 4. The stage draws them with the parameter's value, so they stay true while a student
 *  moves the sliders ("3" stayed "3" with vx at 5). A number two parameters equal is left as it is. {object id:
 *  parameter name}. */
export function liveNumberLabels(scene: Scene3D): Record<string, string> {
  const later = new Set(scene.steps.flatMap((s) => s.show));
  let vis = new Set(scene.objects.filter((o) => !later.has(o.id)).map((o) => o.id));
  let params: Record<string, number> = Object.fromEntries(scene.params.map((p) => [p.name, p.value]));
  const states = [{ vis, params }];
  for (const s of scene.steps) {
    params = { ...params };
    for (const { param, to } of s.set) if (param in params) params[param] = to;
    const hide = new Set(s.hide);
    vis = new Set([...vis, ...s.show].filter((id) => !hide.has(id)));
    states.push({ vis, params });
  }
  const out: Record<string, string> = {};
  for (const o of scene.objects) {
    if (!/^\s*[-+]?\d+(?:\.\d+)?\s*$/.test(o.label)) continue;
    const v = Number(o.label);
    const where = states.filter((st) => st.vis.has(o.id));
    if (!where.length) continue;
    const same = scene.params.filter((p) => where.every((st) => Math.abs(st.params[p.name] - v) <= 1e-9 * Math.max(1, Math.abs(v))));
    if (same.length === 1) out[o.id] = same[0].name;
  }
  return out;
}

/** What the camera frames at each step, in order from before the first one (index -1): the objects visible once the
 *  step is done, and the parameters where its change starts and where it ends (the drawing is framed for both, so
 *  nothing leaves the stage while it changes). The stage frames each step on these instead of on every object of
 *  every moment at once (the cylinder filled a fifth of the stage until its ghosts came). */
export function stepFraming(scene: Scene3D): { visible: string[]; from: Record<string, number>; to: Record<string, number> }[] {
  const later = new Set(scene.steps.flatMap((s) => s.show));
  let vis = scene.objects.filter((o) => !later.has(o.id)).map((o) => o.id);
  let params: Record<string, number> = Object.fromEntries(scene.params.map((p) => [p.name, p.value]));
  const out = [{ visible: vis, from: params, to: params }];
  for (const s of scene.steps) {
    const from = params;
    params = { ...params };
    for (const { param, to } of s.set) if (param in params) params[param] = to;
    const hide = new Set(s.hide);
    vis = [...new Set([...vis, ...s.show])].filter((id) => !hide.has(id));
    out.push({ visible: vis, from, to: params });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// "Try it": only shapes that can exist

/** Degrees an angle may measure off the parameter it is named after (app/popout.py ANGLE_TOL). */
export const ANGLE_TOL = 0.5;
/** Three edges from one corner whose unit directions' determinant, squared, is under this lie nearly in one plane:
 *  a flat plate, past which no cell has those angles (app/popout.py FLAT_MIN). */
export const FLAT_MIN = 0.45;

const GREEK: Record<string, string> = {
  α: 'alpha', β: 'beta', γ: 'gamma', δ: 'delta', θ: 'theta', φ: 'phi', ψ: 'psi', ω: 'omega', λ: 'lambda', μ: 'mu',
  σ: 'sigma', ρ: 'rho', τ: 'tau', ς: 'sigma', ϕ: 'phi', ϑ: 'theta', µ: 'mu',
};
const GREEK_NAMES = new Set(Object.values(GREEK));

/** Lower case, Greek letters as their names, in words (app/popout.py plain_words). */
export function plainWords(s: string): string[] {
  return [...s.toLowerCase()].map((c) => (GREEK[c] ? ` ${GREEK[c]} ` : c)).join('').split(/\s+/).filter(Boolean);
}

/** The check a scene's angles get while a student moves them (`ok`), and the parameters it reads. */
export interface ShapeGuard {
  /** The angle parameters it guards ("deg" parameters an angle object is named after). */
  params: string[];
  /** Whether these values (every parameter; a missing one at its starting value) make a shape that exists. */
  ok(values: Readonly<Record<string, number>>): boolean;
}

/**
 * The server's check of a cell's angles (app/popout.py narrow_angle_ranges), for any values: every angle object
 * named after an angle parameter ("beta" for param beta) measures that parameter's value within ANGLE_TOL, and three
 * edges from one corner never lie nearly in one plane (FLAT_MIN). The server keeps each slider's range to what that
 * angle can take with the others at the lecture's values; this keeps every mix of them a shape that exists. Null
 * when no angle object is named after an angle parameter (nothing to guard).
 */
export function shapeGuard(scene: Scene3D): ShapeGuard | null {
  const deg = scene.params.filter((p) => p.unit === 'deg');
  if (!deg.length) return null;
  const names = [...scene.params.map((p) => p.name), ...(scene.defs ?? []).map((d) => d.name)];
  const defs = (scene.defs ?? []).map((d) => tryCompileVars(d.expr, names)?.fn ?? null);
  if (defs.some((f) => !f)) return null;
  const pairs: { pts: VarsFn[][]; radius: VarsFn | null; param: string; at: number }[] = [];
  for (const o of scene.objects) {
    if (o.type !== 'angle' || o.points.length !== 3) continue;
    const greek = plainWords(o.label).filter((w) => GREEK_NAMES.has(w));
    if (!greek.length) continue;
    const p = deg.find((q) => {
      const words = new Set([...q.name.split(/[\s_]+/).flatMap(plainWords), ...plainWords(q.label)]);
      return greek.some((w) => words.has(w));
    });
    if (!p) continue;
    const pts = o.points.map((pt) => pt.map((s) => tryCompileVars(s, names)?.fn ?? null));
    const radius = o.size.length ? (tryCompileVars(o.size[0], names)?.fn ?? null) : null;
    if (pts.some((pt) => pt.some((f) => !f)) || (o.size.length && !radius)) continue;
    pairs.push({ pts: pts as VarsFn[][], radius, param: p.name, at: names.indexOf(p.name) });
  }
  if (!pairs.length) return null;
  const start = scene.params.map((p) => p.value);
  const np = scene.params.length;
  const buf = new Float64Array(names.length);
  const ok = (values: Readonly<Record<string, number>>): boolean => {
    scene.params.forEach((p, i) => {
      const v = values[p.name];
      buf[i] = typeof v === 'number' && Number.isFinite(v) ? v : start[i];
    });
    defs.forEach((f, k) => (buf[np + k] = f!(buf)));
    const corners = new Map<string, number[][]>();
    for (const { pts, radius, param } of pairs) {
      const [v, a, b] = pts.map((pt) => pt.map((f) => f(buf)) as Vec3);
      const deg = measureAngle(v, a, b);
      if (deg === null || !(Math.abs(deg - buf[names.indexOf(param)]) <= ANGLE_TOL)) return false;
      if (radius) {
        const r = radius(buf);
        if (!(Number.isFinite(r) && r > 0)) return false;
      }
      const key = v.map((x) => Math.round(x * 1e6) / 1e6).join(',');
      let dirs = corners.get(key);
      if (!dirs) corners.set(key, (dirs = []));
      for (const p of [a, b]) {
        const d = [p[0] - v[0], p[1] - v[1], p[2] - v[2]];
        const n = Math.hypot(d[0], d[1], d[2]);
        if (!(n > 1e-9)) continue;
        const u = d.map((x) => x / n);
        if (dirs.every((w) => u[0] * w[0] + u[1] * w[1] + u[2] * w[2] < 1 - 1e-9)) dirs.push(u);
      }
    }
    for (const d of corners.values()) {
      if (d.length !== 3) continue;
      const [a, b, c] = d;
      const det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
      if (det * det < FLAT_MIN) return false;
    }
    return true;
  };
  return { params: [...new Set(pairs.map((p) => p.param))], ok };
}

const SEARCH = 24; // halvings: under a millionth of the way

/**
 * The value parameter `name` takes when the viewer asks for `wanted` with every other value as it is now
 * (`values`): `wanted` when it makes a shape that exists, else the farthest point towards it from the value now that
 * does (the slider stops at the wall). When the values now make no shape already, `wanted` (nothing to keep).
 */
export function clampToShape(guard: ShapeGuard | null, values: Readonly<Record<string, number>>, name: string, wanted: number): number {
  if (!guard) return wanted;
  const at = (v: number) => guard.ok({ ...values, [name]: v });
  const from = values[name];
  if (at(wanted) || typeof from !== 'number' || !Number.isFinite(from) || !at(from)) return wanted;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < SEARCH; i++) {
    const m = (lo + hi) / 2;
    if (at(from + (wanted - from) * m)) lo = m;
    else hi = m;
  }
  return from + (wanted - from) * lo;
}

/**
 * The viewer's values (`own`) on top of the lecture's of the moment, applied as far as they make a shape that
 * exists. Only the angles the guard checks give way: each moves from the lecture's value towards the viewer's by
 * the same share, the largest that does (all of it, mostly); anything else they set (a length) stays as they set it
 * (test2 #7: c set to 6 read 5.42 once the hexagon step took gamma to 120). When the lecture's angles make no shape
 * with the viewer's other values (mid-change), the viewer's are applied as they are.
 */
export function fitToShape(
  guard: ShapeGuard | null,
  lecture: Readonly<Record<string, number>>,
  own: Readonly<Record<string, number>>,
): Record<string, number> {
  const all: Record<string, number> = { ...lecture, ...own };
  if (!guard || guard.ok(all)) return all;
  const angles = guard.params.filter((k) => k in own && k in lecture);
  const base = { ...all };
  for (const k of angles) base[k] = lecture[k];
  if (!angles.length || !guard.ok(base)) return all;
  const mix = (f: number) => {
    const out = { ...base };
    for (const k of angles) out[k] = lecture[k] + (own[k] - lecture[k]) * f;
    return out;
  };
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < SEARCH; i++) {
    const m = (lo + hi) / 2;
    if (guard.ok(mix(m))) lo = m;
    else hi = m;
  }
  return mix(lo);
}

/**
 * Every value as the view last drew it (`frame.params`), for tryValue, when that frame was drawn with the viewer's
 * values as they are now (`frame.own` is `own`, the same object); else those values over it, as the viewer set them.
 * Two sliders moved before the next frame (End on alpha, then on beta) put alpha back to the 90 last drawn.
 */
export function drawnWith(
  frame: { params: Readonly<Record<string, number>>; own: Readonly<Record<string, number>> },
  own: Readonly<Record<string, number>>,
): Record<string, number> {
  return frame.own === own ? { ...frame.params } : { ...frame.params, ...own };
}

/**
 * The viewer's values once they move slider `name` towards `wanted` ("Try it"), from their values so far (`own`)
 * and every value as last drawn (`drawn`):
 * - their angles that gave way to the lecture (fitToShape) are kept where they are shown, so moving one slider never
 *   moves another they did not touch (test2 #7: with alpha shown at 71.6, moving beta to 75 drew beta 77.9 and sent
 *   alpha to 65.7, back towards the 60 they had set before the hexagon step), and the one moved starts from there;
 * - the value stops where the shape would stop existing with the others as they are (clampToShape), on the slider's
 *   own steps (`grid`: its min and step) on the side where the shape still exists, so the thumb, the value shown and
 *   the label agree (it stopped at 85.1 with the thumb at 85).
 */
export function tryValue(
  guard: ShapeGuard | null,
  own: Readonly<Record<string, number>>,
  drawn: Readonly<Record<string, number>>,
  name: string,
  wanted: number,
  grid?: { min: number; step: number },
): Record<string, number> {
  const out: Record<string, number> = { ...own };
  for (const k of guard?.params ?? []) {
    const d = drawn[k];
    if (k in own && typeof d === 'number' && Number.isFinite(d)) out[k] = d;
  }
  const now = { ...drawn, ...out };
  let v = clampToShape(guard, now, name, wanted);
  const from = now[name];
  if (v !== wanted && guard && grid && grid.step > 0 && typeof from === 'number' && Number.isFinite(from)) {
    const n = (v - grid.min) / grid.step;
    const up = wanted > from;
    const r = grid.min + (up ? Math.floor(n + 1e-9) : Math.ceil(n - 1e-9)) * grid.step;
    const on = Math.round(r * 1e9) / 1e9;
    if ((up ? on >= from && on <= v : on <= from && on >= v) && guard.ok({ ...now, [name]: on })) v = on;
  }
  out[name] = v;
  return out;
}
