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
import { GRAMMAR_NAMES, tryCompileVars } from './expr.ts';
import { formatTime } from './timing.ts';

export const OBJECT_TYPES: readonly Scene3DObjectType[] = [
  'point', 'segment', 'vector', 'polygon', 'box', 'sphere', 'cylinder', 'cone', 'torus', 'graph', 'surface', 'curve',
  'psurface', 'label',
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
  paramLabel: 24,
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
}

export type Camera = Exclude<CameraPreset, 'keep'>;

export interface SceneState {
  /** The current step (-1: before the first; the base drawing). */
  index: number;
  /** How far the current step's change has gone, 0..1 (1 when it has none or it is done). */
  progress: number;
  /** Every parameter's value. */
  params: Record<string, number>;
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
  const params: Record<string, number> = {};
  for (const [name, seg] of pseg) {
    params[name] = segAt(seg, clock);
    moving ||= segMoving(seg, clock);
  }
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
