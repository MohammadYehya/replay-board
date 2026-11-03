// replay.json (SPEC.md section 6) plus the job-status shapes served by app/server.py.
// Every field the backend might leave out is optional here, and the player fills in defaults.

export type Box = [number, number, number, number]; // x, y, w, h in board pixels

export interface Span {
  start: number;
  end: number;
}

export interface Reading {
  kind: string;
  text: string;
  latex: string;
}

export interface Piece {
  id: number;
  box: Box;
  image: string;
  order?: string;
  /** Optional map of when each pixel leaves the board (partly erased): 24-bit code R<<16|G<<8|B, 0 = never,
   *  else time = (code - 1) / 10 s on the video's clock. */
  gone?: string | null;
  /** [start, end] seconds: the order map's 1..255 is linear in write time over this window. */
  draw?: [number, number] | null;
  t_seen: number;
  t_shown: number;
  written_after?: number;
  revealed?: boolean;
  removed_at?: number | null;
  pre_existing?: boolean;
  read?: Reading;
  spoken?: Span[];
  redraw_at?: number | null;
}

export interface Erase {
  t: number;
  box: Box;
}

/** Pieces written on one line in bursts, read (and typeset) together. */
export interface BoardLine {
  id: string;
  pieces: number[];
  box: Box;
  t_complete: number;
  removed_at?: number | null;
  pre_existing?: boolean;
  read?: Reading;
}

export interface BoardData {
  size: [number, number];
  mode?: string;
  corners?: number[][];
  background?: string;
  pieces: Piece[];
  lines?: BoardLine[];
  erases?: Erase[];
  /** The teacher's hand pointing at writing (tool "hand"), in board pixels; targets are piece ids. */
  annotations?: SlideAnnotation[];
  t_start?: number;
}

export interface SlideElement {
  id: string;
  kind: string;
  text: string;
  latex: string;
  box: Box; // slide image pixels
  spoken?: Span[];
}

export interface SlideState {
  id: number;
  kind: string; // slide | addition
  t_start: number;
  t_end: number;
  /** When the change to this state began (the crossfade runs from t_change to t_start). */
  t_change?: number;
  blank?: boolean;
  image: string | null;
  elements?: SlideElement[];
}

export type AnnotationType = 'spotlight' | 'circle' | 'underline' | 'arrow' | 'pulse';

export interface SlideAnnotation {
  type: AnnotationType | string;
  tool: string; // cursor | laser (slides) | hand (board)
  t_start: number;
  t_end: number;
  target?: string[];
  box: Box;
  from?: [number, number] | null;
  to?: [number, number] | null;
  why?: string;
  guess?: string;
  state?: number; // id of the slide state it belongs to (slides only)
  path?: [number, number][];
  underline_y?: number | null;
}

/** [t, "c" (cursor) | "l" (laser), x, y] in slide image pixels. */
export type PointerSample = [number, string, number, number];

export interface SlidesData {
  size: [number, number];
  background?: string;
  live?: Box[];
  states: SlideState[];
  annotations?: SlideAnnotation[];
  pointer?: PointerSample[];
}

export interface Word {
  w: string;
  s: number;
  e: number;
}

export interface Turn {
  start: number;
  end: number;
  text: string;
}

export interface Chapter {
  t: number;
  title: string;
}

export interface ReplayStats {
  seconds?: number;
  api?: { calls?: number; live?: number; cached?: number; seconds?: number; cost_usd?: number };
  capture?: Record<string, number>;
}

export interface Replay {
  version: number;
  video: { file: string; duration: number; width: number; height: number; start?: number; end?: number };
  type: string;
  subject?: string;
  board?: BoardData;
  slides?: SlidesData;
  transcript?: { turns?: Turn[]; words?: Word[]; timing?: string };
  chapters?: Chapter[];
  stats?: ReplayStats;
}

export interface StageStatus {
  state: string; // pending | running | done | skipped
  progress?: number;
  seconds?: number | null;
  note?: string;
}

export interface JobStatus {
  id: string;
  state: string; // queued | running | done | failed
  error?: string | null;
  ready?: boolean;
  name?: string;
  started?: number;
  seconds?: number;
  stages?: Record<string, StageStatus>;
}

export interface JobSummary {
  id: string;
  state: string;
  error?: string | null;
  ready: boolean;
  name: string;
  /** "Board lecture · 7:58 · lecture.mp4", when the job has a summary. */
  detail?: string;
}

// 3D view made by Muse Spark for one board drawing (app/popout.py), format 2: a scene plus ordered steps, each tied to
// the moment the teacher explains it. Animations are data from a small fixed set; expressions (E) are strings in the
// whitelisted grammar of player/expr.ts, checked on the server and compiled again by the player. The player cleans
// every field before use (player/scene3dState.ts cleanScene): a server older or newer than this one cannot break it.
// Maths axes: z is up.

/** An expression string: numbers, + - * / ^ ( ), whitelisted functions, pi, e, deg, the scene's parameters and defs
 *  (a def: the params and the defs before it only) and the variables of its field (graph expr: x; surface expr: x, y; curve x/y/z: t; psurface x/y/z: u, v). */
export type Expr = string;
export type Vec3E = [Expr, Expr, Expr];
export type Vec3 = [number, number, number];

export interface Scene3DAxis {
  min: number;
  max: number;
  label: string;
}

/** A slider the steps move ("beta" from 90 to 110 degrees); [min, max] is also the range a student may explore
 *  with "Try it". */
export interface Scene3DParam {
  name: string;
  value: number;
  min: number;
  max: number;
  label: string;
  unit: '' | 'deg';
}

/** "cx" = "c*cos(beta*deg)": a value the objects share, worked out again whenever a parameter moves. */
export interface Scene3DDef {
  name: string;
  expr: Expr;
}

export type Scene3DObjectType =
  | 'point'
  | 'segment'
  | 'vector'
  | 'polygon'
  | 'box'
  | 'sphere'
  | 'cylinder'
  | 'cone'
  | 'torus'
  | 'graph'
  | 'surface'
  | 'curve'
  | 'psurface'
  | 'label'
  | 'angle';

/** Flat: every field is present; the unused ones are "" or []. */
export interface Scene3DObject {
  id: string;
  type: Scene3DObjectType;
  /** point, label: 1; segment, vector: 2 (from, to); polygon: 3..12 corners of a flat face; angle: 3 (the vertex,
   *  then a point on each arm: the angle at the vertex between them, 0..180 degrees). */
  points: Vec3E[];
  /** box, sphere, cylinder, cone, torus. */
  center: Vec3E | [];
  /** box [width, depth, height]; sphere [radius]; cylinder, cone [radius, height]; torus [ring radius, tube radius];
   *  angle [] (auto: 0.3 x the shorter arm) or [radius]. */
  size: Expr[];
  /** Degrees about x, y, z ([] = none). */
  rotation: Vec3E | [];
  /** graph: y = f(x) in the plane z = 0; surface: z = f(x, y). */
  expr: Expr;
  /** curve: in t; psurface: in u and v. */
  x: Expr;
  y: Expr;
  z: Expr;
  /** curve: t range; psurface: u range; graph: x range ([] = the x axis range). */
  range: [Expr, Expr] | [];
  /** psurface: v range. */
  range2: [Expr, Expr] | [];
  color: string;
  opacity: number;
  style: 'solid' | 'wire' | 'dashed';
  label: string;
}

export type CameraPreset = 'keep' | 'front' | 'top' | 'side' | 'iso';

export interface Scene3DStep {
  /** "m:ss" on the video clock when the teacher says it, or "" when the drawing is never talked about. */
  at: string;
  /** Seconds on the video clock (from "at", filled in by the server); null when not synced. */
  t: number | null;
  /** Seconds the change takes. */
  over: number;
  caption: string;
  show: string[];
  hide: string[];
  highlight: string[];
  set: { param: string; to: number }[];
  camera: CameraPreset;
}

export interface Scene3D {
  format: 2;
  version: string;
  piece: number;
  /** The board pieces that make up the drawing shown (the asked piece first). */
  pieces: number[];
  title: string;
  explain: string;
  /** True when the steps carry lecture times. */
  synced: boolean;
  axes: { show: boolean; x: Scene3DAxis; y: Scene3DAxis; z: Scene3DAxis };
  params: Scene3DParam[];
  /** Named values worked out from the params, in order (each may use the params and the defs before it), so a long
   *  formula is written once: every object's expressions may use their names. */
  defs: Scene3DDef[];
  objects: Scene3DObject[];
  steps: Scene3DStep[];
  /** [[minx, miny, minz], [maxx, maxy, maxz]] over every state of every step (the server works it out). */
  bounds: [Vec3, Vec3] | null;
  /** The corner the "iso" camera looks from: "left" when the scene's main vector points into the usual front-right
   *  one (the server picks it). */
  isoFrom?: 'left' | 'right';
}

// Ask Muse (app/ask.py): a student's questions about the lecture, answered by Muse Spark with the whole lecture as
// context. Times are on the original video's clock, like everywhere else.

/** Something from the lecture attached to a question. */
export type AskItem =
  | { kind: 'piece'; id: number }
  | { kind: 'line'; id: string }
  | { kind: 'element'; id: string }
  | { kind: 'span'; t0: number; t1: number }
  | { kind: 'moment'; t: number };

/** An attached item with the chip's label: sent for a span (the selected words) and saved with every turn. */
export type AskLabelledItem = AskItem & { label?: string };

/** A board or slide item cited in an answer as [#12], [#L3] or [#s0e1]. */
export interface AskRef {
  ref: string; // "#12"
  kind: 'piece' | 'line' | 'element';
  id: number | string;
  /** Where to seek to see it explained. */
  t: number;
  label: string;
}

export interface AskRequest {
  question: string;
  selection: AskLabelledItem[];
  history: { q: string; a: string }[];
  t: number | null;
}

/** The "done" event of POST /api/jobs/{id}/ask. */
export interface AskDone {
  id: string;
  answer: string;
  items: AskRef[];
  times: number[];
  cost_usd: number | null;
  cached: boolean;
  /** False when the server could not save the turn (it will be gone after a reload). */
  saved: boolean;
}

/** One saved question and answer (GET /api/jobs/{id}/chat). */
export interface AskTurn {
  id: string;
  q: string;
  /** What was attached, with the labels the server saved ("The board at 1:23", the selected words...). */
  selection: AskLabelledItem[];
  answer: string;
  items: AskRef[];
  times: number[];
  t: number | null;
  at?: string | number;
  cost_usd: number | null;
  cached?: boolean;
}

// Slides (app/export.py, GET /api/jobs/{id}/slides.json): a lecture deck written by Muse Spark that covers the whole
// lecture in order, checked by code, previewed here and downloaded as a PowerPoint (app/pptx_deck.py). Format 2.
// Times are on the video's clock. Text is plain, except inline maths ($...$) in bullet, step, table and callout text.

export type DeckKind = 'board' | 'slides';
export type SlideKind = 'title' | 'agenda' | 'section' | 'content' | 'summary' | 'formulas';

export interface BulletItem {
  text: string;
  level: 0 | 1;
}
export interface BulletsBlock {
  type: 'bullets';
  items: BulletItem[];
}
/** One display formula; refs are the board or slide items it comes from ("#12", "#L3", "#s0e1"). */
export interface EquationBlock {
  type: 'equation';
  latex: string;
  label: string;
  refs: string[];
}
/** One step of a worked example or derivation: LaTeX, text or both. */
export interface StepItem {
  latex: string;
  text: string;
}
export interface StepsBlock {
  type: 'steps';
  label: string;
  items: StepItem[];
}
/** A drawing, graph or diagram cut from the board or a slide; image is "export/fig_NNN.png" ('' when unusable). */
export interface FigureBlock {
  type: 'figure';
  refs: string[];
  caption: string;
  image: string;
  size: [number, number] | null;
}
export interface TableBlock {
  type: 'table';
  header: string[];
  rows: string[][];
  caption: string;
}
export type CalloutStyle = 'definition' | 'key' | 'note';
export interface CalloutBlock {
  type: 'callout';
  style: CalloutStyle;
  title: string;
  text: string;
  latex: string;
}
export type DeckBlock = BulletsBlock | EquationBlock | StepsBlock | FigureBlock | TableBlock | CalloutBlock;

export interface DeckSlide {
  id: string; // "s1"
  kind: SlideKind;
  title: string;
  /** The section the slide is in ('' for the title, agenda, summary and formula slides). */
  section: string;
  /** The part of the lecture it covers (null for the title, agenda and formula slides). */
  t0: number | null;
  t1: number | null;
  /** Where "Play" seeks: the slide's first important moment. */
  t: number | null;
  blocks: DeckBlock[];
  /** What a presenter would say: plain text, Unicode maths. */
  notes: string;
  /** What was said during [t0, t1), as plain text. */
  said: string;
}

export interface Deck {
  format: 2;
  /** Changes when replay.json or the deck rules change. */
  version: string;
  kind: DeckKind;
  title: string;
  subtitle: string;
  /** The job's display name. */
  lecture: string;
  /** Seconds of the analysed span. */
  duration: number;
  clip: [number, number];
  slides: DeckSlide[];
  /** Plain lines about parts written without Muse Spark; [] when all went fine. */
  degraded: string[];
}

export type SlidesState = 'none' | 'writing' | 'ready' | 'failed';

/** GET /api/jobs/{id}/slides.json and POST /api/jobs/{id}/slides/write. */
export interface SlidesStatus {
  state: SlidesState;
  /** A plain line: "Planning the slides", "Writing part 2 of 5". */
  step: string;
  done: number;
  total: number;
  error: string | null;
  /** The deck when ready; while writing, the slides finished so far (in order) or null. */
  deck: Deck | null;
}
