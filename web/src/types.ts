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

export interface BoardData {
  size: [number, number];
  mode?: string;
  corners?: number[][];
  background?: string;
  pieces: Piece[];
  erases?: Erase[];
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
  tool: string; // cursor | laser
  t_start: number;
  t_end: number;
  target?: string[];
  box: Box;
  from?: [number, number] | null;
  to?: [number, number] | null;
  why?: string;
  guess?: string;
  state?: number; // id of the slide state it belongs to
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
}

// 3D pop-out scene made by Muse Spark for one board piece (app/popout.py, SCENE_SCHEMA and clean_scene).
// Expressions are in x, y, t and are evaluated by player/expr.ts. Every field is checked again before use.
export interface PopoutAxis {
  min: number;
  max: number;
  label?: string;
}

export interface PopoutScene {
  kind: 'plot2d' | 'surface' | 'curve3d' | 'solids' | 'none' | string;
  title?: string;
  explain?: string;
  x?: PopoutAxis;
  y?: PopoutAxis;
  z?: PopoutAxis;
  /** plot2d: y = f(x); surface: z = f(x, y). */
  functions?: { expr: string; color?: string; label?: string }[];
  /** curve3d: x(t), y(t), z(t). */
  curves?: { x: string; y: string; z: string; t_min?: number; t_max?: number; color?: string; label?: string }[];
  solids?: {
    shape: string; // box | sphere | cylinder | cone | torus | plane | arrow
    center: number[];
    size: number[];
    rotation?: number[]; // degrees about x, y, z
    color?: string;
    label?: string;
  }[];
  points?: { at: number[]; label?: string }[];
}
