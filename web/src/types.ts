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
  slides?: unknown;
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
