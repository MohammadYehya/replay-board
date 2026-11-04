// Streaming replay (SPEC.md "Streaming replay"): while a board lecture is processed, replay.json is rewritten with
// the writing captured so far. Such a partial replay has a `partial` key {seq, ready_until}; the final one has none.
// The player watches the job (status.json's `replay` {seq, partial, ready_until}), fetches each new version and swaps it
// in without blanking the board. Pure rules for that, no DOM, so `npm test` runs them with plain Node.
import type { JobStatus, Piece, Replay } from '../types';
import { formatTime } from './timing.ts';

/** How often the player looks at a running job, ms. */
export const WATCH_MS = 2500;
/** A swap waits at most this long for the new version's board images (then the late ones pop in as they load). */
export const PRELOAD_MS = 30000;
/** The playhead is "past what is ready" only this far past ready_until (a seek to exactly there is not). */
const SLACK = 0.25;

export const GATE_TITLE = 'Available when processing finishes';
export const GATE_TITLE_STOPPED = 'Not available: processing stopped before the end';

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Whether the job has stopped running (done or failed): nothing more will be written. */
export const jobEnded = (st: Pick<JobStatus, 'state'>) => st.state !== 'running' && st.state !== 'queued';

/** Whether a replay is partial: the board covers only the part up to ready_until. */
export const isPartial = (r: Replay) => !!r.partial && typeof r.partial === 'object';

/** The version number a partial replay carries (null for the final one, or when it carries none). */
export const replaySeq = (r: Replay): number | null => (isPartial(r) ? num(r.partial!.seq) : null);

/** How far a partial replay's board reaches (seconds on the video clock, kept inside the clip), or null when the
 *  replay is final. A partial replay without a usable time counts as ready up to the clip start. */
export function readyUntil(r: Replay): number | null {
  if (!isPartial(r)) return null;
  const start = num(r.video?.start) ?? 0;
  const end = num(r.video?.end) ?? num(r.video?.duration) ?? Infinity;
  const t = num(r.partial!.ready_until) ?? start;
  return Math.min(Math.max(t, start), Math.max(start, end));
}

/** Share of the clip the board is ready for, 0..1 (1 when the replay is final). */
export function readyFraction(ready: number | null, clipStart: number, clipEnd: number): number {
  if (ready == null) return 1;
  const len = clipEnd - clipStart;
  if (!(len > 0)) return 1;
  const f = (ready - clipStart) / len;
  return f <= 0 ? 0 : f >= 1 ? 1 : f;
}

/** Whether the playhead is past what the board is ready for. */
export const pastReady = (t: number, ready: number | null) => ready != null && t > ready + SLACK;

/** The time the board is drawn at: the playhead, held at ready_until past it (frozen there, never blank). */
export const boardTime = (t: number, ready: number | null) => (ready != null && t > ready ? ready : t);

/** Key of a piece's decoded sprite in the cache shared by every board renderer of a lecture: its image, order and
 *  gone maps (a path always means the same bytes: partial replays name images by their content) and the board's
 *  light or dark ink. The id is in it because the frame cache keys partial frames by it. */
export const spriteKey = (p: Pick<Piece, 'id' | 'image' | 'order' | 'gone'>, light: boolean) =>
  `${p.id}|${p.image}|${p.order ?? ''}|${p.gone ?? ''}|${light ? 'light' : 'dark'}`;

/** What the player knows about the replay it holds. seq: the status seq it was fetched at (or the partial's own
 *  seq), null when unknown (a replay JobPage loaded before the player had a status of its own). */
export interface Held {
  seq: number | null;
  partial: boolean;
  /** Its word times are still the proportional first guess (the aligner may rewrite them). */
  proportional: boolean;
}

export const heldOf = (r: Replay, seq: number | null = replaySeq(r)): Held => ({
  seq,
  partial: isPartial(r),
  proportional: r.transcript?.timing === 'proportional',
});

/** Whether to watch the job at all: it runs and its replay can still change. */
export const shouldWatch = (r: Replay, running: boolean | undefined) =>
  running !== false && (isPartial(r) || r.transcript?.timing === 'proportional');

export interface PollPlan {
  /** Fetch replay.json now. */
  fetch: boolean;
  /** Stop watching (nothing more will change). */
  stop: boolean;
  /** The status says which version is on disk (an older server does not). */
  bySeq: boolean;
}

/** What to do after one look at the job's status. A server with `replay` in its status says when replay.json
 *  changed (seq); an older one only says when the aligner is done, which is when its one rewrite comes. */
export function planPoll(held: Held, st: JobStatus): PollPlan {
  const ended = jobEnded(st);
  const seq = num(st.replay?.seq);
  if (seq != null) {
    const changed = seq !== held.seq;
    const final = !held.partial && !held.proportional;
    return { fetch: changed, stop: !changed && (ended || final), bySeq: true };
  }
  // A partial replay held, and a status without a seq once the job ended: the job failed and the server took the
  // partial replay away (there is nothing to fetch: it answers 404). The player keeps the board it has.
  if (held.partial && ended) return { fetch: false, stop: true, bySeq: false };
  const align = st.stages?.align?.state;
  return { fetch: ended || align === 'done' || align === 'skipped', stop: false, bySeq: false };
}

/** Whether a fetched replay replaces the one held. A partial replay never replaces the final one, nor a later
 *  partial; with an older server, the rewrite is the one with refined word times (or whatever is there once the job
 *  ended). */
export function accepts(held: Replay, next: Replay, opts: { bySeq: boolean; ended: boolean }): boolean {
  if (isPartial(next)) {
    if (!isPartial(held)) return false;
    const a = replaySeq(held);
    const b = replaySeq(next);
    return a == null || b == null || b > a;
  }
  if (opts.bySeq || isPartial(held)) return true;
  return next.transcript?.timing !== 'proportional' || opts.ended;
}

/** Whether to stop watching after a fetched replay was taken (an older server rewrites once). */
export const stopAfterTaking = (plan: PollPlan, next: Replay) => !plan.bySeq && !isPartial(next);

/** The fetched replay, with the parts that did not change taken from the one held, so what is built from them (the
 *  transcript's paragraphs, the chapter list) is not built again for every version. */
export function reuseUnchanged(held: Replay, next: Replay): Replay {
  const same = (a: unknown, b: unknown) => a === b || JSON.stringify(a) === JSON.stringify(b);
  const out = { ...next };
  if (same(held.transcript, next.transcript)) out.transcript = held.transcript;
  if (same(held.chapters, next.chapters)) out.chapters = held.chapters;
  return out;
}

/** The title of the Notes, Slides and 3D buttons while they are not available, or null when they are. */
export function gateTitle(partial: boolean, stopped: boolean): string | null {
  if (!partial) return null;
  return stopped ? GATE_TITLE_STOPPED : GATE_TITLE;
}

// The stage the "Processing" line names when several run at once: the one the board is waiting for first.
const DOING: [string, string][] = [
  ['capture', 'capturing the writing'],
  ['read', 'reading the writing'],
  ['gestures', 'finding pointing gestures'],
  ['link', 'linking writing to speech'],
  ['write', 'writing the replay'],
  ['align', 'refining word times'],
  ['people', 'finding the teacher'],
  ['speech', 'transcribing the voice'],
  ['board', 'finding the board'],
  ['classify', 'looking at the video'],
  ['probe', 'reading the video file'],
];

/** The line under the video while the replay is partial: "Processing: capturing the writing, 3:20 of 8:00 ready". */
export function streamLine(st: JobStatus | null, ready: number, clipStart: number, clipEnd: number): string {
  const have = `${formatTime(Math.max(0, ready - clipStart))} of ${formatTime(Math.max(0, clipEnd - clipStart))}`;
  if (st && st.state === 'failed') return `Processing stopped. The board covers ${have}.`;
  // The final replay is written (the status says so) and its board images load before it is shown.
  if (st?.replay && st.replay.partial === false) return 'Processing done: loading the finished board';
  if (st && jobEnded(st)) return `Processing: finishing, ${have} ready`;
  const stages = st?.stages ?? {};
  const doing = DOING.find(([k]) => stages[k]?.state === 'running')?.[1];
  return doing ? `Processing: ${doing}, ${have} ready` : `Processing: ${have} ready`;
}

/** What the board says while the playhead is past what is ready (or nothing is on it yet). */
export const catchUpText = (ready: number, stopped: boolean) =>
  stopped
    ? `Processing stopped: the board is ready up to ${formatTime(ready)}`
    : `Board catching up: ready up to ${formatTime(ready)}`;

/** What Ask Muse knows while the replay is partial. */
export const knowsUpTo = (ready: number) => `Knows the lecture up to ${formatTime(ready)}`;
