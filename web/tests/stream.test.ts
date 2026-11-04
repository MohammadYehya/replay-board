// Streaming replay: how far a partial replay reaches, when the player fetches a new version and takes it, the sprite
// cache key, the gates and the plain lines it shows.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  accepts, boardTime, catchUpText, failureOf, gateTitle, GATE_TITLE, GATE_TITLE_FAILED, GATE_TITLE_INTERRUPTED, heldOf,
  isPartial, jobEnded, knowsUpTo, pastReady, planPoll, readyFraction, readyUntil, replaySeq, reuseUnchanged,
  shouldWatch, spriteKey, stopAfterTaking, streamLine,
} from '../src/player/stream.ts';
import type { JobStatus, Piece, Replay } from '../src/types.ts';

const replay = (over: Partial<Replay> = {}): Replay => ({
  version: 1,
  video: { file: 'lecture.mp4', duration: 480, width: 1280, height: 720, start: 0, end: 480 },
  type: 'board',
  board: { size: [800, 314], pieces: [] },
  transcript: { turns: [], words: [], timing: 'proportional' },
  chapters: [],
  ...over,
});
const partial = (seq: number, ready_until: number, over: Partial<Replay> = {}) => replay({ partial: { seq, ready_until }, ...over });
const final = (timing = 'mms') => replay({ transcript: { turns: [], words: [], timing } });
const status = (over: Partial<JobStatus> = {}): JobStatus => ({ id: 'j', state: 'running', ...over });

test('a final replay is not partial and has no ready time', () => {
  assert.equal(isPartial(final()), false);
  assert.equal(readyUntil(final()), null);
  assert.equal(replaySeq(final()), null);
});

test('a partial replay is ready up to its ready_until, kept inside the clip', () => {
  assert.equal(readyUntil(partial(3, 200)), 200);
  assert.equal(replaySeq(partial(3, 200)), 3);
  const clip = { file: 'lecture.mp4', duration: 480, width: 1, height: 1, start: 380, end: 478 };
  assert.equal(readyUntil(partial(1, 0, { video: clip })), 380); // the first partial: ready up to the clip start
  assert.equal(readyUntil(partial(1, 999, { video: clip })), 478);
  // no usable time: ready up to the clip start
  assert.equal(readyUntil(replay({ partial: { seq: 1, ready_until: NaN } })), 0);
});

test('the ready fraction is the share of the clip that is ready', () => {
  assert.equal(readyFraction(null, 0, 480), 1);
  assert.equal(readyFraction(120, 0, 480), 0.25);
  assert.equal(readyFraction(380, 380, 478), 0);
  assert.equal(readyFraction(429, 380, 478), 0.5);
  assert.equal(readyFraction(-5, 0, 480), 0);
  assert.equal(readyFraction(900, 0, 480), 1);
  assert.equal(readyFraction(10, 5, 5), 1); // an empty clip
});

test('the board holds at ready_until once the playhead is past it', () => {
  assert.equal(pastReady(100, null), false);
  assert.equal(pastReady(100, 120), false);
  assert.equal(pastReady(120.1, 120), false); // a seek to exactly there
  assert.equal(pastReady(121, 120), true);
  assert.equal(boardTime(100, 120), 100);
  assert.equal(boardTime(300, 120), 120);
  assert.equal(boardTime(300, null), 300);
});

test('the sprite key changes with any of the piece files and with the ink, not with timing', () => {
  const p = { id: 4, image: 'stream/aa.png', order: 'stream/bb.png', gone: null } as Piece;
  const k = spriteKey(p, true);
  assert.equal(spriteKey({ ...p, t_seen: 99 } as Piece, true), k);
  assert.notEqual(spriteKey({ ...p, image: 'stream/cc.png' }, true), k);
  assert.notEqual(spriteKey({ ...p, order: 'stream/dd.png' }, true), k);
  assert.notEqual(spriteKey({ ...p, gone: 'stream/ee.png' }, true), k);
  assert.notEqual(spriteKey(p, false), k);
  assert.notEqual(spriteKey({ ...p, id: 5 }, true), k);
  assert.notEqual(spriteKey({ ...p, image: 'pieces/004.png' }, true), k); // the final replay's own paths
});

test('the player watches a running job whose replay can still change', () => {
  assert.equal(shouldWatch(partial(1, 0), true), true);
  assert.equal(shouldWatch(partial(1, 0), undefined), true);
  assert.equal(shouldWatch(final('proportional'), true), true);
  assert.equal(shouldWatch(final('mms'), true), false);
  assert.equal(shouldWatch(partial(1, 0), false), false); // a finished job keeps what it has
  assert.equal(shouldWatch(final('proportional'), false), false);
});

test('with a seq in the status, the replay is fetched when the seq changes', () => {
  const held = heldOf(partial(3, 100));
  assert.deepEqual(held, { seq: 3, partial: true, proportional: true });
  assert.deepEqual(planPoll(held, status({ replay: { seq: 3, partial: true, ready_until: 100 } })), {
    fetch: false, stop: false, bySeq: true,
  });
  assert.deepEqual(planPoll(held, status({ replay: { seq: 4, partial: true, ready_until: 130 } })), {
    fetch: true, stop: false, bySeq: true,
  });
  // a replay JobPage loaded before the player had a status: fetched once to learn its seq
  assert.equal(planPoll(heldOf(final('proportional'), null), status({ replay: { seq: 9, partial: false, ready_until: null } })).fetch, true);
});

test('watching stops once the job ended and nothing changed, or the replay is final', () => {
  const ended = status({ state: 'done', replay: { seq: 7, partial: false, ready_until: null } });
  assert.deepEqual(planPoll(heldOf(final('mms'), 7), ended), { fetch: false, stop: true, bySeq: true });
  assert.deepEqual(planPoll(heldOf(final('mms'), 6), ended), { fetch: true, stop: false, bySeq: true });
  // final word times while the job still finishes: nothing more comes
  const running = status({ replay: { seq: 7, partial: false, ready_until: null } });
  assert.equal(planPoll(heldOf(final('mms'), 7), running).stop, true);
  // proportional word times while it runs: the rewrite with refined ones is still to come
  assert.equal(planPoll(heldOf(final('proportional'), 7), running).stop, false);
  // a job that failed mid-capture keeps its last partial replay
  const failed = status({ state: 'failed', replay: { seq: 5, partial: true, ready_until: 90 } });
  assert.equal(planPoll(heldOf(partial(5, 90), 5), failed).stop, true);
  assert.equal(jobEnded(failed), true);
  assert.equal(jobEnded(status({ state: 'queued' })), false);
});

test('a job that failed and had its partial replay taken away is not fetched again', () => {
  // The backend takes a partial replay.json away when the job fails (and the replay key out of the status): a
  // fetch answers 404, and the player fetched it every 2.5 s for good.
  const held = heldOf(partial(10, 112), 10);
  const failed = status({ state: 'failed', error: 'ran out of memory' });
  assert.deepEqual(planPoll(held, failed), { fetch: false, stop: true, bySeq: false });
  assert.deepEqual(planPoll(held, status({ state: 'done' })), { fetch: false, stop: true, bySeq: false });
  // still running: keep watching (the status may just not have its replay key yet)
  assert.equal(planPoll(held, status()).stop, false);
  // a final replay held with an older server is unchanged: fetched once the job ended
  assert.equal(planPoll(heldOf(final('proportional'), null), failed).fetch, true);
});

test('an older server without a seq: fetched once the aligner is done or the job ended', () => {
  const held = heldOf(final('proportional'), null);
  assert.equal(planPoll(held, status({ stages: { align: { state: 'running' } } })).fetch, false);
  assert.equal(planPoll(held, status({ stages: { align: { state: 'done' } } })).fetch, true);
  assert.equal(planPoll(held, status({ stages: { align: { state: 'skipped' } } })).fetch, true);
  assert.equal(planPoll(held, status({ state: 'done' })).fetch, true);
  assert.equal(planPoll(held, status({ state: 'done' })).bySeq, false);
  // the aligner finishes a moment before the file is rewritten: the old file is not taken
  assert.equal(accepts(final('proportional'), final('proportional'), { bySeq: false, ended: false }), false);
  assert.equal(accepts(final('proportional'), final('mms'), { bySeq: false, ended: false }), true);
  assert.equal(accepts(final('proportional'), final('proportional'), { bySeq: false, ended: true }), true);
  assert.equal(stopAfterTaking({ fetch: true, stop: false, bySeq: false }, final('mms')), true);
  assert.equal(stopAfterTaking({ fetch: true, stop: false, bySeq: true }, final('mms')), false);
});

test('a newer partial, the final replay and its rewrite are taken; an older partial never is', () => {
  const by = { bySeq: true, ended: false };
  assert.equal(accepts(partial(3, 100), partial(4, 130), by), true);
  assert.equal(accepts(partial(4, 130), partial(4, 130), by), false); // the same version
  assert.equal(accepts(partial(4, 130), partial(3, 100), by), false);
  assert.equal(accepts(partial(4, 130), final('proportional'), by), true);
  assert.equal(accepts(final('proportional'), final('mms'), by), true);
  assert.equal(accepts(final('proportional'), partial(9, 400), by), false); // a late partial after the final one
  assert.equal(accepts(final('mms'), partial(9, 400), { bySeq: false, ended: true }), false);
});

test('unchanged transcript and chapters are kept from the replay held', () => {
  const words = [{ w: 'hello', s: 1, e: 1.4 }];
  const a = partial(1, 10, { transcript: { turns: [], words, timing: 'proportional' } });
  const b = partial(2, 40, { transcript: { turns: [], words: [{ ...words[0] }], timing: 'proportional' } });
  const out = reuseUnchanged(a, b);
  assert.equal(out.transcript, a.transcript);
  assert.equal(out.chapters, a.chapters);
  assert.equal(out.partial, b.partial);
  const c = final('mms');
  c.chapters = [{ t: 0, title: 'Start' }];
  const out2 = reuseUnchanged(a, c);
  assert.equal(out2.transcript, c.transcript);
  assert.equal(out2.chapters, c.chapters);
});

// The job's own lines (app/pipeline.py plain_error and INTERRUPTED).
const KEY_ERROR = 'something went wrong (KeyError: 53); details are in error.log';
const INTERRUPTED = 'processing was interrupted before it finished; process the video again';
/** test3.mp4 (7:52): capture crashed at 4:46 with the board ready up to 2:48, the voice transcribed. */
const crashed = (error: string | null = KEY_ERROR, over: Partial<JobStatus> = {}) =>
  status({
    state: 'failed',
    error,
    stages: {
      speech: { state: 'done', progress: 1 },
      capture: { state: 'running', progress: 0.607 },
      read: { state: 'pending' },
    },
    ...over,
  });

test('Notes, Slides and 3D wait for the final replay, and say when it will not come', () => {
  assert.equal(gateTitle(false), null);
  assert.equal(gateTitle(true), GATE_TITLE);
  assert.equal(GATE_TITLE, 'Available when processing finishes');
  assert.equal(gateTitle(true, failureOf(crashed(), 0, 472)), GATE_TITLE_FAILED);
  assert.equal(GATE_TITLE_FAILED, 'Not available: processing failed before the end');
  assert.equal(gateTitle(true, failureOf(crashed(INTERRUPTED), 0, 472)), GATE_TITLE_INTERRUPTED);
  assert.equal(GATE_TITLE_INTERRUPTED, 'Not available: processing was interrupted before the end');
  assert.equal(gateTitle(false, failureOf(crashed(), 0, 472)), null); // a final replay: nothing waits
});

test('a failed job: its own line, how far capture got, and nothing while it runs or once it is done', () => {
  assert.equal(failureOf(null, 0, 472), null);
  assert.equal(failureOf(status(), 0, 472), null);
  assert.equal(failureOf(status({ state: 'done' }), 0, 472), null);
  const f = failureOf(crashed(), 0, 472)!;
  assert.equal(f.interrupted, false);
  assert.equal(f.cause, 'something went wrong (KeyError: 53)'); // where the details are: in the tooltip
  assert.equal(f.error, KEY_ERROR);
  assert.ok(Math.abs(f.at! - 286.5) < 0.01); // capture's progress over the clip, on the video clock
  assert.equal(f.doing, null);
  const i = failureOf(crashed(INTERRUPTED), 0, 472)!;
  assert.equal(i.interrupted, true);
  assert.equal(i.cause, ''); // said in the player's own words
  assert.equal(i.error, INTERRUPTED);
  // a clip: capture's progress is over the clip
  assert.ok(Math.abs(failureOf(crashed(), 380, 478)!.at! - (380 + 0.607 * 98)) < 0.01);
  // capture done: the stage still running says where it stopped
  const read = crashed('boom', {
    stages: { capture: { state: 'done', progress: 1 }, read: { state: 'running', progress: 0.3 } },
  });
  assert.deepEqual(failureOf(read, 0, 472), {
    interrupted: false, cause: 'boom', error: 'boom', at: null, doing: 'reading the writing',
  });
  // capture running without progress yet
  const early = crashed('boom', { stages: { capture: { state: 'running', progress: 0 } } });
  assert.equal(failureOf(early, 0, 472)!.doing, 'capturing the writing');
  // no error, no stages
  assert.deepEqual(failureOf(status({ state: 'failed' }), 0, 472), {
    interrupted: false, cause: '', error: '', at: null, doing: null,
  });
  // a long line is cut (the whole one is the tooltip); one on several lines is one line
  const long = failureOf(crashed(`chat failed: ${'x'.repeat(400)}`), 0, 472)!;
  assert.equal(long.cause.length, 160);
  assert.ok(long.cause.endsWith('…'));
  assert.equal(long.error.length, 413);
  assert.equal(failureOf(crashed('first line\n  second line.'), 0, 472)!.cause, 'first line second line');
});

test('the processing line names the stage the board waits for and how much is ready', () => {
  const st = status({ stages: { speech: { state: 'running' }, capture: { state: 'running' }, people: { state: 'done' } } });
  assert.equal(streamLine(st, 200, 0, 480), 'Processing: capturing the writing, 3:20 of 8:00 ready');
  assert.equal(streamLine(status({ stages: { read: { state: 'running' } } }), 470, 0, 480), 'Processing: reading the writing, 7:50 of 8:00 ready');
  assert.equal(streamLine(status(), 0, 0, 480), 'Processing: 0:00 of 8:00 ready');
  assert.equal(streamLine(null, 30, 0, 480), 'Processing: 0:30 of 8:00 ready');
  // a clip: the times are within the clip
  assert.equal(streamLine(status(), 410, 380, 478), 'Processing: 0:30 of 1:38 ready');
  assert.equal(streamLine(status({ state: 'done' }), 480, 0, 480), 'Processing: finishing, 8:00 of 8:00 ready');
  // the final replay is on disk while the player still loads its board images (the aligner runs on): not "refining"
  const written = status({ stages: { align: { state: 'running' } }, replay: { seq: 20, partial: false, ready_until: null } });
  assert.equal(streamLine(written, 470, 0, 480), 'Processing done: loading the finished board');
});

test('once the job failed, the line under the video says so, why, how far the board goes and what to do', () => {
  // what the owner saw on test3.mp4
  assert.equal(
    streamLine(crashed(), 168, 0, 472),
    'Processing failed at 4:46 of the video: something went wrong (KeyError: 53). ' +
      'The board stops at 2:48; process the video again.',
  );
  assert.equal(
    streamLine(crashed(INTERRUPTED), 168, 0, 472),
    'Processing was interrupted at 4:46 of the video. The board stops at 2:48; process the video again.',
  );
  // nothing known but the state
  const bare = 'Processing failed. The board stops at 1:30; process the video again.';
  assert.equal(streamLine(status({ state: 'failed' }), 90, 0, 480), bare);
  assert.equal(streamLine(status({ state: 'failed', error: null }), 90, 0, 480), bare);
  // the job's line says what to do already
  assert.equal(
    streamLine(crashed('ran out of memory; try a shorter part of the video with --start/--end'), 90, 0, 472),
    'Processing failed at 4:46 of the video: ran out of memory; try a shorter part of the video with --start/--end. ' +
      'The board stops at 1:30.',
  );
  // after capture: the stage it stopped in, and a board that reaches the end but is not finished
  const read = crashed('chat (muse-spark) failed after 3 tries: HTTP 500', {
    stages: { capture: { state: 'done', progress: 1 }, read: { state: 'running', progress: 0.3 } },
  });
  assert.equal(
    streamLine(read, 472, 0, 472),
    'Processing failed while reading the writing: chat (muse-spark) failed after 3 tries: HTTP 500. ' +
      'The board reaches the end but is not finished; process the video again.',
  );
  // failed before any of the board was ready
  assert.equal(
    streamLine(crashed('boom.', { stages: { capture: { state: 'running', progress: 0.01 } } }), 0, 0, 480),
    'Processing failed at 0:04 of the video: boom. None of the board was ready; process the video again.',
  );
  // a clip: times on the video clock
  assert.equal(
    streamLine(crashed(null, { stages: { capture: { state: 'running', progress: 0.5 } } }), 400, 380, 478),
    'Processing failed at 7:09 of the video. The board stops at 6:40; process the video again.',
  );
  // capture is never said to be behind the board
  assert.equal(
    streamLine(crashed(null, { stages: { capture: { state: 'running', progress: 0.1 } } }), 90, 0, 480),
    'Processing failed at 1:30 of the video. The board stops at 1:30; process the video again.',
  );
});

test('the board and Ask Muse say how far they reach', () => {
  assert.equal(catchUpText(200), 'Board catching up: ready up to 3:20');
  assert.equal(catchUpText(200, null), 'Board catching up: ready up to 3:20');
  assert.equal(catchUpText(168, failureOf(crashed(), 0, 472)), 'Processing failed: the board stops at 2:48');
  assert.equal(
    catchUpText(168, failureOf(crashed(INTERRUPTED), 0, 472)),
    'Processing was interrupted: the board stops at 2:48',
  );
  assert.equal(catchUpText(380.5, failureOf(crashed(), 380, 478), 380), 'Processing failed: none of the board was ready');
  assert.equal(knowsUpTo(200), 'Knows the lecture up to 3:20');
});
