// Run with `npm test` (Node 22.18+/24 strips the types itself; no test framework needed).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  activeWord, alphaAt, chapterAt, formatTime, groupWords, isTypesettable, progressAt, schedule, seekTarget,
  stepFor, tagAt, thresholdFor, turnsFromWords, typesetFits, typesetIds, STEPS, FRAME_HZ,
} from '../src/player/timing.ts';

const base = {
  id: 1, box: [0, 0, 10, 10] as [number, number, number, number], image: 'pieces/001.png',
  t_seen: 10, t_shown: 12, revealed: false, removed_at: null, pre_existing: false,
  read: { kind: 'equation', text: 'x', latex: 'x' }, spoken: [], redraw_at: null,
};

test('normal piece reveals from t_seen-0.7 to t_shown', () => {
  const s = schedule(base);
  assert.equal(progressAt(s, 9.2), 0);
  assert.equal(progressAt(s, 9.3), 0);
  assert.ok(progressAt(s, 10.65) > 0.45 && progressAt(s, 10.65) < 0.55);
  assert.equal(progressAt(s, 12), 1);
  assert.equal(progressAt(s, 99), 1);
});

test('t_shown before t_seen-0.4 still gives a short reveal', () => {
  const s = schedule({ ...base, t_shown: 10 });
  assert.equal(s.end, 10); // max(t_shown, t_seen - 0.4)
  const s2 = schedule({ ...base, t_shown: 5 });
  assert.ok(Math.abs(s2.end - 9.6) < 1e-9);
});

test('pre-existing pieces are visible from the start', () => {
  const s = schedule({ ...base, pre_existing: true, t_seen: 0, t_shown: 0 });
  assert.equal(progressAt(s, -5), 1);
  assert.equal(progressAt(s, 0), 1);
  assert.equal(tagAt(s, 0), 0);
});

test('revealed piece is redrawn at redraw_at with a "was hidden" tag', () => {
  const p = { ...base, revealed: true, t_seen: 28.33, t_shown: 33.33, redraw_at: 24.3 };
  const s = schedule(p);
  assert.equal(s.start, 24.3);
  assert.ok(Math.abs(s.end - (24.3 + 5.6)) < 1e-9);
  assert.equal(progressAt(s, 24.2), 0);
  assert.equal(progressAt(s, 30), 1);
  assert.equal(tagAt(s, 24.0), 0);
  assert.equal(tagAt(s, 26), 1);
  assert.equal(tagAt(s, 40), 0);
  // no redraw time: starts at t_seen, at least 1.2 s long
  const s2 = schedule({ ...p, redraw_at: null, t_shown: 28.33 });
  assert.equal(s2.start, 28.33);
  assert.ok(Math.abs(s2.end - s2.start - 1.2) < 1e-9);
  // a hidden piece that kept growing for minutes is still redrawn within 6 s
  const s3 = schedule({ ...p, t_seen: 4, t_shown: 228.67, redraw_at: 0.6 });
  assert.ok(Math.abs(s3.end - 6.6) < 1e-9);
});

test('a hidden piece is never redrawn before it was written or after the camera shows it', () => {
  const p = { ...base, revealed: true, t_seen: 277.33, t_shown: 278.33, written_after: 274, redraw_at: 268.8 };
  assert.equal(schedule(p).start, 274); // talked about before it was written: drawn once it can exist
  assert.equal(schedule({ ...p, redraw_at: 275.5 }).start, 275.5);
  assert.equal(schedule({ ...p, redraw_at: 300 }).start, 277.33); // talked about later: still shown when visible
  assert.equal(schedule({ ...p, written_after: undefined }).start, 268.8);
});

test('removed piece fades out over 0.4 s', () => {
  const s = schedule({ ...base, removed_at: 50 });
  assert.equal(alphaAt(s, 49.9), 1);
  assert.ok(Math.abs(alphaAt(s, 50.2) - 0.5) < 1e-9);
  assert.equal(alphaAt(s, 50.4), 0);
});

test('steps and thresholds', () => {
  assert.equal(STEPS, 255); // one step per level of the order map
  assert.equal(stepFor(0), 0);
  assert.equal(stepFor(0.0001), 0); // rounded down: a level is drawn only once it is due
  assert.equal(stepFor(1 / STEPS), 1);
  assert.equal(stepFor(254 / STEPS), 254);
  for (let k = 0; k <= STEPS; k++) assert.equal(thresholdFor(stepFor(k / STEPS)), k);
  assert.equal(stepFor(1), STEPS);
  assert.equal(thresholdFor(STEPS), 255);
  assert.equal(thresholdFor(0), 0);
});

test('click seeks to the first spoken start, else redraw_at, else t_seen-1', () => {
  assert.equal(seekTarget({ ...base, spoken: [{ start: 61.2, end: 70 }, { start: 40, end: 41 }] }), 40);
  assert.equal(seekTarget({ ...base, redraw_at: 24.3 }), 24.3);
  assert.equal(seekTarget(base), 9);
  assert.equal(seekTarget({ ...base, t_seen: 0 }), 0);
});

test('typeset only for non-drawings with text or latex', () => {
  assert.ok(isTypesettable(base));
  assert.ok(!isTypesettable({ ...base, read: { kind: 'graph', text: 'a graph', latex: '' } }));
  assert.ok(!isTypesettable({ ...base, read: { kind: 'text', text: ' ', latex: '' } }));
  assert.ok(isTypesettable({ ...base, read: { kind: 'text', text: 'Hello', latex: '' } }));
});

test('big merged blocks stay as handwriting in typeset mode', () => {
  const long = String.raw`y=\ln|x|, \text{When } x<0, y=\ln(-x), x>0, y=\ln x, y=\ln(x^2)=2\ln|x|`;
  assert.ok(!typesetFits({ ...base, box: [192, 12, 370, 255], read: { kind: 'equation', text: '', latex: long } }, 800, 317));
  assert.ok(typesetFits({ ...base, box: [594, 28, 68, 24], read: { kind: 'equation', text: '', latex: long } }, 800, 317));
  assert.ok(typesetFits({ ...base, box: [192, 12, 370, 255] }, 800, 317)); // short reading
});

test('transcript and chapter lookups', () => {
  const words = [{ w: 'a', s: 1, e: 1.2 }, { w: 'b', s: 1.3, e: 1.5 }, { w: 'c', s: 5, e: 5.5 }];
  assert.equal(activeWord(words, 0.5), -1);
  assert.equal(activeWord(words, 1.25), 0);
  assert.equal(activeWord(words, 1.4), 1);
  assert.equal(activeWord(words, 3), -1);
  assert.equal(activeWord(words, 5.6), 2);
  assert.deepEqual(groupWords([{ start: 0, end: 1.6 }, { start: 4, end: 6 }], words), [[0, 1], [2]]);
  // an aligned word starting just before its turn (1.99 < 2.14) still belongs to it
  const w2 = [{ w: 'there?', s: 1.609, e: 1.75 }, { w: 'Yeah.', s: 1.99, e: 2.49 }];
  assert.deepEqual(groupWords([{ start: 0, end: 1.66 }, { start: 2.14, end: 20 }], w2), [[0], [1]]);
  const ch = [{ t: 0, title: 'A' }, { t: 24.3, title: 'B' }];
  assert.equal(chapterAt(ch, 10), 0);
  assert.equal(chapterAt(ch, 24.3), 1);
  assert.equal(formatTime(84.6), '1:24');
  assert.equal(formatTime(3725), '1:02:05');
});

test('LaTeX clean-up for KaTeX', async () => {
  const { cleanLatex } = await import('../src/player/texClean.ts');
  assert.equal(cleanLatex('$x$'), 'x');
  assert.equal(cleanLatex('$$a+b$$'), 'a+b');
  assert.equal(cleanLatex(String.raw`\$2.61`), String.raw`\$2.61`);
  assert.equal(cleanLatex('= $2.71826...'), String.raw`= \$2.71826...`);
  assert.equal(cleanLatex(String.raw`\(y=x\)`), 'y=x');
});

test('words without turns are grouped into paragraphs at pauses', () => {
  const w = (s: number, e: number, t: string) => ({ w: t, s, e });
  const turns = turnsFromWords([w(0, 0.3, 'so'), w(0.4, 0.7, 'today'), w(3, 3.2, 'next'), w(3.3, 3.5, 'part')]);
  assert.deepEqual(turns, [
    { start: 0, end: 0.7, text: 'so today' },
    { start: 3, end: 3.5, text: 'next part' },
  ]);
  assert.deepEqual(groupWords(turns, [w(0, 0.3, 'so'), w(0.4, 0.7, 'today'), w(3, 3.2, 'next'), w(3.3, 3.5, 'part')]), [[0, 1], [2, 3]]);
  assert.deepEqual(turnsFromWords([]), []);
});

test('a long reading whose box holds another piece stays handwriting (merged block)', () => {
  const piece = (id: number, box: [number, number, number, number], text: string) => ({
    ...base, id, box, read: { kind: 'equation', text, latex: '' },
  });
  const merged = piece(39, [515, 141, 171, 67], '#2. A = ... = 12.25, #3. A = (1+1/12)^12 $1, $1, $2.61, #4.');
  const inside = piece(55, [605, 163, 30, 16], '$2.61');
  const apart = piece(7, [10, 10, 40, 20], 'log_2 8 = 3, a long reading that is not merged with anything');
  const ids = typesetIds([merged, inside, apart], 800, 314);
  assert.deepEqual([...ids].sort(), [55, 7].sort());
});

test('draw window from the backend drives the reveal', () => {
  const p = { id: 1, box: [0, 0, 1, 1], image: '', t_seen: 125, t_shown: 130, draw: [124.3, 315.0] } as any;
  const s = schedule(p);
  assert.equal(s.start, 124.3);
  assert.equal(s.end, 315.0);
  assert.equal(progressAt(s, 219.65), 127 / STEPS); // half way: levels 1..127 (level 128 holds pixels due up to 220.03)
});

// The order map (board.py piece_image): a pixel at time te in the draw window [a, b] gets level 1 + round(254 f).
const levelOf = (te: number, a: number, b: number) => 1 + Math.round((254 * (te - a)) / (b - a));
/** When the player first draws a pixel of level v (scanning at 60 frames a second). */
const drawnAt = (s: ReturnType<typeof schedule>, v: number, t0: number, t1: number) => {
  for (let t = t0; t <= t1 + 1 / 60; t += 1 / 60) if (thresholdFor(stepFor(progressAt(s, t))) >= v) return t;
  return Infinity;
};

test('no pixel of a long piece is drawn before its time in the order map', () => {
  // test-full #39: 278 pixels shown over 186 s (a few early strokes, then bits added for three minutes). With 64 steps
  // half of them were drawn more than 0.67 s early, up to 2.64 s.
  const [a, b] = [125.0, 311.33];
  const s = schedule({ ...base, t_seen: 125.67, t_shown: 311.33, draw: [a, b] });
  let worstEarly = -Infinity;
  let worstLate = -Infinity;
  for (let k = 0; k <= 400; k++) {
    const te = a + ((b - a) * k) / 400;
    const at = drawnAt(s, levelOf(te, a, b), a - 1, b + 1);
    worstEarly = Math.max(worstEarly, te - at);
    worstLate = Math.max(worstLate, at - te);
  }
  assert.ok(worstEarly <= 0, `drawn ${worstEarly.toFixed(3)} s early`);
  // at most one level late (0.73 s here), plus the frame tick
  assert.ok(worstLate <= (b - a) / 254 + 1 / FRAME_HZ + 1 / 60 + 1e-9, `drawn ${worstLate.toFixed(3)} s late`);
});

test('a piece being drawn gets a new frame at most FRAME_HZ times a second, and still draws smoothly', () => {
  for (const [a, b] of [[10.0, 11.33], [10.0, 15.0], [125.0, 311.33]]) {
    const s = schedule({ ...base, t_seen: a + 0.33, t_shown: b, draw: [a, b] });
    const steps: number[] = [];
    for (let t = a - 0.1; t <= b + 0.1; t += 1 / 60) steps.push(stepFor(progressAt(s, t)));
    for (let i = 1; i < steps.length; i++) assert.ok(steps[i] >= steps[i - 1]);
    let changes = 0;
    for (let i = 1; i < steps.length; i++) if (steps[i] !== steps[i - 1]) changes++;
    const secs = b - a + 0.2;
    assert.ok(changes <= FRAME_HZ * secs + 1, `${changes} new frames in ${secs.toFixed(2)} s`);
    // smooth: a new frame on most ticks while it is drawn (64 steps gave a 5 s piece one every 78 ms)
    assert.ok(changes >= Math.min(STEPS, FRAME_HZ * (b - a)) * 0.9, `${changes} new frames in ${(b - a).toFixed(2)} s`);
    assert.equal(steps[0], 0);
    assert.equal(steps[steps.length - 1], STEPS);
  }
});

test('a hidden piece that kept growing is redrawn early only for what the camera showed when it came into view', () => {
  // test2 #14 as processed before: revealed at 10.33, but it gained ink until 37.67 (t_shown 35.33).
  const [a, b] = [10.01, 37.67];
  const p = { ...base, revealed: true, t_seen: 10.33, t_shown: 35.33, draw: [a, b] as [number, number], redraw_at: 9.5,
    written_after: 0 };
  const s = schedule(p);
  assert.equal(s.start, 9.5);
  assert.ok(s.redraw && Math.abs(s.redraw.end - (9.5 + 6)) < 1e-9);
  assert.equal(s.end, b);
  // ink shown in the first 6 s after it came into view is redrawn from 9.5 on...
  assert.ok(drawnAt(s, levelOf(16.0, a, b), 9, 40) < 16.0);
  // ...ink it gained later is drawn when the replay shows it, not before
  for (const te of [16.6, 20, 30, 37.67]) {
    const at = drawnAt(s, levelOf(te, a, b), 9, 40);
    assert.ok(at >= te - 1e-9, `ink shown at ${te} drawn at ${at.toFixed(2)}`);
  }
  // a piece shown within 6 s of coming into view is redrawn whole, as before (test-full #34)
  const whole = schedule({ ...p, t_seen: 28, t_shown: 31.33, draw: [27.67, 31.33], redraw_at: 24, written_after: 23.33 });
  assert.equal(whole.redraw, undefined);
  assert.ok(Math.abs(whole.end - 28.26) < 1e-9);
});

test('a hidden piece redraws early only the levels the backend allows; ink seen as it was written is drawn when due', () => {
  // The slow-writer scene's #12: a word under the palm 9 s (hidden), uncovered at 97.67 together with the first strokes
  // of the next word, written in view. The backend orders the hidden ink first and allows levels 1..150 early.
  const [a, b] = [97.34, 98.34];
  const p = { ...base, revealed: true, t_seen: 97.67, t_shown: 98.33, draw: [a, b] as [number, number], redraw_at: 88.37,
    written_after: 76.67, redraw_to: 150 };
  const s = schedule(p);
  assert.equal(s.start, 88.37);
  assert.ok(s.redraw && s.redraw.to === 150);
  assert.ok(drawnAt(s, 150, 88, 100) < 90.5); // the hidden word is redrawn from when the link says it was written
  for (let v = 151; v <= 255; v++) {
    const due = a + ((b - a) * (v - 0.5)) / 254; // the latest a pixel of this level can be due
    assert.ok(drawnAt(s, v, 88, 100) >= Math.min(due, b) - 1e-9, `level ${v} drawn before it is due`);
  }
  assert.equal(progressAt(s, b + 0.01), 1);
  // redrawn as long as writing it takes (300 px at 88 a second: the camera showed it at once)...
  const slow = schedule({ ...p, redraw_secs: 3.4 });
  assert.ok(slow.redraw && Math.abs(slow.redraw.end - (88.37 + 3.4)) < 1e-9);
  assert.ok(Math.abs(drawnAt(slow, 150, 88, 100) - 91.77) < 0.02);
  // ...but done soon after the camera shows it: with no link time it starts at t_seen and is as brisk as before
  const late = schedule({ ...p, redraw_at: null, redraw_secs: 3.4 });
  assert.ok(late.redraw && Math.abs(late.redraw.end - late.redraw.start - 1.2) < 1e-9);
  // nothing allowed: drawn when the replay shows it (the tag still shows)...
  const none = schedule({ ...p, redraw_to: 0 });
  assert.ok(drawnAt(none, 1, 88, 100) >= a);
  assert.ok(tagAt(none, 90) > 0);
  // ...all of it allowed, or an older replay without redraw_to: redrawn whole, as before
  assert.equal(schedule({ ...p, redraw_to: 255 }).redraw, undefined);
  assert.equal(schedule({ ...p, redraw_to: undefined }).redraw, undefined);
});
