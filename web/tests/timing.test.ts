// Run with `npm test` (Node 22.18+/24 strips the types itself; no test framework needed).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  activeWord, alphaAt, chapterAt, formatTime, groupWords, isTypesettable, progressAt, schedule, seekTarget,
  stepFor, tagAt, thresholdFor, turnsFromWords, typesetFits, typesetIds, STEPS,
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
  assert.equal(stepFor(0), 0);
  assert.equal(stepFor(0.0001), 1);
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
  assert.ok(Math.abs(progressAt(s, 219.65) - 0.5) < 1e-9);
});
