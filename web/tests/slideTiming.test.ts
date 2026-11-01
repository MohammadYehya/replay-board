// Run with `npm test`. Timing and geometry of the slide view: states, crossfades, annotations, pulse, pointer.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  FADE_OUT, HOLD, POINTER_NEAR, PULSE_PERIOD, PULSE_RINGS, SPOT_FADE_IN, annotationAlpha, annotationTiming,
  arrowEnds, arrowGeometry, arrowHeadAt, circlePath, circleStart, crossfadeAt, drawProgress, elementSeekTime, exitPoint,
  isLightColour, kindOf, pickElement, pointerAt, pulseRings, slideEndAt, sortStates, spotlightHole, stateIndexAt, toolColour,
  underlinePath, underlineY,
} from '../src/player/slideTiming.ts';
import type { PointerSample, SlideAnnotation, SlideElement, SlideState } from '../src/types.ts';

const near = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) <= eps;
const box = [100, 100, 200, 40] as [number, number, number, number];
const ann = (o: Partial<SlideAnnotation>): SlideAnnotation => ({
  type: 'circle', tool: 'cursor', t_start: 10, t_end: 11, target: [], box, from: null, to: null, ...o,
});

// The six states of the generated slides lecture (data/jobs/slides-gen).
const STATES: SlideState[] = [
  { id: 0, kind: 'slide', t_start: 0, t_end: 5.75, t_change: 0, image: 'slides/000.png' },
  { id: 1, kind: 'addition', t_start: 6, t_end: 12.75, t_change: 5.75, image: 'slides/001.png' },
  { id: 2, kind: 'slide', t_start: 13, t_end: 20.5, t_change: 12.75, image: 'slides/002.png' },
  { id: 3, kind: 'addition', t_start: 20.75, t_end: 26.5, t_change: 20.5, image: 'slides/003.png' },
  { id: 4, kind: 'slide', t_start: 27.25, t_end: 33.75, t_change: 26.5, image: 'slides/004.png' },
  { id: 5, kind: 'addition', t_start: 34, t_end: 40, t_change: 33.75, image: 'slides/005.png' },
];

test('the state shown at t is the last one whose change has begun', () => {
  const s = sortStates([...STATES].reverse());
  assert.deepEqual(s.map((x) => x.id), [0, 1, 2, 3, 4, 5]);
  assert.equal(stateIndexAt(s, -3), 0); // before the first state: the first
  assert.equal(stateIndexAt(s, 0), 0);
  assert.equal(stateIndexAt(s, 5.74), 0);
  assert.equal(stateIndexAt(s, 5.75), 1); // the change window belongs to the new state
  assert.equal(stateIndexAt(s, 12.8), 2);
  assert.equal(stateIndexAt(s, 26.7), 4);
  assert.equal(stateIndexAt(s, 39.9), 5);
  assert.equal(stateIndexAt(s, 400), 5);
  assert.equal(stateIndexAt([], 3), -1);
  // no t_change: the change starts at t_start
  assert.equal(stateIndexAt(sortStates([{ ...STATES[0] }, { ...STATES[1], t_change: undefined }]), 5.9), 0);
});

test('a change crossfades from the previous image over [t_change, t_start)', () => {
  assert.deepEqual(crossfadeAt(STATES, 0, 0), { prev: -1, mix: 1 });
  assert.deepEqual(crossfadeAt(STATES, 1, 5.75), { prev: 0, mix: 0 });
  const mid = crossfadeAt(STATES, 1, 5.875);
  assert.equal(mid.prev, 0);
  assert.ok(near(mid.mix, 0.5));
  assert.deepEqual(crossfadeAt(STATES, 1, 6), { prev: -1, mix: 1 });
  const late = crossfadeAt(STATES, 4, 27);
  assert.equal(late.prev, 3);
  assert.ok(late.mix > 0.5 && late.mix < 1);
  // an instant change has no crossfade
  const inst = [STATES[0], { ...STATES[1], t_change: 6 }];
  assert.deepEqual(crossfadeAt(inst, 1, 6), { prev: -1, mix: 1 });
});

test('an annotation shows from t_start to t_end + 1.5 s, then fades out over 0.4 s', () => {
  const tm = annotationTiming(ann({ t_start: 10, t_end: 11 }));
  assert.equal(tm.holdEnd, 11 + HOLD);
  assert.equal(tm.end, 11 + HOLD + FADE_OUT);
  assert.equal(annotationAlpha(tm, 9.99), 0);
  assert.equal(annotationAlpha(tm, 10), 0); // fading in from nothing
  assert.equal(annotationAlpha(tm, 10.5), 1);
  assert.equal(annotationAlpha(tm, 12.5), 1);
  const fading = annotationAlpha(tm, 12.5 + FADE_OUT / 2);
  assert.ok(near(fading, 0.5));
  assert.equal(annotationAlpha(tm, 12.9), 0);
  assert.equal(annotationAlpha(tm, 50), 0);
  assert.equal(annotationAlpha(tm, NaN), 0);
  // fading out is monotonic and smooth
  let prev = 1;
  for (let t = 12.5; t <= 12.9; t += 0.02) {
    const a = annotationAlpha(tm, t);
    assert.ok(a <= prev + 1e-9);
    prev = a;
  }
});

test('the spotlight dims in over 0.25 s', () => {
  const tm = annotationTiming(ann({ type: 'spotlight', t_start: 16.75, t_end: 18.17 }));
  assert.equal(tm.fadeIn, SPOT_FADE_IN);
  const a1 = annotationAlpha(tm, 16.75 + 0.1);
  const a2 = annotationAlpha(tm, 16.75 + 0.2);
  assert.ok(a1 > 0 && a1 < a2 && a2 < 1);
  assert.equal(annotationAlpha(tm, 16.75 + SPOT_FADE_IN), 1);
  assert.equal(annotationAlpha(tm, 17.5), 1);
});

test('an annotation starts fading when its slide changes, even inside the hold', () => {
  // gesture ends 0.5 s before the next slide change begins
  const tm = annotationTiming(ann({ t_start: 30.33, t_end: 33.25, state: 4 }), 33.75);
  assert.equal(tm.holdEnd, 33.75);
  assert.equal(annotationAlpha(tm, 33.7), 1);
  assert.ok(annotationAlpha(tm, 34.0) < 0.5);
  assert.equal(annotationAlpha(tm, 34.2), 0);
  // a state that ends long after does not shorten anything
  assert.equal(annotationTiming(ann({}), 99).holdEnd, 11 + HOLD);
  // nonsense state end (before the gesture) is ignored
  assert.equal(annotationTiming(ann({}), 5).holdEnd, 11 + HOLD);
  assert.equal(annotationTiming(ann({}), NaN).holdEnd, 11 + HOLD);
});

test('a slide lasts through its additions, up to the next new slide', () => {
  assert.equal(slideEndAt(STATES, 0), 12.75); // state 1 only adds a bullet: the slide goes at the change to state 2
  assert.equal(slideEndAt(STATES, 1), 12.75);
  assert.equal(slideEndAt(STATES, 4), 40);
  assert.equal(slideEndAt(STATES, 5), 40);
  assert.ok(Number.isNaN(slideEndAt(STATES, -1)));
  assert.ok(Number.isNaN(slideEndAt(STATES, 6)));
  // so a gesture just before a bullet appears keeps its full hold, while one before a new slide is cut short
  const beforeBullet = annotationTiming(ann({ t_start: 4.5, t_end: 5.5, state: 0 }), slideEndAt(STATES, 0));
  assert.equal(beforeBullet.holdEnd, 5.5 + HOLD);
  const beforeSlide = annotationTiming(ann({ t_start: 11.5, t_end: 12.5, state: 1 }), slideEndAt(STATES, 1));
  assert.equal(beforeSlide.holdEnd, 12.75);
});

test('a circle is drawn from t_start to max(t_end, t_start + 0.6)', () => {
  const quick = annotationTiming(ann({ type: 'circle', t_start: 10, t_end: 10.2 }));
  assert.ok(near(quick.drawEnd, 10.6));
  assert.equal(drawProgress(quick, 10), 0);
  assert.ok(near(drawProgress(quick, 10.3), 0.5));
  assert.equal(drawProgress(quick, 10.6), 1);
  assert.equal(drawProgress(quick, 20), 1);
  const slow = annotationTiming(ann({ type: 'circle', t_start: 14.42, t_end: 16.08 }));
  assert.equal(slow.drawEnd, 16.08);
  // eased (slow start), and monotonic
  assert.ok(drawProgress(slow, 14.42 + 0.1) < 0.1 / 1.66);
  let prev = 0;
  for (let t = 14.42; t <= 16.1; t += 0.05) {
    const p = drawProgress(slow, t);
    assert.ok(p >= prev - 1e-12 && p <= 1);
    prev = p;
  }
});

test('underline and arrow draw over the gesture time; the arrowhead appears at the end', () => {
  const u = annotationTiming(ann({ type: 'underline', t_start: 9.5, t_end: 10.33 }));
  assert.equal(u.drawEnd, 10.33);
  assert.ok(drawProgress(u, 9.9) > 0.3 && drawProgress(u, 9.9) < 0.7);
  const a = annotationTiming(ann({ type: 'arrow', t_start: 35.75, t_end: 36.33 }));
  assert.equal(a.drawEnd, 36.33);
  assert.equal(arrowHeadAt(a, 35.9), 0);
  assert.equal(arrowHeadAt(a, 36.2), 0);
  assert.ok(arrowHeadAt(a, 36.33) > 0 && arrowHeadAt(a, 36.33) < 1);
  assert.equal(arrowHeadAt(a, 36.8), 1);
  // a flick still draws calmly
  assert.ok(near(annotationTiming(ann({ type: 'arrow', t_start: 5, t_end: 5.1 })).drawEnd, 5.45));
});

test('pulse rings repeat while the gesture lasts, then the ones in flight finish', () => {
  const tm = annotationTiming(ann({ type: 'pulse', t_start: 36.42, t_end: 37.58 }));
  assert.equal(pulseRings(tm, 36.0).length, PULSE_RINGS);
  assert.ok(pulseRings(tm, 36.0).every((r) => r.alpha === 0));
  const at = (t: number) => pulseRings(tm, t).filter((r) => r.alpha > 0).length;
  assert.equal(at(36.45), 1); // first ring only
  assert.equal(at(36.42 + PULSE_PERIOD / 3 + 0.05), 2);
  assert.equal(at(37.0), 2);
  assert.equal(at(37.5), 3);
  // a ring expands as it fades
  const r0 = pulseRings(tm, 36.42 + 0.3)[0];
  const r1 = pulseRings(tm, 36.42 + 1.0)[0];
  assert.ok(r1.k > r0.k && r1.alpha < r0.alpha);
  // no ring is born after the gesture ends: every ring is gone one period after the last spawn
  assert.equal(pulseRings(tm, 36.42 + PULSE_PERIOD + 0.05)[0].alpha, 0);
  assert.equal(at(37.58 + PULSE_PERIOD + 0.01), 0);
  // a longer gesture keeps the rings coming
  const long = annotationTiming(ann({ type: 'pulse', t_start: 30.33, t_end: 32.08 }));
  const again = pulseRings(long, 30.33 + PULSE_PERIOD + 0.05)[0];
  assert.ok(again.alpha > 0 && again.k < 0.3);
  // a tap shorter than a round still gets all three rings once
  const tap = annotationTiming(ann({ type: 'pulse', t_start: 7, t_end: 7.1 }));
  assert.equal(pulseRings(tap, 7 + (2 * PULSE_PERIOD) / 3 + 0.05).filter((r) => r.alpha > 0).length, 3);
});

test('pointer dot: interpolated, only when a sample is within 0.2 s', () => {
  const s: PointerSample[] = [
    [1.0, 'c', 100, 100],
    [1.1, 'c', 200, 100],
    [3.0, 'l', 50, 60],
  ];
  const mid = pointerAt(s, 1.05)!;
  assert.ok(near(mid.x, 150) && near(mid.y, 100));
  assert.equal(mid.tool, 'cursor');
  assert.equal(mid.alpha, 1);
  assert.equal(pointerAt(s, 0.7), null);
  assert.ok(pointerAt(s, 0.85)); // 0.15 s before the first sample
  assert.equal(pointerAt(s, 1.5), null); // gap: no sample within 0.2 s
  const held = pointerAt(s, 1.25)!; // past the last cursor sample, not interpolated across the gap
  assert.equal(held.x, 200);
  assert.ok(held.alpha > 0 && held.alpha < 1);
  assert.equal(pointerAt(s, 3.0)!.tool, 'laser');
  assert.equal(pointerAt(s, 3.0 + POINTER_NEAR + 0.01), null);
  assert.equal(pointerAt([], 1), null);
});

test('clicking an element seeks to its first spoken time, else to its slide', () => {
  const st = STATES[2];
  const el: SlideElement = { id: 's2e1', kind: 'title', text: 'x', latex: '', box, spoken: [{ start: 15, end: 16 }, { start: 14.2, end: 14.9 }] };
  assert.equal(elementSeekTime(el, st), 14.2);
  assert.equal(elementSeekTime({ ...el, spoken: [] }, st), 13);
  assert.equal(elementSeekTime({ ...el, spoken: [] }, st, 20), 20); // never before the processed part
});

test('the smallest element under the point is picked', () => {
  const chart: SlideElement = { id: 'c', kind: 'chart', text: '', latex: '', box: [84, 138, 534, 303] };
  const label: SlideElement = { id: 'l', kind: 'label', text: '300', latex: '', box: [84, 188, 29, 13] };
  assert.equal(pickElement([chart, label], 90, 195)?.id, 'l');
  assert.equal(pickElement([chart, label], 300, 300)?.id, 'c');
  assert.equal(pickElement([chart, label], 10, 10), null);
  assert.equal(pickElement([label], 115, 195, 3)?.id, 'l'); // a little slack around small boxes
});

test('arrows point at their targets instead of covering them', () => {
  const from: [number, number] = [276.5, 334]; // centre of "n = 365 gives A = 2.715"
  const to: [number, number] = [603.5, 197]; // centre of "e = 2.71828"
  const fromBox = [66, 318, 421, 32] as [number, number, number, number];
  const toBox = [508, 168, 191, 58] as [number, number, number, number];
  const g = arrowGeometry(from, to, [960, 540], fromBox, toBox);
  const inside = (p: [number, number], b: number[]) => p[0] >= b[0] && p[0] <= b[0] + b[2] && p[1] >= b[1] && p[1] <= b[1] + b[3];
  assert.ok(!inside(g.from, fromBox));
  assert.ok(!inside(g.tip, toBox));
  assert.ok(g.tip[1] > 226 && g.tip[1] < 240); // just below the target box
  assert.match(g.d, /^M[\d.]+ [\d.]+ Q[\d.]+ [\d.]+ [\d.]+ [\d.]+$/);
  // exit point of a segment from inside a box
  const e = exitPoint([10, 10], [50, 10], [0, 0, 20, 20], 0)!;
  assert.ok(near(e[0], 20) && near(e[1], 10));
  assert.equal(exitPoint([10, 10], [15, 10], [0, 0, 20, 20], 0), null); // both ends inside
  // ends from the gesture path when from/to are missing
  assert.deepEqual(arrowEnds(ann({ type: 'arrow', path: [[1, 2], [3, 4], [5, 6]] })), [[1, 2], [5, 6]]);
  assert.deepEqual(arrowEnds(ann({ type: 'arrow', from: [7, 8], to: [9, 10] })), [[7, 8], [9, 10]]);
});

test('an arrow starts and stops where the pointer did when that is on its elements', () => {
  // slides-gen: the pointer went from the end of "n = 365 gives A = 2.715" to "e = 2.71828"; from / to are the
  // element centres, and an arrow from the middle of that line cuts through "n = 12 gives A = 2.613" above it
  const arrow = ann({
    type: 'arrow', from: [233.5, 336], to: [561, 202.5],
    path: [[370, 331.7], [383, 325.7], [475, 275.7], [588, 207], [594.3, 202.7]],
  });
  const fromBox = [66, 324, 335, 24] as [number, number, number, number];
  const toBox = [451, 178, 220, 49] as [number, number, number, number];
  const lineAbove = [66, 276, 320, 24]; // ink of "n = 12 gives A = 2.613"
  assert.deepEqual(arrowEnds(arrow, fromBox, toBox), [[370, 331.7], [594.3, 202.7]]);
  assert.deepEqual(arrowEnds(arrow), [[233.5, 336], [561, 202.5]]); // no element boxes: as given
  // a path that starts elsewhere keeps the backend's start
  assert.deepEqual(arrowEnds({ ...arrow, path: [[700, 420], [594.3, 202.7]] }, fromBox, toBox), [[233.5, 336], [594.3, 202.7]]);
  const crosses = (d: string, b: number[]) => {
    const n = d.match(/-?[\d.]+/g)!.map(Number); // M ax ay Q cx cy bx by
    for (let k = 0; k <= 50; k++) {
      const s = k / 50;
      const x = (1 - s) ** 2 * n[0] + 2 * s * (1 - s) * n[2] + s * s * n[4];
      const y = (1 - s) ** 2 * n[1] + 2 * s * (1 - s) * n[3] + s * s * n[5];
      if (x >= b[0] && x <= b[0] + b[2] && y >= b[1] && y <= b[1] + b[3]) return true;
    }
    return false;
  };
  assert.ok(crosses(arrowGeometry([233.5, 336], [561, 202.5], [960, 540], fromBox, toBox).d, lineAbove));
  const g = arrowGeometry(...arrowEnds(arrow, fromBox, toBox), [960, 540], fromBox, toBox);
  assert.ok(!crosses(g.d, lineAbove));
  const inside = (p: [number, number], b: number[]) => p[0] >= b[0] && p[0] <= b[0] + b[2] && p[1] >= b[1] && p[1] <= b[1] + b[3];
  assert.ok(!inside(g.from, fromBox) && !inside(g.tip, toBox));
});

test('slide arrows are drawn exactly as before the board learned to keep arrows off its writing', () => {
  // Recorded from the player before that change: slides pass no obstacles, so nothing may move.
  const S: [number, number] = [960, 540];
  const arrow = ann({
    type: 'arrow', from: [233.5, 336], to: [561, 202.5],
    path: [[370.0, 331.7], [383.0, 325.7], [422.0, 305.7], [475.7, 276.0], [532.3, 242.7], [568.7, 219.7], [588.0, 207.0], [594.3, 202.7]],
  });
  const fb = [66, 324, 335, 24] as [number, number, number, number];
  const tb = [451, 178, 220, 49] as [number, number, number, number];
  const cases: [() => ReturnType<typeof arrowGeometry>, string, string, [number, number], [number, number]][] = [
    [() => arrowGeometry(...arrowEnds(arrow, fb, tb, 8), S, fb, tb), 'M387.7 321.5 Q454.5 253.5 546.8 230',
      'M530.3 226.2 L546.8 230 L534.1 241.2', [546.8469772984955, 229.99130596742788], [387.7227514760599, 321.5072450271434]],
    [() => arrowGeometry([276.5, 334], [603.5, 197], S, [66, 318, 421, 32], [508, 168, 191, 58]), 'M319.3 316.1 Q411.7 242.9 528.7 228.3',
      'M512.8 222.5 L528.7 228.3 L514.7 237.8', [528.747077143544, 228.31850284811765], [319.301401650453, 316.0679142932353]],
    [() => arrowGeometry([150, 120], [150, 160], S, [100, 100, 200, 40], [100, 150, 200, 30]), 'M305 120 Q332 142.5 305 165',
      'M315.7 162.6 L305 165 L309.3 154.9', [305, 165], [305, 120]],
    [() => arrowGeometry([100, 100], [120, 110], S), 'M100 100 Q111.4 102.2 120 110',
      'M116.1 99.7 L120 110 L109.4 107.1', [120, 110], [100, 100]],
  ];
  for (const [make, d, head, tip, from] of cases) {
    const g = make();
    assert.equal(g.d, d);
    assert.equal(g.head, head);
    assert.deepEqual(g.tip, tip);
    assert.deepEqual(g.from, from);
  }
  // and with an empty or missing obstacle list, the same arrow
  const g = arrowGeometry([276.5, 334], [603.5, 197], S, [66, 318, 421, 32], [508, 168, 191, 58]);
  assert.deepEqual(arrowGeometry([276.5, 334], [603.5, 197], S, [66, 318, 421, 32], [508, 168, 191, 58], []), g);
  assert.deepEqual(arrowGeometry([276.5, 334], [603.5, 197], S, [66, 318, 421, 32], [508, 168, 191, 58], null), g);
});

test('shapes: circle encloses the box, underline sits under it, spotlight hole pads it', () => {
  const pts = [...circlePath(box, [960, 540]).matchAll(/[ML]([\d.-]+) ([\d.-]+)/g)].map((m) => [+m[1], +m[2]]);
  assert.ok(pts.length > 100);
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  assert.ok(Math.min(...xs) < 100 - 8 && Math.max(...xs) > 300 + 8);
  assert.ok(Math.min(...ys) < 100 - 8 && Math.max(...ys) > 140 + 8);
  const ul = underlinePath(box, [960, 540]);
  assert.match(ul, /^M97 145 Q/);
  assert.match(underlinePath(box, [960, 540], 150), /^M97 150 Q/);
  const hole = spotlightHole(box, [960, 540]);
  assert.deepEqual([hole.x, hole.y, hole.w, hole.h], [88, 88, 224, 64]);
  // twice the slide resolution: twice the padding
  assert.deepEqual(spotlightHole([200, 200, 400, 80], [1920, 1080]).x, 176);
});

test('a drawn circle starts where the teacher started and turns the same way', () => {
  const ring = (n: number, sign: number, a0: number): [number, number][] =>
    Array.from({ length: n }, (_, i) => {
      const a = a0 + (sign * i * 2 * Math.PI) / n;
      return [200 + 100 * Math.cos(a), 120 + 30 * Math.sin(a)];
    });
  const cw = circleStart(ring(20, 1, 0), 200, 120, 100, 30); // right, bottom, left, top: clockwise on screen
  assert.equal(cw.dir, 1);
  assert.ok(near(cw.a0, 0, 1e-9));
  const ccw = circleStart(ring(20, -1, Math.PI / 2), 200, 120, 100, 30);
  assert.equal(ccw.dir, -1);
  assert.ok(near(ccw.a0, Math.PI / 2, 1e-9));
  // slides-gen: the laser loop around "338" starts at its lower right and goes clockwise
  const laser: [number, number][] = [[567.1, 179.1], [543.8, 177.9], [516.5, 188.1], [476.5, 187.4], [441.9, 171.1],
    [440.9, 149.6], [470.8, 134.9], [510.7, 134.9], [539.7, 146.6], [547.7, 164.8]];
  const l = circleStart(laser, 493, 158, 39, 30);
  assert.equal(l.dir, 1);
  assert.ok(l.a0 > 0 && l.a0 < Math.PI / 2);
  // a straight stroke or no path: the default start (upper left, clockwise)
  assert.equal(circleStart([[0, 0], [10, 0], [20, 0], [30, 0]], 200, 120, 100, 30).dir, 1);
  assert.ok(circleStart(undefined, 200, 120, 100, 30).a0 < -Math.PI / 2);
  // the path follows: its first point is near the gesture's start
  const d = circlePath([475, 150, 36, 16], [960, 540], laser);
  const [x0, y0] = d.slice(1).split(' ').slice(0, 2).map(Number);
  assert.ok(x0 > 493 && y0 > 158);
});

test('an underline goes at underline_y, else under the measured sweep, else under the box', () => {
  const size: [number, number] = [960, 540];
  // slides-gen: the box read for "Next year you ..." sits a line gap high (ink is y 193-216); the sweep ran at y≈220
  const sweep = ann({
    type: 'underline', box: [86, 166, 565, 28],
    path: [[90.3, 218.7], [139.3, 219.7], [197, 220.3], [259, 220], [308.3, 220]],
  });
  assert.ok(near(underlineY(sweep, size), 222, 0.5));
  assert.equal(underlineY({ ...sweep, underline_y: 212 }, size), 212);
  // a sweep through the upper half of the box (or far away) is not an underline height: under the box
  assert.equal(underlineY({ ...sweep, path: [[90, 170], [300, 171]] }, size), 199);
  assert.equal(underlineY({ ...sweep, path: [[90, 400], [300, 401]] }, size), 199);
  assert.equal(underlineY({ ...sweep, path: [] }, size), 199);
  // never off the slide
  assert.equal(underlineY(ann({ type: 'underline', box: [10, 500, 100, 39] }), size), 537);
});

test('kinds, tools and colours', () => {
  assert.equal(kindOf(ann({ type: 'spotlight' })), 'spotlight');
  assert.equal(kindOf(ann({ type: 'mystery' })), 'pulse');
  assert.equal(toolColour('laser'), '#ef4444');
  assert.equal(toolColour('cursor'), '#f59e0b');
  assert.ok(isLightColour('#fcfffc'));
  assert.ok(!isLightColour('#1a1d22'));
  assert.ok(isLightColour(undefined));
});
