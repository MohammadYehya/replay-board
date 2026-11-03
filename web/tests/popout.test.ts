// Pop-out helpers: sampling y = f(x) around asymptotes and jumps, ticks, ink colours, and which pieces get 3D.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  distinctCurveColors, fitRange, formatTick, looksAlike, niceStep, niceTicks, sampleSegments, stageColors, stageInk,
} from '../src/player/plotMath.ts';
import {
  badgePlaces, badgeSpot, boxOverlap, canPopout, cleanMadeScenes, drawingOnBoard, explainedAt, looksLikeGraphEquation, PILL_AFTER, PILL_BEFORE,
} from '../src/player/popoutable.ts';
import type { Box, Piece } from '../src/types.ts';

const maxGap = (seg: [number, number][], xr: number, yr: number) => {
  let g = 0;
  for (let i = 1; i < seg.length; i++)
    g = Math.max(g, Math.hypot((seg[i][0] - seg[i - 1][0]) / xr, (seg[i][1] - seg[i - 1][1]) / yr));
  return g;
};

test('a smooth function is one segment covering the range', () => {
  const segs = sampleSegments((x) => x * x, -2, 2, 0, 4);
  assert.equal(segs.length, 1);
  const s = segs[0];
  assert.equal(s[0][0], -2);
  assert.equal(s[s.length - 1][0], 2);
  assert.ok(maxGap(s, 4, 4) < 0.01);
});

test('|ln x| runs into the asymptote at 0 and up to the edge of the window', () => {
  const segs = sampleSegments((x) => Math.abs(Math.log(x)), 0, 5, -0.5, 4);
  assert.equal(segs.length, 1);
  const s = segs[0];
  const [x0, y0] = s[0];
  assert.ok(y0 > 4 && y0 <= 4.1, `starts at the top of the window, got ${y0}`);
  assert.ok(Math.abs(x0 - Math.exp(-y0)) < 1e-6);
  // the cusp at (1, 0) is sampled closely
  const nearCusp = s.reduce((b, p) => Math.min(b, Math.hypot(p[0] - 1, p[1])), Infinity);
  assert.ok(nearCusp < 0.01);
  assert.ok(maxGap(s, 5, 4.5) < 0.01, 'steep part is refined');
  assert.ok(s.every(([, y]) => Number.isFinite(y)));
});

test('ln|x| splits into two branches at the pole', () => {
  const segs = sampleSegments((x) => Math.log(Math.abs(x)), -5, 5, -3, 2);
  assert.equal(segs.length, 2);
  assert.ok(segs[0][segs[0].length - 1][0] < 0 && segs[1][0][0] > 0);
  assert.ok(segs[0][segs[0].length - 1][1] < -2.9 && segs[1][0][1] < -2.9);
});

test('tan breaks at each pole, even with a tall window', () => {
  const segs = sampleSegments(Math.tan, -Math.PI, Math.PI, -50, 50);
  assert.equal(segs.length, 3);
  for (const s of segs) for (let i = 1; i < s.length; i++) assert.ok(s[i][1] > s[i - 1][1] - 1e-9, 'each branch rises');
});

test('a jump (floor) splits; sqrt stops at the edge of its domain', () => {
  assert.equal(sampleSegments(Math.floor, 0, 3.5, -1, 4).length, 4);
  // small steps in a tall window are jumps too: no vertical risers between the treads
  const steps = sampleSegments((x) => Math.floor(x) / 2, -5, 5, -3, 3);
  assert.equal(steps.length, 10);
  for (const seg of steps) assert.ok(seg[seg.length - 1][1] - seg[0][1] < 1e-9, 'each tread is flat');
  // a steep but continuous function is not split
  assert.equal(sampleSegments((x) => Math.atan(x * 2000), -1, 1, -2, 2).length, 1);
  const s = sampleSegments(Math.sqrt, -2, 4, 0, 2);
  assert.equal(s.length, 1);
  assert.ok(Math.abs(s[0][0][0]) < 1e-6);
});

test('nothing to draw', () => {
  assert.deepEqual(sampleSegments(() => NaN, 0, 1, 0, 1), []);
  assert.deepEqual(sampleSegments((x) => x + 100, 0, 1, 0, 1), []);
  assert.deepEqual(sampleSegments((x) => x, 1, 1, 0, 1), []);
});

test('a wild function stays bounded', () => {
  const segs = sampleSegments((x) => Math.sin(1 / x), -1, 1, -1.2, 1.2);
  const n = segs.reduce((a, s) => a + s.length, 0);
  assert.ok(n < 60000, `${n} points`);
});

test('nice ticks', () => {
  assert.equal(niceStep(10), 2);
  assert.equal(niceStep(4.5), 1);
  assert.equal(niceStep(0.3), 0.05);
  assert.deepEqual(niceTicks(0, 5).ticks, [0, 1, 2, 3, 4, 5]);
  assert.deepEqual(niceTicks(-0.5, 4).ticks, [0, 1, 2, 3, 4]);
  const t = niceTicks(-1, 1);
  assert.equal(t.step, 0.5);
  assert.deepEqual(t.ticks, [-1, -0.5, 0, 0.5, 1]);
  assert.equal(formatTick(-0.5, 0.5), '−0.5');
  assert.equal(formatTick(2, 1), '2');
  assert.equal(formatTick(-1e-17, 0.5), '0');
  assert.equal(formatTick(1, 0.5), '1');
  assert.equal(formatTick(2.5, 0.5), '2.5');
  assert.equal(formatTick(0.1 + 0.2, 0.1), '0.3');
  assert.equal(formatTick(-0.05, 0.05), '−0.05');
  assert.equal(formatTick(1000, 500), '1000');
});

test('fitRange keeps a sensible window and fixes a hopeless one', () => {
  assert.deepEqual(fitRange([1, 2, 3, NaN], 0, 4), { min: 0, max: 4 });
  const r = fitRange([100, 110, 120, 130], -1, 1);
  assert.ok(r.min < 100 && r.max > 130 && r.max - r.min < 60);
});

test('stage ink: dark board ink is lightened, black and grey become the accent', () => {
  const accent = '#7aa7ff';
  assert.equal(stageInk('#000000', accent), accent);
  assert.equal(stageInk('#555555', accent), accent);
  assert.equal(stageInk('not a colour', accent), accent);
  assert.equal(stageInk(undefined, accent), accent);
  assert.equal(stageInk('#ffffff', accent), '#ffffff');
  const navy = stageInk('#333366', accent);
  assert.match(navy, /^#[0-9a-f]{6}$/);
  const lum = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)).reduce((a, b) => a + b, 0) / 3;
  assert.ok(lum(navy) > 130, `lightened: ${navy}`);
  const red = stageInk('#cc2222', accent);
  assert.ok(parseInt(red.slice(1, 3), 16) > parseInt(red.slice(3, 5), 16) + 60, `still red: ${red}`);
  assert.equal(stageInk('#f00', accent), '#ff3d3d'); // short form, already bright: lightness lifted to 0.62
});

test('stage colours: each dark ink becomes one colour for all its objects, not a rainbow', () => {
  const palette = ['#accent', '#p1', '#p2'];
  // twelve black edges of a cube, a grey label, a red arc, black again, an empty colour
  const colors = [...Array(12).fill('#222222'), '#888888', '#d62728', '#222222', ''];
  const out = stageColors(colors, palette);
  assert.deepEqual(new Set(out.slice(0, 12)), new Set(['#accent']));
  assert.equal(out[12], '#p1'); // a second dark ink: the next palette colour
  assert.equal(out[13], stageInk('#d62728', '')); // a real colour keeps its hue
  assert.equal(out[14], '#accent');
  assert.equal(out[15], '#p2');
});

test('named arcs get colours of their own, not the colour of each other or of an edge', () => {
  const palette = ['#7aa7ff', '#ff9f6b', '#6fdc9a', '#f2c94c', '#c792ea', '#5fd4e6', '#ff7aa8', '#b8c4ff'];
  // test2 #4: red a edges, green b edges, blue c edges, and three black angle arcs (black -> the accent blue)
  const objects = [
    { type: 'segment', label: 'a' },
    { type: 'segment', label: 'b' },
    { type: 'segment', label: 'c' },
    { type: 'segment', label: '' },
    { type: 'curve', label: 'gamma' },
    { type: 'curve', label: 'beta' },
    { type: 'curve', label: 'alpha' },
    { type: 'curve', label: '' }, // an unnamed curve keeps its colour
    { type: 'point', label: 'O' },
  ];
  const before = stageColors(['#e53935', '#43a047', '#1e88e5', '#1e88e5', '#000000', '#000000', '#000000', '#000000', '#000000'], palette);
  assert.ok(looksAlike(before[2], before[4]), 'the arcs start as blue as the c edges');
  const out = distinctCurveColors(objects, before, palette);
  const arcs = out.slice(4, 7);
  for (const a of arcs) for (const e of out.slice(0, 3)) assert.ok(!looksAlike(a, e), `${a} looks like the edge ${e}`);
  assert.ok(!looksAlike(arcs[0], arcs[1]) && !looksAlike(arcs[0], arcs[2]) && !looksAlike(arcs[1], arcs[2]), arcs.join());
  assert.equal(out[7], before[7]);
  assert.deepEqual(out.slice(0, 4), before.slice(0, 4)); // edges are never changed
  // a named curve already distinct is left alone
  const plain = distinctCurveColors([{ type: 'segment', label: '' }, { type: 'curve', label: 'y = ln x' }], ['#e53935', '#43a047'], palette);
  assert.deepEqual(plain, ['#e53935', '#43a047']);
});

const piece = (kind: string, text = '', latex = ''): Piece => ({
  id: 1, box: [0, 0, 10, 10], image: 'p.png', t_seen: 0, t_shown: 0, read: { kind, text, latex },
});

test('which pieces get a 3D view', () => {
  for (const k of ['graph', 'diagram', 'figure', 'plot', 'sketch', 'drawing', 'chart', 'curve', 'Graph', 'graphs'])
    assert.ok(canPopout(piece(k)), k);
  for (const k of ['text', 'arrow', 'number', 'label', 'other', 'scribble', 'box', 'underline'])
    assert.ok(!canPopout(piece(k, 'y = ln x', 'y = \\ln x')), k);
  assert.ok(canPopout(piece('arrow', 'x-axis and y-axis arrows labeled x and y')));
  assert.ok(canPopout(piece('arrow', 'red vector arrow labeled v')));
  assert.ok(canPopout(piece('arrow', 'z-axis arrow labeled Z')));
  assert.ok(!canPopout(piece('arrow', 'arrow from the sum to the result')));
  assert.ok(canPopout(piece('equation', 'y = |ln x| = |log_e x|', 'y = |\\ln x| = |\\log_e x|')));
  assert.ok(canPopout(piece('equation', 'y = ln|x|', 'y=\\ln|x|')));
  assert.ok(canPopout(piece('equation', 'z = x^2 + y^2', '')));
  assert.ok(!canPopout(piece('equation', 'A = P(1+r)^n', 'A = P(1+r)^n')));
  assert.ok(!canPopout(piece('equation', 'e = 2.71828', 'e = 2.71828')));
  assert.ok(!canPopout(null));
  assert.ok(!canPopout({ ...piece('graph'), read: undefined }));
  assert.ok(looksLikeGraphEquation('f(x) = e^{-x}'));
  assert.ok(looksLikeGraphEquation('$y = mx + c$'));
  assert.ok(looksLikeGraphEquation('f(x, y) = \\sin x \\cos y'));
  assert.ok(!looksLikeGraphEquation('y = 3'));
  assert.ok(!looksLikeGraphEquation('y = \\exp(2)'));
  assert.ok(!looksLikeGraphEquation('x = 3'));
});

// ---------------------------------------------------------------- drawings whose 3D view is already made

test('the list of made 3D views is read defensively', () => {
  const made = cleanMadeScenes({
    drawings: [
      { drawing: 4, pieces: [4, 5], title: 'Monoclinic  cell', steps: 8, synced: true, times: [189.4, 14.0, 'x', 49.3] },
      { drawing: 7, pieces: [7], title: 'Hexagonal prism', steps: 12, synced: false, times: [1, 2] }, // not synced: no times
      { drawing: 9, pieces: [], title: '', steps: 3, synced: true, times: [] }, // synced with no times: not synced
      { drawing: 4, pieces: [], title: 'again', steps: 1, synced: false, times: [] }, // the drawing is listed already
      { drawing: 'x' },
      null,
      'nope',
    ],
  });
  assert.deepEqual(made, [
    { drawing: 4, pieces: [4, 5], title: 'Monoclinic cell', steps: 8, synced: true, times: [14.0, 49.3, 189.4] },
    { drawing: 7, pieces: [7], title: 'Hexagonal prism', steps: 12, synced: false, times: [] },
    { drawing: 9, pieces: [], title: '', steps: 3, synced: false, times: [] },
  ]);
  // an older server (its 404 body), a slide lecture, nothing at all: no list
  assert.deepEqual(cleanMadeScenes({ detail: 'Not Found' }), []);
  assert.deepEqual(cleanMadeScenes({ drawings: [] }), []);
  assert.deepEqual(cleanMadeScenes(null), []);
  assert.deepEqual(cleanMadeScenes([{ drawing: 1 }]), []);
});

test('"Watch in 3D" from a second before the first step to five seconds after the last, on one drawing at a time', () => {
  const made = cleanMadeScenes({
    drawings: [
      { drawing: 4, pieces: [4], title: 'a', steps: 3, synced: true, times: [100, 120, 150] },
      { drawing: 7, pieces: [7], title: 'b', steps: 2, synced: true, times: [140, 160] },
      { drawing: 9, pieces: [9], title: 'c', steps: 2, synced: false, times: [] },
    ],
  });
  const at = (t: number) => explainedAt(made, t).map((m) => m.drawing);
  assert.deepEqual(at(98.9), []);
  assert.deepEqual(at(100 - PILL_BEFORE), [4]);
  assert.deepEqual(at(130), [4]);
  // both explained: the one whose step came last (7 at 140, then 4 at 150)
  assert.deepEqual(at(139), [7]);
  assert.deepEqual(at(145), [7]);
  assert.deepEqual(at(149), [4]);
  assert.deepEqual(at(150 + PILL_AFTER), [4]);
  assert.deepEqual(at(155.1), [7]);
  // a drawing not on the board then never takes the pill from one that is
  assert.deepEqual(explainedAt(made, 145, (m) => m.drawing !== 7).map((m) => m.drawing), [4]);
  assert.deepEqual(explainedAt(made, 145, () => false), []);
  assert.deepEqual(at(165.1), []);
  assert.deepEqual(at(NaN), []);
});

// The badge of a made view covers as little writing as it can (test2 #4: inside the drawing's top-right corner it
// covered the edge label "b").
const inkOf = (ink: Box[]) => (r: Box) => ink.reduce((n, b) => n + boxOverlap(r, b), 0);

test('the area two boxes share', () => {
  assert.equal(boxOverlap([0, 0, 10, 10], [5, 5, 10, 10]), 25);
  assert.equal(boxOverlap([0, 0, 10, 10], [10, 0, 10, 10]), 0);
  assert.equal(boxOverlap([0, 0, 10, 10], [20, 20, 1, 1]), 0);
  assert.equal(boxOverlap([2, 2, 2, 2], [0, 0, 10, 10]), 4);
});

test("a made view's badge goes beside the drawing where there is no writing, never off the board", () => {
  const board: [number, number] = [800, 400];
  const drawing: Box = [100, 200, 100, 120];
  const badge: [number, number] = [60, 20];
  // nothing around: just above the drawing's top-right corner (right edges lined up)
  assert.deepEqual(badgeSpot(drawing, board, badge, 4, inkOf([drawing])), { box: [140, 176, 60, 20], align: 'right' });
  // writing above it: beside its top-right corner
  // (growing away from the drawing)
  assert.deepEqual(badgeSpot(drawing, board, badge, 4, inkOf([drawing, [90, 150, 200, 45]])), { box: [204, 200, 60, 20], align: 'left' });
  // writing above and beside: under it
  assert.deepEqual(badgeSpot(drawing, board, badge, 4, inkOf([drawing, [90, 150, 200, 45], [204, 190, 100, 60]])), { box: [140, 324, 60, 20], align: 'right' });
  // above and under are off the board (a drawing as tall as the board): beside it
  assert.deepEqual(badgeSpot([100, 0, 100, 400], board, badge, 4, inkOf([])), { box: [204, 0, 60, 20], align: 'left' });
  // nowhere clear: the spot that covers the least writing, here inside its own corner as before
  const inside = (r: Box) => r[0] >= 100 && r[1] >= 200 && r[0] + r[2] <= 200 && r[1] + r[3] <= 320;
  assert.deepEqual(badgeSpot(drawing, board, badge, 4, (r) => (inside(r) ? 5 : 10)), { box: [136, 204, 60, 20], align: 'right' });
  // a drawing at the board's right edge: never off the board
  const s = badgeSpot([700, 100, 100, 100], board, badge, 4, inkOf([[700, 100, 100, 100], [600, 60, 200, 40]])).box;
  assert.ok(s[0] >= 0 && s[1] >= 0 && s[0] + s[2] <= 800 && s[1] + s[3] <= 400, JSON.stringify(s));
  // a badge bigger than the board still gets a place on it
  const big = badgeSpot([0, 0, 10, 10], [50, 20], [80, 30], 2, inkOf([])).box;
  assert.ok(big[0] === 0 && big[1] === 0, JSON.stringify(big));
});

test('"Watch in 3D" is written out only where it covers no more writing than the small badge', () => {
  const board: [number, number] = [800, 400];
  const drawing: Box = [100, 200, 100, 120];
  // room above the drawing for both
  const free = badgePlaces(drawing, board, [40, 20], [100, 24], 4, inkOf([drawing]));
  assert.equal(free.long, true);
  assert.deepEqual(free.badge, { box: [160, 176, 40, 20], align: 'right' });
  assert.deepEqual(free.pill, { box: [100, 172, 100, 24], align: 'right' });
  // test2 #4: writing above and beside leaves room for the badge only (on its left), so the pill stays the badge's
  // size and place (lit up, not written out) instead of covering the writing above
  const tight = badgePlaces([72, 206, 111, 169], [800, 391], [54, 24], [123, 27], 3.5,
    inkOf([[72, 206, 111, 169], [95, 150, 91, 45], [187, 190, 140, 30]]));
  assert.equal(tight.long, false);
  assert.deepEqual(tight.badge, { box: [14.5, 206, 54, 24], align: 'right' });
  assert.deepEqual(tight.pill, tight.badge);
});

test('a made drawing is on the board as soon as any of its pieces is', () => {
  // synth3d-saddle: the drawing's anchor (its label, piece 23) is written at 40.7 s, its curves (8, 12, 17) from 13 s;
  // three of its five steps came before the badge did
  const m = { drawing: 23, pieces: [23, 8, 12, 17] };
  const drawnAt: Record<number, number> = { 23: 40.67, 8: 13, 12: 23, 17: 31.7 };
  const on = (t: number) => drawingOnBoard(m, (id) => t >= (drawnAt[id] ?? Infinity));
  assert.equal(on(12), false);
  assert.equal(on(13), true);
  assert.equal(on(21.15), true);
  assert.equal(drawingOnBoard({ drawing: 5, pieces: [] }, (id) => id === 5), true);
  assert.equal(drawingOnBoard({ drawing: 5, pieces: [5] }, () => false), false);
});
