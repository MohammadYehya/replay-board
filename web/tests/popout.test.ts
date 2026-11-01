// Pop-out helpers: sampling y = f(x) around asymptotes and jumps, ticks, ink colours, and which pieces get 3D.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fitRange, formatTick, niceStep, niceTicks, sampleSegments, stageInk } from '../src/player/plotMath.ts';
import { canPopout, looksLikeGraphEquation } from '../src/player/popoutable.ts';
import type { Piece } from '../src/types.ts';

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

const piece = (kind: string, text = '', latex = ''): Piece => ({
  id: 1, box: [0, 0, 10, 10], image: 'p.png', t_seen: 0, t_shown: 0, read: { kind, text, latex },
});

test('which pieces get a 3D view', () => {
  for (const k of ['graph', 'diagram', 'figure', 'plot', 'sketch', 'drawing', 'chart', 'curve', 'Graph', 'graphs'])
    assert.ok(canPopout(piece(k)), k);
  for (const k of ['text', 'arrow', 'number', 'label', 'other', 'scribble', 'box', 'underline'])
    assert.ok(!canPopout(piece(k, 'y = ln x', 'y = \\ln x')), k);
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
