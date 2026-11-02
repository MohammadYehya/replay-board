// Where the 3D view's labels go: never on each other, the important ones first; an angle's name out along its
// bisector. Every box and arc is made up for the test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { arcLabelAt, circumcentre, placeLabels, type LabelBox } from '../src/player/labelLayout.ts';

const box = (x: number, y: number, w = 40, h = 18, movable = true): LabelBox => ({ x, y, w, h, movable });
const STAGE = { width: 600, height: 400 };

function overlaps(boxes: LabelBox[], places: ReturnType<typeof placeLabels>, pad = 2) {
  const shown = boxes
    .map((b, i) => ({ x: b.x + places[i].dx, y: b.y + places[i].dy, w: b.w, h: b.h, dim: places[i].dim }))
    .filter((r) => !r.dim);
  let n = 0;
  for (let i = 0; i < shown.length; i++)
    for (let j = i + 1; j < shown.length; j++) {
      const a = shown[i];
      const b = shown[j];
      if (a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad) n++;
    }
  return n;
}

test('labels apart stay where they are', () => {
  const boxes = [box(10, 10), box(100, 10), box(10, 100)];
  assert.deepEqual(placeLabels(boxes, STAGE), boxes.map(() => ({ dx: 0, dy: 0, dim: false })));
});

test('a label on another moves the least it can; the first one keeps its place', () => {
  const boxes = [box(100, 100), box(110, 104)];
  const [a, b] = placeLabels(boxes, STAGE);
  assert.deepEqual(a, { dx: 0, dy: 0, dim: false });
  assert.equal(b.dim, false);
  assert.equal(overlaps(boxes, [a, b]), 0);
  // the smallest move clears it: 4 px lower needs 16 more to clear a box 18 high (pad 2), sideways would be 32
  assert.deepEqual([b.dx, b.dy], [0, 16]);
});

test('five names piled on one corner (alpha, beta, gamma, a, c) all end up readable', () => {
  const boxes = [box(200, 200, 44), box(204, 203, 38), box(198, 206, 50), box(206, 199, 14), box(201, 204, 14)];
  const places = placeLabels(boxes, STAGE);
  assert.equal(overlaps(boxes, places), 0);
  assert.ok(places.every((p) => !p.dim));
  assert.ok(places.every((p) => Math.hypot(p.dx, p.dy) <= 44), 'none is far from where it belongs');
});

test('a label that cannot move (an axis number) is dimmed instead, and so is one with no free spot near', () => {
  const ticks = [box(100, 100, 20, 14), box(105, 102, 20, 14, false)];
  assert.deepEqual(placeLabels(ticks, STAGE)[1], { dx: 0, dy: 0, dim: true });
  // a label boxed in on every side within reach
  const ring: LabelBox[] = [];
  for (let dy = -60; dy <= 60; dy += 20) for (let dx = -84; dx <= 84; dx += 42) ring.push(box(300 + dx, 200 + dy));
  const crowd = [...ring, box(300, 200)];
  const places = placeLabels(crowd, { ...STAGE, reach: 30 });
  assert.equal(places[places.length - 1].dim, true);
  assert.equal(overlaps(crowd, places), 0);
});

test('a label inside the stage stays inside it', () => {
  const boxes = [box(0, 0, 60, 18), box(2, 2, 60, 18)];
  const [, b] = placeLabels(boxes, STAGE);
  assert.ok(!b.dim && b.dy > 0 && 2 + b.dx >= 0, JSON.stringify(b)); // below the first: above would leave the stage
});

test('last frame\'s nudge is kept while its spot is free, so labels do not hop from side to side', () => {
  const boxes = [box(100, 100), box(110, 104)];
  const prev = [undefined, { dx: 0, dy: -24, dim: false }];
  assert.deepEqual(placeLabels(boxes, { ...STAGE, prev })[1], { dx: 0, dy: -24, dim: false });
  // when it is no longer needed, the label goes back to its own place
  assert.deepEqual(placeLabels([box(100, 100), box(300, 104)], { ...STAGE, prev })[1], { dx: 0, dy: 0, dim: false });
});

test('many labels stay quick to lay out (it runs on every frame that is drawn)', () => {
  const boxes: LabelBox[] = [];
  for (let i = 0; i < 80; i++) boxes.push(box(200 + (i % 9) * 6, 150 + Math.floor(i / 9) * 5, 30 + (i % 4) * 8, 16, i % 3 > 0));
  const t0 = performance.now();
  for (let k = 0; k < 10; k++) placeLabels(boxes, STAGE);
  const ms = (performance.now() - t0) / 10;
  assert.ok(ms < 20, `${ms.toFixed(1)} ms`);
  assert.equal(overlaps(boxes, placeLabels(boxes, STAGE)), 0);
});

// ---------------------------------------------------------------- an angle's name

function arc(c: [number, number, number], r: number, u: number[], v: number[], from: number, to: number, n = 40) {
  const pts: [number, number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const a = ((from + ((to - from) * i) / n) * Math.PI) / 180;
    pts.push([0, 1, 2].map((k) => c[k] + r * (Math.cos(a) * u[k] + Math.sin(a) * v[k])) as [number, number, number]);
  }
  return pts;
}

test('the circle through three points', () => {
  const c = circumcentre([1, 0, 0], [0, 1, 0], [-1, 0, 0])!;
  c.forEach((v, i) => assert.ok(Math.abs(v - [0, 0, 0][i]) < 1e-12));
  assert.equal(circumcentre([0, 0, 0], [1, 1, 1], [2, 2, 2]), null);
});

test('an angle\'s arc names it out along its bisector, 1.35 times its radius from the corner', () => {
  // a 90-degree arc of radius 0.8 round the corner (1, 2, 3), in the plane of x and z
  const at = arcLabelAt(arc([1, 2, 3], 0.8, [1, 0, 0], [0, 0, 1], 0, 90))!;
  const k = 0.8 * 1.35 * Math.SQRT1_2;
  [1 + k, 2, 3 + k].forEach((v, i) => assert.ok(Math.abs(at[i] - v) < 1e-9, `${at} ${i}`));
  // a 120-degree arc: the bisector at 60 degrees
  const at2 = arcLabelAt(arc([0, 0, 0], 1, [1, 0, 0], [0, 1, 0], 0, 120))!;
  assert.ok(Math.abs(Math.atan2(at2[1], at2[0]) - Math.PI / 3) < 1e-9 && Math.abs(Math.hypot(at2[0], at2[1]) - 1.35) < 1e-9);
  // three arcs from one corner put their names in three places
  const names = [
    arcLabelAt(arc([0, 0, 0], 0.9, [1, 0, 0], [0, 1, 0], 0, 90))!,
    arcLabelAt(arc([0, 0, 0], 1.1, [1, 0, 0], [0, 0, 1], 0, 90))!,
    arcLabelAt(arc([0, 0, 0], 0.7, [0, 1, 0], [0, 0, 1], 0, 90))!,
  ];
  for (let i = 0; i < 3; i++)
    for (let j = i + 1; j < 3; j++) assert.ok(Math.hypot(...names[i].map((v, k) => v - names[j][k])) > 0.6);
});

test('a curve that is not an arc, or a whole circle, keeps its label at its middle', () => {
  const parabola = Array.from({ length: 41 }, (_, i) => [i / 10 - 2, (i / 10 - 2) ** 2, 0] as [number, number, number]);
  assert.equal(arcLabelAt(parabola), null);
  assert.equal(arcLabelAt(arc([0, 0, 0], 1, [1, 0, 0], [0, 1, 0], 0, 360)), null);
  assert.equal(arcLabelAt(arc([0, 0, 0], 1, [1, 0, 0], [0, 1, 0], 0, 90, 3)), null); // too few points to tell
  // a line
  assert.equal(arcLabelAt(Array.from({ length: 10 }, (_, i) => [i, 0, 0] as [number, number, number])), null);
});
