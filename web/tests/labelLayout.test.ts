// Where the 3D view's labels go: never on each other, the important ones first; an angle's name out along its
// bisector. Every box and arc is made up for the test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  arcLabelAt, besideCenter, circumcentre, glide, KEEP_SLACK, originCenter, placeLabels, tickCenter, tickSide, TICK_GAP,
  type Glide, type LabelBox,
} from '../src/player/labelLayout.ts';

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

test('a kept spot taken by a hair moves the label a hair, not to the spot nearest home and back (test2 #4 step 3)', () => {
  // The cell coming to rest, as the view measured it: "b" creeps right while beta's name, kept just right of it,
  // creeps left. Searching from home sent beta 31 px up once they touched; the layout at rest put it back.
  const avoid = [{ x: 313, y: 214, w: 10, h: 22 }, { x: 313, y: 214, w: 22, h: 29 }, { x: 313, y: 223, w: 45, h: 22 }];
  const opts = { width: 700, height: 390, avoid };
  const at = (k: number) => [
    box(348.7, 224.8, 20, 22), // a
    box(346.4 + 0.02 * k, 188.8, 21, 22), // b
    box(311.7, 184.9, 44, 22), // alpha
    box(338.0 - 0.01 * k, 196, 43, 22), // beta
    box(373.9, 216.3, 43, 22), // gamma
    box(311.1, 138.4, 20, 22), // c
  ];
  let prev = placeLabels(at(0), opts).map((q, i) =>
    i === 2 ? { dx: -15, dy: 0, dim: false } : i === 3 ? { dx: 346.4 + 21 + 2 - 338, dy: 0, dim: false } : i === 4 ? { dx: 0, dy: 4, dim: false } : q,
  );
  let worst = 0;
  for (let k = 1; k <= 20; k++) {
    const now = placeLabels(at(k), { ...opts, prev });
    assert.equal(overlaps(at(k), now), 0);
    worst = Math.max(worst, Math.hypot(now[3].dx - prev[3].dx, now[3].dy - prev[3].dy));
    prev = now;
  }
  assert.ok(worst < 0.5, `beta moved ${worst.toFixed(1)} px in one frame`);
  // and where it ends is where the layout at rest puts it: nothing left to slide
  const rest = placeLabels(at(20), opts)[3];
  assert.ok(Math.hypot(rest.dx - prev[3].dx, rest.dy - prev[3].dy) < 0.5, JSON.stringify([rest, prev[3]]));
});

test('a label keeps near its last spot only while that is about as near its own place: it never drifts far off', () => {
  // the saddle turning (step 3): "z = x²" is laid out first; the axis name "y" sits on it and slides left under it.
  // Kept beside "z = x²", it followed it 39 px right while a spot 9 px above its place was free, then slid back.
  const z = box(386.8, 155.8, 43, 22);
  const y = (k: number) => box(412.3 - 0.95 * k, 150.4 - 0.25 * k, 8, 17);
  let prev = [undefined, { dx: 386.8 + 43 + 2 - 412.3, dy: 0, dim: false }];
  for (let k = 0; k <= 20; k++) {
    const now = placeLabels([z, y(k)], { ...STAGE, prev })[1];
    const fresh = placeLabels([z, y(k)], STAGE)[1];
    const d = Math.hypot(now.dx, now.dy);
    assert.ok(!now.dim && d <= Math.hypot(fresh.dx, fresh.dy) + KEEP_SLACK + 1e-9, `frame ${k}: ${JSON.stringify(now)} vs ${JSON.stringify(fresh)}`);
    prev = [undefined, now];
  }
  // at the end it is a few pixels from where the layout at rest puts it (a short slide, not 39 px back)
  const rest = placeLabels([z, y(20)], STAGE)[1];
  assert.ok(Math.hypot(prev[1]!.dx - rest.dx, prev[1]!.dy - rest.dy) <= KEEP_SLACK, JSON.stringify([prev[1], rest]));
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

test("a label that has to move never lands on an angle's mark (test2 #7: beta's name went onto gamma's arc)", () => {
  // the second label must move: the smallest move (16 px down) is on an angle's wedge, so it goes up instead
  const boxes = [box(100, 100), box(110, 104)];
  const wedge = { x: 95, y: 118, w: 60, h: 30 };
  const [a, b] = placeLabels(boxes, { ...STAGE, avoid: [wedge] });
  assert.deepEqual(a, { dx: 0, dy: 0, dim: false });
  assert.equal(b.dim, false);
  assert.deepEqual([b.dx, b.dy], [0, -24]);
  // a label already where it belongs (its own place) stays there, even on a mark
  const [c] = placeLabels([box(100, 120)], { ...STAGE, avoid: [wedge] });
  assert.deepEqual(c, { dx: 0, dy: 0, dim: false });
  // nowhere free within reach but on marks: dimmed in its place
  const walls = [{ x: 0, y: 0, w: 600, h: 99 }, { x: 0, y: 123, w: 600, h: 277 }, { x: 0, y: 0, w: 99, h: 400 }, { x: 151, y: 0, w: 449, h: 400 }];
  const [, e] = placeLabels(boxes, { ...STAGE, avoid: walls });
  assert.deepEqual(e, { dx: 0, dy: 0, dim: true });
});

test('a label slides to its new place, starting and ending gently, never past it, and ends exactly on it (no hop)', () => {
  // test2 #4: placed afresh once the drawing was still, a label hopped up to 32 px in one frame
  const to = { dx: 0, dy: -32 };
  for (const fps of [60, 30]) {
    let at: Glide = { dx: 0, dy: 0, vx: 0, vy: 0 };
    let frames = 0;
    let biggest = 0;
    let first = 0;
    for (; frames < 120; frames++) {
      const g = glide(at, to, 1 / fps);
      const step = Math.hypot(g.dx - at.dx, g.dy - at.dy);
      if (!frames) first = step;
      biggest = Math.max(biggest, step);
      assert.ok(g.dy >= to.dy - 1e-9 && g.dy <= 0, 'never past its place');
      at = g;
      if (g.done) break;
    }
    assert.deepEqual([at.dx, at.dy, at.vx, at.vy], [0, -32, 0, 0], 'exactly on its place at the end, still');
    assert.ok(frames / fps <= 0.4, `on its place within 0.4 s (${frames} frames at ${fps} fps)`);
    assert.ok(first < 32 * 0.15, `a gentle start (${first.toFixed(1)} px in the first frame at ${fps} fps)`);
    assert.ok(biggest < 32 * 0.25, `no frame moves it more than a quarter of the way (${biggest.toFixed(1)} px)`);
  }
  // a label not on screen last frame, or no time to slide (reduced motion): straight to its place
  assert.deepEqual(glide(undefined, { dx: 5, dy: 6 }, 1 / 60), { dx: 5, dy: 6, vx: 0, vy: 0, done: true });
  assert.deepEqual(glide({ dx: 0, dy: 0, vx: 0, vy: 0 }, { dx: 5, dy: 6 }, 1 / 60, 0), { dx: 5, dy: 6, vx: 0, vy: 0, done: true });
  // already there, or within half a pixel and still: done
  assert.deepEqual(glide({ dx: 5, dy: 6.3, vx: 0, vy: 0 }, { dx: 5, dy: 6 }, 1 / 60), { dx: 5, dy: 6, vx: 0, vy: 0, done: true });
  // a long frame (a busy tab) covers more of the way, never past it
  const long = glide({ dx: 0, dy: 0, vx: 0, vy: 0 }, { dx: 100, dy: 0 }, 0.1);
  assert.ok(long.dx > 30 && long.dx < 100, `${long.dx}`);
  // the place changes while it moves: it turns smoothly towards the new one and gets there
  let at: Glide = { dx: 0, dy: 0, vx: 0, vy: 0 };
  for (let i = 0; i < 6; i++) at = glide(at, { dx: 40, dy: 0 }, 1 / 60);
  for (let i = 0; i < 60 && !glide(at, { dx: -10, dy: 0 }, 1 / 60).done; i++) at = glide(at, { dx: -10, dy: 0 }, 1 / 60);
  assert.deepEqual(glide(at, { dx: -10, dy: 0 }, 1 / 60).dx, -10);
});

test("an axis' numbers sit beside it as seen, below one across the screen and left of one up it", () => {
  const w = 14;
  const h = 12;
  // where the label's box is, from its centre (CSS2DObject.center) with the anchor (the tick) at 0, 0
  const boxAt = ([cx, cy]: [number, number]) => ({ x0: -cx * w, y0: -cy * h, x1: (1 - cx) * w, y1: (1 - cy) * h });
  // x across the screen: below it, centred, its top TICK_GAP under the axis
  const x = boxAt(tickCenter(300, 0, w, h));
  assert.ok(Math.abs(x.y0 - TICK_GAP) < 1e-9 && Math.abs(x.x0 + x.x1) < 1e-9);
  // seen from behind (x pointing left): still below
  assert.ok(Math.abs(boxAt(tickCenter(-300, 0, w, h)).y0 - TICK_GAP) < 1e-9);
  // y up the screen (a graph seen from the top): left of it, its right edge TICK_GAP off the axis, never across it
  // (test-full #3: "-2", "-4" sat on the y axis, under the dashed asymptote x = 0 drawn along it)
  const y = boxAt(tickCenter(0, -300, w, h));
  assert.ok(Math.abs(y.x1 + TICK_GAP) < 1e-9 && Math.abs(y.y0 + y.y1) < 1e-9);
  // and down the screen: left too
  assert.ok(Math.abs(boxAt(tickCenter(0, 300, w, h)).x1 + TICK_GAP) < 1e-9);
  // any slant: the box keeps TICK_GAP off the axis line on one side of it
  for (let deg = 0; deg < 360; deg += 5) {
    const d = [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)];
    const b = boxAt(tickCenter(d[0] * 200, d[1] * 200, w, h));
    const corners = [[b.x0, b.y0], [b.x1, b.y0], [b.x0, b.y1], [b.x1, b.y1]];
    const side = corners.map(([px, py]) => d[0] * py - d[1] * px); // signed distance from the axis line
    assert.ok(side.every((s) => s >= TICK_GAP - 1e-6) || side.every((s) => s <= -TICK_GAP + 1e-6), `${deg} degrees: ${side}`);
  }
  // an axis seen end on (no direction on screen): centred, as before
  assert.deepEqual(tickCenter(0, 0, w, h), [0.5, 0.5]);
  assert.equal(tickSide(0, 0), null);
  // the origin's number (kept on x) goes out between x's side and y's, clear of both axes (gest-test #10: the "0"
  // sat under the asymptote x = 0 running down the y axis past it)
  const xs = tickSide(300, 0)!;
  const ys = tickSide(0, -300)!;
  const o = boxAt(besideCenter(xs[0] + ys[0], xs[1] + ys[1], w, h));
  assert.ok(o.x1 <= -3 && o.y0 >= 3, `${JSON.stringify(o)}`); // left of the y axis and under the x axis
});

/** The box (relative to the anchor) a label w x h takes with the renderer's centre c, and how near it comes to rays
 *  from the anchor (each 600 px long), worked out by stepping along them. */
function originBox(c: [number, number], w: number, h: number) {
  const x = -c[0] * w;
  const y = -c[1] * h;
  return { x, y, w, h };
}
function nearestRay(b: { x: number; y: number; w: number; h: number }, rays: [number, number][]) {
  let best = Infinity;
  for (const [rx, ry] of rays) {
    const n = Math.hypot(rx, ry);
    for (let s = 0; s <= 600; s += 0.25) {
      const px = (rx / n) * s;
      const py = (ry / n) * s;
      const dx = Math.max(b.x - px, 0, px - (b.x + b.w));
      const dy = Math.max(b.y - py, 0, py - (b.y + b.h));
      best = Math.min(best, Math.hypot(dx, dy));
    }
  }
  return best;
}

test("the origin's number sits in a gap between all the axes as seen, clear of each, below-left when it can", () => {
  const w = 9;
  const h = 13;
  // graph paper: x across, y up, both running on past the origin: below-left of the crossing, 6 px off each axis
  const paper: [number, number][] = [[200, 0], [-200, 0], [0, -150], [0, 150]];
  const b = originBox(originCenter(paper, w, h), w, h);
  assert.ok(b.x + b.w <= -TICK_GAP + 1e-6 && b.y >= TICK_GAP - 1e-6, JSON.stringify(b));
  assert.ok(nearestRay(b, paper) >= TICK_GAP - 0.05);
  assert.ok(Math.hypot(b.x + b.w, b.y) < TICK_GAP * Math.SQRT2 + 3, 'near the crossing');
  // x and y from the origin only (a first quadrant): below-left still, out of the way of both
  const quad: [number, number][] = [[200, 0], [0, -150]];
  const q = originBox(originCenter(quad, w, h), w, h);
  assert.ok(q.x + q.w / 2 < 0 && q.y + q.h / 2 > 0 && nearestRay(q, quad) >= TICK_GAP - 0.05, JSON.stringify(q));
  // in space, seen from the iso corner: x to the lower left, y to the lower right, z up, each both ways
  const iso: [number, number][] = [[-160, 100], [160, -100], [180, 80], [-180, -80], [0, -170], [0, 170]];
  const i = originBox(originCenter(iso, w, h), w, h);
  assert.ok(nearestRay(i, iso) >= TICK_GAP - 0.05, JSON.stringify(i));
  // the saddle from the front: x across, z up, y running almost straight away (a short stub)
  const front: [number, number][] = [[200, 0], [-200, 0], [0, -160], [0, 160], [6, 3], [-6, -3]];
  const f = originBox(originCenter(front, w, h), w, h);
  assert.ok(nearestRay(f, front) >= TICK_GAP - 0.05, JSON.stringify(f));
  // nothing to keep clear of: below-left, as before
  assert.deepEqual(originCenter([], w, h), besideCenter(-Math.SQRT1_2, Math.SQRT1_2, w, h));
});
