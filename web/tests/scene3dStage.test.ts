// The 3D view's stage geometry that can be checked without a browser (three.js geometry works in Node; nothing here
// draws). Every shape is made up for the test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CylinderGeometry, Mesh, Vector3 } from 'three';
import {
  axisScales, cornerInsets, cornerSide, CURVE_LABEL_AT, dashesGeom, DASH, EMPH, fitDistanceOf, fullViewOf, GAP, placeRod,
  type Insets,
} from '../src/player/scene3dStage.ts';
import { sampleSegments } from '../src/player/plotMath.ts';

/** The x of every vertex the index uses (the faces actually drawn). */
function drawnXs(g: NonNullable<ReturnType<typeof dashesGeom>>): number[] {
  const pos = g.getAttribute('position');
  const idx = g.index!;
  const xs: number[] = [];
  for (let k = 0; k < idx.count; k++) xs.push(pos.getX(idx.getX(k)));
  return xs;
}

test('a dashed line draws dashes and leaves the gaps empty', () => {
  const period = DASH + GAP;
  const g = dashesGeom([new Vector3(0, 0, 0), new Vector3(10 * period, 0, 0)], 0.016)!;
  assert.ok(g);
  const xs = drawnXs(g);
  const inGap = xs.filter((x) => {
    const m = x % period;
    return m > DASH + 0.01 && m < period - 0.01;
  });
  assert.equal(inGap.length, 0, 'nothing is drawn in a gap');
  for (let i = 0; i < 10; i++) {
    const [a, b] = [i * period, i * period + DASH];
    assert.ok(xs.some((x) => x >= a - 1e-6 && x <= b + 1e-6), `dash ${i + 1} is drawn`);
  }
  // in path order: a draw range draws it from the start
  assert.ok(xs[0] < 0.05 && xs[xs.length - 1] > 9 * period);
});

test('a dashed wiggly graph stays cheap to build (it is rebuilt on every frame a parameter moves it)', () => {
  // sin(200x) over [-5, 5], drawn about 2.4 wide: some 30,000 sample points
  const pts = sampleSegments((x) => Math.sin(200 * x), -5, 5, -1.2, 1.2, 600)[0].map(
    ([x, y]) => new Vector3(x * 0.24, y * 0.35, 0),
  );
  assert.ok(pts.length > 20000);
  const t0 = performance.now();
  const g = dashesGeom(pts, 0.016)!;
  const ms = performance.now() - t0;
  assert.ok(g);
  assert.ok(g.getAttribute('position').count < 150000, `${g.getAttribute('position').count} vertices`);
  assert.ok(ms < 5000, `${Math.round(ms)} ms`); // a rod per dash, merged, took 15 to 25 s here
  g.dispose();
});

test('a dashed line too short or with one point draws nothing, without throwing', () => {
  assert.equal(dashesGeom([new Vector3(1, 1, 1)], 0.016), null);
  assert.equal(dashesGeom([new Vector3(1, 1, 1), new Vector3(1, 1, 1)], 0.016), null);
  assert.equal(dashesGeom([], 0.016), null);
});

test('a rod is placed by its matrix: the unit cylinder from one end to the other, at the radius', () => {
  const rod = new Mesh(new CylinderGeometry(1, 1, 1, 12, 1, false));
  const a = new Vector3(1, 2, 3);
  const b = new Vector3(4, 6, 3);
  assert.equal(placeRod(rod, a, b, 0.02), true);
  const top = new Vector3(0, 0.5, 0).applyMatrix4(rod.matrix);
  const bottom = new Vector3(0, -0.5, 0).applyMatrix4(rod.matrix);
  assert.ok(top.distanceTo(b) < 1e-9, 'its top is at b');
  assert.ok(bottom.distanceTo(a) < 1e-9, 'its bottom is at a');
  const side = new Vector3(1, 0, 0).applyMatrix4(rod.matrix).sub(new Vector3(0, 0, 0).applyMatrix4(rod.matrix));
  assert.ok(Math.abs(side.length() - 0.02) < 1e-9, 'its radius');
  // moved: a new matrix, the same geometry
  const g = rod.geometry;
  assert.equal(placeRod(rod, b, a, 0.02), true);
  assert.ok(new Vector3(0, 0.5, 0).applyMatrix4(rod.matrix).distanceTo(a) < 1e-9);
  assert.equal(rod.geometry, g);
  // ends too close, or not finite: not placed
  assert.equal(placeRod(rod, a, a.clone(), 0.02), false);
  assert.equal(placeRod(rod, a, new Vector3(NaN, 0, 0), 0.02), false);
});

test('a tall surface over x and y gets its z axis drawn as long as x and y', () => {
  // synth3d-saddle: x, y in +-2, z in +-4 drew a narrow shape twice as tall as wide
  const k = axisScales([4, 4, 8], true, true);
  assert.ok(Math.abs(k[0] * 4 - k[2] * 8) < 1e-9 && Math.abs(k[0] - k[1]) < 1e-9);
  // anything else: one scale (a cube stays a cube), a far shorter shown axis stretched to 35% of the longest
  const cube = axisScales([2, 2, 4], false, true);
  assert.ok(Math.abs(cube[0] - cube[2]) < 1e-9);
  const graph = axisScales([10, 1, 0], true, false);
  assert.ok(Math.abs(graph[1] * 1 - 0.35 * graph[0] * 10) < 1e-9);
  const flatSurface = axisScales([4, 4, 2], true, true);
  assert.ok(Math.abs(flatSurface[0] - flatSurface[2]) < 1e-9);
});

test('a step emphasises its lines and names curves away from their middle', () => {
  assert.equal(EMPH, 1.3);
  assert.ok(CURVE_LABEL_AT >= 0.8 && CURVE_LABEL_AT <= 0.85);
});

// "Try it" at 1368 x 825: the stage is 702 x 340, the sliders cover its left 256 px and the caption its bottom 60;
// Reset view covers the top right corner, 111 px in and 44 down.
const STAGE = { w: 702, h: 340, fov: 36 };
const EDGES: Insets = { top: 0, right: 0, bottom: 60, left: 256 };
const RESET = { width: 111, height: 44 };
const box = (x: number, y: number) => {
  const pts: Vector3[] = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) pts.push(new Vector3(sx * x, sy * y, sz * 0.3));
  return pts;
};
const FRONT = new Vector3(0, 0, 1);
const O = new Vector3();
/** Pixels per world unit with the corner kept clear from that side. */
const scaleFrom = (pts: Vector3[], side: 'top' | 'right') => {
  const ins = cornerInsets(EDGES, RESET, side);
  return fullViewOf(STAGE.w, STAGE.h, ins).h / fitDistanceOf(pts, O, FRONT, STAGE.fov, STAGE.w, STAGE.h, ins);
};
const sideFor = (pts: Vector3[], current: 'top' | 'right' = 'top') =>
  cornerSide(pts, O, FRONT, STAGE.fov, STAGE.w, STAGE.h, EDGES, RESET, current);

test('a tall drawing passes Reset view on its left, a wide one below it: whichever shows it larger', () => {
  // test2's tall unit cell: a band across the top took 22% of its size; one down the right costs it nothing
  const tall = box(0.5, 1.2);
  assert.equal(sideFor(tall), 'right');
  assert.ok(scaleFrom(tall, 'right') > scaleFrom(tall, 'top') * 1.1);
  const free = fullViewOf(STAGE.w, STAGE.h, EDGES).h / fitDistanceOf(tall, O, FRONT, STAGE.fov, STAGE.w, STAGE.h, EDGES);
  assert.ok(Math.abs(scaleFrom(tall, 'right') - free) < 1e-9, 'as large as with nothing over the corner');
  // a wide, flat drawing: the band down the right would shrink it, the one across the top does not
  const wide = box(1.6, 0.3);
  assert.equal(sideFor(wide), 'top');
  assert.equal(sideFor(wide, 'right'), 'top');
  assert.ok(scaleFrom(wide, 'top') > scaleFrom(wide, 'right'));
  // the band is the corner box's: across the top as deep as it reaches down, or down the right as wide as it reaches in
  assert.deepEqual(cornerInsets(EDGES, RESET, 'top'), { ...EDGES, top: 44 });
  assert.deepEqual(cornerInsets(EDGES, RESET, 'right'), { ...EDGES, right: 111 });
  assert.deepEqual(cornerInsets({ ...EDGES, top: 70 }, RESET, 'top'), { ...EDGES, top: 70 }); // already clear
});

test('the side changes only when the other one is clearly better, so the drawing does not hop as a slider moves', () => {
  // a drawing about as large either way: it stays on the side it is on
  let near: Vector3[] | null = null;
  for (let y = 0.3; y <= 1.6 && !near; y += 0.01) {
    const pts = box(0.8, y);
    const r = scaleFrom(pts, 'right') / scaleFrom(pts, 'top');
    if (r > 1 && r < 1.02) near = pts;
  }
  assert.ok(near, 'a drawing the two sides show within 2% of each other');
  assert.equal(sideFor(near!, 'top'), 'top');
  assert.equal(sideFor(near!, 'right'), 'right');
  // nothing over the corner: nothing to choose
  assert.equal(cornerSide(box(0.5, 1.2), O, FRONT, STAGE.fov, STAGE.w, STAGE.h, EDGES, { width: 0, height: 0 }, 'top'), 'top');
});
