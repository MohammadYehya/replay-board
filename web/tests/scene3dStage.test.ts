// The 3D view's stage geometry that can be checked without a browser (three.js geometry works in Node; nothing here
// draws). Every shape is made up for the test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CylinderGeometry, Mesh, Vector3 } from 'three';
import { axisScales, CURVE_LABEL_AT, dashesGeom, DASH, EMPH, GAP, placeRod } from '../src/player/scene3dStage.ts';
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
