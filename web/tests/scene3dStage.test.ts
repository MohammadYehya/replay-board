// The 3D view's stage geometry that can be checked without a browser (three.js geometry works in Node; nothing here
// draws). Every shape is made up for the test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Vector3 } from 'three';
import { dashesGeom, DASH, GAP } from '../src/player/scene3dStage.ts';
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
