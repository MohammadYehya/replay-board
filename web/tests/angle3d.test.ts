// The "angle" object of a 3D scene: its value, its arc or right-angle mark, its label on the bisector. Every angle here
// is made up for the test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { angleReach, angleShape, measureAngle, AUTO_RADIUS, LABEL_OUT, RIGHT_SIDE } from '../src/player/angle3d.ts';
import type { Vec3 } from '../src/types.ts';

const near = (a: number, b: number, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);
const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const deg = (d: number) => (d * Math.PI) / 180;

test('the angle at the vertex between its two arms, 0..180 degrees', () => {
  near(measureAngle([0, 0, 0], [1, 0, 0], [0, 1, 0])!, 90);
  near(measureAngle([1, 1, 1], [3, 1, 1], [1, 1, 5])!, 90);
  near(measureAngle([0, 0, 0], [2, 0, 0], [Math.cos(deg(75)), 0, Math.sin(deg(75))])!, 75, 1e-9);
  near(measureAngle([0, 0, 0], [1, 0, 0], [-1, 0, 0])!, 180);
  near(measureAngle([0, 0, 0], [1, 0, 0], [3, 0, 0])!, 0);
  near(measureAngle([0, 0, 0], [1, 0, 0], [Math.cos(deg(120)), Math.sin(deg(120)), 0])!, 120, 1e-9);
});

test('an arc in the plane of the arms, from one arm to the other, at the radius', () => {
  // beta of a monoclinic cell: between a (along x) and c (leaning in the x-z plane)
  const v: Vec3 = [0, 0, 0];
  const c: Vec3 = [5 * Math.cos(deg(75)), 0, 5 * Math.sin(deg(75))];
  const s = angleShape(v, [2.5, 0, 0], c)!;
  assert.ok(s);
  assert.equal(s.right, false);
  near(s.degrees, 75, 1e-9);
  near(s.radius, AUTO_RADIUS * 2.5);
  for (const p of s.line) {
    near(dist(p, v), s.radius, 1e-9);
    near(p[1], 0, 1e-12); // in the plane of the arms
  }
  near(s.line[0][0], s.radius, 1e-12); // starts on the first arm
  const last = s.line[s.line.length - 1];
  near(Math.atan2(last[2], last[0]), deg(75), 1e-9); // ends on the second
  assert.equal(s.fan[0], v);
  assert.equal(s.fan.length, s.line.length + 1);
  // the label: on the bisector, LABEL_OUT x the radius out
  near(dist(s.label, v), LABEL_OUT * s.radius, 1e-9);
  near(Math.atan2(s.label[2], s.label[0]), deg(37.5), 1e-9);
});

test('a given radius wins over the auto one; a radius that is not positive is degenerate', () => {
  const s = angleShape([0, 0, 0], [4, 0, 0], [0, 0, 4], 0.5)!;
  near(s.radius, 0.5);
  assert.equal(angleShape([0, 0, 0], [4, 0, 0], [0, 4, 0], 0), null);
  assert.equal(angleShape([0, 0, 0], [4, 0, 0], [0, 4, 0], -1), null);
  assert.equal(angleShape([0, 0, 0], [4, 0, 0], [0, 4, 0], NaN), null);
});

test('within half a degree of 90: a right-angle mark, a square of 0.7 x the radius in the plane of the arms', () => {
  const v: Vec3 = [1, 2, 3];
  for (const d of [90, 89.6, 90.4]) {
    const s = angleShape(v, [4, 2, 3], [1, 2 + 3 * Math.cos(deg(d)), 3 + 3 * Math.sin(deg(d))])!;
    assert.equal(s.right, true, `${d} degrees`);
    assert.equal(s.line.length, 3);
    const side = RIGHT_SIDE * s.radius;
    near(dist(s.line[0], v), side, 1e-9);
    near(dist(s.line[2], v), side, 1e-9);
    near(dist(s.line[1], s.line[0]), side, 1e-9);
    for (const p of s.line) near(p[0] - 1 >= -1e-12 ? 0 : 1, 0); // the mark is on the arms' side, in their plane
  }
  assert.equal(angleShape([0, 0, 0], [1, 0, 0], [Math.cos(deg(89.4)), Math.sin(deg(89.4)), 0])!.right, false);
  assert.equal(angleShape([0, 0, 0], [1, 0, 0], [Math.cos(deg(90.6)), Math.sin(deg(90.6)), 0])!.right, false);
});

test('degenerate: an arm too short or anything not finite is not drawn', () => {
  assert.equal(angleShape([0, 0, 0], [0, 0, 0], [1, 0, 0]), null);
  assert.equal(angleShape([0, 0, 0], [1, 0, 0], [1e-10, 0, 0]), null);
  assert.equal(angleShape([0, 0, 0], [NaN, 0, 0], [1, 0, 0]), null);
  assert.equal(angleShape([0, 0, Infinity], [1, 0, 0], [0, 1, 0]), null);
});

test('arms in one line still give a shape: a half circle at 180, nothing to fill at 0', () => {
  const flat = angleShape([0, 0, 0], [2, 0, 0], [-2, 0, 0])!;
  near(flat.degrees, 180);
  for (const p of flat.line) near(dist(p, [0, 0, 0]), flat.radius, 1e-9);
  near(flat.line[flat.line.length - 1][0], -flat.radius, 1e-9);
  const none = angleShape([0, 0, 0], [2, 0, 0], [3, 0, 0])!;
  near(none.degrees, 0);
  assert.deepEqual(none.fan, []);
});

test('framing reaches the vertex +/- the radius', () => {
  const s = angleShape([1, 1, 1], [3, 1, 1], [1, 3, 1])!;
  const r = s.radius;
  const pts = angleReach(s);
  assert.equal(pts.length, 6);
  for (let a = 0; a < 3; a++) {
    near(Math.min(...pts.map((p) => p[a])), 1 - r);
    near(Math.max(...pts.map((p) => p[a])), 1 + r);
  }
});
