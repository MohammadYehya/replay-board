// The 3D view's pure logic: reading a format-2 scene defensively, and the state at any moment of the video's clock.
// Every scene here is made up for the test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  beforeLine, chipCaption, clampToShape, cleanScene, drawingFrom, dwell, ease, fitToShape, FLAT_MIN, momentClock,
  openingCamera, playFrom, sceneLoader, liveNumberLabels, sceneState, seekTimeFor, shapeGuard, stepFraming, stepHeading,
  stepTimes, STEP_GAP, tryValue, drawnWith,
} from '../src/player/scene3dState.ts';
import { measureAngle } from '../src/player/angle3d.ts';
import type { Scene3D } from '../src/types.ts';

const near = (a: number, b: number, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

const obj = (id: string, type: string, extra: Record<string, unknown> = {}) => ({
  id, type, points: [], center: [], size: [], rotation: [], expr: '', x: '', y: '', z: '', range: [], range2: [],
  color: '#7aa7ff', opacity: 1, style: 'solid', label: '', ...extra,
});
const step = (t: number | null, extra: Record<string, unknown> = {}) => ({
  at: '', t, over: 2, caption: 'A step', show: [], hide: [], highlight: [], set: [], camera: 'keep', ...extra,
});

/** A made-up lesson: a box whose height grows, a leaning edge, a force that splits into two components. */
function rawScene(over: Record<string, unknown> = {}) {
  return {
    format: 2,
    version: 'v1',
    piece: 3,
    pieces: [3, 4],
    title: 'A box that grows and leans',
    explain: 'The height grows first, then the top edge leans.',
    synced: true,
    axes: { show: true, x: { min: -1, max: 4, label: 'x' }, y: { min: -1, max: 4, label: 'y' }, z: { min: 0, max: 5, label: 'z' } },
    params: [
      { name: 'h', value: 1, min: 1, max: 3, label: 'height', unit: '' },
      { name: 'lean', value: 90, min: 60, max: 120, label: 'lean angle', unit: 'deg' },
    ],
    objects: [
      obj('base', 'box', { center: ['1', '1', 'h/2'], size: ['2', '2', 'h'] }),
      obj('edge', 'segment', { points: [['0', '0', 'h'], ['2*cos(lean*deg)', '0', 'h + 2*sin(lean*deg)']] }),
      obj('force', 'vector', { points: [['0', '0', '0'], ['2', '0', '2']], label: 'F' }),
      obj('fx', 'vector', { points: [['0', '0', '0'], ['2', '0', '0']], label: 'Fx' }),
      obj('fz', 'vector', { points: [['0', '0', '0'], ['0', '0', '2']], label: 'Fz' }),
      obj('note', 'label', { points: [['3', '0', '3']], label: 'the top' }),
    ],
    steps: [
      step(100, { caption: 'The height grows', set: [{ param: 'h', to: 3 }], highlight: ['base'], camera: 'front' }),
      step(110, { caption: 'The edge leans', set: [{ param: 'lean', to: 110 }], show: ['edge'], highlight: ['edge'] }),
      step(120, { caption: 'The force splits', show: ['fx', 'fz'], hide: ['force'], camera: 'iso' }),
    ],
    bounds: [[-1, -1, 0], [4, 4, 5]],
    ...over,
  };
}

function scene(over: Record<string, unknown> = {}): Scene3D {
  const r = cleanScene(rawScene(over));
  assert.equal(r.kind, 'scene');
  return (r as { scene: Scene3D }).scene;
}

// ---------------------------------------------------------------- cleaning

test('a good scene comes through whole', () => {
  const s = scene();
  assert.equal(s.format, 2);
  assert.equal(s.synced, true);
  assert.equal(s.piece, 3);
  assert.deepEqual(s.pieces, [3, 4]);
  assert.equal(s.objects.length, 6);
  assert.equal(s.steps.length, 3);
  assert.deepEqual(s.params.map((p) => p.name), ['h', 'lean']);
  assert.equal(s.params[1].unit, 'deg');
  assert.deepEqual(s.bounds, [[-1, -1, 0], [4, 4, 5]]);
  assert.equal(s.axes.show, true);
  assert.deepEqual(s.steps.map((x) => x.t), [100, 110, 120]);
  assert.equal(s.isoFrom, 'right');
});

test("the iso camera's corner: left only when the server says so", () => {
  assert.equal(scene({ iso_from: 'left' }).isoFrom, 'left');
  assert.equal(scene({ iso_from: 'LEFT' }).isoFrom, 'right');
  assert.equal(scene({ iso_from: [-0.7, -1.2, 0.62] }).isoFrom, 'right');
});

test('an older server (format 1) or something unknown is told apart', () => {
  assert.deepEqual(cleanScene({ kind: 'solids', solids: [] }), { kind: 'old' });
  assert.deepEqual(cleanScene({ format: 1, objects: [] }), { kind: 'old' });
  assert.equal(cleanScene(null).kind, 'bad');
  assert.equal(cleanScene('scene').kind, 'bad');
  assert.equal(cleanScene([]).kind, 'bad');
  assert.equal(cleanScene({ title: 'x' }).kind, 'bad');
  assert.equal(cleanScene({ format: 'two' }).kind, 'bad');
  // a newer format is read for the fields this player knows
  assert.equal(cleanScene({ ...rawScene(), format: 3, extra: { anything: 1 } }).kind, 'scene');
});

test('malformed objects are dropped, and so are the steps\' mentions of them', () => {
  const s = scene({
    objects: [
      ...rawScene().objects,
      obj('Bad-Id', 'point', { points: [['0', '0', '0']] }),
      obj('teapot', 'teapot'),
      obj('evil', 'graph', { expr: 'alert(1)' }),
      obj('leak', 'graph', { expr: 'x + y' }), // a graph is in x only
      obj('leak2', 'curve', { x: 'u', y: 't', z: '0', range: ['0', '1'] }), // a curve is in t only
      obj('nosuch', 'point', { points: [['k', '0', '0']] }), // k is not a parameter
      obj('tri', 'polygon', { points: [['0', '0', '0'], ['1', '0', '0']] }), // too few corners
      obj('silent', 'label', { points: [['0', '0', '0']], label: '' }), // a label with no text
      obj('ball', 'sphere', { center: ['0', '0', '0'], size: [] }), // no radius
      obj('tube', 'curve', { x: 't', y: '0', z: '0' }), // no t range
      obj('sheet', 'psurface', { x: 'u', y: 'v', z: '0', range: ['0', '1'] }), // no v range
      obj('base', 'point', { points: [['0', '0', '0']] }), // the id is taken
      obj('dot', 'point', { points: [[1, 2, 3]] }), // numbers are expressions too
    ],
    steps: [step(10, { show: ['edge', 'evil', 'teapot', 'tri'], highlight: ['leak', 'base'], hide: ['nosuch'] })],
  });
  assert.deepEqual(
    s.objects.map((o) => o.id),
    ['base', 'edge', 'force', 'fx', 'fz', 'note', 'dot'],
  );
  assert.deepEqual(s.objects.find((o) => o.id === 'dot')!.points, [['1', '2', '3']]);
  assert.equal(s.objects.find((o) => o.id === 'base')!.type, 'box');
  assert.deepEqual(s.steps[0].show, ['edge']);
  assert.deepEqual(s.steps[0].highlight, ['base']);
  assert.deepEqual(s.steps[0].hide, []);
});

test('every object type with its fields', () => {
  const s = scene({
    objects: [
      obj('p', 'point', { points: [['1', '2', '3'], ['9', '9', '9']] }),
      obj('poly', 'polygon', { points: Array.from({ length: 15 }, (_, i) => [`cos(${i})`, `sin(${i})`, '0']) }),
      obj('cyl', 'cylinder', { center: ['0', '0', '0'], size: ['1', '2', '3'], rotation: ['90', '0'] }),
      obj('cone', 'cone', { center: ['0', '0', '0'], size: ['1', 'h'], rotation: ['0', '45', '0'] }),
      obj('ring', 'torus', { center: ['0', '0', '0'], size: ['2', '0.5'] }),
      obj('g', 'graph', { expr: 'h*x^2', range: ['-1'] }), // a bad range falls back to the x axis
      obj('surf', 'surface', { expr: 'sin(x) cos(y) h' }),
      obj('helix', 'curve', { x: 'cos(t)', y: 'sin(t)', z: 't/5', range: ['0', '4pi'] }),
      obj('sweep', 'psurface', { x: 'u cos(v)', y: 'u sin(v)', z: 'u^2', range: ['0', '1'], range2: ['0', 'lean*deg'] }),
    ],
    steps: [step(10)],
  });
  const by = (id: string) => s.objects.find((o) => o.id === id)!;
  assert.equal(by('p').points.length, 1);
  assert.equal(by('poly').points.length, 12);
  assert.deepEqual(by('cyl').size, ['1', '2']);
  assert.deepEqual(by('cyl').rotation, []); // malformed rotation: none
  assert.deepEqual(by('cone').rotation, ['0', '45', '0']);
  assert.deepEqual(by('g').range, []);
  assert.equal(by('surf').expr, 'sin(x) cos(y) h');
  assert.deepEqual(by('helix').range, ['0', '4pi']);
  assert.deepEqual(by('sweep').range2, ['0', 'lean*deg']);
});

test('styles, colours, opacity and labels are kept within limits', () => {
  const s = scene({
    objects: [
      obj('a', 'point', { points: [['0', '0', '0']], color: 'red', opacity: 0.01, style: 'glow', label: 'x'.repeat(50) }),
      obj('b', 'point', { points: [['0', '0', '0']], color: '#ABCDEF', opacity: 7, style: 'dashed' }),
      obj('c', 'point', { points: [['0', '0', '0']], opacity: 'half', style: 'wire' }),
    ],
    steps: [step(10)],
  });
  const [a, b, c] = s.objects;
  assert.equal(a.color, '');
  assert.equal(a.opacity, 0.15);
  assert.equal(a.style, 'solid');
  assert.ok(a.label.length <= 30 && a.label.endsWith('…'));
  assert.equal(b.color, '#abcdef');
  assert.equal(b.opacity, 1);
  assert.equal(b.style, 'dashed');
  assert.equal(c.opacity, 1);
  assert.equal(c.style, 'wire');
});

test('parameters: names that clash are dropped, the start value is inside the range', () => {
  const s = scene({
    params: [
      { name: 'x', value: 1, min: 0, max: 2 }, // a field variable
      { name: 'sin', value: 1, min: 0, max: 2 }, // a function
      { name: 'e', value: 1, min: 0, max: 2 }, // a constant
      { name: 'Beta', value: 1, min: 0, max: 2 }, // not lower case
      { name: 'a_very_long_name_indeed', value: 1 },
      { name: 'k', value: 'one' },
      { name: 'h', value: 5, min: 3, max: 1, label: '' }, // swapped range, value outside it
      { name: 'h', value: 2 }, // a second h
      { name: 'w2', value: 0 },
    ],
  });
  assert.deepEqual(s.params.map((p) => p.name), ['h', 'w2']);
  assert.deepEqual([s.params[0].min, s.params[0].max, s.params[0].label], [1, 5, 'h']);
  // objects that use a dropped parameter are dropped with it
  assert.ok(!s.objects.some((o) => o.id === 'edge'));
});

test('steps: limits, unknown targets and empty steps', () => {
  const s = scene({
    steps: [
      step(5, { over: 0.1, caption: 'y'.repeat(200), set: [{ param: 'h', to: 2 }, { param: 'h', to: 3 }, { param: 'nope', to: 1 }, { param: 'lean', to: 'far' }] }),
      step(6, { over: 99, camera: 'fisheye', show: ['edge'], hide: ['edge', 'force'] }),
      step(7, { caption: '', camera: 'keep' }), // nothing at all: dropped
      step(8, { over: 'slow', caption: 'ok' }),
    ],
  });
  assert.equal(s.steps.length, 3);
  const [a, b, c] = s.steps;
  assert.equal(a.over, 0.6);
  assert.ok(a.caption.length <= 140);
  assert.deepEqual(a.set, [{ param: 'h', to: 3 }]);
  assert.equal(b.over, 8);
  assert.equal(b.camera, 'keep');
  assert.deepEqual(b.show, ['edge']);
  assert.deepEqual(b.hide, ['force']); // shown and hidden at once: shown
  assert.equal(c.over, 1.5);
});

test('synced steps are kept in time order; steps without a time are dropped; at most 12', () => {
  const s = scene({
    steps: [step(30, { caption: 'c' }), step(null, { caption: 'n' }), step(10, { caption: 'a' }), step(20, { caption: 'b' })],
  });
  assert.equal(s.synced, true);
  assert.deepEqual(s.steps.map((x) => x.caption), ['a', 'b', 'c']);
  const many = scene({ steps: Array.from({ length: 20 }, (_, i) => step(i * 2, { caption: `s${i}` })) });
  assert.equal(many.steps.length, 12);
  assert.equal(many.steps[11].caption, 's11');
});

test('a scene whose steps carry no times is not synced, whatever it says', () => {
  const s = scene({ steps: [step(null), step(null)] });
  assert.equal(s.synced, false);
  const t = scene({ synced: false });
  assert.equal(t.synced, false);
  assert.ok(t.steps.every((x) => x.t === null));
});

test('bounds, axes and text fields are checked', () => {
  assert.equal(scene({ bounds: [[0, 0, 0], [1, 1]] }).bounds, null);
  assert.equal(scene({ bounds: [[2, 0, 0], [1, 1, 1]] }).bounds, null);
  assert.equal(scene({ bounds: 'big' }).bounds, null);
  assert.deepEqual(scene({ bounds: [[0, 0, 0], [1, 1, 0]] }).bounds, [[0, 0, 0], [1, 1, 0]]); // flat is fine
  const s = scene({ axes: { show: 'yes', x: { min: 3, max: 1 } }, title: 't'.repeat(200), explain: 42 });
  assert.equal(s.axes.show, false);
  assert.deepEqual(s.axes.x, { min: -5, max: 5, label: '' });
  assert.ok(s.title.length <= 80);
  assert.equal(s.explain, '');
});

test('nothing to show in 3D: no objects, the explanation says why', () => {
  const s = scene({ objects: [], steps: [], explain: 'It is a list of words, not a shape.' });
  assert.equal(s.objects.length, 0);
  assert.equal(s.steps.length, 0);
  assert.equal(s.explain, 'It is a list of words, not a shape.');
  assert.equal(scene({ objects: 'none' }).objects.length, 0);
});

// ---------------------------------------------------------------- state at a moment

test('before the first step: the base drawing, every parameter at its value', () => {
  const s = scene();
  for (const clock of [-Infinity, 0, 50, 99.99]) {
    const st = sceneState(s, { clock });
    assert.equal(st.index, -1);
    assert.deepEqual(st.params, { h: 1, lean: 90 });
    assert.deepEqual(st.vis, { base: 1, edge: 0, force: 1, fx: 0, fz: 0, note: 1 });
    assert.deepEqual(st.highlight, []);
    assert.equal(st.glow, 0);
    assert.equal(st.camera, 'front'); // the first preset a step names: the view does not jump at step 1
    assert.equal(st.cameraStep, -1);
    assert.equal(st.moving, false);
  }
});

test('the opening camera is the first one a step names, else the three-quarter view', () => {
  // a graph seen from the top opens from the top, not on empty 3D axes
  const top = scene({ steps: [step(100, { camera: 'keep' }), step(110, { camera: 'top' }), step(120, { camera: 'iso' })] });
  assert.equal(openingCamera(top), 'top');
  assert.equal(sceneState(top, { clock: 0 }).camera, 'top');
  assert.equal(sceneState(top, { clock: 105 }).camera, 'top');
  assert.equal(sceneState(top, { clock: 105 }).cameraStep, -1);
  assert.equal(sceneState(top, { clock: 125 }).camera, 'iso');
  const none = scene({ steps: [step(100), step(110)] });
  assert.equal(openingCamera(none), 'iso');
  assert.equal(sceneState(none, { clock: 0 }).camera, 'iso');
  assert.equal(openingCamera(scene({ steps: [] })), 'iso');
});

test('defs: named values the objects share, each from the parameters and the defs before it', () => {
  const s = scene({
    defs: [
      { name: 'cx', expr: '2*cos(lean*deg)' },
      { name: 'top', expr: 'h + cx' }, // an earlier def
      { name: 'bad', expr: 'later + 1' }, // a def after it
      { name: 'h', expr: '1' }, // a parameter's name
      { name: 'sin', expr: '1' }, // a function's name
      { name: 'wx', expr: 'x + 1' }, // a field's variable
      { name: 'Bad Name', expr: '1' },
      { name: 'later', expr: '2' },
    ],
    objects: [
      obj('edge', 'segment', { points: [['0', '0', 'h'], ['cx', '0', 'top']] }),
      obj('uses_bad', 'point', { points: [['bad', '0', '0']] }),
      obj('arc', 'curve', { x: 'cx*cos(t)', y: 'sin(t)', z: '0', range: ['0', 'top'] }),
    ],
    steps: [step(100, { show: ['edge'] })],
  });
  assert.deepEqual(s.defs, [
    { name: 'cx', expr: '2*cos(lean*deg)' },
    { name: 'top', expr: 'h + cx' },
    { name: 'later', expr: '2' },
  ]);
  assert.deepEqual(s.objects.map((o) => o.id), ['edge', 'arc']);
  // an older server sends no defs
  assert.deepEqual(scene({ defs: undefined }).defs, []);
  assert.equal(scene({ defs: Array.from({ length: 30 }, (_, i) => ({ name: `d${i}`, expr: '1' })) }).defs.length, 16);
});

test('a step applies at its time and its change runs, eased, over its length', () => {
  const s = scene();
  const at = (clock: number) => sceneState(s, { clock });
  assert.equal(at(100).index, 0);
  assert.equal(at(100).params.h, 1); // just started
  assert.equal(at(100).progress, 0);
  assert.equal(at(100).camera, 'front');
  assert.equal(at(100).cameraStep, 0);
  near(at(100.5).params.h, 1 + 2 * ease(0.25));
  near(at(101).params.h, 2); // halfway (the easing is symmetric)
  assert.ok(at(100.5).params.h < 1.5, 'eased: slow at first');
  near(at(101).progress, 0.5);
  assert.equal(at(101).moving, true);
  assert.equal(at(102).params.h, 3);
  assert.equal(at(102).progress, 1);
  assert.equal(at(105).moving, false);
  assert.equal(at(105).params.lean, 90); // the next step has not come yet
});

test('show fades in, hide fades out, over the step\'s length', () => {
  const s = scene();
  const at = (clock: number) => sceneState(s, { clock });
  assert.equal(at(110).vis.edge, 0);
  near(at(111).vis.edge, 0.5);
  assert.equal(at(112).vis.edge, 1);
  assert.equal(at(120).vis.force, 1);
  near(at(121).vis.force, 0.5);
  near(at(121).vis.fx, 0.5);
  assert.equal(at(122).vis.force, 0);
  assert.equal(at(122).vis.fz, 1);
});

test('highlight glows only while its step is the current one', () => {
  const s = scene();
  const at = (clock: number) => sceneState(s, { clock });
  assert.deepEqual(at(100).highlight, ['base']);
  assert.equal(at(100).glow, 0);
  assert.equal(at(101).glow, 1);
  assert.deepEqual(at(109.9).highlight, ['base']);
  assert.deepEqual(at(110.5).highlight, ['edge']);
  assert.deepEqual(at(125).highlight, []); // the last step highlights nothing
  assert.equal(at(125).glow, 0);
});

test('after the last step: everything has arrived and stays', () => {
  const s = scene();
  const st = sceneState(s, { clock: 1e6 });
  assert.equal(st.index, 2);
  assert.equal(st.progress, 1);
  assert.deepEqual(st.params, { h: 3, lean: 110 });
  assert.deepEqual(st.vis, { base: 1, edge: 1, force: 0, fx: 1, fz: 1, note: 1 });
  assert.equal(st.camera, 'iso');
  assert.equal(st.cameraStep, 2);
  assert.equal(st.moving, false);
});

test('seeking anywhere, forward or back, gives the state of playing up to there', () => {
  const s = scene();
  const clocks = [0, 100.3, 101, 104, 110.7, 111.9, 120.2, 130, 99, 100.3, 0, 121, 101];
  const fresh = clocks.map((c) => JSON.stringify(sceneState(scene(), { clock: c })));
  const reused = clocks.map((c) => JSON.stringify(sceneState(s, { clock: c })));
  assert.deepEqual(reused, fresh);
  // and a fine sweep of the clock never jumps: every value is continuous
  let prev = sceneState(s, { clock: 95 });
  for (let c = 95; c <= 130; c += 0.01) {
    const st = sceneState(s, { clock: c });
    for (const k of Object.keys(st.params)) assert.ok(Math.abs(st.params[k] - prev.params[k]) < 0.4, `${k} at ${c}`);
    for (const k of Object.keys(st.vis)) assert.ok(Math.abs(st.vis[k] - prev.vis[k]) < 0.05, `${k} at ${c}`);
    prev = st;
  }
});

test('a change that starts while the last one is still running carries on from where it is', () => {
  const s = scene({
    steps: [
      step(10, { set: [{ param: 'h', to: 3 }], over: 4 }),
      step(11, { set: [{ param: 'h', to: 1 }], over: 2 }),
      step(12, { show: ['edge'], over: 4 }),
      step(13, { hide: ['edge'], over: 2 }),
    ],
  });
  const at = (clock: number) => sceneState(s, { clock });
  const mid = 1 + 2 * ease(0.25); // where h is when the second step starts
  near(at(11).params.h, mid);
  near(at(11 - 1e-9).params.h, mid, 1e-6);
  near(at(12).params.h, (mid + 1) / 2);
  assert.equal(at(13).params.h, 1);
  const vis13 = ease(0.25);
  near(at(13).vis.edge, vis13);
  near(at(14).vis.edge, vis13 / 2);
  assert.equal(at(15).vis.edge, 0);
});

test('reduced motion: every change jumps at its step', () => {
  const s = scene();
  const at = (clock: number) => sceneState(s, { clock }, { reducedMotion: true });
  assert.equal(at(100).params.h, 3);
  assert.equal(at(100).progress, 1);
  assert.equal(at(100).glow, 1);
  assert.equal(at(110).vis.edge, 1);
  assert.equal(at(110).params.lean, 110);
  assert.equal(at(120).vis.force, 0);
  assert.equal(at(100).moving, false);
  assert.equal(at(99).params.h, 1);
});

test('a scene that is not synced is stepped through by the viewer', () => {
  const s = scene({ synced: false });
  assert.deepEqual(stepTimes(s), [0, STEP_GAP, 2 * STEP_GAP]);
  const at = (step: number, elapsed: number) => sceneState(s, { step, elapsed });
  assert.equal(at(-1, 0).index, -1);
  assert.deepEqual(at(-1, 0).params, { h: 1, lean: 90 });
  assert.equal(at(0, 0).index, 0);
  assert.equal(at(0, 0).params.h, 1);
  near(at(0, 1).params.h, 2);
  assert.equal(at(0, 100).index, 0, 'waiting long on a step never runs into the next');
  assert.equal(at(0, 100).params.h, 3);
  // going to step 2 (or back to it) finishes the steps before it and plays step 2's own change
  const two = at(2, 0);
  assert.equal(two.index, 2);
  assert.deepEqual(two.params, { h: 3, lean: 110 });
  assert.equal(two.vis.edge, 1);
  assert.equal(two.vis.fx, 0);
  assert.equal(at(2, 5).vis.fx, 1);
  const back = at(1, 0);
  assert.equal(back.index, 1);
  assert.equal(back.params.lean, 90);
  assert.equal(back.vis.fx, 0);
  assert.equal(back.vis.force, 1);
  assert.equal(at(7, 0).index, 2, 'past the end: the last step');
  // the video's clock does not move a scene that is not synced
  assert.equal(momentClock(s, { step: -1, elapsed: 3 }), -Infinity);
});

test('a scene with no steps shows its drawing', () => {
  const s = scene({ steps: [] });
  const st = sceneState(s, { clock: 50 });
  assert.equal(st.index, -1);
  assert.equal(st.vis.edge, 1); // no step shows it, so it is part of the base drawing
  assert.equal(beforeLine(s), 'Drag to turn the drawing.');
});

test('words and seek targets for the controls', () => {
  const s = scene();
  assert.equal(stepHeading(s, 0), 'Step 1 of 3 · 1:40');
  assert.equal(stepHeading(s, 2), 'Step 3 of 3 · 2:00');
  assert.equal(stepHeading(s, 5), '');
  assert.equal(stepHeading(scene({ synced: false }), 1), 'Step 2 of 3');
  assert.equal(beforeLine(s), 'The explanation starts at 1:40.');
  assert.match(beforeLine(scene({ synced: false })), /not talked about/);
  near(seekTimeFor(s, 1)!, 110.02);
  near(seekTimeFor(s, 1, true)!, 110.02);
  // paused, the video stays paused and shows the step done (its change is over, the next step not begun)
  near(seekTimeFor(s, 1, false)!, 112.02);
  near(seekTimeFor(scene({ steps: [step(100, { over: 8 }), step(104)] }), 0, false)!, 103.95);
  near(seekTimeFor(scene({ steps: [step(100, { over: 0.1 })] }), 0, false)!, 100.62); // a change takes 0.6 s at least
  assert.equal(seekTimeFor(scene({ synced: false }), 1), null);
  assert.equal(playFrom(s, 0), 98);
  assert.equal(playFrom(s, 99.5), 99.5); // never before the analysed clip
  assert.equal(playFrom(scene({ synced: false }), 0), null);
  assert.equal(chipCaption('The force splits into two parts along the axes'), 'The force splits into two parts…');
  assert.equal(chipCaption('Short one'), 'Short one');
});

test('"This drawing from": when the drawing itself is talked about well after the first step', () => {
  // the series starts at 1:40; the teacher talks about this drawing from 3:00
  const s = scene({ steps: [step(100), step(140), step(178), step(185)] });
  assert.equal(drawingFrom(s, [190, 180], 0), 178);
  assert.equal(drawingFrom(s, [180], 179.5), 179.5); // not before the analysed clip
  assert.equal(drawingFrom(s, [110], 0), null); // close to the first step: one button is enough
  assert.equal(drawingFrom(s, [], 0), null);
  assert.equal(drawingFrom(s, [400], 0, 300), null); // after the analysed clip
  assert.equal(drawingFrom(scene({ steps: [step(100), step(105)] }), [200], 0), null); // no step that late
  assert.equal(drawingFrom(scene({ synced: false }), [180], 0), null);
});

test('playing the steps of a scene not tied to the video: each stays for its change and its caption', () => {
  const [a] = scene({ synced: false, steps: [step(null, { over: 2, caption: 'Short' })] }).steps;
  assert.equal(dwell(a), 2 + 2.5);
  const [b] = scene({ synced: false, steps: [step(null, { over: 1, caption: 'x'.repeat(140) })] }).steps;
  assert.equal(dwell(b), 1 + 7);
});

// ---------------------------------------------------------------- one request per drawing

test('one request per drawing, shared; only a usable scene is kept', async () => {
  const answers: unknown[] = [];
  let calls = 0;
  const load = sceneLoader(async () => {
    calls++;
    const a = answers.shift();
    if (a instanceof Error) throw a;
    return a;
  });
  // two opens while the first is still on its way: one request
  answers.push(rawScene());
  const [a, b] = await Promise.all([load('job', 3), load('job', 3)]);
  assert.equal(calls, 1);
  assert.equal(a.kind, 'scene');
  assert.equal(b, a);
  assert.equal((await load('job', 3)).kind, 'scene');
  assert.equal(calls, 1, 'a good scene is kept');
  // a failure is forgotten, so "Try again" asks again
  answers.push(new Error('Muse Spark could not make the 3D view: timed out'));
  await assert.rejects(load('job', 4), /timed out/);
  answers.push(rawScene({ piece: 4 }));
  assert.equal((await load('job', 4)).kind, 'scene');
  assert.equal(calls, 3);
  // an older server's answer is forgotten too: once the server is restarted, opening it again asks the new one
  answers.push({ kind: 'solids', solids: [] });
  assert.equal((await load('job', 5)).kind, 'old');
  answers.push(rawScene({ piece: 5 }));
  assert.equal((await load('job', 5)).kind, 'scene');
  // and so is a malformed one
  answers.push('not a scene');
  assert.equal((await load('job', 6)).kind, 'bad');
  answers.push(rawScene({ piece: 6 }));
  assert.equal((await load('job', 6)).kind, 'scene');
  assert.equal(calls, 7);
  // other lectures and drawings are their own requests
  answers.push(rawScene());
  await load('other', 3);
  assert.equal(calls, 8);
});

// ---------------------------------------------------------------- angles

test('an angle: exactly three points, and no size or one radius', () => {
  const corner = [['0', '0', '0'], ['h', '0', '0'], ['2*cos(lean*deg)', '0', '2*sin(lean*deg)']];
  const s = scene({
    objects: [
      obj('beta', 'angle', { points: corner, label: 'beta' }),
      obj('sized', 'angle', { points: corner, size: ['0.4*h'] }),
      obj('two', 'angle', { points: corner.slice(0, 2) }), // too few points
      obj('four', 'angle', { points: [...corner, ['1', '1', '1']] }), // too many
      obj('badr', 'angle', { points: corner, size: ['k'] }), // k is not a parameter
      obj('rr', 'angle', { points: corner, size: ['1', '2'] }), // one radius at most
      obj('vars', 'angle', { points: [['t', '0', '0'], ['1', '0', '0'], ['0', '1', '0']] }), // no variables
    ],
    steps: [step(10, { show: ['beta', 'two'], highlight: ['sized'] })],
  });
  assert.deepEqual(s.objects.map((o) => o.id), ['beta', 'sized']);
  const beta = s.objects[0];
  assert.equal(beta.type, 'angle');
  assert.equal(beta.points.length, 3);
  assert.deepEqual(beta.size, []);
  assert.equal(beta.label, 'beta');
  assert.deepEqual(s.objects[1].size, ['0.4*h']);
  assert.deepEqual(s.steps[0].show, ['beta']);
  assert.deepEqual(s.steps[0].highlight, ['sized']);
});

// ---------------------------------------------------------------- "Try it": the viewer's own values

test('overrides: the viewer\'s values on top of the lecture, kept in range; the rest still follows the clock', () => {
  const s = scene();
  // in the middle of step 2 (lean 90 -> 110 at 110 s, over 2 s)
  const plain = sceneState(s, { clock: 111 });
  assert.deepEqual(plain.changing, ['lean']);
  assert.deepEqual(plain.overridden, []);
  const own = sceneState(s, { clock: 111 }, { overrides: { lean: 70 } });
  assert.equal(own.params.lean, 70);
  assert.deepEqual(own.overridden, ['lean']);
  assert.deepEqual(own.changing, [], 'an overridden parameter is not "changing" with the step');
  assert.equal(own.params.h, plain.params.h, 'the other parameters follow the lecture');
  assert.equal(own.index, plain.index);
  assert.deepEqual(own.vis, plain.vis);
  // out of range: kept inside [min, max]; not a number, or not a parameter: ignored
  assert.equal(sceneState(s, { clock: 111 }, { overrides: { lean: 500 } }).params.lean, 120);
  assert.equal(sceneState(s, { clock: 111 }, { overrides: { h: -3 } }).params.h, 1);
  const junk = sceneState(s, { clock: 111 }, { overrides: { lean: NaN, nope: 3 } as Record<string, number> });
  assert.equal(junk.params.lean, plain.params.lean);
  assert.equal('nope' in junk.params, false);
  // pure: the same scene, moment and overrides give the same state
  assert.deepEqual(sceneState(s, { clock: 111 }, { overrides: { lean: 70 } }), own);
  // the step that moves h (100 s, over 2 s) moves it while it is not overridden
  assert.deepEqual(sceneState(s, { clock: 101 }).changing, ['h']);
  assert.deepEqual(sceneState(s, { clock: 101 }, { overrides: { h: 2 } }).changing, []);
});

// ---------------------------------------------------------------- fixer's pass

test('a number label the lecture keeps equal to one parameter is drawn live', () => {
  // synth3d-vector3d: "3" by the x component while vx is 3 stayed "3" with the slider at 5
  const s = scene({
    params: [
      { name: 'vx', value: 3, min: 1, max: 5, label: 'x', unit: '' },
      { name: 'vy', value: 2, min: 1, max: 5, label: 'y', unit: '' },
      { name: 'w', value: 2, min: 1, max: 5, label: 'w', unit: '' },
      { name: 'h', value: 1, min: 1, max: 9, label: 'h', unit: '' },
    ],
    objects: [
      obj('l3', 'label', { points: [['0', '0', '0']], label: '3' }),
      obj('l2', 'label', { points: [['0', '0', '0']], label: '2' }), // vy and w are both 2: whose is it?
      obj('lh', 'label', { points: [['0', '0', '0']], label: '1' }), // h moves away from 1 while it shows
      obj('lt', 'label', { points: [['0', '0', '0']], label: 'the top' }),
      obj('lv', 'label', { points: [['0', '0', '0']], label: '3' }), // shown after vx moved to 4
    ],
    steps: [
      step(100, { caption: 'x', show: ['l3', 'l2', 'lh'] }),
      step(110, { caption: 'up', set: [{ param: 'h', to: 5 }] }),
      step(120, { caption: 'on', set: [{ param: 'vx', to: 4 }], show: ['lv'], hide: ['l3'] }),
    ],
  });
  assert.deepEqual(liveNumberLabels(s), { l3: 'vx' });
});

test('each step is framed on what is on screen then, where its change starts and ends', () => {
  const s = scene();
  const f = stepFraming(s);
  assert.equal(f.length, s.steps.length + 1);
  assert.deepEqual(f[0].visible, ['base', 'force', 'note']); // before the first step: what no step shows
  assert.deepEqual([f[0].from.h, f[0].to.h], [1, 1]);
  assert.deepEqual([f[1].from.h, f[1].to.h], [1, 3]); // the height grows
  assert.deepEqual(f[2].visible, ['base', 'force', 'note', 'edge']);
  assert.deepEqual([f[2].from.lean, f[2].to.lean, f[2].to.h], [90, 110, 3]);
  assert.deepEqual(f[3].visible, ['base', 'note', 'edge', 'fx', 'fz']); // the force goes, its parts come
});

// ---------------------------------------------------------------- "Try it": only cells that exist

/** test2 #7's cell: a, b, c and alpha, beta, gamma, built from the defs; the hexagon step takes gamma to 120. */
function cell(): Scene3D {
  const r = cleanScene({
    format: 2,
    synced: true,
    params: [
      { name: 'a', value: 3, min: 1, max: 6, label: 'edge length a' },
      { name: 'b', value: 3, min: 1, max: 6, label: 'edge length b' },
      { name: 'c', value: 3, min: 1, max: 6, label: 'edge length c' },
      { name: 'alpha', value: 90, min: 60, max: 120, label: 'angle alpha', unit: 'deg' },
      { name: 'beta', value: 90, min: 60, max: 120, label: 'angle beta', unit: 'deg' },
      { name: 'gamma', value: 90, min: 60, max: 150, label: 'angle gamma', unit: 'deg' },
    ],
    defs: [
      { name: 'bx', expr: 'b*cos(gamma*deg)' },
      { name: 'by', expr: 'b*sin(gamma*deg)' },
      { name: 'cx', expr: 'c*cos(beta*deg)' },
      { name: 'cy', expr: 'c*(cos(alpha*deg) - cos(beta*deg)*cos(gamma*deg))/sin(gamma*deg)' },
      { name: 'cz', expr: 'sqrt(max(c*c - cx*cx - cy*cy, 0.2))' },
    ],
    objects: [
      obj('alpha_angle', 'angle', { points: [['0', '0', '0'], ['bx', 'by', '0'], ['cx', 'cy', 'cz']], label: 'alpha' }),
      obj('beta_angle', 'angle', { points: [['0', '0', '0'], ['a', '0', '0'], ['cx', 'cy', 'cz']], label: 'β' }),
      obj('gamma_angle', 'angle', {
        points: [['0', '0', '0'], ['a', '0', '0'], ['bx', 'by', '0']],
        label: 'gamma',
        size: ['0.3*min(a, b)'],
      }),
    ],
    steps: [step(100, { caption: 'cube' }), step(200, { caption: 'hexagonal', set: [{ param: 'gamma', to: 120 }] })],
  });
  assert.equal(r.kind, 'scene');
  return (r as { scene: Scene3D }).scene;
}

const gram = (al: number, be: number, ga: number) => {
  const c = (d: number) => Math.cos((d * Math.PI) / 180);
  return 1 - c(al) ** 2 - c(be) ** 2 - c(ga) ** 2 + 2 * c(al) * c(be) * c(ga);
};

test("the shape guard: a cell's angles must measure what the sliders say and never lie flat", () => {
  const g = shapeGuard(cell())!;
  assert.deepEqual(g.params, ['alpha', 'beta', 'gamma']); // the letter names beta as well as the word does
  const at = (alpha: number, beta: number, gamma: number) => g.ok({ a: 3, b: 3, c: 3, alpha, beta, gamma });
  assert.equal(at(90, 90, 90), true);
  assert.equal(at(90, 90, 120), true); // the hexagonal cell
  assert.equal(at(80, 75, 85), true); // triclinic
  assert.equal(at(120, 120, 120), false); // a flat plate: the old ranges' corner (labels read 119.9)
  assert.equal(at(60, 60, 120), false);
  // the edge of what exists is the server's FLAT_MIN on the Gram determinant
  for (const [al, be, ga] of [[123, 90, 120], [123.5, 90, 120], [100, 70, 140], [95, 60, 140]]) {
    assert.equal(at(al, be, ga), gram(al, be, ga) >= FLAT_MIN, `${al} ${be} ${ga}`);
  }
  // nothing to guard: no angle parameter, or no angle object named after one
  const s = cell();
  assert.equal(shapeGuard({ ...s, params: s.params.map((p) => ({ ...p, unit: '' as const })) }), null);
  assert.equal(shapeGuard({ ...s, objects: s.objects.map((o) => ({ ...o, label: 'phi' })) }), null);
  assert.equal(shapeGuard(scene()), null); // a lean with no angle object
});

test('a slider explores what the cell allows with the other values as they are, and stops at the wall', () => {
  // test2 #7: the box rule let alpha explore only 77..94, because gamma must still reach 120 in the same box
  const g = shapeGuard(cell())!;
  const hex = { a: 3, b: 3, c: 4.5, alpha: 90, beta: 90, gamma: 120 };
  // with gamma at 120 and beta at 90, alpha goes up to where the cell would lie flat: cos^2 alpha = 0.75 - FLAT_MIN
  const top = (Math.acos(-Math.sqrt(0.75 - FLAT_MIN)) * 180) / Math.PI; // 123.2
  const up = clampToShape(g, hex, 'alpha', 150);
  assert.ok(Math.abs(up - top) < 0.01, `${up} vs ${top}`);
  assert.ok(g.ok({ ...hex, alpha: up }) && !g.ok({ ...hex, alpha: up + 0.05 }));
  const down = clampToShape(g, hex, 'alpha', 20);
  assert.ok(Math.abs(down - (180 - top)) < 0.01, `${down}`);
  assert.ok(up - down > 46, 'far more than the 17 degrees of 77..94');
  // a value that makes a cell is taken as it is
  assert.equal(clampToShape(g, hex, 'alpha', 110), 110);
  assert.equal(clampToShape(g, hex, 'c', 1), 1); // lengths are free here
  // with gamma at 90 instead, alpha may go further
  assert.ok(clampToShape(g, { ...hex, gamma: 90 }, 'alpha', 150) > top + 5);
  // values that already make no cell (nothing to keep): the wanted value, as it is
  assert.equal(clampToShape(g, { ...hex, alpha: 120, beta: 120, gamma: 120 }, 'alpha', 130), 130);
  assert.equal(clampToShape(null, hex, 'alpha', 170), 170);
  // whatever the slider asks, every value taken makes a cell whose angles read what the sliders say
  for (let want = 0; want <= 180; want += 7.5) {
    const v = clampToShape(g, hex, 'beta', want);
    assert.ok(g.ok({ ...hex, beta: v }), `beta ${want} -> ${v}`);
  }
});

test("when the lecture moves an angle the viewer left alone, the viewer's angles give way as little as it takes", () => {
  const s = cell();
  const g = shapeGuard(s)!;
  // the viewer set alpha and beta to 60 with gamma at 90 (a cell); the hexagon step takes gamma to 120 (none)
  const own = { alpha: 60, beta: 60 };
  assert.ok(g.ok({ a: 3, b: 3, c: 3, gamma: 90, ...own }));
  const lecture = { a: 3, b: 3, c: 3, alpha: 90, beta: 90, gamma: 120 };
  const fit = fitToShape(g, lecture, own);
  assert.ok(g.ok(fit), 'a cell that exists');
  assert.ok(fit.alpha < 90 && fit.alpha > 60 && Math.abs(fit.alpha - fit.beta) < 1e-9, 'both moved back by one share');
  assert.ok(!g.ok({ ...fit, alpha: fit.alpha - 0.05, beta: fit.beta - 0.05 }), 'no more than it takes');
  assert.equal(fit.gamma, 120, "the lecture's own value is untouched");
  // values that make a cell are kept as they are
  assert.deepEqual(fitToShape(g, lecture, { alpha: 100 }), { ...lecture, alpha: 100 });
  // in the view: before the hexagon step the viewer's values stand; after it they give way; without the guard, a
  // flat cell whose angles no longer read what the sliders say
  const before = sceneState(s, { clock: 150 }, { overrides: own, guard: g });
  assert.deepEqual([before.params.alpha, before.params.beta], [60, 60]);
  const after = sceneState(s, { clock: 205 }, { overrides: own, guard: g });
  assert.ok(g.ok(after.params) && after.params.alpha > 60 && after.params.gamma === 120);
  assert.deepEqual(after.overridden, ['alpha', 'beta']);
  const bare = sceneState(s, { clock: 205 }, { overrides: own });
  assert.equal(g.ok(bare.params), false);
  // what the guard lets through, the angle objects measure: the labels read the sliders' values
  const p = after.params;
  const r = (d: number) => (d * Math.PI) / 180;
  const cx = p.c * Math.cos(r(p.beta));
  const cy = (p.c * (Math.cos(r(p.alpha)) - Math.cos(r(p.beta)) * Math.cos(r(p.gamma)))) / Math.sin(r(p.gamma));
  const cz = Math.sqrt(Math.max(p.c * p.c - cx * cx - cy * cy, 0.2));
  const bx = p.b * Math.cos(r(p.gamma));
  const by = p.b * Math.sin(r(p.gamma));
  assert.ok(Math.abs(measureAngle([0, 0, 0], [bx, by, 0], [cx, cy, cz])! - p.alpha) <= 0.5);
});

test("a lecture change moves only the viewer's angles that must give way: their lengths stay as they set them", () => {
  const s = cell();
  const g = shapeGuard(s)!;
  // test2 #7: c set to 6 read 5.42 once the hexagon step took gamma to 120
  const own = { c: 6, alpha: 60, beta: 60 };
  const lecture = { a: 3, b: 3, c: 3, alpha: 90, beta: 90, gamma: 120 };
  const fit = fitToShape(g, lecture, own);
  assert.equal(fit.c, 6);
  assert.ok(g.ok(fit) && fit.alpha > 60 && fit.alpha < 90 && Math.abs(fit.alpha - fit.beta) < 1e-9);
  // the angles give way exactly as far as without the length (a cell's angles do not depend on its edges)
  near(fit.alpha, fitToShape(g, lecture, { alpha: 60, beta: 60 }).alpha, 1e-6);
  const after = sceneState(s, { clock: 205 }, { overrides: own, guard: g });
  assert.equal(after.params.c, 6);
  assert.deepEqual(after.overridden, ['c', 'alpha', 'beta']);
});

test("moving one slider keeps the viewer's other angles where they are shown, and stops on the slider's steps", () => {
  const s = cell();
  const g = shapeGuard(s)!;
  const lecture = { a: 3, b: 3, c: 3, alpha: 90, beta: 90, gamma: 120 };
  // the hexagon step made alpha and beta give way from the 60 the viewer set to about 71.6
  const own = { c: 6, alpha: 60, beta: 60 };
  const drawn = fitToShape(g, lecture, own);
  const shown = drawn.alpha;
  // the viewer moves beta to 75: alpha, which they did not touch, stays at what it shows; beta is 75, as asked
  const next = tryValue(g, own, drawn, 'beta', 75, { min: 60, step: 1 });
  assert.deepEqual([next.alpha, next.beta, next.c], [shown, 75, 6]);
  const frame = fitToShape(g, lecture, next);
  assert.deepEqual([frame.alpha, frame.beta, frame.c], [shown, 75, 6]);
  // the old way (the viewer's 60 kept for alpha): beta drew 77.9 and alpha went to 65.7
  const old = fitToShape(g, lecture, { ...own, beta: 75 });
  assert.ok(Math.abs(old.beta - 75) > 2 && Math.abs(old.alpha - shown) > 5);
  // moving an angle that gave way starts from where it is shown, not from the 60 set before: it stops at the wall
  // there, never jumping to the value asked for
  const back = tryValue(g, own, drawn, 'alpha', 65, { min: 60, step: 1 });
  assert.ok(Math.abs(back.alpha - shown) < 0.05 && back.beta === drawn.beta, JSON.stringify(back));
  assert.ok(g.ok({ ...lecture, ...back }));
  // at a wall the value stops on the slider's own steps, on the side where the cell exists (85.1 had the thumb at 85)
  const hex = { a: 3, b: 3, c: 4.5, alpha: 90, beta: 60, gamma: 120 };
  const wall = clampToShape(g, hex, 'alpha', 60);
  assert.ok(wall > 85 && wall < 86, `${wall}`);
  const taken = tryValue(g, {}, hex, 'alpha', 60, { min: 60, step: 1 });
  assert.equal(taken.alpha, 86);
  assert.ok(g.ok({ ...hex, alpha: taken.alpha }));
  // a value that makes a cell is taken as asked, on or off the steps; with no guard, as asked
  assert.equal(tryValue(g, {}, hex, 'alpha', 100.5, { min: 60, step: 1 }).alpha, 100.5);
  assert.equal(tryValue(null, { alpha: 70 }, hex, 'beta', 20, { min: 60, step: 1 }).beta, 20);
  // whatever is asked, what is taken makes a cell
  for (let want = 0; want <= 180; want += 5) {
    const t = tryValue(g, {}, hex, 'gamma', want, { min: 60, step: 1 });
    assert.ok(g.ok({ ...hex, ...t }), `gamma ${want} -> ${t.gamma}`);
    assert.ok(Number.isInteger(t.gamma) || t.gamma === want, `gamma ${want} -> ${t.gamma} is on the steps`);
  }
});

test('two sliders moved before the next frame: the first keeps the value just set, not the one last drawn', () => {
  const g = shapeGuard(cell())!;
  // test2 #4's last step: End on alpha, then End on beta, with no frame drawn between
  const frame = { params: { a: 3, b: 3, c: 3, alpha: 90, beta: 75, gamma: 90 }, own: {} };
  const grid = { min: 60, step: 1 };
  const one = tryValue(g, frame.own, drawnWith(frame, frame.own), 'alpha', 120, grid);
  assert.equal(one.alpha, 120);
  const two = tryValue(g, one, drawnWith(frame, one), 'beta', 120, grid);
  assert.equal(two.alpha, 120, 'alpha stays where the viewer put it');
  assert.ok(g.ok({ ...frame.params, ...two }));
  // once a frame is drawn with these values, they are what is shown
  const drawn = { params: { ...frame.params, ...two }, own: two };
  assert.deepEqual(drawnWith(drawn, two), drawn.params);
  // the old way (the frame as last drawn, whatever it was drawn with): alpha went back to 90
  assert.equal(tryValue(g, one, frame.params, 'beta', 120, grid).alpha, 90);
});
