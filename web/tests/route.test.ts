// Run with `npm test`. The player's pages from the hash: home, a lecture (optionally at a moment), its notes, its
// slides.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseRoute } from '../src/route.ts';

test('home for anything that is not a lecture', () => {
  assert.deepEqual(parseRoute(''), { page: 'home' });
  assert.deepEqual(parseRoute('#/'), { page: 'home' });
  assert.deepEqual(parseRoute('#/jobs/x'), { page: 'home' });
  assert.deepEqual(parseRoute('#/job/'), { page: 'home' });
  assert.deepEqual(parseRoute('#/job/x/other'), { page: 'home' });
});

test('a lecture, at a moment or not', () => {
  assert.deepEqual(parseRoute('#/job/test-full'), { page: 'job', id: 'test-full', t: null });
  assert.deepEqual(parseRoute('#/job/test-full?t=83'), { page: 'job', id: 'test-full', t: 83 });
  assert.deepEqual(parseRoute('#/job/test-full?t=12.5'), { page: 'job', id: 'test-full', t: 12.5 });
  assert.deepEqual(parseRoute('#/job/test-full?t=-4'), { page: 'job', id: 'test-full', t: null });
  assert.deepEqual(parseRoute('#/job/test-full?t=abc'), { page: 'job', id: 'test-full', t: null });
  assert.deepEqual(parseRoute('#/job/test-full?t=0'), { page: 'job', id: 'test-full', t: 0 });
  // no t, or an empty one, is no moment (not the clip's start)
  assert.deepEqual(parseRoute('#/job/test-full?x=1'), { page: 'job', id: 'test-full', t: null });
  assert.deepEqual(parseRoute('#/job/test-full?t='), { page: 'job', id: 'test-full', t: null });
  assert.deepEqual(parseRoute('#/job/test-full?'), { page: 'job', id: 'test-full', t: null });
});

test('notes and slides pages', () => {
  assert.deepEqual(parseRoute('#/job/test-full/notes'), { page: 'notes', id: 'test-full' });
  assert.deepEqual(parseRoute('#/job/test-full/slides'), { page: 'slides', id: 'test-full' });
  assert.deepEqual(parseRoute('#/job/test-full/slides/'), { page: 'slides', id: 'test-full' });
  assert.deepEqual(parseRoute('#/job/test-full/slides?x=1'), { page: 'slides', id: 'test-full' });
  assert.deepEqual(parseRoute('#/job/my%20lecture/slides'), { page: 'slides', id: 'my lecture' });
  assert.deepEqual(parseRoute('#/job/test-full/slidesx'), { page: 'home' });
});

test('a broken escape in the id goes home instead of throwing', () => {
  assert.deepEqual(parseRoute('#/job/%E0%A4/slides'), { page: 'home' });
});
