// Run with `npm test`. The typeset layer fits its readings a slice at a time (sliceQueue.ts): an hour-long lecture's
// 310 readings, fitted in the commit that first showed the player, froze it for 3 s twice.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SliceQueue } from '../src/player/sliceQueue.ts';

/** A queue on a fake clock: each job advances it by its cost; scheduled slices wait in `pending` until run. */
function harness(budget = 12) {
  let clock = 0;
  const pending: (() => void)[] = [];
  const q = new SliceQueue<string>(budget, () => clock, (run) => pending.push(run));
  const ran: string[] = [];
  const job = (name: string, cost: number) => () => {
    clock += cost;
    ran.push(name);
  };
  const slice = () => {
    const run = pending.shift();
    assert.ok(run, 'a slice is scheduled');
    const before = ran.length;
    const t0 = clock;
    run();
    return { jobs: ran.length - before, ms: clock - t0 };
  };
  return { q, ran, job, slice, pending };
}

test('310 readings of 5 ms each are fitted in slices of at most the budget plus one job, not in one 1.5 s task', () => {
  const { q, ran, job, slice, pending } = harness(12);
  for (let i = 0; i < 310; i++) q.add(`p${i}`, job(`p${i}`, 5));
  assert.equal(ran.length, 0, 'nothing runs when queued');
  assert.equal(pending.length, 1, 'one slice scheduled, not one per job');
  let slices = 0;
  let longest = 0;
  while (pending.length) {
    const s = slice();
    slices++;
    longest = Math.max(longest, s.ms);
  }
  assert.equal(ran.length, 310);
  assert.equal(slices, Math.ceil(310 / 3)); // 5 + 5 + 5 >= 12: three a slice
  assert.ok(longest <= 12 + 5, `longest slice ${longest} ms`);
  assert.deepEqual(ran.slice(0, 4), ['p0', 'p1', 'p2', 'p3'], 'in the order queued');
});

test('a reading queued again before it ran (fonts arrived) is fitted once, with the latest job', () => {
  const { q, ran, job, slice, pending } = harness(100);
  q.add('a', job('a-before-fonts', 1));
  q.add('b', job('b', 1));
  q.add('a', job('a-after-fonts', 1));
  assert.equal(q.size, 2);
  while (pending.length) slice();
  assert.deepEqual(ran, ['b', 'a-after-fonts']);
});

test('a cancelled reading (its piece left the board) is not fitted; a slow job still lets the next slice run', () => {
  const { q, ran, job, slice, pending } = harness(10);
  q.add('slow', job('slow', 40));
  q.add('gone', job('gone', 1));
  q.add('kept', job('kept', 1));
  q.cancel('gone');
  assert.deepEqual(slice(), { jobs: 1, ms: 40 }); // at least one job a slice, however slow
  while (pending.length) slice();
  assert.deepEqual(ran, ['slow', 'kept']);
  assert.equal(q.size, 0);
});

test('a reading that throws is dropped and the others are still fitted', () => {
  const { q, ran, job, slice, pending } = harness(100);
  q.add('bad', () => {
    throw new Error('KaTeX could not lay it out');
  });
  q.add('good', job('good', 1));
  while (pending.length) slice();
  assert.deepEqual(ran, ['good']);
});

test('jobs queued while a slice runs are picked up by a later slice', () => {
  const { q, ran, job, slice, pending } = harness(3);
  q.add('a', () => {
    job('a', 5)();
    q.add('b', job('b', 1));
  });
  slice();
  assert.deepEqual(ran, ['a']);
  assert.equal(pending.length, 1);
  slice();
  assert.deepEqual(ran, ['a', 'b']);
});
