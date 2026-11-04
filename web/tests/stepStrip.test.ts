// Run with `npm test`. The 3D view's step strip scrolls in whole chips: test2 #4's eight steps at 1920x1080, as the
// browser laid them out (widths 194-230 px, a 6 px gap, 2 px padding, a 921 px strip).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cutChips, endRoom, isWhole, scrollFor, startOf, type StripLayout } from '../src/player/stepStrip.ts';

const WIDTHS = [230, 219, 194, 230, 227, 224, 206, 230];
const strip = (view = 921, widths = WIDTHS): StripLayout => {
  const lefts: number[] = [];
  let x = 2;
  for (const w of widths) {
    lefts.push(x);
    x += w + 6;
  }
  return { lefts, widths, pad: 2, view };
};
const END = 1574 + 230 + 2; // the last chip's right edge and the padding: the strip's scrollWidth

test('opening on step 5 of 8: the strip starts at a chip, and the chip cut on the right is hidden', () => {
  const s = strip();
  const to = scrollFor(s, 4, 0);
  assert.equal(to, startOf(s, 1)); // chips 1-4 whole
  assert.ok(isWhole(s, 4, to));
  assert.deepEqual(cutChips(s, to), [5]); // hidden: a gap, not a sliver
  // what the strip did before: the chip's right edge 8 px from the strip's, a sliver at each end
  assert.deepEqual(cutChips(s, 1126 - 921 + 8), [0, 5]);
});

test('"‹" down to the first step and "›" up to the last: whole chips at both ends', () => {
  const s = strip();
  const max = END + endRoom(s, END) - s.view;
  let at = scrollFor(s, 4, 0, max);
  for (let i = 3; i >= -1; i--) {
    at = scrollFor(s, i, at, max);
    assert.ok(isWhole(s, Math.max(0, i), at), `step ${i}`);
    assert.ok(s.lefts.some((_, j) => startOf(s, j) === at), `starts at a chip (${at})`);
  }
  assert.equal(at, 0);
  assert.deepEqual(cutChips(s, 0), [4]); // hidden, as at every position: never a sliver
  for (let i = 0; i < 8; i++) {
    at = scrollFor(s, i, at, max);
    assert.ok(isWhole(s, i, at), `step ${i}`);
  }
  assert.equal(at, startOf(s, 4)); // the last four, whole
  assert.deepEqual(cutChips(s, at), []);
  // before: the browser stopped at the end of the content, 13 px short of chip 4's start, and chip 3 showed a sliver
  assert.deepEqual(cutChips(s, END - s.view), [3]);
});

test('room after the last chip lets the strip end on a chip start', () => {
  const s = strip();
  assert.equal(endRoom(s, END), 12); // chip 4 starts at 897; without room the strip stops at 885
  assert.equal(endRoom(strip(5000), END), 0); // everything fits
  assert.equal(endRoom(strip(921, []), 0), 0);
  // a narrow strip (the owner's laptop with Ask open: about 450 px): two chips at a time, still whole
  const narrow = strip(450);
  const room = endRoom(narrow, END);
  const max = END + room - 450;
  assert.ok(isWhole(narrow, 7, scrollFor(narrow, 7, 0, max)) && isWhole(narrow, 6, scrollFor(narrow, 7, 0, max)));
});

test('a chip stays put while it is whole; a chip wider than the strip shows its start', () => {
  const s = strip();
  assert.equal(scrollFor(s, 2, startOf(s, 1)), startOf(s, 1));
  assert.equal(scrollFor(s, 0, 500), 0);
  assert.equal(scrollFor(s, -1, 500), 0); // before the first step
  const wide = strip(150);
  assert.equal(scrollFor(wide, 3, 0), startOf(wide, 3));
  assert.equal(scrollFor(strip(921, []), 2, 40), 0);
});
