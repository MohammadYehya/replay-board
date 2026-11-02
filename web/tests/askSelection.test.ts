// Run with `npm test`. Words selected in the transcript become a span of speech for Ask Muse: from the first selected
// word's start to the last one's end, labelled with the selected text.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectionLabel, spanEntry, spanFromTurns, spanFromWords } from '../src/ask/selection.ts';

const words = [
  { w: 'So', s: 60.0, e: 60.2 },
  { w: 'the', s: 60.25, e: 60.4 },
  { w: 'interest', s: 60.5, e: 61.0 },
  { w: 'is', s: 61.2, e: 61.3 },
  { w: 'compounded', s: 61.4, e: 62.1 },
  { w: 'yearly.', s: 62.2, e: 62.987 },
];

test('selected words -> span from the first start to the last end', () => {
  const s = spanFromWords(words, [2, 3, 4], 'interest is compounded');
  assert.deepEqual(s, { t0: 60.5, t1: 62.1, label: 'interest is compounded' });
  // any order, duplicates, rounding to 1/100 s
  assert.deepEqual(spanFromWords(words, [5, 1, 5, 3]), { t0: 60.25, t1: 62.99, label: 'the interest is compounded yearly' });
});

test('the label is the selected text, cleaned; else the words', () => {
  assert.equal(spanFromWords(words, [0, 1], '  “So,\n the ')!.label, 'So, the');
  assert.equal(spanFromWords(words, [4, 5], '')!.label, 'compounded yearly');
  const long = 'word '.repeat(40);
  const label = selectionLabel(long);
  assert.ok(label.length <= 60 && label.endsWith('…'), label);
});

test('nothing selected, or indices outside the transcript: no span', () => {
  assert.equal(spanFromWords(words, []), null);
  assert.equal(spanFromWords(words, [-1, 99, 2.5]), null);
  assert.equal(spanFromWords([], [0]), null);
});

test('paragraphs without word times: the whole paragraphs', () => {
  const turns = [
    { start: 10, end: 20, text: 'First paragraph.' },
    { start: 21, end: 35.5, text: 'Second one.' },
  ];
  assert.deepEqual(spanFromTurns(turns, 1, 0, ''), { t0: 10, t1: 35.5, label: 'First paragraph' });
  assert.equal(spanFromTurns(turns, 5, 6), null);
});

test('the tray entry for a span', () => {
  const s = spanFromWords(words, [2, 3])!;
  assert.deepEqual(spanEntry(s), { item: { kind: 'span', t0: 60.5, t1: 61.3 }, label: 'interest is', t: 60.5 });
  assert.equal(spanEntry({ t0: 1, t1: 2, label: '' }).label, 'Speech');
});
