// Run with `npm test`. The lecture's items for Ask Muse: lines and the pieces not in one (including erased writing),
// grouped by chapter; seek times; refs; slide elements without the repeats of build steps; matching a notes line.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { entryFor, filterGroups, lectureIndex, matchNote, parseRef, savedEntry } from '../src/ask/items.ts';
import type { Piece, Replay } from '../src/types.ts';

const piece = (id: number, over: Partial<Piece> = {}): Piece => ({
  id,
  box: [10 * id, 10, 40, 20],
  image: `pieces/${id}.png`,
  t_seen: 10 * id,
  t_shown: 10 * id + 1,
  read: { kind: 'equation', text: `x = ${id}`, latex: `x = ${id}` },
  spoken: [],
  ...over,
});

const boardReplay = (): Replay => ({
  version: 1,
  type: 'board',
  video: { file: 'v.mp4', duration: 600, width: 1280, height: 720, start: 5, end: 300 },
  chapters: [
    { t: 0, title: 'Intro' },
    { t: 40, title: 'Later' },
  ],
  board: {
    size: [960, 540],
    pieces: [
      piece(1, { pre_existing: true, t_seen: 0, t_shown: 0 }),
      piece(2, { spoken: [{ start: 30, end: 31 }, { start: 22, end: 23 }] }),
      piece(3, { removed_at: 90, redraw_at: 25 }),
      piece(4, { read: { kind: 'graph', text: 'graph of a parabola', latex: '' } }),
      piece(5),
      piece(6, { spoken: [{ start: 80, end: 81 }] }),
    ],
    lines: [
      { id: 'L1', pieces: [5, 6], box: [0, 0, 100, 30], t_complete: 61, read: { kind: 'equation', text: 'y = 2x', latex: 'y = 2x' } },
      { id: 'L2', pieces: [2], box: [0, 0, 100, 30], t_complete: 20, read: { kind: 'equation', text: 'one', latex: '' } },
    ],
  },
});

test('refs parse to items', () => {
  assert.deepEqual(parseRef('#12'), { kind: 'piece', id: 12 });
  assert.deepEqual(parseRef('#L3'), { kind: 'line', id: 'L3' });
  assert.deepEqual(parseRef('#s0e1'), { kind: 'element', id: 's0e1' });
  assert.equal(parseRef('#x1'), null);
  assert.equal(parseRef('12'), null);
});

test('board: lines and the pieces not in a line, by chapter, with seek times', () => {
  const idx = lectureIndex(boardReplay());
  // every piece and line can be cited
  assert.deepEqual([...idx.byRef.keys()].sort(), ['#1', '#2', '#3', '#4', '#5', '#6', '#L1', '#L2']);
  const listed = idx.groups.flatMap((g) => g.items.map((i) => i.ref));
  assert.deepEqual(listed.sort(), ['#1', '#2', '#3', '#4', '#L1'].sort()); // L2 has one piece: listed as the piece
  // seek time: first spoken start, else redraw_at, else t_seen; never before the clip start (5 s)
  assert.equal(idx.byRef.get('#1')!.t, 5);
  assert.equal(idx.byRef.get('#2')!.t, 22);
  assert.equal(idx.byRef.get('#3')!.t, 25);
  assert.equal(idx.byRef.get('#4')!.t, 40);
  assert.equal(idx.byRef.get('#L1')!.t, 80); // spoken about through its piece 6
  // erased and pre-existing writing is listed, and says so
  assert.equal(idx.byRef.get('#3')!.erasedAt, 90);
  assert.equal(idx.byRef.get('#1')!.preExisting, true);
  // chapters: items at or after 40 s go to "Later"
  assert.deepEqual(
    idx.groups.map((g) => [g.title, g.items.map((i) => i.ref)]),
    [
      ['Intro', ['#1', '#2', '#3']],
      ['Later', ['#4', '#L1']],
    ],
  );
  assert.equal(idx.byRef.get('#4')!.label, 'graph of a parabola');
  assert.equal(idx.count, 5);
});

test('a line the board does not show as one line is listed as its pieces', () => {
  const idx = lectureIndex(boardReplay(), { isLine: (ln) => ln.id !== 'L1' });
  const listed = idx.groups.flatMap((g) => g.items.map((i) => i.ref)).sort();
  assert.deepEqual(listed, ['#1', '#2', '#3', '#4', '#5', '#6']);
});

test('search filters by text, LaTeX, kind or ref', () => {
  const idx = lectureIndex(boardReplay());
  assert.deepEqual(
    filterGroups(idx.groups, 'parabola').flatMap((g) => g.items.map((i) => i.ref)),
    ['#4'],
  );
  assert.deepEqual(
    filterGroups(idx.groups, 'y = 2x').flatMap((g) => g.items.map((i) => i.ref)),
    ['#L1'],
  );
  assert.equal(filterGroups(idx.groups, '').length, 2);
  assert.equal(filterGroups(idx.groups, 'nothing like this').length, 0);
});

test('slides: every element can be cited; the browse list skips the repeats of build steps', () => {
  const replay: Replay = {
    version: 1,
    type: 'slides',
    video: { file: 'v.mp4', duration: 100, width: 1280, height: 720 },
    slides: {
      size: [1280, 720],
      states: [
        {
          id: 0, kind: 'slide', t_start: 0, t_end: 10, image: 'slides/000.png',
          elements: [{ id: 's0e1', kind: 'title', text: 'Compound interest', latex: '', box: [0, 0, 10, 10], spoken: [] }],
        },
        {
          id: 1, kind: 'addition', t_start: 10, t_end: 20, image: 'slides/001.png',
          elements: [
            { id: 's1e1', kind: 'title', text: 'Compound interest', latex: '', box: [0, 0, 10, 10] },
            { id: 's1e2', kind: 'equation', text: 'A = P(1+r)^n', latex: 'A=P(1+r)^n', box: [0, 0, 10, 10], spoken: [{ start: 14, end: 16 }] },
          ],
        },
      ],
    },
  };
  const idx = lectureIndex(replay);
  assert.deepEqual([...idx.byRef.keys()], ['#s0e1', '#s1e1', '#s1e2']);
  assert.deepEqual(idx.groups[0].items.map((i) => [i.ref, i.t]), [
    ['#s0e1', 0],
    ['#s1e2', 14],
  ]);
  assert.equal(idx.groups[0].title, 'Lecture'); // no chapters: one group
});

test('chips for attached items, and the item behind a line of the notes', () => {
  const idx = lectureIndex(boardReplay());
  assert.deepEqual(entryFor({ kind: 'piece', id: 3 }, idx), { item: { kind: 'piece', id: 3 }, label: 'x = 3', t: 25 });
  assert.deepEqual(entryFor({ kind: 'piece', id: 77 }, idx), { item: { kind: 'piece', id: 77 }, label: '#77', t: 0 });
  assert.deepEqual(entryFor({ kind: 'span', t0: 3, t1: 9 }, idx), { item: { kind: 'span', t0: 3, t1: 9 }, label: 'Speech', t: 3 });
  assert.deepEqual(entryFor({ kind: 'moment', t: 42 }, null), { item: { kind: 'moment', t: 42 }, label: 'Moment', t: 42 });
  assert.equal(matchNote(idx, { t: 80, latex: 'y=2x', text: '' })?.ref, '#L1');
  assert.equal(matchNote(idx, { t: 40, latex: '', text: 'Graph of a  parabola' })?.ref, '#4');
  assert.equal(matchNote(idx, { t: 40, latex: 'z', text: '' }), null);
});

test('chips of a saved question: lecture items named from the lecture, spans and moments by their saved label', () => {
  const idx = lectureIndex(boardReplay());
  assert.deepEqual(savedEntry({ kind: 'piece', id: 3, label: 'old name' }, idx), {
    item: { kind: 'piece', id: 3 },
    label: 'x = 3',
    t: 25,
  });
  assert.deepEqual(savedEntry({ kind: 'piece', id: 77, label: 'gone now' }, idx), {
    item: { kind: 'piece', id: 77 },
    label: 'gone now',
    t: 0,
  });
  assert.deepEqual(savedEntry({ kind: 'span', t0: 3, t1: 9, label: 'the words' }, idx), {
    item: { kind: 'span', t0: 3, t1: 9 },
    label: 'the words',
    t: 3,
  });
  assert.deepEqual(savedEntry({ kind: 'moment', t: 42, label: 'The board at 0:42' }, null), {
    item: { kind: 'moment', t: 42 },
    label: 'The board at 0:42',
    t: 42,
  });
  assert.deepEqual(savedEntry({ kind: 'moment', t: 42 }, null), entryFor({ kind: 'moment', t: 42 }, null));
});

test('a chip jumps to the later of when it is first talked about and when the board starts drawing it', () => {
  const r = boardReplay();
  r.board!.pieces = [
    piece(1, { t_seen: 30, t_shown: 31, spoken: [{ start: 20, end: 25 }] }), // talked about first, written later
    piece(2, { t_seen: 44, t_shown: 46, draw: [40, 45], spoken: [{ start: 38, end: 50 }] }),
    piece(3, { pre_existing: true, t_seen: 0, t_shown: 0, spoken: [{ start: 12, end: 14 }] }), // on the board already
    piece(4, { t_seen: 56, t_shown: 57, draw: [50, 56], spoken: [{ start: 44, end: 60 }] }),
    piece(5, { t_seen: 60, t_shown: 61, draw: [55, 60] }),
  ];
  r.board!.lines = [{ id: 'L1', pieces: [4, 5], box: [0, 0, 100, 30], t_complete: 61, read: { kind: 'equation', text: 'y', latex: 'y' } }];
  const idx = lectureIndex(r);
  assert.equal(idx.byRef.get('#1')!.t, 29.3); // the board starts drawing it 0.7 s before the camera sees it
  assert.equal(idx.byRef.get('#2')!.t, 40);
  assert.equal(idx.byRef.get('#3')!.t, 12);
  assert.equal(idx.byRef.get('#L1')!.t, 50); // the line's first piece starts at 50
});
