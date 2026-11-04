// Run with `npm test`. The lecture's items for Ask Muse: lines and the pieces not in one (including erased writing),
// grouped by chapter; seek times; refs; slide elements without the repeats of build steps; matching a notes line.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CHAPTER_MARGIN, entryFor, entryLatex, entryTime, ERASE_MARGIN, filterGroups, lectureIndex, matchNote, parseRef,
  pieceTime, savedEntry, SHOWN_SHARE, SHOWN_WAIT, shownFrom,
} from '../src/ask/items.ts';
import { alphaAt, chapterAt, progressAt, schedule } from '../src/player/timing.ts';
import type { AskItem, Piece, Replay } from '../src/types.ts';

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
  // seek time: where it is explained (first spoken start, else redraw_at, else t_seen), on to where the board shows
  // it whole; never before the clip start (5 s)
  assert.equal(idx.byRef.get('#1')!.t, 5);
  assert.equal(idx.byRef.get('#2')!.t, 22); // drawn by then
  assert.equal(idx.byRef.get('#3')!.t, 31); // drawn 29.3-31
  assert.equal(idx.byRef.get('#4')!.t, 41); // drawn 39.3-41
  assert.equal(idx.byRef.get('#L1')!.t, 80); // spoken about through its piece 6, long after it was written
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
  assert.deepEqual(entryFor({ kind: 'piece', id: 3 }, idx), { item: { kind: 'piece', id: 3 }, label: 'x = 3', t: 31 });
  assert.deepEqual(entryFor({ kind: 'piece', id: 77 }, idx), { item: { kind: 'piece', id: 77 }, label: '#77', t: 0 });
  assert.deepEqual(entryFor({ kind: 'span', t0: 3, t1: 9 }, idx), { item: { kind: 'span', t0: 3, t1: 9 }, label: 'Speech', t: 3 });
  assert.deepEqual(entryFor({ kind: 'moment', t: 42 }, null), { item: { kind: 'moment', t: 42 }, label: 'Moment', t: 42 });
  assert.equal(matchNote(idx, { t: 80, latex: 'y=2x', text: '' })?.ref, '#L1');
  assert.equal(matchNote(idx, { t: 40, latex: '', text: 'Graph of a  parabola' })?.ref, '#4');
  assert.equal(matchNote(idx, { t: 40, latex: 'z', text: '' }), null);
});

test('an attached chip typesets the formula Browse shows for it; speech, moments and plain text keep their label', () => {
  const r = boardReplay();
  r.board!.pieces.push(p34(), piece(9, { read: { kind: 'equation', text: 'x', latex: '$$x^2$$' } }));
  const idx = lectureIndex(r);
  const chip = (item: AskItem) => entryLatex(entryFor(item, idx), idx);
  assert.equal(chip({ kind: 'piece', id: 34 }), 'A=P(1+r)^n');
  assert.equal(chip({ kind: 'line', id: 'L1' }), 'y = 2x');
  assert.equal(chip({ kind: 'piece', id: 9 }), 'x^2');
  assert.equal(chip({ kind: 'piece', id: 4 }), ''); // a graph: no formula
  assert.equal(chip({ kind: 'piece', id: 77 }), ''); // gone from the lecture
  assert.equal(chip({ kind: 'span', t0: 1, t1: 2 }), '');
  assert.equal(chip({ kind: 'moment', t: 3 }), '');
  assert.equal(entryLatex({ item: { kind: 'piece', id: 34 }, label: 'A=P(1+r)^n', t: 1 }, null), '');
});

test('chips of a saved question: lecture items named from the lecture, spans and moments by their saved label', () => {
  const idx = lectureIndex(boardReplay());
  assert.deepEqual(savedEntry({ kind: 'piece', id: 3, label: 'old name' }, idx), {
    item: { kind: 'piece', id: 3 },
    label: 'x = 3',
    t: 31,
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

test('a chip jumps to where the board shows its writing: from where it is explained on to where it is drawn', () => {
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
  assert.equal(idx.byRef.get('#1')!.t, 31); // the board draws it 29.3-31
  assert.equal(idx.byRef.get('#2')!.t, 45); // drawn 40-45, talked about 38-50
  assert.equal(idx.byRef.get('#3')!.t, 12);
  // half each: #4 whole at 56 (60%), #5 then 80% drawn at about 59 (the board draws in whole levels of its order map)
  const tL1 = idx.byRef.get('#L1')!.t;
  const share = (t: number) => (progressAt(schedule(r.board!.pieces[3]), t) + progressAt(schedule(r.board!.pieces[4]), t)) / 2;
  assert.ok(tL1 >= 59 && tL1 <= 59.1, `${tL1}`);
  assert.ok(share(tL1) >= SHOWN_SHARE && share(tL1 - 0.01) < SHOWN_SHARE, `the first hundredth with 90%: ${tL1}`);
});

// test-full #34, A = P(1+r)^n: hidden while it was written, redrawn from 0:24 when it is talked about (23.3-31.1).
const p34 = (): Piece => ({
  id: 34, box: [592, 24, 68, 23], image: 'pieces/34.png', draw: [27.67, 31.33], t_seen: 28, t_shown: 31.33,
  revealed: true, redraw_at: 24, written_after: 23.33, spoken: [{ start: 23.3, end: 31.13 }],
  read: { kind: 'equation', text: 'A=P(1+r)^n', latex: 'A=P(1+r)^n' },
});

test('test-full #34: the chip shows the redrawn formula, not the empty spot where its redraw starts', () => {
  const p = p34();
  const t = pieceTime(p);
  const drawn = progressAt(schedule(p), t);
  assert.equal(drawn, 1, `drawn ${drawn} at ${t}`); // the old rule jumped to 24.0: nothing drawn
  assert.ok(t > 24 && t < 31.13, `${t} is still while it is talked about`);
  assert.equal(t, 28.26); // the end of its 24-28.26 redraw
});

test('a chip waits at most SHOWN_WAIT for its writing, and stops short of its erasing', () => {
  const slow = piece(1, { t_seen: 10, t_shown: 200, draw: [10, 200], spoken: [{ start: 9, end: 30 }] });
  assert.equal(pieceTime(slow), 10 + SHOWN_WAIT); // a scribble that grew for minutes
  // erased at 14, before it is whole: ERASE_MARGIN before (the board fades it out from 14; jumping to 14 itself showed
  // test-full #39 for an instant)
  const erased = piece(2, { t_seen: 10, t_shown: 20, draw: [10, 20], removed_at: 14, spoken: [{ start: 9, end: 30 }] });
  assert.equal(ERASE_MARGIN, 1);
  assert.equal(pieceTime(erased), 13);
  assert.equal(alphaAt(schedule(erased), pieceTime(erased)), 1);
  // erased within ERASE_MARGIN of where it is explained: there
  const gone = piece(4, { t_seen: 10, t_shown: 20, draw: [10, 20], removed_at: 10.5, spoken: [{ start: 9, end: 30 }] });
  assert.equal(pieceTime(gone), 10);
  const already = piece(3, { t_seen: 10, t_shown: 12, draw: [10, 12], spoken: [{ start: 40, end: 45 }] });
  assert.equal(pieceTime(already), 40); // drawn long before it is talked about: where it is talked about
  assert.equal(shownFrom([], 7), 7);
  assert.equal(SHOWN_SHARE, 0.9);
});

// synth3d-vector3d #8, the x and y axes: talked about 7.47-20.25, in "Coordinate axes" (7.5-20.1); hidden while it was
// drawn, and its order map runs on to 0:37.67 (the labels), two chapters later.
const v8 = (): Piece => ({
  id: 8, box: [59, 179, 372, 114], image: 'pieces/008.png', draw: [15.01, 37.67], t_seen: 16.33, t_shown: 19.67,
  written_after: 13, revealed: true, redraw_at: 13, spoken: [{ start: 7.47, end: 20.25 }],
  read: { kind: 'arrow', text: 'x-axis and y-axis arrows labeled x and y', latex: '' },
});

test('a chip never jumps into the next chapter, and Browse lists it under the chapter it is explained in', () => {
  const r = boardReplay();
  r.video.start = 0;
  r.chapters = [
    { t: 0, title: 'Intro to 3D vector' },
    { t: 7.5, title: 'Coordinate axes' },
    { t: 20.1, title: 'Plotting vector v' },
    { t: 30, title: 'Components and angle theta' },
  ];
  r.board!.pieces = [v8(), piece(2, { t_seen: 21, t_shown: 22, spoken: [{ start: 21, end: 25 }] })];
  r.board!.lines = [];
  const idx = lectureIndex(r);
  const it = idx.byRef.get('#8')!;
  assert.equal(it.explained, 13); // its redraw starts (talked about from 7.47)
  assert.equal(it.t, 20); // CHAPTER_MARGIN before "Plotting vector v" (20.1): partly drawn, still talked about
  assert.equal(CHAPTER_MARGIN, 0.1);
  assert.equal(chapterAt(r.chapters, it.t), 1);
  assert.ok(progressAt(schedule(v8()), it.t) > 0.2, 'some of it is drawn there (none at 0:13)');
  assert.deepEqual(
    idx.groups.map((g) => [g.title, g.items.map((i) => i.ref)]),
    [
      ['Intro to 3D vector', []],
      ['Coordinate axes', ['#8']],
      ['Plotting vector v', ['#2']],
      ['Components and angle theta', []],
    ],
  );
  // without the chapters it would wait for all of it: 0:37.67, listed under "Components and angle theta"
  assert.equal(pieceTime(v8()), 37.67);
});

test('a lecture processed from part-way: the wait for the writing starts where the processed part does', () => {
  // processed from 6:20 (as gest-test): a formula talked about and written from 1:40 on to 6:35 jumps to the end of
  // the chapter the processed part starts in (6:28.9, 98% drawn), not to 6:20 (the wait ran out at 2:10, before it)
  const r = boardReplay();
  r.video.start = 380;
  r.chapters = [{ t: 0, title: 'Before' }, { t: 200, title: 'Earlier' }, { t: 389, title: 'Later' }];
  r.board!.pieces = [piece(1, { t_seen: 100, t_shown: 395, draw: [100, 395], spoken: [{ start: 100, end: 400 }] })];
  r.board!.lines = [];
  const it = lectureIndex(r).byRef.get('#1')!;
  assert.equal(it.explained, 380);
  assert.equal(it.t, 388.9); // still "Earlier" (from 3:20 to 6:29), the most of it drawn there
  assert.ok(progressAt(schedule(r.board!.pieces[0]), it.t) > 0.97);
});

test("a chip shows where it jumps: the lecture's own time for its item, not an older time saved with it", () => {
  const idx = lectureIndex(boardReplay());
  // the Notes page's "Ask" puts the note's time (first talked about) on the tray; the chip jumps to the lecture's
  assert.equal(entryTime({ item: { kind: 'piece', id: 3 }, label: 'x = 3', t: 25 }, idx), 31);
  assert.equal(entryTime({ item: { kind: 'line', id: 'L1' }, label: 'y = 2x', t: 61 }, idx), 80);
  assert.equal(entryTime({ item: { kind: 'piece', id: 77 }, label: 'gone', t: 12 }, idx), 12); // gone from the lecture
  assert.equal(entryTime({ item: { kind: 'moment', t: 42 }, label: 'Moment', t: 42 }, idx), 42);
  assert.equal(entryTime({ item: { kind: 'span', t0: 3, t1: 9 }, label: 'Speech', t: 3 }, idx), 3);
  assert.equal(entryTime({ item: { kind: 'piece', id: 3 }, label: 'x = 3', t: 25 }, null), 25);
});

test('a notes line finds its item by where it is explained, not where its chip jumps', () => {
  const r = boardReplay();
  r.chapters = [];
  const e = { kind: 'equation', text: 'e', latex: 'e' };
  r.board!.pieces = [
    piece(1, { t_seen: 40, t_shown: 41, draw: [40, 70], spoken: [{ start: 40, end: 45 }], read: e }), // written on to 70
    piece(2, { t_seen: 60, t_shown: 61, spoken: [{ start: 60, end: 62 }], read: e }),
  ];
  r.board!.lines = [];
  const idx = lectureIndex(r);
  assert.equal(idx.byRef.get('#1')!.t, 70);
  assert.equal(matchNote(idx, { t: 41, latex: 'e' })?.ref, '#1');
  assert.equal(matchNote(idx, { t: 59, latex: 'e' })?.ref, '#2');
});
