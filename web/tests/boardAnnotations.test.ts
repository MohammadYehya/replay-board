// Run with `npm test`. The teacher's hand gestures on a board: timing (incl. fading early when the writing is
// erased), the colours, an underline aimed at a whole region of the board, an arrow between two stacked lines, arrows
// that keep off the other writing (going round it, or to a free side of their target), and which of the backend's
// lines are really one written line.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  FADE_OUT, HAND_ON_DARK, HAND_ON_LIGHT, HOLD, annotationAlpha, annotationTiming, arrowEnds, arrowGeometry,
  boardAnnotationTiming, boardAnnotations, boardArrowBoxes, boardObstacles, boardTargetBox, drawProgress, fitUnderline,
  pieceIdOf, pulseRings, sideArrow, stacked, targetGoneAt, targetPieceIds, toolColour, underlinePath, underlineY,
  type ArrowGeom,
} from '../src/player/slideTiming.ts';
import { isWrittenLine, schedule } from '../src/player/timing.ts';
import type { BoardData, BoardLine, Box, Piece, SlideAnnotation } from '../src/types.ts';

const near = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) <= eps;
const SIZE: [number, number] = [800, 315];
const piece = (id: number, box: Box, removed_at: number | null = null): Piece => ({
  id, box, image: `pieces/${id}.png`, t_seen: 380, t_shown: 380, removed_at,
});
const hand = (o: Partial<SlideAnnotation>): SlideAnnotation => ({
  type: 'spotlight', tool: 'hand', t_start: 389.33, t_end: 390, target: ['17'], box: [456, 202, 104, 69],
  from: null, to: null, path: [], ...o,
});

// Four gestures from a run of the pipeline on data/jobs/gest-test (board lecture, 380-478 s).
const GEST: SlideAnnotation[] = [
  hand({ path: [[404.3, 234.0], [449.3, 220.1]] }),
  hand({
    type: 'underline', t_start: 463.0, t_end: 463.67, target: ['4'], box: [296, 9, 434, 255],
    path: [[408.6, 142.4], [478.5, 169.3]],
  }),
  hand({ t_start: 464.0, t_end: 465.0, target: ['38'], box: [688, 118, 44, 19] }),
  hand({ type: 'pulse', t_start: 470.67, t_end: 473.33, target: ['19', '18'], box: [186, 234, 23, 17] }),
];
const PIECES = [
  piece(4, [296, 9, 434, 255]),
  piece(17, [456, 202, 104, 69]),
  piece(18, [221, 212, 91, 22]),
  piece(19, [186, 234, 23, 17]),
  piece(38, [688, 118, 44, 19]),
];

test('target ids are piece ids given as strings', () => {
  assert.equal(pieceIdOf('17'), 17);
  assert.equal(pieceIdOf(' 4 '), 4);
  assert.equal(pieceIdOf(38), 38);
  assert.equal(pieceIdOf(''), null);
  assert.equal(pieceIdOf('s2e1'), null); // a slide element id
  assert.equal(pieceIdOf('1.5'), null);
  assert.deepEqual(targetPieceIds(hand({ target: ['19', 'x', '18'] })), [19, 18]);
  assert.deepEqual(targetPieceIds(hand({ target: undefined })), []);
});

test('a hand gesture whose writing stays is timed exactly like a slide annotation', () => {
  for (const a of GEST) assert.deepEqual(boardAnnotationTiming(a, PIECES), annotationTiming(a));
  const tm = boardAnnotationTiming(GEST[0], PIECES);
  assert.equal(tm.holdEnd, 390 + HOLD);
  assert.equal(tm.end, 390 + HOLD + FADE_OUT);
  assert.equal(annotationAlpha(tm, 389.2), 0); // before the gesture
  assert.equal(annotationAlpha(tm, 390.5), 1); // held
  assert.equal(annotationAlpha(tm, 391.95), 0); // gone
  // the underline is drawn over the gesture, the pulse keeps ringing while the hand taps
  const ul = boardAnnotationTiming(GEST[1], PIECES);
  assert.ok(drawProgress(ul, 463.3) > 0.2 && drawProgress(ul, 463.3) < 0.8);
  assert.equal(drawProgress(ul, 463.67), 1);
  const pulse = boardAnnotationTiming(GEST[3], PIECES);
  assert.ok(pulseRings(pulse, 473).some((r) => r.alpha > 0));
});

test('erased writing takes its annotation with it, even inside the hold', () => {
  // "= e" (piece 38) is wiped 0.4 s after the hand leaves it: the spotlight fades with it, not 1.5 s later
  const erased = PIECES.map((p) => (p.id === 38 ? { ...p, removed_at: 465.4 } : p));
  assert.equal(targetGoneAt(GEST[2], erased), 465.4);
  const tm = boardAnnotationTiming(GEST[2], erased);
  assert.equal(tm.holdEnd, 465.4);
  assert.equal(annotationAlpha(tm, 465.35), 1);
  assert.ok(annotationAlpha(tm, 465.6) > 0 && annotationAlpha(tm, 465.6) < 1);
  assert.equal(annotationAlpha(tm, 465.4 + FADE_OUT), 0);
  assert.equal(annotationAlpha(tm, 466.0), 0); // the full hold would still show it here
  assert.equal(annotationAlpha(annotationTiming(GEST[2]), 466.0), 1);
  // erased while the hand is still on it: fades from there
  const during = PIECES.map((p) => (p.id === 38 ? { ...p, removed_at: 464.5 } : p));
  assert.equal(boardAnnotationTiming(GEST[2], during).holdEnd, 464.5);
  // erased long after: the usual hold
  const later = PIECES.map((p) => (p.id === 38 ? { ...p, removed_at: 500 } : p));
  assert.equal(boardAnnotationTiming(GEST[2], later).holdEnd, 465 + HOLD);
  // erased before the gesture started (data that cannot be right): ignored, like a nonsense slide end
  const before = PIECES.map((p) => (p.id === 38 ? { ...p, removed_at: 450 } : p));
  assert.equal(boardAnnotationTiming(GEST[2], before).holdEnd, 465 + HOLD);
});

test('with several targets the first one erased ends the annotation', () => {
  const pieces = PIECES.map((p) => (p.id === 18 ? { ...p, removed_at: 474.0 } : p.id === 19 ? { ...p, removed_at: 476 } : p));
  assert.equal(targetGoneAt(GEST[3], pieces), 474.0);
  assert.equal(boardAnnotationTiming(GEST[3], pieces).holdEnd, 474.0);
  // a target that is not a piece, or no piece erased: nothing to end it early
  assert.equal(targetGoneAt(hand({ target: ['99', 'abc'] }), pieces), null);
  assert.equal(targetGoneAt(GEST[0], PIECES), null);
  assert.equal(targetGoneAt(hand({ target: undefined }), pieces), null);
});

test('the hand is teal, apart from the cursor and the laser, and suits the board', () => {
  assert.equal(toolColour('hand', true), HAND_ON_LIGHT);
  assert.equal(toolColour('hand', false), HAND_ON_DARK);
  assert.equal(toolColour('hand'), HAND_ON_LIGHT);
  assert.notEqual(HAND_ON_LIGHT, toolColour('cursor'));
  assert.notEqual(HAND_ON_LIGHT, toolColour('laser'));
  // slides keep their colours
  assert.equal(toolColour('cursor', false), '#f59e0b');
  assert.equal(toolColour('laser', false), '#ef4444');
});

test('an underline aimed at a whole region goes under the stretch the hand swept', () => {
  // gest-test: the target is piece 4, the middle third of the board; the hand swept under "2 ln|x|"
  const ul = fitUnderline(GEST[1], SIZE);
  const [x, , w] = ul.box;
  assert.ok(near(x, 408.6, 0.01) && near(x + w, 478.5, 0.01));
  assert.ok(near(underlineY(ul, SIZE), 169.3 + 2 * (800 / 960), 0.01)); // just under the sweep
  const d = underlinePath(ul.box, SIZE, underlineY(ul, SIZE));
  const xs = [...d.matchAll(/([\d.]+) ([\d.]+)/g)].map((m) => +m[1]);
  assert.ok(Math.min(...xs) > 400 && Math.max(...xs) < 490); // not across the whole region (296-730)
  assert.deepEqual(ul.target, ['4']); // still about piece 4 (its erasure still ends it)
  // a short sweep still gets a visible line, inside the region
  const short = fitUnderline({ ...GEST[1], path: [[440, 160], [452, 161]] }, SIZE);
  assert.ok(near(short.box[2], 20) && near(short.box[0], 436));
  const atEdge = fitUnderline({ ...GEST[1], path: [[716, 160], [740, 161]] }, SIZE);
  assert.ok(atEdge.box[3] < 30 && atEdge.box[2] > 14 && atEdge.box[0] + atEdge.box[2] <= 730 + 1e-9);
  // the sweep is clamped to the region
  const wide = fitUnderline({ ...GEST[1], path: [[250, 160], [500, 161]] }, SIZE);
  assert.ok(near(wide.box[0], 296) && near(wide.box[0] + wide.box[2], 500));
  // the hand came from the side, or barely reached the region, or swept below it, or left no usable sweep: the
  // region's own box, with the line under it (see the next test)
  const under = 9 + 255 + 5 * (800 / 960);
  for (const path of [[[200, 160], [290, 162]], [[250, 160], [300, 161]], [[400, 290], [470, 291]], [[420, 150]]]) {
    const ul2 = fitUnderline({ ...GEST[1], path: path as [number, number][] }, SIZE);
    assert.deepEqual(ul2.box, GEST[1].box);
    assert.ok(near(ul2.underline_y!, under), `${path}: ${ul2.underline_y}`);
  }
  // a later run: the target is two small lines (57 px) and the hand swept in from the left of it
  const twoLines = hand({
    type: 'underline', t_start: 434, t_end: 435.33, target: ['28'], box: [652, 252, 63, 57],
    path: [[536.6, 286.8], [613.4, 294.6], [606.5, 293.3], [636.5, 291.7]],
  });
  assert.equal(fitUnderline(twoLines, SIZE), twoLines);
  // a line of writing, an explicit height, or no usable sweep: left as it is
  const line = hand({ type: 'underline', box: [100, 100, 300, 24], path: [[110, 126], [200, 127]] });
  assert.equal(fitUnderline(line, SIZE), line);
  const given = { ...GEST[1], underline_y: 150 };
  assert.equal(fitUnderline(given, SIZE), given);
  assert.equal(fitUnderline(GEST[0], SIZE), GEST[0]); // not an underline
});

/** The height an underline is drawn at: the y of its path's start (M x y Q ...). */
const drawnY = (a: SlideAnnotation, size: [number, number]) => +underlinePath(a.box, size, underlineY(a, size)).split(' ')[1];

test('an underline on a region it cannot be fitted to goes under the region, not through its middle', () => {
  // gest-test's piece 4 (y 9-264) with the hand sweeping beside it at y 160, or just touching its left edge: drawn at
  // the sweep's height the line would cross the middle of the region
  const u = 800 / 960;
  for (const path of [[[200, 160], [290, 162]], [[250, 160], [300, 161]]] as [number, number][][]) {
    const raw = { ...GEST[1], path };
    assert.ok(drawnY(raw, SIZE) < 200); // the bug: through the region
    const ul = boardAnnotations({ size: SIZE, pieces: PIECES, annotations: [raw] })[0];
    assert.ok(near(drawnY(ul, SIZE), 264 + 5 * u, 0.06), `drawn at ${drawnY(ul, SIZE)}`);
    const d = underlinePath(ul.box, SIZE, underlineY(ul, SIZE));
    const xs = [...d.matchAll(/([\d.]+) ([\d.]+)/g)].map((m) => +m[1]);
    assert.ok(Math.min(...xs) < 296 && Math.max(...xs) > 730); // across the whole region
  }
  // a region at the bottom of the board: the line stays on the board
  const low = fitUnderline({ ...GEST[1], box: [296, 100, 434, 214], path: [[200, 160], [290, 162]] }, SIZE);
  assert.ok(near(drawnY(low, SIZE), 315 - 3 * u, 0.06));
  // a line of writing (not a region) still follows the sweep, and a fitted sweep is unchanged
  const line = hand({ type: 'underline', box: [100, 100, 300, 24], path: [[110, 126], [200, 127]] });
  assert.ok(near(drawnY(fitUnderline(line, SIZE), SIZE), 127 + 2 * u, 0.06));
  assert.ok(near(drawnY(fitUnderline(GEST[1], SIZE), SIZE), 169.3 + 2 * u, 0.06));
});

test("the board's annotations: the drawable ones, underlines fitted", () => {
  const board: BoardData = {
    size: SIZE, pieces: PIECES,
    annotations: [...GEST, { ...GEST[0], box: [1, 2] as unknown as Box }, { ...GEST[0], t_start: NaN }],
  };
  const list = boardAnnotations(board);
  assert.equal(list.length, 4);
  assert.equal(list[0], GEST[0]);
  assert.notDeepEqual(list[1].box, GEST[1].box);
  assert.deepEqual(boardAnnotations({ size: SIZE, pieces: PIECES }), []);
});

/** Points along an arrow's quadratic shaft (M ax ay Q cx cy bx by). */
const shaft = (d: string): [number, number][] => {
  const n = d.match(/-?[\d.]+/g)!.map(Number);
  return Array.from({ length: 41 }, (_, k) => {
    const s = k / 40;
    return [
      (1 - s) ** 2 * n[0] + 2 * s * (1 - s) * n[2] + s * s * n[4],
      (1 - s) ** 2 * n[1] + 2 * s * (1 - s) * n[3] + s * s * n[5],
    ];
  });
};
const inBox = (p: [number, number], b: Box) => p[0] >= b[0] && p[0] <= b[0] + b[2] && p[1] >= b[1] && p[1] <= b[1] + b[3];

test('an arrow between two lines stacked close curves round their ends instead of sitting on the writing', () => {
  // test-full, 335.7 s: the hand goes from "= $2.714567..." (line L7) up to "A = (1 + 1/365)^365" (line L6), 8 px above
  const SIZE_F: [number, number] = [800, 314];
  const L7: Box = [601, 216, 129, 19];
  const L6: Box = [550, 175, 136, 33];
  const a = hand({
    type: 'arrow', t_start: 335.67, t_end: 336.33, target: ['60', '59'], box: [601, 216, 63, 19],
    from: [632.5, 225.5], to: [646.0, 195.5], path: [[632.5, 225.5], [646.0, 195.5]],
  });
  assert.ok(stacked(L7, L6) && stacked(L6, L7));
  const g = arrowGeometry(...arrowEnds(a, L7, L6), SIZE_F, L7, L6);
  for (const p of shaft(g.d)) assert.ok(!inBox(p, L7) && !inBox(p, L6), `shaft point ${p} on the writing`);
  assert.ok(g.from[0] > 730 && near(g.from[1], 225.5)); // leaves from the end of the result
  assert.ok(g.tip[0] > 686 && g.tip[0] < 700 && near(g.tip[1], 191.5)); // comes back into the end of the formula
  const xs = shaft(g.d).map((p) => p[0]);
  assert.ok(Math.max(...xs) < 760); // a small bow, not a detour
  assert.match(g.head, /^M[\d.]+ [\d.]+ L[\d.]+ [\d.]+ L[\d.]+ [\d.]+$/);
  // no room on the right of the board: round the left ends instead
  const left = sideArrow([700, 216, 95, 19], [690, 175, 105, 33], SIZE_F);
  assert.ok(left.from[0] < 700 && left.tip[0] < 690);
  // lines side by side, far apart, or overlapping are not "stacked"; a long arrow is drawn as before
  assert.ok(!stacked([100, 100, 50, 20], [160, 100, 50, 20]));
  assert.ok(!stacked([100, 100, 50, 20], [100, 110, 50, 20]));
  const long = arrowGeometry([147, 223], [507, 235.5], SIZE_F, [139, 200, 16, 46], [455, 201, 104, 69]);
  assert.ok(long.from[0] > 147 && long.tip[0] < 455); // pulled back to just outside both boxes, straight across
});

test("an arrow's ends are its pieces' lines, unless it goes between two pieces of one line", () => {
  const pieces = [piece(60, [601, 216, 63, 19]), piece(62, [680, 221, 50, 10]), piece(59, [606, 183, 80, 25]), piece(29, [139, 200, 16, 46])];
  const lines: BoardLine[] = [
    { id: 'L6', pieces: [58, 59], box: [550, 175, 136, 33], t_complete: 300 },
    { id: 'L7', pieces: [60, 62], box: [601, 216, 129, 19], t_complete: 320 },
  ];
  const up = hand({ type: 'arrow', target: ['60', '59'] });
  assert.deepEqual(boardTargetBox(up, '60', pieces, lines), [601, 216, 129, 19]);
  assert.deepEqual(boardTargetBox(up, '59', pieces, lines), [550, 175, 136, 33]);
  assert.deepEqual(boardTargetBox(up, '29', pieces, lines), [139, 200, 16, 46]); // no line: the piece
  assert.deepEqual(boardTargetBox(up, '60', pieces), [601, 216, 63, 19]); // no lines given
  // from one fragment of "= $2.714567..." to another: each fragment's own box
  const along = hand({ type: 'arrow', target: ['60', '62'] });
  assert.deepEqual(boardTargetBox(along, '60', pieces, lines), [601, 216, 63, 19]);
  assert.deepEqual(boardTargetBox(along, '62', pieces, lines), [680, 221, 50, 10]);
  assert.equal(boardTargetBox(up, '99', pieces, lines), null);
  assert.equal(boardTargetBox(up, 's2e1', pieces, lines), null);
});

test("an arrow uses a line's box only while that line is finished and still on the board", () => {
  const pieces = [piece(60, [601, 216, 63, 19]), piece(59, [606, 183, 80, 25])];
  const L6: BoardLine = { id: 'L6', pieces: [58, 59], box: [550, 175, 136, 33], t_complete: 315.33 };
  const L7: BoardLine = { id: 'L7', pieces: [60, 62], box: [601, 216, 129, 19], t_complete: 326.33 };
  const at = (t_start: number, t_end: number) => hand({ type: 'arrow', target: ['60', '59'], t_start, t_end });
  // L7 is still being written ("567...~" comes at 326.33): the piece itself, not the line it will become
  assert.deepEqual(boardTargetBox(at(322, 323), '60', pieces, [L6, L7]), [601, 216, 63, 19]);
  assert.deepEqual(boardTargetBox(at(322, 323), '59', pieces, [L6, L7]), [550, 175, 136, 33]);
  // finished by the end of the gesture (or exactly then): the line
  assert.deepEqual(boardTargetBox(at(326, 326.5), '60', pieces, [L6, L7]), [601, 216, 129, 19]);
  assert.deepEqual(boardTargetBox(at(326, 326.33), '60', pieces, [L6, L7]), [601, 216, 129, 19]);
  // the line was erased before the gesture: the piece; erased after it started: still the line
  const gone = { ...L7, removed_at: 330 };
  assert.deepEqual(boardTargetBox(at(330, 331), '60', pieces, [L6, gone]), [601, 216, 63, 19]);
  assert.deepEqual(boardTargetBox(at(340, 341), '60', pieces, [L6, gone]), [601, 216, 63, 19]);
  assert.deepEqual(boardTargetBox(at(329, 331), '60', pieces, [L6, gone]), [601, 216, 129, 19]);
  // no t_end: timed by t_start
  assert.deepEqual(boardTargetBox({ ...at(327, 0), t_end: NaN }, '60', pieces, [L6, L7]), [601, 216, 129, 19]);
  assert.deepEqual(boardTargetBox({ ...at(325, 0), t_end: NaN }, '60', pieces, [L6, L7]), [601, 216, 63, 19]);
});

test('an arrow points to its second target, as the backend does, not its last', () => {
  const pieces = [piece(60, [601, 216, 63, 19]), piece(59, [606, 183, 80, 25]), piece(29, [139, 200, 16, 46])];
  const lines: BoardLine[] = [
    { id: 'L6', pieces: [58, 59], box: [550, 175, 136, 33], t_complete: 300 },
    { id: 'L7', pieces: [60, 62], box: [601, 216, 129, 19], t_complete: 320 },
  ];
  const start = (p: Piece) => schedule(p).start;
  const three = hand({ type: 'arrow', target: ['60', '59', '29'], from: [632.5, 225.5], to: [646, 195.5] });
  const e = boardArrowBoxes(three, pieces, lines, SIZE, start);
  assert.deepEqual(e.from, [601, 216, 129, 19]);
  assert.deepEqual(e.to, [550, 175, 136, 33]); // target[1] ("A = (1 + 1/365)^365"), whose centre `to` is
  const two = boardArrowBoxes({ ...three, target: ['60', '59'] }, pieces, lines, SIZE, start);
  assert.deepEqual([two.from, two.to], [e.from, e.to]);
  // one target: no from box, to that target
  const one = boardArrowBoxes({ ...three, target: ['29'] }, pieces, lines, SIZE, start);
  assert.deepEqual([one.from, one.to], [null, [139, 200, 16, 46]]);
  const none = boardArrowBoxes({ ...three, target: [] }, pieces, lines, SIZE, start);
  assert.deepEqual([none.from, none.to], [null, null]);
});

// test-full (800 x 314): every piece as [id, box, t_seen (-1: on the board from the start), removed_at], and the
// lines that are one written line.
const SIZE_F: [number, number] = [800, 314];
const PIECES_F: Piece[] = ([
  [1, [6, 9, 43, 189], -1], [3, [294, 8, 266, 256], -1], [4, [24, 11, 69, 48], -1], [5, [322, 10, 55, 20], -1],
  [6, [716, 10, 64, 19], -1], [7, [182, 13, 68, 39], -1], [8, [82, 35, 10, 17], -1], [9, [265, 37, 66, 31], -1],
  [10, [46, 45, 107, 132], -1], [14, [371, 170, 18, 35], -1], [15, [265, 173, 55, 8], -1],
  [16, [455, 201, 104, 69], -1], [17, [220, 212, 90, 22], -1], [18, [664, 220, 18, 15], -1],
  [19, [121, 52, 73, 143], -1], [20, [177, 116, 61, 39], -1], [21, [229, 180, 16, 9], -1],
  [22, [185, 234, 23, 17], -1], [23, [39, 70, 43, 39], -1], [24, [155, 149, 7, 16], -1],
  [28, [130, 225, 16, 35], -1], [29, [139, 200, 16, 46], 2.67], [31, [45, 172, 12, 7], 4.67],
  [34, [592, 24, 68, 23], 28], [35, [731, 40, 19, 13], 42.33], [36, [729, 57, 40, 17], 46.33],
  [37, [768, 57, 26, 6], 58.67], [38, [527, 114, 18, 14], 118.67, 141], [39, [515, 141, 115, 59], 125.67, 141],
  [41, [655, 6, 35, 8], 154.67], [42, [557, 65, 18, 16], 179.67], [43, [586, 61, 65, 16], 187.33],
  [45, [600, 62, 50, 27], 192.67], [46, [496, 133, 38, 31], 200.67], [47, [550, 97, 28, 13], 211.67],
  [48, [586, 99, 19, 10], 219.67], [49, [610, 95, 42, 23], 224.33], [52, [598, 124, 7, 4], 246.67],
  [53, [612, 123, 29, 11], 254.33], [54, [549, 135, 116, 36], 260.67], [55, [606, 101, 3, 14], 277.67],
  [56, [606, 124, 5, 13], 283.33], [57, [605, 163, 30, 16], 283.33], [58, [550, 175, 53, 21], 301.67],
  [59, [606, 183, 80, 25], 310.67], [60, [601, 216, 63, 19], 319.33], [62, [680, 221, 50, 10], 326],
  [63, [570, 241, 123, 25], 349.67], [64, [688, 252, 27, 21], 359.33], [65, [715, 246, 56, 15], 362.67],
  [67, [624, 274, 86, 19], 374], [68, [606, 264, 90, 44], 381.67], [71, [678, 89, 108, 25], 402.67],
  [73, [776, 90, 18, 7], 422.67], [74, [688, 117, 43, 19], 426.33], [76, [585, 6, 64, 17], 448.67],
] as [number, Box, number, number?][]).map(([id, box, seen, gone]) => ({
  id, box, image: '', t_seen: Math.max(0, seen), t_shown: Math.max(0, seen), pre_existing: seen < 0, removed_at: gone ?? null,
}));
const LINES_F: BoardLine[] = [
  { id: 'L1', pieces: [4, 8], box: [24, 11, 69, 48], t_complete: 2.33 },
  { id: 'L2', pieces: [42, 43, 45], box: [557, 61, 94, 28], t_complete: 194.67 },
  { id: 'L3', pieces: [47, 48, 55, 49], box: [550, 95, 102, 23], t_complete: 277.67 },
  { id: 'L4', pieces: [52, 56, 53], box: [598, 123, 43, 14], t_complete: 283.33 },
  { id: 'L5', pieces: [54, 57], box: [549, 135, 116, 44], t_complete: 284.67 },
  { id: 'L6', pieces: [58, 59], box: [550, 175, 136, 33], t_complete: 315.33 },
  { id: 'L7', pieces: [60, 62], box: [601, 216, 129, 19], t_complete: 326.33 },
  { id: 'L8', pieces: [63, 64, 65], box: [570, 241, 201, 32], t_complete: 365.33 },
];
const startOf = (p: Piece) => schedule(p).start;
const boxOfPiece = (id: number) => PIECES_F.find((p) => p.id === id)!.box;
const lineBox = (id: string) => LINES_F.find((ln) => ln.id === id)!.box;

/** Where an arrow is drawn: points along its shaft, and along the wings of its head. */
const drawn = (g: ArrowGeom): [number, number][] => {
  const head = g.head.match(/-?[\d.]+/g)!.map(Number); // M w1 L tip L w2
  const seg = (k: number): [number, number][] =>
    Array.from({ length: 9 }, (_, i) => [head[k] + ((head[2] - head[k]) * i) / 8, head[k + 1] + ((head[3] - head[k + 1]) * i) / 8]);
  return [...shaft(g.d), ...seg(0), ...seg(4)];
};
/** The arrow for a test-full gesture as the board draws it: with or without the other writing to keep off. */
const boardArrow = (a: SlideAnnotation, keepOff = true) => {
  const e = boardArrowBoxes(a, PIECES_F, LINES_F, SIZE_F, startOf);
  const g = arrowGeometry(...arrowEnds(a, e.from, e.to, 8 * (800 / 960)), SIZE_F, e.from, e.to, keepOff ? e.obstacles : null);
  return { e, g };
};

test('an arrow on the board goes round the writing between its ends instead of over it', () => {
  // test-full, 436.33 s: from "A = (1+1)^1" (piece 43, line L2) down to "#3. A = (1+1/12)^12 =" (piece 54, line L5),
  // over "= (1+1/2)^2" (49, in L3) and "2.25" (53, in L4), its tip on "=$2.25" (L4) instead of L5
  const a = hand({
    type: 'arrow', t_start: 436.33, t_end: 438.67, target: ['43', '54'], box: [586, 61, 65, 16],
    from: [618.5, 69.0], to: [607.0, 153.0],
    path: [[617.0, 83.7], [633.1, 79.6], [637.0, 71.5], [644.0, 80.9], [639.5, 113.3], [638.3, 121.4], [651.3, 149.8]],
  });
  const before = boardArrow(a, false).g;
  assert.ok(shaft(before.d).some((p) => inBox(p, boxOfPiece(49))) && shaft(before.d).some((p) => inBox(p, boxOfPiece(53))));
  assert.ok(drawn(before).some((p) => inBox(p, lineBox('L4'))));
  const { e, g } = boardArrow(a);
  assert.deepEqual([e.from, e.to], [lineBox('L2'), lineBox('L5')]);
  for (const b of [lineBox('L3'), lineBox('L4'), boxOfPiece(49), boxOfPiece(53)]) assert.ok(e.obstacles!.some((o) => o === b));
  for (const p of drawn(g)) {
    for (const o of e.obstacles!) assert.ok(!inBox(p, o), `${p} on the writing at ${o}`);
    assert.ok(!inBox(p, lineBox('L2')) && !inBox(p, lineBox('L5')));
  }
  // round the right ends, from the end of L2 to the near (right) edge of L5
  assert.ok(g.from[0] > 651 && g.from[0] < 660 && g.from[1] > 61 && g.from[1] < 89);
  assert.ok(g.tip[0] > 665 && g.tip[0] < 672 && g.tip[1] > 135 && g.tip[1] < 179);
  assert.ok(Math.max(...shaft(g.d).map((p) => p[0])) < 678); // inside the gap before "A = (1 + 1/oo?)^oo" (71)
});

test('an arrow to writing beyond other lines goes round their ends to the near edge of its target', () => {
  // test-full, 464 s: from "= $2.7145" (piece 60, line L7) up to "= $e." (piece 74), across "A = (1 + 1/365)^365" (59)
  const a = hand({
    type: 'arrow', t_start: 464.0, t_end: 465.0, target: ['60', '74'], box: [601, 216, 63, 19],
    from: [632.5, 225.5], to: [709.5, 126.5], path: [[651.3, 197.5], [695.5, 129.3], [701.5, 146.8]],
  });
  const before = boardArrow(a, false).g;
  assert.ok(shaft(before.d).some((p) => inBox(p, boxOfPiece(59))));
  const { e, g } = boardArrow(a);
  assert.deepEqual([e.from, e.to], [lineBox('L7'), boxOfPiece(74)]);
  for (const p of drawn(g)) {
    for (const o of e.obstacles!) assert.ok(!inBox(p, o), `${p} on the writing at ${o}`);
    assert.ok(!inBox(p, lineBox('L7')) && !inBox(p, boxOfPiece(74)));
  }
  assert.ok(g.from[0] > 730 && near(g.from[1], 225.5)); // from the end of L7
  assert.ok(g.tip[0] > 731 && g.tip[0] < 738 && near(g.tip[1], 126.5)); // into the right edge of "= $e."
  // "D = (1-r)^n" (76) was written at 448 s: in the way at 464 s, not yet at 436 s; "$5" (38) was wiped at 141 s
  assert.ok(e.obstacles!.includes(boxOfPiece(76)) && !e.obstacles!.includes(boxOfPiece(38)));
  const early = boardArrowBoxes({ ...a, t_start: 436.33, t_end: 438.67 }, PIECES_F, LINES_F, SIZE_F, startOf);
  assert.ok(!early.obstacles!.includes(boxOfPiece(76)));
});

test('arrows clear of the other writing are drawn as before', () => {
  // test-full, 389.33 s: the long arrow from "x=0" (29) across the graph to "x = -1, ..." (16)
  const long = hand({
    type: 'arrow', t_start: 389.33, t_end: 390.0, target: ['29', '16'], box: [139, 200, 16, 46],
    from: [147.0, 223.0], to: [507.0, 235.5], path: [[407.6, 230.4], [448.5, 219.2]],
  });
  const { e, g } = boardArrow(long);
  assert.deepEqual(g, boardArrow(long, false).g);
  assert.ok(e.obstacles!.length > 20);
  assert.ok(!e.obstacles!.includes(boxOfPiece(3))); // a graph is a region: the arrow may cross its blank space
  // test-full, 335.67 s: between two lines stacked close, round their ends as before
  const close = hand({
    type: 'arrow', t_start: 335.67, t_end: 336.33, target: ['60', '59'], box: [601, 216, 63, 19],
    from: [632.5, 225.5], to: [646.0, 195.5], path: [[600.7, 242.9], [629.2, 185.3]],
  });
  assert.deepEqual(boardArrow(close).g, boardArrow(close, false).g);
  assert.deepEqual(boardArrow(close).g, sideArrow(lineBox('L7'), lineBox('L6'), SIZE_F));
  // nothing to keep off: exactly as without obstacles
  const plain = arrowGeometry([147, 223], [507, 235.5], SIZE_F, [139, 200, 16, 46], [455, 201, 104, 69]);
  assert.deepEqual(arrowGeometry([147, 223], [507, 235.5], SIZE_F, [139, 200, 16, 46], [455, 201, 104, 69], []), plain);
  assert.deepEqual(arrowGeometry([147, 223], [507, 235.5], SIZE_F, [139, 200, 16, 46], [455, 201, 104, 69], [[700, 10, 20, 20]]), plain);
});

test('an arrow between writing side by side bows over what lies between, staying on the board', () => {
  const fromBox: Box = [100, 150, 60, 20];
  const toBox: Box = [400, 150, 60, 20];
  const between: Box = [250, 135, 40, 20];
  const plain = arrowGeometry([130, 160], [430, 160], SIZE, fromBox, toBox);
  assert.ok(shaft(plain.d).some((p) => inBox(p, between)));
  const g = arrowGeometry([130, 160], [430, 160], SIZE, fromBox, toBox, [between]);
  assert.ok(!drawn(g).some((p) => inBox(p, between)));
  assert.deepEqual([g.from, g.tip], [plain.from, plain.tip]); // same ends, a bigger bow
  assert.ok(g.ctrl[1] < plain.ctrl[1]);
  // at the top of the board the bigger bow would leave it: under instead
  const top = arrowGeometry([130, 20], [430, 20], SIZE, [100, 10, 60, 20], [400, 10, 60, 20], [[250, 0, 40, 22]]);
  assert.ok(top.ctrl[1] > 20 && !drawn(top).some((p) => inBox(p, [250, 0, 40, 22]) || p[1] < 0));
  // nothing clear: left as it was
  const walled = arrowGeometry([130, 160], [430, 160], SIZE, fromBox, toBox, [[200, 0, 40, 315]]);
  assert.deepEqual(walled, plain);
});

/** An arrow on test-full at t from piece i to piece j, from and to at their boxes' centres (as the backend gives them). */
const centred = (i: number, j: number, t: number) => {
  const c = (b: Box): [number, number] => [b[0] + b[2] / 2, b[1] + b[3] / 2];
  return hand({
    type: 'arrow', t_start: t, t_end: t, target: [`${i}`, `${j}`], box: boxOfPiece(i),
    from: c(boxOfPiece(i)), to: c(boxOfPiece(j)), path: [],
  });
};
/** Points along an arrow 10x as dense as drawn() (for checks that must not slip between its points). */
const dense = (g: ArrowGeom): [number, number][] => {
  const n = g.d.match(/-?[\d.]+/g)!.map(Number);
  const head = g.head.match(/-?[\d.]+/g)!.map(Number);
  const shaftPts = Array.from({ length: 401 }, (_, k): [number, number] => {
    const s = k / 400;
    return [
      (1 - s) ** 2 * n[0] + 2 * s * (1 - s) * n[2] + s * s * n[4],
      (1 - s) ** 2 * n[1] + 2 * s * (1 - s) * n[3] + s * s * n[5],
    ];
  });
  const seg = (k: number) =>
    Array.from({ length: 21 }, (_, i): [number, number] => [
      head[k] + ((head[2] - head[k]) * i) / 20, head[k + 1] + ((head[3] - head[k + 1]) * i) / 20,
    ]);
  return [...shaftPts, ...seg(0), ...seg(4)];
};
/** How far p lies outside box b (0 or less: on or inside it). */
const gapTo = (p: [number, number], b: Box) => Math.max(b[0] - p[0], p[0] - b[0] - b[2], b[1] - p[1], p[1] - b[1] - b[3]);

test('an arrow whose tip would land on other writing points at a free side of its target instead', () => {
  // test-full, 440 s: from "A = (1+r)^n" (34) down to "= $e." (74), right under "A = (1 + 1/oo?)^oo" (71): pulled
  // out of 74 towards 34, the tip sat on 71 (and the shaft crossed it), whatever the bow
  const down = centred(34, 74, 440);
  const before = boardArrow(down, false).g;
  assert.ok(inBox(before.tip, boxOfPiece(71)));
  const { e, g } = boardArrow(down);
  assert.ok(e.obstacles!.includes(boxOfPiece(71)));
  for (const p of drawn(g)) {
    for (const o of e.obstacles!) assert.ok(!inBox(p, o), `${p} on the writing at ${o}`);
    assert.ok(!inBox(p, boxOfPiece(34)) && !inBox(p, boxOfPiece(74)));
  }
  // into the left side of "= $e.", under 71
  const [x, y, , h] = boxOfPiece(74);
  assert.ok(g.tip[0] > x - 8 && g.tip[0] < x && g.tip[1] > y && g.tip[1] < y + h, `tip at ${g.tip}`);

  // from "x = -1, ..." (17) across to "A = (1 + 1/365)^365" (line L6): pulled out of L6, the tip sat on the corner of
  // "x = -1, log_e(-1) ..." (16) that reaches under L6's left end
  const across = centred(17, 59, 440);
  const before2 = boardArrow(across, false).g;
  assert.ok(inBox(before2.tip, boxOfPiece(16)));
  const r = boardArrow(across);
  assert.deepEqual(r.e.to, lineBox('L6'));
  for (const p of drawn(r.g)) {
    for (const o of r.e.obstacles!) assert.ok(!inBox(p, o), `${p} on the writing at ${o}`);
    assert.ok(!inBox(p, boxOfPiece(17)) && !inBox(p, lineBox('L6')));
  }
  // into L6's left end, above 16
  const [lx, ly] = lineBox('L6');
  assert.ok(r.g.tip[0] > lx - 8 && r.g.tip[0] < lx && r.g.tip[1] > ly && r.g.tip[1] < boxOfPiece(16)[1], `tip at ${r.g.tip}`);
});

test('with no clear way round, an arrow still keeps its tip off other writing', () => {
  // a wall of writing between the two boxes (no route is clear) and a scrap just left of the target, where the tip
  // pulled out of it lands
  const fromBox: Box = [100, 150, 60, 20];
  const toBox: Box = [400, 150, 60, 20];
  const wall: Box = [200, 0, 40, 315];
  const scrap: Box = [380, 140, 18, 40];
  const plain = arrowGeometry([130, 160], [430, 160], SIZE, fromBox, toBox);
  assert.ok(inBox(plain.tip, scrap));
  const g = arrowGeometry([130, 160], [430, 160], SIZE, fromBox, toBox, [wall, scrap]);
  assert.ok(!inBox(g.tip, scrap) && !inBox(g.tip, toBox) && !inBox(g.tip, wall));
  assert.ok(!drawn(g).some((p) => inBox(p, scrap) || inBox(p, toBox) || inBox(p, fromBox)));
  assert.ok(gapTo(g.tip, toBox) > 0 && gapTo(g.tip, toBox) < 8); // still right beside its target
});

/** Whether all of an arrow lies on the board. */
const onBoard = (g: ArrowGeom) => dense(g).every((pt) => pt[0] >= 0 && pt[0] <= SIZE_F[0] && pt[1] >= 0 && pt[1] <= SIZE_F[1]);

test('an arrow sent round other writing keeps its head on the board', () => {
  // test-full, 450 s: from "#5. ..." (20) to "100%" (41), 6 px from the board's top: the way round the writing first
  // put the tip above 41, with a wing off the board
  const a = centred(20, 41, 450);
  assert.ok(onBoard(boardArrow(a, false).g));
  const { g } = boardArrow(a);
  assert.ok(onBoard(g), `tip ${g.tip}`);
});

test('across test-full, no arrow between writing side by side ends on other writing', () => {
  // every fifth pair of pieces on the board at three moments, from and to at their centres: none ends with its tip on
  // other writing, far fewer run over it, and one that was well clear of it is drawn exactly as before
  const u = SIZE_F[0] / 960;
  let n = 0;
  let kept = 0;
  let touched = 0;
  let touchedBefore = 0;
  const onWriting = (g: ArrowGeom, obs: readonly Box[]) => drawn(g).some((pt) => obs.some((o) => inBox(pt, o)));
  for (const t of [300, 440, 470]) {
    const shown = PIECES_F.filter((p) => startOf(p) <= t && !(p.removed_at != null && p.removed_at <= t));
    shown.forEach((p, i) =>
      shown.forEach((q, j) => {
        if (i === j || (i * shown.length + j) % 5) return;
        const a = centred(p.id, q.id, t);
        const e = boardArrowBoxes(a, PIECES_F, LINES_F, SIZE_F, startOf);
        if (stacked(e.from!, e.to!)) return; // round the ends of lines one above the other: as before
        n++;
        const obs = e.obstacles!;
        const ends = arrowEnds(a, e.from, e.to, 8 * u);
        const g = arrowGeometry(...ends, SIZE_F, e.from, e.to, obs);
        const plain = arrowGeometry(...ends, SIZE_F, e.from, e.to);
        assert.ok(!obs.some((o) => inBox(g.tip, o)), `${p.id} -> ${q.id} at ${t}: tip ${g.tip} on the writing`);
        if (onBoard(plain)) assert.ok(onBoard(g), `${p.id} -> ${q.id} at ${t}: sent off the board`);
        if (onWriting(g, obs)) touched++;
        if (onWriting(plain, obs)) touchedBefore++;
        else if (
          dense(plain).every(
            (pt) => pt[0] >= 0 && pt[0] <= SIZE_F[0] && pt[1] >= 0 && pt[1] <= SIZE_F[1] && obs.every((o) => gapTo(pt, o) > 2 * u + 1),
          )
        ) {
          assert.deepEqual(g, plain, `${p.id} -> ${q.id} at ${t}: clear, but redrawn`);
          kept++;
        }
      }),
    );
  }
  assert.ok(n > 1000 && kept > 100, `${n} arrows, ${kept} clear`);
  assert.ok(touched < touchedBefore / 2, `${touched} of ${n} arrows on writing, ${touchedBefore} before`);
});

test('what an arrow on the board keeps off', () => {
  const a = hand({ type: 'arrow', t_start: 436.33, t_end: 438.67, target: ['43', '54'] });
  const ends: Box[] = [lineBox('L2'), lineBox('L5')];
  const obs = boardObstacles(a, PIECES_F, LINES_F, SIZE_F, ends, startOf);
  const has = (b: Box) => obs.includes(b);
  assert.ok(!has(boxOfPiece(43)) && !has(boxOfPiece(54))); // the targets
  assert.ok(!has(lineBox('L2')) && !has(lineBox('L5'))); // their lines
  assert.ok(!has(boxOfPiece(45)) && !has(boxOfPiece(57))); // the rest of those lines
  assert.ok(has(lineBox('L4')) && has(boxOfPiece(53))); // touches L5 by 2 px, still other writing
  assert.ok(!has(boxOfPiece(3)) && !has(boxOfPiece(19))); // regions
  assert.ok(!has(boxOfPiece(39)) && !has(boxOfPiece(76))); // erased before, written after
  // a line still being written is not in the way as a whole, only its pieces shown so far
  const writing = boardObstacles({ ...a, t_start: 255, t_end: 256 }, PIECES_F, LINES_F, SIZE_F, ends, startOf);
  assert.ok(!writing.includes(lineBox('L4')) && writing.includes(boxOfPiece(53)) && !writing.includes(boxOfPiece(56)));
});

test('a backend "line" covering a block of the board is not typeset as one line', () => {
  const line = (box: Box): BoardLine => ({ id: 'L', pieces: [1, 2], box, t_complete: 10 });
  // gest-test: 11 pieces chained through a merged block, read as "= $2.71828..."
  assert.equal(isWrittenLine(line([221, 6, 573, 303]), 800), false);
  // test-full's real lines, the tallest a fraction over two rows (44-48 px of an 800-wide board)
  for (const b of [[24, 11, 69, 48], [557, 61, 94, 28], [549, 135, 116, 44], [570, 241, 201, 32]] as Box[])
    assert.equal(isWrittenLine(line(b), 800), true);
  // scales with the board: 90 px is a line on a 1920-wide board, not on a 640-wide one
  assert.equal(isWrittenLine(line([0, 0, 300, 90]), 1920), true);
  assert.equal(isWrittenLine(line([0, 0, 300, 90]), 640), false);
  assert.equal(isWrittenLine({ ...line([0, 0, 10, 10]), box: [1, 2] as unknown as Box }, 800), false);
});
