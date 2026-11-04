// Half-erased writing (D3): a piece the backend marks removed (removed_at: half or more of its shown pixels erased)
// keeps drawing what the teacher left of it while less than 90% of its drawn pixels are ever erased; one erased 90% or
// more, or one without a gone map, fades whole at removed_at as before. The canvas, hover, clicks and isVisible all go
// by the same rules (timing.ts), and the erased share is worked out once from the gone map (sprites.ts goneOf).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  alphaAt, handwritingAlpha, hitBy, inkAlphaAt, keepsRemnant, progressAt, REMNANT_MAX, schedule, STEPS, visibleAt,
} from '../src/player/timing.ts';
import { FrameCache, goneCount, goneOf, type Sprite } from '../src/player/sprites.ts';
import type { Piece } from '../src/types.ts';

// test.mp4 #39 (test-full job): drawn from 120 s, removed_at 141 s; 203 of its 278 drawn pixels are ever erased (73%),
// a "$1" the teacher left is still on the board when he circles it at about 300 s.
const piece: Piece = {
  id: 39, box: [515, 141, 115, 59], image: 'pieces/039.png', gone: 'pieces/039_gone.png',
  t_seen: 120, t_shown: 122, revealed: false, removed_at: 141, pre_existing: false,
  read: { kind: 'text', text: '$1', latex: '' }, spoken: [], redraw_at: null,
};
const R = 141;
const s = schedule(piece);
const times = [119, 121, 140.9, R, R + 0.1, R + 0.2, R + 0.39, R + 0.4, R + 1, 300, 1e6];

test('the remnant rule: a gone map with less than 90% of the drawn pixels ever erased', () => {
  assert.equal(REMNANT_MAX, 0.9);
  assert.equal(keepsRemnant(0.73), true); // test.mp4 #39
  assert.equal(keepsRemnant(0.5), true);
  assert.equal(keepsRemnant(0), true); // a gone map that erases nothing of the drawn pixels: nothing leaves
  assert.equal(keepsRemnant(0.8999), true);
  assert.equal(keepsRemnant(0.9), false); // the boundary: 90% or more fades whole
  assert.equal(keepsRemnant(0.95), false);
  assert.equal(keepsRemnant(1), false); // test.mp4 #38, all of it erased
  assert.equal(keepsRemnant(null), false); // no gone map
  assert.equal(keepsRemnant(undefined), false); // sprite not loaded / old sprite
  assert.equal(keepsRemnant(NaN), false);
});

test('a remnant keeps its handwriting at full opacity after removed_at', () => {
  for (const t of times) assert.equal(inkAlphaAt(s, t, true), 1, `t=${t}`);
  // it stays on the board (visible, so hover is not dropped) long after removed_at
  assert.equal(visibleAt(s, R + 1, true), true);
  assert.equal(visibleAt(s, 300, true), true);
  // and is drawn like any other handwriting
  assert.equal(handwritingAlpha(inkAlphaAt(s, 300, true), alphaAt(s, 300), false, 0, 0), 1);
});

test('a piece that does not keep a remnant fades whole at removed_at, as before', () => {
  for (const t of times) assert.equal(inkAlphaAt(s, t, false), alphaAt(s, t), `t=${t}`);
  assert.equal(inkAlphaAt(s, R - 0.1, false), 1);
  assert.ok(Math.abs(inkAlphaAt(s, R + 0.2, false) - 0.5) < 1e-9);
  assert.equal(inkAlphaAt(s, R + 0.4, false), 0);
  assert.equal(visibleAt(s, R + 0.4, false), false);
  assert.equal(handwritingAlpha(inkAlphaAt(s, R + 0.4, false), alphaAt(s, R + 0.4), false, 0, 0), 0);
});

test('the remnant decision follows the erased share at the boundary', () => {
  const at = (erased: number | null) => inkAlphaAt(s, R + 1, keepsRemnant(erased));
  assert.equal(at(0.89), 1);
  assert.equal(at(0.9), 0);
  assert.equal(at(null), 0);
});

test('a piece that is never removed is drawn the same either way', () => {
  const kept = schedule({ ...piece, removed_at: null });
  for (const t of times) {
    assert.equal(inkAlphaAt(kept, t, true), 1);
    assert.equal(inkAlphaAt(kept, t, false), 1);
  }
});

test('visibleAt without a remnant is the old isVisible (drawn and not faded); before it is drawn nothing is visible', () => {
  for (let t = 115; t < 145; t += 0.05) {
    const old = progressAt(s, t) > 0 && alphaAt(s, t) >= 0.05;
    assert.equal(visibleAt(s, t, false), old, `t=${t}`);
    if (progressAt(s, t) <= 0) assert.equal(visibleAt(s, t, true), false, `t=${t}`);
  }
});

test('a streamed piece that gains removed_at later is rescheduled, nothing kept from before', () => {
  const before = schedule({ ...piece, removed_at: null });
  assert.equal(inkAlphaAt(before, 200, false), 1);
  const after = schedule(piece); // the next partial replay: same piece, now removed at 141 s
  assert.equal(inkAlphaAt(after, 200, false), 0);
  assert.equal(inkAlphaAt(after, 200, true), 1);
});

test('typeset mode: the reading fades at removed_at and the remnant comes in as handwriting as it goes', () => {
  for (const t of times) {
    const whole = alphaAt(s, t);
    const ink = inkAlphaAt(s, t, true);
    const hand = handwritingAlpha(ink, whole, true, 0, 0);
    // the reading (whole) and the handwriting never show the piece twice over
    assert.ok(Math.abs(hand + whole - 1) < 1e-9, `t=${t}`);
  }
  assert.equal(handwritingAlpha(1, alphaAt(s, R - 1), true, 0, 0), 0); // the reading stands in for it until then
  assert.ok(Math.abs(handwritingAlpha(1, alphaAt(s, R + 0.2), true, 0, 0) - 0.5) < 1e-9);
  assert.equal(handwritingAlpha(1, alphaAt(s, R + 1), true, 0, 0), 1);
  // a piece shown typeset that keeps no remnant draws no handwriting at all, as before
  for (const t of times) assert.equal(handwritingAlpha(inkAlphaAt(s, t, false), alphaAt(s, t), true, 0, 0), 0, `t=${t}`);
  // a bigger reading that repeats it still hides its remnant; its finished line does too while it shows
  assert.equal(handwritingAlpha(1, 0, true, 1, 0), 0);
  assert.equal(handwritingAlpha(1, 0, false, 0, 0.99), 0);
  assert.ok(Math.abs(handwritingAlpha(1, 0, false, 0, 0.5) - 0.5) < 1e-9);
});

test('hover and clicks find exactly what is drawn: canvas, hit test and isVisible agree', () => {
  const cases = [];
  for (const remnant of [true, false])
    for (const reading of [true, false])
      for (const hide of [0, 0.5, 0.97])
        for (const lineHide of [0, 0.6, 0.99]) cases.push({ remnant, reading, hide, lineHide });
  for (let t = 115; t < 146; t += 0.01) {
    for (const o of cases) {
      const ink = inkAlphaAt(s, t, o.remnant);
      const whole = alphaAt(s, t);
      const drawn = progressAt(s, t) > 0;
      const hand = drawn ? handwritingAlpha(ink, whole, o.reading, o.hide, o.lineHide) : 0;
      const shown = drawn && o.reading ? whole * (1 - o.hide) * (1 - o.lineHide) : 0; // its typeset reading
      const by = hitBy(s, t, o);
      const msg = `t=${t.toFixed(2)} ${JSON.stringify(o)} hand=${hand} by=${by}`;
      // found only while on the board (hover is not dropped on something that can be found)
      if (by) assert.equal(visibleAt(s, t, o.remnant), true, msg);
      // found by its ink only when its handwriting is drawn; by its box only when a reading stands there
      if (by === 'ink') assert.ok(hand > 0, msg);
      if (by === 'box') assert.ok(shown > 0 || o.lineHide > 0.5, msg);
      // anything drawn plainly can be found (during the 0.4 s cross-fade in typeset mode, by the reading's box)
      if (hand >= 0.05 || (shown >= 0.05 && o.hide <= 0.95)) assert.notEqual(by, null, msg);
      if (hand >= 0.05 && !o.reading && o.lineHide <= 0.5) assert.equal(by, 'ink', msg);
      if (shown >= 0.05 && o.hide <= 0.95) assert.equal(by, 'box', msg);
    }
  }
  // the remnant, long after removed_at: found by its ink in both modes (its reading has gone); a piece that fades
  // whole is not found at all
  assert.equal(hitBy(s, 300, { remnant: true, reading: false, hide: 0, lineHide: 0 }), 'ink');
  assert.equal(hitBy(s, 300, { remnant: true, reading: true, hide: 0, lineHide: 0 }), 'ink');
  assert.equal(hitBy(s, 300, { remnant: false, reading: false, hide: 0, lineHide: 0 }), null);
  assert.equal(hitBy(s, 300, { remnant: false, reading: true, hide: 0, lineHide: 0 }), null);
  // before removed_at, a typeset reading is found by its box, as before
  assert.equal(hitBy(s, 130, { remnant: true, reading: true, hide: 0, lineHide: 0 }), 'box');
  assert.equal(hitBy(s, 130, { remnant: false, reading: false, hide: 0, lineHide: 0 }), 'ink');
});

/** A gone map image: RGB code per pixel (0 = never, else 1 + tenths of a second), alpha 255. */
function goneMap(codes: number[], w: number, h: number) {
  const data = new Uint8ClampedArray(w * h * 4);
  codes.forEach((c, i) => {
    data[i * 4] = (c >> 16) & 255;
    data[i * 4 + 1] = (c >> 8) & 255;
    data[i * 4 + 2] = c & 255;
    data[i * 4 + 3] = 255;
  });
  return { width: w, height: h, data };
}
const code = (sec: number) => Math.round(sec * 10) + 1;

test('the erased share counts the drawn pixels (order > 0) whose gone time is finite', () => {
  // 10 drawn pixels and 2 blank ones (order 0); 7 drawn pixels erased at 140.7 or 141 s, one more at 177.7 s
  const order = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0]);
  const codes = [code(140.7), code(141), code(141), code(140.7), code(141), code(141), code(141), code(177.7), 0, 0,
    code(150), code(150)]; // a code under a blank pixel does not count
  const g = goneOf(order, goneMap(codes, 4, 3), 4, 3);
  assert.ok(g.goneAt);
  assert.equal(g.erased, 0.8);
  assert.equal(keepsRemnant(g.erased), true);
  assert.deepEqual(g.goneTimes, [140.7, 141, 177.7]);
  assert.equal(g.goneAt![8], Infinity); // what the teacher left
  assert.equal(g.goneAt![10], Infinity); // blank pixel: never drawn, never erased
  assert.ok(Math.abs(g.goneAt![7] - 177.7) < 1e-4);
});

test('the erased share at the boundary: 9 of 10 drawn pixels erased fades whole', () => {
  const order = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0].map(() => code(141));
  codes[9] = 0;
  const g = goneOf(order, goneMap(codes, 10, 1), 10, 1);
  assert.equal(g.erased, 0.9);
  assert.equal(keepsRemnant(g.erased), false);
  const g2 = goneOf(order, goneMap(codes.map((c, i) => (i >= 8 ? 0 : c)), 10, 1), 10, 1);
  assert.equal(g2.erased, 0.8);
  assert.equal(keepsRemnant(g2.erased), true);
});

test('no gone map, one of the wrong size, or one that erases nothing: no erase times and no share (fades whole)', () => {
  const order = new Uint8Array([1, 2, 3, 4]);
  for (const gone of [null, goneMap([code(1), code(1), code(1)], 3, 1), goneMap([0, 0, 0, 0], 4, 1)]) {
    const g = goneOf(order, gone, 4, 1);
    assert.equal(g.goneAt, null);
    assert.deepEqual(g.goneTimes, []);
    assert.equal(g.erased, null);
    assert.equal(keepsRemnant(g.erased), false);
  }
});

test('goneCount still counts the partial erases that have happened (the frame leaves those pixels out)', () => {
  const order = new Uint8Array([1, 2, 3, 4]);
  const g = goneOf(order, goneMap([code(140.7), code(141), 0, 0], 4, 1), 4, 1);
  const sprite = { goneAt: g.goneAt, goneTimes: g.goneTimes } as Sprite;
  assert.equal(goneCount(sprite, 140), 0);
  assert.equal(goneCount(sprite, 140.7), 1);
  assert.equal(goneCount(sprite, 300), 2);
});

test('pixels erased at a time that rounds up in float32 leave the drawn frame (141.3 s)', () => {
  // goneAt is a Float32Array: 141.3 is 141.30000305 there. The frame's cut must compare in float32 too, or a pixel
  // erased at the piece's last erase time stays drawn for good, which a remnant (drawn after removed_at) shows.
  const g = globalThis as Record<string, unknown>;
  const saved = { ImageData: g.ImageData, document: g.document };
  g.ImageData = class {
    width: number; height: number; data: Uint8ClampedArray;
    constructor(w: number, h: number) { this.width = w; this.height = h; this.data = new Uint8ClampedArray(w * h * 4); }
  };
  g.document = { createElement: () => ({ getContext() { const c = this as { img?: unknown }; return { putImageData(i: unknown) { c.img = i; } }; } }) };
  try {
    const times = [140.7, 140.7, 141.3, 0]; // two erased at 140.7, one at 141.3 (its last erase), one never
    const n = times.length;
    const order = new Uint8Array(n).fill(1);
    const data = new Uint8ClampedArray(n * 4);
    times.forEach((t, i) => {
      const c = t ? Math.round(t * 10) + 1 : 0;
      data.set([(c >> 16) & 255, (c >> 8) & 255, c & 255, 255], i * 4);
    });
    const sp = {
      id: 1, w: n, h: 1, full: {} as HTMLCanvasElement, rgba: new Uint8ClampedArray(n * 4).fill(255),
      sorted: Uint32Array.from({ length: n }, (_, i) => i), upto: new Uint32Array(256).fill(n), ink: '', cy: 0.5,
      ...goneOf(order, { width: n, height: 1, data } as ImageData, n, 1),
    } as Sprite;
    sp.upto[0] = 0;
    const drawn = (t: number) => {
      const c = new FrameCache().frame(sp, STEPS, goneCount(sp, t)) as unknown as { img: ImageData };
      let k = 0;
      for (let i = 0; i < n; i++) if (c.img.data[i * 4 + 3]) k++;
      return k;
    };
    assert.equal(drawn(141.0), 2);
    assert.equal(drawn(141.3), 1);
    assert.equal(drawn(400), 1);
  } finally {
    g.ImageData = saved.ImageData;
    g.document = saved.document;
  }
});
