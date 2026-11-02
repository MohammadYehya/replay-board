// Run with `npm test`. The slides export: checking the deck the server sends, the typeset list of a slide (drawings
// named in plain text), present-mode keys, and the page's toggles kept per lecture.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  clampIndex, cleanDeck, cleanImagePath, DEFAULT_PREFS, drawingName, loadPrefs, playerAt, pointsBeside, POINTS_MAX,
  pptxFileName, prefsKey, presentKey, rangeLabel, savePrefs, sentence, typesetList,
} from '../src/slides/deck.ts';
import type { DeckItem } from '../src/types.ts';

const item = (over: Partial<DeckItem> = {}): DeckItem => ({ ref: '#1', kind: 'equation', text: 'x = 1', latex: 'x = 1', t: 5, ...over });

const serverDeck = () => ({
  type: 'board',
  title: 'compound interest',
  version: 'abc123',
  points_ready: true,
  pages: [
    {
      id: 'p1',
      t0: 0,
      t1: 95.4,
      chapter: 'Simple interest',
      title: 'Simple interest grows by the same amount',
      image: 'export/page_001.png',
      image_size: [1600, 900],
      items: [
        { ref: '#L3', kind: 'equation', text: 'A = P(1 + rt)', latex: 'A = P(1 + rt)', t: 12.3 },
        { ref: '#7', kind: 'graph', text: 'straight line of A against t', latex: '', t: 40, drawing: true },
      ],
      said: 'Today we look at interest...',
      points: ['Interest is paid on the principal only.', '  A grows   linearly  ', ''],
    },
  ],
});

test('a server deck comes through as it is', () => {
  const d = cleanDeck(serverDeck());
  assert.equal(d.type, 'board');
  assert.equal(d.title, 'compound interest');
  assert.equal(d.version, 'abc123');
  assert.equal(d.points_ready, true);
  assert.equal(d.pages.length, 1);
  const p = d.pages[0];
  assert.equal(p.id, 'p1');
  assert.equal(p.t1, 95.4);
  assert.equal(p.image, 'export/page_001.png');
  assert.deepEqual(p.image_size, [1600, 900]);
  assert.equal(p.items.length, 2);
  assert.equal(p.items[1].drawing, true);
  assert.equal(p.items[0].drawing, undefined);
  // blank points dropped, spaces tidied
  assert.deepEqual(p.points, ['Interest is paid on the principal only.', 'A grows linearly']);
});

test('not a deck: one plain line', () => {
  for (const bad of [null, 'x', 3, {}, { pages: 'no' }, []]) {
    assert.throws(() => cleanDeck(bad), (e: Error) => /slides came back empty/.test(e.message) && !e.message.includes('\n'));
  }
});

test('missing and odd fields get safe defaults', () => {
  const d = cleanDeck({
    type: 'weird',
    pages: [
      { t0: 30, t1: 10, chapter: 'Limits', items: [null, 5, { kind: 'Diagram', t: -1 }], points: 'not a list' },
      null,
      { image: '../../secret.txt', image_size: [0, 10], title: '  ', points: Array(10).fill('a point') },
      { image: 'https://evil.example/x.png', items: 'nope', said: 42 },
    ],
  });
  assert.equal(d.type, 'board');
  assert.equal(d.title, 'Lecture');
  assert.equal(d.version, '');
  assert.equal(d.points_ready, false);
  assert.equal(d.pages.length, 3);
  const [a, b, c] = d.pages;
  // t0 and t1 the right way round; the chapter is the title until Muse Spark writes one
  assert.equal(a.t0, 10);
  assert.equal(a.t1, 30);
  assert.equal(a.title, 'Limits');
  assert.equal(a.id, 'p1');
  assert.equal(a.points, null);
  // junk items dropped; a drawing kind is a drawing even without the flag; a bad time falls back to the page start
  assert.equal(a.items.length, 1);
  assert.deepEqual(a.items[0], { ref: '', kind: 'diagram', text: '', latex: '', t: 10, drawing: true });
  assert.equal(b.image, '');
  assert.equal(b.image_size, null);
  assert.equal(b.title, 'Slide 3');
  assert.equal(b.points?.length, POINTS_MAX);
  assert.equal(c.image, '');
  assert.deepEqual(c.items, []);
  assert.equal(c.said, '');
});

test('image paths stay inside the job folder', () => {
  assert.equal(cleanImagePath('export/page_001.png'), 'export/page_001.png');
  assert.equal(cleanImagePath('export\\page_002.png'), 'export/page_002.png');
  assert.equal(cleanImagePath('slides/003.png'), 'slides/003.png');
  for (const bad of ['', '/etc/passwd', '../x.png', 'export/../../x', 'C:/x.png', 'http://x/y.png', 'a//b', 7, null]) {
    assert.equal(cleanImagePath(bad), '', String(bad));
  }
});

test('typeset list: writing typeset, drawings named, reading order kept', () => {
  const list = typesetList({
    items: [
      item({ ref: '#L1', latex: '\\frac{a}{b}', text: 'a over b', t: 3 }),
      item({ ref: '#2', kind: 'graph', text: 'parabola y = x²', latex: '', drawing: true, t: 8 }),
      item({ ref: '#3', kind: 'text', text: 'Note: always check', latex: '', t: 9 }),
      item({ ref: '#4', kind: 'diagram', text: '', latex: '', drawing: true, t: 11 }),
      item({ ref: '#5', kind: 'equation', text: 'unreadable', latex: '', t: 12 }),
      item({ ref: '#6', kind: 'arrow', text: 'arrow', latex: '', drawing: true, t: 13 }),
    ],
  });
  assert.deepEqual(
    list.map((e) => e.ref),
    ['#L1', '#2', '#3', '#4', '#5', '#6'],
  );
  assert.deepEqual(list[0], { ref: '#L1', t: 3, tex: '\\frac{a}{b}', text: 'a over b', drawing: false });
  assert.deepEqual(list[1], { ref: '#2', t: 8, tex: null, text: 'Graph: parabola y = x²', drawing: true });
  assert.deepEqual(list[2], { ref: '#3', t: 9, tex: null, text: 'Note: always check', drawing: false });
  assert.equal(list[3].text, 'Diagram');
  assert.equal(list[4].text, 'Equation');
  assert.equal(list[5].text, 'Arrow');
});

test('drawing names', () => {
  assert.equal(drawingName(item({ kind: 'chart', text: 'bar chart of sales' })), 'Chart: bar chart of sales');
  assert.equal(drawingName(item({ kind: '', text: '' })), 'Drawing');
  assert.equal(drawingName(item({ kind: 'figure', text: 'unreadable' })), 'Figure');
});

test('time range, player link and layout', () => {
  assert.equal(rangeLabel(0, 95.4), '0:00–1:35');
  assert.equal(rangeLabel(3600, 3725), '1:00:00–1:02:05');
  assert.equal(playerAt('test-full', 83.9), '#/job/test-full?t=83');
  assert.equal(playerAt('a b', -3), '#/job/a%20b?t=0');
  assert.equal(pointsBeside([1600, 900]), false);
  assert.equal(pointsBeside([900, 900]), true);
  assert.equal(pointsBeside(null), false);
});

test('present mode: next, previous, ends, leaving', () => {
  const n = 5;
  assert.equal(presentKey('ArrowRight', false, 0, n), 1);
  assert.equal(presentKey('ArrowDown', false, 1, n), 2);
  assert.equal(presentKey('PageDown', false, 2, n), 3);
  assert.equal(presentKey(' ', false, 3, n), 4);
  assert.equal(presentKey(' ', false, 4, n), 4); // stays on the last slide
  assert.equal(presentKey(' ', true, 4, n), 3); // Shift+Space goes back
  assert.equal(presentKey('ArrowLeft', false, 3, n), 2);
  assert.equal(presentKey('PageUp', false, 0, n), 0); // stays on the first
  assert.equal(presentKey('Backspace', false, 2, n), 1);
  assert.equal(presentKey('Home', false, 3, n), 0);
  assert.equal(presentKey('End', false, 0, n), 4);
  assert.equal(presentKey('Escape', false, 2, n), 'exit');
  assert.equal(presentKey('a', false, 2, n), null);
  assert.equal(presentKey('Tab', false, 2, n), null);
  assert.equal(presentKey('End', false, 0, 0), 0);
  assert.equal(clampIndex(9, 3), 2);
  assert.equal(clampIndex(-2, 3), 0);
});

class MemoryStorage {
  map = new Map<string, string>();
  getItem(k: string) {
    return this.map.has(k) ? this.map.get(k)! : null;
  }
  setItem(k: string, v: string) {
    this.map.set(k, v);
  }
}

test('toggles are kept per lecture, and junk or blocked storage gives the defaults', () => {
  const store = new MemoryStorage();
  assert.deepEqual(loadPrefs(store, 'a'), DEFAULT_PREFS);
  savePrefs(store, 'a', { view: 'typeset', said: true });
  assert.deepEqual(loadPrefs(store, 'a'), { view: 'typeset', said: true });
  assert.deepEqual(loadPrefs(store, 'b'), DEFAULT_PREFS);
  store.setItem(prefsKey('c'), '{not json');
  assert.deepEqual(loadPrefs(store, 'c'), DEFAULT_PREFS);
  store.setItem(prefsKey('d'), JSON.stringify({ view: 'fancy', said: 'yes' }));
  assert.deepEqual(loadPrefs(store, 'd'), DEFAULT_PREFS);
  assert.deepEqual(loadPrefs(null, 'a'), DEFAULT_PREFS);
  const blocked = {
    getItem(): string | null {
      throw new Error('SecurityError');
    },
    setItem() {
      throw new Error('QuotaExceededError');
    },
  };
  assert.deepEqual(loadPrefs(blocked, 'a'), DEFAULT_PREFS);
  assert.doesNotThrow(() => savePrefs(blocked, 'a', { view: 'image', said: false }));
});

test('two pages with the same id get distinct ids (they key the slide cards)', () => {
  const d = cleanDeck({ pages: [{ id: 'p1' }, { id: 'p1' }, { id: 'p1-2' }, { id: 'p1' }, {}] });
  const ids = d.pages.map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.deepEqual(ids, ['p1', 'p1-2', 'p1-2-2', 'p1-3', 'p5']);
});

test('the real deck shape from app/export.py (test-full) passes through unchanged', () => {
  const page = {
    id: 'p2', t0: 110.4, t1: 176.3, chapter: 'One dollar at 100% interest', title: 'One dollar at 100% interest',
    image: 'export/page_002.png', image_size: [723, 405],
    items: [
      { ref: '#41', kind: 'text', text: '100%.', latex: '', t: 154.36, drawing: false },
      { ref: '#73', kind: 'scribble', text: 'blurry blue mark', latex: '', t: 422.35, drawing: true },
    ],
    said: 'So if you have one dollar...', points: null,
  };
  const d = cleanDeck({ type: 'board', title: 'Logarithms and exponentials', version: 'd88a5f09ffca75ac', points_ready: false, pages: [page] });
  const p = d.pages[0];
  assert.deepEqual(
    { ...p, items: p.items.map((it) => ({ ...it, drawing: !!it.drawing })) },
    page,
  );
});

test('error lines read as sentences, on one line', () => {
  assert.equal(sentence('this lecture is not processed yet, so there are no slides yet'), 'This lecture is not processed yet, so there are no slides yet');
  assert.equal(sentence('  Muse Spark could not\nwrite   the key points '), 'Muse Spark could not write the key points');
  assert.equal(sentence(''), '');
});

test("PowerPoint file name: the server's, else the lecture title; never a path", () => {
  const both = `attachment; filename="Logarithms and exponentials - slides.pptx"; filename*=UTF-8''Logarithmes%20et%20exponentielles%20%C3%A9t%C3%A9%20-%20slides.pptx`;
  assert.equal(pptxFileName(both, 'x'), 'Logarithmes et exponentielles été - slides.pptx');
  assert.equal(pptxFileName('attachment; filename="Limits - slides.pptx"', 'x'), 'Limits - slides.pptx');
  assert.equal(pptxFileName('attachment; filename=plain.pptx', 'x'), 'plain.pptx');
  // a broken UTF-8 escape falls back to the ASCII name
  assert.equal(pptxFileName(`attachment; filename="ascii - slides.pptx"; filename*=UTF-8''%E0%A4`, 'x'), 'ascii - slides.pptx');
  // no header, or nothing usable in it: the lecture title
  assert.equal(pptxFileName(null, 'Compound interest'), 'Compound interest - slides.pptx');
  assert.equal(pptxFileName('attachment', 'a/b:c?'), 'a b c - slides.pptx');
  assert.equal(pptxFileName(null, '  '), 'lecture - slides.pptx');
  // path parts and odd characters never reach the saved name; the extension is always there
  assert.equal(pptxFileName('attachment; filename="../../evil"', 'x'), 'evil.pptx');
  assert.equal(pptxFileName('attachment; filename=C:\\Users\\me\\evil.pptx', 'x'), 'C Users me evil.pptx');
  assert.equal(pptxFileName('attachment; filename="say \\"hi\\".pptx"', 'x'), 'say hi .pptx');
  assert.equal(pptxFileName(`attachment; filename*=UTF-8''..%2F..%2Fx.pptx`, 'x'), 'x.pptx');
});
