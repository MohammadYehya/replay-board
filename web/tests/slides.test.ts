// Run with `npm test`. The lecture deck (format 2): checking what the server sends (junk, unsafe paths, limits,
// unknown blocks), inline maths, the content layout, present-mode keys, getting the deck (one write, polling, try
// again) and the PowerPoint's file name. All lecture content here is made up.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  agendaLayout, clampIndex, cleanBlock, cleanDeck, cleanFigurePath, cleanStatus, contentLayout, coverFacts, cutInline,
  deckSections, eyebrow, formulaGrid, frameBlocks, inlinePlain, LIMITS, mathShown, plainMath, playerAt, pptxFileName,
  presentKey, progressShare, rangeLabel, sectionFacts, sentence, shownLength, sideTable, slideBlocks, splitInline,
  stepLine,
} from '../src/slides/deck.ts';
import { fitAll, FitQueue } from '../src/slides/fit.ts';
import { joinCall, MISS_LIMIT, NONE_LIMIT, NOT_STARTED, POLL_MS, runFlow, sleep, type FlowCalls } from '../src/slides/flow.ts';
import type { DeckSlide, SlidesStatus } from '../src/types.ts';

// ---------- a made-up deck with every slide kind and block type ----------

const serverDeck = () => ({
  format: 2,
  version: 'v7f3a',
  kind: 'board',
  title: 'Growth that compounds',
  subtitle: 'From simple interest to the number e, one step at a time',
  lecture: 'Week 3 - interest',
  duration: 478.2,
  clip: [12, 490.2],
  slides: [
    { id: 's1', kind: 'title', title: 'Growth that compounds', section: '', t0: null, t1: null, t: null, blocks: [], notes: 'Welcome.', said: '' },
    {
      id: 's2', kind: 'agenda', title: 'Agenda', section: '', t0: null, t1: null, t: null,
      blocks: [{ type: 'bullets', items: [{ text: 'Simple interest', level: 0 }, { text: 'Compounding', level: 0 }] }],
      notes: '', said: '',
    },
    { id: 's3', kind: 'section', title: 'Simple interest', section: 'Simple interest', t0: 12, t1: 140, t: 12, blocks: [], notes: '', said: '' },
    {
      id: 's4', kind: 'content', title: 'Interest on the principal only', section: 'Simple interest', t0: 12, t1: 70.5, t: 20.4,
      blocks: [
        { type: 'bullets', items: [{ text: 'The rate $r$ applies to $P$ each year', level: 0 }, { text: 'no interest on interest', level: 1 }] },
        { type: 'equation', latex: 'A = P(1 + rt)', label: 'Simple interest', refs: ['#L3', '#12'] },
      ],
      notes: 'Start from the definition.', said: 'so today we start with the simplest case...',
    },
    {
      id: 's5', kind: 'content', title: 'A worked example', section: 'Simple interest', t0: 70.5, t1: 110, t: 75,
      blocks: [
        { type: 'steps', label: 'Worked example', items: [{ latex: 'A = 100(1 + 0.05 \\cdot 3)', text: 'put in the numbers' }, { latex: 'A = 115', text: '' }] },
        { type: 'figure', refs: ['#20'], caption: 'Growth is a straight line', image: 'export/fig_001.png', size: [640, 400] },
      ],
      notes: '', said: '',
    },
    {
      id: 's6', kind: 'content', title: 'Comparing the two', section: 'Simple interest', t0: 110, t1: 140, t: 112,
      blocks: [
        { type: 'table', header: ['Year', 'Simple', 'Compound'], rows: [['1', '105', '105'], ['2', '110', '110.25']], caption: 'At $r = 5\\%$' },
        { type: 'callout', style: 'key', title: 'The gap grows', text: 'Compounding pays interest on $A$, not $P$', latex: '' },
      ],
      notes: '', said: '',
    },
    {
      id: 's7', kind: 'content', title: 'Compounding n times a year', section: 'Compounding', t0: 140, t1: 300, t: 150,
      blocks: [{ type: 'callout', style: 'definition', title: 'Compound interest', text: '', latex: 'A = P\\left(1 + \\tfrac{r}{n}\\right)^{nt}' }],
      notes: '', said: '',
    },
    {
      id: 's8', kind: 'summary', title: 'Key takeaways', section: '', t0: 12, t1: 490.2, t: null,
      blocks: [{ type: 'bullets', items: [{ text: 'Simple interest grows linearly', level: 0 }, { text: 'Compound interest grows exponentially', level: 0 }, { text: 'As $n \\to \\infty$ we reach $e$', level: 0 }] }],
      notes: '', said: '',
    },
    {
      id: 's9', kind: 'formulas', title: 'Formula sheet', section: '', t0: null, t1: null, t: null,
      blocks: [
        { type: 'equation', latex: 'A = P(1 + rt)', label: 'Simple interest', refs: [] },
        { type: 'equation', latex: 'A = P(1 + r/n)^{nt}', label: 'Compound interest', refs: [] },
        { type: 'equation', latex: 'A = Pe^{rt}', label: 'Continuous', refs: [] },
      ],
      notes: '', said: '',
    },
  ],
  degraded: [],
});

const status = (state: SlidesStatus['state'], over: Partial<Record<string, unknown>> = {}) => ({
  state, step: '', done: 0, total: 0, error: null, deck: state === 'ready' ? serverDeck() : null, ...over,
});

test('a server deck comes through as it is', () => {
  const raw = serverDeck();
  const d = cleanDeck(raw);
  assert.equal(d.format, 2);
  assert.equal(d.version, 'v7f3a');
  assert.equal(d.kind, 'board');
  assert.equal(d.title, 'Growth that compounds');
  assert.equal(d.lecture, 'Week 3 - interest');
  assert.equal(d.duration, 478.2);
  assert.deepEqual(d.clip, [12, 490.2]);
  assert.deepEqual(d.degraded, []);
  // every slide, every block, unchanged
  assert.deepEqual(d.slides, raw.slides);
});

test('the slides state: ready, writing with the slides so far, failed, none', () => {
  const ready = cleanStatus(status('ready'));
  assert.equal(ready.state, 'ready');
  assert.equal(ready.deck?.slides.length, 9);

  const partial = serverDeck();
  partial.slides = partial.slides.slice(0, 3);
  const writing = cleanStatus(status('writing', { step: 'Writing part 2 of 5', done: 2, total: 5, deck: partial }));
  assert.equal(writing.deck?.slides.length, 3);
  assert.equal(writing.done, 2);
  assert.equal(writing.total, 5);
  // a damaged partial deck is dropped while writing, not an error
  assert.equal(cleanStatus(status('writing', { deck: { slides: 'no' } })).deck, null);
  // done never passes total; junk counts are 0
  assert.equal(cleanStatus(status('writing', { done: 9, total: 4 })).done, 4);
  assert.equal(cleanStatus(status('writing', { done: -1, total: 'x' })).done, 0);

  const failed = cleanStatus(status('failed', { error: 'muse Spark answered 502\n twice' }));
  assert.equal(failed.error, 'Muse Spark answered 502 twice');
  assert.equal(cleanStatus(status('failed')).error, 'Muse Spark could not write the slides.');
  assert.equal(cleanStatus(status('none')).error, null);
});

test('not a deck or not a state: one plain line', () => {
  for (const bad of [null, 'x', 3, [], {}, { state: 'done' }, { state: 'ready', deck: null }, { state: 'ready', deck: { slides: 3 } }]) {
    assert.throws(
      () => cleanStatus(bad),
      (e: Error) => /slides came back damaged/.test(e.message) && !e.message.includes('\n'),
      JSON.stringify(bad),
    );
  }
  for (const bad of [null, 'x', {}, { slides: 'no' }, []]) assert.throws(() => cleanDeck(bad), /slides came back damaged/);
  // an older server still sends the page-image deck
  const old = { type: 'board', title: 'x', version: 'v', points_ready: false, pages: [] };
  assert.throws(() => cleanStatus(old), /old slides: restart `python -m app.server`/);
  assert.throws(() => cleanDeck(old), /old slides/);
});

test('missing and odd fields get safe defaults', () => {
  const d = cleanDeck({
    kind: 'weird',
    duration: -3,
    clip: [50, 'x'],
    slides: [
      { kind: 'mystery', t0: 30, t1: 10, section: 'Limits', blocks: [{ type: 'bullets', items: ['a plain string item', 7, { text: '  ' }] }] },
      null,
      7,
      { kind: 'agenda' },
      { kind: 'summary', t1: 40 },
      { kind: 'formulas', title: '   ' },
      { kind: 'section', t0: -5, t: 'later' },
    ],
    degraded: ['  the agenda was made by code\n', 4, ''],
  });
  assert.equal(d.kind, 'board');
  assert.equal(d.title, 'Lecture');
  assert.equal(d.lecture, 'Lecture');
  assert.equal(d.subtitle, '');
  assert.equal(d.version, '');
  assert.deepEqual(d.degraded, ['The agenda was made by code']);
  assert.equal(d.slides.length, 5);
  const [a, agenda, summary, formulas, section] = d.slides;
  // an unknown kind is a content slide; t0 and t1 the right way round; Play goes to the start
  assert.equal(a.kind, 'content');
  assert.equal(a.id, 's1');
  assert.deepEqual([a.t0, a.t1, a.t], [10, 30, 10]);
  assert.equal(a.title, 'Limits');
  assert.equal(a.section, 'Limits');
  assert.deepEqual(a.blocks, [{ type: 'bullets', items: [{ text: 'a plain string item', level: 0 }] }]);
  assert.equal(a.notes, '');
  assert.equal(a.said, '');
  // titles fall back by kind; times stay null when there are none (an end without a start is no range)
  assert.equal(agenda.title, 'Agenda');
  assert.equal(summary.title, 'Key takeaways');
  assert.deepEqual([summary.t0, summary.t1, summary.t], [null, null, null]);
  assert.equal(formulas.title, 'Formula sheet');
  assert.deepEqual([section.t0, section.t], [null, null]);
  assert.equal(section.title, 'Slide 7');
  // the duration falls back to the slides' end, the clip to [0, duration]
  assert.equal(d.duration, 30);
  assert.deepEqual(d.clip, [0, 30]);
});

test('Play: the server\'s moment, none when it says none, else the slide\'s start', () => {
  const [a, b, c] = cleanDeck({
    slides: [
      { kind: 'content', t0: 5, t1: 9, t: 6.5 },
      { kind: 'summary', t0: 5, t1: 9, t: null },
      { kind: 'content', t0: 5, t1: 9, t: 'soon' },
    ],
  }).slides;
  assert.deepEqual([a.t, b.t, c.t], [6.5, null, 5]);
});

test('slide ids are unique (they key the frames)', () => {
  const d = cleanDeck({ slides: [{ id: 's1' }, { id: 's1' }, { id: 's1-2' }, { id: 's1' }, {}] });
  const ids = d.slides.map((s) => s.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.deepEqual(ids, ['s1', 's1-2', 's1-2-2', 's1-3', 's5']);
});

test('figure pictures must be export/fig_NNN.png inside the job folder', () => {
  assert.equal(cleanFigurePath('export/fig_001.png'), 'export/fig_001.png');
  assert.equal(cleanFigurePath(' export\\fig_042.png '), 'export/fig_042.png');
  assert.equal(cleanFigurePath('export/fig_1234.png'), 'export/fig_1234.png');
  for (const bad of [
    '', 'fig_001.png', 'export/fig_1.png', 'export/page_001.png', 'export/fig_001.jpg', '/export/fig_001.png',
    '../export/fig_001.png', 'export/../fig_001.png', 'export/fig_001.png/../../x', 'export//fig_001.png',
    'C:/export/fig_001.png', 'http://evil.example/export/fig_001.png', 'pieces/012.png', 7, null,
  ]) {
    assert.equal(cleanFigurePath(bad), '', String(bad));
  }
  // an unsafe picture keeps the figure (its caption, and a plain line instead of the picture)
  const fig = cleanBlock({ type: 'figure', image: '../../secret.png', caption: 'A parabola', size: [0, 4], refs: ['#3', 'evil', 5] });
  assert.deepEqual(fig, { type: 'figure', refs: ['#3'], caption: 'A parabola', image: '', size: null });
});

test('unknown block types are dropped; each slide kind keeps only what it may show', () => {
  assert.equal(cleanBlock({ type: 'video', src: 'x.mp4' }), null);
  assert.equal(cleanBlock({ type: 'chart' }), null);
  assert.equal(cleanBlock(null), null);
  assert.equal(cleanBlock(['bullets']), null);
  // empty blocks are dropped
  assert.equal(cleanBlock({ type: 'bullets', items: [] }), null);
  assert.equal(cleanBlock({ type: 'equation', latex: '  ' }), null);
  assert.equal(cleanBlock({ type: 'steps', items: [{ latex: '', text: '' }] }), null);
  assert.equal(cleanBlock({ type: 'table', header: [], rows: [] }), null);
  assert.equal(cleanBlock({ type: 'callout', style: 'key', title: '', text: '', latex: '' }), null);

  const eq = (n: number) => ({ type: 'equation', latex: `x_${n}`, label: '', refs: [] });
  const fig = (n: number) => ({ type: 'figure', image: `export/fig_00${n}.png`, caption: '', refs: [], size: [4, 3] });
  const table = { type: 'table', header: ['a', 'b'], rows: [['1', '2']], caption: '' };
  // content: at most 3 blocks, the first figure / table / steps only, unknown types skipped
  const content = slideBlocks('content', [{ type: 'poll' }, fig(1), fig(2), table, table, eq(1), eq(2), eq(3)]);
  assert.deepEqual(content.map((b) => b.type), ['figure', 'table', 'equation']);
  assert.equal((content[0] as { image: string }).image, 'export/fig_001.png');
  // title and section slides show no blocks; a formula sheet only formulas, at most 8
  assert.deepEqual(slideBlocks('title', [eq(1)]), []);
  assert.deepEqual(slideBlocks('section', [eq(1)]), []);
  const sheet = slideBlocks('formulas', [table, ...Array.from({ length: 11 }, (_, i) => eq(i))]);
  assert.equal(sheet.length, LIMITS.formulas);
  assert.ok(sheet.every((b) => b.type === 'equation'));
  assert.deepEqual(slideBlocks('content', 'nope'), []);
});

test('limits: lists, text, formulas, tables, steps and callouts', () => {
  const long = 'word '.repeat(60).trim();
  const bullets = cleanBlock({
    type: 'bullets',
    items: [{ text: 'first', level: 1 }, ...Array.from({ length: 9 }, (_, i) => ({ text: `${long} ${i}`, level: i % 2 }))],
  });
  assert.ok(bullets && bullets.type === 'bullets');
  assert.equal(bullets.items.length, LIMITS.bullets);
  assert.equal(bullets.items[0].level, 0); // a list never starts indented
  assert.ok(bullets.items.every((it) => it.text.length <= LIMITS.bullet));
  assert.ok(bullets.items[1].text.endsWith('…'));

  const tooLong = `x = ${'a + '.repeat(60)}b`;
  assert.equal(cleanBlock({ type: 'equation', latex: tooLong }), null);
  const eq = cleanBlock({ type: 'equation', latex: 'E = mc^2', label: 'A label that goes on far beyond forty characters', refs: ['#1', '#1', '#L2'] });
  assert.ok(eq && eq.type === 'equation');
  assert.ok(eq.label.length <= LIMITS.label);
  assert.deepEqual(eq.refs, ['#1', '#L2']);

  const table = cleanBlock({
    type: 'table',
    header: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    rows: [...Array.from({ length: 12 }, (_, i) => [String(i), 'a cell with far more than forty characters in it, really']), ['', ''], 'junk'],
    caption: 'Made up',
  });
  assert.ok(table && table.type === 'table');
  assert.equal(table.header.length, LIMITS.columns);
  assert.equal(table.rows.length, LIMITS.rows);
  assert.ok(table.rows.every((r) => r.length === LIMITS.columns)); // short rows padded
  assert.ok(table.rows[0][1].length <= LIMITS.cell);
  // numbers in cells are text
  const nums = cleanBlock({ type: 'table', header: [], rows: [[1, 2.5]], caption: '' });
  assert.deepEqual(nums && nums.type === 'table' && nums.rows, [['1', '2.5']]);

  const steps = cleanBlock({
    type: 'steps',
    label: 'Derivation',
    items: [...Array.from({ length: 8 }, (_, i) => ({ latex: `x = ${i}`, text: long })), 'a plain string step', { latex: tooLong, text: '' }],
  });
  assert.ok(steps && steps.type === 'steps');
  assert.equal(steps.items.length, LIMITS.steps);
  assert.ok(steps.items.every((s) => s.text.length <= LIMITS.stepText));

  const callout = cleanBlock({ type: 'callout', style: 'shouty', title: 'T', text: long + long, latex: tooLong });
  assert.ok(callout && callout.type === 'callout');
  assert.equal(callout.style, 'note');
  assert.equal(callout.latex, '');
  assert.ok(callout.text.length <= LIMITS.calloutText);

  // speaker notes and what was said keep their paragraphs but not their size
  const d = cleanDeck({ slides: [{ kind: 'content', notes: 'one\n\n\n\ntwo   three', said: 'x'.repeat(5000) }] });
  assert.equal(d.slides[0].notes, 'one\n\ntwo three');
  assert.ok(d.slides[0].said.length <= LIMITS.said);
  // the title is cut at a word
  const t = cleanDeck({ title: 'A title '.repeat(20), slides: [] }).title;
  assert.ok(t.length <= LIMITS.title && t.endsWith('…'));
});

test('inline maths: $...$ is maths, money and stray dollars are text', () => {
  assert.deepEqual(splitInline('the slope $m$ of $y = mx + c$'), [
    { math: false, text: 'the slope ' },
    { math: true, tex: 'm' },
    { math: false, text: ' of ' },
    { math: true, tex: 'y = mx + c' },
  ]);
  assert.deepEqual(splitInline('costs $5 and $10 today'), [{ math: false, text: 'costs $5 and $10 today' }]);
  assert.deepEqual(splitInline('a \\$5 fee and $x$'), [
    { math: false, text: 'a $5 fee and ' },
    { math: true, tex: 'x' },
  ]);
  assert.deepEqual(splitInline('$$\\frac{a}{b}$$ done'), [
    { math: true, tex: '\\frac{a}{b}' },
    { math: false, text: ' done' },
  ]);
  assert.deepEqual(splitInline('open $x but never closed'), [{ math: false, text: 'open $x but never closed' }]);
  assert.deepEqual(splitInline('$ x$ is not maths'), [{ math: false, text: '$ x$ is not maths' }]);
  // LaTeX with spaces inside the dollars is maths, as the server reads it
  assert.deepEqual(splitInline('then $ x^2 $ grows'), [
    { math: false, text: 'then ' },
    { math: true, tex: 'x^2' },
    { math: false, text: ' grows' },
  ]);
  assert.deepEqual(splitInline('a$b$c'), [
    { math: false, text: 'a' },
    { math: true, tex: 'b' },
    { math: false, text: 'c' },
  ]);
  // an escaped dollar inside maths stays in the TeX
  assert.deepEqual(splitInline('$\\$5$'), [{ math: true, tex: '\\$5' }]);
  assert.deepEqual(splitInline(''), []);
  assert.equal(inlinePlain('as $n \\to \\infty$'), 'as n \\to \\infty');
});

test('text with maths is never cut inside a formula', () => {
  const s = `${'word '.repeat(20)}then $\\frac{a+b}{c}$ and more words after it`;
  const cut = cutInline(s, 110);
  assert.ok(cut.length <= 110 && shownLength(cut) <= 110);
  assert.ok(cut.endsWith('…'));
  // every "$" that opens maths in the result is closed again
  const maths = splitInline(cut).filter((p) => p.math);
  assert.equal((cut.match(/\$/g) ?? []).length, maths.length * 2);
  // short text is left alone
  assert.equal(cutInline('$x$ is short', 120), '$x$ is short');
  // nothing before a long formula: its words without typesetting
  const only = `$${'x + '.repeat(40)}y$`;
  assert.ok(!cutInline(only, 50).includes('$'));
});

test('limits count what a reader sees, as the server does: a formula as its typeset characters', () => {
  // The server (app/export.py shown_len) counts "$\sigma\le\tau$" as 3 characters, not 16. A 43-character cell that
  // shows 14 came through cut to raw LaTeX ("\sigma\le\tau\le\omega, \ka…") before.
  const cell = '$\\sigma\\le\\tau\\le\\omega, \\kappa\\neq\\lambda$';
  assert.ok(cell.length > LIMITS.cell);
  const table = cleanBlock({ type: 'table', header: ['Case', 'Rule'], rows: [['Ordered', cell]], caption: '' });
  assert.ok(table && table.type === 'table');
  assert.equal(table.rows[0][1], cell);
  // a bullet whose LaTeX makes it longer than 120 characters, though it shows 77
  const bullet =
    'Set $p_{1}$, $p_{2}$ and $\\rho$ so that $\\rho\\ge\\frac{1}{2}$, then $\\sigma_{0}\\to\\infty$ while ' +
    '$\\tau\\le\\omega$ and $\\kappa\\neq\\lambda$ stay fixed';
  assert.ok(bullet.length > LIMITS.bullet);
  const list = cleanBlock({ type: 'bullets', items: [{ text: bullet, level: 0 }] });
  assert.ok(list && list.type === 'bullets');
  assert.equal(list.items[0].text, bullet);
  // the same in a step
  const step = 'Set $p_{1}$, $p_{2}$ and $\\rho$ so that $\\rho\\ge\\frac{1}{2}$, then $\\sigma_{0}\\to\\infty$ while $\\tau\\le\\omega$';
  assert.ok(step.length > LIMITS.stepText);
  const steps = cleanBlock({ type: 'steps', label: '', items: [{ latex: '', text: step }, { latex: 'x = 1', text: '' }] });
  assert.ok(steps && steps.type === 'steps');
  assert.equal(steps.items[0].text, step);
  // never more than the server's count (its Unicode rendering), so the page never cuts what the server kept
  const server: [string, number][] = [
    ['\\frac{1}{12}', 4], ['\\left(1+\\frac{1}{n}\\right)^{n}', 8], ['x_{i}^{2}', 4], ['\\{x\\}', 1], ['10\\%', 3],
    ['n \\to \\infty', 5], ['\\mathbb{R}', 7], ['\\overline{AB}', 10], ['\\sqrt[3]{x}', 8], ['\\binom{n}{k}', 7],
    ['365 \\times 24 \\times 60', 13], ['\\boldsymbol{v}', 11], ['\\text{if } x', 4],
  ];
  for (const [tex, n] of server) assert.ok(mathShown(tex) <= n, `${tex}: ${mathShown(tex)} > ${n}`);
  assert.equal(shownLength('a \\$5 fee and $\\alpha^{2}$'), 'a $5 fee and '.length + 2);
  // junk is still bounded: LaTeX longer than a display formula counts in full, and a cut keeps to the limit as shown
  assert.equal(mathShown(`${'\\left('.repeat(50)}x`), 301);
  const many = `${'$\\alpha$ and '.repeat(40)}end`;
  const cut = cutInline(many, LIMITS.bullet);
  assert.ok(shownLength(cut) <= LIMITS.bullet && cut.endsWith('…'));
  assert.equal(splitInline(cut).filter((p) => p.math).length * 2, (cut.match(/\$/g) ?? []).length);
});

test('a steps label keeps the 60 characters the server allows it', () => {
  const label = 'Finding the break-even point from the two cost lines, slowly';
  assert.equal(label.length, 60);
  const steps = cleanBlock({ type: 'steps', label, items: [{ latex: 'x = 1', text: '' }, { latex: 'y = 2', text: '' }] });
  assert.ok(steps && steps.type === 'steps');
  assert.equal(steps.label, label);
  const eq = cleanBlock({ type: 'equation', latex: 'x = 1', label, refs: [] });
  assert.ok(eq && eq.type === 'equation' && eq.label.length <= LIMITS.label);
});

test('while writing: the step line and the progress', () => {
  assert.equal(stepLine({ state: 'writing', step: 'Planning the slides' }), 'Muse Spark is planning the slides…');
  assert.equal(stepLine({ state: 'writing', step: 'Writing part 2 of 5' }), 'Writing part 2 of 5…');
  assert.equal(stepLine({ state: 'writing', step: 'checking the slides...' }), 'Checking the slides…');
  assert.equal(stepLine({ state: 'writing', step: '' }), 'Muse Spark is writing the slides…');
  assert.equal(stepLine({ state: 'none', step: '' }), 'Muse Spark is starting on the slides…');
  assert.equal(progressShare({ done: 2, total: 5 }), 0.4);
  assert.equal(progressShare({ done: 0, total: 0 }), null);
});

test('content layout follows the PowerPoint: one column, or the figure in its own column', () => {
  const eq = { type: 'equation' as const, latex: 'x', label: '', refs: [] };
  const fig = (size: [number, number] | null) => ({ type: 'figure' as const, refs: [], caption: '', image: 'export/fig_001.png', size });
  assert.equal(contentLayout([eq]).mode, 'stack');
  assert.equal(contentLayout([]).mode, 'stack');
  assert.equal(contentLayout([fig([4, 3])]).mode, 'figure');
  // a figure beside the text, on the side where it comes in the list; wider for a wider figure
  const right = contentLayout([eq, fig([1600, 1000])]);
  assert.equal(right.mode, 'split');
  assert.deepEqual(right.main, [eq]);
  assert.equal(right.side, 0.56);
  assert.equal(right.figureFirst, false);
  const left = contentLayout([fig([300, 600]), eq]);
  assert.equal(left.side, 0.38);
  assert.equal(left.figureFirst, true);
  assert.equal(contentLayout([eq, fig(null)]).side, 0.5);
  // a very wide figure stays in the column, in its place
  const banner = contentLayout([eq, fig([1600, 500])]);
  assert.equal(banner.mode, 'stack');
  assert.equal(banner.main.length, 2);
});

test('a tall, narrow table sits beside the other blocks, as in the PowerPoint; other tables are stacked', () => {
  const pts = { type: 'bullets' as const, items: [{ text: 'Rates double', level: 0 as const }] };
  const table = (rows: number, cols: number) => ({
    type: 'table' as const,
    header: Array.from({ length: cols }, (_, j) => `H${j}`),
    rows: Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => `${i}.${j}`)),
    caption: '',
  });
  const tall = table(5, 2);
  const lay = contentLayout([tall, pts]);
  assert.deepEqual([lay.mode, lay.table, lay.figure, lay.side, lay.figureFirst], ['split', tall, null, 0.5, true]);
  assert.deepEqual(lay.main, [pts]);
  assert.equal(contentLayout([pts, tall]).figureFirst, false);
  assert.equal(sideTable([pts, table(3, 2)]), null); // short
  assert.equal(sideTable([pts, table(6, 4)]), null); // wide
  assert.equal(sideTable([tall]), null); // alone
  const fig = { type: 'figure' as const, refs: [], caption: '', image: 'export/fig_001.png', size: [400, 300] as [number, number] };
  assert.equal(contentLayout([fig, tall, pts]).figure, fig); // a figure takes the side; the table stays with the text
  assert.equal(contentLayout([fig, tall, pts]).table, null);
  assert.equal(contentLayout([pts, table(3, 2)]).mode, 'stack');
});

test('what the frame draws: formulas in a row, takeaways as numbered rows, a number step as text', () => {
  const eq = (latex: string) => ({ type: 'equation' as const, latex, label: '', refs: [] });
  const pts = { type: 'bullets' as const, items: [{ text: 'One', level: 0 as const }, { text: 'Two $x$', level: 0 as const }] };
  assert.deepEqual(frameBlocks('content', [eq('a=1'), eq('b=2'), pts, eq('c=3')]), [
    { type: 'equations', items: [eq('a=1'), eq('b=2')] },
    pts,
    eq('c=3'),
  ]);
  assert.deepEqual(frameBlocks('summary', [pts]), [
    { type: 'steps', label: '', items: [{ latex: '', text: 'One' }, { latex: '', text: 'Two $x$' }] },
  ]);
  assert.deepEqual(frameBlocks('content', [pts]), [pts]);
  // numbers only: text in the body font, as the PowerPoint sets them (same cases as tests/test_pptx_deck.py)
  const plain: [string, string][] = [
    ['1331', '1331'], [' \\$2.25 ', '$2.25'], ['1{,}000', '1,000'], ['50\\%', '50%'], ['-3.5', '−3.5'],
    ['1000 \\to 1100', '1000 → 1100'], ['225\\times 1.5=337.5', '225 × 1.5 = 337.5'], ['1+1', '1 + 1'], ['7 - 2', '7 − 2'],
  ];
  for (const [latex, text] of plain) assert.equal(plainMath(latex), text, latex);
  for (const not of ['x', '2x', '10^{3}', 'A=2', '\\frac{1}{2}', '', '+']) assert.equal(plainMath(not), '', not);
});

test('formula sheet and agenda grids', () => {
  assert.deepEqual(formulaGrid(2), { columns: 1, rows: 2, cardHeight: 150 });
  assert.deepEqual(formulaGrid(3), { columns: 2, rows: 2, cardHeight: 150 });
  assert.deepEqual(formulaGrid(8), { columns: 2, rows: 4, cardHeight: (360 - 18 * 3) / 4 });
  const d = cleanDeck(serverDeck());
  const agenda = d.slides.find((s) => s.kind === 'agenda') as DeckSlide;
  const one = agendaLayout(agenda, d.slides);
  assert.equal(one.columns, 1);
  assert.equal(one.size, 22);
  assert.equal(one.rowHeight, 52);
  // past 7 sections: two columns, smaller text, rows sharing the body
  const many = Array.from({ length: 10 }, (_, i) => ({ ...d.slides[3], id: `c${i}`, section: `Part ${i}` }));
  const two = agendaLayout(agenda, many);
  assert.deepEqual([two.columns, two.rows, two.size], [2, 5, 18]);
  assert.equal(two.rowHeight, 52);
  const lots = agendaLayout(agenda, Array.from({ length: 18 }, (_, i) => ({ ...d.slides[3], id: `c${i}`, section: `Part ${i}` })));
  assert.equal(lots.rowHeight, 40);
});

test('sections, the agenda, "Part n" and the eyebrow', () => {
  const d = cleanDeck(serverDeck());
  const secs = deckSections(d.slides);
  assert.deepEqual(
    secs.map((s) => [s.title, s.t0, s.t1, s.first]),
    [
      ['Simple interest', 12, 140, 2],
      ['Compounding', 140, 300, 6],
    ],
  );
  const section = d.slides.find((s) => s.kind === 'section') as DeckSlide;
  assert.deepEqual(sectionFacts(d.slides, section), { part: 1, range: '0:12–2:20' });
  // a section slide without times shows its section's
  assert.deepEqual(sectionFacts(d.slides, { ...section, t0: null, t1: null }), { part: 1, range: '0:12–2:20' });
  // the agenda lists the deck's sections with their times, whatever the slide's own list says
  const agenda = d.slides.find((s) => s.kind === 'agenda') as DeckSlide;
  assert.deepEqual(agendaLayout({ ...agenda, blocks: [] }, d.slides).entries, [
    { text: 'Simple interest', range: '0:12–2:20' },
    { text: 'Compounding', range: '2:20–5:00' },
  ]);
  // no sections: what the slide lists
  const bare = d.slides.filter((s) => s.kind !== 'content' && s.kind !== 'section');
  assert.deepEqual(agendaLayout(agenda, bare).entries, [
    { text: 'Simple interest', range: '' },
    { text: 'Compounding', range: '' },
  ]);
  // a run of slides is one section; the same name again later is a new one
  const again = [d.slides[3], d.slides[6], { ...d.slides[4], id: 'x' }];
  assert.deepEqual(deckSections(again).map((s) => s.title), ['Simple interest', 'Compounding', 'Simple interest']);
  // the section above a content slide's title, when the deck has two or more
  assert.equal(eyebrow(d.slides[3], 2), 'Simple interest');
  assert.equal(eyebrow(d.slides[3], 1), '');
  assert.equal(eyebrow(d.slides[7], 2), '');
  assert.equal(coverFacts(d), 'Board lecture · 7:58 · made with Replay Board from the recording');
  assert.equal(coverFacts({ kind: 'slides', duration: 0 }), 'Slide lecture · made with Replay Board from the recording');
});

test('time range and player link', () => {
  assert.equal(rangeLabel(0, 95.4), '0:00–1:35');
  assert.equal(rangeLabel(3600, 3725), '1:00:00–1:02:05');
  assert.equal(rangeLabel(null, null), '');
  assert.equal(rangeLabel(65, null), '1:05');
  assert.equal(playerAt('test-job', 83.9), '#/job/test-job?t=83');
  assert.equal(playerAt('a b', -3), '#/job/a%20b?t=0');
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
  assert.equal(presentKey('ArrowUp', false, 3, n), 2);
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

// ---------- getting the deck ----------

/** Fake server calls: GET answers from a script (its last answer repeats); every call is logged. */
function fakeCalls(gets: (SlidesStatus | Error)[], writes: (SlidesStatus | Error)[] = []) {
  const log: string[] = [];
  const shown: string[] = [];
  let live = true;
  const next = (list: (SlidesStatus | Error)[], what: string) => {
    log.push(what);
    const v = list.length > 1 ? list.shift()! : list[0];
    return v instanceof Error ? Promise.reject(v) : Promise.resolve(v);
  };
  const calls: FlowCalls = {
    get: () => next(gets, 'GET'),
    write: () => next(writes, 'POST'),
    wait: (ms) => {
      log.push(`wait ${ms}`);
      return Promise.resolve();
    },
    show: (s) => shown.push(s.state),
    alive: () => live,
  };
  return { calls, log, shown, stop: () => (live = false) };
}

const S = (state: SlidesStatus['state'], over: Partial<SlidesStatus> = {}): SlidesStatus => cleanStatus(status(state, over));

test('nothing written: one POST, then polling every 1.5 s until ready', async () => {
  const f = fakeCalls([S('none'), S('writing'), S('writing'), S('ready')], [S('writing', { step: 'Planning the slides' })]);
  const last = await runFlow(f.calls);
  assert.equal(last.state, 'ready');
  assert.deepEqual(f.log, ['GET', 'POST', `wait ${POLL_MS}`, 'GET', `wait ${POLL_MS}`, 'GET', `wait ${POLL_MS}`, 'GET']);
  assert.deepEqual(f.shown, ['none', 'writing', 'writing', 'writing', 'ready']);
  assert.equal(POLL_MS, 1500);
});

test('already ready or being written: no POST', async () => {
  const ready = fakeCalls([S('ready')]);
  await runFlow(ready.calls);
  assert.deepEqual(ready.log, ['GET']);
  const writing = fakeCalls([S('writing'), S('ready')]);
  await runFlow(writing.calls);
  assert.deepEqual(writing.log, ['GET', `wait ${POLL_MS}`, 'GET']);
});

test('failed: shown, no automatic retry; Try again POSTs once', async () => {
  const f = fakeCalls([S('failed', { error: 'Muse Spark is busy' })]);
  const last = await runFlow(f.calls);
  assert.equal(last.state, 'failed');
  assert.equal(last.error, 'Muse Spark is busy');
  assert.deepEqual(f.log, ['GET']);
  const again = fakeCalls([S('writing'), S('ready')], [S('writing')]);
  assert.equal((await runFlow(again.calls, { write: true })).state, 'ready');
  assert.deepEqual(again.log, ['POST', `wait ${POLL_MS}`, 'GET', `wait ${POLL_MS}`, 'GET']);
});

test('never two POSTs: not in one run, and not from two runs at once (a page mounted twice)', async () => {
  // the server keeps saying "none" after the POST: poll a while, then one plain line; still one POST
  const stuck = fakeCalls([S('none')], [S('none')]);
  await assert.rejects(runFlow(stuck.calls), (e: Error) => e.message === NOT_STARTED);
  assert.equal(stuck.log.filter((x) => x === 'POST').length, 1);
  assert.equal(stuck.log.filter((x) => x === 'GET').length, NONE_LIMIT + 1);

  const calls = new Map<string, Promise<SlidesStatus>>();
  let posts = 0;
  let release!: (s: SlidesStatus) => void;
  const post = () => {
    posts++;
    return new Promise<SlidesStatus>((r) => (release = r));
  };
  const a = joinCall(calls, 'job-1', post);
  const b = joinCall(calls, 'job-1', post);
  assert.equal(a, b);
  assert.equal(posts, 1);
  const other = joinCall(calls, 'job-2', () => Promise.resolve(S('writing')));
  assert.notEqual(other, a);
  release(S('writing'));
  await a;
  await other;
  // once it has answered, a later Try again makes a new request
  assert.equal(calls.size, 0);
  joinCall(calls, 'job-1', post);
  assert.equal(posts, 2);
});

test('polling rides out a blip but not a dead server; a page that went away stops', async () => {
  const down = new Error('Cannot reach the Replay Board server. Is `python -m app.server` running?');
  const blip = fakeCalls([S('writing'), down, S('ready')]);
  assert.equal((await runFlow(blip.calls)).state, 'ready');
  assert.deepEqual(blip.shown, ['writing', 'writing', 'ready']);

  const dead = fakeCalls([S('writing'), down]);
  await assert.rejects(runFlow(dead.calls), (e: Error) => e === down);
  assert.equal(dead.log.filter((x) => x === 'GET').length, 1 + MISS_LIMIT);

  // the page went away while waiting: no more reads
  const gone = fakeCalls([S('writing')]);
  gone.calls.wait = () => {
    gone.stop();
    return Promise.resolve();
  };
  assert.equal((await runFlow(gone.calls)).state, 'writing');
  assert.deepEqual(gone.log, ['GET']);

  // a first read that fails is the page's error line
  const first = fakeCalls([new Error('This lecture is not processed yet.')]);
  await assert.rejects(runFlow(first.calls), /not processed yet/);
});

test('sleep ends early with an AbortError', async () => {
  const ac = new AbortController();
  const p = sleep(10_000, ac.signal);
  ac.abort();
  await assert.rejects(p, (e: Error) => e.name === 'AbortError');
  await sleep(1);
});

// ---------- small things ----------

test('error lines read as sentences, on one line', () => {
  assert.equal(sentence('this lecture is not processed yet, so there are no slides yet'), 'This lecture is not processed yet, so there are no slides yet');
  assert.equal(sentence('  Muse Spark could not\nwrite   the slides '), 'Muse Spark could not write the slides');
  assert.equal(sentence(''), '');
});

test("PowerPoint file name: the server's, else the lecture title; never a path", () => {
  const both = `attachment; filename="Vectors and matrices - slides.pptx"; filename*=UTF-8''Vecteurs%20et%20matrices%20%C3%A9t%C3%A9%20-%20slides.pptx`;
  assert.equal(pptxFileName(both, 'x'), 'Vecteurs et matrices été - slides.pptx');
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

// ---------- fitting the text (fit.ts) ----------

/** A made-up page: boxes whose text fits at `fit` points or less; setting a size makes the page dirty, and the first
 *  measurement after that lays the page out (what costs time in a browser). */
function fakePage() {
  const page = { layouts: 0, dirty: false };
  const box = (fit: number, max = 24, min = 10) => {
    const b = {
      max,
      min,
      size: 0,
      set(s: number) {
        b.size = s;
        page.dirty = true;
      },
      spills() {
        if (page.dirty) {
          page.layouts++;
          page.dirty = false;
        }
        return b.size > fit;
      },
    };
    return b;
  };
  return { page, box };
}

test('fitting: each box gets the largest size that does not spill, all boxes measured together', () => {
  const { page, box } = fakePage();
  const fits = [24, 30, 23, 17, 10, 4, 12, 21];
  const boxes = Array.from({ length: 80 }, (_, i) => box(fits[i % fits.length]));
  const fixed = box(5, 24, 24); // min = max: no search, stays at its size even though it spills
  const steps = fitAll([...boxes, fixed]);
  boxes.forEach((b, i) => assert.equal(b.size, Math.min(24, Math.max(10, fits[i % fits.length]))));
  assert.equal(fixed.size, 24);
  // one layout per step for the whole page: a binary search over 10-23 is at most 4 steps, plus trying the largest
  assert.ok(steps <= 5, `steps ${steps}`);
  assert.ok(page.layouts <= steps, `layouts ${page.layouts}`);
  // one box at a time, the same page took a layout per box per step
  const alone = fakePage();
  const lone = Array.from({ length: 80 }, (_, i) => alone.box(fits[i % fits.length]));
  for (const b of lone) fitAll([b]);
  assert.ok(alone.page.layouts > 10 * page.layouts);
  assert.equal(fitAll([]), 0);
});

test('fitting queue: boxes added while rendering are fitted once, together, before the next paint', async () => {
  const { page, box } = fakePage();
  const q = new FitQueue(20);
  const a = box(18);
  const b = box(30);
  const gone = { ...box(12), live: () => false };
  const dropped = box(11);
  q.add(a);
  q.add(b);
  q.add(gone);
  q.add(dropped);
  q.delete(dropped);
  assert.equal(a.size, 0); // not yet: the other frames of this render join first
  await Promise.resolve();
  assert.deepEqual([a.size, b.size, gone.size, dropped.size], [18, 24, 0, 0]);
  const after = page.layouts;
  // after a resize or when fonts arrive: once things are quiet, however many times it was asked
  q.add(a, true);
  q.add(a, true);
  q.add(b, true);
  await Promise.resolve();
  assert.equal(page.layouts, after);
  await new Promise((r) => setTimeout(r, 60));
  assert.ok(page.layouts > after && page.layouts <= after + 5);
  assert.deepEqual([a.size, b.size], [18, 24]);
});
