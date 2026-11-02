// Run with `npm test`. Ask Muse answers: the Markdown subset (paragraphs, lists, bold, italic, code), maths in $ and
// $$, times [m:ss] / [h:mm:ss], refs [#12] [#L3] [#s0e1], half-written maths while streaming, and that nothing in an
// answer can become HTML (rendered with React's server renderer, as the drawer would).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createElement, Fragment } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { citedTimes, parseClock, parseInline, parseMarkdown, type Block, type Inline } from '../src/ask/markdown.ts';
import { renderBlocks, type RenderContext } from '../src/ask/render.ts';

const kinds = (xs: Inline[]) => xs.map((x) => x.t);
const text = (xs: Inline[]): string =>
  xs
    .map((x) =>
      x.t === 'text' || x.t === 'code' || x.t === 'pending'
        ? x.v
        : x.t === 'b' || x.t === 'i'
          ? text(x.c)
          : x.t === 'math'
            ? `<${x.tex}>`
            : x.t === 'time'
              ? `{${x.s}}`
              : x.t === 'ref'
                ? `{${x.ref}}`
                : '\n',
    )
    .join('');

const ctx = (over: Partial<RenderContext> = {}): RenderContext => ({
  timeOk: () => true,
  refInfo: (ref) => (ref === '#12' ? { label: 'Area formula', t: 61 } : null),
  onTime: () => {},
  onRef: () => {},
  clock: (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`,
  ...over,
});
const html = (md: string, c = ctx(), streaming = false) =>
  renderToStaticMarkup(createElement(Fragment, null, ...renderBlocks(parseMarkdown(md, { streaming }), c)));

test('clock times: m:ss and h:mm:ss', () => {
  assert.equal(parseClock('0:05'), 5);
  assert.equal(parseClock('12:34'), 754);
  assert.equal(parseClock('1:02:03'), 3723);
  assert.equal(parseClock('75:10'), 4510); // minutes past an hour, written as m:ss
  assert.equal(parseClock('1:60'), null);
  assert.equal(parseClock('abc'), null);
});

test('times become time nodes, also as a range', () => {
  const xs = parseInline('See [2:05] and [1:02:03], then [1:00–1:30].');
  assert.deepEqual(
    xs.filter((x) => x.t === 'time').map((x) => (x as { s: number }).s),
    [125, 3723, 60, 90],
  );
  assert.equal(text(xs), 'See {125} and {3723}, then {60}–{90}.');
  // not a time: brackets around other things stay text
  assert.equal(text(parseInline('[note] [1:5] [ab:cd]')), '[note] [1:5] [ab:cd]');
});

test('refs to pieces, lines and slide elements; several in one bracket; bad ones stay text', () => {
  const xs = parseInline('Compare [#12] with [#L3] and [#s0e1]; also [#4, #5]. Not [#foo] or [#12a].');
  assert.deepEqual(
    xs.filter((x) => x.t === 'ref').map((x) => (x as { ref: string }).ref),
    ['#12', '#L3', '#s0e1', '#4', '#5'],
  );
  assert.match(text(xs), /Not \[#foo\] or \[#12a\]\.$/);
});

test('bold, italic and code; snake_case and lone stars are not emphasis', () => {
  const xs = parseInline('**Key**: the *rate* is `r`, not snake_case_name or 2 * 3 * 4.');
  assert.deepEqual(kinds(xs).slice(0, 5), ['b', 'text', 'i', 'text', 'code']);
  assert.equal(text(xs), 'Key: the rate is r, not snake_case_name or 2 * 3 * 4.');
  assert.deepEqual(kinds(parseInline('_word_ and __strong__')), ['i', 'text', 'b']);
  // inside code nothing is parsed
  assert.deepEqual(parseInline('`[1:00] **x** $y$`'), [{ t: 'code', v: '[1:00] **x** $y$' }]);
});

test('inline and display maths; prices are not maths', () => {
  const xs = parseInline('So $A = P(1+r)^n$ and $$\\int_0^1 x\\,dx$$ but it costs $5 and $10.');
  const maths = xs.filter((x) => x.t === 'math') as { tex: string; display: boolean }[];
  assert.deepEqual(
    maths.map((m) => [m.tex, m.display]),
    [
      ['A = P(1+r)^n', false],
      ['\\int_0^1 x\\,dx', true],
    ],
  );
  assert.match(text(xs), /costs \$5 and \$10\.$/);
  assert.deepEqual(
    (parseInline('\\(x^2\\) and \\[y\\]').filter((x) => x.t === 'math') as { display: boolean }[]).map((m) => m.display),
    [false, true],
  );
  assert.equal(text(parseInline('a \\$ sign and \\*stars\\*')), 'a $ sign and *stars*');
});

test('blocks: paragraphs, lists (with nesting and a start number), display maths, headings', () => {
  const md = [
    '### Summary',
    'First line',
    'second line.',
    '',
    '- one',
    '- two [1:00]',
    '  - two point one',
    '',
    '3. third',
    '4. fourth',
    '',
    '$$',
    'A = P(1+r)^n',
    '$$',
    'After.',
  ].join('\n');
  const blocks = parseMarkdown(md);
  assert.deepEqual(
    blocks.map((b) => b.t),
    ['h', 'p', 'ul', 'ol', 'math', 'p'],
  );
  const p = blocks[1] as { c: Inline[] };
  assert.deepEqual(kinds(p.c), ['text', 'br', 'text']);
  const ul = blocks[2] as Extract<Block, { t: 'ul' }>;
  assert.equal(ul.items.length, 2);
  assert.equal(ul.items[1].sub[0].t, 'ul');
  const ol = blocks[3] as Extract<Block, { t: 'ol' }>;
  assert.equal(ol.start, 3);
  assert.equal(ol.items.length, 2);
  assert.equal((blocks[4] as { tex: string }).tex, 'A = P(1+r)^n');
  assert.deepEqual(citedTimes(blocks), [60]);
  // CRLF input parses the same
  assert.deepEqual(parseMarkdown(md.replace(/\n/g, '\r\n')), blocks);
});

test('a list item can hold maths and a ref; a blank line between items keeps one list', () => {
  const blocks = parseMarkdown('1. $x$ at [#12]\n\n2. done');
  assert.equal(blocks.length, 1);
  const ol = blocks[0] as Extract<Block, { t: 'ol' }>;
  assert.deepEqual(kinds(ol.items[0].c), ['math', 'text', 'ref']);
});

test('streaming: half-written maths is pending, not a broken formula; finished text is not', () => {
  const tail = parseMarkdown('The rule is $\\frac{a', { streaming: true });
  const p = tail[0] as { c: Inline[] };
  assert.deepEqual(kinds(p.c), ['text', 'pending']);
  assert.equal((p.c[1] as { v: string }).v, '$\\frac{a');
  // the same text once the answer is complete: just text
  assert.deepEqual(kinds((parseMarkdown('The rule is $\\frac{a')[0] as { c: Inline[] }).c), ['text']);
  // an open $$ block at the end
  assert.equal(parseMarkdown('Look:\n\n$$\nA = ', { streaming: true }).at(-1)?.t, 'pending');
  // bold still being written shows as bold without its marker
  const b = (parseMarkdown('This is **impor', { streaming: true })[0] as { c: Inline[] }).c;
  assert.deepEqual(kinds(b), ['text', 'b']);
  // only the end of the answer is "still being written": an earlier unclosed $ stays text
  const two = parseMarkdown('It costs $5 to\n\nNext $x', { streaming: true });
  assert.deepEqual(kinds((two[0] as { c: Inline[] }).c), ['text']);
  assert.deepEqual(kinds((two[1] as { c: Inline[] }).c), ['text', 'pending']);
});

test('no HTML injection: tags, attributes and links in an answer stay text', () => {
  const out = html(
    '<script>alert(1)</script> <img src=x onerror="alert(2)"> **<b>bold</b>** [link](javascript:alert(3)) `<i>`',
  );
  assert.ok(!/<script|<img|<b>|<i>|<a /i.test(out), out);
  assert.match(out, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.match(out, /&lt;img src=x onerror=&quot;alert\(2\)&quot;&gt;/);
  assert.match(out, /<strong>&lt;b&gt;bold&lt;\/b&gt;<\/strong>/);
  assert.match(out, /\[link\]\(javascript:alert\(3\)\)/);
  // KaTeX renders maths with trust off: \href and raw HTML commands do not become links or elements
  const tex = html('$\\href{javascript:alert(4)}{x}$ and $\\htmlId{evil}{y}$');
  assert.ok(!/<a |href=|id="evil"/i.test(tex), tex);
  assert.match(tex, /class="katex"/);
});

test('rendering: times and known refs are buttons; unknown refs and out-of-range times stay text', () => {
  const out = html('At [1:01] see [#12] and [#99]. Not [9:59:59].', ctx({ timeOk: (s) => s < 3600 }));
  assert.match(out, /<button type="button" class="md-time"[^>]*>1:01<\/button>/);
  assert.match(out, /<button type="button" class="md-ref" data-ref="#12"[^>]*><span class="md-ref-label">Area formula<\/span><\/button>/);
  assert.match(out, /and \[#99\]\./);
  assert.match(out, /Not \[9:59:59\]\./);
  // a display formula gets its own block; a broken formula is drawn by KaTeX (no exception)
  assert.match(html('$$x^2$$'), /<div class="md-dmath"><span class="katex-display">/);
  assert.match(html('$\\frac{1}{$ ok'), /katex|md-code/);
  // lists render as lists, with their start number
  assert.match(html('3. a\n4. b'), /<ol class="md-list" start="3"><li>a<\/li><li>b<\/li><\/ol>/);
});

test('rendering: no empty line around a display formula inside a list item or a paragraph', () => {
  // a real answer: "Using the rule ...:\n$$A=P(1+r)^n$$\nwhat is $A$?" inside "1. ..."
  const out = html('1. Using the rule:\n$$A=P(1+r)^n$$\nwhat is $A$?\n\nThe rule:\n$$x^2$$\nso on');
  assert.doesNotMatch(out, /<br\/?>\s*<span class="md-dmath">/);
  assert.doesNotMatch(out, /<span class="md-dmath">(?:(?!<\/li>|<\/p>)[\s\S])*?<\/span><br\/?>what/);
  assert.equal((out.match(/<br\/?>/g) ?? []).length, 0);
  // other line breaks stay
  assert.match(html('a\nb'), /a<br\/>b/);
});

test('code: a run of backticks closes only on a run as long; fenced blocks are kept as they are', () => {
  assert.deepEqual(parseInline('``a`b`` and ```x```'), [
    { t: 'code', v: 'a`b' },
    { t: 'text', v: ' and ' },
    { t: 'code', v: 'x' },
  ]);
  // an unclosed run is text, and "``" cannot close a later single "`"
  assert.equal(text(parseInline('two `` then `one`')), 'two `` then one');
  // emphasis skips over a code span that holds its marker
  assert.deepEqual(kinds(parseInline('**see `a**b`**')), ['b']);
  const md = 'Try this:\n```python\nx = [1:00]  # **not bold** $not maths$\n<b>tag</b>\n```\nAfter [0:05].';
  const blocks = parseMarkdown(md);
  assert.deepEqual(
    blocks.map((b) => b.t),
    ['p', 'pre', 'p'],
  );
  assert.equal((blocks[1] as { v: string }).v, 'x = [1:00]  # **not bold** $not maths$\n<b>tag</b>');
  assert.deepEqual(citedTimes(blocks), [5]); // a time inside code is not a link
  const out = html(md);
  assert.match(out, /<pre class="md-code md-pre"><code>x = \[1:00\]  # \*\*not bold\*\* \$not maths\$\n&lt;b&gt;tag&lt;\/b&gt;<\/code><\/pre>/);
  // ~~~ fences, and a fence still open while the answer streams in: the code so far
  assert.equal((parseMarkdown('~~~\na\n~~~')[0] as { v: string }).v, 'a');
  const open = parseMarkdown('Code:\n\n```js\nlet a = 1;\nlet b', { streaming: true });
  assert.deepEqual(open.at(-1), { t: 'pre', v: 'let a = 1;\nlet b' });
});

test('a ref chip shows the item typeset when it has LaTeX, else its label', () => {
  const tex = html('see [#12]', ctx({ refInfo: () => ({ label: 'A=pi r^2', t: 61, latex: '$A=\pi r^2$' }) }));
  assert.match(tex, /class="md-ref-label md-ref-math"><span class="katex">/);
  assert.ok(!tex.includes('A=pi r^2</span>'), tex);
  const plain = html('see [#12]', ctx({ refInfo: () => ({ label: 'Area formula', t: 61, latex: '' }) }));
  assert.match(plain, /<span class="md-ref-label">Area formula<\/span>/);
});

test('money is not maths: a formula never holds a bare dollar sign', () => {
  const maths = (md: string) => JSON.stringify(parseMarkdown(md, { streaming: false })).match(/"t":"math","tex":"[^"]*"/g) ?? [];
  assert.deepEqual(maths('Start from $1 at 100% for a year [#L2] and you get $2.'), []);
  assert.deepEqual(maths('twice a year = $2.25 and monthly = $2.61.'), []);
  assert.deepEqual(maths('Cost is $3, not $x$.'), ['"t":"math","tex":"x"']);
  assert.deepEqual(maths('Inline $x^2$ and money $5 and $10 together.'), ['"t":"math","tex":"x^2"']);
});
