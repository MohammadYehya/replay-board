// Run with `npm test`. Line layouts for typeset readings, and the plain-text clean-up.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { clauses, relationLines, texLayouts, writtenText } from '../src/player/texLayout.ts';
import { isTypesettable } from '../src/player/timing.ts';

const show = (latex: string) => texLayouts(latex).map((l) => l.map((x) => (x.cont ? '+' : '') + x.tex));

test('a chain of equalities is written one relation per line, like on the board', () => {
  assert.deepEqual(relationLines(String.raw`x=-1, \log_e(- -1)=\log_e 1=0`).map((l) => [l.tex, l.cont]), [
    [String.raw`x=-1, \log_e(- -1)`, false],
    [String.raw`=\log_e 1`, true],
    ['=0', true],
  ]);
  assert.deepEqual(show(String.raw`y=|\ln x|=|\log_e x|`), [[String.raw`y=|\ln x|=|\log_e x|`], [String.raw`y=|\ln x|`, String.raw`+=|\log_e x|`]]);
});

test('clauses split at top-level commas, sentence stops and \\quad, never inside groups or numbers', () => {
  assert.deepEqual(clauses(String.raw`\log_2 8 = 3. \quad \log_2(-8)=`), [String.raw`\log_2 8 = 3.`, String.raw`\log_2(-8)=`]);
  assert.deepEqual(clauses('f(x, y) = 1,000'), ['f(x, y) = 1,000']);
  assert.deepEqual(clauses(String.raw`e = 2.71828 \dots`), [String.raw`e = 2.71828 \dots`]);
  assert.deepEqual(clauses(String.raw`\left(a, b\right), c`), [String.raw`\left(a, b\right),`, 'c']);
});

test('relations inside groups, \\left..\\right and environments are not split points', () => {
  assert.deepEqual(show(String.raw`\sum_{i=1}^n x_i = 5`), [[String.raw`\sum_{i=1}^n x_i = 5`]]);
  assert.equal(relationLines(String.raw`\begin{cases} a=1 \\ b=2 \end{cases} = c`).length, 1);
  assert.equal(relationLines(String.raw`\left(x = 1\right) = 2`).length, 1);
});

test('a trailing relation stays on its line', () => {
  assert.deepEqual(relationLines('a = b =').map((l) => l.tex), ['a = b =']);
});

test('the single line is always the first candidate and candidates are distinct', () => {
  const c = show(String.raw`x=-1, \log_e(- -1)=\log_e 1=0`);
  assert.deepEqual(c[0], [String.raw`x=-1, \log_e(- -1)=\log_e 1=0`]);
  assert.equal(new Set(c.map((x) => x.join('\n'))).size, c.length);
  assert.deepEqual(texLayouts('  '), []);
});

test('plain readings lose descriptions of the marks', () => {
  assert.equal(writtenText('16/2/15 with underline'), '16/2/15');
  assert.equal(writtenText('circled 1!'), '1!');
  assert.equal(writtenText('x (underlined)'), 'x');
  assert.equal(writtenText('A = 5'), 'A = 5');
});

test('vague kinds are typeset only when they carry LaTeX', () => {
  const p = (kind: string, text: string, latex: string) => ({
    id: 1, box: [0, 0, 10, 10] as [number, number, number, number], image: 'x.png', t_seen: 0, t_shown: 0,
    read: { kind, text, latex },
  });
  assert.equal(isTypesettable(p('other', 'small red and black marks', '')), false);
  assert.equal(isTypesettable(p('other', 'circled 1!', '$1!$')), true);
  assert.equal(isTypesettable(p('text', '100%', '')), true);
});
