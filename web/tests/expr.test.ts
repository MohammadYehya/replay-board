// The 3D view's expression compiler: grammar, precedence, functions, and what it must refuse.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { compile, compileVars, tryCompile, tryCompileVars, ExprError, GRAMMAR_NAMES } from '../src/player/expr.ts';

const ev = (s: string, x = 0, y = 0, t = 0) => compile(s)(x, y, t);
const near = (a: number, b: number, eps = 1e-12) =>
  assert.ok(Math.abs(a - b) <= eps * Math.max(1, Math.abs(b)), `${a} != ${b}`);

test('numbers, including decimals and exponents', () => {
  assert.equal(ev('42'), 42);
  assert.equal(ev('3.5'), 3.5);
  assert.equal(ev('.25'), 0.25);
  assert.equal(ev('7.'), 7);
  near(ev('2.5e-3'), 0.0025);
  near(ev('1E3'), 1000);
  near(ev('4e+2'), 400);
});

test('precedence and associativity of + - * /', () => {
  assert.equal(ev('2 + 3 * 4'), 14);
  assert.equal(ev('(2 + 3) * 4'), 20);
  assert.equal(ev('10 - 4 - 3'), 3);
  assert.equal(ev('24 / 4 / 2'), 3);
  assert.equal(ev('2 * 3 + 4 * 5'), 26);
  assert.equal(ev('1 + 2 * 3 ^ 2'), 19);
});

test('^ is right-associative and binds tighter than unary minus', () => {
  assert.equal(ev('2 ^ 3 ^ 2'), 512);
  assert.equal(ev('(2 ^ 3) ^ 2'), 64);
  assert.equal(ev('-2 ^ 2'), -4);
  assert.equal(ev('(-2) ^ 2'), 4);
  assert.equal(ev('2 ^ -1'), 0.5);
  assert.equal(ev('2 ^ -1 ^ 2'), 0.5); // 2^(-(1^2))
  assert.equal(ev('2 ** 3'), 8);
  assert.equal(ev('-x^2', 3), -9);
});

test('unary minus and plus', () => {
  assert.equal(ev('-3'), -3);
  assert.equal(ev('--3'), 3);
  assert.equal(ev('+3'), 3);
  assert.equal(ev('4 - -2'), 6);
  assert.equal(ev('-(1 + 2) * 2'), -6);
  assert.equal(ev('2 * -x', 5), -10);
});

test('variables x, y and t', () => {
  assert.equal(ev('x', 2), 2);
  assert.equal(ev('y', 0, 3), 3);
  assert.equal(ev('t', 0, 0, 4), 4);
  assert.equal(ev('x^2 + y^2', 3, 4), 25);
  assert.equal(ev('X + Y', 1, 2), 3); // names are case-insensitive
});

test('functions', () => {
  near(ev('sin(pi / 2)'), 1);
  near(ev('cos(0)'), 1);
  near(ev('tan(pi / 4)'), 1);
  near(ev('asin(1)'), Math.PI / 2);
  near(ev('acos(1)'), 0);
  near(ev('atan(1)'), Math.PI / 4);
  near(ev('sinh(1)'), Math.sinh(1));
  near(ev('cosh(1)'), Math.cosh(1));
  near(ev('tanh(1)'), Math.tanh(1));
  near(ev('exp(1)'), Math.E);
  near(ev('ln(e)'), 1);
  near(ev('log(e^2)'), 2); // log is the natural log
  near(ev('log10(1000)'), 3);
  near(ev('log2(8)'), 3);
  near(ev('sqrt(16)'), 4);
  assert.equal(ev('abs(-3)'), 3);
  assert.equal(ev('pow(2, 10)'), 1024);
  assert.equal(ev('min(3, 1, 2)'), 1);
  assert.equal(ev('max(3, 1, 2)'), 3);
  assert.equal(ev('floor(2.7)'), 2);
  assert.equal(ev('ceil(2.2)'), 3);
  assert.equal(ev('sign(-5)'), -1);
  near(ev('abs(ln(x))', Math.E), 1);
  near(ev('SIN(PI)'), 0, 1e-12);
  near(ev('LOG10(100)'), 2);
});

test('functions without parentheses and implicit multiplication', () => {
  near(ev('ln x', Math.E), 1);
  near(ev('sin x^2', 2), Math.sin(4));
  assert.equal(ev('2x', 3), 6);
  assert.equal(ev('2x^2', 3), 18);
  assert.equal(ev('3(x + 1)', 1), 6);
  assert.equal(ev('(x + 1)(x - 1)', 3), 8);
  assert.equal(ev('x(x - 1)', 3), 6);
  near(ev('2pi'), 2 * Math.PI);
  near(ev('2e'), 2 * Math.E); // "2e" is 2 times e, not a malformed exponent
  near(ev('sin x cos x', 1), Math.sin(1) * Math.cos(1));
});

test('constants', () => {
  near(ev('pi'), Math.PI);
  near(ev('e'), Math.E);
  near(ev('E'), Math.E);
});

test('domain errors give NaN, poles give Infinity', () => {
  assert.ok(Number.isNaN(ev('sqrt(-1)')));
  assert.ok(Number.isNaN(ev('ln(-2)')));
  assert.ok(Number.isNaN(ev('asin(3)')));
  assert.ok(Number.isNaN(ev('(-8) ^ (1/3)')));
  assert.equal(ev('ln(0)'), -Infinity);
  assert.equal(ev('1 / x', 0), Infinity);
  assert.equal(ev('abs(ln(x))', 0), Infinity);
});

test('rejects unknown names, characters and malformed input', () => {
  const bad = [
    '', '   ', 'foo(x)', 'z', 'alert(1)', 'x.constructor', 'window', 'x; y', 'x = 1', 'x & 1', 'x > 1', '"x"',
    '`x`', '[1]', '{x}', 'x!', '2 +', '(x', 'x)', 'sin()', 'pow(2)', 'sin(1, 2)', 'xy', 'sinx', 'constructor',
    '__proto__', 'Math.sin(x)', 'x # 1', '1 ++', '* 2', 'sin', 'toString(x)', 'hasOwnProperty(x)',
  ];
  for (const s of bad) {
    assert.throws(() => compile(s), ExprError, `should reject ${JSON.stringify(s)}`);
    assert.equal(tryCompile(s), null);
  }
  assert.throws(() => compile('x+'.repeat(300) + 'x'), ExprError); // too long
  assert.throws(() => compile('('.repeat(100) + 'x' + ')'.repeat(100)), ExprError); // too deep
});

test('the cached pop-out expression compiles', () => {
  const f = compile('abs(ln(x))');
  near(f(1, 0, 0), 0);
  near(f(1 / Math.E, 0, 0), 1);
});

// ---- named variables (format-2 scenes: field variables plus the scene's parameters) ----
const evv = (s: string, names: string[], vals: number[]) => compileVars(s, names).fn(vals);

test('named variables: field variables and parameters, by position', () => {
  assert.equal(evv('a + 2b', ['a', 'b'], [1, 3]), 7);
  assert.equal(evv('c_len * 2', ['c_len'], [1.5]), 3);
  assert.equal(evv('theta2 + 1', ['theta2'], [4]), 5);
  near(evv('cos(beta*deg)', ['beta'], [60]), 0.5);
  near(evv('r cos(v)', ['u', 'v', 'r'], [0, Math.PI, 2]), -2);
  assert.equal(evv('Beta', ['beta'], [3]), 3); // names are case-insensitive
  near(evv('deg', [], []), Math.PI / 180);
  near(evv('180 deg', [], []), Math.PI);
  assert.equal(evv('2', [], []), 2);
});

test('which variables an expression reads', () => {
  assert.deepEqual([...compileVars('x^2 + k*x', ['x', 'k', 'm']).uses].sort(), ['k', 'x']);
  assert.deepEqual([...compileVars('3 + pi', ['x', 'k']).uses], []);
  assert.deepEqual([...compileVars('sin(t) * amp', ['t', 'amp']).uses].sort(), ['amp', 't']);
});

test('names are whole words: no hidden multiplication inside a name', () => {
  assert.equal(tryCompileVars('x2', ['x']), null); // "x2" is a name, not x times 2
  assert.equal(evv('2x', ['x'], [3]), 6);
  assert.equal(evv('2 beta', ['beta'], [3]), 6);
  assert.equal(tryCompileVars('betax', ['beta', 'x']), null);
  assert.equal(evv('beta x', ['beta', 'x'], [2, 5]), 10);
  assert.equal(tryCompileVars('e2', []), null);
  near(evv('2e', [], []), 2 * Math.E);
  near(evv('2e3', [], []), 2000);
});

test('only the given variables: field variables do not leak between fields', () => {
  assert.equal(tryCompileVars('x + y', ['x']), null); // a graph's expression has no y
  assert.equal(tryCompileVars('t', ['u', 'v']), null); // a psurface has no t
  assert.equal(tryCompileVars('k', []), null); // a parameter the scene does not have
  assert.ok(tryCompileVars('u v', ['u', 'v']));
});

test('a variable cannot be named like a function or constant', () => {
  for (const n of ['sin', 'pi', 'e', 'deg', 'log10', 'max']) {
    assert.ok(GRAMMAR_NAMES.has(n));
    assert.throws(() => compileVars('1', [n]), ExprError);
  }
});

test('object-prototype names are never functions, constants or variables', () => {
  for (const s of ['constructor(1)', 'toString(1)', 'valueOf(1)', 'hasOwnProperty(1)', '__proto__', 'prototype']) {
    assert.equal(tryCompileVars(s, ['x']), null, s);
  }
});
