// A small, safe maths-expression compiler for the 3D views (app/popout.py writes the expressions; the server checks
// them against the same whitelist). Recursive descent straight to closures: no eval, no new Function, nothing but
// the whitelisted names.
//
// Grammar (lowest precedence first):
//   sum     := product (('+' | '-') product)*
//   product := unary (('*' | '/') unary | <implicit> power)*      2x, 3(x+1), x(x-1), (x+1)(x-1), 2 beta
//   unary   := ('-' | '+') unary | power                            -x^2 = -(x^2)
//   power   := call ('^' exponent)?                                 right-associative: 2^3^2 = 2^9
//   exponent:= ('-' | '+') exponent | power                         2^-x
//   call    := FUNC '(' sum (',' sum)* ')' | FUNC power | atom      sin(x), ln x
//   atom    := NUMBER | VARIABLE | CONSTANT | '(' sum ')'
// '**' is accepted for '^'. Names are case-insensitive and are whole words: letters, digits and '_' after a first
// letter ("x2" is a name, not x times 2; "2x" is 2 times x). "log" is the natural log, like "ln".
// Constants: pi, e and deg (= pi/180, so cos(beta*deg) takes beta in degrees).

/** An expression in x, y and t (the first 3D pop-out's variables). */
export type Fn = (x: number, y: number, t: number) => number;
/** An expression over named variables: vals[i] is the value of names[i]. */
export type VarsFn = (vals: ArrayLike<number>) => number;

export interface Compiled {
  fn: VarsFn;
  /** The variable names the expression reads (a subset of the names it was compiled with). */
  uses: Set<string>;
}

export class ExprError extends Error {}

// Maps, not plain objects: "constructor" or "__proto__" must never look like a known name.
const FUNCS1 = new Map<string, (a: number) => number>([
  ['sin', Math.sin],
  ['cos', Math.cos],
  ['tan', Math.tan],
  ['asin', Math.asin],
  ['acos', Math.acos],
  ['atan', Math.atan],
  ['sinh', Math.sinh],
  ['cosh', Math.cosh],
  ['tanh', Math.tanh],
  ['exp', Math.exp],
  ['log', Math.log],
  ['ln', Math.log],
  ['log10', Math.log10],
  ['log2', Math.log2],
  ['sqrt', Math.sqrt],
  ['abs', Math.abs],
  ['floor', Math.floor],
  ['ceil', Math.ceil],
  ['sign', Math.sign],
]);
const FUNCSN = new Map<string, { min: number; max: number; f: (args: number[]) => number }>([
  ['pow', { min: 2, max: 2, f: (a) => Math.pow(a[0], a[1]) }],
  ['min', { min: 1, max: 16, f: (a) => Math.min(...a) }],
  ['max', { min: 1, max: 16, f: (a) => Math.max(...a) }],
]);
const CONSTS = new Map<string, number>([
  ['pi', Math.PI],
  ['e', Math.E],
  ['deg', Math.PI / 180],
]);

/** Every name the grammar itself uses (functions and constants): never a variable or parameter name. */
export const GRAMMAR_NAMES: ReadonlySet<string> = new Set([...FUNCS1.keys(), ...FUNCSN.keys(), ...CONSTS.keys()]);

type Tok =
  | { k: 'num'; v: number; at: number }
  | { k: 'name'; v: string; at: number }
  | { k: 'op'; v: string; at: number }
  | { k: 'end'; at: number };

const MAX_LEN = 500;
const MAX_DEPTH = 60;

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  const re = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z][A-Za-z0-9_]*)|(\*\*|[-+*/^(),]))/y;
  let pos = 0;
  while (pos < src.length) {
    re.lastIndex = pos;
    const m = re.exec(src);
    if (!m || m.index !== pos) {
      if (/^\s*$/.test(src.slice(pos))) break;
      const at = pos + (src.slice(pos).length - src.slice(pos).trimStart().length);
      throw new ExprError(`Unexpected character "${src[at]}" at ${at + 1}`);
    }
    const at = m.index + m[0].length - (m[1] ?? m[2] ?? m[3]).length;
    if (m[1] !== undefined) out.push({ k: 'num', v: parseFloat(m[1]), at });
    else if (m[2] !== undefined) out.push({ k: 'name', v: m[2].toLowerCase(), at });
    else out.push({ k: 'op', v: m[3] === '**' ? '^' : m[3], at });
    pos = re.lastIndex;
  }
  out.push({ k: 'end', at: src.length });
  return out;
}

/**
 * Compile an expression over the given variable names (case-insensitive) to a function of their values, in the
 * same order. Throws ExprError on anything outside the grammar, on unknown names, and on a variable named like a
 * function or constant. The function returns NaN where the maths is undefined (sqrt(-1), ln(-2), asin(3)).
 */
export function compileVars(src: string, names: readonly string[]): Compiled {
  if (typeof src !== 'string' || !src.trim()) throw new ExprError('Empty expression');
  if (src.length > MAX_LEN) throw new ExprError('Expression too long');
  const index = new Map<string, number>();
  names.forEach((n, i) => {
    const name = String(n).toLowerCase();
    if (GRAMMAR_NAMES.has(name)) throw new ExprError(`"${name}" cannot be a variable`);
    if (!index.has(name)) index.set(name, i);
  });
  const uses = new Set<string>();
  const toks = tokenize(src);
  let i = 0;
  let depth = 0;
  const peek = () => toks[i];
  const isOp = (v: string) => {
    const t = toks[i];
    return t.k === 'op' && t.v === v;
  };
  const expect = (v: string) => {
    if (!isOp(v)) throw new ExprError(`Expected "${v}" at ${peek().at + 1}`);
    i++;
  };
  const deeper = () => {
    if (++depth > MAX_DEPTH) throw new ExprError('Expression nested too deeply');
  };

  function sum(): VarsFn {
    deeper();
    let left = product();
    while (isOp('+') || isOp('-')) {
      const op = (toks[i++] as { v: string }).v;
      const a = left;
      const b = product();
      left = op === '+' ? (v) => a(v) + b(v) : (v) => a(v) - b(v);
    }
    depth--;
    return left;
  }

  // An operand can follow directly (implicit multiplication) when the next token starts an atom.
  const startsOperand = () => {
    const t = peek();
    return t.k === 'num' || t.k === 'name' || (t.k === 'op' && t.v === '(');
  };

  function product(): VarsFn {
    let left = unary();
    for (;;) {
      if (isOp('*') || isOp('/')) {
        const op = (toks[i++] as { v: string }).v;
        const a = left;
        const b = unary();
        left = op === '*' ? (v) => a(v) * b(v) : (v) => a(v) / b(v);
      } else if (startsOperand()) {
        const a = left;
        const b = power();
        left = (v) => a(v) * b(v);
      } else return left;
    }
  }

  function unary(): VarsFn {
    if (isOp('-') || isOp('+')) {
      const neg = (toks[i++] as { v: string }).v === '-';
      deeper();
      const a = unary();
      depth--;
      return neg ? (v) => -a(v) : a;
    }
    return power();
  }

  function power(): VarsFn {
    const base = call();
    if (!isOp('^')) return base;
    i++;
    const ex = exponent();
    return (v) => Math.pow(base(v), ex(v));
  }

  function exponent(): VarsFn {
    if (isOp('-') || isOp('+')) {
      const neg = (toks[i++] as { v: string }).v === '-';
      deeper();
      const a = exponent();
      depth--;
      return neg ? (v) => -a(v) : a;
    }
    deeper();
    const p = power();
    depth--;
    return p;
  }

  function call(): VarsFn {
    const tok = peek();
    if (tok.k === 'name' && (FUNCS1.has(tok.v) || FUNCSN.has(tok.v))) {
      i++;
      const name = tok.v;
      let args: VarsFn[];
      if (isOp('(')) {
        i++;
        args = [sum()];
        while (isOp(',')) {
          i++;
          args.push(sum());
        }
        expect(')');
      } else if (startsOperand()) {
        deeper();
        args = [power()]; // "ln x", "sin 2" (binds like a power: sin x^2 = sin(x^2))
        depth--;
      } else throw new ExprError(`Expected "(" after ${name} at ${peek().at + 1}`);
      const f1 = FUNCS1.get(name);
      if (f1) {
        if (args.length !== 1) throw new ExprError(`${name} takes one argument`);
        const a = args[0];
        return (v) => f1(a(v));
      }
      const spec = FUNCSN.get(name)!;
      if (args.length < spec.min || args.length > spec.max)
        throw new ExprError(`Wrong number of arguments for ${name}`);
      if (name === 'pow') {
        const [a, b] = args;
        return (v) => Math.pow(a(v), b(v));
      }
      const f = spec.f;
      return (v) => f(args.map((a) => a(v)));
    }
    return atom();
  }

  function atom(): VarsFn {
    const tok = toks[i];
    if (tok.k === 'num') {
      i++;
      const c = tok.v;
      return () => c;
    }
    if (tok.k === 'name') {
      i++;
      const at = index.get(tok.v);
      if (at !== undefined) {
        uses.add(tok.v);
        return (v) => v[at];
      }
      const c = CONSTS.get(tok.v);
      if (c !== undefined) return () => c;
      throw new ExprError(`Unknown name "${tok.v}" at ${tok.at + 1}`);
    }
    if (tok.k === 'op' && tok.v === '(') {
      i++;
      const inner = sum();
      expect(')');
      return inner;
    }
    if (tok.k === 'end') throw new ExprError('Unexpected end of expression');
    throw new ExprError(`Unexpected "${tok.v}" at ${tok.at + 1}`);
  }

  const fn = sum();
  if (peek().k !== 'end') {
    const t = peek() as Tok & { v?: unknown };
    throw new ExprError(`Unexpected "${String(t.v)}" at ${t.at + 1}`);
  }
  // Infinity and -Infinity are kept (they mark asymptotes); undefined maths is NaN, as Math gives it.
  return { fn, uses };
}

/** compileVars() that returns null instead of throwing. */
export function tryCompileVars(src: string, names: readonly string[]): Compiled | null {
  try {
    return compileVars(src, names);
  } catch {
    return null;
  }
}

const XYT = ['x', 'y', 't'] as const;

/** Compile an expression in x, y and t to a function. Throws ExprError on anything outside the grammar. */
export function compile(src: string): Fn {
  const { fn } = compileVars(src, XYT);
  const vals = new Float64Array(3); // evaluation is synchronous and never re-entrant, so one buffer will do
  return (x, y, t) => {
    vals[0] = x;
    vals[1] = y;
    vals[2] = t;
    return fn(vals);
  };
}

/** compile() that returns null instead of throwing. */
export function tryCompile(src: string): Fn | null {
  try {
    return compile(src);
  } catch {
    return null;
  }
}
