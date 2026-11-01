// A small, safe maths-expression compiler for the 3D pop-out scenes (app/popout.py writes the expressions).
// Recursive descent straight to closures: no eval, no new Function, nothing but the whitelisted names.
//
// Grammar (lowest precedence first):
//   sum     := product (('+' | '-') product)*
//   product := unary (('*' | '/') unary | <implicit> power)*      2x, 3(x+1), x(x-1), (x+1)(x-1)
//   unary   := ('-' | '+') unary | power                            -x^2 = -(x^2)
//   power   := call ('^' exponent)?                                 right-associative: 2^3^2 = 2^9
//   exponent:= ('-' | '+') exponent | power                         2^-x
//   call    := FUNC '(' sum (',' sum)* ')' | FUNC power | atom      sin(x), ln x
//   atom    := NUMBER | x | y | t | pi | e | '(' sum ')'
// '**' is accepted for '^'. Names are case-insensitive. "log" is the natural log, like "ln".

export type Fn = (x: number, y: number, t: number) => number;

export class ExprError extends Error {}

const FUNCS1: Record<string, (a: number) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  exp: Math.exp,
  log: Math.log,
  ln: Math.log,
  log10: Math.log10,
  log2: Math.log2,
  sqrt: Math.sqrt,
  abs: Math.abs,
  floor: Math.floor,
  ceil: Math.ceil,
  sign: Math.sign,
};
const FUNCSN: Record<string, { min: number; max: number; f: (args: number[]) => number }> = {
  pow: { min: 2, max: 2, f: (a) => Math.pow(a[0], a[1]) },
  min: { min: 1, max: 16, f: (a) => Math.min(...a) },
  max: { min: 1, max: 16, f: (a) => Math.max(...a) },
};
const CONSTS: Record<string, number> = { pi: Math.PI, e: Math.E };

type Tok =
  | { k: 'num'; v: number; at: number }
  | { k: 'name'; v: string; at: number }
  | { k: 'op'; v: string; at: number }
  | { k: 'end'; at: number };

const MAX_LEN = 500;
const MAX_DEPTH = 60;

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  const re = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|(log10|log2|[A-Za-z_]+)|(\*\*|[-+*/^(),]))/iy;
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

/** Compile an expression in x, y and t to a function. Throws ExprError on anything outside the grammar.
 *  The function returns NaN where the maths is undefined (sqrt(-1), ln(-2), asin(3)). */
export function compile(src: string): Fn {
  if (typeof src !== 'string' || !src.trim()) throw new ExprError('Empty expression');
  if (src.length > MAX_LEN) throw new ExprError('Expression too long');
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

  function sum(): Fn {
    deeper();
    let left = product();
    while (isOp('+') || isOp('-')) {
      const op = (toks[i++] as { v: string }).v;
      const a = left;
      const b = product();
      left = op === '+' ? (x, y, t) => a(x, y, t) + b(x, y, t) : (x, y, t) => a(x, y, t) - b(x, y, t);
    }
    depth--;
    return left;
  }

  // An operand can follow directly (implicit multiplication) when the next token starts an atom.
  const startsOperand = () => {
    const t = peek();
    return t.k === 'num' || t.k === 'name' || (t.k === 'op' && t.v === '(');
  };

  function product(): Fn {
    let left = unary();
    for (;;) {
      if (isOp('*') || isOp('/')) {
        const op = (toks[i++] as { v: string }).v;
        const a = left;
        const b = unary();
        left = op === '*' ? (x, y, t) => a(x, y, t) * b(x, y, t) : (x, y, t) => a(x, y, t) / b(x, y, t);
      } else if (startsOperand()) {
        const a = left;
        const b = power();
        left = (x, y, t) => a(x, y, t) * b(x, y, t);
      } else return left;
    }
  }

  function unary(): Fn {
    if (isOp('-') || isOp('+')) {
      const neg = (toks[i++] as { v: string }).v === '-';
      deeper();
      const a = unary();
      depth--;
      return neg ? (x, y, t) => -a(x, y, t) : a;
    }
    return power();
  }

  function power(): Fn {
    const base = call();
    if (!isOp('^')) return base;
    i++;
    const ex = exponent();
    return (x, y, t) => Math.pow(base(x, y, t), ex(x, y, t));
  }

  function exponent(): Fn {
    if (isOp('-') || isOp('+')) {
      const neg = (toks[i++] as { v: string }).v === '-';
      deeper();
      const a = exponent();
      depth--;
      return neg ? (x, y, t) => -a(x, y, t) : a;
    }
    deeper();
    const p = power();
    depth--;
    return p;
  }

  function call(): Fn {
    const tok = peek();
    if (tok.k === 'name' && (tok.v in FUNCS1 || tok.v in FUNCSN)) {
      i++;
      const name = tok.v;
      let args: Fn[];
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
      if (name in FUNCS1) {
        if (args.length !== 1) throw new ExprError(`${name} takes one argument`);
        const f = FUNCS1[name];
        const a = args[0];
        return (x, y, t) => f(a(x, y, t));
      }
      const spec = FUNCSN[name];
      if (args.length < spec.min || args.length > spec.max)
        throw new ExprError(`Wrong number of arguments for ${name}`);
      if (name === 'pow') {
        const [a, b] = args;
        return (x, y, t) => Math.pow(a(x, y, t), b(x, y, t));
      }
      const f = spec.f;
      return (x, y, t) => f(args.map((a) => a(x, y, t)));
    }
    return atom();
  }

  function atom(): Fn {
    const tok = toks[i];
    if (tok.k === 'num') {
      i++;
      const v = tok.v;
      return () => v;
    }
    if (tok.k === 'name') {
      i++;
      if (tok.v === 'x') return (x) => x;
      if (tok.v === 'y') return (_x, y) => y;
      if (tok.v === 't') return (_x, _y, t) => t;
      if (tok.v in CONSTS) {
        const v = CONSTS[tok.v];
        return () => v;
      }
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
  return fn;
}

/** compile() that returns null instead of throwing. */
export function tryCompile(src: string): Fn | null {
  try {
    return compile(src);
  } catch {
    return null;
  }
}
