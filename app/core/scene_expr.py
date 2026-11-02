"""The 3D views' maths expressions: a parser that mirrors the player's own (web/src/player/expr.ts), so the server
accepts exactly what the player can draw, and evaluates them (numpy, many points at once) to check a scene's
geometry and frame it.

Nothing is ever run as code: an expression is parsed into a small tree of numbers, names and whitelisted
functions, and the tree is evaluated here, or written out again in a canonical form for the player.

Grammar (lowest precedence first), as in the player:
    sum      := product (('+' | '-') product)*
    product  := unary (('*' | '/') unary | <implicit> power)*      2x, 3(x+1), (x+1)(x-1)
    unary    := ('-' | '+') unary | power                          -x^2 = -(x^2)
    power    := call ('^' exponent)?                               right-associative: 2^3^2 = 2^9
    exponent := ('-' | '+') exponent | power                       2^-x
    call     := FUNC '(' sum (',' sum)* ')' | FUNC power | atom     sin(x), ln x
    atom     := NUMBER | NAME | '(' sum ')'
'**' is read as '^'. Names are case-insensitive. "log" is the natural log, like "ln". Constants: pi, e and deg
(= pi/180, so cos(beta*deg)). Other names are the variables a field allows (x; x, y; t; u, v) and the scene's
parameters. A name is a letter followed by letters, digits and '_' ("log10", "beta_1"; "x2" is a name,
not x times 2).
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np

FUNCS1 = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "exp": np.exp, "log": np.log, "ln": np.log,
    "log10": np.log10, "log2": np.log2, "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
    "sign": np.sign,
}
FUNCSN = {"pow": (2, 2), "min": (1, 16), "max": (1, 16)}
CONSTS = {"pi": math.pi, "e": math.e, "deg": math.pi / 180}
VARIABLES = ("x", "y", "t", "u", "v")
# Names a parameter may not take: every name the grammar already gives a meaning.
RESERVED = frozenset(FUNCS1) | frozenset(FUNCSN) | frozenset(CONSTS) | frozenset(VARIABLES)

MAX_LEN = 500
MAX_DEPTH = 60

_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z][A-Za-z0-9_]*)"
                    r"|(\*\*|[-+*/^(),]))")


class ExprError(ValueError):
    """An expression outside the grammar, or with a name the field does not allow."""


def _tokens(src: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m or m.end() == pos:
            if src[pos:].strip() == "":
                break
            raise ExprError(f"unexpected character {src[pos:].strip()[0]!r}")
        if m.group(1) is not None:
            v = float(m.group(1))
            if not math.isfinite(v):
                raise ExprError("number too big")
            out.append(("num", v))
        elif m.group(2) is not None:
            out.append(("name", m.group(2).lower()))
        else:
            out.append(("op", "^" if m.group(3) == "**" else m.group(3)))
        pos = m.end()
    out.append(("end", None))
    return out


def parse(src: Any, names: Iterable[str] = ()) -> tuple:
    """An expression -> its tree. `names`: the variables and parameters it may use (the constants are always
    allowed). ExprError for anything else."""
    if isinstance(src, bool) or not isinstance(src, (str, int, float)):
        raise ExprError("not an expression")
    if isinstance(src, (int, float)):
        try:
            value = float(src)  # an int too big for a float (JSON can carry one) raises OverflowError
        except OverflowError:
            raise ExprError("number too big") from None
        if not math.isfinite(value):
            raise ExprError("not a finite number")
        src = repr(value)
    if not src.strip():
        raise ExprError("empty expression")
    if len(src) > MAX_LEN:
        raise ExprError("expression too long")
    allowed = {n.lower() for n in names}
    toks = _tokens(src)
    i = 0
    depth = 0

    def peek():
        return toks[i]

    def is_op(v: str) -> bool:
        return toks[i][0] == "op" and toks[i][1] == v

    def take():
        nonlocal i
        t = toks[i]
        i += 1
        return t

    def deeper():
        nonlocal depth
        depth += 1
        if depth > MAX_DEPTH:
            raise ExprError("expression nested too deeply")

    def up():
        nonlocal depth
        depth -= 1

    def starts_operand() -> bool:
        k, v = peek()
        return k in ("num", "name") or (k == "op" and v == "(")

    def p_sum():
        deeper()
        left = p_product()
        while is_op("+") or is_op("-"):
            op = take()[1]
            left = ("bin", op, left, p_product())
        up()
        return left

    def p_product():
        left = p_unary()
        while True:
            if is_op("*") or is_op("/"):
                op = take()[1]
                left = ("bin", op, left, p_unary())
            elif starts_operand():
                left = ("bin", "*", left, p_power())
            else:
                return left

    def p_unary():
        if is_op("-") or is_op("+"):
            neg = take()[1] == "-"
            deeper()
            a = p_unary()
            up()
            return ("neg", a) if neg else a
        return p_power()

    def p_power():
        base = p_call()
        if not is_op("^"):
            return base
        take()
        return ("bin", "^", base, p_exponent())

    def p_exponent():
        if is_op("-") or is_op("+"):
            neg = take()[1] == "-"
            deeper()
            a = p_exponent()
            up()
            return ("neg", a) if neg else a
        deeper()
        p = p_power()
        up()
        return p

    def p_call():
        k, v = peek()
        if k == "name" and (v in FUNCS1 or v in FUNCSN):
            take()
            if is_op("("):
                take()
                args = [p_sum()]
                while is_op(","):
                    take()
                    args.append(p_sum())
                if not is_op(")"):
                    raise ExprError(f"expected ')' after the arguments of {v}")
                take()
            elif starts_operand():
                deeper()
                args = [p_power()]  # "ln x", "sin 2": binds like a power (sin x^2 = sin(x^2))
                up()
            else:
                raise ExprError(f"expected '(' after {v}")
            if v in FUNCS1:
                if len(args) != 1:
                    raise ExprError(f"{v} takes one argument")
            else:
                lo, hi = FUNCSN[v]
                if not lo <= len(args) <= hi:
                    raise ExprError(f"wrong number of arguments for {v}")
            return ("call", v, tuple(args))
        return p_atom()

    def p_atom():
        k, v = take()
        if k == "num":
            return ("num", v)
        if k == "name":
            if v in CONSTS:
                return ("const", v)
            if v in allowed:
                return ("name", v)
            raise ExprError(f"unknown name {v!r}")
        if k == "op" and v == "(":
            inner = p_sum()
            if not is_op(")"):
                raise ExprError("expected ')'")
            take()
            return inner
        if k == "end":
            raise ExprError("unexpected end of expression")
        raise ExprError(f"unexpected {v!r}")

    tree = p_sum()
    if peek()[0] != "end":
        raise ExprError(f"unexpected {peek()[1]!r}")
    return tree


def names_in(tree: tuple) -> set[str]:
    """The variable and parameter names a tree uses."""
    if tree[0] == "name":
        return {tree[1]}
    if tree[0] == "neg":
        return names_in(tree[1])
    if tree[0] == "bin":
        return names_in(tree[2]) | names_in(tree[3])
    if tree[0] == "call":
        out: set[str] = set()
        for a in tree[2]:
            out |= names_in(a)
        return out
    return set()


def evaluate(tree: tuple, env: dict[str, Any]) -> Any:
    """The tree's value, element-wise over numpy arrays in `env` (NaN where the maths is undefined, as in the
    player: sqrt(-1), ln(-2), asin(3))."""
    with np.errstate(all="ignore"):
        return _ev(tree, env)


def _power(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a^b as the player's Math.pow gives it: NaN for a NaN exponent and for (+-1)^(+-inf), where numpy gives 1;
    and always through numpy's array loop, since for single numbers it gives NaN for (-inf)^0.5, not inf."""
    shape = np.broadcast_shapes(np.shape(a), np.shape(b))
    a1, b1 = np.atleast_1d(a), np.atleast_1d(b)
    out = np.where(np.isnan(b1) | (np.isinf(b1) & (np.abs(a1) == 1)), np.nan, np.power(a1, b1))
    return out.reshape(shape)


def _ev(t: tuple, env: dict[str, Any]) -> Any:
    k = t[0]
    if k == "num":
        return t[1]
    if k == "const":
        return CONSTS[t[1]]
    if k == "name":
        return env[t[1]]
    if k == "neg":
        return -_ev(t[1], env)
    if k == "bin":
        a = np.asarray(_ev(t[2], env), dtype=float)
        b = np.asarray(_ev(t[3], env), dtype=float)
        op = t[1]
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return np.true_divide(a, b)
        return _power(a, b)
    if k == "call":
        args = [np.asarray(_ev(a, env), dtype=float) for a in t[2]]
        if t[1] in FUNCS1:
            return FUNCS1[t[1]](args[0])
        if t[1] == "pow":
            return _power(args[0], args[1])
        args = np.broadcast_arrays(*args)
        return (np.minimum if t[1] == "min" else np.maximum).reduce(np.stack(args), axis=0)
    raise ExprError(f"bad tree node {k!r}")


# ---------- the canonical form sent to the player ----------
def _num_text(v: float) -> str:
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


def to_text(tree: tuple) -> str:
    """The tree written out with explicit '*' and the parentheses it needs: the player reads it the same way
    whatever implicit products or shorthand ("ln x", "2x") the original used."""
    return _text(tree, 0)


def _text(t: tuple, ctx: int) -> str:
    # precedence: 1 sum, 2 product, 3 unary, 4 power, 5 atom
    k = t[0]
    if k == "num":
        return _num_text(t[1])
    if k in ("const", "name"):
        return t[1]
    if k == "call":
        return f"{t[1]}(" + ", ".join(_text(a, 0) for a in t[2]) + ")"
    if k == "neg":
        s = "-" + _text(t[1], 3)
        return f"({s})" if ctx > 3 else s
    op, a, b = t[1], t[2], t[3]
    if op in "+-":
        s = f"{_text(a, 1)} {op} {_text(b, 2)}"
        return f"({s})" if ctx > 1 else s
    if op in "*/":
        s = f"{_text(a, 2)}{op}{_text(b, 3)}"
        return f"({s})" if ctx > 2 else s
    base = _text(a, 5)
    if b[0] in ("num", "const", "name", "call"):
        ex = _text(b, 5)
    else:
        ex = f"({_text(b, 0)})"
    s = f"{base}^{ex}"
    return f"({s})" if ctx > 4 else s


def renamed(tree: tuple, mapping: dict[str, str]) -> tuple:
    """The tree with its names changed as `mapping` says (a param called like a variable, "t", renamed "p_t")."""
    k = tree[0]
    if k == "name":
        return ("name", mapping.get(tree[1], tree[1]))
    if k == "neg":
        return ("neg", renamed(tree[1], mapping))
    if k == "bin":
        return ("bin", tree[1], renamed(tree[2], mapping), renamed(tree[3], mapping))
    if k == "call":
        return ("call", tree[1], [renamed(a, mapping) for a in tree[2]])
    return tree


def canonical(src: Any, names: Iterable[str] = (), rename: dict[str, str] | None = None) -> str:
    """parse() then to_text(): the expression as the scene stores it. ExprError when it is not allowed. `rename`:
    names it may use under another name, written out with the new one."""
    if not rename:
        return to_text(parse(src, names))
    return to_text(renamed(parse(src, set(names) | set(rename)), rename))
