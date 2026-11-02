"""The lecture deck (export/deck.json, format 2) as a 16:9 PowerPoint.

build_pptx(job_dir, deck) -> bytes. Code only: Muse Spark wrote the deck; this lays it out and draws it.

- Theme: white slides, slate text, one blue accent, Calibri Light titles and Calibri body text. Every slide title
  is the layout's real title placeholder (the outline, navigation and screen readers rely on it).
- Layout: text is measured with the real fonts (PIL), every block's height is computed for a width and a body
  size, and the body size shrinks from 24 pt to 14 pt until the slide's blocks fit: one column, or two when there is
  a figure (or a table that does not fit under the text). Nothing overflows the slide or overlaps.
- Maths: display formulas are pictures typeset by MathJax (app/core/texrender.py), with their LaTeX as alt text;
  without node they are Unicode text in Cambria Math. Inline $...$ maths becomes Unicode runs with real
  superscripts and subscripts.
- Speaker notes: what a presenter would say, then what was said in the lecture over the slide's time range.
"""
from __future__ import annotations

import functools
import io
import math
import os
import re
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .ask import clock
from .core import texrender

RULES_VERSION = "4"  # change it when the PowerPoint's look or layout rules change, so cached files are rebuilt


def rules_version() -> str:
    """RULES_VERSION, marked when formulas cannot be typeset now: a PowerPoint cached with its formulas as text is
    made again once node and its packages are installed, or once a run that failed (a crash, a timeout) works."""
    ok = texrender.available() and not texrender.last_run_failed()
    return RULES_VERSION if ok else RULES_VERSION + "-text"

# ---------- theme ----------
SLATE = "1F2A37"
ACCENT = "2563EB"
MUTED = "6B7280"
WHITE = "FFFFFF"
ACCENT_SOFT = "EEF3FE"   # callout fill
PANEL = "F4F6F9"         # equation cards
BAND = "F3F6FC"          # table banding
RULE = "E3E7ED"          # dividers, figure frames
NOTE_FILL = "F3F4F6"
NOTE_EDGE = "9CA3AF"
ON_ACCENT = "D6E4FF"     # quieter text on the accent background

TITLE_FONT = "Calibri Light"
BODY_FONT = "Calibri"
INLINE_MATH_FONT = "Cambria"
MATH_FONT = "Cambria Math"

# ---------- geometry (points: the slide is 960 x 540 pt = 13.333 x 7.5 in) ----------
EMU_PER_PT = 12700
SW, SH = 960.0, 540.0
MX = 50.0                    # side margins
CW = SW - 2 * MX             # content width
EYEBROW_Y, EYEBROW_H = 24.0, 17.0
TITLE_Y, TITLE_H = 41.0, 58.0
BAR_Y, BAR_H, BAR_W = 106.0, 3.5, 48.0
BODY_Y, BODY_B = 128.0, 488.0
BODY_H = BODY_B - BODY_Y
FOOT_Y, FOOT_H = 507.0, 16.0
COL_GAP = 36.0

BODY_SIZES = list(range(24, 13, -1))    # 24 pt down to 14 pt
LAST_RESORT_SIZES = [13, 12, 11, 10]    # only when a single block cannot fit at 14 pt
MIN_BODY = 14
FOOTER_SIZE = 11
LINE = 1.2                   # line spacing of body text (exact, so the measured height is the drawn height)
EQ_K = 1.25                  # a display formula's font size, relative to the body size
STEP_EQ_K = 1.05
READABLE_SHRINK = 0.72       # a formula scaled below this is a last resort
MIN_FIGURE_SHARE = 0.2       # a figure gets at least this share of the body area
TABLE_MIN_SHARE = 0.75       # a table is at least this share of its column's width, centred in it
# A table this tall and this narrow sits beside a slide's other blocks (no figure on the slide); any other table is
# stacked with them. The player's preview (web/src/slides/deck.ts, contentLayout) follows the same rule.
SIDE_TABLE_ROWS = 4
SIDE_TABLE_COLS = 3


def side_table(blocks: list[dict]) -> dict | None:
    """The table that goes in a column of its own beside the other blocks: a slide without a figure whose table has
    at least SIDE_TABLE_ROWS rows and at most SIDE_TABLE_COLS columns, next to other blocks."""
    if len(blocks) < 2 or any(b.get("type") == "figure" for b in blocks):
        return None
    t = next((b for b in blocks if b.get("type") == "table"), None)
    if t is None:
        return None
    cols = len(t.get("header") or []) or max((len(r) for r in t.get("rows") or [] if isinstance(r, list)), default=0)
    rows = len([r for r in t.get("rows") or [] if isinstance(r, list)])
    return t if rows >= SIDE_TABLE_ROWS and cols <= SIDE_TABLE_COLS else None


def E(pt: float) -> int:
    return int(round(pt * EMU_PER_PT))


# ---------- small helpers ----------
def _safe(s: Any) -> str:
    from .export import safe_text  # late: export uses this module
    return safe_text(s)


def _flat(s: Any) -> str:
    return " ".join(_safe(s).split())


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def span(t0: Any, t1: Any) -> str:
    """'2:56–6:29', or '' without both times."""
    if _num(t0) and _num(t1):
        return f"{clock(t0)}–{clock(max(t0, t1))}"
    return ""


# ---------- fonts and measuring ----------
_FONT_FILES: dict[tuple[str, bool, bool], list[tuple[str, int]]] = {
    ("Calibri", False, False): [("calibri.ttf", 0), ("Carlito-Regular.ttf", 0)],
    ("Calibri", True, False): [("calibrib.ttf", 0), ("Carlito-Bold.ttf", 0)],
    ("Calibri", False, True): [("calibrii.ttf", 0), ("Carlito-Italic.ttf", 0)],
    ("Calibri", True, True): [("calibriz.ttf", 0), ("Carlito-BoldItalic.ttf", 0)],
    ("Calibri Light", False, False): [("calibril.ttf", 0), ("calibri.ttf", 0), ("Carlito-Regular.ttf", 0)],
    ("Calibri Light", True, False): [("calibrib.ttf", 0), ("Carlito-Bold.ttf", 0)],
    ("Cambria", False, False): [("cambria.ttc", 0), ("Caladea-Regular.ttf", 0)],
    ("Cambria", True, False): [("cambriab.ttf", 0), ("Caladea-Bold.ttf", 0)],
    ("Cambria", False, True): [("cambriai.ttf", 0), ("Caladea-Italic.ttf", 0)],
    ("Cambria", True, True): [("cambriaz.ttf", 0), ("Caladea-BoldItalic.ttf", 0)],
    ("Cambria Math", False, False): [("cambria.ttc", 1)],
}
AVG_EM = 0.56  # a generous average character width when a font is not installed
_measure_lock = threading.Lock()


def _font_dirs() -> list[Path]:
    dirs = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    home = Path.home()
    dirs += [Path("/Library/Fonts"), home / "Library" / "Fonts", Path("/System/Library/Fonts"),
             Path("/System/Library/Fonts/Supplemental"),
             Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts"),
             Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".fonts", home / ".local/share/fonts"]
    return [d for d in dirs if d.is_dir()]


@functools.lru_cache(maxsize=None)
def _font_index() -> dict[str, Path]:
    """Lower-case file name -> path, for the fonts this module may measure with."""
    wanted = {name.lower() for files in _FONT_FILES.values() for name, _ in files}
    found: dict[str, Path] = {}
    for d in _font_dirs():
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            if n.lower() in wanted and n.lower() not in found:
                found[n.lower()] = d / n
    for d in _font_dirs():  # Linux keeps fonts in sub-folders
        if str(d).startswith(("/usr", str(Path.home()))):
            for p in d.rglob("*"):
                if p.name.lower() in wanted and p.name.lower() not in found:
                    found[p.name.lower()] = p
    return found


@functools.lru_cache(maxsize=None)
def _pil_font(family: str, bold: bool, italic: bool):
    from PIL import ImageFont
    for name, index in _FONT_FILES.get((family, bold, italic)) or _FONT_FILES.get((family, bold, False)) or []:
        path = _font_index().get(name.lower())
        if path is None:
            continue
        try:
            return ImageFont.truetype(str(path), 1000, index=index)
        except OSError:
            continue
    if (family, bold, italic) != (family, False, False) and (family, False, False) in _FONT_FILES:
        return _pil_font(family, False, False)
    return None


@functools.lru_cache(maxsize=200_000)
def text_em(text: str, family: str, bold: bool = False, italic: bool = False) -> float:
    """The advance width of this text, in em of its font size."""
    if not text:
        return 0.0
    font = _pil_font(family, bold, italic)
    if font is None:
        return len(text) * AVG_EM * (1.06 if bold else 1.0)
    try:
        with _measure_lock:
            return float(font.getlength(text)) / 1000.0
    except Exception:
        return len(text) * AVG_EM


# ---------- rich text ----------
@dataclass(frozen=True)
class Run:
    text: str
    script: int = 0       # 1 superscript, -1 subscript
    italic: bool = False
    bold: bool = False
    math: bool = False    # set in the maths font
    op: str = ""          # spacing class in maths: "fn" (sin), "rel" (=, →), "bin" (+), "punct" (,), "open"/"close" (a/b)


@dataclass
class Para:
    runs: list[Run]
    size: float
    color: str = SLATE
    font: str = BODY_FONT
    bold: bool = False
    align: str = "l"      # l, c, r
    before: float = 0.0   # space before (pt)
    bullet: str = ""      # bullet character ("" for none)
    bullet_color: str = ACCENT
    marl: float = 0.0     # left margin of the text (pt)
    hang: float = 0.0     # how far left of the text the bullet hangs (pt)
    line: float = LINE
    math_font: str = INLINE_MATH_FONT

    @property
    def line_h(self) -> float:
        return round(self.size * self.line, 1)


SCRIPT_K = 0.67  # superscripts and subscripts are drawn about this much smaller
# Inline maths without letters (numbers, operators, arrows) is set in the body font: live, "365 × 24 × 60" and
# "2.71828" switched to a serif font mid-sentence. Letters (variables, Greek) stay italic Cambria.
BODY_FONT_MATH = re.compile(r"[\d\s.,:;!?'\"()\[\]{}/|+\-=<>%°′″·×÷−±≈≠≤≥→←↔⇒∞…   ]*")


def _run_font(run: Run, para: Para) -> tuple[str, float, bool, bool]:
    family = para.math_font if run.math and not BODY_FONT_MATH.fullmatch(run.text) else para.font
    size = para.size * (SCRIPT_K if run.script else 1.0)
    return family, size, bool(run.bold or para.bold), run.italic


def _words(runs: list[Run], para: Para) -> list[tuple[float, float]]:
    """(width, space before it) of every word, in points."""
    words: list[tuple[float, float]] = []
    cur, has, space = 0.0, False, 0.0
    for r in runs:
        family, size, bold, italic = _run_font(r, para)
        for part in re.split(r"([ \u00a0\u2009]+)", r.text):
            if not part:
                continue
            if part[0] in " \u00a0\u2009" and not part.strip(" \u00a0\u2009"):
                if has:
                    words.append((cur, space0))
                    cur, has, space = 0.0, False, 0.0
                space += text_em(part, family, bold, italic) * size
            else:
                if not has:
                    space0, has = space, True
                cur += text_em(part, family, bold, italic) * size
    if has:
        words.append((cur, space0))
    return words


def _wrap(para: Para, width: float) -> tuple[int, float]:
    """(lines, width of the last line) of the paragraph wrapped in this width (greedy, like PowerPoint; long words
    are broken)."""
    width = max(1.0, width * 0.985 - 1.0)
    lines, x = 1, 0.0
    for w, sp in _words(para.runs, para):
        if x > 0 and x + sp + w <= width:
            x += sp + w
            continue
        if x > 0:
            lines += 1
        if w > width:
            extra = math.ceil(w / width) - 1
            lines += extra
            x = w - extra * width
        else:
            x = w
    return lines, x


def line_count(para: Para, width: float) -> int:
    """How many lines the paragraph wraps to in this width."""
    return _wrap(para, width)[0]


WIDOW = 0.3  # a title's last line at least this share of its box: "…leads to" / "e" alone looked broken (test-full)


def natural_width(para: Para) -> float:
    words = _words(para.runs, para)
    return sum(w + (sp if i else 0.0) for i, (w, sp) in enumerate(words))


def para_height(para: Para, width: float) -> float:
    return para.before + line_count(para, width - para.marl) * para.line_h


def paras_height(paras: list[Para], width: float) -> float:
    return sum(para_height(p, width) for p in paras)


# ---------- inline maths: LaTeX -> Unicode runs ----------
GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "omicron": "ο", "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς",
    "tau": "τ", "upsilon": "υ", "phi": "ϕ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMBOLS = {
    "infty": "∞", "to": "→", "rightarrow": "→", "Rightarrow": "⇒", "longrightarrow": "⟶", "leftarrow": "←",
    "gets": "←", "Leftarrow": "⇐", "leftrightarrow": "↔", "Leftrightarrow": "⇔", "iff": "⇔", "implies": "⇒",
    "mapsto": "↦", "uparrow": "↑", "downarrow": "↓", "approx": "≈", "simeq": "≃", "cong": "≅", "equiv": "≡",
    "sim": "∼", "propto": "∝", "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "ll": "≪", "gg": "≫", "in": "∈", "notin": "∉", "ni": "∋",
    "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇", "cup": "∪", "cap": "∩", "emptyset": "∅",
    "varnothing": "∅", "forall": "∀", "exists": "∃", "neg": "¬", "lnot": "¬", "land": "∧", "lor": "∨",
    "wedge": "∧", "vee": "∨", "partial": "∂", "nabla": "∇", "sum": "∑", "prod": "∏", "int": "∫", "iint": "∬",
    "oint": "∮", "circ": "∘", "degree": "°", "angle": "∠", "perp": "⊥", "parallel": "∥", "ldots": "…",
    "dots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱", "prime": "′", "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ",
    "Im": "ℑ", "aleph": "ℵ", "star": "⋆", "ast": "∗", "bullet": "•", "oplus": "⊕", "otimes": "⊗",
    "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "vert": "|",
    "mid": "|", "Vert": "‖", "triangle": "△", "square": "□", "therefore": "∴", "because": "∵", "dagger": "†",
    "setminus": "∖", "lbrace": "{", "rbrace": "}", "colon": ":", "quad": "\u2003", "qquad": "\u2003\u2003",
    "checkmark": "✓", "pounds": "£", "S": "§", "%": "%",
}
RELATIONS = {"approx", "simeq", "cong", "equiv", "sim", "propto", "le", "leq", "ge", "geq", "ne", "neq", "ll", "gg",
             "in", "notin", "ni", "subset", "subseteq", "supset", "supseteq", "to", "rightarrow", "Rightarrow",
             "longrightarrow", "leftarrow", "gets", "Leftarrow", "leftrightarrow", "Leftrightarrow", "iff", "implies",
             "mapsto", "perp", "parallel", "mid"}
BINARY = {"times", "cdot", "div", "pm", "mp", "cup", "cap", "oplus", "otimes", "setminus", "ast", "star", "land",
          "lor", "wedge", "vee"}
UNICODE_RELATIONS = "≤≥≠≈≡∝→←↔⇒⇐⇔∈∉⊂⊆⊃⊇∼≃≅↦≪≫"  # typed as Unicode in the maths
UNICODE_BINARY = "×·÷±∓∪∩−"
FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "log",
         "ln", "lg", "exp", "lim", "max", "min", "sup", "inf", "det", "dim", "ker", "deg", "gcd", "arg", "Pr",
         "mod", "bmod"}
UPRIGHT = {"mathrm", "operatorname", "mathsf", "mathtt", "mathcal", "mathscr", "mathfrak", "mathbin", "mathrel",
           "mathop", "mathord", "textstyle", "displaystyle"}
TEXTS = {"text", "textrm", "textup", "textnormal", "mbox", "textsf", "texttt", "hbox"}
BOLD = {"mathbf", "textbf", "boldsymbol", "bm", "pmb"}
ITALIC = {"mathit", "textit", "emph"}
ACCENTS = {"vec": "\u20d7", "hat": "\u0302", "widehat": "\u0302", "bar": "\u0304", "overline": "\u0305",
           "tilde": "\u0303", "widetilde": "\u0303", "dot": "\u0307", "ddot": "\u0308", "check": "\u030c",
           "acute": "\u0301", "grave": "\u0300", "breve": "\u0306", "overrightarrow": "\u20d7"}
DROP = {"left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl",
        "Biggr", "middle", "limits", "nolimits", "nonumber", "notag", "displaystyle", "textstyle", "scriptstyle",
        "allowbreak", "relax"}
SKIP_ARG = {"label", "tag", "phantom", "hphantom", "vphantom", "color", "ref", "eqref"}
BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "P": "ℙ", "E": "𝔼"}
SPACES = {",": "\u2009", ";": " ", ":": " ", ">": " ", " ": " ", "!": "", "\\": " ", "/": ""}
LITERAL = {"{": "{", "}": "}", "$": "$", "%": "%", "&": "&", "#": "#", "_": "_", "|": "‖"}


@dataclass(frozen=True)
class _Mode:
    script: int = 0
    italic: bool = True   # Latin letters are maths variables
    bold: bool = False


class _TeX:
    """A small LaTeX reader for inline maths: enough for super/subscripts, Greek, relations, arrows, short
    fractions and roots. Unknown commands are dropped (an argument they have is kept as plain content)."""

    def __init__(self, src: str):
        self.s, self.i = src, 0

    def parse(self) -> list[Run]:
        return self.seq(_Mode())

    def seq(self, mode: _Mode, stop: str | None = None) -> list[Run]:
        out: list[Run] = []
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "}" or (c == "]" and stop == "]"):
                self.i += 1
                if stop is not None:
                    return out
                continue
            out += self.atom(mode)
        return out

    def _skip_space(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def _run(self, text: str, mode: _Mode, italic: bool | None = None, op: str = "") -> Run:
        return Run(text, mode.script, mode.italic if italic is None else italic, mode.bold, True, op)

    def atom(self, mode: _Mode) -> list[Run]:
        c = self.s[self.i]
        if c == "{":
            self.i += 1
            return self.seq(mode, "}")
        if c in "^_":
            self.i += 1
            self._skip_space()
            rest = self.s[self.i:]
            if c == "^":
                m = re.match(r"\{?\s*\\circ\s*\}?", rest)
                if m:
                    self.i += m.end()
                    return [self._run("°", replace(mode, script=0), False)]
                m = re.match(r"\{?\s*(\\prime|')+\s*\}?", rest)
                if m:
                    self.i += m.end()
                    return [self._run("′" * max(1, m.group(0).count("prime") + m.group(0).count("'")), mode, False)]
            sub = replace(mode, script=mode.script or (1 if c == "^" else -1))
            return self.arg(sub)
        if c == "\\":
            return self.command(mode)
        if c.isspace():
            self._skip_space()  # TeX ignores spaces in maths; operators are spaced by _space_ops
            return []
        self.i += 1
        if c == "~":
            return [self._run("\u00a0", mode, False)]
        if c == "&":
            return [self._run(" ", mode, False)]
        if c == "'":
            return [self._run("′", mode, False)]
        if c in "=<>" or c in UNICODE_RELATIONS:
            return [self._run(c, mode, False, op="rel")]
        if c in "+-*" or c in UNICODE_BINARY:
            return [self._run({"-": "−", "*": "∗"}.get(c, c), mode, False, op="bin")]
        if c in ",;":
            return [self._run(c, mode, False, op="punct")]
        if c.isascii() and c.isalpha():
            return [self._run(c, mode)]
        return [self._run(c, mode, False)]

    def arg(self, mode: _Mode) -> list[Run]:
        """One argument: a {group}, a command or a single character."""
        self._skip_space()
        if self.i >= len(self.s):
            return []
        c = self.s[self.i]
        if c == "{":
            self.i += 1
            return self.seq(mode, "}")
        if c == "}":
            return []
        return self.atom(mode)

    def raw_group(self) -> str:
        """A {group} as raw text (for \\text)."""
        self._skip_space()
        if self.i >= len(self.s) or self.s[self.i] != "{":
            return ""
        depth, start = 0, self.i
        while self.i < len(self.s):
            ch = self.s[self.i]
            if ch == "\\":
                self.i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    self.i += 1
                    return self.s[start + 1:self.i - 1]
            self.i += 1
        return self.s[start + 1:]

    def command(self, mode: _Mode) -> list[Run]:
        self.i += 1
        if self.i >= len(self.s):
            return []
        c = self.s[self.i]
        if not (c.isascii() and c.isalpha()):  # TeX command names are ASCII letters: '\μ' is '\' then 'μ'
            self.i += 1
            if c in SPACES:
                return [self._run(SPACES[c], mode, False)] if SPACES[c] else []
            if c in LITERAL:
                return [self._run(LITERAL[c], mode, False)]
            return [self._run(c, mode, False)]
        m = re.match(r"[A-Za-z]+\*?", self.s[self.i:])
        name = m.group(0).rstrip("*")
        self.i += m.end()
        if name in GREEK:  # lower-case Greek is a variable, italic as TeX sets it (next to italic Latin letters)
            return [self._run(GREEK[name], mode, None if GREEK[name].islower() else False)]
        if name in SYMBOLS:
            op = "rel" if name in RELATIONS else "bin" if name in BINARY else ""
            return [self._run(SYMBOLS[name], mode, False, op=op)]
        if name in FUNCS:
            return [self._run(name, mode, False, op="fn")]
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            a, b = self.arg(mode), self.arg(mode)
            return ([self._run("", mode, False, op="open")] + _paren(a, mode) + [self._run("/", mode, False)]
                    + _paren(b, mode) + [self._run("", mode, False, op="close")])
        if name == "sqrt":
            self._skip_space()
            index: list[Run] = []
            if self.i < len(self.s) and self.s[self.i] == "[":
                self.i += 1
                index = self.seq(replace(mode, script=1), "]")
            a = self.arg(mode)
            return index + [self._run("√", mode, False)] + _paren(a, mode, root=True)
        if name in ("binom", "dbinom", "tbinom", "choose"):
            a, b = self.arg(mode), self.arg(mode)
            return ([self._run("C(", mode, False)] + a + [self._run(", ", mode, False)] + b
                    + [self._run(")", mode, False)])
        if name in TEXTS:
            raw = self.raw_group().replace("\\", "")
            return [self._run(" ".join(raw.split()) if raw.strip() else raw, mode, False)]
        if name in BOLD:
            return self.arg(replace(mode, bold=True, italic=name in ("boldsymbol", "bm", "pmb")))
        if name in ITALIC:
            return self.arg(replace(mode, italic=True))
        if name == "mathbb":
            return [replace(r, text="".join(BLACKBOARD.get(ch, ch) for ch in r.text), italic=False)
                    for r in self.arg(replace(mode, italic=False))]
        if name in UPRIGHT:
            return self.arg(replace(mode, italic=False)) if name not in ("textstyle", "displaystyle") else []
        if name in ACCENTS:
            a = self.arg(mode)
            if a:
                a[-1] = replace(a[-1], text=a[-1].text + ACCENTS[name])
            return a
        if name == "not":
            a = self.arg(mode)
            if a:
                a[-1] = replace(a[-1], text=a[-1].text + "\u0338")
            return a
        if name in ("overset", "underset", "stackrel"):
            self.arg(mode)
            return self.arg(mode)
        if name == "textcolor":
            self.raw_group()
            return self.arg(mode)
        if name in SKIP_ARG:
            self.raw_group()
            return []
        if name in ("hspace", "vspace", "mspace", "hskip", "mskip", "kern", "mkern"):  # a length, not content
            self._skip_space()
            m = re.match(r"\{[^{}]*\}|-?[\d.]+\s*[a-z]{2}", self.s[self.i:])
            self.i += m.end() if m else 0
            return [self._run(" ", mode, False)] if name != "vspace" else []
        if name in DROP:
            self._skip_space()
            if name in ("left", "right", "middle", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr",
                        "biggl", "biggr", "Biggl", "Biggr") and self.s[self.i:self.i + 1] == ".":
                self.i += 1
            return []
        return []  # unknown: dropped; a {group} after it is read as ordinary content


def _plain(runs: list[Run]) -> str:
    return "".join(r.text for r in runs)


def _paren(runs: list[Run], mode: _Mode, root: bool = False) -> list[Run]:
    """A fraction's part, or a root's argument, in brackets unless it reads as one term: no operator, relation or
    operator name in it (a root: one number or one letter)."""
    text = _plain(runs).strip()
    if not text:
        return []
    if text[0] in "([{" and text[-1] in ")]}" and text.count(text[0]) == 1:
        return runs
    marks = "\u0300-\u036f\u20d7"  # combining accents (\hat, \vec)
    simple = (re.fullmatch(rf"\d+(\.\d+)?|.[{marks}]?", text) if root
              else re.fullmatch(rf"[\w.′°{marks}]+", text) and not any(r.op for r in runs))
    if simple:
        return runs
    return [Run("(", mode.script, False, mode.bold, True)] + runs + [Run(")", mode.script, False, mode.bold, True)]


def _space_ops(runs: list[Run]) -> list[Run]:
    """TeX's spacing, roughly: a space on both sides of a relation (=, ≤, →) and of a binary operator (+, ×) unless
    it is a sign (−x), a space after a comma, a thin space between an operator name (sin, log) or a fraction and
    the letters next to it. None inside superscripts and subscripts."""
    out: list[Run] = []
    last_op = ""  # the class of the previous run, before spacing
    for i, r in enumerate(runs):
        if not r.op:
            out.append(r)
            last_op = ""
            continue
        text = r.text
        prev = out[-1] if out else None
        nxt = next((x for x in runs[i + 1:] if not x.script or r.script), None)
        if r.op in ("open", "close"):  # a fraction set as a/b: a thin space from the letters around it
            near = prev.text[-1:] if r.op == "open" and prev is not None else (
                nxt.text[:1] if r.op == "close" and nxt is not None else "")
            text = "\u2009" if near and (near.isalnum() or near in ")]") and not r.script else ""
        elif r.op == "fn":
            if prev is not None and prev.text[-1:] and (prev.text[-1].isalnum() or prev.text[-1] in ")]′"):
                text = "\u2009" + text
            if nxt is not None and nxt.text[:1].isalnum():
                after = runs.index(nxt, i + 1)
                if after == i + 1:
                    text += "\u2009"
                else:  # sin²θ: the space goes after the superscript
                    runs[after] = replace(nxt, text="\u2009" + nxt.text)
        elif r.script:
            pass
        elif r.op == "rel":
            text = (" " if prev is not None else "") + text + " "
        elif r.op == "bin":
            sign = prev is None or last_op in ("rel", "bin", "punct") or prev.text.rstrip()[-1:] in ("(", "[", "{", "")
            if not sign:
                text = " " + text + " "
        elif r.op == "punct":
            text += " "
        out.append(replace(r, text=text, op=""))
        last_op = r.op
    return out


def merge_runs(runs: list[Run]) -> list[Run]:
    """Neighbouring runs of the same style joined, repeated spaces collapsed, ends trimmed."""
    out: list[Run] = []
    for r in runs:
        if not r.text:
            continue
        if out and (out[-1].script, out[-1].italic, out[-1].bold, out[-1].math) == (r.script, r.italic, r.bold, r.math):
            out[-1] = replace(out[-1], text=out[-1].text + r.text)
        else:
            out.append(r)
    clean: list[Run] = []
    for r in out:
        text = re.sub(r" {2,}", " ", r.text)
        if clean and clean[-1].text.endswith(" ") and text.startswith(" "):
            text = text[1:]
        if text:
            clean.append(replace(r, text=text))
    if clean:
        clean[0] = replace(clean[0], text=clean[0].text.lstrip(" "))
        clean[-1] = replace(clean[-1], text=clean[-1].text.rstrip(" "))
    return [r for r in clean if r.text]


def tex_runs(latex: str) -> list[Run]:
    """Inline LaTeX as Unicode runs (maths font, italic variables, superscript/subscript runs)."""
    return list(_tex_runs(_safe(latex)))


@functools.lru_cache(maxsize=4096)
def _tex_runs(latex: str) -> tuple[Run, ...]:
    try:
        runs = _TeX(latex).parse()
    except RecursionError:
        runs = [Run(_flat(latex).replace("\\", ""), math=True)]
    return tuple(merge_runs(_space_ops(runs)))


def _find_close(text: str, a: int) -> int:
    """The $ closing maths that opens at a, or -1. Pandoc's rule, stricter: the next $ closes it only if it has no
    space before it and no digit after it ('$5 or $10' is money)."""
    b = text.find("$", a + 1)
    while b > 0 and text[b - 1] == "\\":
        b = text.find("$", b + 1)
    if b < 0 or text[b - 1].isspace() or text[b + 1:b + 2].isdigit():
        return -1
    return b


def rich(text: Any) -> list[Run]:
    """Plain text with inline $...$ maths -> runs. A $ that does not open maths ('$5 or $10') stays."""
    return list(_rich(_flat(text)))


@functools.lru_cache(maxsize=4096)
def _rich(text: str) -> tuple[Run, ...]:
    out: list[Run] = []
    i = 0
    while i < len(text):
        a = text.find("$", i)
        if a < 0:
            out.append(Run(text[i:]))
            break
        if a > 0 and text[a - 1] == "\\":
            out.append(Run(text[i:a - 1] + "$"))
            i = a + 1
            continue
        if text.startswith("$$", a):
            b = text.find("$$", a + 2)
            if b > a + 2:
                out.append(Run(text[i:a]))
                out += tex_runs(text[a + 2:b])
                i = b + 2
                continue
        b = _find_close(text, a) if a + 1 < len(text) and not text[a + 1].isspace() else -1
        if b > a + 1:
            out.append(Run(text[i:a]))
            out += tex_runs(text[a + 1:b])
            i = b + 1
        else:
            out.append(Run(text[i:a + 1]))
            i = a + 1
    return tuple(merge_runs(out))


_MATH_ITALIC = {**{chr(ord("A") + k): chr(0x1D434 + k) for k in range(26)},
                **{chr(ord("a") + k): chr(0x1D44E + k) for k in range(26)}, "h": "ℎ"}
_MATH_GREEK = {**{chr(0x3B1 + k): chr(0x1D6FC + k) for k in range(25)},  # α…ω in mathematical italic, as TeX sets them
               "ϵ": chr(0x1D716), "ϑ": chr(0x1D717), "ϕ": chr(0x1D719), "ϱ": chr(0x1D71A), "ϖ": chr(0x1D71B)}


_PLAIN_CMDS = {"to": "→", "rightarrow": "→", "times": "×", "cdot": "·", "approx": "≈", "neq": "≠", "ne": "≠",
               "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "pm": "±", "div": "÷"}
_PLAIN_MATH = re.compile(r"(?:\s|\d|[.,=+\-<>()]|\\\$|\\%|\{,\}|\\(?:" + "|".join(_PLAIN_CMDS) + r")(?![A-Za-z]))+")


def plain_math(latex: str) -> str:
    """A formula of numbers only ("1331", "\\$2.25", "1{,}000", "50\\%", "1000 \\to 1100", "225 \\times 1.5 = 337.5")
    as text, for the body font: no letters, powers, indices or fractions. '' for anything else. The player's preview
    does the same (web/src/slides/deck.ts, plainMath)."""
    s = " ".join(latex.split())
    if not re.search(r"\d", s) or not _PLAIN_MATH.fullmatch(s):
        return ""
    s = s.replace("{,}", ",").replace("\\$", "$").replace("\\%", "%")
    s = re.sub(r"\\([A-Za-z]+)", lambda m: f" {_PLAIN_CMDS[m[1]]} ", s)
    s = re.sub(r"(?<=[\d)%])\s*-\s*", " − ", s).replace("-", "−")  # a minus between numbers, else a sign
    s = re.sub(r"\s*([=<>+→≈≠≤≥×÷±·])\s*", r" \1 ", s)
    return " ".join(s.split())


def display_runs(latex: str) -> list[Run]:
    """A display formula as Unicode text for Cambria Math (when it could not be typeset): variables in the
    mathematical italic letters, as Word's equations write them."""
    return list(_display_runs(_safe(latex)))


@functools.lru_cache(maxsize=1024)
def _display_runs(latex: str) -> tuple[Run, ...]:
    out = []
    for r in tex_runs(latex):
        text = "".join(_MATH_ITALIC.get(ch, ch) for ch in r.text) if r.italic else r.text
        text = "".join(_MATH_GREEK.get(ch, ch) for ch in text) if not r.op and r.math else text
        out.append(replace(r, text=text, italic=False))
    return tuple(out or [Run(_flat(latex), math=True)])


# ---------- drawing primitives ----------
class Draw:
    """python-pptx calls, in points."""

    def __init__(self):
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
        from pptx.oxml.ns import qn
        from pptx.util import Pt
        self.RGB, self.SHAPE, self.ANCHOR, self.AUTO, self.ALIGN, self.qn, self.Pt = (
            RGBColor, MSO_SHAPE, MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN, qn, Pt)

    def rgb(self, hex6: str):
        return self.RGB.from_string(hex6)

    def fill_frame(self, tf, paras: list[Para], anchor: str = "t", inset: tuple[float, float, float, float] = (0, 0, 0, 0)):
        tf.word_wrap = True
        tf.auto_size = self.AUTO.NONE
        tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (E(v) for v in inset)
        tf.vertical_anchor = {"t": self.ANCHOR.TOP, "m": self.ANCHOR.MIDDLE, "b": self.ANCHOR.BOTTOM}[anchor]
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            self.paragraph(p, para)

    def paragraph(self, p, para: Para) -> None:
        p.alignment = {"l": self.ALIGN.LEFT, "c": self.ALIGN.CENTER, "r": self.ALIGN.RIGHT}[para.align]
        p.line_spacing = self.Pt(para.line_h)
        p.space_before = self.Pt(para.before)
        p.space_after = self.Pt(0)
        if para.bullet:
            self.bullet(p, para)
        for r in para.runs:
            family, _, bold, italic = _run_font(r, para)
            run = p.add_run()
            run.text = r.text
            f = run.font
            f.size = self.Pt(para.size)
            f.name = family
            f.bold = bold
            f.italic = italic
            f.color.rgb = self.rgb(para.color)
            if r.script:
                f._rPr.set("baseline", "30000" if r.script > 0 else "-25000")
            if para.font == TITLE_FONT:
                f._rPr.set("cap", "none")

    def bullet(self, p, para: Para) -> None:
        """A real paragraph bullet hanging in the left margin, so wrapped lines line up with the text."""
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(E(para.marl)))
        pPr.set("indent", str(-E(para.hang or para.marl)))
        for tag in ("a:buClrTx", "a:buClr", "a:buSzTx", "a:buSzPct", "a:buSzPts", "a:buFontTx", "a:buFont",
                    "a:buNone", "a:buAutoNum", "a:buChar", "a:buBlip"):
            for el in pPr.findall(self.qn(tag)):
                pPr.remove(el)
        clr = pPr.makeelement(self.qn("a:buClr"), {})
        clr.append(clr.makeelement(self.qn("a:srgbClr"), {"val": para.bullet_color}))
        font = pPr.makeelement(self.qn("a:buFont"), {"typeface": "Arial"})
        char = pPr.makeelement(self.qn("a:buChar"), {"char": para.bullet})
        for el in (clr, font, char):
            pPr.insert_element_before(el, "a:tabLst", "a:defRPr", "a:extLst")

    def textbox(self, shapes, x: float, y: float, w: float, h: float, paras: list[Para], anchor: str = "t", name: str = ""):
        tb = shapes.add_textbox(E(x), E(y), E(w), E(h))
        if name:
            tb.name = name
        self.fill_frame(tb.text_frame, paras, anchor)
        return tb

    def rect(self, shapes, x: float, y: float, w: float, h: float, color: str, radius: float = 0.0, oval: bool = False):
        kind = self.SHAPE.OVAL if oval else self.SHAPE.ROUNDED_RECTANGLE if radius else self.SHAPE.RECTANGLE
        shp = shapes.add_shape(kind, E(x), E(y), E(w), E(h))
        if radius and not oval:
            shp.adjustments[0] = max(0.0, min(0.5, radius / max(1.0, min(w, h))))
        shp.fill.solid()
        shp.fill.fore_color.rgb = self.rgb(color)
        shp.line.fill.background()
        shp.shadow.inherit = False
        style = shp._element.find(self.qn("p:style"))  # its effect reference draws a shadow in some viewers
        if style is not None:
            shp._element.remove(style)
        return shp

    def picture(self, shapes, path: Path, x: float, y: float, w: float, h: float, alt: str, frame: str = ""):
        pic = shapes.add_picture(str(path), E(x), E(y), E(w), E(h))
        pic._element.nvPicPr.cNvPr.set("descr", _safe(alt)[:1500])
        if frame:
            pic.line.color.rgb = self.rgb(frame)
            pic.line.width = self.Pt(0.75)
        return pic


# ---------- plans ----------
@dataclass
class Plan:
    """A block laid out for a width and a body size: its height, and how to draw it with its top-left at (x, y)."""
    h: float
    draw: Callable[[Any, float, float], None]
    w: float = 0.0
    shrink: float = 1.0   # the smallest scale given to a formula (1: none)
    fig: float = 0.0      # area of a figure (pt²)


@dataclass
class EqView:
    w: float
    h: float
    k: float
    draw: Callable[[Any, float, float], None]


def _rows(blocks: list[dict]) -> list[dict]:
    """Formulas that follow one another become one block, so they can sit side by side."""
    out: list[dict] = []
    for b in blocks:
        if b.get("type") == "equation" and out and out[-1].get("type") in ("equation", "equations"):
            prev = out.pop()
            items = prev["items"] if prev.get("type") == "equations" else [prev]
            out.append({"type": "equations", "items": items + [b]})
        else:
            out.append(b)
    return out


class Builder:
    def __init__(self, job_dir: Path, deck: dict):
        from pptx import Presentation
        self.job_dir = Path(job_dir)
        self.deck = deck if isinstance(deck, dict) else {}
        self.d = Draw()
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = E(SW), E(SH)
        self._widescreen_layouts()
        self.layouts = {l.name: l for l in self.prs.slide_layouts}
        self.slides_in = [s for s in self.deck.get("slides") or [] if isinstance(s, dict)]
        self.lecture = _flat(self.deck.get("lecture")) or _flat(self.deck.get("title")) or "Lecture"
        # The slides' footer names the deck: the job's subject can be writing left from an earlier lesson (live, a
        # deck on compound interest had "Logarithms and exponentials" on every slide).
        self.footer_name = _flat(self.deck.get("title")) or self.lecture
        self.sections = self._sections()
        self.tex = texrender.render(self._formulas(), color="#" + SLATE, display=True)
        self.number = 0
        self.figure_cache: dict[str, tuple[Path, int, int] | None] = {}

    # ----- deck facts -----
    def _widescreen_layouts(self) -> None:
        """The default template is 4:3: its master and layouts are stretched to 16:9, so a slide added or reset in
        PowerPoint looks right too. A layout placeholder without a position of its own follows the master's."""
        k = SW / 720.0
        for layout in self.prs.slide_layouts:
            for ph in layout.placeholders:
                if ph._element.xfrm is not None:
                    ph.left, ph.width = int(ph.left * k), int(ph.width * k)
        for ph in self.prs.slide_master.placeholders:
            ph.left, ph.width = int(ph.left * k), int(ph.width * k)

    def _sections(self) -> list[dict]:
        """[{name, t0, t1}] in order: the deck's sections, from its section and content slides."""
        out: list[dict] = []
        for sl in self.slides_in:
            kind = sl.get("kind")
            if kind not in ("section", "content"):
                continue
            name = _flat(sl.get("section")) or (_flat(sl.get("title")) if kind == "section" else "")
            if not name:
                continue
            if not out or out[-1]["name"] != name:
                out.append({"name": name, "t0": None, "t1": None})
            cur = out[-1]
            for key, pick in (("t0", min), ("t1", max)):
                v = sl.get(key)
                if _num(v):
                    cur[key] = v if cur[key] is None else pick(cur[key], v)
        return out

    def _blocks(self, sl: dict) -> list[dict]:
        return [b for b in sl.get("blocks") or [] if isinstance(b, dict)]

    def _formulas(self) -> list[str]:
        out = []
        for sl in self.slides_in:
            for b in self._blocks(sl):
                t = b.get("type")
                if t in ("equation", "callout") and _flat(b.get("latex")):
                    out.append(_flat(b["latex"]))
                elif t == "steps":
                    out += [_flat(it.get("latex")) for it in b.get("items") or []
                            if isinstance(it, dict) and _flat(it.get("latex"))]
        return list(dict.fromkeys(x for x in out if not plain_math(x)))  # numbers only: set as text

    def _figure_file(self, block: dict) -> tuple[Path, int, int] | None:
        rel = block.get("image")
        if not isinstance(rel, str) or not re.fullmatch(r"export/fig_\d{3,}\.png", rel):
            return None
        if rel in self.figure_cache:
            return self.figure_cache[rel]
        found = None
        path = (self.job_dir / rel).resolve()
        try:
            if path.is_file() and path.is_relative_to(self.job_dir.resolve()):
                from PIL import Image
                with Image.open(path) as im:
                    found = (path, int(im.width), int(im.height))
        except Exception:
            found = None
        self.figure_cache[rel] = found
        return found

    # ----- formulas -----
    def eq_view(self, latex: str, size: float, max_w: float, align_paras: str = "c") -> EqView:
        """A display formula at this font size, scaled down to max_w if it is wider: its picture, or Unicode text in
        Cambria Math when it could not be typeset."""
        latex = _flat(latex)
        number = plain_math(latex)
        if number:  # numbers only ("1331", "1000 → 1100"): text in the body font, not a serif picture
            para = Para([Run(number)], size, SLATE, align=align_paras)
            w = min(max_w, natural_width(para) * 1.02 + 2)
            h = para_height(para, max_w)

            def draw_number(shapes, x, y, para=para, w=w, h=h):
                self.d.textbox(shapes, x, y, w + 4, h + 2, [replace(para, align="l")])
            return EqView(w + 4, h + 2, 1.0, draw_number)
        img = self.tex.get(latex)
        if img is not None:
            w, h = img.size_pt(size)
            k = min(1.0, max_w / w) if w > 0 else 1.0
            w, h = w * k, h * k

            def draw(shapes, x, y, img=img, w=w, h=h, latex=latex):
                self.d.picture(shapes, img.path, x, y, w, h, latex)
            return EqView(w, h, k, draw)
        para = Para(display_runs(latex), size, SLATE, font=MATH_FONT, math_font=MATH_FONT, align=align_paras,
                    line=1.25)
        k = 1.0
        while line_count(para, max_w) > 2 and para.size > size * 0.7:
            para = replace(para, size=para.size - 1)
            k = para.size / size
        w = min(max_w, natural_width(para) * 1.02 + 2)
        h = para_height(para, max_w if w >= max_w else w + 4)

        def draw_text(shapes, x, y, para=para, w=w, h=h):
            self.d.textbox(shapes, x, y, w + 4, h + 2, [replace(para, align="l")] if w < max_w else [para])
        return EqView(w + 4, h + 2, k, draw_text)

    # ----- blocks -----
    def plan_bullets(self, b: dict, w: float, s: float) -> Plan | None:
        paras = []
        for i, it in enumerate(x for x in b.get("items") or [] if isinstance(x, dict) and _flat(x.get("text"))):
            sub = it.get("level") == 1 and i > 0
            size = max(MIN_BODY if s >= MIN_BODY else s, round(s * 0.88)) if sub else s
            marl = s * 0.95 + (size * 1.05 if sub else 0)
            paras.append(Para(rich(it["text"]), size, SLATE, bullet="–" if sub else "•",
                              bullet_color=MUTED if sub else ACCENT, marl=marl, hang=size * 1.05 if sub else marl,
                              before=0 if i == 0 else (s * 0.28 if sub else s * 0.5)))
        if not paras:
            return None
        h = paras_height(paras, w) + 2

        def draw(shapes, x, y):
            self.d.textbox(shapes, x, y, w, h, paras)
        return Plan(h, draw, w)

    def _label_para(self, text: str, s: float, color: str = MUTED, bold: bool = True) -> Para:
        return Para(rich(text), max(13.0, round(s * 0.66)), color, bold=bold)

    def plan_equation(self, b: dict, w: float, s: float, min_h: float = 0.0) -> Plan | None:
        """A display formula on a light card: its label at the top left, the formula centred under it."""
        latex = _flat(b.get("latex"))
        if not latex:
            return None
        padx, pady = 0.9 * s, 0.62 * s
        inner = w - 2 * padx
        label = _flat(b.get("label"))
        lab = self._label_para(label, s) if label else None
        lab_h = para_height(lab, inner) if lab else 0.0
        gap = 0.3 * s if lab else 0.0
        eq = self.eq_view(latex, s * EQ_K, inner)
        h = max(min_h, pady + lab_h + gap + eq.h + pady)
        room = h - 2 * pady - lab_h - gap  # a card made taller to match its neighbour centres its formula

        def draw(shapes, x, y):
            grp = shapes.add_group_shape()
            grp.name = "Equation"
            self.d.rect(grp.shapes, x, y, w, h, PANEL, radius=8)
            if lab:
                self.d.textbox(grp.shapes, x + padx, y + pady, inner, lab_h, [lab])
            eq.draw(grp.shapes, x + (w - eq.w) / 2, y + pady + lab_h + gap + (room - eq.h) / 2)
        return Plan(h, draw, w, shrink=eq.k)

    def plan_equation_row(self, b: dict, w: float, s: float) -> Plan | None:
        """Formulas that follow one another: side by side on cards of one height when each fits its share of the
        width at full size, else stacked."""
        items = [x for x in b.get("items") or [] if isinstance(x, dict)]
        gap = 0.85 * s
        n = len(items)
        cw = (w - gap * (n - 1)) / max(1, n)
        side = [self.plan_equation(x, cw, s) for x in items]
        if n > 1 and all(p is not None and p.shrink >= 0.999 for p in side):
            h = max(p.h for p in side)
            side = [self.plan_equation(x, cw, s, h) for x in items]

            def draw_row(shapes, x, y):
                for k, p in enumerate(side):
                    p.draw(shapes, x + k * (cw + gap), y)
            return Plan(h, draw_row, w)
        stack = [self.plan_equation(x, w, s) for x in items]
        stack = [p for p in stack if p is not None]
        if not stack:
            return None

        def draw_stack(shapes, x, y):
            for p in stack:
                p.draw(shapes, x, y)
                y += p.h + gap
        return Plan(sum(p.h for p in stack) + gap * (len(stack) - 1), draw_stack, w,
                    shrink=min(p.shrink for p in stack))

    def plan_steps(self, b: dict, w: float, s: float) -> Plan | None:
        steps = [it for it in b.get("items") or [] if isinstance(it, dict)
                 and (_flat(it.get("latex")) or _flat(it.get("text")))]
        if not steps:
            return None
        label = _flat(b.get("label"))
        lab = Para(rich(label), max(MIN_BODY if s >= MIN_BODY else s, round(s * 0.85)), SLATE, bold=True) if label else None
        lab_h = para_height(lab, w) if lab else 0.0
        lab_gap = 0.45 * s if lab else 0.0
        d = round(s * 1.3, 1)
        g1 = 0.6 * s
        inner = w - d - g1
        tsize = max(MIN_BODY if s >= MIN_BODY else s, round(s * 0.92))
        e = s * STEP_EQ_K
        has_eq = any(_flat(it.get("latex")) for it in steps)
        has_text = any(_flat(it.get("text")) for it in steps)
        grid = False
        eq_col = inner
        if has_eq and has_text:
            widest = max(self.eq_view(it["latex"], e, 10_000).w for it in steps if _flat(it.get("latex")))
            eq_col = min(widest, inner * 0.6)
            text_col = inner - eq_col - s
            grid = text_col >= max(150.0, inner * 0.3)
        rows = []
        shrink = 1.0
        for n, it in enumerate(steps, 1):
            latex, text = _flat(it.get("latex")), _flat(it.get("text"))
            eq = self.eq_view(latex, e, eq_col if grid else inner, "l") if latex else None
            if eq:
                shrink = min(shrink, eq.k)
            tw = (inner - eq_col - s) if (grid and eq) else inner
            para = Para(rich(text), tsize, SLATE) if text else None
            th = para_height(para, tw) + 1 if para else 0.0
            if grid or not eq or not para:
                rh = max(d, eq.h if eq else 0.0, th)
            else:
                rh = max(d, eq.h + 0.25 * s + th)
            rows.append((n, eq, para, tw, th, rh))
        gap = 0.5 * s
        h = lab_h + lab_gap + sum(r[5] for r in rows) + gap * (len(rows) - 1)

        def draw(shapes, x, y):
            if lab:
                self.d.textbox(shapes, x, y, w, lab_h, [lab])
            yy = y + lab_h + lab_gap
            for n, eq, para, tw, th, rh in rows:
                cy = yy + rh / 2
                top_aligned = not grid and eq is not None and para is not None
                circle_y = yy + max(0.0, (eq.h - d) / 2) if top_aligned else cy - d / 2
                dot = self.d.rect(shapes, x, circle_y, d, d, ACCENT, oval=True)
                self.d.fill_frame(dot.text_frame, [Para([Run(str(n))], max(11.0, round(s * 0.62)), WHITE, bold=True,
                                                         align="c", line=1.0)], "m")
                cx = x + d + g1
                # Text beside a number is centred on it, whatever line count PowerPoint wraps it to (a row measured
                # as two lines but drawn as one sat above its number, test-full's takeaways).
                if grid:
                    if eq:
                        eq.draw(shapes, cx, cy - eq.h / 2)
                        if para:
                            self.d.textbox(shapes, cx + eq_col + s, cy - th / 2, tw, th, [para], "m")
                    elif para:
                        self.d.textbox(shapes, cx, cy - th / 2, inner, th, [para], "m")
                elif eq and para:
                    eq.draw(shapes, cx, yy)
                    self.d.textbox(shapes, cx, yy + eq.h + 0.25 * s, tw, th, [para])
                elif eq:
                    eq.draw(shapes, cx, cy - eq.h / 2)
                else:
                    self.d.textbox(shapes, cx, cy - th / 2, tw, th, [para], "m")
                yy += rh + gap
        return Plan(h, draw, w, shrink=shrink)

    def plan_figure(self, b: dict, w: float, hmax: float, s: float) -> Plan | None:
        found = self._figure_file(b)
        if found is None:
            return None
        path, iw, ih = found
        caption = _flat(b.get("caption"))
        cap = Para(rich(caption), max(12.0, round(s * 0.62)), MUTED, align="c") if caption else None
        cap_h = para_height(cap, w) if cap else 0.0
        gap = 0.4 * s if cap else 0.0
        room = hmax - cap_h - gap
        if room <= 20:
            return None
        k = min(w / iw, room / ih)
        fw, fh = iw * k, ih * k
        h = fh + gap + cap_h

        def draw(shapes, x, y):
            self.d.picture(shapes, path, x + (w - fw) / 2, y, fw, fh, caption or "A figure from the lecture",
                           frame=RULE)
            if cap:
                self.d.textbox(shapes, x, y + fh + gap, w, cap_h, [cap])
        return Plan(h, draw, w, fig=fw * fh)

    def plan_table(self, b: dict, w: float, s: float) -> Plan | None:
        header = [_flat(c) for c in b.get("header") or []][:6]
        rows = [[_flat(c) for c in r][:len(header)] for r in b.get("rows") or [] if isinstance(r, list)][:8]
        if not header:
            return None
        rows = [r + [""] * (len(header) - len(r)) for r in rows]
        t = max(MIN_BODY if s >= MIN_BODY else s, round(s * 0.8))
        mx, my = 0.55 * t, 0.32 * t
        head = [Para(rich(c), t, WHITE, bold=True) for c in header]
        body = [[Para(rich(c), t, SLATE) for c in r] for r in rows]
        cols = len(header)
        natural, minimum = [], []
        for j in range(cols):
            cells = [head[j]] + [r[j] for r in body]
            natural.append(max(natural_width(p) for p in cells) * 1.02 + 2 * mx + 4)
            minimum.append(max(max((wd for wd, _ in _words(p.runs, p)), default=0.0) for p in cells) + 2 * mx + 2)
        if sum(minimum) > w:
            if s > LAST_RESORT_SIZES[-1]:
                return None  # smaller type may fit the longest words
            # The smallest type and the words still do not fit: they break inside their cells, as PowerPoint breaks
            # them, rather than the table being left out.
            total = sum(minimum)
            minimum = [w * m / total for m in minimum]
        if sum(natural) <= w:
            widths = natural[:]
            total = max(sum(natural), w * TABLE_MIN_SHARE)
            extra = total - sum(widths)
            widths = [x + extra * x / sum(natural) for x in widths]
        else:
            room = w - sum(minimum)
            want = [n - m for n, m in zip(natural, minimum)]
            widths = [m + room * wt / max(1e-6, sum(want)) for m, wt in zip(minimum, want)]
        heights = []
        for r in [head] + body:
            heights.append(max(para_height(p, wd - 2 * mx) for p, wd in zip(r, widths)) + 2 * my + 2)
        caption = _flat(b.get("caption"))
        cap = Para(rich(caption), max(12.0, round(s * 0.62)), MUTED) if caption else None
        tw = sum(widths)
        cap_h = para_height(cap, tw) if cap else 0.0
        gap = 0.35 * s if cap else 0.0
        h = sum(heights) + gap + cap_h

        def draw(shapes, x, y):
            x += max(0.0, (w - tw) / 2)  # centred in its column
            gf = shapes.add_table(len(heights), cols, E(x), E(y), E(tw), E(sum(heights)))
            gf.name = "Table"
            table = gf.table
            table.first_row = True
            table.horz_banding = False
            col_emu = [E(v) for v in widths]
            col_emu[-1] = E(tw) - sum(col_emu[:-1])
            for j, v in enumerate(col_emu):
                table.columns[j].width = v
            row_emu = [E(v) for v in heights]
            row_emu[-1] = E(sum(heights)) - sum(row_emu[:-1])
            for i, v in enumerate(row_emu):
                table.rows[i].height = v
            for i, r in enumerate([head] + body):
                for j, para in enumerate(r):
                    cell = table.cell(i, j)
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = self.d.rgb(ACCENT if i == 0 else BAND if i % 2 == 0 else WHITE)
                    cell.margin_left = cell.margin_right = E(mx)
                    cell.margin_top = cell.margin_bottom = E(my)
                    cell.vertical_anchor = self.d.ANCHOR.MIDDLE
                    tf = cell.text_frame
                    tf.word_wrap = True
                    self.d.paragraph(tf.paragraphs[0], para)
            if cap:
                self.d.textbox(shapes, x, y + sum(heights) + gap, tw, cap_h, [cap])
        return Plan(h, draw, w)

    def plan_callout(self, b: dict, w: float, s: float) -> Plan | None:
        style = b.get("style") if b.get("style") in ("definition", "key", "note") else "key"
        title, text, latex = _flat(b.get("title")), _flat(b.get("text")), _flat(b.get("latex"))
        if not (title or text or latex):
            return None
        edge = 5.0
        padl, padr, pady = edge + 0.8 * s, 0.8 * s, 0.6 * s
        inner = w - padl - padr
        tag = {"definition": "Definition", "key": "Key idea", "note": "Note"}[style]
        tone = MUTED if style == "note" else ACCENT
        paras = [Para([Run(tag.upper())], max(12.0, round(s * 0.56)), tone, bold=True)]
        if title:
            paras.append(Para(rich(title), s, SLATE, bold=True, before=0.18 * s))
        if text:
            paras.append(Para(rich(text), s, SLATE, before=0.22 * s))
        text_h = paras_height(paras, inner) + 1
        eq = self.eq_view(latex, s * 1.15, inner, "l") if latex else None
        eq_gap = 0.35 * s if eq else 0.0
        h = pady + text_h + eq_gap + (eq.h if eq else 0.0) + pady

        def draw(shapes, x, y):
            grp = shapes.add_group_shape()
            grp.name = f"Callout ({tag})"
            self.d.rect(grp.shapes, x, y, w, h, NOTE_FILL if style == "note" else ACCENT_SOFT, radius=6)
            self.d.rect(grp.shapes, x, y, edge, h, NOTE_EDGE if style == "note" else ACCENT)
            self.d.textbox(grp.shapes, x + padl, y + pady, inner, text_h, paras)
            if eq:
                eq.draw(grp.shapes, x + padl, y + pady + text_h + eq_gap)
        return Plan(h, draw, w, shrink=eq.k if eq else 1.0)

    def plan(self, b: dict, w: float, s: float, hmax: float = BODY_H) -> Plan | None:
        t = b.get("type")
        if t == "bullets":
            return self.plan_bullets(b, w, s)
        if t == "equation":
            return self.plan_equation(b, w, s)
        if t == "equations":
            return self.plan_equation_row(b, w, s)
        if t == "steps":
            return self.plan_steps(b, w, s)
        if t == "figure":
            return self.plan_figure(b, w, hmax, s)
        if t == "table":
            return self.plan_table(b, w, s)
        if t == "callout":
            return self.plan_callout(b, w, s)
        return None

    # ----- content layout -----
    def _usable(self, blocks: list[dict]) -> list[dict]:
        """The blocks that can be drawn: not a figure whose picture is missing, an empty list or a table too wide
        for the slide even in small type."""
        return [b for b in blocks if self.plan(b, CW, LAST_RESORT_SIZES[-1]) is not None]

    def _column(self, blocks: list[dict], w: float, s: float) -> tuple[list[Plan], float] | None:
        """Plans stacked in one column; a figure gets the height the others leave."""
        gap = 0.85 * s
        plans: dict[int, Plan] = {}
        for i, b in enumerate(blocks):
            if b.get("type") != "figure":
                p = self.plan(b, w, s)
                if p is None:
                    return None
                plans[i] = p
        used = sum(p.h for p in plans.values()) + gap * (len(blocks) - 1)
        for i, b in enumerate(blocks):
            if b.get("type") == "figure":
                p = self.plan_figure(b, w, BODY_H - used, s)
                if p is None:
                    return None
                plans[i] = p
                used += p.h
        return [plans[i] for i in range(len(blocks))], used

    def _arrangements(self, blocks: list[dict]):
        """(name, [(x, width, blocks)]) for each way of placing the blocks: one column, or a figure in a column of
        its own; a tall, narrow table (side_table) always in a column of its own."""
        table = side_table(blocks)
        if table is None:
            yield "stack", [(MX, CW, blocks)]
        if len(blocks) < 2:
            return
        side = next((b for b in blocks if b.get("type") == "figure"), None)
        fractions = (0.54, 0.48, 0.6, 0.42)
        if side is None:
            side = table
            fractions = (0.5, 0.42, 0.58)
        if side is None:
            return
        rest = [b for b in blocks if b is not side]
        side_first = blocks.index(side) == 0
        for f in fractions:  # f: the share of the side block's column
            ws = (CW - COL_GAP) * f
            wr = CW - COL_GAP - ws
            if side_first:
                yield f"split{f}", [(MX, ws, [side]), (MX + ws + COL_GAP, wr, rest)]
            else:
                yield f"split{f}", [(MX, wr, rest), (MX + wr + COL_GAP, ws, [side])]

    def layout(self, blocks: list[dict]) -> tuple[float, list[tuple[float, list[Plan]]]] | None:
        """(body size, [(x, plans)]) for the largest body size at which the blocks fit; None if they never do."""
        area = CW * BODY_H
        has_fig = any(b.get("type") == "figure" for b in blocks)
        table = side_table(blocks)
        phases = [] if has_fig or table else [(True, BODY_SIZES[:4], True)]  # one column if it fits at 21 pt or more
        phases += [(True, BODY_SIZES, False), (False, BODY_SIZES, False)]
        if table:  # beside the others it never fits: stacked, rather than split over two slides
            phases.append((False, BODY_SIZES, True))
        for strict, sizes, stack_only in phases:
            for s in sizes:
                best, best_score = None, -math.inf
                arrangements = [("stack", [(MX, CW, blocks)])] if stack_only else self._arrangements(blocks)
                for name, cols in arrangements:
                    if stack_only and name != "stack":
                        continue
                    placed, fig, shrink, ok = [], 0.0, 1.0, True
                    for x, w, bl in cols:
                        col = self._column(bl, w, s)
                        if col is None or col[1] > BODY_H + 0.01:
                            ok = False
                            break
                        placed.append((x, col[0]))
                        fig += sum(p.fig for p in col[0])
                        shrink = min([shrink] + [p.shrink for p in col[0]])
                    if not ok:
                        continue
                    if strict and (shrink < READABLE_SHRINK or (has_fig and fig < MIN_FIGURE_SHARE * area)):
                        continue
                    score = fig / area + 0.3 * shrink + (0.04 if name == "stack" else -0.5 * abs(float(name[5:]) - 0.52))
                    if score > best_score:
                        best, best_score = placed, score
                if best is not None:
                    return s, best
        return None

    def forced(self, blocks: list[dict]) -> tuple[float, list[tuple[float, list[Plan]]]]:
        """A single block that does not fit at 14 pt: smaller, as a last resort."""
        for s in LAST_RESORT_SIZES:
            col = self._column(blocks, CW, s)
            if col is not None and col[1] <= BODY_H:
                return s, [(MX, col[0])]
        col = self._column(blocks, CW, LAST_RESORT_SIZES[-1])
        return LAST_RESORT_SIZES[-1], [(MX, col[0] if col else [])]

    def pages(self, blocks: list[dict]) -> list[tuple[float, list[tuple[float, list[Plan]]]]]:
        """The blocks on as few slides as they fit (one, unless they cannot fit at 14 pt)."""
        out = []
        todo = blocks[:]
        while todo:
            for k in range(len(todo), 0, -1):
                got = self.layout(todo[:k])
                if got is not None:
                    out.append(got)
                    todo = todo[k:]
                    break
            else:
                out.append(self.forced(todo[:1]))
                todo = todo[1:]
        return out or [(BODY_SIZES[0], [])]

    # ----- slide furniture -----
    def new_slide(self, layout: str):
        slide = self.prs.slides.add_slide(self.layouts.get(layout) or self.layouts["Title Only"])
        self.number += 1
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = self.d.rgb(WHITE)
        return slide

    def _place(self, shape, x: float, y: float, w: float, h: float) -> None:
        shape.left, shape.top, shape.width, shape.height = E(x), E(y), E(w), E(h)

    def fit_title(self, text: str, w: float, h: float, one: tuple[int, ...], two: tuple[int, ...],
                  line: float = 1.08) -> Para:
        runs = rich(text)
        for size in one:
            p = Para(runs, size, SLATE, font=TITLE_FONT, line=line)
            if line_count(p, w) == 1 and p.line_h <= h:
                return p
        fits = []
        for size in two:
            p = Para(runs, size, SLATE, font=TITLE_FONT, line=line)
            lines, last = _wrap(p, w)
            if lines * p.line_h <= h:
                if lines == 1 or last >= WIDOW * w:  # no word left alone on the last line
                    return p
                fits.append(p)
        return fits[0] if fits else Para(runs, two[-1], SLATE, font=TITLE_FONT, line=line)

    def title(self, slide, text: str) -> None:
        para = self.fit_title(text, CW, TITLE_H, (32, 30, 28, 26), (28, 26, 24, 22, 20))
        ph = slide.shapes.title
        self._place(ph, MX, TITLE_Y, CW, TITLE_H)
        self.d.fill_frame(ph.text_frame, [para], "b")
        bar = self.d.rect(slide.shapes, MX, BAR_Y, BAR_W, BAR_H, ACCENT)
        bar.name = "Accent bar"

    def eyebrow(self, slide, text: str) -> None:
        para = Para([Run(text)], 13, ACCENT, bold=True)
        if line_count(para, CW * 0.8) > 1:
            while text and line_count(Para([Run(text + "…")], 13, ACCENT, bold=True), CW * 0.8) > 1:
                text = text[:-1]
            para = Para([Run(text.rstrip() + "…")], 13, ACCENT, bold=True)
        self.d.textbox(slide.shapes, MX, EYEBROW_Y, CW * 0.8, EYEBROW_H, [para], "b", name="Section")

    def footer(self, slide, sl: dict) -> None:
        when = span(sl.get("t0"), sl.get("t1"))
        name = self.footer_name
        left_w = CW - 80
        while True:
            text = name + (f"  ·  {when}" if when else "")
            para = Para([Run(text)], FOOTER_SIZE, MUTED, line=1.15)
            if line_count(para, left_w) == 1 or len(name) <= 4:
                break
            name = name.rstrip("…")[:-1].rstrip() + "…"  # a long lecture name is cut to one line
        self.d.textbox(slide.shapes, MX, FOOT_Y, left_w, FOOT_H, [para], "m", name="Footer")
        num = Para([Run(str(self.number))], FOOTER_SIZE, MUTED, align="r", line=1.15)
        self.d.textbox(slide.shapes, SW - MX - 60, FOOT_Y, 60, FOOT_H, [num], "m", name="Slide number")

    def notes(self, slide, sl: dict, extra: str = "") -> None:
        parts = []
        notes = _safe(sl.get("notes")).strip()
        if extra:
            notes = (extra + " " + notes).strip() if notes else extra
        if notes:
            parts.append(notes)
        said = _safe(sl.get("said")).strip()
        if said:
            when = span(sl.get("t0"), sl.get("t1"))
            parts.append((f"From the lecture ({when}):" if when else "From the lecture:") + "\n" + said)
        slide.notes_slide.notes_text_frame.text = "\n\n".join(parts)

    # ----- slides -----
    def title_slide(self, sl: dict) -> None:
        slide = self.new_slide("Title Slide")
        x, w = 72.0, SW - 144.0
        title = _flat(sl.get("title")) or _flat(self.deck.get("title")) or self.lecture
        para = self.fit_title(title, w, 140, (44, 42, 40), (40, 38, 36, 34, 32, 30), line=1.05)
        n = line_count(para, w)
        th = n * para.line_h
        ty = 248 - th
        self.d.rect(slide.shapes, x, ty - 30, 64, 5, ACCENT).name = "Accent bar"
        ph = slide.shapes.title
        self._place(ph, x, ty, w, th + 4)
        self.d.fill_frame(ph.text_frame, [para], "b")
        kind = "Slide lecture" if self.deck.get("kind") == "slides" else "Board lecture"
        dur = self.deck.get("duration")
        facts = " · ".join(v for v in (kind, clock(dur) if _num(dur) and dur > 0 else "",
                                        "made with Replay Board from the recording") if v)
        subtitle = _flat(self.deck.get("subtitle"))
        sub = slide.placeholders[1]
        if subtitle:
            sp = Para(rich(subtitle), 22, MUTED, line=1.2)
            sh = min(3, line_count(sp, w)) * sp.line_h
            self._place(sub, x, 272, w, sh + 2)
            self.d.fill_frame(sub.text_frame, [sp], "t")
            self.d.rect(slide.shapes, x, 452, w, 1, RULE).name = "Rule"
            self.d.textbox(slide.shapes, x, 466, w, 20, [Para([Run(facts)], 14, MUTED)], "t", name="About")
        else:
            self._place(sub, x, 272, w, 24)
            self.d.fill_frame(sub.text_frame, [Para([Run(facts)], 16, MUTED)], "t")
        self.notes(slide, sl)

    def agenda_slide(self, sl: dict) -> None:
        slide = self.new_slide("Title Only")
        self.title(slide, _flat(sl.get("title")) or "Agenda")
        items = [(sec["name"], span(sec["t0"], sec["t1"])) for sec in self.sections]
        if not items:  # no sections: whatever the slide lists
            for b in self._blocks(sl):
                for it in b.get("items") or []:
                    if isinstance(it, dict) and _flat(it.get("text")):
                        items.append((_flat(it["text"]), ""))
        cols = 2 if len(items) > 7 else 1
        per = math.ceil(len(items) / cols) if items else 0
        col_w = (CW - (COL_GAP if cols == 2 else 0)) / cols
        row_h = min(52.0, BODY_H / max(1, per))
        size = 22 if cols == 1 else 18
        num_w, time_w = 2.2 * size, 5.6 * FOOTER_SIZE * 1.6
        name_w = col_w - num_w - time_w - 12
        while size > 14 and any(line_count(Para(rich(n), size, SLATE), name_w) * size * 1.15 > row_h - 8
                                for n, _ in items):
            size -= 1
        for k, (name, when) in enumerate(items):
            c, r = divmod(k, per) if per else (0, 0)
            x = MX + c * (col_w + COL_GAP)
            y = BODY_Y + r * row_h
            mid = y + row_h / 2
            nh = line_count(Para(rich(name), size, SLATE), name_w) * round(size * 1.15, 1)
            self.d.textbox(slide.shapes, x, mid - size * 0.65, num_w, size * 1.3,
                           [Para([Run(f"{k + 1:02d}")], size, ACCENT, bold=True, line=1.15)], "m")
            self.d.textbox(slide.shapes, x + num_w, mid - nh / 2, name_w, nh,
                           [Para(rich(name), size, SLATE, line=1.15)], "m")
            if when:
                self.d.textbox(slide.shapes, x + col_w - time_w, mid - 10, time_w, 20,
                               [Para([Run(when)], max(13, size - 6), MUTED, align="r", line=1.15)], "m")
            if r < per - 1 and k < len(items) - 1:
                self.d.rect(slide.shapes, x, y + row_h - 0.5, col_w, 0.75, RULE).name = "Divider"
        self.footer(slide, sl)
        self.notes(slide, sl)

    def section_slide(self, sl: dict) -> None:
        slide = self.new_slide("Section Header")
        slide.background.fill.fore_color.rgb = self.d.rgb(ACCENT)
        name = _flat(sl.get("title")) or _flat(sl.get("section"))
        key = _flat(sl.get("section")) or name
        idx = next((i for i, s in enumerate(self.sections) if s["name"] in (key, name)), None)
        sec = self.sections[idx] if idx is not None else {"t0": None, "t1": None}
        when = span(sl.get("t0"), sl.get("t1")) or span(sec["t0"], sec["t1"])
        x, w = 72.0, SW - 144.0
        para = self.fit_title(name, w, 120, (44, 42, 40), (40, 38, 36, 34, 32), line=1.05)
        para.color = WHITE
        th = line_count(para, w) * para.line_h
        total = 24 + 14 + th + (16 + 26 if when else 0)
        y = (SH - total) / 2 - 8
        part = slide.placeholders[1]
        self._place(part, x, y, w, 24)
        self.d.fill_frame(part.text_frame, [Para([Run(f"PART {idx + 1 if idx is not None else ''}".strip())],
                                                 16, ON_ACCENT, bold=True, line=1.2)], "b")
        ph = slide.shapes.title
        self._place(ph, x, y + 38, w, th + 4)
        self.d.fill_frame(ph.text_frame, [para], "t")
        if when:
            self.d.textbox(slide.shapes, x, y + 38 + th + 16, w, 26, [Para([Run(when)], 18, ON_ACCENT, line=1.2)],
                           "t", name="Time")
        self.notes(slide, sl)

    def content_slide(self, sl: dict) -> None:
        blocks = self._blocks(sl)
        if sl.get("kind") == "summary":  # the takeaways as numbered rows
            blocks = [{"type": "steps", "label": "", "items": [{"latex": "", "text": it.get("text")}
                                                               for it in b.get("items") or [] if isinstance(it, dict)]}
                      if b.get("type") == "bullets" else b for b in blocks]
        blocks = _rows(self._usable(blocks))
        pages = self.pages(blocks)
        for n, (s, cols) in enumerate(pages):
            slide = self.new_slide("Title Only")
            title = _flat(sl.get("title")) or "Untitled"
            self.title(slide, title if n == 0 else f"{title} (continued)")
            if sl.get("kind") == "content" and len(self.sections) >= 2 and _flat(sl.get("section")):
                self.eyebrow(slide, _flat(sl.get("section")))
            gap = 0.85 * s
            heights = [sum(p.h for p in plans) + gap * (len(plans) - 1) for _, plans in cols]
            # The body sits in the middle of the space under the title (live, slides with a formula and two points
            # left the bottom 40% empty); two columns start at the same height.
            top = BODY_Y + max(0.0, (BODY_H - max(heights, default=0.0)) / 2)
            for (x, plans), h in zip(cols, heights):
                y = top
                for p in plans:
                    p.draw(slide.shapes, x, y)
                    y += p.h + gap
            self.footer(slide, sl)
            self.notes(slide, sl, "(continued)" if n else "")

    def formulas_slide(self, sl: dict) -> None:
        slide = self.new_slide("Title Only")
        self.title(slide, _flat(sl.get("title")) or "Formula sheet")
        eqs = [b for b in self._blocks(sl) if b.get("type") == "equation" and _flat(b.get("latex"))][:8]
        n = len(eqs)
        cols = 1 if n <= 2 else 2
        rows = max(1, math.ceil(n / cols))
        gap = 18.0
        cw = (CW - gap * (cols - 1)) / cols
        ch = min(150.0, (BODY_H - gap * (rows - 1)) / rows)
        pad = 14.0
        for k, b in enumerate(eqs):
            r, c = divmod(k, cols)
            x, y = MX + c * (cw + gap), BODY_Y + r * (ch + gap)
            label = _flat(b.get("label"))
            lab = Para(rich(label), 13, MUTED, bold=True) if label else None
            lab_h = para_height(lab, cw - 2 * pad) if lab else 0.0
            room_w, room_h = cw - 2 * pad, ch - 2 * pad - lab_h - (6 if lab else 0)
            size = 28.0
            eq = self.eq_view(b["latex"], size, room_w)
            if eq.h > room_h:
                size = max(10.0, size * room_h / eq.h)
                eq = self.eq_view(b["latex"], size, room_w)
            grp = slide.shapes.add_group_shape()
            grp.name = "Formula"
            self.d.rect(grp.shapes, x, y, cw, ch, PANEL, radius=8)
            if lab:
                self.d.textbox(grp.shapes, x + pad, y + pad, room_w, lab_h, [lab])
            top = y + pad + lab_h + (6 if lab else 0)
            eq.draw(grp.shapes, x + (cw - eq.w) / 2, top + (room_h - eq.h) / 2)
        self.footer(slide, sl)
        self.notes(slide, sl)

    def build(self) -> bytes:
        for sl in self.slides_in:
            kind = sl.get("kind")
            if kind == "title":
                self.title_slide(sl)
            elif kind == "agenda":
                self.agenda_slide(sl)
            elif kind == "section":
                self.section_slide(sl)
            elif kind == "formulas":
                self.formulas_slide(sl)
            else:  # content, summary
                self.content_slide(sl)
        if not self.slides_in:
            self.title_slide({"title": self.deck.get("title")})
        props = self.prs.core_properties
        props.title = _flat(self.deck.get("title"))[:250]
        props.subject = _flat(self.deck.get("subtitle"))[:250]
        props.author = props.last_modified_by = "Replay Board"  # not python-pptx's template author, from 2013
        props.created = props.modified = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        props.revision = 1
        props.comments = f"Made with Replay Board from the recording of {self.lecture}."[:250]
        buf = io.BytesIO()
        self.prs.save(buf)
        return buf.getvalue()


def build_pptx(job_dir: Path, deck: dict) -> bytes:
    """The deck as a 16:9 PowerPoint (bytes of a .pptx file)."""
    return Builder(Path(job_dir), deck).build()
