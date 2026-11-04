"""Slides export: the lecture as the deck a good teacher would have prepared for it, written by Muse Spark and
checked by code.

- Plan (one Muse Spark call): sections and slides that follow the lecture in order and cover all of it; each
  slide has its stretch of the lecture and the board lines and writing (or slide elements) it presents.
- Code repairs the plan: known ids only, spans in order without gaps or overlaps, and every substantive item of
  the lecture presented by some slide or skipped on purpose.
- Write (one Muse Spark call per section, at most PARALLEL at a time): each slide's blocks (bullets, display
  formulas, worked steps, callouts, tables, figures) and what a presenter would say.
- Code checks everything (limits, LaTeX, ids), draws the figures from the board's own strokes (or cuts them from
  the clean slide), adds the title, agenda, section, summary and formula-sheet slides, and saves the deck as
  export/deck.json (format 2). A part whose write call fails is made by code from the plan, and the deck says so.
- PowerPoint: app/pptx_deck.py renders the saved deck; export/slides.pptx is kept per deck and renderer version.

Writing runs in a background thread per lecture: POST /slides/write starts it (a second POST joins it), GET
slides.json reports it and never calls the model. All times are on the video's clock, like replay.json.

Strings are plain text, except bullet text, step text, table cells and callout text, which may hold short inline
maths as $...$ (pandoc's rule, as the player's Markdown: no space just inside the dollars). In those fields every
other dollar sign is escaped as \\$ ("costs \\$5"), so a renderer can tell money from maths.
"""
from __future__ import annotations

import decimal
import hashlib
import importlib
import json
import math
import re
import threading
import time
import traceback
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import cv2
import numpy as np

from . import ask
from .core.media import imread
from .core.meta_api import MetaAPIError, chat_text, parse_json_object
from .pipeline import plain_error, replace_file

DECK_RULES_VERSION = "11"  # change it when the deck rules or prompts change, so every deck is written again
FORMAT = 2
EXPORT = "export"
REASONING = "low"
PARALLEL = 4             # write calls at a time
SAID_CHARS = 2500
TITLE_CHARS = 70
SUBTITLE_CHARS = 110
BULLET_CHARS = 120
STEP_CHARS = 90
CELL_CHARS = 40
LABEL_CHARS = 40
STEPS_LABEL_CHARS = 60
CAPTION_CHARS = 140
CALLOUT_TITLE_CHARS = 60
CALLOUT_TEXT_CHARS = 240
NOTES_CHARS = 700
NOTES_SENTENCES = 4
FOCUS_CHARS = 400
LATEX_CHARS = 200
INLINE_TEX_CHARS = 60
SHORT_FRACTION = 12      # characters in each part of an inline \frac
MAX_BULLETS = 6
MAX_STEPS = 6
MAX_COLS = 6
MAX_ROWS = 8
MAX_BLOCKS = 3
MAX_TAKEAWAYS = 6
MAX_CONTENT = 60         # content slides in a deck
MAX_SECTIONS = 18        # the agenda holds 3 lists of 6
AGENDA_ITEMS = 6
SECTION_SLIDE_MIN = 3    # a section gets a section slide when it has this many content slides
FORMULAS_MIN = 3
FORMULAS_PER_SLIDE = 8
SPEECH_CHARS = 5000      # transcript sent with one slide to its writer
FIG_SIDE = 1200          # px, the long side of a board figure
FIG_MAX_UP = 6.0         # a board figure is grown with crisp strokes: at 3, test2's hexagon was placed at 87 ppi
SLIDE_FIG_MAX_UP = 2.0
FIG_PAD = 12             # board px (of an 800-px-wide board) around a figure's strokes
SLIDE_PAD = 16           # slide px around a slide figure
STRAY_AREA = 0.02        # a drawing smaller than this share of the board, never talked about, is a stray mark
DRAWINGS = ask.DRAWINGS
# Kinds a figure is made of. Marks (an arrow, a scribble, an underline, a box, pen ink on a slide) can be part of a
# figure, next to the drawing they point at, but are never one alone: live, a figure of a lone "blurry blue mark"
# was the result (test-full).
MARKS = {"arrow", "scribble", "underline", "box", "ink"}
FIGURE_KINDS = {"diagram", "graph", "figure", "chart", "drawing", "image", "picture", "photo", "illustration", "plot",
                "bar", "map", "table"}
VAGUE = {"other", "unknown", "mark", "marks"}  # kinds whose text describes a mark, not its writing (as in the player)
NO_TEXT = {"", "unreadable"}
BLOCK_TYPES = ("bullets", "equation", "steps", "figure", "table", "callout")
CALLOUT_STYLES = ("definition", "key", "note")
ONE_PER_SLIDE = ("figure", "table", "steps")
FIXES = "_fixes"  # a written slide's corrected numbers ({misread: correct}), applied to the whole deck by assemble

_num = ask._num
_flat = ask._flat
clock = ask.clock
# Characters XML (so PowerPoint) cannot hold, and lone surrogates (they cannot be written as UTF-8): a reading or a
# model's answer with one of these would make the deck, its JSON or the PowerPoint fail.
_NOT_XML = re.compile("[^\t\n\r\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")
_DOLLAR = "\ue000"  # a literal dollar sign while inline text is being cleaned


class NotReady(Exception):
    """No replay.json yet, it is being rewritten, or no deck to download yet (one plain line, HTTP 409)."""


# While a board lecture is still processed, replay.json is written in parts (SPEC "Streaming replay"): what needs the
# whole lecture (the deck, the notes, 3D views) waits for the final one.
STILL_PROCESSING = "Available when processing finishes"


def is_partial(replay) -> bool:
    """True for a partial replay.json (written while the lecture is still processed)."""
    return isinstance(replay, dict) and "partial" in replay


class _Cancelled(Exception):
    """A newer run (replay.json changed) took over this lecture's slides."""


# ---------- small helpers ----------
def safe_text(s: Any) -> str:
    """Text that JSON, UTF-8 and XML (the PowerPoint) can all hold: control characters and lone surrogates removed."""
    return _NOT_XML.sub("", str(s or ""))


def _safe_strings(x: Any) -> Any:
    """Every string in a JSON-like value made safe_text."""
    if isinstance(x, str):
        return safe_text(x)
    if isinstance(x, list):
        return [_safe_strings(v) for v in x]
    if isinstance(x, dict):
        return {k: _safe_strings(v) for k, v in x.items()}
    return x


def _is_box(b: Any) -> bool:
    return isinstance(b, list) and len(b) >= 4 and all(_num(v) for v in b[:4])


def _board_size(replay: dict) -> tuple[int, int]:
    size = (replay.get("board") or {}).get("size") or [800, 450]
    try:
        return max(1, int(size[0])), max(1, int(size[1]))
    except (TypeError, ValueError, IndexError):
        return 800, 450


def _cut(text: str, n: int) -> str:
    """At most n characters, cut at a word with an ellipsis."""
    text = _flat(text)
    if len(text) <= n:
        return text
    cut = text[:n - 1]
    sp = cut.rfind(" ")
    if sp > n * 0.5:
        cut = cut[:sp]
        words = cut.split(" ")
        while len(words) > 1 and words[-1].lower() in _DANGLING_WORDS:  # never "…triclinic and…"
            words.pop()
        cut = " ".join(words)
    return cut.rstrip(" ,;:-") + "…"


_DANGLING_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of", "on", "or",
                   "the", "to", "with"}


def _title(text: Any, n: int = TITLE_CHARS) -> str:
    """A title: plain text, at most n characters, no full stop at the end (an ellipsis or '2.71...' stays)."""
    return re.sub(r"(?<![.\d])\.$", "", _cut(plain_text(text), n))


def said_text(words: list[dict], a: float, b: float, limit: int = SAID_CHARS) -> str:
    """The words spoken from a (inclusive) to b (exclusive), as plain text; a long stretch is cut at a sentence
    (or else a word) with '...'."""
    text = " ".join(str(w["w"]).strip() for w in words if a <= w["s"] < b and str(w["w"]).strip())
    if len(text) <= limit:
        return text
    cut = text[:limit - 4]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end >= 0.6 * limit:
        return cut[:end + 1] + " ..."
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else cut).rstrip(" ,;:") + " ..."


def parse_clock(v: Any) -> float | None:
    """'1:05', '1:02:03', '[1:05]' or seconds (a number or '65.5') -> seconds; None when it is none of these."""
    if _num(v):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().strip("[]").strip()
    m = re.fullmatch(r"(\d{1,3}):([0-5]\d)(?::([0-5]\d))?(\.\d+)?", s)
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        t = int(a) * 3600 + int(b) * 60 + int(c) if c is not None else int(a) * 60 + int(b)
        return t + (float(m.group(4)) if m.group(4) else 0.0)
    try:
        t = float(s)
    except ValueError:
        return None
    return t if math.isfinite(t) else None


def _vague(read: Any) -> bool:
    """A reading that only describes a mark ("person obscuring board, no legible writing"): a vague kind, no LaTeX."""
    read = read if isinstance(read, dict) else {}
    return _flat(read.get("kind")).lower() in VAGUE and not _flat(read.get("latex")).strip("$").strip()


def _has_content(read: Any) -> bool:
    read = read if isinstance(read, dict) else {}
    return bool(_flat(read.get("latex")).strip("$").strip()) or _flat(read.get("text")).lower() not in NO_TEXT


# ---------- LaTeX and plain text ----------
_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "theta": "θ",
          "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
          "omega": "ω", "Delta": "Δ", "Gamma": "Γ", "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ",
          "Omega": "Ω", "Pi": "Π"}
_SYMBOLS = {"infty": "∞", "to": "→", "rightarrow": "→", "Rightarrow": "⇒", "leftarrow": "←", "approx": "≈",
            "times": "×", "cdot": "·", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "ne": "≠",
            "pm": "±", "circ": "°", "degree": "°", "ldots": "…", "dots": "…", "cdots": "⋯", "sim": "~",
            "in": "∈", "sum": "Σ", "int": "∫", "partial": "∂", "nabla": "∇", "propto": "∝", "div": "÷"}
_DROP = re.compile(r"\\(left|right|bigl|bigr|Bigl|Bigr|big|Big|displaystyle|textstyle|mathrm|mathbf|mathit|"
                   r"text|operatorname)(?![A-Za-z])")
_SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_CITE = re.compile(r"\[#[A-Za-z0-9]+\]|\[\d{1,3}:[0-5]\d(?::[0-5]\d)?\]")
# Commands short inline maths may use: what the PowerPoint can write as Unicode with superscripts and subscripts.
_INLINE_CMDS = set(_GREEK) | set(_SYMBOLS) | {
    "zeta", "eta", "vartheta", "iota", "kappa", "nu", "xi", "upsilon", "chi", "psi", "Xi", "Psi", "Upsilon",
    "leftrightarrow", "Leftarrow", "Leftrightarrow", "mapsto", "implies", "iff", "simeq", "cong", "equiv", "ll",
    "gg", "notin", "subset", "subseteq", "mp", "ast", "star", "prime", "angle", "perp", "parallel", "ln", "log",
    "exp", "sin", "cos", "tan", "lim", "max", "min", "frac", "tfrac", "dfrac", "sqrt", "text", "mathrm", "mathbf",
    "mathit", "left", "right", "quad"}


def _group(s: str) -> str:
    s = s.strip()
    return s if re.fullmatch(r"[\w.]+", s) else f"({s})"


def _unmath(s: str) -> str:
    """$...$ (or $$...$$) around LaTeX -> its content. Any other $ stays: it is money ('$5 or $1')."""
    out, i = [], 0
    while True:
        a = s.find("$", i)
        if a < 0:
            out.append(s[i:])
            return "".join(out)
        if s.startswith("$$", a) and s.find("$$", a + 2) > a:
            b = s.find("$$", a + 2)
            out += [s[i:a], s[a + 2:b]]
            i = b + 2
            continue
        b = s.find("$", a + 1)
        if b > a and re.search(r"[\\^_{}]", s[a + 1:b]):
            out += [s[i:a], s[a + 1:b]]
            i = b + 1
        else:
            out.append(s[i:a + 1])
            i = a + 1


def _unicode_tex(s: str) -> str:
    """LaTeX commands, powers, indices and braces -> Unicode, spacing kept."""
    if "\\" in s:
        s = re.sub(r"\\(begin|end)\{[^{}]*\}", "", s)
        s = s.replace("\\\\", "; ")
        s = _DROP.sub("", s)
        s = re.sub(r"\\[,;:! ]", " ", s)
        s = re.sub(r"\^\{?\\circ\}?", "°", s)
        for _ in range(3):  # nested fractions and roots, inside out
            s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", lambda m: f"{_group(m[1])}/{_group(m[2])}", s)
            s = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: "√" + _group(m[1]), s)
        s = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m[1]) or _SYMBOLS.get(m[1]) or m[1], s)
        s = s.replace("\\", "")
    s = re.sub(r"\^\{([0-9+\-=()n]+)\}", lambda m: m[1].translate(_SUP), s)
    s = re.sub(r"\^\{([^{}]*[^\w{}.][^{}]*)\}", lambda m: f"^({_flat(m[1])})", s)  # ^{365 × 24}: kept together
    s = re.sub(r"\^([0-9]+|n)(?![\w{(])", lambda m: m[1].translate(_SUP), s)
    return re.sub(r"_\{([^{}]*)\}", r"_\1", s).replace("{", "").replace("}", "")


def plain_text(s: Any) -> str:
    """A model's line as plain text: no citations, markdown or LaTeX (turned into Unicode where it is simple)."""
    s = _CITE.sub("", safe_text(s))
    s = _unicode_tex(_unmath(s))
    s = re.sub(r"\*\*|__|`", "", s)
    s = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s+", "", s)
    return _flat(s)


def _balanced(s: str) -> bool:
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


_DANGLING = re.compile(r"(?<!\\)[\^_]\s*(?:$|\})")  # "x^", "a_": KaTeX cannot parse them


def _left_right_ok(s: str) -> bool:
    return len(re.findall(r"\\left(?![A-Za-z])", s)) == len(re.findall(r"\\right(?![A-Za-z])", s))


def _escape_tex(s: str) -> str:
    """A bare $ or % in LaTeX escaped: KaTeX reads % as a comment ("r=100%" showed "r=100") and $ as the end."""
    return re.sub(r"(?<!\\)([$%])", r"\\\1", s)


_GREEK_NAMES = sorted(set(_GREEK) | {"zeta", "eta", "vartheta", "iota", "kappa", "nu", "xi", "upsilon", "chi", "psi",
                                     "Xi", "Psi", "Upsilon"}, key=len, reverse=True)
_BARE_GREEK = re.compile(r"(?<![\\A-Za-z])(" + "|".join(_GREEK_NAMES) + r")(?![A-Za-z])")
_TEXT_GROUP = re.compile(r"(\\(?:text|textrm|textit|textbf|mathrm|operatorname|mbox)\s*\{[^{}]*\})")


def greek_commands(s: str) -> str:
    """Greek letters written as bare words in maths -> commands: "$pi$" showed the letters p and i (seen live in a
    takeaway). Words inside \\text{...} are left alone."""
    parts = _TEXT_GROUP.split(s)
    return "".join(p if k % 2 else _BARE_GREEK.sub(r"\\\1", p) for k, p in enumerate(parts))


_SPACE_CMD = re.compile(r"\s*(?:(?<!\\)\\[ ,;:]|(?<!\\)~)\s*")
_OPEN_SIZED = re.compile(r"\\(?:left|bigl|Bigl|biggl|Biggl|big|Big|bigg|Bigg)\s*$")
_PLAIN_GAP = re.compile(r"(?:(?<=\d)|(?<=\})|(?<=\\circ))\s+(?=[A-Za-z]|\\(?:" + "|".join(_GREEK_NAMES)
                        + r")(?![A-Za-z]))")
_NUMBER_SIDE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\^\{?\\circ\}?)?")
_PLAIN_MATH = re.compile(r"[A-Za-z]{4,}|(?:\\\$)?\d+(?:\.\d+)?(?:\\%|\.\.\.|\\ldots|\\dots)?")
# A case number written before a formula on the board ("#2. y=...", seen live): a label, not maths.
_CASE_LABEL = re.compile(r"^\\#\s*\d+\s*[.:)]?\s*(?:\\[ ,;:]|~|\\quad(?![A-Za-z]))?\s*(?=\S)")
_UNIT_START = re.compile(r"\\(?:text|textrm|mathrm|mbox|operatorname|quad|qquad)(?![A-Za-z])")
_CLOSE_SIZED = re.compile(r"\\(?:right|bigr|Bigr|biggr|Biggr|big|Big|bigg|Bigg)\s*$")


def _top_level(s: str, i: int) -> bool:
    """Position i of LaTeX s is outside every group: braces, brackets, parentheses, \\left...\\right and |...|."""
    depth = bars = 0
    k = 0
    while k < i:
        c = s[k]
        if c == "\\":
            m = re.match(r"\\(left|right)(?![A-Za-z])", s[k:])
            if m:
                depth += 1 if m[1] == "left" else -1
                k += m.end()
                k += 2 if s[k:k + 1] == "\\" else 1  # its delimiter
                continue
            k += 2
            continue
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        elif c == "|":
            bars += 1
        k += 1
    return depth == 0 and bars % 2 == 0


def separate_conditions(s: str) -> str:
    """Two conditions with only a space between them ("a=b=c \\ \\alpha=\\beta=\\gamma=90^\\circ", as the board reads
    and Muse Spark copied them): TeX shows them run together, "a = b = c α = β = γ = 90°" (seen live, test2).
    Joined with ",\\quad" instead: each side must hold a relation, at the top level. A plain space (TeX ignores it)
    counts only after a number ("\\alpha=90^\\circ \\gamma=120^\\circ"): "F = m a = 10" is one statement."""
    if "\\\\" in s or "\\begin" in s:  # line breaks and environments: laid out on purpose
        return s
    spaced = [m for m in _SPACE_CMD.finditer(s) if _top_level(s, m.start())]
    plain = [m for m in _PLAIN_GAP.finditer(s) if _top_level(s, m.start())
             and not any(o.start() <= m.start() < o.end() for o in spaced)]
    cands = sorted(spaced + plain, key=lambda m: m.start())
    if not cands:
        return s
    bounds = [0] + [x for m in cands for x in (m.start(), m.end())] + [len(s)]
    segs = [s[bounds[k]:bounds[k + 1]].strip() for k in range(0, len(bounds), 2)]

    def statement(seg: str) -> bool:
        sides = [x.strip() for x in _RELATION.split(seg)]
        return len(sides) >= 2 and all(sides)

    def starts_one(seg: str) -> bool:
        """Its subject is a name ("c", "\\alpha"), not a unit or a list ("\\text{m}, t" in "x=2\\ \\text{m}, t=3")."""
        subject = _RELATION.split(seg)[0].strip()
        return "," not in subject and not _UNIT_START.match(subject)

    out = [segs[0]]
    for k, m in enumerate(cands):
        left, right = segs[k], segs[k + 1]
        joined = statement(left) and statement(right) and starts_one(right) and not left.endswith(",")
        if joined and m in plain:
            joined = bool(_NUMBER_SIDE.fullmatch(_RELATION.split(left)[-1].strip()))
        out.append(",\\quad " if joined else m.group(0))
        out.append(right)
    return "".join(out)


def grow_parentheses(s: str) -> str:
    """Parentheses around a fraction grow with it: "(1+\\frac{1}{12})^{12}" -> "\\left(1+\\frac{1}{12}\\right)^{12}".
    Written plain (as board reads often are), they stay text-sized beside a tall fraction. Unmatched ones, or ones
    split across groups, are left as written."""
    stack: list[tuple[int, int]] = []
    pairs: list[tuple[int, int]] = []
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "(" and not _OPEN_SIZED.search(s[:i]):
            stack.append((i, depth))
        elif c == ")" and not _CLOSE_SIZED.search(s[:i]):
            if not stack or stack[-1][1] != depth:
                return s
            pairs.append((stack.pop()[0], i))
        i += 1
    if stack:
        return s
    grow = [(a, b) for a, b in pairs if re.search(r"\\[dt]?frac(?![A-Za-z])", s[a:b])]
    if not grow:
        return s
    edits = sorted([(a, "\\left(") for a, _ in grow] + [(b, "\\right)") for _, b in grow], reverse=True)
    for pos, rep in edits:
        s = s[:pos] + rep + s[pos + 1:]
    return s


def simple_tex(v: str) -> bool:
    """Short inline maths the PowerPoint can write in Unicode: powers, indices, Greek letters, relations, arrows and
    short fractions; no environments, line breaks or big operators."""
    if not v or len(v) > INLINE_TEX_CHARS or not _balanced(v) or re.search(r"\\\\|&|\$", v.replace("\\$", "")):
        return False
    if _DANGLING.search(v) or not _left_right_ok(v):
        return False
    if any(cmd not in _INLINE_CMDS for cmd in re.findall(r"\\([A-Za-z]+)", v)):
        return False
    if re.search(r"\\[dt]?frac\{[^{}]*\{", v):  # a fraction inside a fraction
        return False
    return all(len(m[1]) <= SHORT_FRACTION and len(m[2]) <= SHORT_FRACTION
               for m in re.finditer(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", v))


_UNI_TEX = {"α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "δ": "\\delta", "ε": "\\varepsilon", "θ": "\\theta",
            "λ": "\\lambda", "μ": "\\mu", "π": "\\pi", "ρ": "\\rho", "σ": "\\sigma", "τ": "\\tau", "φ": "\\phi",
            "ω": "\\omega", "Γ": "\\Gamma", "Δ": "\\Delta", "Θ": "\\Theta", "Λ": "\\Lambda", "Π": "\\Pi",
            "Σ": "\\Sigma", "Φ": "\\Phi", "Ω": "\\Omega", "≠": "\\neq", "≤": "\\le", "≥": "\\ge", "≈": "\\approx",
            "→": "\\to", "⇒": "\\Rightarrow", "←": "\\leftarrow", "×": "\\times", "·": "\\cdot", "÷": "\\div",
            "±": "\\pm", "∞": "\\infty", "°": "^\\circ", "∈": "\\in", "…": "\\ldots", "⋯": "\\cdots", "−": "-",
            "∂": "\\partial", "∑": "\\sum", "∫": "\\int", "√": "\\sqrt"}
_SUP_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ]+")
_SUB_RUN = re.compile("[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]+")
_FROM_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ", "0123456789+-=()n")
_FROM_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎", "0123456789+-=()")


def unicode_to_tex(s: str) -> str:
    """Unicode maths written inside LaTeX ("α=β=90°", "x²") -> LaTeX commands ("\\alpha=\\beta=90^\\circ", "x^{2}"),
    so KaTeX and the PowerPoint read standard LaTeX (seen live: table cells like "$α=β=γ=90°$")."""
    s = _SUP_RUN.sub(lambda m: "^{" + m[0].translate(_FROM_SUP) + "}", s)
    s = _SUB_RUN.sub(lambda m: "_{" + m[0].translate(_FROM_SUB) + "}", s)
    out = []
    for i, ch in enumerate(s):
        rep = _UNI_TEX.get(ch)
        if rep is None:
            out.append(ch)
            continue
        nxt = s[i + 1] if i + 1 < len(s) else ""
        out.append(rep + " " if rep[-1].isalpha() and nxt.isascii() and nxt.isalnum() else rep)
    return "".join(out)


def _prose(latex: str) -> bool:
    """Words outside \\text{...}: a sentence put in a LaTeX field ("\\$100\\%\\$ for one year", seen live)."""
    s = re.sub(r"\\(text|mathrm|textrm|textbf|mathit|operatorname)\{[^{}]*\}", " ", latex)
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    return bool(re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}\s+[A-Za-z]{2,}", s))


_POWER_NUMBER = re.compile(r"(\^(?:\{[^{}]*\}|[A-Za-z]))\s+(?=\d)")


def explicit_products(s: str) -> str:
    """A number right after a power is multiplied by it: "D=(1-r)^n 100\\%" (as the board read it) showed as
    "(1−r)ⁿ 100%", which reads as garbled; "(1-r)^n \\cdot 100\\%" says what it means."""
    return _POWER_NUMBER.sub(r"\1 \\cdot ", s)


def display_tex(v: Any) -> str:
    """A display formula's LaTeX, cleaned: wrappers ($$, $, \\[ \\]) removed, a bare $ or % escaped. '' when it
    cannot be used: empty, longer than LATEX_CHARS, unbalanced braces, \\left/\\right or environments, a power or
    index with nothing after it, or prose."""
    s = _flat(safe_text(v))
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if len(s) > len(a) + len(b) and s.startswith(a) and s.endswith(b):
            s = s[len(a):-len(b)].strip()
            break
    s = _CASE_LABEL.sub("", greek_commands(_escape_tex(unicode_to_tex(s))))
    if not s or len(s) > LATEX_CHARS or not _balanced(s) or _prose(s):
        return ""
    if not _left_right_ok(s) or _DANGLING.search(s):
        return ""
    if sorted(re.findall(r"\\begin\{([^{}]*)\}", s)) != sorted(re.findall(r"\\end\{([^{}]*)\}", s)):
        return ""
    if not s.replace("\\$", "").strip():
        return ""
    nicer = grow_parentheses(separate_conditions(explicit_products(s)))  # typesetting only: same formula
    return nicer if len(nicer) <= LATEX_CHARS and _balanced(nicer) and _left_right_ok(nicer) else s


_LAYOUT_CMDS = re.compile(r"\\(frac|dfrac|tfrac|left|right|times|cdot|div|dots|ldots|cdots|quad|qquad|text|mathrm|"
                          r"displaystyle|big|Big|bigl|bigr)(?![A-Za-z])|\\[,;:! %$]")
_RELATION = re.compile(r"=|<|>|\\(?:approx|sim|simeq|equiv|le|leq|ge|geq|neq|ne|to)(?![A-Za-z])")
_LONE_NAME = re.compile(r"[A-Za-z](?:_\{?\w+\}?)?")


def is_formula(latex: str) -> bool:
    """A formula for the formula sheet: a relation whose sides hold a variable beyond a lone name on the left
    ("A=P(1+r)^n", "A=(1+1/\\infty)^\\infty=e"), not a worked number ("A=(1+1/12)^{12}=2.61"), a bare expression
    ("x^{300}") or a fragment ("= e")."""
    s = _LAYOUT_CMDS.sub(" ", latex).strip()
    sides = [x.strip() for x in _RELATION.split(s)]
    if len(sides) < 2 or not all(sides):
        return False
    return any(re.search(r"[A-Za-z]", x) and not (k == 0 and _LONE_NAME.fullmatch(x)) for k, x in enumerate(sides))


# ---------- the arithmetic of a formula ----------
# Sides that are plain numbers are worked out and compared. Live, a board reading lost an exponent and the deck showed
# "A = (1 + 1/(365×24×60)) = $2.71826…" (the left side is 1.0000019), put "= e" after (1 + 1/1000000)^1000000
# (2.7182805) and "2.71826" for 2.7182792 (gest-test, test-full).
ARITH_SAME = decimal.Decimal("1e-12")  # relative difference of two sides that are the same number
ARITH_NEAR = decimal.Decimal("0.01")   # two worked-out sides at most this far apart (relative) are "≈"; farther: false
FIX_ULPS = 3        # a stated number off by at most this many units of its last digit has a misread digit, ...
FIX_DIGITS = 3      # ... when it has at least this many significant digits: it is corrected
_ARITH = decimal.Context(prec=40, Emax=100_000, Emin=-100_000,
                         traps=[decimal.Overflow, decimal.DivisionByZero, decimal.InvalidOperation])
_PI = decimal.Decimal("3.141592653589793238462643383279502884197")
_E = _ARITH.exp(decimal.Decimal(1))
_SIDES = re.compile(r"=|<|>|\\(?:approx|sim|simeq|equiv|le|leq|ge|geq|neq|ne|to|rightarrow|Rightarrow|propto|"
                    r"mapsto)(?![A-Za-z])")
_ELLIPSIS_END = re.compile(r"(?:\\(?:ldots|dots|cdots|dotsc)(?![A-Za-z])|…|\.\.\.)\s*$")
_LITERAL = re.compile(r"[+-]?(\d+)(?:\.(\d+))?(\\%)?")
_ARITH_TOKEN = re.compile(r"\s*(\d+(?:\.\d+)?|\.\d+|\\[A-Za-z]+|\\[{}%]|\S)")
_SIZED = re.compile(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|big|Big|bigg|Bigg|displaystyle|textstyle)"
                    r"(?![A-Za-z])")
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_OPEN = {"(": ")", "[": "]", "{": "}", "\\{": "\\}", "\\lbrace": "\\rbrace"}
_TIMES = ("\\times", "\\cdot", "*", "×", "·", "\\ast")
_OVER = ("/", "\\div", "÷")


class _Arith:
    """One side of a formula as a number (decimal, 40 digits): digits, + − × ÷ /, powers, brackets, \\frac, \\sqrt,
    \\%, e and π. Anything else (a variable, ∞, a unit) raises ValueError."""

    def __init__(self, s: str):
        self.toks = [m.group(1) for m in _ARITH_TOKEN.finditer(s)]
        self.i = 0

    def peek(self, k: int = 0) -> str | None:
        return self.toks[self.i + k] if self.i + k < len(self.toks) else None

    def take(self) -> str:
        t = self.peek()
        if t is None:
            raise ValueError("unexpected end")
        self.i += 1
        return t

    def close(self, want: str) -> None:
        if self.take() != want:
            raise ValueError("unbalanced")

    def value(self) -> decimal.Decimal:
        v = self.expr()
        if self.peek() is not None:
            raise ValueError("unexpected token")
        return v

    def expr(self) -> decimal.Decimal:
        v = self.term()
        while self.peek() in ("+", "-", "−"):
            op = self.take()
            w = self.term()
            v = _ARITH.add(v, w) if op == "+" else _ARITH.subtract(v, w)
        return v

    def starts_atom(self, t: str | None) -> bool:
        return t is not None and (bool(_NUMBER.fullmatch(t)) or t in _OPEN or t in ("\\frac", "\\dfrac", "\\tfrac",
                                                                                     "\\sqrt", "e", "\\pi", "π"))

    def term(self) -> decimal.Decimal:
        v = self.unary()
        while True:
            t = self.peek()
            if t in _TIMES or (t == "x" and _NUMBER.fullmatch(self.peek(1) or "")):  # "365x24": the board's ×
                self.take()
                v = _ARITH.multiply(v, self.unary())
            elif t in _OVER:
                self.take()
                v = _ARITH.divide(v, self.unary())
            elif self.starts_atom(t):  # 2(1+r), 2\pi
                v = _ARITH.multiply(v, self.power())
            else:
                return v

    def unary(self) -> decimal.Decimal:
        t = self.peek()
        if t in ("-", "−"):
            self.take()
            return _ARITH.minus(self.unary())
        if t == "+":
            self.take()
            return self.unary()
        return self.power()

    def power(self) -> decimal.Decimal:
        v = self.atom()
        while self.peek() == "\\%":
            self.take()
            v = _ARITH.multiply(v, decimal.Decimal("0.01"))
        if self.peek() == "^":
            self.take()
            e = self.unary() if self.peek() in ("-", "−", "+") else self.atom()
            if abs(e) > 10 ** 15:
                raise ValueError("too big")
            v = _ARITH.power(v, e)
        return v

    def group(self) -> decimal.Decimal:
        """A command's argument: {...}, or one digit ("\\frac12")."""
        t = self.peek()
        if t == "{":
            self.take()
            v = self.expr()
            self.close("}")
            return v
        if t is not None and t.isdigit() and len(t) > 1:
            self.toks[self.i] = t[1:]
            return decimal.Decimal(t[0])
        return self.atom()

    def atom(self) -> decimal.Decimal:
        t = self.take()
        if _NUMBER.fullmatch(t):
            return decimal.Decimal(t)
        if t in _OPEN:
            v = self.expr()
            self.close(_OPEN[t])
            return v
        if t in ("\\frac", "\\dfrac", "\\tfrac"):
            a = self.group()
            return _ARITH.divide(a, self.group())
        if t == "\\sqrt":
            n = None
            if self.peek() == "[":
                self.take()
                n = self.expr()
                self.close("]")
            a = self.group()
            return _ARITH.sqrt(a) if n is None else _ARITH.power(a, _ARITH.divide(1, n))
        if t == "e":
            return _E
        if t in ("\\pi", "π"):
            return _PI
        raise ValueError(f"not a number: {t}")


def _side_value(side: str) -> tuple[decimal.Decimal, dict | None] | None:
    """(its value, the stated number when the side is just one) of a formula's side made of numbers only; None when
    it holds anything else. A stated number: its digits as written, decimals, significant digits, whether it ends
    with "…" (its digits go on) and whether it is a percentage."""
    s = side.strip()
    trunc = bool(_ELLIPSIS_END.search(s))
    s = _ELLIPSIS_END.sub("", s).strip()
    s = _SIZED.sub("", s)
    s = re.sub(r"\\(?:mathrm|text|textrm|operatorname)\s*\{\s*e\s*\}", "e", s)
    s = re.sub(r"\\[,;:! ]|~", " ", s).replace("{,}", "")
    s = re.sub(r"(?<=\d)[, ](?=\d{3}(?![\d.]))", "", s)  # 525,600 and 525 600
    s = re.sub(r"^\\\$\s*", "", s).strip()  # money: "= \$2.25"
    if not s or len(s) > LATEX_CHARS or re.search(r"\d\s+\d", s):
        return None
    try:
        v = _Arith(s).value()
    except (ValueError, IndexError, ArithmeticError):  # decimal's errors are ArithmeticErrors
        return None
    lit = _LITERAL.fullmatch(s)
    if not lit:
        return v, None
    digits = lit[1] + ("." + lit[2] if lit[2] else "")
    return v, {"digits": digits, "decimals": len(lit[2] or ""), "sig": len((lit[1] + (lit[2] or "")).lstrip("0")),
               "trunc": trunc, "pct": bool(lit[3])}


def _verdict(a: tuple, b: tuple, named: bool) -> tuple:
    """Two sides joined by = or ≈: ("same",), ("approx",) when the stated number is rounded, ("fix", k, digits)
    when side k states a number with a misread last digit, ("false",) or ("unsure",). `named`: a side holds the
    letter e, which may be a variable rather than the number e: never "false" then."""
    (va, la), (vb, lb) = a, b
    diff = abs(va - vb)
    if diff <= ARITH_SAME * max(decimal.Decimal(1), abs(va), abs(vb)):
        return ("same",)
    if la is None and lb is None:
        if diff <= ARITH_NEAR * max(abs(va), abs(vb)):
            return ("approx",)
        return ("unsure",) if named else ("false",)
    k, lit, v = (0, la, vb) if la is not None and (lb is None or la["decimals"] <= lb["decimals"]) else (1, lb, va)
    ulp = decimal.Decimal(1).scaleb(-lit["decimals"]) * (decimal.Decimal("0.01") if lit["pct"] else 1)
    tol = ulp if lit["trunc"] else ulp / 2
    if not lit["trunc"] and lit["decimals"] == 0 and lit["sig"] < FIX_DIGITS:
        tol = decimal.Decimal(0)  # a small whole number is exact: "(1+1/2)^2 = 2" is false, not rounded
    if diff <= tol * decimal.Decimal("1.000000001"):
        return ("same",) if lit["trunc"] else ("approx",)
    if (la is None or lb is None) and lit["sig"] >= FIX_DIGITS and diff <= FIX_ULPS * ulp:
        shown = v * 100 if lit["pct"] else v
        new = shown.quantize(decimal.Decimal(1).scaleb(-lit["decimals"]), rounding=decimal.ROUND_HALF_UP)
        return ("fix", k, format(new, "f"))
    return ("unsure",) if named else ("false",)


def check_arithmetic(latex: str) -> tuple[str, dict[str, str]] | None:
    """A display formula with its arithmetic checked, where its sides are numbers only: (the formula, {a misread
    number: its correction}), or None when it states something false (the formula is then left out). A rounded
    result gets ≈ instead of = (unless its digits go on, "e = 2.71828…"); a number off in its last digit or two is
    corrected, and the deck corrects it everywhere. Other formulas are returned unchanged."""
    cuts = [(m.start(), m.end()) for m in _SIDES.finditer(latex) if _top_level(latex, m.start())]
    if not cuts:
        return latex, {}
    bounds = [0] + [x for a, b in cuts for x in (a, b)] + [len(latex)]
    sides = [latex[bounds[k]:bounds[k + 1]] for k in range(0, len(bounds), 2)]
    signs = [latex[a:b] for a, b in cuts]
    values = [_side_value(x) for x in sides]
    fixes: dict[str, str] = {}
    for k, sign in enumerate(signs):
        if sign not in ("=", "\\approx") or values[k] is None or values[k + 1] is None:
            continue
        named = any(re.search(r"(?<![\\A-Za-z])e(?![A-Za-z])", x) for x in (sides[k], sides[k + 1]))
        v = _verdict(values[k], values[k + 1], named)
        if v[0] == "false":
            return None
        if v[0] == "fix":
            j = k + v[1]
            old = values[j][1]["digits"]
            if old not in sides[j]:
                return None
            sides[j] = sides[j].replace(old, v[2], 1)
            fixes[old] = v[2]
            exact = decimal.Decimal(v[2]) * (decimal.Decimal("0.01") if values[j][1]["pct"] else 1) == values[k + 1 - v[1]][0]
            v = ("same",) if values[j][1]["trunc"] or exact else ("approx",)
        if v[0] == "approx" and sign == "=":
            signs[k] = "\\approx" if sides[k + 1][:1].isspace() else "\\approx "
    out = [sides[0]]
    for sign, side in zip(signs, sides[1:]):
        out += [sign, side]
    return "".join(out), fixes


_FIX_SKIP = {"said", "version", "lecture", "image", "refs", "id", "kind", "type", "style", "size", "clip"}


def fix_numbers(x: Any, fixes: dict[str, str]) -> Any:
    """Every misread number the formulas corrected (check_arithmetic), corrected in every text of the deck: titles,
    bullets, cells, notes, other formulas. What was said (the transcript) is left as it was heard."""
    if not fixes:
        return x
    if isinstance(x, str):
        for old, new in fixes.items():
            x = re.sub(r"(?<![\d.])" + re.escape(old) + r"(?!\d)", new, x)
        return x
    if isinstance(x, list):
        return [fix_numbers(v, fixes) for v in x]
    if isinstance(x, dict):
        return {k: v if k in _FIX_SKIP else fix_numbers(v, fixes) for k, v in x.items()}
    return x


def _inline_end(s: str, i: int) -> int:
    """Index of the '$' closing inline maths opened at i, or -1 (the player's rule, pandoc's: '$5 and $10' is not
    maths). The next unescaped '$' must close it."""
    if i + 1 >= len(s) or s[i + 1].isspace() or s[i + 1] == "$":
        return -1
    j = i + 1
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "$":
            ok = not s[j - 1].isspace() and not (j + 1 < len(s) and s[j + 1].isdigit())
            return j if ok else -1
        j += 1
    return -1


def _math_spans(s: str) -> list[tuple[bool, str]]:
    """Text split into (is maths, piece). In the text pieces a literal dollar sign is _DOLLAR."""
    out: list[tuple[bool, str]] = []
    buf: list[str] = []

    def flush():
        if buf:
            out.append((False, "".join(buf)))
            buf.clear()

    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and s.startswith("\\$", i):
            buf.append(_DOLLAR)
            i += 2
            continue
        if c == "$":
            if s.startswith("$$", i):
                j = s.find("$$", i + 2)
                if j > i + 2:
                    flush()
                    out.append((True, s[i + 2:j]))
                    i = j + 2
                    continue
            j = _inline_end(s, i)
            if j < 0:  # "$ x^2 $": LaTeX with spaces inside the dollars is maths too
                k = s.find("$", i + 1)
                j = k if k > i + 1 and re.search(r"[\\^_{}]", s[i + 1:k]) and s[i + 1:k].strip() else -1
            if j > 0:
                flush()
                out.append((True, s[i + 1:j]))
                i = j + 1
                continue
            buf.append(_DOLLAR)
            i += 1
            continue
        buf.append(c)
        i += 1
    flush()
    return out


def _text_piece(v: str) -> str:
    """Text between inline maths: LaTeX commands outside the dollars -> Unicode; spacing at its ends kept."""
    core = _unicode_tex(v.strip()) if "\\" in v or "^" in v or "_{" in v else v.strip()
    core = " ".join(core.split())
    if not core:
        return " " if v.strip() != v and v else ""
    return (" " if v[:1].isspace() else "") + core + (" " if v[-1:].isspace() else "")


def _math_shown(v: str) -> str:
    """Inline maths as a reader sees it, about: its Unicode rendering."""
    return _flat(_unicode_tex(v.replace("\\$", _DOLLAR)))


def shown_len(text: str) -> int:
    """Characters a reader sees in an inline-maths field: a formula counts as its Unicode rendering ("$\\alpha^{2}$"
    is 2), "\\$" as 1. The limits of these fields (BULLET_CHARS, STEP_CHARS, CELL_CHARS, CALLOUT_TEXT_CHARS) count
    these: measured on the LaTeX source, a cell "$\\alpha=\\gamma=90^\\circ, \\beta\\neq 90^\\circ$" was 43 long and
    was dropped (test2)."""
    return sum(len(_math_shown(v)) if m else len(v) for m, v in _math_spans(text))


def inline_text(s: Any, limit: int) -> str:
    """A field that may hold short inline maths (bullet text, step text, table cells, callout text): citations and
    markdown removed, maths kept as $...$ when it is simple (else turned into Unicode), every other dollar sign
    escaped as \\$, and at most `limit` characters as shown (shown_len), cut at a word outside the maths."""
    s = _CITE.sub("", safe_text(s).replace(_DOLLAR, ""))
    s = re.sub(r"\*\*|__|`", "", s)
    s = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m[1].strip() + "$", s)
    s = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s+", "", s)
    parts: list[tuple[str, str]] = []
    for is_math, v in _math_spans(s):
        if is_math:
            v = greek_commands(re.sub(r"(?<!\\)%", r"\\%", unicode_to_tex(_flat(v))))
            if not v:
                continue
            if _PLAIN_MATH.fullmatch(v):  # a word or a number set as maths reads better as text: a hyphenated word
                # written as "$word$-$word$" showed italic letters (seen live), "$40$ grams" or "$\$2.71828...$"
                # a font change mid-sentence
                parts.append(("t", re.sub(r"\\(?:ldots|dots)$", "…", v).replace("\\%", "%").replace("\\$", _DOLLAR)))
                continue
            if simple_tex(v):
                parts.append(("m", v))
            else:  # too big for inline maths: Unicode
                parts.append(("t", _flat(_unicode_tex(v.replace("\\$", _DOLLAR)))))
            continue
        parts.append(("t", _text_piece(v)))
    # Join, cutting at `limit` characters as shown (a formula counts as its Unicode rendering, \$ as one).
    out: list[str] = []
    used = 0
    cut = False
    for kind, v in parts:
        n = len(_math_shown(v)) if kind == "m" else len(v)
        if used + n <= limit:
            out.append("$" + v + "$" if kind == "m" else v.replace(_DOLLAR, "\\$"))
            used += n
            continue
        cut = True
        room = max(0, limit - 1 - used)
        if kind == "m":
            v = _math_shown(v) if not out else ""  # maths alone too long for the field: its Unicode, cut
        piece = v[:room]
        sp = piece.rfind(" ")
        if sp > 0:
            piece = piece[:sp]
        elif out or v[:1].isspace():  # never cut inside a word, unless it is the only one
            piece = ""
        out.append(piece.replace(_DOLLAR, "\\$"))
        break
    text = " ".join("".join(out).split())
    text = re.sub(r"(…|\.\.\.|\\ldots\$|\\dots\$)\.(?=\s|$)", r"\1", text)  # "2.71828…." (seen live): one stop
    if cut:
        text = text.rstrip(" ,;:-") + "…"
    return text if text != "…" else ""


def latex_as_text(raw: Any, limit: int) -> str:
    """A LaTeX field that cannot be used as display maths, as an inline-maths text field: a sentence written in a
    LaTeX field stays a sentence (its \\$...\\$ become maths), anything else is tried as inline maths."""
    raw = _flat(safe_text(raw)).strip("$ ")
    if not raw:
        return ""
    return inline_text(raw.replace("\\$", "$") if _prose(raw) else "$" + raw + "$", limit)


def notes_text(s: Any) -> str:
    """What a presenter would say: plain text, at most NOTES_SENTENCES sentences and NOTES_CHARS characters. Inline
    maths written as $...$ by the pandoc rule is unwrapped too ("powers of $x$" kept its dollars, seen live); money
    ("$5 or $10") stays."""
    s = "".join(_math_shown(v) if m else v.replace(_DOLLAR, "$") for m, v in _math_spans(safe_text(s)))
    text = plain_text(s)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return _cut(" ".join(parts[:NOTES_SENTENCES]), NOTES_CHARS)


# ---------- the lecture's items ----------
class Lecture:
    """replay.json with what the deck needs: its items by ref, the clip, the transcript, the board's size."""

    def __init__(self, job_dir: Path, replay: dict):
        self.job_dir = job_dir
        self.replay = replay
        self.kind = "slides" if replay.get("type") == "slides" else "board"
        _, self.clip0, self.clip1 = ask._video(replay)
        self.clip1 = max(self.clip0, self.clip1)
        self.words = sorted(ask._words(replay), key=lambda w: w["s"])
        self.index = ask.item_index(replay)
        self.W, self.H = _board_size(replay)
        self.slide_size = (replay.get("slides") or {}).get("size")

    def read(self, it: ask.Item) -> dict:
        if it.kind == "element":
            return it.data
        r = it.data.get("read")
        return r if isinstance(r, dict) else {}

    def kind_of(self, it: ask.Item) -> str:
        return _flat(self.read(it).get("kind")).lower()

    def is_drawing(self, it: ask.Item) -> bool:
        if self.kind_of(it) in FIGURE_KINDS:
            return True
        return it.kind == "line" and any(isinstance(m.get("read"), dict) and
                                         _flat(m["read"].get("kind")).lower() in FIGURE_KINDS for m in it.members)

    def _piece_drawable(self, p: dict) -> bool:
        return (isinstance(p.get("image"), str) and bool(p["image"]) and _is_box(p.get("box"))
                and (self.job_dir / p["image"]).is_file())

    def drawable(self, it: ask.Item) -> bool:
        """It can be drawn as a figure: a piece (or a line's piece) with its image, a slide element with a box."""
        if it.kind == "element":
            st = it.state or {}
            return (_is_box(it.data.get("box")) and isinstance(st.get("image"), str) and bool(st["image"])
                    and (self.job_dir / st["image"]).is_file())
        return any(self._piece_drawable(p) for p in (it.members or [it.data]))

    def ref(self, raw: Any) -> str | None:
        """'#L3', 'L3', '#12', '12', '#s0e1' -> the item's ref, or None when there is no such item."""
        if not isinstance(raw, str):
            return None
        r = raw.strip().strip("[]").strip()
        r = r if r.startswith("#") else "#" + r
        if r in self.index:
            return r
        up = "#" + r[1:].upper() if re.fullmatch(r"#l\d+", r) else None
        return up if up in self.index else None

    def refs(self, raw: Any) -> list[str]:
        out: list[str] = []
        for x in raw if isinstance(raw, list) else []:
            r = self.ref(x)
            if r and r not in out:
                out.append(r)
        return out

    def coverage_items(self) -> tuple[dict[str, set[str]], set[str]]:
        """({ref: the refs that present it} for every item the deck must present or skip on purpose, the refs code
        leaves out by itself). Board: each line, and each piece not in a line, that has text or LaTeX and is not a
        vague mark; left out: writing on the board from before the clip that is never talked about (an earlier
        lesson), and small drawings never talked about (stray marks). Slides: each element with text or LaTeX, a
        repeat on the next build step counted once."""
        must: dict[str, set[str]] = {}
        auto: set[str] = set()
        if self.kind == "slides":
            prev: dict[tuple, str] = {}
            for st in (self.replay.get("slides") or {}).get("states") or []:
                seen: dict[tuple, str] = {}
                for e in st.get("elements") or [] if isinstance(st, dict) else []:
                    ref = "#" + str(e.get("id")) if isinstance(e, dict) else ""
                    if ref not in self.index:
                        continue
                    key = (_flat(e.get("kind")).lower(), _flat(e.get("text")), _flat(e.get("latex")))
                    rep = prev.get(key, ref)
                    seen[key] = rep
                    if rep != ref:
                        if rep in must:
                            must[rep].add(ref)
                    elif _has_content(e) and not _vague(e):
                        must[ref] = {ref}
                prev = seen
            return must, auto
        for ref, it in self.index.items():
            if it.kind == "piece" and it.line is not None:
                continue
            read = self.read(it)
            if not _has_content(read) or _vague(read):
                continue
            ps = it.members or [it.data]
            if not it.spoken and all(p.get("pre_existing") for p in ps):
                auto.add(ref)
                continue
            box = it.data.get("box")
            if (it.kind == "piece" and not it.spoken and self.kind_of(it) in DRAWINGS and _is_box(box)
                    and float(box[2]) * float(box[3]) < STRAY_AREA * self.W * self.H):
                auto.add(ref)
                continue
            must[ref] = {ref} | {f"#{m['id']}" for m in it.members}
        return must, auto

    def item_line(self, ref: str) -> str:
        """One item for a writer: ref, kind, what it reads, its LaTeX, when it was written or shown and talked about."""
        it = self.index[ref]
        read = self.read(it)
        kind = _flat(read.get("kind")) or "writing"
        text = _flat(read.get("text"))
        latex = _flat(read.get("latex")).strip("$").strip()
        parts = [f"{ref} {kind}"]
        if self.is_drawing(it) and self.drawable(it):
            parts[0] += " [drawing]"
        if text and text.lower() != "unreadable":
            parts.append(f'reads "{text}"')
        if latex:
            parts.append(f"LaTeX {latex}")
        if it.kind == "element":
            st = it.state or {}
            parts.append(f"on screen {clock(st.get('t_start', 0) or 0)}-{clock(st.get('t_end', st.get('t_start', 0)) or 0)}")
            if it.spoken:
                parts.append(f"talked about {ask._ranges(it.spoken)}")
        else:
            parts.append(ask._board_when(it, self.clip0))
        return " | ".join(parts)


# ---------- the plan ----------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "sections": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "t0": {"type": "string"}, "t1": {"type": "string"},
            "slides": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "t0": {"type": "string"}, "t1": {"type": "string"},
                "covers": {"type": "array", "items": {"type": "string"}},
                "focus": {"type": "string"}}}}}}},
        "takeaways": {"type": "array", "items": {"type": "string"}},
        "closing": {"type": "string"},
        "skip": {"type": "array", "items": {"type": "string"}},
    },
}

PLAN_TASK = """This request is not a student's question. Replay Board is turning this lecture into a slide deck: \
the slides a good teacher would have prepared for it, so that a student who missed the lecture can learn \
everything it taught from the deck alone. Plan that deck. Reply only with JSON in the given format. For this reply, \
ignore the answer style above: no markdown, no LaTeX, and no [m:ss] times or [#id] citations inside any text.

How to plan:
- Follow the lecture's order and cover all of it, from {start} to {end}. Sections are its parts: usually one per \
chapter; merge a chapter shorter than about a minute into a neighbour, and split a long one where the topic changes.
- One idea per slide: about one slide for every 45 to 90 seconds of real teaching (at least 3 slides for any real \
lecture, at most about 60). Greetings, logistics, jokes and pauses get no slide of their own.
- Each worked example, derivation, definition, comparison and drawing gets the slide it needs. When the lecture \
works through a series of cases (the same calculation with other numbers, or a list of types with their \
properties), plan a slide per case or two, then one slide that compares all the cases side by side, placed at the \
moment the series ends.
- Titles say what the slide teaches: "Compounding monthly gives $2.61", not "Compound interest". At most 70 \
characters, plain text, maths in Unicode (x², √2, π, ∞, →, ≈), in sentence case: a capital only for the first word \
and for names.
- t0 and t1 are m:ss times on the video's clock, as above: the stretch of the lecture a section or a slide \
presents. Sections and slides follow each other without gaps or overlaps.
- covers: the ids of the board lines and writing (or slide elements) the slide presents, as listed above ("#L3", \
"#12", "#s0e1"). Every item listed above must be in some slide's covers or in skip.
- skip: ids that carry nothing to teach: dates, scribbles, stray marks, repeats, and writing left on the board from \
an earlier lesson that this lecture does not use.
- focus: one or two sentences for the slide's writer: what the slide must get across, including the explanation the \
teacher gave aloud.
- title and subtitle: what this lecture teaches (at most 70 characters, sentence case; not the subject of writing \
left from an earlier lesson) and one line saying what it covers (at most 110 characters).
- takeaways: 3 to 6 key takeaways for the closing slide, one sentence each, at most 120 characters, stating only \
what the lecture states (no comparisons or numbers of your own); write maths in them as $...$ with LaTeX inside \
($a=b=c$, $90^\\circ$, Greek letters as commands: $\\pi$, never $pi$).
- closing: one or two sentences a presenter would say with the takeaways.
- Be faithful: only what the lecture says or shows, no facts of your own. The transcript is automatic and may \
mishear words; the board's reading (or the slides') is automatic too and may misread handwriting. Take symbols and \
names from the board (the number e is "e", not "E"), but when the board's reading and what the teacher says \
disagree about a number or a formula, follow the teacher."""


def context_messages(job_dir: Path, replay: dict) -> tuple[str, str]:
    """(the system message, the prompt cache key): Ask Muse's, word for word, so the lecture is cached once for
    Ask, the plan and every part."""
    system = safe_text(ask.SYSTEM + "\n\n" + ask.lecture_context(replay))  # unchanged unless a reading is odd
    key = f"replay-board-{job_dir.name}-{hashlib.sha256(system.encode('utf-8')).hexdigest()[:16]}"
    return system, key


def plan_messages(lec: Lecture, system: str) -> list[dict]:
    task = PLAN_TASK.replace("{start}", clock(lec.clip0)).replace("{end}", clock(lec.clip1))
    return [{"role": "system", "content": system}, {"role": "user", "content": safe_text(task)}]


def _spread(slides: list[dict], clip1: float) -> None:
    """Slides that start at the same time share the stretch up to the next start evenly."""
    i = 0
    while i < len(slides):
        j = i
        while j + 1 < len(slides) and slides[j + 1]["t0"] <= slides[i]["t0"] + 1e-6:
            j += 1
        if j > i:
            a = slides[i]["t0"]
            b = slides[j + 1]["t0"] if j + 1 < len(slides) else clip1
            step = max(0.0, b - a) / (j - i + 1)
            for k in range(i, j + 1):
                slides[k]["t0"] = a + step * (k - i)
        i = j + 1


def repair_plan(data: Any, lec: Lecture, lecture: str = "") -> dict:
    """Muse Spark's plan, checked: titles cut to size, sections and slides in time order, following each other from
    the clip's start to its end with no gaps or overlaps, known refs only, and every substantive item presented by
    a slide or skipped (a missing one joins the slide whose stretch holds its time). MetaAPIError: no slides."""
    data = data if isinstance(data, dict) else {}
    c0, c1 = lec.clip0, lec.clip1
    sections = []
    for sec in data.get("sections") if isinstance(data.get("sections"), list) else []:
        if not isinstance(sec, dict):
            continue
        slides = []
        for sl in sec.get("slides") if isinstance(sec.get("slides"), list) else []:
            if not isinstance(sl, dict):
                continue
            slides.append({"title": _title(sl.get("title")), "t0": parse_clock(sl.get("t0")),
                           "covers": lec.refs(sl.get("covers")), "focus": _cut(plain_text(sl.get("focus")), FOCUS_CHARS)})
        if not slides:
            continue
        starts = [s["t0"] for s in slides if s["t0"] is not None]
        t0 = parse_clock(sec.get("t0"))
        sections.append({"title": _title(sec.get("title")), "t0": min(starts) if starts else t0, "slides": slides})
    if not sections:
        raise MetaAPIError("the plan had no slides; try again")
    # Time order: a section or slide without a time stays after the one before it.
    last = c0
    for sec in sections:
        sec["t0"] = last if sec["t0"] is None else sec["t0"]
        last = sec["t0"]
    sections.sort(key=lambda s: s["t0"])
    flat = []
    for n, sec in enumerate(sections):
        last = sec["t0"]
        for sl in sec["slides"]:
            sl["t0"] = last if sl["t0"] is None else sl["t0"]
            last = sl["t0"]
        sec["slides"].sort(key=lambda s: s["t0"])
        for sl in sec["slides"]:
            sl["section"] = n
            flat.append(sl)
    prev = c0
    for k, sl in enumerate(flat):
        sl["t0"] = c0 if k == 0 else min(c1, max(prev, sl["t0"], c0))
        prev = sl["t0"]
    _spread(flat, c1)
    for k, sl in enumerate(flat):
        sl["t1"] = flat[k + 1]["t0"] if k + 1 < len(flat) else c1
    groups = [[sl for sl in flat if sl["section"] == n] for n in range(len(sections))]
    secs = [{"title": sections[n]["title"], "slides": g} for n, g in enumerate(groups) if g]

    def span(sec):
        return sec["slides"][-1]["t1"] - sec["slides"][0]["t0"]

    while len(secs) > MAX_SECTIONS:  # merge the shortest part into its neighbour
        i = min(range(len(secs)), key=lambda k: span(secs[k]))
        j = i - 1 if i > 0 else i + 1
        a, b = sorted((i, j))
        title = secs[i]["title"] if span(secs[i]) > span(secs[j]) else secs[j]["title"]
        secs[a:b + 1] = [{"title": title, "slides": secs[a]["slides"] + secs[b]["slides"]}]
    while sum(len(s["slides"]) for s in secs) > MAX_CONTENT:  # merge the shortest slide into its neighbour
        cands = [(sl["t1"] - sl["t0"], si, k) for si, s in enumerate(secs) if len(s["slides"]) > 1
                 for k, sl in enumerate(s["slides"])]
        _, si, k = min(cands)
        sl = secs[si]["slides"]
        j = k - 1 if k > 0 else k + 1
        a, b = sorted((k, j))
        keep = sl[j]
        merged = {**keep, "t0": sl[a]["t0"], "t1": sl[b]["t1"],
                  "covers": list(dict.fromkeys(sl[a]["covers"] + sl[b]["covers"]))}
        sl[a:b + 1] = [merged]
    for n, sec in enumerate(secs, 1):
        sec["title"] = sec["title"] or (sec["slides"][0]["title"] if sec["slides"][0]["title"] else f"Part {n}")
        sec["t0"], sec["t1"] = sec["slides"][0]["t0"], sec["slides"][-1]["t1"]
        for k, sl in enumerate(sec["slides"], 1):
            sl.pop("section", None)
            sl["title"] = sl["title"] or (sec["title"] if len(sec["slides"]) == 1 else f"{sec['title']} ({k})")
            sl["t0"], sl["t1"] = round(sl["t0"], 2), round(sl["t1"], 2)
        sec["t0"], sec["t1"] = round(sec["t0"], 2), round(sec["t1"], 2)
    # Coverage: every substantive item is presented by a slide or skipped.
    skip = lec.refs(data.get("skip"))
    must, auto = lec.coverage_items()
    covered = {r for sec in secs for sl in sec["slides"] for r in sl["covers"]}
    skipped = set(skip)
    slides = [sl for sec in secs for sl in sec["slides"]]
    n_cov = n_skip = 0
    attached = []
    for ref, by in must.items():
        if by & covered:
            n_cov += 1
            continue
        if by & skipped:
            n_skip += 1
            continue
        t = lec.index[ref].t
        sl = next((s for s in slides if s["t0"] <= t < s["t1"]), None)
        if sl is None:
            sl = min(slides, key=lambda s: min(abs(t - s["t0"]), abs(t - s["t1"])))
        sl["covers"].append(ref)
        attached.append(ref)
    takeaways = [x for x in (inline_text(t, BULLET_CHARS) for t in data.get("takeaways") or []
                             if isinstance(t, str)) if x][:MAX_TAKEAWAYS]
    return {"title": _title(data.get("title")) or _title(lecture) or "Lecture",
            "subtitle": _cut(plain_text(data.get("subtitle")), SUBTITLE_CHARS),
            "sections": secs, "takeaways": takeaways, "closing": notes_text(data.get("closing")), "skip": skip,
            "coverage": {"items": len(must), "covered": n_cov, "skipped": n_skip, "attached": len(attached),
                         "attached_refs": attached, "left_out_by_code": len(auto - covered - skipped)}}


# ---------- writing a part ----------
_ITEM = {"type": "object", "properties": {"text": {"type": "string"}, "level": {"type": "integer"}}}
_STEP = {"type": "object", "properties": {"latex": {"type": "string"}, "text": {"type": "string"}}}
_STRS = {"type": "array", "items": {"type": "string"}}
WRITE_SCHEMA = {
    "type": "object",
    "properties": {"slides": {"type": "array", "items": {"type": "object", "properties": {
        "title": {"type": "string"}, "t": {"type": "string"}, "notes": {"type": "string"},
        "blocks": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string"},
            "items": {"type": "array", "items": _ITEM},
            "latex": {"type": "string"}, "label": {"type": "string"}, "refs": _STRS,
            "steps": {"type": "array", "items": _STEP},
            "caption": {"type": "string"}, "header": _STRS, "rows": {"type": "array", "items": _STRS},
            "style": {"type": "string"}, "title": {"type": "string"}, "text": {"type": "string"}}}}}}}},
}

WRITE_TASK = """This request is not a student's question. Replay Board is writing a slide deck of this lecture, \
planned below: the slides a good teacher would have prepared, so that a student who missed the lecture can learn it \
from them. Write the slides of one part. Reply only with JSON in the given format. For this reply, ignore the \
answer style above: no markdown, and no [m:ss] times or [#id] citations inside any text.

- Write exactly the planned slides of this part, in order, one for each. You may sharpen a title (at most 70 \
characters, plain text, Unicode maths, sentence case) but keep each slide's idea.
- Give each slide 1 to 3 blocks, chosen for its content:
  - equation: one display formula in "latex" (at most 200 characters), a short "label" (at most 40 characters) and \
the ids it comes from in "refs". Typeset the board's formulas as written; fix a misreading (a dropped bracket or \
exponent, a misread symbol or digit) from what the teacher says and the context, but never invent. Write each \
formula whole: a result written on its own ("= 2.25", "= e") joins the formula it completes, in one block. A \
formula states something (it has =, <, ≈ or →): a lone expression such as x³ is not an equation block, write it \
inline in a bullet. Two statements in one formula are separated with ",\\quad" (a=b=c,\\quad \\alpha=90^\\circ). \
The formula of one case in a series of cases states all its conditions, completed from the teacher's words. Use = \
only for exact results; a rounded result uses \\approx. When the teacher says that informal board notation means \
something standard (a limit written with ∞ in place of n), write the standard notation (\\lim_{n\\to\\infty}).
  - steps: a worked example or derivation in 2 to 6 steps, with the lecture's own numbers and only the calculations \
the lecture does. Each step has "latex", "text" (at most 90 characters) or both; "label" names the example.
  - callout: a definition (style "definition"), a key result ("key") or an aside or warning ("note"), with a short \
"title", the "text", and "latex" when it has a formula.
  - table: a comparison or a set of cases with their properties: "header" (2 to 6 columns), "rows" (1 to 8, each \
with as many cells as the header), cells at most 40 characters, and a "caption". A table never repeats what a \
figure on the same slide shows.
  - figure: a drawing, graph or diagram from the lecture, shown as a picture: "refs" are ids of items marked \
[drawing] (with the labels and arrows drawn on it, if any), and the "caption" says what it shows. A slide that \
presents a [drawing] shows it.
  - bullets: 1 to 6 short points in "items" (level 0, or 1 for a point under the one before), aim for 12 words or \
fewer (at most 120 characters). They carry the explanation the teacher gave aloud: why, not only what. Say what \
a formula's letters stand for, as the lecture uses them, in a bullet or the notes.
  At most one figure, one table and one steps block per slide. Fill only the fields of the block's type and leave \
the others empty ("" or []). Labels, captions and callout titles are in sentence case, starting with a capital.
- Numbers: no arithmetic, comparisons or conclusions of your own anywhere (bullets, tables, steps, notes). A \
difference, a trend or a comparison appears only as the lecture states it; a rounded number is written with ≈.
- Maths: in bullet text, step text, table cells and callout text, write short maths inline as $...$ (powers, \
indices, Greek letters, relations, arrows, short fractions); anything bigger goes in an equation, steps or callout \
"latex" field. In those four fields a bare $ always starts maths, so write money there with a backslash: \\$2.61. \
Titles, labels, captions and notes are plain text with Unicode maths (x², √2, π, →, ≈), and money there is just \
$2.61. "latex" fields hold LaTeX only, never a sentence. In maths, Greek letters and symbols are LaTeX commands \
($\\pi$, $\\alpha$, $\\infty$), never bare words.
- Each slide adds something new: within this part, do not repeat a formula, table or point an earlier slide already \
shows, unless the slide compares it with something new.
- t: the m:ss moment, inside the slide's stretch, where a student should start watching for this slide.
- notes: what a presenter would say with the slide: 1 to 4 sentences of plain text.
- Be faithful: only what the lecture says or shows; no facts, examples or numbers of your own. The transcript is \
automatic and may mishear words; take symbols and names from the board (or the slides) (the number e is "e", not \
"E"). The board's reading is automatic too and may misread handwriting, digits included: when a reading and what \
the teacher says disagree (the teacher says "x squared" where the board reads x^z, or "infinity" where it reads a \
number), follow the teacher."""


def plan_outline(plan: dict) -> str:
    lines = ["# The deck, as planned", f"Deck title: {plan['title']}"]
    for n, sec in enumerate(plan["sections"], 1):
        lines.append(f'Part {n}: "{sec["title"]}" ({clock(sec["t0"])}-{clock(sec["t1"])})')
        for k, sl in enumerate(sec["slides"], 1):
            lines.append(f'  {n}.{k} "{sl["title"]}" ({clock(sl["t0"])}-{clock(sl["t1"])})')
    return "\n".join(lines)


def section_brief(lec: Lecture, plan: dict, si: int) -> str:
    sec = plan["sections"][si]
    n = len(plan["sections"])
    lines = [f'# Write part {si + 1} of {n}: "{sec["title"]}" ({clock(sec["t0"])}-{clock(sec["t1"])}), '
             f'{len(sec["slides"])} slide{"s" if len(sec["slides"]) != 1 else ""}']
    for k, sl in enumerate(sec["slides"], 1):
        lines += ["", f'## Slide {si + 1}.{k}: "{sl["title"]}" ({clock(sl["t0"])}-{clock(sl["t1"])})']
        if sl.get("focus"):
            lines.append(f"Focus: {sl['focus']}")
        if sl["covers"]:
            lines.append("It presents:")
            lines += [f"- {lec.item_line(r)}" for r in sl["covers"]]
        else:
            lines.append("It presents no writing: it is about what was said.")
        speech = ask.speech_lines(lec.words, sl["t0"], sl["t1"] - 1e-3)
        text = "\n".join(speech)
        if len(text) > SPEECH_CHARS:
            text = text[:SPEECH_CHARS].rsplit("\n", 1)[0] + "\n..."
        lines.append(f"What was said ({clock(sl['t0'])}-{clock(sl['t1'])}):" if text else "Nothing was said then.")
        if text:
            lines.append(text)
    return "\n".join(lines)


def write_messages(lec: Lecture, system: str, plan: dict, si: int) -> list[dict]:
    user = WRITE_TASK + "\n\n" + plan_outline(plan) + "\n\n" + section_brief(lec, plan, si)
    return [{"role": "system", "content": system}, {"role": "user", "content": safe_text(user)}]


# ---------- checking what Muse Spark wrote ----------
def _bullets(items: list[dict]) -> dict | None:
    out = []
    for it in items:
        text = inline_text(it.get("text"), BULLET_CHARS)
        if text:
            level = it.get("level")
            out.append({"text": text, "level": 1 if _num(level) and level >= 1 and out else 0})
    return {"type": "bullets", "items": out[:MAX_BULLETS]} if out else None


def figure_refs(lec: Lecture, refs: list[str]) -> list[str]:
    """The refs a figure can draw; [] unless one of them is a drawing (a figure is never a picture of writing)."""
    ok = [r for r in refs if lec.drawable(lec.index[r])]
    return ok if any(lec.is_drawing(lec.index[r]) for r in ok) else []


_NOT_CAPITALISED = {"log", "sin", "cos", "tan", "sec", "csc", "cot", "exp", "lim", "max", "min", "mod", "det", "gcd",
                    "lcm", "arg", "mol", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan"}


def capital(s: str) -> str:
    """A label, caption or callout title starting with a capital, like the deck's titles: live, a formula sheet mixed
    "compound interest formula" with "Yearly compounding". Only a lower-case first word of three letters or more is
    changed: never a symbol ("e as a limit", "x-axis"), a name in mixed case ("pH") or a function ("ln x", "sin")."""
    m = re.match(r"([a-z]{3,})(?![A-Za-z0-9_(])", s)
    if not m or m[1] in _NOT_CAPITALISED:
        return s
    return s[0].upper() + s[1:]


def _formula(raw: Any, fixes: dict[str, str] | None) -> str | None:
    """A display formula checked (display_tex, check_arithmetic): its LaTeX, '' when it cannot be used, None when it
    states something false. The numbers it corrects go into `fixes`."""
    latex = display_tex(raw)
    if not latex:
        return ""
    checked = check_arithmetic(latex)
    if checked is None:
        return None
    if fixes is not None:
        fixes.update(checked[1])
    return checked[0]


def map_block(lec: Lecture, b: Any, fixes: dict[str, str] | None = None) -> dict | None:
    """One flat block of the model's answer -> a block of the deck format, checked; None when nothing usable. A
    formula whose arithmetic is false is left out (a step or callout keeps its words); numbers it corrects go into
    `fixes`."""
    if not isinstance(b, dict):
        return None
    kind = _flat(b.get("type")).lower()
    refs = lec.refs(b.get("refs"))
    if kind == "bullets":
        items = b.get("items") if isinstance(b.get("items"), list) else []
        return _bullets([x for x in items if isinstance(x, dict)])
    if kind == "equation":
        latex = _formula(b.get("latex"), fixes)
        if not latex:
            return None
        return {"type": "equation", "latex": latex, "label": capital(_title(b.get("label"), LABEL_CHARS)),
                "refs": refs}
    if kind == "steps":
        steps = []
        for s in b.get("steps") if isinstance(b.get("steps"), list) else []:
            if not isinstance(s, dict):
                continue
            latex = _formula(s.get("latex"), fixes)
            text = inline_text(s.get("text"), STEP_CHARS)
            if latex == "" and not text and _flat(s.get("latex")):  # unusable LaTeX: its text instead
                text = latex_as_text(s.get("latex"), STEP_CHARS)
            if latex or text:
                steps.append({"latex": latex or "", "text": text})
        steps = steps[:MAX_STEPS]
        label = capital(_title(b.get("label"), STEPS_LABEL_CHARS))
        if len(steps) >= 2:
            return {"type": "steps", "label": label, "items": steps}
        if steps and steps[0]["latex"]:
            return {"type": "equation", "latex": steps[0]["latex"], "label": _title(label, LABEL_CHARS), "refs": refs}
        return _bullets([{"text": steps[0]["text"], "level": 0}]) if steps else None
    if kind == "figure":
        refs = figure_refs(lec, refs)
        if not refs:
            return None
        return {"type": "figure", "refs": refs, "caption": capital(_cut(plain_text(b.get("caption")), CAPTION_CHARS)),
                "image": "", "size": [0, 0]}
    if kind == "table":
        header = [inline_text(h, CELL_CHARS) for h in (b.get("header") or []) if isinstance(h, str)][:MAX_COLS]
        while header and not header[-1]:
            header.pop()
        if len(header) < 2:
            return None
        rows = []
        for row in b.get("rows") if isinstance(b.get("rows"), list) else []:
            if not isinstance(row, list):
                continue
            cells = [inline_text(c, CELL_CHARS) if isinstance(c, (str, int, float)) else "" for c in row]
            cells = (cells + [""] * len(header))[:len(header)]
            if any(cells):
                rows.append(cells)
        if not rows:
            return None
        return {"type": "table", "header": header, "rows": rows[:MAX_ROWS],
                "caption": capital(_cut(plain_text(b.get("caption")), CAPTION_CHARS))}
    if kind == "callout":
        style = _flat(b.get("style")).lower()
        style = style if style in CALLOUT_STYLES else "key"
        latex = _formula(b.get("latex"), fixes)
        text = inline_text(b.get("text"), CALLOUT_TEXT_CHARS)
        if latex == "" and not text and _flat(b.get("latex")):
            text = latex_as_text(b.get("latex"), CALLOUT_TEXT_CHARS)
        if not latex and not text:
            return None
        return {"type": "callout", "style": style, "title": capital(_title(b.get("title"), CALLOUT_TITLE_CHARS)),
                "text": text, "latex": latex or ""}
    return None


def check_blocks(blocks: list[dict]) -> list[dict]:
    """At most MAX_BLOCKS, and at most one figure, one table and one steps block: later ones are dropped."""
    out, seen = [], set()
    for b in blocks:
        if b["type"] in ONE_PER_SLIDE:
            if b["type"] in seen:
                continue
            seen.add(b["type"])
        out.append(b)
    return out[:MAX_BLOCKS]


def fallback_blocks(lec: Lecture, covers: list[str], focus: str = "") -> list[dict]:
    """A slide's blocks made by code from what it presents: its formulas as equations (two at most), its other
    readable items as bullets, a drawing as a figure if there is room."""
    eqs, points, drawings = [], [], []
    for r in covers:
        it = lec.index[r]
        read = lec.read(it)
        if lec.is_drawing(it) or lec.kind_of(it) in MARKS | DRAWINGS:
            if lec.drawable(it):
                drawings.append(r)  # a mark only joins a figure of a drawing
            continue
        latex = _formula(_flat(read.get("latex")).strip("$"), None)
        if latex is None:  # its arithmetic is false (a misreading): left out, its reading too
            continue
        text = _flat(read.get("text"))
        if latex and len(eqs) < 2:
            eqs.append({"type": "equation", "latex": latex, "label": "", "refs": [r]})
        elif text and text.lower() not in NO_TEXT and not _vague(read):
            points.append({"text": text, "level": 0})
        elif latex:
            points.append({"text": "$" + latex + "$" if simple_tex(latex) else plain_text("$" + latex + "$"),
                           "level": 0})
    blocks: list[dict] = list(eqs)
    b = _bullets(points)
    if b:
        blocks.append(b)
    if drawings and len(blocks) < MAX_BLOCKS:
        refs = figure_refs(lec, drawings)
        if refs:
            blocks.append({"type": "figure", "refs": refs, "caption": "", "image": "", "size": [0, 0]})
    if not blocks and focus:
        blocks.append({"type": "bullets", "items": [{"text": inline_text(focus, BULLET_CHARS), "level": 0}]})
    return check_blocks(blocks)


def content_slide(lec: Lecture, planned: dict, section: str, written: Any) -> dict:
    """A planned slide with what Muse Spark wrote for it (None: made by code from the plan)."""
    t0, t1 = planned["t0"], planned["t1"]
    w = written if isinstance(written, dict) else None
    blocks = []
    fixes: dict[str, str] = {}
    if w is not None:
        blocks = check_blocks([x for x in (map_block(lec, b, fixes) for b in (w.get("blocks") or [])
                                           if isinstance(w.get("blocks"), list)) if x])
    if not blocks:
        fixes = {}
        blocks = fallback_blocks(lec, planned["covers"], planned.get("focus", ""))
    t = parse_clock(w.get("t")) if w is not None else None
    t = t if t is not None and t0 <= t < t1 else t0
    return {"id": "", "kind": "content", "title": (_title(w.get("title")) if w is not None else "") or planned["title"],
            "section": section, "t0": t0, "t1": t1, "t": round(t, 2), "blocks": blocks,
            "notes": notes_text(w.get("notes")) if w is not None else "", "said": said_text(lec.words, t0, t1),
            FIXES: fixes}


# ---------- figures ----------
def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(1e-6, b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _dark(replay: dict) -> bool:
    """A chalkboard. The board's mode decides when it has one: a whiteboard filmed in dim light can measure darker
    than mid-grey (test2's is #8f8d8d), and inverting its black ink would make it white on the white slide."""
    mode = (replay.get("board") or {}).get("mode")
    if mode in ("white", "chalk"):
        return mode == "chalk"
    b, g, r = (float(v) for v in ask._bg(replay))
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0 < 0.5


INK_ALPHA = 40 / 255  # a piece's pixel is ink above this opacity
BAND_GAP = 2          # empty lines (of an 800-px board) between two bands of a figure's ink
# A band of a figure's ink is other writing when this share of it lies in that writing's box and this share is that
# writing's own ink. Measured on test2's monoclinic cell: its words 37% and 59% in the box, 10% and 17% shared ink;
# the cell itself 6.5% and 1.8%; a label under a heading's large box 100% but no shared ink.
OTHER_IN_BOX = 0.3
OTHER_INK = 0.05


def _other_writing(lec: Lecture, ids: set, x0: int, y0: int, w: int, h: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(its box, its ink grown by a pixel)] as masks of a figure's crop, for each other piece of readable writing
    (words or formulas: not labels, drawings or marks) that overlaps the crop."""
    out = []
    for p in (lec.replay.get("board") or {}).get("pieces") or []:
        if not isinstance(p, dict) or p.get("id") in ids or not lec._piece_drawable(p):
            continue
        read = p.get("read") if isinstance(p.get("read"), dict) else {}
        if (_flat(read.get("kind")).lower() in DRAWINGS | MARKS | FIGURE_KINDS | {"label"}
                or not _has_content(read) or _vague(read)):
            continue
        bx, by, bw, bh = ask._box(p["box"])
        x, y = bx - x0, by - y0
        if x >= w or y >= h or x + bw <= 0 or y + bh <= 0:
            continue
        img = ask._bgra(lec.job_dir, p["image"])
        if img is None:
            continue
        ih, iw = img.shape[:2]
        box = np.zeros((h, w), bool)
        box[max(0, y):max(0, y + bh), max(0, x):max(0, x + bw)] = True
        ink = np.zeros((h, w), np.uint8)
        ax0, ay0, ax1, ay1 = max(0, x), max(0, y), min(w, x + iw), min(h, y + ih)
        if ax1 > ax0 and ay1 > ay0:
            ink[ay0:ay1, ax0:ax1] = img[ay0 - y:ay1 - y, ax0 - x:ax1 - x, 3] > 40
        out.append((box, cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0))
    return out


def _bands(profile: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """[start, end) of the runs of a profile's non-zero entries, joined across fewer than `gap` zeros."""
    idx = np.flatnonzero(profile)
    if not len(idx):
        return []
    out = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if i - out[-1][1] - 1 >= gap:
            out.append([int(i), int(i)])
        else:
            out[-1][1] = int(i)
    return [(a, b + 1) for a, b in out]


def drawing_region(ink: np.ndarray, others: list[tuple[np.ndarray, np.ndarray]], gap: int):
    """The part of a figure's crop that is the drawing, when board capture put nearby writing into its pieces (test2:
    "MONOCLINIC a≠b≠c" and the cell's angles came out under the monoclinic cell, cut off). Across the rows, then the
    columns: bands of ink at least `gap` empty lines apart; a band is other writing when enough of its ink lies in
    another piece's box and is that piece's own ink (OTHER_IN_BOX, OTHER_INK). The drawing is the biggest band that
    is not, with
    its neighbours out to the first band of other writing on each side (beyond it may be more of that writing).
    (rows, columns) slices; None when all of it is the drawing; "none" when none of it is."""
    if not others:
        return None
    region = [slice(0, ink.shape[0]), slice(0, ink.shape[1])]
    changed = False
    for axis in (0, 1):
        sub = ink[region[0], region[1]]
        prof = sub.sum(axis=1 - axis)
        bands = _bands(prof, gap)
        if len(bands) < 2:
            continue

        def other(a: int, b: int) -> bool:
            cut = (slice(a, b), slice(None)) if axis == 0 else (slice(None), slice(a, b))
            band = sub[cut]
            n = int(band.sum())
            for box, own in others:
                if n and (band & box[region[0], region[1]][cut]).sum() >= OTHER_IN_BOX * n \
                        and (band & own[region[0], region[1]][cut]).sum() >= max(3, OTHER_INK * n):
                    return True
            return False

        flags = [other(a, b) for a, b in bands]
        if not any(flags):
            continue
        mine = [k for k in range(len(bands)) if not flags[k]]
        if not mine:
            return "none"
        main = max(mine, key=lambda k: int(prof[bands[k][0]:bands[k][1]].sum()))
        lo = hi = main
        while lo > 0 and not flags[lo - 1]:
            lo -= 1
        while hi + 1 < len(bands) and not flags[hi + 1]:
            hi += 1
        start = region[axis].start
        region[axis] = slice(start + bands[lo][0], start + bands[hi][1])
        changed = True
    return (region[0], region[1]) if changed else None


def board_figure(lec: Lecture, refs: list[str]) -> np.ndarray | None:
    """The referenced pieces (a line's pieces) drawn at their places on a transparent background, cropped to them
    (padded), grown smoothly to about FIG_SIDE px on the long side (at most FIG_MAX_UP times) with crisp strokes.
    Ink on a dark board has its lightness inverted (hue kept) for a white slide. BGRA."""
    pieces, seen = [], set()
    for r in refs:
        it = lec.index.get(r)
        if it is None or it.kind == "element":
            continue
        for p in it.members or [it.data]:
            if p.get("id") not in seen and lec._piece_drawable(p):
                seen.add(p.get("id"))
                img = ask._bgra(lec.job_dir, p["image"])
                if img is not None:
                    pieces.append((p, img))
    if not pieces:
        return None
    pad = max(2, round(FIG_PAD * lec.W / 800))
    boxes = [ask._box(p["box"]) for p, _ in pieces]
    x0, y0 = min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad
    x1 = max(b[0] + b[2] for b in boxes) + pad
    y1 = max(b[1] + b[3] for b in boxes) + pad
    w, h = x1 - x0, y1 - y0
    col = np.zeros((h, w, 3), np.float32)  # premultiplied
    alpha = np.zeros((h, w), np.float32)
    for (p, img), (bx, by, _, _) in zip(pieces, boxes):
        ih, iw = img.shape[:2]
        x, y = bx - x0, by - y0
        a = img[..., 3].astype(np.float32) / 255.0
        c = img[..., :3].astype(np.float32)
        col[y:y + ih, x:x + iw] = c * a[..., None] + col[y:y + ih, x:x + iw] * (1 - a[..., None])
        alpha[y:y + ih, x:x + iw] = a + alpha[y:y + ih, x:x + iw] * (1 - a)
    keep = drawing_region(alpha > INK_ALPHA, _other_writing(lec, seen, x0, y0, w, h),
                          max(2, round(BAND_GAP * lec.W / 800)))
    if keep is not None:
        if keep == "none":
            return None
        mask = np.zeros_like(alpha, bool)
        mask[keep[0], keep[1]] = True
        alpha[~mask] = 0
        col[~mask] = 0
        ys, xs = np.nonzero(alpha > INK_ALPHA)
        if not len(ys):
            return None
        sy = slice(max(0, ys.min() - pad), min(h, ys.max() + pad + 1))
        sx = slice(max(0, xs.min() - pad), min(w, xs.max() + pad + 1))
        alpha, col = alpha[sy, sx].copy(), col[sy, sx].copy()
        h, w = alpha.shape
    ink = alpha > 0
    color = np.zeros_like(col)
    color[ink] = col[ink] / alpha[ink][:, None]
    if _dark(lec.replay):
        hls = cv2.cvtColor(np.clip(color + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HLS)
        hls[..., 1] = 255 - hls[..., 1]
        color = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR).astype(np.float32)
    # Colour defined around the strokes too (normalised blur), so growing them never pulls in black.
    num = cv2.GaussianBlur(color * alpha[..., None], (0, 0), 2.0)
    den = cv2.GaussianBlur(alpha, (0, 0), 2.0)[..., None]
    fill = np.where(den > 1e-4, num / np.maximum(den, 1e-4), 0.0)
    color = np.where(ink[..., None], color, fill).astype(np.float32)
    k = min(FIG_MAX_UP, FIG_SIDE / max(w, h))
    size = (max(1, round(w * k)), max(1, round(h * k)))
    if k > 1.001:
        color = cv2.resize(color, size, interpolation=cv2.INTER_CUBIC)
        a = cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)
        s = 0.5 * k
        a = cv2.GaussianBlur(a, (0, 0), s)
        d = min(0.4, max(0.08, 0.3 / s))
        alpha = _smoothstep(0.42 - d, 0.42 + d, a)  # round the steps of the grown pixels, keep the edges sharp
    elif k < 0.999:
        color = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    out = np.zeros((size[1], size[0], 4), np.uint8)
    out[..., :3] = np.clip(color + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def slide_figure(lec: Lecture, refs: list[str]) -> np.ndarray | None:
    """The elements' union box (padded) cut from the clean slide: the latest of their slide states that has an
    image, with the elements of the other states found there by what they say. Grown at most SLIDE_FIG_MAX_UP
    times toward FIG_SIDE. BGR (opaque)."""
    its = [lec.index[r] for r in refs if r in lec.index and lec.index[r].kind == "element" and lec.drawable(lec.index[r])]
    if not its:
        return None
    state = max((it.state for it in its), key=lambda st: float(st.get("t_start") or 0.0))
    els = {(_flat(e.get("kind")).lower(), _flat(e.get("text")), _flat(e.get("latex"))): e
           for e in state.get("elements") or [] if isinstance(e, dict)}
    boxes = []
    for it in its:
        if it.state is state:
            boxes.append(it.data["box"])
            continue
        e = els.get((_flat(it.data.get("kind")).lower(), _flat(it.data.get("text")), _flat(it.data.get("latex"))))
        if e is not None and _is_box(e.get("box")):
            boxes.append(e["box"])
    if not boxes:
        return None
    x0 = min(float(b[0]) for b in boxes)
    y0 = min(float(b[1]) for b in boxes)
    x1 = max(float(b[0]) + float(b[2]) for b in boxes)
    y1 = max(float(b[1]) + float(b[3]) for b in boxes)
    img = ask._slide_crop(lec.job_dir, state, [x0, y0, x1 - x0, y1 - y0], lec.slide_size, SLIDE_PAD)
    if img is None or not img.size:
        return None
    h, w = img.shape[:2]
    k = min(SLIDE_FIG_MAX_UP, FIG_SIDE / max(h, w))
    if abs(k - 1) > 1e-3:
        img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))),
                         interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    return img


def render_figure(lec: Lecture, refs: list[str]) -> np.ndarray | None:
    return slide_figure(lec, refs) if lec.kind == "slides" else board_figure(lec, refs)


def figure_name(slide_id: str) -> str:
    """export/fig_NNN.png, numbered by its slide (a content slide has at most one figure)."""
    return f"{EXPORT}/fig_{int(slide_id.lstrip('s')):03d}.png"


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError("could not encode a figure")
    return buf.tobytes()


# ---------- the deck ----------
def _bullet_lists(texts: list[str]) -> list[dict]:
    """Bullet lists of at most AGENDA_ITEMS items each, about equally long."""
    if not texts:
        return []
    n = math.ceil(len(texts) / AGENDA_ITEMS)
    size = math.ceil(len(texts) / n)
    return [{"type": "bullets", "items": [{"text": inline_text(t, BULLET_CHARS), "level": 0}
                                          for t in texts[i:i + size]]} for i in range(0, len(texts), size)]


def layout(plan: dict) -> dict:
    """The deck's slide ids, fixed once the plan is known: title, agenda, then per section its section slide (when it
    has SECTION_SLIDE_MIN content slides) and its content slides, then the summary; formula sheets come after."""
    ids = iter(f"s{n}" for n in range(1, 10 ** 6))
    out = {"title": next(ids), "agenda": next(ids) if len(plan["sections"]) >= 2 else None, "sections": []}
    for sec in plan["sections"]:
        head = next(ids) if len(sec["slides"]) >= SECTION_SLIDE_MIN else None
        out["sections"].append({"head": head, "slides": [next(ids) for _ in sec["slides"]]})
    out["summary"] = next(ids)
    out["next"] = int(out["summary"][1:]) + 1
    return out


def _slide(sid: str, kind: str, title: str, section: str = "", t0: float | None = None, t1: float | None = None,
           t: float | None = None, blocks: list | None = None, notes: str = "", said: str = "") -> dict:
    return {"id": sid, "kind": kind, "title": title, "section": section, "t0": t0, "t1": t1, "t": t,
            "blocks": blocks or [], "notes": notes, "said": said}


def formula_slides(content: list[dict], first_id: int) -> list[dict]:
    """The formula sheet: every distinct display formula of the deck (equations, and callouts' formulas) with its
    label, when there are at least FORMULAS_MIN, at most FORMULAS_PER_SLIDE per slide. Worked numbers, bare
    expressions and fragments are left out (is_formula): live, they made the sheet 11 lines long (test-full)."""
    seen, eqs = set(), []
    for sl in content:
        for b in sl["blocks"]:
            if b["type"] == "equation":
                latex, label, refs = b["latex"], b["label"], b["refs"]
            elif b["type"] == "callout" and b["latex"]:
                latex, label, refs = b["latex"], b["title"], []
            else:
                continue
            key = re.sub(r"\s+", "", latex)
            if key in seen or not is_formula(latex):
                continue  # a repeat; a worked number, bare expression or fragment means nothing on a sheet
            seen.add(key)
            eqs.append({"type": "equation", "latex": latex, "label": _title(label or sl["title"], LABEL_CHARS),
                        "refs": list(refs)})
    if len(eqs) < FORMULAS_MIN:
        return []
    out = []
    for n, i in enumerate(range(0, len(eqs), FORMULAS_PER_SLIDE)):
        title = "Formula sheet" if n == 0 else "Formula sheet (continued)"
        out.append(_slide(f"s{first_id + n}", "formulas", title, blocks=eqs[i:i + FORMULAS_PER_SLIDE],
                          notes="The lecture's formulas on one sheet, in the order they came up." if n == 0 else ""))
    return out


def assemble(lec: Lecture, plan: dict, lay: dict, written: list[list[dict] | None], version: str, lecture: str,
             degraded: list[str], final: bool) -> dict:
    """The deck (format 2). `written`: each section's content slides, or None while it is being written. Unless
    `final`, the deck holds the slides finished so far, in order (title, agenda, the sections done before the first
    one still being written), without the summary and formula sheet."""
    secs = plan["sections"]
    lead = plan["subtitle"] or plan["title"]
    lead = lead if lead.endswith((".", "!", "?", "…")) else lead + "."  # a subtitle may end with its own full stop
    slides = [_slide(lay["title"], "title", plan["title"],
                     notes=_cut(f"{lead} The lecture runs {clock(lec.clip1 - lec.clip0)}"
                                f" and the deck follows it in {len(secs)} part{'s' if len(secs) != 1 else ''}.",
                                NOTES_CHARS))]
    if lay["agenda"]:
        slides.append(_slide(lay["agenda"], "agenda", "Agenda", blocks=_bullet_lists([s["title"] for s in secs]),
                             notes=f"The lecture has {len(secs)} parts."))
    content: list[dict] = []
    fixes: dict[str, str] = {}
    for n, (sec, ids, done) in enumerate(zip(secs, lay["sections"], written), 1):
        if done is None:
            if not final:
                break
            done = [content_slide(lec, sl, sec["title"], None) for sl in sec["slides"]]
        if ids["head"]:
            slides.append(_slide(ids["head"], "section", sec["title"], sec["title"], sec["t0"], sec["t1"], sec["t0"],
                                 notes=_cut(f"Part {n}: {sec['title']}. It covers: "
                                            + "; ".join(s["title"] for s in done) + ".", NOTES_CHARS)))
        for sid, sl in zip(ids["slides"], done):
            fixes.update(sl.get(FIXES) or {})
            sl = {k: v for k, v in sl.items() if k != FIXES}
            sl["id"] = sid
            slides.append(sl)
            content.append(sl)
    if final:
        takeaways = plan["takeaways"] or [inline_text(s["title"], BULLET_CHARS) for s in secs][:MAX_TAKEAWAYS]
        slides.append(_slide(lay["summary"], "summary", "Key takeaways", t0=lec.clip0, t1=lec.clip1, t=lec.clip0,
                             blocks=[{"type": "bullets", "items": [{"text": t, "level": 0} for t in takeaways]}],
                             notes=plan["closing"]))
        slides += formula_slides(content, lay["next"])
    return _safe_strings(fix_numbers({
        "format": FORMAT, "version": version, "kind": lec.kind, "title": plan["title"], "subtitle": plan["subtitle"],
        "lecture": lecture, "duration": round(lec.clip1 - lec.clip0, 2), "clip": [round(lec.clip0, 2), round(lec.clip1, 2)],
        "slides": slides, "degraded": list(degraded)}, fixes))


# ---------- files ----------
def _export_dir(job_dir: Path) -> Path:
    return job_dir / EXPORT


def _save_bytes(path: Path, data: bytes, tries: int = 40) -> None:
    """Written whole or not at all: the player may be reading the old file (Windows then refuses the replace for
    a moment: `tries` attempts, about 6 s for 40)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(data)
    if not replace_file(tmp, path, tries):
        tmp.unlink(missing_ok=True)
        raise OSError(f"could not save {path.name}: the file is in use")


def _save_json(path: Path, data: dict) -> None:
    _save_bytes(path, json.dumps(_safe_strings(data), ensure_ascii=False, indent=1).encode("utf-8"))


def _load_json(path: Path) -> dict:
    for _ in range(20):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)  # Windows can refuse a read while the file is being replaced
    return {}


def read_replay(job_dir: Path) -> tuple[bytes, dict]:
    """replay.json as (bytes, data). NotReady while there is none, or while it is being rewritten."""
    path = job_dir / "replay.json"
    for _ in range(20):
        try:
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict):
                return raw, data
            break
        except FileNotFoundError:
            raise NotReady("this lecture is not processed yet, so there are no slides yet") from None
        except (OSError, ValueError):
            time.sleep(0.05)
    raise NotReady("the lecture is being updated; try again in a moment")


def deck_version(raw: bytes) -> str:
    return hashlib.sha256(raw + b"|deck-rules-" + DECK_RULES_VERSION.encode()).hexdigest()[:16]


def lecture_title(job_dir: Path, replay: dict) -> str:
    """The subject, else the uploaded file's name, else the job's id."""
    subject = _flat(replay.get("subject"))
    if subject:
        return subject[:1].upper() + subject[1:]
    status = _load_json(job_dir / "status.json")
    video = replay.get("video") or {}
    for name in (status.get("name"), video.get("source"), video.get("file")):
        if isinstance(name, str) and name.strip() and Path(name).stem not in ("upload", "lecture"):
            return Path(name.strip()).stem
    return job_dir.name


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(job_dir: Path, what: str) -> threading.Lock:
    key = f"{job_dir.resolve()}|{what}"
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


_FIGURE = re.compile(rf"{EXPORT}/fig_\d{{3,}}\.png")


def _figures(deck: dict) -> list[tuple[dict, dict]]:
    return [(sl, b) for sl in deck.get("slides") or [] if isinstance(sl, dict)
            for b in sl.get("blocks") or [] if isinstance(b, dict) and b.get("type") == "figure"]


def ready_deck(job_dir: Path, replay: dict, version: str) -> dict | None:
    """The saved deck when it is this lecture version's, with its figures (a missing one is drawn again: code only)."""
    deck = _load_json(_export_dir(job_dir) / "deck.json")
    if deck.get("format") != FORMAT or deck.get("version") != version or not isinstance(deck.get("slides"), list):
        return None
    lec = None
    for _, b in _figures(deck):
        image = b.get("image")
        if not (isinstance(image, str) and _FIGURE.fullmatch(image)):
            return None
        if not (job_dir / image).is_file():
            lec = lec or Lecture(job_dir, replay)
            img = render_figure(lec, lec.refs(b.get("refs")))
            if img is None:
                return None
            _save_bytes(job_dir / image, _encode_png(img))
    return deck


def _clean_old_files(job_dir: Path, deck: dict) -> None:
    """Figures no deck refers to any more, and the page images and key points of the old page-image decks."""
    keep = {b.get("image") for _, b in _figures(deck)}
    out = _export_dir(job_dir)
    for p in list(out.glob("fig_*.png")) + list(out.glob("page_*.png")) + [out / "points.json"]:
        if f"{EXPORT}/{p.name}" in keep:
            continue
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass  # being sent to the player (Windows refuses): nothing refers to it any more


# ---------- writing the deck, in the background ----------
class Run:
    """One writing of a lecture's deck: its progress for slides.json."""

    def __init__(self, job_dir: Path, version: str, attempt: int = 1):
        self.job_dir = job_dir
        self.version = version
        self.attempt = attempt
        self.state = "writing"
        self.step = "Planning the slides"
        self.done = 0
        self.total = 1
        self.error: str | None = None
        self.deck: dict | None = None
        self.cancelled = False
        self.thread: threading.Thread | None = None

    def status(self) -> dict:
        return {"state": self.state, "step": self.step, "done": self.done, "total": self.total, "error": self.error,
                "deck": self.deck}


_runs: dict[str, Run] = {}
_guard = threading.Lock()


def _key(job_dir: Path) -> str:
    return str(job_dir.resolve())


def _ready_status(deck: dict) -> dict:
    parts = len({sl.get("section") for sl in deck["slides"] if isinstance(sl, dict) and sl.get("kind") == "content"})
    return {"state": "ready", "step": "Ready", "done": parts + 1, "total": parts + 1, "error": None, "deck": deck}


def slides_status(job_dir: Path) -> dict:
    """GET slides.json: the deck when it is ready, else how the writing is going (never calls the model)."""
    raw, replay = read_replay(job_dir)
    version = deck_version(raw)
    with _guard:
        deck = ready_deck(job_dir, replay, version)
        if deck is not None:
            return _ready_status(deck)
        run = _runs.get(_key(job_dir))
        if run is not None and run.version == version and run.state != "ready":  # a finished run: deck.json says
            return run.status()
    return {"state": "none", "step": "", "done": 0, "total": 0, "error": None, "deck": None}


def start_writing(job_dir: Path, make_client: Callable[[], Any]) -> dict:
    """POST slides/write: start writing the deck in the background, unless it is ready or being written (then this
    joins that run: one lecture is never paid for twice at once). A failed run is tried again. NotReady while the
    replay is partial."""
    raw, replay = read_replay(job_dir)
    if is_partial(replay):
        raise NotReady(STILL_PROCESSING)
    version = deck_version(raw)
    with _guard:
        deck = ready_deck(job_dir, replay, version)
        if deck is not None:
            return _ready_status(deck)
        key = _key(job_dir)
        run = _runs.get(key)
        if run is not None and run.version == version and run.state == "writing":
            return run.status()
        if run is not None:
            run.cancelled = True  # an older lecture version's run stops at its next step; its deck is never saved
        attempt = 1
        if run is not None and run.version == version:
            # After a failure the plan is asked as a new request (the cache could hold the bad answer); a deck that
            # was written and then lost is written again with the same requests, from the cache, for free.
            attempt = run.attempt + (1 if run.state == "failed" else 0)
        run = _runs[key] = Run(job_dir, version, attempt)
        run.thread = threading.Thread(target=write_deck, args=(run, raw, replay, make_client), daemon=True,
                                      name=f"slides-{job_dir.name}")
        run.thread.start()
        return run.status()


def _current(run: Run) -> bool:
    return not run.cancelled and _runs.get(_key(run.job_dir)) is run


def _call(client: Any, messages: list[dict], schema: dict, name: str, max_tokens: int, key: str,
          attempt: int, log: list) -> dict:
    """One Muse Spark call for a JSON answer. An answer cut short or not JSON is asked once more (a new request, so
    the disk cache does not hand back the same answer)."""
    for k in range(2):
        msgs = messages
        tries = attempt + k
        if tries > 1:  # a later try is a different request
            msgs = messages[:-1] + [{"role": "user", "content": messages[-1]["content"]
                                     + f"\n\n(Try {tries}: reply with complete, valid JSON.)"}]
        t = time.perf_counter()
        reply = client.chat(msgs, schema=schema, schema_name=name, reasoning=REASONING,
                            max_tokens=int(max_tokens * (1.5 if k else 1)), extra={"prompt_cache_key": key})
        usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
        log.append({"call": name, "seconds": round(time.perf_counter() - t, 2), "usage": usage})
        finish = ((reply.get("choices") or [{}])[0] or {}).get("finish_reason")
        try:
            if finish == "length":
                raise MetaAPIError("the answer was cut short")
            return parse_json_object(chat_text(reply))
        except MetaAPIError:
            if k:
                raise
    raise MetaAPIError("no answer")  # not reached


def plan_max_tokens(seconds: float) -> int:
    """Room for the plan's answer: live, 4.4k tokens for 8 minutes at low effort (3.4k of them thinking); about 80
    more per slide, so an hour-long lecture (60 slides) gets more."""
    return int(min(20000, max(12000, 4000 + 200 * seconds / 60)))


def make_plan(client: Any, lec: Lecture, system: str, key: str, lecture: str, attempt: int = 1,
              log: list | None = None) -> dict:
    data = _call(client, plan_messages(lec, system), PLAN_SCHEMA, "deck_plan",
                 plan_max_tokens(lec.clip1 - lec.clip0), key, attempt, log if log is not None else [])
    return repair_plan(data, lec, lecture)


def write_section(client: Any, lec: Lecture, system: str, key: str, plan: dict, si: int, attempt: int = 1,
                  log: list | None = None) -> tuple[list[dict], str | None]:
    """(the section's content slides, a plain line when it was made without Muse Spark, else None)."""
    sec = plan["sections"][si]
    n = len(sec["slides"])
    problem = None
    written: list = []
    try:
        data = _call(client, write_messages(lec, system, plan, si), WRITE_SCHEMA, "deck_section",
                     min(16000, 4000 + 1500 * n), key, attempt, log if log is not None else [])
        written = [s for s in data.get("slides") or [] if isinstance(s, dict)] if isinstance(data.get("slides"), list) else []
        if not written:
            raise MetaAPIError("the answer had no slides")
        if len(written) < n:
            problem = (f'Part {si + 1} ("{sec["title"]}"): {n - len(written)} of its {n} slides were made from the plan '
                       f"without Muse Spark (its answer was short).")
    except MetaAPIError as e:
        text = str(e).splitlines()[0] if str(e) else "unknown error"
        problem = f'Part {si + 1} ("{sec["title"]}") was made from the plan without Muse Spark ({text}).'
        written = []
    slides = [content_slide(lec, sl, sec["title"], written[k] if k < len(written) else None)
              for k, sl in enumerate(sec["slides"])]
    return slides, problem


def block_coverage(lec: Lecture, deck: dict) -> dict:
    """What the written deck shows by reference, for the log: the plan's "covered" counts what its slides said they
    would present; here, the items that a formula or figure of the deck names in its refs (bullets, tables and steps
    carry no refs, so this is a floor)."""
    must, _ = lec.coverage_items()
    refs = {r for sl in deck["slides"] for b in sl["blocks"] for r in b.get("refs") or []}
    shown = sorted(ref for ref, by in must.items() if by & refs)
    return {"in_formulas_or_figures": len(shown), "in_formulas_or_figures_refs": shown}


def draw_figures(run: Run | None, lec: Lecture, slides: list[dict], ids: list[str], planned: list[dict]) -> None:
    """Each figure block's picture, saved as export/fig_NNN.png (NNN: its slide's number). A figure that cannot be
    drawn is dropped (and a slide left with nothing gets its blocks from what it presents)."""
    for sl, sid, plan_sl in zip(slides, ids, planned):
        keep = []
        for b in sl["blocks"]:
            if b["type"] != "figure":
                keep.append(b)
                continue
            img = render_figure(lec, b["refs"])
            if img is None:
                continue
            name = figure_name(sid)
            data = _encode_png(img)
            with _guard:
                if run is not None and not _current(run):
                    raise _Cancelled()
                _save_bytes(lec.job_dir / name, data)
            keep.append({**b, "image": name, "size": [int(img.shape[1]), int(img.shape[0])]})
        if not keep:
            keep = [b for b in fallback_blocks(lec, plan_sl["covers"], plan_sl.get("focus") or plan_sl["title"])
                    if b["type"] != "figure"]
            keep = keep or [{"type": "bullets", "items": [{"text": inline_text(plan_sl["title"], BULLET_CHARS),
                                                           "level": 0}]}]
        sl["blocks"] = keep


def write_deck(run: Run, raw: bytes, replay: dict, make_client: Callable[[], Any]) -> None:
    """The background writing of one deck: plan, parts in parallel, figures, then deck.json (if this run is still
    the lecture's current one and replay.json has not changed)."""
    job_dir = run.job_dir
    client = None
    started = time.perf_counter()
    log: list = []
    try:
        client = make_client()
        calls0 = len(getattr(client, "calls", None) or [])
        lec = Lecture(job_dir, replay)
        lecture = lecture_title(job_dir, replay)
        system, key = context_messages(job_dir, replay)
        plan = make_plan(client, lec, system, key, lecture, run.attempt, log)
        lay = layout(plan)
        n = len(plan["sections"])
        written: list[list[dict] | None] = [None] * n
        degraded: list[str] = [None] * n  # type: ignore[list-item]
        with _guard:
            if not _current(run):
                raise _Cancelled()
            run.done, run.total, run.step = 1, n + 1, f"Writing part 1 of {n}"
            run.deck = assemble(lec, plan, lay, written, run.version, lecture, [], final=False)

        def one(si: int):
            if not _current(run):
                raise _Cancelled()
            # Write calls are the same request on a later try, so parts already written come from the disk cache.
            slides, problem = write_section(client, lec, system, key, plan, si, 1, log)
            draw_figures(run, lec, slides, lay["sections"][si]["slides"], plan["sections"][si]["slides"])
            return slides, problem

        pool = ThreadPoolExecutor(max_workers=PARALLEL, thread_name_prefix=f"slides-{job_dir.name}")
        try:
            futures = {pool.submit(one, si): si for si in range(n)}
            for fut in as_completed(futures):
                si = futures[fut]
                written[si], degraded[si] = fut.result()
                with _guard:
                    if not _current(run):
                        raise _Cancelled()
                    run.done = 1 + sum(w is not None for w in written)
                    first = next((k for k, w in enumerate(written) if w is None), None)
                    run.step = f"Writing part {first + 1} of {n}" if first is not None else "Putting the deck together"
                    run.deck = assemble(lec, plan, lay, written, run.version, lecture,
                                        [d for d in degraded if d], final=False)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        deck = assemble(lec, plan, lay, written, run.version, lecture, [d for d in degraded if d], final=True)
        recs = (getattr(client, "calls", None) or [])[calls0:]
        costs = [float(r.cost_usd) for r in recs if getattr(r, "cost_usd", None) is not None]
        try:  # read_replay tries again while Windows refuses the read (the pipeline replacing the file)
            now = deck_version(read_replay(job_dir)[0])
        except NotReady:
            now = None
        with _guard:
            if not _current(run):
                raise _Cancelled()
            if now != run.version:  # replay.json changed while writing: this deck is for an older lecture
                run.state, run.step, run.error = "failed", "Failed", "the lecture changed while its slides were written; try again"
                return
            _save_json(_export_dir(job_dir) / "deck.json", deck)
            run.state, run.step, run.done, run.deck = "ready", "Ready", run.total, deck
        _clean_old_files(job_dir, deck)
        try:
            _save_json(_export_dir(job_dir) / "deck-log.json", {
                "version": run.version, "seconds": round(time.perf_counter() - started, 1),
                "cost_usd": round(sum(costs), 6) if costs else None,
                "coverage": {**plan["coverage"], **block_coverage(lec, deck)},
                "slides": len(deck["slides"]), "degraded": deck["degraded"], "calls": log,
                "plan": {"sections": plan["sections"], "skip": plan["skip"]}})
        except OSError:
            pass
    except _Cancelled:
        with _guard:
            if run.state == "writing":
                run.state, run.step = "failed", "Stopped"
                run.error = "a newer version of the lecture is being written"
    except Exception as e:
        if not isinstance(e, (MetaAPIError, NotReady)):
            try:
                (job_dir / "export-error.log").write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass
        with _guard:
            run.state, run.step, run.error, run.deck = "failed", "Failed", plain(e), None
    finally:
        close = getattr(getattr(client, "http", None), "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


def plain(e: BaseException) -> str:
    """One line for the player."""
    if isinstance(e, MetaAPIError):
        text = str(e).splitlines()[0] if str(e) else "unknown error"
        return f"Muse Spark could not write the slides: {text}"
    if isinstance(e, NotReady):
        return str(e)
    if isinstance(e, OSError) and (e.errno is not None or e.filename):
        return plain_error(e)
    text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return f"could not make the slides ({type(e).__name__}: {text[:160]}); details are in export-error.log"


# ---------- PowerPoint ----------
def file_name(title: str) -> tuple[str, str]:
    """(an ASCII file name, the same in Unicode) for the deck: '<title> - slides.pptx'."""
    uni = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', " ", title)
    uni = " ".join(uni.split()).strip(" .-")[:80].strip(" .-") or "lecture"
    asc = unicodedata.normalize("NFKD", uni).encode("ascii", "ignore").decode("ascii")
    asc = " ".join(re.sub(r"[^A-Za-z0-9 _.-]+", " ", asc).split()).strip(" .-") or "lecture"
    return f"{asc} - slides.pptx", f"{uni} - slides.pptx"


def content_disposition(title: str) -> str:
    asc, uni = file_name(title)
    return f"attachment; filename=\"{asc}\"; filename*=UTF-8''{quote(uni)}"


NO_DECK = "Open the Slides page first so Muse Spark can write the slides."


def write_pptx(job_dir: Path) -> tuple[bytes, dict]:
    """(the PowerPoint, the deck it shows) of the ready deck (NotReady when there is none), rendered by
    app/pptx_deck.py and kept as export/slides.pptx for this deck and renderer version. No model call."""
    raw, replay = read_replay(job_dir)
    if is_partial(replay):
        raise NotReady(STILL_PROCESSING)
    version = deck_version(raw)
    with _guard:
        deck = ready_deck(job_dir, replay, version)
    if deck is None:
        raise NotReady(NO_DECK)
    renderer = importlib.import_module(f"{__package__}.pptx_deck")
    # The renderer's rules as it can apply them now (rules_version() marks formulas left as text while node is
    # missing, so that PowerPoint is made again once it can typeset them), and the deck itself: a deck written
    # again for the same lecture version (deck.json lost) must not get the old file.
    rules = renderer.rules_version() if callable(getattr(renderer, "rules_version", None)) else \
        getattr(renderer, "RULES_VERSION", "")
    content = hashlib.sha256(json.dumps(deck, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    key = f"{deck['version']}|{content}|{rules}"
    out = _export_dir(job_dir)
    with _lock(job_dir, "pptx"):
        if _load_json(out / "slides-pptx.json").get("key") == key:
            try:
                return (out / "slides.pptx").read_bytes(), deck
            except OSError:
                pass
        blob = renderer.build_pptx(job_dir, deck)
        try:  # a copy for the owner; open in PowerPoint, it is skipped at once rather than holding up the download
            _save_bytes(out / "slides.pptx", blob, tries=2)
            _save_json(out / "slides-pptx.json", {"key": key})
        except OSError:
            pass
    return blob, deck
