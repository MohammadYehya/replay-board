"""Slides export (app/export.py and its endpoints), offline, with made-up lectures and a fake Muse Spark: the plan's
repair and coverage, the written blocks and their limits, figures, the background writing (status, joining,
failures, a lecture that changes) and the endpoints."""
import copy
import json
import sys
import threading
import types
import warnings

import cv2
import numpy as np
import pytest

with warnings.catch_warnings():  # starlette's notice about httpx, at import time
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app import ask, export, server
from app.core.media import imread, imwrite
from app.core.meta_api import CallRecord, MetaAPIError

BG = "#f0f0e8"
INK = (200, 40, 10)  # BGR


# ---------- lectures made up for the tests ----------
def piece(i, box, t, kind="text", text=None, latex="", spoken=(), **kw):
    p = {"id": i, "box": list(box), "image": f"pieces/{i:03d}.png", "draw": [t, t + 1.0], "t_seen": t + 0.3,
         "t_shown": t + 1.0, "written_after": t, "revealed": False, "removed_at": None, "pre_existing": False,
         "read": {"kind": kind, "text": f"item {i}" if text is None else text, "latex": latex},
         "spoken": [{"start": a, "end": b} for a, b in spoken], "redraw_at": None}
    p.update(kw)
    return p


def old(i, box, **kw):
    """Writing that was on the board before the recording."""
    return piece(i, box, 0.0, pre_existing=True, t_seen=0.0, t_shown=0.0, **kw)


def board(pieces, chapters, lines=(), start=0.0, end=180.0, size=(800, 400), bg=BG, subject="growth and logs"):
    words = [{"w": f"w{i}.", "s": start + i + 0.1, "e": start + i + 0.5} for i in range(int(end - start))]
    return {
        "version": 1, "type": "board", "subject": subject,
        "video": {"file": "lecture.mp4", "duration": max(end, 200.0), "width": 1280, "height": 720,
                  "start": start, "end": end},
        "chapters": [{"t": t, "title": title} for t, title in chapters],
        "transcript": {"turns": [], "words": words, "timing": "mms"},
        "board": {"size": list(size), "background": bg, "pieces": pieces, "lines": list(lines), "erases": [],
                  "annotations": []},
    }


def stroke_image(w, h, color=INK):
    """A BGRA piece: an ellipse outline and a diagonal, 2 px wide, on a transparent background (binary alpha)."""
    img = np.zeros((h, w, 4), np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (max(1, w // 2 - 3), max(1, h // 2 - 3)), 0, 0, 360, (*color, 255), 2)
    cv2.line(img, (2, h - 3), (w - 3, 2), (*color, 255), 2)
    img[..., 3] = np.where(img[..., 3] > 0, 255, 0)
    return img


def write_job(d, replay, colors=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    if replay["type"] == "board":
        for p in replay["board"]["pieces"]:
            imwrite(d / p["image"], stroke_image(p["box"][2], p["box"][3], (colors or {}).get(p["id"], INK)))
    else:
        for st in replay["slides"]["states"]:
            if st.get("image"):
                img = np.full((540, 960, 3), 255, np.uint8)
                cv2.rectangle(img, (100, 200), (400, 400), (30, 120, 220), -1)  # the chart s1e3 shows
                cv2.putText(img, f"state {st['id']}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                imwrite(d / st["image"], img)
    return d


def lecture_replay():
    pieces = [
        piece(1, (100, 40, 120, 40), 10, kind="equation", text="A = P(1+r)^n", latex="A=P(1+r)^n", spoken=[(11, 14)]),
        piece(2, (300, 100, 200, 150), 30, kind="graph", text="graph of growth", spoken=[(31, 35)]),
        piece(3, (100, 200, 60, 30), 70, text="A=(1+1)", spoken=[(70, 72)]),
        piece(4, (170, 200, 60, 30), 72, text="^1=2", spoken=[(72, 74)]),
        piece(5, (100, 260, 160, 40), 100, kind="equation", text="A=(1+1/2)^2=2.25", latex="A=(1+1/2)^2=2.25",
              spoken=[(101, 105)]),
        piece(6, (500, 40, 80, 30), 130, text="decay", spoken=[(131, 133)]),
        piece(7, (500, 100, 120, 40), 140, kind="equation", text="D=(1-r)^n", latex="D=(1-r)^n", spoken=[(141, 145)]),
        old(8, (650, 300, 100, 60), kind="equation", text="y = ln x", latex="y=\\ln x"),  # an earlier lesson
        piece(9, (700, 20, 12, 10), 2, kind="arrow", text="small arrow"),  # a stray mark
        old(10, (20, 360, 60, 10), kind="underline", text="16/2/15"),
    ]
    lines = [{"id": "L1", "pieces": [3, 4], "box": [100, 200, 130, 30], "t_complete": 74,
              "read": {"kind": "equation", "text": "A=(1+1)^1=2", "latex": "A=(1+1)^1=2"}}]
    return board(pieces, [(0, "Growth"), (60, "Compounding"), (120, "Decay")], lines=lines)


PLAN = {
    "title": "Exponential **growth** and compounding [#1]",
    "subtitle": "Why money that earns interest on interest grows by the same factor each year",
    "sections": [
        {"title": "Compounding", "t0": "1:00", "t1": "2:00", "slides": [  # out of order: sorted by time
            {"title": "Yearly compounding gives $2", "t0": "1:00", "t1": "1:20", "covers": ["L1"], "focus": "One year."},
            {"title": "Twice a year gives $2.25", "t0": "1:20", "t1": "1:40", "covers": ["#5", "#nope"], "focus": ""},
            {"title": "More often, more money", "t0": "1:40", "t1": "2:00", "covers": [], "focus": "The trend."}]},
        {"title": "Growth", "t0": "0:00", "t1": "1:00", "slides": [
            {"title": "The growth formula", "t0": "0:05", "t1": "0:30", "covers": ["#1"], "focus": "The formula."},
            {"title": "The growth curve", "t0": "0:30", "t1": "1:00", "covers": ["#2"], "focus": "The curve."}]},
        {"title": "Decay", "t0": "2:00", "t1": "3:00", "slides": [
            {"title": "Decay shrinks by a share", "t0": "2:00", "t1": "3:10", "covers": ["#6"], "focus": "Decay."}]},
    ],
    "takeaways": ["Growth is proportional to size [1:02]", "Compounding more often gives more, up to a limit",
                  "Decay is growth with $-r$"],
    "closing": "Growth and decay follow one rule. [#7]",
    "skip": ["#8"],
}


def blk(kind, **kw):
    """A flat block as the model writes it: every field present, the unused ones empty."""
    b = {"type": kind, "items": [], "latex": "", "label": "", "refs": [], "steps": [], "caption": "", "header": [],
         "rows": [], "style": "", "title": "", "text": ""}
    b.update(kw)
    return b


SECTIONS = {
    1: {"slides": [
        {"title": "The growth formula", "t": "0:11", "notes": "Money grows by a share each year. [#1] That is the formula.",
         "blocks": [blk("equation", latex="$$A=P(1+r)^n$$", label="Compound growth", refs=["#1"]),
                    blk("bullets", items=[{"text": "Each year adds $r$ times the balance [#1]", "level": 0},
                                          {"text": "So $A$ grows by **the same factor**", "level": 1}])]},
        {"title": "The growth curve.", "t": "9:59", "notes": "",
         "blocks": [blk("figure", refs=["#2", "#99"], caption="The growth curve $y=e^x$"),
                    blk("figure", refs=["#2"], caption="again"),
                    blk("bullets", items=[{"text": "It gets steeper as it grows", "level": 0}])]}]},
    2: {"slides": [
        {"title": "Yearly compounding gives $2", "t": "1:02", "notes": "One year at 100% doubles the dollar.",
         "blocks": [blk("equation", latex="A=(1+1)^1=2", label="One year", refs=["L1"]),
                    blk("steps", label="One dollar, one year", steps=[{"latex": "A=(1+1)^1", "text": ""},
                                                                      {"latex": "", "text": "so \\$1 becomes $2"}])]},
        {"title": "Twice a year gives $2.25", "t": "1:21", "notes": "Half the rate, twice.",
         "blocks": [blk("callout", style="definition", title="Compound interest",
                        text="Interest is paid on the interest", latex="A=P\\left(1+\\frac{r}{n}\\right)^{n}"),
                    blk("table", header=["Compounding", "Formula", "Value", ""],
                        rows=[["yearly", "$(1+1)^1$", "2"], ["twice", "$(1+\\tfrac12)^2$", "2.25", "extra"]],
                        caption="Two cases")]},
        {"title": "More often, more money", "t": "1:45", "notes": "Monthly is better still.",
         "blocks": [blk("bullets", items=[{"text": "Monthly gives \\$2.61", "level": 0}])]}]},
    3: {"slides": [
        {"title": "Decay shrinks by a share", "t": "2:21", "notes": "Decay is the same rule, shrinking.",
         "blocks": [blk("callout", style="key", title="Decay", text="Each step keeps $1-r$ of it",
                        latex="D=(1-r)^n"),
                    blk("bullets", items=[{"text": "Half-lives work this way", "level": 0}])]}]},
}


def reply(data):
    return {"choices": [{"message": {"content": json.dumps(data) if not isinstance(data, str) else data},
                         "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


class FakeClient:
    """Muse Spark as the tests want it: the plan, then each part's answer (by the part number in the request).
    `gates`: {"plan" or part number: threading.Event} holds that call until the test sets it. `fail`: {same keys:
    message} raises MetaAPIError instead."""

    def __init__(self, plan=PLAN, sections=SECTIONS, gates=None, fail=None, bad=None):
        self.plan, self.sections = plan, sections
        self.gates, self.fail, self.bad = gates or {}, fail or {}, bad or {}
        self.calls, self.seen = [], []
        self.lock = threading.Lock()

    def chat(self, messages, **kw):
        with self.lock:
            self.seen.append((messages, kw))
        user = messages[-1]["content"]
        if kw["schema_name"] == "deck_plan":
            what, answer = "plan", self.plan
        else:
            import re
            what = int(re.search(r"# Write part (\d+) of", user).group(1))
            answer = self.sections.get(what, {"slides": []})
        if what in self.gates:
            assert self.gates[what].wait(10)
        if what in self.fail:
            raise MetaAPIError(self.fail[what])
        with self.lock:
            self.calls.append(CallRecord("/chat/completions", "muse-spark-1.3", 200, 0.1, False, 1, {}, 0.01, "k"))
            n = self.bad.get(what, 0)
            if n:
                self.bad[what] = n - 1
                return reply("{not json")
        return reply(answer)

    def calls_for(self, name):
        return [m for m, kw in self.seen if kw["schema_name"] == name]


def wait(d):
    run = export._runs[export._key(d)]
    run.thread.join(20)
    assert not run.thread.is_alive()
    return export.slides_status(d)


def write(d, fake):
    export.start_writing(d, lambda: fake)
    return wait(d)


@pytest.fixture
def job(tmp_path):
    return write_job(tmp_path / "lec", lecture_replay())


def lec_of(d):
    return export.Lecture(d, json.loads((d / "replay.json").read_text(encoding="utf-8")))


# ---------- text ----------
@pytest.mark.parametrize("raw, plain", [
    ("$A = P\\left(1+\\frac{r}{n}\\right)^{nt}$", "A = P(1+r/n)^nt"),
    ("\\sqrt{2} \\approx 1.41 and \\pi r^2", "√2 ≈ 1.41 and π r²"),
    ("as $n \\to \\infty$, $(1+1/n)^n \\to e$", "as n → ∞, (1+1/n)ⁿ → e"),
    ("**Bold** and `code` [#L3] at [1:02]", "Bold and code at"),
    ("1. a numbered point", "a numbered point"),
    ("\\alpha = \\beta \\neq 90^{\\circ}", "α = β ≠ 90°"),
    ("$5 earns the same as five separate $1 accounts", "$5 earns the same as five separate $1 accounts"),
    ("$1 grows to $(1+1/12)^{12} \\approx$ $2.61", "$1 grows to (1+1/12)¹² ≈ $2.61"),
    ("$x^{365 \\times 24}$", "x^(365 × 24)"),
])
def test_plain_text_from_the_model(raw, plain):
    assert export.plain_text(raw) == plain


@pytest.mark.parametrize("raw, text", [
    ("Monthly gives $2.61 and $x^2$ grows", "Monthly gives \\$2.61 and $x^2$ grows"),  # money escaped, maths kept
    ("From $2.25 to $2.61", "From \\$2.25 to \\$2.61"),
    ("costs \\$5 now", "costs \\$5 now"),
    ("$1$ becomes $2", "1 becomes \\$2"), ("$k$ becomes $2", "$k$ becomes \\$2"),
    ("$ x^2 $ with spaces", "$x^2$ with spaces"),
    ("- **Bold** $\\alpha=\\beta$ [#L3] at [1:02]", "Bold $\\alpha=\\beta$ at"),
    ("x \\to \\infty outside the dollars", "x → ∞ outside the dollars"),
    ("\\(a_n\\) in brackets", "$a_n$ in brackets"),
    ("too big: ${\\frac{a+b+c+d+e+f+g}{c+d}}$.", "too big: (a+b+c+d+e+f+g)/(c+d)."),
    ("$\\begin{matrix}1\\end{matrix}$ is not inline", "1 is not inline"),
    ("$α=β=90°$ and $x²³$, 90° outside", "$\\alpha=\\beta=90^\\circ$ and $x^{23}$, 90° outside"),  # seen live
])
def test_inline_text_keeps_simple_maths_and_escapes_money(raw, text):
    assert export.inline_text(raw, 120) == text


def test_a_cut_never_ends_on_a_dangling_word():
    s = ("Edge lengths and angles for cubic, tetragonal, orthorhombic, rhombohedral, monoclinic, triclinic and "
         "hexagonal unit cells")  # seen live: cut to "…triclinic and…"
    assert export._cut(s, export.SUBTITLE_CHARS) == (
        "Edge lengths and angles for cubic, tetragonal, orthorhombic, rhombohedral, monoclinic, triclinic…")
    assert export._cut("one two three of the", 18) == "one two three…"
    assert export._cut("short enough", 40) == "short enough"


def test_inline_text_is_cut_at_a_word_outside_the_maths():
    s = "We get $(1+\\tfrac{1}{2})^2 = 2.25$ dollars after a year of compounding twice, and more words after that"
    cut = export.inline_text(s, 60)  # the limit counts what a reader sees: the formula as "(1+1/2)² = 2.25"
    assert cut == "We get $(1+\\tfrac{1}{2})^2 = 2.25$ dollars after a year of compounding…"
    assert export.shown_len(cut) == 59
    assert export.inline_text("short $x^{2}$ " + "word " * 30 + "$y^2$", 40).endswith("word…")
    assert export.inline_text("a" * 200, 50) == "a" * 49 + "…"  # one word longer than the field: cut inside it
    many = "$x_{1}+x_{2}+x_{3}+x_{4}+x_{5}+x_{6}+x_{7}$"
    assert export.inline_text("A " + many + " tail", 20) == "A…"  # never cut inside maths
    assert export.inline_text(many + " tail", 20) == "x_1+x_2+x_3+x_4+x_5…"  # maths alone too long: its Unicode
    cell = export.inline_text("$α=γ=90°, β≠90°$", export.CELL_CHARS)  # seen live (test2): 43 long as LaTeX
    assert cell == "$\\alpha=\\gamma=90^\\circ, \\beta\\neq 90^\\circ$" and export.shown_len(cell) == 15
    assert export.shown_len("costs \\$5 and $x^{2}$") == 15


def test_display_and_inline_latex_checks():
    assert export.display_tex("$$A=P(1+r)^n$$") == "A=P(1+r)^n"
    assert export.display_tex("\\[x\\]") == "x" and export.display_tex(" $y$ ") == "y"
    assert export.display_tex("= $2.71") == "= \\$2.71"  # a bare dollar would break KaTeX
    assert export.display_tex("α=β=90° γ≠120°") == "\\alpha=\\beta=90^\\circ,\\quad \\gamma\\neq 120^\\circ"
    assert export.display_tex("\\begin{aligned}a&=b\\end{aligned}") == "\\begin{aligned}a&=b\\end{aligned}"
    for bad in ("{a", "a}", "\\left( x", "\\begin{aligned}a", "$", "", "x" * 201, None):
        assert export.display_tex(bad) == ""
    ok = ["A=(1+1/12)^{12}", "\\frac{1}{12}", "\\alpha \\to \\infty", "e^{-x}"]
    bad = ["\\frac{\\frac{1}{2}}{3}", "\\begin{x}", "\\mathcal{A}", "a \\\\ b", "\\frac{1234567890123}{2}", "x" * 61]
    assert all(export.simple_tex(v) for v in ok) and not any(export.simple_tex(v) for v in bad)


def test_formulas_are_typeset_better_without_changing_them():
    # Two statements side by side with only a space between them: TeX would run them together ("p = q = r θ = …").
    assert export.display_tex(r"p=q=r\ \theta=\phi=60^\circ") == r"p=q=r,\quad \theta=\phi=60^\circ"
    assert export.display_tex(r"m\neq n \; \lambda\neq 0") == r"m\neq n,\quad \lambda\neq 0"
    assert export.display_tex(r"\theta=30^{\circ} \phi=45^{\circ}") == r"\theta=30^{\circ},\quad \phi=45^{\circ}"
    # Left as written: units, statements already separated, line breaks, one statement, a plain space after a name.
    for keep in (r"g = 9.8\ \mathrm{m/s^2}", r"x = 2\ \text{m}, t = 3\ \text{s}", r"x=-1,\ y=2", r"F = m a = 10",
                 r"\begin{aligned} a&=1 \\ b&=2 \end{aligned}", r"u=v \\ w=z", r"\alpha=\beta \quad \gamma=1",
                 r"y=|\ln x|=|\log_e x|", r"\sin^2 x + \cos^2 x = 1", r"(a=1\ b=2)"):
        assert export.display_tex(keep) == keep, keep
    # Parentheses around a fraction grow with it; others, sized ones and unmatched ones stay.
    assert export.display_tex(r"P=(1+\frac{r}{4})^{4t}") == r"P=\left(1+\frac{r}{4}\right)^{4t}"
    assert export.display_tex(r"((\frac{a}{b}) + c)") == r"\left(\left(\frac{a}{b}\right) + c\right)"
    for keep in (r"f(x)=\frac{1}{x}", r"\left(\frac{a}{b}\right)^2", r"\big(\frac{1}{2}\big)", r"\frac{(a}{b)}",
                 r"(\frac{a}{b}"):
        assert export.display_tex(keep) == keep, keep
    assert len(export.display_tex("(" + r"\frac{1}{2}" * 16 + ")")) <= export.LATEX_CHARS  # never past the limit
    for s in (r"p=q=r\ \theta=60^\circ", r"P=(1+\frac{r}{4})^{4t}", r"A = pi r^2"):  # cleaning twice: no change
        once = export.display_tex(s)
        assert export.display_tex(once) == once
    # Greek letters written as words become commands ("$pi$" showed the letters p and i), but not inside \text.
    assert export.display_tex(r"A = pi r^2") == r"A = \pi r^2"
    assert export.display_tex(r"N = N_0 e^{-lambda t} \text{ (beta decay)}") == \
        r"N = N_0 e^{-\lambda t} \text{ (beta decay)}"
    assert export.inline_text("Like $pi$, the ratio $alpha/beta$ is fixed", 120) == \
        r"Like $\pi$, the ratio $\alpha/\beta$ is fixed"
    assert export.inline_text(r"$\theta$, $\epsilon$, $\upsilon$ and $\text{pi}$ stay", 120) == \
        r"$\theta$, $\epsilon$, $\upsilon$ and $\text{pi}$ stay"
    assert export.inline_text("spin, pipe and alphabet stay words", 120) == "spin, pipe and alphabet stay words"
    # A word or a plain number written as maths is text: "$word$-$word$" showed italic letters (seen live).
    assert export.inline_text("Some $well$-$known$ cells: $40$ grams, $30%$ more, $3.5$ m, but $x$, $ab$ and $2x$",
                              120) == "Some well-known cells: 40 grams, 30% more, 3.5 m, but $x$, $ab$ and $2x$"
    # A case number written before a formula on the board is a label, not maths.
    assert export.display_tex(r"\#4.\ v=(1+\frac{1}{k})^k") == r"v=\left(1+\frac{1}{k}\right)^k"
    assert export.display_tex(r"\#12 y=3") == "y=3" and export.display_tex(r"\#x = 1") == r"\#x = 1"


@pytest.mark.parametrize("latex, checked", [
    # exact, or a variable on a side: unchanged
    (r"v=(1+\frac{1}{4})^2=1.5625", r"v=(1+\frac{1}{4})^2=1.5625"), (r"F=ma=12", r"F=ma=12"),
    (r"q=(1+1)^1=2", r"q=(1+1)^1=2"), (r"2^{10} = 1024", r"2^{10} = 1024"), (r"x^{300}", r"x^{300}"),
    (r"365x24x60 = 525{,}600", r"365x24x60 = 525{,}600"), (r"\frac12 + 0.5 = 1", r"\frac12 + 0.5 = 1"),
    (r"a=b=c,\quad \alpha=90^\circ", r"a=b=c,\quad \alpha=90^\circ"), (r"40\% = 0.4", r"40\% = 0.4"),
    # rounded: = becomes ≈, unless its digits go on ("…")
    (r"v=(1+\frac{1}{8})^{8}=2.57", r"v=(1+\frac{1}{8})^{8}\approx 2.57"), (r"\sqrt{3} = 1.732", r"\sqrt{3}\approx 1.732"),
    (r"e = 2.71828", r"e\approx 2.71828"), (r"e = 2.71828\ldots", r"e = 2.71828\ldots"),
    (r"340 \times 1.25 = 425", r"340 \times 1.25 = 425"), (r"170 \times 1.25 = 213", r"170 \times 1.25\approx 213"),
    (r"v=(1+\frac{1}{10^{9}})^{10^{9}} = e", r"v=(1+\frac{1}{10^{9}})^{10^{9}}\approx e"),  # 40 digits: no float error
    (r"\pi \approx 3.14", r"\pi \approx 3.14"),
    # a misread last digit is corrected
    (r"v=(1+\frac{1}{1000})^{1000}\approx 2.71694\ldots", r"v=(1+\frac{1}{1000})^{1000}\approx 2.71692\ldots"),
    (r"\sqrt{2} = \$1.41419", r"\sqrt{2}\approx \$1.41421"),
    # false: left out
    (r"v=\left(1+\frac{1}{50}\right)=2.69159", None), (r"(1+\frac{1}{2})^2 = 2", None), (r"2^{10} = 1000", None),
    (r"v=(1+\frac{1}{12})^{12} = 3.1", None), (r"\frac{1}{3} = 0.4", None),
    # the letter e may be a variable: never false because of it
    (r"e = 1.6\times 10^{-19}", r"e = 1.6\times 10^{-19}"), (r"2e = 4", r"2e = 4"),
])
def test_the_arithmetic_of_a_formula_is_checked(latex, checked):
    got = export.check_arithmetic(latex)
    assert (got[0].replace(" ", "") if got else None) == (checked.replace(" ", "") if checked else None)


def test_misread_numbers_are_corrected_and_false_formulas_left_out(job):
    lec = lec_of(job)
    fixes = {}
    eq = export.map_block(lec, blk("equation", latex=r"v=(1+\frac{1}{1000})^{1000}=2.71694", label="thousand steps"),
                          fixes)
    assert eq["latex"].replace(" ", "") == r"v=\left(1+\frac{1}{1000}\right)^{1000}\approx2.71692"
    assert fixes == {"2.71694": "2.71692"} and eq["label"] == "Thousand steps"
    assert export.map_block(lec, blk("equation", latex=r"v=(1+\frac{1}{50})=2.69159")) is None  # its power was lost
    st = export.map_block(lec, blk("steps", label="", steps=[{"latex": r"(1+\frac{1}{2})^2=2", "text": "twice"},
                                                            {"latex": r"(1+\frac{1}{2})^2=2.25", "text": ""}]))
    assert st["items"] == [{"latex": "", "text": "twice"}, {"latex": r"\left(1+\frac{1}{2}\right)^2=2.25", "text": ""}]
    co = export.map_block(lec, blk("callout", style="key", title="half", text="Twice", latex=r"(1+\frac{1}{2})^2=2"))
    assert co["latex"] == "" and co["text"] == "Twice"
    assert export.map_block(lec, blk("callout", style="key", title="T", text="", latex=r"2^{10}=1000")) is None
    # The whole deck gets the corrected number (titles, cells, notes, other formulas), not what was said.
    deck = {"title": "Steps give 2.71694", "slides": [{"title": "About 2.71694…", "said": "two point seven 2.71694",
            "blocks": [{"type": "table", "rows": [["1000", "2.71694"], ["12.71694", "2.716940"]]}],
                                                       "refs": ["#2.71694"]}]}
    fixed = export.fix_numbers(deck, {"2.71694": "2.71692"})
    assert fixed["title"] == "Steps give 2.71692" and fixed["slides"][0]["title"] == "About 2.71692…"
    assert fixed["slides"][0]["said"] == "two point seven 2.71694" and fixed["slides"][0]["refs"] == ["#2.71694"]
    assert fixed["slides"][0]["blocks"][0]["rows"] == [["1000", "2.71692"], ["12.71694", "2.716940"]]


def test_the_corrected_number_reaches_every_slide_of_the_deck(job):
    sections = copy.deepcopy(SECTIONS)
    sections[2]["slides"][0]["blocks"][0] = blk("equation", latex=r"A=(1+\frac{1}{1000})^{1000}=2.71694",
                                                label="A thousand times", refs=["L1"])
    sections[2]["slides"][2]["blocks"] = [blk("bullets", items=[{"text": "A thousand steps give 2.71694", "level": 0}])]
    sections[2]["slides"][2]["notes"] = "It reaches 2.71694."
    sections[3]["slides"][0]["blocks"][0] = blk("callout", style="key", title="Decay", text="Shrinks",
                                                latex=r"D=(1-r)^n 100\%")
    deck = write(job, FakeClient(sections=sections))["deck"]
    text = json.dumps([{k: v for k, v in s.items() if k != "said"} for s in deck["slides"]], ensure_ascii=False)
    assert "2.71694" not in text and text.count("2.71692") >= 3
    c3 = next(s for s in deck["slides"] if s["title"] == "More often, more money")
    assert c3["blocks"][0]["items"][0]["text"] == "A thousand steps give 2.71692" and c3["notes"] == "It reaches 2.71692."
    decay = next(s for s in deck["slides"] if s["section"] == "Decay")
    assert decay["blocks"][0]["latex"] == r"D=(1-r)^n \cdot 100\%"  # a number after a power is multiplied
    assert export.FIXES not in json.dumps(deck)


def test_labels_start_with_a_capital_but_symbols_and_names_stay():
    for raw, want in (("compound interest formula", "Compound interest formula"), ("monoclinic edges", "Monoclinic edges"),
                      ("e as a limit", "e as a limit"), ("x-axis crossing", "x-axis crossing"), ("pH scale", "pH scale"),
                      ("ln x and its graph", "ln x and its graph"), ("log rules", "log rules"), ("", ""),
                      ("Already", "Already"), ("n2o levels", "n2o levels"), ("sin(x) graph", "sin(x) graph")):
        assert export.capital(raw) == want, raw
    assert export.notes_text("Powers of $x$ and $y^2$ grow; it costs $5 or $10.") == \
        "Powers of x and y² grow; it costs $5 or $10."  # notes are plain text; money stays
    assert export.inline_text("It tends to 2.71828….", 120) == "It tends to 2.71828…"
    assert export.inline_text("It stays near $\\$2.71828...$ and $2.718\\ldots$, then $e$", 120) == \
        "It stays near \\$2.71828... and 2.718…, then $e$"  # money and numbers as text, not maths
    assert export.inline_text("It tends to $2.71828\\ldots$.", 120) == "It tends to 2.71828…"
    assert export.inline_text("It tends to $x\\ldots$.", 120) == "It tends to $x\\ldots$"


@pytest.mark.parametrize("latex, ok", [
    ("A=P(1+r)^n", True), ("A = \\left(1 + \\frac{1}{\\infty}\\right)^{\\infty} = e", True), ("D=(1-r)^{n}", True),
    ("\\sin^2 x + \\cos^2 x = 1", True), ("A=(1+1/12)^{12}=2.61", False), ("A=(1+1/(365\\times 24))^{365}", False),
    ("= e", False), ("x^{300}", False), ("", False)])
def test_what_goes_on_the_formula_sheet(latex, ok):
    assert export.is_formula(latex) is ok


def test_notes_titles_times_and_said():
    assert export.notes_text("One. Two! [#3] Three? Four. Five.") == "One. Two! Three? Four."
    assert len(export.notes_text("word " * 400)) <= export.NOTES_CHARS
    assert export._title("The growth curve.") == "The growth curve" and export._title("e = 2.718...") == "e = 2.718..."
    assert len(export._title("long title " * 20)) <= export.TITLE_CHARS
    assert [export.parse_clock(v) for v in ("1:05", "[1:02:03]", 65.5, "7.5", "x", None, True)] == [
        65.0, 3723.0, 65.5, 7.5, None, None, None]
    words = [{"w": w, "s": i * 0.3, "e": i * 0.3 + 0.2}
             for i, w in enumerate(("This is a fairly long sentence about growth that goes on. " * 80).split())]
    text = export.said_text(words, 0, 1e9)
    assert len(text) <= export.SAID_CHARS and text.endswith(" ...") and text[:-4].endswith(".")
    assert export.said_text(words, 0.3, 0.85) == "is a"


def test_lecture_title_file_name_and_safe_text(tmp_path):
    r = board([], [(0, "Intro")], subject="")
    d = write_job(tmp_path / "lec", r)
    (d / "status.json").write_text(json.dumps({"name": "Week 3 — logs.mp4"}), encoding="utf-8")
    assert export.lecture_title(d, r) == "Week 3 — logs"
    assert export.lecture_title(d, {**r, "subject": "growth"}) == "Growth"
    asc, uni = export.file_name('Logs: été / "growth"?')
    assert asc == "Logs ete growth - slides.pptx" and uni == "Logs été growth - slides.pptx"
    cd = export.content_disposition("Logs: été")
    assert cd.startswith('attachment; filename="Logs ete - slides.pptx"; filename*=UTF-8\'\'Logs%20%C3%A9t%C3%A9')
    assert export.safe_text("a\x00b\tc\n\U0001f600￾") == "ab\tc\n\U0001f600"


# ---------- the plan ----------
def test_the_plan_is_put_in_order_without_gaps_and_covers_every_item(job):
    lec = lec_of(job)
    plan = export.repair_plan(json.loads(json.dumps(PLAN)), lec, "Growth and logs")
    assert plan["title"] == "Exponential growth and compounding"
    assert [s["title"] for s in plan["sections"]] == ["Growth", "Compounding", "Decay"]
    spans = [(sl["t0"], sl["t1"]) for s in plan["sections"] for sl in s["slides"]]
    assert spans == [(0, 30), (30, 60), (60, 80), (80, 100), (100, 120), (120, 180)]  # from the clip's start to its end
    assert [(s["t0"], s["t1"]) for s in plan["sections"]] == [(0, 60), (60, 120), (120, 180)]
    slides = [sl for s in plan["sections"] for sl in s["slides"]]
    assert slides[2]["covers"] == ["#L1"] and slides[3]["covers"] == ["#5"]  # "L1" known, "#nope" dropped
    assert slides[5]["covers"] == ["#6", "#7"]  # #7 was in no slide: it joins the slide its time is in
    assert plan["skip"] == ["#8"]
    cov = plan["coverage"]
    assert (cov["items"], cov["covered"], cov["skipped"], cov["attached"]) == (6, 5, 0, 1)
    assert cov["attached_refs"] == ["#7"] and cov["left_out_by_code"] == 2  # the stray arrow and the old date
    assert plan["takeaways"] == ["Growth is proportional to size", "Compounding more often gives more, up to a limit",
                                 "Decay is growth with $-r$"]
    assert plan["closing"] == "Growth and decay follow one rule."


def test_plan_times_missing_equal_or_outside_the_clip(job):
    lec = lec_of(job)
    data = {"title": "", "subtitle": "", "skip": [], "takeaways": [], "closing": "", "sections": [
        {"title": "", "t0": "", "t1": "", "slides": [
            {"title": "A", "t0": "0:40", "t1": "", "covers": [], "focus": ""},
            {"title": "B", "t0": "0:40", "t1": "", "covers": [], "focus": ""},
            {"title": "C", "t0": "", "t1": "", "covers": [], "focus": ""},  # no time: after the one before
            {"title": "", "t0": "1:20", "t1": "", "covers": [], "focus": ""},
            {"title": "E", "t0": "9:00", "t1": "", "covers": [], "focus": ""}]}]}
    plan = export.repair_plan(data, lec, "Growth and logs")
    (sec,) = plan["sections"]
    assert sec["title"] == "A" and plan["title"] == "Growth and logs"
    assert [(s["title"], s["t0"], s["t1"]) for s in sec["slides"]] == [
        ("A", 0, 40), ("B", 40, 60), ("C", 60, 80), ("A (4)", 80, 180), ("E", 180, 180)]
    with pytest.raises(MetaAPIError, match="no slides"):
        export.repair_plan({"sections": [{"title": "x", "slides": []}]}, lec)
    with pytest.raises(MetaAPIError, match="no slides"):
        export.repair_plan("not a plan", lec)


def test_too_many_sections_and_slides_are_merged(job):
    lec = lec_of(job)
    secs = [{"title": f"S{i}", "t0": "", "t1": "", "slides": [
        {"title": f"S{i}.{k}", "t0": f"0:{(i * 3 + k) % 60:02d}" if i < 20 else "2:00", "t1": "", "covers": [],
         "focus": ""} for k in range(3)]} for i in range(22)]
    plan = export.repair_plan({"sections": secs}, lec)
    assert len(plan["sections"]) == export.MAX_SECTIONS
    assert sum(len(s["slides"]) for s in plan["sections"]) <= export.MAX_CONTENT
    flat = [sl for s in plan["sections"] for sl in s["slides"]]
    assert flat[0]["t0"] == 0 and flat[-1]["t1"] == 180
    assert all(a["t1"] == b["t0"] for a, b in zip(flat, flat[1:]))


def test_slide_repeats_count_once_for_coverage(tmp_path):
    d = write_job(tmp_path / "sl", slides_replay())
    must, auto = lec_of(d).coverage_items()
    assert set(must) == {"#s0e1", "#s0e2", "#s1e3", "#s1e4"} and auto == set()  # s1e1, s1e2 repeat s0e1, s0e2
    assert must["#s0e1"] == {"#s0e1", "#s1e1"}


# ---------- what Muse Spark writes, checked ----------
def test_blocks_are_checked_and_kept_within_their_limits(job):
    lec = lec_of(job)
    b = export.map_block(lec, blk("bullets", items=[{"text": f"point {i} " + "word " * 40, "level": 1}
                                                    for i in range(9)]))
    assert len(b["items"]) == export.MAX_BULLETS and b["items"][0]["level"] == 0 and b["items"][1]["level"] == 1
    assert all(len(it["text"]) <= export.BULLET_CHARS and it["text"].endswith("…") for it in b["items"])
    assert export.map_block(lec, blk("bullets", items=[{"text": " ", "level": 0}])) is None

    eq = export.map_block(lec, blk("equation", latex="$A=P(1+r)^n$", label="The compound interest formula, in full",
                                   refs=["#1", "#nope", "1"]))
    assert eq == {"type": "equation", "latex": "A=P(1+r)^n", "label": "The compound interest formula, in full",
                  "refs": ["#1"]}
    assert len(export.map_block(lec, blk("equation", latex="x", label="word " * 20))["label"]) <= export.LABEL_CHARS
    assert export.map_block(lec, blk("equation", latex="A=(1+r")) is not None  # brackets are not braces
    assert export.map_block(lec, blk("equation", latex="A=\\frac{1}{2")) is None

    st = export.map_block(lec, blk("steps", label="Twice", steps=[{"latex": "{bad", "text": ""}] +
                                   [{"latex": f"x_{i}", "text": "t " * 60} for i in range(7)]))
    assert st["type"] == "steps" and len(st["items"]) == export.MAX_STEPS
    assert st["items"][0] == {"latex": "", "text": "bad"}  # unusable LaTeX moved to text
    assert all(len(s["text"]) <= export.STEP_CHARS for s in st["items"])
    one = export.map_block(lec, blk("steps", label="One", steps=[{"latex": "x=1", "text": ""}]))
    assert one == {"type": "equation", "latex": "x=1", "label": "One", "refs": []}

    tb = export.map_block(lec, blk("table", header=list("ABCDEFG"), rows=[["x" * 50] + ["y"] * 7] * 10 + [["", ""]],
                                   caption="$x^2$ cases"))
    assert tb["header"] == list("ABCDEF") and len(tb["rows"]) == export.MAX_ROWS
    assert len(tb["rows"][0]) == 6 and len(tb["rows"][0][0]) <= export.CELL_CHARS and tb["caption"] == "x² cases"
    assert export.map_block(lec, blk("table", header=["only"], rows=[["a"]])) is None
    assert export.map_block(lec, blk("table", header=["a", "b"], rows=[])) is None

    prose = export.map_block(lec, blk("callout", style="key", title="T", text="", latex="\\$100\\%\\$ for one year"))
    assert prose["latex"] == "" and prose["text"] == "100% for one year"  # a sentence is not LaTeX (seen live)
    co = export.map_block(lec, blk("callout", style="warning", title="T", text="", latex="\\frac{1}{"))
    assert co is not None and co["style"] == "key" and co["latex"] == "" and co["text"]
    assert export.map_block(lec, blk("callout", style="note", title="T", text="", latex="")) is None

    assert export.map_block(lec, blk("figure", refs=["#1"])) is None  # writing is never a figure
    assert export.map_block(lec, blk("figure", refs=["#9"])) is None  # nor is a mark (an arrow) alone
    assert export.map_block(lec, blk("figure", refs=["#9", "#2"]))["refs"] == ["#9", "#2"]  # next to its graph
    assert export.map_block(lec, blk("figure", refs=["#2", "#1"]))["refs"] == ["#2", "#1"]  # a graph and its label
    assert export.map_block(lec, blk("video")) is None and export.map_block(lec, "x") is None

    blocks = export.check_blocks([{"type": "figure"}, {"type": "figure"}, {"type": "table"}, {"type": "bullets"},
                                  {"type": "steps"}])
    assert [b["type"] for b in blocks] == ["figure", "table", "bullets"]


def test_a_slide_without_usable_blocks_is_made_from_what_it_presents(job):
    lec = lec_of(job)
    planned = {"title": "Two formulas", "t0": 120.0, "t1": 180.0, "covers": ["#6", "#7", "#2"], "focus": "f"}
    sl = export.content_slide(lec, planned, "Decay", {"title": "", "t": "0:10", "notes": "n",
                                                      "blocks": [blk("equation", latex="{")]})
    assert sl["title"] == "Two formulas" and sl["t"] == 120.0 and sl["notes"] == "n"
    assert [b["type"] for b in sl["blocks"]] == ["equation", "bullets", "figure"]
    assert sl["blocks"][0]["latex"] == "D=(1-r)^n" and sl["blocks"][1]["items"] == [{"text": "decay", "level": 0}]
    assert sl["said"].startswith("w120.") and "w179." in sl["said"] and "w119." not in sl["said"]
    empty = export.content_slide(lec, {**planned, "covers": []}, "Decay", None)
    assert empty["blocks"] == [{"type": "bullets", "items": [{"text": "f", "level": 0}]}] and empty["notes"] == ""


# ---------- figures ----------
def test_a_board_figure_is_the_drawing_alone_grown_with_crisp_strokes(job):
    lec = lec_of(job)
    img = export.board_figure(lec, ["#2"])
    pad = export.FIG_PAD
    w, h = 200 + 2 * pad, 150 + 2 * pad
    k = min(export.FIG_MAX_UP, export.FIG_SIDE / max(w, h))
    assert img.shape == (round(h * k), round(w * k), 4)
    a = img[..., 3]
    assert a[0, 0] == 0 and a[-1, -1] == 0  # transparent around it
    assert (a == 255).sum() > 0 and ((a > 0) & (a < 255)).sum() < 0.6 * (a == 255).sum()  # crisp: few soft pixels
    ink = img[a == 255][:, :3].astype(int)
    assert np.abs(ink - np.array(INK)).max() <= 3  # light board: the ink keeps its colour
    big = export.board_figure(lec, ["#2", "#9"])  # 436 px wide with its padding: grown 2.75x, to 1200 px
    assert abs(max(big.shape[:2]) - export.FIG_SIDE) <= 1
    line = export.board_figure(lec, ["#L1"])  # a line: its pieces
    assert line.shape[1] / line.shape[0] == pytest.approx((130 + 2 * pad) / (30 + 2 * pad), rel=0.02)
    assert export.board_figure(lec, ["#nope"]) is None


def test_writing_that_board_capture_put_into_a_drawing_is_left_out(tmp_path):
    # A cube drawn at the top of piece 1; under it, two lines of words that board capture put into the same piece.
    # The first line is also piece 2 (its box and some of its ink): the figure stops above it, and the words after
    # it go too.
    r = board([piece(1, (100, 100, 120, 170), 5, kind="diagram", text="a cube", spoken=[(5, 9)]),
               piece(2, (100, 205, 120, 25), 8, kind="text", text="CUBE a=b=c", latex="a=b=c"),
               piece(3, (300, 100, 40, 40), 9, kind="label", text="a")], [(0, "Cells")])
    d = write_job(tmp_path / "cells", r)
    img = np.zeros((170, 120, 4), np.uint8)
    cv2.rectangle(img, (10, 5), (110, 95), (*INK, 255), 2)  # the drawing
    for x in range(10, 110, 14):  # words: short strokes
        cv2.line(img, (x, 112), (x + 8, 124), (*INK, 255), 2)
        cv2.line(img, (x, 142), (x + 8, 158), (*INK, 255), 2)
    imwrite(d / "pieces/001.png", img)
    words = np.zeros((25, 120, 4), np.uint8)
    words[:, :] = img[105:130]  # piece 2 holds some of the same strokes
    words[:, 60:] = 0
    imwrite(d / "pieces/002.png", words)
    lec = lec_of(d)
    fig = export.board_figure(lec, ["#1"])
    pad = export.FIG_PAD
    k = min(export.FIG_MAX_UP, export.FIG_SIDE / (120 + 2 * pad))
    assert fig.shape[0] == pytest.approx((95 - 5 + 3 + 2 * pad) * k, abs=2 * k)  # the drawing only
    assert fig.shape[1] == pytest.approx((101 + 3 + 2 * pad) * k, abs=2 * k)
    # Without other writing in it (piece 2 is a label now), the figure is the whole piece.
    r["board"]["pieces"][1]["read"]["kind"] = "label"
    (d / "replay.json").write_text(json.dumps(r), encoding="utf-8")
    assert export.board_figure(lec_of(d), ["#1"]).shape[0] == pytest.approx((170 + 2 * pad) * k, abs=2)
    # The region itself, on masks: bands from the drawing out to the first band of other writing.
    ink = np.zeros((40, 10), bool)
    ink[2:10], ink[14:18], ink[22:26], ink[30:38] = True, True, True, True
    box, own = np.zeros_like(ink), np.zeros_like(ink)
    box[20:28], own[22:24] = True, True
    assert export.drawing_region(ink, [(box, own)], 2) == (slice(2, 18), slice(0, 10))
    assert export.drawing_region(ink, [], 2) is None
    assert export.drawing_region(ink[20:28], [(box[20:28], own[20:28])], 2) is None  # one band: cannot tell
    all_words = np.ones_like(ink)
    assert export.drawing_region(ink, [(all_words, all_words)], 2) == "none"


def test_on_a_dark_board_ink_lightness_is_inverted_keeping_its_hue(tmp_path):
    chalk, white = (40, 210, 230), (235, 235, 235)
    r = board([piece(1, (100, 100, 80, 60), 5, kind="graph", spoken=[(5, 6)]),
               piece(2, (300, 100, 80, 60), 5, kind="graph", spoken=[(5, 6)])], [(0, "Intro")], bg="#1e2222")
    d = write_job(tmp_path / "chalk", r, colors={1: chalk, 2: white})
    lec = lec_of(d)
    for i, color in ((1, chalk), (2, white)):
        img = export.board_figure(lec, [f"#{i}"])
        px = np.median(img[img[..., 3] == 255][:, :3], axis=0).astype(np.uint8)
        h_in, l_in, _ = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HLS)[0, 0].astype(int)
        h_out, l_out, _ = cv2.cvtColor(px[None, None], cv2.COLOR_BGR2HLS)[0, 0].astype(int)
        assert abs(l_out - (255 - l_in)) <= 3 and (abs(h_out - h_in) <= 2 or l_in > 230)


def slides_replay():
    def el(i, kind, text, box, latex="", spoken=()):
        return {"id": i, "kind": kind, "text": text, "latex": latex, "box": list(box),
                "spoken": [{"start": a, "end": b} for a, b in spoken]}

    def state(i, kind, t0, t1, els, blank=False):
        return {"id": i, "kind": kind, "t_start": t0, "t_end": t1, "blank": blank,
                "image": f"slides/{i:03d}.png", "elements": els}

    return {
        "version": 1, "type": "slides", "subject": "compound interest",
        "video": {"file": "lecture.mp4", "duration": 40.0, "width": 1920, "height": 1080, "start": 0.0, "end": 40.0},
        "chapters": [],
        "transcript": {"words": [{"w": "hello.", "s": 0.5, "e": 0.9}, {"w": "interest", "s": 3.0, "e": 3.5},
                                 {"w": "grows.", "s": 12.5, "e": 13.0}, {"w": "bye.", "s": 39.0, "e": 39.5}]},
        "slides": {"size": [960, 540], "states": [
            state(0, "slide", 0.0, 10.0, [el("s0e1", "title", "Compound interest", (60, 40, 300, 40)),
                                          el("s0e2", "bullet", "Interest on interest", (60, 100, 300, 20),
                                             spoken=[(4, 6)])]),
            state(1, "addition", 10.0, 30.0, [el("s1e1", "title", "Compound interest", (60, 40, 300, 40)),
                                              el("s1e2", "bullet", "Interest on interest", (60, 100, 300, 20)),
                                              el("s1e3", "chart", "bar chart of growth", (100, 200, 300, 200)),
                                              el("s1e4", "equation", "A = P(1+r)^n", (500, 300, 200, 40),
                                                 "A=P(1+r)^n", [(12, 15)])]),
            state(2, "slide", 30.0, 40.0, [], blank=True),
        ], "annotations": []},
    }


def test_a_slide_figure_is_cut_from_the_clean_slide(tmp_path):
    d = write_job(tmp_path / "sl", slides_replay())
    lec = lec_of(d)
    img = export.slide_figure(lec, ["#s1e3"])
    pad = export.SLIDE_PAD
    assert img.ndim == 3 and img.shape[2] == 3  # opaque
    assert img.shape[:2] == (round((200 + 2 * pad) * 2), round((300 + 2 * pad) * 2))  # grown at most 2x
    assert np.abs(img[img.shape[0] // 2, img.shape[1] // 2].astype(int) - [30, 120, 220]).max() <= 3
    both = export.slide_figure(lec, ["#s1e3", "#s0e2"])  # s0e2 is found on the later state as s1e2
    assert both.shape[1] > img.shape[1] and both.shape[0] > img.shape[0]
    assert export.slide_figure(lec, ["#s0e1x"]) is None


# ---------- writing the deck ----------
def test_the_deck_end_to_end(job):
    fake = FakeClient()
    st = write(job, fake)
    assert st["state"] == "ready" and st["error"] is None and st["step"] == "Ready" and st["done"] == st["total"] == 4
    deck = st["deck"]
    assert deck == json.loads((job / "export" / "deck.json").read_text(encoding="utf-8"))
    assert {k: deck[k] for k in ("format", "kind", "title", "lecture", "duration", "clip", "degraded")} == {
        "format": 2, "kind": "board", "title": "Exponential growth and compounding", "lecture": "Growth and logs",
        "duration": 180.0, "clip": [0.0, 180.0], "degraded": []}
    assert deck["subtitle"].startswith("Why money")
    slides = deck["slides"]
    assert [s["kind"] for s in slides] == ["title", "agenda", "content", "content", "section", "content", "content",
                                           "content", "content", "summary", "formulas"]
    assert [s["id"] for s in slides] == [f"s{i}" for i in range(1, 12)]
    for s in slides:
        assert set(s) == {"id", "kind", "title", "section", "t0", "t1", "t", "blocks", "notes", "said"}
    title, agenda, g1, g2, sec, c1, c2, c3, d1, summary, formulas = slides
    assert (title["t0"], title["t"], title["blocks"], title["said"]) == (None, None, [], "")
    assert agenda["blocks"] == [{"type": "bullets", "items": [{"text": t, "level": 0}
                                                              for t in ("Growth", "Compounding", "Decay")]}]
    assert [(s["section"], s["t0"], s["t1"], s["t"]) for s in (g1, g2, sec, c1, d1)] == [
        ("Growth", 0, 30, 11.0), ("Growth", 30, 60, 30), ("Compounding", 60, 120, 60), ("Compounding", 60, 80, 62.0),
        ("Decay", 120, 180, 141.0)]  # 2:21 is inside Decay's stretch
    assert g1["blocks"] == [
        {"type": "equation", "latex": "A=P(1+r)^n", "label": "Compound growth", "refs": ["#1"]},
        {"type": "bullets", "items": [{"text": "Each year adds $r$ times the balance", "level": 0},
                                      {"text": "So $A$ grows by the same factor", "level": 1}]}]
    assert g1["notes"] == "Money grows by a share each year. That is the formula."
    assert g1["said"].startswith("w0. w1.") and "w29." in g1["said"] and "w30." not in g1["said"]
    assert g2["title"] == "The growth curve" and [b["type"] for b in g2["blocks"]] == ["figure", "bullets"]
    fig = g2["blocks"][0]
    assert fig["image"] == "export/fig_004.png" and fig["refs"] == ["#2"] and fig["caption"] == "The growth curve y=e^x"
    png = imread(job / fig["image"], cv2.IMREAD_UNCHANGED)
    assert png.shape[2] == 4 and fig["size"] == [png.shape[1], png.shape[0]]
    assert (sec["title"], sec["blocks"], sec["said"]) == ("Compounding", [], "")
    assert "Yearly compounding gives $2; Twice a year gives $2.25; More often, more money" in sec["notes"]
    assert c1["blocks"][1] == {"type": "steps", "label": "One dollar, one year", "items": [
        {"latex": "A=(1+1)^1", "text": ""}, {"latex": "", "text": "so \\$1 becomes \\$2"}]}
    assert c2["blocks"][1]["header"] == ["Compounding", "Formula", "Value"]
    assert c2["blocks"][1]["rows"] == [["yearly", "$(1+1)^1$", "2"], ["twice", "$(1+\\tfrac12)^2$", "2.25"]]
    assert c3["blocks"] == [{"type": "bullets", "items": [{"text": "Monthly gives \\$2.61", "level": 0}]}]
    assert d1["blocks"][0] == {"type": "callout", "style": "key", "title": "Decay", "text": "Each step keeps $1-r$ of it",
                               "latex": "D=(1-r)^n"}
    assert summary["title"] == "Key takeaways" and len(summary["blocks"][0]["items"]) == 3
    assert summary["notes"] == "Growth and decay follow one rule." and (summary["t0"], summary["t1"]) == (0, 180)
    assert formulas["title"] == "Formula sheet" and formulas["t0"] is None
    assert [(b["latex"], b["label"]) for b in formulas["blocks"]] == [  # not the worked number A=(1+1)^1=2
        ("A=P(1+r)^n", "Compound growth"), ("A=P\\left(1+\\frac{r}{n}\\right)^{n}", "Compound interest"),
        ("D=(1-r)^n", "Decay")]

    # One plan and one call per part; Ask Muse's own first message and cache key, so the lecture is cached once.
    assert len(fake.calls_for("deck_plan")) == 1 and len(fake.calls_for("deck_section")) == 3
    r = json.loads((job / "replay.json").read_text(encoding="utf-8"))
    system = ask.SYSTEM + "\n\n" + ask.lecture_context(r)
    key = ask.prepare(job, r, {"question": "q"}).cache_key
    for messages, kw in fake.seen:
        assert messages[0] == {"role": "system", "content": system}
        assert kw["extra"]["prompt_cache_key"] == key and kw["reasoning"] == "low"
    plan_task = fake.calls_for("deck_plan")[0][1]["content"]
    assert "from 0:00 to 3:00" in plan_task and 'the number e is "e", not "E"' in plan_task
    assert "ignore the answer style above" in plan_task and "Try" not in plan_task
    part2 = next(m[1]["content"] for m in fake.calls_for("deck_section") if "# Write part 2 of 3" in m[1]["content"])
    assert 'Part 1: "Growth" (0:00-1:00)' in part2 and '  2.3 "More often, more money" (1:40-2:00)' in part2
    assert '## Slide 2.1: "Yearly compounding gives $2" (1:00-1:20)' in part2 and "Focus: One year." in part2
    assert '- #L1 equation | reads "A=(1+1)^1=2" | LaTeX A=(1+1)^1=2 | written 1:10-1:14' in part2
    assert "What was said (1:00-1:20):\n[1:00] w60." in part2 and "It presents no writing" in part2
    part1 = next(m[1]["content"] for m in fake.calls_for("deck_section") if "# Write part 1 of 3" in m[1]["content"])
    assert "#2 graph [drawing]" in part1 and "#1 equation [drawing]" not in part1

    log = json.loads((job / "export" / "deck-log.json").read_text(encoding="utf-8"))
    assert log["coverage"]["attached_refs"] == ["#7"] and log["cost_usd"] == pytest.approx(0.04)
    # What the slides name in their formulas and figures (the plan's "covered" is what it said it would present).
    assert {"#1", "#2", "#L1"} <= set(log["coverage"]["in_formulas_or_figures_refs"])
    assert log["coverage"]["in_formulas_or_figures"] == len(log["coverage"]["in_formulas_or_figures_refs"])
    assert len(log["calls"]) == 4

    # Ready: asking again neither writes nor pays.
    assert export.start_writing(job, lambda: pytest.fail("the deck is ready"))["state"] == "ready"


def test_a_second_request_joins_the_run_and_the_deck_grows_in_order(job):
    gates = {"plan": threading.Event(), 2: threading.Event()}
    fake = FakeClient(gates=gates)
    first = export.start_writing(job, lambda: fake)
    assert (first["state"], first["step"], first["deck"]) == ("writing", "Planning the slides", None)
    again = export.start_writing(job, lambda: pytest.fail("joined the running one"))
    assert again["state"] == "writing" and export.slides_status(job)["step"] == "Planning the slides"
    gates["plan"].set()
    for _ in range(200):  # parts 1 and 3 finish; part 2 waits
        st = export.slides_status(job)
        if st["done"] == 3:
            break
        threading.Event().wait(0.02)
    assert (st["state"], st["step"], st["done"], st["total"]) == ("writing", "Writing part 2 of 3", 3, 4)
    assert [s["id"] for s in st["deck"]["slides"]] == ["s1", "s2", "s3", "s4"]  # part 3 waits for part 2
    assert st["deck"]["slides"][3]["blocks"][0]["image"] == "export/fig_004.png"
    assert (job / "export/fig_004.png").exists() and not (job / "export/deck.json").exists()
    gates[2].set()
    assert wait(job)["state"] == "ready" and len(fake.calls_for("deck_plan")) == 1


def test_a_part_that_fails_is_made_from_the_plan(job):
    fake = FakeClient(fail={3: "HTTP 500: busy\nmore"}, sections={**SECTIONS, 1: {"slides": SECTIONS[1]["slides"][:1]}})
    deck = write(job, fake)["deck"]
    assert deck["degraded"] == [
        'Part 1 ("Growth"): 1 of its 2 slides were made from the plan without Muse Spark (its answer was short).',
        'Part 3 ("Decay") was made from the plan without Muse Spark (HTTP 500: busy).']
    g2 = next(s for s in deck["slides"] if s["id"] == "s4")
    assert g2["notes"] == "" and [b["type"] for b in g2["blocks"]] == ["figure"]  # the graph it presents
    d1 = next(s for s in deck["slides"] if s["id"] == "s9")
    assert d1["title"] == "Decay shrinks by a share" and d1["t"] == 120 and d1["notes"] == ""
    assert d1["blocks"] == [{"type": "equation", "latex": "D=(1-r)^n", "label": "", "refs": ["#7"]},
                            {"type": "bullets", "items": [{"text": "decay", "level": 0}]}]


def test_a_failed_plan_fails_the_deck_and_a_later_request_tries_again(job):
    st = write(job, FakeClient(fail={"plan": "/chat/completions (muse-spark-1.3) failed: HTTP 500: busy\nmore"}))
    assert st == {"state": "failed", "step": "Failed", "done": 0, "total": 1, "deck": None,
                  "error": "Muse Spark could not write the slides: /chat/completions (muse-spark-1.3) failed: HTTP 500: busy"}
    assert export.slides_status(job)["state"] == "failed"
    fake = FakeClient()
    assert write(job, fake)["state"] == "ready"
    plan_task = fake.calls_for("deck_plan")[0][1]["content"]
    assert plan_task.endswith("(Try 2: reply with complete, valid JSON.)")  # a new request, not the cached answer
    assert all("Try" not in m[1]["content"] for m in fake.calls_for("deck_section"))  # parts: same requests


def test_an_answer_that_is_not_json_is_asked_once_more(job):
    fake = FakeClient(bad={"plan": 1, 2: 2})
    deck = write(job, fake)["deck"]
    assert len(fake.calls_for("deck_plan")) == 2 and len(fake.calls_for("deck_section")) == 4
    assert deck["degraded"] == ['Part 2 ("Compounding") was made from the plan without Muse Spark '
                                "(model answer was not JSON: '{not json')."]


def test_an_unexpected_failure_is_one_plain_line_with_a_log(job, monkeypatch):
    monkeypatch.setattr(export, "layout", lambda plan: 1 / 0)
    st = write(job, FakeClient())
    assert st["state"] == "failed" and "ZeroDivisionError" in st["error"] and "\n" not in st["error"]
    assert (job / "export-error.log").exists()

    def no_key():
        raise MetaAPIError("MODEL_API_KEY is missing: put it in .env (see .env.example).")

    monkeypatch.undo()
    export.start_writing(job, no_key)
    assert wait(job)["error"] == "Muse Spark could not write the slides: MODEL_API_KEY is missing: put it in .env (see .env.example)."


def test_a_lecture_that_changes_while_its_deck_is_written(job):
    gate = threading.Event()
    fake = FakeClient(gates={"plan": gate})
    export.start_writing(job, lambda: fake)
    old_run = export._runs[export._key(job)]
    r = json.loads((job / "replay.json").read_text(encoding="utf-8"))
    r["chapters"][0]["title"] = "Growth, again"  # the pipeline rewrote replay.json
    (job / "replay.json").write_text(json.dumps(r), encoding="utf-8")
    assert export.slides_status(job)["state"] == "none"  # nothing for this version yet
    gate.set()
    old_run.thread.join(20)
    assert old_run.state == "failed" and not (job / "export" / "deck.json").exists()  # never saved as this version's

    # A newer run takes over from an older one still writing: the older one stops and never saves its deck.
    gate2 = threading.Event()
    slow = FakeClient(gates={"plan": gate2})
    export.start_writing(job, lambda: slow)
    older = export._runs[export._key(job)]
    r["chapters"][0]["title"] = "Growth, third"
    (job / "replay.json").write_text(json.dumps(r), encoding="utf-8")
    fake2 = FakeClient()
    export.start_writing(job, lambda: fake2)
    assert older.cancelled
    gate2.set()
    older.thread.join(20)
    st = wait(job)
    assert st["state"] == "ready" and st["deck"]["version"] == export.deck_version((job / "replay.json").read_bytes())
    assert older.state == "failed" and slow.calls_for("deck_section") == []


def test_a_missing_figure_is_drawn_again_and_old_files_are_cleaned(job):
    (job / "export").mkdir()
    for name in ("page_001.png", "points.json", "fig_099.png"):  # the old page-image deck, an old figure
        (job / "export" / name).write_bytes(b"x")
    write(job, FakeClient())
    assert sorted(p.name for p in (job / "export").glob("*.png")) == ["fig_004.png"]
    assert not (job / "export" / "points.json").exists()
    (job / "export" / "fig_004.png").unlink()
    st = export.slides_status(job)
    assert st["state"] == "ready" and (job / "export" / "fig_004.png").exists()
    deck = json.loads((job / "export" / "deck.json").read_text(encoding="utf-8"))
    deck["slides"][3]["blocks"][0]["image"] = "replay.json"  # pointing outside its figures: not a ready deck
    export._save_json(job / "export" / "deck.json", deck)
    assert export.slides_status(job)["state"] == "none"


def test_a_slide_lecture_deck(tmp_path):
    d = write_job(tmp_path / "sl", slides_replay())
    plan = {"title": "Compound interest", "subtitle": "Interest on interest", "takeaways": ["It compounds"],
            "closing": "", "skip": [], "sections": [{"title": "Compound interest", "t0": "0:00", "t1": "0:40", "slides": [
                {"title": "Interest on interest", "t0": "0:00", "t1": "0:10", "covers": ["#s0e1", "#s0e2"], "focus": ""},
                {"title": "The growth chart", "t0": "0:10", "t1": "0:40", "covers": ["#s1e3"], "focus": ""}]}]}
    sections = {1: {"slides": [
        {"title": "Interest on interest", "t": "0:04", "notes": "",
         "blocks": [blk("bullets", items=[{"text": "Interest earns interest", "level": 0}])]},
        {"title": "The growth chart", "t": "0:12", "notes": "",
         "blocks": [blk("figure", refs=["#s1e3"], caption="Growth"), blk("equation", latex="A=P(1+r)^n", refs=["#s1e4"])]}]}}
    deck = write(d, FakeClient(plan=plan, sections=sections))["deck"]
    assert deck["kind"] == "slides" and [s["kind"] for s in deck["slides"]] == ["title", "content", "content", "summary"]
    fig = deck["slides"][2]["blocks"][0]
    assert fig["image"] == "export/fig_003.png" and imread(d / fig["image"], cv2.IMREAD_UNCHANGED).ndim == 3
    assert deck["slides"][1]["said"] == "hello. interest" and deck["slides"][2]["said"] == "grows. bye."


def test_formula_sheets_and_long_agendas():
    content = [{"title": f"Slide {i}", "blocks": [
        {"type": "equation", "latex": f"y_{i} = a x + {i}", "label": "", "refs": []},
        {"type": "equation", "latex": "y _1 = a x + 1", "label": "dup", "refs": []},  # a repeat
        {"type": "equation", "latex": f"y = {i} + {i}", "label": "", "refs": []}]}  # a worked number
        for i in range(1, 11)]
    sheets = export.formula_slides(content, 20)
    assert [s["id"] for s in sheets] == ["s20", "s21"] and [len(s["blocks"]) for s in sheets] == [8, 2]
    assert sheets[0]["blocks"][0]["label"] == "Slide 1" and sheets[1]["title"] == "Formula sheet (continued)"
    assert export.formula_slides(content[:2], 5) == []  # fewer than three formulas: no sheet
    lists = export._bullet_lists([f"Part {i}" for i in range(8)])
    assert [len(b["items"]) for b in lists] == [4, 4]


# ---------- the endpoints ----------
@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JOBS", tmp_path)
    write_job(tmp_path / "lec", lecture_replay())

    def use(fake):
        monkeypatch.setattr(server, "_meta_client", fake if callable(fake) else (lambda: fake))
        return fake

    return TestClient(server.app), tmp_path / "lec", use


@pytest.fixture
def renderer(monkeypatch):
    """app.pptx_deck as the PowerPoint track's module would be: build_pptx(job_dir, deck) -> bytes."""
    mod = types.ModuleType("app.pptx_deck")
    mod.RULES_VERSION = "1"
    mod.built = []

    def build_pptx(job_dir, deck):
        mod.built.append(deck["version"])
        return b"PK\x03\x04 a deck called " + deck["title"].encode("utf-8")

    mod.build_pptx = build_pptx
    monkeypatch.setitem(sys.modules, "app.pptx_deck", mod)
    return mod


def test_slides_need_a_processed_lecture(api):
    client, d, use = api
    use(lambda: pytest.fail("no model call before the lecture is processed"))
    (d.parent / "raw").mkdir()
    for method, url in (("get", "slides.json"), ("post", "slides/write"), ("get", "slides.pptx")):
        r = getattr(client, method)(f"/api/jobs/raw/{url}")
        assert r.status_code == 409 and "not processed yet" in r.json()["detail"] and "\n" not in r.json()["detail"]
    assert client.get("/api/jobs/nope/slides.json").status_code == 404
    assert client.post("/api/jobs/lec/slides/points").status_code == 405  # the old key points endpoint is gone


def test_writing_through_the_endpoints(api, renderer):
    client, d, use = api
    fake = use(FakeClient())
    r = client.get("/api/jobs/lec/slides.json")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    assert r.json() == {"state": "none", "step": "", "done": 0, "total": 0, "error": None, "deck": None}
    r = client.get("/api/jobs/lec/slides.pptx")
    assert r.status_code == 409 and r.json()["detail"] == "Open the Slides page first so Muse Spark can write the slides."

    r = client.post("/api/jobs/lec/slides/write", headers={"origin": "https://evil.example"})
    assert r.status_code == 403 and fake.seen == []
    r = client.post("/api/jobs/lec/slides/write", headers={"origin": "http://localhost:5173"})
    assert r.status_code == 200 and r.json()["state"] in ("writing", "ready")
    wait(d)
    data = client.get("/api/jobs/lec/slides.json").json()
    assert data["state"] == "ready" and data["deck"]["slides"][3]["blocks"][0]["image"] == "export/fig_004.png"
    assert client.get(f"/api/jobs/lec/files/{data['deck']['slides'][3]['blocks'][0]['image']}").content[:4] == b"\x89PNG"
    assert client.post("/api/jobs/lec/slides/write").json() == data and len(fake.seen) == 4

    r = client.get("/api/jobs/lec/slides.pptx")
    assert r.status_code == 200 and r.content.startswith(b"PK\x03\x04")
    assert r.headers["content-type"] == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    assert r.headers["content-disposition"].startswith(
        'attachment; filename="Exponential growth and compounding - slides.pptx"')
    assert (d / "export" / "slides.pptx").read_bytes() == r.content
    assert client.get("/api/jobs/lec/slides.pptx").content == r.content and len(renderer.built) == 1  # kept
    renderer.RULES_VERSION = "2"  # the renderer changed: made again
    assert client.get("/api/jobs/lec/slides.pptx").status_code == 200 and len(renderer.built) == 2


def test_a_powerpoint_open_in_powerpoint_does_not_hold_up_the_download(api, renderer, monkeypatch):
    client, d, use = api
    use(FakeClient())
    client.post("/api/jobs/lec/slides/write")
    wait(d)
    tries = []

    def locked(tmp, path, n=40):  # Windows: export/slides.pptx is open in PowerPoint
        tries.append(n)
        return False

    monkeypatch.setattr(export, "replace_file", locked)
    r = client.get("/api/jobs/lec/slides.pptx")
    assert r.status_code == 200 and tries == [2] and not list((d / "export").glob("*.tmp"))


def test_an_unexpected_endpoint_failure_is_one_line_with_a_log(api, monkeypatch):
    client, d, use = api
    monkeypatch.setattr(export, "slides_status", lambda d: 1 / 0)
    r = client.get("/api/jobs/lec/slides.json")
    assert r.status_code == 500 and "ZeroDivisionError" in r.json()["detail"] and "\n" not in r.json()["detail"]
    assert (d / "export-error.log").exists()


def test_control_characters_and_lone_surrogates_are_dropped_everywhere(job):
    sections = copy.deepcopy(SECTIONS)
    sections[1]["slides"][0]["title"] = "Growth\x02 \udc00rate"
    sections[1]["slides"][0]["blocks"][1]["items"][0]["text"] = "Bad\x0b char \ud83d here"
    deck = write(job, FakeClient(sections=sections))["deck"]
    g1 = deck["slides"][2]
    assert g1["title"] == "Growth rate" and g1["blocks"][1]["items"][0]["text"] == "Bad char here"
    json.dumps(deck).encode("utf-8")


# ---------- found in review ----------
def test_percent_signs_and_broken_latex_never_reach_katex():
    # KaTeX reads a bare % as a comment: "r=100%" showed "r=100", and "\frac{50%}{2}" failed to render.
    assert export.display_tex("r=100%") == "r=100\\%" and export.display_tex("r=100\\%") == "r=100\\%"
    assert export.display_tex("\\frac{50%}{2}") == "\\frac{50\\%}{2}"
    assert export.inline_text("at $r=100%$ a year", 120) == "at $r=100\\%$ a year"
    assert export.inline_text("at 100% a year", 120) == "at 100% a year"  # text, not maths: unchanged
    # LaTeX KaTeX cannot parse: an unmatched \left, a power or index with nothing after it.
    for bad in ("x^", "a_ ", "{x^}", "\\left( x"):
        assert export.display_tex(bad) == "" and not export.simple_tex(bad), bad
    assert export.inline_text("so $\\left(1+x$ grows", 120) == "so (1+x grows"
    assert export.simple_tex("\\left(1+x\\right)^2") and export.display_tex("x^2_1") == "x^2_1"


def test_a_dim_whiteboard_keeps_its_ink_colours(tmp_path):
    # The board's own mode decides: a whiteboard filmed in dim light (its background measured darker than mid-grey,
    # test2's is #8f8d8d) must not have black ink turned white on the white slide.
    dark_ink = (30, 30, 30)
    r = board([piece(1, (100, 100, 80, 60), 5, kind="graph", spoken=[(5, 6)])], [(0, "Intro")], bg="#6e6e6c")
    r["board"]["mode"] = "white"
    d = write_job(tmp_path / "dim", r, colors={1: dark_ink})
    img = export.board_figure(lec_of(d), ["#1"])
    px = np.median(img[img[..., 3] == 255][:, :3], axis=0)
    assert np.abs(px - dark_ink).max() <= 3
    r["board"]["mode"] = "chalk"  # a chalkboard: light chalk becomes dark on the white slide
    d = write_job(tmp_path / "chalk2", r, colors={1: (235, 235, 235)})
    img = export.board_figure(lec_of(d), ["#1"])
    assert np.median(img[img[..., 3] == 255][:, :3]) < 40


def test_the_powerpoint_is_made_again_when_formulas_can_be_typeset_or_the_deck_changes(api, renderer):
    client, d, use = api
    use(FakeClient())
    client.post("/api/jobs/lec/slides/write")
    wait(d)
    renderer.typeset = False  # node missing: formulas as text, and app.pptx_deck says so in rules_version()
    renderer.rules_version = lambda: renderer.RULES_VERSION + ("" if renderer.typeset else "-text")
    assert client.get("/api/jobs/lec/slides.pptx").status_code == 200 and len(renderer.built) == 1
    assert client.get("/api/jobs/lec/slides.pptx").status_code == 200 and len(renderer.built) == 1  # kept
    renderer.typeset = True  # node installed: the PowerPoint is made again, with typeset formulas
    assert client.get("/api/jobs/lec/slides.pptx").status_code == 200 and len(renderer.built) == 2
    # The same lecture version written again (deck.json lost, then rewritten): its PowerPoint is not the old one.
    deck = json.loads((d / "export" / "deck.json").read_text(encoding="utf-8"))
    deck["title"] = "Another title"
    export._save_json(d / "export" / "deck.json", deck)
    r = client.get("/api/jobs/lec/slides.pptx")
    assert len(renderer.built) == 3 and r.content.endswith(b"Another title")


def test_a_lost_deck_is_written_again_from_the_cache(job):
    write(job, FakeClient())
    (job / "export" / "deck.json").unlink()  # the owner deleted it, the server still running
    fake = FakeClient()
    assert write(job, fake)["state"] == "ready"
    # The same requests as the first time (the disk cache answers them for free), not a new paid plan.
    assert len(fake.seen) == 4 and all("Try" not in messages[-1]["content"] for messages, _ in fake.seen)


def test_replay_json_busy_at_the_end_of_a_run_does_not_fail_it(job, monkeypatch):
    # Windows refuses a read for a moment while the pipeline replaces a file: the run checks again.
    real = export.Path.read_bytes
    fails = {"n": 0}

    def flaky(self):
        if self.name == "replay.json" and threading.current_thread().name.startswith("slides-"):
            if fails["n"] == 0:
                fails["n"] += 1
                raise PermissionError(13, "The process cannot access the file", str(self))
        return real(self)

    monkeypatch.setattr(export.Path, "read_bytes", flaky)
    st = write(job, FakeClient())
    assert fails["n"] == 1 and st["state"] == "ready" and st["error"] is None


def test_a_clip_that_starts_late(tmp_path):
    r = board([piece(1, (100, 40, 120, 40), 120, kind="equation", latex="A=P(1+r)^n", spoken=[(121, 124)]),
               old(2, (500, 40, 100, 30), text="an earlier lesson")], [(100, "Intro")], start=100.0, end=280.0)
    d = write_job(tmp_path / "late", r)
    plan = {"title": "Late", "subtitle": "What it covers.", "takeaways": ["a"], "closing": "", "skip": [],
            "sections": [{"title": "Intro", "t0": "1:40", "t1": "4:40", "slides": [
                {"title": "One", "t0": "1:40", "t1": "2:30", "covers": ["#1"], "focus": ""},
                {"title": "Two", "t0": "2:30", "t1": "4:40", "covers": [], "focus": ""}]}]}
    sections = {1: {"slides": [
        {"title": "One", "t": "2:01", "notes": "", "blocks": [blk("equation", latex="A=P(1+r)^n", refs=["#1"])]},
        {"title": "Two", "t": "0:10", "notes": "", "blocks": [blk("bullets", items=[{"text": "b", "level": 0}])]}]}}
    fake = FakeClient(plan=plan, sections=sections)
    deck = write(d, fake)["deck"]
    assert deck["clip"] == [100.0, 280.0] and deck["duration"] == 180.0
    assert [(s["t0"], s["t1"], s["t"]) for s in deck["slides"][1:3]] == [(100, 150, 121.0), (150, 280, 150)]
    assert deck["slides"][1]["said"].startswith("w0. ") and "w49." in deck["slides"][1]["said"]
    assert "from 1:40 to 4:40" in fake.calls_for("deck_plan")[0][1]["content"]
    assert deck["slides"][0]["notes"].startswith("What it covers. The lecture runs 3:00")  # one full stop
