"""The PowerPoint of a format-2 deck (app/pptx_deck.py), from hand-written decks of a made-up lecture.

The figures are drawn by code here; nothing comes from a real job.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app import pptx_deck
from app.core import texrender

EMU_PER_PT = 12700


# ---------- fixtures: a made-up lecture on projectile motion ----------
def draw_trajectory(path: Path, size=(900, 520)) -> None:
    """A board-like sketch: axes and three launch arcs."""
    w, h = size
    img = Image.new("RGB", size, (247, 247, 245))
    d = ImageDraw.Draw(img)
    ox, oy = 60, h - 60
    d.line([(ox, oy), (w - 30, oy)], fill=(40, 40, 50), width=4)
    d.line([(ox, oy), (ox, 30)], fill=(40, 40, 50), width=4)
    k = min(w - 140, (oy - 40) / (math.tan(math.radians(60)) / 4))  # px per unit of range: the 60° arc fits
    for deg, col in ((30, (30, 80, 200)), (45, (200, 40, 40)), (60, (30, 140, 60))):
        a = math.radians(deg)
        rng = math.sin(2 * a)
        pts = [(ox + x * k, oy - (x * math.tan(a) - x * x * math.tan(a) / rng) * k)
               for x in (rng * j / 100 for j in range(101))]
        d.line(pts, fill=col, width=5)
    d.ellipse([ox - 8, oy - 8, ox + 8, oy + 8], fill=(40, 40, 50))
    img.save(path)


def draw_chart(path: Path, size=(1280, 720)) -> None:
    """A slide-like bar chart."""
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 90], fill=(30, 60, 120))
    for i, v in enumerate((0.5, 0.87, 1.0, 0.87, 0.5)):
        x = 140 + i * 210
        d.rectangle([x, h - 80 - v * 480, x + 140, h - 80], fill=(37, 99, 235))
    d.line([(100, h - 80), (w - 80, h - 80)], fill=(60, 60, 60), width=4)
    img.save(path)


def figures(job: Path) -> None:
    (job / "export").mkdir(parents=True, exist_ok=True)
    draw_trajectory(job / "export" / "fig_001.png")
    draw_chart(job / "export" / "fig_002.png")
    draw_trajectory(job / "export" / "fig_003.png", (1400, 360))


def bullets(*items):
    return {"type": "bullets", "items": [{"text": t, "level": lv} for t, lv in
                                         ((i, 0) if isinstance(i, str) else i for i in items)]}


def eq(latex, label="", refs=()):
    return {"type": "equation", "latex": latex, "label": label, "refs": list(refs)}


def slide(sid, kind, title, section="", t0=None, t1=None, blocks=(), notes="", said=""):
    return {"id": sid, "kind": kind, "title": title, "section": section, "t0": t0, "t1": t1,
            "t": t0, "blocks": list(blocks), "notes": notes, "said": said}


def lecture_deck() -> dict:
    """A board lecture with every block type, short and long content, three sections."""
    S1, S2, S3 = "Splitting the launch velocity", "How long and how far", "Choosing the angle"
    slides = [
        slide("s1", "title", "Projectile motion: range and height", notes="Today: how far and how high a launched "
              "ball goes."),
        slide("s2", "agenda", "Agenda"),
        slide("s3", "section", S1, S1, 12.0, 190.0),
        slide("s4", "content", "Two motions at once", S1, 12.0, 70.5, [
            bullets("A launched ball moves sideways and up at the same time",
                    "Sideways: no force, so the speed $v_x$ stays constant",
                    "Up and down: gravity pulls with $g \\approx 9.8\\ \\mathrm{m/s^2}$",
                    ("The two motions do not affect each other", 1)),
            {"type": "figure", "refs": ["#12"], "caption": "The board's sketch: launches at 30°, 45° and 60°",
             "image": "export/fig_001.png", "size": [900, 520]},
        ], notes="Treat the motion as two independent parts.", said="so what happens when you throw a ball " * 8),
        slide("s5", "content", "Components of the launch velocity", S1, 70.5, 130.0, [
            {"type": "callout", "style": "definition", "title": "Components",
             "text": "A velocity $v_0$ at angle $\\theta$ splits into a sideways part and an upward part.",
             "latex": ""},
            eq(r"v_x = v_0\cos\theta", "Sideways"),
            eq(r"v_y = v_0\sin\theta - g t", "Upward, at time t"),
        ], notes="Resolve the launch velocity.", said="we split v naught into two parts"),
        slide("s6", "content", "Time of flight", S1, 130.0, 190.0, [
            {"type": "steps", "label": "Worked example: when does it land?", "items": [
                {"latex": r"y(t) = v_0 \sin\theta\, t - \tfrac{1}{2} g t^2", "text": "Height at time $t$"},
                {"latex": r"0 = t\left(v_0\sin\theta - \tfrac{1}{2} g t\right)", "text": "It lands when $y = 0$"},
                {"latex": r"T = \frac{2 v_0 \sin\theta}{g}", "text": "The other root is the flight time"},
                {"latex": "", "text": "Doubling the launch speed doubles the time in the air"},
            ]},
        ], notes="Set the height to zero and solve.", said="when does it come back down"),
        slide("s7", "content", "Range for different angles", S2, 190.0, 260.0, [
            {"type": "table", "header": ["Angle $\\theta$", "Range (m)", "Max height (m)", "Flight time (s)"],
             "rows": [["15°", "20.4", "1.4", "1.06"], ["30°", "35.3", "5.1", "2.04"], ["45°", "40.8", "10.2", "2.89"],
                      ["60°", "35.3", "15.3", "3.53"], ["75°", "20.4", "19.0", "3.94"]],
             "caption": "For $v_0 = 20\\ \\mathrm{m/s}$ and $g = 9.8\\ \\mathrm{m/s^2}$"},
            bullets("Complementary angles (30° and 60°) give the same range",
                    "The height keeps growing with the angle"),
        ], notes="Read the table row by row.", said="here are the numbers for twenty metres per second"),
        slide("s8", "content", "The range formula", S2, 260.0, 330.0, [
            eq(r"R = \frac{v_0^2 \sin 2\theta}{g}", "Range on level ground"),
            bullets("Range grows with the square of the launch speed: $R \\propto v_0^2$",
                    "It is largest when $\\sin 2\\theta = 1$, so at $\\theta = 45^\\circ$"),
        ], notes="Multiply flight time by sideways speed.", said="the range is v x times capital T"),
        slide("s9", "content", "Maximum height", S2, 330.0, 372.0, [
            eq(r"H = \frac{v_0^2 \sin^2\theta}{2g}", "Highest point"),
            bullets("At the top the upward speed is zero: $v_y = 0$",
                    ("Half the flight time, by symmetry", 1),
                    ("It does not depend on the sideways speed", 1),
                    "A steeper launch goes higher but not always further"),
        ], notes="At the top the vertical velocity is zero.", said="at the top it stops going up"),
        slide("s10", "content", "Why 45° gives the longest range", S3, 372.0, 430.0, [
            {"type": "callout", "style": "key", "title": "The trade-off",
             "text": "A steeper launch stays up longer but moves sideways more slowly; at 45° the product of the two "
                     "is largest.", "latex": r"R \propto \sin\theta\cos\theta = \tfrac{1}{2}\sin 2\theta"},
            bullets("Below 45°: fast sideways, but it lands quickly",
                    "Above 45°: long flight, slow sideways motion",
                    "Air resistance moves the best angle lower, to about 35°–40° for a real ball"),
        ], notes="Explain the trade-off.", said="it is a trade off between time and speed"),
        slide("s11", "content", "Everything on one slide: a crowded summary of the angle study", S3, 430.0, 470.0, [
            bullets(*["A long point about how the launch angle, the launch speed and gravity together decide where "
                      "the ball lands on the ground" for _ in range(5)]),
            {"type": "table", "header": ["Quantity", "Formula", "Depends on"],
             "rows": [["Range", "$R = v_0^2 \\sin 2\\theta / g$", "speed, angle, gravity"],
                      ["Height", "$H = v_0^2 \\sin^2\\theta / 2g$", "speed, angle, gravity"],
                      ["Time", "$T = 2 v_0 \\sin\\theta / g$", "speed, angle, gravity"],
                      ["Sideways speed", "$v_x = v_0 \\cos\\theta$", "speed, angle"]],
             "caption": "All three come from the same two components"},
            eq(r"y = x\tan\theta - \frac{g x^2}{2 v_0^2 \cos^2\theta}", "The path is a parabola"),
        ], notes="A lot on one slide.", said="let me put it all together"),
        slide("s12", "content", "The path of the ball", S3, 470.0, 478.0, [
            {"type": "figure", "refs": ["#20"], "caption": "Three launches drawn to scale",
             "image": "export/fig_003.png", "size": [1400, 360]},
        ], notes="Show the drawing.", said=""),
        slide("s13", "summary", "Key takeaways", "", 12.0, 478.0, [
            bullets("Split the launch velocity into $v_0\\cos\\theta$ and $v_0\\sin\\theta$",
                    "The flight time is $T = 2 v_0 \\sin\\theta / g$",
                    "The range $R = v_0^2 \\sin 2\\theta / g$ is largest at 45°",
                    "Complementary angles give the same range"),
        ], notes="Recap."),
        slide("s14", "formulas", "Formula sheet", blocks=[
            eq(r"v_x = v_0\cos\theta", "Sideways"),
            eq(r"v_y = v_0\sin\theta - g t", "Upward, at time t"),
            eq(r"T = \frac{2 v_0 \sin\theta}{g}", "Time of flight"),
            eq(r"R = \frac{v_0^2 \sin 2\theta}{g}", "Range on level ground"),
            eq(r"H = \frac{v_0^2 \sin^2\theta}{2g}", "Highest point"),
            eq(r"y = x\tan\theta - \frac{g x^2}{2 v_0^2 \cos^2\theta}", "The path is a parabola"),
        ], notes="All formulas from the lecture."),
    ]
    return {"format": 2, "version": "test-1", "kind": "board", "title": "Projectile motion: range and height",
            "subtitle": "How far and how high a launched ball goes, from the two parts of its velocity",
            "lecture": "Physics 101 - projectile motion", "duration": 478.1, "clip": [0.0, 478.1],
            "slides": slides, "degraded": []}


def slides_lecture_deck() -> dict:
    """A slide lecture: one section (no agenda, no section slide), a slide picture, no formulas."""
    S = "Measuring launch speed"
    slides = [
        slide("s1", "title", "Measuring the launch speed of a toy rocket"),
        slide("s2", "content", "The experiment", S, 0.0, 41.0, [
            bullets("Launch the rocket five times at each angle", "Film it at 120 frames per second",
                    "Measure where it lands with a tape"),
            {"type": "figure", "refs": ["#s1e2"], "caption": "Results slide: landing distance by launch angle",
             "image": "export/fig_002.png", "size": [1280, 720]},
        ], notes="Describe the setup.", said="we launched it five times"),
        slide("s3", "content", "What we found", S, 41.0, 95.0, [
            {"type": "callout", "style": "note", "title": "Not quite 45°",
             "text": "The longest flights came at about 40°, because of air drag.", "latex": ""},
            bullets("Launch speed estimated at $v_0 \\approx 14\\ \\mathrm{m/s}$", "Spread between runs about 5%"),
        ], notes="Summarise the results.", said="the best angle was around forty degrees"),
        slide("s4", "summary", "Key takeaways", "", 0.0, 95.0, [
            bullets("Drag lowers the best angle", "Five runs per angle were enough", "Film fast to time the flight")]),
    ]
    return {"format": 2, "version": "test-2", "kind": "slides", "title": "Measuring the launch speed of a toy rocket",
            "subtitle": "A short lab report", "lecture": "Lab 3", "duration": 95.0, "clip": [0.0, 95.0],
            "slides": slides, "degraded": []}


# ---------- helpers ----------
def open_pptx(blob: bytes):
    from pptx import Presentation
    return Presentation(io.BytesIO(blob))


def boxes(slide):
    """(name, left, top, right, bottom) in points of every top-level shape."""
    return [(sh.name, sh.left / EMU_PER_PT, sh.top / EMU_PER_PT, (sh.left + sh.width) / EMU_PER_PT,
             (sh.top + sh.height) / EMU_PER_PT) for sh in slide.shapes]


def overlap(a, b) -> float:
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1])) * max(0.0, min(a[4], b[4]) - max(a[2], b[2]))


def no_overlaps(slide) -> bool:
    bs = boxes(slide)
    return all(overlap(bs[i], bs[j]) < 0.5 for i in range(len(bs)) for j in range(i + 1, len(bs)))


def walk(shapes):
    """Every shape, inside groups too."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from walk(sh.shapes)
        else:
            yield sh


def runs_of(shape):
    """(the shape, run) for every text run in a shape (groups and tables included)."""
    for sh in walk([shape]):
        if getattr(sh, "has_table", False) and sh.has_table:
            for row in sh.table.rows:
                for cell in row.cells:
                    for p in cell.text_frame.paragraphs:
                        yield from ((sh, r) for r in p.runs)
        elif getattr(sh, "has_text_frame", False) and sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                yield from ((sh, r) for r in p.runs)


def pictures(slide):
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    return [sh for sh in walk(slide.shapes) if sh.shape_type == MSO_SHAPE_TYPE.PICTURE]


def alt(pic) -> str:
    return pic._element.nvPicPr.cNvPr.get("descr")


def fake_render(folder: Path, broken=()):
    """texrender.render without node: every formula a plain picture sized from its length."""
    folder.mkdir(parents=True, exist_ok=True)

    def render(latexes, **kw):
        out = {}
        for k, latex in enumerate(latexes):
            if latex in broken:
                out[latex] = None
                continue
            w_em, h_em = 0.25 * len(latex) + 1, 2.2 if "frac" in latex else 1.3
            px = (int(w_em * 60), int(h_em * 60))
            path = folder / f"tex_{k}.png"
            Image.new("RGBA", px, (31, 42, 55, 255)).save(path)
            out[latex] = texrender.TexImage(latex, path, w_em, h_em, 0.3, px)
        return out
    return render


def build(job: Path, deck: dict, monkeypatch, tmp: Path, broken=()):
    monkeypatch.setattr(pptx_deck.texrender, "render", fake_render(tmp / "tex", broken))
    return open_pptx(pptx_deck.build_pptx(job, deck))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Both fixture decks, built once (formulas as plain pictures) and re-opened from their bytes."""
    tmp = tmp_path_factory.mktemp("built")
    job = tmp / "job"
    figures(job)
    with pytest.MonkeyPatch.context() as mp:
        return {"lecture": build(job, lecture_deck(), mp, tmp / "a"),
                "slides": build(job, slides_lecture_deck(), mp, tmp / "b")}


@pytest.fixture
def job(tmp_path):
    d = tmp_path / "job"
    figures(d)
    return d


def title_of(slide) -> str:
    return slide.shapes.title.text_frame.text


# ---------- the deck ----------
def test_slides_in_the_deck_order_with_real_title_placeholders(built):
    prs = built["lecture"]
    assert (prs.slide_width, prs.slide_height) == (12192000, 6858000)  # 13.333 x 7.5 in
    deck = lecture_deck()
    assert [title_of(s) for s in prs.slides] == [s["title"] for s in deck["slides"]]
    layouts = [s.slide_layout.name for s in prs.slides]
    assert layouts[0] == "Title Slide" and layouts[2] == "Section Header"
    assert all(name == "Title Only" for i, name in enumerate(layouts) if i not in (0, 2))
    for s in prs.slides:
        ph = s.shapes.title
        assert ph.is_placeholder and ph.placeholder_format.idx == 0
        for p in s.placeholders:  # no empty placeholder left ("Click to add text")
            assert p.text_frame.text.strip()
    assert [title_of(s) for s in built["slides"].slides] == [s["title"] for s in slides_lecture_deck()["slides"]]


def test_document_properties_name_the_lecture_not_the_template(built):
    props = built["lecture"].core_properties
    assert props.title == "Projectile motion: range and height" and props.author == "Replay Board"
    assert props.last_modified_by == "Replay Board"  # python-pptx's template says "Steve Canny", 2013
    assert props.created.year >= 2026 and props.modified == props.created


def test_every_shape_inside_the_slide_and_none_overlapping(built):
    for name in ("lecture", "slides"):
        for n, s in enumerate(built[name].slides, 1):
            bs = boxes(s)
            for b in bs:
                assert b[1] >= -0.01 and b[2] >= -0.01 and b[3] <= 960.01 and b[4] <= 540.01, (name, n, b)
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    assert overlap(bs[i], bs[j]) < 0.5, (name, n, bs[i], bs[j])


def test_font_sizes_are_readable(built):
    quiet = {pptx_deck.MUTED, pptx_deck.ACCENT, pptx_deck.ON_ACCENT, pptx_deck.WHITE}
    body_sizes = []
    for name in ("lecture", "slides"):
        for n, s in enumerate(built[name].slides, 1):
            for sh in s.shapes:
                for owner, r in runs_of(sh):
                    size = r.font.size.pt
                    color = str(r.font.color.rgb)
                    if owner.name in ("Footer", "Slide number") or color in quiet:
                        assert size >= 11, (name, n, r.text, size)  # footers, captions, labels, step numbers
                    else:
                        assert size >= 14, (name, n, owner.name, r.text, size)
                        body_sizes.append(size)
    assert max(body_sizes) >= 24 and min(body_sizes) >= 14


def test_footer_and_speaker_notes(built):
    prs = built["lecture"]
    s4 = prs.slides[3]
    footer = next(sh for sh in s4.shapes if sh.name == "Footer").text_frame.text
    # The deck's title, not the job's name (that can be the subject of writing left from an earlier lesson).
    assert footer == "Projectile motion: range and height  ·  0:12–1:10"
    assert next(sh for sh in s4.shapes if sh.name == "Slide number").text_frame.text == "4"
    notes = s4.notes_slide.notes_text_frame.text
    assert notes.startswith("Treat the motion as two independent parts.\n\nFrom the lecture (0:12–1:10):\nso what")
    assert prs.slides[0].notes_slide.notes_text_frame.text == "Today: how far and how high a launched ball goes."
    assert prs.slides[12].notes_slide.notes_text_frame.text == "Recap."  # nothing said: no "From the lecture"
    for i in (0, 2):  # no footer on the title and section slides
        assert not any(sh.name == "Footer" for sh in prs.slides[i].shapes)
    formulas = next(sh for sh in prs.slides[13].shapes if sh.name == "Footer").text_frame.text
    assert formulas == "Projectile motion: range and height"  # no time range


def test_inline_maths_has_real_superscripts_and_subscripts(built):
    s = built["lecture"].slides[7]  # "The range formula"
    runs = [r for sh in s.shapes for _, r in runs_of(sh)]
    subs = [r.text for r in runs if r.font._rPr.get("baseline") == "-25000"]
    sups = [r.text for r in runs if r.font._rPr.get("baseline") == "30000"]
    assert "0" in subs and "2" in sups
    text = "".join(r.text for r in runs)
    assert "$" not in text and "\\" not in text and "R ∝ v02" in text and "θ = 45°" in text
    assert any(r.font.name == "Cambria" and r.font.italic and r.text == "R" for r in runs)


def test_summary_takeaways_are_numbered_rows(built):
    s = built["lecture"].slides[12]
    assert title_of(s) == "Key takeaways"
    numbers = [sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.isdigit()
               and sh.name != "Slide number"]
    assert numbers == ["1", "2", "3", "4"]


def test_tables_are_native_with_an_accent_header(built):
    s = built["lecture"].slides[6]
    gf = next(sh for sh in s.shapes if getattr(sh, "has_table", False) and sh.has_table)
    t = gf.table
    assert len(t.rows) == 6 and len(t.columns) == 4
    head = t.cell(0, 0)
    assert str(head.fill.fore_color.rgb) == pptx_deck.ACCENT
    run = head.text_frame.paragraphs[0].runs[0]
    assert run.font.bold and str(run.font.color.rgb) == "FFFFFF"
    assert "".join(r.text for r in head.text_frame.paragraphs[0].runs) == "Angle θ"
    assert str(t.cell(2, 0).fill.fore_color.rgb) == pptx_deck.BAND


def test_pictures_have_alt_text_and_keep_their_aspect(built):
    pics = [p for s in built["lecture"].slides for p in pictures(s)]
    alts = [alt(p) for p in pics]
    assert r"R = \frac{v_0^2 \sin 2\theta}{g}" in alts
    assert "The board's sketch: launches at 30°, 45° and 60°" in alts
    for p in pics:
        with Image.open(io.BytesIO(p.image.blob)) as im:
            assert abs(p.width / p.height - im.width / im.height) < 0.01 * im.width / im.height


def test_section_slides_and_agenda(built):
    prs = built["lecture"]
    sec = prs.slides[2]
    assert sec.placeholders[1].text_frame.text == "PART 1"
    assert sec.shapes.title.text_frame.paragraphs[0].runs[0].font._rPr.get("cap") == "none"
    agenda = " ".join(sh.text_frame.text for sh in prs.slides[1].shapes if sh.has_text_frame)
    for want in ("01", "Splitting the launch velocity", "0:12–3:10", "03", "Choosing the angle", "6:12–7:58"):
        assert want in agenda
    eyebrows = [sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.name == "Section"]
    assert eyebrows[0] == "Splitting the launch velocity" and len(eyebrows) == 9
    assert not any(sh.name == "Section" for s in built["slides"].slides for sh in s.shapes)  # one section only


# ---------- inline maths ----------
def text_of(latex: str) -> str:
    return "".join(r.text for r in pptx_deck.tex_runs(latex))


def test_tex_runs_scripts_and_symbols():
    assert [(r.text, r.script) for r in pptx_deck.tex_runs(r"v_0^2")] == [("v", 0), ("0", -1), ("2", 1)]
    assert text_of(r"\frac{v_0^2 \sin 2\theta}{g}") == "(v02\u2009sin\u20092θ)/g"
    assert text_of(r"\frac{1}{2} g t^2") == "1/2\u2009gt2"
    assert text_of(r"45^\circ") == "45°"
    assert text_of(r"\alpha \le \beta \to \infty") == "α ≤ β → ∞"
    assert text_of(r"f(x) = -2x + 1") == "f(x) = −2x + 1"
    assert text_of(r"\sin^2\theta") == "sin2\u2009θ"
    assert text_of(r"\unknowncmd{x} + \left( y \right)") == "x + (y)"
    assert text_of(r"x \in \mathbb{R}, \; \vec{F} = m\vec{a}") == "x ∈ ℝ, F\u20d7 = ma\u20d7"
    assert text_of(r"\sqrt{2gh}") == "√(2gh)" and text_of(r"\sqrt{x}") == "√x"
    assert text_of(r"g \approx 9.8\ \mathrm{m/s^2}") == "g ≈ 9.8 m/s2"
    assert text_of(r"x^{2") == "x2" and text_of("}{") == ""  # broken LaTeX still gives text
    # a length is a space, not text ('x1emy' once)
    assert text_of(r"x \hspace{1em} y") == "x y" and text_of(r"a\kern 2pt b\vspace{3pt}") == "a b"
    italic = {r.text: r.italic for r in pptx_deck.tex_runs(r"x \sin y")}
    assert italic["x"] and not italic["\u2009sin\u2009"]


def test_lower_case_greek_is_italic_like_other_variables():
    # TeX sets lower-case Greek in italic: "a = b, α = β" had upright α, β next to italic a, b (seen live, test2)
    assert [(r.text, r.italic) for r in pptx_deck.tex_runs(r"\alpha=\Gamma")] == [("α", True), (" = Γ", False)]
    assert [(r.text, r.italic) for r in pptx_deck.tex_runs(r"\mathrm{\mu m}")] == [("μm", False)]


def test_rich_text_keeps_money_and_reads_maths():
    assert "".join(r.text for r in pptx_deck.rich(r"It costs $5 or $10, and $x^2$ grows")) == \
        "It costs $5 or $10, and x2 grows"
    runs = pptx_deck.rich(r"speed $v_x$ here")
    assert [(r.text, r.script, r.math) for r in runs] == [("speed ", 0, False), ("v", 0, True), ("x", -1, True),
                                                           (" here", 0, False)]
    assert "".join(r.text for r in pptx_deck.rich(r"a \$5 note and $ x$ alone")) == "a $5 note and $ x$ alone"


def test_display_fallback_uses_maths_italic_letters():
    text = "".join(r.text for r in pptx_deck.display_runs(r"h_{max} = \frac{v_0^2}{2g}"))
    assert text.startswith("ℎ𝑚𝑎𝑥 = ") and "𝑣" in text and "/" in text


def test_a_backslash_before_a_non_ascii_letter_does_not_break_the_deck(job, tmp_path, monkeypatch):
    # '\μ' is not a TeX command (their names are ASCII letters): it once crashed the whole PowerPoint.
    assert text_of(r"\μ + \Ω") == "μ + Ω"
    assert "".join(r.text for r in pptx_deck.rich(r"a mass of $3\,\μ g$ here")) == "a mass of 3\u2009μg here"
    assert "𝜇" in "".join(r.text for r in pptx_deck.display_runs(r"\μ = 1"))  # maths italic, as \mu would be
    deck = slides_lecture_deck()
    deck["slides"][2]["blocks"] = [bullets(r"Odd maths $\μ_0$ in a point"), eq(r"\μ_0 = 4\pi", "Odd")]
    prs = build(job, deck, monkeypatch, tmp_path, broken={r"\μ_0 = 4\pi"})  # MathJax fails it: shown as text
    runs = [r for sh in prs.slides[2].shapes for _, r in runs_of(sh)]
    assert "Odd maths μ0 in a point" in "".join(r.text for r in runs)
    assert any(r.font.name == "Cambria Math" and "𝜇" in r.text for r in runs)


# ---------- layout ----------
def all_latex(deck: dict) -> set:
    out = set()
    for s in deck["slides"]:
        for b in s["blocks"]:
            out.add(b.get("latex"))
            out |= {it.get("latex") for it in b.get("items") or []}
    return out - {None, ""}


def test_builds_without_node_formulas_as_text(job, tmp_path, monkeypatch):
    deck = lecture_deck()
    prs = build(job, deck, monkeypatch, tmp_path, broken=all_latex(deck))
    runs = [r for s in prs.slides for sh in s.shapes for _, r in runs_of(sh)]
    maths = [r for r in runs if r.font.name == "Cambria Math"]
    assert maths and any("𝑣" in r.text for r in maths)
    assert sorted(alt(p) for s in prs.slides for p in pictures(s)) == sorted(
        ["The board's sketch: launches at 30°, 45° and 60°", "Three launches drawn to scale"])
    assert all(no_overlaps(s) for s in prs.slides)


def test_missing_or_outside_figure_is_left_out(job, tmp_path, monkeypatch):
    deck = lecture_deck()
    (job / "export" / "fig_001.png").unlink()
    deck["slides"][11]["blocks"][0]["image"] = "export/../../fig_003.png"
    prs = build(job, deck, monkeypatch, tmp_path)
    assert not any("drawn to scale" in alt(p) or "sketch" in alt(p) for s in prs.slides for p in pictures(s))
    assert len(prs.slides) == len(deck["slides"])


def test_short_formulas_in_a_row_sit_side_by_side(built):
    cards = [sh for sh in built["lecture"].slides[4].shapes if sh.name == "Equation"]  # "Components of ..."
    assert len(cards) == 2
    a, b = sorted(cards, key=lambda sh: sh.left)
    assert a.top == b.top and a.height == b.height and a.left + a.width < b.left
    wide = [sh for sh in built["lecture"].slides[10].shapes if sh.name == "Equation"]  # a long formula: own row
    assert len(wide) == 1


def test_numbers_in_inline_maths_use_the_body_font_and_a_number_step_is_text(job, tmp_path, monkeypatch):
    # Live, "365 × 24 × 60" and "2.71828" switched to a serif font mid-sentence, and steps "1000" to "1331" were
    # serif pictures larger than the words beside them.
    deck = slides_lecture_deck()
    deck["slides"][2]["blocks"] = [
        bullets("A year has $60 \\times 24 \\times 365$ minutes and $v_0 \\approx 14$ m/s"),
        {"type": "steps", "label": "A ball's height", "items": [
            {"latex": "120", "text": "At launch"}, {"latex": "1{,}080", "text": "After one second"},
            {"latex": r"h = \frac{v^2}{2g}", "text": "At the top"}]}]
    prs = build(job, deck, monkeypatch, tmp_path)
    runs = [r for sh in prs.slides[2].shapes for _, r in runs_of(sh)]
    by_text = {r.text: r.font for r in runs}
    assert by_text["60 × 24 × 365"].name == "Calibri"
    assert by_text["v"].name == "Cambria" and by_text["v"].italic and by_text["0"].name == "Calibri"
    assert by_text[" ≈ 14"].name == "Calibri"
    assert "120" in by_text and "1,080" in by_text and by_text["1,080"].name == "Calibri"
    assert all(alt(p) not in ("120", "1{,}080") for p in pictures(prs.slides[2]))
    assert any(alt(p) == r"h = \frac{v^2}{2g}" for p in pictures(prs.slides[2]))
    for latex, text in ((r" \$2.25 ", "$2.25"), ("50\\%", "50%"), ("1{,}000", "1,000"), ("-3.5", "−3.5"),
                        (r"1000 \to 1100", "1000 → 1100"), (r"225\times 1.5=337.5", "225 × 1.5 = 337.5"),
                        ("1+1", "1 + 1"), ("7 - 2", "7 − 2")):
        assert pptx_deck.plain_math(latex) == text, latex
    assert all(pptx_deck.plain_math(x) == "" for x in ("x", "2x", "10^{3}", "A=2", r"\frac{1}{2}", "", "+"))


def test_a_tall_narrow_table_sits_beside_the_text_tops_aligned(job, tmp_path, monkeypatch):
    deck = slides_lecture_deck()
    rows = [[f"{a}°", f"{r}"] for a, r in ((15, 20.4), (30, 35.3), (45, 40.8), (60, 35.3), (75, 20.4))]
    table = {"type": "table", "header": ["Angle", "Range (m)"], "rows": rows, "caption": "Range by angle"}
    deck["slides"][2]["blocks"] = [table, bullets("Complementary angles give the same range", "45° goes furthest")]
    prs = build(job, deck, monkeypatch, tmp_path)
    s = prs.slides[2]
    gf = next(sh for sh in s.shapes if getattr(sh, "has_table", False) and sh.has_table)
    text = next(sh for sh in s.shapes if sh.has_text_frame and "Complementary" in sh.text_frame.text)
    assert gf.left + gf.width < text.left and gf.top == text.top  # beside it, and both start at one height
    col = (pptx_deck.CW - pptx_deck.COL_GAP) * 0.5
    assert gf.width >= pptx_deck.E(col * pptx_deck.TABLE_MIN_SHARE) - 2
    # The pair sits in the middle of the body.
    top, bottom = gf.top / EMU_PER_PT, max(gf.top + gf.height, text.top + text.height) / EMU_PER_PT
    cap = next(sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text == "Range by angle")
    bottom = max(bottom, (cap.top + cap.height) / EMU_PER_PT)
    assert abs((top - pptx_deck.BODY_Y) - (pptx_deck.BODY_B - bottom)) < 6
    # A short or wide table is stacked with the text instead.
    assert pptx_deck.side_table([table, bullets("x")]) is table
    assert pptx_deck.side_table([{**table, "rows": rows[:3]}, bullets("x")]) is None
    assert pptx_deck.side_table([{**table, "header": ["a", "b", "c", "d"]}, bullets("x")]) is None
    assert pptx_deck.side_table([table]) is None
    assert pptx_deck.side_table([table, bullets("x"), {"type": "figure", "image": "export/fig_001.png"}]) is None


def test_a_title_never_leaves_one_short_word_on_its_last_line(job, tmp_path, monkeypatch):
    # Live, a cover title wrapped as "How compounding interest every minute leads to" / "e".
    b = pptx_deck.Builder(job, {"slides": []})
    title = "How a launched ball's range depends on its angle of x"
    p = b.fit_title(title, 816, 140, (44, 42, 40), (40, 38, 36, 34, 32, 30), line=1.05)
    lines, last = pptx_deck._wrap(p, 816)
    assert lines == 1 or last >= pptx_deck.WIDOW * 816
    loose = pptx_deck.Para(pptx_deck.rich(title), 40, font=pptx_deck.TITLE_FONT, line=1.05)
    lines, last = pptx_deck._wrap(loose, 816)
    if lines == 2 and last < pptx_deck.WIDOW * 816:  # with Calibri Light (Windows): "x" alone at 40 pt
        assert p.size < 40


def test_takeaway_rows_are_centred_on_their_numbers(built):
    s = built["lecture"].slides[12]
    dots = sorted((sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.isdigit()
                   and sh.name != "Slide number"), key=lambda sh: sh.top)
    texts = sorted((sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text[:5] in
                    ("Split", "The f", "The r", "Compl")), key=lambda sh: sh.top)
    assert len(dots) == len(texts) == 4
    for dot, text in zip(dots, texts):
        assert abs((dot.top + dot.height / 2) - (text.top + text.height / 2)) < EMU_PER_PT
        assert str(text.text_frame.vertical_anchor) .startswith("MIDDLE")


def test_a_wide_formula_is_scaled_to_its_column(job, tmp_path, monkeypatch):
    deck = slides_lecture_deck()
    long = r"x = " + " + ".join(f"a_{{{k}}} b_{{{k}}}" for k in range(40))
    deck["slides"][2]["blocks"] = [eq(long, "A long sum")]
    prs = build(job, deck, monkeypatch, tmp_path)
    pic = next(p for p in pictures(prs.slides[2]) if alt(p) == long)
    assert pic.left >= 50 * EMU_PER_PT and pic.left + pic.width <= 910 * EMU_PER_PT


def test_too_much_for_one_slide_continues_on_the_next(job, tmp_path, monkeypatch):
    deck = lecture_deck()
    deck["slides"][10]["blocks"].append({"type": "steps", "label": "", "items": [
        {"latex": r"\frac{a}{b} = \frac{c}{d}", "text": "A step with a long explanation " * 3} for _ in range(6)]})
    prs = build(job, deck, monkeypatch, tmp_path)
    titles = [title_of(s) for s in prs.slides]
    k = titles.index("Everything on one slide: a crowded summary of the angle study (continued)")
    assert titles[k - 1] == "Everything on one slide: a crowded summary of the angle study"
    assert len(prs.slides) == len(deck["slides"]) + 1
    assert prs.slides[k].notes_slide.notes_text_frame.text.startswith("(continued) A lot on one slide.")
    assert all(no_overlaps(s) and all(b[4] <= 540.01 for b in boxes(s)) for s in prs.slides)


def test_a_table_of_words_too_long_for_any_size_is_kept(job, tmp_path, monkeypatch):
    # Six columns of 40-letter words cannot fit even in 10-pt type: the words break inside their cells (as PowerPoint
    # breaks them) instead of the whole table being left out.
    word = "Pneumonoultramicroscopicsilicovolcanoco"
    deck = slides_lecture_deck()
    deck["slides"][2]["blocks"] = [{"type": "table", "header": [word] * 6, "rows": [[word] * 6 for _ in range(8)],
                                    "caption": "Long words"}]
    prs = build(job, deck, monkeypatch, tmp_path)
    tables = [sh for s in prs.slides for sh in s.shapes if getattr(sh, "has_table", False) and sh.has_table]
    assert len(tables) == 1 and len(tables[0].table.rows) == 9 and len(tables[0].table.columns) == 6
    assert tables[0].table.cell(8, 5).text_frame.text == word
    s = prs.slides[2]
    assert no_overlaps(s) and all(b[1] >= 49.9 and b[3] <= 910.1 and b[4] <= 540.01 for b in boxes(s))


def test_empty_and_odd_decks_still_build(job, tmp_path, monkeypatch):
    assert len(build(job, {"slides": []}, monkeypatch, tmp_path).slides) == 1
    odd = {"title": "x\x00y", "slides": [{"kind": "content", "title": None, "blocks": [
        {"type": "nope"}, "junk", {"type": "bullets", "items": [{"text": "a\x07b $\\frac{1}{$", "level": 7}]},
        {"type": "table", "header": [], "rows": []}, {"type": "figure", "image": "export/fig_999.png"},
        {"type": "steps", "items": [{}]}, {"type": "callout", "style": "weird", "title": "T", "text": "", "latex": ""}]},
        {"kind": "agenda", "title": "Agenda", "blocks": []}, {"kind": "section", "title": "Alone"},
        {"kind": "formulas", "title": "Formula sheet", "blocks": []}]}
    prs = build(job, odd, monkeypatch, tmp_path)
    assert len(prs.slides) == 4 and title_of(prs.slides[0]) == "Untitled"


def test_many_agenda_sections_use_two_columns(job, tmp_path, monkeypatch):
    slides = [slide("s1", "agenda", "Agenda")] + [
        slide(f"c{k}", "content", f"Slide {k}", f"Section number {k} with a longer name", 10.0 * k, 10.0 * k + 9,
              [bullets("point")]) for k in range(10)]
    prs = build(job, {"title": "Many", "lecture": "L", "slides": slides}, monkeypatch, tmp_path)
    agenda = prs.slides[0]
    lefts = {round(sh.left / EMU_PER_PT) for sh in agenda.shapes if sh.has_text_frame and sh.text_frame.text.isdigit()
             and sh.name != "Slide number"}
    assert len(lefts) == 2 and no_overlaps(agenda)


def test_real_typesetting_when_node_is_there(job, tmp_path, monkeypatch):
    if texrender.shutil.which("node") is None or not (texrender.ROOT / "web" / "node_modules" / "mathjax-full").is_dir():
        pytest.skip("node or the MathJax package is not installed")
    monkeypatch.setattr(texrender, "CACHE_DIR", tmp_path / "tex")
    deck = slides_lecture_deck()
    deck["slides"][2]["blocks"].append(eq(r"R = \frac{v_0^2 \sin 2\theta}{g}", "Range"))
    prs = open_pptx(pptx_deck.build_pptx(job, deck))
    pic = next(p for p in pictures(prs.slides[2]) if alt(p) == r"R = \frac{v_0^2 \sin 2\theta}{g}")
    with Image.open(io.BytesIO(pic.image.blob)) as im:
        assert im.mode == "RGBA" and im.getpixel((0, 0))[3] == 0  # transparent
        assert abs(pic.width / pic.height - im.width / im.height) < 0.01 * im.width / im.height
    assert len(list((tmp_path / "tex").glob("*.png"))) == 1
