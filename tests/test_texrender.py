"""Equations typeset as pictures for the PowerPoint (app/core/texrender.py): cache, errors, fallbacks."""
from __future__ import annotations

import base64
import io
import logging

import pytest
from PIL import Image

from app.core import texrender


def png_bytes(w=40, h=20) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


class FakeNode:
    """Stands in for the node run: records each batch, answers per formula."""

    def __init__(self, bad=(), broken_png=()):
        self.batches: list[list[str]] = []
        self.bad, self.broken_png = set(bad), set(broken_png)

    def __call__(self, payload: dict, timeout: float) -> dict:
        assert timeout > 0
        items = payload["items"]
        self.batches.append([it["latex"] for it in items])
        out = []
        for it in items:
            assert it["px_per_em"] == texrender.PX_PER_EM and it["color"].startswith("#")
            if it["latex"] in self.bad:
                out.append({"id": it["id"], "ok": False, "error": "Undefined control sequence \\foo"})
            elif it["latex"] in self.broken_png:
                out.append({"id": it["id"], "ok": True, "png": "not base64!", "width_em": 1, "height_em": 1,
                            "depth_em": 0, "px": [1, 1]})
            else:
                out.append({"id": it["id"], "ok": True, "png": base64.b64encode(png_bytes()).decode(),
                            "width_em": 0.25 * len(it["latex"]), "height_em": 1.2, "depth_em": 0.3, "px": [40, 20]})
        return {"results": out}


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch):
    texrender._warned.clear()
    monkeypatch.setattr(texrender, "_last_run_failed", False)
    yield
    texrender._warned.clear()


def test_one_batch_then_the_disk_cache(tmp_path):
    node = FakeNode()
    got = texrender.render(["x^2", "y = mx + c", "x^2"], cache_dir=tmp_path, runner=node)
    assert node.batches == [["x^2", "y = mx + c"]]  # one run for the batch, duplicates once
    img = got["x^2"]
    assert img is not None and img.path.is_file() and img.path.parent == tmp_path
    assert (img.width_em, img.height_em, img.depth_em, img.px) == (0.75, 1.2, 0.3, (40, 20))
    assert img.size_pt(20) == (15.0, 24.0)
    again = texrender.render(["x^2", " y = mx + c "], cache_dir=tmp_path, runner=node)
    assert len(node.batches) == 1  # all from the cache (a formula is trimmed before its key is made)
    assert again["x^2"] == img and again[" y = mx + c "].path == got["y = mx + c"].path
    assert len(list(tmp_path.glob("*.png"))) == 2 and len(list(tmp_path.glob("*.json"))) == 2


def test_colour_and_display_style_are_part_of_the_key(tmp_path):
    node = FakeNode()
    a = texrender.render(["x"], cache_dir=tmp_path, runner=node)["x"]
    b = texrender.render(["x"], cache_dir=tmp_path, runner=node, color="#FFFFFF")["x"]
    c = texrender.render(["x"], cache_dir=tmp_path, runner=node, display=False)["x"]
    assert len({a.path, b.path, c.path}) == 3 and len(node.batches) == 3


def test_a_formula_that_does_not_typeset_is_none_and_cached(tmp_path, caplog):
    node = FakeNode(bad={r"\foo{x}"})
    with caplog.at_level(logging.WARNING, logger=texrender.__name__):
        got = texrender.render([r"\foo{x}", "x"], cache_dir=tmp_path, runner=node)
        assert got[r"\foo{x}"] is None and got["x"] is not None
        got = texrender.render([r"\foo{x}"], cache_dir=tmp_path, runner=node)
    assert got[r"\foo{x}"] is None and len(node.batches) == 1  # the failure is remembered
    lines = [r.getMessage() for r in caplog.records]
    assert lines == ["Some equations could not be typeset; the PowerPoint shows them as plain text."]


def test_a_broken_picture_is_not_cached(tmp_path):
    node = FakeNode(broken_png={"z"})
    assert texrender.render(["z"], cache_dir=tmp_path, runner=node)["z"] is None
    assert texrender.render(["z"], cache_dir=tmp_path, runner=node)["z"] is None
    assert len(node.batches) == 2 and not list(tmp_path.glob("*"))


def test_no_node_means_text_and_one_plain_line(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(texrender.shutil, "which", lambda name: None)
    with caplog.at_level(logging.WARNING, logger=texrender.__name__):
        for _ in range(3):
            got = texrender.render(["x^2", "y"], cache_dir=tmp_path)
            assert got == {"x^2": None, "y": None}
    assert [r.getMessage() for r in caplog.records] == [
        "Equations are shown as plain text in the PowerPoint: node was not found."]
    assert not list(tmp_path.glob("*"))  # nothing cached: installing node later works at once


def test_runner_errors_never_raise(tmp_path, monkeypatch):
    def crash(payload, timeout):
        raise RuntimeError("boom")

    def missing(payload, timeout):
        raise texrender.TexRenderError("a package for typesetting equations is missing")
    assert texrender.render(["a"], cache_dir=tmp_path, runner=crash) == {"a": None}
    assert texrender.render(["a"], cache_dir=tmp_path, runner=missing) == {"a": None}
    assert texrender.render(["a"], cache_dir=tmp_path, runner=lambda p, t: {"results": "nope"}) == {"a": None}
    assert texrender.render(["", "   ", "x" * 5000], cache_dir=tmp_path, runner=crash) == {
        "": None, "   ": None, "x" * 5000: None}


def test_node_runner_reports_a_failed_script_in_one_line(monkeypatch, tmp_path):
    class Done:
        returncode, stdout, stderr = 3, b"", b"a package for typesetting equations is missing (run npm install)\nat ..."
    monkeypatch.setattr(texrender.shutil, "which", lambda name: "node")
    monkeypatch.setattr(texrender.subprocess, "run", lambda *a, **k: Done())
    with pytest.raises(texrender.TexRenderError, match=r"^a package for typesetting equations is missing \(run"):
        texrender.node_runner({"items": []}, 5)
    Done.returncode, Done.stdout = 0, b"not json"
    with pytest.raises(texrender.TexRenderError, match="not JSON"):
        texrender.node_runner({"items": []}, 5)
    monkeypatch.setattr(texrender, "SCRIPT", tmp_path / "missing.mjs")
    with pytest.raises(texrender.TexRenderError, match="tex2png.mjs is missing"):
        texrender.node_runner({"items": []}, 5)


def test_available_and_the_pptx_rules_version(monkeypatch):
    from app import pptx_deck
    monkeypatch.setattr(texrender.shutil, "which", lambda name: None)
    assert texrender.available() is False
    assert pptx_deck.rules_version() == pptx_deck.RULES_VERSION + "-text"


def test_a_failed_run_marks_the_rules_version_until_a_run_works(tmp_path, monkeypatch):
    # node is installed but a run fails (a timeout on a busy machine): that PowerPoint has its formulas as text, so
    # the version it is cached under must not claim typeset formulas, or the download keeps the text for good.
    from app import pptx_deck
    monkeypatch.setattr(texrender, "available", lambda: True)
    assert pptx_deck.rules_version() == pptx_deck.RULES_VERSION

    def timeout(payload, t):
        raise texrender.TexRenderError("typesetting took longer than 31 s")
    assert texrender.render(["a"], cache_dir=tmp_path, runner=timeout) == {"a": None}
    assert texrender.last_run_failed() and pptx_deck.rules_version() == pptx_deck.RULES_VERSION + "-text"
    assert texrender.render(["a"], cache_dir=tmp_path, runner=FakeNode())["a"] is not None
    assert not texrender.last_run_failed() and pptx_deck.rules_version() == pptx_deck.RULES_VERSION


def test_a_cache_that_cannot_be_written_never_raises(tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    got = texrender.render(["x^2", r"\foo{x}"], cache_dir=blocker / "tex", runner=FakeNode(bad={r"\foo{x}"}))
    assert got == {"x^2": None, r"\foo{x}": None}  # shown as text, and the PowerPoint is still made


def node_ready() -> bool:
    return texrender.shutil.which("node") is not None and (
        texrender.ROOT / "web" / "node_modules" / "mathjax-full").is_dir() and (
        texrender.ROOT / "web" / "node_modules" / "@resvg" / "resvg-js").is_dir()


@pytest.mark.skipif(not node_ready(), reason="node, MathJax or resvg is not installed")
def test_real_node_typesets_sharp_transparent_pictures(tmp_path):
    got = texrender.render([r"R = \frac{v_0^2 \sin 2\theta}{g}", r"x^2", r"\undefinedmacro{x}", r"x^{2"],
                           cache_dir=tmp_path, color="#2563EB")
    frac, x2 = got[r"R = \frac{v_0^2 \sin 2\theta}{g}"], got["x^2"]
    assert got[r"\undefinedmacro{x}"] is None and got["x^{2"] is None
    assert frac.height_em > 2 * x2.height_em * 0.8 and frac.width_em > x2.width_em  # a display fraction is tall
    with Image.open(frac.path) as im:
        assert im.mode == "RGBA" and im.size == frac.px
        assert abs(im.width / texrender.PX_PER_EM - frac.width_em) < 0.02
        assert im.getpixel((0, 0))[3] == 0  # transparent margin
        inked = [px for px in im.getdata() if px[3] > 200]
        assert inked and all(abs(px[0] - 0x25) < 40 and abs(px[2] - 0xEB) < 40 for px in inked)  # the colour asked


@pytest.mark.skipif(not node_ready(), reason="node, MathJax or resvg is not installed")
def test_real_node_fails_a_huge_formula_alone_and_draws_latin_text_quickly():
    items = [{"id": k, "latex": latex, "display": True, "color": "#1F2A37", "px_per_em": texrender.PX_PER_EM}
             for k, latex in (("huge", r"x\hspace{3000em}y"), ("tall", r"\rule{1em}{2000em}"), ("ok", "x^2"),
                              ("angstrom", r"a = 5.6\,\text{Å}"))]
    # A picture of 360,000 px once made resvg ask for 230 GB: node died and every formula of the deck was text.
    got = {r["id"]: r for r in texrender.node_runner({"items": items}, 60)["results"]}
    assert not got["huge"]["ok"] and "too big" in got["huge"]["error"] and not got["tall"]["ok"]
    assert got["ok"]["ok"] and got["ok"]["fonts"] == "none" and got["angstrom"]["ok"]
    with Image.open(io.BytesIO(base64.b64decode(got["angstrom"]["png"]))) as im:
        assert sum(1 for px in im.getdata() if px[3] > 128) > 500
    if (texrender.Path(texrender.os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "cambria.ttc").is_file():
        assert got["angstrom"]["fonts"] == "quick"  # not every system font (about a second per formula)
