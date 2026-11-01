"""3D pop-out scenes: only safe, whitelisted maths reaches the player."""
from app.popout import clean_scene, safe_expr


def test_safe_expr_allows_maths_only():
    for ok in ["x^2 + y^2", "ln(abs(x))", "sin(t)*cos(t)", "2.5e-3*x", "exp(-x^2)", "sqrt(1 - x^2)", "pi*x"]:
        assert safe_expr(ok), ok
    for bad in ["", "__import__('os')", "alert(x)", "x; fetch(1)", "window.x", "x" * 300, "x[0]", "`x`"]:
        assert not safe_expr(bad), bad


def test_clean_scene_drops_unsafe_parts_and_fixes_ranges():
    scene = {
        "kind": "plot2d", "title": "t" * 200,
        "x": {"min": 5, "max": -5, "label": "x"}, "y": {"min": float("nan"), "max": 1},
        "functions": [{"expr": "ln(x)", "color": "#ff0000"}, {"expr": "eval(x)", "color": "#00ff00"}],
        "curves": [{"x": "t", "y": "t", "z": "process(t)"}],
        "solids": [{"shape": "box", "center": [0, 0], "size": [1]}],
        "points": [{"at": [1, 0, 0], "label": "(1,0)"}],
    }
    out = clean_scene(scene)
    assert [f["expr"] for f in out["functions"]] == ["ln(x)"]
    assert out["curves"] == [] and out["solids"] == [] and len(out["points"]) == 1
    assert out["x"]["min"] < out["x"]["max"] and out["y"]["min"] < out["y"]["max"]
    assert len(out["title"]) <= 80


def test_scene_without_drawable_content_is_none():
    assert clean_scene({"kind": "surface", "functions": [{"expr": "rm -rf"}]})["kind"] == "none"
