"""3D views (app/popout.py, format 2) and their expressions (app/core/scene_expr.py), offline: only safe,
whitelisted maths reaches the player; steps sit on the video's clock; the geometry is checked through every state;
scenes are cached per lecture version, and two requests make one model call. The model is a fake, and every
fixture is made up (pieces are drawn by code)."""
import json
import math
import threading
import time
import warnings

import numpy as np
import pytest

with warnings.catch_warnings():  # starlette's notice about httpx, at import time
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app import popout, server
from app.core import scene_expr as sx
from app.core.media import imwrite
from app.core.meta_api import CallRecord, MetaAPIError

CLIP = (10.0, 130.0)


# ---------- the expressions ----------
@pytest.mark.parametrize("src, names", [
    ("x^2 + 1", {"x"}), ("ln(abs(x))", {"x"}), ("ln x", {"x"}), ("2x", {"x"}), ("exp(-x^2)", {"x"}),
    ("x^2 + y^2", {"x", "y"}), ("sin(x)*cos(y)", {"x", "y"}),
    ("cos(t)", {"t"}), ("a*cos(beta*deg)", {"a", "beta"}), ("u*cos(v)", {"u", "v"}),
    ("pi*e", set()), ("1.5", set()), ("2.5e-3", set()), ("max(1, 2, 3)", set()), ("pow(2, 3)", set()),
    ("log10(100) + log2(8)", set()), ("x**2", {"x"}), ("Beta", {"beta"}), ("c_1 + 1", {"c_1"}),
])
def test_allowed_expressions(src, names):
    sx.parse(src, names)


@pytest.mark.parametrize("src, names", [
    ("", set()), ("__import__('os')", {"x"}), ("alert(x)", {"x"}), ("x; fetch(1)", {"x"}), ("window.x", {"x"}),
    ("x" * 600, {"x"}), ("x[0]", {"x"}), ("`x`", {"x"}), ("constructor", {"x"}),
    ("x + 1", set()),           # a variable the field does not have
    ("y", {"x"}), ("t", {"u", "v"}), ("x2", {"x"}),  # "x2" is one name, not x times 2
    ("pow(1)", set()), ("sin(1, 2)", set()), ("(1 + 2", set()), ("1 +", set()), ("1e999", set()),
    ("(" * 70 + "1" + ")" * 70, set()),
])
def test_refused_expressions(src, names):
    with pytest.raises(sx.ExprError):
        sx.parse(src, names)


def test_each_field_allows_only_its_own_variables():
    params = {"k"}

    def ok(obj):
        return popout.clean_object({"id": "o", "color": "#000000", **obj}, 0, params, []) is not None

    assert ok({"type": "graph", "expr": "k*x^2"})
    assert not ok({"type": "graph", "expr": "x*y"})
    assert ok({"type": "surface", "expr": "x*y"})
    assert not ok({"type": "surface", "expr": "t"})
    assert ok({"type": "curve", "x": "cos(t)", "y": "sin(t)", "z": "k*t", "range": ["0", "2*pi"]})
    assert not ok({"type": "curve", "x": "cos(x)", "y": "0", "z": "0", "range": ["0", "1"]})
    assert not ok({"type": "curve", "x": "t", "y": "0", "z": "0", "range": ["0", "t"]})  # ranges: params only
    assert ok({"type": "psurface", "x": "u", "y": "v", "z": "u*v", "range": ["0", "1"], "range2": ["0", "k"]})
    assert not ok({"type": "psurface", "x": "u", "y": "t", "z": "0", "range": ["0", "1"], "range2": ["0", "1"]})
    assert ok({"type": "segment", "points": [["0", "0", "0"], ["k", "1", "2"]]})
    assert not ok({"type": "segment", "points": [["0", "0", "0"], ["x", "1", "2"]]})
    assert not ok({"type": "sphere", "center": ["t", "0", "0"], "size": ["1"]})
    assert ok({"type": "box", "center": [0, 0, 0], "size": ["1", "2", "k"], "rotation": ["0", "0", "45"]})


@pytest.mark.parametrize("src", ["2x(x-1)", "-x^2", "2^-x", "2^3^2", "ln x^2", "a - (b - c)", "a/(b*c)",
                                 "-(a+b)*c", "--a", "a*-b", "sin x + cos(2 x)", "(x+1)(x-1)", "1e-5*x", "-2^0.5"])
def test_the_canonical_form_means_the_same(src):
    names = {"x", "a", "b", "c"}
    text = sx.canonical(src, names)
    env = {"x": np.linspace(0.3, 2.2, 7), "a": 1.7, "b": -0.6, "c": 2.9}
    want = sx.evaluate(sx.parse(src, names), env)
    got = sx.evaluate(sx.parse(text, names), env)
    np.testing.assert_allclose(np.broadcast_to(got, (7,)), np.broadcast_to(want, (7,)), rtol=1e-12)
    assert sx.canonical(text, names) == text  # stable


def test_evaluation_follows_the_player():
    ev = lambda s, **env: float(sx.evaluate(sx.parse(s, set(env)), env))  # noqa: E731
    assert ev("2^3^2") == 512 and ev("-x^2", x=3) == -9 and ev("2x", x=4) == 8
    assert ev("cos(beta*deg)", beta=60) == pytest.approx(0.5)
    assert ev("log(e)") == pytest.approx(1) and ev("ln x", x=math.e) == pytest.approx(1)
    assert math.isnan(ev("sqrt(-1)")) and math.isnan(ev("ln(-2)")) and math.isnan(ev("(-8)^(1/3)"))
    assert ev("1/0") == math.inf and ev("min(3, 1, 2)") == 1 and ev("sign(-2)") == -1


def test_powers_follow_the_player():
    # The player uses JavaScript's Math.pow: NaN for a NaN exponent and for (+-1)^(+-inf), where numpy gives 1;
    # and (-inf)^0.5 is inf, which numpy gives for arrays but not for single numbers.
    ev = lambda s, **env: np.asarray(sx.evaluate(sx.parse(s, set(env)), env), dtype=float)  # noqa: E731
    assert math.isnan(ev("1^sqrt(-1)")) and math.isnan(ev("pow(1, 1/0)")) and math.isnan(ev("(-1)^(-1/0)"))
    assert ev("(-1/0)^0.5") == math.inf and ev("sqrt(-1)^0") == 1 and ev("0^0") == 1 and ev("(-2)^3") == -8
    np.testing.assert_array_equal(ev("x^(1/0)", x=np.array([1.0, -1.0, 2.0, 0.5])), [np.nan, np.nan, np.inf, 0])
    assert ev("x^2", x=np.ones((3, 4))).shape == (3, 4)


def test_numbers_too_big_for_a_float_are_refused_not_a_crash():
    big = 10 ** 400  # JSON can carry it; float() cannot
    with pytest.raises(sx.ExprError):
        sx.parse(big)
    k = {"name": "k", "value": 0, "min": -big, "max": big, "label": "", "unit": ""}
    raw = answer([obj("p", "point", points=[[big, "0", "0"]]),
                  obj("q", "point", points=[["1", "0", "0"]], opacity=big)],
                 [step("0:20", highlight=["q"], set=[{"param": "k", "to": big}], over=big)], [k])
    raw["axes"]["x"] = {"min": -big, "max": big, "label": ""}
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    assert [o["id"] for o in scene["objects"]] == ["q"] and scene["params"] == []
    assert scene["objects"][0]["opacity"] == 1 and scene["steps"][0]["over"] == popout.OVER_DEFAULT
    assert scene["axes"]["x"]["min"] == -5
    json.dumps(scene, allow_nan=False)


def test_ranges_far_out_stay_in_order():
    # Clamped to +-LIMIT one end at a time, a range wholly beyond it came out upside down; the player then threw
    # the axis away (-5..5) while the server had framed and sampled the upside-down one.
    axes = popout.clean_axes({"show": True, "x": {"min": 2e4, "max": 3e4, "label": ""},
                              "y": {"min": -3e4, "max": -2e4, "label": ""},
                              "z": {"min": 0, "max": 1, "label": ""}}, [])
    assert all(axes[k]["min"] < axes[k]["max"] for k in "xyz") and axes["z"] == {"min": 0, "max": 1, "label": ""}
    (p,) = popout.clean_params([{"name": "m", "value": 25000, "min": 20000, "max": 30000, "label": "", "unit": ""}],
                               [])
    assert p["min"] < p["max"] and p["min"] <= p["value"] <= p["max"]


# ---------- the steps on the video's clock ----------
def step(at="", **kw):
    return {"at": at, "over": kw.pop("over", 1.0), "caption": kw.pop("caption", "c"), "show": [], "hide": [],
            "highlight": [], "set": [], "camera": "keep", **kw}


def test_times_are_parsed_sorted_spaced_and_kept_in_the_clip():
    notes = []
    steps, synced = popout.clean_steps(
        [step("0:40", caption="b"), step("[0:30]", caption="a"), step("0:40", caption="c", over=8),
         step("0:05", caption="early"), step("9:00", caption="late"), step("0:41", caption="d")],
        [], [], CLIP, notes)
    assert synced
    ts = [s["t"] for s in steps]
    assert [s["caption"] for s in steps] == ["early", "a", "b", "c", "d", "late"]
    assert ts[0] == CLIP[0] and ts[-1] == pytest.approx(CLIP[1] - popout.OVER_MIN)
    assert all(b - a >= 1 - 1e-9 for a, b in zip(ts, ts[1:]))
    assert ts[1:5] == [30, 40, 41, 42]
    assert [s["at"] for s in steps][:3] == ["0:10", "0:30", "0:40"]
    # a change is over before the next step starts
    assert all(s["over"] <= n["t"] - s["t"] + 1e-9 for s, n in zip(steps, steps[1:]))
    assert steps[3]["over"] == 1.0  # 8 s asked, the next step 1 s later


def test_steps_crowded_at_the_end_of_the_clip_move_back():
    steps, _ = popout.clean_steps([step("2:09"), step("2:10"), step("2:10")], [], [], CLIP, [])
    assert [s["t"] for s in steps] == pytest.approx([127.4, 128.4, 129.4])


def test_every_change_is_over_by_the_end_of_the_clip():
    # Playback stops at the end of the analysed part (the player pauses there), so a change still running then
    # would stay half done: the last one starts early enough and is short enough to finish.
    for end_steps in ([step("2:05", over=3), step("2:09", over=3)], [step("2:10", over=8)], [step("9:00", over=2)]):
        steps, synced = popout.clean_steps(end_steps, [], [], CLIP, [])
        last = steps[-1]
        assert synced and last["t"] + last["over"] <= CLIP[1] + 1e-9 and last["over"] >= popout.OVER_MIN
        assert all(s["t"] + s["over"] <= n["t"] + 1e-9 for s, n in zip(steps, steps[1:]))


def test_untimed_steps_follow_the_step_before_them():
    steps, synced = popout.clean_steps([step("0:20", caption="a"), step("", caption="b"), step("0:50", caption="c")],
                                       [], [], CLIP, [])
    assert synced and [(s["caption"], s["t"]) for s in steps] == [("a", 20), ("b", 21), ("c", 50)]


def test_a_drawing_never_talked_about_has_no_times():
    steps, synced = popout.clean_steps([step(""), step("", caption="b"), step("1:00", caption="c")], [], [], CLIP, [])
    assert not synced and all(s["t"] is None and s["at"] == "" for s in steps) and len(steps) == 3


def test_step_references_are_repaired():
    params = popout.clean_params([{"name": "theta", "value": 0, "min": 0, "max": 90, "label": "", "unit": "deg"}], [])
    notes = []
    steps, _ = popout.clean_steps([
        step("0:20", show=["A", "ghost", "a"], hide=["a", "b"], highlight=["zz"],
             set=[{"param": "theta", "to": 400}, {"param": "phi", "to": 1}], over=99, camera="sideways"),
        step("0:25", caption="", show=["ghost"]),  # nothing left: dropped
    ], ["a", "b"], params, CLIP, notes)
    assert len(steps) == 1
    s = steps[0]
    assert s["show"] == ["a"] and s["hide"] == ["b"] and s["highlight"] == []
    # rules 6: a value past the param's range is kept (the range is widened to hold it: fit_range), not cut to it
    assert s["set"] == [{"param": "theta", "to": 400}] and s["camera"] == "keep" and s["over"] == 8
    assert any("ghost" in n for n in notes)
    steps, _ = popout.clean_steps([step("0:20", set=[{"param": "theta", "to": 1e9}])], [], params, CLIP, [])
    assert steps[0]["set"] == [{"param": "theta", "to": popout.LIMIT}]


def test_params_are_checked():
    notes = []
    ps = popout.clean_params([
        {"name": "Beta", "value": 200, "min": 120, "max": 60, "label": "β", "unit": "deg"},
        {"name": "x", "value": 1, "min": 0, "max": 2, "label": "", "unit": ""},       # a variable's name
        {"name": "sin", "value": 1, "min": 0, "max": 2, "label": "", "unit": ""},     # a function's name
        {"name": "beta", "value": 1, "min": 0, "max": 2, "label": "", "unit": ""},    # repeated
        {"name": "k", "value": "1", "min": 0, "max": 2, "label": "", "unit": "cm"},  # not a number
        {"name": "flat", "value": 3, "min": 3, "max": 3, "label": "", "unit": "kg"},
    ], notes)
    assert [p["name"] for p in ps] == ["beta", "flat"]
    # rules 6: the range is what a student explores, so it is widened to hold the value instead of moving the value
    assert ps[0] == {"name": "beta", "value": 200, "min": 60, "max": 200, "label": "β", "unit": "deg"}
    assert ps[1]["min"] < ps[1]["value"] < ps[1]["max"] and ps[1]["unit"] == ""
    assert len(popout.clean_params([{"name": f"p{i}", "value": 0, "min": 0, "max": 1, "label": "", "unit": ""}
                                    for i in range(12)], [])) == popout.MAX_PARAMS


# ---------- whole scenes ----------
def obj(oid, typ, **kw):
    base = {"id": oid, "type": typ, "points": [], "center": [], "size": [], "rotation": [], "expr": "", "x": "",
            "y": "", "z": "", "range": [], "range2": [], "color": "#336699", "opacity": 1, "style": "solid",
            "label": ""}
    return {**base, **kw}


def answer(objects, steps, params=(), show_axes=False, **kw):
    ax = {"min": -2, "max": 2, "label": ""}
    return {"title": kw.get("title", "A turning rod"), "explain": kw.get("explain", "It turns."),
            "axes": {"show": show_axes, "x": ax, "y": ax, "z": ax}, "params": list(params), "objects": objects,
            "steps": steps}


THETA = {"name": "theta", "value": 0, "min": 0, "max": 180, "label": "angle", "unit": "deg"}


def test_bounds_cover_the_whole_animation():
    # A rod from the origin to (cos theta, sin theta, 0): at the start and at the end y = 0, but half-way it
    # points straight up (y = 1), so the bounds must hold that state too.
    rod = obj("rod", "segment", points=[["0", "0", "0"], ["cos(theta*deg)", "sin(theta*deg)", "0"]])
    scene = popout.clean_scene(answer([rod], [step("0:20", set=[{"param": "theta", "to": 180}], over=4)], [THETA]),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    (x0, y0, z0), (x1, y1, z1) = scene["bounds"]
    assert x0 == pytest.approx(-1) and x1 == pytest.approx(1) and y1 == pytest.approx(1, abs=0.01)
    assert y0 == pytest.approx(0) and z0 == z1 == 0


def test_bounds_include_shown_axes_and_turned_solids():
    box = obj("b", "box", center=["0", "0", "0"], size=["2", "2", "2"], rotation=["0", "0", "45"])
    scene = popout.clean_scene(answer([box], [step("0:20", highlight=["b"])]), piece=1, pieces=[1], version="v",
                               clip=CLIP)
    assert scene["bounds"][1][0] == pytest.approx(math.sqrt(2), abs=1e-4)  # the corner turned onto the x axis
    scene = popout.clean_scene(answer([box], [step("0:20", highlight=["b"])], show_axes=True), piece=1,
                               pieces=[1], version="v", clip=CLIP)
    assert scene["bounds"] == [[-2, -2, -2], [2, 2, 2]]


def test_a_graph_is_framed_by_its_value_axis():
    # ln x dives to -inf at 0: the frame stops at the y axis' range instead of the samples' depth
    g = obj("g", "graph", expr="ln(x)", range=["0", "2"])
    scene = popout.clean_scene(answer([g], [step("", highlight=["g"])]), piece=1, pieces=[1], version="v", clip=CLIP)
    assert scene["objects"] and scene["bounds"][0][1] >= -2.2 and scene["bounds"][1][0] == pytest.approx(2)


def test_objects_that_cannot_be_drawn_are_dropped():
    notes = []
    objects = [
        obj("ok", "point", points=[["1", "0", "0"]]),
        obj("nan", "point", points=[["sqrt(-1)", "0", "0"]]),
        obj("huge", "point", points=[["1e9", "0", "0"]]),
        obj("negative", "sphere", center=["0", "0", "0"], size=["1 - theta/90"]),  # radius < 0 past 90 degrees
        obj("never", "graph", expr="sqrt(-1 - x^2)"),
        obj("code", "graph", expr="process.exit()"),
        obj("few", "segment", points=[["0", "0", "0"]]),
        obj("toomany", "polygon", points=[["0", "0", "0"]] * 13),
        obj("bad type", "teapot"),
        obj("nolabel", "label", points=[["0", "0", "0"]], label=""),
        obj("ok", "point", points=[["2", "0", "0"]]),  # a repeated id
    ]
    scene = popout.clean_scene(
        answer(objects, [step("0:20", show=["negative", "nan", "ok"], set=[{"param": "theta", "to": 180}])],
               [THETA]), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert [o["id"] for o in scene["objects"]] == ["ok"]
    assert scene["objects"][0]["points"] == [["1", "0", "0"]]
    # steps no longer name the dropped objects (and "ok", the only one left, is there from the start: it glows)
    (s,) = scene["steps"]
    assert s["show"] == [] and s["highlight"] == ["ok"] and s["hide"] == []
    assert len(notes) >= 10


def test_objects_are_flat_with_every_field_and_safe_values():
    o = obj("Edge-A", "segment", points=[["0", "0", "0"], ["2x", "1", "1"]], color="red", opacity=7, style="glow",
            label="a" * 50)
    scene = popout.clean_scene(answer([o], [step("0:20", show=["edge_a"])]), piece=1, pieces=[1], version="v",
                               clip=CLIP)
    assert scene["objects"] == []  # "2x": x is not allowed in a point
    o["points"][1][0] = "2*1"
    scene = popout.clean_scene(answer([o], [step("0:20", show=["edge_a"])]), piece=1, pieces=[1], version="v",
                               clip=CLIP)
    (out,) = scene["objects"]
    assert set(out) == {"id", "type", "points", "center", "size", "rotation", "expr", "x", "y", "z", "range",
                        "range2", "color", "opacity", "style", "label"}
    assert out["id"] == "edge_a" and out["color"].startswith("#") and len(out["color"]) == 7
    assert out["opacity"] == 1 and out["style"] == "solid" and len(out["label"]) <= popout.OBJECT_LABEL_CHARS
    assert scene["steps"][0]["show"] + scene["steps"][0]["highlight"] == ["edge_a"]  # the step's id, normalised
    assert set(scene) == {"format", "version", "piece", "pieces", "title", "explain", "synced", "axes", "params",
                          "defs", "objects", "steps", "bounds"}
    json.dumps(scene, allow_nan=False)  # no NaN or Infinity reaches the player


def test_with_no_usable_step_one_step_points_at_everything():
    # The drawing is there from the start (a step that showed it would leave the view empty until then, maybe
    # minutes); the one step makes it glow when the teacher first talks about it.
    objects = [obj("a", "point", points=[["0", "0", "0"]]), obj("b", "point", points=[["1", "1", "1"]])]
    for talked, at in ((55.4, "0:55"), (None, ""), (500.0, "2:09")):
        scene = popout.clean_scene(answer(objects, [step("", caption="", show=["ghost"])]), piece=1, pieces=[1],
                                   version="v", clip=CLIP, talked=talked)
        (s,) = scene["steps"]
        assert s["show"] == [] and s["highlight"] == ["a", "b"]
        assert s["at"] == at and scene["synced"] == (talked is not None)
        assert s["caption"] == "It turns."


def test_the_view_is_never_empty_before_the_first_step():
    objects = [obj("a", "point", points=[["0", "0", "0"]]), obj("b", "point", points=[["1", "0", "0"]]),
               obj("c", "point", points=[["2", "0", "0"]])]
    steps = [step("0:30", caption="a and b", show=["a", "b"]), step("0:40", hide=["b"]),
             step("0:50", show=["b", "c"])]
    scene = popout.clean_scene(answer(objects, steps), piece=1, pieces=[1], version="v", clip=CLIP)
    s0, _, s2 = scene["steps"]
    # a is there from the start and glows at the first step; b comes back later, so the first step still shows it
    assert s0["show"] == ["b"] and s0["highlight"] == ["a"] and s2["show"] == ["b", "c"]
    shown = {i for s in scene["steps"] for i in s["show"]}
    assert [o["id"] for o in scene["objects"] if o["id"] not in shown] == ["a"]
    # an object hidden before the step that shows it is left alone (it would vanish for good otherwise)
    scene = popout.clean_scene(answer(objects[:2], [step("0:20", hide=["a"], highlight=["b"]),
                                                    step("0:30", show=["a"])]),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    assert scene["steps"][1]["show"] == ["a"]
    # rules 6: alone, that hide changes nothing (a is not shown yet), so its step goes, and a is there from the start
    scene = popout.clean_scene(answer(objects[:1], [step("0:20", hide=["a"]), step("0:30", show=["a"])]), piece=1,
                               pieces=[1], version="v", clip=CLIP)
    assert [(s["show"], s["highlight"], s["hide"]) for s in scene["steps"]] == [([], ["a"], [])]
    # shown twice, then hidden: there from the start, glowing where it was first shown, and still hidden at the end
    # (rules 6: the second show changed nothing, a was there already, so that step goes)
    scene = popout.clean_scene(answer(objects[:1], [step("0:20", caption="", show=["a"]),
                                                    step("0:30", caption="", show=["a"]), step("0:40", hide=["a"])]),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    assert [(s["show"], s["highlight"], s["hide"]) for s in scene["steps"]] == [([], ["a"], []), ([], [], ["a"])]


def test_every_state_of_the_final_steps_is_checked():
    # 13 steps: the first only shows an object that cannot be drawn, so it goes with it, and the 13th (which
    # shrinks the ball below zero) moves up into the 12 kept. Its states must be checked too.
    k = {"name": "k", "value": 1, "min": -1, "max": 1, "label": "", "unit": ""}
    objects = [obj("bad", "point", points=[["sqrt(-1)", "0", "0"]]),
               obj("ball", "sphere", center=["0", "0", "0"], size=["k"]),
               obj("dot", "point", points=[["2", "0", "0"]])]
    steps = ([step("0:20", caption="", show=["bad"])] + [step(f"0:{21 + i}", highlight=["dot"]) for i in range(11)]
             + [step("0:40", set=[{"param": "k", "to": -1}])])
    scene = popout.clean_scene(answer(objects, steps, [k]), piece=1, pieces=[1], version="v", clip=CLIP)
    assert len(scene["steps"]) == 12 and scene["steps"][-1]["set"]
    again = json.loads(json.dumps(scene))
    popout.check_geometry(again, [])
    assert [o["id"] for o in again["objects"]] == [o["id"] for o in scene["objects"]] == ["dot"]


def test_a_drawing_with_nothing_to_show_in_3d():
    scene = popout.clean_scene({"title": "Notes", "explain": "This is writing, not a drawing.", "objects": [],
                                "steps": [step("0:20")]}, piece=3, pieces=[3], version="v", clip=CLIP)
    assert scene["objects"] == [] and scene["steps"] == [] and scene["explain"].startswith("This is writing")
    scene = popout.clean_scene("not even a dict", piece=3, pieces=[3], version="v", clip=CLIP)
    assert scene["objects"] == [] and scene["explain"] and scene["format"] == 2


def test_limits_on_counts_and_lengths():
    objects = [obj(f"p{i}", "point", points=[[str(i), "0", "0"]]) for i in range(50)]
    steps = [step(f"0:{20 + i:02d}", caption="x" * 300, highlight=[f"p{i}"]) for i in range(20)]
    scene = popout.clean_scene(answer(objects, steps, title="t" * 200, explain="e" * 500), piece=1, pieces=[1],
                               version="v", clip=CLIP)
    assert len(scene["objects"]) == popout.MAX_OBJECTS and len(scene["steps"]) == popout.MAX_STEPS
    assert len(scene["title"]) <= 80 and len(scene["explain"]) <= 300
    assert all(len(s["caption"]) <= 140 for s in scene["steps"])


# ---------- the drawing, the request, the cache and the lock ----------
BG = "#f0f0e8"


def board_replay(start=10.0, end=130.0):
    def piece(i, box, read, t=start, spoken=(), **kw):
        return {"id": i, "box": list(box), "image": f"pieces/{i:03d}.png", "t_seen": t, "t_shown": t,
                "written_after": t, "removed_at": kw.pop("removed_at", None), "pre_existing": kw.pop("pre", False),
                "read": read, "spoken": [{"start": a, "end": b} for a, b in spoken], "redraw_at": None, **kw}

    words = [{"w": f"word{i}", "s": start + i * 0.5, "e": start + i * 0.5 + 0.4} for i in range(int((end - start) * 2))]
    words[40]["w"] = "cone"
    return {
        "version": 1, "type": "board", "subject": "solids",
        "video": {"file": "lecture.mp4", "duration": 200.0, "width": 1280, "height": 720, "start": start, "end": end},
        "chapters": [{"t": start, "title": "Solids"}],
        "transcript": {"turns": [], "words": words},
        "board": {"size": [400, 200], "background": BG, "pieces": [
            piece(1, (100, 40, 80, 100), {"kind": "diagram", "text": "a cone with height h", "latex": ""},
                  spoken=[(30, 40)], pre=True),
            piece(2, (130, 60, 8, 10), {"kind": "label", "text": "h", "latex": "h"}, t=12),          # inside it
            piece(3, (184, 70, 10, 10), {"kind": "label", "text": "r", "latex": "r"}, t=14),         # touching it
            piece(4, (10, 5, 380, 20), {"kind": "heading", "text": "SOLIDS OF REVOLUTION", "latex": ""}, pre=True),
            piece(5, (300, 150, 60, 20), {"kind": "equation", "text": "V = pi r^2 h / 3", "latex": ""}, t=50),
            piece(6, (110, 100, 30, 20), {"kind": "text", "text": "2nd lesson", "latex": ""}, t=120),  # much later
            piece(7, (195, 90, 40, 20), {"kind": "equation", "text": "r = 2, h = 4", "latex": ""}, t=20),  # near
            piece(8, (250, 20, 100, 80), {"kind": "graph", "text": "graph of y = x^2", "latex": ""}, t=60,
                  spoken=[(70, 80)]),
            piece(9, (262, 26, 30, 12), {"kind": "equation", "text": "y = x^2", "latex": "y = x^2"}, t=61),  # in #8
        ]},
    }


def write_job(d, replay):
    d.mkdir(parents=True, exist_ok=True)
    (d / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    (d / "pieces").mkdir(exist_ok=True)
    for p in replay["board"]["pieces"]:
        img = np.zeros((p["box"][3], p["box"][2], 4), np.uint8)
        img[1:-1, 1:-1] = (40, 40, 200, 255)
        imwrite(d / p["image"], img)
    return d


def test_the_drawing_is_its_piece_with_its_labels_and_nothing_else():
    replay = board_replay()
    asked, anchor, pieces, nearby = popout.gather(replay, 1)
    assert [p["id"] for p in pieces] == [1, 2, 3] and [p["id"] for p in nearby] == [7]
    asked, anchor, pieces, _ = popout.gather(replay, 2)  # a label opens its drawing
    assert anchor["id"] == 1 and [p["id"] for p in pieces][:2] == [2, 1]
    with pytest.raises(popout.NoScene):
        popout.gather(replay, 99)


def test_a_sketch_drawn_in_several_pieces_is_one_drawing():
    """synth3d-vector3d once the arrowheads were captured: the z axis, the x and y axes and the vector became
    separate "arrow" pieces, and each half opened its own scene."""
    replay = board_replay()

    def p(i, box, kind, text, t):
        return {"id": i, "box": list(box), "image": f"pieces/{i:03d}.png", "t_seen": t, "t_shown": t,
                "written_after": t, "removed_at": None, "pre_existing": False,
                "read": {"kind": kind, "text": text, "latex": ""}, "spoken": [], "redraw_at": None}

    replay["board"]["pieces"] = [
        p(1, (200, 20, 20, 90), "arrow", "z-axis arrow labeled z", 20),
        p(2, (150, 90, 140, 60), "arrow", "x-axis and y-axis arrows labeled x and y", 25),
        p(3, (180, 40, 45, 60), "arrow", "red vector arrow labeled v", 35),
        p(4, (185, 95, 40, 30), "diagram", "components 3, 2, 4 of v", 45),
        p(5, (60, 100, 40, 12), "arrow", "arrow from the sum to the result", 30),  # not a figure
        p(6, (170, 80, 150, 80), "graph", "graph of y = x^2", 200),  # over it, minutes later
        p(7, (330, 20, 60, 60), "diagram", "a cube", 30),  # drawn at the same time, elsewhere
    ]
    replay["video"]["end"] = 260.0
    whole = {popout.drawing_id(replay, i) for i in (1, 2, 3, 4)}
    assert whole == {2}  # the biggest piece of the sketch
    asked, anchor, pieces, _ = popout.gather(replay, 1)
    assert anchor["id"] == 2 and [q["id"] for q in pieces][0] == 1 and {q["id"] for q in pieces} >= {1, 2, 3, 4}
    assert 6 not in {q["id"] for q in pieces} and popout.drawing_id(replay, 6) == 6
    assert popout.drawing_id(replay, 7) == 7
    assert not popout.can_popout(replay["board"]["pieces"][4])


def test_the_images_show_every_figure_piece_of_a_drawing_not_only_the_first(tmp_path):
    """synth3d-vector3d: the photo of the main piece (the x and y axes) was taken before the vector was drawn, so no
    image Muse Spark saw showed the vector. Each other figure piece's own photo goes too, in the order they were
    drawn, at most PART_PHOTOS."""
    replay = board_replay()

    def p(i, box, kind, text, t):
        return {"id": i, "box": list(box), "image": f"pieces/{i:03d}.png", "t_seen": t, "t_shown": t,
                "written_after": t, "removed_at": None, "pre_existing": False,
                "read": {"kind": kind, "text": text, "latex": ""}, "spoken": [], "redraw_at": None}

    replay["board"]["pieces"] = [
        p(1, (200, 20, 20, 90), "arrow", "z-axis arrow labeled z", 20),
        p(2, (150, 90, 140, 60), "arrow", "x-axis and y-axis arrows labeled x and y", 25),
        p(3, (180, 40, 45, 60), "arrow", "red vector arrow labeled v", 35),
        p(4, (185, 95, 40, 30), "diagram", "components 3, 2, 4 of v", 45),
        p(5, (195, 60, 8, 10), "label", "v", 36),  # a label of it: no photo of its own
    ]
    job = write_job(tmp_path / "job", replay)
    (job / "crops").mkdir()
    for i in (1, 2, 3, 4, 5):  # each photo a colour of its own, to tell them apart
        imwrite(job / "crops" / f"{i:03d}.jpg", np.full((60, 80, 3), (40 * i, 20, 200 - 30 * i), np.uint8))
    asked, anchor, pieces, _ = popout.gather(replay, 1)
    assert anchor["id"] == 2 and {q["id"] for q in pieces} >= {1, 2, 3, 4, 5}
    images = popout.drawing_images(job, replay, anchor, pieces)
    caps = [c for c, _, _ in images]
    assert caps[0].startswith("the drawing as captured from the board")
    assert caps[1].startswith("a photo of the drawing from the video")
    assert [c.split("(#")[1].split(")")[0] for c in caps[2:]] == ["1", "3", "4"]  # in the order they were drawn
    assert "red vector arrow labeled v (#3)" in caps[3]
    blue = [int(img[..., 0].mean() + 0.5) for _, img, fmt in images[1:]]
    assert blue == pytest.approx([80, 40, 120, 160], abs=6)  # #2's photo, then #1, #3, #4's
    # at most PART_PHOTOS of them
    replay["board"]["pieces"] += [p(10 + k, (150 + k, 90, 140, 60), "diagram", f"more of it {k}", 26 + k)
                                  for k in range(4)]
    for k in range(4):
        imwrite(job / "crops" / f"{10 + k:03d}.jpg", np.full((60, 80, 3), 128, np.uint8))
    asked, anchor, pieces, _ = popout.gather(replay, anchor["id"])
    assert len(popout.drawing_images(job, replay, anchor, pieces)) == 2 + popout.PART_PHOTOS
    # and the prompt names each image
    msgs, _, _ = popout.scene_messages(job, replay, anchor["id"])
    head = msgs[1]["content"][0]["text"]
    assert "Image 3: a photo of one part of the drawing" in head


GOOD = answer([obj("cone", "cone", center=["0", "0", "1"], size=["r", "2"]),
               obj("axis", "segment", points=[["0", "0", "0"], ["0", "0", "2"]], style="dashed")],
              [step("0:30", caption="The cone", show=["axis"]),
               step("0:35", caption="It widens", set=[{"param": "r", "to": 2}], over=3)],
              [{"name": "r", "value": 1, "min": 0.5, "max": 2, "label": "radius", "unit": ""}])


class FakeClient:
    def __init__(self, answers, delay=0.0, cost=0.01):
        self.answers = list(answers)
        self.delay, self.cost = delay, cost
        self.calls, self.seen = [], []
        self._lock = threading.Lock()

    def chat(self, messages, **kw):
        time.sleep(self.delay)
        with self._lock:
            self.seen.append((messages, kw))
            a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a
        finish = "stop"
        if isinstance(a, tuple):
            a, finish = a
        text = a if isinstance(a, str) else json.dumps(a)
        self.calls.append(CallRecord("/chat/completions", "muse-spark-1.3", 200, 0.1, False, 1,
                                     {"prompt_tokens": 100, "completion_tokens": 50}, self.cost, "k"))
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


@pytest.fixture
def job(tmp_path):
    popout._failures.clear()
    return write_job(tmp_path / "lec", board_replay())


def test_a_scene_is_made_once_and_kept_for_this_lecture_version(job):
    fake = FakeClient([GOOD])
    scene = popout.make_scene(lambda: fake, job, 1)
    assert scene["format"] == 2 and scene["piece"] == 1 and scene["pieces"] == [1, 2, 3] and scene["synced"]
    assert [s["t"] for s in scene["steps"]] == [30, 35]
    assert len(fake.seen) == 1
    messages, kw = fake.seen[0]
    assert messages[0]["role"] == "system" and "# Transcript" in messages[0]["content"]  # the whole lecture
    assert kw["schema_name"] == "scene_3d" and kw["extra"]["prompt_cache_key"].startswith("replay-board-lec-")
    content = messages[1]["content"]
    assert "#1 diagram" in content[0]["text"] and "#2 label" in content[0]["text"]
    assert sum(1 for c in content if c["type"] == "image_url") == 1  # the drawing on the board colour
    log = json.loads((job / "popout" / "001.v2-log.json").read_text(encoding="utf-8"))
    assert log["cost_usd"] == pytest.approx(0.01) and log["calls"][0]["usage"]["prompt_tokens"] == 100

    again = popout.make_scene(lambda: pytest.fail("no model call for a kept scene"), job, 1)
    assert again == scene

    # A reprocessed lecture (or new rules) makes it again; a format-1 file is never read.
    replay = json.loads((job / "replay.json").read_text(encoding="utf-8"))
    replay["subject"] = "cones"
    (job / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    (job / "popout" / "008.json").write_text(json.dumps({"kind": "solids", "solids": []}), encoding="utf-8")
    fake2 = FakeClient([GOOD])
    newer = popout.make_scene(lambda: fake2, job, 1)
    assert len(fake2.seen) == 1 and newer["version"] != scene["version"]
    popout.make_scene(lambda: fake2, job, 8)
    assert len(fake2.seen) == 2


def test_two_requests_at_once_make_one_model_call(job):
    fake = FakeClient([GOOD], delay=0.3)
    made, out = [], []

    def client():
        made.append(1)
        return fake

    threads = [threading.Thread(target=lambda: out.append(popout.make_scene(client, job, 1))) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(fake.seen) == 1 and len(made) == 1 and len(out) == 3 and out[0] == out[1] == out[2]


def test_an_answer_cut_short_is_asked_again_with_other_text(job):
    fake = FakeClient([(GOOD, "length"), GOOD])
    scene = popout.make_scene(lambda: fake, job, 1)
    assert scene["objects"] and len(fake.seen) == 2
    first, second = (m[1]["content"][0]["text"] for m, _ in fake.seen)
    assert first != second and "Try 2" in second


def test_answers_with_nothing_drawable_are_asked_again_then_fail(job):
    junk = answer([obj("x", "graph", expr="eval(x)")], [step("0:30")])
    fake = FakeClient(["not json", junk])
    with pytest.raises(MetaAPIError):
        popout.make_scene(lambda: fake, job, 1)
    assert len(fake.seen) == 2
    # The next request asks with new text (the disk cache may hold the answers that failed).
    fake2 = FakeClient([GOOD])
    popout.make_scene(lambda: fake2, job, 1)
    assert "Try 3" in fake2.seen[0][0][1]["content"][0]["text"]


def test_after_an_unexpected_failure_the_next_request_asks_with_new_text(job, monkeypatch):
    # An answer the checks trip over must not come back from the disk cache on every later request (a 500 for
    # good): "Try again" asks with new text.
    real = popout.clean_scene
    monkeypatch.setattr(popout, "clean_scene", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        popout.make_scene(lambda: FakeClient([GOOD]), job, 1)
    monkeypatch.setattr(popout, "clean_scene", real)
    fake = FakeClient([GOOD])
    assert popout.make_scene(lambda: fake, job, 1)["objects"]
    assert "(Try " in fake.seen[0][0][1]["content"][0]["text"]


def test_slide_lectures_and_unknown_pieces_have_no_scene(job, tmp_path):
    with pytest.raises(popout.NoScene):
        popout.make_scene(lambda: pytest.fail("no call"), job, 42)
    (tmp_path / "new").mkdir()
    with pytest.raises(popout.NoScene, match="not processed yet"):
        popout.make_scene(lambda: pytest.fail("no call"), tmp_path / "new", 1)
    replay = json.loads((job / "replay.json").read_text(encoding="utf-8"))
    replay["type"] = "slides"
    (job / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    with pytest.raises(popout.NoScene):
        popout.make_scene(lambda: pytest.fail("no call"), job, 1)


# ---------- the endpoint ----------
@pytest.fixture
def api(tmp_path, monkeypatch):
    popout._failures.clear()
    monkeypatch.setattr(server, "JOBS", tmp_path)
    write_job(tmp_path / "lec", board_replay())

    def use(fake):
        monkeypatch.setattr(server, "_meta_client", lambda: fake)
        return fake

    return TestClient(server.app), tmp_path / "lec", use


def test_endpoint_returns_the_scene(api):
    client, d, use = api
    fake = use(FakeClient([GOOD]))
    r = client.post("/api/jobs/lec/popout/1")
    assert r.status_code == 200 and r.json()["format"] == 2 and len(r.json()["steps"]) == 2
    assert client.post("/api/jobs/lec/popout/1").json() == r.json() and len(fake.seen) == 1


def test_endpoint_codes_are_one_plain_line(api):
    client, d, use = api
    fake = use(FakeClient([MetaAPIError("/chat/completions (muse-spark-1.3) failed: HTTP 500: boom\nmore")]))
    r = client.post("/api/jobs/lec/popout/1", headers={"origin": "https://evil.example"})
    assert r.status_code == 403 and fake.seen == []
    for url, code in (("/api/jobs/nope/popout/1", 404), ("/api/jobs/lec/popout/99", 404),
                      ("/api/jobs/lec/popout/1", 502)):
        r = client.post(url)
        assert r.status_code == code and "\n" not in r.json()["detail"], (url, r.json())
    assert r.json()["detail"].startswith("Muse Spark could not make the 3D view")


def test_endpoint_unexpected_failure_is_one_line_with_a_log(api, monkeypatch):
    client, d, use = api
    monkeypatch.setattr(popout, "ask_scene", lambda *a, **k: 1 / 0)
    use(FakeClient([GOOD]))
    r = client.post("/api/jobs/lec/popout/1")
    assert r.status_code == 500 and "\n" not in r.json()["detail"] and (d / "popout-error.log").exists()


# ---------- rules 4: cues, defs, one point per point, one scene per drawing ----------
def said(*pairs):
    """Transcript words: (start, word) pairs."""
    return [{"w": w, "s": float(t), "e": float(t) + 0.3} for t, w in pairs]


WORDS = said((13.1, "the"), (13.5, "simple"), (14.6, "cubic,"), (15.2, "where"), (16.0, "all"), (17.0, "sides"),
             (49.2, "We"), (49.6, "have"), (50.1, "what"), (50.5, "we"), (51.0, "call"), (52.4, "Tetragonal"),
             (53.0, "tetragonal"), (64.0, "it"), (64.5, "is"), (65.0, "pushed"), (65.4, "over"), (66.0, "to"),
             (66.3, "the"), (66.7, "right"))


def test_a_cue_moves_its_step_to_the_word():
    assert popout.find_cue(WORDS, "cubic", 13) == 14.6
    assert popout.find_cue(WORDS, "Tetragonal", 49) == 52.4  # the first time it is said, any case and punctuation
    assert popout.find_cue(WORDS, "pushed over", 63) == 65.0   # several words, in order
    assert popout.find_cue(WORDS, "push over", 63) is None     # not said like that, and no long word of it is
    assert popout.find_cue(WORDS, "leaning to the right", 63) is None
    assert popout.find_cue(WORDS, "pushed sideways", 63) == 65.0  # its first long word, said there
    assert popout.find_cue(WORDS, "tetragonal", 13) is None    # 39 s later: not this line's
    assert popout.find_cue(WORDS, "", 13) is None and popout.find_cue([], "cubic", 13) is None
    notes = []
    steps, synced = popout.clean_steps(
        [step("0:13", caption="cubic", cue="cubic"), step("0:49", caption="tetra", cue="tetragonal"),
         step("0:50", caption="no cue"), step("1:03", caption="lost", cue="rhombohedral")],
        [], [], CLIP, notes, words=WORDS)
    assert synced and [s["t"] for s in steps] == [14.6, 52.4, 53.4, 63]  # the step after "tetragonal" moves on
    assert [s["at"] for s in steps] == ["0:14", "0:52", "0:53", "1:03"]
    assert any("rhombohedral" in n for n in notes)
    # a scene that is not synced has no times, cue or not
    steps, synced = popout.clean_steps([step("", cue="cubic"), step("", cue="")], [], [], CLIP, [], words=WORDS)
    assert not synced and all(s["t"] is None for s in steps)
    assert "cue" not in steps[0]  # the scene's steps keep their format


def test_a_point_given_several_points_becomes_several_points():
    notes = []
    atoms = obj("atoms", "point", points=[["0", "0", "0"], ["1", "0", "0"], ["2", "0", "0"]], label="atom")
    tag = obj("tag", "label", points=[["0", "0", "1"], ["2", "0", "1"]], label="corner")
    edge = obj("edge", "segment", points=[["0", "0", "0"], ["2", "0", "0"]])
    scene = popout.clean_scene(
        answer([edge, atoms, tag], [step("0:20", caption="atoms", show=["atoms"]),
                                    step("0:30", caption="tags", show=["Tag"], highlight=["atoms"])]),
        piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    ids = [o["id"] for o in scene["objects"]]
    assert ids == ["edge", "atoms", "atoms_2", "atoms_3", "tag", "tag_2"]
    by = {o["id"]: o for o in scene["objects"]}
    assert [by[i]["points"] for i in ("atoms", "atoms_2", "atoms_3")] == [[["0", "0", "0"]], [["1", "0", "0"]],
                                                                          [["2", "0", "0"]]]
    assert by["atoms"]["label"] == "atom" and by["atoms_2"]["label"] == "" and by["tag_2"]["label"] == "corner"
    s0, s1 = scene["steps"]
    assert s1["show"] == ["tag", "tag_2"] and s1["highlight"] == ["atoms", "atoms_2", "atoms_3"]
    assert set(s0["show"] + s0["highlight"]) == {"atoms", "atoms_2", "atoms_3"}
    assert any("3 points made 3 points" in n for n in notes)
    # never past the limit on objects
    many = obj("many", "point", points=[[str(i), "0", "0"] for i in range(60)])
    scene = popout.clean_scene(answer([many], [step("0:20", highlight=["many"])]), piece=1, pieces=[1],
                               version="v", clip=CLIP)
    assert len(scene["objects"]) == popout.MAX_OBJECTS and len(scene["steps"][0]["highlight"]) == popout.MAX_OBJECTS


def test_defs_write_a_formula_once():
    notes = []
    params = [THETA, {"name": "r", "value": 2, "min": 1, "max": 3, "label": "radius", "unit": ""}]
    defs = [{"name": "cx", "expr": "r*cos(theta*deg)"}, {"name": "cy", "expr": "r*sin(theta*deg)"},
            {"name": "mid", "expr": "cx/2"},             # uses an earlier def
            {"name": "late", "expr": "later + 1"},       # a def after it: not allowed
            {"name": "theta", "expr": "1"},              # a param's name
            {"name": "sin", "expr": "1"},                # a function's name
            {"name": "wx", "expr": "x + 1"},             # a field's variable
            {"name": "later", "expr": "2"}]
    rod = obj("rod", "segment", points=[["0", "0", "0"], ["cx", "cy", "0"]])
    dot = obj("dot", "point", points=[["mid", "0", "0"]])
    bad = obj("bad", "point", points=[["late", "0", "0"]])  # names a def that was dropped
    raw = answer([rod, dot, bad], [step("0:20", caption="turn", set=[{"param": "theta", "to": 90}])], params)
    raw["defs"] = defs
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert [d["name"] for d in scene["defs"]] == ["cx", "cy", "mid", "later"]
    assert scene["defs"][0] == {"name": "cx", "expr": "r*cos(theta*deg)"}
    assert [o["id"] for o in scene["objects"]] == ["rod", "dot"]
    # the geometry is evaluated with the defs, through every state: the rod turns from (2, 0) to (0, 2)
    (x0, y0, _), (x1, y1, _) = scene["bounds"]
    assert x1 == pytest.approx(2) and y1 == pytest.approx(2) and x0 == pytest.approx(0) and y0 == pytest.approx(0)
    assert sum(n.startswith("def ") for n in notes) == 4
    # a def that is not finite in some state drops the objects that use it
    raw["defs"] = [{"name": "h", "expr": "sqrt(cos(theta*deg) - 0.5)"}]  # past 60 degrees: no square root
    raw["objects"] = [obj("tip", "point", points=[["h", "0", "0"]]), obj("o", "point", points=[["0", "0", "0"]])]
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    assert [o["id"] for o in scene["objects"]] == ["o"]
    assert len(popout.clean_defs([{"name": f"d{i}", "expr": "1"} for i in range(30)], set(), [])) == popout.MAX_DEFS


def test_a_view_with_one_lone_line_before_the_first_step_gets_the_first_step_objects():
    edges = [obj(f"e{i}", "segment", points=[["0", "0", "0"], [str(i + 1), "0", "1"]]) for i in range(5)]
    notes = []
    scene = popout.clean_scene(answer(edges, [step("0:20", caption="cube", show=["e1", "e2", "e3", "e4"]),
                                              step("0:30", caption="more")]),
                               piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    s0 = scene["steps"][0]
    assert s0["show"] == [] and s0["highlight"] == ["e1", "e2", "e3", "e4"]
    assert any("there from the start" in n for n in notes)
    # three or more there already: left alone
    scene = popout.clean_scene(answer(edges, [step("0:20", caption="two", show=["e3", "e4"])]), piece=1,
                               pieces=[1], version="v", clip=CLIP)
    assert scene["steps"][0]["show"] == ["e3", "e4"]


def test_axes_drawn_again_are_dropped_unless_a_step_names_them():
    notes = []
    xa = obj("x_axis", "segment", points=[["-2", "0", "0"], ["2", "0", "0"]])
    ya = obj("y_axis", "vector", points=[["0", "-2", "0"], ["0", "2", "0"]])
    asym = obj("asym", "segment", points=[["0", "-2", "0"], ["0", "2", "0"]], style="dashed")
    short = obj("tick", "segment", points=[["0", "0", "0"], ["0.5", "0", "0"]])
    moving = obj("rod", "segment", points=[["-2", "0", "0"], ["2*cos(theta*deg)", "0", "0"]])
    g = obj("g", "graph", expr="x^2")
    raw = answer([xa, ya, asym, short, moving, g], [step("", caption="asymptote", highlight=["asym"])], [THETA],
                 show_axes=True)
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert [o["id"] for o in scene["objects"]] == ["asym", "tick", "rod", "g"]
    assert sum("axis again" in n for n in notes) == 2
    # without axes there is nothing to draw twice
    raw["axes"]["show"] = False
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    assert len(scene["objects"]) == 6


def test_a_caption_naming_an_angle_nothing_shows_is_logged():
    gamma = {"name": "gamma", "value": 90, "min": 60, "max": 120, "label": "gamma", "unit": "deg"}
    arc = obj("arc", "curve", x="cos(t*deg)", y="sin(t*deg)", z="0", range=["0", "gamma"], label="γ = 90°")
    edge = obj("edge", "segment", points=[["0", "0", "0"], ["1", "0", "0"]], label="a")
    edge2 = obj("edge2", "segment", points=[["0", "0", "0"], ["0", "1", "0"]], label="b")
    edge3 = obj("edge3", "segment", points=[["0", "0", "0"], ["0", "0", "1"]], label="c")
    notes = []
    popout.clean_scene(answer([edge, edge2, edge3, arc],
                              [step("0:20", caption="Gamma is 120 degrees", set=[{"param": "gamma", "to": 120}]),
                               step("0:30", caption="Gamma again", show=["arc"])], [gamma]),
                       piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert [n for n in notes if "caption names" in n] == [
        "step 1: the caption names gamma, but nothing shown is labelled gamma"]  # the arc (gamma) comes at step 2


@pytest.mark.parametrize("kind, text, latex, ok", [
    ("graph", "", "", True), ("diagram", "a cell", "", True), ("Graphs", "", "", True), ("sketch", "", "", True),
    ("text", "y = ln x", "y = \\ln x", False), ("label", "a", "a", False), ("arrow", "", "", False),
    ("equation", "y = |ln x| = |log_e x|", "y = |\\ln x| = |\\log_e x|", True),
    ("equation", "y = ln|x|", "y=\\ln|x|", True), ("equation", "z = x^2 + y^2", "", True),
    ("formula", "", "f(x) = e^{-x}", True), ("function", "", "$y = mx + c$", True),
    ("equation", "", "f(x, y) = \\sin x \\cos y", True),
    ("arrow", "x-axis and y-axis arrows labeled x and y", "", True), ("arrow", "red vector arrow labeled v", "", True),
    ("arrow", "z-axis arrow labeled Z", "", True), ("arrow", "arrow from the sum to the result", "", False),
    ("arrow", "y = ln x", "y = \\ln x", False),
    ("equation", "A = P(1+r)^n", "A = P(1+r)^n", False), ("equation", "e = 2.71828", "e = 2.71828", False),
    ("equation", "y = 3", "", False), ("equation", "y = \\exp(2)", "", False), ("equation", "x = 3", "", False),
])
def test_only_drawings_and_graph_equations_have_a_3d_view_as_in_the_player(kind, text, latex, ok):
    assert popout.can_popout({"read": {"kind": kind, "text": text, "latex": latex}}) is ok
    assert not popout.can_popout({"read": None}) and not popout.can_popout(None)


def test_every_piece_of_a_drawing_opens_one_scene(job):
    fake = FakeClient([GOOD])
    by_eq = popout.make_scene(lambda: fake, job, 9)  # the graph's equation, inside it
    assert by_eq["piece"] == 9 and by_eq["pieces"][0] == 9 and 8 in by_eq["pieces"]
    by_graph = popout.make_scene(lambda: pytest.fail("one call for the drawing"), job, 8)
    assert by_graph["piece"] == 8 and by_graph["pieces"][0] == 8 and set(by_graph["pieces"]) == set(by_eq["pieces"])
    assert {k: v for k, v in by_eq.items() if k not in ("piece", "pieces")} == \
        {k: v for k, v in by_graph.items() if k not in ("piece", "pieces")}
    assert len(fake.seen) == 1 and (job / "popout" / "008.v2.json").exists()
    assert not (job / "popout" / "009.v2.json").exists()
    text = fake.seen[0][0][1]["content"][0]["text"]
    assert "The drawing: #8" in text and "opened" not in text  # the same request from any of its pieces
    # writing that is not a drawing nor a graph's equation: no scene, and no model call
    for pid in (2, 4, 5, 6):  # a label, a heading, V = pi r^2 h / 3, a note
        with pytest.raises(popout.NoScene):
            popout.make_scene(lambda: pytest.fail("no call"), job, pid)


def test_two_pieces_of_one_drawing_at_once_make_one_model_call(job):
    fake = FakeClient([GOOD], delay=0.3)
    out = []
    threads = [threading.Thread(target=lambda p=p: out.append(popout.make_scene(lambda: fake, job, p)))
               for p in (8, 9, 8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(fake.seen) == 1 and sorted(s["piece"] for s in out) == [8, 8, 9]


def test_no_scene_while_the_word_times_are_still_being_worked_out(job):
    replay = json.loads((job / "replay.json").read_text(encoding="utf-8"))
    replay["transcript"]["timing"] = "proportional"
    (job / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    (job / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    with pytest.raises(popout.NotReady, match="word times"):
        popout.make_scene(lambda: pytest.fail("no call yet"), job, 1)
    # the job ended with rough times (the aligner failed): they are the lecture's times now
    (job / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
    fake = FakeClient([GOOD])
    assert popout.make_scene(lambda: fake, job, 1)["objects"] and len(fake.seen) == 1


def test_endpoint_refuses_writing_that_has_no_3d_view(api):
    client, d, use = api
    fake = use(FakeClient([GOOD]))
    r = client.post("/api/jobs/lec/popout/2")
    assert r.status_code == 404 and "drawings" in r.json()["detail"] and fake.seen == []
    r = client.post("/api/jobs/lec/popout/9")
    assert r.status_code == 200 and r.json()["piece"] == 9


def test_making_scenes_ahead_of_time_asks_first(job, monkeypatch, capsys):
    import app.core.meta_api as meta_api
    fake = FakeClient([GOOD])
    monkeypatch.setattr(meta_api, "MetaClient", lambda: fake)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert popout.main([str(job), "1", "2"]) == 1  # #2 is a label: one plain line, nothing spent
    assert "error: #2" in capsys.readouterr().err and fake.seen == []
    assert popout.main([str(job), "1", "8"]) == 0
    out = capsys.readouterr().out
    assert "2 3D view(s) to make (#1, #8): about $0.06 to $0.30" in out and "nothing made" in out and fake.seen == []
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert popout.main([str(job), "1", "9", "8"]) == 0  # #9 opens #8: made once
    out = capsys.readouterr().out
    assert len(fake.seen) == 2 and "spent $0.020" in out
    assert popout.main([str(job), "9", "--yes"]) == 0
    assert "already has its 3D view" in capsys.readouterr().out and len(fake.seen) == 2


def test_a_param_named_like_a_variable_is_renamed_not_dropped():
    # test-full #3 (rules 4): the place of a point sliding along y = ln(x^2) was a param called "t"; dropping it took
    # the point, its def and the step that moved it, and the caption promised a point that was not there.
    notes = []
    t = {"name": "t", "value": 1, "min": 0.5, "max": 3, "label": "place", "unit": ""}
    raw = answer([obj("mover", "point", points=[["t", "yt", "0"]]),
                  obj("g", "graph", expr="ln(x^2) + 0*t"),
                  obj("arc", "curve", x="cos(t)", y="sin(t)", z="0", range=["0", "t"]),  # t: the curve's own
                  obj("band", "box", center=["t", "0", "0"], size=["1", "1", "1"])],
                 [step("", caption="the point slides", set=[{"param": "t", "to": 2.71828}], show=["mover"])], [t])
    raw["defs"] = [{"name": "yt", "expr": "ln(t^2)"}]
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert [p["name"] for p in scene["params"]] == ["p_t"]
    assert scene["defs"] == [{"name": "yt", "expr": "ln(p_t^2)"}]
    by = {o["id"]: o for o in scene["objects"]}
    assert by["mover"]["points"] == [["p_t", "yt", "0"]] and by["g"]["expr"] == "ln(x^2) + 0*p_t"
    assert by["arc"]["x"] == "cos(t)" and by["arc"]["range"] == ["0", "p_t"]  # inside a curve, t is still t
    assert by["band"]["center"] == ["p_t", "0", "0"]
    assert scene["steps"][0]["set"] == [{"param": "p_t", "to": 2.71828}]
    assert any("renamed p_t" in n for n in notes)
    # x would clash with a graph's variable the same way
    xp = {"name": "x", "value": 1, "min": 0, "max": 2, "label": "", "unit": ""}
    raw = answer([obj("dot", "point", points=[["x", "0", "0"]]), obj("g", "graph", expr="x^2")], [step("")], [xp])
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    by = {o["id"]: o for o in scene["objects"]}
    assert by["dot"]["points"] == [["p_x", "0", "0"]] and by["g"]["expr"] == "x^2"


# ---------- rules 6: angle objects, explorable ranges, steps that change something, captions that point ----------
def angle(oid, pts, size=(), label="", **kw):
    return obj(oid, "angle", points=[list(p) for p in pts], size=list(size), label=label, **kw)


def measured(scene, oid, finals=True):
    """An angle object's measured value (degrees) at the start and after each step that moves something."""
    o = next(o for o in scene["objects"] if o["id"] == oid)
    shape = popout.Shape(o, popout._names(scene), scene["axes"])
    envs = popout._envs(scene, popout.param_states(scene["params"], scene["steps"], finals=finals))
    return [None if (a := shape.angle_at(env)) is None else round(a, 6) for env in envs]


CELL = [{"name": "a", "value": 3, "min": 1, "max": 6, "label": "a", "unit": ""},
        {"name": "c", "value": 4, "min": 1, "max": 6, "label": "c", "unit": ""},
        {"name": "beta", "value": 90, "min": 60, "max": 120, "label": "beta", "unit": "deg"}]
CELL_DEFS = [{"name": "cx", "expr": "c*cos(beta*deg)"}, {"name": "cz", "expr": "c*sin(beta*deg)"}]


def cell(steps, extra=()):
    edge_a = obj("edge_a", "segment", points=[["0", "0", "0"], ["a", "0", "0"]], label="a")
    edge_c = obj("edge_c", "segment", points=[["0", "0", "0"], ["cx", "0", "cz"]], label="c")
    beta = angle("beta_angle", [["0", "0", "0"], ["a", "0", "0"], ["cx", "0", "cz"]], label="beta")
    raw = answer([edge_a, edge_c, beta, *extra], steps, CELL)
    raw["defs"] = CELL_DEFS
    return raw


def test_the_scene_format_has_angles():
    assert "angle" in popout.TYPES and "angle" in popout.SCENE_SCHEMA["properties"]["objects"]["items"][
        "properties"]["type"]["enum"]
    assert popout.SCENE_RULES_VERSION == "6"
    task = popout.SCENE_TASK
    assert "angle: points [vertex, end1, end2]" in task and "never a curve" in task and "slider" in task
    assert "Every length the teacher names" in task and "Every step visibly changes" in task


def test_an_angle_follows_its_edges_and_measures_what_the_steps_say():
    scene = popout.clean_scene(cell([step("0:20", caption="Monoclinic: beta is 75 degrees",
                                          set=[{"param": "beta", "to": 75}]),
                                     step("0:30", caption="tilted further", set=[{"param": "beta", "to": 105}])]),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    (o,) = [o for o in scene["objects"] if o["type"] == "angle"]
    assert o["points"] == [["0", "0", "0"], ["a", "0", "0"], ["cx", "0", "cz"]] and o["size"] == []
    assert o["label"] == "beta" and set(o) == set(scene["objects"][0])  # flat, with every field
    assert measured(scene, "beta_angle") == [90, 75, 105]
    json.dumps(scene, allow_nan=False)


def test_an_angle_has_three_points_and_at_most_one_radius():
    notes = []
    v, p, q = ["0", "0", "0"], ["1", "0", "0"], ["0", "1", "0"]
    objs = [angle("two", [v, p]), angle("four", [v, p, q, q]), angle("bad_r", [v, p, q], size=["x"]),
            angle("two_r", [v, p, q], size=["0.5", "2"]), angle("blank_r", [v, p, q], size=[""]),
            angle("ok", [v, p, q], size=["0.25"]), angle("var", [v, ["t", "0", "0"], q])]
    scene = popout.clean_scene(answer(objs, [step("0:20", highlight=["ok"])]), piece=1, pieces=[1], version="v",
                               clip=CLIP, notes=notes)
    by = {o["id"]: o for o in scene["objects"]}
    assert set(by) == {"two_r", "blank_r", "ok"}
    assert by["two_r"]["size"] == ["0.5"] and by["blank_r"]["size"] == [] and by["ok"]["size"] == ["0.25"]
    assert any("2 points for an angle" in n for n in notes) and any("4 points for an angle" in n for n in notes)
    assert any("only its first size kept" in n for n in notes)
    assert any(n.startswith("object bad_r dropped") for n in notes)
    assert any(n.startswith("object var dropped") for n in notes)  # no variables in an angle's points


@pytest.mark.parametrize("label, name", [("beta = 75°", "beta"), ("β 75°", "β"), ("γ=120", "γ"),
                                         ("theta: 30 degrees", "theta"), ("θ1", "θ1"), ("angle ABC", "angle ABC"),
                                         ("alpha", "alpha"), ("= 90", ""),
                                         # a label that is only a value would show it twice ("90° 90°")
                                         ("90°", ""), ("120 degrees", ""), ("β = 75º", "β"), ("γ 120˚", "γ"),
                                         ("1", "1"), ("angle 1", "angle 1")])
def test_an_angle_label_is_its_name_without_a_value(label, name):
    notes = []
    o = popout.clean_object(angle("ang", [["0", "0", "0"], ["1", "0", "0"], ["0", "1", "0"]], label=label), 0,
                            set(), notes)
    assert o["label"] == name
    assert any("the value taken out of its label" in n for n in notes) == (label != name)
    # other objects keep their label as it is
    seg = popout.clean_object(obj("s", "segment", points=[["0", "0", "0"], ["1", "0", "0"]], label="a = 3 cm"), 0,
                              set(), [])
    assert seg["label"] == "a = 3 cm"


def test_an_angle_is_framed_by_its_vertex_and_radius():
    only = angle("ang", [["1", "1", "1"], ["3", "1", "1"], ["1", "5", "1"]])  # arms 2 and 4: auto radius 0.6
    scene = popout.clean_scene(answer([only], [step("0:20", highlight=["ang"])]), piece=1, pieces=[1], version="v",
                               clip=CLIP)
    assert scene["bounds"] == [[0.4, 0.4, 0.4], [1.6, 1.6, 1.6]]
    only["size"] = ["0.5"]
    scene = popout.clean_scene(answer([only], [step("0:20", highlight=["ang"])]), piece=1, pieces=[1], version="v",
                               clip=CLIP)
    assert scene["bounds"] == [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
    assert measured(scene, "ang") == [90]


@pytest.mark.parametrize("p2", [["1", "1e-8", "0"], ["-1", "1e-8", "0"], ["1", "1e-12", "0"], ["-2", "3e-9", "0"],
                                ["0.3", "0.7", "-0.2"], ["0", "1", "0"]])
def test_an_angle_measures_what_the_player_measures(p2):
    # the player measures atan2(|a x b|, a.b) (angle3d.ts): acos lost every angle under about 1e-6 degrees and
    # the same near 180
    o = popout.clean_object(angle("ang", [["0", "0", "0"], ["1", "0", "0"], p2]), 0, set(), [])
    a, b = np.array([1.0, 0, 0]), np.array([float(v) for v in p2])
    want = math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))
    assert popout.Shape(o, set(), {}).angle_at({}) == pytest.approx(want, rel=1e-12, abs=1e-15)


def test_a_degenerate_angle_is_dropped_only_where_the_view_rests():
    k = {"name": "k", "value": 1, "min": -1, "max": 1, "label": "", "unit": ""}
    r = {"name": "r", "value": 0.5, "min": 0, "max": 1, "label": "", "unit": ""}
    v, q = ["0", "0", "0"], ["0", "1", "0"]
    objs = [angle("swing", [v, ["k", "0", "0"], q]),           # its arm passes through 0 half-way: allowed
            angle("ends_flat", [v, ["k + 1", "0", "0"], q]),   # an arm of length 0 after the step: dropped
            angle("starts_flat", [v, ["k - 1", "0", "0"], q]),  # an arm of length 0 at the start: dropped
            angle("nan", [v, ["sqrt(k - 1.5)", "0", "0"], q]),  # not finite at the start: dropped
            angle("no_radius", [v, ["1", "0", "0"], q], size=["r"]),  # radius 0 after the second step: dropped
            obj("dot", "point", points=[["0", "0", "1"]])]
    notes = []
    scene = popout.clean_scene(answer(objs, [step("0:20", caption="swing", set=[{"param": "k", "to": -1}], over=4),
                                             step("0:30", caption="shrink", set=[{"param": "r", "to": 0}])],
                                      [k, r]), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert [o["id"] for o in scene["objects"]] == ["swing", "dot"]
    assert sum("the angle is degenerate" in n for n in notes) == 4
    assert measured(scene, "swing") == [90, 90, 90]
    assert measured(scene, "swing", finals=False)[2] is None  # half-way through the swing, k = 0: not drawn then
    json.dumps(scene, allow_nan=False)


def test_params_get_a_range_to_explore():
    notes = []
    ps = popout.clean_params([
        {"name": "beta", "value": 90, "min": 90, "max": 90, "label": "", "unit": "deg"},  # no range: +-30 degrees
        {"name": "len", "value": 3, "min": 3, "max": 3, "label": "", "unit": ""},         # +-half of it
        {"name": "zero", "value": 0, "min": 0, "max": 0, "label": "", "unit": ""},        # +-1
        {"name": "out", "value": 7, "min": 1, "max": 6, "label": "", "unit": ""},         # widened to hold 7
    ], notes)
    assert [(p["min"], p["value"], p["max"]) for p in ps] == [(60, 90, 120), (1.5, 3, 4.5), (-1, 0, 1), (1, 7, 7)]
    assert "param beta: no range to explore, so 60..120" in notes
    assert "param out: range widened from 1..6 to 1..7 to hold its values" in notes
    # every step's "to" is inside the range too, never cut to it: the caption says that value
    gamma = {"name": "gamma", "value": 90, "min": 80, "max": 100, "label": "gamma", "unit": "deg"}
    notes = []
    scene = popout.clean_scene(answer([obj("p", "point", points=[["cos(gamma*deg)", "0", "0"]])],
                                      [step("0:20", caption="hexagonal: gamma is 120",
                                            set=[{"param": "gamma", "to": 120}])], [gamma]),
                               piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    (p,) = scene["params"]
    assert (p["min"], p["value"], p["max"]) == (80, 90, 120) and scene["steps"][0]["set"][0]["to"] == 120
    assert "param gamma: range widened from 80..100 to 80..120 to hold its values" in notes


def test_a_value_cut_to_the_limit_is_never_silent():
    # a value beyond +-LIMIT is cut to it: the run log says so, never "already there" or nothing at all
    n = {"name": "n", "value": 3e5, "min": 0, "max": 1e6, "label": "", "unit": ""}
    notes = []
    scene = popout.clean_scene(answer([obj("q", "point", points=[["n/1e5", "0", "0"]])],
                                      [step("0:20", caption="n doubles", set=[{"param": "n", "to": 6e5}])], [n]),
                               piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert scene["params"][0]["value"] == popout.LIMIT
    assert "param n: value 300000 cut to 10000 (at most 10000 either way)" in notes
    assert "param n: max 1e+06 cut to 10000 (at most 10000 either way)" in notes
    assert "step: n set to 600000 cut to 10000 (at most 10000 either way)" in notes


def test_a_step_that_changes_nothing_is_dropped():
    k = {"name": "k", "value": 0, "min": 0, "max": 1, "label": "", "unit": ""}
    objs = [obj(i, "point", points=[[str(n), "0", "0"]]) for n, i in enumerate("abcd")]
    steps = [step("0:20", caption="k stays", set=[{"param": "k", "to": 0}]),   # 1: moves nothing
             step("0:25", caption="d appears", show=["d"]),                     # 2
             step("0:30", caption="d again", show=["d"]),                       # 3: d is there already
             step("0:35", caption="d goes", hide=["d"]),                        # 4
             step("0:40", caption="d gone", hide=["d"]),                        # 5: d is gone already
             step("0:45", caption="iso", camera="iso"),                         # 6: the start camera
             step("0:50", caption="from above", camera="top"),                  # 7
             step("0:55", caption="above", camera="top"),                       # 8: already from above
             step("1:00", caption="k moves", set=[{"param": "k", "to": 1}]),    # 9
             step("1:05", caption="only words")]                                # 10
    notes = []
    scene = popout.clean_scene(answer(objs, steps, [k]), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert [s["caption"] for s in scene["steps"]] == ["d appears", "d goes", "from above", "k moves"]
    assert [n for n in notes if "changes nothing" in n] == [
        "step 1 dropped: it changes nothing on screen ('k stays')",
        "step 3 dropped: it changes nothing on screen ('d again')",
        "step 5 dropped: it changes nothing on screen ('d gone')",
        "step 6 dropped: it changes nothing on screen ('iso')",
        "step 8 dropped: it changes nothing on screen ('above')",
        "step 10 dropped: it changes nothing on screen ('only words')"]
    assert "step 1: set of k removed: already there" in notes


def test_sets_that_move_nothing_are_taken_out_of_a_step():
    raw = cell([step("0:20", caption="tetragonal: c is longer",
                     set=[{"param": "a", "to": 3}, {"param": "c", "to": 5}, {"param": "beta", "to": 90}])])
    notes = []
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert scene["steps"][0]["set"] == [{"param": "c", "to": 5}]
    assert "step 1: set of a, beta removed: already there" in notes


def test_the_start_camera_stays_when_the_step_that_named_it_goes():
    k = {"name": "k", "value": 0, "min": 0, "max": 1, "label": "", "unit": ""}
    dots = [obj("dot", "point", points=[["k", "0", "0"]])] + [obj(f"p{i}", "point", points=[[str(i), "1", "0"]])
                                                             for i in range(3)]
    scene = popout.clean_scene(answer(dots, [step("0:20", caption="look", camera="front"),
                                             step("0:25", caption="move", set=[{"param": "k", "to": 1}]),
                                             step("0:30", caption="turn", camera="iso")], [k]),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    assert [(s["caption"], s["camera"]) for s in scene["steps"]] == [("move", "front"), ("turn", "iso")]


def test_a_caption_points_at_what_it_names():
    k = {"name": "k", "value": 0, "min": 0, "max": 1, "label": "", "unit": ""}
    v = ["0", "0", "0"]
    objs = [obj("edge_a", "segment", points=[v, ["1", "0", "0"]], label="a"),
            angle("alpha_angle", [v, ["0", "1", "0"], ["0", "0", "1"]], label="α"),
            angle("beta_angle", [v, ["1", "0", "0"], ["0", "0", "1"]], label="beta"),
            angle("gamma_angle", [v, ["1", "0", "0"], ["0", "1", "0"]], label="Gamma"),
            obj("tag", "label", points=[["0", "0", "2"]], label="apex"),
            obj("corner", "point", points=[["1", "1", "1"]], label="corner"),
            obj("mover", "point", points=[["k", "0", "0"]])]
    steps = [step("0:20", caption="Alpha and γ stay at 90 degrees; a is the same; the alphabet",
                  set=[{"param": "k", "to": 1}]),
             step("0:30", caption="Beta is not 90"),                          # beta is shown later: shown here
             step("0:40", caption="beta again", show=["beta_angle"], set=[{"param": "k", "to": 0}]),
             step("0:50", caption="the apex, the corner, alpha, beta and gamma"),  # 5 named: 3 pointed at
             step("1:00", caption="the apex goes", hide=["tag"]),             # its own hide is left alone
             step("1:10", caption="apex"),                                    # base object hidden: not shown again
             step("1:20", caption="mu", highlight=["mover"])]
    notes = []
    scene = popout.clean_scene(answer(objs, steps, [k]), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    s = {x["caption"]: x for x in scene["steps"]}
    first = s["Alpha and γ stay at 90 degrees; a is the same; the alphabet"]
    assert first["highlight"] == ["alpha_angle", "gamma_angle"] and first["show"] == []  # never "a", nor "alphabet"
    assert s["Beta is not 90"]["show"] == ["beta_angle"] and s["Beta is not 90"]["highlight"] == []
    # its own show stays, and what it shows is visible then, so it glows too
    assert s["beta again"]["show"] == ["beta_angle"] and s["beta again"]["highlight"] == ["beta_angle"]
    assert s["the apex, the corner, alpha, beta and gamma"]["highlight"] == ["tag", "corner", "alpha_angle"]
    assert s["the apex goes"]["hide"] == ["tag"] and s["the apex goes"]["highlight"] == []
    assert "apex" not in s  # nothing to point at (the apex is hidden and part of the base): nothing changes
    assert "step 1: alpha_angle highlighted: its caption names 'α'" in notes
    assert "step 2: beta_angle shown: its caption names 'beta'" in notes
    assert sum(" highlighted: " in n or " shown: " in n for n in notes) == 7


def test_a_caption_naming_an_angle_is_satisfied_by_an_angle_object():
    gamma = {"name": "gamma", "value": 90, "min": 60, "max": 120, "label": "gamma", "unit": "deg"}
    v = ["0", "0", "0"]
    objs = [obj("edge", "segment", points=[v, ["1", "0", "0"]]),
            obj("edge2", "segment", points=[v, ["cos(gamma*deg)", "sin(gamma*deg)", "0"]]),
            obj("edge3", "segment", points=[v, ["0", "0", "1"]]),
            angle("g", [v, ["1", "0", "0"], ["cos(gamma*deg)", "sin(gamma*deg)", "0"]], label="γ")]
    notes = []
    scene = popout.clean_scene(answer(objs, [step("0:20", caption="Gamma is 120 degrees",
                                                  set=[{"param": "gamma", "to": 120}])], [gamma]),
                               piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert not [n for n in notes if "caption names gamma" in n]
    assert scene["steps"][0]["highlight"] == ["g"] and measured(scene, "g") == [90, 120]


@pytest.mark.parametrize("text, words", [("Angle β", "angle beta"), ("Γ", "gamma"),
                                         ("Σ and ς", "sigma and sigma"), ("αβ", "alpha beta"),
                                         ("  Beta=75°", "beta=75°")])
def test_plain_words(text, words):
    assert popout.plain_words(text) == words


# ---------- the list of drawings that have a 3D view ----------
def test_the_list_of_made_scenes(api, monkeypatch):
    client, d, use = api
    r = client.get("/api/jobs/lec/popouts")
    assert r.status_code == 200 and r.json() == {"drawings": []} and r.headers["cache-control"] == "no-cache"
    use(FakeClient([GOOD]))
    assert client.post("/api/jobs/lec/popout/1").status_code == 200
    assert client.post("/api/jobs/lec/popout/9").status_code == 200  # drawing 8
    monkeypatch.setattr(server, "_meta_client", lambda: pytest.fail("the list never calls the model"))
    r = client.get("/api/jobs/lec/popouts")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    one, eight = r.json()["drawings"]
    assert one == {"drawing": 1, "pieces": [1, 2, 3], "title": "A turning rod", "steps": 2, "synced": True,
                   "times": [30.0, 35.0]}
    assert eight["drawing"] == 8 and eight["pieces"][0] == 8 and 9 in eight["pieces"] and eight["synced"]
    # files the POST would not hand back (an old format, a half-written file, a broken one) are not listed
    (d / "popout" / "005.json").write_text("{}", encoding="utf-8")
    (d / "popout" / "006.v2.abcd1234.tmp").write_text("{", encoding="utf-8")
    (d / "popout" / "007.v2.json").write_text("{not json", encoding="utf-8")
    assert [x["drawing"] for x in client.get("/api/jobs/lec/popouts").json()["drawings"]] == [1, 8]
    # new rules: every scene is made again on its next open, so none is listed
    monkeypatch.setattr(popout, "SCENE_RULES_VERSION", "test-next")
    assert client.get("/api/jobs/lec/popouts").json() == {"drawings": []}


def test_the_list_follows_the_lecture_version_and_type(api):
    client, d, use = api
    use(FakeClient([answer([obj("p", "point", points=[["0", "0", "0"]])],
                            [step("", highlight=["p"]), step("", caption="b", highlight=["p"])])]))
    assert client.post("/api/jobs/lec/popout/1").json()["synced"] is False
    (drawing,) = client.get("/api/jobs/lec/popouts").json()["drawings"]
    assert drawing["synced"] is False and drawing["times"] == [] and drawing["steps"] == 2
    replay = json.loads((d / "replay.json").read_text(encoding="utf-8"))
    replay["subject"] = "cones"  # reprocessed: the scene is out of date
    (d / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    assert client.get("/api/jobs/lec/popouts").json() == {"drawings": []}
    replay["type"] = "slides"
    (d / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    assert client.get("/api/jobs/lec/popouts").json() == {"drawings": []}
    (d / "replay.json").unlink()  # not processed yet
    assert client.get("/api/jobs/lec/popouts").json() == {"drawings": []}
    r = client.get("/api/jobs/nope/popouts")
    assert r.status_code == 404 and r.json()["detail"] == "no such job"


def test_the_list_holds_only_what_a_click_on_its_badge_opens_without_a_model_call(api, monkeypatch):
    # The board puts a "3D" badge on each listed drawing, and a click asks POST popout/{drawing}: a file under any
    # other number (a piece with no 3D view, one that opens another drawing, no such piece) or a lecture whose word
    # times are still being worked out would make that click a paid model call or an error.
    client, d, use = api
    use(FakeClient([GOOD]))
    assert client.post("/api/jobs/lec/popout/1").status_code == 200
    made = (d / "popout" / "001.v2.json").read_text(encoding="utf-8")
    for other in (2, 6, 42):  # a label inside drawing 1, plain text, no such piece
        (d / "popout" / f"{other:03d}.v2.json").write_text(made, encoding="utf-8")
    monkeypatch.setattr(server, "_meta_client", lambda: pytest.fail("the list never calls the model"))
    assert [x["drawing"] for x in client.get("/api/jobs/lec/popouts").json()["drawings"]] == [1]
    for other in (6, 42):
        assert client.post(f"/api/jobs/lec/popout/{other}").status_code == 404
    replay = json.loads((d / "replay.json").read_text(encoding="utf-8"))
    replay["transcript"]["timing"] = "proportional"  # the aligner still running: POST answers 409
    raw = json.dumps(replay).encode("utf-8")
    (d / "replay.json").write_bytes(raw)
    (d / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    scene = json.loads(made)
    scene["version"] = popout.scene_version(raw)
    (d / "popout" / "001.v2.json").write_text(json.dumps(scene), encoding="utf-8")
    assert client.post("/api/jobs/lec/popout/1").status_code == 409
    assert client.get("/api/jobs/lec/popouts").json() == {"drawings": []}


@pytest.mark.parametrize("broken", [{"steps": 5}, {"pieces": 5}, {"steps": "abc"}, {"title": None},
                                    {"pieces": [True, 2.5, "x"]}, {"steps": [{"t": 1e400}, 7]}])
def test_a_broken_scene_file_never_breaks_the_list(api, broken):
    client, d, use = api
    use(FakeClient([GOOD]))
    assert client.post("/api/jobs/lec/popout/1").status_code == 200
    path = d / "popout" / "001.v2.json"
    scene = {**json.loads(path.read_text(encoding="utf-8")), **broken}
    path.write_text(json.dumps(scene), encoding="utf-8")
    r = client.get("/api/jobs/lec/popouts")
    assert r.status_code == 200
    (one,) = r.json()["drawings"]
    assert one["drawing"] == 1 and one["pieces"] and all(type(p) is int for p in one["pieces"])
    assert isinstance(one["title"], str) and all(math.isfinite(t) for t in one["times"])
    # as the player reads it (cleanScene): a synced scene keeps only its steps with a time
    assert one["steps"] == len(one["times"]) if one["synced"] else one["times"] == []


def test_an_unexpected_failure_of_the_list_is_one_plain_line(api, monkeypatch):
    client, d, _ = api
    monkeypatch.setattr(popout, "list_scenes", lambda job_dir: 1 / 0)
    r = TestClient(server.app, raise_server_exceptions=False).get("/api/jobs/lec/popouts")
    assert r.status_code == 500 and "\n" not in r.json()["detail"] and "ZeroDivisionError" in r.json()["detail"]


def test_a_label_object_that_only_repeats_a_name_on_screen_is_dropped():
    # test2 #7 (rules 6): lab_a, lab_b, lab_c beside the vectors a, b, c put each name twice on screen.
    o = ["0", "0", "0"]
    objs = [obj("vec_a", "vector", points=[o, ["2", "0", "0"]], label="a"),
            obj("vec_b", "vector", points=[o, ["0", "2", "0"]], label="b"),
            angle("beta_angle", [o, ["1", "0", "0"], ["0", "0", "1"]], label="beta"),
            obj("lab_a", "label", points=[["1", "-0.3", "0"]], label="a"),        # vec_a is always there: dropped
            obj("lab_b", "label", points=[["0", "1", "-0.3"]], label="b"),        # vec_b is hidden then: kept
            obj("lab_beta", "label", points=[["0.5", "0", "0.5"]], label="β"),    # the angle's name as a letter
            obj("apex", "label", points=[["0", "0", "2"]], label="apex"),         # nothing else says it: kept
            obj("edge", "segment", points=[o, ["0", "0", "2"]])]
    steps = [step("0:20", caption="the edges", show=["lab_a", "lab_b"], highlight=["vec_a"]),
             step("0:30", caption="b goes", hide=["vec_b"]),
             step("0:40", caption="the apex", show=["apex", "lab_beta"])]
    notes = []
    scene = popout.clean_scene(answer(objs, steps), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    ids = [x["id"] for x in scene["objects"]]
    assert "lab_a" not in ids and "lab_beta" not in ids
    assert {"vec_a", "vec_b", "beta_angle", "lab_b", "apex", "edge"} <= set(ids)
    assert "object lab_a dropped: its label 'a' is said already by vec_a, there whenever it is" in notes
    assert any(n.startswith("object lab_beta dropped") and "beta_angle" in n for n in notes)
    first = scene["steps"][0]
    assert first["show"] == ["lab_b"] and first["highlight"] == ["vec_a"]
    assert all("lab_a" not in s[k] for s in scene["steps"] for k in ("show", "hide", "highlight"))
    assert scene["steps"][2]["show"] == ["apex"]
    # a step that only showed a dropped label changes nothing: dropped too, with its note
    steps = [step("0:20", caption="name it", show=["lab_a"]), step("0:30", caption="edge", highlight=["edge"])]
    notes = []
    scene = popout.clean_scene(answer(objs, steps), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert [s["caption"] for s in scene["steps"]] == ["edge"]
    assert any("changes nothing" in n and "name it" in n for n in notes)


@pytest.mark.parametrize("text, n, cut", [
    ("position along x parabola", 24, "position along x…"),          # synth3d-saddle: never "parabo…"
    ("half length of traces", 24, "half length of traces"),          # fits: as it is
    ("a b c d e f g h i j k l m", 10, "a b c d e…"),
    ("supercalifragilistic expialidocious", 24, "supercalifragilistic…"),
    ("averyveryverylongwordwithnospace", 12, "averyveryve…"),          # one word longer than the field: cut inside
    ("x " + "y" * 30, 12, "x yyyyyyyyy…"),                           # the word boundary would keep too little
    ("one two, three", 10, "one two…"),                              # no comma left before the dots
])
def test_a_long_text_is_cut_at_the_end_of_a_word(text, n, cut):
    out = popout._str(text, n)
    assert out == cut and len(out) <= n


def test_an_angle_with_a_short_arm_gets_a_radius_that_shows():
    # synth3d-vector3d: theta between the x axis, given by the point (1, 0, 0), and v = (3, 2, 4): the auto radius
    # (0.3 of the 1-unit arm) drew a wedge nobody could see.
    o = ["0", "0", "0"]
    L = {"name": "L", "value": 5, "min": 1, "max": 6, "label": "", "unit": ""}
    # each at a vertex of its own (angles that share one are nested: test_angles_at_one_corner_are_nested)
    objs = [angle("theta", [o, ["1", "0", "0"], ["3", "2", "4"]], label="theta"),
            angle("cell", [["0", "20", "0"], ["2.5", "20", "0"], ["0", "20", "5"]], label="beta"),  # 2.5, 5: as it is
            angle("given", [["0", "40", "0"], ["1", "40", "0"], ["0", "40", "9"]], size=["0.5"]),  # its own: kept
            angle("shrinks", [["0", "60", "0"], ["1", "60", "0"], ["0", "60 + L", "0"]], label="phi")]  # L 5 -> 2
    notes = []
    scene = popout.clean_scene(answer(objs, [step("0:20", caption="smaller", set=[{"param": "L", "to": 2}])], [L]),
                               piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    size = {x["id"]: x["size"] for x in scene["objects"]}
    assert size["theta"] == ["0.808"]  # 0.15 x 5.385, under 0.9 x 1
    assert size["cell"] == [] and size["given"] == ["0.5"]
    assert size["shrinks"] == []  # with L = 2 the most it may have is 0.3, the auto radius: nothing gained
    assert "object theta: radius 0.808 (its arms are 1 and 5.39 long: 0.3 of the shorter would hardly show)" in notes
    lo, hi = scene["bounds"]
    assert lo[0] == pytest.approx(-0.808) and hi[0] == pytest.approx(0.808)  # framed by vertex +- its radius
    assert measured(scene, "theta") == [pytest.approx(56.145, abs=1e-3)] * 2


# ---------- rules from the critic's pass (fixer, rules 6) ----------
TRICLINIC = [{"name": "a", "value": 3, "min": 1, "max": 6, "label": "edge a", "unit": ""},
             {"name": "b", "value": 3, "min": 1, "max": 6, "label": "edge b", "unit": ""},
             {"name": "c", "value": 3, "min": 1, "max": 6, "label": "edge c", "unit": ""},
             {"name": "alpha", "value": 90, "min": 60, "max": 120, "label": "angle alpha", "unit": "deg"},
             {"name": "beta", "value": 90, "min": 60, "max": 120, "label": "angle beta", "unit": "deg"},
             {"name": "gamma", "value": 90, "min": 60, "max": 120, "label": "angle gamma", "unit": "deg"}]
# test2 #4's defs (its cz is clamped, so a cell that cannot exist is drawn as a flat plate)
TRICLINIC_DEFS = [{"name": "bx", "expr": "b*cos(gamma*deg)"}, {"name": "by", "expr": "b*sin(gamma*deg)"},
                  {"name": "cx", "expr": "c*cos(beta*deg)"},
                  {"name": "cy", "expr": "c*(cos(alpha*deg) - cos(beta*deg)*cos(gamma*deg))/sin(gamma*deg)"},
                  {"name": "cz", "expr": "sqrt(max(c*c - cx*cx - cy*cy, 0.2))"}]


def corner_cell(steps):
    o = ["0", "0", "0"]
    objs = [obj("face", "polygon", points=[o, ["a", "0", "0"], ["a + bx", "by", "0"], ["bx", "by", "0"]]),
            obj("edge_c", "segment", points=[o, ["cx", "cy", "cz"]], label="c"),
            obj("top", "polygon", points=[["cx", "cy", "cz"], ["a + cx", "cy", "cz"],
                                          ["a + bx + cx", "by + cy", "cz"]]),
            angle("alpha_angle", [o, ["bx", "by", "0"], ["cx", "cy", "cz"]], label="alpha"),
            angle("beta_angle", [o, ["a", "0", "0"], ["cx", "cy", "cz"]], label="beta"),
            angle("gamma_angle", [o, ["a", "0", "0"], ["bx", "by", "0"]], label="gamma")]
    raw = answer(objs, steps, TRICLINIC)
    raw["defs"] = TRICLINIC_DEFS
    return raw


SERIES = [step("0:20", caption="Cubic: all angles 90 degrees"),
          step("0:40", caption="Rhombohedral", show=["alpha_angle", "beta_angle", "gamma_angle"],
               set=[{"param": "alpha", "to": 80}, {"param": "beta", "to": 80}, {"param": "gamma", "to": 80}]),
          step("1:00", caption="Monoclinic: beta is not 90",
               set=[{"param": "a", "to": 2.5}, {"param": "b", "to": 4}, {"param": "c", "to": 5},
                    {"param": "alpha", "to": 90}, {"param": "beta", "to": 75}, {"param": "gamma", "to": 90}])]


def test_the_word_angles_shows_the_angles_not_on_screen_yet():
    # test2 #4: "all angles 90 degrees" for 90 s with no angle marked; they came only at "Rhombohedral"
    notes = []
    scene = popout.clean_scene(corner_cell(SERIES), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert scene["steps"][0]["show"] == ["alpha_angle", "beta_angle", "gamma_angle"]
    assert "step 1: alpha_angle shown: its caption names 'angles (alpha)'" in notes
    # a whole word only: "triangle" and "angled" name no angle
    steps = [step("0:20", caption="a triangle, angled", set=[{"param": "a", "to": 4}]),
             step("0:40", caption="the angle", show=["alpha_angle", "beta_angle", "gamma_angle"])]
    scene = popout.clean_scene(corner_cell(steps), piece=1, pieces=[1], version="v", clip=CLIP)
    assert scene["steps"][0]["show"] == [] and len(scene["steps"]) == 2


def test_angles_at_one_corner_are_nested():
    # test2 #4: alpha, beta and gamma at one corner, all of radius 0.75: three marks on top of each other
    notes = []
    raw = corner_cell(SERIES)
    raw["objects"].append(angle("lone", [["9", "0", "0"], ["10", "0", "0"], ["9", "1", "0"]], label="phi"))
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    size = {o["id"]: o["size"] for o in scene["objects"] if o["type"] == "angle"}
    assert size["lone"] == []  # at a vertex of its own: the auto radius
    envs = popout._envs(scene, popout.param_states(scene["params"], scene["steps"], finals=True))
    ids = ("alpha_angle", "beta_angle", "gamma_angle")
    shapes = {i: popout.Shape(next(o for o in scene["objects"] if o["id"] == i), popout._names(scene), scene["axes"])
              for i in ids}
    for env in envs:
        arms = {i: s.arms(env) for i, s in shapes.items()}
        r = [arms[i][3] for i in ids]
        shortest = min(float(np.linalg.norm(v)) for a in arms.values() for v in (a[1], a[2]))
        assert r == pytest.approx([0.3 * shortest, 0.435 * shortest, 0.57 * shortest])
        assert max(r) < shortest  # inside every arm
    assert any("share a vertex: radii 1, 1.45, 1.9" in n for n in notes)


def gram(al, be, ga):
    cos = lambda d: math.cos(math.radians(d))  # noqa: E731
    return 1 - cos(al) ** 2 - cos(be) ** 2 - cos(ga) ** 2 + 2 * cos(al) * cos(be) * cos(ga)


HEXAGONAL = SERIES + [step("1:20", caption="Hexagonal: gamma is 120",
                           set=[{"param": "a", "to": 3}, {"param": "b", "to": 3}, {"param": "c", "to": 4.5},
                                {"param": "alpha", "to": 90}, {"param": "beta", "to": 90},
                                {"param": "gamma", "to": 120}])]


def test_each_angle_range_holds_what_that_angle_can_take_with_the_lecture_values():
    # test2 #7: one box whose every corner made a cell shrank all three angles together (alpha 77..94, because gamma
    # had to reach 120 in the same box). Each range now holds what its angle can take with the other values as the
    # lecture has them; the player stops a slider where a mix of the student's values would make no cell
    # (scene3dState.ts shapeGuard).
    raw = corner_cell(HEXAGONAL)
    gamma = next(p for p in raw["params"] if p["name"] == "gamma")
    raw["params"] = [p for p in raw["params"] if p["name"] != "gamma"] + [{**gamma, "min": 20, "max": 170}]
    notes = []
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    rng = {p["name"]: (p["min"], p["max"]) for p in scene["params"]}
    assert rng["a"] == (1, 6)  # lengths are not touched
    # alpha and beta: 60..120 is a cell with the others at 90 (the cubic start): left as they are, wider than 77..94
    assert rng["alpha"] == (60, 120) and rng["beta"] == (60, 120)
    # gamma 20..170: with alpha = beta = 90 the cell lies flat past |cos gamma| = sqrt(1 - FLAT_MIN) (42.1..137.9)
    lim = math.degrees(math.acos(math.sqrt(1 - popout.FLAT_MIN)))
    assert rng["gamma"] == (math.ceil(lim), math.floor(180 - lim)) == (43, 137)
    assert "param gamma: range narrowed from 20..170 to 43..137" in " ".join(notes)
    # each end makes a cell with the other values as the lecture has them somewhere (here the cubic start)
    for g in rng["gamma"]:
        assert gram(90, 90, g) >= popout.FLAT_MIN
    # but not every mix of the ends does (the player checks those): 120/120/120 is a flat plate
    assert gram(120, 120, 120) < popout.FLAT_MIN
    # the lecture's own values always stay inside, even past where the rest would narrow them
    for s in scene["steps"]:
        for st in s["set"]:
            p = next(p for p in scene["params"] if p["name"] == st["param"])
            assert p["min"] <= st["to"] <= p["max"]
    # one angle alone (the monoclinic beta of `cell`) exists over its whole range: left as it is
    scene = popout.clean_scene(cell([step("0:20", caption="lean", set=[{"param": "beta", "to": 75}])]),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    assert [(p["min"], p["max"]) for p in scene["params"] if p["name"] == "beta"] == [(60, 120)]


def test_a_range_narrowed_by_one_lecture_state_is_widened_by_another():
    # gamma 60..170 with alpha = beta = 80 (rhombohedral) reaches less far than with alpha = beta = 90 (cubic): the
    # range is the widest any resting state of the lecture allows
    raw = corner_cell(SERIES)
    raw["params"] = [{**p, "max": 170} if p["name"] == "gamma" else p for p in raw["params"]]
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    hi = next(p["max"] for p in scene["params"] if p["name"] == "gamma")
    assert hi == 137  # the cubic start's limit, not the rhombohedral one
    assert gram(80, 80, 138) < popout.FLAT_MIN <= gram(90, 90, 137)


def faces_cell(labels, steps=None):
    """test2 #4 as Muse Spark made it: a cell of six faces, and a, b, c as loose labels."""
    o = ["0", "0", "0"]
    a, ab, b = ["a", "0", "0"], ["a + bx", "by", "0"], ["bx", "by", "0"]
    c, ac, abc, bc = ["cx", "cy", "cz"], ["a + cx", "cy", "cz"], ["a + bx + cx", "by + cy", "cz"], ["bx + cx", "by + cy", "cz"]
    faces = [obj("face_bottom", "polygon", points=[o, a, ab, b], color="#a0c4ff", opacity=0.25),
             obj("face_top", "polygon", points=[c, ac, abc, bc], color="#a0c4ff", opacity=0.25),
             obj("face_front", "polygon", points=[o, a, ac, c], color="#90befa", opacity=0.3),
             obj("face_back", "polygon", points=[b, ab, abc, bc], color="#90befa", opacity=0.25),
             obj("face_left", "polygon", points=[o, b, bc, c], color="#b8d4ff", opacity=0.25),
             obj("face_right", "polygon", points=[a, ab, abc, ac], color="#b8d4ff", opacity=0.25)]
    raw = answer(faces + labels, steps if steps is not None else [
        step("0:20", caption="Cubic: a b c equal", show=[x["id"] for x in labels]),
        step("0:40", caption="Tetragonal: c longer", highlight=["label_c"], set=[{"param": "c", "to": 4.5}]),
        step("1:00", caption="Monoclinic: pushed over",
             set=[{"param": "a", "to": 2.5}, {"param": "b", "to": 4}, {"param": "c", "to": 5},
                  {"param": "beta", "to": 75}])], TRICLINIC)
    raw["defs"] = TRICLINIC_DEFS
    return raw


T4_LABELS = [obj("label_a", "label", points=[["a/2", "-0.5", "0.2"]], color="#d62728", label="a"),
             obj("label_b", "label", points=[["bx/2", "by/2 + 0.5", "0.2"]], color="#2ca02c", label="b"),
             obj("label_c", "label", points=[["cx + 0.6", "cy", "cz/2"]], color="#ff7f0e", label="c")]


def test_a_loose_label_naming_a_length_becomes_its_edge_in_its_colour():
    # test2 #4: "c" sat mid-face (0.6 off its edge, 1.2 once the cell leaned) and every edge was the faces' blue
    notes = []
    scene = popout.clean_scene(faces_cell(T4_LABELS), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    by = {o["id"]: o for o in scene["objects"]}
    o = ["0", "0", "0"]
    assert by["label_a"]["type"] == "segment" and by["label_a"]["points"] == [o, ["a", "0", "0"]]
    assert by["label_b"]["type"] == "segment" and by["label_b"]["points"] == [o, ["bx", "by", "0"]]
    assert by["label_c"]["type"] == "segment" and by["label_c"]["points"] == [o, ["cx", "cy", "cz"]]
    # each in its own colour, with its name, solid; the faces keep theirs
    assert [(by[i]["color"], by[i]["label"], by[i]["style"], by[i]["opacity"]) for i in ("label_a", "label_b", "label_c")] \
        == [("#d62728", "a", "solid", 1.0), ("#2ca02c", "b", "solid", 1.0), ("#ff7f0e", "c", "solid", 1.0)]
    assert by["face_front"]["color"] == "#90befa"
    # the steps still show and point at them under the same ids
    assert scene["steps"][0]["show"] == ["label_a", "label_b", "label_c"]
    assert scene["steps"][1]["highlight"] == ["label_c"]
    # edge c is c long in every state the view rests in, the cell leaning or not
    shape = popout.Shape(by["label_c"], popout._names(scene), scene["axes"])
    for env in popout._envs(scene, popout.param_states(scene["params"], scene["steps"], finals=True)):
        pts, _ = shape.points_at(env)
        assert float(np.linalg.norm(pts[1] - pts[0])) == pytest.approx(env["c"])
    assert "object label_c: the label 'c' is the edge c beside it (0.6 off it), drawn in its colour" in notes


def test_a_label_stays_a_label_unless_an_edge_beside_it_is_that_length():
    far = obj("label_c", "label", points=[["9", "9", "9"]], color="#ff7f0e", label="c")  # nowhere near edge c
    angle_name = obj("label_beta", "label", points=[["0.5", "0", "0.5"]], label="beta")  # an angle, not a length
    named_other = obj("label_d", "label", points=[["1", "0", "0"]], label="d")  # names no param
    raw = faces_cell([far, angle_name, named_other], steps=[step("0:20", caption="the cell")])
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    assert {o["id"]: o["type"] for o in scene["objects"] if o["id"].startswith("label")} == \
        {"label_c": "label", "label_beta": "label", "label_d": "label"}
    # a side c long only at the start (it is a, which a step stretches) is not edge c
    labels = [obj("label_c", "label", points=[["a/2", "-0.3", "0"]], label="c")]
    raw = faces_cell(labels, steps=[step("0:20", caption="a longer", set=[{"param": "a", "to": 5}])])
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    assert next(o["type"] for o in scene["objects"] if o["id"] == "label_c") == "label"
    # only a polygon's sides: an edge drawn as a segment of its own is left to that segment
    edge = obj("edge_c", "segment", points=[["0", "0", "0"], ["0", "0", "c"]])
    lab = obj("label_c", "label", points=[["0.3", "0", "c/2"]], label="c")
    scene = popout.clean_scene(answer([edge, lab], [step("0:20", caption="c")], TRICLINIC),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    assert [o["type"] for o in scene["objects"]] == ["segment", "label"]


def test_a_side_drawn_as_its_own_segment_is_not_drawn_again_from_a_label():
    # the cell's faces, its edge c drawn as a segment of its own (written the other way round), and "c" beside it:
    # turning the label into edge c drew a second, orange c on the blue one
    for ends in ([["0", "0", "0"], ["cx", "cy", "cz"]], [["cx", "cy", "cz"], ["0", "0", "0"]]):
        edge = obj("edge_c", "segment", points=ends, color="#336699")
        lab = obj("label_c", "label", points=[["cx + 0.6", "cy", "cz/2"]], color="#ff7f0e", label="c")
        raw = faces_cell([lab], steps=[step("0:20", caption="Cubic: a b c equal", show=["label_c"])])
        raw["objects"].append(edge)
        notes = []
        scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
        by = {o["id"]: o for o in scene["objects"]}
        assert by["label_c"]["type"] == "label" and by["edge_c"]["type"] == "segment"
        assert not [n for n in notes if "is the edge" in n]
    # the same side written with other expressions that land on the same corners counts as drawn too
    edge = obj("edge_c", "vector", points=[["0*a", "0", "0"], ["c*cos(beta*deg)", "cy", "cz"]])
    raw = faces_cell([obj("label_c", "label", points=[["cx + 0.6", "cy", "cz/2"]], label="c")],
                     steps=[step("0:20", caption="c", show=["label_c"])])
    raw["objects"].append(edge)
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    assert next(o["type"] for o in scene["objects"] if o["id"] == "label_c") == "label"
    # the other sides still become edges: a and b, with c drawn already
    raw = faces_cell(T4_LABELS)
    raw["objects"].append(obj("edge_c", "segment", points=[["0", "0", "0"], ["cx", "cy", "cz"]]))
    scene = popout.clean_scene(raw, piece=1, pieces=[1], version="v", clip=CLIP)
    assert {o["id"]: o["type"] for o in scene["objects"] if o["id"].startswith(("label", "edge"))} ==         {"label_a": "segment", "label_b": "segment", "label_c": "label", "edge_c": "segment"}


def test_a_label_nearer_another_drawn_length_names_that_one_not_a_side():
    # a regular hexagon: its sides are r long, and so is its dashed radius, which "r" sits beside
    pts = [[f"r*cos({k * 60}*deg)", f"r*sin({k * 60}*deg)", "0"] for k in range(6)]
    radius = obj("radius", "segment", points=[["0", "0", "0"], ["r", "0", "0"]], style="dashed")
    params = [{"name": "r", "value": 2, "min": 1, "max": 3, "label": "radius r"}]
    for at, becomes in ((["r/2", "-0.15", "0"], "label"),  # 0.15 off the radius, 0.79 off a side: the radius's
                        (["1.6", "-0.6", "0"], "segment")):  # 0.05 off a side, 0.6 off the radius: the side's
        objs = [obj("hex", "polygon", points=pts, color="#1f6fd1", opacity=0.3), radius,
                obj("label_r", "label", points=[at], color="#d0342c", label="r")]
        scene = popout.clean_scene(answer(objs, [step("0:20", caption="The radius r", show=["radius", "label_r"])], params),
                                   piece=1, pieces=[1], version="v", clip=CLIP)
        assert next(o["type"] for o in scene["objects"] if o["id"] == "label_r") == becomes, at
    # with no radius drawn, "r" beside a side is that side
    objs = [obj("hex", "polygon", points=pts, color="#1f6fd1", opacity=0.3),
            obj("label_r", "label", points=[["1.6", "-0.6", "0"]], color="#d0342c", label="r")]
    scene = popout.clean_scene(answer(objs, [step("0:20", caption="A side r", show=["label_r"])], params),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    assert next(o["type"] for o in scene["objects"] if o["id"] == "label_r") == "segment"


def test_a_step_looks_from_where_what_it_points_at_is_seen():
    # synth3d-vector3d: v = (2, -3, 4) seen from "iso" (24 degrees off its line) showed at 0.41 of its length
    o = ["0", "0", "0"]
    objs = [obj("x_axis", "vector", points=[o, ["0", "-5", "0"]], label="x"),
            obj("y_axis", "vector", points=[o, ["5", "0", "0"]], label="y"),
            obj("z_axis", "vector", points=[o, ["0", "0", "5"]], label="z"),
            obj("v", "vector", points=[o, ["2", "-3", "4"]], label="v"),
            obj("comp", "segment", points=[o, ["0", "-3", "0"]])]
    steps = [step("0:10", caption="axes", highlight=["x_axis", "y_axis", "z_axis"], camera="iso"),
             step("0:20", caption="the vector", show=["v"], camera="iso"),
             step("0:30", caption="still it", highlight=["v"]),  # keeps the camera the step before turned to
             step("0:40", caption="first along x", show=["comp"], camera="iso")]
    notes = []
    scene = popout.clean_scene(answer(objs, steps), piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    # v points into iso's usual front-right corner: the scene's iso looks from the front-left, where v is 0.85 of its
    # length and the components of steps 3-5 still show (from "side", the y axis pointed at the viewer)
    assert scene["iso_from"] == "left"
    assert [s["camera"] for s in scene["steps"]] == ["iso", "iso", "keep", "iso"]
    assert "iso looks from the front-left: the worst line on screen is seen 0.61 face on, not 0.41" in notes
    assert sum("camera" in n for n in notes) == 0
    # an angle: its plane faces the camera when its normal is along the line of sight
    theta = angle("theta", [o, ["0", "-1", "0"], ["2", "-3", "4"]], label="theta")
    notes = []
    scene = popout.clean_scene(answer(objs + [theta], [step("0:10", caption="theta", show=["theta"], camera="iso")]),
                               piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert scene["steps"][0]["camera"] == "side" and "theta seen 0.75 face on, not 0.59" in " ".join(notes)


def test_iso_stays_at_the_front_right_when_nothing_points_into_it():
    o = ["0", "0", "0"]
    objs = [obj("x_axis", "vector", points=[o, ["5", "0", "0"]], label="x"),
            obj("y_axis", "vector", points=[o, ["0", "5", "0"]], label="y"),
            obj("z_axis", "vector", points=[o, ["0", "0", "5"]], label="z"),
            obj("v", "vector", points=[o, ["-2", "3", "4"]], label="v")]
    notes = []
    scene = popout.clean_scene(answer(objs, [step("0:10", caption="the vector", show=["v"], camera="iso")]),
                               piece=1, pieces=[1], version="v", clip=CLIP, notes=notes)
    assert "iso_from" not in scene and scene["steps"][0]["camera"] == "iso"
    assert not any("front-left" in n for n in notes)


def test_a_param_label_up_to_32_characters_is_whole():
    # synth3d-saddle: "position along x parabola" (25) was cut to "position along x…" in the value chips
    assert popout.PARAM_LABEL_CHARS == 32
    p = {"name": "px", "value": 1, "min": -2, "max": 2, "label": "position along x parabola", "unit": ""}
    assert popout.clean_params([p], [])[0]["label"] == "position along x parabola"


def test_the_prompt_asks_for_points_built_from_the_defs_and_angles_where_marked():
    task = popout.SCENE_TASK
    assert "Build every point from the params and defs" in task  # test2 #7: the prism walls ignored alpha
    assert "goes at the corner where it is marked" in task  # test2 #7: gamma at the hexagon's centre
    assert "at most 32 characters" in task


def _deg_scene(captions, params=(("beta", "β", "deg", 90), ("gamma", "gamma", "deg", 120), ("a", "a", "", 3)),
               sets=(("beta", 104.5), ("beta", 75))):
    steps = [{"caption": c, "set": []} for c in captions]
    steps[0]["set"] = [{"param": n, "to": v} for n, v in sets]
    return {"params": [{"name": n, "label": l, "unit": u, "value": v} for n, l, u, v in params], "steps": steps}


@pytest.mark.parametrize("caption, expected", [
    # test2 #7's captions, in the teacher's words (its angles are 90 and 120)
    ("Tetragonal: stretched cube with a equals b, c longer, angles 90",
     "Tetragonal: stretched cube with a equals b, c longer, angles 90°"),
    ("Orthorhombic: shoe box with a b c all different, angles still 90",
     "Orthorhombic: shoe box with a b c all different, angles still 90°"),
    ("Monoclinic: leans one way with alpha gamma 90 and beta not 90",
     "Monoclinic: leans one way with alpha gamma 90° and beta not 90°"),
    ("Rhombohedral: cube squished so sides equal and angles equal not 90",
     "Rhombohedral: cube squished so sides equal and angles equal not 90°"),
    ("β = 75 now", "β = 75° now"),
    ("beta 104.5, gamma 120", "beta 104.5°, gamma 120°"),
    ("beta=90, so it stands straight", "beta=90°, so it stands straight"),
    ("All angles are still 90 here", "All angles are still 90° here"),
    ("gamma 120 at the base", "gamma 120° at the base"),
    ("angles 90.", "angles 90°."),
    # left alone: not a value the scene's angles take
    ("beta is 45 now", "beta is 45 now"),
    ("a is 3 and c is 4", "a is 3 and c is 4"),
    ("angle 720", "angle 720"),
    # left alone: a unit, a ratio, pi or radians
    ("Simple cubic: all angles 90 degrees", "Simple cubic: all angles 90 degrees"),
    ("Hexagon corner with gamma 120° at the base", "Hexagon corner with gamma 120° at the base"),
    ("beta 90 degs", "beta 90 degs"),
    ("Cos beta is 3 over root 29", "Cos beta is 3 over root 29"),
    ("The cosine of beta is 90", "The cosine of beta is 90"),
    ("beta is 90 radians", "beta is 90 radians"),
    ("beta and gamma are 90 and 120 radians", "beta and gamma are 90 and 120 radians"),
    ("beta is 90 pi over 3", "beta is 90 pi over 3"),
    ("beta now 90 and a bit radians", "beta now 90 and a bit radians"),
    # left alone: a count, an index, a coefficient, a list of several values
    ("The angles at all 8 corners are 90", "The angles at all 8 corners are 90"),
    ("Of the angles only 1 is not 90", "Of the angles only 1 is not 90"),
    ("angles of 90 bonds", "angles of 90 bonds"),
    ("the number of angles is 90", "the number of angles is 90"),
    ("Angles 1 and 2 add up to 180", "Angles 1 and 2 add up to 180"),
    ("beta 1 is 30 and beta 2 is 19", "beta 1 is 30 and beta 2 is 19"),
    ("beta is 2 gamma", "beta is 2 gamma"),
    ("F and beta are 90 and 120", "F and beta are 90 and 120"),
    ("Angle 90 is marked", "Angle 90 is marked"),
    # left alone: glued to text
    ("Angles in 90D look different", "Angles in 90D look different"),
    ("beta 90s style", "beta 90s style"),
    ("beta at 90:30 in the video", "beta at 90:30 in the video"),
    ("beta is a 90-degree angle", "beta is a 90-degree angle"),
    ("β90 and β120", "β90 and β120"),
    ("betamax 90", "betamax 90"),
    ("beta, then 90 more", "beta, then 90 more"),
])
def test_a_bare_angle_value_in_a_caption_gets_its_degree_sign(caption, expected):
    scene = _deg_scene([caption])
    popout.degree_captions(scene, [])
    assert scene["steps"][0]["caption"] == expected


def test_a_one_letter_angle_name_is_no_trigger_and_a_caption_never_outgrows_its_limit():
    # a triangle's angles named a, b, c: "a" is also the article (link_captions skips labels under 2 letters too)
    scene = _deg_scene(["Take a 60 degree turn", "a is 60 here"], params=(("a", "a", "deg", 60), ("b", "b", "deg", 60)),
                       sets=())
    popout.degree_captions(scene, [])
    assert [s["caption"] for s in scene["steps"]] == ["Take a 60 degree turn", "a is 60 here"]
    full = "x" * (popout.CAPTION_CHARS - len(" beta 90")) + " beta 90"
    scene, notes = _deg_scene([full]), []
    popout.degree_captions(scene, notes)
    assert scene["steps"][0]["caption"] == full and "would pass" in notes[0]


def test_no_degree_sign_without_an_angle_param():
    scene = _deg_scene(["angles 90", "theta 45"], params=(("a", "a", "", 90),), sets=())
    notes = []
    popout.degree_captions(scene, notes)
    assert [s["caption"] for s in scene["steps"]] == ["angles 90", "theta 45"] and notes == []
