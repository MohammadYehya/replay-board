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
    assert s["set"] == [{"param": "theta", "to": 90}] and s["camera"] == "keep" and s["over"] == 8
    assert any("ghost" in n for n in notes)


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
    assert ps[0] == {"name": "beta", "value": 120, "min": 60, "max": 120, "label": "β", "unit": "deg"}
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
    scene = popout.clean_scene(answer(objects[:1], [step("0:20", hide=["a"]), step("0:30", show=["a"])]), piece=1,
                               pieces=[1], version="v", clip=CLIP)
    assert scene["steps"][1]["show"] == ["a"]
    # shown twice, then hidden: there from the start, glowing where it was shown, and still hidden at the end
    scene = popout.clean_scene(answer(objects[:1], [step("0:20", caption="", show=["a"]),
                                                    step("0:30", caption="", show=["a"]), step("0:40", hide=["a"])]),
                               piece=1, pieces=[1], version="v", clip=CLIP)
    assert [(s["show"], s["highlight"], s["hide"]) for s in scene["steps"]] == [([], ["a"], []), ([], ["a"], []),
                                                                                ([], [], ["a"])]


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
