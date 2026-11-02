"""Fast checks of the 3D-figure lecture generator (tests/gen_scene_lectures.py): the timeline and the truth agree.

No rendering, speech or ffmpeg here: the plans use estimated speech times, and a built truth JSON (data/synth3d/)
is checked only when it exists.
"""
import copy
import json
import math

import pytest

from tests import gen_scene_lectures as g


@pytest.mark.parametrize("name", g.LECTURES)
def test_plan_is_consistent(name):
    pl = g.plan(g.lecture(name))
    assert g.check_truth(pl) == []
    assert g.MIN_DURATION <= pl["duration"] <= g.MAX_DURATION
    starts = [s["start"] for s in pl["stages"]]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


@pytest.mark.parametrize("name", g.LECTURES)
def test_every_stage_cue_is_said_in_its_stage(name):
    lec = g.lecture(name)
    pl = g.plan(lec)
    narration = " ".join(s["text"] for s in pl["segments"])
    for beat, st in zip(lec.beats, pl["stages"]):
        assert beat.cues, beat.stage
        for cue in beat.cues:
            assert g.has_word(beat.say, cue), (beat.stage, cue)
            assert g.has_word(narration, cue), (beat.stage, cue)
        for c in st["cues"]:
            assert st["start"] <= c["t"] <= st["end"], (beat.stage, c)


@pytest.mark.parametrize("name", g.LECTURES)
def test_each_ink_is_drawn_once_at_its_word(name):
    lec = g.lecture(name)
    pl = g.plan(lec)
    drawn = [d.ink for b in lec.beats for d in b.draws]
    assert sorted(drawn) == sorted(k.id for k in lec.inks)            # every ink drawn, none twice
    assert len(set(drawn)) == len(drawn)
    for i, beat in enumerate(lec.beats):
        seg = pl["segments"][i]
        for d in beat.draws:
            k = pl["inks"][d.ink]
            assert k["stage"] == beat.stage
            if d.word is not None:
                said = [w["s"] for w in seg["words"] if w["w"].lower() == d.word.lower()]
                assert said, (beat.stage, d.word)
                assert k["t_start"] >= said[0] + d.offset - 1e-6         # drawn while/after the word is said


@pytest.mark.parametrize("name", g.LECTURES)
def test_figure_and_formulas(name):
    lec = g.lecture(name)
    figure = [k for k in lec.inks if k.figure]
    assert len(figure) >= 5                                          # a figure drawn in stages
    for k in lec.inks:
        if k.kind == "text":
            g.parse_markup(k.text)                                    # every glyph is known
        else:
            for p in k.paths:
                for x, y in p:                                        # inside the board, clear of its edge band
                    assert g.BOARD[0] + 40 <= x <= g.BOARD[2] - 40 and g.BOARD[1] + 25 <= y <= g.BOARD[3] - 25


def test_plan_follows_real_speech_times():
    """With (fake) slower speech, everything moves later but stays consistent: drawings wait for their words."""
    lec = g.lecture("vector3d")
    slow = {}
    for i, b in enumerate(lec.beats):
        est = g.estimate_speech(b.say)
        slow[f"vector3d_{i:02d}"] = {"duration": est["duration"] * 1.05,
                                     "words": [{**w, "s": w["s"] * 1.05, "e": w["e"] * 1.05} for w in est["words"]]}
    fast, late = g.plan(lec), g.plan(lec, slow)
    assert g.check_truth(late) == []
    assert late["duration"] > fast["duration"]
    assert late["inks"]["comp_z"]["t_start"] > fast["inks"]["comp_z"]["t_start"]


def test_check_truth_finds_problems():
    pl = g.plan(g.lecture("cylinder"))
    assert g.check_truth(pl) == []

    bad = copy.deepcopy(pl)
    bad["stages"][2]["cues"].append({"w": "banana", "t": bad["stages"][2]["start"]})
    assert any("banana" in p for p in g.check_truth(bad))

    bad = copy.deepcopy(pl)
    bad["inks"]["top"]["t_end"] = bad["duration"] + 5
    assert any("top" in p for p in g.check_truth(bad))

    bad = copy.deepcopy(pl)
    bad["stages"][3]["start"], bad["stages"][4]["start"] = bad["stages"][4]["start"], bad["stages"][3]["start"]
    assert g.check_truth(bad)

    bad = copy.deepcopy(pl)
    bad["inks"]["sides"]["t_start"] = bad["inks"]["top"]["t_start"]   # two inks at once
    assert any("same time" in p or "outside" in p for p in g.check_truth(bad))

    bad = copy.deepcopy(pl)
    bad["duration"] = 30.0
    assert g.check_truth(bad)


def test_merge_sapi_words():
    text = "That is the square root of 29, about 5.4."
    events = [{"c": 0, "t": 0.1}, {"c": 5, "t": 0.3}, {"c": 8, "t": 0.4}, {"c": 12, "t": 0.5}, {"c": 19, "t": 0.9},
              {"c": 24, "t": 1.1}, {"c": 27, "t": 1.2}, {"c": 27, "t": 1.6},     # "29" spoken as two words
              {"c": 31, "t": 2.3}, {"c": 37, "t": 2.6}, {"c": 37, "t": 2.9}]   # so is "5.4"
    words = g.merge_sapi_words(text, events, speech_end=3.4)
    assert [w["w"] for w in words] == ["That", "is", "the", "square", "root", "of", "29", "about", "5.4"]
    assert [w["s"] for w in words] == [0.1, 0.3, 0.4, 0.5, 0.9, 1.1, 1.2, 2.3, 2.6]
    assert words[6]["e"] == pytest.approx(2.28)
    assert words[-1]["e"] == 3.4
    # a word SAPI did not report takes the time of the one before it
    words = g.merge_sapi_words("one two three", [{"c": 0, "t": 0.2}, {"c": 8, "t": 0.9}], speech_end=1.4)
    assert [w["s"] for w in words] == [0.2, 0.2, 0.9]


def test_markup():
    assert g.parse_markup("cos {theta} = 3/{sqrt:29}") == [("t", "cos "), ("theta", ""), ("t", " = 3/"),
                                                           ("sqrt", "29")]
    with pytest.raises(ValueError):
        g.parse_markup("{alpha}")


@pytest.mark.parametrize("name", g.LECTURES)
def test_built_truth(name):
    """The truth JSON next to a built video (skipped until `python -m tests.gen_scene_lectures` has run)."""
    path = g.OUT_DIR / f"{name}.json"
    if not path.is_file():
        pytest.skip("not built")
    truth = json.loads(path.read_text(encoding="utf-8"))
    if truth.get("GEN_VERSION") != g.GEN_VERSION or truth.get("TRUTH_VERSION") != g.TRUTH_VERSION:
        pytest.skip("built by an older generator: run `python -m tests.gen_scene_lectures` (no --force)")
    assert g.check_truth(truth) == []
    assert truth["narration_text"] == " ".join(b.say for b in g.lecture(name).beats)
    fx, fy, fw, fh = truth["figure"]["board_box"]
    assert 0 < fx and 0 < fy and fx + fw < 1 and fy + fh < 1
    for k in truth["inks"].values():
        x, y, w, h = k["box"]
        assert w > 0 and h > 0
    assert [s["check"] for s in truth["stages"]] == [json.loads(json.dumps(b.check)) for b in g.lecture(name).beats]
    assert truth["pipeline"]["open_piece"] == g.OPEN_ON[name]["piece"]


# ---------- each stage's check: the numbers the 3D view of the drawing should show by the stage's end ----------
def _checks(name):
    return [(b.stage, b.check) for b in g.lecture(name).beats]


@pytest.mark.parametrize("name", g.LECTURES)
def test_every_stage_has_a_check_and_the_figure_only_grows(name):
    drawn = []
    for stage, check in _checks(name):
        assert check, stage
        if "drawn" in check:
            assert check["drawn"][:len(drawn)] == drawn, stage           # nothing drawn is taken away
            drawn = check["drawn"]
    assert drawn, "the figure is never drawn"


def test_vector3d_checks_agree():
    facts = g.lecture("vector3d").facts
    c = dict(_checks("vector3d"))
    v = c["vector"]["v"]["to"]
    assert v == facts["vector"]
    length = math.sqrt(sum(x * x for x in v))
    assert c["vector"]["v"]["length"] == pytest.approx(length, abs=1e-4) == facts["length"]
    # the three components are a path from the origin to the tip of v, each along one axis
    path = [c["x_component"]["x_component"], c["y_component"]["y_component"], c["z_component"]["z_component"]]
    assert path[0]["from"] == [0, 0, 0] and path[-1]["to"] == v
    for a, b in zip(path, path[1:]):
        assert a["to"] == b["from"]
    for axis, seg in enumerate(path):
        d = [t - f for t, f in zip(seg["to"], seg["from"])]
        assert [i for i, x in enumerate(d) if x] == [axis] and d[axis] == seg["length"] == int(seg["label"])
    theta = math.degrees(math.acos(v[0] / length))
    ang = c["angle"]["theta"]
    assert ang["deg"] == pytest.approx(theta, abs=0.01) == facts["theta_deg"]
    assert ang["label_shows"] == "θ 56.1°"                    # the 3D view's angle label: 1 decimal
    assert c["length"]["length_squared"] == sum(x * x for x in v)
    assert c["cos_theta"]["cos_theta"] == pytest.approx(v[0] / length, abs=1e-4)


def test_cylinder_checks_agree():
    lec = g.lecture("cylinder")
    c = dict(_checks("cylinder"))
    for stage, check in c.items():
        if "V_factor" in check:                                      # V = pi r^2 h
            assert check["V_factor"] == check["r_factor"] ** 2 * check["h_factor"], stage
    assert (c["double_height"]["r_factor"], c["double_height"]["h_factor"], c["double_height"]["V_factor"]) == (1, 2, 2)
    # "if we double the radius instead": the height is h again, so the volume is x4, not x8
    assert (c["double_radius"]["r_factor"], c["double_radius"]["h_factor"], c["double_radius"]["V_factor"]) == (2, 1, 4)
    assert c["summary"] == {"V_factor_when_h_doubles": lec.facts["double_height"]["volume_factor"],
                            "V_factor_when_r_doubles": lec.facts["double_radius"]["volume_factor"]}
    d = lec.facts["drawing"]
    assert c["sides"]["drawn_h_over_r"] == pytest.approx(d["height_px"] / d["radius_px"], abs=0.01)
    assert d["bottom_y_px"] - d["twice_height_top_y_px"] == 2 * d["height_px"]   # 2h is drawn twice as tall
    assert d["twice_radius_px"] == 2 * d["radius_px"]


def test_saddle_checks_agree():
    c = dict(_checks("saddle"))

    def on_surface(p):
        return p[2] == pytest.approx(p[0] ** 2 - p[1] ** 2, abs=1e-9)

    ux, dy = c["along_x"]["curve_along_x"], c["along_y"]["curve_along_y"]
    assert all(on_surface(p) and p[1] == 0 for p in ux["points"]) and ux["d2z_dx2"] > 0 and ux["opens"] == "up"
    assert all(on_surface(p) and p[0] == 0 for p in dy["points"]) and dy["d2z_dy2"] < 0 and dy["opens"] == "down"
    s = c["saddle_shape"]["surface"]
    assert all(on_surface(p) for p in s["corners"]) and s["z_range"] == [-s["x_range"][1] ** 2, s["x_range"][1] ** 2]
    h = c["saddle_point"]["saddle_point"]
    assert h["point"] == [0, 0, 0] and h["gradient"] == [0, 0]
    assert h["hessian"] == [[ux["d2z_dx2"], 0], [0, dy["d2z_dy2"]]]
    assert h["hessian_det"] == h["hessian"][0][0] * h["hessian"][1][1] < 0   # a saddle: neither max nor min
    assert (c["why"]["d2z_dx2"], c["why"]["d2z_dy2"]) == (ux["d2z_dx2"], dy["d2z_dy2"])


# ---------- the truth is rewritten without the video only when the video still matches ----------
def test_truth_differs_ignores_words_but_not_the_timeline():
    pl = g.plan(g.lecture("saddle"))
    old = json.loads(json.dumps(g.truth_dict(g.lecture("saddle"), pl)))
    new = copy.deepcopy(old)
    new["stages"][3]["check"] = {"changed": True}
    new["stages"][3]["true"] = "said differently"
    new["facts"]["extra"] = 1
    assert g.truth_differs(old, new) == []
    late = copy.deepcopy(old)
    late["inks"]["para_x"]["t_start"] += 0.5
    assert g.truth_differs(old, late) == ["inks"]
    moved = copy.deepcopy(old)
    moved["stages"][2]["cues"][0]["t"] += 0.1
    assert g.truth_differs(old, moved) == ["stage axes.cues"]


def test_speak_cached_only_never_speaks(tmp_path):
    with pytest.raises(RuntimeError, match="--force"):
        g.speak(g.lecture("saddle"), tmp_path, cached_only=True)
    assert not (tmp_path / "tts_in.json").exists()


# ---------- the processed jobs: the piece to open is a drawing with a 3D view, and it is the figure ----------
@pytest.mark.parametrize("name", g.LECTURES)
def test_open_piece_in_the_processed_job(name):
    """data/jobs/synth3d-<name> (skipped when it is not there): the piece in OPEN_ON passes the 3D view's rule
    (can_popout, as the player's canPopout), opens its own drawing, is where the figure is, is talked about while the
    figure is drawn, and the hover point is on its ink and clear of other pieces' ink."""
    job = g.ROOT / "data" / "jobs" / f"synth3d-{name}"
    if not (job / "replay.json").is_file():
        pytest.skip("not processed")
    import cv2
    import numpy as np

    from app import popout

    replay = json.loads((job / "replay.json").read_text(encoding="utf-8"))
    truth = json.loads((g.OUT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    o = g.OPEN_ON[name]
    pieces = {p["id"]: p for p in replay["board"]["pieces"]}
    p = pieces[o["piece"]]
    assert replay["type"] == "board"
    assert popout.can_popout(p) and popout.is_figure(p)
    assert popout.drawing_id(replay, o["piece"]) == o["piece"]
    for q in o["same_scene"]:
        assert popout.can_popout(pieces[q]) and popout.drawing_id(replay, q) == o["piece"], q
    # it lies on the figure: its box, in video px, is mostly inside the truth's figure box
    bw, bh = replay["board"]["size"]
    hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]]),
                                     np.float32(replay["board"]["corners"]))
    x, y, w, h = p["box"]
    q = cv2.perspectiveTransform(np.float32([[[x, y]], [[x + w, y + h]]]), hm).reshape(-1, 2)
    fx, fy, fw, fh = truth["figure"]["box"]
    ix = max(0.0, min(q[1][0], fx + fw) - max(q[0][0], fx))
    iy = max(0.0, min(q[1][1], fy + fh) - max(q[0][1], fy))
    assert ix * iy >= 0.8 * (q[1][0] - q[0][0]) * (q[1][1] - q[0][1])
    # talked about while the figure is being drawn
    fig_stages = [s for s in truth["stages"] if any(truth["inks"][k]["figure"] for k in s["draws"])]
    assert any(sp["start"] < s["end"] and s["start"] < sp["end"] for sp in p["spoken"] for s in fig_stages)
    # the hover point is on its ink, and no other piece's ink is near it (the board's hit test picks it)
    hx, hy = round(o["hover"][0] * bw), round(o["hover"][1] * bh)
    own = cv2.imread(str(job / p["image"]), cv2.IMREAD_UNCHANGED)[..., 3]
    assert own[max(0, hy - y - 1):hy - y + 2, max(0, hx - x - 1):hx - x + 2].max() > 200
    for other in replay["board"]["pieces"]:
        if other["id"] == o["piece"]:
            continue
        ox, oy, ow, oh = other["box"]
        a = cv2.imread(str(job / other["image"]), cv2.IMREAD_UNCHANGED)[..., 3]
        ys, xs = np.nonzero(a > 200)
        if len(xs):
            assert np.hypot(xs + ox - hx, ys + oy - hy).min() > 12, other["id"]
