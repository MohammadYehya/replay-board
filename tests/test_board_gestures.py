"""Pointing on a filmed board: which hand samples count, and how they group into gestures."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.core import board_gestures as bg
from app.core.board import BoardGeometry
from app.core.media import VideoInfo
from app.core.sam import HandTrack

INFO = VideoInfo(Path("x.mp4"), 60.0, 1280, 720, 25.0, True, 0)
CORNERS = [[0, 0], [1279, 0], [1279, 719], [0, 719]]  # the whole frame is the board


def setup(pieces, adds=(), geom=None):
    geom = geom or BoardGeometry(INFO, CORNERS, 800, 1280)
    W, H = geom.width, geom.height
    add_t = np.array([a[0] for a in adds], float)
    add_idx = np.array([int(round(a[2])) * W + int(round(a[1])) for a in adds], np.int64)
    board = SimpleNamespace(width=W, height=H, pieces=pieces, add_t=add_t, add_idx=add_idx)
    body = np.zeros((216, 384), bool)
    body[60:216, 40:80] = True  # a person standing at the left (x ~ 130-270 video px)
    person = lambda t: body  # noqa: E731
    return geom, board, person


def hand_at(track, i, x, y):
    local = np.zeros((10, 10), bool)
    local[4:6, 4:6] = True
    track.samples.setdefault(i, []).append({"box": [x - 10, y - 10, x + 10, y + 10], "local": local, "scale": 2.0,
                                            "score": 0.8})


def piece(i, box, t_seen=1.0, written_after=0.5, t_shown=2.0, pre=False):
    return {"id": i, "box": box, "t_seen": t_seen, "written_after": written_after, "t_shown": t_shown,
            "removed_at": None, "pre_existing": pre}


def test_extended_hand_on_writing_points_but_not_near_the_body():
    geom, board, person = setup([piece(1, [450, 150, 80, 30]), piece(2, [80, 150, 60, 30])])
    hands = HandTrack(0.0, 3.0)
    for i in range(30, 33):
        hand_at(hands, i, 785, 290)  # far from the body, on piece 1
    for i in range(40, 43):
        hand_at(hands, i, 185, 290)  # right next to the body, over piece 2
    s = bg.pointing_samples(hands, person, board, geom, INFO)
    assert [round(x[0], 2) for x in s] == [10.0, 10.33, 10.67]
    assert all(440 <= x[1] <= 540 and 140 <= x[2] <= 190 for x in s)


def test_writing_is_not_pointing():
    # piece 1 is being written at 10 s; new ink appears next to the hand at 14 s near piece 3
    geom, board, person = setup([piece(1, [450, 150, 80, 30], t_seen=10.5, written_after=9.5, t_shown=11.0),
                                 piece(3, [450, 250, 80, 30])],
                                adds=[(14.0, 470 + k, 262) for k in range(20)])
    hands = HandTrack(0.0, 3.0)
    for i in (30, 31):
        hand_at(hands, i, 785, 290)  # on piece 1 while it is written
    for i in (36, 37):
        hand_at(hands, i, 785, 470)  # on piece 3, where ink appears soon after
    assert bg.pointing_samples(hands, person, board, geom, INFO) == []


def test_the_other_stretched_hand_can_be_the_one_pointing():
    geom, board, person = setup([piece(1, [450, 150, 80, 30]), piece(3, [660, 360, 60, 30])])
    hands = HandTrack(0.0, 3.0)
    for i in (30, 31):
        hand_at(hands, i, 1100, 450)  # stretched out most, over empty board
        hand_at(hands, i, 785, 290)  # the other hand, on piece 1
    for i in (40, 41):
        hand_at(hands, i, 785, 290)  # both on writing: the most stretched-out one counts
        hand_at(hands, i, 1100, 600)  # on piece 3
    s = bg.pointing_samples(hands, person, board, geom, INFO)
    assert [round(x[0], 2) for x in s] == [10.0, 10.33, 13.33, 13.67]
    assert all(440 <= x[1] <= 540 and 140 <= x[2] <= 190 for x in s[:2])
    assert all(650 <= x[1] <= 730 and 350 <= x[2] <= 400 for x in s[2:])


def scene(width):
    """One physical scene on a board flattened to `width` px (sizes below are for 800 px)."""
    k = width / 800
    geom = BoardGeometry(INFO, CORNERS, width, 1280)
    g800 = BoardGeometry(INFO, CORNERS, 800, 1280)
    pieces = [piece(1, [450 * k, 150 * k, 80 * k, 30 * k]), piece(2, [450 * k, 300 * k, 80 * k, 30 * k])]
    # 8 px of new ink near piece 2 at 800 px (too little to be writing): 4x as many pixels at twice the width
    adds = [(21.0, 485 * k + dx, 318 * k + dy) for dx in range(int(4 * k)) for dy in range(int(2 * k))]
    _, board, person = setup(pieces, adds, geom)
    hands = HandTrack(0.0, 3.0)
    # 16 px right of piece 1, rocking up and down 18 px (a hover); then to and fro 60 px across piece 2
    for i, (x, y) in enumerate([(546, 155), (546, 173), (546, 160), (546, 168)]):
        vx, vy = g800.board_to_video([[x, y]])[0]
        hand_at(hands, 30 + i, vx, vy + 2)
    for i, x in enumerate([455, 515, 455, 515]):
        vx, vy = g800.board_to_video([[x, 315]])[0]
        hand_at(hands, 60 + i, vx, vy + 2)
    return bg.pointing_samples(hands, person, board, geom, INFO)


def test_sizes_scale_with_the_board_width():
    s1, s2 = scene(800), scene(1600)
    assert [x[0] for x in s1] == [x[0] for x in s2] and len(s1) == 8
    assert all(abs(b[1] - 2 * a[1]) < 3 and abs(b[2] - 2 * a[2]) < 3 and a[3] == b[3] for a, b in zip(s1, s2))
    e1, e2 = bg.episodes(s1, 3.0), bg.episodes(s2, 3.0, scale=2.0)
    assert [e["type"] for e in e1] == [e["type"] for e in e2] == ["hover", "move"]
    assert [(e["t_start"], e["t_end"]) for e in e1] == [(e["t_start"], e["t_end"]) for e in e2]


def test_hand_off_writing_or_off_board_is_ignored():
    geom, board, person = setup([piece(1, [450, 150, 80, 30])])
    hands = HandTrack(0.0, 3.0)
    hand_at(hands, 30, 1100, 600)  # empty board
    hand_at(hands, 31, 2000, 290)  # outside the board
    assert bg.pointing_samples(hands, person, board, geom, INFO) == []


def test_episodes_split_on_gaps_and_guess_the_shape():
    rest = [(10.0 + i / 3, 300.0, 200.0, 0.5) for i in range(4)]
    sweep = [(20.0 + i / 3, 300.0 + 40 * i, 210.0, 0.5) for i in range(5)]
    lone = [(30.0, 100.0, 100.0, 0.5)]
    eps = bg.episodes(rest + sweep + lone, 3.0)
    assert [e["type"] for e in eps] == ["hover", "sweep"]
    assert eps[1]["from"] == [300.0, 210.0] and eps[1]["to"] == [460.0, 210.0]


def test_video_to_board_inverts_board_to_video():
    geom = BoardGeometry(INFO, [[100, 50], [1180, 60], [1170, 660], [110, 650]], 800, 1280)
    pts = np.array([[10.0, 20.0], [400.0, 300.0], [790.0, 500.0]])
    back = bg.video_to_board(geom, geom.board_to_video(pts))
    assert np.abs(back - pts).max() < 0.5



def test_zoom_region_is_the_target_around_the_path():
    assert bg.zoom_region([100, 50, 400, 200], [[300, 120], [340, 130]], 800, 315) == [220, 50, 201, 161]
    # a path beside the target, not over it: the whole target
    assert bg.zoom_region([100, 50, 40, 20], [[600, 300]], 800, 315) == [100, 50, 40, 20]


def test_tighten_fits_the_ink_inside_a_rough_box():
    ink = np.zeros((100, 200), bool)
    ink[40:52, 60:120] = True
    ink[70:80, 60:120] = True  # the next line down
    assert bg.tighten(ink, [58, 37, 70, 18]) == [60, 40, 60, 12]
    assert bg.tighten(ink, [150, 10, 20, 10]) is None


def test_union_box():
    assert bg.union_box([[10, 10, 20, 5], [5, 30, 10, 10]]) == [5, 10, 25, 30]


def test_zoom_region_and_tighten_scale_with_the_board():
    assert bg.zoom_region([200, 100, 800, 400], [[600, 240], [680, 260]], 1600, 630) == [440, 100, 401, 321]
    ink = np.zeros((200, 400), bool)
    ink[40:52, 60:120] = True
    assert bg.tighten(ink, [66, 46, 70, 18]) == [62, 42, 58, 10]  # pad 4
    assert bg.tighten(ink, [66, 46, 70, 18], scale=2.0) == [60, 40, 60, 12]  # pad 8


def test_clean_target_keeps_known_ids_once_in_order():
    ids = ["1", "2", "3"]
    assert bg.clean_target("spotlight", ["2", "2", "9"], ids) == ("spotlight", ["2"])
    assert bg.clean_target("circle", [3, "1"], ids) == ("circle", ["3", "1"])
    assert bg.clean_target("arrow", ["1", "1", "3", "2"], ids) == ("arrow", ["1", "3"])
    assert bg.clean_target("arrow", ["3", "3", "7"], ids) == ("spotlight", ["3"])
    assert bg.clean_target("pulse", None, ids) == ("pulse", [])


def test_box_distance():
    assert bg.box_distance([15, 15], [10, 10, 20, 20]) == 0
    assert bg.box_distance([40, 15], [10, 10, 20, 20]) == 10
    assert bg.box_distance([33, 34], [10, 10, 20, 20]) == 5


def test_arrow_must_follow_the_hand():
    # test-full at 389.33 s: Spark said from 29 to 16, but the hand only moved on 16 (piece 29 is 250 px away)
    path = [[407.6, 230.4], [448.5, 219.2]]
    boxes = [[139, 200, 16, 46], [455, 201, 104, 69]]
    assert bg.check_arrow(path, ["29", "16"], boxes) == ("spotlight", ["16"])
    # the same at twice the width
    path2 = [[2 * x, 2 * y] for x, y in path]
    boxes2 = [[2 * v for v in b] for b in boxes]
    assert bg.check_arrow(path2, ["29", "16"], boxes2, scale=2.0) == ("spotlight", ["16"])
    # real arrows: kept, or turned round when the hand ran the other way
    a, b = [586, 61, 65, 16], [549, 135, 116, 36]
    path = [[617.0, 83.7], [651.3, 149.8]]
    assert bg.check_arrow(path, ["43", "54"], [a, b]) == ("arrow", ["43", "54"])
    assert bg.check_arrow(path, ["54", "43"], [b, a]) == ("arrow", ["43", "54"])
    # side by side, both ways are within reach: the way the hand went
    a, b = [601, 216, 63, 19], [606, 183, 80, 25]
    path = [[600.7, 242.9], [629.2, 185.3]]
    assert bg.check_arrow(path, ["60", "59"], [a, b]) == ("arrow", ["60", "59"])
    assert bg.check_arrow(path, ["59", "60"], [b, a]) == ("arrow", ["60", "59"])
    # an end 50 px from its box: within reach at 800 px; at 1600 px it is 100 px, still within (scaled) reach
    assert bg.check_arrow([[0, 50], [300, 50]], ["1", "2"], [[50, 40, 20, 20], [350, 40, 20, 20]]) == ("arrow", ["1", "2"])
    assert bg.check_arrow([[0, 100], [600, 100]], ["1", "2"], [[100, 80, 40, 40], [700, 80, 40, 40]],
                          scale=2.0) == ("arrow", ["1", "2"])
    assert bg.check_arrow([[0, 100], [600, 100]], ["1", "2"], [[100, 80, 40, 40], [700, 80, 40, 40]])[0] == "spotlight"


class FakeClient:
    def __init__(self, answer):
        self.answer, self.prompts, self.images = answer, [], []

    def ask(self, prompt, images, schema=None, reasoning=None):
        self.prompts.append(prompt)
        self.images.append(images)
        return dict(self.answer)


def test_board_gesture_part_maps_the_zoomed_box_back_to_the_board():
    from app.core import understand
    board_img = np.full((315, 800, 3), 255, np.uint8)
    region = [300, 100, 240, 120]  # zoomed x4 to 960x480
    c = FakeClient({"whole": False, "box_2d": [250, 500, 750, 750]})
    got = understand.board_gesture_part(c, board_img, region, [[350, 160], [450, 170]],
                                        {"type": "underline", "why": "x"}, "y = 2 ln|x|", "this one")
    assert [round(v) for v in got] == [360, 160, 120, 30]
    assert "y = 2 ln|x|" in c.prompts[0] and "this one" in c.prompts[0]
    c = FakeClient({"whole": True, "box_2d": []})
    assert understand.board_gesture_part(c, board_img, region, [[350, 160]], {"type": "spotlight"}, "", "") is None
