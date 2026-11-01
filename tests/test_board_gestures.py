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


def setup(pieces, adds=()):
    geom = BoardGeometry(INFO, CORNERS, 800, 1280)
    W, H = geom.width, geom.height
    add_t = np.array([a[0] for a in adds], float)
    add_idx = np.array([int(a[2]) * W + int(a[1]) for a in adds], np.int64)
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
