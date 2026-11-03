"""Too-thick ink (a filled dot, an arrowhead, where strokes cross) that stays put is writing;
a thick thing that moves is still something in front of the board (SPEC 5.1)."""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from app.core import media
from app.core.board import BoardParams, capture_board

W, H, FPS, SECONDS = 800, 450, 15, 15.0
DOT_STAYS = (200, 150)      # appears at 4.0 s (after the first 2.5 s, which count as already there), stays
DOT_ERASED = (500, 150)     # appears at 4.0 s, wiped at 12.5 s (lives longer than min_life)
R = 9                       # 18 px across: thicker than max_stroke (11), thinner than the 21 px closing
MOVER_Y = 330               # a thick dark blob crossing the board the whole time


def _frame(t: float) -> np.ndarray:
    f = np.full((H, W, 3), 232, np.uint8)
    if t >= 4.0:
        cv2.circle(f, DOT_STAYS, R, (30, 30, 30), -1, cv2.LINE_AA)
        if t < 12.5:
            cv2.circle(f, DOT_ERASED, R, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.circle(f, (int(60 + 45 * t), MOVER_Y), R, (40, 40, 40), -1, cv2.LINE_AA)
    return f


def _encode(path, frame):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)]
    frames = b"".join(frame(i / FPS).tobytes() for i in range(int(SECONDS * FPS)))
    subprocess.run(cmd, input=frames, check=True, capture_output=True, timeout=60)
    return media.probe(path)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return _encode(tmp_path_factory.mktemp("thick") / "dots.mp4", _frame)


def _disk(c, r=R - 2):
    m = np.zeros((H, W), bool)
    cv2.circle(m.view(np.uint8), c, r, 1, -1)
    return m


@pytest.fixture(scope="module")
def runs(clip):
    return {n: capture_board(clip, params=BoardParams(width=W, stay_n=n)) for n in (0, 3)}


def test_a_thick_dot_that_stays_put_is_writing(runs):
    res = runs[3]
    assert res.ink_at(SECONDS - 0.2)[_disk(DOT_STAYS)].mean() > 0.95
    dots = [p for p in res.pieces if p["box"][0] <= DOT_STAYS[0] <= p["box"][0] + p["box"][2]]
    assert len(dots) == 1
    # shown soon after it is written, and not as "written while hidden"
    assert 4.0 <= dots[0]["t_seen"] <= 6.0 and not dots[0]["revealed"] and not dots[0]["pre_existing"]


def test_without_the_rule_it_was_lost(runs):
    assert runs[0].ink_at(SECONDS - 0.2)[_disk(DOT_STAYS)].mean() < 0.1


def test_a_moving_thick_thing_is_never_writing(runs):
    band = np.zeros((H, W), bool)
    band[MOVER_Y - 2 * R: MOVER_Y + 2 * R] = True
    for t in np.arange(0.5, SECONDS, 0.5):
        assert not runs[3].ink_at(float(t))[band].any(), t


def test_an_erased_thick_dot_is_removed(runs):
    res = runs[3]
    assert res.ink_at(10.0)[_disk(DOT_ERASED)].mean() > 0.95
    assert not res.ink_at(SECONDS - 0.2)[_disk(DOT_ERASED)].any()


def _held_frame(t: float) -> np.ndarray:
    """A thin line written at 4 s while a thick blob darts to and fro below it until 9 s (never over it)."""
    f = np.full((H, W, 3), 232, np.uint8)
    if t >= 4.0:
        cv2.line(f, (200, 150), (320, 150), (30, 30, 30), 3, cv2.LINE_AA)
        if t < 9.0:
            x = 200 + abs((240 * (t - 4.0)) % 240 - 120)  # 80 px per 1/3 s: never stays put
            cv2.circle(f, (int(x), 260), R, (40, 40, 40), -1, cv2.LINE_AA)
    return f


def test_ink_held_back_only_by_the_thick_mask_is_not_written_while_hidden(tmp_path):
    # A huge thick-mask reach makes the moving blob hold the line back until it leaves at 9 s.
    clip = _encode(tmp_path / "held.mp4", _held_frame)
    res = capture_board(clip, params=BoardParams(width=W, thick_dilate=361))
    lines = [p for p in res.pieces if p["box"][1] <= 150 <= p["box"][1] + p["box"][3]]
    assert len(lines) == 1
    p = lines[0]
    # It is accepted only after the blob leaves, but it was in view and stayed put the whole time: the replay
    # shows it from when thin ink would have shown (accept_n samples after it was written)...
    assert 4.0 <= p["t_seen"] <= 5.5
    line = np.zeros((H, W), bool)
    line[149:152, 205:316] = True
    assert res.ink_at(5.5)[line].mean() > 0.95
    # ...and not as written while hidden.
    assert not p["revealed"]


BODY = (560, 90, 700, 420)   # x0, y0, x1, y1: the teacher, in the person mask
WRIST = ((430, 250), (560, 250))  # a still 14 px bar joined to the body that SAM missed
FREE_DOT = (200, 150)        # thick writing far from the teacher, as a control


def _wrist_frame(t: float) -> np.ndarray:
    f = np.full((H, W, 3), 232, np.uint8)
    x0, y0, x1, y1 = BODY
    cv2.rectangle(f, (x0, y0), (x1, y1), (70, 60, 50), -1)
    if 3.0 <= t < 13.0:
        cv2.line(f, *WRIST, (60, 60, 60), 14, cv2.LINE_AA)
    if t >= 4.0:
        cv2.circle(f, FREE_DOT, R, (30, 30, 30), -1, cv2.LINE_AA)
    return f


def _body_mask(t: float) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = BODY
    m[y0:y1 + 1, x0:x1 + 1] = 1
    return m.astype(bool)


@pytest.fixture(scope="module")
def wrist_runs(tmp_path_factory):
    clip = _encode(tmp_path_factory.mktemp("wrist") / "wrist.mp4", _wrist_frame)
    return {c: capture_board(clip, person_mask=_body_mask, params=BoardParams(width=W, stay_clear=c)) for c in (0.0, 6.0)}


def test_a_still_thick_thing_joined_to_the_teacher_is_not_writing(wrist_runs):
    band = np.zeros((H, W), bool)
    band[250 - 12: 250 + 13, WRIST[0][0] - 5: BODY[0]] = True
    res = wrist_runs[6.0]
    for t in np.arange(3.0, SECONDS, 1.0 / 3):
        assert not res.ink_at(float(t))[band].any(), t
    # thick writing away from the teacher still counts
    assert res.ink_at(SECONDS - 0.2)[_disk(FREE_DOT)].mean() > 0.95
    # without the gate the wrist was taken for writing
    assert wrist_runs[0.0].ink_at(12.5)[band].sum() > 200


def _jumpy_frame(t: float) -> np.ndarray:
    """A thick dot written at 4 s while the camera's exposure jumps 15 grey levels every 2/3 s."""
    f = np.full((H, W, 3), 225, np.int16)
    if t >= 4.0:
        cv2.circle(f, DOT_STAYS, R, (30, 30, 30), -1, cv2.LINE_AA)
        if int(t * 1.5) % 2:
            f += 15
    return np.clip(f, 0, 255).astype(np.uint8)


def test_exposure_jumps_do_not_stop_thick_ink_settling(tmp_path):
    clip = _encode(tmp_path / "jumpy.mp4", _jumpy_frame)
    res = capture_board(clip, params=BoardParams(width=W))
    assert res.ink_at(SECONDS - 0.2)[_disk(DOT_STAYS)].mean() > 0.95
    dots = [p for p in res.pieces if p["box"][0] <= DOT_STAYS[0] <= p["box"][0] + p["box"][2]]
    assert len(dots) == 1 and 4.0 <= dots[0]["t_seen"] <= 6.0 and not dots[0]["revealed"]
