"""Pieces merged part way through an acceptance step (SPEC 5.1).

In one step the new ink's components are handled in turn. A component bridging two young pieces merges one into the
other; a small bit handled after it in the same step must join the piece that is left, not the one merged away. The
small-bits join reads piece ids from a dilation of the piece map made once per step, so it can still hold the old id:
on data/test3.mp4 that crashed capture at 4:47 (KeyError: 53) and failed the job.

The frames are made here and fed straight to capture_board (no video file): every pixel and every step is exact.
"""
from pathlib import Path

import numpy as np
import pytest

from app.core import board
from app.core.board import BoardParams, capture_board
from app.core.media import VideoInfo

W, H, FPS = 800, 200, 3.0
BOARD, INK = 220, 40
SECONDS = 10.0
INFO = VideoInfo(Path("synthetic.mp4"), SECONDS, W, H, FPS, False, 0)


def _box(x0, y0, x1, y1):
    """Ink rows y0..y1-1, columns x0..x1-1."""
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


# 3 px strokes. Gaps are chosen against the defaults at 800 px: drift reaches 2 px (k_drift), a small bit touches ink
# 4 px away (k_touch), components join within 12 px (k_group), pieces within 25 px (join_pad).
P0 = _box(100, 20, 161, 23)  # older writing, no longer recent when the step comes
A = _box(120, 25, 126, 28)  # 18 px, 3 px below P0: a small bit that joins P0 (and makes the step's touch map)
Y1 = _box(100, 80, 161, 83)  # two young strokes, 39 px apart: two pieces
Y2 = _box(200, 80, 261, 83)
B = _box(166, 80, 195, 83)  # between them, 5-6 px from each: bridges them, so Y1 is merged into Y2 (the newer)
C = _box(110, 85, 116, 88)  # 18 px, 3 px below Y1: a small bit touching ink of the piece just merged away

# When each mark appears (it is accepted accept_n samples later): P0 at 3.67 s (after the 2.5 s pre-window),
# Y1 at 6.67 s, Y2 at 7.0 s, and A, B and C together at 8.67 s, in that order (top to bottom).
APPEAR = [(3.0, P0), (6.0, Y1), (6.0 + 1 / FPS, Y2), (8.0, A), (8.0, B), (8.0, C)]


def _frame(t):
    f = np.full((H, W, 3), BOARD, np.uint8)
    for t0, m in APPEAR:
        if t >= t0 - 1e-6:
            f[m] = INK
    return f


@pytest.fixture
def synthetic(monkeypatch):
    def frames(info, fps, width, start=0.0, end=None, prefetch=8):
        for k in range(int(round((end - start) * fps))):
            t = start + k / fps
            yield t, _frame(t)

    monkeypatch.setattr(board, "iter_frames", frames)
    monkeypatch.setattr(board, "detect_mode", lambda info, geom, s0, s1, p: ("white", float(BOARD)))


def _piece_at(res, mask):
    """The piece ids the replay log gives the pixels of `mask`."""
    idx = np.flatnonzero(mask.reshape(-1))
    return set(np.unique(res.add_piece[np.isin(res.add_idx, idx)]).tolist())


def test_a_small_bit_touching_a_piece_merged_earlier_in_the_same_step_joins_the_merged_piece(synthetic):
    res = capture_board(INFO, params=BoardParams(width=W), mode="white")
    ids = {q["id"] for q in res.pieces}
    # Y1, the bridge B and the bit C are one piece with Y2, the one Y1 was merged into; A joined P0.
    assert _piece_at(res, Y1 | Y2 | B | C) == {3}
    assert _piece_at(res, P0 | A) == {1}
    assert ids == {1, 3}


def test_the_merge_and_the_join_happen_in_one_step(synthetic, monkeypatch, capsys):
    """The scene really is the one that crashed: Y1 is merged into Y2, and C joins, in the same step (debug log)."""
    monkeypatch.setenv("RB_DEBUG_BOARD", "1")
    capture_board(INFO, params=BoardParams(width=W), mode="white")
    log = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[board]")]
    step = [i for i, line in enumerate(log) if line.startswith("[board] t=8.67 ")]
    merged = log.index("[board] merged piece 2 into 3")
    join_c = next(i for i, line in enumerate(log) if line.startswith("[board] t=8.67 join 3 box=[110, 85, 116, 88]"))
    join_a = next(i for i, line in enumerate(log) if line.startswith("[board] t=8.67 join 1 box=[120, 25, 126, 28]"))
    assert step and step[0] <= join_a < merged < join_c <= step[-1]
