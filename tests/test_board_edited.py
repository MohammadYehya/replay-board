"""Board capture on real-world chalkboard and edited videos (SPEC 5.1).

- On a chalkboard the grey image is 255 - max(B, G, R): a dark shirt is brighter than the board there. The "large dark
  things" fallback (for what SAM missed) keeps a running maximum of the board as its reference, so a shirt passing
  lifted it, and the clean board behind him counted as "in front" until the reference sank back (half a minute and
  more on data/test4.mp4): writing showed late, tagged "written while hidden". Where SAM looked, it is not used.
- Edited videos cut to title and end cards: a sample that does not show the board is skipped, so a card is never
  writing, never erases the board and never gives the board its colour.

The frames are made here and fed straight to capture_board (no video file), like test_board_merge.
"""
from pathlib import Path

import numpy as np
import pytest

from app.core import board
from app.core.board import BoardParams, capture_board
from app.core.media import VideoInfo
from app.core.sam import PersonTrack

W, H, FPS = 800, 300, 3.0


def _box(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _feed(monkeypatch, frame, seconds, mode, typical):
    def frames(info, fps, width, start=0.0, end=None, prefetch=8):
        for k in range(int(round((end - start) * fps))):
            t = start + k / fps
            yield t, frame(t)

    monkeypatch.setattr(board, "iter_frames", frames)
    monkeypatch.setattr(board, "detect_mode", lambda info, geom, s0, s1, p: (mode, float(typical)))
    return VideoInfo(Path("synthetic.mp4"), seconds, W, H, FPS, False, 0)


def _first_shown(res, mask):
    sel = np.isin(res.add_idx, np.flatnonzero(mask.reshape(-1)))
    return res.add_t[sel]


# --- a dark shirt passing in front of a chalkboard ------------------------------------------------------------------
CHALK_BOARD = (60, 110, 70)  # BGR: chalk grey 145, gray ~92 (a chalkboard)
SHIRT = (12, 12, 12)  # chalk grey 243: ~100 above the board
CHALK = (215, 215, 215)
TEACHER = _box(300, 40, 520, 260)  # in front 2-5 s, in SAM's mask
STROKE = _box(350, 140, 470, 143)  # written at 6 s where he stood, after he left
SHIRT_SECONDS = 20.0


def _shirt_frame(t):
    f = np.full((H, W, 3), CHALK_BOARD, np.uint8)
    if t >= 6.0 - 1e-6:
        f[STROKE] = CHALK
    if 2.0 - 1e-6 <= t < 5.0 - 1e-6:
        f[TEACHER] = SHIRT
    return f


def _sam(t):
    return TEACHER.copy() if 2.0 - 1e-6 <= t < 5.0 - 1e-6 else None


def test_writing_after_a_dark_shirt_passed_a_chalkboard_shows_when_written(monkeypatch):
    info = _feed(monkeypatch, _shirt_frame, SHIRT_SECONDS, "chalk", 92)
    res = capture_board(info, person_mask=_sam, params=BoardParams(width=W), mode="chalk")
    t = _first_shown(res, STROKE)
    assert t.size > 200 and t.max() <= 6.0 + 1.5, (t.size, t.max() if t.size else None)
    p = [q for q in res.pieces if q["box"][1] <= 141 <= q["box"][1] + q["box"][3]]
    assert len(p) == 1 and p[0]["t_seen"] <= 7.5 and not p[0]["revealed"], p


class _NotLooked:
    """A person track whose SAM chunk failed everywhere: no masks, and it says so."""

    def __call__(self, t):
        return None

    def looked(self, t):
        return False


def test_where_sam_did_not_look_the_chalk_fallback_still_runs(monkeypatch):
    # The same scene with SAM's chunk failed: the reference is used as before, so the shirt still blinds the spot.
    info = _feed(monkeypatch, _shirt_frame, SHIRT_SECONDS, "chalk", 92)
    res = capture_board(info, person_mask=_NotLooked(), params=BoardParams(width=W), mode="chalk")
    t = _first_shown(res, STROKE)
    assert t.size == 0 or t.min() > 6.0 + 5.0


def test_person_track_knows_where_sam_did_not_look():
    tr = PersonTrack(start=10.0, fps=2.0, size=(4, 4))
    tr.chunks = [(0, 241, None), (240, 300, None)]  # samples of two 120 s / 30 s chunks
    tr.failed_chunks = {1}
    assert tr.looked(10.0) and tr.looked(129.0)
    assert not tr.looked(131.0) and not tr.looked(150.0)
    assert tr.looked(500.0)  # outside every chunk: nothing to say


# --- title and end cards in an edited video -------------------------------------------------------------------------
WHITE_BOARD, INK = (232, 232, 232), (40, 40, 40)
WRITING = _box(100, 100, 300, 104)  # on the board from 3 s
CARD_TEXT = _box(250, 140, 550, 160)  # the card's big text


def _card_frame(board_bgr, ink, card_bgr, card_ink):
    def frame(t):
        on_card = 8.0 - 1e-6 <= t < 10.0 - 1e-6 or t >= 13.0 - 1e-6  # a card mid-clip, and the end card
        f = np.full((H, W, 3), card_bgr if on_card else board_bgr, np.uint8)
        if on_card:
            f[CARD_TEXT] = card_ink
        elif t >= 3.0 - 1e-6:
            f[WRITING] = ink
        return f
    return frame


CARDS = {
    # board, ink, card, card text, mode, typical gray
    "whiteboard-black-card": (WHITE_BOARD, INK, (8, 8, 8), (240, 240, 240), "white", 232),
    "chalkboard-black-card": (CHALK_BOARD, CHALK, (8, 8, 8), (240, 240, 240), "chalk", 92),
    "chalkboard-white-card": (CHALK_BOARD, CHALK, (250, 250, 250), (30, 30, 200), "chalk", 92),
}


@pytest.mark.parametrize("case", list(CARDS))
def test_title_and_end_cards_are_not_the_board(monkeypatch, case):
    board_bgr, ink, card, card_ink, mode, typical = CARDS[case]
    info = _feed(monkeypatch, _card_frame(board_bgr, ink, card, card_ink), 16.0, mode, typical)
    res = capture_board(info, params=BoardParams(width=W), mode=mode)
    ys, xs = np.divmod(res.add_idx, W)
    assert not CARD_TEXT[ys, xs].any()  # the card's text is not writing
    t = _first_shown(res, WRITING)
    assert t.size > 300 and t.max() <= 3.0 + 1.5
    p = [q for q in res.pieces if q["box"][1] <= 102 <= q["box"][1] + q["box"][3]]
    assert len(p) == 1 and p[0]["removed_at"] is None, p  # neither card erased the board
    assert res.rem_t.size == 0
    assert max(abs(a - b) for a, b in zip(res.background, board_bgr)) <= 6, res.background  # the board's colour
    assert res.stats["skipped"] == 6 + 9  # 8-10 s and 13-16 s at 3 a second


def test_a_normal_clip_skips_nothing(monkeypatch):
    board_bgr, ink, *_ = CARDS["whiteboard-black-card"]

    def frame(t):
        f = np.full((H, W, 3), board_bgr, np.uint8)
        if t >= 3.0 - 1e-6:
            f[WRITING] = ink
        if 5.0 - 1e-6 <= t < 9.0 - 1e-6:
            f[TEACHER] = (70, 60, 50)  # the teacher, half the board's width, in SAM's mask
        return f

    info = _feed(monkeypatch, frame, 12.0, "white", 232)
    res = capture_board(info, person_mask=lambda t: TEACHER.copy() if 5.0 - 1e-6 <= t < 9.0 - 1e-6 else None,
                        params=BoardParams(width=W), mode="white")
    assert res.stats["skipped"] == 0 and res.stats["steps"] == 36
