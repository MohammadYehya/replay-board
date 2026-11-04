"""Written while hidden ("revealed", SPEC 5.1): only writing that really stayed out of view after it was written.

The writing hand, arm or body covers a spot before the pen gets there; that says nothing about when the ink was written,
and a redraw at the time it was talked about would show it before the teacher wrote it. Also: the rest of a stroke
accepted a sample after the arrowhead beside it is writing, not camera drift, so the piece's image keeps it.
"""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from app.core import media
from app.core.board import BoardParams, capture_board

W, H, FPS, SECONDS = 800, 450, 15, 15.0
BOARD, INK, SHIRT = 232, (30, 30, 30), (70, 60, 50)


def _encode(path, frame):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)]
    frames = b"".join(frame(i / FPS).tobytes() for i in range(int(SECONDS * FPS)))
    subprocess.run(cmd, input=frames, check=True, capture_output=True, timeout=60)
    return media.probe(path)


def _lecture(tmp_path, name, ink_at, front_at):
    """A clip where `ink_at(t)` is the ink written by t (bool H x W) and `front_at(t)` the teacher (bool H x W,
    also the person mask). Returns the capture."""
    def frame(t):
        f = np.full((H, W, 3), BOARD, np.uint8)
        f[ink_at(t)] = INK
        f[front_at(t)] = SHIRT
        return f
    clip = _encode(tmp_path / f"{name}.mp4", frame)
    # (the teacher fills much of the middle, which would make the board look dark: say it is a whiteboard)
    return capture_board(clip, person_mask=front_at, params=BoardParams(width=W), mode="white")


def _rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[max(0, y0):y1, max(0, x0):x1] = True
    return m


LINE_Y = 250


def _line(x0, x1, y=LINE_Y):
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (x0, y), (x1, y), 255, 3, cv2.LINE_AA)
    return m > 127


def _on_line(p, x0, x1):
    x, y, w, h = p["box"]
    return y <= LINE_Y <= y + h and x < x1 and x + w > x0


def test_writing_seen_as_it_is_written_after_the_teacher_stood_there_is_not_revealed(tmp_path):
    # The teacher stands on the spot from 2 s, then writes a line left to right (150 -> 450 px, 6-9 s) with his body
    # just right of the pen: each bit of ink is in view right after the pen leaves it, but its spot was out of view for
    # up to 4 s before it was written.
    full = _line(150, 450)
    xs = np.arange(W)

    def pen(t):
        return 150 + 100 * min(max(t - 6.0, 0.0), 3.0)

    def ink_at(t):
        return full & (xs[None, :] <= pen(t)) if t >= 6.0 else np.zeros((H, W), bool)

    def front_at(t):
        if t < 2.0:
            return _rect(620, 80, 760, 380)
        x = pen(t)
        return _rect(int(x) + 30, 80, int(x) + 170, 380) | _rect(int(x) - 8, LINE_Y - 15, int(x) + 31, LINE_Y + 16)

    res = _lecture(tmp_path, "precover", ink_at, front_at)
    ps = [p for p in res.pieces if _on_line(p, 150, 450)]
    assert ps and not any(p["revealed"] for p in ps), ps
    assert all(p["redraw_after"] is None for p in ps)
    assert min(p["t_seen"] for p in ps) <= 8.0  # shown as it was written


def test_writing_done_behind_the_teacher_is_revealed_and_never_redrawn_before_its_spot_was_last_seen_blank(tmp_path):
    # The teacher stands on the spot from 2 s to 12.5 s and writes there at 4-6 s, out of view.
    full = _line(180, 310)

    def ink_at(t):
        return full if t >= 5.0 else np.zeros((H, W), bool)

    def front_at(t):
        return _rect(150, 80, 340, 380) if 2.0 <= t < 12.5 else _rect(640, 80, 780, 380)

    res = _lecture(tmp_path, "hidden", ink_at, front_at)
    ps = [p for p in res.pieces if _on_line(p, 180, 310)]
    assert len(ps) == 1 and ps[0]["revealed"], ps
    p = ps[0]
    assert 12.5 <= p["t_seen"] <= 14.0
    assert p["written_after"] <= p["redraw_after"] <= 2.0  # last seen blank just before the teacher came


def test_writing_seen_as_it_is_written_after_a_long_stand_on_the_spot_is_not_revealed(tmp_path):
    # The teacher stands on the blank spot 1-11 s (talking), then writes a line there left to right (11-14 s) with his
    # body right against the pen: each bit is in view as soon as it is written, but its spot was out of view for 10-13 s
    # first. It came into view bit by bit as the pen moved on (not in one go, as when he steps away from hidden
    # writing), so it was not written while hidden; flagged, a redraw could put it up to 10 s before the pen got there.
    full = _line(150, 300)
    xs = np.arange(W)

    def pen(t):
        return 150 + 50 * min(max(t - 11.0, 0.0), 3.0)

    def ink_at(t):
        return full & (xs[None, :] <= pen(t)) if t >= 11.0 else np.zeros((H, W), bool)

    def front_at(t):
        if t < 1.0:
            return _rect(620, 80, 760, 380)
        if t < 11.0:
            return _rect(140, 150, 320, 380)
        x = int(pen(t))
        return _rect(x + 4, 150, x + 184, 380)

    res = _lecture(tmp_path, "stand_then_write", ink_at, front_at)
    ps = [p for p in res.pieces if _on_line(p, 150, 300)]
    assert ps and not any(p["revealed"] for p in ps), ps
    assert min(p["t_seen"] for p in ps) <= 12.5  # shown as it was written


def test_ink_written_in_view_that_joins_hidden_writing_is_not_redrawn_early(tmp_path):
    # A line written behind the teacher (on the spot 2-12.5 s, writing at 5 s), then, once he has stepped away, more of
    # the line written in view at 13-13.4 s: one piece, written while hidden. Only the hidden part may be redrawn before
    # the camera shows it (from when the link says it was written); the part seen as it was written is drawn when the
    # replay shows it, or the board would show it while the video still shows its spot blank.
    hidden, more = _line(180, 310), _line(316, 344)
    xs = np.arange(W)

    def ink_at(t):
        m = hidden if t >= 5.0 else np.zeros((H, W), bool)
        return m | (more & (xs[None, :] <= 316 + 70 * (t - 13.0))) if t >= 13.0 else m

    def front_at(t):
        if 2.0 <= t < 12.5:
            return _rect(150, 80, 360, 380)
        if 12.9 <= t < 13.4:
            x = int(316 + 70 * (t - 13.0))
            return _rect(x + 4, 150, x + 140, 380)
        return _rect(640, 80, 780, 380)

    res = _lecture(tmp_path, "hidden_then_more", ink_at, front_at)
    ps = [p for p in res.pieces if _on_line(p, 180, 344)]
    assert len(ps) == 1 and ps[0]["revealed"], ps
    p = ps[0]
    part = res.redraw_part(p["id"])
    to = part["redraw_to"]
    assert 0 < to < 255, to
    _, order, (x, y, w, h), _, _ = res.piece_image(p["id"])
    o = np.zeros((H, W), np.int64)
    o[y:y + h, x:x + w] = order
    on_more = more & (o > 0)
    on_hidden = hidden & ~(cv2.dilate(more.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (o > 0)
    assert on_more.sum() > 20 and (o[on_more] > to).all(), (to, np.sort(o[on_more])[:5])  # drawn when due
    assert (o[on_hidden] <= to).mean() >= 0.95, (o[on_hidden] <= to).mean()  # redrawn early
    # every pixel it may redraw early had its spot last seen blank before the redraw can start
    floor = min(p["t_seen"], p["redraw_after"])
    assert floor <= 2.5
    early = (o > 0) & (o <= to)
    blank = np.full(H * W, -np.inf)
    js = np.nonzero(res.add_piece == p["id"])[0]
    np.maximum.at(blank, res.add_idx[js], res.add_blank[js])
    assert (blank.reshape(H, W)[early] <= floor + 0.01).all()
    # ...and it is redrawn as long as writing that much ink takes (board.WRITE_RATE pixels a second)
    from app.core.board import WRITE_RATE
    assert part["redraw_secs"] == pytest.approx(early.sum() / WRITE_RATE, abs=0.01)
    # other pieces: nothing to cut
    assert all(res.redraw_part(q["id"]) is None for q in res.pieces if not q["revealed"])


def test_counting_on_a_4x1_slice_does_not_crash():
    # cv2.add(a, 1) on a 4x1 uint8 array fails ("input arrays ... different types": OpenCV reads it as a 4-value
    # scalar); the box capture counts in shrinks to whatever ink is still counting, so a real clip can hit it.
    from app.core import board

    for shape in [(4, 1), (1, 4), (3, 1), (5, 1), (4, 2), (1, 1), (6, 800)]:
        a = np.zeros(shape, np.uint8)
        a.flat[0] = 255
        out = board._inc(a)
        assert out.shape == shape and out.dtype == np.uint8
        assert out.flat[0] == 255 and (out.size == 1 or out.flat[-1] == 1)
    big = np.zeros((40, 30), np.uint8)
    assert (board._inc(big[3:7, 5:6]) == 1).all()  # a 4x1 view of a bigger array
    # ...and no counter in capture adds a bare scalar with OpenCV any more
    import inspect
    import re

    assert not re.search(r"cv2\.add\([^()]*,\s*\d+\s*\)", inspect.getsource(board))


def test_writing_behind_the_teacher_since_the_clip_started_is_not_revealed(tmp_path):
    # The line is on the board from the start, behind the teacher until 8 s: nothing says it was written in the clip.
    full = _line(180, 310)

    def front_at(t):
        return _rect(150, 80, 340, 380) if t < 8.0 else _rect(640, 80, 780, 380)

    res = _lecture(tmp_path, "from_start", lambda t: full, front_at)
    ps = [p for p in res.pieces if _on_line(p, 180, 310)]
    assert ps and not any(p["revealed"] for p in ps), ps


def test_the_shaft_beside_an_arrowhead_is_in_the_pieces_image(tmp_path):
    # A thin shaft written up to a filled arrowhead (thick ink): the thick mask holds the shaft next to the head until
    # the head has stayed put, so that part is accepted a sample after the head. It is writing, not camera drift.
    tail, tip = np.array([150.0, 330.0]), np.array([350.0, 190.0])
    d = (tip - tail) / np.linalg.norm(tip - tail)
    n = np.array([-d[1], d[0]])
    head = np.round(np.array([tip, tip - 26 * d + 12 * n, tip - 26 * d - 12 * n]) * 16).astype(np.int32)
    shaft_end = tip - 20 * d

    def shaft_upto(u):
        m = np.zeros((H, W), np.uint8)
        e = tail + u * (shaft_end - tail)
        cv2.line(m, tuple(np.round(tail * 16).astype(int)), tuple(np.round(e * 16).astype(int)), 255, 4, cv2.LINE_AA, 4)
        return m > 127

    hm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(hm, [head], 255, cv2.LINE_AA, 4)
    head_m = hm > 127

    def ink_at(t):
        if t < 4.0:
            return np.zeros((H, W), bool)
        m = shaft_upto(min(1.0, (t - 4.0) / 0.8))
        return m | head_m if t >= 4.9 else m

    res = _lecture(tmp_path, "arrow", ink_at, lambda t: np.zeros((H, W), bool))
    ps = [p for p in res.pieces if p["box"][0] <= 250 <= p["box"][0] + p["box"][2]]
    assert len(ps) == 1, res.pieces
    bgra, _, (x, y, w, h), _, _ = res.piece_image(ps[0]["id"])
    alpha = np.zeros((H, W), bool)
    alpha[y:y + h, x:x + w] = bgra[..., 3] > 0
    near = cv2.dilate(alpha.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # 1 px of slack
    shaft = shaft_upto(1.0) & ~(cv2.dilate(head_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    assert near[shaft].mean() >= 0.97, near[shaft].mean()
    assert near[head_m].mean() >= 0.9, near[head_m].mean()
    # ...and the replayed board has it all too
    on = res.ink_at(8.0)
    assert (cv2.dilate(on.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)[shaft].mean() >= 0.97


def _thick_line(width):
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (200, 300), (500, 300), 255, width, cv2.LINE_AA)
    return m > 127


def test_ink_beside_ink_accepted_a_sample_before_is_writing_not_camera_drift(tmp_path):
    # A line, then gone over again a moment later with a wider stroke: the new ink lies within 2 px of ink accepted one
    # sample before (as the rest of a stroke does beside an arrowhead the thick mask held back). The camera did not
    # move it: it stays in the piece's image (ink the camera moved is left out of it).
    thin, wide = _thick_line(3), _thick_line(7)

    def ink_at(t):
        return wide if t >= 4.4 else thin if t >= 4.0 else np.zeros((H, W), bool)

    res = _lecture(tmp_path, "retrace", ink_at, lambda t: np.zeros((H, W), bool))
    assert len(res.pieces) == 1, res.pieces
    bgra, _, (x, y, w, h), _, _ = res.piece_image(res.pieces[0]["id"])
    alpha = np.zeros((H, W), bool)
    alpha[y:y + h, x:x + w] = bgra[..., 3] > 0
    assert alpha[wide].mean() >= 0.97, alpha[wide].mean()
    assert not res.add_moved.any()


def test_ink_the_camera_moved_is_still_drift(tmp_path):
    # The same line written once; at 9 s the camera moves 2 px: ink re-accepted next to old ink is drift, not writing,
    # and is left out of the piece's image (the original marks stand for it).
    line = _thick_line(5)

    def frame(t):
        f = np.full((H, W, 3), BOARD, np.uint8)
        if t >= 3.0:
            f[line] = INK
        if t >= 9.0:
            f = np.ascontiguousarray(np.roll(f, (2, 2), axis=(0, 1)))
        return f

    clip = _encode(tmp_path / "drift.mp4", frame)
    res = capture_board(clip, params=BoardParams(width=W), mode="white")
    late = res.add_t >= 9.0
    assert late.any() and res.add_drift[late].mean() >= 0.9, (late.sum(), res.add_drift[late].mean())
    assert res.add_moved[late].mean() >= 0.9, res.add_moved[late].mean()
    bgra, _, (x, y, w, h), _, _ = res.piece_image(res.pieces[0]["id"])
    assert (bgra[..., 3] > 0).sum() == np.unique(res.add_idx[~late]).size
    assert len(res.pieces) == 1 and res.pieces[0]["t_seen"] < 5.0 and not res.erases


def test_a_redraw_never_comes_before_the_hidden_inks_spots_were_last_seen_blank():
    from app.pipeline import redraw_time

    p = {"revealed": True, "t_seen": 24.33, "written_after": 7.33, "redraw_after": 8.67}
    assert redraw_time(p, 5.0, []) == 8.67  # talked about before it could exist: drawn once it can
    assert redraw_time(p, 13.5, []) == 13.5
    assert redraw_time(p, None, [{"start": 6.0, "end": 9.0}]) == 8.67  # first mentioned, early
    assert redraw_time(p, None, [{"start": 30.0, "end": 31.0}]) is None  # first mentioned once the camera showed it
    assert redraw_time({**p, "redraw_after": None}, 5.0, []) == 7.33
    assert redraw_time({**p, "revealed": False}, 13.5, []) is None
