"""Typed caption overlays in an edited video are not writing (SPEC 5.1).

An edited lecture burns typed captions into the picture for a few seconds ("Catenate means...", "Ring Chain" on
data/test4.mp4): white text that appears in one frame, stays a while and goes. On a chalkboard white is ink, and a
caption stays longer than short-lived ink, so it became a piece of writing, and an erase when it went. Writing comes a
stroke at a time; a caption comes all at once, on board that was blank a moment before, and goes within a minute.

Here: captions alone, next to writing done just before (they join its piece) and across writing (some of their ink
stays accepted when they go); writing done as a caption comes up, kept even when it is wiped soon after; and streamed
partials, which leave out what may still turn out a caption and never show writing behind their ready line. Then
captions that come in two goes: faded in (their edges come a sample late, as drift), or in two parts whose first part
leaves first; and the next line written in a piece a caption joined, while it is up.

The frames are made here and fed straight to capture_board (no video file), like test_board_merge.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core import board
from app.core.board import BoardParams, capture_board
from app.core.media import VideoInfo
from app.stream import ready_pieces

W, H, FPS = 800, 300, 3.0
SECONDS = 64.0
BOARD_BGR, CHALK, CAPTION = (60, 110, 70), (215, 215, 215), (250, 250, 250)  # chalkboard grey 92
INFO = VideoInfo(Path("synthetic.mp4"), SECONDS, W, H, FPS, False, 0)

# Real writing: 3 px horizontal strokes written left to right at an even speed, (x0, x1, y, t0, t1).
STROKES = {
    "R0": (60, 300, 40, 4.0, 8.0),  # wiped at 45 s: a real erase
    "R3": (60, 240, 50, 14.5, 16.5),  # written as C1 appears, wiped with R0 (gone within a minute, like a caption)
    "R1": (60, 340, 100, 9.0, 13.0),
    "R1b": (60, 340, 150, 15.0, 19.0),  # written while caption C1 is up
    "R2": (60, 260, 230, 26.0, 30.0),  # caption C2 appears just after it, next to it
    "W": (470, 690, 70, 27.0, 31.0),  # caption C3 is later written across it
}
WIPE_R0 = 45.0
WIPED = ("R0", "R3")


def _caption(lines, x, y):
    """Anti-aliased typed text (0..255), lines 34 px apart from baseline y."""
    m = np.zeros((H, W), np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(m, s, (x, y + 34 * i), cv2.FONT_HERSHEY_DUPLEX, 0.8, 255, 2, cv2.LINE_AA)
    return m


# Captions on screen for 10 s each. C1 is far from any writing; C2's first letter is ~22 px right of R2
# (pieces join within 25 px, join_pad), shown a second after R2 was finished (within join_recent), so it joins R2's
# piece. C3's first line crosses W: where its letters come within 2 px of W's chalk they stay accepted when it goes.
CAPTIONS = {
    "C1": (_caption(["Catenate means the", "ability of carbon", "to combine with one"], 450, 60), 14.0, 24.0),
    "C2": (_caption(["Ring Chain and", "straight chains", "of carbon atoms"], 282, 222), 31.0, 41.0),
    "C3": (_caption(["Hybrid orbitals", "are the mixing", "of orbitals"], 470, 80), 50.0, 60.0),
}


def _stroke_mask(name, upto=np.inf):
    """A stroke of STROKES by name, or an (x0, x1, y, t0, t1) one, as written by time `upto`."""
    x0, x1, y, t0, t1 = STROKES[name] if isinstance(name, str) else name
    m = np.zeros((H, W), bool)
    xs = np.arange(x0, x1)
    tw = t0 + (xs - x0) / (x1 - x0) * (t1 - t0)  # when each column is written
    m[y:y + 3, xs[tw <= upto + 1e-6]] = True
    return m


def _write_time(name):
    """Per pixel of the stroke, when it was written (inf off the stroke)."""
    x0, x1, y, t0, t1 = STROKES[name] if isinstance(name, str) else name
    tw = np.full((H, W), np.inf)
    xs = np.arange(x0, x1)
    tw[y:y + 3, xs] = t0 + (xs - x0) / (x1 - x0) * (t1 - t0)
    return tw


def _over(f, m, alpha=1.0):
    """The frame with typed text m (0..255) laid over it, at `alpha` (fading in)."""
    a = (m.astype(np.float32) / 255.0 * alpha)[..., None]
    return np.round(f * (1 - a) + np.float32(CAPTION) * a).astype(np.uint8)


def _frame(t):
    f = np.full((H, W, 3), BOARD_BGR, np.uint8)
    for name in STROKES:
        if name in WIPED and t >= WIPE_R0 - 1e-6:
            continue
        f[_stroke_mask(name, t)] = CHALK
    for m, t0, t1 in CAPTIONS.values():
        if t0 - 1e-6 <= t < t1 - 1e-6:
            f = _over(f, m)
    return f


@pytest.fixture
def synthetic(monkeypatch):
    def frames(info, fps, width, start=0.0, end=None, prefetch=8):
        for k in range(int(round((end - start) * fps))):
            t = start + k / fps
            yield t, _frame(t)

    monkeypatch.setattr(board, "iter_frames", frames)
    monkeypatch.setattr(board, "detect_mode", lambda info, geom, s0, s1, p: ("chalk", 92.0))


def _on(mask, idx):
    return mask.reshape(-1)[idx]


def _caption_px(grow=2):
    """Caption pixels and those within `grow` of them, but not the real writing's own."""
    m = np.zeros((H, W), bool)
    for c, _, _ in CAPTIONS.values():
        m |= c > 0
    m = cv2.dilate(m.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)) > 0
    for name in STROKES:
        m &= ~_stroke_mask(name)
    return m


def _boxes_overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _xywh(mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return [x, y, w, h]


def _capture(**kw):
    return capture_board(INFO, params=BoardParams(width=W), mode="chalk", **kw)


def test_the_scene_is_the_one_meant(synthetic, monkeypatch, capsys):
    """The captions are big blocks of ink in one step, C2 joins R2's piece (debug log), and part of C3 stays accepted
    after it goes (without the overlay rule)."""
    for m, _, _ in CAPTIONS.values():
        assert (m > 128).sum() > 3 * 600
    monkeypatch.setenv("RB_DEBUG_BOARD", "1")
    res = _capture()
    log = capsys.readouterr().out.splitlines()
    r2 = np.unique(res.add_piece[_on(_stroke_mask("R2"), res.add_idx)])
    assert r2.size == 1
    assert any(line.startswith(f"[board] t=31.67 join {int(r2[0])} ") for line in log)
    c3 = CAPTIONS["C3"][0] > 0
    plain = capture_board(INFO, params=BoardParams(width=W, overlay_burst=0), mode="chalk")
    stays = _on(c3 & ~_stroke_mask("W"), plain.add_idx[np.isinf(plain.add_gone)])
    assert 100 < stays.sum() < 0.3 * c3.sum(), (stays.sum(), c3.sum())


def test_typed_captions_are_not_writing(synthetic):
    res = _capture()
    cap = _caption_px()
    assert not _on(cap, res.add_idx).any(), "a caption pixel is on the clean board"
    assert len(res.pieces) == len(STROKES), res.pieces
    # The only erase is the wipe of R0 and R3: a caption going away rubs nothing out.
    assert len(res.erases) == 1, res.erases
    e = res.erases[0]
    assert WIPE_R0 - 0.5 <= e["t"] <= WIPE_R0 + 1.0
    assert all(_boxes_overlap(e["box"], _xywh(_stroke_mask(n))) for n in WIPED)
    for name, (m, _, _) in CAPTIONS.items():
        assert not _boxes_overlap(e["box"], _xywh(m > 0)), name


@pytest.mark.parametrize("name", list(STROKES))
def test_real_writing_is_kept_with_its_timing(synthetic, name):
    res = _capture()
    stroke = _stroke_mask(name)
    tw = _write_time(name).reshape(-1)
    sel = _on(stroke, res.add_idx)
    idx, t = res.add_idx[sel], res.add_t[sel]
    # every pixel of the stroke is on the board, shown within a second of when it was written
    assert np.unique(idx).size == stroke.sum()
    first = np.full(W * H, np.inf)
    np.minimum.at(first, idx, t)
    late = first[np.flatnonzero(stroke)] - tw[np.flatnonzero(stroke)]
    assert late.min() >= 0.0 and late.max() <= 1.0 + 1e-3, (late.min(), late.max())
    # one piece, seen while it was written, not "written while hidden"
    ids = np.unique(res.add_piece[sel])
    assert ids.size == 1
    q = next(q for q in res.pieces if q["id"] == int(ids[0]))
    x0, x1, y, t0, t1 = STROKES[name]
    assert t0 <= q["t_seen"] <= t1 + 1.5 and not q["revealed"], q
    assert q["box"] == [x0, y, x1 - x0, 3], q  # nothing of a caption in its box
    if name in WIPED:
        assert q["removed_at"] is not None and abs(q["removed_at"] - WIPE_R0) <= 0.5, q
    else:
        assert q["removed_at"] is None, q


def test_a_streamed_replay_never_shows_a_caption(synthetic):
    plain = _capture()
    seen = []
    res = _capture(checkpoint=lambda part, t, ready: seen.append((part, t, ready)), checkpoint_every=1.0,
                   checkpoint_wall=0.0, checkpoint_share=1.0)
    assert len(seen) >= 40
    assert res.pieces == plain.pieces and res.erases == plain.erases
    assert np.array_equal(res.add_idx, plain.add_idx) and np.array_equal(res.add_t, plain.add_t)
    final = {q["id"] for q in plain.pieces}
    cap = _caption_px()
    prev = 0.0
    for part, t, ready in seen:
        assert prev <= ready <= t
        prev = ready
        shown = {q["id"] for q in ready_pieces(part, ready)}
        assert shown <= final, (t, ready, shown - final)  # what a partial shows is still there at the end
        ink = part.add_idx[np.isin(part.add_piece, list(shown))]
        assert not _on(cap, ink).any(), (t, ready)
    assert ready_pieces(seen[-1][0], seen[-1][2])


# --- writing that comes into view at once and stays (the cut to another board, a banner going) ---------------------
BLOCK = np.zeros((H, W), bool)
for _i in range(8):
    BLOCK[150 + 12 * _i:153 + 12 * _i, 420:700] = True  # 8 lines of 280 px: 6720 px in one frame from 10 s


def _reveal_frame(t):
    f = np.full((H, W, 3), BOARD_BGR, np.uint8)
    f[_stroke_mask("R0", t)] = CHALK  # written 4-8 s
    if t >= 10.0 - 1e-6:
        f[BLOCK] = CHALK
    f[_stroke_mask("R1", t - 12.0)] = CHALK  # written 21-25 s
    return f


def test_a_streamed_replay_never_shows_writing_behind_its_ready_line(monkeypatch):
    """A burst that stays is writing, known once it outlived overlay_life: until then the partials leave it out and
    hold their ready line before it, so it never shows up behind a line a partial already passed."""
    def frames(info, fps, width, start=0.0, end=None, prefetch=8):
        for k in range(int(round((end - start) * fps))):
            yield start + k / fps, _reveal_frame(start + k / fps)

    monkeypatch.setattr(board, "iter_frames", frames)
    monkeypatch.setattr(board, "detect_mode", lambda info, geom, s0, s1, p: ("chalk", 92.0))
    info = VideoInfo(Path("synthetic.mp4"), 90.0, W, H, FPS, False, 0)
    seen = []
    res = capture_board(info, params=BoardParams(width=W), mode="chalk",
                        checkpoint=lambda part, t, ready: seen.append((part, t, ready)), checkpoint_every=1.0,
                        checkpoint_wall=0.0, checkpoint_share=1.0)
    ys, xs = np.divmod(res.add_idx, W)
    assert BLOCK[ys, xs].sum() == BLOCK.sum()  # the block is writing, on the board from 10.67 s
    assert any(q["t_seen"] == 10.67 for q in res.pieces), res.pieces
    before, prev = set(), 0.0
    for part, t, ready in seen:
        now = {q["id"]: q for q in ready_pieces(part, ready)}
        for pid in now.keys() - before:
            assert now[pid]["t_seen"] > prev, (t, ready, prev, now[pid])
        before, prev = set(now), ready
    assert {q["id"] for q in res.pieces} <= before  # and every piece is shown by the end


# --- captions that come in two goes, and writing done while one is up ---------------------------------------------
# Five lines written one under the other, 14 px apart (one piece), done at 26.5 s. A caption comes up 1.5 s later
# with its first letter 22 px right of them: it joins their piece and is about half of it (below overlay_share).
LINES = [(60, 360, 40 + 14 * i, 6.5 + 4 * i, 10.5 + 4 * i) for i in range(5)]
NEXT = (60, 360, 110, 28.4, 34.4)  # the next line, begun just after the caption came up: it joins the piece too
DEFN = _caption(["Catenate means the", "ability of carbon"], 382, 70)
T_DEFN, T_DEFN_GONE = 28.0, 38.0
# A title, and a subtitle a second later right under it (it joins the title's piece); the title goes first. The title
# is 44% of the piece: once it has gone, less than overlay_gone of the piece's window is gone, and the subtitle up is
# less than overlay_share of it.
FAR = (60, 260, 230, 4.0, 8.0)
TITLE = (_caption(["Ring Chain and"], 300, 80), 20.0, 28.0)
SUBTITLE = (_caption(["and straight chains"], 300, 114), 21.0, 36.0)
SCENE_SECONDS = 50.0


def _ink(strokes, upto=np.inf):
    m = np.zeros((H, W), bool)
    for s in strokes:
        m |= _stroke_mask(s, upto)
    return m


def _grown(m, grow=2):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)) > 0


def _faded_frame(t):
    """The caption next to the lines fades in: at 45% for its first sample (its anti-aliased edges are not ink yet)."""
    f = np.full((H, W, 3), BOARD_BGR, np.uint8)
    f[_ink(LINES, t)] = CHALK
    if T_DEFN - 1e-6 <= t < T_DEFN_GONE - 1e-6:
        f = _over(f, DEFN, 0.45 if t < T_DEFN + 1 / FPS - 1e-6 else 1.0)
    return f


def _next_line_frame(t):
    f = np.full((H, W, 3), BOARD_BGR, np.uint8)
    f[_ink(LINES + [NEXT], t)] = CHALK
    if T_DEFN - 1e-6 <= t < T_DEFN_GONE - 1e-6:
        f = _over(f, DEFN)
    return f


def _two_part_frame(t):
    f = np.full((H, W, 3), BOARD_BGR, np.uint8)
    f[_stroke_mask(FAR, t)] = CHALK
    for m, t0, t1 in (TITLE, SUBTITLE):
        if t0 - 1e-6 <= t < t1 - 1e-6:
            f = _over(f, m)
    return f


SCENES = {  # frames, the real writing, the caption's pixels and those within 2 px of them (not the writing's own)
    "faded in": (_faded_frame, _ink(LINES), _grown(DEFN > 0) & ~_ink(LINES)),
    "next line": (_next_line_frame, _ink(LINES + [NEXT]), _grown(DEFN > 0) & ~_ink(LINES + [NEXT])),
    "two parts": (_two_part_frame, _ink([FAR]), _grown((TITLE[0] > 0) | (SUBTITLE[0] > 0)) & ~_ink([FAR])),
}


def _run(monkeypatch, scene, stream=False, **params):
    frame = SCENES[scene][0]

    def frames(info, fps, width, start=0.0, end=None, prefetch=8):
        for k in range(int(round((end - start) * fps))):
            yield start + k / fps, frame(start + k / fps)

    monkeypatch.setattr(board, "iter_frames", frames)
    monkeypatch.setattr(board, "detect_mode", lambda info, geom, s0, s1, p: ("chalk", 92.0))
    info = VideoInfo(Path("synthetic.mp4"), SCENE_SECONDS, W, H, FPS, False, 0)
    seen = []
    kw = {}
    if stream:
        kw = dict(checkpoint=lambda part, t, ready: seen.append((part, t, ready)), checkpoint_every=1.0,
                  checkpoint_wall=0.0, checkpoint_share=1.0)
    res = capture_board(info, params=BoardParams(width=W, **params), mode="chalk", **kw)
    return (res, seen) if stream else res


def _one_piece_of_writing(res, writing, cap):
    """The capture is the writing alone, every pixel of it, in one piece that stays: no caption ink, no erase."""
    assert not _on(cap, res.add_idx).any(), np.unique(res.add_idx[_on(cap, res.add_idx)]).size
    assert np.unique(res.add_idx[_on(writing, res.add_idx)]).size == writing.sum()
    assert len(res.pieces) == 1, res.pieces
    q = res.pieces[0]
    assert q["box"] == _xywh(writing) and q["removed_at"] is None and not q["revealed"], q
    assert not res.erases, res.erases


def test_a_caption_faded_in_leaves_no_edges(monkeypatch):
    """Faded in, a caption's anti-aliased edges become ink a sample after the letters next to them, so they are
    accepted as drift (old ink moved within drift_r). They go with the caption even when it joined writing done just
    before and is less than overlay_share of its piece."""
    _, writing, cap = SCENES["faded in"]
    plain = _run(monkeypatch, "faded in", overlay_burst=0)
    on = _on(cap, plain.add_idx)
    assert plain.add_drift[on].sum() > 100  # the scene is the one meant: edges came as drift, in the lines' piece
    assert np.unique(plain.add_piece[on | _on(writing, plain.add_idx)]).size == 1
    _one_piece_of_writing(_run(monkeypatch, "faded in"), writing, cap)


def test_the_next_line_written_while_a_caption_is_up_is_writing(monkeypatch):
    """He goes on writing as a caption comes up next to his lines: the caption joined his piece, and so does the next
    line, accepted within overlay_win after the caption. The caption goes; the line stays, and is writing, shown as
    it was written."""
    _, writing, cap = SCENES["next line"]
    plain = _run(monkeypatch, "next line", overlay_burst=0)
    nxt = _stroke_mask(NEXT)
    assert np.unique(plain.add_piece[_on(nxt | cap, plain.add_idx)]).size == 1  # one piece: lines, caption, line
    res = _run(monkeypatch, "next line")
    _one_piece_of_writing(res, writing, cap)
    sel = _on(nxt, res.add_idx)
    first = np.full(W * H, np.inf)
    np.minimum.at(first, res.add_idx[sel], res.add_t[sel])
    px = np.flatnonzero(nxt)
    late = first[px] - _write_time(NEXT).reshape(-1)[px]
    assert late.min() >= 0.0 and late.max() <= 1.0 + 1e-3, (late.min(), late.max())


@pytest.mark.parametrize("scene", ["two parts", "faded in"])
def test_a_streamed_replay_never_shows_what_the_final_drops(monkeypatch, scene):
    """Two parts: once the title has gone, the piece's window ink is not mostly gone yet (the subtitle is up), so it
    is no caption yet; it is when the subtitle goes. Faded in: the edges accepted as drift are taken with the caption
    only once they go. Until it is known, a partial shows none of it, nor a piece the final replay drops."""
    _, writing, cap = SCENES[scene]
    if scene == "two parts":  # the scene is the one meant: the subtitle joined the title's piece
        both = _run(monkeypatch, scene, overlay_burst=0)
        assert np.unique(both.add_piece[_on(cap, both.add_idx)]).size == 1
    plain = _run(monkeypatch, scene)
    res, seen = _run(monkeypatch, scene, stream=True)
    assert len(seen) >= 30
    assert res.pieces == plain.pieces and res.erases == plain.erases
    assert np.array_equal(res.add_idx, plain.add_idx) and np.array_equal(res.add_t, plain.add_t)
    final = {q["id"] for q in plain.pieces}
    assert not _on(cap, plain.add_idx).any()
    prev = 0.0
    for part, t, ready in seen:
        assert prev <= ready <= t
        prev = ready
        shown = {q["id"] for q in ready_pieces(part, ready)}
        assert shown <= final, (t, ready, shown - final, part.pieces)
        ink = part.add_idx[np.isin(part.add_piece, list(shown))]
        assert not _on(cap, ink).any(), (t, ready, int(_on(cap, ink).sum()))
    assert ready_pieces(seen[-1][0], seen[-1][2])  # the writing is shown by the end
