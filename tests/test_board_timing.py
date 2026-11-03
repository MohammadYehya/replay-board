"""Thick ink (a filled dot, an arrowhead, a heavy underline, where strokes cross) appears on the replayed board
when thin ink written at the same moment does, never before it was written (SPEC 5.1)."""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from app.core import media
from app.core.board import BoardGeometry, BoardParams, capture_board

W, H, FPS, SECONDS = 800, 450, 15, 15.0
R = 9  # 18 px across: thicker than max_stroke (11), thinner than the 21 px closing
INK = (30, 30, 30)


def _encode(path, frame):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)]
    frames = b"".join(frame(i / FPS).tobytes() for i in range(int(SECONDS * FPS)))
    subprocess.run(cmd, input=frames, check=True, capture_output=True, timeout=60)
    return media.probe(path)


def _mask(draw) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    draw(m)
    return m > 0


def _shown(res, mask) -> np.ndarray:
    """When each ink pixel of `mask` first appears on the replayed board."""
    sel = mask.reshape(-1)[res.add_idx]
    idx, t = res.add_idx[sel], res.add_t[sel]
    first = {}
    for i, v in zip(idx.tolist(), t.tolist()):
        first.setdefault(i, v)
    return np.array(sorted(first.values()))


# Written at 4 s, all at once: a thin line, a filled dot, a heavy (14 px) underline.
LINE = lambda m: cv2.line(m, (100, 150), (220, 150), 255, 3, cv2.LINE_AA)  # noqa: E731
DOT = lambda m: cv2.circle(m, (300, 150), R, 255, -1, cv2.LINE_AA)  # noqa: E731
BAR = lambda m: cv2.line(m, (380, 150), (520, 150), 255, 14, cv2.LINE_AA)  # noqa: E731
# The teacher (in the person mask) rests a hand 10 px from a second dot and a second line for 4 s after writing them.
HAND = (300, 250, 360, 330)  # x0, y0, x1, y1
DOT2 = lambda m: cv2.circle(m, (300 - R - 10, 290), R, 255, -1, cv2.LINE_AA)  # noqa: E731
LINE2 = lambda m: cv2.line(m, (370, 270), (470, 270), 255, 3, cv2.LINE_AA)  # noqa: E731


def _together_frame(t: float) -> np.ndarray:
    f = np.full((H, W, 3), 232, np.uint8)
    if t >= 4.0:
        for draw in (LINE, DOT, BAR, DOT2, LINE2):
            m = _mask(draw)
            f[m] = INK
    if 4.0 <= t < 8.0:
        x0, y0, x1, y1 = HAND
        cv2.rectangle(f, (x0, y0), (x1, y1), (70, 60, 50), -1)
    return f


def _hand_mask(t: float) -> np.ndarray:
    m = np.zeros((H, W), bool)
    if 4.0 <= t < 8.0:
        x0, y0, x1, y1 = HAND
        m[y0:y1 + 1, x0:x1 + 1] = True
    return m


@pytest.fixture(scope="module")
def together(tmp_path_factory):
    clip = _encode(tmp_path_factory.mktemp("together") / "together.mp4", _together_frame)
    return capture_board(clip, person_mask=_hand_mask, params=BoardParams(width=W))


def test_thick_ink_shows_with_thin_ink_written_at_the_same_moment(together):
    line = np.median(_shown(together, _mask(LINE)))
    for draw in (DOT, BAR):
        t = _shown(together, _mask(draw))
        assert t.size > 50 and t.min() >= 4.0  # never before it was written
        assert abs(np.median(t) - line) <= 0.4, (np.median(t), line)


def test_thick_ink_next_to_the_teachers_hand_shows_with_thin_ink(together):
    # The hand stops a thick shape from counting as "stays put" (a still wrist looks the same) until it
    # leaves at 8 s; once the shape has stayed put, it is shown from when it was written, like thin ink.
    line = np.median(_shown(together, _mask(LINE2)))
    t = _shown(together, _mask(DOT2))
    assert line < 5.5 and t.size > 50 and t.min() >= 4.0
    assert abs(np.median(t) - line) <= 0.4, (np.median(t), line)
    dots = [p for p in together.pieces if p["box"][0] <= 300 - R - 10 <= p["box"][0] + p["box"][2]
            and p["box"][1] <= 290 <= p["box"][1] + p["box"][3]]
    assert len(dots) == 1 and dots[0]["t_seen"] < 5.5 and not dots[0]["revealed"]


BODY = (560, 90, 700, 420)        # the teacher, in the person mask
WRIST = ((430, 250), (560, 250))  # a still 14 px bar joined to the body that SAM missed, 3 s to 10 s


def _under(width: int):
    return lambda m: cv2.line(m, (450, 250), (530, 250), 255, width, cv2.LINE_AA)


def _wrist_frame(wrist_bgr, width: int):
    """At 10 s the wrist lifts and writing (`width` px) appears exactly where it rested."""
    under = _mask(_under(width))

    def frame(t: float) -> np.ndarray:
        f = np.full((H, W, 3), 232, np.uint8)
        x0, y0, x1, y1 = BODY
        cv2.rectangle(f, (x0, y0), (x1, y1), (70, 60, 50), -1)
        if 3.0 <= t < 10.0:
            cv2.line(f, *WRIST, wrist_bgr, 14, cv2.LINE_AA)
        if t >= 10.0:
            f[under] = INK
        return f
    return frame


def _body_mask(t: float) -> np.ndarray:
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = BODY
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


@pytest.mark.parametrize("wrist_bgr, width", [((60, 60, 60), 3), (INK, 3), (INK, 14)],
                         ids=["lighter-wrist-thin-ink", "same-dark-thin-ink", "same-dark-thick-ink"])
def test_writing_where_a_missed_wrist_rested_is_not_shown_before_it_was_written(tmp_path, wrist_bgr, width):
    # The wrist counts as ink in view, held back by the thick mask. A change of darkness restarts the count;
    # when it is as dark as the ink, its middle turning into clean board as it lifts drops the count.
    clip = _encode(tmp_path / "wrist_then_ink.mp4", _wrist_frame(wrist_bgr, width))
    res = capture_board(clip, person_mask=_body_mask, params=BoardParams(width=W))
    band = np.zeros((H, W), bool)
    band[250 - 12: 250 + 13, WRIST[0][0] - 5: BODY[0]] = True
    ys, xs = np.divmod(res.add_idx, W)
    assert not (band[ys, xs] & (res.add_t < 10.0)).any()  # neither the wrist nor the writing before 10 s
    t = _shown(res, _mask(_under(width)))
    assert t.size > 50 and t.min() >= 10.0 and np.median(t) <= 11.5


# --- a thing SAM missed, replaced by writing where it lay without ever being seen clean -------------------------
SKIN = (120, 150, 200)
BODY2 = (620, 60, 760, 440)  # the teacher, in the person mask


def _rects(t: float, layers, board=(232, 232, 232)) -> np.ndarray:
    """Board with (box or fn(t) -> box, BGR, t0, t1) layers, softened: a blurred edge has every contrast."""
    f = np.full((H, W, 3), board, np.float32)
    for box, col, t0, t1 in layers:
        b = box(t) if callable(box) else box
        if t0 <= t < t1 and b is not None:
            x0, y0, x1, y1 = b
            f[y0:y1, x0:x1] = col
    return np.clip(cv2.GaussianBlur(f, (3, 3), 0.7), 0, 255).astype(np.uint8)


def _sam(t: float, layers) -> np.ndarray:
    m = np.zeros((H, W), bool)
    for box, t0, t1 in layers:
        b = box(t) if callable(box) else box
        if t0 <= t < t1 and b is not None:
            x0, y0, x1, y1 = b
            m[y0:y1, x0:x1] = True
    return m


def _box_mask(x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


CUFF = (430, 193, 625, 207)    # 14 px, joined to the body, missed by SAM: rests 3-9 s
CUFF_HAND = (410, 160, 625, 240)  # the teacher's hand (SAM) over it 9-11 s


def _cuff_line(width: int):
    return (430, 200 - width // 2, 625, 200 - width // 2 + width)  # written under the hand at 10 s, where the cuff lay


# (mode, board, body, cuff, line, line width): on chalk the grey image is 255 - max(B, G, R), so skin and chalk
# look alike
CUFF_CASES = {"ink-dark-cuff": ("white", (232, 232, 232), (70, 60, 50), INK, INK, 4),
              "ink-dark-cuff-8px-line": ("white", (232, 232, 232), (70, 60, 50), INK, INK, 8),
              "chalk-skin-cuff": ("chalk", (40, 60, 45), (200, 200, 200), SKIN, (200, 200, 200), 4)}


@pytest.mark.parametrize("case", list(CUFF_CASES))
def test_writing_under_the_hand_where_a_missed_cuff_rested_is_not_shown_before_it_was_written(tmp_path, case):
    # The cuff counts as ink in view, held back only by its own thick mask; under the hand it is replaced by a
    # line as dark as it, so the middle rows look the same when the hand leaves. Only the rows near the line's
    # edges change contrast: that must restart the count across the line's width (jump_zone).
    mode, board, body, cuff, ink, width = CUFF_CASES[case]
    line = _cuff_line(width)

    def frame(t):
        return _rects(t, [(BODY2, body, 0, 99), (CUFF, cuff, 3.0, 9.0), (line, ink, 10.0, 99),
                          (CUFF_HAND, SKIN, 9.0, 11.0)], board)

    clip = _encode(tmp_path / "cuff.mp4", frame)
    res = capture_board(clip, person_mask=lambda t: _sam(t, [(BODY2, 0, 99), (CUFF_HAND, 9.0, 11.0)]),
                        params=BoardParams(width=W), mode=mode)
    t = _shown(res, _box_mask(*line))
    assert t.size > 300 and t.min() >= 10.0 and np.median(t) <= 12.5, (t.size, t.min(), np.median(t))
    p = [q for q in res.pieces if q["box"][1] <= 200 <= q["box"][1] + q["box"][3]]
    assert len(p) == 1 and p[0]["t_seen"] >= 10.0 and p[0]["revealed"]  # written while hidden, shown on uncover


UNDERLINE = (250, 150, 400, 164)  # heavy (14 px) writing at 3 s, the hand (SAM) resting at its end until 9 s
UNDERLINE_HAND = (405, 120, 470, 200)


def _tip(t):
    return int(round(360 - 60 * min(1.0, max(0.0, t - 8.0))))  # the marker is pulled 60 px left over 8-9 s


def _barrel(t):
    return (_tip(t) - 60, 167, _tip(t), 173)  # 6 px, no thick core, 3 px under the underline, missed by SAM


def _line2(t):
    return (_tip(t), 168, 360, 172) if _tip(t) < 360 else None  # drawn where the barrel lay


@pytest.mark.parametrize("barrel_bgr", [INK, (70, 70, 70)], ids=["ink-dark-barrel", "grey-barrel"])
def test_a_marker_pulled_along_itself_next_to_thick_ink_does_not_show_its_line_early(tmp_path, barrel_bgr):
    # The barrel lies in the thick mask's reach of writing that cannot settle yet (the hand rests on it), so it
    # counts as ink in view, held back. Pulled along its own axis, it leaves a narrower line exactly where it lay,
    # with no clean sample in between: only some of the line's rows change contrast.
    def hold(t):
        return (_tip(t) - 110, 168, _tip(t) - 57, 215)

    def frame(t):
        return _rects(t, [(BODY2, (70, 60, 50), 0, 99), (UNDERLINE, INK, 3.0, 99), (UNDERLINE_HAND, SKIN, 3.0, 9.0),
                          (_line2, INK, 8.0, 99), (_barrel, barrel_bgr, 3.0, 9.3), (hold, SKIN, 3.0, 9.3)])

    clip = _encode(tmp_path / "barrel.mp4", frame)
    res = capture_board(clip, person_mask=lambda t: _sam(t, [(BODY2, 0, 99), (UNDERLINE_HAND, 3.0, 9.0),
                                                             (hold, 3.0, 9.3)]),
                        params=BoardParams(width=W))
    # when the pen reached each column of the line (2 px of slack for the blur)
    wt = np.full(W, np.inf)
    for i in range(int(8 * FPS), int(10 * FPS)):
        x = _tip(i / FPS)
        wt[x:360] = np.minimum(wt[x:360], i / FPS)
    wt = np.array([wt[max(0, x - 2):x + 3].min() for x in range(W)])
    line = _box_mask(300, 168, 360, 172).reshape(-1)[res.add_idx]
    first = {}
    for i, v in zip(res.add_idx[line].tolist(), res.add_t[line].tolist()):
        first.setdefault(i, v)
    idx = np.fromiter(first, np.int64)
    early = np.fromiter(first.values(), np.float64) < wt[idx % W] - 1e-6
    assert idx.size > 150 and not early.any(), (idx.size, int(early.sum()))
    under = _shown(res, _box_mask(*UNDERLINE))
    assert under.size > 1500 and under.min() >= 3.0


# --- the generated lecture with thick writing (slow: pytest -m selftest) -----------------------------------
@pytest.mark.selftest
@pytest.mark.parametrize("oracle", [False, True], ids=["no-person", "person"])
def test_arrows_lecture_thick_ink_on_time(oracle):
    from app.config import DATA_DIR
    from tests import gen_board as gen

    video = gen.make_board_lecture(DATA_DIR / "selftest", "arrows")
    info = media.probe(video)
    p = BoardParams(width=800)
    corners = gen.board_corners()
    person = (lambda t: gen.teacher_mask(t, "arrows")) if oracle else None
    res = capture_board(info, corners=corners, person_mask=person, params=p)
    geom = BoardGeometry(info, corners, 800, p.decode_width)
    # when each board pixel was written: the nearest written video pixel within 2 px
    wt = gen.write_time("arrows").astype(np.float32)
    wt = cv2.erode(np.where(np.isfinite(wt), wt, np.float32(1e9)), np.ones((5, 5), np.uint8))
    sel = ~res.add_drift
    first = {}
    for i, v in zip(res.add_idx[sel].tolist(), res.add_t[sel].tolist()):
        first.setdefault(i, v)
    idx = np.fromiter(first, np.int64)
    shown = np.fromiter(first.values(), np.float64)
    ys, xs = np.divmod(idx, res.width)
    vx, vy = np.round(geom.board_to_video(np.stack([xs, ys], 1))).astype(int).T
    written = wt[vy.clip(0, gen.H - 1), vx.clip(0, gen.W - 1)].astype(np.float64)
    lag = shown - written
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    cats = {}
    for it in gen.ITEMS:
        if it["id"] == gen.HIDDEN_ITEM:
            continue
        ink = geom.warp_mask(gen.item_ink(it["id"], "arrows"))
        core = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3) > p.max_stroke / 2
        near = cv2.dilate(core.astype(np.uint8), ring) > 0
        on_it = ink[ys, xs] & (written < 1e8)
        cats[it["id"]] = {"thick": on_it & core[ys, xs], "beside": on_it & near[ys, xs] & ~core[ys, xs],
                          "thin": on_it & ~near[ys, xs]}
        assert (lag[on_it] >= 0).all(), (it["id"], float(lag[on_it].min()))  # never before it was written
    thin_all = np.median(lag[np.any([c["thin"] for c in cats.values()], axis=0)])
    for i, c in cats.items():
        thin = np.median(lag[c["thin"]]) if c["thin"].sum() >= 50 else thin_all
        for k in ("thick", "beside"):
            if c[k].sum() >= 20:
                assert np.median(lag[c[k]]) - thin <= 0.4, (i, k, float(np.median(lag[c[k]])), float(thin))
