"""Board capture (SPEC 5.1): writing stays put; the teacher moves.

Per-pixel counters run only where the board is visible. A pixel is accepted after
`accept_n` ink samples in a row and removed after `remove_n` clean samples in a row.
Newly accepted pixels are grouped into pieces of writing; removals into erases.
The result is a replay log from which the board at any time can be rebuilt.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .media import VideoInfo, clip_range, grab_frame, iter_frames, scaled_size

PersonMask = Callable[[float], "np.ndarray | None"]


@dataclass(frozen=True)
class BoardParams:
    width: int = 800  # analysis width; every pixel size below is for 800 and scales with s
    fps: float = 3.0
    decode_width: int = 1280
    white_level: float = 110.0  # median brightness above this means a whiteboard
    mode_moments: int = 9
    close_size: int = 21
    ink_thr: float = 22.0
    keep_thr: float = 12.0  # hysteresis for accepted pixels
    border_frac: float = 0.008
    bg_decay: float = 0.5
    dark_thr: float = 45.0
    dark_dilate: int = 15
    max_stroke: float = 11.0
    thick_dilate: int = 31
    # Writing stays put: too-thick ink (an arrowhead, a dot, where strokes cross) that keeps the
    # same spot and grey for `stay_n` samples is writing, not something in front. 0 = never.
    stay_n: int = 3
    stay_tol: float = 10.0  # contrast levels (vs the clean board) a staying thick pixel may change between samples
    stay_zone: float = 9.0  # thick pixels this close (diameter) to staying ones stay too (edges flicker)
    # A thick shape starts its count only while all of it is more than this far (px at 800) from
    # anything else in front: a still hand, wrist or sleeve SAM missed is joined to the teacher.
    stay_clear: float = 6.0
    person_dilate: int = 15
    accept_n: int = 3
    remove_n: int = 6
    drift_r: float = 2.0
    drift_share: float = 0.5  # a new component this much within drift_r of old ink is drift, not writing
    touch_r: float = 4.0  # a small bit "touches" a piece if its ink is this close
    settle_tol: float = 25.0
    settle_give_up: float = 30.0  # seconds without settling before we start anyway
    pre_window: float = 2.5
    group_dilate: int = 13
    join_recent: float = 3.0
    join_pad: int = 25
    update_min_px: int = 6  # new pixels in one step for a piece to count as recently updated
    min_px: int = 25
    min_life: float = 6.0
    short_life_frac: float = 0.6
    seen_frac: float = 0.10
    hair_thick: float = 1.5
    hair_ratio: float = 15.0
    band_frac: float = 0.03
    erase_min_px: int = 60
    erase_merge: float = 2.5
    reveal_gap: float = 3.0
    darken: float = 0.72


def _odd(d: float) -> int:
    d = max(1, int(round(d)))
    return d if d % 2 else d + 1


def _ellipse(d: float) -> np.ndarray:
    d = _odd(d)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def order_corners(pts) -> np.ndarray:
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, np.float32).reshape(4, 2)
    c = p.mean(0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]  # clockwise in image coords, starting near top-left/left
    start = int(np.argmin(p.sum(1)))
    return np.roll(p, -start, axis=0)


class BoardGeometry:
    """Maps decoded video frames onto the flattened board."""

    def __init__(self, info: VideoInfo, corners, width: int, decode_width: int):
        self.info = info
        self.dw, self.dh = scaled_size(info, min(decode_width, info.width) if info.width >= 640 else decode_width)
        self.width = int(width)
        k = self.dw / info.width
        if corners is not None:
            c = order_corners(corners)
            self.corners_video = c.copy()
            cd = c * np.float32(k)
            top = np.linalg.norm(cd[1] - cd[0]); bottom = np.linalg.norm(cd[2] - cd[3])
            left = np.linalg.norm(cd[3] - cd[0]); right = np.linalg.norm(cd[2] - cd[1])
            aspect = (left + right) / max(1e-6, top + bottom)
            self.height = max(8, int(round(self.width * aspect)))
            dst = np.float32([[0, 0], [self.width - 1, 0], [self.width - 1, self.height - 1], [0, self.height - 1]])
            self.M = cv2.getPerspectiveTransform(cd.astype(np.float32), dst)
        else:
            self.corners_video = np.float32([[0, 0], [info.width - 1, 0], [info.width - 1, info.height - 1],
                                             [0, info.height - 1]])
            self.height = max(8, int(round(self.dh * self.width / self.dw)))
            self.M = np.float64([[self.width / self.dw, 0, 0], [0, self.height / self.dh, 0], [0, 0, 1]])

    def warp(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != self.dw or frame.shape[0] != self.dh:
            frame = cv2.resize(frame, (self.dw, self.dh), interpolation=cv2.INTER_AREA)
        return cv2.warpPerspective(frame, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8) * 255
        if m.shape[1] != self.dw or m.shape[0] != self.dh:
            m = cv2.resize(m, (self.dw, self.dh), interpolation=cv2.INTER_NEAREST)
        out = cv2.warpPerspective(m, self.M, (self.width, self.height), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out > 0

    def board_to_video(self, pts) -> np.ndarray:
        """Board pixel coordinates -> original video pixel coordinates."""
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        inv = np.linalg.inv(self.M)
        q = cv2.perspectiveTransform(p, inv).reshape(-1, 2)
        return q * (self.info.width / self.dw)


@dataclass
class Piece:
    id: int
    box: list  # x0, y0, x1, y1 (inclusive-exclusive)
    last_update: float
    written_after: float
    pre_existing: bool = False
    born: float = 0.0


@dataclass
class BoardResult:
    width: int
    height: int
    mode: str  # "white" or "chalk"
    corners: list  # 4 corners in video pixels (tl, tr, br, bl)
    background: tuple  # BGR
    t_start: float  # first settled sample
    t_end: float
    fps: float
    pieces: list  # dicts, see _finish
    erases: list  # dicts {t, box}
    # replay log, kept marks only: adds sorted by time, removes sorted by time
    add_t: np.ndarray
    add_idx: np.ndarray
    add_piece: np.ndarray
    add_color: np.ndarray  # N x 3 BGR, already darkened
    rem_t: np.ndarray
    rem_idx: np.ndarray
    add_drift: np.ndarray | None = None  # per add: ink re-accepted after the camera drifted
    add_gone: np.ndarray | None = None  # per add: when it leaves the replayed board (inf = never)
    stats: dict = field(default_factory=dict)

    def ink_at(self, t: float) -> np.ndarray:
        """Boolean ink mask of the board at time t, rebuilt from the replay log."""
        state = np.full(self.width * self.height, -1.0)
        a = self.add_t <= t
        # A pixel is ink at t if its latest add <= t is later than its latest remove <= t.
        np.maximum.at(state, self.add_idx[a], self.add_t[a])
        rem = np.full(self.width * self.height, -1.0)
        r = self.rem_t <= t
        np.maximum.at(rem, self.rem_idx[r], self.rem_t[r])
        return ((state >= 0) & (state > rem)).reshape(self.height, self.width)

    def image_at(self, t: float) -> np.ndarray:
        img = np.empty((self.height * self.width, 3), np.uint8)
        img[:] = self.background
        a = self.add_t <= t
        img[self.add_idx[a]] = self.add_color[a]  # later adds overwrite earlier ones
        ink = self.ink_at(t).reshape(-1)
        out = np.empty_like(img)
        out[:] = self.background
        out[ink] = img[ink]
        return out.reshape(self.height, self.width, 3)

    def piece_image(self, piece_id: int) -> tuple[np.ndarray, np.ndarray, list, np.ndarray | None, list]:
        """A piece for the player: BGRA crop, drawing-order map, box, and removal map.

        - BGRA: the strokes on a transparent background.
        - order: 8-bit, 1..255 linear in write time over the returned [draw_start, draw_end]
          (within one sample, left to right); 0 where there is no ink.
        - removal map: 24-bit RGB per pixel = deciseconds + 1 when that pixel leaves the board
          (partial erases), 0 = stays; None when nothing of the piece is ever removed.
        Ink re-accepted after the camera drifted is left out: the original marks stand for it.
        A pixel added more than once keeps its first order and its last colour.
        """
        sel = self.add_piece == piece_id
        if self.add_drift is not None and (sel & ~self.add_drift).any():
            sel &= ~self.add_drift
        idx, t, col = self.add_idx[sel], self.add_t[sel], self.add_color[sel]
        gone = self.add_gone[sel] if self.add_gone is not None else np.full(idx.size, np.inf)
        if idx.size == 0:
            return np.zeros((1, 1, 4), np.uint8), np.zeros((1, 1), np.uint8), [0, 0, 1, 1], None, [0.0, 0.0]
        ys, xs = np.divmod(idx, self.width)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        bgra = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        bgra[ys - y0, xs - x0, :3] = col  # adds are in time order, so later colours win
        bgra[ys - y0, xs - x0, 3] = 255
        # Order follows real time: each sample's new pixels were written during the sample
        # interval before it, so spread them over it left to right. Then 1..255 maps linearly
        # onto [draw_start, draw_end], so a piece written in bursts replays in bursts.
        te = t.astype(np.float64).copy()
        dt = 1.0 / max(1e-6, self.fps)
        for tv in np.unique(t):
            m = t == tv
            r = np.argsort(np.argsort(xs[m], kind="stable"), kind="stable")
            te[m] = tv - dt + dt * (r + 1) / max(1, m.sum())
        lo, hi = float(te.min()), float(te.max())
        val = (1 + np.round(254 * (te - lo) / max(1e-6, hi - lo))).astype(np.int64).clip(1, 255).astype(np.uint8)
        omap = np.zeros((y1 - y0, x1 - x0), np.uint8)
        rev = np.argsort(-te, kind="stable")  # write latest first so the earliest order wins
        omap[ys[rev] - y0, xs[rev] - x0] = val[rev]
        gmap = None
        if np.isfinite(gone).any():
            # The latest add of a pixel decides when it leaves (it may have been rewritten).
            code = np.where(np.isfinite(gone), np.round(np.nan_to_num(gone, posinf=0) * 10) + 1, 0).astype(np.int64)
            last = np.zeros((y1 - y0, x1 - x0), np.int64)
            last[ys - y0, xs - x0] = code  # adds are in time order
            gmap = np.stack([last & 255, (last >> 8) & 255, (last >> 16) & 255], axis=2).astype(np.uint8)  # BGR
        return bgra, omap, [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], gmap, [round(lo, 2), round(hi, 2)]


def _roi(ys: np.ndarray, xs: np.ndarray, pad: int, H: int, W: int) -> tuple[int, int, int, int]:
    """Bounding box of some pixels grown by `pad`, clipped: work on this instead of the whole board."""
    return (max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad + 1),
            max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad + 1))


def _rect(mask: np.ndarray, ox: int = 0, oy: int = 0):
    """(x0, y0, x1, y1) around the nonzero pixels of a bool/uint8 mask (offset by ox, oy), or None."""
    x, y, w, h = cv2.boundingRect(mask.view(np.uint8) if mask.dtype == bool else mask)
    return (ox + x, oy + y, ox + x + w, oy + y + h) if w else None


def _union(a, b):
    if a is None or b is None:
        return a or b
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _middle(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    return gray[h // 4: h - h // 4, w // 4: w - w // 4]


def detect_mode(info: VideoInfo, geom: BoardGeometry, start: float, end: float, p: BoardParams):
    """Median middle brightness at several moments (never the first frame)."""
    n = p.mode_moments
    times = [start + (end - start) * (i + 0.5) / n for i in range(n)]

    def one(t):
        try:
            g = cv2.cvtColor(geom.warp(grab_frame(info, t, geom.dw)), cv2.COLOR_BGR2GRAY)
            return float(np.median(_middle(g)))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(n, 4)) as ex:
        vals = [v for v in ex.map(one, times) if v is not None]
    typical = float(np.median(vals)) if vals else 128.0
    return ("white" if typical > p.white_level else "chalk"), typical


def capture_board(info: VideoInfo, corners=None, person_mask: PersonMask | None = None,
                  params: BoardParams | None = None, start: float = 0.0, end: float | None = None,
                  progress: Callable[[float], None] | None = None, mode: str | None = None) -> BoardResult:
    p = params or BoardParams()
    debug = bool(os.environ.get("RB_DEBUG_BOARD"))
    t_wall = time.perf_counter()
    s0, s1 = clip_range(info, start, end)
    geom = BoardGeometry(info, corners, p.width, p.decode_width)
    W, H = geom.width, geom.height
    s = W / 800.0
    detected_mode, typical = detect_mode(info, geom, s0, s1, p)
    mode = mode or detected_mode

    k_close = _ellipse(p.close_size * s)
    k_close_800 = _ellipse(p.close_size)
    # Coarse ("in front") masks: at 800 px scale when analysing wider than 800.
    W8, H8 = (800, max(8, int(round(H / s)))) if s > 1.05 else (W, H)
    sc = 1.0 if s > 1.05 else s
    k_dark_c, k_thick_c, k_person_c = _ellipse(p.dark_dilate * sc), _ellipse(p.thick_dilate * sc), _ellipse(p.person_dilate * sc)
    k_zone = _ellipse(p.stay_zone * s) if p.stay_zone > 0 else None
    k_comp = _ellipse(5 * s)
    clear_px = p.stay_clear * s
    clear_pad = int(np.ceil(clear_px + 5 * s)) + 2
    k_max = np.ones((max(1, int(np.ceil(s))),) * 2, np.uint8)  # keep thin thick-marks when shrinking
    geom8 = BoardGeometry(info, corners, W8, p.decode_width) if s > 1.05 else geom
    k_drift = _ellipse(2 * p.drift_r * s + 1)
    k_group = _ellipse(p.group_dilate * s)
    k_touch = _ellipse(2 * p.touch_r * s + 1)
    margin = max(1, int(round(p.border_frac * W)))
    min_px = p.min_px * s * s
    pad = p.join_pad * s
    dt = 1.0 / p.fps

    accepted = np.zeros((H, W), bool)
    on = np.zeros((H, W), np.uint8)
    off = np.zeros((H, W), np.uint8)
    last_clean = np.zeros((H, W), np.float32)
    never_clean = np.ones((H, W), bool)
    acc_time = np.zeros((H, W), np.float32)
    cur_add = np.full((H, W), -1, np.int64)
    piece_img = np.zeros((H, W), np.float32)  # float so cv2.dilate can max it
    bg_ref = None
    # Thick ink that stays put: how many samples each thick pixel has kept its spot and grey.
    stay_run = np.zeros((H, W), np.uint8)
    stay_grey = np.zeros((H, W), np.uint8)  # its grey when last in view
    stay_box = None  # (x0, y0, x1, y1) around every nonzero run
    # Ink in view but held back by the thick mask alone: since when (inf = not held). The wait
    # before thick ink counts must not make it look written while hidden.
    held_t = np.full((H, W), np.inf, np.float32)
    held_box = None
    held_pad = int(np.ceil(p.thick_dilate * s / 2 + 2 * s)) + 2  # reach of the thick mask around thick ink

    pieces: dict[int, Piece] = {}
    next_piece = 1
    add_t, add_idx, add_piece, add_col = [], [], [], []
    add_seen = []  # per add: when it would have been accepted without the thick mask's wait
    add_drift = []  # True for ink re-accepted next to old ink (camera drift), not new writing
    n_adds = 0
    rem_add, rem_t = [], []  # removal events: which add, when the clean run began
    erases: list[dict] = []
    t_settle = None
    last_board = None
    steps = 0
    flat_x = np.arange(W)

    def new_piece(box, t, wa, pre=False) -> int:
        nonlocal next_piece
        pid = next_piece
        next_piece += 1
        pieces[pid] = Piece(pid, list(box), t, wa, pre, born=t)
        return pid

    alias: dict[int, int] = {}

    def resolve(pid: int) -> int:
        while pid in alias:
            pid = alias[pid]
        return pid

    def merge_into(src: int, dst: int):
        a, b = pieces.pop(src), pieces[dst]
        b.box = [min(a.box[0], b.box[0]), min(a.box[1], b.box[1]), max(a.box[2], b.box[2]), max(a.box[3], b.box[3])]
        b.written_after = min(a.written_after, b.written_after)
        b.last_update = max(a.last_update, b.last_update)
        b.born = min(a.born, b.born)
        alias[src] = dst
        piece_img[piece_img == src] = dst
        if debug:
            print(f"[board] merged piece {src} into {dst}")

    def touches(a, b, padding) -> bool:
        return not (a[2] + padding <= b[0] or b[2] + padding <= a[0] or a[3] + padding <= b[1] or b[3] + padding <= a[1])

    for t, frame in iter_frames(info, p.fps, geom.dw, s0, s1):
        board = geom.warp(frame)
        if t_settle is None:
            g = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
            if abs(float(np.median(_middle(g))) - typical) > p.settle_tol and t - s0 < p.settle_give_up:
                continue
            t_settle = t
            last_clean[:] = t
        steps += 1
        last_board = board
        b, g, r = cv2.split(board)
        if mode == "white":
            grey = cv2.min(cv2.min(b, g), r)
        else:
            grey = cv2.bitwise_not(cv2.max(cv2.max(b, g), r))
        coarse = s > 1.05
        if coarse:
            # The closing only estimates the clean board behind thin strokes, and the "in front"
            # masks are coarse: at wider analysis sizes compute them at the 800 px scale (a
            # fraction of the cost at 1200) and scale them back. Ink itself stays full size.
            small = cv2.resize(grey, (W8, H8), interpolation=cv2.INTER_AREA)
            closed_s = cv2.morphologyEx(small, cv2.MORPH_CLOSE, k_close_800)
            closed = cv2.max(cv2.resize(closed_s, (W, H), interpolation=cv2.INTER_LINEAR), grey)  # never darker
        else:
            closed_s = closed = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, k_close)
        contrast = closed.astype(np.int16) - grey.astype(np.int16)
        contrast[:margin] = 0; contrast[-margin:] = 0; contrast[:, :margin] = 0; contrast[:, -margin:] = 0
        ink = contrast > p.ink_thr
        faint = contrast > p.keep_thr

        # What's in front of the board: the person mask, then large dark things, then too-thick ink.
        closed_f = closed_s.astype(np.float32)
        if bg_ref is None:
            bg_ref = closed_f.copy()
        else:
            np.maximum(closed_f, bg_ref - p.bg_decay, out=bg_ref)
        front = cv2.dilate(((bg_ref - closed_f) > p.dark_thr).astype(np.uint8), k_dark_c)
        if person_mask is not None:
            pm = person_mask(t)
            if pm is not None:
                pw = (geom8 if coarse else geom).warp_mask(pm).astype(np.uint8)
                front = cv2.max(front, cv2.dilate(pw, k_person_c))
        # In view unless something other than too-thick ink is in front.
        visible_nt = (cv2.resize(front, (W, H), interpolation=cv2.INTER_NEAREST) if coarse else front) == 0
        dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
        core = dist > (p.max_stroke * s) / 2
        core_box = _rect(core)
        if p.stay_n > 0 and (core_box or stay_box):
            # Writing stays put; the teacher moves. A thick pixel in view that was thick when last
            # in view, with the same grey, extends its run; one that just turned thick starts at 1.
            # Behind the teacher the run waits, like the ink counters. (Only around thick pixels.)
            bx0, by0, bx1, by1 = _union(core_box, stay_box)
            sl = np.s_[by0:by1, bx0:bx1]
            run, vis, c = stay_run[sl], visible_nt[sl], core[sl]
            if core_box:
                start = c
                if p.stay_clear > 0:
                    # A count starts only while all of its thick shape is clear of the teacher and
                    # dark things (a still wrist, sleeve or fingers SAM missed are joined to them).
                    # A count that began while clear keeps going, so ink written just before the
                    # teacher came near still settles.
                    gx0, gy0 = max(0, bx0 - clear_pad), max(0, by0 - clear_pad)
                    gx1, gy1 = min(W, bx1 + clear_pad), min(H, by1 + clear_pad)
                    cc = cv2.dilate(core[gy0:gy1, gx0:gx1].view(np.uint8), k_comp)
                    _, lab = cv2.connectedComponents(cc, connectivity=8)
                    gap = cv2.distanceTransform(visible_nt[gy0:gy1, gx0:gx1].view(np.uint8), cv2.DIST_L2, 3)
                    bad = np.unique(lab[(gap <= clear_px) & (cc > 0)])
                    bad = bad[bad > 0]
                    if bad.size:
                        start = c & ~np.isin(lab[by0 - gy0:by1 - gy0, bx0 - gx0:bx1 - gx0], bad)
                # Compare contrast with the clean board, not raw grey: exposure jumps move both.
                con = np.clip(contrast[sl], 0, 255).astype(np.uint8)
                same = c & (run > 0) & (cv2.absdiff(con, stay_grey[sl]) <= p.stay_tol)
                np.copyto(run, np.where(same, cv2.add(run, 1), start.view(np.uint8)), where=vis)
                np.copyto(stay_grey[sl], con, where=vis)
            else:
                run[vis] = 0
            stay_box = _rect(run, bx0, by0)
            if core_box and stay_box:
                settled = run >= p.stay_n
                if k_zone is not None and settled.any():
                    settled = cv2.dilate(settled.view(np.uint8), k_zone) > 0
                c &= ~settled  # a view: thick ink that stays put leaves `core`
                core_box = _rect(core)
        thick = None
        if core_box:
            thick = core.astype(np.uint8)
            if coarse:
                thick = cv2.resize(cv2.dilate(thick, k_max), (W8, H8), interpolation=cv2.INTER_NEAREST)
            front = cv2.max(front, cv2.dilate(thick, k_thick_c))
        if coarse:
            front = cv2.resize(front, (W, H), interpolation=cv2.INTER_NEAREST)
        visible = (front == 0) if thick is not None else visible_nt

        # Drift tolerance: accepted pixels stay ink while faint ink lies within ~2 px.
        near = cv2.dilate(faint.astype(np.uint8), k_drift) > 0
        is_ink = np.where(accepted, near, ink)
        upd_ink = visible & is_ink
        upd_clean = visible & ~is_ink
        # Ink held back by the thick mask alone: remember since when, while it stays in view.
        if held_box:
            sl = np.s_[held_box[1]:held_box[3], held_box[0]:held_box[2]]
            np.copyto(held_t[sl], np.float32(np.inf), where=~(visible_nt[sl] & is_ink[sl]))
            if np.isinf(held_t[sl]).all():
                held_box = None
        if thick is not None:
            x0, y0, x1, y1 = core_box
            x0, y0, x1, y1 = max(0, x0 - held_pad), max(0, y0 - held_pad), min(W, x1 + held_pad), min(H, y1 + held_pad)
            sl = np.s_[y0:y1, x0:x1]
            hold = visible_nt[sl] & ~visible[sl] & is_ink[sl] & ~accepted[sl] & np.isinf(held_t[sl])
            if hold.any():
                held_t[sl][hold] = t
                held_box = _union(held_box, _rect(hold, x0, y0))
        on = np.where(upd_ink, cv2.add(on, 1), np.where(upd_clean, np.uint8(0), on))
        off = np.where(upd_clean, cv2.add(off, 1), np.where(upd_ink, np.uint8(0), off))
        np.copyto(last_clean, np.float32(t), where=upd_clean)
        np.copyto(never_clean, False, where=upd_clean)

        # ---- removals ----
        gone = accepted & (off >= p.remove_n)
        if gone.any():
            t_gone = t - (p.remove_n - 1) * dt  # when the clean run began
            ys, xs = np.nonzero(gone)
            rem_add.append(cur_add[ys, xs].copy())
            rem_t.append(np.full(ys.size, t_gone, np.float32))
            accepted[gone] = False
            piece_img[gone] = 0
            cur_add[gone] = -1
            # erase components (in the region around the removed pixels only)
            ry0, ry1, rx0, rx1 = _roi(ys, xs, k_group.shape[0], H, W)
            sub = gone[ry0:ry1, rx0:rx1].astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(sub, k_group), connectivity=8)
            labs = lab[ys - ry0, xs - rx0]
            for c in range(1, n):
                sel = labs == c
                cnt = int(sel.sum())
                if cnt == 0:
                    continue
                x0, y0, w, h = stats[c, :4]
                box = [int(xs[sel].min()), int(ys[sel].min()), int(xs[sel].max()) + 1, int(ys[sel].max()) + 1]
                adds = rem_add[-1][sel]
                # Merge with every recent erase it touches (it may bridge two of them).
                hit = [e for e in erases if t_gone - e["t_last"] <= p.erase_merge and touches(e["box"], box, pad)]
                if not hit:
                    erases.append({"t": t_gone, "t_last": t_gone, "box": box, "adds": [adds]})
                    continue
                into = hit[0]
                for e in hit[1:]:
                    into["adds"] += e["adds"]
                    into["t"] = min(into["t"], e["t"])
                    box = [min(box[0], e["box"][0]), min(box[1], e["box"][1]),
                           max(box[2], e["box"][2]), max(box[3], e["box"][3])]
                    erases.remove(e)
                into["box"] = [min(into["box"][0], box[0]), min(into["box"][1], box[1]),
                               max(into["box"][2], box[2]), max(into["box"][3], box[3])]
                into["adds"].append(adds)
                into["t_last"] = t_gone

        # ---- acceptances ----
        new = ~accepted & (on >= p.accept_n)
        if new.any():
            ys, xs = np.nonzero(new)
            nnew = ys.size
            ids = np.arange(n_adds, n_adds + nnew)
            n_adds += nnew
            pid = np.zeros(nnew, np.int64)
            drift = np.zeros(nnew, bool)
            stale = np.zeros(nnew, bool)  # small bits that joined an old piece: not an update
            pre_window = t - t_settle <= p.pre_window

            # Drift: ink within ~2 px of accepted ink. Decided per component below: a component
            # that is mostly such ink is drift and joins its owners; otherwise it is new writing
            # (new writing that merely touches old writing must not be swallowed by it).
            # Work in the region around the new pixels (padded by the largest kernel used below).
            ry0, ry1, rx0, rx1 = _roi(ys, xs, max(k_group.shape[0], k_touch.shape[0]) + 2, H, W)
            ly, lx = ys - ry0, xs - rx0
            pimg = piece_img[ry0:ry1, rx0:rx1]
            owner = np.zeros(nnew, np.int64)
            if not pre_window and accepted.any():
                owner = cv2.dilate(pimg, k_drift)[ly, lx].astype(np.int64)
            if True:
                rmask = np.zeros(pimg.shape, np.uint8)
                rmask[ly, lx] = 1
                dil = cv2.dilate(rmask, k_group)
                n, lab, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
                ri = np.arange(nnew)
                labs = lab[ly, lx]
                order = np.argsort(labs, kind="stable")
                labs_sorted = labs[order]
                bounds = np.searchsorted(labs_sorted, np.arange(1, n + 1))
                touch_img = None
                for c in range(1, n):
                    members = ri[order[bounds[c - 1]:bounds[c]]]
                    if members.size == 0:
                        continue
                    own = owner[members]
                    if (own > 0).mean() >= p.drift_share:
                        # Drift: each pixel keeps its owner; the rest go with the majority.
                        major = int(np.bincount(own[own > 0]).argmax())
                        pid[members] = np.where(own > 0, own, major)
                        drift[members] = True
                        continue
                    cx, cy = xs[members], ys[members]
                    box = [int(cx.min()), int(cy.min()), int(cx.max()) + 1, int(cy.max()) + 1]
                    wa = float(np.median(last_clean[cy, cx]))
                    target = 0
                    bx, by, bw, bh = stats[c, :4]
                    under = pimg[by:by + bh, bx:bx + bw][lab[by:by + bh, bx:bx + bw] == c]
                    under = under[under > 0].astype(np.int64)
                    if pre_window:
                        # Writing already there: group by touching ink only, so a full board
                        # doesn't chain into one giant piece.
                        pre_under = [int(u) for u in np.unique(under) if pieces[int(u)].pre_existing]
                        target = pre_under[0] if pre_under else new_piece(box, t, t_settle, pre=True)
                    elif never_clean[cy, cx].all() and under.size and any(pieces[u].pre_existing for u in np.unique(under)):
                        target = next(int(u) for u in np.unique(under) if pieces[u].pre_existing)
                    if not target and not pre_window and members.size < min_px:
                        # Small bits join a piece whose ink they touch, however old.
                        if touch_img is None:
                            touch_img = cv2.dilate(pimg, k_touch)
                        near = touch_img[cy - ry0, cx - rx0].astype(np.int64)
                        if (near > 0).any():
                            target = int(np.bincount(near[near > 0]).argmax())
                            if t - pieces[target].last_update > p.join_recent:
                                # Joining an old piece must not make it "recent", or the rest of
                                # new writing that starts next to it would chain into it.
                                stale[members] = True
                    if not target and not pre_window:
                        hits = [q for q in pieces.values() if not q.pre_existing
                                and t - q.last_update <= p.join_recent and touches(q.box, box, pad)]
                        if hits:
                            best = max(hits, key=lambda q: (q.last_update, q.id))
                            target = best.id
                            # A stroke whose middle was hidden (by the hand) shows up as separate
                            # bits: a component bridging young pieces merges them.
                            for q in hits:
                                if q.id != target and t - q.born <= p.join_recent:
                                    merge_into(q.id, target)
                    if not target:
                        target = new_piece(box, t, wa)
                        if debug:
                            near = [(q.id, q.box, round(q.last_update, 2)) for q in pieces.values()
                                    if q.id != target and touches(q.box, box, 3 * pad)]
                            print(f"[board] t={t:.2f} new piece {target} box={box} n={members.size} "
                                  f"wa={wa:.2f} nearby={near}")
                    elif debug:
                        print(f"[board] t={t:.2f} join {target} box={box} n={members.size} "
                              f"last_update={pieces[target].last_update:.2f} piece_box={pieces[target].box}")
                    pid[members] = target
                    q = pieces[target]
                    q.written_after = min(q.written_after, wa)
                    # Grow the box now, so later components of this step see it.
                    q.box = [min(q.box[0], box[0]), min(q.box[1], box[1]), max(q.box[2], box[2]), max(q.box[3], box[3])]
                    if members.size >= p.update_min_px * s * s and not stale[members[0]]:
                        q.last_update = t

            pid = np.array([resolve(int(v)) for v in pid], np.int64) if alias else pid
            for u in np.unique(pid):
                q = pieces[int(u)]
                sel = pid == u
                if debug and t - q.last_update > 1.0:
                    print(f"[board] t={t:.2f} piece {q.id} +{int(sel.sum())} px after {t - q.last_update:.2f}s "
                          f"at x={int(xs[sel].min())}-{int(xs[sel].max())} y={int(ys[sel].min())}-{int(ys[sel].max())} "
                          f"last_clean median={float(np.median(last_clean[ys[sel], xs[sel]])):.2f}")
                q.box = [min(q.box[0], int(xs[sel].min())), min(q.box[1], int(ys[sel].min())),
                         max(q.box[2], int(xs[sel].max()) + 1), max(q.box[3], int(ys[sel].max()) + 1)]
                # Only real writing counts as an update: not a stray edge pixel or two, and not
                # ink re-accepted after the camera drifted.
                if (sel & ~drift & ~stale).sum() >= p.update_min_px * s * s:
                    q.last_update = t

            accepted[ys, xs] = True
            acc_time[ys, xs] = t
            cur_add[ys, xs] = ids
            piece_img[ys, xs] = pid.astype(np.float32)
            add_seen.append(np.minimum(np.float32(t), held_t[ys, xs] + np.float32((p.accept_n - 1) * dt)))
            held_t[ys, xs] = np.inf
            add_t.append(np.full(nnew, t, np.float32))
            add_idx.append((ys * W + xs).astype(np.int64))
            add_piece.append(pid)
            add_drift.append(drift)
            add_col.append(board[ys, xs].copy())

        if progress is not None and steps % 30 == 0:
            progress(min(1.0, (t - s0) / max(1e-6, s1 - s0)))

    if t_settle is None or last_board is None:
        t_settle = s0
        last_board = np.full((H, W, 3), 255 if mode == "white" else 30, np.uint8)

    add_piece = [np.array([resolve(int(v)) for v in a], np.int64) if alias else a for a in add_piece]
    res = _finish(p, geom, mode, pieces, erases, add_t, add_idx, add_piece, add_col, add_drift, rem_add, rem_t,
                  accepted, last_board, t_settle, s1, add_seen)
    res.stats.update(steps=steps, seconds=time.perf_counter() - t_wall, typical_brightness=typical,
                     clip_seconds=s1 - s0, speed=(s1 - s0) / max(1e-6, time.perf_counter() - t_wall))
    if progress is not None:
        progress(1.0)
    return res


def _cat(lst, dtype, shape=(0,)):
    return np.concatenate(lst).astype(dtype) if lst else np.zeros(shape, dtype)


def _finish(p: BoardParams, geom: BoardGeometry, mode, pieces, erases, add_t, add_idx, add_piece, add_col, add_drift,
            rem_add, rem_t, accepted, last_board, t_settle, t_end, add_seen=None) -> BoardResult:
    W, H = geom.width, geom.height
    s = W / 800.0
    A_t = _cat(add_t, np.float64)
    A_seen = _cat(add_seen, np.float64) if add_seen else A_t
    A_idx = _cat(add_idx, np.int64)
    A_pid = _cat(add_piece, np.int64)
    A_drift = _cat(add_drift, bool)
    A_col = _cat(add_col, np.uint8, (0, 3))
    R_add = _cat(rem_add, np.int64)
    R_t = _cat(rem_t, np.float64)
    n = A_t.size
    removed_at = np.full(n, np.inf)
    if R_add.size:
        removed_at[R_add] = R_t
    life = removed_at - A_t
    short = life < p.min_life
    shown_removed, replaced = _display_removals(A_t, A_idx, A_drift, removed_at, W, H, s, p)

    bx, by = p.band_frac * W, p.band_frac * H
    out_pieces = []
    keep_piece = {}
    order = np.argsort(A_pid, kind="stable")
    pid_sorted = A_pid[order]
    uniq, starts = np.unique(pid_sorted, return_index=True)
    ends = list(starts[1:]) + [pid_sorted.size]
    for u, a, b in zip(uniq, starts, ends):
        members = order[a:b]
        q = pieces[int(u)]
        uniq_px = np.unique(A_idx[members])
        ys, xs = np.divmod(uniq_px, W)
        npx = uniq_px.size
        reason = None
        if npx < p.min_px * s * s:
            reason = "small"
        elif short[members].mean() > p.short_life_frac:
            reason = "short-lived"
        else:
            pts = np.stack([xs, ys], 1).astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            length = max(rw, rh) + 1
            thick = npx / length
            if thick <= p.hair_thick * s and length >= p.hair_ratio * thick:
                reason = "hairline"
            elif ((xs < bx) | (xs >= W - bx) | (ys < by) | (ys >= H - by)).all():
                reason = "edge band"
        kept = members[~short[members]]
        if reason is None and kept.size == 0:
            reason = "short-lived"
        keep_piece[int(u)] = reason is None
        if reason is not None:
            continue
        # Timing from writing, not from ink re-accepted after the camera drifted.
        written = kept[~A_drift[kept]]
        times = np.sort(A_t[written if written.size else kept])
        t_seen = float(times[int(np.floor(p.seen_frac * (times.size - 1)))])
        t_shown = float(times[int(np.floor(0.95 * (times.size - 1)))])
        # Written while hidden is judged from when the ink was in view, before any wait for
        # thick ink to show it stays put.
        in_view = np.sort(A_seen[written if written.size else kept])
        t_view = float(in_view[int(np.floor(p.seen_frac * (in_view.size - 1)))])
        shown = kept[~A_drift[kept]] if (~A_drift[kept]).any() else kept
        rem = shown_removed[shown]
        removed = rem[np.isfinite(rem)]
        rem_when = float(np.median(removed)) if removed.size >= 0.5 * shown.size and removed.size else None
        kys, kxs = np.divmod(A_idx[kept], W)
        if q.pre_existing:
            t_seen = t_shown = float(t_settle)
        wa = float(q.written_after)
        out_pieces.append({
            "id": int(u), "box": [int(kxs.min()), int(kys.min()), int(kxs.max() - kxs.min() + 1), int(kys.max() - kys.min() + 1)],
            "t_seen": round(t_seen, 2), "t_commit": round(t_seen, 2), "t_shown": round(t_shown, 2),
            "written_after": round(wa, 2),
            "revealed": bool(not q.pre_existing and t_view - wa > p.reveal_gap),
            "removed_at": None if rem_when is None else round(rem_when, 2),
            "pre_existing": bool(q.pre_existing), "pixels": int(kept.size),
        })

    out_pieces.sort(key=lambda d: (d["t_seen"], d["box"][0]))

    out_erases = []
    debug = bool(os.environ.get("RB_DEBUG_BOARD"))
    for e in erases:
        adds = np.concatenate(e["adds"])
        if debug and adds.size >= 20:
            print(f"[board] erase candidate t={e['t']:.2f}-{e['t_last']:.2f} box={e['box']} px={adds.size} "
                  f"short={short[adds].mean():.2f}")
        if adds.size < p.erase_min_px * s * s:
            continue
        if short[adds].mean() > p.short_life_frac:
            continue
        if replaced[adds].mean() > 0.5:
            continue  # the camera moved; the ink came back next to where it was
        x0, y0, x1, y1 = e["box"]
        out_erases.append({"t": round(float(e["t"]), 2), "box": [x0, y0, x1 - x0, y1 - y0], "pixels": int(adds.size)})

    keep = ~short & np.array([keep_piece.get(int(v), False) for v in A_pid], bool) if n else np.zeros(0, bool)
    kept_ids = np.nonzero(keep)[0]
    o = np.argsort(A_t[kept_ids], kind="stable")
    kept_ids = kept_ids[o]
    col = np.clip(A_col[kept_ids].astype(np.float32) * p.darken, 0, 255).astype(np.uint8)
    rmask = np.isfinite(removed_at[kept_ids])
    rem_ids = kept_ids[rmask]
    ro = np.argsort(removed_at[rem_ids], kind="stable")
    rem_ids = rem_ids[ro]

    # Board colour: median of the clean, visible middle of the last frame.
    mid = last_board[H // 4: H - H // 4, W // 4: W - W // 4].reshape(-1, 3)
    mask = ~accepted[H // 4: H - H // 4, W // 4: W - W // 4].reshape(-1)
    bg = tuple(int(v) for v in np.median(mid[mask] if mask.any() else mid, axis=0))

    return BoardResult(
        width=W, height=H, mode=mode, corners=[[round(float(x), 1), round(float(y), 1)] for x, y in geom.corners_video],
        background=bg, t_start=float(t_settle), t_end=float(t_end), fps=p.fps,
        pieces=out_pieces, erases=out_erases,
        add_t=A_t[kept_ids], add_idx=A_idx[kept_ids], add_piece=A_pid[kept_ids], add_color=col,
        rem_t=removed_at[rem_ids], rem_idx=A_idx[rem_ids],
        add_drift=A_drift[kept_ids], add_gone=shown_removed[kept_ids],
        stats={"pieces_raw": len(pieces), "pieces_kept": len(out_pieces), "erases": len(out_erases)},
    )


def _display_removals(A_t, A_idx, A_drift, removed_at, W, H, s, p):
    """When each mark should disappear from the replayed board.

    A removal followed within a few seconds by drift-accepted ink within ~2 px is the camera
    moving, not an erase: the original mark stays, and inherits the removal time of the ink that
    replaced it (newest first, so chains of drift resolve). Returns (times, replaced flags).
    """
    shown = removed_at.copy()
    replaced = np.zeros(removed_at.size, bool)
    drift_ids = np.nonzero(A_drift)[0]
    rem_ids = np.nonzero(np.isfinite(removed_at))[0]
    if drift_ids.size == 0 or rem_ids.size == 0:
        return shown, replaced
    d_order = drift_ids[np.argsort(A_t[drift_ids], kind="stable")]
    d_times = A_t[d_order]
    k = _ellipse(2 * p.drift_r * s + 1)
    rem_times = removed_at[rem_ids]
    for tr in np.unique(rem_times)[::-1]:
        lo, hi = np.searchsorted(d_times, [tr - 1.0, tr + 4.0])
        if hi <= lo:
            continue
        cand = d_order[lo:hi]
        img = np.zeros(W * H, np.float32)
        img[A_idx[cand]] = (cand + 1).astype(np.float32)  # ids up to 2^24 are exact in float32
        near = cv2.dilate(img.reshape(H, W), k).reshape(-1)
        batch = rem_ids[rem_times == tr]
        by = near[A_idx[batch]].astype(np.int64) - 1
        hit = by >= 0
        replaced[batch[hit]] = True
        shown[batch[hit]] = shown[by[hit]]
    return shown, replaced
