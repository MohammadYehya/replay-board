"""Slide analysis (SPEC 5.2): live regions, slide states, clean slides, pointer track, gesture guesses.

Everything runs on one working copy (12 fps, 960 px wide), so the original is decoded once.
Pixel sizes are for 960 px and scale with s = width / 960.
"""
from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .media import MediaError, VideoInfo, _tool, clip_range, iter_frames, probe


@dataclass(frozen=True)
class SlideParams:
    width: int = 960
    work_fps: float = 12.0
    live_fps: float = 2.0
    state_fps: float = 4.0
    # live regions
    busy_diff: float = 12
    busy_share: float = 0.35
    quiet_screen: float = 0.05
    flicker_diff: float = 6
    flicker_share: float = 0.10
    busy_min_area: float = 0.004
    flicker_min_area: float = 0.01
    flicker_fill: float = 0.40
    grow_max: float = 0.15
    strip_bg_diff: float = 25
    strip_bg_share: float = 0.50
    strip_flicker: float = 0.02
    strip_flicker_share: float = 0.10
    # states
    change_diff: float = 30
    change_area: float = 1500
    change_side: float = 60
    group_px: float = 11  # changed pixels this close form one component (glyphs -> words)
    settle_s: float = 0.5
    new_slide_screen: float = 0.25
    new_slide_text: float = 0.25
    text_thr: float = 40
    median_samples: int = 11
    blank_std: float = 4
    # pointer
    ptr_diff: float = 30
    laser_comp: float = 60
    laser_bright: float = 170
    laser_sat: float = 120
    laser_box: float = 30
    laser_fill: float = 0.30
    laser_aspect: float = 3.0
    cur_min: float = 25
    cur_max: float = 700
    cur_side: float = 60
    shapes_max: int = 6
    shape_min_px: int = 20
    shape_min_std: float = 40
    app_good: float = 0.12
    app_ok: float = 0.35
    near_px: float = 60
    min_contrast: float = 100
    lost_s: float = 0.5
    park_match: float = 0.08
    match_pad: int = 8
    parked_rms: float = 0.08
    seen_app: float = 0.2  # a copied sighting must look this much like a learned cursor
    inpaint_rounds: int = 3
    # gestures
    gap_s: float = 0.35
    move_speed: float = 55
    hover_s: float = 0.6
    parked_s: float = 8.0
    content_px: float = 40
    circle_max_s: float = 4.0
    circle_turn: float = 300
    circle_consistency: float = 1.35
    circle_min_side: float = 18
    circle_gap: float = 0.6
    wiggle_s: tuple = (0.4, 2.0)
    wiggle_box: tuple = (12, 150)
    wiggle_reversals: int = 3
    wiggle_path: float = 2.5
    wiggle_speed: float = 80
    rdp_eps: float = 8
    sweep_straight: float = 0.85
    sweep_ratio: float = 2.5
    sweep_min: float = 80
    sweep_max_speed: float = 700
    sweep_min_s: float = 0.35
    move_straight: float = 0.75
    move_min: float = 60
    arrow_rest: float = 0.25
    underline_gap: float = 4


@dataclass
class SlideState:
    id: int
    kind: str  # "slide" or "addition"
    t_start: float  # when the content stopped changing
    t_end: float
    t_change: float  # when the change began (transition start)
    image: np.ndarray | None = None  # clean slide (BGR), None for blank states
    blank: bool = False
    samples: list = field(default_factory=list, repr=False)


@dataclass
class SlidesResult:
    width: int
    height: int
    live: list  # [x, y, w, h] boxes, working-copy pixels
    background: tuple
    states: list  # SlideState
    track: list  # [(t, tool, x, y)]
    gestures: list  # dicts
    transitions: list  # [(t0, t1)]
    stats: dict = field(default_factory=dict)


# ------------------------------------------------------------------ working copy

def make_working_copy(info: VideoInfo, out: Path, start: float, end: float, p: SlideParams) -> Path:
    """Cut, 12 fps, 960 px wide, x264 crf 12 yuv444p. Every later pass reads this."""
    if out.exists() and out.stat().st_size > 0:
        return out
    w = p.width
    h = max(2, int(round(w * info.height / info.width / 2)) * 2)
    tmp = out.with_suffix(".part.mp4")
    args = [_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", str(info.path),
            "-t", f"{end - start:.3f}", "-an", "-sn", "-vf", f"fps={p.work_fps}:round=up,scale={w}:{h}:flags=area,setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv444p", str(tmp)]
    res = subprocess.run(args, capture_output=True)
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", "replace").strip().splitlines()
        raise MediaError(f"could not make the working copy: {err[-1] if err else 'ffmpeg error'}")
    tmp.replace(out)
    return out


def _frames(work: VideoInfo, fps: float, width: int):
    """Frames from the working copy at `fps` (a divisor of 12 keeps exact samples)."""
    for t, f in iter_frames(work, fps, width):
        yield t, f


# ------------------------------------------------------------------ live regions

def _absdiff_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(a, b)
    return cv2.max(cv2.max(d[..., 0], d[..., 1]), d[..., 2])


def find_live_regions(work: VideoInfo, p: SlideParams, bg: np.ndarray) -> tuple[list, np.ndarray, np.ndarray]:
    """Webcam / embedded video boxes from 2 fps busy and flicker statistics."""
    s = p.width / 960.0
    prev = None
    busy = flick = None
    pairs = quiet = 0
    for _, f in _frames(work, p.live_fps, p.width):
        if prev is not None:
            d = _absdiff_max(f, prev)
            if busy is None:
                busy = np.zeros(d.shape, np.float32)
                flick = np.zeros(d.shape, np.float32)
            ch = d > p.busy_diff
            busy += ch
            pairs += 1
            if ch.mean() < p.quiet_screen:
                flick += d > p.flicker_diff
                quiet += 1
        prev = f
    if busy is None:
        h, w = prev.shape[:2] if prev is not None else (1, 1)
        return [], np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    busy /= max(1, pairs)
    flick /= max(1, quiet)
    H, W = busy.shape
    area = H * W
    boxes = []
    core = (busy > p.busy_share).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _ell(15 * s))
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, _ell(5 * s))
    n, _, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= p.busy_min_area * area:
            boxes.append(list(map(int, st[i, :4])))
    if quiet:
        fl = (flick > p.flicker_share).astype(np.uint8)
        fl = cv2.morphologyEx(fl, cv2.MORPH_CLOSE, _ell(9 * s))
        n, _, st, _ = cv2.connectedComponentsWithStats(fl, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a >= p.flicker_min_area * area and a >= p.flicker_fill * w * h:
                boxes.append([int(x), int(y), int(w), int(h)])
    boxes = _merge_boxes(boxes)
    grown = [_grow_box(b, bg, flick, p) for b in boxes]
    return _merge_boxes(grown), busy, flick


def _grow_box(b, bg: np.ndarray, flick: np.ndarray, p: SlideParams) -> list:
    """Grow each side, by at most 15% of the size, to the furthest strip that still belongs."""
    H, W = flick.shape
    x0, y0, w, h = b
    x1, y1 = x0 + w, y0 + h
    bgv = np.median(bg.reshape(-1, 3), axis=0)
    diff = np.abs(bg.astype(np.int16) - bgv.astype(np.int16)).max(axis=2) > p.strip_bg_diff

    def belongs(ys, xs):
        if ys.size == 0:
            return False
        return (diff[ys, xs].mean() > p.strip_bg_share
                or (flick[ys, xs] > p.strip_flicker).mean() > p.strip_flicker_share)

    def furthest(max_k, strip):
        best = 0
        for k in range(1, max_k + 1):
            ys, xs = strip(k)
            if belongs(ys, xs):
                best = k
        return best

    mx, my = int(math.ceil(p.grow_max * w)), int(math.ceil(p.grow_max * h))
    xs_in, ys_in = np.arange(x0, x1), np.arange(y0, y1)
    left = furthest(min(mx, x0), lambda k: (ys_in, np.full(ys_in.size, x0 - k)))
    right = furthest(min(mx, W - x1), lambda k: (ys_in, np.full(ys_in.size, x1 - 1 + k)))
    top = furthest(min(my, y0), lambda k: (np.full(xs_in.size, y0 - k), xs_in))
    bottom = furthest(min(my, H - y1), lambda k: (np.full(xs_in.size, y1 - 1 + k), xs_in))
    nx0, ny0 = x0 - left, y0 - top
    return [nx0, ny0, (x1 + right) - nx0, (y1 + bottom) - ny0]


def _merge_boxes(boxes: list) -> list:
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] <= b[0] + b[2] and b[0] <= a[0] + a[2] and a[1] <= b[1] + b[3] and b[1] <= a[1] + a[3]:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def _ell(d: float) -> np.ndarray:
    d = max(1, int(round(d)))
    d += 1 - d % 2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _live_mask(shape, live: list) -> np.ndarray:
    m = np.zeros(shape[:2], bool)
    for x, y, w, h in live:
        m[y:y + h, x:x + w] = True
    return m


# ------------------------------------------------------------------ slide states

def _content_change(a: np.ndarray, b: np.ndarray, ignore: np.ndarray, p: SlideParams, s: float) -> np.ndarray:
    """Mask of changed components big enough to be content (not a cursor)."""
    ch = (_absdiff_max(a, b) > p.change_diff) & ~ignore
    if not ch.any():
        return ch
    # Group nearby changes first: new text appears as separate glyphs, each far below the
    # size rule, while a line or word of them is clearly content.
    grouped = cv2.dilate(ch.astype(np.uint8), _ell(p.group_px * s))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    lab_ch = lab[ch]
    if lab_ch.size == 0:
        return np.zeros_like(ch)
    count = np.bincount(lab_ch, minlength=n)
    ys, xs = np.nonzero(ch)
    x0 = np.full(n, 1 << 30); x1 = np.full(n, -1); y0 = np.full(n, 1 << 30); y1 = np.full(n, -1)
    np.minimum.at(x0, lab_ch, xs); np.maximum.at(x1, lab_ch, xs)
    np.minimum.at(y0, lab_ch, ys); np.maximum.at(y1, lab_ch, ys)
    side = np.maximum(x1 - x0, y1 - y0) + 1
    keep = (count >= p.change_area * s * s) | (side >= p.change_side * s)
    keep[0] = False
    return ch & keep[lab]


def _text_pixels(img: np.ndarray, p: SlideParams) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.abs(g.astype(np.int16) - cv2.medianBlur(g, 31).astype(np.int16)) > p.text_thr


def find_states(work: VideoInfo, live: list, bg_colour, p: SlideParams,
                progress: Callable[[float], None] | None = None) -> tuple[list, list]:
    """Slide states at 4 fps with clean (median) images. Returns (states, transitions)."""
    s = p.width / 960.0
    ignore = None
    states: list[SlideState] = []
    transitions = []
    cur: SlideState | None = None
    ref = None  # reference frame of the current state
    prev = None
    changing_since = None
    last_change = None
    dt = 1.0 / p.state_fps
    t = 0.0
    for t, f in _frames(work, p.state_fps, p.width):
        if ignore is None:
            ignore = _live_mask(f.shape, live)
        if cur is None:
            cur = SlideState(0, "slide", t, t, t)
            ref = f
            prev = f
            _sample(cur, f, t)
            continue
        moved = _content_change(f, prev, ignore, p, s).any()
        prev = f
        if not moved and changing_since is None and _content_change(f, ref, ignore, p, s).any():
            # A slow change (ink drawn stroke by stroke, a slow fade) never jumps between two
            # frames: catch it against the state's reference instead.
            moved = True
        if moved:
            if changing_since is None:
                changing_since = t
            last_change = t
            continue
        if changing_since is not None:
            if t - last_change < p.settle_s - 1e-6:
                continue
            # Settled: compare with the current state's reference.
            diff = _content_change(f, ref, ignore, p, s)
            changing_since_ = changing_since
            changing_since = None
            if not diff.any():
                transitions.append((changing_since_ - dt, last_change + dt))
                continue
            text = _text_pixels(ref, p) & ~ignore
            screen = diff.mean()
            text_frac = (diff & text).sum() / max(1, text.sum())
            kind = "slide" if (screen >= p.new_slide_screen or text_frac >= p.new_slide_text) else "addition"
            t_new = last_change + dt  # first frame after the content stopped changing
            cur.t_end = changing_since_
            _close_state(cur, p, bg_colour, ignore)
            states.append(cur)
            transitions.append((changing_since_ - dt, t_new))
            cur = SlideState(len(states), kind, t_new, t_new, changing_since_)
            ref = f
        _sample(cur, f, t)
        if progress and len(states) % 5 == 0:
            progress(0.0)
    if cur is not None:
        cur.t_end = t + dt
        _close_state(cur, p, bg_colour, ignore if ignore is not None else np.zeros((1, 1), bool))
        states.append(cur)
    return states, transitions


def _sample(st: SlideState, f: np.ndarray, t: float):
    """Keep a bounded, evenly thinned set of frames for the median."""
    st.samples.append((t, f))
    if len(st.samples) > 44:
        st.samples = st.samples[::2]


def _close_state(st: SlideState, p: SlideParams, bg_colour, ignore: np.ndarray):
    smp = st.samples
    if not smp:
        return
    k = min(p.median_samples, len(smp))
    idx = np.linspace(0, len(smp) - 1, k).round().astype(int)
    stack = np.stack([smp[i][1] for i in idx])
    img = np.median(stack, axis=0).astype(np.uint8)
    if ignore is not None and ignore.shape == img.shape[:2]:
        img[ignore] = bg_colour
    st.image = img
    st.blank = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < p.blank_std
    st.samples = []


# ------------------------------------------------------------------ pointer

@dataclass
class Shape:
    patch: np.ndarray  # grey, float32
    mask: np.ndarray  # bool
    hot: tuple  # (dx, dy) of the pointer tip inside the patch


class PointerTracker:
    def __init__(self, p: SlideParams):
        self.p = p
        self.s = p.width / 960.0
        self.shapes: list[Shape] = []
        self.last = None  # (t, x, y, box)
        self.track: list = []
        self.seen: dict = {}  # time -> (x0, y0, grey crop) around each cursor sighting

    # -- appearance
    def _match(self, grey: np.ndarray, shape: Shape, x0: int, y0: int, pad: int):
        """Best masked match of a shape around (x0, y0): returns (appearance, x, y) of the patch corner."""
        h, w = shape.patch.shape
        H, W = grey.shape
        ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
        ax1, ay1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
        if ax1 - ax0 < w or ay1 - ay0 < h:
            return 1.0, x0, y0
        area = grey[ay0:ay1, ax0:ax1]
        m = shape.mask.astype(np.float32)
        res = cv2.matchTemplate(area, shape.patch, cv2.TM_SQDIFF, mask=m)
        res = np.nan_to_num(res, nan=1e18, posinf=1e18)
        _, _, loc, _ = cv2.minMaxLoc(res)
        bx, by = ax0 + loc[0], ay0 + loc[1]
        crop = grey[by:by + h, bx:bx + w]
        app = float(np.abs(crop - shape.patch)[shape.mask].mean() / 255.0)
        return app, bx, by

    def _learn(self, grey: np.ndarray, box, changed: np.ndarray):
        x, y, w, h = box
        patch = grey[y:y + h, x:x + w].copy()
        mask = changed[y:y + h, x:x + w].copy()
        if mask.sum() < self.p.shape_min_px or patch[mask].std() < self.p.shape_min_std:
            return None
        ys, xs = np.nonzero(mask)
        k = int(np.argmin(ys * 10000 + xs))
        sh = Shape(patch, mask, (int(xs[k]), int(ys[k])))
        if len(self.shapes) >= self.p.shapes_max:
            self.shapes.pop(0)
        self.shapes.append(sh)
        return sh

    def best_shape(self, grey, box):
        best = (1.0, None, box[0], box[1])
        for sh in self.shapes:
            app, bx, by = self._match(grey, sh, box[0], box[1], self.p.match_pad)
            if app < best[0]:
                best = (app, sh, bx, by)
        return best

    # -- one frame
    def step(self, t: float, f: np.ndarray, clean: np.ndarray, ignore: np.ndarray, in_transition: bool):
        p, s = self.p, self.s
        if in_transition:
            return None
        d = _absdiff_max(f, clean)
        changed = (d > p.ptr_diff) & ~ignore
        grey = None
        cands, lasers = [], []
        if changed.any():
            n, lab, st, cen = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                x, y, w, h, a = (int(v) for v in st[i])
                if max(w, h) > max(p.laser_comp, p.cur_side) * s:
                    continue
                own = lab[y:y + h, x:x + w] == i
                if max(w, h) <= p.laser_comp * s:
                    lz = self._laser_blob(f[y:y + h, x:x + w], own, x, y, s)
                    if lz is not None:
                        lasers.append(lz)
                if p.cur_min * s * s <= a <= p.cur_max * s * s and max(w, h) <= p.cur_side * s:
                    contrast = float(d[y:y + h, x:x + w][own].max())
                    cands.append({"box": (x, y, w, h), "c": (float(cen[i][0]), float(cen[i][1])),
                                  "contrast": contrast, "lab": i})
        if lasers or cands or self.last is not None:
            grey = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if lasers:
            k, lx, ly = max(lasers)
            # Coloured ink next to a moving cursor is not a laser.
            beside = [c for c in cands if self.shapes and math.hypot(c["c"][0] - lx, c["c"][1] - ly) <= 30 * s
                      and self.best_shape(grey, c["box"])[0] <= p.app_good]
            if not beside:
                self.track.append((t, "laser", lx, ly))
                return ("laser", lx, ly)
        lost = self.last is None or t - self.last[0] > p.lost_s
        pick = None
        if cands and self.shapes and not lost:
            lx, ly = self.last[1], self.last[2]
            scored = []
            for c in cands:
                app, sh, bx, by = self.best_shape(grey, c["box"])
                dist = math.hypot(c["c"][0] - lx, c["c"][1] - ly)
                scored.append((app, dist, c, sh, bx, by))
            good = [x for x in scored if x[0] <= p.app_good]
            if good:
                pick = min(good, key=lambda x: x[1])
            else:
                cand = min(scored, key=lambda x: x[0] + x[1] / 900.0)
                if cand[1] <= p.near_px * s or (cand[0] <= p.app_ok and cand[2]["contrast"] >= p.min_contrast):
                    pick = cand
                    if cand[0] > p.app_good:
                        self._learn(grey, cand[2]["box"], changed)
        if pick is None and cands and (lost or not self.shapes):
            c = max(cands, key=lambda c: c["contrast"])
            if c["contrast"] >= p.min_contrast:
                app, sh, bx, by = self.best_shape(grey, c["box"]) if self.shapes else (1.0, None, 0, 0)
                if sh is None or app > p.app_good:
                    sh = self._learn(grey, c["box"], changed)
                    bx, by = c["box"][0], c["box"][1]
                    app = 0.0
                if sh is not None:
                    pick = (app, 0.0, c, sh, bx, by)
        if pick is None:
            # Parked: nothing new changed, but the last spot still looks like the cursor.
            if self.last is not None and self.last[4] is not None and not lost:
                sh = self.last[4]
                app, bx, by = self._match(grey, sh, self.last[3][0], self.last[3][1], 2)
                if app < p.park_match and changed[by:by + sh.patch.shape[0], bx:bx + sh.patch.shape[1]].any():
                    x, y = bx + sh.hot[0], by + sh.hot[1]
                    self.last = (t, x, y, (bx, by), sh)
                    self.track.append((t, "cursor", x, y))
                    self._keep(t, grey, x, y)
                    return ("cursor", x, y)
            return None
        app, dist, c, sh, bx, by = pick
        if sh is None:
            return None
        # Position from template matching (the blob centre jumps when crossing text).
        app2, bx, by = self._match(grey, sh, bx, by, p.match_pad)
        x, y = bx + sh.hot[0], by + sh.hot[1]
        self.last = (t, x, y, (bx, by), sh)
        self.track.append((t, "cursor", x, y))
        self._keep(t, grey, x, y)
        return ("cursor", x, y)

    def _keep(self, t, grey, x, y, r: int = 32):
        H, W = grey.shape
        x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
        self.seen[t] = (x0, y0, grey[y0:min(H, int(y) + r), x0:min(W, int(x) + r)].copy())

    def _laser_blob(self, crop: np.ndarray, own: np.ndarray, x: int, y: int, s: float):
        """A laser dot inside one small change component: bright, saturated, compact."""
        p = self.p
        b, g, r = cv2.split(crop)
        mx = cv2.max(cv2.max(b, g), r)
        mn = cv2.min(cv2.min(b, g), r)
        col = (mx > p.laser_bright) & (cv2.subtract(mx, mn) > p.laser_sat) & own
        k = int(col.sum())
        if k < max(10, 15 * s * s):
            return None
        ys, xs = np.nonzero(col)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if (max(bw, bh) > p.laser_box * s or k < p.laser_fill * bw * bh
                or max(bw, bh) / max(1, min(bw, bh)) > p.laser_aspect):
            return None
        return (k, float(x + xs.mean()), float(y + ys.mean()))


def track_pointer(work: VideoInfo, states: list, transitions: list, live: list, p: SlideParams,
                  shapes: list | None = None, only: set | None = None) -> tuple[list, list]:
    """Pointer at 12 fps against each state's clean slide. Returns (track, learned shapes)."""
    tr = PointerTracker(p)
    if shapes:
        tr.shapes = list(shapes)
    ignore = None
    si = 0
    for t, f in _frames(work, p.work_fps, p.width):
        if ignore is None:
            ignore = _live_mask(f.shape, live)
        while si + 1 < len(states) and t >= states[si + 1].t_change:
            si += 1
        st = states[si]
        if only is not None and st.id not in only:
            continue
        in_tr = any(a <= t < b for a, b in transitions) or st.image is None
        clean = st.image if st.image is not None else f
        tr.step(t, f, clean, ignore, in_tr)
    return tr.track, tr.shapes, tr.seen


def _filled(mask: np.ndarray) -> np.ndarray:
    """A cursor shape's mask with its outline filled in (the white inside of an arrow)."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ell(3)), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    out = m.copy()
    cv2.drawContours(out, cnts, -1, 1, thickness=-1)
    return out.astype(bool)


def remove_parked_cursor(states: list, track: list, shapes: list, p: SlideParams, seen: dict | None = None) -> set:
    """Find a cursor baked into clean slides, near places it was seen, and inpaint it.

    Places: inside the state, the last sighting before it and the first after it. The cursor's
    exact look is copied from the frame where it was seen at that spot (scaling makes its pixels
    depend on the sub-pixel position, so a template learned elsewhere matches poorly); learned
    shapes are the fallback. Returns the ids of the states that changed.
    """
    fixed = set()
    if not shapes:
        return fixed
    seen = seen or {}
    cursor = [x for x in track if x[1] == "cursor"]
    for st in states:
        if st.image is None or st.blank:
            continue
        inside = [x for x in cursor if st.t_change <= x[0] < st.t_end]
        before = [x for x in cursor if x[0] < st.t_change][-1:]
        after = [x for x in cursor if x[0] >= st.t_end][:1]
        spots: dict[tuple, tuple] = {}
        for t, _, x, y in before + after + inside:
            spots.setdefault((int(round(x)) // 3, int(round(y)) // 3), (t, x, y))
        if not spots:
            continue
        grey = cv2.cvtColor(st.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = grey.shape
        mask = np.zeros((H, W), np.uint8)
        for (t, x, y) in list(spots.values())[:40]:
            for sh in shapes:
                h, w = sh.patch.shape
                x0, y0 = int(round(x)) - sh.hot[0], int(round(y)) - sh.hot[1]
                if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
                    continue
                full = _filled(sh.mask)
                tmpls = []
                if t in seen:
                    cx0, cy0, crop = seen[t]
                    tx, ty = x0 - cx0, y0 - cy0
                    if 0 <= tx and 0 <= ty and tx + w <= crop.shape[1] and ty + h <= crop.shape[0]:
                        tm = crop[ty:ty + h, tx:tx + w]
                        # Only if it really shows the cursor (not background at a misplaced spot).
                        app = float(np.abs(tm - sh.patch)[sh.mask].mean() / 255.0)
                        if app <= p.seen_app and tm[full].std() >= p.shape_min_std:
                            tmpls.append((tm, 2))
                tmpls.append((sh.patch, 6))
                for patch, pad in tmpls:
                    ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
                    ax1, ay1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
                    res = cv2.matchTemplate(grey[ay0:ay1, ax0:ax1], np.ascontiguousarray(patch), cv2.TM_SQDIFF,
                                            mask=full.astype(np.float32))
                    res = np.nan_to_num(res, nan=1e18, posinf=1e18)
                    mn, _, loc, _ = cv2.minMaxLoc(res)
                    rms = math.sqrt(max(0.0, mn) / max(1, full.sum())) / 255.0
                    if rms <= p.parked_rms:
                        bx, by = ax0 + loc[0], ay0 + loc[1]
                        m = cv2.dilate(full.astype(np.uint8), _ell(3)) * 255
                        mask[by:by + h, bx:bx + w] = np.maximum(mask[by:by + h, bx:bx + w], m)
                        break
        if mask.any():
            st.image = cv2.inpaint(st.image, mask, 3, cv2.INPAINT_TELEA)
            fixed.add(st.id)
    return fixed


# ------------------------------------------------------------------ gestures

def _rdp(pts: np.ndarray, eps: float) -> list[int]:
    """Indices of points kept by Ramer-Douglas-Peucker."""
    if len(pts) < 3:
        return list(range(len(pts)))
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.hypot(*ab)
    if n < 1e-9:
        d = np.hypot(*(pts - a).T)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] > eps:
        left = _rdp(pts[: i + 1], eps)
        right = _rdp(pts[i:], eps)
        return left[:-1] + [x + i for x in right]
    return [0, len(pts) - 1]


def guess_gestures(track: list, states: list, p: SlideParams, content: Callable[[int, float, float], bool]) -> list:
    """Code's gesture guesses (hints for Muse Spark) from the pointer track.

    `content(state_index, x, y)` says whether there is slide content within 40 px.
    """
    s = p.width / 960.0
    if not track:
        return []
    # split at gaps and tool changes
    segs, cur = [], [track[0]]
    for a, b in zip(track, track[1:]):
        if b[0] - a[0] > p.gap_s or b[1] != a[1]:
            segs.append(cur)
            cur = []
        cur.append(b)
    segs.append(cur)
    out = []
    for seg in segs:
        if len(seg) < 2:
            continue
        tool = seg[0][1]
        t = np.array([x[0] for x in seg])
        xy = np.array([[x[2], x[3]] for x in seg], float)
        if len(seg) >= 3:
            sm = xy.copy()
            sm[1:-1] = (xy[:-2] + xy[1:-1] + xy[2:]) / 3.0
            xy = sm
        out += _segment_gestures(t, xy, tool, states, p, s, content)
    return _merge_sweeps(sorted(out, key=lambda g: g["t_start"]), p, s)


def _state_at(states, t):
    for i, st in enumerate(states):
        if st.t_change <= t < st.t_end:
            return i
    return len(states) - 1


def _segment_gestures(t, xy, tool, states, p, s, content) -> list:
    n = len(t)
    v = np.zeros(n)
    v[1:] = np.hypot(*(np.diff(xy, axis=0).T)) / np.maximum(1e-6, np.diff(t))
    v[0] = v[1] if n > 1 else 0
    moving = v > p.move_speed * s
    used = np.zeros(n, bool)
    gest = []

    # rests (hovers)
    rests = []
    i = 0
    while i < n:
        if not moving[i]:
            j = i
            while j + 1 < n and not moving[j + 1]:
                j += 1
            dur = t[j] - t[i]
            rests.append((i, j, dur))
            i = j + 1
        else:
            i += 1
    for i, j, dur in rests:
        if dur >= p.hover_s:
            cx, cy = xy[i:j + 1].mean(axis=0)
            si = _state_at(states, t[i])
            parked = dur > p.parked_s or not content(si, cx, cy)
            gest.append(_g("hover", tool, t[i], t[j], xy[i:j + 1], parked=parked))

    # circles: windows <= 4 s whose heading turns >= 300 degrees consistently
    head = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    step = np.hypot(*(np.diff(xy, axis=0).T))
    turn = np.zeros(max(0, n - 2))
    for k in range(n - 2):
        if step[k] < 0.5 * s or step[k + 1] < 0.5 * s:
            continue
        d = head[k + 1] - head[k]
        turn[k] = (d + math.pi) % (2 * math.pi) - math.pi
    circles = []
    for a in range(n - 2):
        tot = 0.0
        absum = 0.0
        for b in range(a, n - 2):
            if t[b + 2] - t[a] > p.circle_max_s:
                break
            if abs(turn[b]) > math.radians(150):
                break  # a reversal (back and forth) is not part of a circle
            tot += turn[b]
            absum += abs(turn[b])
            if abs(tot) >= math.radians(p.circle_turn) and absum <= p.circle_consistency * abs(tot):
                pts = xy[a:b + 3]
                w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
                gap = np.hypot(*(pts[-1] - pts[0]))
                if min(w, h) >= p.circle_min_side * s and gap <= p.circle_gap * max(w, h):
                    circles.append((a, b + 2))
                break
    for a, b in _merge_spans(circles):
        gest.append(_g("circle", tool, t[a], t[b], xy[a:b + 1]))
        used[a:b + 1] = True

    # wiggles
    for a in range(n):
        if used[a]:
            continue
        for b in range(n - 1, a, -1):
            dur = t[b] - t[a]
            if dur < p.wiggle_s[0] or dur > p.wiggle_s[1] or used[a:b + 1].any():
                continue
            pts = xy[a:b + 1]
            w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
            big = max(w, h)
            if not (p.wiggle_box[0] * s <= big <= p.wiggle_box[1] * s):
                continue
            path = np.hypot(*(np.diff(pts, axis=0).T)).sum()
            if path < p.wiggle_path * big or path / dur < p.wiggle_speed * s:
                continue
            axis = 0 if w >= h else 1
            dx = np.diff(pts[:, axis])
            dx = dx[np.abs(dx) > 1.0 * s]
            rev = int((np.sign(dx[1:]) != np.sign(dx[:-1])).sum()) if dx.size > 1 else 0
            if rev < p.wiggle_reversals:
                continue
            while a < b and not moving[a]:
                a += 1
            while b > a and not moving[b]:
                b -= 1
            gest.append(_g("wiggle", tool, t[a], t[b], xy[a:b + 1]))
            used[a:b + 1] = True
            break

    # the rest: straight pieces
    i = 0
    while i < n:
        if used[i] or not moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and moving[j + 1] and not used[j + 1]:
            j += 1
        a0 = max(0, i - 1)
        pts = xy[a0:j + 1]
        keep = _rdp(pts, p.rdp_eps * s)
        for u, w_ in zip(keep, keep[1:]):
            a, b = a0 + u, a0 + w_
            if b <= a:
                continue
            seg = xy[a:b + 1]
            chord = np.hypot(*(seg[-1] - seg[0]))
            path = np.hypot(*(np.diff(seg, axis=0).T)).sum()
            straight = chord / max(path, 1e-6)
            dur = t[b] - t[a]
            dx, dy = seg[-1] - seg[0]
            if (straight >= p.sweep_straight and abs(dx) >= p.sweep_ratio * abs(dy) and chord >= p.sweep_min * s
                    and chord / max(dur, 1e-6) <= p.sweep_max_speed * s and dur >= p.sweep_min_s):
                gest.append(_g("sweep", tool, t[a], t[b], seg))
            elif straight >= p.move_straight and chord >= p.move_min * s:
                # an arrow starts after a rest on content and ends in a non-parked hover
                before = [r for r in rests if r[1] <= a + 1 and t[a] - t[r[1]] <= 0.35]
                after = [g for g in gest if g["type"] == "hover" and abs(g["t_start"] - t[b]) <= 0.35]
                si = _state_at(states, t[a])
                is_arrow = (bool(before) and before[-1][2] >= p.arrow_rest
                            and content(si, *xy[before[-1][0]:before[-1][1] + 1].mean(axis=0))
                            and bool(after) and not after[0]["parked"])
                g = _g("arrow" if is_arrow else "move", tool, t[a], t[b], seg)
                g["from"] = [round(float(seg[0][0]), 1), round(float(seg[0][1]), 1)]
                g["to"] = [round(float(seg[-1][0]), 1), round(float(seg[-1][1]), 1)]
                gest.append(g)
        i = j + 1
    return gest


def _merge_spans(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _g(kind, tool, t0, t1, pts, parked=False):
    pts = np.asarray(pts, float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    g = {"type": kind, "tool": tool, "t_start": round(float(t0), 2), "t_end": round(float(t1), 2),
         "box": [round(float(x0), 1), round(float(y0), 1), round(float(x1 - x0), 1), round(float(y1 - y0), 1)],
         "from": None, "to": None, "path": [[round(float(x), 1), round(float(y), 1)] for x, y in pts]}
    if kind == "hover":
        g["parked"] = bool(parked)
    return g


def _merge_sweeps(gs: list, p: SlideParams, s: float) -> list:
    out = []
    for g in gs:
        if (g["type"] == "sweep" and out and out[-1]["type"] == "sweep" and out[-1]["tool"] == g["tool"]
                and g["t_start"] - out[-1]["t_end"] <= 0.5
                and abs((g["box"][1] + g["box"][3] / 2) - (out[-1]["box"][1] + out[-1]["box"][3] / 2)) <= 10 * s):
            a = out[-1]
            x0, y0 = min(a["box"][0], g["box"][0]), min(a["box"][1], g["box"][1])
            x1 = max(a["box"][0] + a["box"][2], g["box"][0] + g["box"][2])
            y1 = max(a["box"][1] + a["box"][3], g["box"][1] + g["box"][3])
            a.update(t_end=g["t_end"], box=[x0, y0, round(x1 - x0, 1), round(y1 - y0, 1)], path=a["path"] + g["path"])
        else:
            out.append(g)
    return out


def snap_underline(g: dict, text_mask: np.ndarray, p: SlideParams) -> dict:
    """Snap a sweep 4 px below the nearest text line above it."""
    x0, y0, w, h = (int(round(v)) for v in g["box"])
    yc = y0 + h // 2
    H, W = text_mask.shape
    xs = slice(max(0, x0), min(W, x0 + max(1, w)))
    rows = text_mask[:, xs].any(axis=1)
    above = np.nonzero(rows[max(0, yc - 40): yc + 6])[0]
    if above.size:
        bottom = max(0, yc - 40) + int(above.max())
        g = dict(g)
        g["underline_y"] = bottom + p.underline_gap * (p.width / 960.0)
    return g


# ------------------------------------------------------------------ driver

def analyse_slides(info: VideoInfo, workdir: Path, start: float = 0.0, end: float | None = None,
                   params: SlideParams | None = None, progress: Callable[[float], None] | None = None) -> SlidesResult:
    p = params or SlideParams()
    t_wall = time.perf_counter()
    s0, s1 = clip_range(info, start, end)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    work_path = make_working_copy(info, workdir / "work.mp4", s0, s1, p)
    work = probe(work_path)
    t_copy = time.perf_counter() - t_wall
    if progress:
        progress(0.15)

    # Slide background colour: the most common colour over a few frames.
    frames = [f for _, (_t, f) in zip(range(5), _frames(work, 0.5, p.width))]
    bg_img = np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None
    bg_colour = tuple(int(v) for v in np.median(bg_img.reshape(-1, 3), axis=0)) if bg_img is not None else (255, 255, 255)

    live, busy, flick = find_live_regions(work, p, bg_img)
    if progress:
        progress(0.3)
    states, transitions = find_states(work, live, bg_colour, p)
    if progress:
        progress(0.5)
    track, shapes, seen = track_pointer(work, states, transitions, live, p)
    for _ in range(p.inpaint_rounds):
        fixed = remove_parked_cursor(states, track, shapes, p, seen)
        if not fixed:
            break
        sub, _, seen2 = track_pointer(work, states, transitions, live, p, shapes=shapes, only=fixed)
        seen.update(seen2)
        keep = [x for x in track if _state_at(states, x[0]) not in {i for i, st in enumerate(states) if st.id in fixed}]
        track = sorted(keep + sub, key=lambda x: x[0])
    if progress:
        progress(0.85)

    texts = {}

    def content(si, x, y):
        st = states[si]
        if st.image is None:
            return False
        if si not in texts:
            g = cv2.cvtColor(st.image, cv2.COLOR_BGR2GRAY)
            nonbg = _absdiff_max(st.image, np.full_like(st.image, bg_colour)) > 40
            texts[si] = cv2.dilate(nonbg.astype(np.uint8), _ell(2 * p.content_px * p.width / 960 + 1)) > 0
            del g
        H, W = texts[si].shape
        xi, yi = int(min(W - 1, max(0, x))), int(min(H - 1, max(0, y)))
        return bool(texts[si][yi, xi])

    gestures = guess_gestures(track, states, p, content)
    for g in gestures:
        if g["type"] == "sweep":
            si = _state_at(states, g["t_start"])
            if states[si].image is not None:
                g.update(snap_underline(g, _text_pixels(states[si].image, p), p))
    # back to the original clock
    for st in states:
        st.t_start += s0
        st.t_end += s0
        st.t_change += s0
    track = [(t + s0, tool, x, y) for t, tool, x, y in track]
    for g in gestures:
        g["t_start"] = round(g["t_start"] + s0, 2)
        g["t_end"] = round(g["t_end"] + s0, 2)
    transitions = [(a + s0, b + s0) for a, b in transitions]
    H, W = (states[0].image.shape[:2] if states and states[0].image is not None else (work.height, work.width))
    seconds = time.perf_counter() - t_wall
    return SlidesResult(width=W, height=H, live=live, background=bg_colour, states=states, track=track,
                        gestures=gestures, transitions=transitions,
                        stats={"seconds": seconds, "copy_seconds": t_copy, "speed": (s1 - s0) / max(1e-6, seconds),
                               "shapes": len(shapes), "track_points": len(track)})
