"""Generated "filmed whiteboard lecture" with exact ground truth (SPEC 5.4).

A camera looks at a whiteboard on a wall. A teacher walks in front of it and
writes 5 items left to right (4 script-font expressions and one underline).
Item 3 is written while the teacher's body hides it; item 1 is wiped off with
an eraser at 33-35 s. Every frame is rendered from the same per-pixel timing
arrays that `truth_ink` uses, so the truth is exact at any time t.

Variants (all share the scene and, for a given seed, the camera noise):
  base   the plain lecture
  chalk  every base frame negated (255 - frame), like ffmpeg `negate`
  fade   base with a fade-in from black over 0..1.5 s
  drift  the whole camera image shifted by `drift_offset(t)` (sub-pixel)

Usage:
    .venv\\Scripts\\python.exe -m tests.gen_board            # all variants -> data\\selftest\\
    .venv\\Scripts\\python.exe -m tests.gen_board --force    # regenerate
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Public constants
# --------------------------------------------------------------------------------------
SIZE = (1280, 720)            # (width, height)
FPS = 25
DURATION = 38.0
VARIANTS = ("base", "chalk", "fade", "drift")
GEN_VERSION = 2               # bump whenever the rendering or the truth changes

HIDDEN_ITEM = 3
ERASE_ITEM = 1
ERASE_T = (33.0, 35.0)

W, H = SIZE
N_FRAMES = int(round(DURATION * FPS))          # frame i is at t = i / FPS
NOISE_SIGMA = 2.5
WOBBLE_AMP, WOBBLE_PERIOD = 0.01, 3.7          # frame *= 1 + amp * sin(2 pi t / period)
FADE_SECONDS = 1.5
# (t0, t1, dx, dy): the offset ramps linearly to (dx, dy) during t0..t1, then holds.
DRIFT_STEPS = ((20.0, 20.5, 2.0, -1.5), (28.0, 28.5, 2.5, -2.0))

_EPS = 1e-9
_BOARD = (90, 60, 1190, 560)   # white surface: pixels x in [90, 1190), y in [60, 560)
_FRAME_PX = 14                 # aluminium frame width
_FONT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
_COLORS = {"black": (30, 30, 35), "blue": (170, 70, 25), "red": (40, 40, 190)}

# --------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------
# The writing timeline (seconds), set by how SPEC 5.1 groups and times writing:
# - item 1 starts at 4.0: in the fade variant the picture settles at ~1.7 s and anything
#   accepted in the next 2.5 s is "pre-existing", so item 1 must not be accepted before ~4.2;
# - item 5 starts 4.9 s after item 4 ends: item 4's last pixels are accepted ~1.1 s after
#   it ends and the underline's first ~0.6-1 s after it starts, and the underline (13 px
#   below item 4) would join item 4's piece if accepted within 3 s of its last update;
# - the teacher then walks ~540 px from the underline to item 1 and erases at 33.0.
_TEXT_SPECS = (
    dict(id=1, text="x + y", color_name="black", scale=1.9, left=150, top=114, t_start=4.0, t_end=6.6),
    dict(id=2, text="a = b", color_name="blue", scale=1.9, left=404, top=124, t_start=7.5, t_end=11.0),
    dict(id=3, text="f(t)", color_name="red", scale=1.8, centre_x=640, top=368, t_start=13.0, t_end=17.0),
    dict(id=4, text="dx = 2", color_name="black", scale=1.7, left=754, top=372, t_start=24.4, t_end=26.0),
)
_UNDERLINE_SPEC = dict(id=5, color_name="red", t_start=30.9, t_end=31.5, gap=16.0)
# Ink-to-ink gap between words. The script font's own space leaves ~34 px, right on SPEC
# 5.1's join padding (25 px at 800 px wide = ~34 video px), so an item could split in two.
_WORD_GAP = 16


def _render_text(text: str, scale: float, left: int, top: int) -> np.ndarray:
    """Anti-aliased coverage (uint8 0..255) of `text` with its ink box's top-left at (left, top).

    Words share one baseline and sit `_WORD_GAP` px apart, ink to ink.
    """
    words = text.split()
    ext = []                                   # each word's ink box relative to its text origin
    for word in words:
        probe = np.zeros((H, W), np.uint8)
        cv2.putText(probe, word, (300, 300), _FONT, scale, 255, 3, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect((probe >= 128).astype(np.uint8))
        ext.append((x - 300, y - 300, w))
    base_y = top - min(e[1] for e in ext)
    cov = np.zeros((H, W), np.uint8)
    cursor = left
    for word, (ex, _, ew) in zip(words, ext):
        tmp = np.zeros((H, W), np.uint8)
        cv2.putText(tmp, word, (cursor - ex, base_y), _FONT, scale, 255, 3, cv2.LINE_AA)
        np.maximum(cov, tmp, out=cov)
        cursor += ew + _WORD_GAP
    return cov


def _ink_box(cov: np.ndarray) -> list[int]:
    x, y, w, h = cv2.boundingRect((cov >= 128).astype(np.uint8))
    return [int(x), int(y), int(w), int(h)]


def _underline_curve(box4: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """A slightly tilted, slightly bowed hand-drawn line under item 4."""
    x4, y4, w4, h4 = box4
    bottom = y4 + h4 - 1
    u = np.linspace(0.0, 1.0, 33)
    xs = (x4 - 4) + (w4 + 6) * u
    ys = bottom + _UNDERLINE_SPEC["gap"] + 3.0 * u - 1.5 * np.sin(np.pi * u)
    return xs, ys


def _render_underline(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    cov = np.zeros((H, W), np.uint8)
    pts = np.round(np.stack([xs, ys], 1) * 16.0).astype(np.int32)
    cv2.polylines(cov, [pts], False, 255, 4, cv2.LINE_AA, 4)
    return cov


def _build_items():
    items, covs = [], {}
    for s in _TEXT_SPECS:
        if "left" in s:
            cov = _render_text(s["text"], s["scale"], s["left"], s["top"])
        else:  # centred horizontally on centre_x
            tmp = _render_text(s["text"], s["scale"], 0, s["top"])
            w = _ink_box(tmp)[2]
            cov = _render_text(s["text"], s["scale"], int(round(s["centre_x"] - w / 2)), s["top"])
        covs[s["id"]] = cov
        items.append(dict(id=s["id"], kind="text", text=s["text"], color_name=s["color_name"],
                          bgr=_COLORS[s["color_name"]], box=_ink_box(cov),
                          t_start=s["t_start"], t_end=s["t_end"]))
    u = _UNDERLINE_SPEC
    xs, ys = _underline_curve(items[3]["box"])
    cov = _render_underline(xs, ys)
    covs[5] = cov
    items.append(dict(id=5, kind="underline", text="", color_name=u["color_name"],
                      bgr=_COLORS[u["color_name"]], box=_ink_box(cov),
                      t_start=u["t_start"], t_end=u["t_end"]))
    return items, covs, (xs, ys)


ITEMS, _COV, _UL_CURVE = _build_items()
_ITEM = {it["id"]: it for it in ITEMS}
_ITEM_INK = {k: (c >= 128) for k, c in _COV.items()}   # alpha >= 0.5


def _check_layout() -> None:
    bx0, by0, bx1, by1 = _BOARD
    for it in ITEMS:
        x, y, w, h = it["box"]
        assert x - bx0 >= 50 and y - by0 >= 50 and bx1 - (x + w) >= 50 and by1 - (y + h) >= 50, it
    for a in ITEMS:
        for b in ITEMS:
            if a["id"] >= b["id"] or {a["id"], b["id"]} == {4, 5}:
                continue
            ax, ay, aw, ah = a["box"]
            bx, by, bw, bh = b["box"]
            gap = max(bx - (ax + aw), ax - (bx + bw), by - (ay + ah), ay - (by + bh))
            assert gap >= 60, (a["id"], b["id"], gap)
    # underline at least 12 px below item 4's lowest ink pixel
    y4, h4 = _ITEM[4]["box"][1], _ITEM[4]["box"][3]
    assert _ITEM[5]["box"][1] - (y4 + h4 - 1) >= 12
    # coverages never overlap, so one alpha/colour map holds every item
    total = sum((c > 0).astype(np.int32) for c in _COV.values())
    assert total.max() <= 1


_check_layout()

# --------------------------------------------------------------------------------------
# Pen and eraser paths, per-pixel reveal and erase times
# --------------------------------------------------------------------------------------
def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _pen_x(item: dict, t: float) -> float:
    x0, _, w, _ = item["box"]
    ts, te = item["t_start"], item["t_end"]
    return x0 + w * (t - ts) / (te - ts)


def _pen_point(item_id: int, t: float) -> tuple[float, float]:
    it = _ITEM[item_id]
    ts, te = it["t_start"], it["t_end"]
    t = min(max(t, ts), te)
    px = _pen_x(it, t)
    if it["kind"] == "underline":
        xs, ys = _UL_CURVE
        return px, float(np.interp(px, xs, ys))
    _, y, _, h = it["box"]
    return px, y + h * (0.5 + 0.3 * math.sin(2 * math.pi * 2.2 * (t - ts)))


_ERASER_W, _ERASER_H = 70, 36


def _eraser_centre(t: float) -> tuple[float, float]:
    """Three horizontal passes over item 1 (bottom, middle, top) during ERASE_T."""
    x1, y1, w1, h1 = _ITEM[ERASE_ITEM]["box"]
    xl = x1 - 12 + _ERASER_W / 2            # rect reaches 12 px past both ends
    xr = x1 + w1 + 12 - _ERASER_W / 2
    yb = y1 + h1 + 4 - _ERASER_H / 2        # rect reaches 4 px past top and bottom
    yt = y1 - 4 + _ERASER_H / 2
    ym = (yb + yt) / 2
    t0 = ERASE_T[0]
    # (start, end, x0, x1, y0, y1)
    legs = ((t0 + 0.0, t0 + 0.6, xl, xr, yb, yb),
            (t0 + 0.6, t0 + 0.7, xr, xr, yb, ym),
            (t0 + 0.7, t0 + 1.3, xr, xl, ym, ym),
            (t0 + 1.3, t0 + 1.4, xl, xl, ym, yt),
            (t0 + 1.4, t0 + 2.0, xl, xr, yt, yt))
    t = min(max(t, legs[0][0]), legs[-1][1])
    for a, b, xa, xb, ya, yb_ in legs:
        if t <= b + _EPS:
            s = _smooth((t - a) / (b - a))
            wob = 1.5 * math.sin(2 * math.pi * 3.0 * (t - t0))
            return xa + (xb - xa) * s, ya + (yb_ - ya) * s + wob
    raise AssertionError


def _eraser_rect(cx: float, cy: float) -> tuple[int, int, int, int]:
    x0 = int(math.floor(cx - _ERASER_W / 2 + 0.5))
    y0 = int(math.floor(cy - _ERASER_H / 2 + 0.5))
    return x0, y0, x0 + _ERASER_W, y0 + _ERASER_H


def _build_timing():
    alpha = np.zeros((H, W), np.float32)
    col = np.zeros((H, W, 3), np.float32)
    reveal = np.full((H, W), np.inf)
    cols = np.arange(W, dtype=np.float64)
    for it in ITEMS:
        cov = _COV[it["id"]]
        m = cov > 0
        alpha[m] = cov[m] / 255.0
        col[m] = it["bgr"]
        x0, _, w, _ = it["box"]
        ts, te = it["t_start"], it["t_end"]
        # pixel column c appears once pen_x(t) >= c
        rt = np.clip(ts + (cols - x0) * (te - ts) / w, ts, te)
        ys, xs = np.nonzero(m)
        reveal[ys, xs] = rt[xs]
    erase = np.full((H, W), np.inf)
    k0 = int(math.ceil(ERASE_T[0] * FPS - 1e-6))
    k1 = int(math.floor(ERASE_T[1] * FPS + 1e-6))
    for k in range(k0, k1 + 1):           # the eraser wipes what it covers at each frame
        t = k / FPS
        x0, y0, x1, y1 = _eraser_rect(*_eraser_centre(t))
        sub = erase[y0:y1, x0:x1]
        np.minimum(sub, t, out=sub)
    return alpha, col, reveal, erase


_ALPHA, _INKCOL, _REVEAL, _ERASE = _build_timing()
# the same data for the few pixels that ever carry ink (fast per-frame compositing)
_INK_YS, _INK_XS = np.nonzero(_ALPHA > 0)
_INK_A, _INK_C = _ALPHA[_INK_YS, _INK_XS], _INKCOL[_INK_YS, _INK_XS]
_INK_RT, _INK_ET = _REVEAL[_INK_YS, _INK_XS], _ERASE[_INK_YS, _INK_XS]
# item 1 (every anti-aliased pixel of it) must be gone by the end of the erase
assert np.all(_ERASE[_COV[ERASE_ITEM] > 0] <= ERASE_T[1] + _EPS)
assert np.all(np.isinf(_ERASE[(_ALPHA > 0) & (_COV[ERASE_ITEM] == 0)]))


def _visible(t: float) -> np.ndarray:
    return (_REVEAL <= t + _EPS) & (_ERASE > t + _EPS)


# --------------------------------------------------------------------------------------
# Teacher motion
# --------------------------------------------------------------------------------------
_OFF_FOLLOW = 130.0                       # body centre - pen x while writing items 1, 2
_C3 = _ITEM[3]["box"][0] + _ITEM[3]["box"][2] / 2.0
_CX45 = _ITEM[4]["box"][0] + _ITEM[4]["box"][2] + 110.0   # stands right of item 4 for 4 and 5
_CX_ERASE = 470.0
_WALK3 = (23.2, 24.2)                     # leaves item 3 (this sets UNCOVER_T)
_WALK_ERASE = (31.9, 32.95)               # brisk walk from item 5 to item 1
_RAISE = 0.4                              # arm up/down transition time
_SHOULDER_DX, _SHOULDER_Y = -68.0, 318.0
_REST_DX, _REST_Y = -88.0, 500.0


def _follow(item_id: int):
    it = _ITEM[item_id]

    def f(t: float) -> float:
        return _pen_x(it, min(max(t, it["t_start"]), it["t_end"])) + _OFF_FOLLOW
    return f


def _const(v: float):
    return lambda t: v


# (t0, t1, x(t)); gaps between segments are smooth walks
_BODY_SEGS = (
    (0.0, 1.8, _const(1000.0)),
    (3.5, 6.7, _follow(1)),
    (7.1, 11.4, _follow(2)),
    (12.5, _WALK3[0], _const(_C3)),
    (_WALK3[1], _WALK_ERASE[0], _const(_CX45)),
    (_WALK_ERASE[1], DURATION + 1.0, _const(_CX_ERASE)),
)


def _body_x(t: float) -> float:
    segs = _BODY_SEGS
    if t <= segs[0][1]:
        return segs[0][2](t)
    for (a0, a1, fa), (b0, b1, fb) in zip(segs, segs[1:]):
        if t <= a1:
            return fa(t)
        if t < b0:
            xa, xb = fa(a1), fb(b0)
            return xa + (xb - xa) * _smooth((t - a1) / (b0 - a1))
    return segs[-1][2](t)


# (t_start, t_end, tip(t), tool, pen colour)
_ARM_PHASES = tuple(
    (it["t_start"], it["t_end"], functools.partial(_pen_point, it["id"]), "pen", it["bgr"])
    for it in ITEMS) + ((ERASE_T[0], ERASE_T[1], _eraser_centre, "eraser", None),)
_ERASER_FROM = _ITEM[5]["t_end"] + _RAISE  # swaps pen for eraser with the arm down
for _a, _b in zip(_ARM_PHASES, _ARM_PHASES[1:]):
    assert _a[1] + _RAISE <= _b[0] - _RAISE + _EPS


def _hand(t: float, cx: float):
    """Returns (tip (x, y), tool, pen colour)."""
    rest = (cx + _REST_DX, _REST_Y)
    tool = "eraser" if t >= _ERASER_FROM else "pen"
    pen_bgr = ITEMS[0]["bgr"]
    for ts, te, fn, ph_tool, bgr in _ARM_PHASES:
        if ph_tool == "pen" and t >= ts - _RAISE:
            pen_bgr = bgr
    for ts, te, fn, _tool, _bgr in _ARM_PHASES:
        if ts - _RAISE <= t < ts:
            p, s = fn(ts), _smooth((t - (ts - _RAISE)) / _RAISE)
            return (rest[0] + (p[0] - rest[0]) * s, rest[1] + (p[1] - rest[1]) * s), tool, pen_bgr
        if ts <= t <= te:
            return fn(t), tool, pen_bgr
        if te < t < te + _RAISE:
            p, s = fn(te), _smooth((t - te) / _RAISE)
            return (p[0] + (rest[0] - p[0]) * s, p[1] + (rest[1] - p[1]) * s), tool, pen_bgr
    return rest, tool, pen_bgr


_SHIRT, _TROUSERS, _SLEEVE = (60, 50, 45), (48, 42, 40), (66, 55, 49)
_SKIN, _HAIR = (140, 170, 210), (32, 42, 58)
_FELT, _ERASER_TOP = (52, 52, 58), (112, 112, 118)


def _P(pts, ox: float, oy: float) -> np.ndarray:
    """Points in 1/16 px fixed point (cv2 `shift=4`) relative to (ox, oy)."""
    a = (np.asarray(pts, np.float64) - (ox, oy)) * 16.0
    return np.round(a).astype(np.int32)


def _pt(x: float, y: float, ox: float, oy: float) -> tuple[int, int]:
    return int(round((x - ox) * 16.0)), int(round((y - oy) * 16.0))


def _teacher_parts(t: float):
    """Draw list for the teacher at time t: [(colour, shaded, draw(canvas, ox, oy))], roi, cx.

    Parts are drawn back to front. `shaded` parts get a per-column darkening at the sides.
    """
    cx = _body_x(t)
    (tx, ty), tool, pen_bgr = _hand(t, cx)
    sx, sy = cx + _SHOULDER_DX, _SHOULDER_Y
    L = math.hypot(tx - sx, ty - sy) or 1.0
    ux, uy = (tx - sx) / L, (ty - sy) / L
    parts = []

    def poly(pts):
        return lambda c, ox, oy: cv2.fillPoly(c, [_P(pts, ox, oy)], 255, cv2.LINE_AA, 4)

    def circle(x, y, r):
        return lambda c, ox, oy: cv2.circle(c, _pt(x, y, ox, oy), int(round(r * 16)),
                                            255, -1, cv2.LINE_AA, 4)

    def line(x0, y0, x1, y1, th):
        return lambda c, ox, oy: cv2.line(c, _pt(x0, y0, ox, oy), _pt(x1, y1, ox, oy),
                                          255, th, cv2.LINE_AA, 4)

    def half_ellipse(x, y, ax, ay):
        return lambda c, ox, oy: cv2.ellipse(c, _pt(x, y, ox, oy),
                                             (int(round(ax * 16)), int(round(ay * 16))),
                                             0, 180, 360, 255, -1, cv2.LINE_AA, 4)

    def rect(x0, y0, x1, y1):  # integer, no anti-aliasing (the eraser's footprint is exact)
        return lambda c, ox, oy: cv2.rectangle(c, (int(x0 - ox), int(y0 - oy)),
                                               (int(x1 - 1 - ox), int(y1 - 1 - oy)), 255, -1)

    yb = H + 2
    parts.append((_TROUSERS, True, poly([(cx - 76, 560), (cx - 4, 560), (cx - 8, yb), (cx - 70, yb)])))
    parts.append((_TROUSERS, True, poly([(cx + 4, 560), (cx + 76, 560), (cx + 70, yb), (cx + 8, yb)])))
    parts.append((_SHIRT, True, poly([(cx - 58, 296), (cx + 58, 296), (cx + 80, 306), (cx + 86, 322),
                                      (cx + 84, 450), (cx + 80, 575), (cx - 80, 575), (cx - 84, 450),
                                      (cx - 86, 322), (cx - 80, 306)])))
    parts.append((_SKIN, False, poly([(cx - 13, 262), (cx + 13, 262), (cx + 14, 300), (cx - 14, 300)])))
    parts.append((_SKIN, False, circle(cx, 240.0, 38.0)))
    parts.append((_HAIR, False, half_ellipse(cx, 236.0, 40.0, 34.0)))
    if tool == "pen":
        wx, wy = tx - 34 * ux, ty - 34 * uy
        hx, hy = tx - 26 * ux, ty - 26 * uy
    else:
        wx, wy = tx - 12 * ux, ty - 12 * uy
        hx, hy = tx - 4 * ux, ty - 4 * uy
    parts.append((_SLEEVE, False, line(sx, sy, wx, wy, 22)))
    if tool == "eraser":
        ex0, ey0, ex1, ey1 = _eraser_rect(tx, ty)
        parts.append((_FELT, False, rect(ex0, ey0, ex1, ey1)))
        parts.append((_ERASER_TOP, False, rect(ex0 + 3, ey0, ex1 - 3, ey0 + 20)))
    parts.append((_SKIN, False, circle(hx, hy, 12.0)))
    if tool == "pen":
        parts.append((pen_bgr, False, line(tx - 24 * ux, ty - 24 * uy, tx, ty, 6)))

    x0 = int(math.floor(min(cx - 92, tx - 46))) - 2
    x1 = int(math.ceil(max(cx + 92, tx + 46))) + 2
    y0 = int(math.floor(min(198.0, ty - 30))) - 2
    roi = (max(x0, 0), max(y0, 0), min(x1, W), H)
    return parts, roi, cx


def _teacher_alpha(t: float, roi=None) -> np.ndarray:
    """uint8 union coverage of everything in front of the board, in `roi` (x0, y0, x1, y1)."""
    parts, _, _ = _teacher_parts(t)
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, W, H)
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for _, _, draw in parts:
        draw(canvas, x0, y0)
    return canvas


def _draw_teacher(frame: np.ndarray, t: float) -> None:
    parts, (x0, y0, x1, y1), cx = _teacher_parts(t)
    R = frame[y0:y1, x0:x1]
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for colour, shaded, draw in parts:
        canvas.fill(0)
        draw(canvas, x0, y0)
        bx, by, bw, bh = cv2.boundingRect(canvas)
        if bw == 0 or bh == 0:
            continue
        a = canvas[by:by + bh, bx:bx + bw].astype(np.float32) * (1.0 / 255.0)
        sub = R[by:by + bh, bx:bx + bw]
        c = np.asarray(colour, np.float32)
        if shaded:
            xs = np.arange(x0 + bx, x0 + bx + bw, dtype=np.float32) - cx
            shade = 0.80 + 0.20 * np.cos(np.clip(xs / 92.0, -1, 1) * (np.pi / 2))
            c = shade[None, :, None] * c[None, None, :]
        sub += a[..., None] * (c - sub)


def _compute_uncover_t() -> float:
    ink = _ITEM_INK[HIDDEN_ITEM]
    x, y, w, h = _ITEM[HIDDEN_ITEM]["box"]
    sub = ink[y:y + h, x:x + w]
    k0 = int(math.ceil(_ITEM[HIDDEN_ITEM]["t_end"] * FPS - 1e-6))
    for k in range(k0, N_FRAMES):
        t = k / FPS
        if not np.any(_teacher_alpha(t, (x, y, x + w, y + h))[sub] > 0):
            return round(t, 6)
    raise AssertionError("item 3 is never uncovered")


UNCOVER_T = _compute_uncover_t()

# --------------------------------------------------------------------------------------
# Public truth functions
# --------------------------------------------------------------------------------------
def board_corners() -> list[list[int]]:
    """Corners (tl, tr, br, bl) of the white writing surface in video px, no drift."""
    x0, y0, x1, y1 = _BOARD
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def drift_offset(t: float, variant: str = "base") -> tuple[float, float]:
    _check_variant(variant)
    if variant != "drift":
        return 0.0, 0.0
    dx = dy = 0.0
    for t0, t1, tx, ty in DRIFT_STEPS:
        if t <= t0:
            break
        s = min((t - t0) / (t1 - t0), 1.0)
        dx, dy = dx + (tx - dx) * s, dy + (ty - dy) * s
    return dx, dy


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def truth_ink(t: float, variant: str = "base") -> np.ndarray:
    """Every ink pixel on the board at t (written, not erased), occluded or not; drift applied."""
    _check_variant(variant)
    dx, dy = drift_offset(t, variant)
    vis = _visible(t)
    if dx == 0.0 and dy == 0.0:
        return (_ALPHA >= 0.5) & vis
    a = np.where(vis, _ALPHA, np.float32(0))
    return _shift(a, dx, dy) >= 0.5


def teacher_mask(t: float, variant: str = "base") -> np.ndarray:
    """Everything in front of the board at t (body, head, arm, hand, pen, eraser); drift applied."""
    _check_variant(variant)
    a = _teacher_alpha(t)
    dx, dy = drift_offset(t, variant)
    if dx == 0.0 and dy == 0.0:
        return a > 0
    return _shift(a.astype(np.float32) * (1.0 / 255.0), dx, dy) > (0.5 / 255.0)


def item_ink(item_id: int) -> np.ndarray:
    """Final full ink (alpha >= 0.5) of one item, no drift."""
    return _ITEM_INK[item_id].copy()


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _background() -> np.ndarray:
    """Static scene without ink or teacher: wall, floor, board frame, tray, white surface."""
    rng = np.random.default_rng(20261104)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.empty((H, W, 3), np.float32)
    wall = np.array([158, 176, 192], np.float32)               # warm grey-beige (BGR)
    grad = 1.03 - 0.07 * (yy / H) - 0.03 * (xx / W)
    low = cv2.resize(rng.normal(0, 1, (12, 20)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    tex = 2.5 * low + rng.normal(0, 1.2, (H, W)).astype(np.float32)
    img[:] = wall * grad[..., None] + tex[..., None]

    # skirting board and floor
    img[656:668] = (212, 215, 218)
    img[667:668] = (170, 172, 175)
    floor = np.array([82, 106, 136], np.float32)
    fy = np.arange(668, H, dtype=np.float32)
    img[668:] = floor * (0.92 + 0.08 * (fy - 668) / (H - 668))[:, None, None]
    img[668:] += (rng.normal(0, 1.5, (H - 668, W)).astype(np.float32)
                  + 4 * np.sin(xx[668:] / 9.0) * 0.3)[..., None]
    for y in (684, 703):
        img[y] *= 0.88

    # soft shadow of the board on the wall
    fx0, fy0, fx1, fy1 = (_BOARD[0] - _FRAME_PX, _BOARD[1] - _FRAME_PX,
                          _BOARD[2] + _FRAME_PX, _BOARD[3] + _FRAME_PX)
    sh = np.zeros((H, W), np.float32)
    sh[fy0 + 7:fy1 + 7 + 16, fx0 + 5:fx1 + 5] = 1.0
    sh = cv2.GaussianBlur(sh, (0, 0), 6)
    img *= (1.0 - 0.18 * sh)[..., None]

    # aluminium frame with a bevel
    img[fy0:fy1, fx0:fx1] = (178, 180, 178)
    img[fy0:fy0 + 3, fx0:fx1] = (214, 216, 214)
    img[fy0:fy1, fx0:fx0 + 2] = (205, 207, 205)
    img[fy1 - 3:fy1, fx0:fx1] = (128, 130, 128)
    img[fy0:fy1, fx1 - 2:fx1] = (140, 142, 140)
    img[fy0, fx0:fx1] = img[fy1 - 1, fx0:fx1] = (110, 112, 110)
    img[fy0:fy1, fx0] = img[fy0:fy1, fx1 - 1] = (110, 112, 110)
    bx0, by0, bx1, by1 = _BOARD
    img[by0 - 1:by1 + 1, bx0 - 1:bx1 + 1] = (138, 140, 138)

    # pen tray with three markers
    img[fy1:fy1 + 12, fx0 + 30:fx1 - 30] = (166, 168, 166)
    img[fy1 + 10:fy1 + 12, fx0 + 30:fx1 - 30] = (115, 117, 115)
    img[fy1 + 12:fy1 + 16, fx0 + 34:fx1 - 26] *= 0.85
    for i, name in enumerate(("black", "blue", "red")):
        x = 990 + 62 * i
        img[fy1 + 2:fy1 + 9, x:x + 44] = (205, 205, 205)
        img[fy1 + 2:fy1 + 9, x + 30:x + 44] = _COLORS[name]
        img[fy1 + 2:fy1 + 9, x:x + 6] = _COLORS[name]

    # white writing surface with a soft +-4 vignette
    sy, sx = yy[by0:by1, bx0:bx1], xx[by0:by1, bx0:bx1]
    r2 = ((sx - (bx0 + bx1) / 2) / ((bx1 - bx0) / 2)) ** 2 + ((sy - (by0 + by1) / 2) / ((by1 - by0) / 2)) ** 2
    vig = 4.0 - 4.0 * r2
    img[by0:by1, bx0:bx1] = np.array([242, 243, 240], np.float32) + vig[..., None]
    return img


def _scene(t: float) -> np.ndarray:
    """Clean float32 frame at t: background, ink, teacher (no camera effects)."""
    f = _background().copy()
    vis = (_INK_RT <= t + _EPS) & (_INK_ET > t + _EPS)
    ys, xs = _INK_YS[vis], _INK_XS[vis]
    under = f[ys, xs]
    f[ys, xs] = under + _INK_A[vis][:, None] * (_INK_C[vis] - under)
    _draw_teacher(f, t)
    return f


def _noise_lut() -> np.ndarray:
    """256 equal-probability quantiles of N(0, NOISE_SIGMA): one random byte -> one Gaussian sample.

    8-bit resolution is far finer than the 1-level output quantisation; tails stop at 2.66 sigma.
    The table is rescaled so its standard deviation is exactly NOISE_SIGMA.
    """
    from statistics import NormalDist
    z = np.array([NormalDist().inv_cdf((k + 0.5) / 256) for k in range(256)], np.float64)
    return (z * (NOISE_SIGMA / z.std())).astype(np.float32).reshape(1, 256)


_NOISE_LUT = _noise_lut()


def _camera_noise(seed: int, i: int) -> np.ndarray:
    """Fresh per-pixel, per-channel Gaussian noise for frame i (deterministic in seed, i)."""
    raw = np.random.default_rng([seed, i]).bytes(H * W * 3)
    return cv2.LUT(np.frombuffer(raw, np.uint8).reshape(H, W, 3), _NOISE_LUT)


def _render_frame(i: int, seed: int, variants: tuple[str, ...]) -> dict[str, np.ndarray]:
    t = i / FPS
    scene = _scene(t)
    wob = 1.0 + WOBBLE_AMP * math.sin(2 * math.pi * t / WOBBLE_PERIOD)
    noise = _camera_noise(seed, i)
    out = {}
    # cv2.convertScaleAbs rounds and saturates to uint8; values here are never negative.
    if {"base", "chalk", "fade"} & set(variants):
        cam = cv2.scaleAdd(scene, wob, noise)                  # scene * wobble + noise
        if "fade" in variants:
            out["fade"] = cv2.convertScaleAbs(cam, alpha=min(1.0, t / FADE_SECONDS))
        if {"base", "chalk"} & set(variants):
            base = cv2.convertScaleAbs(cam)
            if "base" in variants:
                out["base"] = base
            if "chalk" in variants:
                out["chalk"] = cv2.bitwise_not(base)
    if "drift" in variants:
        dx, dy = drift_offset(t, "drift")
        moved = scene if (dx == 0.0 and dy == 0.0) else _shift(scene, dx, dy)
        out["drift"] = cv2.convertScaleAbs(cv2.scaleAdd(moved, wob, noise))
    return out


class _Encoder:
    """Pipes raw bgr24 frames into ffmpeg/libx264; the file appears only when complete."""

    def __init__(self, path: Path):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found on PATH")
        self.path = path
        self.tmp = path.with_name(path.stem + ".partial.mp4")
        self.err = tempfile.TemporaryFile()
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
               str(self.tmp)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).data)

    def close(self) -> None:
        self.proc.stdin.close()
        rc = self.proc.wait()
        self.err.seek(0)
        msg = self.err.read().decode("utf-8", "replace").strip().splitlines()
        self.err.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path.name}: {msg[-1] if msg else rc}")
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait()
        finally:
            self.err.close()
            self.tmp.unlink(missing_ok=True)


def _truth_summary(variant: str, seed: int) -> dict:
    return {
        "GEN_VERSION": GEN_VERSION,
        "variant": variant,
        "seed": seed,
        "SIZE": list(SIZE),
        "FPS": FPS,
        "DURATION": DURATION,
        "n_frames": N_FRAMES,
        "board_type": "chalk" if variant == "chalk" else "whiteboard",
        "corners": board_corners(),
        "items": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in it.items()} for it in ITEMS],
        "HIDDEN_ITEM": HIDDEN_ITEM,
        "UNCOVER_T": UNCOVER_T,
        "ERASE_ITEM": ERASE_ITEM,
        "ERASE_T": list(ERASE_T),
        "drift_steps": ([{"t0": a, "t1": b, "dx": dx, "dy": dy} for a, b, dx, dy in DRIFT_STEPS]
                        if variant == "drift" else []),
        "fade_in_s": FADE_SECONDS if variant == "fade" else 0.0,
        "noise_sigma": NOISE_SIGMA,
        "brightness_wobble": {"amp": WOBBLE_AMP, "period_s": WOBBLE_PERIOD},
    }


def _paths(out_dir: Path, variant: str) -> tuple[Path, Path]:
    return out_dir / f"board_{variant}.mp4", out_dir / f"board_{variant}.json"


def _up_to_date(out_dir: Path, variant: str, seed: int) -> bool:
    video, meta = _paths(out_dir, variant)
    if not (video.is_file() and meta.is_file()):
        return False
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return info.get("GEN_VERSION") == GEN_VERSION and info.get("seed", seed) == seed


def _generate(out_dir, variants, seed: int = 0, force: bool = False, workers: int | None = None) -> list[str]:
    """Renders every requested variant in one pass (one scene + one noise field per frame)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in variants:
        _check_variant(v)
    todo = tuple(v for v in dict.fromkeys(variants) if force or not _up_to_date(out_dir, v, seed))
    if not todo:
        return []
    _background()  # build once before the worker threads start
    encoders = {v: _Encoder(_paths(out_dir, v)[0]) for v in todo}
    workers = workers or max(2, min(4, (os.cpu_count() or 4) // 2))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            for i in range(N_FRAMES):
                pending.append(pool.submit(_render_frame, i, seed, todo))
                if len(pending) >= 2 * workers:
                    frames = pending.popleft().result()
                    for v in todo:
                        encoders[v].write(frames[v])
            while pending:
                frames = pending.popleft().result()
                for v in todo:
                    encoders[v].write(frames[v])
        for v in todo:
            encoders[v].close()
    except BaseException:
        for enc in encoders.values():
            enc.abort()
        raise
    for v in todo:
        meta = _paths(out_dir, v)[1]
        tmp = meta.with_name(meta.name + ".partial")
        tmp.write_text(json.dumps(_truth_summary(v, seed), indent=2), encoding="utf-8")
        os.replace(tmp, meta)
    return list(todo)


def make_board_lecture(out_dir, variant: str = "base", seed: int = 0, force: bool = False) -> Path:
    """Writes out_dir/board_<variant>.mp4 and .json (skipped when already up to date)."""
    _check_variant(variant)
    _generate(out_dir, (variant,), seed=seed, force=force)
    return _paths(Path(out_dir), variant)[0]


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate the board-lecture self-test videos.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "selftest"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    done = _generate(args.out, tuple(args.variants), seed=args.seed, force=args.force)
    dt = time.perf_counter() - t0
    if done:
        print(f"generated {', '.join(done)} in {dt:.1f} s ({dt / len(done):.1f} s per variant, "
              f"{N_FRAMES / dt:.0f} frames/s) -> {args.out}")
    else:
        print(f"all variants up to date in {args.out}")
    print(f"UNCOVER_T = {UNCOVER_T:.2f} s, items: "
          + ", ".join(f"{it['id']}:{it['t_start']}-{it['t_end']}" for it in ITEMS))


if __name__ == "__main__":
    _main()
