"""Generated "screen-recorded slide lecture" with exact ground truth (SPEC 5.2, 5.4).

A 40 s screen recording of a 3-slide deck, the fixture for the slide analysis:

- 6 slide states: slide 1 (a bullet builds in), slide 2 (a bar chart; red pen ink is
  drawn on it), slide 3 (reached by a 0.6 s crossfade; a callout builds in);
- a noisy webcam picture in the bottom-right corner with a person who sways and talks;
- a mouse cursor (white arrow, black outline) doing a hover, a sweep, a circle, a wiggle
  and an A->B arrow move, plus travel moves that are neither arrows nor sweeps, and parked
  on empty space for the whole of states 0 and 3, so it is baked into those clean slides;
- a red laser dot doing a circle and a hover over the coloured bars (never shown together
  with the cursor).

Every frame is rendered from the same timeline that `states`, `gestures` and `pointer`
report, so the truth is exact. Screen recordings are clean: the only noise is in the
webcam picture (sigma 6); x264 crf 18 adds the compression artefacts.

Variants (all share the timeline and, for a given seed, the webcam noise):
  base  the plain lecture
  dark  every base frame negated (255 - frame), like ffmpeg `negate`
  fade  base with a fade-in from black over 0..1.5 s (the self-test checks gestures only)
  pip   one still slide (slide 3, fully built), the same webcam corner with a person who
        barely moves (breathing + noise only), no pointer: expect 1 state, the webcam
        found, no gestures

Slide numbers are 1-based; state ids are 0-based. All coordinates are video px (1280x720).

Usage:
    .venv\\Scripts\\python.exe -m tests.gen_slides            # all variants -> data\\selftest\\
    .venv\\Scripts\\python.exe -m tests.gen_slides --force    # regenerate
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
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
DURATION = 40.0
VARIANTS = ("base", "dark", "fade", "pip")
GEN_VERSION = 3               # bump whenever the rendering or the truth changes

W, H = SIZE
N_FRAMES = int(round(DURATION * FPS))          # frame i is at t = i / FPS
FADE_SECONDS = 1.5
CROSSFADE_SECONDS = 0.6
WEBCAM_BOX = (1040, 535, 220, 165)             # x, y, w, h incl. a 2 px border
WEBCAM_NOISE_SIGMA = 6.0
PARK_CLEARANCE = 90           # parked pointer spots are this far (video px) from any content
ON_CONTENT_PX = 12            # hover / arrow-rest spots are this close to content

# State changes sit on 0.2 s steps, so every change starts exactly on a frame.
T_BUILD_1 = 5.6               # bullet 3 appears on slide 1 (instant build)
T_CUT = 12.6                  # cut to slide 2
T_INK = (20.0, 21.0)          # red pen underline drawn on slide 2 (the new state starts when it is complete)
T_XFADE = (26.4, 27.0)        # crossfade to slide 3 (the new state starts when it ends)
T_BUILD_2 = 33.6              # callout appears on slide 3 (instant build)

_EPS = 1e-9

# --------------------------------------------------------------------------------------
# Slide layout
# --------------------------------------------------------------------------------------
_DUPLEX = cv2.FONT_HERSHEY_DUPLEX
_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
_ITALIC = cv2.FONT_ITALIC
_BG = 255                                      # slide background (white)
_TITLE_COL = (105, 55, 25)                     # BGR
_BODY_COL = (45, 45, 45)
_MUTED_COL = (150, 150, 150)
_EXAMPLE_COL = (95, 95, 95)
_ACCENT_COL = (200, 140, 60)
_GRID_COL = (222, 222, 222)
_AXIS_COL = (90, 90, 90)
_INK_COL = (40, 40, 215)                       # red pen
_CALLOUT_FILL = (246, 230, 208)
_BAR_VALUES = (100, 150, 225, 338)
_BAR_COLS = ((200, 125, 45), (85, 170, 65), (165, 80, 150), (40, 175, 235))   # blue, green, purple, gold
# Coloured rectangles sit on even pixel boundaries, so yuv420p's 2x2 chroma blocks don't smear them.
_BAR_W, _BAR_X0, _BAR_STEP, _CHART_BASE = 100, 200, 136, 560


def _ink_box_rel(text: str, font: int, scale: float, thick: int, start: int = 0, end: int | None = None):
    """Ink box (coverage >= 50%) of text[start:end], relative to the putText origin of `text`.

    putText draws characters one after another from the same origin, so the ink of a word is
    exactly what text[:end] adds to text[:start].
    """
    end = len(text) if end is None else end
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 12 + 2 * thick + int(12 * scale)
    shape = (th + base + 2 * pad, tw + 2 * pad)
    org = (pad, pad + th)

    def draw(s: str) -> np.ndarray:
        c = np.zeros(shape, np.uint8)
        if s:
            cv2.putText(c, s, org, font, scale, 255, thick, cv2.LINE_AA)
        return c >= 128

    m = draw(text[:end]) & ~draw(text[:start])
    x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
    return x - org[0], y - org[1], w, h


def _union(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _el_text(eid, text, x, y, *, font=_DUPLEX, scale=1.0, thick=2, colour=_BODY_COL, kind="text", bullet=False):
    bx, by, bw, bh = _ink_box_rel(text, font, scale, thick)
    words = {}
    for m in re.finditer(r"\S+", text):
        wx, wy, ww, wh = _ink_box_rel(text, font, scale, thick, m.start(), m.end())
        words.setdefault(m.group(), [x + wx, y + wy, ww, wh])
    box = [x + bx, y + by, bw, bh]
    if bullet:
        box = _union(box, [x - 30, y - 16, 12, 12])

    def draw(img):
        if bullet:
            cv2.circle(img, ((x - 24) * 16, (y - 10) * 16), 5 * 16, _ACCENT_COL, -1, cv2.LINE_AA, 4)
        cv2.putText(img, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)
    return dict(id=eid, kind=kind, text=text, box=box, words=words, org=(x, y), draw=draw)


def _el_rect(eid, kind, x0, y0, x1, y1, colour, **extra):
    """Filled rectangle covering pixels x0..x1-1, y0..y1-1."""
    def draw(img):
        img[y0:y1, x0:x1] = colour
    return dict(id=eid, kind=kind, text="", box=[x0, y0, x1 - x0, y1 - y0], words={}, draw=draw, **extra)


def _rounded_rect(img, x0, y0, x1, y1, r, colour, thick):
    if thick < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), colour, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), colour, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r)):
            cv2.circle(img, (cx, cy), r, colour, -1, cv2.LINE_AA)
        return
    for p, q in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                 ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
        cv2.line(img, p, q, colour, thick, cv2.LINE_AA)
    for (cx, cy), a0 in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                         ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, colour, thick, cv2.LINE_AA)


def _el_callout(eid, text_el, pad_x=22, pad_y=16, r=10):
    x, y, w, h = text_el["box"]
    x0, y0, x1, y1 = x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y

    def draw(img):
        _rounded_rect(img, x0, y0, x1, y1, r, _CALLOUT_FILL, -1)
        _rounded_rect(img, x0, y0, x1, y1, r, _ACCENT_COL, 2)
        text_el["draw"](img)
    return dict(id=eid, kind="callout", text=text_el["text"], box=[x0 - 1, y0 - 1, x1 - x0 + 3, y1 - y0 + 3],
                words=text_el["words"], draw=draw)


def _title(sid, s):
    return _el_text(f"{sid}_title", s, 80, 96, scale=1.5, thick=2, colour=_TITLE_COL, kind="title")


def _rule(sid):
    return _el_rect(f"{sid}_rule", "rule", 80, 120, 1200, 124, _ACCENT_COL)


def _footer(sid, n):
    return _el_text(f"{sid}_footer", f"Replay Board self-test   {n} / 3", 80, 697,
                    font=_SIMPLEX, scale=0.55, thick=1, colour=_MUTED_COL, kind="footer")


def _bullet(eid, s, y):
    return _el_text(eid, s, 118, y, bullet=True, kind="bullet")


def _centred_text(eid, s, cx, y, **kw):
    tw = cv2.getTextSize(s, kw.get("font", _DUPLEX), kw.get("scale", 1.0), kw.get("thick", 2))[0][0]
    return _el_text(eid, s, int(round(cx - tw / 2)), y, **kw)


def _build_slides():
    s1 = [_title("s1", "Compound interest"), _rule("s1"),
          _bullet("s1_b1", "Interest is added to the balance each year", 210),
          _bullet("s1_b2", "Next year you earn interest on the interest", 280),
          _bullet("s1_b3", "So the balance grows faster and faster", 350),          # build step
          _el_text("s1_ex", "Example: 1000 at 10% gives 1100, 1210, 1331", 118, 455,
                   font=_SIMPLEX | _ITALIC, scale=0.85, thick=2, colour=_EXAMPLE_COL),
          _footer("s1", 1)]

    s2 = [_title("s2", "Balance with 50% interest"), _rule("s2")]
    for v in (100, 200, 300):
        s2.append(_el_rect(f"s2_grid{v}", "grid", 160, _CHART_BASE - v, 740, _CHART_BASE - v + 1, _GRID_COL))
    for k, (val, col) in enumerate(zip(_BAR_VALUES, _BAR_COLS)):
        x0, top = _BAR_X0 + _BAR_STEP * k, _CHART_BASE - 2 * int(round(val / 2))
        s2.append(_el_rect(f"s2_bar{k}", "bar", x0, top, x0 + _BAR_W, _CHART_BASE, col, value=val))
        s2.append(_centred_text(f"s2_val{k}", str(val), x0 + _BAR_W / 2, top - 10, scale=0.75, thick=2, kind="label"))
        s2.append(_centred_text(f"s2_year{k}", f"Year {k}", x0 + _BAR_W / 2, 590,
                                font=_SIMPLEX, scale=0.65, thick=1, colour=_AXIS_COL, kind="label"))
    s2.append(_el_rect("s2_yaxis", "axis", 159, 175, 161, 562, _AXIS_COL))
    s2.append(_el_rect("s2_xaxis", "axis", 159, 559, 742, 561, _AXIS_COL))
    for v in (0, 100, 200, 300):
        s = str(v)
        tw = cv2.getTextSize(s, _SIMPLEX, 0.6, 1)[0][0]
        s2.append(_el_text(f"s2_tick{v}", s, 150 - tw, _CHART_BASE - v + 6, font=_SIMPLEX, scale=0.6, thick=1,
                           colour=_AXIS_COL, kind="label"))
    s2 += [_el_text("s2_c1", "Each year adds 50%", 830, 225, scale=0.9),
           _el_text("s2_c2", "of the balance, so", 830, 262, scale=0.9),
           _el_text("s2_c3", "growth speeds up", 830, 299, scale=0.9),
           _footer("s2", 2)]

    b3 = _bullet("s3_b3", "n = 365 gives A = 2.715", 455)
    eq = _el_text("s3_eq", "A = (1 + 1/n)", 150, 300, scale=1.5, thick=2, kind="equation")
    sup = _el_text("s3_eq_sup", "n", eq["box"][0] + eq["box"][2] + 6, 300 - 26, scale=0.95, thick=2, kind="equation")
    # The callout is placed so the A->B arrow ('2.715' -> '2.71828') is 350 px long, 30 degrees up
    # to the right: steep enough (|dx| = 1.7 |dy|) that it can never pass as a sweep (|dx| >= 2.5 |dy|).
    ax_, ay_ = _box_centre(b3["words"]["2.715"])
    bx_, by_ = ax_ + 350 * math.cos(math.radians(-30)), ay_ + 350 * math.sin(math.radians(-30))
    probe = _el_text("probe", "e = 2.71828", 0, 0, scale=1.2, thick=2)
    wx, wy = _box_centre(probe["words"]["2.71828"])
    call_txt = _el_text("s3_callout_text", "e = 2.71828", int(round(bx_ - wx)), int(round(by_ - wy)),
                        scale=1.2, thick=2, colour=_TITLE_COL, kind="callout")
    s3 = [_title("s3", "Towards the number e"), _rule("s3"),
          _bullet("s3_b1", "Rate 100%, n times a year", 210),
          eq, sup,
          _bullet("s3_b2", "n = 12 gives A = 2.613", 390),
          b3,
          _el_callout("s3_callout", call_txt),                                        # build step
          _footer("s3", 3)]
    return {"s1": s1, "s2": s2, "s3": s3}


def _box_centre(box) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


_SLIDE_ELS = _build_slides()
_EL = {el["id"]: el for els in _SLIDE_ELS.values() for el in els}
# what each state image shows (element ids left out of the first state of a slide)
_IMAGE_ELS = {
    "s1a": [e for e in _SLIDE_ELS["s1"] if e["id"] != "s1_b3"],
    "s1b": _SLIDE_ELS["s1"],
    "s2a": _SLIDE_ELS["s2"],
    "s2b": _SLIDE_ELS["s2"],          # + the full pen ink
    "s3a": [e for e in _SLIDE_ELS["s3"] if e["id"] != "s3_callout"],
    "s3b": _SLIDE_ELS["s3"],
}

# --------------------------------------------------------------------------------------
# Red pen ink (a hand-drawn underline under the last caption line of slide 2)
# --------------------------------------------------------------------------------------
_INK_THICK = 5


def _ink_curve() -> np.ndarray:
    x, y, w, h = _EL["s2_c3"]["box"]
    u = np.linspace(0.0, 1.0, 80)
    xs = x - 8 + (w + 18) * u
    ys = (y + h + 9) + 3.0 * u - 2.0 * np.sin(np.pi * u) + 1.2 * np.sin(2 * np.pi * 2.3 * u + 0.4)
    ys -= 5.0 * np.clip((u - 0.94) / 0.06, 0, 1) ** 2          # pen flicks up as it lifts
    return np.stack([xs, ys], 1)


_INK_PTS = _ink_curve()
_INK_CUM = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(_INK_PTS, axis=0).T))])


def _ink_progress(t: float) -> float:
    """Share of the stroke's length drawn at t (fast, nearly even pen speed)."""
    v = min(max((t - T_INK[0]) / (T_INK[1] - T_INK[0]), 0.0), 1.0)
    return 0.85 * v + 0.15 * _smoothstep(v)


def _draw_ink(img: np.ndarray, progress: float) -> None:
    if progress <= 0.0:
        return
    L = _INK_CUM[-1] * min(progress, 1.0)
    k = int(np.searchsorted(_INK_CUM, L, side="right"))
    pts = _INK_PTS[:k]
    if k < len(_INK_PTS):
        a, b = _INK_PTS[k - 1], _INK_PTS[k]
        f = (L - _INK_CUM[k - 1]) / max(_INK_CUM[k] - _INK_CUM[k - 1], 1e-9)
        pts = np.vstack([pts, a + (b - a) * f])
    if len(pts) == 1:
        pts = np.vstack([pts, pts + 0.5])
    cv2.polylines(img, [np.round(pts * 16).astype(np.int32)], False, _INK_COL, _INK_THICK, cv2.LINE_AA, 4)


def _ink_box() -> list[int]:
    m = np.zeros((H, W), np.uint8)
    cv2.polylines(m, [np.round(_INK_PTS * 16).astype(np.int32)], False, 255, _INK_THICK, cv2.LINE_AA, 4)
    x, y, w, h = cv2.boundingRect((m >= 128).astype(np.uint8))
    return [int(x), int(y), int(w), int(h)]


@functools.lru_cache(maxsize=None)
def _image(key: str) -> np.ndarray:
    """uint8 BGR slide image for a state (no webcam, no pointer). Treat as read-only."""
    img = np.full((H, W, 3), _BG, np.uint8)
    for el in _IMAGE_ELS[key]:
        el["draw"](img)
    if key == "s2b":
        _draw_ink(img, 1.0)
    img.setflags(write=False)
    return img


# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------
_STATES = (
    dict(id=0, kind="slide", slide=1, image="s1a", t_start=0.0, t_end=T_BUILD_1, transition="start",
         note="slide 1: title, bullets 1-2, example line; the cursor is parked on empty space for the whole state"),
    dict(id=1, kind="addition", slide=1, image="s1b", t_start=T_BUILD_1, t_end=T_CUT, transition="build",
         note="bullet 3 appears (instant build)"),
    dict(id=2, kind="slide", slide=2, image="s2a", t_start=T_CUT, t_end=T_INK[0], transition="cut",
         note="slide 2: bar chart with 4 coloured bars (cut); the laser is used here"),
    dict(id=3, kind="addition", slide=2, image="s2b", t_start=T_INK[1], t_end=T_XFADE[0], transition="ink",
         note=f"red pen underline under 'growth speeds up', drawn {T_INK[0]:.1f}-{T_INK[1]:.1f} s; "
              "the cursor is parked on empty space for the whole state"),
    dict(id=4, kind="slide", slide=3, image="s3a", t_start=T_XFADE[1], t_end=T_BUILD_2, transition="crossfade",
         note=f"slide 3 after a {CROSSFADE_SECONDS} s crossfade {T_XFADE[0]:.1f}-{T_XFADE[1]:.1f} s"),
    dict(id=5, kind="addition", slide=3, image="s3b", t_start=T_BUILD_2, t_end=DURATION, transition="build",
         note="callout 'e = 2.71828' appears (instant build)"),
)
_PIP_STATES = (
    dict(id=0, kind="slide", slide=3, image="s3b", t_start=0.0, t_end=DURATION, transition="start",
         note="one still slide (slide 3, fully built) with a picture-in-picture webcam; no pointer"),
)
# (t_start, t_end, kind) of every change of the slide picture in base/dark/fade
_CHANGES = ((T_BUILD_1, T_BUILD_1, "build"), (T_CUT, T_CUT, "cut"), (T_INK[0], T_INK[1], "ink"),
            (T_XFADE[0], T_XFADE[1], "crossfade"), (T_BUILD_2, T_BUILD_2, "build"))


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def _state_defs(variant: str):
    _check_variant(variant)
    return _PIP_STATES if variant == "pip" else _STATES


# --------------------------------------------------------------------------------------
# Pointer path
# --------------------------------------------------------------------------------------
def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _minjerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def _circle_xy(seg, s: float) -> tuple[float, float]:
    """Hand-drawn loop: an ellipse with a slight wobble that closes a little inside itself."""
    cx, cy = seg["c"]
    th = math.radians(seg["a0"] + seg["sweep"] * s)
    r = (1.0 - 0.07 * s) * (1.0 + 0.035 * math.sin(2.0 * th + 0.6))
    return cx + seg["ax"] * r * math.cos(th), cy + seg["ay"] * r * math.sin(th)


def _seg_xy(seg, u: float) -> tuple[float, float]:
    u = min(max(u, 0.0), 1.0)
    kind = seg["kind"]
    if kind == "rest":
        return seg["p"]
    if kind == "move":                       # minimum-jerk reach with a slight bow
        (px, py), (qx, qy) = seg["p"], seg["q"]
        s = _minjerk(u)
        L = math.hypot(qx - px, qy - py) or 1.0
        nx, ny = -(qy - py) / L, (qx - px) / L
        off = seg["bow"] * L * 4.0 * s * (1.0 - s)
        return px + (qx - px) * s + nx * off, py + (qy - py) * s + ny * off
    if kind == "sweep":                      # near-constant speed, tiny sag
        (px, py), (qx, qy) = seg["p"], seg["q"]
        s = 0.5 * u + 0.5 * _smoothstep(u)
        return px + (qx - px) * s, py + (qy - py) * s + 0.8 * math.sin(math.pi * s)
    if kind == "circle":
        return _circle_xy(seg, 0.75 * u + 0.25 * _smoothstep(u))
    if kind == "wiggle":                     # figure-of-eight: the turns alternate, so it never reads as a loop
        cx, cy = seg["c"]
        e = _smoothstep(u / 0.12) * _smoothstep((1.0 - u) / 0.12)
        ph = 2.0 * math.pi * seg["cycles"] * u
        return cx + seg["amp"][0] * e * math.sin(ph), cy + seg["amp"][1] * e * math.sin(2.0 * ph)
    raise AssertionError(kind)


class _Path:
    """Builds the pointer timeline as contiguous segments [t0, t1)."""

    def __init__(self, tool: str, p):
        self.t, self.tool, self.p, self.segs = 0.0, tool, (float(p[0]), float(p[1])), []

    def _add(self, t1, kind, label, gesture=None, **kw):
        seg = dict(t0=self.t, t1=float(t1), tool=self.tool, kind=kind, label=label, p=self.p, gesture=gesture, **kw)
        end = _seg_xy(seg, 1.0)
        self.segs.append(seg)
        self.t, self.p = float(t1), (float(end[0]), float(end[1]))
        return self

    def rest(self, t1, label, gesture=None):
        return self._add(t1, "rest", label, gesture)

    def move(self, t1, q, label="travel", bow=0.05, gesture=None):
        return self._add(t1, "move", label, gesture, q=(float(q[0]), float(q[1])), bow=bow)

    def sweep(self, t1, q, gesture=None):
        return self._add(t1, "sweep", "sweep", gesture, q=(float(q[0]), float(q[1])))

    def circle(self, t1, spec, gesture=None):
        """A loop (see `_circle_spec`); the pointer must already be at its start, `_circle_xy(spec, 0)`."""
        return self._add(t1, "circle", "circle", gesture, **spec)

    def wiggle(self, t1, amp, cycles, gesture=None):
        return self._add(t1, "wiggle", "wiggle", gesture, c=self.p, amp=amp, cycles=cycles)

    def use(self, tool):
        self.tool = tool
        return self


def _circle_spec(box, pad_x, pad_y, a0, sweep=370.0):
    c = _box_centre(box)
    return dict(c=c, ax=box[2] / 2.0 + pad_x, ay=box[3] / 2.0 + pad_y, a0=a0, sweep=sweep)


def _on_word(box, fx=0.4, fy=0.55):
    return (round(box[0] + fx * box[2]), round(box[1] + fy * box[3]))


def _build_path():
    # Parked spots: empty space in every state they are visible in (checked below). They are placed so
    # every travel move is steep (|dx| < 2|dy|) or fast, so no travel move can pass as a sweep.
    P0, P1, P2, P3, P4 = (720, 610), (900, 620), (860, 640), (660, 590), (1100, 460)
    # targets
    w_interest = _EL["s1_b1"]["words"]["Interest"]
    H1 = _on_word(w_interest)
    b2 = _EL["s1_b2"]
    sweep_words = [b2["words"][w] for w in ("Next", "year", "you", "earn")]
    SW0 = (sweep_words[0][0] - 4, b2["org"][1] + 12)
    SW1 = (SW0[0] + 300, SW0[1] + 2)
    bar1, bar3, val3 = _EL["s2_bar1"]["box"], _EL["s2_bar3"]["box"], _EL["s2_val3"]["box"]
    laser_circle_box = _union(val3, [bar3[0], bar3[1], bar3[2], 12])
    LC = _circle_spec(laser_circle_box, 30, 22, a0=20.0)
    LH = (round(bar1[0] + bar1[2] / 2), bar1[1] + 70)
    w_2613 = _EL["s3_b2"]["words"]["2.613"]
    CC = _circle_spec(w_2613, 30, 22, a0=15.0)
    w_100 = _EL["s3_b1"]["words"]["100%,"]
    WC = (round(w_100[0] + w_100[2] / 2), round(w_100[1] + w_100[3] / 2))
    w_2715 = _EL["s3_b3"]["words"]["2.715"]
    w_e = _EL["s3_callout"]["words"]["2.71828"]
    A = (round(w_2715[0] + w_2715[2] / 2), round(w_2715[1] + w_2715[3] / 2))
    B = (round(w_e[0] + w_e[2] / 2), round(w_e[1] + w_e[3] / 2))

    def g(type_, target, target_box, **kw):
        return dict(type=type_, target=target, target_box=[int(v) for v in target_box], **kw)

    p = _Path("cursor", P0)
    p.rest(6.4, "parked")                                                     # all of state 0
    p.move(7.0, H1)                                                           # travel from empty space
    p.rest(8.6, "hover", g("hover", "bullet 1 'Interest is added to the balance each year' (word 'Interest')",
                           _EL["s1_b1"]["box"]))
    p.move(9.0, SW0, bow=-0.05)                                               # travel, ends in a short rest
    p.rest(9.3, "short rest")
    p.sweep(10.1, SW1, g("sweep", "underline of bullet 2 'Next year you earn'",
                         _union(_union(sweep_words[0], sweep_words[1]), _union(sweep_words[2], sweep_words[3])),
                         from_=SW0, to=SW1))
    p.rest(10.4, "short rest")
    p.move(11.0, P1)                                                          # travel, ends parked on empty space
    p.rest(13.6, "parked")                                                    # across the cut
    p.use("laser").rest(13.9, "laser on")                                     # laser replaces the cursor
    p.move(14.5, _circle_xy(LC, 0.0))                                         # travel from empty space
    p.circle(16.0, LC, g("circle", "tallest bar (Year 3) and its label '338'", laser_circle_box))
    p.move(16.6, LH, bow=0.04)                                                # no rest before it
    p.rest(18.2, "hover", g("hover", "green bar 'Year 1' (150)", bar1))
    p.move(18.54, P2)                                                         # quick flick; laser off on arrival
    p.use("cursor").rest(28.0, "parked")                                      # ink, all of state 3, crossfade
    p.move(28.6, _circle_xy(CC, 0.0))                                         # travel from empty space
    p.circle(30.1, CC, g("circle", "'2.613' in bullet 'n = 12 gives A = 2.613'", w_2613))
    p.move(30.5, WC, bow=0.04)                                                # no rest before it
    # Short rests on both sides of the wiggle: they break the heading chain, so the wiggle's
    # back-and-forth turns can't add up with the turns of the moves into and out of it (and the
    # moves' height) into a false circle once the tracked positions carry ~1 px of jitter.
    p.rest(30.8, "short rest")
    p.wiggle(32.0, amp=(40.0, 5.0), cycles=2.5,
             gesture=g("wiggle", "'100%' in bullet 'Rate 100%, n times a year'", w_100))
    p.rest(32.3, "short rest")
    p.move(32.9, P3)                                                          # travel, ends parked
    p.rest(34.6, "parked")                                                    # across the build
    p.move(35.2, A, bow=-0.05)                                                # travel from empty space
    p.rest(35.7, "rest before arrow")                                         # 0.5 s on '2.715' (not a hover)
    p.move(36.3, B, label="arrow", bow=0.02,
           gesture=g("arrow", "from '2.715' to the callout 'e = 2.71828'", _union(w_2715, w_e), from_=A, to=B))
    p.rest(37.6, "hover", g("hover", "callout 'e = 2.71828' (end of the arrow)", _EL["s3_callout"]["box"]))
    p.move(38.2, P4)                                                          # travel, ends parked
    p.rest(DURATION, "parked")
    assert abs(p.t - DURATION) < _EPS
    return p.segs


_SEGS = _build_path()
_SEG_T1 = np.array([s["t1"] for s in _SEGS])


def _seg_at(t: float):
    k = int(np.searchsorted(_SEG_T1, t + _EPS, side="right"))
    return _SEGS[min(k, len(_SEGS) - 1)]


def _pointer_base(t: float):
    if t < -_EPS or t > DURATION + _EPS:
        return None
    seg = _seg_at(t)
    x, y = _seg_xy(seg, (t - seg["t0"]) / (seg["t1"] - seg["t0"]))
    return seg["tool"], float(math.floor(x + 0.5)), float(math.floor(y + 0.5))   # screen pointers sit on whole px


# --------------------------------------------------------------------------------------
# Public truth functions
# --------------------------------------------------------------------------------------
def states(variant: str = "base") -> list[dict]:
    """True slide states: {id, kind: slide|addition, t_start, t_end, slide (1-based), note, transition}.

    t_start is when the state's picture is complete (end of a crossfade, end of the pen ink);
    t_end is when the picture starts to change into the next state.
    """
    return [{k: v for k, v in s.items() if k != "image"} for s in _state_defs(variant)]


def webcam_box(variant: str = "base") -> list[int]:
    """[x, y, w, h] of the webcam (live) region in video px, including its 2 px border."""
    _check_variant(variant)
    return list(WEBCAM_BOX)


def _path_box(t0: float, t1: float) -> list[int]:
    pts = [_pointer_base(t)[1:] for t in np.linspace(t0, t1 - 1e-6, 241)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [int(min(xs)), int(min(ys)), int(max(xs) - min(xs) + 1), int(max(ys) - min(ys) + 1)]


def gestures(variant: str = "base") -> list[dict]:
    """Gestures the analysis must find: {type, tool, t_start, t_end, box, from, to, target, target_box}.

    box is the bounding box of the pointer path during the gesture (cursor tip / laser centre);
    target_box is the slide content the gesture is about. The arrow's time is the A->B move
    only; the hover that ends it is listed on its own.
    """
    _check_variant(variant)
    if variant == "pip":
        return []
    out = []
    for seg in _SEGS:
        g = seg["gesture"]
        if not g:
            continue
        fr, to = g.get("from_"), g.get("to")
        out.append(dict(type=g["type"], tool=seg["tool"], t_start=round(seg["t0"], 3), t_end=round(seg["t1"], 3),
                        box=_path_box(seg["t0"], seg["t1"]),
                        **{"from": [float(fr[0]), float(fr[1])] if fr else None},
                        to=[float(to[0]), float(to[1])] if to else None,
                        target=g["target"], target_box=list(g["target_box"])))
    return out


def parked(variant: str = "base") -> list[dict]:
    """Pointer rests on empty space (>= PARK_CLEARANCE px from any content): not hovers."""
    _check_variant(variant)
    if variant == "pip":
        return []
    return [dict(tool=s["tool"], t_start=s["t0"], t_end=s["t1"], pos=list(s["p"]))
            for s in _SEGS if s["label"] == "parked"]


def pointer(t: float, variant: str = "base"):
    """Ground-truth pointer at time t: None or (tool, x, y) with tool 'cursor' (tip) or 'laser' (centre).

    The cursor is drawn through transitions too (it is an overlay); the pip variant has none.
    """
    _check_variant(variant)
    if variant == "pip":
        return None
    return _pointer_base(t)


def _state_image_key(state_id: int, variant: str) -> str:
    defs = _state_defs(variant)
    if not 0 <= state_id < len(defs):
        raise ValueError(f"no state {state_id} in variant {variant!r}")
    return defs[state_id]["image"]


def clean_slide(state_id: int, variant: str = "base") -> np.ndarray:
    """BGR uint8 (720, 1280, 3) slide of a state as it should be reconstructed: no pointer,
    the webcam box filled with the slide background colour."""
    img = _image(_state_image_key(state_id, variant)).copy()
    x, y, w, h = WEBCAM_BOX
    img[y:y + h, x:x + w] = _BG
    if variant == "dark":
        img = cv2.bitwise_not(img)
    return img


def transitions(variant: str = "base") -> list[dict]:
    """Every change of the slide picture: {t_start, t_end, kind: build|cut|ink|crossfade}."""
    _check_variant(variant)
    if variant == "pip":
        return []
    return [dict(t_start=a, t_end=b, kind=k) for a, b, k in _CHANGES]


# --------------------------------------------------------------------------------------
# Layout checks (run at import, like gen_board)
# --------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _content_mask(key: str) -> np.ndarray:
    """Pixels differing from the white background by more than 25 in any channel (webcam box excluded)."""
    b, g, r = cv2.split(_image(key))
    m = cv2.min(cv2.min(b, g), r) < _BG - 25
    x, y, w, h = WEBCAM_BOX
    m[y:y + h, x:x + w] = False
    return m


@functools.lru_cache(maxsize=None)
def _dist_to_content(key: str) -> np.ndarray:
    return cv2.distanceTransform((~_content_mask(key)).astype(np.uint8), cv2.DIST_L2, 5)


def _keys_during(t0: float, t1: float) -> set[str]:
    return {s["image"] for s in _STATES if s["t_start"] - 1.0 < t1 and t0 < s["t_end"] + 1.0}


def _check_layout() -> None:
    wx, wy, ww, wh = WEBCAM_BOX
    # content keeps clear of the webcam box (its detected box may grow by up to 15%)
    for key in _IMAGE_ELS:
        m = _content_mask(key)
        assert not m[wy - 45:wy + wh, wx - 45:wx + ww].any(), key
    # path is continuous, stays on screen and away from the webcam
    for a, b in zip(_SEGS, _SEGS[1:]):
        assert abs(a["t1"] - b["t0"]) < _EPS
        ea, sb = _seg_xy(a, 1.0), _seg_xy(b, 0.0)
        assert math.hypot(ea[0] - sb[0], ea[1] - sb[1]) < 0.5, (a["label"], b["label"])
    for i in range(N_FRAMES + 1):
        _, x, y = _pointer_base(i / FPS)
        assert 40 <= x <= W - 40 and 40 <= y <= H - 40, (i, x, y)
        dx = max(wx - x, 0, x - (wx + ww))
        dy = max(wy - y, 0, y - (wy + wh))
        assert math.hypot(dx, dy) >= 60, (i, x, y)
    for s in _SEGS:
        keys = _keys_during(s["t0"], s["t1"])
        x, y = (int(round(v)) for v in s["p"])
        if s["label"] == "parked":                       # empty space in every state it is seen in
            for key in keys:
                assert _dist_to_content(key)[y, x] >= PARK_CLEARANCE, (s["t0"], key, _dist_to_content(key)[y, x])
        if s["label"] in ("hover", "rest before arrow"):  # on content
            key = next(st["image"] for st in _STATES if st["t_start"] <= s["t0"] < st["t_end"])
            assert _dist_to_content(key)[y, x] <= ON_CONTENT_PX, (s["t0"], key)
    # nothing moves within 0.5 s of a change of the slide picture; gestures stay clear of them
    for s in _SEGS:
        if s["kind"] == "rest":
            continue
        for a, b, _ in _CHANGES:
            assert s["t1"] + 0.5 <= a + _EPS or s["t0"] >= b + 0.5 - _EPS, (s["t0"], s["label"], a)
    for s in _SEGS:
        if s["gesture"]:
            assert any(st["t_start"] <= s["t0"] and s["t1"] <= st["t_end"] for st in _STATES), s["t0"]
    # the cursor is parked for the whole of states 0 and 3
    for sid in (0, 3):
        st = _STATES[sid]
        assert any(s["label"] == "parked" and s["t0"] <= st["t_start"] and s["t1"] >= st["t_end"] for s in _SEGS), sid
    # SPEC 5.2 sweep: straight, |dx| >= 2.5|dy|, >= 0.35 s, <= 700 px/s at 960 px (933 video px/s).
    # Travel moves are steep (|dx| < 2|dy|) or fast (>= 1250 video px/s), so none can pass as one;
    # the true sweep is well under the speed limit.
    for s in _SEGS:
        if s["kind"] in ("move", "sweep"):
            (px, py), (qx, qy) = s["p"], s["q"]
            speed = math.hypot(qx - px, qy - py) / (s["t1"] - s["t0"])
            if s["kind"] == "sweep":
                assert abs(qx - px) >= 4 * abs(qy - py) and speed <= 0.6 * 933, s["t0"]
            elif s["label"] == "travel":
                assert abs(qx - px) < 2 * abs(qy - py) or speed >= 1250, (s["t0"], speed)
    # the arrow is ~350 px and too steep to pass as a sweep
    arrow = next(s for s in _SEGS if s["label"] == "arrow")
    adx, ady = arrow["q"][0] - arrow["p"][0], arrow["q"][1] - arrow["p"][1]
    assert 330 <= math.hypot(adx, ady) <= 380 and abs(adx) <= 2.0 * abs(ady)
    # text elements of a slide never touch; the callout keeps 20 px clear of everything
    for els in _SLIDE_ELS.values():
        texts = [e for e in els
                 if e["kind"] in ("title", "bullet", "text", "equation", "callout") and e["id"] != "s3_eq_sup"]
        for i, ea in enumerate(texts):
            for eb in texts[i + 1:]:
                a, b = ea["box"], eb["box"]
                gap = max(b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]), b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]))
                assert gap >= (20 if "callout" in (ea["kind"], eb["kind"]) else 6), (ea["id"], eb["id"], gap)


_check_layout()

# --------------------------------------------------------------------------------------
# Pointer sprites
# --------------------------------------------------------------------------------------
_CURSOR_ROWS = (
    "B",
    "BB",
    "BWB",
    "BWWB",
    "BWWWB",
    "BWWWWB",
    "BWWWWWB",
    "BWWWWWWB",
    "BWWWWWWWB",
    "BWWWWWWWWB",
    "BWWWWWWWWWB",
    "BWWWWWWBBBBB",
    "BWWWBWWB",
    "BWWBBWWB",
    "BWB  BWWB",
    "BB   BWWB",
    "B     BWWB",
    "      BWWB",
    "       BB",
)


def _cursor_masks():
    h, w = len(_CURSOR_ROWS), max(len(r) for r in _CURSOR_ROWS)
    black = np.zeros((h, w), bool)
    white = np.zeros((h, w), bool)
    for y, row in enumerate(_CURSOR_ROWS):
        for x, ch in enumerate(row):
            black[y, x] = ch == "B"
            white[y, x] = ch == "W"
    return black, white


_CUR_BLACK, _CUR_WHITE = _cursor_masks()          # 19 x 12, tip (hot spot) at (0, 0)


def _laser_sprite():
    R = 12
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.hypot(xx, yy).astype(np.float32)
    core = np.clip(5.5 - r, 0.0, 1.0)                                   # ~10 px dot
    glow = 0.45 * np.exp(-(np.maximum(r - 3.0, 0.0) / 3.5) ** 2)        # soft halo
    return R, core[..., None], glow[..., None]


_LASER_R, _LASER_CORE, _LASER_GLOW = _laser_sprite()
_LASER_CORE_COL = np.array((35, 35, 255), np.float32)
_LASER_GLOW_COL = np.array((60, 60, 255), np.float32)


def _draw_pointer(frame: np.ndarray, ptr) -> None:
    if ptr is None:
        return
    tool, x, y = ptr
    x, y = int(x), int(y)
    if tool == "cursor":
        h, w = _CUR_BLACK.shape
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0:
            return
        sub = frame[y0:y1, x0:x1]
        mb = _CUR_BLACK[y0 - y:y1 - y, x0 - x:x1 - x]
        mw = _CUR_WHITE[y0 - y:y1 - y, x0 - x:x1 - x]
        sub[mb] = 0
        sub[mw] = 255
        return
    R = _LASER_R
    x0, y0, x1, y1 = max(x - R, 0), max(y - R, 0), min(x + R + 1, W), min(y + R + 1, H)
    sl = (slice(y0 - (y - R), y1 - (y - R)), slice(x0 - (x - R), x1 - (x - R)))
    sub = frame[y0:y1, x0:x1].astype(np.float32)
    g, c = _LASER_GLOW[sl], _LASER_CORE[sl]
    sub = sub + g * (_LASER_GLOW_COL - sub)
    sub = sub + c * (_LASER_CORE_COL - sub)
    frame[y0:y1, x0:x1] = np.clip(np.rint(sub), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Webcam picture (a person in a room)
# --------------------------------------------------------------------------------------
_CAM_BORDER = 2
_CAM_BORDER_COL = (58, 58, 58)
_CX0, _CY0 = WEBCAM_BOX[0] + _CAM_BORDER, WEBCAM_BOX[1] + _CAM_BORDER
_CW, _CH = WEBCAM_BOX[2] - 2 * _CAM_BORDER, WEBCAM_BOX[3] - 2 * _CAM_BORDER
_SHIRT = np.array((128, 86, 52), np.float32)
_SKIN = np.array((128, 160, 205), np.float32)
_HAIR = np.array((36, 46, 60), np.float32)
_PERSON_X, _HEAD_Y = 116.0, 64.0
_CAM_YY, _CAM_XX = np.mgrid[0:_CH, 0:_CW].astype(np.float32)


@functools.lru_cache(maxsize=1)
def _cam_room() -> np.ndarray:
    """Static float32 room behind the person: wall, bookshelf, a picture, a desk edge."""
    rng = np.random.default_rng(20261101)
    yy, xx = _CAM_YY, _CAM_XX
    v = yy / _CH
    img = (np.array((160, 170, 180), np.float32) * (1 - v)[..., None]
           + np.array((126, 138, 150), np.float32) * v[..., None])
    img *= (1.0 - 0.18 * ((xx - 0.6 * _CW) / _CW) ** 2)[..., None]
    # bookshelf
    img[12:, 0:50] = (44, 66, 92)
    img[16:, 4:46] = (30, 44, 62)
    palette = ((40, 40, 150), (130, 90, 40), (60, 120, 60), (40, 150, 190), (120, 60, 110), (170, 170, 160))
    for sy in (56, 100, 144):
        img[sy:sy + 4, 0:50] = (52, 78, 108)
        x = 6
        while x < 42:
            bw, bh = int(rng.integers(4, 8)), int(rng.integers(26, 37))
            col = np.array(palette[int(rng.integers(len(palette)))], np.float32) * float(rng.uniform(0.8, 1.1))
            img[sy - bh:sy, x:min(x + bw, 44)] = col
            x += bw + 1
    # framed picture
    img[18:54, 174:208] = (62, 62, 70)
    img[21:51, 177:205] = (170, 140, 110)
    tri = np.array([[178, 50], [190, 32], [204, 50]], np.int32)
    cv2.fillPoly(img, [tri], (90, 120, 80))
    # desk edge at the bottom
    img[150:, :] = (72, 98, 128)
    img[150:152, :] = (95, 124, 150)
    img += rng.normal(0, 1.0, img.shape).astype(np.float32)          # fixed texture, not noise
    return img


def _cam_pose(t: float, calm: bool) -> dict:
    tau = 2 * math.pi
    if calm:   # breathing only
        b = math.sin(tau * t / 4.2)
        return dict(hx=0.35 * math.sin(tau * t / 6.3), hy=0.6 * b, bx=0.0, by=0.8 * b, mouth=0.0, blink=False)
    hx = 11.0 * math.sin(tau * t / 2.9) + 5.0 * math.sin(tau * t / 1.13 + 1.3) + 2.5 * math.sin(tau * t / 0.71 + 0.2)
    hy = 3.5 * math.sin(tau * t / 1.7 + 0.4) + 2.0 * math.sin(tau * t / 0.83 + 2.0)
    bx = 8.0 * math.sin(tau * (t - 0.25) / 2.9) + 2.0 * math.sin(tau * t / 1.9 + 0.5)
    by = 1.8 * math.sin(tau * t / 3.7) + 1.0 * math.sin(tau * t / 1.3 + 0.9)
    syll = 0.5 + 0.5 * math.sin(tau * 4.1 * t + 0.3 * math.sin(tau * 0.7 * t))
    mouth = (0.75 + 0.25 * math.sin(tau * t / 6.1)) * syll ** 1.5
    blink = (t + 0.7) % 3.3 < 0.14
    return dict(hx=hx, hy=hy, bx=bx, by=by, mouth=mouth, blink=blink)


def _P(pts) -> np.ndarray:
    return np.round(np.asarray(pts, np.float64) * 16.0).astype(np.int32)


def _pt(x, y):
    return int(round(x * 16)), int(round(y * 16))


def _cam_frame(t: float, calm: bool) -> np.ndarray:
    """float32 (CH, CW, 3) clean webcam picture at t."""
    img = _cam_room().copy()
    q = _cam_pose(t, calm)
    bx, by = _PERSON_X + q["bx"], q["by"]
    hx, hy = _PERSON_X + q["hx"], _HEAD_Y + q["hy"]
    xx, yy = _CAM_XX, _CAM_YY
    canvas = np.zeros((_CH, _CW), np.uint8)

    def comp(colour, draw):
        canvas.fill(0)
        draw(canvas)
        x, y, w, h = cv2.boundingRect(canvas)
        if w == 0 or h == 0:
            return
        a = canvas[y:y + h, x:x + w].astype(np.float32)[..., None] * (1.0 / 255.0)
        c = colour[y:y + h, x:x + w] if isinstance(colour, np.ndarray) and colour.ndim == 3 else np.asarray(colour, np.float32)
        sub = img[y:y + h, x:x + w]
        sub += a * (c - sub)

    def poly(pts):
        return lambda c: cv2.fillPoly(c, [_P(pts)], 255, cv2.LINE_AA, 4)

    def ell(x, y, ax, ay, ang=0.0):
        return lambda c: cv2.ellipse(c, _pt(x, y), (int(round(ax * 16)), int(round(ay * 16))), ang, 0, 360,
                                     255, -1, cv2.LINE_AA, 4)

    def line(x0, y0, x1, y1, th):
        return lambda c: cv2.line(c, _pt(x0, y0), _pt(x1, y1), 255, th, cv2.LINE_AA, 4)

    # shirt with a small check pattern that moves with the body
    check = np.sin(2 * np.pi * (xx - bx) / 9.0) * np.sin(2 * np.pi * (yy - by) / 9.0)
    shade = 0.85 + 0.15 * np.cos(np.clip((xx - bx) / 100.0, -1, 1) * np.pi / 2)
    shirt = _SHIRT * ((1.0 + 0.22 * check) * shade)[..., None]
    comp(shirt, poly([(bx - 100, _CH + 4), (bx - 94, 138 + by), (bx - 72, 121 + by), (bx - 24, 112 + by),
                      (bx + 24, 112 + by), (bx + 72, 121 + by), (bx + 94, 138 + by), (bx + 100, _CH + 4)]))
    comp(_SKIN * 0.82, poly([(hx - 13, hy + 30), (hx + 13, hy + 30), (bx + 16, 120 + by), (bx - 16, 120 + by)]))
    comp(_SKIN * 0.88, poly([(bx - 22, 113 + by), (bx, 136 + by), (bx + 22, 113 + by)]))
    hair_tex = _HAIR * (1.0 + 0.18 * np.sin(2 * np.pi * (xx - hx) / 5.0 + 0.6 * (yy - hy) / 5.0))[..., None]
    comp(hair_tex, ell(hx, hy - 6, 36, 40))
    comp(_SKIN * 0.86, ell(hx - 33, hy + 4, 6, 10))
    comp(_SKIN * 0.86, ell(hx + 33, hy + 4, 6, 10))
    face = _SKIN * (0.80 + 0.30 * np.clip(0.5 - (xx - hx) / 62.0, 0, 1) - 0.06 * np.clip((yy - hy) / 40.0, 0, 1))[..., None]
    comp(face, ell(hx, hy + 4, 31, 39))
    comp(hair_tex, ell(hx + 3, hy - 27, 31, 15, -8.0))
    comp(_HAIR, line(hx - 19, hy - 9, hx - 6, hy - 10, 2))
    comp(_HAIR, line(hx + 6, hy - 10, hx + 19, hy - 9, 2))
    eye_ry = 0.6 if q["blink"] else 2.6
    comp((40, 35, 35), ell(hx - 12, hy - 1, 4.5, eye_ry))
    comp((40, 35, 35), ell(hx + 12, hy - 1, 4.5, eye_ry))
    comp(_SKIN * 0.72, line(hx + 2, hy + 2, hx + 4, hy + 13, 2))
    comp((55, 55, 120), ell(hx, hy + 22, 9, 1.4 + 4.5 * q["mouth"]))
    return img


def _cam_noise(seed: int, i: int) -> np.ndarray:
    """Webcam sensor noise for frame i: luma sigma 5 + per-channel sigma 3.3 = sigma 6 per channel."""
    rng = np.random.default_rng([seed, i, 7])
    luma = rng.standard_normal((_CH, _CW), dtype=np.float32) * 5.0
    chroma = rng.standard_normal((_CH, _CW, 3), dtype=np.float32) * 3.3
    return chroma + luma[..., None]


def _paste_cam(frame: np.ndarray, cam: np.ndarray) -> None:
    x, y, w, h = WEBCAM_BOX
    frame[y:y + h, x:x + w] = _CAM_BORDER_COL
    frame[_CY0:_CY0 + _CH, _CX0:_CX0 + _CW] = np.clip(np.rint(cam), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def _slide_frame(t: float) -> np.ndarray:
    """Slide picture at t without webcam or pointer (base timeline). May be a cached read-only image."""
    if t < T_BUILD_1 - _EPS:
        return _image("s1a")
    if t < T_CUT - _EPS:
        return _image("s1b")
    if t <= T_INK[0] + _EPS:
        return _image("s2a")
    if t < T_INK[1] - _EPS:
        img = _image("s2a").copy()
        _draw_ink(img, _ink_progress(t))
        return img
    if t <= T_XFADE[0] + _EPS:
        return _image("s2b")
    if t < T_XFADE[1] - _EPS:
        a = (t - T_XFADE[0]) / (T_XFADE[1] - T_XFADE[0])
        return cv2.addWeighted(_image("s2b"), 1.0 - a, _image("s3a"), a, 0.0)
    if t < T_BUILD_2 - _EPS:
        return _image("s3a")
    return _image("s3b")


def _render_frame(i: int, seed: int, variants: tuple[str, ...]) -> dict[str, np.ndarray]:
    t = i / FPS
    out = {}
    if {"base", "dark", "fade"} & set(variants):
        f = _slide_frame(t).copy()
        _paste_cam(f, _cam_frame(t, calm=False) + _cam_noise(seed, i))
        _draw_pointer(f, _pointer_base(t))
        if "base" in variants:
            out["base"] = f
        if "dark" in variants:
            out["dark"] = cv2.bitwise_not(f)
        if "fade" in variants:
            a = min(1.0, t / FADE_SECONDS)
            out["fade"] = f if a >= 1.0 else cv2.convertScaleAbs(f, alpha=a)
    if "pip" in variants:
        f = _image(_PIP_STATES[0]["image"]).copy()
        _paste_cam(f, _cam_frame(t, calm=True) + _cam_noise(seed, i))
        out["pip"] = f
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
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
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


def _elements_json(key: str) -> list[dict]:
    out = [dict(id=e["id"], kind=e["kind"], text=e["text"], box=list(e["box"])) for e in _IMAGE_ELS[key]]
    if key == "s2b":
        out.append(dict(id="s2_ink", kind="ink", text="", box=_ink_box()))
    return out


def _truth_summary(variant: str, seed: int) -> dict:
    sts = states(variant)
    for s, d in zip(sts, _state_defs(variant)):
        s["elements"] = _elements_json(d["image"])
    segs = [] if variant == "pip" else [
        dict(t_start=s["t0"], t_end=s["t1"], tool=s["tool"], kind=s["kind"], label=s["label"],
             start=list(_pointer_base(s["t0"])[1:]), end=list(_pointer_base(s["t1"] - 1e-6)[1:]))
        for s in _SEGS]
    return {
        "GEN_VERSION": GEN_VERSION,
        "variant": variant,
        "seed": seed,
        "SIZE": list(SIZE),
        "FPS": FPS,
        "DURATION": DURATION,
        "n_frames": N_FRAMES,
        "negated": variant == "dark",
        "fade_in_s": FADE_SECONDS if variant == "fade" else 0.0,
        "webcam_box": webcam_box(variant),
        "webcam": {"noise_sigma": WEBCAM_NOISE_SIGMA,
                   "person": "breathing only" if variant == "pip" else "sways, nods, talks, blinks"},
        "states": sts,
        "transitions": transitions(variant),
        "gestures": gestures(variant),
        "parked": parked(variant),
        "pointer_segments": segs,
        "pointer_track": "__TRACK__",
    }


def _truth_json(variant: str, seed: int) -> str:
    track = [None if p is None else [p[0], int(p[1]), int(p[2])]
             for p in (pointer(i / FPS, variant) for i in range(N_FRAMES))]
    text = json.dumps(_truth_summary(variant, seed), indent=2)
    return text.replace('"__TRACK__"', "{\"fps\": %d, \"frames\": %s}" % (FPS, json.dumps(track, separators=(",", ":"))))


def _paths(out_dir: Path, variant: str) -> tuple[Path, Path]:
    return out_dir / f"slides_{variant}.mp4", out_dir / f"slides_{variant}.json"


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
    """Renders every requested variant in one pass (one slide picture + one webcam per frame)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in variants:
        _check_variant(v)
    todo = tuple(v for v in dict.fromkeys(variants) if force or not _up_to_date(out_dir, v, seed))
    if not todo:
        return []
    _cam_room()                       # build the caches once before the worker threads start
    for key in _IMAGE_ELS:
        _image(key)
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
        tmp.write_text(_truth_json(v, seed), encoding="utf-8")
        os.replace(tmp, meta)
    return list(todo)


def make_slides_lecture(out_dir, variant: str = "base", seed: int = 0, force: bool = False) -> Path:
    """Writes out_dir/slides_<variant>.mp4 and .json (skipped when already up to date)."""
    _check_variant(variant)
    _generate(out_dir, (variant,), seed=seed, force=force)
    return _paths(Path(out_dir), variant)[0]


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate the slide-lecture self-test videos.")
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
    print("states: " + ", ".join(f"{s['id']}:{s['kind'][0]}@{s['t_start']:.1f}" for s in states()))
    print("gestures: " + ", ".join(f"{g['tool']} {g['type']} {g['t_start']:.1f}-{g['t_end']:.1f}" for g in gestures()))


if __name__ == "__main__":
    _main()
