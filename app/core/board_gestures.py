"""Pointing on a filmed board: when the teacher's hand rests on, taps, circles or sweeps under writing.

Code measures (SAM 3.1 hands + person masks + the board capture); Muse Spark then decides what each
gesture means, as for slides. A sample counts as pointing when:
- the hand is stretched away from the body (gesturing in front of the chest is not pointing),
- its tip is on the board, on or next to writing that is there at that moment,
- and no new ink appears next to it then (that would be writing, not pointing).
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def video_to_board(geom, pts) -> np.ndarray:
    """Original video pixels -> board pixels."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2) * (geom.dw / geom.info.width)
    return cv2.perspectiveTransform(p, np.asarray(geom.M, np.float64)).reshape(-1, 2)


def _body(person, t, video_w: int):
    """Centre and height of the person in video pixels at t, or None."""
    if person is None:
        return None
    m = person(t)
    if m is None or not m.any():
        return None
    ys, xs = np.nonzero(m)
    k = video_w / m.shape[1]
    # The torso, not the arms: the median column, and the vertical middle of the body.
    return float(np.median(xs)) * k, float(ys.min() + 0.4 * (ys.max() - ys.min())) * k, float(ys.max() - ys.min()) * k


def _tip(hand: dict, cx: float, cy: float) -> tuple[float, float]:
    """The point of the hand farthest from the body (a fingertip or the marker's end)."""
    x0, y0 = hand["box"][0], hand["box"][1]
    ys, xs = np.nonzero(hand["local"])
    if ys.size == 0:
        return (hand["box"][0] + hand["box"][2]) / 2, (hand["box"][1] + hand["box"][3]) / 2
    px, py = x0 + xs * hand["scale"], y0 + ys * hand["scale"]
    i = int(np.argmax((px - cx) ** 2 + (py - cy) ** 2))
    return float(px[i]), float(py[i])


def pointing_samples(hands, person, board, geom, info, min_extension: float = 0.3, near_px: float = 22.0,
                     write_px: float = 35.0, write_after: float = 8.0) -> list[tuple[float, float, float, float]]:
    """[(t, x, y, extension)] in board pixels for every sample where the teacher points at writing."""
    W, H = board.width, board.height
    visible = []  # (t0, t1, box) of pieces, for "is there writing here at t"
    writing = []  # (t0, t1, box): while a piece is being written (ink often shows only after the hand moves)
    for p in board.pieces:
        t0 = -math.inf if p.get("pre_existing") else p["t_seen"] - 0.5
        t1 = p["removed_at"] if p.get("removed_at") is not None else math.inf
        visible.append((t0, t1, p["box"]))
        if not p.get("pre_existing"):
            writing.append((min(p["written_after"], p["t_seen"]) - 0.7, p["t_shown"] + 1.0, p["box"]))
    add_t, add_idx = board.add_t, board.add_idx
    order = np.argsort(add_t)
    add_t, add_idx = add_t[order], add_idx[order]
    out = []
    for t, hs in hands.times():
        if not hs:
            continue
        body = _body(person, t, info.width)
        if body is None:
            continue
        cx, cy, height = body
        best = None
        for h in hs:
            tx, ty = _tip(h, cx, cy)
            ext = math.hypot(tx - cx, ty - cy) / max(1.0, height)
            if best is None or ext > best[2]:
                best = (tx, ty, ext)
        tx, ty, ext = best
        if ext < min_extension:
            continue
        bx, by = video_to_board(geom, [[tx, ty]])[0]
        if not (0 <= bx < W and 0 <= by < H):
            continue
        near = any(t0 <= t < t1 and x - near_px <= bx <= x + w + near_px and y - near_px <= by <= y + h + near_px
                   for t0, t1, (x, y, w, h) in visible)
        if not near:
            continue
        if any(t0 <= t <= t1 and x - write_px <= bx <= x + w + write_px and y - write_px <= by <= y + h + write_px
               for t0, t1, (x, y, w, h) in writing):
            continue  # the hand is on a piece that is being written right now
        # Writing: new ink appears next to the hand soon after (often only once the hand moves away).
        a, b = np.searchsorted(add_t, [t - 1.0, t + write_after])
        if b > a:
            ay, ax = np.divmod(add_idx[a:b], W)
            if (np.hypot(ax - bx, ay - by) <= write_px).sum() >= 12:
                continue
        out.append((float(t), float(bx), float(by), float(ext)))
    return out


def episodes(samples: list, fps: float, max_gap: float = 0.75, max_jump: float = 90.0, min_len: int = 2) -> list[dict]:
    """Group pointing samples into gestures, with the code's geometric guess as a hint for Muse Spark."""
    groups, cur = [], []
    for s in samples:
        if cur and (s[0] - cur[-1][0] > max_gap or math.hypot(s[1] - cur[-1][1], s[2] - cur[-1][2]) > max_jump):
            groups.append(cur)
            cur = []
        cur.append(s)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        if len(g) < min_len:
            continue
        pts = np.array([[s[1], s[2]] for s in g])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        w, h = x1 - x0, y1 - y0
        path = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0
        chord = float(np.hypot(*(pts[-1] - pts[0])))
        if max(w, h) < 25:
            guess = "hover"
        elif w >= 2.5 * h and w >= 50 and chord >= 0.7 * path:
            guess = "sweep"
        elif len(pts) >= 4 and chord <= 0.5 * max(w, h) and path >= 2.2 * max(w, h):
            guess = "circle"
        else:
            guess = "move"
        out.append({"type": guess, "tool": "hand", "t_start": round(g[0][0], 2), "t_end": round(g[-1][0] + 1 / fps, 2),
                    "box": [round(float(x0), 1), round(float(y0), 1), round(float(w), 1), round(float(h), 1)],
                    "from": [round(float(pts[0][0]), 1), round(float(pts[0][1]), 1)],
                    "to": [round(float(pts[-1][0]), 1), round(float(pts[-1][1]), 1)],
                    "path": [[round(float(x), 1), round(float(y), 1)] for x, y in pts]})
    return out
