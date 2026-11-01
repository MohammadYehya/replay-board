"""Pointing on a filmed board: when the teacher's hand rests on, taps, circles or sweeps under writing.

Code measures (SAM 3.1 hands + person masks + the board capture); Muse Spark then decides what each
gesture means, as for slides. A sample counts as pointing when:
- the hand is stretched away from the body (gesturing in front of the chest is not pointing),
- its tip is on the board, on or next to writing that is there at that moment,
- and no new ink appears next to it then (that would be writing, not pointing).

Pixel sizes are set for an 800 px board and scale with s = board width / 800 (areas with s * s).
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
    s = W / 800
    near_px, write_px, min_ink = near_px * s, write_px * s, 12 * s * s
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

    def points(t, bx, by) -> bool:
        if not (0 <= bx < W and 0 <= by < H):
            return False
        near = any(t0 <= t < t1 and x - near_px <= bx <= x + w + near_px and y - near_px <= by <= y + h + near_px
                   for t0, t1, (x, y, w, h) in visible)
        if not near:
            return False
        if any(t0 <= t <= t1 and x - write_px <= bx <= x + w + write_px and y - write_px <= by <= y + h + write_px
               for t0, t1, (x, y, w, h) in writing):
            return False  # the hand is on a piece that is being written right now
        # Writing: new ink appears next to the hand soon after (often only once the hand moves away).
        a, b = np.searchsorted(add_t, [t - 1.0, t + write_after])
        if b > a:
            ay, ax = np.divmod(add_idx[a:b], W)
            if (np.hypot(ax - bx, ay - by) <= write_px).sum() >= min_ink:
                return False
        return True

    out = []
    for t, hs in hands.times():
        if not hs:
            continue
        body = _body(person, t, info.width)
        if body is None:
            continue
        cx, cy, height = body
        tips = []
        for h in hs:
            tx, ty = _tip(h, cx, cy)
            tips.append((math.hypot(tx - cx, ty - cy) / max(1.0, height), tx, ty))
        # Every stretched-out hand, most extended first: the other one may be the hand that points.
        for ext, tx, ty in sorted(tips, key=lambda c: -c[0]):
            if ext < min_extension:
                break
            bx, by = video_to_board(geom, [[tx, ty]])[0]
            if points(t, bx, by):
                out.append((float(t), float(bx), float(by), float(ext)))
                break
    return out


# Writing a gesture can be narrowed to one part of (graphs and drawings are always meant whole).
NARROW_KINDS = {"text", "equation", "number", "heading", "table"}


def zoom_region(box, path, board_w: int, board_h: int, reach: float = 80.0) -> list[int]:
    """The part of a big target around the hand's path, [x, y, w, h] in board pixels: what Muse Spark sees zoomed
    in to say which part was meant."""
    s = board_w / 800
    reach = reach * s
    pts = np.asarray(path, np.float64).reshape(-1, 2)
    x0 = int(max(0, box[0], pts[:, 0].min() - reach))
    y0 = int(max(0, box[1], pts[:, 1].min() - reach))
    x1 = int(min(board_w, box[0] + box[2], pts[:, 0].max() + reach + 1))
    y1 = int(min(board_h, box[1] + box[3], pts[:, 1].max() + reach + 1))
    if x1 - x0 < 8 * s or y1 - y0 < 8 * s:  # the path is beside the target, not over it
        return [int(v) for v in box]
    return [x0, y0, x1 - x0, y1 - y0]


def tighten(ink: np.ndarray, box, pad: int = 4, scale: float = 1.0) -> list[int] | None:
    """The ink inside a model's box (grown by pad), as a tight box; None if there is no ink there."""
    H, W = ink.shape
    pad = int(round(pad * scale))
    x0, y0 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad))
    x1, y1 = int(min(W, box[0] + box[2] + pad)), int(min(H, box[1] + box[3] + pad))
    if x1 <= x0 or y1 <= y0:
        return None
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    if ys.size == 0:
        return None
    return [x0 + int(xs.min()), y0 + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def union_box(boxes: list) -> list:
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def clean_target(kind: str, target, ids) -> tuple[str, list[str]]:
    """The model's target ids: only writing on the board then, each once, in its order. An arrow keeps its first
    two (from, to); with fewer it is a spotlight."""
    ids = {str(x) for x in ids}
    out = []
    for x in target or []:
        x = str(x)
        if x in ids and x not in out:
            out.append(x)
    if kind == "arrow":
        return ("arrow", out[:2]) if len(out) >= 2 else ("spotlight", out)
    return kind, out


def box_distance(pt, box) -> float:
    """From a point to a box [x, y, w, h]; 0 inside it."""
    dx = max(box[0] - pt[0], 0.0, pt[0] - (box[0] + box[2]))
    dy = max(box[1] - pt[1], 0.0, pt[1] - (box[1] + box[3]))
    return math.hypot(dx, dy)


def check_arrow(path, target: list, boxes: list, scale: float = 1.0, reach: float = 60.0) -> tuple[str, list]:
    """Muse Spark's arrow from target[0] to target[1] (boxes[0], boxes[1]) against the hand's path: the path must
    start near the first and end near the second. Run the other way (or fitting better that way), the targets swap;
    otherwise it is a spotlight on the target nearest the path's end."""
    a, b, r = path[0], path[-1], reach * scale
    fwd = max(box_distance(a, boxes[0]), box_distance(b, boxes[1]))
    rev = max(box_distance(a, boxes[1]), box_distance(b, boxes[0]))
    if fwd <= r and fwd <= rev:
        return "arrow", [target[0], target[1]]
    if rev <= r:
        return "arrow", [target[1], target[0]]
    i = 0 if box_distance(b, boxes[0]) <= box_distance(b, boxes[1]) else 1
    return "spotlight", [target[i]]


def episodes(samples: list, fps: float, max_gap: float = 0.75, max_jump: float = 90.0, min_len: int = 2,
             scale: float = 1.0) -> list[dict]:
    """Group pointing samples into gestures, with the code's geometric guess as a hint for Muse Spark.
    scale = board width / 800."""
    max_jump = max_jump * scale
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
        if max(w, h) < 25 * scale:
            guess = "hover"
        elif w >= 2.5 * h and w >= 50 * scale and chord >= 0.7 * path:
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
