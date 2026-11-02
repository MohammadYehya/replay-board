"""Slides export: a slide deck of what the teacher presented, made from replay.json.

- Pages (code only, deterministic). A board lecture is cut at chapter starts and big erases; each page
  shows the board at the end of its stretch, with what was written or talked about then at full strength
  and the rest of the writing faded, cropped to that writing. A slide lecture gets one page per slide,
  with its build steps.
- Key points: Muse Spark writes each page's title and 2-4 key points, in one call per deck.
- PowerPoint: the deck as a .pptx (python-pptx).

Everything is written to <job>/export/: deck.json, page_NNN.png, points.json and slides.pptx. The deck is
rebuilt when replay.json changes (the pipeline rewrites it when the word times are refined), and the key
points of an older deck are then ignored. All times are on the video's clock, like replay.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import cv2
import numpy as np

from . import ask
from .core.media import imread
from .core.meta_api import MetaAPIError, chat_text, parse_json_object
from .pipeline import plain_error, replace_file

PAGE_RULES_VERSION = "3"  # change it when the page rules change, so every deck is rebuilt
EXPORT = "export"
MIN_PAGE_S = 20.0      # a shorter stretch joins its neighbour
ERASE_AREA = 0.08      # an erase covering this share of the board starts a new page...
ERASE_INK = 0.25       # ...and so does one that removes this share of the writing on the board
ERASE_NEAR = 3.0       # pieces removed this close to an erase (s) were removed by it
ERASE_CHAIN = 4.0      # erases closer than this (s) are one wipe
FADE = 0.3             # strength of the writing that is not the page's focus
CROP_PAD = 24          # board px (of an 800-px-wide board) around the focus
WHOLE_BOARD = 0.65     # a crop at least this share of the board shows the whole board
MIN_CROP = 0.3         # a crop is at least this share of the board's width
ASPECT = 16 / 9
IMAGE_SIDE = 1600
MAX_UPSCALE = 3.0
SAID_CHARS = 2500
TITLE_CHARS = 60
POINT_CHARS = 110
MAX_POINTS = 4
REASONING = "low"
LINE_MAX_H = 100       # px of a 960-wide board: a taller "line" is not one written line (as in the player)
STRAY_AREA = 0.02      # a mark smaller than this share of the board, not talked about in the page, is a stray mark
DRAWINGS = ask.DRAWINGS
VAGUE = {"other", "unknown", "mark", "marks"}  # kinds whose text describes a mark, not its writing (as in the player)
NO_TEXT = {"", "unreadable"}

_num = ask._num
_flat = ask._flat
clock = ask.clock
# Characters XML (so PowerPoint) cannot hold, and lone surrogates (they cannot be written as UTF-8): a reading or a
# model's answer with one of these would make the deck, its JSON or the PowerPoint fail.
_NOT_XML = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")


class NotReady(Exception):
    """The lecture has no replay.json yet, or it is being rewritten (one plain line, HTTP 409)."""


# ---------- small helpers ----------
def safe_text(s: Any) -> str:
    """Text that JSON, UTF-8 and XML (the PowerPoint) can all hold: control characters and lone surrogates removed."""
    return _NOT_XML.sub("", str(s or ""))


def _safe_strings(x: Any) -> Any:
    """Every string in a JSON-like value made safe_text."""
    if isinstance(x, str):
        return safe_text(x)
    if isinstance(x, list):
        return [_safe_strings(v) for v in x]
    if isinstance(x, dict):
        return {k: _safe_strings(v) for k, v in x.items()}
    return x


def _is_box(b: Any) -> bool:
    return isinstance(b, list) and len(b) >= 4 and all(_num(v) for v in b[:4])


def _pieces(replay: dict) -> list[dict]:
    return [p for p in (replay.get("board") or {}).get("pieces") or []
            if isinstance(p, dict) and isinstance(p.get("id"), int) and not isinstance(p.get("id"), bool)
            and _is_box(p.get("box"))]


def _board_size(replay: dict) -> tuple[int, int]:
    size = (replay.get("board") or {}).get("size") or [800, 450]
    try:
        return max(1, int(size[0])), max(1, int(size[1]))
    except (TypeError, ValueError, IndexError):
        return 800, 450


def _box(b: Any) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in b[:4])
    return x, y, max(1.0, w), max(1.0, h)


def draw_start(p: dict, clip0: float = 0.0) -> float:
    """When the board starts drawing a piece: the player's rule (web/src/player/timing.ts, schedule). Writing that
    was there before the recording counts as drawn before the clip (-inf); the rest is never before the clip."""
    if p.get("pre_existing"):
        return -math.inf
    t_seen = float(p["t_seen"]) if _num(p.get("t_seen")) else 0.0
    draw = p.get("draw")
    if p.get("revealed"):
        redraw = float(p["redraw_at"]) if _num(p.get("redraw_at")) else None
        after = float(p["written_after"]) if _num(p.get("written_after")) else -math.inf
        start = t_seen if redraw is None else min(t_seen, max(redraw, after))
    elif (isinstance(draw, list) and len(draw) >= 2 and _num(draw[0]) and _num(draw[1])
          and draw[1] >= draw[0]):
        start = float(draw[0])
    else:
        start = t_seen - 0.7
    return max(start, clip0)


def _removed(p: dict, t: float) -> bool:
    return _num(p.get("removed_at")) and float(p["removed_at"]) <= t


def on_board(pieces: list[dict], t: float, clip0: float) -> list[dict]:
    """The pieces on the board at t: drawing started (a piece being written counts whole) and not erased."""
    return [p for p in pieces if draw_start(p, clip0) <= t and not _removed(p, t)]


def _talked(spans: list[tuple[float, float]], t0: float, t1: float) -> bool:
    """Talked about in [t0, t1): for at least a second of it (half the span, for a shorter one). A span that only
    grazes a page boundary (word times snap a little early) does not count."""
    for a, b in spans:
        if b <= a:
            if t0 <= a < t1:
                return True
        elif min(b, t1) - max(a, t0) >= min(1.0, 0.5 * (b - a)):
            return True
    return False


def _cut(text: str, n: int) -> str:
    """At most n characters, cut at a word with an ellipsis."""
    text = _flat(text)
    if len(text) <= n:
        return text
    cut = text[:n - 1]
    sp = cut.rfind(" ")
    if sp > n * 0.5:
        cut = cut[:sp]
    return cut.rstrip(" ,;:-") + "…"


def said_text(words: list[dict], a: float, b: float, limit: int = SAID_CHARS) -> str:
    """The words spoken from a (inclusive) to b (exclusive), as plain text; a long stretch is cut at a sentence
    (or else a word) with '...'."""
    text = " ".join(str(w["w"]).strip() for w in words if a <= w["s"] < b and str(w["w"]).strip())
    if len(text) <= limit:
        return text
    cut = text[:limit - 4]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end >= 0.6 * limit:
        return cut[:end + 1] + " ..."
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else cut).rstrip(" ,;:") + " ..."


def reading_order(entries: list, box: Callable[[Any], Any]) -> list:
    """Top to bottom, then left to right: things whose heights overlap by half form a row."""
    rows: list[list] = []
    for e in sorted(entries, key=lambda e: (_box(box(e))[1], _box(box(e))[0])):
        x, y, w, h = _box(box(e))
        if rows:
            ax, ay, aw, ah = _box(box(rows[-1][0]))
            if min(y + h, ay + ah) - max(y, ay) >= 0.5 * min(h, ah):
                rows[-1].append(e)
                continue
        rows.append([e])
    return [e for row in rows for e in sorted(row, key=lambda e: _box(box(e))[0])]


def _vague(read: Any) -> bool:
    """A reading that only describes a mark ("person obscuring board, no legible writing"): a vague kind, no LaTeX."""
    read = read if isinstance(read, dict) else {}
    return _flat(read.get("kind")).lower() in VAGUE and not _flat(read.get("latex")).strip("$").strip()


def _mark(read: Any) -> bool:
    """A drawing, or a vague mark: shown as a picture and named, never typeset."""
    read = read if isinstance(read, dict) else {}
    return _flat(read.get("kind")).lower() in DRAWINGS or _vague(read)


def _item(ref: str, read: dict | None, t: float) -> dict | None:
    read = read if isinstance(read, dict) else {}
    kind = _flat(read.get("kind")) or "writing"
    text = _flat(read.get("text"))
    latex = _flat(read.get("latex")).strip("$").strip()
    drawing = _mark(read)
    if not latex and text.lower() in NO_TEXT and kind.lower() not in DRAWINGS:
        return None  # nothing to show or read (a drawing is named by its kind)
    return {"ref": ref, "kind": kind, "text": "" if text.lower() == "unreadable" else text, "latex": latex,
            "t": round(t, 2), "drawing": drawing}


def _seek(spans: list[tuple[float, float]], start: float, t0: float, t1: float, clip0: float,
          here: float | None = None) -> float:
    """Where the player should seek for an item of the page [t0, t1): when it was first talked about, but not
    before it starts being drawn; else when it is drawn. What happens in this page comes first: an item drawn
    here and talked about only in a later page seeks to its drawing. `here`: when the part of it drawn in this
    page starts (a line whose first piece was on the board before the page), if not `start`."""
    if here is None and t0 <= start < t1:
        here = start
    inside = [s for s in spans if _talked([s], t0, t1)]
    if inside:
        t = max(inside[0][0], start)
    elif here is not None:
        t = here
    elif spans:
        t = max(spans[0][0], start)
    else:
        t = start
    return max(clip0, t) if math.isfinite(t) else clip0


# ---------- piece images ----------
class Pixels:
    """Piece images, read once: their ink and, for partly erased pieces, when each pixel leaves the board."""

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self._cache: dict[Any, tuple[np.ndarray | None, np.ndarray | None]] = {}

    def get(self, p: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        key = (p.get("image"), p.get("gone"))
        if key not in self._cache:
            bgra = ask._bgra(self.job_dir, p.get("image"))
            gone = None
            if bgra is not None and isinstance(p.get("gone"), str) and p["gone"]:
                try:
                    g = imread(self.job_dir / p["gone"], cv2.IMREAD_COLOR).astype(np.int64)
                    if g.shape[:2] == bgra.shape[:2]:  # 24-bit code B | G<<8 | R<<16; time = (code - 1) / 10
                        code = g[..., 0] | (g[..., 1] << 8) | (g[..., 2] << 16)
                        gone = np.where(code > 0, (code - 1) / 10.0, np.inf).astype(np.float32)
                except Exception:
                    gone = None
            self._cache[key] = (bgra, gone)
        return self._cache[key]

    def ink(self, p: dict, t: float = math.inf) -> float:
        """Pixels of the piece still on the board at t (an estimate from its box without an image)."""
        bgra, gone = self.get(p)
        if bgra is None:
            x, y, w, h = _box(p["box"])
            return 0.15 * w * h
        a = bgra[..., 3] > 0
        if gone is not None and math.isfinite(t):
            a &= gone > t
        return float(np.count_nonzero(a))

    def ink_gone(self, p: dict, a: float, b: float) -> float:
        """Pixels of a partly erased piece that leave the board in (a, b]."""
        bgra, gone = self.get(p)
        if bgra is None or gone is None:
            return 0.0
        return float(np.count_nonzero((bgra[..., 3] > 0) & (gone > a) & (gone <= b)))


# ---------- board pages ----------
def big_erases(replay: dict, pix: Pixels) -> list[tuple[float, float]]:
    """(time, the board is shown just before this) of each erase that starts a new page: erases closer than
    ERASE_CHAIN s are one wipe, which is big when its boxes cover ERASE_AREA of the board or the pieces it
    removes hold ERASE_INK of the writing on the board just before."""
    _, clip0, clip1 = ask._video(replay)
    W, H = _board_size(replay)
    pieces = _pieces(replay)
    erases = sorted((e for e in (replay.get("board") or {}).get("erases") or []
                     if isinstance(e, dict) and _num(e.get("t")) and clip0 < e["t"] < clip1),
                    key=lambda e: e["t"])
    wipes: list[list[dict]] = []
    for e in erases:
        if wipes and e["t"] - wipes[-1][-1]["t"] <= ERASE_CHAIN:
            wipes[-1].append(e)
        else:
            wipes.append([e])
    out = []
    for wipe in wipes:
        ta, tb = float(wipe[0]["t"]), float(wipe[-1]["t"])
        removed = [p for p in pieces if _num(p.get("removed_at")) and ta - ERASE_NEAR <= p["removed_at"] <= tb + ERASE_NEAR]
        before = min([ta] + [float(p["removed_at"]) for p in removed]) - 0.05
        there = on_board(pieces, before, clip0)
        total = sum(pix.ink(p, before) for p in there)
        gone_ids = {p["id"] for p in removed}
        gone = sum(pix.ink(p, before) for p in there if p["id"] in gone_ids)
        gone += sum(pix.ink_gone(p, before, tb + ERASE_NEAR) for p in there if p["id"] not in gone_ids)
        area = 0.0
        for e in wipe:
            b = e.get("box")
            if isinstance(b, list) and len(b) >= 4 and all(_num(v) for v in b[:4]):
                area += max(0.0, float(b[2])) * max(0.0, float(b[3]))
        if area >= ERASE_AREA * W * H or (total > 0 and gone >= ERASE_INK * total):
            out.append((ta, max(clip0, before)))
    return out


def focus_pieces(pieces: list[dict], t0: float, t1: float, clip0: float) -> list[dict]:
    """The page's focus: pieces the board starts drawing in [t0, t1), and pieces on the board then that were
    talked about then."""
    out = []
    for p in pieces:
        s = draw_start(p, clip0)
        if t0 <= s < t1:
            out.append(p)
        elif s < t1 and not _removed(p, t0) and _talked(ask._merge(p.get("spoken") or []), t0, t1):
            out.append(p)
    return out


def board_segments(replay: dict, pix: Pixels) -> list[dict]:
    """The board lecture cut into pages: [{t0, t1, before}], `before` = the board is shown just before this
    (an erase ends the page), else None. Cuts at chapter starts and big erases; a page shorter than MIN_PAGE_S,
    or with no focus, joins the page before it (the first joins the next, and so does one that starts with a big
    erase, so that the page before still shows the board before the wipe). The last stretch has no next page:
    when it starts with a big erase (the board wiped at the end of the lecture), the page it joins still shows
    the board just before the wipe, not the clean board after it."""
    _, clip0, clip1 = ask._video(replay)
    pieces = _pieces(replay)
    cuts: list[tuple[float, float | None]] = []
    for c in replay.get("chapters") or []:
        if isinstance(c, dict) and _num(c.get("t")) and clip0 + 0.5 < c["t"] < clip1 - 0.5:
            cuts.append((float(c["t"]), None))
    for t, before in big_erases(replay, pix):
        if clip0 + 0.5 < t < clip1 - 0.5:
            cuts.append((t, before))
    cuts.sort(key=lambda c: c[0])
    kept: list[tuple[float, float | None]] = []
    for t, before in cuts:
        if kept and t - kept[-1][0] < 0.5:  # a chapter starting with an erase: one cut, at the erase
            if before is not None:
                kept[-1] = (t, before)
            continue
        kept.append((t, before))
    bounds = [clip0] + [t for t, _ in kept] + [clip1]
    segs = [{"t0": bounds[i], "t1": bounds[i + 1], "before": kept[i][1] if i < len(kept) else None}
            for i in range(len(bounds) - 1)]

    def bad(i: int) -> bool:
        s = segs[i]
        end = math.inf if i == len(segs) - 1 else s["t1"]
        return s["t1"] - s["t0"] < MIN_PAGE_S or not focus_pieces(pieces, s["t0"], end, clip0)

    while len(segs) > 1:
        i = next((k for k in range(len(segs)) if bad(k)), None)
        if i is None:
            break
        if i == 0:
            j = 1
        elif segs[i - 1]["before"] is not None and i + 1 < len(segs):
            j = i + 1  # it starts with a big erase: the page before keeps the board as it was before the wipe
        else:
            j = i - 1
        a, b = sorted((i, j))
        before = segs[a]["before"] if j == i - 1 and segs[a]["before"] is not None else segs[b]["before"]
        segs[a:b + 1] = [{"t0": segs[a]["t0"], "t1": segs[b]["t1"], "before": before}]
    return segs


def crop_box(boxes: list, W: int, H: int) -> list[int] | None:
    """[x, y, w, h] around these boxes, padded, at least MIN_CROP of the board's width, and widened toward 16:9
    inside the board; None for the whole board (no boxes, or a crop of at least WHOLE_BOARD of it)."""
    if not boxes:
        return None
    pad = CROP_PAD * W / 800
    bs = [_box(b) for b in boxes]
    x0 = max(0.0, min(b[0] for b in bs) - pad)
    y0 = max(0.0, min(b[1] for b in bs) - pad)
    x1 = min(float(W), max(b[0] + b[2] for b in bs) + pad)
    y1 = min(float(H), max(b[1] + b[3] for b in bs) + pad)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    if w < MIN_CROP * W:  # one short formula: show some of the board around it, not a few blurry pixels
        nw = MIN_CROP * W
        x0 = min(max(0.0, x0 - (nw - w) / 2), W - nw)
        x1, w = x0 + nw, nw
    if w / h < ASPECT:
        nw = min(float(W), h * ASPECT)
        x0 = min(max(0.0, x0 - (nw - w) / 2), W - nw)
        x1 = x0 + nw
    else:
        nh = min(float(H), w / ASPECT)
        y0 = min(max(0.0, y0 - (nh - h) / 2), H - nh)
        y1 = y0 + nh
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
    if (x1 - x0) * (y1 - y0) >= WHOLE_BOARD * W * H:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def fit_side(img: np.ndarray, side: int = IMAGE_SIDE, max_up: float = MAX_UPSCALE) -> np.ndarray:
    """Longest side `side`: smaller images grown (at most max_up times), bigger ones shrunk."""
    h, w = img.shape[:2]
    k = min(side / max(h, w), max_up)
    if abs(k - 1.0) < 1e-3:
        return img
    size = (max(1, round(w * k)), max(1, round(h * k)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)


def render_board(replay: dict, pix: Pixels, shown: list[dict], focus_ids: set, t: float,
                 crop: list[int] | None = None) -> np.ndarray:
    """The board at t (these pieces): the focus at full strength, the rest faded toward the board colour,
    cut to `crop` (None: the whole board). BGR, longest side about IMAGE_SIDE."""
    W, H = _board_size(replay)
    canvas = np.empty((H, W, 3), np.float32)
    canvas[:] = ask._bg(replay)
    for p in sorted(shown, key=lambda p: p["id"] in focus_ids):  # the focus is drawn last, on top
        bgra, gone = pix.get(p)
        if bgra is None:
            continue
        a = bgra[..., 3:4].astype(np.float32) / 255.0
        if gone is not None:
            a = a * (gone > t)[..., None]
        if p["id"] not in focus_ids:
            a = a * FADE
        x, y = int(round(float(p["box"][0]))), int(round(float(p["box"][1])))
        h, w = bgra.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        src, al = bgra[y0 - y:y1 - y, x0 - x:x1 - x, :3].astype(np.float32), a[y0 - y:y1 - y, x0 - x:x1 - x]
        canvas[y0:y1, x0:x1] = src * al + canvas[y0:y1, x0:x1] * (1 - al)
    img = np.clip(canvas + 0.5, 0, 255).astype(np.uint8)
    if crop is not None:
        x, y, w, h = crop
        img = img[y:y + h, x:x + w]
    return fit_side(img)


def _written_line(ln: dict, W: int) -> bool:
    b = ln.get("box")
    if not _is_box(b):
        return False
    return b[3] <= LINE_MAX_H * max(0.25, W / 960)


def board_items(replay: dict, focus: list[dict], t0: float, t1: float, clip0: float) -> list[dict]:
    """The focus in reading order, a piece of a written line replaced by its line (once)."""
    W, _ = _board_size(replay)
    pieces = {p["id"]: p for p in _pieces(replay)}
    line_of: dict[int, dict] = {}
    for ln in (replay.get("board") or {}).get("lines") or []:
        # A line with no reading of its own would drop its pieces: they are then listed one by one.
        if (isinstance(ln, dict) and isinstance(ln.get("id"), str) and isinstance(ln.get("pieces"), list)
                and _written_line(ln, W) and _item("", ln.get("read"), 0.0) is not None):
            for i in ln["pieces"]:
                if isinstance(i, int):
                    line_of.setdefault(i, ln)
    entries, seen = [], set()
    for p in focus:
        ln = line_of.get(p["id"])
        if ln is not None:
            if ln["id"] in seen:
                continue
            seen.add(ln["id"])
            members = [pieces[i] for i in ln["pieces"] if isinstance(i, int) and i in pieces] or [p]
            starts = [draw_start(m, clip0) for m in members]
            here = [s for s in starts if t0 <= s < t1]  # the line's part drawn in this page
            spans = ask._merge([s for m in members for s in m.get("spoken") or []])
            entries.append((ln["box"], "#" + ln["id"], ln.get("read"), min(starts), spans,
                            min(here) if here else None))
        else:
            entries.append((p["box"], f"#{p['id']}", p.get("read"), draw_start(p, clip0),
                            ask._merge(p.get("spoken") or []), None))
    out = []
    for box, ref, read, start, spans, here in reading_order(entries, lambda e: e[0]):
        it = _item(ref, read, _seek(spans, start, t0, t1, clip0, here))
        if it is not None:
            out.append(it)
    return out


def _inside(box: Any, crop: list[int] | None) -> bool:
    """The box's centre is in the crop (None: the whole board)."""
    if crop is None:
        return True
    x, y, w, h = _box(box)
    cx, cy = x + w / 2, y + h / 2
    return crop[0] <= cx <= crop[0] + crop[2] and crop[1] <= cy <= crop[1] + crop[3]


def page_view(focus: list[dict], shown: list[dict], t0: float, t1: float, W: int,
              H: int) -> tuple[list[int] | None, list[dict]]:
    """(crop, the focus listed as the page's items). Marks not talked about in the page do not choose the crop:
    a vague reading ("person obscuring board, no legible writing") is never listed, and a small stray drawing (an
    arrow seen as the camera settles) is listed only when it is on the page's picture. So a stray mark far from the
    page's writing does not turn the slide into the whole board."""
    ids = {p["id"] for p in shown}
    quiet = [p for p in focus if _mark(p.get("read")) and not _talked(ask._merge(p.get("spoken") or []), t0, t1)]
    vague = {p["id"] for p in quiet if _vague(p.get("read"))}
    stray = {p["id"] for p in quiet if p["id"] not in vague and _box(p["box"])[2] * _box(p["box"])[3] < STRAY_AREA * W * H}
    on_page = [p for p in focus if p["id"] in ids]
    key = [p for p in on_page if p["id"] not in vague and p["id"] not in stray]
    crop = crop_box([p["box"] for p in key or [p for p in on_page if p["id"] not in vague]], W, H)
    listed = [p for p in focus if p["id"] not in vague and (p["id"] not in stray or (p["id"] in ids and _inside(p["box"], crop)))]
    return crop, listed


def board_pages(job_dir: Path, replay: dict, write: Callable[[int, np.ndarray], tuple[str, list[int]]]) -> list[dict]:
    _, clip0, _ = ask._video(replay)
    W, H = _board_size(replay)
    pix = Pixels(job_dir)
    pieces = _pieces(replay)
    segs = board_segments(replay, pix)
    pages = []
    for n, s in enumerate(segs, 1):
        last = n == len(segs)
        t0, t1 = s["t0"], s["t1"]
        end = math.inf if last else t1
        focus = focus_pieces(pieces, t0, end, clip0)
        if s["before"] is not None:
            t_end = max(t0, s["before"])
        else:
            t_end = t1 if last else max(t0, t1 - 0.01)
        shown = on_board(pieces, t_end, clip0)
        crop, listed = page_view(focus, shown, t0, end, W, H)
        strong = {p["id"] for p in focus if p in listed or not _vague(p.get("read"))}  # a vague mark is faded
        img = render_board(replay, pix, shown, strong, t_end, crop)
        image, size = write(n, img)
        pages.append({"t0": t0, "t1": t1, "image": image, "image_size": size,
                      "items": board_items(replay, listed, t0, end, clip0), "fallback": f"Part {n}"})
    return pages


# ---------- slide pages ----------
def slide_groups(replay: dict) -> list[list[dict]]:
    """One group per page: a state that is not a build step ("addition") and not blank starts one, and the build
    steps after it join it. Blank states are left out."""
    states = sorted((st for st in (replay.get("slides") or {}).get("states") or []
                     if isinstance(st, dict) and _num(st.get("t_start"))), key=lambda st: st["t_start"])
    groups: list[list[dict]] = []
    for st in states:
        if st.get("blank"):
            continue
        if st.get("kind") == "addition" and groups:
            groups[-1].append(st)
        else:
            groups.append([st])
    return groups


def _el_key(e: dict) -> tuple:
    return (_flat(e.get("kind")).lower(), _flat(e.get("text")), _flat(e.get("latex")))


def slide_items(group: list[dict], clip0: float) -> list[dict]:
    """The last state's elements, in their order. An element shown in an earlier build step of the page keeps
    that step's time and what was said about it there."""
    first: dict[tuple, float] = {}
    spoken: dict[tuple, list] = {}
    for st in group:
        for e in st.get("elements") or []:
            if isinstance(e, dict):
                k = _el_key(e)
                first.setdefault(k, float(st["t_start"]))
                spoken.setdefault(k, []).extend(e.get("spoken") or [])
    t0 = float(group[0]["t_start"])
    t1 = float(group[-1].get("t_end", group[-1]["t_start"])) if _num(group[-1].get("t_end")) else math.inf
    out = []
    for e in group[-1].get("elements") or []:
        if not isinstance(e, dict) or not isinstance(e.get("id"), str):
            continue
        k = _el_key(e)
        it = _item("#" + e["id"], e, _seek(ask._merge(spoken.get(k, [])), first.get(k, t0), t0, t1, clip0))
        if it is not None:
            out.append(it)
    return out


def _slide_title(group: list[dict]) -> str:
    for e in group[-1].get("elements") or []:
        if isinstance(e, dict) and _flat(e.get("kind")).lower() in ("title", "heading") and _flat(e.get("text")):
            return _cut(e["text"], TITLE_CHARS)
    return ""


def slide_pages(job_dir: Path, replay: dict, write: Callable[[int, np.ndarray], tuple[str, list[int]]]) -> list[dict]:
    _, clip0, _ = ask._video(replay)
    pages = []
    for n, group in enumerate(slide_groups(replay), 1):
        img = None
        for st in reversed(group):  # the last state's image (an earlier one if it has none)
            img = ask._slide_full(job_dir, st)
            if img is not None:
                break
        # No image at all: no picture (the player says so), rather than a blank white slide that looks real.
        image, isize = write(n, fit_side(img, IMAGE_SIDE, 1.0)) if img is not None else ("", None)
        last = group[-1]
        t1 = float(last["t_end"]) if _num(last.get("t_end")) else float(last["t_start"])
        pages.append({"t0": float(group[0]["t_start"]), "t1": t1, "image": image, "image_size": isize,
                      "items": slide_items(group, clip0), "fallback": _slide_title(group) or f"Slide {n}"})
    return pages


# ---------- the deck ----------
def _export_dir(job_dir: Path) -> Path:
    return job_dir / EXPORT


def _save_bytes(path: Path, data: bytes, tries: int = 40) -> None:
    """Written whole or not at all: the player may be reading the old file (Windows then refuses the replace for
    a moment: `tries` attempts, about 6 s for 40)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(data)
    if not replace_file(tmp, path, tries):
        tmp.unlink(missing_ok=True)
        raise OSError(f"could not save {path.name}: the file is in use")


def _save_json(path: Path, data: dict) -> None:
    _save_bytes(path, json.dumps(_safe_strings(data), ensure_ascii=False, indent=1).encode("utf-8"))


def _load_json(path: Path) -> dict:
    for _ in range(20):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)  # Windows can refuse a read while the file is being replaced
    return {}


def read_replay(job_dir: Path) -> tuple[bytes, dict]:
    """replay.json as (bytes, data). NotReady while there is none, or while it is being rewritten."""
    path = job_dir / "replay.json"
    for _ in range(20):
        try:
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict):
                return raw, data
            break
        except FileNotFoundError:
            raise NotReady("this lecture is not processed yet, so there are no slides yet") from None
        except (OSError, ValueError):
            time.sleep(0.05)
    raise NotReady("the lecture is being updated; try again in a moment")


def deck_version(raw: bytes) -> str:
    return hashlib.sha256(raw + b"|export-rules-" + PAGE_RULES_VERSION.encode()).hexdigest()[:16]


def lecture_title(job_dir: Path, replay: dict) -> str:
    """The subject, else the uploaded file's name, else the job's id."""
    subject = _flat(replay.get("subject"))
    if subject:
        return subject[:1].upper() + subject[1:]
    status = _load_json(job_dir / "status.json")
    video = replay.get("video") or {}
    for name in (status.get("name"), video.get("source"), video.get("file")):
        if isinstance(name, str) and name.strip() and Path(name).stem not in ("upload", "lecture"):
            return Path(name.strip()).stem
    return job_dir.name


def _chapter_at(chapters: list[dict], t0: float, t1: float) -> str:
    cur = None
    for c in chapters:
        if c["t"] <= t0 + 0.5:
            cur = c
    if cur is None:  # the page starts before the first chapter: the one that starts in it
        cur = next((c for c in chapters if c["t"] < t1), None)
    return _flat(cur["title"]) if cur else ""


def build_deck(job_dir: Path, replay: dict, version: str) -> dict:
    """Make the pages and their images (export/page_NNN.png). Code only, no model calls."""
    out = _export_dir(job_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(n: int, img: np.ndarray) -> tuple[str, list[int]]:
        name = f"page_{n:03d}.png"
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise OSError(f"could not encode {name}")
        _save_bytes(out / name, buf.tobytes())
        return f"{EXPORT}/{name}", [int(img.shape[1]), int(img.shape[0])]

    kind = "slides" if replay.get("type") == "slides" else "board"
    raw = slide_pages(job_dir, replay, write) if kind == "slides" else board_pages(job_dir, replay, write)
    chapters = sorted((c for c in replay.get("chapters") or []
                       if isinstance(c, dict) and _num(c.get("t")) and _flat(c.get("title"))), key=lambda c: c["t"])
    words = sorted(ask._words(replay), key=lambda w: w["s"])
    pages, used = [], {}
    for i, pg in enumerate(raw):
        chapter = _chapter_at(chapters, pg["t0"], pg["t1"])
        title = chapter or pg["fallback"]
        used[title] = used.get(title, 0) + 1
        if used[title] > 1:
            title = f"{title} ({used[title]})"
        a = -math.inf if i == 0 else pg["t0"]  # the words before the first page and after the last are theirs
        b = raw[i + 1]["t0"] if i + 1 < len(raw) else math.inf
        pages.append({"id": f"p{i + 1}", "t0": round(pg["t0"], 2), "t1": round(pg["t1"], 2), "chapter": chapter,
                      "title": title, "image": pg["image"], "image_size": pg["image_size"], "items": pg["items"],
                      "said": said_text(words, a, b)})
    for old in out.glob("page_*.png"):  # pages of an older, longer deck
        m = re.fullmatch(r"page_(\d+)\.png", old.name)
        if m and int(m.group(1)) > len(pages):
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass  # being sent to the player (Windows refuses): nothing refers to it any more
    _, clip0, clip1 = ask._video(replay)
    return _safe_strings({"version": version, "rules": PAGE_RULES_VERSION, "type": kind,
                          "title": lecture_title(job_dir, replay), "clip": [round(clip0, 2), round(clip1, 2)],
                          "pages": pages})


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(job_dir: Path, what: str) -> threading.Lock:
    key = f"{job_dir.resolve()}|{what}"
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


_PAGE_IMAGE = re.compile(rf"{EXPORT}/page_\d{{3,}}\.png")


def _deck_ok(job_dir: Path, deck: dict, version: str) -> bool:
    """The saved deck is this version's, and every page's image is one of ours and is there (or it has none)."""
    if deck.get("version") != version or not isinstance(deck.get("pages"), list):
        return False
    for p in deck["pages"]:
        image = p.get("image") if isinstance(p, dict) else None
        if image == "":
            continue
        if not (isinstance(image, str) and _PAGE_IMAGE.fullmatch(image) and (job_dir / image).is_file()):
            return False
    return True


def load_deck(job_dir: Path) -> tuple[dict, dict]:
    """(deck, replay): the cached deck, rebuilt (with its images) when replay.json changed."""
    raw, replay = read_replay(job_dir)
    version = deck_version(raw)
    with _lock(job_dir, "deck"):
        path = _export_dir(job_dir) / "deck.json"
        deck = _load_json(path)
        if _deck_ok(job_dir, deck, version):
            return deck, replay
        deck = build_deck(job_dir, replay, version)
        _save_json(path, deck)
    return deck, replay


# ---------- key points (Muse Spark) ----------
POINTS_SCHEMA = {
    "type": "object",
    "properties": {"pages": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "title": {"type": "string"},
        "points": {"type": "array", "items": {"type": "string"}}}}}},
}

POINTS_TASK = """This request is not a student's question. Replay Board is making a slide deck of this lecture: \
one page for each part listed below, showing the board (or the slide) at the end of that part, with what was \
written or talked about then. Write each page's title and key points. Reply only with JSON in the given format; \
for this reply, ignore the answer style above (no LaTeX, no markdown, no [m:ss] times or [#id] citations).
- title: at most 60 characters, specific to what this page shows and says (for example "Compounding monthly \
gives 2.61", not "Compound interest").
- points: 2 to 4 key points a student should take away from this part, each at most 110 characters, in plain \
text. Write maths with Unicode symbols (x², √2, π, ∞, →, ≈, ×, ≠), never LaTeX.
- Stay faithful to the lecture: say only what it says or shows in that part, and do not add facts of your own.
- Write symbols and names as the board or slide writes them: the transcript is automatic and can mishear or \
capitalise them (the number e is "e", not "E").
- Give every page below, with its id."""


def _points_max_tokens(n_pages: int) -> int:
    return min(16000, 2500 + 350 * max(1, n_pages))


def points_messages(job_dir: Path, replay: dict, deck: dict) -> tuple[list[dict], str]:
    """(messages, prompt cache key). The first message is Ask Muse's, word for word, with the same cache key, so
    the lecture is cached once for both; the pages follow."""
    system = safe_text(ask.SYSTEM + "\n\n" + ask.lecture_context(replay))  # unchanged unless a reading is odd
    key = f"replay-board-{job_dir.name}-{hashlib.sha256(system.encode('utf-8')).hexdigest()[:16]}"
    lines = [POINTS_TASK, "", "# Pages"]
    for pg in deck["pages"]:
        head = f"Page {pg['id']} | {clock(pg['t0'])}-{clock(pg['t1'])}"
        if pg.get("chapter"):
            head += f' | chapter "{pg["chapter"]}"'
        lines.append(head)
        shows = [f"{it['ref']} " + ask._what(it) + (" (a drawing)" if it.get("drawing") else "")
                 for it in pg["items"]]
        lines.append("Shows: " + ("; ".join(shows) if shows else "nothing new"))
        lines.append(f'Said: "{pg["said"]}"' if pg.get("said") else "Said: nothing")
        lines.append("")
    return [{"role": "system", "content": system},
            {"role": "user", "content": safe_text("\n".join(lines).rstrip())}], key


_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "theta": "θ",
          "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
          "omega": "ω", "Delta": "Δ", "Gamma": "Γ", "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ",
          "Omega": "Ω", "Pi": "Π"}
_SYMBOLS = {"infty": "∞", "to": "→", "rightarrow": "→", "Rightarrow": "⇒", "leftarrow": "←", "approx": "≈",
            "times": "×", "cdot": "·", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "ne": "≠",
            "pm": "±", "circ": "°", "degree": "°", "ldots": "…", "dots": "…", "cdots": "⋯", "sim": "~",
            "in": "∈", "sum": "Σ", "int": "∫", "partial": "∂", "nabla": "∇", "propto": "∝", "div": "÷"}
_DROP = re.compile(r"\\(left|right|bigl|bigr|Bigl|Bigr|big|Big|displaystyle|textstyle|mathrm|mathbf|mathit|"
                   r"text|operatorname)(?![A-Za-z])")
_SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


def _group(s: str) -> str:
    s = s.strip()
    return s if re.fullmatch(r"[\w.]+", s) else f"({s})"


def _unmath(s: str) -> str:
    """$...$ (or $$...$$) around LaTeX -> its content. Any other $ stays: it is money ('$5 or $1')."""
    out, i = [], 0
    while True:
        a = s.find("$", i)
        if a < 0:
            out.append(s[i:])
            return "".join(out)
        if s.startswith("$$", a) and s.find("$$", a + 2) > a:
            b = s.find("$$", a + 2)
            out += [s[i:a], s[a + 2:b]]
            i = b + 2
            continue
        b = s.find("$", a + 1)
        if b > a and re.search(r"[\\^_{}]", s[a + 1:b]):
            out += [s[i:a], s[a + 1:b]]
            i = b + 1
        else:
            out.append(s[i:a + 1])
            i = a + 1


def plain_text(s: str) -> str:
    """A model's line as plain text: no citations, markdown or LaTeX (turned into Unicode where it is simple)."""
    s = re.sub(r"\[#[A-Za-z0-9]+\]", "", safe_text(s))
    s = re.sub(r"\[\d{1,3}:[0-5]\d(?::[0-5]\d)?\]", "", s)
    s = _unmath(s)
    if "\\" in s:
        s = _DROP.sub("", s)
        s = re.sub(r"\\[,;:! ]", " ", s)
        s = re.sub(r"\^\{?\\circ\}?", "°", s)
        for _ in range(3):  # nested fractions and roots, inside out
            s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", lambda m: f"{_group(m[1])}/{_group(m[2])}", s)
            s = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: "√" + _group(m[1]), s)
        s = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m[1]) or _SYMBOLS.get(m[1]) or m[1], s)
        s = s.replace("\\", "")
    s = re.sub(r"\^\{([0-9+\-=()n]+)\}", lambda m: m[1].translate(_SUP), s)
    s = re.sub(r"\^([0-9]+|n)(?![\w{(])", lambda m: m[1].translate(_SUP), s)
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s).replace("{", "").replace("}", "")
    s = re.sub(r"\*\*|__|`", "", s)
    s = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s+", "", s)
    return _flat(s)


def parse_points(data: Any, ids: list[str]) -> dict[str, dict]:
    """The model's {pages: [{id, title, points}]} -> {id: {title, points}}, cleaned. Unknown ids are left out."""
    out: dict[str, dict] = {}
    known = set(ids)
    pages = data.get("pages") if isinstance(data, dict) else None
    for p in pages if isinstance(pages, list) else []:
        if not isinstance(p, dict):
            continue
        pid = _flat(p.get("id")).lower()
        if pid not in known and f"p{pid}" in known:
            pid = f"p{pid}"
        if pid not in known or pid in out:
            continue
        title = _cut(plain_text(p.get("title") or ""), TITLE_CHARS).rstrip(".")
        pts = [_cut(plain_text(x), POINT_CHARS) for x in p.get("points") or [] if isinstance(x, str)]
        out[pid] = {"title": title, "points": [x for x in pts if x][:MAX_POINTS]}
    return out


def load_points(job_dir: Path, version: str) -> dict[str, dict] | None:
    """The saved key points of this deck version, or None (none yet, or made for an older deck)."""
    data = _load_json(_export_dir(job_dir) / "points.json")
    if data.get("version") != version or not isinstance(data.get("pages"), dict):
        return None
    return {k: v for k, v in data["pages"].items() if isinstance(v, dict)}


def make_points(client: Any, job_dir: Path, replay: dict, deck: dict) -> dict[str, dict]:
    """One Muse Spark call for the whole deck; saved to export/points.json under the deck's version."""
    messages, key = points_messages(job_dir, replay, deck)
    calls = getattr(client, "calls", None)
    n0 = len(calls) if isinstance(calls, list) else 0
    try:
        reply = client.chat(messages, schema=POINTS_SCHEMA, schema_name="slide_points", reasoning=REASONING,
                            max_tokens=_points_max_tokens(len(deck["pages"])), extra={"prompt_cache_key": key})
    finally:
        close = getattr(getattr(client, "http", None), "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
    points = parse_points(parse_json_object(chat_text(reply)), [p["id"] for p in deck["pages"]])
    if not points:
        raise MetaAPIError("the answer had no key points for these slides; try again")
    recs = (getattr(client, "calls", None) or [])[n0:]
    costs = [float(r.cost_usd) for r in recs if getattr(r, "cost_usd", None) is not None]
    _save_json(_export_dir(job_dir) / "points.json",
               {"version": deck["version"], "pages": points, "cost_usd": round(sum(costs), 6) if costs else None,
                "cached": bool(recs) and all(getattr(r, "cached", False) for r in recs), "at": round(time.time(), 3)})
    return points


def with_points(deck: dict, points: dict[str, dict] | None) -> dict:
    """The deck as the player gets it (GET slides.json): Muse Spark's titles and key points where they exist."""
    pages = []
    for pg in deck["pages"]:
        pt = (points or {}).get(pg["id"])
        title = pt.get("title") if pt and pt.get("title") else pg["title"]
        pages.append({"id": pg["id"], "t0": pg["t0"], "t1": pg["t1"], "chapter": pg["chapter"], "title": title,
                      "image": pg["image"], "image_size": pg["image_size"], "items": pg["items"],
                      "said": pg["said"], "points": list(pt.get("points") or []) if pt else None})
    return {"type": deck["type"], "title": deck["title"], "version": deck["version"],
            "points_ready": points is not None, "pages": pages}


def slides_data(job_dir: Path, make_client: Callable[[], Any] | None = None) -> dict:
    """The deck with its key points. With `make_client`, the key points are written first if this deck has none
    (one model call; a second request waits for the first instead of paying again)."""
    deck, replay = load_deck(job_dir)
    points = load_points(job_dir, deck["version"])
    if points is None and make_client is not None and not deck["pages"]:
        points = {}  # no pages (a slide lecture with only blank slides): nothing to write, so no model call
    if points is None and make_client is not None:
        with _lock(job_dir, "points"):
            points = load_points(job_dir, deck["version"])
            if points is None:
                points = make_points(make_client(), job_dir, replay, deck)
    return with_points(deck, points)


def plain(e: BaseException) -> str:
    """One line for the player."""
    if isinstance(e, MetaAPIError):
        text = str(e).splitlines()[0] if str(e) else "unknown error"
        return f"Muse Spark could not write the key points: {text}"
    if isinstance(e, NotReady):
        return str(e)
    if isinstance(e, OSError) and (e.errno is not None or e.filename):
        return plain_error(e)
    text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return f"could not make the slides ({type(e).__name__}: {text[:160]}); details are in export-error.log"


# ---------- PowerPoint ----------
def file_name(title: str) -> tuple[str, str]:
    """(an ASCII file name, the same in Unicode) for the deck: '<title> - slides.pptx'."""
    uni = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', " ", title)
    uni = " ".join(uni.split()).strip(" .-")[:80].strip(" .-") or "lecture"
    asc = unicodedata.normalize("NFKD", uni).encode("ascii", "ignore").decode("ascii")
    asc = " ".join(re.sub(r"[^A-Za-z0-9 _.-]+", " ", asc).split()).strip(" .-") or "lecture"
    return f"{asc} - slides.pptx", f"{uni} - slides.pptx"


def content_disposition(title: str) -> str:
    asc, uni = file_name(title)
    return f"attachment; filename=\"{asc}\"; filename*=UTF-8''{quote(uni)}"


def _span(t0: float, t1: float) -> str:
    return f"{clock(t0)}–{clock(t1)}"


EMU_PER_INCH = 914400


def points_height(points: list[str], width_in: float, size: float) -> float:
    """About how tall (inches) these bullets are at this font size in a text box this wide: wrapped at a generous
    average character width (0.55 of the font size), less the box's insets and the bullet's indent, plus the space
    after each bullet."""
    per_line = max(8, int((width_in - 0.2 - 0.3) * 72 / (size * 0.55)))
    lines = sum(max(1, math.ceil(len(p) / per_line)) for p in points)
    return lines * size * 1.2 / 72 + len(points) * 8 / 72 + 0.1


def points_size(points: list[str], width_in: float, height_in: float, largest: int, smallest: int = 11) -> int:
    """The largest font size (pt) at which the bullets fit the box: PowerPoint does not shrink text by itself."""
    for size in range(largest, smallest - 1, -1):
        if points_height(points, width_in, size) <= height_in:
            return size
    return smallest


def notes_text(page: dict) -> str:
    out = [_span(page["t0"], page["t1"]) + (f" ({page['chapter']})" if page.get("chapter") else "")]
    if page.get("points"):
        out += ["", "Key points:"] + [f"- {p}" for p in page["points"]]
    out += ["", "What was said:", page.get("said") or "(nothing)"]
    return "\n".join(out)


def build_pptx(job_dir: Path, data: dict, seconds: float | None = None) -> bytes:
    """The deck as a 16:9 PowerPoint: a title slide, then one slide per page (title, image, key points, a footer
    with the time range), with the key points and what was said in the speaker notes."""
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Inches, Pt

    dark, grey, white = RGBColor(0x1F, 0x23, 0x28), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xFF, 0xFF, 0xFF)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    SW, SH = prs.slide_width, prs.slide_height
    margin = Inches(0.5)

    def fill(tf, paras: list[str], size: float, color, bold: bool = False, align=None, after: float = 0,
             bullets: bool = False):
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        for i, text in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = safe_text(text)
            if align is not None:
                p.alignment = align
            if after:
                p.space_after = Pt(after)
            if bullets:  # a real bullet with a hanging indent, so a wrapped line lines up with the text
                pPr = p._p.get_or_add_pPr()
                pPr.set("marL", str(int(Inches(0.3))))
                pPr.set("indent", str(-int(Inches(0.3))))
                pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))  # after spcAft, as the schema wants
            for r in p.runs:
                r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color

    def white_background(slide):
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = white

    def place(shape, left, top, width, height):
        shape.left, shape.top, shape.width, shape.height = int(left), int(top), int(width), int(height)

    # The title slide.
    s = prs.slides.add_slide(prs.slide_layouts[0])
    white_background(s)
    place(s.shapes.title, margin, Inches(2.0), SW - 2 * margin, Inches(1.8))
    s.shapes.title.text_frame.vertical_anchor = MSO_ANCHOR.BOTTOM
    fill(s.shapes.title.text_frame, [data["title"]], 40, dark, bold=True, align=PP_ALIGN.CENTER)
    sub = s.placeholders[1]
    place(sub, margin, Inches(4.0), SW - 2 * margin, Inches(1.4))
    kind = "Slide lecture" if data.get("type") == "slides" else "Board lecture"
    facts = " · ".join(x for x in (kind, clock(seconds) if seconds else "",
                                   f"{len(data['pages'])} slides") if x)
    fill(sub.text_frame, ["Made with Replay Board from the recorded lecture", facts], 20, grey, align=PP_ALIGN.CENTER)

    n = len(data["pages"])
    top = Inches(1.3)
    bottom = SH - Inches(0.65)
    cw, ch = SW - 2 * margin, bottom - top
    gap = Inches(0.3)
    for i, pg in enumerate(data["pages"], 1):
        s = prs.slides.add_slide(prs.slide_layouts[5])  # "Title Only": a real title for the outline and readers
        white_background(s)
        place(s.shapes.title, margin, Inches(0.3), cw, Inches(0.9))
        s.shapes.title.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        title = pg["title"]  # a chapter's title can be long: one line at 28 pt holds about 60 characters
        fill(s.shapes.title.text_frame, [title], 28 if len(title) <= 60 else 24 if len(title) <= 85 else 20, dark,
             bold=True, align=PP_ALIGN.LEFT)

        points = pg.get("points") or []
        path = job_dir / pg["image"] if pg.get("image") else None
        has_pic = path is not None and path.is_file()
        size_ = pg.get("image_size")
        iw, ih = (size_ if isinstance(size_, list) and len(size_) >= 2 and all(_num(v) for v in size_[:2])
                  else [16, 9])[:2]
        iw, ih = max(1, int(iw)), max(1, int(ih))

        def fit(bw, bh):
            k = min(bw / iw, bh / ih)
            return iw * k, ih * k

        # The key points beside the image, or under it: whichever leaves the image bigger. Under it, they get the
        # height they need at 16 pt (within limits); either way their size is then fitted to their box.
        side_w = cw * 0.6 - gap / 2
        text_h = Inches(min(2.6, max(1.0, points_height(points, cw / EMU_PER_INCH, 16) + 0.2))) if points else 0
        beside = has_pic and bool(points) and math.prod(fit(side_w, ch)) >= math.prod(fit(cw, ch - text_h))
        pic_h = 0
        if has_pic:
            if not points:
                w, h = fit(cw, ch)
                x, y = margin + (cw - w) / 2, top + (ch - h) / 2
            elif beside:
                w, h = fit(side_w, ch)
                x, y = margin + (side_w - w) / 2, top + (ch - h) / 2
            else:
                w, h = fit(cw, ch - text_h)
                x, y = margin + (cw - w) / 2, top
            pic = s.shapes.add_picture(io.BytesIO(path.read_bytes()), int(x), int(y), int(w), int(h))
            what = "The slide" if data.get("type") == "slides" else "The board"
            pic._element.nvPicPr.cNvPr.set("descr", safe_text(f"{what} at {clock(pg['t1'])}: {title}"))
            pic_h = h
        if points:
            if beside:
                bw, bh = cw - side_w - gap, ch
                tb = s.shapes.add_textbox(int(margin + side_w + gap), int(top), int(bw), int(bh))
                tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
                largest = 18
            else:
                ty = top + pic_h + Inches(0.2)
                bw, bh = cw, max(Inches(0.5), bottom - ty)
                tb = s.shapes.add_textbox(int(margin), int(ty), int(bw), int(bh))
                tb.text_frame.vertical_anchor = MSO_ANCHOR.TOP
                largest = 16 if has_pic else 20
            size = points_size(points, bw / EMU_PER_INCH, bh / EMU_PER_INCH, largest)
            fill(tb.text_frame, points, size, dark, after=8, bullets=True)

        foot = s.shapes.add_textbox(int(margin), int(SH - Inches(0.55)), int(cw * 0.8), int(Inches(0.35)))
        chapter = pg.get("chapter") or ""
        fill(foot.text_frame, [_span(pg["t0"], pg["t1"]) + (f"  ·  {chapter}" if chapter and chapter != pg["title"]
                                                          else "")], 11, grey)
        num = s.shapes.add_textbox(int(margin + cw * 0.8), int(SH - Inches(0.55)), int(cw * 0.2), int(Inches(0.35)))
        fill(num.text_frame, [f"{i} / {n}"], 11, grey, align=PP_ALIGN.RIGHT)
        s.notes_slide.notes_text_frame.text = safe_text(notes_text(pg))

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def write_pptx(job_dir: Path) -> tuple[bytes, dict]:
    """(the PowerPoint, the deck it shows), also saved as export/slides.pptx. Uses the key points if there are some
    (no model call)."""
    deck, _ = load_deck(job_dir)  # one read of replay.json: the length below is this deck's
    data = with_points(deck, load_points(job_dir, deck["version"]))
    clip = deck.get("clip")
    seconds = (max(0.0, float(clip[1]) - float(clip[0]))
               if isinstance(clip, list) and len(clip) >= 2 and all(_num(v) for v in clip[:2]) else None)
    blob = build_pptx(job_dir, data, seconds)
    try:  # a copy for the owner; open in PowerPoint, it is skipped at once rather than holding up the download
        _save_bytes(_export_dir(job_dir) / "slides.pptx", blob, tries=2)
    except OSError:
        pass
    return blob, data
