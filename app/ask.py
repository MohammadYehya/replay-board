"""Ask Muse: a student's questions about the lecture they are watching, answered by Muse Spark.

Every question gets the whole lecture as context (Muse Spark reads about a million tokens): the board
writing or the slides with their ids and times, the chapters and the full transcript. That context is
the same for every question about a lecture and goes first, so Meta's prompt cache makes later
questions cheap. What the student attached (writing from anywhere in the lecture, a moment, a stretch
of speech) is looked up here in replay.json, with images. The answer cites [m:ss] times and [#id]
items, which the player turns into links.

All times are on the original video's clock, like replay.json and the player.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from .core.media import imread
from .core.meta_api import MetaAPIError, image_part, text_part
from .pipeline import plain_error, replace_file

MAX_QUESTION = 2000
MAX_ITEMS = 12
MAX_HISTORY = 6
MAX_HISTORY_CHARS = 2000
MAX_IMAGES = 6
MAX_SIDE = 1024
MAX_TOKENS = 2000
REASONING = "low"  # "none" is rejected by the API
MAX_SAVED_TURNS = 200
DRAWINGS = {"diagram", "graph", "arrow", "scribble", "underline", "box", "figure", "chart", "drawing"}
CUT_NOTE = "\n\n*(The answer was cut short. Ask me to go on.)*"

SYSTEM = """You are Muse, the study helper in Replay Board, a player for recorded lectures. A student is \
watching the lecture described below and asks you about it. You have all of it: the writing on the board \
(or the slides) with ids and times, the chapters, and the whole transcript with times.

How to answer:
- Answer from the lecture first, and point to where it happens. When you go beyond what the lecture says \
or shows, say so briefly (for example "Beyond the lecture: ...").
- Be concise: about 250 words, unless the student asks for more. Answer in the language of the question.
- Cite moments as [m:ss] ([h:mm:ss] past an hour) on the video's clock, the clock used below, and board \
or slide items by id: [#12] for a piece of writing, [#L3] for a line of writing, [#s0e1] for a slide \
element. Put one time or one id in each pair of brackets. Only cite times and ids that appear below; \
never make one up. The student sees each cited id as a chip showing that item's own writing, so do not \
copy the item's formula again right after its id.
- Write plain paragraphs, "- " lists and "1. " lists, with **bold**, *italic* and `code`. Write maths in \
LaTeX: $...$ inline, $$...$$ on a line of its own. No headings, tables, links, images or HTML.
- The transcript comes from speech recognition and the board was read by a model, so a word or a symbol \
may be wrong; go by the sense of the lecture.
- The student may attach items from the lecture (with images of them) and say where they are in the \
video. "This" and "these" mean what they attached, or else what is on the board at their time."""


class AskError(ValueError):
    """A bad request, as one plain line (HTTP 400)."""


# ---------- small helpers ----------
def clock(t: float) -> str:
    """m:ss, or h:mm:ss past an hour. Rounded down, so a link lands just before the moment."""
    t = max(0, int(math.floor(float(t) + 1e-6)))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _short(text: str, n: int = 60) -> str:
    text = _flat(text)
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _what(read: dict | None) -> str:
    """'equation $A=P(1+r)^n$' or 'graph "graph of y = ln x"': kind, then LaTeX for maths or the text."""
    read = read or {}
    kind = _flat(read.get("kind")) or "writing"
    latex = _flat(read.get("latex")).strip("$").strip()
    text = _flat(read.get("text"))
    if latex and kind.lower() not in DRAWINGS:
        return f"{kind} ${latex}$"
    return f'{kind} "{text}"' if text else kind


def _label(read: dict | None, fallback: str = "") -> str:
    read = read or {}
    text = _flat(read.get("text"))
    if not text or text.lower() == "unreadable":
        text = _flat(read.get("latex")) or _flat(read.get("kind")) or fallback
    return _short(text, 60)


def _merge(spans: list[dict]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s in sorted((float(s["start"]), float(s.get("end", s["start"]))) for s in spans
                    if isinstance(s, dict) and _num(s.get("start"))):
        if out and s[0] <= out[-1][1] + 0.5:
            out[-1][1] = max(out[-1][1], s[1])
        else:
            out.append([s[0], s[1]])
    return [(a, b) for a, b in out]


def _ranges(spans: list[tuple[float, float]], limit: int = 4) -> str:
    parts = []
    for a, b in spans[:limit]:
        ca, cb = clock(a), clock(b)
        parts.append(ca if ca == cb else f"{ca}-{cb}")
    if len(spans) > limit:
        parts.append(f"and {len(spans) - limit} more times")
    return ", ".join(parts)


def _words(replay: dict) -> list[dict]:
    return [w for w in (replay.get("transcript") or {}).get("words") or []
            if isinstance(w, dict) and _num(w.get("s")) and w.get("w")]


def speech_lines(words: list[dict], a: float = -math.inf, b: float = math.inf, max_words: int = 12,
                 gap: float = 1.2) -> list[str]:
    """Words spoken from a to b -> '[m:ss] text' lines of about 12 words, broken on pauses."""
    lines, cur, t0, last_e = [], [], None, None
    for w in words:
        if not a <= w["s"] <= b:
            continue
        if cur and (len(cur) >= max_words or (last_e is not None and w["s"] - last_e > gap)):
            lines.append(f"[{clock(t0)}] " + " ".join(cur))
            cur, t0 = [], None
        if t0 is None:
            t0 = w["s"]
        cur.append(str(w["w"]))
        last_e = w.get("e", w["s"])
    if cur:
        lines.append(f"[{clock(t0)}] " + " ".join(cur))
    return lines


def _said(words: list[dict], a: float, b: float, limit: int = 60) -> str:
    ws = [str(w["w"]) for w in words if a <= w["s"] <= b]
    return " ".join(ws[:limit]) + (" ..." if len(ws) > limit else "")


def _video(replay: dict) -> tuple[float, float, float]:
    """(duration, clip start, clip end) of the video, in seconds."""
    v = replay.get("video") or {}
    words = _words(replay)
    dur = float(v["duration"]) if _num(v.get("duration")) else (words[-1].get("e", words[-1]["s"]) if words else 0.0)
    a = float(v["start"]) if _num(v.get("start")) else 0.0
    b = float(v["end"]) if _num(v.get("end")) else dur
    return dur, a, b


# ---------- the lecture's items: board pieces and lines, slide elements ----------
@dataclass
class Item:
    ref: str           # "#12", "#L3", "#s0e1"
    kind: str          # piece | line | element
    id: Any            # 12 | "L3" | "s0e1"
    t: float           # where the player seeks for it
    label: str
    first: float       # when it first appears (for ordering)
    data: dict = field(repr=False, default_factory=dict)
    spoken: list = field(repr=False, default_factory=list)  # merged (start, end)
    members: list = field(repr=False, default_factory=list)  # a line's pieces
    state: dict | None = field(repr=False, default=None)  # an element's slide state
    line: str | None = None  # the line a piece belongs to


def item_index(replay: dict) -> dict[str, Item]:
    """Every board piece, board line and slide element of the lecture, by ref."""
    _, clip0, _ = _video(replay)
    out: dict[str, Item] = {}
    if replay.get("type") == "slides":
        for st in (replay.get("slides") or {}).get("states") or []:
            for e in st.get("elements") or []:
                if not isinstance(e.get("id"), str):
                    continue
                sp = _merge(e.get("spoken") or [])
                t0 = float(st.get("t_start") or 0.0)
                out["#" + e["id"]] = Item("#" + e["id"], "element", e["id"], max(clip0, sp[0][0] if sp else t0),
                                          _label(e, "slide element"), t0, e, sp, state=st)
        return out
    board = replay.get("board") or {}
    pieces = {p["id"]: p for p in board.get("pieces") or [] if isinstance(p.get("id"), int)}
    for p in pieces.values():
        sp = _merge(p.get("spoken") or [])
        t = sp[0][0] if sp else (p.get("redraw_at") if _num(p.get("redraw_at")) else p.get("t_seen", clip0))
        out[f"#{p['id']}"] = Item(f"#{p['id']}", "piece", p["id"], max(clip0, float(t)),
                                  _label(p.get("read"), "writing"), float(p.get("t_seen", clip0)), p, sp)
    for ln in board.get("lines") or []:
        members = [pieces[i] for i in ln.get("pieces") or [] if i in pieces]
        if not isinstance(ln.get("id"), str) or not members:
            continue
        sp = _merge([s for p in members for s in p.get("spoken") or []])
        if sp:
            t = sp[0][0]
        else:
            t = min(float(p["redraw_at"]) if _num(p.get("redraw_at")) else float(p.get("t_seen", clip0))
                    for p in members)
        out["#" + ln["id"]] = Item("#" + ln["id"], "line", ln["id"], max(clip0, t), _label(ln.get("read"), "line"),
                                   min(float(p.get("t_seen", clip0)) for p in members), ln, sp, members)
        for p in members:
            if f"#{p['id']}" in out:
                out[f"#{p['id']}"].line = ln["id"]
    return out


def _board_when(it: Item, clip0: float) -> str:
    """'written 1:03 | erased 2:22 | talked about 1:01-1:10' for a piece or a line."""
    ps = it.members or [it.data]
    parts = []
    if it.data.get("pre_existing") or all(p.get("pre_existing") for p in ps):
        parts.append("on the board from the start" + (f" ({clock(clip0)})" if clip0 > 0.5 else ""))
    else:
        a = min(float(p.get("t_seen", 0.0)) for p in ps)
        b = float(it.data.get("t_complete", a)) if it.kind == "line" else a
        hidden = [bool(p.get("revealed")) for p in ps]
        if all(hidden):  # seen only once the teacher moved away: written before that
            parts.append(f"written by {clock(b)}, hidden behind the teacher until then")
        else:
            parts.append(f"written {clock(a)}" + (f"-{clock(b)}" if clock(b) != clock(a) else ""))
            if any(hidden):
                parts.append("partly written while hidden behind the teacher")
    if _num(it.data.get("removed_at")):
        parts.append(f"erased {clock(it.data['removed_at'])}")
    if it.spoken:
        parts.append(f"talked about {_ranges(it.spoken)}")
    return " | ".join(parts)


# ---------- the whole lecture, as text ----------
def lecture_context(replay: dict) -> str:
    """The lecture as compact text: the same for every question about it (so the prompt cache hits)."""
    dur, clip0, clip1 = _video(replay)
    kind = replay.get("type")
    out = ["# The lecture"]
    if replay.get("subject"):
        out.append(f"Subject: {_flat(replay['subject'])}")
    if kind == "slides":
        out.append("Kind: a slide lecture (a screen recording with a voice-over). Each slide state has elements "
                   "(#s0e1); a state that 'adds to the one before' is a build step of the previous slide.")
    else:
        out.append("Kind: a filmed board lecture. The writing was captured piece by piece (#12), and pieces "
                   "written along one line were grouped into lines (#L3).")
    out.append(f"Video: {clock(dur)} long. Every time here is on the video's clock.")
    if clip0 > 0.5 or clip1 < dur - 0.5:
        out.append(f"Only {clock(clip0)} to {clock(clip1)} of the video was analysed: the "
                   f"{'slides' if kind == 'slides' else 'board'} and the transcript cover just that part.")

    chapters = sorted((c for c in replay.get("chapters") or [] if _num(c.get("t"))), key=lambda c: c["t"])
    if chapters:
        out += ["", "# Chapters"] + [f"[{clock(c['t'])}] {_flat(c.get('title'))}" for c in chapters]

    index = item_index(replay)
    if kind == "slides":
        out += ["", "# Slides", "Each element: id, kind, what it says, and when it was talked about."]
        prev: dict[tuple, str] = {}
        for st in (replay.get("slides") or {}).get("states") or []:
            head = f"Slide state {st.get('id')}"
            if st.get("kind") == "addition":
                head += " (adds to the one before)"
            head += f", on screen {clock(st.get('t_start', 0))}-{clock(st.get('t_end', st.get('t_start', 0)))}"
            if st.get("blank"):
                head += " (blank)"
            out.append(head + ":")
            seen: dict[tuple, str] = {}
            for e in st.get("elements") or []:
                it = index.get("#" + str(e.get("id")))
                if it is None:
                    continue
                key = (_flat(e.get("kind")).lower(), _flat(e.get("text")), _flat(e.get("latex")))
                talk = f" | talked about {_ranges(it.spoken)}" if it.spoken else ""
                if key in prev:
                    out.append(f"{it.ref} same as #{prev[key]}{talk}")
                else:
                    out.append(f"{it.ref} {_what(e)}{talk}")
                seen[key] = prev.get(key, e["id"])
            prev = seen
    else:
        board = [it for it in index.values() if it.kind == "line" or it.line is None]
        board.sort(key=lambda it: (it.first, 0 if it.kind == "line" else 1, str(it.id).zfill(6)))
        out += ["", "# Board",
                "Each item: id, kind, what it says, then when it was written, erased and talked about."]
        for it in board:
            read = it.data.get("read") or {}
            out.append(f"{it.ref} {_what(read)} | {_board_when(it, clip0)}")

    lines = speech_lines(_words(replay))
    out += ["", "# Transcript"] + (lines or ["(no speech)"])
    return "\n".join(out)


# ---------- the request ----------
def check_request(body: Any) -> dict:
    """The POST body -> {question, selection, history, t}. AskError (one plain line) when it is wrong."""
    if not isinstance(body, dict):
        raise AskError("the request must be a JSON object with a question")
    q = body.get("question")
    if not isinstance(q, str) or not q.strip():
        raise AskError("please type a question")
    q = q.strip()
    if len(q) > MAX_QUESTION:
        raise AskError(f"the question is too long ({len(q)} characters; at most {MAX_QUESTION})")

    sel = body.get("selection") or []
    if not isinstance(sel, list):
        raise AskError("selection must be a list of attached items")
    if len(sel) > MAX_ITEMS:
        raise AskError(f"attach at most {MAX_ITEMS} items (got {len(sel)})")
    items = []
    for s in sel:
        if not isinstance(s, dict):
            raise AskError("each attached item must be an object with a kind")
        k = s.get("kind")
        it: dict[str, Any] = {"kind": k}
        if k == "piece":
            i = s.get("id")
            if isinstance(i, str) and i.strip().isdigit():
                i = int(i)
            if not isinstance(i, int) or isinstance(i, bool):
                raise AskError("an attached piece needs a number id")
            it["id"] = i
        elif k in ("line", "element"):
            if not isinstance(s.get("id"), str) or not s["id"].strip():
                raise AskError(f"an attached {k} needs an id")
            it["id"] = s["id"].strip()
        elif k == "span":
            if not (_num(s.get("t0")) and _num(s.get("t1"))):
                raise AskError("an attached span needs t0 and t1 in seconds")
            it["t0"], it["t1"] = sorted((float(s["t0"]), float(s["t1"])))
        elif k == "moment":
            if not _num(s.get("t")):
                raise AskError("an attached moment needs t in seconds")
            it["t"] = float(s["t"])
        else:
            raise AskError(f"unknown kind of attached item: {str(k)[:20]!r}")
        if isinstance(s.get("label"), str) and s["label"].strip():
            it["label"] = _short(s["label"], 80)
        items.append(it)

    hist = body.get("history") or []
    if not isinstance(hist, list):
        raise AskError("history must be a list of earlier questions and answers")
    history = []
    for h in hist[-MAX_HISTORY:]:
        if not isinstance(h, dict) or not isinstance(h.get("q"), str) or not isinstance(h.get("a"), str):
            raise AskError("each history entry needs a question q and an answer a")
        q_, a_ = h["q"].strip()[:MAX_QUESTION], h["a"].strip()[:MAX_HISTORY_CHARS]
        if q_ and a_:  # an empty turn says nothing, and the model's API can refuse an empty message
            history.append({"q": q_, "a": a_})

    t = body.get("t")
    if t is not None and not _num(t):
        raise AskError("t must be the video time in seconds, or null")
    return {"question": q, "selection": items, "history": history, "t": None if t is None else float(t)}


# ---------- what the student attached ----------
@dataclass
class Attachment:
    item: dict                      # what is saved and shown: kind, id or times, label, t
    text: str                       # what Muse Spark reads about it
    image: Callable[[], np.ndarray | None] | None = None
    fmt: str = ".png"
    name: str = ""                  # how the prompt names its image
    grow: int = 0                   # small images are enlarged towards this size


def _bg(replay: dict) -> np.ndarray:
    h = str((replay.get("board") or {}).get("background") or "#f4f4f2").lstrip("#")
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        r, g, b = 244, 244, 242
    return np.array([b, g, r], np.float32)


def _bgra(job_dir: Path, rel: str | None) -> np.ndarray | None:
    if not rel:
        return None
    try:
        img = imread(job_dir / rel)
    except Exception:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _paste(canvas: np.ndarray, bgra: np.ndarray, x: int, y: int) -> None:
    H, W = canvas.shape[:2]
    h, w = bgra.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4].astype(np.float32) / 255.0
    canvas[y0:y1, x0:x1] = src[..., :3].astype(np.float32) * a + canvas[y0:y1, x0:x1] * (1 - a)


def fit_image(img: np.ndarray, grow_to: int = 0) -> np.ndarray:
    """At most MAX_SIDE on the longest side; small images grown towards `grow_to` (at most 4x)."""
    h, w = img.shape[:2]
    side = max(h, w)
    k = 1.0
    if side > MAX_SIDE:
        k = MAX_SIDE / side
    elif grow_to and side < grow_to:
        k = min(4.0, min(grow_to, MAX_SIDE) / side)
    if abs(k - 1.0) < 1e-3:
        return img
    size = (max(1, min(MAX_SIDE, round(w * k))), max(1, min(MAX_SIDE, round(h * k))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)


def _box(b: Any) -> tuple[int, int, int, int]:
    x, y, w, h = (int(round(float(v))) for v in b[:4])
    return x, y, max(1, w), max(1, h)


def compose_pieces(job_dir: Path, replay: dict, pieces: list[dict], pad: int = 10,
                   whole_board: bool = False) -> np.ndarray | None:
    """Pieces drawn at their places on the board colour: cropped to them (padded), or the whole board."""
    imgs = [(p, _bgra(job_dir, p.get("image"))) for p in pieces]
    imgs = [(p, im) for p, im in imgs if im is not None]
    if whole_board:
        W, H = (int(v) for v in (replay.get("board") or {}).get("size") or (800, 450))
        x0, y0 = 0, 0
    else:
        if not imgs:
            return None
        boxes = [_box(p["box"]) for p, _ in imgs]
        x0 = min(b[0] for b in boxes) - pad
        y0 = min(b[1] for b in boxes) - pad
        W = max(b[0] + b[2] for b in boxes) + pad - x0
        H = max(b[1] + b[3] for b in boxes) + pad - y0
    canvas = np.empty((max(1, H), max(1, W), 3), np.float32)
    canvas[:] = _bg(replay)
    for p, im in imgs:
        x, y, _, _ = _box(p["box"])
        _paste(canvas, im, x - x0, y - y0)
    return np.clip(canvas + 0.5, 0, 255).astype(np.uint8)


def _slide_crop(job_dir: Path, state: dict, box: Any, size: Any = None, pad: int = 16) -> np.ndarray | None:
    """The element's box cut from its slide image (padded). `size` is the slides' size the box is in."""
    try:
        img = imread(job_dir / state["image"], cv2.IMREAD_COLOR)
    except Exception:
        return None
    H, W = img.shape[:2]
    x, y, w, h = _box(box)
    if size and _num(size[0]) and size[0] > 0 and abs(W / size[0] - 1) > 0.01:  # an image of another size
        k = W / size[0]
        x, y, w, h = round(x * k), round(y * k), max(1, round(w * k)), max(1, round(h * k))
        pad = round(pad * k)
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


def _slide_full(job_dir: Path, state: dict | None) -> np.ndarray | None:
    if not state or not state.get("image"):
        return None
    try:
        return imread(job_dir / state["image"], cv2.IMREAD_COLOR)
    except Exception:
        return None


def visible_pieces(replay: dict, t: float) -> list[dict]:
    """The board's pieces showing at t: already there or shown by t, and not erased by t."""
    out = []
    for p in (replay.get("board") or {}).get("pieces") or []:
        shown = p.get("pre_existing") or float(p.get("t_shown", p.get("t_seen", math.inf))) <= t
        gone = _num(p.get("removed_at")) and p["removed_at"] <= t
        if shown and not gone:
            out.append(p)
    return sorted(out, key=lambda p: (float(p.get("t_shown", 0.0)), p.get("id", 0)))


def state_at(replay: dict, t: float) -> dict | None:
    states = sorted((replay.get("slides") or {}).get("states") or [], key=lambda s: s.get("t_start", 0.0))
    cur = states[0] if states else None
    for s in states:
        if s.get("t_start", 0.0) <= t:
            cur = s
    return cur


def resolve_selection(job_dir: Path, replay: dict, selection: list[dict],
                      index: dict[str, Item] | None = None) -> list[Attachment]:
    """Checked selection items -> attachments with their details (and images, made later). Unknown ids
    are dropped; repeats are kept once."""
    index = item_index(replay) if index is None else index
    words = _words(replay)
    dur, clip0, clip1 = _video(replay)
    slides = replay.get("type") == "slides"
    out: list[Attachment] = []
    seen = set()
    for s in selection:
        k = s["kind"]
        if k in ("piece", "line", "element"):
            ref = f"#{s['id']}"
            it = index.get(ref)
            if it is None or it.kind != k or ref in seen:
                continue
            seen.add(ref)
            item = {"kind": k, "id": it.id, "label": s.get("label") or it.label, "t": round(it.t, 2)}
            if k == "element":
                st = it.state or {}
                text = (f"Slide element {ref} on slide state {st.get('id')} (on screen "
                        f"{clock(st.get('t_start', 0))}-{clock(st.get('t_end', 0))}): {_what(it.data)}")
                if it.spoken:
                    text += f" | talked about {_ranges(it.spoken)}"
                image = (lambda st=st, e=it.data: _slide_crop(job_dir, st, e.get("box") or [0, 0, 1, 1],
                                                              (replay.get("slides") or {}).get("size")))
                fmt = ".jpg"
            else:
                name = "Line of writing" if k == "line" else "Piece of writing"
                text = f"{name} {ref}: {_what(it.data.get('read'))} | {_board_when(it, clip0)}"
                if k == "line":
                    text += f" | made of pieces {', '.join('#' + str(p['id']) for p in it.members)}"
                elif it.line:
                    text += f" | part of line #{it.line}"
                ps = it.members or [it.data]
                image = (lambda ps=ps: compose_pieces(job_dir, replay, ps))
                fmt = ".png"
            if it.spoken:
                said = " / ".join(_said(words, a, b) for a, b in it.spoken[:2])
                if said.strip():
                    text += f'\n   Said then ({clock(it.spoken[0][0])}): "{said}"'
            elif words:
                a = it.first if k != "element" else float((it.state or {}).get("t_start", 0.0))
                said = _said(words, a - 10, a + 15)
                if said.strip():
                    text += f'\n   Said around when it appeared ({clock(max(0, a - 10))}): "{said}"'
            out.append(Attachment(item, text, image, fmt, ref, 512))
        elif k == "moment":
            t = min(max(s["t"], 0.0), dur) if dur else max(s["t"], 0.0)
            key = ("moment", round(t, 1))
            if key in seen:
                continue
            seen.add(key)
            what = "slide" if slides else "board"
            item = {"kind": "moment", "t": round(t, 2), "label": s.get("label") or f"The {what} at {clock(t)}"}
            text = f"The {what} at [{clock(t)}]."
            if slides:
                st = state_at(replay, t)
                if st is not None:
                    els = " ".join(f"#{e['id']}" for e in st.get("elements") or [] if e.get("id"))
                    text += f" It shows slide state {st.get('id')}" + (f", with elements {els}" if els else "") + "."
                image = (lambda st=st: _slide_full(job_dir, st))
                fmt = ".jpg"
            else:
                vis = visible_pieces(replay, t)
                ids = {p.get("id") for p in vis}
                lines = [it for it in index.values() if it.kind == "line"
                         and all(p.get("id") in ids for p in it.members)]
                in_lines = {p.get("id") for it in lines for p in it.members}
                refs = [it.ref for it in lines] + [f"#{p['id']}" for p in vis if p.get("id") not in in_lines]
                if refs:
                    text += " On the board then: " + ", ".join(refs[:40]) + (" ..." if len(refs) > 40 else "") + "."
                    image = (lambda t=t: compose_pieces(job_dir, replay, visible_pieces(replay, t), whole_board=True))
                else:  # no picture of an empty board: it would only use up one of the images
                    text += " The board was empty then."
                    image = None
                fmt = ".png"
            sp = speech_lines(words, t - 20, t + 10)
            text += (f"\n   Speech from {clock(max(0, t - 20))} to {clock(t + 10)}:\n   " + "\n   ".join(sp)
                     if sp else "\n   Nothing was said around then.")
            out.append(Attachment(item, text, image, fmt, f"the {what} at {clock(t)}", 0 if slides else MAX_SIDE))
        elif k == "span":
            t0, t1 = s["t0"], s["t1"]
            key = ("span", round(t0, 1), round(t1, 1))
            if key in seen:
                continue
            seen.add(key)
            sp = speech_lines(words, t0 - 0.05, t1 + 0.05)
            label = s.get("label") or _short(_said(words, t0 - 0.05, t1 + 0.05, 12), 60) or f"Speech at {clock(t0)}"
            item = {"kind": "span", "t0": round(t0, 2), "t1": round(t1, 2), "label": label, "t": round(t0, 2)}
            if len(sp) > 40:
                sp = sp[:40] + ["..."]
            text = (f"What was said from [{clock(t0)}] to [{clock(t1)}]:\n   " + "\n   ".join(sp)
                    if sp else f"The stretch from [{clock(t0)}] to [{clock(t1)}] (nothing was said then).")
            out.append(Attachment(item, text, None))
    return out


def attachment_parts(atts: list[Attachment], max_images: int = MAX_IMAGES) -> tuple[list[str], list[dict]]:
    """(the attachments' text lines, image content parts with a caption before each): at most `max_images`."""
    lines, parts = [], []
    n = 0
    for i, a in enumerate(atts, 1):
        img = None
        if a.image is not None and n < max_images:
            try:
                img = a.image()
            except Exception:
                img = None
        text = f"{i}. {a.text}"
        if img is not None and img.size:
            img = fit_image(img, a.grow)
            n += 1
            text += f"\n   (image {n} below)"
            parts += [text_part(f"Image {n}: {a.name}"), image_part(img, fmt=a.fmt, quality=90)]
        lines.append(text)
    return lines, parts


def build_messages(system: str, question: str, atts: list[Attachment], history: list[dict],
                   t: float | None) -> list[dict]:
    """The whole lecture first (stable, so cached), then earlier turns, then this question."""
    msgs: list[dict] = [{"role": "system", "content": system}]
    for h in history:
        msgs += [{"role": "user", "content": h["q"]}, {"role": "assistant", "content": h["a"]}]
    head = []
    if t is not None:
        head.append(f"The student is at [{clock(t)}] in the video.")
    lines, images = attachment_parts(atts)
    if lines:
        head.append("They attached these from the lecture:")
        head += lines
    if not head and not images:
        msgs.append({"role": "user", "content": question})
        return msgs
    content = [text_part("\n".join(head))] if head else []
    content += images
    content.append(text_part(f"Question: {question}"))
    msgs.append({"role": "user", "content": content})
    return msgs


# ---------- the answer's citations ----------
_REF = re.compile(r"\[#([A-Za-z0-9]+)\]")
_REF_LIST = re.compile(r"\[(#[A-Za-z0-9]+(?:\s*[,;]\s*#?[A-Za-z0-9]+)+)\]")
_T = r"\d{1,3}:[0-5]\d(?::[0-5]\d)?"
_TIME = re.compile(r"\[(\d{1,3}):([0-5]\d)(?::([0-5]\d))?\]")
_TIME_RANGE = re.compile(rf"\[({_T})\s*(?:-|–|—|to)\s*({_T})\]")


def tidy_answer(answer: str) -> str:
    """'[#3, #4]' -> '[#3] [#4]' and '[1:02-1:10]' -> '[1:02]–[1:10]', so each becomes a link."""
    answer = _REF_LIST.sub(lambda m: " ".join("[#" + r.strip().lstrip("#") + "]"
                                              for r in re.split(r"[,;]", m.group(1))), answer)
    return _TIME_RANGE.sub(lambda m: f"[{m.group(1)}]–[{m.group(2)}]", answer)


def parse_time(m: re.Match) -> float:
    a, b, c = m.group(1), m.group(2), m.group(3)
    if c is not None:
        return int(a) * 3600 + int(b) * 60 + int(c)
    return int(a) * 60 + int(b)


def answer_refs(answer: str, replay: dict, index: dict[str, Item] | None = None) -> tuple[list[dict], list[float]]:
    """(the items the answer cites that exist, the cited times inside the video), in order of first use."""
    index = item_index(replay) if index is None else index
    dur, _, _ = _video(replay)
    items, seen = [], set()
    for m in _REF.finditer(answer):
        ref = "#" + m.group(1)
        it = index.get(ref)
        if it is None or ref in seen:
            continue
        seen.add(ref)
        items.append({"ref": ref, "kind": it.kind, "id": it.id, "t": round(it.t, 2), "label": it.label})
    times: list[float] = []
    for m in _TIME.finditer(answer):
        t = float(parse_time(m))
        if 0 <= t <= dur + 0.5 and t not in times:
            times.append(t)
    return items, times


# ---------- the conversation, saved per job ----------
_chat_lock = threading.Lock()


def load_chat(job_dir: Path, strict: bool = False) -> dict:
    """The saved conversation. A broken file reads as an empty one. A file that cannot be read at all (Windows:
    held by another program) also reads as empty, unless `strict`: then it is an OSError, so a save does not
    replace the whole conversation with its one new turn."""
    p = job_dir / "chat.json"
    data: Any = None
    unreadable = False
    for _ in range(20):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            unreadable = False
            break
        except FileNotFoundError:
            return {"turns": []}
        except json.JSONDecodeError:
            unreadable = False
            time.sleep(0.05)
        except OSError:
            unreadable = True
            time.sleep(0.05)  # Windows can refuse a read while the file is being replaced
    if unreadable and strict:
        raise OSError("could not save the conversation: chat.json cannot be read")
    turns = data.get("turns") if isinstance(data, dict) else None
    return {"turns": [t for t in turns if isinstance(t, dict)] if isinstance(turns, list) else []}


def _write_chat(job_dir: Path, data: dict) -> None:
    p = job_dir / "chat.json"
    tmp = p.with_name(f"chat.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if not replace_file(tmp, p):
        tmp.unlink(missing_ok=True)
        raise OSError("could not save the conversation: chat.json is in use")


def save_turn(job_dir: Path, turn: dict) -> None:
    with _chat_lock:
        data = load_chat(job_dir, strict=True)
        data["turns"] = (data["turns"] + [turn])[-MAX_SAVED_TURNS:]
        _write_chat(job_dir, data)


def clear_chat(job_dir: Path) -> None:
    with _chat_lock:
        _write_chat(job_dir, {"turns": []})


# ---------- one question, end to end ----------
def sse(event: str, data: dict) -> str:
    """One server-sent event. json.dumps writes no newlines, so the data is one line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@dataclass
class Prepared:
    job_dir: Path
    replay: dict
    question: str
    selection: list[dict]
    messages: list[dict]
    cache_key: str
    t: float | None
    index: dict[str, Item]


def prepare(job_dir: Path, replay: dict, body: Any) -> Prepared:
    """Check the request and build the messages (raises AskError for a bad request)."""
    req = check_request(body)
    index = item_index(replay)
    atts = resolve_selection(job_dir, replay, req["selection"], index)
    system = SYSTEM + "\n\n" + lecture_context(replay)
    messages = build_messages(system, req["question"], atts, req["history"], req["t"])
    key = f"replay-board-{job_dir.name}-{hashlib.sha256(system.encode('utf-8')).hexdigest()[:16]}"
    return Prepared(job_dir, replay, req["question"], [a.item for a in atts], messages, key, req["t"], index)


def plain(e: BaseException) -> str:
    if isinstance(e, MetaAPIError):
        return f"Muse Spark could not answer: {str(e).splitlines()[0] if str(e) else 'unknown error'}"
    if isinstance(e, OSError) and (e.errno is not None or e.filename):
        return plain_error(e)  # the system's own error ('[WinError 32] ... C:\\Users\\...'): file problem, file name
    if isinstance(e, (AskError, OSError)):
        return str(e).splitlines()[0] if str(e) else type(e).__name__
    text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return f"something went wrong ({type(e).__name__}: {text[:160]}); details are in ask-error.log"


def answer_events(prep: Prepared, make_client: Callable[[], Any]) -> Iterator[str]:
    """Server-sent events for one question: delta... then done, or error (one plain line)."""
    parts: list[str] = []
    client = stream = None
    try:
        client = make_client()
        n0 = len(getattr(client, "calls", None) or [])
        stream = client.chat_stream(prep.messages, reasoning=REASONING, max_tokens=MAX_TOKENS,
                                    extra={"prompt_cache_key": prep.cache_key})
        for d in stream:
            if d:
                parts.append(d)
                yield sse("delta", {"text": d})
        # The calls this answer made: one, or two when an empty stream was asked again without streaming.
        recs = (getattr(client, "calls", None) or [])[n0:]
        if any("cut at max tokens" in (getattr(r, "note", "") or "") for r in recs):
            parts.append(CUT_NOTE)
            yield sse("delta", {"text": CUT_NOTE})
        answer = tidy_answer("".join(parts))
        items, times = answer_refs(answer, prep.replay, prep.index)
        costs = [float(r.cost_usd) for r in recs if getattr(r, "cost_usd", None) is not None]
        cost = round(sum(costs), 6) if costs else None
        cached = bool(recs) and all(getattr(r, "cached", False) for r in recs)
        turn = {"id": uuid.uuid4().hex[:12], "q": prep.question, "selection": prep.selection, "answer": answer,
                "items": items, "times": times, "t": prep.t, "at": round(time.time(), 3),
                "cost_usd": cost, "cached": cached}
    except Exception as e:
        if not isinstance(e, (MetaAPIError, AskError)):
            try:
                (prep.job_dir / "ask-error.log").write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass
        yield sse("error", {"error": plain(e)})
        return
    finally:  # also when the student stops the answer and the connection is dropped
        for close in (getattr(stream, "close", None), getattr(getattr(client, "http", None), "close", None)):
            if close is not None:
                try:
                    close()  # the model's stream first (it logs what the call did), then the connection
                except Exception:
                    pass
    # Saved before "done", so a reload right after shows it. A failed save does not cost the student the answer:
    # "done" says saved=false (an error event after "done" would make the player drop the answer).
    saved = True
    try:
        save_turn(prep.job_dir, turn)
    except OSError:
        saved = False
    yield sse("done", {"id": turn["id"], "answer": answer, "items": items, "times": times,
                       "cost_usd": turn["cost_usd"], "cached": cached, "selection": prep.selection, "saved": saved})
