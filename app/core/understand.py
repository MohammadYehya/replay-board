"""Muse Spark: every judgement about meaning (SPEC section 3).

- classify: board or slides, board colour, webcam corner
- read_pieces: kind, text and LaTeX of each piece of board writing
- link: when each element is talked about, the redraw time of hidden writing, chapters
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np

from .meta_api import MetaClient, image_part, text_part

KINDS = ["text", "equation", "number", "heading", "diagram", "graph", "arrow", "underline", "box",
         "table", "label", "scribble", "other"]

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["board", "slides", "other"]},
        "board_colour": {"type": "string", "enum": ["white", "chalk", "none"]},
        "has_webcam_corner": {"type": "boolean"},
        "subject": {"type": "string"},
    },
}


def classify(client: MetaClient, frames: list[np.ndarray]) -> dict:
    """4 frames sampled across the lecture -> {type, board_colour, has_webcam_corner, subject}."""
    prompt = (
        "These frames are sampled across one recorded lecture. Decide what kind of recording it is.\n"
        "- type: 'board' if a camera films a physical whiteboard or chalkboard that the teacher writes on; "
        "'slides' if it is a screen recording of slides or a document (maybe with a cursor, pen ink or a "
        "small webcam picture of the speaker); 'other' otherwise.\n"
        "- board_colour: 'white' for a whiteboard, 'chalk' for a dark chalkboard, 'none' for slides.\n"
        "- has_webcam_corner: true if a small live video of the speaker is embedded in the screen.\n"
        "- subject: the topic in a few words.")
    return client.ask(prompt, [image_part(f, max_side=768) for f in frames], schema=CLASSIFY_SCHEMA,
                      reasoning="low")


# ---------------------------------------------------------------- reading

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "pieces": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "kind": {"type": "string", "enum": KINDS},
                "text": {"type": "string"},
                "latex": {"type": "string"},
            }}},
    },
}


def piece_crop(board_img: np.ndarray, own: np.ndarray, box: list, margin: float = 0.35,
               min_side: int = 112, max_side: int = 640) -> np.ndarray:
    """Crop around a piece: its own ink in colour, the other ink faded, for context."""
    H, W = board_img.shape[:2]
    x, y, w, h = box
    m = int(max(24, margin * max(w, h)))
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    crop = board_img[y0:y1, x0:x1].astype(np.float32)
    bg = np.median(board_img.reshape(-1, 3), axis=0)
    faded = bg + (crop - bg) * 0.25
    mine = own[y0:y1, x0:x1]
    out = np.where(mine[..., None], crop, faded).clip(0, 255).astype(np.uint8)
    k = max(1.0, min_side / max(1, min(out.shape[:2])))
    k = min(k, max_side / max(out.shape[:2]))
    if abs(k - 1) > 0.05:
        out = cv2.resize(out, (int(out.shape[1] * k), int(out.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    return out


def read_pieces(client: MetaClient, board, board_img: np.ndarray, pieces: list[dict], subject: str = "",
                per_call: int = 10, workers: int = 6, progress: Callable[[float], None] | None = None) -> dict:
    """{piece id: {kind, text, latex}} read by Muse Spark, several pieces per call, calls in parallel."""
    if not pieces:
        return {}
    W = board.width
    overview = image_part(board_img, max_side=1024)
    batches = [pieces[i:i + per_call] for i in range(0, len(pieces), per_call)]
    done = [0]

    def one(batch):
        parts = [text_part(
            "You are reading handwriting on a lecture whiteboard/chalkboard"
            + (f" (topic: {subject})" if subject else "") + ". The first image is the whole board for context. "
            "Each following image is one piece of writing, shown in colour; nearby writing is faded and is NOT "
            "part of the piece. For each piece give: kind; text = what it says in plain words/symbols "
            "(empty for pure drawings); latex = LaTeX for maths (empty if not maths). Describe drawings briefly "
            "in text (e.g. 'graph of y = ln|x|', 'arrow pointing right'). Do not guess beyond what is written."),
            overview]
        for p in batch:
            own = np.zeros(board_img.shape[:2], bool)
            idx = board.add_idx[board.add_piece == p["id"]]
            own.reshape(-1)[idx] = True
            parts.append(text_part(f"Piece id {p['id']}:"))
            parts.append(image_part(piece_crop(board_img, own, p["box"]), fmt=".png"))
        reply = client.chat([{"role": "user", "content": parts}], schema=READ_SCHEMA, reasoning="low")
        from .meta_api import chat_text, parse_json_object
        data = parse_json_object(chat_text(reply))
        done[0] += 1
        if progress:
            progress(done[0] / len(batches))
        return data.get("pieces", [])

    out = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
        for res in ex.map(one, batches):
            for r in res:
                out[int(r["id"])] = {"kind": r.get("kind", "other"), "text": r.get("text", ""),
                                     "latex": r.get("latex", "")}
    return out


# ---------------------------------------------------------------- slides

SLIDE_KINDS = ["title", "heading", "text", "bullet", "equation", "figure", "chart", "bar", "table", "label",
               "code", "ink", "other"]
SLIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": SLIDE_KINDS},
                "text": {"type": "string"},
                "latex": {"type": "string"},
                "box_2d": {"type": "array", "items": {"type": "number"},
                           "description": "[x_min, y_min, x_max, y_max] normalised 0-1000"},
            }}},
    },
}


def read_slide(client: MetaClient, img: np.ndarray, subject: str = "") -> list[dict]:
    """Elements of one clean slide: [{kind, text, latex, box:[x,y,w,h] in image px}]."""
    H, W = img.shape[:2]
    prompt = ("This is one slide from a recorded lecture" + (f" about {subject}" if subject else "") + ". "
              "List its visible elements in reading order: each title, heading, bullet or line of text, "
              "equation, figure, chart and each bar or labelled part of a chart, table, and any hand-drawn ink. "
              "text = the exact words; latex = LaTeX for maths (else empty); box_2d = a tight box "
              "[x_min, y_min, x_max, y_max] normalised to 0-1000 from the top-left. Reply in JSON.")
    data = client.ask(prompt, [image_part(img, fmt=".png")], schema=SLIDE_SCHEMA, reasoning="low")
    out = []
    for e in data.get("elements", []):
        b = e.get("box_2d") or []
        if len(b) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in b)
        x0, x1 = sorted((max(0, min(1000, x0)), max(0, min(1000, x1))))
        y0, y1 = sorted((max(0, min(1000, y0)), max(0, min(1000, y1))))
        box = [round(x0 * W / 1000), round(y0 * H / 1000), max(1, round((x1 - x0) * W / 1000)),
               max(1, round((y1 - y0) * H / 1000))]
        out.append({"kind": e.get("kind", "other"), "text": e.get("text", ""), "latex": e.get("latex", ""),
                    "box": box})
    return out


GESTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["spotlight", "circle", "underline", "arrow", "pulse", "none"]},
        "target": {"type": "array", "items": {"type": "string"}},
        "why": {"type": "string"},
    },
}


def draw_path(img: np.ndarray, path: list, colour=(255, 0, 200)) -> np.ndarray:
    """The slide with the pointer path drawn on it: a line, a start dot and an end arrowhead."""
    out = img.copy()
    pts = np.array(path, np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return out
    if len(pts) == 1 or np.ptp(pts, axis=0).max() < 3:
        c = tuple(int(v) for v in pts.mean(axis=0))
        cv2.circle(out, c, 10, colour, 2, cv2.LINE_AA)
        cv2.circle(out, c, 3, colour, -1, cv2.LINE_AA)
        return out
    cv2.polylines(out, [pts.round().astype(np.int32)], False, colour, 2, cv2.LINE_AA)
    cv2.circle(out, tuple(int(v) for v in pts[0]), 5, colour, -1, cv2.LINE_AA)
    k = max(0, len(pts) - 4)
    cv2.arrowedLine(out, tuple(int(v) for v in pts[k]), tuple(int(v) for v in pts[-1]), colour, 2, cv2.LINE_AA,
                    tipLength=0.5)
    return out


def gesture_meaning(client: MetaClient, slide: np.ndarray, gesture: dict, elements: list[dict], words: str) -> dict:
    """What a pointing gesture means: spotlight/circle/underline/arrow/pulse/none + target element ids."""
    els = "\n".join(json.dumps({"id": e["id"], "kind": e["kind"], "text": e["text"][:80], "box": e["box"]},
                               ensure_ascii=False) for e in elements)
    prompt = (
        "A teacher is pointing at this slide with a " + gesture["tool"] + ". The magenta line is the pointer's "
        "path (dot = start, arrowhead = end). Decide how to replay this gesture on a clean copy of the slide:\n"
        "- spotlight: they rest on one thing to draw attention to it;\n"
        "- circle: they circle something;\n- underline: they sweep under a line of text;\n"
        "- arrow: they move from one thing to another to connect them (from first target to second);\n"
        "- pulse: a quick emphasis on a small thing (a wiggle or a tap);\n"
        "- none: the pointer is just travelling or resting with no meaning.\n"
        f"The code's geometric guess is '{gesture['type']}' (a hint, may be wrong). "
        f"Words spoken meanwhile: \"{words or '(none)'}\".\n"
        "target: the ids of the slide elements meant (1 for most gestures, 2 for an arrow: from, to). "
        "why: one short sentence.\n\nELEMENTS (boxes are [x, y, w, h] in image pixels):\n" + els)
    return client.ask(prompt, [image_part(draw_path(slide, gesture.get("path") or []), fmt=".png")],
                      schema=GESTURE_SCHEMA, reasoning="low")


# ---------------------------------------------------------------- linking

LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "spoken": {"type": "array", "items": {
                    "type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}}}},
                "redraw_at": {"type": "number", "description": "-1 when not applicable"},
            }}},
        "chapters": {"type": "array", "items": {
            "type": "object", "properties": {"t": {"type": "number"}, "title": {"type": "string"}}}},
    },
}


def transcript_lines(words: list[dict], max_words: int = 12, gap: float = 1.2) -> list[str]:
    """Words -> '[12.3] text' lines, breaking on pauses and every few words."""
    lines, cur, t0, last_e = [], [], None, None
    for w in words:
        if cur and (len(cur) >= max_words or (last_e is not None and w["s"] - last_e > gap)):
            lines.append(f"[{t0:.1f}] " + " ".join(cur))
            cur, t0 = [], None
        if t0 is None:
            t0 = w["s"]
        cur.append(w["w"])
        last_e = w["e"]
    if cur:
        lines.append(f"[{t0:.1f}] " + " ".join(cur))
    return lines


def link(client: MetaClient, elements: list[dict], words: list[dict], duration: float, kind: str = "board",
         window: float = 1200.0, workers: int = 4) -> dict:
    """Spoken ranges per element, redraw time for hidden writing, and chapters.

    `elements`: [{id (str), kind, text, t_seen, t_shown, written_after, revealed, removed_at}] for a
    board, or [{id, kind, text, t_start, t_end}] for slides. Long lectures are linked in time windows
    of `window` seconds (with 2 min of overlap), in parallel.
    """
    if not elements:
        return {"elements": {}, "chapters": []}
    starts = np.arange(0.0, max(duration, 1.0), window)
    jobs = []
    for a in starts:
        b = a + window
        els = [e for e in elements if a <= _el_time(e) < b]
        if not els:
            continue
        ws = [w for w in words if a - 120 <= w["s"] <= b + 120]
        jobs.append((a, b, els, ws, a == 0.0))
    if not jobs:
        jobs = [(0.0, duration, elements, words, True)]

    def one(job):
        a, b, els, ws, first = job
        el_lines = [json.dumps(_el_brief(e), ensure_ascii=False) for e in els]
        tr = "\n".join(transcript_lines(ws)) or "(no speech)"
        what = ("pieces of handwriting on a board" if kind == "board" else "elements of slides")
        prompt = (
            f"Below are {what} from a recorded lecture (times in seconds), and the teacher's speech with "
            "start times. For EVERY element, list the time ranges when the teacher is talking about it "
            "(spoken: 1-3 ranges, each tight to the relevant words, a few seconds to ~30 s; empty if never "
            "mentioned). Writing is usually explained while it is being written (between written_after and "
            "t_shown) and sometimes referred back to later.\n"
            "redraw_at: for an element with revealed=true (written while the teacher's body hid it, so it "
            "became visible only at t_seen), the time when the teacher actually wrote/said it, normally "
            "between written_after and t_seen; -1 for all other elements.\n"
            + ("chapters: split the whole lecture into 3-10 topic chapters with short titles (first at "
               f"{max(0.0, a):.0f}).\n" if first else "chapters: return an empty list.\n")
            + "Use only times from the transcript and the elements. Reply in JSON.\n\n"
            "ELEMENTS:\n" + "\n".join(el_lines) + "\n\nSPEECH:\n" + tr)
        return client.ask(prompt, schema=LINK_SCHEMA, reasoning="medium")

    results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        results = list(ex.map(one, jobs))

    out: dict[str, dict] = {}
    chapters = []
    ids = {str(e["id"]) for e in elements}
    for res in results:
        for r in res.get("elements", []):
            eid = str(r.get("id"))
            if eid not in ids:
                continue
            spoken = []
            for s in r.get("spoken", [])[:4]:
                a, b = float(s.get("start", -1)), float(s.get("end", -1))
                if 0 <= a < b <= duration + 1:
                    spoken.append({"start": round(a, 2), "end": round(min(b, duration), 2)})
            ra = r.get("redraw_at", -1)
            out[eid] = {"spoken": spoken, "redraw_at": round(float(ra), 2) if ra is not None and ra >= 0 else None}
        chapters += [c for c in res.get("chapters", []) if 0 <= float(c.get("t", -1)) <= duration]
    chapters = sorted({round(float(c["t"]), 1): c["title"].strip() for c in chapters if c.get("title")}.items())
    return {"elements": out, "chapters": [{"t": t, "title": title} for t, title in chapters]}


def _el_time(e: dict) -> float:
    return float(e.get("t_seen", e.get("t_start", 0.0)) or 0.0)


def _el_brief(e: dict) -> dict:
    keep = ("id", "kind", "text", "t_seen", "t_shown", "written_after", "revealed", "removed_at",
            "t_start", "t_end", "slide")
    return {k: e[k] for k in keep if k in e and e[k] is not None and e[k] != ""}


def snap_to_words(ranges: list[dict], words: list[dict]) -> list[dict]:
    """Move range edges onto the nearest word boundaries."""
    if not words:
        return ranges
    starts = np.array([w["s"] for w in words])
    ends = np.array([w["e"] for w in words])
    out = []
    for r in ranges:
        a = float(starts[np.argmin(np.abs(starts - r["start"]))])
        b = float(ends[np.argmin(np.abs(ends - r["end"]))])
        if b > a:
            out.append({"start": round(a, 2), "end": round(b, 2)})
        else:
            out.append(r)
    return out
