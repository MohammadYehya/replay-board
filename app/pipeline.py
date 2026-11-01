"""The whole pipeline for one lecture: video in, job folder with replay.json out.

Used by the CLI (`python -m app.process`) and by the server's background worker.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import DATA_DIR
from .core import media, speech
from .core.board import BoardParams, capture_board
from .core.meta_api import MetaAPIError, MetaClient
from .core import understand

JOBS = DATA_DIR / "jobs"
STAGES = ["probe", "classify", "board", "people", "capture", "speech", "read", "link", "write", "align"]


class JobError(Exception):
    """Something the owner should read as one plain line."""


def replace_file(tmp: Path, path: Path, tries: int = 40) -> bool:
    """os.replace that waits out a reader: on Windows it fails while the server has the file open."""
    for i in range(tries):
        try:
            tmp.replace(path)
            return True
        except PermissionError:
            time.sleep(0.05 * (1 + i % 5))
    return False


def plain_error(e: BaseException) -> str:
    """One line for the owner, whatever went wrong."""
    if isinstance(e, (media.MediaError, MetaAPIError, JobError)):
        return str(e)
    if isinstance(e, MemoryError):
        return "ran out of memory; try a shorter part of the video with --start/--end"
    if isinstance(e, OSError):
        return f"file problem: {e.strerror or e} ({Path(e.filename).name if getattr(e, 'filename', None) else 'disk'})"
    text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return f"something went wrong ({type(e).__name__}: {text[:160]}); details are in error.log"


def job_id_for(video: Path, start: float | None, end: float | None) -> str:
    st = video.stat()
    h = hashlib.sha1(f"{video.resolve()}|{st.st_size}|{int(st.st_mtime)}|{start}|{end}".encode()).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "-", video.stem.lower()).strip("-")[:32] or "lecture"
    return f"{slug}-{h}"


class Status:
    """Per-stage progress written to status.json (read by the server and the upload page)."""

    def __init__(self, job_dir: Path, on_change: Callable[[dict], None] | None = None):
        self.path = job_dir / "status.json"
        self.lock = threading.Lock()
        self.on_change = on_change
        self.data = {"state": "running", "error": None, "started": time.time(),
                     "stages": {s: {"state": "pending", "progress": 0.0, "seconds": None, "note": ""} for s in STAGES}}
        self._t0: dict[str, float] = {}
        self._last_write = 0.0
        self.write(force=True)

    def write(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_write < 0.5:
            return
        self._last_write = now
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        if not replace_file(tmp, self.path) and not force:
            return  # a reader held the file; the next update will get through
        if self.on_change:
            self.on_change(self.data)

    def start(self, stage: str, note: str = ""):
        with self.lock:
            self._t0[stage] = time.perf_counter()
            self.data["stages"][stage].update(state="running", progress=0.0, note=note)
            self.write(force=True)

    def progress(self, stage: str, frac: float):
        with self.lock:
            self.data["stages"][stage]["progress"] = round(float(min(1.0, max(0.0, frac))), 3)
            self.write()

    def done(self, stage: str, note: str = "", state: str = "done"):
        with self.lock:
            t0 = self._t0.get(stage)
            st = self.data["stages"][stage]
            st.update(state=state, progress=1.0, seconds=None if t0 is None else round(time.perf_counter() - t0, 2))
            if note:
                st["note"] = note
            self.write(force=True)

    def skip(self, stage: str, note: str = ""):
        self.done(stage, note, state="skipped")

    def finish(self, error: str | None = None):
        with self.lock:
            self.data["state"] = "failed" if error else "done"
            self.data["error"] = error
            self.data["seconds"] = round(time.time() - self.data["started"], 1)
            self.write(force=True)


class SpeechJob:
    """Transcribe at once; align word times with MMS on its own thread.

    MMS on CPU runs at only ~1-3x real time, so for long lectures the replay is first written
    with turn-level (proportional) word times and rewritten when MMS lands.
    """

    def __init__(self, info, s0: float, s1: float, client: MetaClient, status: "Status"):
        self.info, self.s0, self.s1, self.client, self.status = info, s0, s1, client, status
        # Cleared while a CPU-heavy stage (board capture, slide analysis) runs: alignment waits,
        # so the stages on the way to the first replay get the whole CPU.
        self.cpu_free = threading.Event()
        self.cpu_free.set()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.aligned = None
        self.transcribed = self.pool.submit(self._transcribe)

    def _transcribe(self):
        st, s0 = self.status, self.s0
        st.start("speech")
        pcm = media.extract_audio(self.info, s0, self.s1) if self.info.has_audio else None
        if pcm is None or pcm.size < speech.RATE // 2:
            st.skip("speech", "no sound track")
            st.skip("align", "no sound track")
            return None, []
        turns = speech.transcribe(pcm, self.client, progress=lambda f: st.progress("speech", f))
        st.done("speech", f"{len(turns)} turns")
        self.aligned = self.pool.submit(self._align, pcm, [dict(t) for t in turns])
        for t in turns:
            t["start"] = round(t["start"] + s0, 3)
            t["end"] = round(t["end"] + s0, 3)
        return pcm, turns

    def _align(self, pcm, turns):
        st = self.status
        st.start("align", "MMS word times")
        words, how = speech.align_words(pcm, turns, progress=lambda f: st.progress("align", f),
                                        cpu_free=self.cpu_free)
        for w in words:
            w["s"] = round(w["s"] + self.s0, 3)
            w["e"] = round(w["e"] + self.s0, 3)
        st.done("align", f"{len(words)} words, timing: {how}")
        return words, how

    def first(self):
        """(turns, words, timing) now: MMS times if they are ready, else proportional ones."""
        pcm, turns = self.transcribed.result()
        if not turns:
            return [], [], "none"
        if self.aligned is not None and self.aligned.done():
            words, how = self.aligned.result()
            return turns, words, how
        words = [w for t in turns for w in speech._proportional(t, t["text"].split())]
        return turns, words, "proportional"

    def final(self):
        pcm, turns = self.transcribed.result()
        if self.aligned is None:
            return turns, [], "none"
        words, how = self.aligned.result()
        return turns, words, how


def _write_replay(job_dir: Path, replay: dict):
    path = job_dir / "replay.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(replay, ensure_ascii=False, indent=1), encoding="utf-8")
    if not replace_file(tmp, path, tries=200):
        raise JobError("could not save replay.json: the file is locked by another program")


def _finish_with_speech(speech_job: SpeechJob, link_and_write, job_dir: Path, pool, log) -> Path:
    """Write the replay now; if word times were still refining, link again and rewrite it."""
    turns, words, how = speech_job.first()
    log(f"speech: {len(turns)} turns, {len(words)} words ({how})")
    link_and_write(turns, words, how)
    if how == "proportional":
        log("first version written; refining word times with MMS")
        turns, words, how = speech_job.final()
        link_and_write(turns, words, how)
    shutil.rmtree(job_dir / "tmp", ignore_errors=True)
    pool.shutdown(wait=False)
    return job_dir / "replay.json"


def _link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def hexcolor(bgr) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def run(video: str | Path, job_dir: Path | None = None, start: float | None = None, end: float | None = None,
        use_sam: bool = True, width: int = 800, log: Callable[[str], None] = print,
        on_status: Callable[[dict], None] | None = None, use_cache: bool = True) -> Path:
    """Process one lecture. Returns the path of replay.json."""
    video = Path(video)
    info = media.probe(video)
    s0, s1 = media.clip_range(info, start, end)
    job_dir = Path(job_dir) if job_dir else JOBS / job_id_for(video, start, end)
    job_dir.mkdir(parents=True, exist_ok=True)
    status = Status(job_dir, on_status)
    try:
        out = _run(info, s0, s1, job_dir, status, use_sam, width, log, use_cache)
        status.finish()
        return out
    except Exception as e:  # report one line; keep the traceback in the job folder
        (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        msg = plain_error(e)
        status.finish(msg)
        raise


def _run(info: media.VideoInfo, s0: float, s1: float, job_dir: Path, status: Status, use_sam: bool,
         width: int, log, use_cache: bool = True) -> Path:
    t_all = time.perf_counter()
    status.start("probe")
    if s1 - s0 < 1.0:
        raise JobError("the clip is shorter than 1 second")
    lecture = job_dir / ("lecture" + (info.path.suffix.lower() or ".mp4"))
    _link_or_copy(info.path, lecture)
    status.done("probe", f"{info.width}x{info.height}, {s1 - s0:.0f}s")
    log(f"video: {info.path.name}, {info.width}x{info.height}, {s1 - s0:.1f}s, audio={'yes' if info.has_audio else 'no'}")

    client = MetaClient(use_cache=use_cache)
    pool = ThreadPoolExecutor(max_workers=3)

    # Speech runs alongside everything else.
    speech_job = SpeechJob(info, s0, s1, client, status)

    status.start("classify")
    times = [s0 + (s1 - s0) * f for f in (0.15, 0.4, 0.65, 0.9)]
    frames = [media.grab_frame(info, t, 768) for t in times]
    kind = understand.classify(client, frames)
    status.done("classify", f"{kind.get('type')}, {kind.get('board_colour')}")
    log(f"classify: {kind}")
    if kind.get("type") == "slides":
        return _run_slides(info, s0, s1, job_dir, status, client, kind, speech_job, lecture, pool, t_all, log)

    status.start("board")
    corners, board_info = None, None
    if use_sam:
        from .core.sam import find_board
        board_info = find_board(client, info, s0, s1, colour=kind.get("board_colour"))
        corners = board_info["corners"] if board_info else None
    status.done("board", f"SAM '{board_info['prompt']}' {board_info['score']}" if board_info else "whole frame")
    log(f"board: {board_info or 'not found, using the whole frame'}")

    person = None
    if use_sam:
        from .core.sam import track_people
        status.start("people")
        # SAM chunks come back in the background, in time order; capture starts at once and
        # waits only when it needs a chunk that isn't back yet.
        person, people_done = track_people(client, info, s0, s1, workdir=job_dir / "tmp", background=True,
                                           progress=lambda f: status.progress("people", f))

        def people_finished(fut):
            failed = fut.result()
            if failed:
                status.done("people", f"{failed} SAM chunk(s) failed: built-in detector there")
            else:
                status.done("people", f"person in {len(person.frames)}/{person.n} samples")

        people_done.add_done_callback(people_finished)
    else:
        status.skip("people", "SAM off")

    status.start("capture")
    mode = "chalk" if kind.get("board_colour") == "chalk" else ("white" if kind.get("board_colour") == "white" else None)
    speech_job.cpu_free.clear()
    try:
        board = capture_board(info, corners=corners, person_mask=person, params=BoardParams(width=width),
                              start=s0, end=s1, progress=lambda f: status.progress("capture", f), mode=mode)
    finally:
        speech_job.cpu_free.set()
    pieces = board.pieces
    status.done("capture", f"{len(pieces)} pieces, {len(board.erases)} erases, {board.stats['speed']:.1f}x real time")
    log(f"capture: {len(pieces)} pieces, {len(board.erases)} erases in {board.stats['seconds']:.1f}s "
        f"({board.stats['speed']:.1f}x real time), mode {board.mode}")

    # Piece images.
    (job_dir / "pieces").mkdir(exist_ok=True)
    final = board.image_at(s1 + 1)
    everything = _board_all(board)
    media.imwrite(job_dir / "board_final.png", final)
    for p in pieces:
        bgra, order, box, gone, draw = board.piece_image(p["id"])
        media.imwrite(job_dir / "pieces" / f"{p['id']:03d}.png", bgra)
        media.imwrite(job_dir / "pieces" / f"{p['id']:03d}_order.png", order)
        p["image"], p["order"] = f"pieces/{p['id']:03d}.png", f"pieces/{p['id']:03d}_order.png"
        p["gone"] = None
        if gone is not None:
            media.imwrite(job_dir / "pieces" / f"{p['id']:03d}_gone.png", gone)
            p["gone"] = f"pieces/{p['id']:03d}_gone.png"
        p["image_box"] = box
        p["draw"] = draw

    status.start("read")
    # Sharp crops from the video itself (the 800 px board is too coarse for small handwriting).
    try:
        crops = understand.hires_crops(info, board.corners, board, pieces, person, t_end=s1)
        for pid, img in crops.items():  # kept for the 3D pop-out
            media.imwrite(job_dir / "crops" / f"{pid:03d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception as e:  # reading still works from the rebuilt board
        log(f"read: sharp crops failed ({e}); reading from the rebuilt board")
        crops = {}
    reads = understand.read_pieces(client, board, everything, pieces, subject=kind.get("subject", ""),
                                   progress=lambda f: status.progress("read", f), crops=crops)
    status.done("read", f"{len(reads)} read")

    elements = []
    for p in pieces:
        r = reads.get(p["id"], {})
        elements.append({"id": str(p["id"]), "kind": r.get("kind", ""), "text": r.get("text", ""),
                         "t_seen": p["t_seen"], "t_shown": p["t_shown"], "written_after": p["written_after"],
                         "revealed": p["revealed"], "removed_at": p["removed_at"],
                         **({"pre_existing": True} if p["pre_existing"] else {})})

    def link_and_write(turns, words, how):
        status.start("link")
        links = understand.link(client, elements, words, s1, kind="board") if words else {"elements": {}, "chapters": []}
        status.done("link", f"{len(links['elements'])} linked, {len(links['chapters'])} chapters")
        status.start("write")
        out_pieces = []
        for p in pieces:
            r = reads.get(p["id"], {"kind": "other", "text": "", "latex": ""})
            lk = links["elements"].get(str(p["id"]), {})
            spoken = understand.snap_to_words(lk.get("spoken", []), words)
            redraw = lk.get("redraw_at") if p["revealed"] else None
            if p["revealed"] and redraw is None:
                redraw = spoken[0]["start"] if spoken and spoken[0]["start"] < p["t_seen"] else None
            out_pieces.append({
                "id": p["id"], "box": p["image_box"], "image": p["image"], "order": p["order"], "gone": p["gone"],
                "draw": p["draw"],
                "t_seen": p["t_seen"], "t_shown": p["t_shown"], "written_after": p["written_after"],
                "revealed": p["revealed"], "removed_at": p["removed_at"], "pre_existing": p["pre_existing"],
                "read": r, "spoken": spoken, "redraw_at": redraw,
            })
        replay = {
            "version": 1,
            "video": {"file": lecture.name, "duration": round(info.duration, 3), "width": info.width,
                      "height": info.height, "start": round(s0, 3), "end": round(s1, 3)},
            "type": "board",
            "subject": kind.get("subject", ""),
            "board": {
                "size": [board.width, board.height], "mode": board.mode, "corners": board.corners,
                "background": hexcolor(board.background), "pieces": out_pieces, "erases": board.erases,
                "t_start": round(board.t_start, 2),
            },
            "transcript": {"turns": turns, "words": words, "timing": how},
            "chapters": links["chapters"],
            "stats": {"seconds": round(time.perf_counter() - t_all, 1), "api": client.summary(),
                      "capture": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in board.stats.items()}},
        }
        _write_replay(job_dir, replay)
        status.done("write", "word times refining" if how == "proportional" else "")
        log(f"wrote replay.json ({how} word times) after {replay['stats']['seconds']}s, API {client.summary()}")

    return _finish_with_speech(speech_job, link_and_write, job_dir, pool, log)


def _words_between(words: list[dict], a: float, b: float) -> str:
    return " ".join(w["w"] for w in words if a <= w["s"] <= b)


def _run_slides(info, s0, s1, job_dir: Path, status: Status, client: MetaClient, kind: dict, speech_job,
                lecture: Path, pool, t_all: float, log) -> Path:
    from .core.slides import analyse_slides

    status.skip("board", "slides")
    status.skip("people", "slides")
    status.start("capture", "slide analysis")
    speech_job.cpu_free.clear()
    try:
        res = analyse_slides(info, job_dir / "tmp", s0, s1, progress=lambda f: status.progress("capture", f))
    finally:
        speech_job.cpu_free.set()
    status.done("capture", f"{len(res.states)} states, {len(res.gestures)} gesture guesses, "
                           f"{len(res.live)} live region(s), {res.stats['speed']:.1f}x real time")
    log(f"slides: {len(res.states)} states, {len(res.gestures)} gestures, live {res.live}, {res.stats['speed']:.1f}x")

    (job_dir / "slides").mkdir(exist_ok=True)
    status.start("read")
    readable = [st for st in res.states if st.image is not None and not st.blank]
    for st in res.states:
        if st.image is not None:
            media.imwrite(job_dir / "slides" / f"{st.id:03d}.png", st.image)

    from .core.slides import content_mask, snap_box, word_at

    def read_one(st):
        els = understand.read_slide(client, st.image, kind.get("subject", ""))
        ink = content_mask(st.image, res.background, res.live)
        for i, e in enumerate(els):
            e["id"] = f"s{st.id}e{i + 1}"
            # The model's boxes can be a line off: snap them onto the slide's real ink.
            e["box"] = snap_box(e["box"], ink, e.get("kind", "text"))
        return st.id, els

    elements: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, (sid, els) in enumerate(ex.map(read_one, readable)):
            elements[sid] = els
            status.progress("read", 0.5 * (k + 1) / max(1, len(readable)))

    turns, words, how = speech_job.first()

    # What each pointing gesture means (Muse Spark), with the code's guess as a hint.
    def state_of(t):
        for st in res.states:
            if st.t_change <= t < st.t_end:
                return st
        return res.states[-1] if res.states else None

    cands = [g for g in res.gestures if not (g["type"] == "hover" and g.get("parked")) and g["type"] != "move"]

    inks = {st.id: content_mask(st.image, res.background, res.live) for st in readable}

    def mean_one(g):
        st = state_of(g["t_start"])
        if st is None or st.image is None or st.blank:
            return g, None
        said = _words_between(words, g["t_start"] - 1.0, g["t_end"] + 1.0)
        return g, (st, understand.gesture_meaning(client, st.image, g, elements.get(st.id, []), said))

    annotations = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, (g, out) in enumerate(ex.map(mean_one, cands)):
            status.progress("read", 0.5 + 0.5 * (k + 1) / max(1, len(cands)))
            if out is None:
                continue
            st, m = out
            if m.get("type", "none") == "none":
                continue
            ids = {e["id"]: e for e in elements.get(st.id, [])}
            target = [x for x in m.get("target", []) if x in ids]
            box = g["box"]
            if target and m["type"] in ("spotlight", "underline"):
                box = ids[target[0]]["box"]
            elif target and m["type"] in ("circle", "pulse"):
                # A circle or a wiggle round one word of a long line: the part of the element inside
                # the gesture, tightened to the ink; else the gesture's own extent.
                el, gb = ids[target[0]]["box"], g["box"]
                ix0, iy0 = max(el[0], gb[0]), max(el[1], gb[1])
                ix1, iy1 = min(el[0] + el[2], gb[0] + gb[2]), min(el[1] + el[3], gb[1] + gb[3])
                if ix1 - ix0 > 4 and iy1 - iy0 > 4:
                    box = snap_box([ix0, iy0, ix1 - ix0, iy1 - iy0], inks[st.id], "figure")
                elif gb[2] <= 4 or gb[3] <= 4:
                    # A rest or a tap: the word under the pointer, else the element.
                    box = word_at(inks[st.id], gb[0] + gb[2] / 2, gb[1] + gb[3] / 2) or el
            fr, to = g.get("from"), g.get("to")
            if m["type"] == "arrow" and len(target) >= 2:
                a, b = ids[target[0]]["box"], ids[target[1]]["box"]
                fr = [a[0] + a[2] / 2, a[1] + a[3] / 2]
                to = [b[0] + b[2] / 2, b[1] + b[3] / 2]
            annotations.append({"type": m["type"], "tool": g["tool"], "t_start": g["t_start"], "t_end": g["t_end"],
                                "target": target, "box": [round(float(v), 1) for v in box], "from": fr, "to": to,
                                "why": m.get("why", ""), "guess": g["type"], "state": st.id,
                                "path": g.get("path", [])[:200]})
    status.done("read", f"{sum(len(v) for v in elements.values())} elements, {len(annotations)} annotations")

    link_els = [{"id": e["id"], "kind": e["kind"], "text": e["text"], "t_start": round(st.t_start, 2),
                 "t_end": round(st.t_end, 2), "slide": st.id} for st in readable for e in elements.get(st.id, [])]

    def link_and_write(turns, words, how):
        status.start("link")
        links = understand.link(client, link_els, words, s1, kind="slides") if words else {"elements": {}, "chapters": []}
        status.done("link", f"{len(links['elements'])} linked, {len(links['chapters'])} chapters")
        status.start("write")
        out_states = []
        for st in res.states:
            els = []
            for e in elements.get(st.id, []):
                lk = links["elements"].get(e["id"], {})
                els.append({**e, "spoken": understand.snap_to_words(lk.get("spoken", []), words)})
            out_states.append({"id": st.id, "kind": st.kind, "t_start": round(st.t_start, 2), "t_end": round(st.t_end, 2),
                               "t_change": round(st.t_change, 2), "blank": st.blank,
                               "image": f"slides/{st.id:03d}.png" if st.image is not None else None, "elements": els})
        replay = {
            "version": 1,
            "video": {"file": lecture.name, "duration": round(info.duration, 3), "width": info.width,
                      "height": info.height, "start": round(s0, 3), "end": round(s1, 3)},
            "type": "slides",
            "subject": kind.get("subject", ""),
            "slides": {"size": [res.width, res.height], "background": hexcolor(res.background), "live": res.live,
                       "states": out_states, "annotations": annotations,
                       "pointer": [[round(t, 2), tool[0], round(x, 1), round(y, 1)] for t, tool, x, y in res.track]},
            "transcript": {"turns": turns, "words": words, "timing": how},
            "chapters": links["chapters"],
            "stats": {"seconds": round(time.perf_counter() - t_all, 1), "api": client.summary(),
                      "slides": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.stats.items()}},
        }
        _write_replay(job_dir, replay)
        status.done("write", "word times refining" if how == "proportional" else "")
        log(f"wrote replay.json ({how} word times) after {replay['stats']['seconds']}s, API {client.summary()}")

    return _finish_with_speech(speech_job, link_and_write, job_dir, pool, log)


def _board_all(board) -> np.ndarray:
    """Every kept mark ever written, drawn on the board colour (for reading, incl. erased pieces)."""
    img = np.empty((board.height * board.width, 3), np.uint8)
    img[:] = board.background
    img[board.add_idx] = board.add_color
    return img.reshape(board.height, board.width, 3)
