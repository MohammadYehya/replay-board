"""Streaming replay for board lectures (SPEC "Streaming replay"): while capture runs, replay.json is written in
parts, so the player opens early and the clean board fills in as processing goes.

A partial replay has the final replay's shape plus "partial": {seq, ready_until}. Its piece images are
content-addressed files in stream/ (same bytes, same name; never rewritten), so a path always means the same picture.
The final replay is written by the pipeline exactly as without streaming.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .core import media

STREAM_DIR = "stream"
NO_READ = {"kind": "other", "text": "", "latex": ""}


@dataclass
class Snapshot:
    """What one partial replay is made of: the board so far (None before capture starts), the capture time it is
    from, how far it is ready, and whether pieces new in it may be read now."""
    board: Any
    t: float
    ready_until: float
    read_new: bool = True


def content_file(job_dir: Path, img: np.ndarray) -> str:
    """Write a PNG under stream/ named by its content (sha1, 12 hex) unless it is there: same content, same name, so
    a path always means the same bytes and a file is never rewritten. Returns the path relative to the job folder."""
    data = media.encode_image(img, ".png")
    name = hashlib.sha1(data).hexdigest()[:12] + ".png"
    path = job_dir / STREAM_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{name}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        for i in range(40):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # a reader has the (identical) file open: it is there already
                if path.exists():
                    break
                time.sleep(0.05 * (1 + i % 5))
        try:
            tmp.unlink()
        except OSError:
            pass
    return f"{STREAM_DIR}/{name}"


def ready_pieces(board, ready_until: float) -> list[dict]:
    """The board's pieces a partial replay shows: seen by ready_until (t_seen), and with their first ink shown by
    then too. Writing already on the board when the clip starts has t_seen = the first settled moment whenever it was
    accepted, so without the second rule the teacher standing there at the start (no person mask) showed up as
    writing in the first partials and went away later."""
    if not board.pieces:
        return []
    first: dict[int, float] = {}
    if board.add_t.size:
        order = np.argsort(board.add_piece, kind="stable")
        pids, starts = np.unique(board.add_piece[order], return_index=True)
        mins = np.minimum.reduceat(board.add_t[order], starts)
        first = {int(k): float(v) for k, v in zip(pids, mins)}
    return [q for q in board.pieces if q["t_seen"] <= ready_until and first.get(q["id"], np.inf) <= ready_until]


def _overlap(a, b) -> bool:
    """Whether two [x, y, w, h] boxes overlap."""
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def partial_pieces(job_dir: Path, board, ready_until: float, reads: dict, shown: dict | None = None) -> list[dict]:
    """The pieces of ready_pieces as the replay lists them, with content-addressed images. No speech yet: spoken is
    empty and redraw_at None.

    shown (kept by the caller across the partials of a job: id -> the entry last written) keeps a piece on the board
    once a partial showed it: a growing piece can fail capture's shape checks for a while (a few new pixels make a
    short stroke look like a hairline) or have its seen time move past ready_until, and it went away for one partial
    and came back in the next. It stays as last shown until it is there again, or until the final replay, which
    decides; unless the partial has an erase over it after it was seen (then it was rubbed out)."""
    out = []
    for q in ready_pieces(board, ready_until):
        bgra, order, box, gone, draw = board.piece_image(q["id"])
        redraw = board.redraw_part(q["id"]) or {}
        out.append({
            "id": q["id"], "box": box, "image": content_file(job_dir, bgra), "order": content_file(job_dir, order),
            "gone": None if gone is None else content_file(job_dir, gone), "draw": draw,
            "t_seen": q["t_seen"], "t_shown": q["t_shown"], "written_after": q["written_after"],
            "revealed": q["revealed"], "removed_at": q["removed_at"], "pre_existing": q["pre_existing"],
            "read": dict(reads.get(q["id"], NO_READ)), "spoken": [], "redraw_at": None,
            "redraw_to": redraw.get("redraw_to"), "redraw_secs": redraw.get("redraw_secs"),
        })
    if shown is None:
        return out
    now = {q["id"] for q in out}
    erases = [e for e in board.erases if e["t"] <= ready_until]
    for pid, old in list(shown.items()):
        if pid in now:
            continue
        if any(e["t"] >= old["t_seen"] and _overlap(e["box"], old["box"]) for e in erases):
            del shown[pid]
            continue
        out.append({**old, "read": dict(reads.get(pid, NO_READ))})
    out.sort(key=lambda d: (d["t_seen"], d["box"][0]))  # as capture lists them (a stable sort keeps its order)
    shown.update({q["id"]: q for q in out if q["id"] in now})
    return out


class StreamWriter:
    """One background thread owns every partial write. A snapshot that arrives while it is busy replaces the one
    waiting (the older is dropped). After writing a snapshot, pieces in it that were never read are read in the
    background (at most 2 small batches at once) and the partial is written again with their readings; a read that
    fails leaves its pieces unread and never fails the job. New pieces wait until there are min_batch of them, or the
    first has waited max_wait seconds: every read call also sends the whole board, so one or two pieces a call cost
    nearly as much as ten. end_reads() (capture is done: the read stage reads everything) drops the reads not started
    yet. stop() waits for the write in progress and drops the rest.

    build(snapshot, reads) -> (replay dict, ready_until); read(snapshot, pieces) -> {piece id: read}; write(replay,
    ready_until) -> False once the final replay is written (ReplayWriter.write_partial)."""

    min_batch = 5
    max_wait = 20.0

    def __init__(self, write: Callable[[dict, float], bool], build: Callable[[Snapshot, dict], tuple[dict, float]],
                 read: Callable[[Snapshot, list[dict]], dict] | None = None, log: Callable[[str], None] = print,
                 batch: int = 10, readers: int = 2):
        self._write, self._build, self._read, self._log = write, build, read, log
        self.batch = batch
        self._waiting: dict[int, float] = {}  # shown, never read: since when (time.monotonic)
        self._reads_off = False
        self._futures: list = []
        self._cv = threading.Condition()
        self._next: Snapshot | None = None  # waiting to be written
        self._last: Snapshot | None = None  # written last: written again when readings arrive
        self._again = False
        self._stopped = False
        self._busy = False
        self.reads: dict[int, dict] = {}
        self._asked: set[int] = set()
        self._warned: set[str] = set()
        self.written: list[dict] = []  # one entry per partial written (seq, times, counts): for logs and tests
        self._pool = ThreadPoolExecutor(max_workers=readers, thread_name_prefix="stream-read") if read else None
        self._thread = threading.Thread(target=self._loop, name="stream-writer", daemon=True)
        self._thread.start()

    def submit(self, snap: Snapshot):
        """Queue a snapshot (from the capture thread or the pipeline): never blocks."""
        with self._cv:
            if self._stopped:
                return
            self._next = snap
            self._cv.notify_all()

    def use_reads(self, reads: dict):
        """Readings that arrived some other way (the final read stage): the last partial is written again with them."""
        with self._cv:
            if self._stopped:
                return
            self.reads.update({int(k): v for k, v in reads.items()})
            self._asked.update(int(k) for k in reads)
            self._again = self._last is not None
            self._cv.notify_all()

    def end_reads(self):
        """Capture is done and the read stage reads every piece: partial reads not started yet are dropped (they
        would be paid for twice); the ones running finish and are used."""
        with self._cv:
            self._reads_off = True
            self._waiting = {}
            futures, self._futures = self._futures, []
        for f in futures:
            f.cancel()

    def idle(self, timeout: float = 10.0) -> bool:
        """Wait until nothing is waiting to be written (for tests)."""
        end = time.monotonic() + timeout
        with self._cv:
            while (self._next is not None or self._again or self._busy) and not self._stopped:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def stop(self, timeout: float = 120.0):
        """No more partial writes: wait for the one in progress, drop what waits, let running reads go unused."""
        with self._cv:
            self._stopped = True
            self._next = None
            self._again = False
            self._cv.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _warn(self, what: str, e: BaseException):
        if what in self._warned:
            return
        self._warned.add(what)
        text = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
        self._log(f"stream: {what} failed ({text}); the final replay is not affected")

    def _loop(self):
        while True:
            with self._cv:
                while self._next is None and not self._again and not self._stopped:
                    self._cv.wait()
                if self._stopped:
                    return
                if self._next is not None:
                    snap, fresh = self._next, True
                    self._next = None
                else:
                    snap, fresh = self._last, False
                self._again = False
                reads = dict(self.reads)
                self._busy = True
            try:
                self._one(snap, fresh, reads)
            except Exception as e:  # a partial replay is a bonus: the job goes on
                self._warn("a partial replay", e)
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _one(self, snap: Snapshot, fresh: bool, reads: dict):
        t0 = time.perf_counter()
        replay, ready = self._build(snap, reads)
        if not self._write(replay, ready):
            with self._cv:
                self._stopped = True
            return
        pieces = replay["board"]["pieces"]
        with self._cv:
            self._last = snap
            self.written.append({"seq": replay["partial"]["seq"], "t": round(snap.t, 2), "ready_until": ready,
                                 "pieces": len(pieces), "read": sum(1 for q in pieces if q["id"] in reads),
                                 "at": time.time(), "seconds": round(time.perf_counter() - t0, 3)})
        if not (fresh and snap.read_new and self._pool is not None and snap.board is not None):
            return
        shown = {q["id"] for q in pieces}
        now = time.monotonic()
        with self._cv:
            if self._reads_off:
                return
            new = [q for q in snap.board.pieces if q["id"] in shown and q["id"] not in self._asked]
            self._waiting = {q["id"]: self._waiting.get(q["id"], now) for q in new}
            if not new or (len(new) < self.min_batch and now - min(self._waiting.values()) < self.max_wait):
                return
            self._asked.update(q["id"] for q in new)
            self._waiting = {}
            n = -(-len(new) // self.batch)  # as few calls as the batch size allows, of even sizes
            size = -(-len(new) // n)
            for i in range(0, len(new), size):
                try:
                    self._futures.append(self._pool.submit(self._read_batch, snap, new[i:i + size]))
                except RuntimeError:  # stopped meanwhile
                    return
            self._futures = [f for f in self._futures if not f.done()]

    def _read_batch(self, snap: Snapshot, batch: list[dict]):
        if self._stopped or self._reads_off:
            return
        try:
            got = self._read(snap, batch)
        except Exception as e:  # those pieces stay unread until the final read
            self._warn("reading pieces for a partial replay", e)
            return
        with self._cv:
            if self._stopped or not got:
                return
            self.reads.update({int(k): v for k, v in got.items()})
            self._again = self._last is not None
            self._cv.notify_all()


def board_colour(info, geom, times, mode: str | None) -> tuple:
    """The clean board's colour (BGR) before capture has a frame of its own: the median over the middles of a few
    flattened frames, never the first one alone (an edited video opens on a title card). Cosmetic: capture's own
    colour replaces it at the first checkpoint."""
    try:
        cols = []
        for t in times:
            img = geom.warp(media.grab_frame(info, t, geom.dw))
            h, w = img.shape[:2]
            cols.append(np.median(img[h // 4: h - h // 4, w // 4: w - w // 4].reshape(-1, 3), axis=0))
        return tuple(int(v) for v in np.median(np.array(cols), axis=0))
    except Exception:
        return (40, 48, 44) if mode == "chalk" else (245, 245, 245)

