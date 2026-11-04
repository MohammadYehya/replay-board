"""The whole pipeline for one lecture: video in, job folder with replay.json out.

Used by the CLI (`python -m app.process`) and by the server's background worker.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import DATA_DIR, load_settings
from .core import media, speech
from .core.board import BoardGeometry, BoardParams, capture_board, detect_mode
from .core.meta_api import MetaAPIError, MetaClient
from .core import understand
from .stream import STREAM_DIR, Snapshot, StreamWriter, board_colour, partial_pieces

JOBS = DATA_DIR / "jobs"
STAGES = ["probe", "classify", "board", "people", "capture", "speech", "read", "gestures", "link", "write", "align"]


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


# A job whose run stopped before the end without an error of its own: Ctrl+C, or the server stopped while it ran
INTERRUPTED = "processing was interrupted before it finished; process the video again"


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


def _saved_name(status_path: Path) -> str:
    """The lecture's file name an earlier status.json gives ("" when there is none)."""
    try:
        name = json.loads(status_path.read_text(encoding="utf-8")).get("name")
    except (OSError, ValueError, AttributeError):
        return ""
    return name.strip() if isinstance(name, str) else ""


def source_name(video: Path, job_dir: Path) -> str:
    """The lecture's own file name, shown in the lecture list and on its pages: the video's name, except for a video
    the server saved in the job folder as upload.mp4, whose real name the server wrote in status.json."""
    video = Path(video)
    try:
        uploaded = video.resolve().parent == Path(job_dir).resolve()
    except OSError:
        uploaded = False
    return (_saved_name(Path(job_dir) / "status.json") if uploaded else "") or video.name


class Status:
    """Per-stage progress written to status.json (read by the server and the upload page), with the lecture's file
    name. Without a name it keeps the one status.json has: the server writes the uploaded file's name there."""

    def __init__(self, job_dir: Path, on_change: Callable[[dict], None] | None = None, name: str | None = None):
        self.path = job_dir / "status.json"
        self.lock = threading.Lock()
        self.on_change = on_change
        name = (name or "").strip() or _saved_name(self.path)
        # pid: the process running the job, so a server starting later can tell a job that is still running (python
        # -m app.process) from one whose process is gone (recover_interrupted)
        self.data = {"state": "running", "error": None, "started": time.time(), "pid": os.getpid(),
                     "stages": {s: {"state": "pending", "progress": 0.0, "seconds": None, "note": ""} for s in STAGES}}
        if name:
            self.data["name"] = name
        self._t0: dict[str, float] = {}
        self._last_write = 0.0
        self.write(force=True)

    @property
    def name(self) -> str:
        return self.data.get("name") or ""

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

    def replay(self, seq: int | None, partial: bool = False, ready_until: float | None = None):
        """Which replay.json is on disk (seq goes up with every write during the job), so the player fetches it
        only when it changed; seq None: there is none (a partial one was taken away)."""
        with self.lock:
            if seq is None:
                self.data.pop("replay", None)
            else:
                self.data["replay"] = {"seq": int(seq), "partial": bool(partial),
                                       "ready_until": None if ready_until is None else round(float(ready_until), 2)}
            self.write(force=True)

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
        st.done("speech", f"{len(turns)} spoken passages")
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
        st.done("align", f"{len(words)} words timed" + ("" if how == "mms" else " (approximately)"))
        return words, how

    def first(self):
        """(turns, words, timing) now: MMS times if they are ready, else proportional ones."""
        pcm, turns = self.transcribed.result()
        if not turns:
            return [], [], "none"
        if self.aligned is not None and self.aligned.done():
            words, how = self.aligned.result()
            return turns, words, how
        return turns, self.proportional(), "proportional"

    def proportional(self) -> list[dict]:
        """Turn-level (proportional) word times: always the same, whenever MMS finishes."""
        pcm, turns = self.transcribed.result()
        return [w for t in turns for w in speech._proportional(t, t["text"].split())]

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
    # A few facts for the lecture list, so it never has to read a big replay.
    v = replay.get("video", {})
    summary = {"subject": replay.get("subject", ""), "type": replay.get("type", ""),
               "seconds": round(float(v.get("end", v.get("duration", 0))) - float(v.get("start", 0)), 1),
               "source": v.get("source", "")}
    text = json.dumps(summary, ensure_ascii=False)
    spath = job_dir / "summary.json"
    try:
        same = spath.read_text(encoding="utf-8") == text
    except OSError:
        same = False
    if not same:  # a streamed lecture writes replay.json many times: the lecture list must not catch this half written
        spath.write_text(text, encoding="utf-8")


def _replay_kind(path: Path) -> str | None:
    """What replay.json is on disk: "partial" (written while the lecture is still processed), "final", or None (no
    file). A file that cannot be read now (locked, being replaced) counts as final, so it is never taken away."""
    for i in range(10):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return "partial" if isinstance(data, dict) and "partial" in data else "final"
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(0.05)
    return None if not path.exists() else "final"


def drop_partial_replay(job_dir: Path) -> bool:
    """Take a partial replay.json and its stream/ pictures away (after a failure or an interrupted run, the job then
    reads as failed, as before streaming); a final replay.json stays. True when none is left that is partial."""
    path = job_dir / "replay.json"
    if _replay_kind(path) != "partial":
        return True
    for i in range(40):
        try:
            path.unlink()
            break
        except FileNotFoundError:
            break
        except PermissionError:
            time.sleep(0.05 * (1 + i % 5))
    if path.exists():
        return False
    shutil.rmtree(job_dir / STREAM_DIR, ignore_errors=True)
    return True


def _alive(pid) -> bool:
    """Whether a process with this id is running (a process that cannot be looked at counts as running)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":  # os.kill(pid, 0) would stop the process on Windows
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5  # access denied: it is there
        try:
            code = wintypes.DWORD()
            return not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def recover_interrupted(jobs: Path | None = None, log: Callable[[str], None] = print) -> list[str]:
    """When the server starts: a job left "queued" (the queue went with the server) or "running" by a process that
    is gone (the server or python -m app.process stopped mid-job) would show as processing for good, and a partial
    replay would keep its "Processing" line and closed gates. Such a job is marked failed and its partial replay
    taken away; a job whose final replay was written already (stopped while refining word times) is done, with the
    replay it has. A job whose process still runs is left alone. Returns the ids of the jobs changed."""
    jobs = JOBS if jobs is None else Path(jobs)
    changed = []
    if not jobs.is_dir():
        return changed
    for d in sorted(jobs.iterdir()):
        path = d / "status.json"
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(st, dict) or st.get("state") not in ("queued", "running"):
            continue
        if st.get("state") == "running" and _alive(st.get("pid")):
            continue
        try:
            final = _replay_kind(d / "replay.json") == "final"
            if not final and drop_partial_replay(d):
                st.pop("replay", None)
            for stage in (st.get("stages") or {}).values() if final else ():  # failed: the stage it stopped in shows
                if isinstance(stage, dict) and stage.get("state") in ("running", "pending"):
                    stage.update(state="skipped", note="interrupted")
            st["state"], st["error"] = ("done", None) if final else ("failed", INTERRUPTED)
            if st.get("started"):
                st.setdefault("seconds", round(max(0.0, path.stat().st_mtime - float(st["started"])), 1))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
            if replace_file(tmp, path):
                changed.append(d.name)
        except (OSError, ValueError, TypeError) as e:
            log(f"could not mark the interrupted job {d.name} ({plain_error(e)})")
    if changed:
        log(f"{len(changed)} job(s) were interrupted when the server last stopped: " + ", ".join(changed))
    return changed


class ReplayWriter:
    """Every write of a job's replay.json goes through here: one lock, a seq that only goes up (in the replay's
    "partial" key and in status.json), and once the final replay is written no partial one is ever written again."""

    def __init__(self, job_dir: Path, status: Status):
        self.job_dir, self.status = job_dir, status
        self.lock = threading.Lock()
        self.seq = 0
        self.final = False
        # A final replay.json from an earlier run in this job folder (python -m app.process on the same video): this
        # run does not stream, so that replay stays until this run's own final one replaces it (as before streaming),
        # also when this run fails or is stopped.
        self.had_final = _replay_kind(job_dir / "replay.json") == "final"

    def write_partial(self, replay: dict, ready_until: float) -> bool:
        """Write a partial replay (it gets its "partial" key here). False, and nothing written, after the final."""
        with self.lock:
            if self.final:
                return False
            self.seq += 1
            replay["partial"] = {"seq": self.seq, "ready_until": round(float(ready_until), 2)}
            _write_replay(self.job_dir, replay)
            self.status.replay(self.seq, True, ready_until)
            return True

    def write_final(self, replay: dict):
        with self.lock:
            self.final = True
            self.seq += 1
            _write_replay(self.job_dir, replay)
            self.status.replay(self.seq, False, None)

    def drop_partial(self):
        """After a failure: a partial replay.json is taken away (the job then reads as failed, as before streaming);
        a final one stays."""
        with self.lock:
            self.final = True
            if _replay_kind(self.job_dir / "replay.json") == "partial" and drop_partial_replay(self.job_dir):
                self.status.replay(None)


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


def _clock(seconds: float) -> str:
    t = int(round(seconds))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def hexcolor(bgr) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def run(video: str | Path, job_dir: Path | None = None, start: float | None = None, end: float | None = None,
        use_sam: bool = True, width: int = 800, log: Callable[[str], None] = print,
        on_status: Callable[[dict], None] | None = None, use_cache: bool = True,
        deck: str | Path | None = None, gestures: bool = True, name: str | None = None) -> Path:
    """Process one lecture. Returns the path of replay.json. gestures=False skips the teacher's pointing on a
    board (hand tracking and its Muse Spark calls, about half the API cost). name is the lecture's file name as the
    owner knows it (default: source_name)."""
    video = Path(video)
    info = media.probe(video)
    s0, s1 = media.clip_range(info, start, end)
    job_dir = Path(job_dir) if job_dir else JOBS / job_id_for(video, start, end)
    job_dir.mkdir(parents=True, exist_ok=True)
    if deck is not None:
        deck = Path(deck)
        if not deck.is_file():
            raise JobError(f"slide deck not found: {deck}")
        if deck.resolve() != (job_dir / "deck.pdf").resolve():
            shutil.copy2(deck, job_dir / "deck.pdf")
    status = Status(job_dir, on_status, name=(name or "").strip() or source_name(video, job_dir))
    writer = ReplayWriter(job_dir, status)
    try:
        out = _run(info, s0, s1, job_dir, status, use_sam, width, log, use_cache, gestures, writer=writer)
        shutil.rmtree(job_dir / STREAM_DIR, ignore_errors=True)  # the partial replays' pictures: the final has its own
        status.finish()
        return out
    except BaseException as e:  # report one line; keep the traceback in the job folder
        (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        writer.drop_partial()
        # Ctrl+C (or the process being stopped some other way) is not an error to report, but the job must not stay
        # "running" with a partial replay for good
        status.finish(plain_error(e) if isinstance(e, Exception) else INTERRUPTED)
        raise


def _run(info: media.VideoInfo, s0: float, s1: float, job_dir: Path, status: Status, use_sam: bool,
         width: int, log, use_cache: bool = True, gestures: bool = True, writer: "ReplayWriter | None" = None) -> Path:
    t_all = time.perf_counter()
    status.start("probe")
    if s1 - s0 < 1.0:
        raise JobError("the clip is shorter than 1 second")
    lecture = job_dir / ("lecture" + (info.path.suffix.lower() or ".mp4"))
    _link_or_copy(info.path, lecture)
    status.done("probe", f"{info.width}×{info.height}, {_clock(s1 - s0)} long" + ("" if info.has_audio else ", no sound"))
    log(f"video: {info.path.name}, {info.width}x{info.height}, {s1 - s0:.1f}s, audio={'yes' if info.has_audio else 'no'}")

    client = MetaClient(use_cache=use_cache)
    pool = ThreadPoolExecutor(max_workers=3)

    # Speech runs alongside everything else.
    speech_job = SpeechJob(info, s0, s1, client, status)

    status.start("classify")
    times = [s0 + (s1 - s0) * f for f in (0.15, 0.4, 0.65, 0.9)]
    frames = [media.grab_frame(info, t, 768) for t in times]
    kind = understand.classify(client, frames)
    status.done("classify", {"board": "a filmed " + ("chalkboard" if kind.get("board_colour") == "chalk" else "whiteboard"),
                             "slides": "a slide recording" + (" with a webcam" if kind.get("has_webcam_corner") else "")}
                .get(kind.get("type"), "not sure: treated as a board"))
    log(f"classify: {kind}")
    if kind.get("type") == "slides":
        return _run_slides(info, s0, s1, job_dir, status, client, kind, speech_job, lecture, pool, t_all, log,
                           writer=writer)
    stop = threading.Event()  # set when the board path fails: the hand tracking then sends no more SAM calls
    try:
        return _run_board(info, s0, s1, job_dir, status, client, kind, speech_job, lecture, pool, t_all, log,
                          use_sam, width, gestures, stop, writer=writer)
    except BaseException:
        stop.set()
        raise


def _run_board(info, s0, s1, job_dir: Path, status: Status, client: MetaClient, kind: dict, speech_job, lecture: Path,
               pool, t_all: float, log, use_sam: bool, width: int, gestures: bool, stop: threading.Event,
               writer: "ReplayWriter | None" = None) -> Path:
    writer = writer or ReplayWriter(job_dir, status)
    streams: list = []  # the partial-replay writer, once there is one: stopped whatever happens
    try:
        return _board_job(info, s0, s1, job_dir, status, client, kind, speech_job, lecture, pool, t_all, log,
                          use_sam, width, gestures, stop, writer, streams)
    finally:
        for sw in streams:
            sw.stop()


def _board_job(info, s0, s1, job_dir: Path, status: Status, client: MetaClient, kind: dict, speech_job, lecture: Path,
               pool, t_all: float, log, use_sam: bool, width: int, gestures: bool, stop: threading.Event,
               writer: "ReplayWriter", streams: list) -> Path:
    status.start("board")
    corners, board_info = None, None
    if use_sam:
        from .core.sam import find_board
        board_info = find_board(client, info, s0, s1, colour=kind.get("board_colour"))
        corners = board_info["corners"] if board_info else None
    status.done("board", f"found the {board_info['prompt']} (confidence {board_info['score']:.2f})" if board_info
                else "no board found: using the whole picture")
    log(f"board: {board_info or 'not found, using the whole frame'}")

    person = None
    hands_job = None
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
                status.done("people", f"teacher tracked in {len(person.frames)} of {person.n} moments")

        people_done.add_done_callback(people_finished)
        # The teacher's hands, for pointing gestures (SAM 3.1 video mode, in the background too). They are needed
        # only after reading, so they start once the person chunks are back: their video copies would otherwise
        # take the CPU from the person chunks that capture is waiting for.
        from .core.sam import track_hands

        def hands_after_people():
            wait([people_done])
            if stop.is_set():
                return None
            return track_hands(client, info, s0, s1, workdir=job_dir / "tmp", stop=stop)

        hands_job = pool.submit(hands_after_people) if gestures else None
    else:
        status.skip("people", "SAM off")

    status.start("capture")
    mode = "chalk" if kind.get("board_colour") == "chalk" else ("white" if kind.get("board_colour") == "white" else None)
    stream = None
    if writer.had_final:
        log("stream: no partial replays (the replay of an earlier run stays until this one is done)")
    elif load_settings().stream_replay != "off":
        # The player opens on this first partial replay (no writing yet) and the board fills in as capture goes.
        mode, stream = _start_stream(info, s0, s1, job_dir, status, client, kind, speech_job, lecture, t_all, log,
                                     corners, person, width, mode, writer)
        if stream is not None:
            streams.append(stream)
    speech_job.cpu_free.clear()
    try:
        board = capture_board(info, corners=corners, person_mask=person, params=BoardParams(width=width),
                              start=s0, end=s1, progress=lambda f: status.progress("capture", f), mode=mode,
                              checkpoint=None if stream is None else
                              (lambda b, t, ready: stream.submit(Snapshot(b, t, ready))))
    finally:
        speech_job.cpu_free.set()
    if stream is not None:
        # The whole board, while reading, gestures and linking run (its own copy of the pieces: they are filled in
        # below). Its pieces are read by the read stage, not again here, so partial reads not started are dropped.
        stream.end_reads()
        stream.submit(Snapshot(dataclasses.replace(board, pieces=[dict(p) for p in board.pieces]), s1, s1,
                               read_new=False))
    pieces = board.pieces
    hidden = sum(1 for p in pieces if p["revealed"])
    status.done("capture", f"{len(pieces)} pieces of writing" + (f", {hidden} written while hidden" if hidden else "")
                + (f", {len(board.erases)} erased" if board.erases else ""))
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
        p.update(board.redraw_part(p["id"]) or {})  # written while hidden: what may be redrawn early, and how fast

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
    # Lines: pieces written on one line in bursts are read (and typeset) together.
    from .core.lines import group_lines
    view = [{**p, "box": p["image_box"]} for p in pieces]
    lines = group_lines(view, reads)
    if lines:
        pseudo = [{"id": 100000 + i, "members": ln["pieces"], "box": ln["box"], "t_shown": ln["t_complete"],
                   "removed_at": ln["removed_at"], "pre_existing": ln["pre_existing"]} for i, ln in enumerate(lines)]
        try:
            line_crops = understand.hires_crops(info, board.corners, board, pseudo, person, t_end=s1)
        except Exception:
            line_crops = {}
        line_reads = understand.read_pieces(client, board, everything, pseudo, subject=kind.get("subject", ""),
                                            crops=line_crops)
        for i, ln in enumerate(lines):
            ln["read"] = line_reads.get(100000 + i, {"kind": "other", "text": "", "latex": ""})
    status.done("read", f"{len(reads)} pieces read" + (f", {len(lines)} lines" if lines else ""))
    if stream is not None:
        stream.use_reads(reads)  # the partial replay gets the readings while gestures and linking run

    annotations = []
    if hands_job is not None:
        annotations = _gestures(hands_job, lambda hands: _board_gestures(client, info, board, pieces, reads, person,
                                                                         hands, speech_job, status, log),
                                status, job_dir, log)
    else:
        status.skip("gestures", "SAM off" if not use_sam else "turned off")

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
        status.done("link", f"{len(links['elements'])} matched to the speech, {len(links['chapters'])} chapters")
        status.start("write")
        out_pieces = []
        for p in pieces:
            r = reads.get(p["id"], {"kind": "other", "text": "", "latex": ""})
            lk = links["elements"].get(str(p["id"]), {})
            spoken = understand.snap_to_words(lk.get("spoken", []), words)
            redraw = redraw_time(p, lk.get("redraw_at"), spoken)
            out_pieces.append({
                "id": p["id"], "box": p["image_box"], "image": p["image"], "order": p["order"], "gone": p["gone"],
                "draw": p["draw"],
                "t_seen": p["t_seen"], "t_shown": p["t_shown"], "written_after": p["written_after"],
                "revealed": p["revealed"], "removed_at": p["removed_at"], "pre_existing": p["pre_existing"],
                "read": r, "spoken": spoken, "redraw_at": redraw,
                "redraw_to": p.get("redraw_to"), "redraw_secs": p.get("redraw_secs"),
            })
        replay = {
            "version": 1,
            "video": {"file": lecture.name, "duration": round(info.duration, 3), "width": info.width,
                      "height": info.height, "start": round(s0, 3), "end": round(s1, 3),
                      "source": status.name or info.path.name},
            "type": "board",
            "subject": kind.get("subject", ""),
            "board": {
                "size": [board.width, board.height], "mode": board.mode, "corners": board.corners,
                "background": hexcolor(board.background), "pieces": out_pieces, "erases": board.erases,
                "lines": lines, "annotations": annotations,
                "t_start": round(board.t_start, 2),
            },
            "transcript": {"turns": turns, "words": words, "timing": how},
            "chapters": links["chapters"],
            "stats": {"seconds": round(time.perf_counter() - t_all, 1), "api": client.summary(),
                      "capture": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in board.stats.items()}},
        }
        if stream is not None and not writer.final:
            stream.stop()  # waits for a partial write in progress; none comes after this
            if stream.written:
                log(f"stream: {len(stream.written)} partial replays written, the first after "
                    f"{stream.written[0]['at'] - status.data['started']:.1f}s")
        writer.write_final(replay)
        status.done("write", "word times refining" if how == "proportional" else "")
        log(f"wrote replay.json ({how} word times) after {replay['stats']['seconds']}s, API {client.summary()}")

    return _finish_with_speech(speech_job, link_and_write, job_dir, pool, log)


def _partial_transcript(speech_job) -> dict:
    """The transcript for a partial replay: all of it, with proportional word times, once transcription is done
    (never waited for); else none yet."""
    fut = getattr(speech_job, "transcribed", None)
    if fut is None or not fut.done() or fut.exception() is not None:
        return {"turns": [], "words": [], "timing": "proportional"}
    _, turns = fut.result()
    if not turns:
        return {"turns": [], "words": [], "timing": "none"}
    return {"turns": turns, "words": speech_job.proportional(), "timing": "proportional"}


def _start_stream(info, s0, s1, job_dir: Path, status: Status, client, kind: dict, speech_job, lecture: Path,
                  t_all: float, log, corners, person, width: int, mode: str | None, writer: ReplayWriter):
    """The partial-replay writer of a board lecture, with its first partial (no writing yet) on its way. Returns
    (mode, writer): the board mode found here when classify gave none (capture then finds the same one), so the
    first partial already has the final size, corners and mode. (mode, None) if streaming could not start."""
    try:
        params = BoardParams(width=width)
        geom = BoardGeometry(info, corners, params.width, params.decode_width)
        if mode is None:
            mode = detect_mode(info, geom, s0, s1, params)[0]
        size = [geom.width, geom.height]
        board_corners = [[round(float(x), 1), round(float(y), 1)] for x, y in geom.corners_video]
        first_bg = board_colour(info, geom, s0, mode)
        shutil.rmtree(job_dir / "stream", ignore_errors=True)  # a run before this one's pictures
    except Exception as e:
        log(f"stream: no partial replays ({plain_error(e)})")
        return mode, None
    video = {"file": lecture.name, "duration": round(info.duration, 3), "width": info.width, "height": info.height,
             "start": round(s0, 3), "end": round(s1, 3), "source": status.name or info.path.name}
    subject = kind.get("subject", "")
    shown: dict = {}  # every piece a partial showed, as last written: it stays on the board (stream.partial_pieces)

    def build(snap: Snapshot, reads: dict) -> tuple[dict, float]:
        b, ready = snap.board, float(snap.ready_until)
        if b is None:
            pieces, erases, bg, t_start = [], [], first_bg, s0
        else:
            pieces = partial_pieces(job_dir, b, ready, reads, shown)
            erases = [e for e in b.erases if e["t"] <= ready]
            bg, t_start = b.background, b.t_start
        replay = {
            "version": 1, "video": dict(video), "type": "board", "subject": subject,
            "board": {"size": list(size), "mode": mode, "corners": board_corners, "background": hexcolor(bg),
                      "pieces": pieces, "erases": erases, "lines": [], "annotations": [],
                      "t_start": round(t_start, 2)},
            "transcript": _partial_transcript(speech_job),
            "chapters": [],
            "stats": {"seconds": round(time.perf_counter() - t_all, 1), "api": client.summary(),
                      "capture": {"t": round(snap.t, 2), "pieces": len(pieces)}},
        }
        return replay, ready

    def read(snap: Snapshot, batch: list[dict]) -> dict:
        b = snap.board
        try:
            crops = understand.hires_crops(info, b.corners, b, batch, person, t_end=snap.t, workers=1)
        except Exception:
            crops = {}
        return understand.read_pieces(client, b, _board_all(b), batch, subject=subject, crops=crops, workers=1)

    stream = StreamWriter(writer.write_partial, build, read, log=log)
    stream.submit(Snapshot(None, s0, s0, read_new=False))
    return mode, stream


def redraw_time(piece: dict, redraw_at, spoken: list[dict]) -> float | None:
    """When the board redraws writing done while hidden (piece["revealed"]): when the link says it was written, else
    its first mention before the camera showed it; never before any of its hidden ink's spots was last seen blank
    (redraw_after, else written_after): talked about first and written later, it is drawn once it can exist."""
    if not piece.get("revealed"):
        return None
    t = redraw_at if isinstance(redraw_at, (int, float)) and redraw_at >= 0 else None
    if t is None and spoken and spoken[0]["start"] < piece["t_seen"]:
        t = spoken[0]["start"]
    if t is None:
        return None
    floor = piece.get("redraw_after")
    floor = piece.get("written_after") if floor is None else floor
    return round(max(float(t), float(floor)), 2) if floor is not None else round(float(t), 2)


def _gestures(hands_job, find, status: Status, job_dir: Path, log) -> list[dict]:
    """The optional pointing stage on a board: whatever goes wrong, the lecture carries on without it.
    find(hands) -> (annotations, gestures not judged, gestures not narrowed)."""
    status.start("gestures", "pointing at the board")
    try:
        hands = hands_job.result()
        if hands is None or (hands.failed and hands.failed >= hands.n_chunks):
            log(f"gestures: skipped, hand tracking failed ({hands.error if hands is not None else 'stopped'})")
            status.skip("gestures", "skipped: hand tracking failed in every SAM chunk")
            return []
        annotations, unjudged, whole = find(hands)
    except Exception as e:  # an optional stage never fails the lecture
        try:
            (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        log(f"gestures: skipped ({plain_error(e)})")
        status.skip("gestures", f"skipped: {plain_error(e)}")
        return []
    note = f"{len(annotations)} pointing gestures redrawn"
    if unjudged:
        note += f", {unjudged} could not be judged"
    if whole:
        note += f", {whole} shown whole (the part meant is unknown)"
    if hands.failed:
        note += f"; hands not tracked in {hands.failed} of {hands.n_chunks} SAM chunks"
    status.done("gestures", note)
    return annotations


def _board_gestures(client, info, board, pieces, reads, person, hands, speech_job, status, log):
    """Pointing gestures on a filmed board: code finds them, Muse Spark decides what each one means.

    Returns (annotations, gestures not judged, gestures not narrowed): a failed call loses only its own gesture.
    """
    from .core import board_gestures as bg
    from .core.board import BoardGeometry

    s = board.width / 800  # pixel sizes are set for an 800 px board
    geom = BoardGeometry(info, board.corners, board.width, 1280)
    episodes = bg.episodes(bg.pointing_samples(hands, person, board, geom, info), hands.fps, scale=s)
    if not episodes:
        return [], 0, 0
    # Proportional times, not whatever alignment has reached by now: the same prompts (and cache hits) every run.
    words = speech_job.proportional()
    by_id = {p["id"]: p for p in pieces}

    def elements_at(t):
        out = []
        for p in pieces:
            if p["pre_existing"] or p["t_seen"] - 0.5 <= t:
                if p["removed_at"] is not None and p["removed_at"] <= t:
                    continue
                r = reads.get(p["id"], {})
                out.append({"id": str(p["id"]), "kind": r.get("kind", ""), "text": r.get("text", ""),
                            "box": p["image_box"]})
        return out

    # A piece much taller than a line of writing is a block (often everything already on the board when the
    # recording starts): a gesture on it may be about one line or term only.
    heights = sorted(p["image_box"][3] for p in pieces if reads.get(p["id"], {}).get("kind") in bg.NARROW_KINDS)
    line_h = heights[len(heights) // 2] if heights else 18 * s

    def own_ink(pid, t):
        ink = board.ink_at(t).reshape(-1)
        own = np.zeros_like(ink)
        own[board.add_idx[(board.add_piece == pid) & (board.add_t <= t)]] = True
        return (ink & own).reshape(board.height, board.width)

    def part_of(g, m, pid, said):
        """Box of the part of a big block the gesture was about, or None for the whole target."""
        p = by_id[int(pid)]
        r = reads.get(p["id"], {})
        if r.get("kind") not in bg.NARROW_KINDS or p["image_box"][3] <= 2.2 * line_h:
            return None
        img = board.image_at(g["t_end"])
        region = bg.zoom_region(p["image_box"], g["path"], board.width, board.height)
        part = understand.board_gesture_part(client, img, region, g["path"], m, r.get("text", ""), said)
        return bg.tighten(own_ink(p["id"], g["t_end"]), part, scale=s) if part else None

    def one(g):
        """(gesture, type, target ids, meaning, part box, what failed)"""
        els = elements_at(g["t_end"])
        said = _words_between(words, g["t_start"] - 1.5, g["t_end"] + 1.5)
        try:
            t = (g["t_start"] + g["t_end"]) / 2
            frame = media.grab_frame(info, t, 960)
            k = frame.shape[1] / info.width
            fpath = [[float(x) * k, float(y) * k] for x, y in geom.board_to_video(g["path"])]
            m = understand.board_gesture_meaning(client, frame, board.image_at(g["t_end"]), g, els, said, fpath)
        except Exception as e:  # loses this gesture only
            log(f"gestures: the one at {g['t_start']:.1f}s could not be judged ({e})")
            return g, "none", [], {}, None, "meaning"
        kind, target = bg.clean_target(m.get("type") or "none", m.get("target"), [e["id"] for e in els])
        if kind == "arrow":  # Spark's two ends must fit where the hand went
            kind, target = bg.check_arrow(g["path"], target, [by_id[int(x)]["image_box"] for x in target], s)
        part = None
        if kind not in ("none", "arrow") and len(target) == 1:
            try:
                part = part_of(g, {**m, "type": kind}, target[0], said)
            except Exception as e:  # the whole target, then
                log(f"gestures: the part meant at {g['t_start']:.1f}s is unknown ({e})")
                return g, kind, target, m, None, "part"
        return g, kind, target, m, part, None

    out, unjudged, whole = [], 0, 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, (g, kind, target, m, part, failed) in enumerate(ex.map(one, episodes)):
            status.progress("gestures", (k + 1) / len(episodes))
            unjudged += failed == "meaning"
            whole += failed == "part"
            if kind == "none" or not target:
                continue
            boxes = [by_id[int(x)]["image_box"] for x in target]
            box, fr, to = boxes[0], None, None
            if kind == "arrow":
                a, b = boxes
                fr = [a[0] + a[2] / 2, a[1] + a[3] / 2]
                to = [b[0] + b[2] / 2, b[1] + b[3] / 2]
            elif part:  # a big block: just the part the teacher meant
                box = part
            elif len(boxes) > 1:  # several targets: one box round them all
                box = bg.union_box(boxes)
            out.append({"type": kind, "tool": "hand", "t_start": g["t_start"], "t_end": g["t_end"], "target": target,
                        "box": [round(float(v), 1) for v in box], "from": fr, "to": to, "why": m.get("why", ""),
                        "guess": g["type"], "path": g["path"]})
    log(f"gestures: {len(episodes)} candidates, {len(out)} kept by Muse Spark"
        + (f", {unjudged} not judged" if unjudged else ""))
    return out, unjudged, whole


def _words_between(words: list[dict], a: float, b: float) -> str:
    return " ".join(w["w"] for w in words if a <= w["s"] <= b)


def _run_slides(info, s0, s1, job_dir: Path, status: Status, client: MetaClient, kind: dict, speech_job,
                lecture: Path, pool, t_all: float, log, writer: "ReplayWriter | None" = None) -> Path:
    from .core.slides import analyse_slides

    writer = writer or ReplayWriter(job_dir, status)

    status.skip("board", "slides")
    status.skip("people", "slides")
    status.skip("gestures", "slides: the pointer's gestures are read with the slides")
    status.start("capture", "slide analysis")
    speech_job.cpu_free.clear()
    try:
        res = analyse_slides(info, job_dir / "tmp", s0, s1, progress=lambda f: status.progress("capture", f))
    finally:
        speech_job.cpu_free.set()
    status.done("capture", f"{sum(1 for st in res.states if st.kind == 'slide')} slides, {len(res.states)} steps, {len(res.gestures)} pointer moves, "
                           f"{len(res.live)} live region(s), {res.stats['speed']:.1f}x real time")
    log(f"slides: {len(res.states)} states, {len(res.gestures)} gestures, live {res.live}, {res.stats['speed']:.1f}x")

    (job_dir / "slides").mkdir(exist_ok=True)
    status.start("read")
    readable = [st for st in res.states if st.image is not None and not st.blank]

    # Optional PDF deck: read the crisp page; show it where it matches the screen exactly.
    deck_pages, matches, video_img = [], {}, {}
    if (job_dir / "deck.pdf").exists():
        from .core.deck import match_pages, render_pages, same_content
        deck_pages = render_pages(job_dir / "deck.pdf", width=1920)
        matches = match_pages(res.states, deck_pages, res.live)
        for st in readable:
            video_img[st.id] = st.image
            if st.id in matches:
                H, W = st.image.shape[:2]
                page = cv2.resize(deck_pages[matches[st.id][0]], (W, H), interpolation=cv2.INTER_AREA)
                if same_content(st.image, page, res.background, res.live):
                    st.image = page
        log(f"deck: {len(deck_pages)} pages, {len(matches)}/{len(readable)} slide states matched")
    for st in res.states:
        if st.image is not None:
            media.imwrite(job_dir / "slides" / f"{st.id:03d}.png", st.image)

    from .core.slides import content_mask, snap_box, word_at

    def read_one(st):
        H, W = st.image.shape[:2]
        if st.id in matches:
            from .core.deck import visible_on
            page_hi = deck_pages[matches[st.id][0]]
            els = understand.read_slide(client, page_hi, kind.get("subject", ""))
            kx, ky = W / page_hi.shape[1], H / page_hi.shape[0]
            page = cv2.resize(page_hi, (W, H), interpolation=cv2.INTER_AREA)
            for e in els:
                e["box"] = [round(e["box"][0] * kx), round(e["box"][1] * ky), max(1, round(e["box"][2] * kx)),
                            max(1, round(e["box"][3] * ky))]
            # A build step: keep only what is already on screen.
            els = [e for e in els if visible_on(video_img[st.id], e["box"], res.background, res.live, page)]
        else:
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
            target = list(dict.fromkeys(x for x in m.get("target", []) if x in ids))
            if m["type"] == "arrow":
                target = target[:2]  # from, to: the two ends the player draws between
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
    status.done("read", f"{sum(len(v) for v in elements.values())} things on the slides, {len(annotations)} gestures redrawn")

    link_els = [{"id": e["id"], "kind": e["kind"], "text": e["text"], "t_start": round(st.t_start, 2),
                 "t_end": round(st.t_end, 2), "slide": st.id} for st in readable for e in elements.get(st.id, [])]

    def link_and_write(turns, words, how):
        status.start("link")
        links = understand.link(client, link_els, words, s1, kind="slides") if words else {"elements": {}, "chapters": []}
        status.done("link", f"{len(links['elements'])} matched to the speech, {len(links['chapters'])} chapters")
        status.start("write")
        out_states = []
        for st in res.states:
            els = []
            for e in elements.get(st.id, []):
                lk = links["elements"].get(e["id"], {})
                els.append({**e, "spoken": understand.snap_to_words(lk.get("spoken", []), words)})
            out_states.append({"id": st.id, "kind": st.kind, "t_start": round(st.t_start, 2), "t_end": round(st.t_end, 2),
                               "t_change": round(st.t_change, 2), "blank": st.blank,
                           "deck_page": (matches[st.id][0] + 1) if st.id in matches else None,
                               "image": f"slides/{st.id:03d}.png" if st.image is not None else None, "elements": els})
        replay = {
            "version": 1,
            "video": {"file": lecture.name, "duration": round(info.duration, 3), "width": info.width,
                      "height": info.height, "start": round(s0, 3), "end": round(s1, 3),
                      "source": status.name or info.path.name},
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
        writer.write_final(replay)
        status.done("write", "word times refining" if how == "proportional" else "")
        log(f"wrote replay.json ({how} word times) after {replay['stats']['seconds']}s, API {client.summary()}")

    return _finish_with_speech(speech_job, link_and_write, job_dir, pool, log)


def _board_all(board) -> np.ndarray:
    """Every kept mark ever written, drawn on the board colour (for reading, incl. erased pieces)."""
    img = np.empty((board.height * board.width, 3), np.uint8)
    img[:] = board.background
    img[board.add_idx] = board.add_color
    return img.reshape(board.height, board.width, 3)
