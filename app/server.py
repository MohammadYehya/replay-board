"""FastAPI server: upload a lecture, watch the stages, then load replay.json and the media.

    .venv\\Scripts\\python -m app.server            # http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import logging
import time
import queue
import re
import threading
import traceback
import uuid
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import ask
from .config import ROOT
from .pipeline import JOBS, STAGES, plain_error, replace_file, run

app = FastAPI(title="Replay Board")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_work: queue.Queue = queue.Queue()
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
NO_CACHE = {"Cache-Control": "no-cache"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def _process(job_dir: Path, video: Path, start, end):
    """One queued job. The pipeline reports its own failures, except those before it starts its status (a bad
    file, a start past the end): those are written here, or the job would stay 'queued' for good."""
    try:
        run(video, job_dir, start, end, log=lambda m: None)
    except Exception as e:
        path = job_dir / "status.json"
        st = _read_json(path) if path.exists() else {}
        if st.get("state") not in (None, "queued", "running"):
            return  # status.json and error.log already say what went wrong
        (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        started = st.get("started") or time.time()
        st.update(state="failed", error=plain_error(e), started=started, seconds=round(time.time() - started, 1))
        if not st.get("stages"):
            st["stages"] = {s: {"state": "pending", "progress": 0.0, "seconds": None, "note": ""} for s in STAGES}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
        replace_file(tmp, path)


def _worker():
    while True:
        job = _work.get()
        try:
            _process(*job)
        except Exception:
            pass  # never stop the worker
        finally:
            _work.task_done()


threading.Thread(target=_worker, daemon=True).start()


def _job_dir(job_id: str) -> Path:
    if not _ID.fullmatch(job_id):  # match() with "$" would let a trailing newline through
        raise HTTPException(404, "no such job")
    d = JOBS / job_id
    if not d.is_dir():
        raise HTTPException(404, "no such job")
    return d


def _duration(seconds) -> str:
    if not seconds:
        return ""
    t = int(round(float(seconds)))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _read_json(p: Path, tries: int = 20) -> dict:
    # The pipeline replaces these files atomically, but Windows can refuse a read mid-replace.
    for i in range(tries):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if not p.exists():
                return {}
            time.sleep(0.05)
    return {}


@app.get("/api/jobs")
def list_jobs():
    out = []
    if JOBS.is_dir():
        for d in sorted(JOBS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            st = _read_json(d / "status.json")
            rp = d / "replay.json"
            sm = _read_json(d / "summary.json", tries=1) if (d / "summary.json").exists() else {}
            subject = (sm.get("subject") or "").strip()
            source = sm.get("source") or st.get("name") or ""
            detail = " · ".join(x for x in (
                {"board": "Board lecture", "slides": "Slide lecture"}.get(sm.get("type", ""), ""),
                _duration(sm.get("seconds")), source) if x)
            out.append({"id": d.name, "state": st.get("state", "unknown"), "error": st.get("error"),
                        "ready": rp.exists(), "name": (subject[:1].upper() + subject[1:]) if subject else (source or d.name),
                        "detail": detail})
    return out


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), start: float | None = Form(None), end: float | None = Form(None),
                     deck: UploadFile | None = File(None)):
    name = Path(file.filename or "lecture.mp4").name
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"please upload a video file ({', '.join(sorted(VIDEO_EXT))})")
    stem = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")[:32] or "lecture"
    job_id = f"{stem}-{uuid.uuid4().hex[:8]}"
    d = JOBS / job_id
    d.mkdir(parents=True)
    upload = d / f"upload{ext}"
    with open(upload, "wb") as f:
        while chunk := await file.read(8 << 20):
            f.write(chunk)
    if deck is not None and deck.filename:
        (d / "deck.pdf").write_bytes(await deck.read())
    (d / "status.json").write_text(json.dumps({"state": "queued", "error": None, "name": name, "stages": {}}),
                                   encoding="utf-8")
    _work.put((d, upload, start, end))
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    d = _job_dir(job_id)
    st = _read_json(d / "status.json")
    st["id"] = job_id
    st["ready"] = (d / "replay.json").exists()
    return st


@app.get("/api/jobs/{job_id}/replay.json")
def job_replay(job_id: str):
    p = _job_dir(job_id) / "replay.json"
    if not p.exists():
        raise HTTPException(404, "not processed yet")
    data = _read_json(p)
    if not data:
        raise HTTPException(503, "replay.json is being written; try again")
    return JSONResponse(data, headers=NO_CACHE)


@app.get("/api/jobs/{job_id}/notes.md")
def job_notes(job_id: str):
    from fastapi.responses import PlainTextResponse

    from .notes import build_notes

    p = _job_dir(job_id) / "replay.json"
    if not p.exists():
        raise HTTPException(404, "not processed yet")
    md = build_notes(_read_json(p))
    return PlainTextResponse(md, media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition": f'inline; filename="{job_id}-notes.md"'})


@app.post("/api/jobs/{job_id}/popout/{piece_id}")
def job_popout(job_id: str, piece_id: int, request: Request):
    """3D scene for one drawn piece (Muse Spark), made on demand and cached in the job folder."""
    from .core.meta_api import MetaAPIError, MetaClient
    from .popout import make_scene

    if _cross_site(request):  # like /ask: other websites must not spend the owner's API credit
        raise HTTPException(403, "3D pop-outs are only made for the Replay Board player itself")
    d = _job_dir(job_id)
    replay = _read_json(d / "replay.json")
    if replay.get("type") != "board":
        raise HTTPException(404, "3D pop-out is for board lectures")
    try:
        return make_scene(MetaClient(), d, replay, piece_id)
    except KeyError:
        raise HTTPException(404, "no such piece") from None
    except MetaAPIError as e:
        raise HTTPException(502, str(e)) from None


@app.get("/api/jobs/{job_id}/notes.json")
def job_notes_json(job_id: str):
    from .notes import notes_data

    p = _job_dir(job_id) / "replay.json"
    if not p.exists():
        raise HTTPException(404, "not processed yet")
    return JSONResponse(notes_data(_read_json(p)), headers=NO_CACHE)


def _meta_client():
    """The client for Ask Muse (tests replace this with a fake)."""
    from .core.meta_api import MetaClient

    return MetaClient()


def _ask_prepare(d: Path, body) -> ask.Prepared:
    p = d / "replay.json"
    if not p.exists():
        raise HTTPException(409, "this lecture is not processed yet, so there is nothing to ask about")
    replay = _read_json(p)
    if not replay:
        raise HTTPException(409, "the lecture is being updated; try again in a moment")
    try:
        return ask.prepare(d, replay, body)
    except ask.AskError as e:
        raise HTTPException(400, str(e)) from None
    except Exception as e:  # e.g. an odd replay.json: one plain line for the player, the traceback in a log
        try:
            (d / "ask-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        raise HTTPException(500, ask.plain(e)) from None


MAX_ASK_BODY = 1 << 20  # a question, 12 attached items and 6 earlier turns fit many times over


# Hosts of local or LAN pages: a bare machine name (localhost, my-pc), a local suffix, a loopback or private address.
_LOCAL = re.compile(r"[^.:]+|::1|.+\.(localhost|local|test|lan|internal|home\.arpa)|127(\.\d{1,3}){3}"
                    r"|10(\.\d{1,3}){3}|192\.168(\.\d{1,3}){2}|172\.(1[6-9]|2\d|3[01])(\.\d{1,3}){2}")


def _cross_site(request: Request) -> bool:
    """True for a request a page on some other website made the browser send. The CORS policy lets every origin
    in, and a text/plain POST is not even preflighted, so without this any page the owner visits could ask
    Muse Spark questions on their API credit. Allowed: no Origin (curl, tests), this server's own host, and
    local or LAN pages (the Vite dev server proxies with its own Origin, e.g. http://localhost:5173)."""
    origin = request.headers.get("origin")
    if origin is None:
        return False
    try:
        host = (urlsplit(origin).hostname or "").lower()
        own = (urlsplit("//" + request.headers.get("host", "")).hostname or "").lower()
    except ValueError:
        return True
    if not host:  # "null": a sandboxed frame or a file
        return True
    return host != own and not _LOCAL.fullmatch(host)


async def _read_body(request: Request, limit: int) -> bytes:
    """The request body, refused (400) past `limit` bytes without reading the rest."""
    too_big = HTTPException(400, f"the request is too big (at most {limit // (1 << 20)} MB)")
    n = request.headers.get("content-length", "")
    if n.isdigit() and int(n) > limit:
        raise too_big
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_big
        chunks.append(chunk)
    return b"".join(chunks)


class EventStream(StreamingResponse):
    """A StreamingResponse over a sync generator that is closed as soon as the response ends. When the student
    presses Stop, Starlette cancels the stream but leaves the generator to the garbage collector, which can take
    minutes: until then the answer's connection to Meta stays open and the stopped call is not logged. Closing
    it here runs its `finally` at once (the model's stream is closed and the call logged as stopped)."""

    def __init__(self, events: Iterator[str], **kw):
        super().__init__(events, **kw)  # a sync iterator: Starlette reads it in a worker thread
        self._events = events

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Shielded: this also runs while the request is being cancelled. Starlette waits for a worker thread's
            # next() to return before it cancels, so the generator is not running here.
            with anyio.CancelScope(shield=True):
                try:
                    await anyio.to_thread.run_sync(self._events.close)
                except (ValueError, RuntimeError):
                    pass  # still running somewhere, or it misbehaved on close: the collector closes it later


@app.post("/api/jobs/{job_id}/ask")
async def job_ask(job_id: str, request: Request):
    """Ask Muse: one question about the lecture, answered as server-sent events (delta..., then done or
    error). A bad request is a 400 with one plain line, before any streaming."""
    if _cross_site(request):
        raise HTTPException(403, "Ask Muse only answers the Replay Board player itself")
    d = _job_dir(job_id)
    raw = await _read_body(request, MAX_ASK_BODY)
    try:
        body = json.loads(raw or b"null")
    except (ValueError, RecursionError):
        raise HTTPException(400, "the request is not valid JSON") from None
    prep = await run_in_threadpool(_ask_prepare, d, body)
    # A sync generator: Starlette runs it in a worker thread, so the model call never blocks the loop.
    return EventStream(ask.answer_events(prep, _meta_client), media_type="text/event-stream",
                       headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/chat")
def job_chat(job_id: str):
    return JSONResponse(ask.load_chat(_job_dir(job_id)), headers=NO_CACHE)


@app.delete("/api/jobs/{job_id}/chat")
def job_chat_clear(job_id: str, request: Request):
    if _cross_site(request):
        raise HTTPException(403, "only the Replay Board player can clear the conversation")
    try:
        ask.clear_chat(_job_dir(job_id))
    except OSError as e:
        raise HTTPException(503, ask.plain(e)) from None
    return {"turns": []}


@app.get("/api/jobs/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str):
    d = _job_dir(job_id).resolve()
    p = (d / path).resolve()
    if d not in p.parents or not p.is_file():
        raise HTTPException(404, "no such file")
    # Revalidate: a re-processed job rewrites its piece images under the same names.
    return FileResponse(p, headers=NO_CACHE)  # supports Range requests, so the video can seek


# The built player (web/dist), if present, at /
_dist = ROOT / "web" / "dist"


@app.get("/{path:path}", include_in_schema=False)
def player(path: str):
    if not _dist.is_dir():
        return JSONResponse({"message": "player not built: run `npm run build` in web/, or use `npm run dev`"})
    p = (_dist / path).resolve()
    if path and _dist.resolve() in p.parents and p.is_file():
        # Vite's assets carry a content hash in their name: cache them for good.
        hashed = path.startswith("assets/")
        return FileResponse(p, headers={"Cache-Control": "public, max-age=31536000, immutable"} if hashed else NO_CACHE)
    return FileResponse(_dist / "index.html", headers=NO_CACHE)


def quiet_dropped_connections(record: logging.LogRecord) -> bool:
    """Log filter: on Windows, asyncio logs a traceback when the browser drops a connection (a student stopping an
    answer, closing the tab). It is harmless, and the console should show only plain lines."""
    exc = record.exc_info[1] if record.exc_info else None
    return not isinstance(exc, (ConnectionResetError, ConnectionAbortedError))


def main():
    import argparse

    import uvicorn

    logging.getLogger("asyncio").addFilter(quiet_dropped_connections)
    ap = argparse.ArgumentParser(prog="python -m app.server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"Replay Board on http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
