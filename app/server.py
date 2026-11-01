"""FastAPI server: upload a lecture, watch the stages, then load replay.json and the media.

    .venv\\Scripts\\python -m app.server            # http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import time
import queue
import re
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import ROOT
from .pipeline import JOBS, run

app = FastAPI(title="Replay Board")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_work: queue.Queue = queue.Queue()
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
NO_CACHE = {"Cache-Control": "no-cache"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def _worker():
    while True:
        job_dir, video, start, end = _work.get()
        try:
            run(video, job_dir, start, end, log=lambda m: None)
        except Exception:
            pass  # status.json and error.log already say what went wrong
        finally:
            _work.task_done()


threading.Thread(target=_worker, daemon=True).start()


def _job_dir(job_id: str) -> Path:
    if not _ID.match(job_id):
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
def job_popout(job_id: str, piece_id: int):
    """3D scene for one drawn piece (Muse Spark), made on demand and cached in the job folder."""
    from .core.meta_api import MetaAPIError, MetaClient
    from .popout import make_scene

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


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m app.server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"Replay Board on http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
