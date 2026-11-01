"""FastAPI server: upload a lecture, watch the stages, then load replay.json and the media.

    .venv\\Scripts\\python -m app.server            # http://127.0.0.1:8000
"""
from __future__ import annotations

import json
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


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
            out.append({"id": d.name, "state": st.get("state", "unknown"), "error": st.get("error"),
                        "ready": rp.exists(), "name": (st.get("name") or d.name)})
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
    return JSONResponse(_read_json(p))


@app.get("/api/jobs/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str):
    d = _job_dir(job_id).resolve()
    p = (d / path).resolve()
    if d not in p.parents or not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p)  # supports Range requests, so the video can seek


# The built player (web/dist), if present, at /
_dist = ROOT / "web" / "dist"


@app.get("/{path:path}", include_in_schema=False)
def player(path: str):
    if not _dist.is_dir():
        return JSONResponse({"message": "player not built: run `npm run build` in web/, or use `npm run dev`"})
    p = (_dist / path).resolve()
    if path and _dist.resolve() in p.parents and p.is_file():
        return FileResponse(p)
    return FileResponse(_dist / "index.html")


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
