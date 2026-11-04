"""A job that fails before the pipeline starts its status must not stay 'queued' for good; an uploaded lecture keeps
its own file name (it is saved as upload.mp4)."""
import json
from pathlib import Path

from app import pipeline, server
from app.core.media import MediaError, VideoInfo
from app.pipeline import STAGES, Status


def status(d):
    return json.loads((d / "status.json").read_text(encoding="utf-8"))


def test_a_job_failing_before_the_pipeline_starts_is_marked_failed(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"state": "queued", "error": None, "name": "talk.mp4",
                                                      "stages": {}}), encoding="utf-8")

    def run(*a, **k):
        raise MediaError("the start time is after the end of the video")

    monkeypatch.setattr(server, "run", run)
    server._process(tmp_path, tmp_path / "upload.mp4", 9999.0, None)
    st = status(tmp_path)
    assert st["state"] == "failed" and st["error"] == "the start time is after the end of the video"
    assert st["name"] == "talk.mp4" and isinstance(st["seconds"], float)
    assert list(st["stages"]) == STAGES and all(s["state"] == "pending" for s in st["stages"].values())
    assert "MediaError" in (tmp_path / "error.log").read_text(encoding="utf-8")


def test_no_status_file_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "run", lambda *a, **k: 1 / 0)
    server._process(tmp_path, tmp_path / "upload.mp4", None, None)
    st = status(tmp_path)
    assert st["state"] == "failed" and "ZeroDivisionError" in st["error"] and "\n" not in st["error"]


def test_a_failure_the_pipeline_reported_is_left_alone(tmp_path, monkeypatch):
    def run(*a, **k):
        Status(tmp_path).finish("could not read the video")
        raise RuntimeError("late")

    monkeypatch.setattr(server, "run", run)
    server._process(tmp_path, tmp_path / "upload.mp4", None, None)
    assert status(tmp_path)["error"] == "could not read the video"
    assert not (tmp_path / "error.log").exists()


# ---------- the uploaded file's own name ----------
def queued(d, name="Week 3 - limits.mp4"):
    (d / "status.json").write_text(json.dumps({"state": "queued", "error": None, "name": name, "stages": {}}),
                                   encoding="utf-8")


def test_the_pipeline_status_keeps_the_name_the_server_wrote(tmp_path):
    queued(tmp_path)
    st = Status(tmp_path)
    st.start("probe")
    st.finish()
    assert status(tmp_path)["name"] == "Week 3 - limits.mp4" and st.name == "Week 3 - limits.mp4"
    assert Status(tmp_path, name="other.mp4").name == "other.mp4"  # a name given wins
    empty = tmp_path / "fresh"
    empty.mkdir()
    Status(empty)
    assert "name" not in status(empty)


def test_the_source_name_of_an_upload_is_the_name_it_was_uploaded_with(tmp_path):
    queued(tmp_path)
    assert pipeline.source_name(tmp_path / "upload.mp4", tmp_path) == "Week 3 - limits.mp4"
    # a video elsewhere keeps its own name, whatever an old status.json says
    assert pipeline.source_name(tmp_path / "elsewhere" / "talk.mp4", tmp_path) == "talk.mp4"
    other = tmp_path / "other"
    other.mkdir()
    assert pipeline.source_name(other / "upload.mp4", other) == "upload.mp4"  # nothing better known


def run_until_status(monkeypatch, video, job_dir, **kw):
    """pipeline.run up to the moment it has made its Status: the name it was given."""
    seen = {}
    monkeypatch.setattr(pipeline.media, "probe", lambda p: VideoInfo(Path(p), 60.0, 640, 360, 25.0, True, 0))

    def fake_run(info, s0, s1, job_dir, status, *a, **k):
        seen["name"] = status.name
        return job_dir / "replay.json"

    monkeypatch.setattr(pipeline, "_run", fake_run)
    pipeline.run(video, job_dir, log=lambda m: None, **kw)
    return seen["name"], status(job_dir)["name"]


def test_run_names_the_lecture_after_the_upload_or_the_file(tmp_path, monkeypatch):
    queued(tmp_path)
    assert run_until_status(monkeypatch, tmp_path / "upload.mp4", tmp_path) == ("Week 3 - limits.mp4",) * 2
    assert run_until_status(monkeypatch, tmp_path / "upload.mp4", tmp_path, name="given.mp4") == ("given.mp4",) * 2
    cli = tmp_path / "cli"
    cli.mkdir()
    assert run_until_status(monkeypatch, tmp_path / "videos" / "talk.mp4", cli) == ("talk.mp4",) * 2


def test_the_worker_hands_the_uploaded_name_to_the_pipeline(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(server, "run", lambda *a, **k: got.update(k))
    server._process(tmp_path, tmp_path / "upload.mp4", None, None, "Week 3 - limits.mp4")
    assert got["name"] == "Week 3 - limits.mp4"


def test_an_upload_is_queued_with_its_own_name(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    class Q:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    q = Q()
    monkeypatch.setattr(server, "JOBS", tmp_path)
    monkeypatch.setattr(server, "_work", q)
    r = TestClient(server.app).post("/api/jobs", files={"file": ("Week 3 - limits.MP4", b"not really a video")})
    assert r.status_code == 200
    job = tmp_path / r.json()["id"]
    assert r.json()["id"].startswith("week-3-limits-")
    assert status(job)["name"] == "Week 3 - limits.MP4"
    d, upload, start, end, name = q.items[0]
    assert (d, upload.name, name) == (job, "upload.mp4", "Week 3 - limits.MP4")
