"""A job that fails before the pipeline starts its status must not stay 'queued' for good."""
import json

from app import server
from app.core.media import MediaError
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
