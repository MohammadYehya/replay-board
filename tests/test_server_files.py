"""The job's files as the player gets them: the video in byte ranges (one request per fragment of a fragmented MP4,
answered without a hop to a worker thread), and the poster frame shown until the video has its first frame."""
import json
import subprocess

import anyio.to_thread
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import server

DATA = bytes(range(256)) * 1200  # 307,200 bytes, every offset recognisable


@pytest.fixture
def client(tmp_path, monkeypatch):
    job = tmp_path / "lec-1"
    job.mkdir()
    (job / "lecture.mp4").write_bytes(DATA)
    (job / "replay.json").write_text(json.dumps({"video": {"file": "lecture.mp4", "start": 0, "end": 2}}),
                                     encoding="utf-8")
    monkeypatch.setattr(server, "JOBS", tmp_path)
    return TestClient(server.app), job


URL = "/api/jobs/lec-1/files/lecture.mp4"


@pytest.mark.parametrize("rng, start, end", [
    ("bytes=0-99", 0, 100),
    ("bytes=1000-", 1000, len(DATA)),
    ("bytes=-500", len(DATA) - 500, len(DATA)),
    ("bytes=307000-999999", 307000, len(DATA)),
    ("bytes=0-0", 0, 1),
])
def test_one_range_is_the_bytes_asked_for(client, rng, start, end):
    c, _ = client
    r = c.get(URL, headers={"Range": rng})
    assert r.status_code == 206
    assert r.content == DATA[start:end]
    assert r.headers["content-range"] == f"bytes {start}-{end - 1}/{len(DATA)}"
    assert int(r.headers["content-length"]) == end - start
    assert r.headers["content-type"] == "video/mp4" and r.headers["accept-ranges"] == "bytes"
    assert r.headers["cache-control"] == "no-cache" and r.headers["etag"]


def test_the_rest_is_answered_as_before(client):
    c, _ = client
    r = c.get(URL)
    assert r.status_code == 200 and r.content == DATA and r.headers["accept-ranges"] == "bytes"
    assert c.get(URL, headers={"Range": f"bytes={len(DATA)}-"}).status_code == 416
    several = c.get(URL, headers={"Range": "bytes=0-9,20-29"})
    assert several.status_code == 206 and "multipart/byteranges" in several.headers["content-type"]
    stale = c.get(URL, headers={"Range": "bytes=0-9", "If-Range": '"not-the-etag"'})
    assert stale.status_code == 200 and stale.content == DATA  # the file changed: all of it
    assert c.get("/api/jobs/lec-1/files/../lec-1/replay.json").status_code in (200, 404)
    assert c.get("/api/jobs/lec-1/files/..%2F..%2Fsecret.txt").status_code == 404
    assert c.get("/api/jobs/lec-1/files/nothing.mp4").status_code == 404


def test_a_range_takes_no_hop_to_a_worker_thread(client, monkeypatch):
    """Chrome opens test-full's 8-minute fragmented MP4 with 88 range requests in a row; each hop to a worker thread
    costs about a millisecond on Windows, and FileResponse made four per request."""
    c, _ = client
    hops = []
    real = anyio.to_thread.run_sync

    async def counting(*a, **k):
        hops.append(a[0])
        return await real(*a, **k)

    monkeypatch.setattr(anyio.to_thread, "run_sync", counting)
    r = c.get(URL, headers={"Range": "bytes=5000-"})
    assert r.status_code == 206 and r.content == DATA[5000:]
    assert hops == []


def make_video(path):
    """2 s of plain blue, 320x180."""
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                    "color=c=0x3060c0:size=320x180:rate=25", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path)], check=True, capture_output=True, timeout=60)


def test_the_poster_is_the_video_frame_where_the_player_opens(client, monkeypatch):
    c, job = client
    make_video(job / "lecture.mp4")
    from app.core import media

    grabs = []
    real = media.grab_frame
    monkeypatch.setattr(media, "grab_frame", lambda *a, **k: grabs.append(a[1]) or real(*a, **k))
    r = c.get("/api/jobs/lec-1/poster.jpg?t=0.2")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (180, 320)  # the video's own size (never blown up)
    b, g, rr = img[90, 160].astype(int)
    assert b > rr  # the blue first part
    assert c.get("/api/jobs/lec-1/poster.jpg?t=0.2").content == r.content and grabs == [0.2]  # kept in memory
    assert "max-age" in r.headers["cache-control"]  # the progress page fetches it before the player asks for it
    # no time: where the processed part starts
    (job / "replay.json").write_text(json.dumps({"video": {"file": "lecture.mp4", "start": 1.5, "end": 2}}),
                                     encoding="utf-8")
    assert c.get("/api/jobs/lec-1/poster.jpg").status_code == 200 and grabs == [0.2, 1.5]


def test_the_poster_time_stays_in_the_processed_part(client, monkeypatch):
    """The player clamps where it opens to the processed part (gest-test, processed from 6:20, opened at ?t=0 starts
    at 6:20); the poster is the frame there, not the 0:00 frame."""
    c, job = client
    make_video(job / "lecture.mp4")
    (job / "replay.json").write_text(
        json.dumps({"video": {"file": "lecture.mp4", "start": 0.5, "end": 1.5, "duration": 2}}), encoding="utf-8")
    from app.core import media

    grabs = []
    real = media.grab_frame
    monkeypatch.setattr(media, "grab_frame", lambda *a, **k: grabs.append(a[1]) or real(*a, **k))
    for t in ("0", "-3", "1.0", "1.9", "nan"):
        assert c.get(f"/api/jobs/lec-1/poster.jpg?t={t}").status_code == 200
    assert grabs == [0.5, 1.0, 1.5]  # 0, -3 and nan: the start (kept); 1.9: the end
    # the same frame as the URL without a time (the progress page warms that one for a plain link)
    assert c.get("/api/jobs/lec-1/poster.jpg").content == c.get("/api/jobs/lec-1/poster.jpg?t=0").content
    assert grabs == [0.5, 1.0, 1.5]


def test_no_poster_without_a_video(client):
    c, job = client
    (job / "replay.json").write_text(json.dumps({"video": {"file": "gone.mp4"}}), encoding="utf-8")
    assert c.get("/api/jobs/lec-1/poster.jpg").status_code == 404
    (job / "replay.json").unlink()
    assert c.get("/api/jobs/lec-1/poster.jpg").status_code == 404
    (job / "replay.json").write_text(json.dumps({"video": {"file": "lecture.mp4"}}), encoding="utf-8")
    assert c.get("/api/jobs/lec-1/poster.jpg").status_code == 404  # not a video at all: one plain 404
