"""The board's pointing stage: one failed call loses one gesture, a failed stage never fails the lecture."""
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app import pipeline
from app.core import board_gestures as bg
from app.core import media, sam, understand
from app.core.media import VideoInfo
from app.core.meta_api import MetaAPIError

INFO = VideoInfo(Path("x.mp4"), 60.0, 1280, 720, 25.0, True, 0)
CORNERS = [[0, 0], [1279, 0], [1279, 719], [0, 719]]


def done(value=None, error=None) -> Future:
    f = Future()
    f.set_exception(error) if error else f.set_result(value)
    return f


def stage(tmp_path):
    return json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["stages"]["gestures"]


def test_a_failed_stage_is_skipped_with_a_plain_note(tmp_path):
    st = pipeline.Status(tmp_path)
    hands = SimpleNamespace(failed=0, n_chunks=2, error=None)

    def spark_down(h):
        raise MetaAPIError("chat (muse-spark) failed after 4 tries: HTTP 503")

    assert pipeline._gestures(done(hands), spark_down, st, tmp_path, lambda m: None) == []
    s = stage(tmp_path)
    assert s["state"] == "skipped" and "HTTP 503" in s["note"] and "hand" not in s["note"]
    # anything at all, e.g. a bug: still only this stage
    assert pipeline._gestures(done(error=KeyError("box")), spark_down, st, tmp_path, lambda m: None) == []
    s = stage(tmp_path)
    assert s["state"] == "skipped" and "KeyError" in s["note"] and "\n" not in s["note"]
    assert "KeyError" in (tmp_path / "error.log").read_text(encoding="utf-8")


def test_hand_tracking_failing_everywhere_or_in_some_chunks(tmp_path):
    st = pipeline.Status(tmp_path)
    called = []
    find = lambda h: called.append(h) or ([{"type": "spotlight"}] * 16, 1, 1)  # noqa: E731
    all_failed = SimpleNamespace(failed=2, n_chunks=2, error=MetaAPIError("segment failed"))
    assert pipeline._gestures(done(all_failed), find, st, tmp_path, lambda m: None) == []
    assert not called and stage(tmp_path)["state"] == "skipped" and "every SAM chunk" in stage(tmp_path)["note"]
    some = SimpleNamespace(failed=1, n_chunks=3, error=MetaAPIError("segment failed"))
    assert len(pipeline._gestures(done(some), find, st, tmp_path, lambda m: None)) == 16
    s = stage(tmp_path)
    assert s["state"] == "done"
    assert s["note"] == ("16 pointing gestures redrawn, 1 could not be judged, 1 shown whole (the part meant is "
                         "unknown); hands not tracked in 1 of 3 SAM chunks")


def test_speech_job_proportional_words_do_not_depend_on_alignment():
    job = pipeline.SpeechJob.__new__(pipeline.SpeechJob)
    turns = [{"start": 10.0, "end": 12.0, "text": "look at this"}]
    job.transcribed = done((None, turns))
    job.aligned = None
    before = job.proportional()
    job.aligned = done(([{"w": "look", "s": 10.5, "e": 10.7}], "mms"))
    assert job.proportional() == before and [w["w"] for w in before] == ["look", "at", "this"]
    assert before[0]["s"] == 10.0 and before[-1]["e"] == 12.0
    assert job.first()[1] == [{"w": "look", "s": 10.5, "e": 10.7}]  # first() still takes MMS when it is there


def ep(t, path):
    return {"type": "move", "tool": "hand", "t_start": t, "t_end": round(t + 0.67, 2), "box": [0, 0, 1, 1],
            "from": path[0], "to": path[-1], "path": path}


def test_board_gestures_lose_only_the_failed_ones(monkeypatch):
    W, H = 800, 450
    pieces = [{"id": i, "image_box": b, "t_seen": 1.0, "removed_at": None, "pre_existing": False}
              for i, b in ((1, [100, 100, 40, 20]), (2, [400, 100, 60, 20]), (3, [300, 200, 200, 100]))]
    reads = {1: {"kind": "text", "text": "a"}, 2: {"kind": "text", "text": "b"}, 3: {"kind": "equation", "text": "c"}}
    ys, xs = np.mgrid[240:250, 320:400]  # piece 3's ink: one line of its block
    add_idx = (ys * W + xs).reshape(-1)
    board = SimpleNamespace(width=W, height=H, corners=CORNERS, pieces=pieces, add_idx=add_idx,
                            add_t=np.ones(add_idx.size), add_piece=np.full(add_idx.size, 3),
                            image_at=lambda t: np.zeros((H, W, 3), np.uint8), ink_at=lambda t: np.ones((H, W), bool))
    eps = [ep(10.0, [[110, 110], [120, 112]]),  # Spark fails
           ep(20.0, [[110, 110], [120, 112]]),  # no frame
           ep(30.0, [[110, 110], [420, 110]]),  # an arrow from 1 to 2 that Spark gave the wrong way round
           ep(40.0, [[350, 250], [380, 245]]),  # an "arrow" from 1, but the hand stayed on 3
           ep(50.0, [[350, 250], [380, 245]])]  # a circle on 3; which part fails
    monkeypatch.setattr(bg, "pointing_samples", lambda *a, **k: [])
    monkeypatch.setattr(bg, "episodes", lambda samples, fps, scale: eps if scale == 1.0 else [])

    def grab_frame(info, t, width):
        if 20 <= t < 21:
            raise media.MediaError("could not read a frame at 20.3s")
        return np.zeros((540, 960, 3), np.uint8)

    said, parts = {}, []
    answers = {30.0: {"type": "arrow", "target": ["2", "1"], "why": "links them"},
               40.0: {"type": "arrow", "target": ["1", "3"], "why": "this"},
               50.0: {"type": "circle", "target": ["3", "3"], "why": "that"}}

    def meaning(client, frame, board_img, g, els, words, fpath):
        said[g["t_start"]] = words
        if g["t_start"] == 10.0:
            raise MetaAPIError("chat failed: HTTP 500")
        return dict(answers[g["t_start"]])

    def part(client, img, region, path, m, text, words):
        parts.append(m["type"])
        if m["why"] == "that":
            raise MetaAPIError("chat failed: HTTP 500")
        return [310, 238, 100, 14]

    monkeypatch.setattr(media, "grab_frame", grab_frame)
    monkeypatch.setattr(understand, "board_gesture_meaning", meaning)
    monkeypatch.setattr(understand, "board_gesture_part", part)
    words = [{"w": "this", "s": 29.0, "e": 29.5}, {"w": "to", "s": 31.0, "e": 31.2}, {"w": "far", "s": 35.0, "e": 36}]

    def first():
        raise AssertionError("gestures must not depend on how far alignment got")

    speech_job = SimpleNamespace(proportional=lambda: words, first=first)
    status = SimpleNamespace(progress=lambda *a: None)
    out, unjudged, whole = pipeline._board_gestures(object(), INFO, board, pieces, reads, None,
                                                    SimpleNamespace(fps=3.0), speech_job, status, lambda m: None)
    assert (unjudged, whole) == (2, 1)
    assert [(a["type"], a["target"]) for a in out] == [("arrow", ["1", "2"]), ("spotlight", ["3"]), ("circle", ["3"])]
    assert out[0]["from"] == [120.0, 110.0] and out[0]["to"] == [430.0, 110.0]
    assert out[1]["box"] == [320.0, 240.0, 80.0, 10.0]  # narrowed to the line, tightened to its ink
    assert out[2]["box"] == [300.0, 200.0, 200.0, 100.0]  # the whole block
    assert parts == ["spotlight", "circle"]
    assert said[30.0] == "this to"


class FakeSam:
    def __init__(self, fail_at=(), on_call=None):
        self.fail_at, self.on_call, self.calls = fail_at, on_call, []

    def segment_video(self, data, prompt, frames):
        start = float(data.decode())
        self.calls.append(start)
        if start in self.fail_at:
            raise MetaAPIError("segment failed: HTTP 500")
        return [{"frame": 0, "box": [10, 10, 30, 30], "local": np.ones((4, 4), bool), "score": 0.9}]


def fake_encode(encoded):
    def encode(info, start, dur, fps, width, out):
        encoded.append(start)
        out.write_bytes(str(start).encode())
    return encode


LONG = VideoInfo(Path("x.mp4"), 300.0, 1280, 720, 25.0, True, 0)  # 3 chunks of 120 s


def test_track_hands_keeps_the_chunks_that_worked(tmp_path, monkeypatch):
    encoded = []
    monkeypatch.setattr(sam, "_encode_chunk", fake_encode(encoded))
    client = FakeSam(fail_at={120.0})
    track = sam.track_hands(client, LONG, workdir=tmp_path)
    assert (track.failed, track.n_chunks) == (1, 3) and isinstance(track.error, MetaAPIError)
    assert sorted(track.samples) == [0, 720]  # chunks 0 and 2 (3 fps)
    assert track.samples[0][0]["box"] == [20.0, 20.0, 60.0, 60.0]  # back in video pixels
    assert not list(tmp_path.glob("hand_chunk_*.mp4"))


def test_track_hands_stops_sending_once_the_job_failed(tmp_path, monkeypatch):
    import threading
    stop = threading.Event()
    encoded = []
    encode = fake_encode(encoded)

    def encode_then_fail(*a):
        encode(*a)
        stop.set()  # the job fails while the first copy is made

    monkeypatch.setattr(sam, "_encode_chunk", encode_then_fail)
    client = FakeSam()
    track = sam.track_hands(client, LONG, workdir=tmp_path, stop=stop)
    assert len(encoded) == 1 and client.calls == []
    assert track.failed == 0 and track.samples == {}



def test_slide_lectures_skip_the_board_gestures_stage(tmp_path, monkeypatch):
    import threading

    import app.core.slides as slides

    def stop_here(*a, **k):
        raise RuntimeError("stop here")

    monkeypatch.setattr(slides, "analyse_slides", stop_here)
    status = pipeline.Status(tmp_path)
    speech_job = SimpleNamespace(cpu_free=threading.Event())
    try:
        pipeline._run_slides(None, 0, 10, tmp_path, status, None, {}, speech_job, tmp_path / "x.mp4", None, 0.0,
                             lambda m: None)
    except RuntimeError:
        pass
    stage = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["stages"]["gestures"]
    assert stage["state"] == "skipped" and "slides" in stage["note"]
