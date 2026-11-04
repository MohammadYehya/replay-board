"""Streaming replay (SPEC "Streaming replay"): while a board lecture is processed, replay.json is written in parts.

- capture's checkpoints never change capture's result;
- each partial keeps the contract (partial key, seq up by one, ready_until only grows and never passes capture's t,
  pieces seen by then, content-addressed pictures that exist);
- the final replay.json is the one written with STREAM_REPLAY=off (apart from stats), with no partial key;
- partial reads that fail never fail the job, and nothing partial is written after the final;
- the server's gates while the replay is partial, and the replay info in the job status.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest

from app import pipeline
from app.config import _stream_mode, load_settings
from app.core import board as board_mod
from app.core import media
from app.core.board import BoardParams, capture_board
from app.core.meta_api import MetaAPIError
from app.stream import Snapshot, StreamWriter, content_file, partial_pieces, ready_pieces

W, H, FPS, SECONDS = 800, 450, 15, 26.0
BOARD, INK, SHIRT = (232, 232, 232), (40, 35, 30), (70, 60, 50)


def _segment(x0, y0, x1, y1, t0, t1):
    """A stroke written from t0 to t1, left to right (bool mask written by t)."""
    full = np.zeros((H, W), np.uint8)
    cv2.line(full, (x0, y0), (x1, y1), 255, 4, cv2.LINE_AA)
    full = full > 127
    xs = np.arange(W)[None, :]

    def at(t):
        if t < t0:
            return np.zeros((H, W), bool)
        f = min(1.0, (t - t0) / max(1e-6, t1 - t0))
        return full & (xs <= x0 + f * (x1 - x0))
    return at


STROKES = [
    (_segment(80, 60, 300, 60, -1.0, -0.5), None),  # on the board when the clip starts
    (_segment(80, 150, 360, 150, 2.0, 4.0), 18.0),  # erased at 18 s
    (_segment(420, 150, 700, 180, 8.0, 10.0), None),
    (_segment(100, 300, 380, 330, 14.0, 16.0), None),
    (_segment(450, 320, 720, 300, 20.0, 21.5), None),
]


def _teacher(t):
    """A shirt crossing the lower right from 5 to 7.5 s (also the person mask)."""
    m = np.zeros((H, W), bool)
    if 5.0 <= t <= 7.5:
        x = int(420 + 120 * (t - 5.0))
        m[200:440, x:x + 120] = True
    return m


def _frame(t):
    f = np.full((H, W, 3), BOARD, np.uint8)
    for at, erased in STROKES:
        if erased is None or t < erased:
            f[at(t)] = INK
    f[_teacher(t)] = SHIRT
    return f


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    path = tmp_path_factory.mktemp("stream") / "lecture.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)]
    frames = b"".join(_frame(i / FPS).tobytes() for i in range(int(SECONDS * FPS)))
    subprocess.run(cmd, input=frames, check=True, capture_output=True, timeout=120)
    return path


# ---------------------------------------------------------------- capture checkpoints

def _same_result(a, b):
    for k in ("width", "height", "mode", "corners", "background", "t_start", "t_end", "fps", "pieces", "erases"):
        assert getattr(a, k) == getattr(b, k), k
    for k in ("add_t", "add_idx", "add_piece", "add_color", "rem_t", "rem_idx", "add_drift", "add_moved", "add_gone",
              "add_blank"):
        x, y = getattr(a, k), getattr(b, k)
        assert x.dtype == y.dtype and np.array_equal(x, y), k
    assert ({k: v for k, v in a.stats.items() if k not in ("seconds", "speed")}
            == {k: v for k, v in b.stats.items() if k not in ("seconds", "speed")})
    for q in a.pieces:
        for u, v in zip(a.piece_image(q["id"]), b.piece_image(q["id"])):
            if isinstance(u, np.ndarray):
                assert isinstance(v, np.ndarray) and u.shape == v.shape and np.array_equal(u, v)
            else:
                assert u == v
        assert a.redraw_part(q["id"]) == b.redraw_part(q["id"])


@pytest.mark.parametrize("person", [False, True], ids=["no-person-mask", "person-mask"])
def test_capture_with_checkpoints_gives_the_same_result(clip, person):
    info = media.probe(clip)
    mask = _teacher if person else None
    plain = capture_board(info, person_mask=mask, params=BoardParams(width=W), mode="white")
    seen = []
    withck = capture_board(info, person_mask=mask, params=BoardParams(width=W), mode="white",
                           checkpoint=lambda part, t, ready: seen.append((part, t, ready)),
                           checkpoint_every=1.0, checkpoint_wall=0.0, checkpoint_share=1.0)
    assert len(seen) >= 20
    _same_result(plain, withck)
    assert len(plain.pieces) >= 4 and plain.erases  # the clip has writing to stream and an erase
    final = {q["id"] for q in plain.pieces}
    last = info.duration
    prev = 0.0
    for part, t, ready in seen:
        assert prev <= ready <= t  # only grows, never past capture's time
        prev = ready
        assert part.t_end == t and part.width == plain.width and part.height == plain.height
        # what a partial shows is still there at the end (the teacher standing there is never shown as writing)
        assert {q["id"] for q in ready_pieces(part, ready)} <= final
        assert t <= last
    assert ready_pieces(seen[-1][0], seen[-1][2])  # and writing is shown before the end


def test_checkpoints_wait_for_video_time_and_wall_clock_time(clip):
    info = media.probe(clip)
    times = []
    capture_board(info, params=BoardParams(width=W), mode="white", checkpoint=lambda p, t, r: times.append(t),
                  checkpoint_every=10.0, checkpoint_wall=0.0)
    assert len(times) == 2 and times[0] >= 10.0 and times[1] - times[0] >= 10.0
    times.clear()
    capture_board(info, params=BoardParams(width=W), mode="white", checkpoint=lambda p, t, r: times.append(t),
                  checkpoint_every=1.0, checkpoint_wall=3600.0)
    assert times == []

    def slow(p, t, r):  # a checkpoint that takes 0.1 s: with a 10% share the next one waits 0.9 s more
        times.append(time.perf_counter())
        time.sleep(0.1)

    capture_board(info, params=BoardParams(width=W), mode="white", checkpoint=slow, checkpoint_every=0.3,
                  checkpoint_wall=0.0, checkpoint_share=0.1)
    gaps = np.diff(times)
    assert len(times) >= 2 and gaps.min() >= 1.0


def test_a_failing_checkpoint_never_stops_the_capture(clip, caplog):
    info = media.probe(clip)
    plain = capture_board(info, params=BoardParams(width=W), mode="white")
    calls = []

    def bad(part, t, ready):
        calls.append(t)
        raise RuntimeError("player went away\nsecond line")

    res = capture_board(info, params=BoardParams(width=W), mode="white", checkpoint=bad, checkpoint_every=2.0,
                        checkpoint_wall=0.0, checkpoint_share=1.0)
    _same_result(plain, res)
    assert len(calls) > 3
    warned = [r for r in caplog.records if "partial replay" in r.getMessage()]
    assert len(warned) == 1 and "\n" not in warned[0].getMessage()  # logged once, one line


def test_ready_until_waits_for_ink_that_may_still_be_shown_earlier():
    p = BoardParams()
    acc = np.zeros((4, 4), bool)
    due = np.full((4, 4), np.inf, np.float32)
    horizon = p.min_life + (p.remove_n - 1) / p.fps
    assert board_mod._ready_until(p, 100.0, 0.0, due, acc, [], []) == pytest.approx(100.0 - horizon)
    due[1, 1] = 80.0  # thick ink not accepted yet, shown from 80 s once it is
    assert board_mod._ready_until(p, 100.0, 0.0, due, acc, [], []) == 80.0
    acc[1, 1] = True  # accepted: no longer pending
    young = [np.full(3, 95.0, np.float32), np.full(2, 99.0, np.float32)]
    shown = [np.float32([70.0, 95.0, 95.0]), np.float32([99.0, 99.0])]  # accepted at 95 s, shown from 70 s
    assert board_mod._ready_until(p, 100.0, 0.0, due, acc, young, shown) == 70.0
    old = [np.full(3, 50.0, np.float32)]  # accepted long ago: final whatever its show time
    assert board_mod._ready_until(p, 100.0, 0.0, due, acc, old, [np.float32([10.0, 10.0, 10.0])]) \
        == pytest.approx(100.0 - horizon)
    assert board_mod._ready_until(p, 100.0, 95.0, due, acc, [], []) == 95.0  # never below the last one
    assert board_mod._ready_until(p, 3.0, 0.0, due, acc, [], []) == 0.0


def test_resolving_merged_pieces_matches_the_loops_resolve():
    alias = {3: 2, 2: 1, 7: 5}
    adds = [np.array([1, 2, 3], np.int64), np.array([], np.int64), np.array([7, 5, 9, 3], np.int64)]
    out = board_mod._resolved(adds, alias)
    assert [a.tolist() for a in out] == [[1, 1, 1], [], [5, 5, 9, 1]]
    assert adds[0].tolist() == [1, 2, 3]  # the loop's arrays are left alone
    assert board_mod._resolved(adds, {})[2] is adds[2]


# ---------------------------------------------------------------- the writers

def test_content_files_are_named_by_their_bytes(tmp_path):
    img = np.zeros((5, 7, 4), np.uint8)
    img[2, 3] = (1, 2, 3, 255)
    a = content_file(tmp_path, img)
    data = (tmp_path / a).read_bytes()
    assert a == f"stream/{hashlib.sha1(data).hexdigest()[:12]}.png"
    assert content_file(tmp_path, img.copy()) == a
    img[2, 4] = (9, 9, 9, 255)
    assert content_file(tmp_path, img) != a
    assert sorted(p.name for p in (tmp_path / "stream").iterdir()) == sorted({a[7:], content_file(tmp_path, img)[7:]})


def test_replay_seq_goes_up_and_nothing_partial_comes_after_the_final(tmp_path):
    st = pipeline.Status(tmp_path)
    w = pipeline.ReplayWriter(tmp_path, st)
    status = lambda: json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))  # noqa: E731
    replay = lambda: json.loads((tmp_path / "replay.json").read_text(encoding="utf-8"))  # noqa: E731
    assert w.write_partial({"video": {}, "n": 1}, 10.0)
    assert replay()["partial"] == {"seq": 1, "ready_until": 10.0}
    assert status()["replay"] == {"seq": 1, "partial": True, "ready_until": 10.0}
    assert w.write_partial({"video": {}, "n": 2}, 20.123)
    assert replay()["partial"]["seq"] == 2 and status()["replay"]["ready_until"] == 20.12
    w.write_final({"video": {}, "n": 3})
    assert "partial" not in replay() and status()["replay"] == {"seq": 3, "partial": False, "ready_until": None}
    assert not w.write_partial({"video": {}, "n": 4}, 30.0)  # late: dropped
    assert replay()["n"] == 3 and status()["replay"]["seq"] == 3
    w.write_final({"video": {}, "n": 5})  # the refined one
    assert replay()["n"] == 5 and status()["replay"]["seq"] == 4
    w.drop_partial()  # a failure after the final keeps the final replay
    assert (tmp_path / "replay.json").exists() and status()["replay"]["seq"] == 4


def test_a_failure_takes_a_partial_replay_away(tmp_path):
    st = pipeline.Status(tmp_path)
    w = pipeline.ReplayWriter(tmp_path, st)
    w.write_partial({"video": {}}, 5.0)
    w.drop_partial()
    assert not (tmp_path / "replay.json").exists()
    assert "replay" not in json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert not w.write_partial({"video": {}}, 6.0)


class _Board:
    """Enough of a BoardResult for the stream writer."""

    def __init__(self, ids):
        self.pieces = [{"id": i, "t_seen": float(i)} for i in ids]


def _writer(build_wait=None, read=None, written=None, accept=lambda: True):
    written = [] if written is None else written
    seq = [0]

    def write(replay, ready):
        if not accept():
            return False
        seq[0] += 1
        replay["partial"] = {"seq": seq[0], "ready_until": ready}
        written.append(json.loads(json.dumps(replay)))
        return True

    def build(snap, reads):
        if build_wait is not None:
            build_wait(snap)
        ids = [] if snap.board is None else [q["id"] for q in snap.board.pieces if q["t_seen"] <= snap.ready_until]
        return {"board": {"pieces": [{"id": i, "read": reads.get(i)} for i in ids]}, "t": snap.t}, snap.ready_until

    logs = []
    sw = StreamWriter(write, build, read, log=logs.append)
    return sw, written, logs


def test_a_snapshot_arriving_while_the_writer_is_busy_replaces_the_waiting_one():
    gate, entered = threading.Event(), threading.Event()

    def wait(snap):
        if snap.t == 1.0:
            entered.set()
            gate.wait(5)

    sw, written, _ = _writer(build_wait=wait)
    sw.submit(Snapshot(None, 1.0, 0.0))
    assert entered.wait(5)
    for t in (2.0, 3.0, 4.0):
        sw.submit(Snapshot(None, t, 0.0))
    gate.set()
    assert sw.idle()
    assert [w["t"] for w in written] == [1.0, 4.0]  # 2 and 3 were dropped
    assert [w["partial"]["seq"] for w in written] == [1, 2]
    sw.stop()


def test_new_pieces_are_read_once_and_the_partial_is_written_again_with_the_readings():
    asked = []

    def read(snap, batch):
        asked.append([q["id"] for q in batch])
        return {q["id"]: {"kind": "text", "text": f"p{q['id']}", "latex": ""} for q in batch}

    sw, written, _ = _writer(read=read)
    sw.batch, sw.min_batch = 2, 1
    sw.submit(Snapshot(_Board([1, 2, 3, 9]), 10.0, 5.0))  # 9 is not ready yet
    assert sw.idle()
    time.sleep(0.2)
    assert sw.idle()
    assert sorted(i for b in asked for i in b) == [1, 2, 3] and sorted(len(b) for b in asked) == [1, 2]
    last = written[-1]["board"]["pieces"]
    assert [q["read"]["text"] for q in last] == ["p1", "p2", "p3"]
    sw.submit(Snapshot(_Board([1, 2, 3, 9]), 20.0, 10.0))
    assert sw.idle()
    time.sleep(0.2)
    assert sw.idle()
    assert sorted(i for b in asked for i in b) == [1, 2, 3, 9]  # only the new one asked again
    assert [q["read"]["text"] for q in written[-1]["board"]["pieces"]] == ["p1", "p2", "p3", "p9"]
    seqs = [w["partial"]["seq"] for w in written]
    assert seqs == list(range(1, len(written) + 1))
    sw.stop()


def test_a_failing_partial_read_leaves_its_pieces_unread_and_says_so_once():
    def read(snap, batch):
        raise MetaAPIError("chat (muse-spark) failed: HTTP 401: bad key\nmore")

    sw, written, logs = _writer(read=read)
    sw.batch, sw.min_batch = 1, 1
    sw.submit(Snapshot(_Board([1, 2, 3]), 10.0, 5.0))
    assert sw.idle()
    time.sleep(0.2)
    assert sw.idle()
    assert all(q["read"] is None for q in written[-1]["board"]["pieces"])
    assert len(logs) == 1 and "HTTP 401" in logs[0] and "\n" not in logs[0]
    sw.use_reads({1: {"kind": "text", "text": "final", "latex": ""}})  # the read stage's readings still get in
    assert sw.idle()
    assert written[-1]["board"]["pieces"][0]["read"]["text"] == "final"
    sw.stop()


def test_partial_reads_wait_for_a_few_new_pieces_or_a_while():
    asked = []

    def read(snap, batch):
        asked.append([q["id"] for q in batch])
        return {q["id"]: {"kind": "text", "text": f"p{q['id']}", "latex": ""} for q in batch}

    sw, written, _ = _writer(read=read)
    assert (sw.min_batch, sw.max_wait) == (5, 20.0)
    sw.max_wait = 0.4

    def settle():
        assert sw.idle()
        time.sleep(0.15)
        assert sw.idle()

    sw.submit(Snapshot(_Board([1, 2]), 10.0, 5.0))
    settle()
    assert asked == []  # two new pieces: they wait for more
    sw.submit(Snapshot(_Board([1, 2, 3, 4, 5]), 20.0, 10.0))
    settle()
    assert asked == [[1, 2, 3, 4, 5]]  # five: one call
    sw.submit(Snapshot(_Board([1, 2, 3, 4, 5, 6]), 30.0, 10.0))
    settle()
    assert len(asked) == 1  # one new piece waits...
    time.sleep(0.4)
    sw.submit(Snapshot(_Board([1, 2, 3, 4, 5, 6]), 40.0, 10.0))
    settle()
    assert asked[1:] == [[6]]  # ...until it has waited max_wait
    sw.batch = 4
    sw.submit(Snapshot(_Board(range(1, 16)), 50.0, 20.0))
    settle()
    assert sorted(len(b) for b in asked[2:]) == [3, 3, 3]  # nine new, at most four a call: three even calls
    assert [q["read"]["text"] for q in written[-1]["board"]["pieces"]] == [f"p{i}" for i in range(1, 16)]
    sw.stop()


def test_partial_reads_not_started_when_capture_ends_are_dropped():
    gate, started = threading.Event(), []

    def read(snap, batch):
        started.append([q["id"] for q in batch])
        gate.wait(5)
        return {q["id"]: {"kind": "text", "text": f"p{q['id']}", "latex": ""} for q in batch}

    sw, written, _ = _writer(read=read)
    sw.batch, sw.min_batch = 1, 1
    sw.submit(Snapshot(_Board([1, 2, 3, 4]), 10.0, 5.0))  # four calls: two run at once, two wait
    end = time.monotonic() + 5
    while len(started) < 2 and time.monotonic() < end:
        time.sleep(0.01)
    sw.end_reads()  # capture is done: the read stage reads them all
    gate.set()
    assert sw.idle()
    time.sleep(0.2)
    assert sw.idle()
    assert len(started) == 2  # the two waiting were never sent
    got = sorted(i for b in started for i in b)
    assert sorted(q["id"] for q in written[-1]["board"]["pieces"] if q["read"]) == got  # the two that ran are used
    sw.submit(Snapshot(_Board([1, 2, 3, 4, 5, 6]), 20.0, 10.0))
    assert sw.idle()
    time.sleep(0.1)
    assert len(started) == 2  # and no new ones are read after that
    sw.stop()


class _Result:
    """Enough of a BoardResult for partial_pieces: each piece's picture is filled with its own number."""

    def __init__(self, pieces, erases=(), shade=0):
        self.pieces = [dict(q) for q in pieces]
        self.erases = list(erases)
        self.shade = shade
        self.add_t = np.array([q["t_seen"] for q in pieces], np.float64)
        self.add_piece = np.array([q["id"] for q in pieces], np.int64)

    def piece_image(self, pid):
        img = np.full((4, 4, 4), (pid * 10 + self.shade) % 256, np.uint8)
        box = next(q["box"] for q in self.pieces if q["id"] == pid)
        return img, img[:, :, 0].copy(), list(box), None, [0.0, 1.0]

    def redraw_part(self, pid):
        return None


def _piece(i, t, box):
    return {"id": i, "t_seen": t, "t_shown": t, "written_after": t - 1, "revealed": False, "removed_at": None,
            "pre_existing": False, "box": box}


def test_a_piece_once_shown_stays_until_it_is_back_or_rubbed_out(tmp_path):
    shown, reads = {}, {2: {"kind": "text", "text": "x^2", "latex": ""}}
    a, b = _piece(1, 1.0, [0, 0, 50, 20]), _piece(2, 2.0, [100, 0, 50, 20])
    first = partial_pieces(tmp_path, _Result([a, b]), 5.0, {}, shown)
    assert [q["id"] for q in first] == [1, 2]
    # piece 2 fails capture's shape checks for a while; piece 1 grew (a new picture)
    second = partial_pieces(tmp_path, _Result([a], shade=1), 10.0, reads, shown)
    assert [q["id"] for q in second] == [1, 2]
    assert second[1]["image"] == first[1]["image"] and second[1]["read"]["text"] == "x^2"  # as last shown, read now
    assert second[0]["image"] != first[0]["image"]
    # its seen time moved past what is ready: it stays as it was shown too
    later = dict(b, t_seen=14.0)
    third = partial_pieces(tmp_path, _Result([a, later], shade=1), 12.0, {}, shown)
    assert [q["id"] for q in third] == [1, 2] and third[1]["t_seen"] == 2.0 and third[1]["image"] == first[1]["image"]
    # back: its new entry
    fourth = partial_pieces(tmp_path, _Result([a, b], shade=2), 15.0, {}, shown)
    assert fourth[1]["image"] != first[1]["image"] and shown[2]["image"] == fourth[1]["image"]
    # gone again, with an erase over it after it was seen: it was rubbed out
    rubbed = partial_pieces(tmp_path, _Result([a], [{"t": 16.0, "box": [110, 5, 20, 10], "pixels": 400}], shade=2),
                            20.0, {}, shown)
    assert [q["id"] for q in rubbed] == [1] and 2 not in shown
    # without shown, only what the snapshot has (as before)
    assert [q["id"] for q in partial_pieces(tmp_path, _Result([a]), 20.0, {})] == [1]


def test_after_stop_or_the_final_nothing_partial_is_written():
    gate, entered = threading.Event(), threading.Event()

    def wait(snap):
        if snap.t == 1.0:
            entered.set()
            gate.wait(5)

    sw, written, _ = _writer(build_wait=wait)
    sw.submit(Snapshot(None, 1.0, 0.0))
    assert entered.wait(5)
    sw.submit(Snapshot(None, 2.0, 0.0))  # waiting: dropped by stop
    threading.Timer(0.2, gate.set).start()
    sw.stop()  # waits for the write in progress
    assert [w["t"] for w in written] == [1.0]
    sw.submit(Snapshot(None, 3.0, 0.0))
    time.sleep(0.1)
    assert [w["t"] for w in written] == [1.0]
    final = [False]
    sw2, written2, _ = _writer(accept=lambda: not final[0])
    sw2.submit(Snapshot(None, 1.0, 0.0))
    assert sw2.idle()
    final[0] = True  # the final replay was written: the next partial is refused and the writer stops
    sw2.submit(Snapshot(None, 2.0, 0.0))
    sw2.idle()
    sw2.submit(Snapshot(None, 3.0, 0.0))
    time.sleep(0.1)
    assert [w["t"] for w in written2] == [1.0]
    sw2.stop()


# ---------------------------------------------------------------- the pipeline, end to end (no model calls)

class FakeClient:
    """Classify says whiteboard; reading answers every piece it is shown. Never a network call."""

    fail_partial_reads = False

    def __init__(self, *a, **k):
        self.calls = []
        self.lock = threading.Lock()

    def ask(self, prompt, images=(), **kw):
        return {"type": "board", "board_colour": "white", "has_webcam_corner": False, "subject": "lines"}

    def chat(self, messages, **kw):
        parts = messages[-1]["content"]
        ids = [int(m) for p in parts if p.get("type") == "text" for m in re.findall(r"Piece id (\d+):", p["text"])]
        with self.lock:
            self.calls.append(ids)
        pieces = [{"id": i, "kind": "text", "text": f"line {i}", "latex": ""} for i in ids]
        return {"choices": [{"message": {"content": json.dumps({"pieces": pieces})}}]}

    def summary(self):
        return {"calls": len(self.calls), "live": 0, "cached": 0, "seconds": 0.0, "cost_usd": 0.0}


_REAL_CAPTURE, _REAL_WRITE = pipeline.capture_board, pipeline._write_replay


def _run(clip, tmp_path, monkeypatch, name, stream="auto", capture=None, read_pieces=None, writes=None):
    monkeypatch.setenv("STREAM_REPLAY", stream)
    monkeypatch.setattr(pipeline, "MetaClient", FakeClient)
    monkeypatch.setattr(StreamWriter, "min_batch", 1)  # read new pieces at once (the rule has its own test)
    def fast_checkpoints(*a, **k):  # checkpoints every 2 s of video, whatever the wall clock
        if k.get("checkpoint") is not None:
            k.update(checkpoint_every=2.0, checkpoint_wall=0.0, checkpoint_share=1.0)
        return (capture or _REAL_CAPTURE)(*a, **k)

    monkeypatch.setattr(pipeline, "capture_board", fast_checkpoints)
    if read_pieces is not None:
        monkeypatch.setattr(pipeline.understand, "read_pieces", read_pieces)
    writes = [] if writes is None else writes
    def record(job_dir, replay):
        _REAL_WRITE(job_dir, replay)
        files = {}
        for q in (replay.get("board") or {}).get("pieces", []):
            for k in ("image", "order", "gone"):
                if q.get(k):
                    files[q[k]] = (job_dir / q[k]).is_file() and (job_dir / q[k]).read_bytes()
        writes.append((json.loads(json.dumps(replay)), files, time.perf_counter()))

    monkeypatch.setattr(pipeline, "_write_replay", record)
    job = tmp_path / name
    pipeline.run(clip, job, use_sam=False, gestures=False, log=lambda m: None, use_cache=False)
    return job, writes


def _no_stats(replay):
    return {k: v for k, v in replay.items() if k != "stats"}


def test_partial_replays_keep_the_contract_and_the_final_is_the_unstreamed_one(clip, tmp_path, monkeypatch):
    job, writes = _run(clip, tmp_path, monkeypatch, "auto")
    off, off_writes = _run(clip, tmp_path, monkeypatch, "off", stream="off")
    partials = [(r, f) for r, f, _ in writes if "partial" in r]
    finals = [r for r, f, _ in writes if "partial" not in r]
    assert len(partials) >= 5 and len(finals) == 1
    assert all("partial" in r for r, _, _ in writes[:len(partials)])  # every partial comes before the final
    final = finals[0]
    # The final replay.json: today's, with no partial key and no stream pictures.
    assert [r for r, _, _ in off_writes if "partial" in r] == [] and not (off / "stream").exists()
    assert _no_stats(final) == _no_stats(off_writes[-1][0])
    assert _no_stats(json.loads((job / "replay.json").read_text(encoding="utf-8"))) == _no_stats(final)
    assert all(q["image"].startswith("pieces/") for q in final["board"]["pieces"])
    first = partials[0][0]
    assert first["partial"] == {"seq": 1, "ready_until": final["video"]["start"]} and first["board"]["pieces"] == []
    prev = -1.0
    shown_ids = set()
    for k, (r, files) in enumerate(partials):
        assert r["partial"]["seq"] == k + 1
        ready = r["partial"]["ready_until"]
        assert prev <= ready <= r["stats"]["capture"]["t"] + 1e-6
        prev = ready
        assert r["video"] == final["video"] and r["type"] == "board" and r["subject"] == final["subject"]
        for key in ("size", "corners", "mode"):
            assert r["board"][key] == final["board"][key], key
        assert r["board"]["lines"] == [] and r["board"]["annotations"] == [] and r["chapters"] == []
        for q in r["board"]["pieces"]:
            assert q["t_seen"] <= ready and q["spoken"] == [] and q["redraw_at"] is None
            assert set(q) == set(final["board"]["pieces"][0])
            for key in ("image", "order", "gone"):
                if q[key] is None:
                    continue
                assert re.fullmatch(r"stream/[0-9a-f]{12}\.png", q[key])
                data = files[q[key]]
                assert data and hashlib.sha1(data).hexdigest()[:12] == q[key][7:19]  # there, and its own name
            shown_ids.add(q["id"])
    assert shown_ids <= {q["id"] for q in final["board"]["pieces"]}
    # the whole board before the final: the partial after capture has every piece, read by the read stage
    last = partials[-1][0]
    assert last["partial"]["ready_until"] == final["video"]["end"]
    assert [q["id"] for q in last["board"]["pieces"]] == [q["id"] for q in final["board"]["pieces"]]
    assert [q["read"] for q in last["board"]["pieces"]] == [q["read"] for q in final["board"]["pieces"]]
    st = json.loads((job / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "done" and st["replay"] == {"seq": len(partials) + 1, "partial": False, "ready_until": None}
    off_st = json.loads((off / "status.json").read_text(encoding="utf-8"))
    assert off_st["replay"] == {"seq": 1, "partial": False, "ready_until": None}


def test_a_failing_partial_read_never_fails_the_job(clip, tmp_path, monkeypatch):
    real = pipeline.understand.read_pieces
    tried = []

    def read_pieces(client, board, img, pieces, **kw):
        if kw.get("workers") == 1:  # the partial replays' reads
            tried.append([q["id"] for q in pieces])
            raise MetaAPIError("chat (muse-spark) failed: HTTP 401: invalid key")
        return real(client, board, img, pieces, **kw)

    job, writes = _run(clip, tmp_path, monkeypatch, "badread", read_pieces=read_pieces)
    st = json.loads((job / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "done" and tried
    final = json.loads((job / "replay.json").read_text(encoding="utf-8"))
    assert "partial" not in final and all(q["read"]["text"] for q in final["board"]["pieces"])
    during = [r for r, _, _ in writes if "partial" in r and r["partial"]["ready_until"] < final["video"]["end"]]
    assert during and all(q["read"]["text"] == "" for r in during for q in r["board"]["pieces"])


def test_a_job_failing_during_capture_takes_its_partial_replay_away(clip, tmp_path, monkeypatch):
    def broken(*a, **k):
        capture_board(*a, **k)  # with its checkpoints: partial replays are written
        time.sleep(0.3)
        raise MemoryError()

    writes = []
    with pytest.raises(MemoryError):
        _run(clip, tmp_path, monkeypatch, "broken", capture=broken, writes=writes)
    assert len(writes) >= 2 and all("partial" in r for r, _, _ in writes)
    job = tmp_path / "broken"
    st = json.loads((job / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "failed" and "memory" in st["error"] and "replay" not in st
    assert not (job / "replay.json").exists()


def test_a_rerun_that_fails_keeps_the_replay_of_the_run_before(clip, tmp_path, monkeypatch):
    job, _ = _run(clip, tmp_path, monkeypatch, "rerun")
    before = (job / "replay.json").read_bytes()

    def broken(*a, **k):
        capture_board(*a, **k)
        time.sleep(0.3)
        raise MemoryError()

    writes = []
    with pytest.raises(MemoryError):
        _run(clip, tmp_path, monkeypatch, "rerun", capture=broken, writes=writes)
    assert writes == []  # nothing partial over the replay that was there
    assert (job / "replay.json").read_bytes() == before
    st = json.loads((job / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "failed" and "memory" in st["error"]
    # and a re-run that works writes its final replay once, still without partial ones
    writes = []
    _run(clip, tmp_path, monkeypatch, "rerun", writes=writes)
    assert [("partial" in r) for r, _, _ in writes] == [False]


def test_a_run_stopped_with_ctrl_c_takes_its_partial_replay_away(clip, tmp_path, monkeypatch):
    def stopped(*a, **k):
        capture_board(*a, **k)
        time.sleep(0.3)
        raise KeyboardInterrupt()

    writes = []
    with pytest.raises(KeyboardInterrupt):
        _run(clip, tmp_path, monkeypatch, "ctrlc", capture=stopped, writes=writes)
    assert len(writes) >= 2 and all("partial" in r for r, _, _ in writes)
    job = tmp_path / "ctrlc"
    st = json.loads((job / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "failed" and st["error"] == pipeline.INTERRUPTED and "replay" not in st
    assert not (job / "replay.json").exists() and not (job / "stream").exists()


def test_a_run_whose_server_is_stopping_says_interrupted_not_failed(clip, tmp_path, monkeypatch):
    # Stopping the server while a job runs: Python's shutdown refuses new thread-pool work, so a stage fails with
    # this RuntimeError (data/test4.mp4). That is the server stopping, not the job going wrong.
    def stopping(*a, **k):
        capture_board(*a, **k)
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    with pytest.raises(RuntimeError):
        _run(clip, tmp_path, monkeypatch, "stopping", capture=stopping, writes=[])
    st = json.loads((tmp_path / "stopping" / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "failed" and st["error"] == pipeline.INTERRUPTED and "replay" not in st


def test_a_finished_job_keeps_no_stream_pictures(clip, tmp_path, monkeypatch):
    writes = []
    job, _ = _run(clip, tmp_path, monkeypatch, "tidy", writes=writes)
    assert any("partial" in r for r, _, _ in writes)  # it did stream
    assert not (job / "stream").exists() and (job / "pieces").is_dir()


def test_the_lecture_summary_is_written_only_when_it_changes(tmp_path):
    replay = {"video": {"start": 0, "end": 10, "source": "a.mp4"}, "subject": "limits", "type": "board"}
    pipeline._write_replay(tmp_path, replay)
    path = tmp_path / "summary.json"
    os.utime(path, (1000, 1000))
    pipeline._write_replay(tmp_path, dict(replay, partial={"seq": 2, "ready_until": 3.0}))
    assert path.stat().st_mtime == 1000  # the same facts: not written again (the lecture list reads it any time)
    pipeline._write_replay(tmp_path, dict(replay, subject="series"))
    assert json.loads(path.read_text(encoding="utf-8"))["subject"] == "series"


def _job(root, name, state, pid=None, replay=None, stages=None):
    d = root / name
    d.mkdir()
    st = {"state": state, "error": None, "started": time.time() - 60, "stages": stages or {}}
    if pid is not None:
        st["pid"] = pid
    if replay is not None:
        (d / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
        st["replay"] = {"seq": 3, "partial": "partial" in replay, "ready_until": None}
        if "partial" in replay:
            (d / "stream").mkdir()
            (d / "stream" / "abc.png").write_bytes(b"png")
    (d / "status.json").write_text(json.dumps(st), encoding="utf-8")
    return d


def test_a_server_starting_marks_jobs_whose_process_is_gone(tmp_path):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    partial = {"video": {}, "partial": {"seq": 3, "ready_until": 20.0}}
    final = {"video": {}, "transcript": {"timing": "proportional"}}
    running = {"capture": {"state": "running", "progress": 0.4, "seconds": None, "note": ""},
               "align": {"state": "pending", "progress": 0.0, "seconds": None, "note": ""}}
    _job(tmp_path, "queued", "queued")
    _job(tmp_path, "stopped", "running", pid=gone.pid, replay=partial, stages=running)
    _job(tmp_path, "refining", "running", pid=gone.pid, replay=final,
         stages={"write": {"state": "done", "note": ""}, "align": {"state": "running", "note": ""}})
    _job(tmp_path, "older", "running")  # from before status.json named its process
    _job(tmp_path, "live", "running", pid=os.getpid(), replay=partial)  # python -m app.process, still at it
    _job(tmp_path, "done", "done", pid=gone.pid, replay=final)
    logs = []
    changed = pipeline.recover_interrupted(tmp_path, log=logs.append)
    assert sorted(changed) == ["older", "queued", "refining", "stopped"]
    st = lambda n: json.loads((tmp_path / n / "status.json").read_text(encoding="utf-8"))  # noqa: E731
    for n in ("queued", "stopped", "older"):
        assert st(n)["state"] == "failed" and st(n)["error"] == pipeline.INTERRUPTED, n
    assert "replay" not in st("stopped") and not (tmp_path / "stopped" / "replay.json").exists()
    assert not (tmp_path / "stopped" / "stream").exists()
    assert st("stopped")["stages"] == running  # the stage it stopped in still shows where
    # stopped while refining word times: its replay is complete, the job is done with it
    assert st("refining")["state"] == "done" and st("refining")["error"] is None
    assert st("refining")["stages"]["align"] == {"state": "skipped", "note": "interrupted"}
    assert (tmp_path / "refining" / "replay.json").exists() and st("refining")["replay"]["partial"] is False
    assert st("live")["state"] == "running" and (tmp_path / "live" / "replay.json").exists()
    assert st("done")["state"] == "done"
    assert len(logs) == 1 and "\n" not in logs[0]
    assert pipeline.recover_interrupted(tmp_path, log=logs.append) == []  # once is enough


def test_the_server_marks_interrupted_jobs_when_it_starts(monkeypatch):
    import uvicorn

    from app import server

    calls = []
    monkeypatch.setattr(server, "recover_interrupted", lambda jobs, log=print: calls.append(("recover", jobs)))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append(("run", k.get("port"))))
    monkeypatch.setattr(sys, "argv", ["app.server", "--port", "8019"])
    server.main()
    assert calls == [("recover", server.JOBS), ("run", 8019)]


def test_a_job_status_names_its_process(tmp_path):
    pipeline.Status(tmp_path)
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_the_switch():
    for word in ("off", "OFF", "no", "false", "0", "none", "cpu", " disabled "):
        assert _stream_mode(word) == "off", word
    for word in ("auto", "", "on", "yes", "1", "whatever"):
        assert _stream_mode(word) == "auto", word


def test_the_switch_from_the_environment(monkeypatch):
    monkeypatch.setenv("STREAM_REPLAY", "false")
    assert load_settings().stream_replay == "off"
    monkeypatch.setenv("STREAM_REPLAY", "auto")
    assert load_settings().stream_replay == "auto"


# ---------------------------------------------------------------- the server while the replay is partial

@pytest.fixture
def server_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import server

    job = tmp_path / "lec"
    job.mkdir()
    replay = {"version": 1, "video": {"file": "lecture.mp4", "start": 0, "end": 60}, "type": "board",
              "subject": "limits",
              "board": {"size": [800, 400], "mode": "white", "corners": [], "background": "#ffffff",
                        "pieces": [{"id": 1, "box": [0, 0, 10, 10], "t_seen": 5.0, "t_shown": 6.0,
                                    "read": {"kind": "graph", "text": "graph of y = x^2", "latex": ""},
                                    "spoken": [], "redraw_at": None, "removed_at": None, "pre_existing": False,
                                    "revealed": False, "written_after": 4.0, "image": "stream/a.png",
                                    "order": "stream/b.png", "gone": None}],
                        "erases": [], "lines": [], "annotations": [], "t_start": 0.0},
              "transcript": {"turns": [{"start": 1.0, "end": 4.0, "text": "a parabola"}],
                             "words": [{"w": "a", "s": 1.0, "e": 2.0}, {"w": "parabola", "s": 2.0, "e": 4.0}],
                             "timing": "proportional"},
              "chapters": [], "stats": {}, "partial": {"seq": 3, "ready_until": 30.0}}
    (job / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    (job / "status.json").write_text(json.dumps({"state": "running", "error": None, "stages": {},
                                                 "replay": {"seq": 3, "partial": True, "ready_until": 30.0}}),
                                     encoding="utf-8")
    monkeypatch.setattr(server, "JOBS", tmp_path)
    monkeypatch.setattr(server, "_meta_client", lambda: pytest.fail("no model call while the replay is partial"))
    return TestClient(server.app), job, replay


def test_the_job_status_says_which_replay_is_there(server_job):
    c, job, _ = server_job
    st = c.get("/api/jobs/lec").json()
    assert st["ready"] is True and st["replay"] == {"seq": 3, "partial": True, "ready_until": 30.0}
    assert c.get("/api/jobs/lec/replay.json").json()["partial"] == {"seq": 3, "ready_until": 30.0}


@pytest.mark.parametrize("method, url", [
    ("post", "/api/jobs/lec/popout/1"),
    ("post", "/api/jobs/lec/slides/write"),
    ("get", "/api/jobs/lec/notes.md"),
    ("get", "/api/jobs/lec/notes.json"),
    ("get", "/api/jobs/lec/slides.pptx"),
])
def test_what_needs_the_whole_lecture_waits_for_the_final_replay(server_job, method, url):
    c, job, _ = server_job
    r = getattr(c, method)(url)
    assert r.status_code == 409 and r.json()["detail"] == "Available when processing finishes"


def test_no_3d_badges_while_partial_and_everything_once_final(server_job):
    c, job, replay = server_job
    assert c.get("/api/jobs/lec/popouts").json() == {"drawings": []}
    final = {k: v for k, v in replay.items() if k != "partial"}
    final["transcript"]["timing"] = "mms"
    (job / "replay.json").write_text(json.dumps(final), encoding="utf-8")
    assert c.get("/api/jobs/lec/notes.json").status_code == 200
    assert c.get("/api/jobs/lec/notes.md").status_code == 200


def test_ask_still_answers_from_a_partial_replay(server_job):
    from app import server

    _, job, _ = server_job
    prep = server._ask_prepare(job, {"question": "what is drawn?", "selection": [{"kind": "piece", "id": 1}],
                                     "history": [], "t": 10.0})
    assert prep.question == "what is drawn?" and prep.messages


def test_the_first_partial_board_colour_is_not_a_title_card(monkeypatch):
    # An edited video opens on a black title card: the colour before capture's first checkpoint comes from frames
    # across the lecture, not the first one alone.
    from app import stream

    green, black = (65, 66, 43), (0, 0, 0)
    monkeypatch.setattr(stream.media, "grab_frame",
                        lambda info, t, width: np.full((90, 160, 3), black if t < 10 else green, np.uint8))

    class Geom:
        dw = 160

        @staticmethod
        def warp(img):
            return img

    assert stream.board_colour(None, Geom, [0.0, 400.0, 800.0], "chalk") == green
    assert stream.board_colour(None, Geom, [0.0, 1.0, 800.0], "chalk") == black  # mostly cards: nothing to go on
