"""GPU decoding (D3D11VA) must give exactly the frames software decoding gives, or not be used: model answers are
cached by the bytes sent, so a changed pixel would mean paying again. Argument building, the fallbacks, and a GPU
ffmpeg that dies part way. The tests marked `gpu` run only where the GPU path is available (Windows with D3D11VA)."""
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import load_settings
from app.core import media, sam
from app.core.media import VideoInfo


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], check=True, capture_output=True, timeout=60)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A clean per-file cache, and "auto" whatever the owner's .env says."""
    monkeypatch.setattr(media, "_gpu_ok", {})
    monkeypatch.setenv("VIDEO_DECODE", "auto")


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("gpu clips")
    src = "testsrc2=size=640x360:rate=30"
    out = {}

    def make(name, *args, src=src, t="2"):
        out[name] = d / name
        ffmpeg("-f", "lavfi", "-i", src, "-t", t, *args, str(out[name]))

    make("h264.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "15")
    make("hevc.mp4", "-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error")
    make("full_range.mp4", "-c:v", "libx264", "-pix_fmt", "yuvj420p")
    make("ten_bit.mp4", "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error")
    make("sar.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=720:576,setsar=16/15",
         src="testsrc2=size=720x576:rate=25")
    make("not_mod16.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", src="testsrc2=size=642x362:rate=25")
    make("vfr.mkv", "-vf", "select='not(mod(n\\,3))+gt(n\\,40)',setpts=N/FRAME_RATE/TB", "-fps_mode", "vfr",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", t="3")
    make("mpeg2.mpg", "-c:v", "mpeg2video", "-q:v", "4")
    # 1 s of black, then a test pattern: the check at 0 sees only black, which looks the same flipped
    out["black_start.mp4"] = d / "black_start.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=black:size=640x360:rate=30:d=1", "-f", "lavfi", "-i", src + ":d=1",
           "-filter_complex", "[0][1]concat=n=2:v=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "15",
           str(out["black_start.mp4"]))

    def copy(name, source, *args, bsf=None):
        out[name] = d / name
        ffmpeg(*args, "-i", str(out[source]), "-c", "copy", *(["-bsf:v", bsf] if bsf else []), str(out[name]))

    copy("rotated.mp4", "h264.mp4", "-display_rotation", "90")
    copy("vflip.mp4", "h264.mp4", "-display_vflip")  # the container's display matrix; ffprobe says rotation 0
    copy("vflip_black.mp4", "black_start.mp4", "-display_vflip")
    # The same flip asked for by the H.264 stream itself (an SEI message in its first frame only), not the container
    copy("sei_flip_black.mp4", "black_start.mp4", bsf="h264_metadata=display_orientation=insert:flip=vertical")
    # Damaged: one frame's NAL length made impossible, 1.1 s in (after the frames the check looks at). No B-frames,
    # so the frames after it up to the next keyframe (1.5 s) depend on it, and the two decoders hide that differently.
    make("p_frames.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "15", "-bf", "0")
    out["broken_nal.mp4"] = d / "broken_nal.mp4"
    packets = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                         "packet=pos", "-of", "json", str(out["p_frames.mp4"])],
                                        capture_output=True, check=True).stdout)["packets"]
    data = bytearray(out["p_frames.mp4"].read_bytes())
    pos = int(packets[33]["pos"])
    data[pos:pos + 4] = (0x7FFFFFF0).to_bytes(4, "big")
    out["broken_nal.mp4"].write_bytes(bytes(data))
    return out


def frames(info, fps=3, width=320, start=0.0, end=None):
    return [(t, f.tobytes()) for t, f in media.iter_frames(info, fps, width, start, end)]


def software(monkeypatch, fn):
    with monkeypatch.context() as m:
        m.setenv("VIDEO_DECODE", "cpu")
        return fn()


def chunk_bytes(info, path, start=0.3, dur=1.5):
    sam._encode_chunk(info, start, dur, 2.0, 320, path)
    return path.read_bytes()


# ------------------------------------------------------------------ everywhere

def test_decode_args():
    info = VideoInfo(Path("x y.mp4"), 60.0, 1280, 720, 25.0, True, 0, "h264", "yuv420p")
    ff = media._tool("ffmpeg")
    # Software: exactly the arguments used before GPU decoding existed (the cached SAM copies depend on them).
    assert media.decode_args(info, 1.5, 2.0, 3, 640, 360) == [
        ff, "-v", "error", "-nostdin", "-ss", "1.500", "-i", "x y.mp4", "-t", "2.000", "-an", "-sn",
        "-vf", "fps=3:round=up,scale=640:360:flags=area,setsar=1"]
    gpu = media.decode_args(info, 1.5, 2.0, 3, 640, 360, gpu=True, overwrite=True)
    # -xerror: damaged data stops the GPU run (and software takes over) instead of being hidden differently
    assert gpu[:11] == [ff, "-v", "error", "-nostdin", "-y", "-xerror", "-hwaccel", "d3d11va",
                        "-hwaccel_output_format", "d3d11", "-ss"]
    assert gpu[11:] == ["1.500", "-i", "x y.mp4", "-t", "2.000", "-an", "-sn", "-vf",
                        "fps=3:round=up,hwdownload,format=nv12,format=yuv420p,scale=640:360:flags=area,setsar=1"]


def test_video_decode_setting(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("VIDEO_DECODE=CPU\n", encoding="utf-8")
    monkeypatch.delenv("VIDEO_DECODE")
    assert load_settings(env).video_decode == "cpu"
    assert load_settings(tmp_path / "none.env").video_decode == "auto"
    monkeypatch.setenv("VIDEO_DECODE", "auto")  # the environment wins over .env
    assert load_settings(env).video_decode == "auto"


@pytest.mark.parametrize("value, mode", [("off", "cpu"), ("Software", "cpu"), ("none", "cpu"), ("0", "cpu"),
                                         ("false", "cpu"), ("no", "cpu"), (" GPU ", "auto"), ("on", "auto"),
                                         ("1", "auto")])
def test_video_decode_words(monkeypatch, value, mode):
    monkeypatch.setenv("VIDEO_DECODE", value)
    assert media.decode_setting() == mode


def test_unknown_video_decode_value_means_software(monkeypatch, caplog):
    """A value that is neither: software (outputs can't change that way), and one warning saying so."""
    monkeypatch.setattr(media, "_warned", set())
    monkeypatch.setenv("VIDEO_DECODE", "d3d11va")
    with caplog.at_level(logging.WARNING, logger=media.__name__):
        assert media.decode_setting() == "cpu"
        assert media.decode_setting() == "cpu"
    assert [r.getMessage() for r in caplog.records] == ["VIDEO_DECODE=d3d11va is not auto or cpu; decoding in software"]


def test_no_gpu_without_running_a_check(clips, monkeypatch):
    """Cases decided before any GPU run: setting, OS, codec, pixel format, rotation, flips."""
    monkeypatch.setattr(media, "_same_frames", lambda *a: pytest.fail("no check expected"))
    h264 = media.probe(clips["h264.mp4"])
    for info in (media.probe(clips["ten_bit.mp4"]), media.probe(clips["mpeg2.mpg"]),
                 media.probe(clips["rotated.mp4"]), media.probe(clips["vflip.mp4"])):
        assert not media.gpu_decode_ok(info, 0.0)
    with monkeypatch.context() as m:
        m.setenv("VIDEO_DECODE", "cpu")
        assert not media.gpu_decode_ok(h264, 0.0)
    with monkeypatch.context() as m:
        m.setattr(sys, "platform", "linux")
        assert not media.gpu_decode_ok(h264, 0.0)


def test_unknown_hwaccel_falls_back_to_the_same_frames(clips, monkeypatch):
    """An ffmpeg without the GPU decoder (or no GPU): the check fails once, software gives the frames."""
    monkeypatch.setattr(media, "HWACCEL", "no_such_hwaccel")
    info = media.probe(clips["h264.mp4"])
    want = software(monkeypatch, lambda: frames(info, start=0.4))
    assert frames(info, start=0.4) == want and len(want) >= 4
    assert not media.gpu_decode_ok(info, 0.4)
    assert list(media._gpu_ok.values()) == ([False] if sys.platform == "win32" else [])
    path = clips["h264.mp4"].parent / "chunk_bogus.mp4"
    assert chunk_bytes(info, path) == software(monkeypatch, lambda: chunk_bytes(info, path))


class _DiesAfter:
    """A running ffmpeg whose output stops after `n` bytes and which then dies (exit code 1)."""

    def __init__(self, proc, n):
        self.proc, self.left = proc, n
        self.stdout, self.stderr = self, proc.stderr

    def read(self, size):
        if self.left <= 0:
            if self.proc.poll() is None:
                self.proc.kill()
            return b""
        buf = self.proc.stdout.read(min(size, self.left))
        self.left -= len(buf)
        return buf

    def close(self):
        self.proc.stdout.close()

    def poll(self):
        return self.proc.poll()

    def kill(self):
        self.proc.kill()

    def wait(self, timeout=None):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout)
        return 1


def _die_after(monkeypatch, n_frames, frame_bytes, marker):
    real = subprocess.Popen

    def popen(args, *a, **k):
        p = real(args, *a, **k)
        return _DiesAfter(p, n_frames * frame_bytes) if marker in args and "rawvideo" in args else p
    monkeypatch.setattr(subprocess, "Popen", popen)


@pytest.mark.parametrize("n", [0, 1, 5])
def test_gpu_dying_part_way_continues_in_software(clips, monkeypatch, n):
    """Frame k is at start + k/fps and has software's pixels, whether it came from the GPU run or the takeover.
    The "GPU" here is software decoding marked with -hwaccel none, so this runs on any machine."""
    info = media.probe(clips["vfr.mkv"])
    want = software(monkeypatch, lambda: frames(info, 3, 320, 0.4))
    monkeypatch.setattr(media, "gpu_decode_ok", lambda *a: True)
    monkeypatch.setattr(media, "_gpu_parts", lambda info: (["-hwaccel", "none"], ""))
    _die_after(monkeypatch, n, 320 * 180 * 3, "none")
    got = frames(info, 3, 320, 0.4)
    assert len(want) > n + 1 and got == want


def test_damaged_data_part_way_continues_in_software(clips, monkeypatch, caplog):
    """Damaged data stops the "GPU" run (its -xerror) and software carries on with its own frames; the owner gets
    one plain line, without ffmpeg's "[name @ address]" prefixes. Software marked -hwaccel none plays the GPU."""
    info = media.probe(clips["broken_nal.mp4"])
    want = software(monkeypatch, lambda: frames(info, 3, 320, 0.0))
    monkeypatch.setattr(media, "gpu_decode_ok", lambda *a: True)
    monkeypatch.setattr(media, "_gpu_parts", lambda info: (["-xerror", "-hwaccel", "none"], ""))
    with caplog.at_level(logging.WARNING, logger=media.__name__):
        assert frames(info, 3, 320, 0.0) == want
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1 and re.search(r"stopped after \d+ frames", lines[0]) and "Invalid data" in lines[0], lines
    assert not re.search(r"\[[^\]]* @ ", lines[0]), lines


def test_replaced_file_is_checked_again(clips, tmp_path, monkeypatch):
    """The check's result belongs to the file's content: another video copied over it with the same size and
    modified time is checked again."""
    monkeypatch.setattr(sys, "platform", "win32")
    checks = []
    monkeypatch.setattr(media, "_same_frames", lambda info, start: checks.append(info.path) or True)
    a, b = clips["h264.mp4"].read_bytes(), clips["not_mod16.mp4"].read_bytes()
    size = max(len(a), len(b))
    video = tmp_path / "lecture.mp4"
    video.write_bytes(a.ljust(size, b"\0"))
    st = video.stat()
    assert media.gpu_decode_ok(media.probe(video), 0.0) and len(checks) == 1
    assert media.gpu_decode_ok(media.probe(video), 0.0) and len(checks) == 1  # same file: no second check
    video.write_bytes(b.ljust(size, b"\0"))
    os.utime(video, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert video.stat().st_size == st.st_size and video.stat().st_mtime_ns == st.st_mtime_ns
    media.gpu_decode_ok(media.probe(video), 0.0)
    assert len(checks) == 2


# ------------------------------------------------------------------ only where the GPU path is available

@pytest.fixture(scope="module")
def gpu_here(clips) -> bool:
    return sys.platform == "win32" and bool(media._same_frames(media.probe(clips["h264.mp4"]), 0.0))


@pytest.fixture
def gpu(gpu_here):
    if not gpu_here:
        pytest.skip("GPU decoding (D3D11VA) is not available here")


@pytest.mark.parametrize("name", ["h264.mp4", "hevc.mp4", "full_range.mp4", "sar.mp4", "not_mod16.mp4", "vfr.mkv"])
def test_gpu_frames_are_software_frames(gpu, clips, monkeypatch, name):
    info = media.probe(clips[name])
    assert media.gpu_decode_ok(info, 0.0)
    for fps, width, start, end in [(3, info.width, 0.0, None), (2, 402, 0.7, None), (3, 320, info.duration - 0.2, None)]:
        assert frames(info, fps, width, start, end) == software(monkeypatch, lambda: frames(info, fps, width, start, end))
    path = clips[name].parent / f"chunk_{name}.mp4"
    assert chunk_bytes(info, path) == software(monkeypatch, lambda: chunk_bytes(info, path))


@pytest.mark.parametrize("name", ["rotated.mp4", "vflip.mp4", "ten_bit.mp4", "mpeg2.mpg"])
def test_gpu_not_used_where_it_would_differ(gpu, clips, monkeypatch, name):
    info = media.probe(clips[name])
    assert not media.gpu_decode_ok(info, 0.0)
    assert frames(info) == software(monkeypatch, lambda: frames(info))
    # The GPU chain itself, forced: on these it fails or gives other pixels.
    w, h = media.scaled_size(info, 320)
    tail = ["-frames:v", "3", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    sw = subprocess.run(media.decode_args(info, 0.3, 1, 3, w, h) + tail, capture_output=True)
    hw = subprocess.run(media.decode_args(info, 0.3, 1, 3, w, h, gpu=True) + tail, capture_output=True)
    assert hw.returncode != 0 or hw.stdout != sw.stdout


def test_real_gpu_dying_part_way(gpu, clips, monkeypatch):
    info = media.probe(clips["h264.mp4"])
    want = software(monkeypatch, lambda: frames(info, 3, 320, 0.2))
    assert media.gpu_decode_ok(info, 0.2)
    _die_after(monkeypatch, 2, 320 * 180 * 3, "d3d11va")
    assert frames(info, 3, 320, 0.2) == want
    assert not media.gpu_decode_ok(info, 0.2)  # software for this file from now on
    assert all(abs(t - (0.2 + k / 3)) < 1e-9 for k, (t, _) in enumerate(want))


@pytest.mark.parametrize("name", ["vflip_black.mp4", "sei_flip_black.mp4"])
def test_flip_behind_a_black_start(gpu, clips, monkeypatch, name):
    """ffmpeg flips software frames only. A black start looks the same flipped, so comparing frames there can't
    see it: the container's flag (vflip_black) and the stream's own (sei_flip_black, first frame only, so the
    first check at 0.6 s doesn't decode it) must keep these on software."""
    info = media.probe(clips[name])
    assert not media.gpu_decode_ok(info, 0.6)
    assert frames(info, 3, 320, 0.0) == software(monkeypatch, lambda: frames(info, 3, 320, 0.0))
    path = clips[name].parent / f"chunk_{name}"
    assert chunk_bytes(info, path, 0.0) == software(monkeypatch, lambda: chunk_bytes(info, path, 0.0))


def test_damaged_stream_falls_back_to_software(gpu, clips, monkeypatch):
    """Damage after the checked frames: the GPU run stops there and software takes over, for frames and SAM's
    copy alike, which then match software exactly (without -xerror the GPU hid the damage its own way)."""
    info = media.probe(clips["broken_nal.mp4"])
    want = software(monkeypatch, lambda: frames(info, 10, 640, 0.0))
    assert media.gpu_decode_ok(info, 0.0)
    assert frames(info, 10, 640, 0.0) == want
    assert media._gpu_ok and not any(media._gpu_ok.values())  # software for this file from now on
    media._gpu_ok.clear()
    assert media.gpu_decode_ok(info, 0.0)
    path = clips["broken_nal.mp4"].parent / "chunk_broken.mp4"
    assert chunk_bytes(info, path, 0.2) == software(monkeypatch, lambda: chunk_bytes(info, path, 0.2))


def test_check_compares_colour_too(gpu, clips, monkeypatch):
    """The check also compares the frames as BGR, which depends on the colour range and matrix the frames carry,
    not only their pixels: GPU frames tagged full range (same pixels) must fail it."""
    info = media.probe(clips["h264.mp4"])
    assert media._same_frames(info, 0.0)
    parts = media._gpu_parts
    monkeypatch.setattr(media, "_gpu_parts", lambda i: (parts(i)[0], parts(i)[1] + "setparams=range=pc,"))
    assert media._same_frames(info, 0.0) is False
