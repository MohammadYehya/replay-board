"""Edge cases from SPEC section 10: rotated phone video, odd sizes, variable frame rate, no audio,
clips under 3 s, bad --start/--end, paths with spaces and accents. Errors must be one plain line."""
import subprocess
import sys

import numpy as np
import pytest

from app.core import media
from app.core.board import BoardParams, capture_board
from app.core.media import MediaError


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], check=True, capture_output=True, timeout=60)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("données lecture")  # spaces and an accent in the path
    src = "testsrc2=size=641x359:rate=25"
    out = {}
    out["odd"] = d / "odd size é.mp4"
    ffmpeg("-f", "lavfi", "-i", src, "-f", "lavfi", "-i", "sine=f=440:sample_rate=44100",
           "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv444p", "-vf", "scale=641:359,setsar=1", "-c:a", "aac",
           "-shortest", str(out["odd"]))
    out["silent"] = d / "no audio.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-t", "2", "-c:v", "libx264", "-pix_fmt",
           "yuv420p", str(out["silent"]))
    out["rotated"] = d / "phone.mp4"
    ffmpeg("-display_rotation", "90", "-i", str(out["silent"]), "-c", "copy", str(out["rotated"]))
    out["vfr"] = d / "vfr.mkv"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-t", "3",
           "-vf", "select='not(mod(n\\,3))+gt(n\\,45)',setpts=N/FRAME_RATE/TB", "-fps_mode", "vfr",
           "-c:v", "libx264", str(out["vfr"]))
    out["late_audio"] = d / "late audio.mp4"
    sine = d / "sine.m4a"
    ffmpeg("-f", "lavfi", "-t", "3", "-i", "sine=f=300:sample_rate=16000", "-c:a", "aac", str(sine))
    ffmpeg("-f", "lavfi", "-t", "4", "-i", "testsrc2=size=320x240:rate=25", "-itsoffset", "1", "-i", str(sine),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
           str(out["late_audio"]))
    return out


def test_odd_size_and_accented_path(clips):
    info = media.probe(clips["odd"])
    assert (info.width, info.height) == (641, 359) and info.has_audio
    frames = list(media.iter_frames(info, 3, 400))
    assert len(frames) >= 11 and frames[0][1].shape[1] == 400 and frames[0][1].shape[0] % 2 == 0
    pcm = media.extract_audio(info)
    assert pcm is not None and abs(pcm.size / 16000 - 4.0) < 0.1
    img = media.grab_frame(info, 1.0, 320)
    path = clips["odd"].parent / "frame à.png"
    media.imwrite(path, img)
    assert np.array_equal(media.imread(path), img)


def test_no_audio_track(clips):
    info = media.probe(clips["silent"])
    assert not info.has_audio and media.extract_audio(info) is None


def test_rotated_phone_video_reports_display_size(clips):
    info = media.probe(clips["rotated"])
    assert abs(info.rotation) == 90 and (info.width, info.height) == (240, 320)
    t, f = next(media.iter_frames(info, 3, 240))
    assert f.shape[:2] == (320, 240)


def test_variable_frame_rate(clips):
    info = media.probe(clips["vfr"])
    ts = [t for t, _ in media.iter_frames(info, 3, 160)]
    assert len(ts) >= 8 and np.allclose(np.diff(ts), 1 / 3)


def test_late_audio_stays_on_video_clock(clips):
    info = media.probe(clips["late_audio"])
    pcm = media.extract_audio(info).astype(np.float32)
    first_sound = np.argmax(np.abs(pcm) > 1000) / 16000
    assert 0.9 <= first_sound <= 1.15  # silence padded at the start, not shifted to 0


def test_bad_ranges_are_one_line(clips):
    info = media.probe(clips["silent"])
    for start, end in [(10.0, None), (1.5, 1.0)]:
        with pytest.raises(MediaError) as e:
            media.clip_range(info, start, end)
        assert "\n" not in str(e.value)
    with pytest.raises(MediaError, match="not found"):
        media.probe(clips["silent"].parent / "missing.mp4")


def test_board_capture_on_a_two_second_clip(clips):
    info = media.probe(clips["silent"])
    res = capture_board(info, params=BoardParams(width=320))
    assert res.width == 320 and res.stats["steps"] >= 1


def test_cli_errors_are_one_line(clips):
    for extra in (["--start", "99"], ["--start", "1.5", "--end", "1"]):
        r = subprocess.run([sys.executable, "-m", "app.process", str(clips["silent"]), *extra],
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 1
        lines = [x for x in r.stderr.strip().splitlines() if x.strip()]
        assert len(lines) == 1 and lines[0].startswith("error:"), r.stderr
