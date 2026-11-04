"""ffmpeg/ffprobe helpers. The original video is only ever read, never changed."""
from __future__ import annotations

import io
import json
import logging
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..config import load_settings

log = logging.getLogger(__name__)


class MediaError(Exception):
    """A problem with the input video, reported to the owner as one plain line."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int  # display size, after rotation and sample aspect ratio
    height: int
    fps: float
    has_audio: bool
    rotation: int
    codec: str = ""  # ffprobe's codec_name, e.g. "h264"
    pix_fmt: str = ""  # what the software decoder gives, e.g. "yuv420p"
    display_matrix: bool = False  # the file asks players to turn or flip the picture (a flip has rotation 0)


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise MediaError(f"{name} was not found on PATH; install ffmpeg and reopen the terminal.")
    return exe


def _run(args: list[str], what: str) -> bytes:
    try:
        res = subprocess.run(args, capture_output=True, check=False)
    except OSError as e:
        raise MediaError(f"could not run {args[0]} for {what}: {e}") from None
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", "replace").strip().splitlines()
        raise MediaError(f"{what} failed: {err[-1] if err else 'ffmpeg error'}")
    return res.stdout


def _ratio(text: str | None) -> float:
    if not text or text in ("0/0", "0:1", "N/A"):
        return 0.0
    for sep in ("/", ":"):
        if sep in text:
            a, b = text.split(sep, 1)
            try:
                return float(a) / float(b) if float(b) else 0.0
            except ValueError:
                return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"video not found: {path}")
    out = _run([_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
                "-show_streams", str(path)], f"reading {path.name}")
    meta = json.loads(out.decode("utf-8", "replace") or "{}")
    streams = meta.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise MediaError(f"{path.name} has no video track")
    w, h = int(video.get("width") or 0), int(video.get("height") or 0)
    if w <= 0 or h <= 0:
        raise MediaError(f"{path.name}: could not read the video size")
    sar = _ratio(video.get("sample_aspect_ratio"))
    if sar > 0 and abs(sar - 1) > 1e-3:
        w = int(round(w * sar))
    rotation, display_matrix = 0, False
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
        # Listed only when it isn't the identity: a turn, or a flip that ffprobe reports as rotation 0
        if "displaymatrix" in sd or str(sd.get("side_data_type", "")).lower().replace(" ", "") == "displaymatrix":
            display_matrix = True
    if "rotate" in (video.get("tags") or {}):
        rotation = int(video["tags"]["rotate"])
    if abs(rotation) % 180 == 90:
        w, h = h, w
    duration = float(meta.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise MediaError(f"{path.name}: could not read the video duration")
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate")) or 25.0
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return VideoInfo(path, duration, w, h, fps, has_audio, rotation, str(video.get("codec_name") or ""),
                     str(video.get("pix_fmt") or ""), display_matrix)


def clip_range(info: VideoInfo, start: float | None, end: float | None) -> tuple[float, float]:
    s = max(0.0, float(start or 0.0))
    e = info.duration if end is None else min(float(end), info.duration)
    if s >= info.duration:
        raise MediaError(f"--start {s:.1f}s is past the end of the video ({info.duration:.1f}s)")
    if e <= s:
        raise MediaError(f"--end {e:.1f}s is not after --start {s:.1f}s")
    return s, e


def even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def scaled_size(info: VideoInfo, width: int) -> tuple[int, int]:
    width = even(width)
    return width, even(width * info.height / info.width)


# ------------------------------------------------------------------ GPU decoding
# Decoding a 1080p lecture is most of board capture's ffmpeg work and of the low-fps copies made for SAM. Windows'
# D3D11VA lets the GPU decode instead; the fps filter then runs on the GPU frames, so only the frames used are
# copied back. It is used only where it gives the very same pixels as software decoding: model answers are cached
# by the bytes sent (board pieces, SAM's copies), so one changed pixel would mean paying for them again, and
# different board results.

HWACCEL = "d3d11va"  # tests point this at a name ffmpeg doesn't know, to check the fallback
# Codecs whose standard fixes the decoded pixels to the bit (integer transforms), so any conforming decoder,
# GPU or not, must give the same frames. MPEG-2 and MPEG-4 part 2 leave the IDCT's rounding to the decoder.
_GPU_CODECS = ("h264", "hevc", "vp9", "av1")
_GPU_PIX_FMTS = ("yuv420p", "yuvj420p")  # 8-bit 4:2:0: the GPU hands it back as nv12, losslessly
_gpu_ok: dict[tuple, bool] = {}  # (path, size, mtime, probed facts) -> checked result, for this process
_gpu_lock = threading.Lock()
_warned: set[str] = set()  # VIDEO_DECODE values already warned about
_FFMPEG_PREFIX = re.compile(r"^(?:\[[^\]]* @ [^\]]*\]\s*)+(?:Terminating thread with error: )?")


def last_line(stderr: bytes) -> str:
    """The last line ffmpeg printed, without its "[h264 @ 000001c5...]" prefixes: one plain line for the owner."""
    lines = stderr.decode("utf-8", "replace").strip().splitlines()
    return _FFMPEG_PREFIX.sub("", lines[-1].strip()) if lines else ""


def _gpu_parts(info: VideoInfo) -> tuple[list[str], str]:
    """Input options and the filters that bring GPU frames back as the software decoder's own pixel format, so
    scaling and colour conversion after it run exactly the same code as with software decoding. -xerror: damaged
    data stops the GPU run, so software takes over, rather than the GPU hiding the damage in its own way (software
    hides it differently)."""
    return (["-xerror", "-hwaccel", HWACCEL, "-hwaccel_output_format", "d3d11"],
            f"hwdownload,format=nv12,format={info.pix_fmt},")


def decode_args(info: VideoInfo, start: float, dur: float, fps: float, w: int, h: int, gpu: bool = False,
                overwrite: bool = False) -> list[str]:
    """ffmpeg up to its output options: `dur` seconds from `start` at `fps`, scaled to w x h. gpu=True decodes
    with D3D11VA (see gpu_decode_ok)."""
    pre, back = _gpu_parts(info) if gpu else ([], "")
    vf = f"fps={fps}:round=up,{back}scale={w}:{h}:flags=area,setsar=1"
    return [_tool("ffmpeg"), "-v", "error", "-nostdin", *(["-y"] if overwrite else []), *pre,
            "-ss", f"{start:.3f}", "-i", str(info.path), "-t", f"{dur:.3f}", "-an", "-sn", "-vf", vf]


def decode_setting() -> str:
    """VIDEO_DECODE: "auto" (default; the GPU where it gives the same frames) or "cpu" (never the GPU); words like
    "off" or "gpu" are understood too (config). Any other value means cpu, with one warning. Read on every call,
    so a running server picks up a change in .env on its next job."""
    mode = load_settings().video_decode
    if mode in ("auto", "cpu"):
        return mode
    if mode not in _warned:
        _warned.add(mode)
        log.warning("VIDEO_DECODE=%s is not auto or cpu; decoding in software", mode)
    return "cpu"


def _file_key(info: VideoInfo) -> tuple | None:
    """Which file a check's result belongs to. The probed facts are part of it, so another video copied over this
    one with the same size and modified time is checked again."""
    try:
        st = info.path.stat()
    except OSError:
        return None
    return (str(info.path.resolve()), st.st_size, st.st_mtime_ns, info.duration, info.width, info.height, info.fps,
            info.codec, info.pix_fmt)


def gpu_decode_ok(info: VideoInfo, start: float) -> bool:
    """Whether to decode this video on the GPU: Windows, a codec and pixel format that can match software, no
    display matrix (ffmpeg turns and flips software frames, not GPU frames: they came out different), and a check
    that the frames do match (_same_frames). The check is done once per file in this process. Any failure means
    software, with one log line."""
    if (sys.platform != "win32" or info.codec not in _GPU_CODECS or info.pix_fmt not in _GPU_PIX_FMTS
            or info.rotation % 360 or info.display_matrix or decode_setting() == "cpu"):
        return False
    key = _file_key(info)
    if key is None:
        return False
    with _gpu_lock:  # one check per file, even when SAM's chunks and the capture all start at once
        ok = _gpu_ok.get(key)
        if ok is None:
            ok = _same_frames(info, start)
            if ok is not None:
                _gpu_ok[key] = ok
        return bool(ok)


def gpu_decode_failed(info: VideoInfo, why: str) -> None:
    """A GPU run failed after its check passed: software for this file from now on. `why` completes "GPU decoding
    ...", e.g. "stopped after 9 frames (Invalid data found when processing input)"."""
    key = _file_key(info)
    if key is None:
        return
    with _gpu_lock:
        first = _gpu_ok.get(key) is not False
        _gpu_ok[key] = False
    if first:
        log.warning("%s: GPU decoding %s; decoding in software", info.path.name, why)


def _same_frames(info: VideoInfo, start: float, n: int = 8) -> bool | None:
    """n frames at `start`, decoded both ways at full size: True if every timestamp and checksum matches, in the
    software decoder's pixel format (what SAM's copies are scaled from) and as BGR (which also depends on the
    colour range and matrix the frames carry). False as well if a frame read asks for the picture to be turned or
    flipped, which ffmpeg does to software frames only: a start that looks the same flipped (black, say) would hide
    it from the comparison. The frames read for that are the first of the video and those from the keyframe
    before `start`, which is where ffmpeg takes it from. None if software gives no frame at `start` (too near the
    end to tell)."""
    with tempfile.TemporaryDirectory(prefix="replay-gpu-check-", ignore_cleanup_errors=True) as tmp:
        bgr = [Path(tmp) / "cpu.framemd5", Path(tmp) / "gpu.framemd5"]

        def args(gpu: bool) -> list[str]:
            pre, back = _gpu_parts(info) if gpu else ([], "")
            outs = []
            for pix_fmt, dest in ((info.pix_fmt, "-"), ("bgr24", str(bgr[gpu]))):
                outs += ["-an", "-sn", *(["-vf", back.rstrip(",")] if gpu else []), "-frames:v", str(n),
                         "-fps_mode", "passthrough", "-enc_time_base", "demux", "-pix_fmt", pix_fmt, "-f", "framemd5",
                         dest]
            return [_tool("ffmpeg"), "-v", "error", "-nostdin", *pre, "-ss", f"{start:.3f}", "-i", str(info.path),
                    *outs]

        turned = [_tool("ffprobe"), "-v", "error", "-select_streams", "V", "-read_intervals",
                  f"%+#1,{start:.3f}%+#{n}", "-show_entries", "frame_side_data=side_data_type", "-of", "csv=p=0",
                  str(info.path)]
        procs: list[subprocess.Popen] = []
        try:
            procs = [subprocess.Popen(a, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                     for a in (args(False), args(True), turned)]
            (sw, _), (hw, hw_err), (sides, _) = (p.communicate(timeout=60) for p in procs)
            if procs[0].returncode != 0 or not any(line and not line.startswith(b"#") for line in sw.splitlines()):
                return None
            if procs[2].returncode != 0:
                why = "could not read the frames' side data"
            elif b"displaymatrix" in sides.lower().replace(b" ", b""):
                why = "the video asks for its picture to be turned or flipped"
            elif procs[1].returncode != 0:
                why = last_line(hw_err) or "ffmpeg error"
            elif hw != sw or bgr[1].read_bytes() != bgr[0].read_bytes():
                why = "frames differ from software decoding"
            else:
                return True
        except (OSError, subprocess.SubprocessError) as e:
            for p in procs:
                p.kill()
                p.wait()
            why = f"check failed: {e if isinstance(e, OSError) else 'ffmpeg took too long'}"
    log.info("%s: not decoding on the GPU (%s)", info.path.name, why)
    return False


def _read_frames(args: list[str], w: int, h: int, prefetch: int, status: dict) -> Iterator[np.ndarray]:
    """BGR frames from an ffmpeg writing rawvideo to stdout. At the end, status gets its exit code ("rc") and the
    last line it printed ("err")."""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 4)
    err_chunks: list[bytes] = []
    err_reader = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()), daemon=True)
    err_reader.start()
    size = w * h * 3
    # Read ahead on a thread so ffmpeg keeps decoding while the caller processes a frame.
    frames: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def pump():
        try:
            while not stop.is_set():
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                frames.put(buf)
        finally:
            frames.put(None)

    pumper = threading.Thread(target=pump, daemon=True)
    pumper.start()
    ended = False
    try:
        while True:
            buf = frames.get()
            if buf is None:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        ended = True
    finally:
        stop.set()
        if not ended and proc.poll() is None:
            proc.kill()  # the caller stopped early
        while pumper.is_alive():
            try:
                frames.get(timeout=0.1)
            except queue.Empty:
                pass
        proc.stdout.close()
        try:
            status["rc"] = proc.wait(timeout=60)  # at the end of the output: its own exit code, not a kill's
        except subprocess.TimeoutExpired:
            proc.kill()
            status["rc"] = proc.wait()
        err_reader.join(timeout=1)
        status["err"] = last_line(b"".join(err_chunks))


def iter_frames(info: VideoInfo, fps: float, width: int, start: float = 0.0,
                end: float | None = None, prefetch: int = 8) -> Iterator[tuple[float, np.ndarray]]:
    """Decode frames at `fps`, scaled to `width` (height keeps the aspect). Yields (t, BGR).

    On the GPU where gpu_decode_ok says it gives the same frames. If the GPU's ffmpeg fails part way, a software
    run carries on from the next frame: frame k is at start + k/fps and has the pixels software alone gives it.
    That run starts at `start` like a plain software run and drops the frames already given: starting it later,
    or dropping them inside ffmpeg, changes which frames ffmpeg's -t lets through at the end."""
    s, e = clip_range(info, start, end)
    w, h = scaled_size(info, width)
    out = ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    k = 0
    status: dict = {}
    if gpu_decode_ok(info, s):
        with closing(_read_frames(decode_args(info, s, e - s, fps, w, h, gpu=True) + out, w, h, prefetch,
                                  status)) as frames:
            for frame in frames:
                yield s + k / fps, frame
                k += 1
        if status["rc"] == 0 and k:
            return
        if status["rc"] != 0:
            gpu_decode_failed(info, f"stopped after {k} frames ({status['err'] or 'no message from ffmpeg'})")
    given = k  # by the GPU run
    with closing(_read_frames(decode_args(info, s, e - s, fps, w, h) + out, w, h, prefetch, status)) as frames:
        for i, frame in enumerate(frames):
            if i >= given:
                yield s + i / fps, frame
                k = i + 1
    if k == 0:
        raise MediaError(f"no frames decoded from {info.path.name}" + (f": {status['err']}" if status["err"] else ""))


def grab_frame(info: VideoInfo, t: float, width: int) -> np.ndarray:
    """One frame at time t (accurate seek), scaled to `width`."""
    t = min(max(0.0, t), max(0.0, info.duration - 0.05))
    w, h = scaled_size(info, width)
    out = _run([_tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(info.path),
                "-frames:v", "1", "-an", "-sn", "-vf", f"scale={w}:{h}:flags=area,setsar=1",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], f"reading a frame at {t:.1f}s")
    if len(out) < w * h * 3:
        # Seeking right at the end can return nothing; step back a little.
        if t > 0.5:
            return grab_frame(info, t - 0.5, width)
        raise MediaError(f"could not read a frame at {t:.1f}s from {info.path.name}")
    return np.frombuffer(out[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def extract_audio(info: VideoInfo, start: float = 0.0, end: float | None = None,
                  rate: int = 16000) -> np.ndarray | None:
    """Mono int16 PCM on the video's clock (sample 0 = `start`). None if there is no sound track."""
    if not info.has_audio:
        return None
    s, e = clip_range(info, start, end)
    out = _run([_tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{s:.3f}", "-i", str(info.path),
                "-t", f"{e - s:.3f}", "-vn", "-sn", "-ac", "1", "-ar", str(rate),
                "-af", "aresample=async=1:first_pts=0", "-f", "s16le", "-acodec", "pcm_s16le", "-"],
               "extracting the sound")
    pcm = np.frombuffer(out, np.int16).copy()
    return pcm if pcm.size else None


def wav_bytes(pcm: np.ndarray, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.ascontiguousarray(pcm, np.int16).tobytes())
    return buf.getvalue()


# cv2.imread/imwrite fail silently on non-ASCII Windows paths: go through bytes.
def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> None:
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", img, params or [])
    if not ok:
        raise MediaError(f"could not encode image {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def imread(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise MediaError(f"could not read image {Path(path).name}")
    return img


def encode_image(img: np.ndarray, fmt: str = ".png", quality: int = 90) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(fmt, img, params)
    if not ok:
        raise MediaError("could not encode an image")
    return buf.tobytes()
