"""ffmpeg/ffprobe helpers. The original video is only ever read, never changed."""
from __future__ import annotations

import io
import json
import queue
import shutil
import subprocess
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


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
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if "rotate" in (video.get("tags") or {}):
        rotation = int(video["tags"]["rotate"])
    if abs(rotation) % 180 == 90:
        w, h = h, w
    duration = float(meta.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise MediaError(f"{path.name}: could not read the video duration")
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate")) or 25.0
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return VideoInfo(path, duration, w, h, fps, has_audio, rotation)


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


def iter_frames(info: VideoInfo, fps: float, width: int, start: float = 0.0,
                end: float | None = None, prefetch: int = 8) -> Iterator[tuple[float, np.ndarray]]:
    """Decode frames at `fps`, scaled to `width` (height keeps the aspect). Yields (t, BGR)."""
    s, e = clip_range(info, start, end)
    w, h = scaled_size(info, width)
    args = [_tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{s:.3f}", "-i", str(info.path),
            "-t", f"{e - s:.3f}", "-an", "-sn",
            "-vf", f"fps={fps}:round=up,scale={w}:{h}:flags=area,setsar=1",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
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
    k = 0
    try:
        while True:
            buf = frames.get()
            if buf is None:
                break
            yield s + k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        stop.set()
        if proc.poll() is None:
            proc.kill()
        while pumper.is_alive():
            try:
                frames.get(timeout=0.1)
            except queue.Empty:
                pass
        proc.stdout.close()
        proc.wait()
        err_reader.join(timeout=1)
    if k == 0:
        err = b"".join(err_chunks).decode("utf-8", "replace").strip().splitlines()
        raise MediaError(f"no frames decoded from {info.path.name}" + (f": {err[-1]}" if err else ""))


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
