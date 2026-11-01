"""SAM 3.1 jobs: find the board's 4 corners, and track the person in front of it."""
from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .media import MediaError, VideoInfo, _tool, clip_range, grab_frame, scaled_size
from .meta_api import MetaAPIError, MetaClient, paste_mask

BOARD_PROMPTS = {"white": ["whiteboard", "blackboard", "chalkboard", "projector screen"],
                 "chalk": ["blackboard", "chalkboard", "whiteboard", "projector screen"]}


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Largest region -> convex hull -> 4 corners (approxPolyDP; minAreaRect fallback)."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = np.column_stack(np.nonzero(lab == big)[::-1]).astype(np.int32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.12, 23):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _order(pts: np.ndarray) -> np.ndarray:
    from .board import order_corners
    return order_corners(pts)


def find_board(client: MetaClient, info: VideoInfo, start: float = 0.0, end: float | None = None,
               colour: str | None = None, width: int = 960, min_area: float = 0.08) -> dict | None:
    """Board corners in video pixels from SAM masks on 3 frames. None if no board is found."""
    s, e = clip_range(info, start, end)
    w, h = scaled_size(info, width)
    k = info.width / w
    times = [s + (e - s) * f for f in (0.5, 0.25, 0.75)]
    frames = {}

    def frame(t):
        if t not in frames:
            frames[t] = grab_frame(info, t, width)
        return frames[t]

    for prompt in BOARD_PROMPTS.get(colour or "white", BOARD_PROMPTS["white"]):
        objs = client.segment(frame(times[0]), prompt, min_score=0.5)
        objs = [o for o in objs if o["mask"].mean() >= min_area]
        if not objs:
            continue
        quads, scores = [], []
        for t in times:
            found = objs if t == times[0] else [o for o in client.segment(frame(t), prompt, min_score=0.5)
                                                 if o["mask"].mean() >= min_area]
            if not found:
                continue
            # Several panels of one board come back as several objects: join the big ones.
            biggest = max(o["mask"].sum() for o in found)
            union = np.zeros((h, w), bool)
            for o in found:
                if o["mask"].sum() >= 0.25 * biggest:
                    union |= o["mask"]
            q = quad_from_mask(union)
            if q is not None:
                quads.append(_order(q))
                scores.append(max(o["score"] or 0 for o in found))
        if not quads:
            continue
        corners = np.median(np.stack(quads), axis=0) * k
        return {"corners": [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
                "prompt": prompt, "score": round(float(np.mean(scores)), 3), "frames": len(quads)}
    return None


class PersonTrack:
    """Person masks sampled at `fps`, stored small and bit-packed. Call with a time to get a mask.

    The masks may still be arriving (SAM chunks run in the background): asking for a time whose
    chunk isn't back yet waits for it, so board capture can start before SAM has finished.
    """

    def __init__(self, start: float, fps: float, size: tuple[int, int]):
        self.start, self.fps = start, fps
        self.h, self.w = size
        self.frames: dict[int, np.ndarray] = {}  # sample index -> packed bits
        self.n = 0
        self.chunks: list[tuple[int, int, threading.Event]] = []  # (first sample, end sample, ready)
        self.failed = 0
        self.waited = 0.0

    def _wait_for(self, i: int) -> None:
        for a, b, ready in self.chunks:
            if a <= i < b:
                if not ready.is_set():
                    t0 = time.perf_counter()
                    ready.wait(timeout=900)
                    self.waited += time.perf_counter() - t0
                return

    def put(self, index: int, mask: np.ndarray) -> None:
        if mask.shape != (self.h, self.w):
            mask = cv2.resize(mask.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
        if mask.any():
            prev = self.frames.get(index)
            packed = np.packbits(mask.reshape(-1))
            self.frames[index] = packed if prev is None else (prev | packed)
        self.n = max(self.n, index + 1)

    def _get(self, i: int) -> np.ndarray | None:
        p = self.frames.get(i)
        return None if p is None else np.unpackbits(p, count=self.h * self.w).reshape(self.h, self.w).astype(bool)

    def __call__(self, t: float) -> np.ndarray | None:
        """Union of the samples just before and after t, to cover motion between samples."""
        x = (t - self.start) * self.fps
        i0 = int(np.floor(x + 1e-6))
        out = None
        for i in {i0, i0 + 1} if x - i0 > 1e-3 else {i0}:
            self._wait_for(i)
            m = self._get(i)
            if m is not None:
                out = m if out is None else (out | m)
        return out

    @property
    def coverage(self) -> float:
        return len(self.frames) / max(1, self.n)


def _encode_chunk(info: VideoInfo, start: float, dur: float, fps: float, width: int, out: Path) -> None:
    w, h = scaled_size(info, width)
    args = [_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", str(info.path),
            "-t", f"{dur:.3f}", "-an", "-sn", "-vf", f"fps={fps}:round=up,scale={w}:{h}:flags=area,setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-threads", "4", "-movflags", "+faststart", str(out)]
    res = subprocess.run(args, capture_output=True)
    if res.returncode != 0 or not out.exists():
        err = res.stderr.decode("utf-8", "replace").strip().splitlines()
        raise MediaError(f"could not make the low-fps copy for SAM: {err[-1] if err else 'ffmpeg error'}")


def track_people(client: MetaClient, info: VideoInfo, start: float = 0.0, end: float | None = None,
                 fps: float = 2.0, width: int = 640, store_width: int = 384, chunk_s: float = 120.0,
                 prompt: str = "person", min_score: float = 0.3, workdir: Path | None = None,
                 progress=None, background: bool = False):
    """SAM 3.1 video mode on a low-fps copy, in parallel chunks (in time order). Billed per frame sent.

    With background=True, returns (track, future) at once: the track's masks fill in as chunks come
    back, and asking it for a time waits for that chunk. The future's result is the number of
    chunks that failed.
    """
    s, e = clip_range(info, start, end)
    w, h = scaled_size(info, width)
    sw, sh = scaled_size(info, store_width)
    track = PersonTrack(s, fps, (sh, sw))
    track.n = int(np.floor((e - s) * fps)) + 1
    chunks = []
    t = s
    while t < e - 1e-3:
        d = min(chunk_s, e - t)
        chunks.append((t, d))
        t += d
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="rb_sam_"))
    tmp.mkdir(parents=True, exist_ok=True)
    done = [0]
    for cs, cd in chunks:
        a = int(round((cs - s) * fps))
        track.chunks.append((a, a + int(np.ceil(cd * fps)) + 1, threading.Event()))

    def one(i_cd):
        i, (cs, cd) = i_cd
        try:
            return _one(i, cs, cd)
        except Exception:
            track.failed += 1
            if not background:
                raise
            return 0
        finally:
            track.chunks[i][2].set()
            done[0] += 1
            if progress:
                progress(done[0] / len(chunks))

    def _one(i, cs, cd):
        path = tmp / f"sam_chunk_{i:03d}.mp4"
        _encode_chunk(info, cs, cd, fps, width, path)
        data = path.read_bytes()
        recs = client.segment_video(data, prompt, frames=int(np.ceil(cd * fps)))
        base = int(round((cs - s) * fps))
        by_frame: dict[int, np.ndarray] = {}
        for r in recs:
            if r["score"] is not None and r["score"] < min_score:
                continue
            m = by_frame.setdefault(r["frame"], np.zeros((h, w), bool))
            paste_mask(r, (h, w), m)
        for f, m in by_frame.items():
            track.put(base + f, m)
        try:
            path.unlink()
        except OSError:
            pass
        return len(by_frame)

    ex = ThreadPoolExecutor(max_workers=min(6, len(chunks)))
    futures = [ex.submit(one, ic) for ic in enumerate(chunks)]  # submitted in time order
    ex.shutdown(wait=False)
    if background:
        from concurrent.futures import Future

        result: Future = Future()

        def finish():
            for f in futures:
                f.exception()
            result.set_result(track.failed)

        threading.Thread(target=finish, daemon=True).start()
        return track, result
    for f in futures:
        f.result()
    return track


class HandTrack:
    """SAM 3.1 'hand' detections sampled at `fps`: per sample, each hand's box and box-local mask (video px)."""

    def __init__(self, start: float, fps: float):
        self.start, self.fps = start, fps
        self.samples: dict[int, list[dict]] = {}  # sample index -> [{box, local, score}]

    def at(self, i: int) -> list[dict]:
        return self.samples.get(i, [])

    def times(self) -> list[tuple[float, list[dict]]]:
        return [(self.start + i / self.fps, hs) for i, hs in sorted(self.samples.items())]


def track_hands(client: MetaClient, info: VideoInfo, start: float = 0.0, end: float | None = None,
                fps: float = 3.0, width: int = 640, chunk_s: float = 120.0, min_score: float = 0.4,
                workdir: Path | None = None, progress=None) -> HandTrack:
    """The teacher's hands with SAM 3.1 video mode on a low-fps copy, in parallel chunks.

    Boxes and masks are scaled back to original video pixels. Billed per frame sent.
    """
    s, e = clip_range(info, start, end)
    w, h = scaled_size(info, width)
    k = info.width / w
    track = HandTrack(s, fps)
    chunks, t = [], s
    while t < e - 1e-3:
        d = min(chunk_s, e - t)
        chunks.append((t, d))
        t += d
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="rb_hands_"))
    tmp.mkdir(parents=True, exist_ok=True)
    done = [0]
    lock = threading.Lock()

    def one(i_cd):
        i, (cs, cd) = i_cd
        path = tmp / f"hand_chunk_{i:03d}.mp4"
        _encode_chunk(info, cs, cd, fps, width, path)
        recs = client.segment_video(path.read_bytes(), "hand", frames=int(np.ceil(cd * fps)))
        base = int(round((cs - s) * fps))
        out: dict[int, list[dict]] = {}
        for r in recs:
            if r["score"] is not None and r["score"] < min_score:
                continue
            x0, y0, x1, y1 = r["box"]
            out.setdefault(base + r["frame"], []).append(
                {"box": [x0 * k, y0 * k, x1 * k, y1 * k], "local": r["local"], "scale": k, "score": r["score"]})
        with lock:
            for f, hs in out.items():
                track.samples.setdefault(f, []).extend(hs)
            done[0] += 1
            if progress:
                progress(done[0] / len(chunks))
        try:
            path.unlink()
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as ex:
        list(ex.map(one, enumerate(chunks)))
    return track
