"""The one module that talks to Meta's model API (SPEC section 4).

- Every answer is cached on disk, keyed by a hash of endpoint + model + request (+ uploaded bytes),
  so re-runs are free.
- 408/429/5xx and network errors are retried with exponential backoff and jitter (about 60 s in
  total), honouring Retry-After.
- Uploads are sent as bytes, so a retry resends the whole upload.
- Every call is recorded (latency, usage, estimated cost) in `client.calls` and in
  `<cache>/calls.jsonl`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ..config import Settings, load_settings

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
RETRY_BUDGET_S = 60.0
MIN_NETWORK_TRIES = 3  # timeouts and dropped connections are retried at least this many times

# USD, from dev.meta.ai/docs/pricing-rate-limits (checked 2026-11-01): per million tokens
# (input, cached input, output), per audio minute, or per image. Unknown models report cost None.
_SPARK = {"input": 1.25, "cached": 0.15, "output": 4.25}
PRICES: dict[str, dict[str, float]] = {
    "muse-spark-1.3": _SPARK, "muse-spark-1.2": _SPARK, "muse-spark-1.1": _SPARK,
    "muse-wonder-1.0": {"input": 4.00, "cached": 0.20, "output": 20.00},
    "muse-voice-transcribe-1.0": {"per_minute": 0.18 / 60},
    "sam-3.1": {"per_image": 2.50 / 1000, "per_video_frame": 0.20 / 1000},
}


class MetaAPIError(RuntimeError):
    """A failed model call, as one plain line."""


@dataclass
class CallRecord:
    endpoint: str
    model: str
    status: int
    latency_s: float
    cached: bool
    attempts: int
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    key: str = ""


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def image_data_url(img: Any, fmt: str = ".jpg", quality: int = 90, max_side: int | None = None) -> str:
    """BGR numpy image, encoded bytes, or a path -> data URL."""
    from .media import encode_image

    if isinstance(img, np.ndarray):
        import cv2

        if max_side and max(img.shape[:2]) > max_side:
            k = max_side / max(img.shape[:2])
            img = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))),
                             interpolation=cv2.INTER_AREA)
        data = encode_image(img, fmt, quality)
        mime = "image/png" if fmt == ".png" else "image/jpeg"
    else:
        if isinstance(img, (str, Path)):
            p = Path(img)
            data = p.read_bytes()
            fmt = p.suffix.lower()
        else:
            data = bytes(img)
        mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def image_part(img: Any, **kw) -> dict:
    return {"type": "image_url", "image_url": {"url": image_data_url(img, **kw)}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def strict_schema(schema: dict) -> dict:
    """Make a JSON schema strict: additionalProperties false and every property required."""
    s = json.loads(json.dumps(schema))

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node.setdefault("additionalProperties", False)
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    return s


class MetaClient:
    def __init__(self, settings: Settings | None = None, use_cache: bool = True, timeout: float = 180.0,
                 transport: httpx.BaseTransport | None = None):
        self.settings = settings or load_settings()
        if not self.settings.api_key:
            raise MetaAPIError("MODEL_API_KEY is missing: put it in .env (see .env.example).")
        self.use_cache = use_cache
        self.cache_dir = self.settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = httpx.Client(
            base_url=self.settings.api_base,
            headers={"Authorization": f"Bearer {self.settings.api_key}", "Accept": "application/json"},
            timeout=httpx.Timeout(timeout, connect=20.0),
            transport=transport,
        )
        self.calls: list[CallRecord] = []
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max(1, self.settings.max_parallel))

    # ---------- plumbing ----------
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _record(self, rec: CallRecord) -> None:
        with self._lock:
            self.calls.append(rec)
            try:
                with open(self.cache_dir / "calls.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({**asdict(rec), "at": time.time()}) + "\n")
            except OSError:
                pass

    def _send(self, endpoint: str, model: str, key_material: dict, send, cost_fn=None) -> dict:
        key = hashlib.sha256(_canonical({"endpoint": endpoint, "model": model, **key_material}).encode()).hexdigest()
        path = self._cache_path(key)
        if self.use_cache and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                self._record(CallRecord(endpoint, model, 200, 0.0, True, 0, cached.get("usage") or {}, 0.0, key))
                return cached
            except (OSError, json.JSONDecodeError):
                pass

        deadline = time.monotonic() + RETRY_BUDGET_S
        attempt = 0
        last_err = ""
        while True:
            attempt += 1
            t0 = time.perf_counter()
            status = 0
            try:
                with self._sem:
                    resp = send()
                status = resp.status_code
            except httpx.TransportError as e:
                resp = None
                last_err = f"network error: {type(e).__name__}"
            latency = time.perf_counter() - t0
            if resp is not None and status < 400:
                try:
                    data = resp.json()
                except ValueError:
                    raise MetaAPIError(f"{endpoint}: the reply was not JSON ({resp.text[:120]!r})") from None
                if not isinstance(data, dict):
                    raise MetaAPIError(f"{endpoint}: expected a JSON object, got {type(data).__name__}")
                usage = data.get("usage") or {}
                cost = cost_fn(data) if cost_fn else None
                self._record(CallRecord(endpoint, model, status, latency, False, attempt, usage, cost, key))
                if self.use_cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(data), encoding="utf-8")
                    tmp.replace(path)
                return data
            if resp is not None:
                last_err = f"HTTP {status}: {_error_text(resp)}"
                if status not in RETRY_STATUS:
                    self._record(CallRecord(endpoint, model, status, latency, False, attempt, {}, None, key))
                    raise MetaAPIError(f"{endpoint} ({model}) failed: {last_err}")
            wait = min(20.0, 1.0 * 2 ** (attempt - 1)) * (0.5 + random.random())
            ra = _retry_after(resp) if resp is not None else None
            if ra is not None:
                wait = max(wait, ra)
            # The budget bounds waiting, not slow attempts: a timed-out call still gets retried.
            now = time.monotonic()
            if resp is None and attempt < MIN_NETWORK_TRIES:
                deadline = max(deadline, now + wait)
            if now + wait > deadline:
                self._record(CallRecord(endpoint, model, status, latency, False, attempt, {}, None, key))
                raise MetaAPIError(f"{endpoint} ({model}) failed after {attempt} tries: {last_err}")
            time.sleep(wait)

    # ---------- API surface ----------
    def list_models(self) -> list[str]:
        r = self.http.get("/models")
        if r.status_code >= 400:
            raise MetaAPIError(f"/models failed: HTTP {r.status_code}: {_error_text(r)}")
        return [m.get("id") for m in r.json().get("data", [])]

    def chat(self, messages: list[dict], *, model: str | None = None, schema: dict | None = None,
             schema_name: str = "answer", temperature: float | None = None, max_tokens: int | None = None,
             reasoning: str | None = None, extra: dict | None = None) -> dict:
        """POST /chat/completions. Returns the raw reply.

        `reasoning` is Meta's reasoning_effort (minimal/low/medium/high/xhigh; "none" is rejected).
        Muse Spark is tuned for the default temperature, so we leave it unset unless asked.
        """
        model = model or self.settings.model_spark
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        if reasoning:
            body["reasoning_effort"] = reasoning
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": schema_name, "schema": strict_schema(schema), "strict": True}}
        if extra:
            body.update(extra)
        return self._send("/chat/completions", model, {"body": body},
                          lambda: self.http.post("/chat/completions", json=body),
                          lambda d: _chat_cost(model, d))

    def ask(self, prompt: str, images: list | tuple = (), *, schema: dict | None = None, system: str | None = None,
            model: str | None = None, **kw) -> Any:
        """One user turn (text + images). Returns a dict when `schema` is given, else text."""
        content: list[dict] = [text_part(prompt)] + [i if isinstance(i, dict) else image_part(i) for i in images]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        reply = self.chat(messages, model=model, schema=schema, **kw)
        text = chat_text(reply)
        if schema is None:
            return text
        return parse_json_object(text)

    def transcribe(self, wav: bytes, *, model: str | None = None, mode: str = "ENDPOINTING",
                   encoding: str = "WAV") -> dict:
        """POST /asr/transcribe (multipart). `wav` is the whole file as bytes."""
        model = model or self.settings.model_transcribe
        if not isinstance(wav, (bytes, bytearray)):
            raise TypeError("transcribe() needs the audio as bytes")
        wav = bytes(wav)
        req = {"model": model, "audioEncoding": encoding, "mode": mode}

        def send():
            # Fresh multipart from the bytes on every attempt, so a retry resends the whole upload.
            files = {"request": (None, json.dumps(req), "application/json"),
                     "audio": ("audio.wav", wav, "audio/wav")}
            return self.http.post("/asr/transcribe", files=files, headers={"Accept": "application/json"})

        seconds = max(0.0, (len(wav) - 44) / 32000.0)
        return self._send("/asr/transcribe", model, {"request": req, "audio_sha256": _hash_bytes(wav)}, send,
                          lambda d: _audio_cost(model, seconds))

    def segment_raw(self, image: Any, text: str, *, model: str | None = None, extra_content: list | None = None,
                    metadata: dict | None = None) -> dict:
        """POST /responses with SAM 3.1: a text prompt and an image. Returns the raw reply."""
        model = model or self.settings.model_sam
        url = image if isinstance(image, str) and image.startswith("data:") else image_data_url(image)
        content = [{"type": "input_text", "text": text}, {"type": "input_image", "image_url": url}]
        if extra_content:
            content += extra_content
        # Responses metadata values must be strings.
        body = {"model": model, "stream": False,
                "input": [{"type": "message", "role": "user", "content": content}],
                "metadata": metadata or {"include_confidence": "true", "mask_encoding": "one_bit"}}
        return self._send("/responses", model, {"body": body}, lambda: self.http.post("/responses", json=body),
                          lambda d: PRICES.get(model, {}).get("per_image"))

    def segment_video(self, video: bytes, text: str, *, model: str | None = None, frames: int = 0,
                      timeout: float = 900.0) -> list[dict]:
        """SAM 3.1 on a whole MP4 (sent inline, at most 50 MB): records for every frame.

        Billed per frame of the uploaded clip, so send a low-fps copy.
        """
        model = model or self.settings.model_sam
        if len(video) > 48_000_000:
            raise MetaAPIError("video for SAM is over the 50 MB inline limit; send a smaller copy")
        url = "data:video/mp4;base64," + base64.b64encode(video).decode("ascii")
        body = {"model": model, "stream": False,
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": text}, {"type": "input_video", "video_url": url}]}],
                "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"}}
        per_frame = PRICES.get(model, {}).get("per_video_frame", 0.0)
        reply = self._send("/responses", model, {"text": text, "video_sha256": _hash_bytes(video)},
                           lambda: self.http.post("/responses", json=body, timeout=timeout),
                           lambda d: max(50, frames) * per_frame if per_frame else None)
        return parse_sam_records(response_text(reply), "video")

    def segment(self, image: Any, text: str, *, model: str | None = None, min_score: float = 0.0) -> list[dict]:
        """SAM 3.1 objects for a noun phrase: [{id, box:[x0,y0,x1,y1] (half-open, image px), score, mask}].

        `mask` is a full-image boolean array (the API's mask covers only the box).
        """
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            import cv2
            from .media import imread

            img = imread(image, cv2.IMREAD_COLOR) if isinstance(image, (str, Path)) else None
            if img is None:
                raise TypeError("segment() needs a BGR image array or an image path")
            image, (h, w) = img, img.shape[:2]
        reply = self.segment_raw(image, text, model=model)
        return parse_sam_output(response_text(reply), (h, w), min_score)

    # ---------- reporting ----------
    def summary(self) -> dict:
        live = [c for c in self.calls if not c.cached]
        return {
            "calls": len(self.calls), "live": len(live), "cached": len(self.calls) - len(live),
            "seconds": round(sum(c.latency_s for c in live), 2),
            "cost_usd": round(sum(c.cost_usd or 0.0 for c in live), 4),
        }


def _error_text(resp: httpx.Response) -> str:
    try:
        d = resp.json()
        if isinstance(d, dict):
            e = d.get("error", d)
            if isinstance(e, dict):
                return str(e.get("message") or e.get("detail") or e.get("title") or e)[:300]
            return str(e)[:300]
    except ValueError:
        pass
    return resp.text[:300].replace("\n", " ")


def _retry_after(resp: httpx.Response) -> float | None:
    v = resp.headers.get("retry-after")
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(v).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def chat_text(reply: dict) -> str:
    """The assistant text of a chat reply (OpenAI-style or Llama-style)."""
    ch = reply.get("choices")
    if ch:
        msg = ch[0].get("message") or {}
        c = msg.get("content")
    else:
        msg = reply.get("completion_message") or {}
        c = msg.get("content")
    if isinstance(c, dict):
        c = c.get("text")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def response_text(reply: dict) -> str:
    """Output text of a Responses API reply."""
    if isinstance(reply.get("output_text"), str):
        return reply["output_text"]
    parts = []
    for item in reply.get("output", []) or []:
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                parts.append(c.get("text", ""))
    return "".join(parts)


def parse_sam_records(text: str, media: str = "image") -> list[dict]:
    """SAM 3.1 special-token output -> [{frame, id, score, box:[l,t,r,b] half-open, local: bool HxW}].

    Each mask covers only its box. Uses the fast decoder in sam_codec when its first mask matches
    Meta's meta-sam-parser exactly; otherwise parses everything with meta-sam-parser.
    """
    from meta_sam_parser import (CompletedOutcome, SegmentationMask, decode_mask_to_raster,
                                 image_segmentation_format, video_segmentation_format)

    from .sam_codec import fast_records

    try:
        fast = fast_records(text)
    except Exception:
        fast = None
    if fast is not None:
        if not fast:
            return []
        payload, w, h = fast[0]["_payload"]
        ref = np.frombuffer(decode_mask_to_raster(SegmentationMask("one_bit", payload, w, h)), np.uint8)
        if np.array_equal(ref.reshape(h, w).astype(bool), fast[0]["local"]):
            for r in fast:
                r.pop("_payload", None)
            return fast

    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    out = []
    for r in result.records:
        if r.kind != "mask":
            continue
        m = r.mask
        local = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width).astype(bool)
        out.append({"frame": r.frame.frame_index if r.frame is not None else 0, "id": r.object_id,
                    "score": r.confidence,
                    "box": [r.bounds.left, r.bounds.top, r.bounds.right, r.bounds.bottom], "local": local})
    return out


def paste_mask(rec: dict, size: tuple[int, int], out: np.ndarray | None = None) -> np.ndarray:
    """OR a record's box-local mask into a full (h, w) boolean image."""
    h, w = size
    full = np.zeros((h, w), bool) if out is None else out
    local = rec["local"]
    x0, y0 = int(np.floor(rec["box"][0])), int(np.floor(rec["box"][1]))
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(h, y0 + local.shape[0]), min(w, x0 + local.shape[1])
    if ys1 > ys0 and xs1 > xs0:
        full[ys0:ys1, xs0:xs1] |= local[ys0 - y0: ys1 - y0, xs0 - x0: xs1 - x0]
    return full


def parse_sam_output(text: str, size: tuple[int, int], min_score: float = 0.0) -> list[dict]:
    """Image output -> [{id, score, box, mask (full-image bool)}], best first."""
    out = []
    for r in parse_sam_records(text, "image"):
        if r["score"] is not None and r["score"] < min_score:
            continue
        out.append({"id": r["id"], "score": r["score"], "box": r["box"], "mask": paste_mask(r, size)})
    out.sort(key=lambda o: -(o["score"] if o["score"] is not None else 0))
    return out


def parse_json_object(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b <= a:
            raise MetaAPIError(f"model answer was not JSON: {text[:120]!r}") from None
        try:
            data = json.loads(t[a:b + 1])
        except json.JSONDecodeError:
            raise MetaAPIError(f"model answer was not JSON: {text[:120]!r}") from None
    if not isinstance(data, dict):
        raise MetaAPIError(f"model answer was JSON but not an object: {text[:120]!r}")
    return data


def _chat_cost(model: str, reply: dict) -> float | None:
    p = PRICES.get(model)
    u = reply.get("usage") or {}
    if not p:
        return None
    tin = u.get("prompt_tokens") or u.get("input_tokens") or 0
    tout = u.get("completion_tokens") or u.get("output_tokens") or 0
    details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    cached = min(tin, details.get("cached_tokens") or 0)
    return ((tin - cached) * p.get("input", 0) + cached * p.get("cached", 0)
            + tout * p.get("output", 0)) / 1e6


def _audio_cost(model: str, seconds: float) -> float | None:
    p = PRICES.get(model)
    if not p:
        return None
    return seconds / 60.0 * p.get("per_minute", 0)
