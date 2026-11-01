"""Speech (SPEC 5.3): split, transcribe with Muse Voice Transcribe, align words with MMS_FA."""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

RATE = 16000
# SPEC allows up to 540 s; shorter pieces transcribe in parallel and a failed one costs less.
MAX_PIECE_S = 180.0
SEARCH_S = 20.0
PAD_S = 0.15


def split_points(pcm: np.ndarray, rate: int = RATE, max_len: float | None = None,
                 search: float | None = None) -> list[tuple[int, int]]:
    """Cut into pieces of at most `max_len` s, at the quietest point in the last `search` s."""
    max_len = MAX_PIECE_S if max_len is None else max_len
    search = SEARCH_S if search is None else search
    n = pcm.size
    limit = int(max_len * rate)
    if n <= limit:
        return [(0, n)]
    hop = int(0.05 * rate)
    frames = n // hop
    x = pcm[: frames * hop].astype(np.float32).reshape(frames, hop)
    energy = np.sqrt((x * x).mean(1) + 1e-9)
    k = max(1, int(round(0.3 / 0.05)))
    smooth = np.convolve(energy, np.ones(k) / k, mode="same")
    pieces = []
    start = 0
    while n - start > limit:
        lo = (start + limit - int(search * rate)) // hop
        hi = (start + limit) // hop
        lo = max(lo, start // hop + 1)
        cut = (lo + int(np.argmin(smooth[lo:hi]))) * hop if hi > lo else start + limit
        pieces.append((start, cut))
        start = cut
    pieces.append((start, n))
    return pieces


def transcribe(pcm: np.ndarray, client, rate: int = RATE, max_parallel: int = 4,
               progress: Callable[[float], None] | None = None) -> list[dict]:
    """Turns [{start, end, text}] on the clip's clock. Pieces are transcribed in parallel."""
    from .media import wav_bytes

    pieces = split_points(pcm, rate)
    done = [0]
    lock = threading.Lock()

    def one(se):
        s, e = se
        reply = client.transcribe(wav_bytes(pcm[s:e], rate))
        off = s / rate
        out = []
        for turn in reply.get("turns", []) or []:
            text = (turn.get("transcript") or turn.get("text") or "").strip()
            if not text:
                continue
            a = off + float(turn.get("startMs", 0)) / 1000.0
            b = off + float(turn.get("endMs", turn.get("startMs", 0))) / 1000.0
            out.append({"start": round(a, 3), "end": round(max(a, b), 3), "text": text})
        with lock:
            done[0] += 1
            if progress:
                progress(done[0] / len(pieces))
        return out

    with ThreadPoolExecutor(max_workers=max(1, min(max(max_parallel, 6), len(pieces)))) as ex:
        results = list(ex.map(one, pieces))
    turns = [t for r in results for t in r]
    turns.sort(key=lambda t: t["start"])
    return turns


_NUM = re.compile(r"\d+(?:\.\d+)?")


def spoken_form(word: str) -> str:
    """Lowercase a-z and apostrophes; numbers spelled out ("2a" -> "two a")."""
    from num2words import num2words

    def num(m):
        s = m.group(0)
        try:
            return " " + num2words(float(s) if "." in s else int(s)) + " "
        except Exception:
            return " "

    w = _NUM.sub(num, word.lower().replace("’", "'"))
    w = re.sub(r"[^a-z' ]+", " ", w)
    w = re.sub(r"\s+", " ", w).strip(" '")
    return w


def _proportional(turn: dict, words: list[str]) -> list[dict]:
    """Fallback: share the turn's time by word length + 2."""
    if not words:
        return []
    weights = np.array([len(w) + 2 for w in words], float)
    edges = turn["start"] + (turn["end"] - turn["start"]) * np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
    return [{"w": w, "s": round(float(edges[i]), 3), "e": round(float(edges[i + 1]), 3)} for i, w in enumerate(words)]


class Aligner:
    """MMS forced aligner (torchaudio < 2.9). Loaded once; thread-safe via a lock."""

    def __init__(self):
        import warnings

        import torch
        import torchaudio

        # MMS_FA works through torchaudio 2.8; it only warns that 2.9 removes it (hence our pin).
        warnings.filterwarnings("ignore", message=".*forced_align has been deprecated.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
        self.torch = torch
        bundle = torchaudio.pipelines.MMS_FA
        self.rate = bundle.sample_rate
        self.model = bundle.get_model()
        self.model.eval()
        # int8 linear layers: ~1.4x faster on CPU; word times move by ~4 ms on average (p95 20 ms).
        try:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception:
            pass
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.lock = threading.Lock()

    def align_turn(self, pcm: np.ndarray, rate: int, turn: dict) -> list[dict]:
        words = turn["text"].split()
        if not words:
            return []
        spoken = [spoken_form(w) for w in words]
        a = max(0, int((turn["start"] - PAD_S) * rate))
        b = min(pcm.size, int((turn["end"] + PAD_S) * rate))
        if b - a < int(0.1 * rate):
            return _proportional(turn, words)
        # Each written word may become several spoken tokens ("2a" -> "two a").
        tok_words, owner = [], []
        for i, sp in enumerate(spoken):
            for part in sp.split():
                tok_words.append(part)
                owner.append(i)
        if not tok_words:
            return _proportional(turn, words)
        torch = self.torch
        wav = torch.from_numpy(pcm[a:b].astype(np.float32) / 32768.0).unsqueeze(0)
        if rate != self.rate:
            import torchaudio.functional as F
            wav = F.resample(wav, rate, self.rate)
        try:
            with self.lock, torch.inference_mode():
                emission, _ = self.model(wav)
                spans = self.aligner(emission[0], self.tokenizer(tok_words))
        except Exception:
            return _proportional(turn, words)
        sec_per_frame = wav.size(1) / emission.size(1) / self.rate
        t0 = a / rate
        start = [None] * len(words)
        end = [None] * len(words)
        for span, i in zip(spans, owner):
            if not span:
                continue
            s = t0 + span[0].start * sec_per_frame
            e = t0 + span[-1].end * sec_per_frame
            start[i] = s if start[i] is None else min(start[i], s)
            end[i] = e if end[i] is None else max(end[i], e)
        out = []
        for i, w in enumerate(words):
            if start[i] is None:
                # Symbols like "=": the gap between their neighbours.
                prev_e = next((end[j] for j in range(i - 1, -1, -1) if end[j] is not None), turn["start"])
                next_s = next((start[j] for j in range(i + 1, len(words)) if start[j] is not None), turn["end"])
                start[i], end[i] = prev_e, max(prev_e, next_s)
            out.append({"w": w, "s": round(float(start[i]), 3), "e": round(float(end[i]), 3)})
        return out


_ALIGNER: Aligner | None = None


def get_aligner() -> Aligner | None:
    global _ALIGNER
    if _ALIGNER is None:
        try:
            _ALIGNER = Aligner()
        except Exception:
            return None
    return _ALIGNER


def align_words(pcm: np.ndarray, turns: list[dict], rate: int = RATE,
                progress: Callable[[float], None] | None = None) -> tuple[list[dict], str]:
    """Word times for every turn. Returns (words, method) with method "mms" or "proportional"."""
    aligner = get_aligner()
    words: list[dict] = []
    for k, turn in enumerate(turns):
        if aligner is None:
            words += _proportional(turn, turn["text"].split())
        else:
            words += aligner.align_turn(pcm, rate, turn)
        if progress:
            progress((k + 1) / max(1, len(turns)))
    return words, ("mms" if aligner is not None else "proportional")
