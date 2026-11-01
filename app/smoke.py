"""M0 smoke test: one tiny call each to Muse Spark (with an image), Voice Transcribe and SAM 3.1.

    python -m app.smoke [video] [--at SECONDS] [--no-cache]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import DATA_DIR
from .core import media
from .core.meta_api import MetaAPIError, MetaClient

SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["board", "slides", "other"]},
        "board_colour": {"type": "string", "enum": ["white", "chalk", "none"]},
        "has_webcam_corner": {"type": "boolean"},
        "visible_text_sample": {"type": "string"},
    },
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.smoke")
    ap.add_argument("video", nargs="?", default=str(DATA_DIR / "test.mp4"))
    ap.add_argument("--at", type=float, default=120.0, help="time of the test frame and audio (s)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)

    try:
        info = media.probe(a.video)
        t = min(a.at, max(0.0, info.duration - 9))
        frame = media.grab_frame(info, t, 960)
        pcm = media.extract_audio(info, t, t + 8)
        client = MetaClient(use_cache=not a.no_cache)
        print("models:", ", ".join(client.list_models()))

        rows = []

        def run(name, fn):
            n0 = len(client.calls)
            t0 = time.perf_counter()
            try:
                out = fn()
                err = None
            except MetaAPIError as e:
                out, err = None, str(e)
            wall = time.perf_counter() - t0
            rec = client.calls[n0] if len(client.calls) > n0 else None
            rows.append((name, rec, wall, err))
            return out

        ans = run("Muse Spark", lambda: client.ask(
            "Is this a filmed board lecture or a slide screen recording? Answer in JSON. "
            "visible_text_sample: copy one short line of writing you can read.",
            [frame], schema=SCHEMA, reasoning="low"))
        if ans is not None:
            print("Muse Spark ->", ans)

        if pcm is None:
            print("Voice Transcribe -> skipped: the video has no sound track")
        else:
            asr = run("Voice Transcribe", lambda: client.transcribe(media.wav_bytes(pcm)))
            if asr is not None:
                turns = asr.get("turns") or []
                print(f"Voice Transcribe -> {len(turns)} turn(s):",
                      [(x.get("startMs"), x.get("endMs"), x.get("transcript")) for x in turns][:4])

        objs = run("SAM 3.1", lambda: client.segment(frame, "whiteboard"))
        if objs is not None:
            print(f"SAM 3.1 -> {len(objs)} object(s):",
                  [{"box": [round(v) for v in o["box"]], "score": o["score"],
                    "mask_px": int(o["mask"].sum())} for o in objs][:3])

        print()
        print(f"{'call':18} {'model':28} {'status':>6} {'latency':>8} {'tokens in/out':>14} {'cost $':>9}  note")
        for name, rec, wall, err in rows:
            if rec is None:
                print(f"{name:18} {'-':28} {'-':>6} {wall:7.2f}s {'-':>14} {'-':>9}  {err or ''}")
                continue
            u = rec.usage or {}
            toks = f"{u.get('prompt_tokens', u.get('input_tokens', '-'))}/{u.get('completion_tokens', u.get('output_tokens', '-'))}"
            cost = "-" if rec.cost_usd is None else f"{rec.cost_usd:.5f}"
            note = "cached" if rec.cached else (err or "")
            print(f"{name:18} {rec.model:28} {rec.status:>6} {rec.latency_s:7.2f}s {toks:>14} {cost:>9}  {note}")
        print("total:", client.summary())
        return 0 if all(r[3] is None for r in rows) else 1
    except (media.MediaError, MetaAPIError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
