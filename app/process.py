"""CLI: python -m app.process <video> [--start S] [--end E] [--no-sam] [--job DIR]

Writes data/jobs/<id>/replay.json plus piece images. Errors are one plain line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.media import MediaError
from .core.meta_api import MetaAPIError
from .pipeline import JobError, run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.process", description="Process a recorded lecture.")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=None, help="start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="end time in seconds")
    ap.add_argument("--deck", default=None, help="PDF of the slides (slide lectures, milestone 3)")
    ap.add_argument("--job", default=None, help="job folder (default data/jobs/<name>-<hash>)")
    ap.add_argument("--no-sam", action="store_true", help="skip SAM (whole frame, built-in person detector)")
    ap.add_argument("--width", type=int, default=800, help="board analysis width (default 800)")
    a = ap.parse_args(argv)
    try:
        path = run(a.video, Path(a.job) if a.job else None, a.start, a.end, use_sam=not a.no_sam, width=a.width)
        print(f"wrote {path}")
        return 0
    except (MediaError, MetaAPIError, JobError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"error: {type(e).__name__}: {e} (details in the job folder's error.log)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
