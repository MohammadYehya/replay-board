"""CLI: python -m app.process <video> [--start S] [--end E] [--no-sam] [--no-gestures] [--job DIR]

Writes data/jobs/<id>/replay.json plus piece images. Errors are one plain line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.media import MediaError
from .core.meta_api import MetaAPIError
from .pipeline import JobError, run


class _BadArgs(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """argparse, except that a bad argument is one plain line for the owner, not five lines of usage text."""

    def error(self, message: str):
        raise _BadArgs(message)


def _seconds(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number of seconds") from None
    if v != v or v in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError(f"'{text}' is not a number of seconds")
    return v


WIDTHS = (320, 3840)  # board analysis widths that make sense: pixel sizes are set for 800 (SPEC "units")


def _width(text: str) -> int:
    """A board analysis width, checked before anything runs: --width 0 or -5 went through probe, classify and the
    board's capture (paid calls with a real key) before failing with a numpy error."""
    try:
        v = int(text)
    except ValueError:
        v = None
    if v is None or not WIDTHS[0] <= v <= WIDTHS[1]:
        raise argparse.ArgumentTypeError(f"'{text}' is not a width in pixels from {WIDTHS[0]} to {WIDTHS[1]}")
    return v


def main(argv: list[str] | None = None) -> int:
    ap = _Parser(prog="python -m app.process", description="Process a recorded lecture.")
    ap.add_argument("video")
    ap.add_argument("--start", type=_seconds, default=None, help="start time in seconds")
    ap.add_argument("--end", type=_seconds, default=None, help="end time in seconds")
    ap.add_argument("--deck", default=None, help="PDF of the slides, for slide lectures (optional)")
    ap.add_argument("--job", default=None, help="job folder (default data/jobs/<name>-<hash>)")
    ap.add_argument("--no-sam", action="store_true", help="skip SAM (whole frame, built-in person detector)")
    ap.add_argument("--no-gestures", action="store_true",
                    help="skip the teacher's pointing on a board (saves about half the API cost)")
    ap.add_argument("--width", type=_width, default=800,
                    help=f"board analysis width, {WIDTHS[0]} to {WIDTHS[1]} (default 800)")
    ap.add_argument("--no-cache", action="store_true", help="ask the models again instead of using saved answers")
    try:
        a = ap.parse_args(argv)
    except _BadArgs as e:
        print(f"error: {e} (see python -m app.process --help)", file=sys.stderr)
        return 2
    try:
        path = run(a.video, Path(a.job) if a.job else None, a.start, a.end, use_sam=not a.no_sam, width=a.width,
                   use_cache=not a.no_cache, deck=a.deck, gestures=not a.no_gestures)
        print(f"wrote {path}")
        return 0
    except (MediaError, MetaAPIError, JobError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    except Exception as e:
        from .pipeline import plain_error
        print(f"error: {plain_error(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
