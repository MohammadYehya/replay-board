"""LaTeX formulas as crisp transparent PNGs, for the PowerPoint export.

One node process per batch runs web/scripts/tex2png.mjs: MathJax 3 typesets each formula (TeX in, SVG out, no
browser) and resvg rasterises it at about 300 dpi for a 28-pt display size. Each picture's size is given in em of
the formula's font size, so the layout can size it for any font size.

Results are cached on disk (data/cache/tex/<sha>.png + .json), so a deck is typeset once. A formula that does not
typeset is cached as a failure too. When node, the script or its packages are missing, nothing is cached: those
formulas are None, the PowerPoint shows them as Unicode text, and one plain line is logged once.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..config import DATA_DIR, ROOT

RENDER_VERSION = "1"   # change it when the picture rules change, so every formula is typeset again
PX_PER_EM = 120         # 28 pt at 300 dpi is 117 px per em
SCRIPT = ROOT / "web" / "scripts" / "tex2png.mjs"
CACHE_DIR = DATA_DIR / "cache" / "tex"
MAX_LATEX = 2000

log = logging.getLogger(__name__)
_warned: set[str] = set()
_warn_lock = threading.Lock()
_last_run_failed = False  # the last batch could not be typeset at all (node crashed, timed out, a package missing)

Runner = Callable[[dict, float], dict]


@dataclass(frozen=True)
class TexImage:
    """A typeset formula: a transparent PNG and its size in em of the formula's font size."""
    latex: str
    path: Path
    width_em: float
    height_em: float
    depth_em: float          # below the baseline
    px: tuple[int, int]

    def size_pt(self, font_pt: float) -> tuple[float, float]:
        """(width, height) in points when the formula is set at this font size."""
        return self.width_em * font_pt, self.height_em * font_pt


class TexRenderError(Exception):
    """The renderer could not run (no node, no script, a missing package, a crash): one plain line."""


def warn_once(message: str) -> None:
    with _warn_lock:
        if message in _warned:
            return
        _warned.add(message)
    log.warning(message)


def _key(latex: str, color: str, display: bool) -> str:
    raw = json.dumps([RENDER_VERSION, PX_PER_EM, latex, color.upper(), bool(display)], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _write(path: Path, data: bytes) -> bool:
    """Whole or not at all (another thread may be writing the same formula). False when the file is not there
    afterwards (a full disk, a folder that cannot be made): the cache is a help, never a reason to fail."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return path.is_file()  # replacing fails on Windows while another thread reads the same, finished file


def _num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _from_meta(latex: str, png: Path, meta: dict) -> TexImage | None:
    w, h, d, px = meta.get("width_em"), meta.get("height_em"), meta.get("depth_em"), meta.get("px")
    if not (_num(w) and _num(h) and w > 0 and h > 0 and isinstance(px, list) and len(px) == 2
            and all(isinstance(v, int) and v > 0 for v in px)):
        return None
    return TexImage(latex, png, float(w), float(h), float(d) if _num(d) else 0.0, (px[0], px[1]))


def _cached(cache: Path, key: str, latex: str) -> tuple[bool, TexImage | None]:
    """(found, image): a cached failure is (True, None)."""
    meta_path = cache / f"{key}.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, None
    if not isinstance(meta, dict):
        return False, None
    if meta.get("error"):
        return True, None
    png = cache / f"{key}.png"
    img = _from_meta(latex, png, meta) if png.is_file() else None
    return (img is not None), img


def available() -> bool:
    """Node, the script and its two packages are there (so formulas can be typeset; cheap to ask)."""
    modules = ROOT / "web" / "node_modules"
    return (shutil.which("node") is not None and SCRIPT.is_file() and (modules / "mathjax-full").is_dir()
            and (modules / "@resvg" / "resvg-js").is_dir())


def last_run_failed() -> bool:
    """The last batch in this process could not be typeset at all (a crash, a timeout): its formulas were left as
    text, though available() may say yes. Cleared by the next batch that runs."""
    return _last_run_failed


def node_runner(payload: dict, timeout: float) -> dict:
    """Run tex2png.mjs once for the whole batch."""
    node = shutil.which("node")
    if not node:
        raise TexRenderError("node was not found")
    if not SCRIPT.is_file():
        raise TexRenderError("web/scripts/tex2png.mjs is missing")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        proc = subprocess.run([node, str(SCRIPT)], input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              capture_output=True, timeout=timeout, cwd=str(SCRIPT.parent), creationflags=flags)
    except subprocess.TimeoutExpired:
        raise TexRenderError(f"typesetting took longer than {timeout:.0f} s") from None
    except OSError as e:
        raise TexRenderError(f"node could not start ({e.strerror or e})") from None
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise TexRenderError(err[0][:200] if err else f"node stopped with exit code {proc.returncode}")
    try:
        data = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise TexRenderError("the typesetter's answer was not JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise TexRenderError("the typesetter's answer had no results")
    return data


def render(latexes: Iterable[str], *, color: str = "#1F2A37", display: bool = True,
           cache_dir: Path | None = None, runner: Runner | None = None,
           timeout: float | None = None) -> dict[str, TexImage | None]:
    """{latex: TexImage, or None when it cannot be typeset}, for every formula given (one node run for those not
    cached yet). The colour is '#RRGGBB'."""
    cache = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    color = color if isinstance(color, str) and len(color) == 7 and color.startswith("#") else "#1F2A37"
    out: dict[str, TexImage | None] = {}
    todo: dict[str, list[str]] = {}  # key -> the formulas given for it (the same once trimmed)
    for latex in latexes:
        if not isinstance(latex, str) or latex in out:
            continue
        src = latex.strip()
        if not src or len(src) > MAX_LATEX:
            out[latex] = None
            continue
        key = _key(src, color, display)
        found, img = _cached(cache, key, latex)
        if found:
            out[latex] = img
        else:
            out[latex] = None
            todo.setdefault(key, []).append(latex)
    if not todo:
        return out
    items = [{"id": k, "latex": v[0].strip(), "display": bool(display), "color": color, "px_per_em": PX_PER_EM}
             for k, v in todo.items()]
    global _last_run_failed
    try:
        data = (runner or node_runner)({"items": items}, timeout or 30.0 + 1.0 * len(items))
    except TexRenderError as e:
        _last_run_failed = True
        warn_once(f"Equations are shown as plain text in the PowerPoint: {e}.")
        return out
    except Exception as e:  # a fake runner, or anything unexpected: never break the PowerPoint over a formula
        _last_run_failed = True
        warn_once(f"Equations are shown as plain text in the PowerPoint: {type(e).__name__}: {str(e)[:120]}.")
        return out
    _last_run_failed = False
    failed = 0
    for r in data.get("results") or []:
        if not isinstance(r, dict) or r.get("id") not in todo:
            continue
        key = r["id"]
        latex = todo[key][0].strip()
        if not r.get("ok"):
            failed += 1
            _write(cache / f"{key}.json", json.dumps({"latex": latex, "error": str(r.get("error") or "")[:200]},
                                                     ensure_ascii=False).encode("utf-8"))
            continue
        try:
            png = base64.b64decode(str(r.get("png") or ""), validate=True)
            from PIL import Image
            with Image.open(io.BytesIO(png)) as im:
                im.verify()
        except Exception:
            failed += 1
            continue  # a broken picture is not cached: the next deck tries again
        meta = {"latex": latex, "color": color, "display": bool(display), "width_em": r.get("width_em"),
                "height_em": r.get("height_em"), "depth_em": r.get("depth_em"), "px": r.get("px")}
        if _from_meta(latex, cache / f"{key}.png", meta) is None or not _write(cache / f"{key}.png", png):
            failed += 1
            continue
        _write(cache / f"{key}.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        for given in todo[key]:
            out[given] = _from_meta(given, cache / f"{key}.png", meta)
    if failed:
        warn_once("Some equations could not be typeset; the PowerPoint shows them as plain text.")
    return out
