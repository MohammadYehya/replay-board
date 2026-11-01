"""3D pop-out (stretch, SPEC 3): Muse Spark turns a drawn graph, surface or solid into a JSON scene that the
player renders with three.js. The scene is pure data: shapes, and maths expressions in a small grammar
(x, y, t, numbers, + - * / ^, sin cos tan exp log ln sqrt abs, pi, e) that the player evaluates itself."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from .core.media import imread, imwrite
from .core.meta_api import MetaClient, image_part

AXIS = {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"},
                                         "label": {"type": "string"}}}
SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["plot2d", "surface", "curve3d", "solids", "none"]},
        "title": {"type": "string"},
        "explain": {"type": "string"},
        "x": AXIS, "y": AXIS, "z": AXIS,
        "functions": {"type": "array", "items": {"type": "object", "properties": {
            "expr": {"type": "string"}, "color": {"type": "string"}, "label": {"type": "string"}}}},
        "curves": {"type": "array", "items": {"type": "object", "properties": {
            "x": {"type": "string"}, "y": {"type": "string"}, "z": {"type": "string"},
            "t_min": {"type": "number"}, "t_max": {"type": "number"}, "color": {"type": "string"}}}},
        "solids": {"type": "array", "items": {"type": "object", "properties": {
            "shape": {"type": "string", "enum": ["box", "sphere", "cylinder", "cone", "torus", "plane", "arrow"]},
            "center": {"type": "array", "items": {"type": "number"}},
            "size": {"type": "array", "items": {"type": "number"}},
            "rotation": {"type": "array", "items": {"type": "number"}},
            "color": {"type": "string"}, "label": {"type": "string"}}}},
        "points": {"type": "array", "items": {"type": "object", "properties": {
            "at": {"type": "array", "items": {"type": "number"}}, "label": {"type": "string"}}}},
    },
}

_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:e[+-]?\d+)?|\.\d+)|([A-Za-z_]+)|(\*\*|[-+*/^(),]))")
_NAMES = {"x", "y", "t", "pi", "e", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
          "exp", "log", "ln", "log10", "log2", "sqrt", "abs", "pow", "min", "max", "floor", "ceil", "sign"}


def safe_expr(expr: str) -> bool:
    """Only numbers, the whitelisted names and arithmetic: nothing the player can't evaluate safely."""
    expr = expr.strip()
    if not expr or len(expr) > 200:
        return False
    pos = 0
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            return expr[pos:].strip() == ""
        if m.group(2) and m.group(2).lower() not in _NAMES:
            return False
        pos = m.end()
    return True


def clean_scene(scene: dict) -> dict:
    """Drop anything the player can't safely draw; clamp sizes and ranges."""
    def axis(a, lo=-10.0, hi=10.0):
        a = a or {}
        mn, mx = float(a.get("min", lo)), float(a.get("max", hi))
        if not (np.isfinite(mn) and np.isfinite(mx)) or mx <= mn:
            mn, mx = lo, hi
        mn, mx = max(-1e4, mn), min(1e4, mx)
        return {"min": mn, "max": mx, "label": str(a.get("label", ""))[:20]}

    out = {"kind": scene.get("kind", "none"), "title": str(scene.get("title", ""))[:80],
           "explain": str(scene.get("explain", ""))[:300],
           "x": axis(scene.get("x")), "y": axis(scene.get("y")), "z": axis(scene.get("z"))}
    out["functions"] = [{"expr": f["expr"], "color": str(f.get("color", ""))[:9], "label": str(f.get("label", ""))[:40]}
                        for f in scene.get("functions", [])[:6] if safe_expr(f.get("expr", ""))]
    out["curves"] = [c for c in scene.get("curves", [])[:4]
                     if all(safe_expr(c.get(k, "")) for k in ("x", "y", "z"))]
    out["solids"] = [s for s in scene.get("solids", [])[:24]
                     if len(s.get("center", [])) == 3 and 1 <= len(s.get("size", [])) <= 3]
    out["points"] = [p for p in scene.get("points", [])[:24] if len(p.get("at", [])) == 3]
    if out["kind"] in ("plot2d", "surface") and not out["functions"]:
        out["kind"] = "none"
    if out["kind"] == "curve3d" and not out["curves"]:
        out["kind"] = "none"
    if out["kind"] == "solids" and not out["solids"]:
        out["kind"] = "none"
    return out


def piece_view(job_dir: Path, replay: dict, piece: dict) -> np.ndarray:
    """The sharp crop saved by the pipeline, else the piece image on the board colour."""
    crop = job_dir / "crops" / f"{piece['id']:03d}.jpg"
    if crop.exists():
        return imread(crop, cv2.IMREAD_COLOR)
    bgra = imread(job_dir / piece["image"])
    bg = replay["board"].get("background", "#f4f4f2").lstrip("#")
    colour = np.array([int(bg[4:6], 16), int(bg[2:4], 16), int(bg[0:2], 16)], np.float32)
    a = bgra[..., 3:4].astype(np.float32) / 255.0
    img = (bgra[..., :3] * a + colour * (1 - a)).astype(np.uint8)
    k = max(1.0, 300 / max(1, min(img.shape[:2])))
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)


def make_scene(client: MetaClient, job_dir: Path, replay: dict, piece_id: int) -> dict:
    cache = job_dir / "popout" / f"{piece_id:03d}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    piece = next((p for p in replay["board"]["pieces"] if p["id"] == piece_id), None)
    if piece is None:
        raise KeyError(piece_id)
    read = piece.get("read") or {}
    prompt = (
        "This is a drawing from a lecture board" + (f" about {replay.get('subject')}" if replay.get("subject") else "")
        + f". It was read as: kind={read.get('kind')!r}, text={read.get('text')!r}, latex={read.get('latex')!r}.\n"
        "Turn it into a small 3D scene a student can rotate, to see the idea better:\n"
        "- plot2d: a graph y = f(x): give functions[].expr in x;\n"
        "- surface: z = f(x, y): give functions[].expr in x and y;\n"
        "- curve3d: a space curve: curves[] with x, y, z in t and t_min/t_max;\n"
        "- solids: drawn 3D shapes (cube, cylinder, cone, sphere...) with center, size and rotation (degrees);\n"
        "- none: if it is not a graph, surface or shape.\n"
        "Expressions may use only x, y, t, numbers, + - * / ^, parentheses, sin cos tan exp ln log sqrt abs, pi, e "
        "(e.g. 'ln(abs(x))', 'x^2 + y^2'). Choose axis ranges that show what was drawn (asymptotes, intercepts), "
        "colours as #rrggbb like the ink, and a few labelled points (e.g. intercepts). explain: one sentence on "
        "what the 3D view shows. Reply in JSON.")
    scene = client.ask(prompt, [image_part(piece_view(job_dir, replay, piece), fmt=".png")], schema=SCENE_SCHEMA,
                       reasoning="medium")
    scene = clean_scene(scene)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(scene), encoding="utf-8")
    return scene
