"""3D views (format 2): a drawing from a board lecture as an animated 3D model that plays along with the lecture.

The player opens it in place of the board, beside the video, which keeps playing. Muse Spark gets the drawing
(its board pieces, pictures of it, when it was drawn and talked about) and the whole lecture, and returns a scene:
objects, parameters the animation moves, named values worked out from them (defs), and ordered steps, each at the
moment the teacher explains it (the transcript line's [m:ss] time, then the cue word looked up in the word times).
Then code checks everything, never trusting the answer:

- every expression is parsed with the player's own grammar (app/core/scene_expr.py) and only the variables of its
  field plus the scene's parameters and defs; what fails is dropped, and what passes is written out in a canonical
  form;
- step times are parsed onto the video's clock (moved to the cue word when it is found), sorted, kept at least 1 s
  apart and inside the analysed clip, and a change never runs into the next step or past the clip's end (playback
  stops there);
- references to unknown objects and parameters are dropped, values clamped, counts and lengths limited; a point
  object given several points becomes several points;
- the geometry is evaluated at the start and through every step (graphs, curves and surfaces sampled): an object
  that gives no finite values is dropped (an angle only where the view rests: at the start and after each step),
  and `bounds` frames the whole animation;
- a param's range holds its value and every step's "to" (it is what a student explores with a slider); a "set" that
  moves nothing and a step that changes nothing on screen are dropped; a step whose caption names an object's label
  ("beta", "β") highlights it, or shows it when it is hidden ("angles" names every angle not
  shown yet);
- angles at one vertex get nested radii, angle params get ranges where the shape exists at every corner, and a step
  whose camera sees what it points at edge-on turns to the preset that sees it best;
- the view is never empty: something is shown before the first step; with no usable step left, one step points at
  the whole drawing (when it is first talked about, if it is).

The scene is pure data: nothing in it is ever run as code. A scene belongs to a drawing, not to the piece it was
opened from: a label or a graph's equation inside a drawing opens that drawing's scene (one model call, one
explanation). It is made on the first request and kept in the job folder (popout/NNN.v2.json, NNN the drawing's
piece) with a version made from replay.json and SCENE_RULES_VERSION, so a reprocessed lecture or new rules make it
again. One request per drawing calls the model at a time; others wait for its answer. A run log
(popout/NNN.v2-log.json) keeps the time, tokens and cost. `python -m app.popout <job> <piece>` makes one ahead of
time (for a demo), after showing what it may cost.

All times are on the video's clock, like replay.json and the player.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from . import ask
from .core import scene_expr as sx
from .core.media import imread
from .core.meta_api import MetaAPIError, chat_text, image_part, parse_json_object, text_part
from .export import NotReady, context_messages, read_replay, safe_text
from .pipeline import replace_file

SCENE_RULES_VERSION = "6"  # change it when the prompt, the checks or REASONING change, so every scene is made again
# 2: angle arcs sized, spanning the named angle and following moving edges; every part of the board drawn; angles in
#    different planes seen from "iso" (end-to-end check on test2: arcs collapsed and labels piled up seen from the front,
#    a 120-degree arc spanned 60 degrees, three spokes drawn where the board has six).
# 3: a series the lecture works through is followed case by case (it "may" be: a v2 scene of test2 #4 skipped the
#    tetragonal and rhombohedral cases); every named angle's arc is labelled with its name; every number in a
#    caption is true of the model (test-full #3 said a point at x = -e on y = ln(x^2) was at height 1, the board's
#    value for ln|x|; it is at 2). The scenes made before that last sentence (test2 #4 and #7, gest-test #10) were
#    checked by hand and have no false numbers, so the version stayed.
# 4: after each step the model is a true example of everything its caption names (test2 #4 showed a triclinic cell
#    under "Monoclinic" for 18 s); the board's orientation is kept (both test2 cells leaned left beside a board cell
#    leaning right); a point is one point (test2 #7 put 13 atoms into two point objects: 2 were drawn); with axes
#    shown they are not drawn again; each step names its cue word and happens when it is said (steps sat at the start
#    of 12-word transcript lines, up to 5 s early); defs write a long formula once (test2 #4 repeated the c edge's
#    formula in 9 edges, 11-16k tokens out, and #7's version ignored gamma); the drawing itself gets a step per thing
#    said about it; graphs animate how they are built. A scene is the drawing's: the asked piece is not named.
# 5: a param named like a field's variable ("t" for a point's place along a curve, test-full #3 rules 4) is renamed
#    instead of dropped with everything that used it. The prompt is the same as 4's, so the scenes made with 4 come
#    back from the disk cache, checked again, at no cost.
# 6: angles are "angle" objects (vertex and two arm ends: the view draws the arc in the right plane and shows its
#    measured value), never curves written by formula (test2 #4's alpha and beta arcs, rules 5, ended at one point;
#    test2 #7 named alpha and beta with nothing labelled so); every named length is
#    labelled; a param's min and max are the range a student explores with a slider; a step that changes nothing is
#    dropped, and one whose caption names a labelled object points at it.
# Kept at "6" when degree_captions and iso_from came (2026-11-03): they only polish scenes made from now on; a new
# version would remake (and bill) every saved scene and replace takes checked by hand. Deleting
# popout/NNN.v2.json remakes one.
FORMAT = 2
# Measured live on test2 #4 (a monoclinic cell, 2026-11-02), same prompt: "low" took 22 s and $0.028 (4.8k tokens
# out) and got the geometry right, but turned the camera to where the lean cannot be seen and drew no angles;
# "medium" took 57 s and $0.056 (11k out) and also drew the alpha, beta and gamma arcs and showed the lean from the
# front. With the final prompt, medium took 68 s and $0.075 and walked the lecture's whole series (cube, tetragonal,
# orthorhombic, rhombohedral, monoclinic) at the teacher's times; low drew test2 #7's 120-degree angle as a 60-degree
# arc. A scene is made once per drawing and kept, so quality wins over the minute's wait.
REASONING = "medium"
# A series of seven cases (test2 #7, rules 3) took 15.8k tokens out, reasoning included: at 16k the first answer
# was cut short and paid for in full before the retry. Both tries get 24k (accepted by the API; tokens are paid as used).
MAX_TOKENS = 24000
POPOUT = "popout"

TITLE_CHARS, EXPLAIN_CHARS, CAPTION_CHARS = 80, 300, 140
# a param's label is on screen all the time (the value chips): 24 cut "position along x parabola" (25)
AXIS_LABEL_CHARS, PARAM_LABEL_CHARS, OBJECT_LABEL_CHARS = 20, 32, 30
MAX_PARAMS, MAX_DEFS, MAX_OBJECTS, MAX_STEPS = 8, 16, 40, 12
OVER_MIN, OVER_MAX, OVER_DEFAULT = 0.6, 8.0, 1.5
STEP_GAP = 1.0           # s between steps, at least
CUE_BEFORE, CUE_AFTER = 1.0, 15.0  # where a step's cue word is looked for, around its line's [m:ss] time
LIMIT = 1e4              # |coordinate| beyond this is not a drawing
SAMPLES, GRID = 241, 41  # samples of a graph or curve; grid of a surface
IMAGE_SIDE = 768

TYPES = ("point", "segment", "vector", "polygon", "box", "sphere", "cylinder", "cone", "torus", "graph", "surface",
         "curve", "psurface", "label", "angle")
STYLES = ("solid", "wire", "dashed")
CAMERAS = ("keep", "front", "top", "side", "iso")
UNITS = ("", "deg")
POINTS = {"point": (1, 1), "label": (1, 1), "segment": (2, 2), "vector": (2, 2), "polygon": (3, 12)}
ARM_MIN = 1e-9           # an angle's arm shorter than this has no direction: the angle is degenerate
AUTO_RADIUS = 0.3        # an angle's radius when it gives none: this share of its shorter arm
NEST = 0.45              # angles at one vertex: radii 1, 1.45, 1.9... times the group's auto radius
ANGLE_TOL = 0.5          # degrees an angle object named like a param may measure off that param's value
# Three edges from one corner whose unit vectors' volume squared (Gram determinant) is under this make a plate, not
# a cell: 60/60/60 (a real rhombohedral cell) is 0.5, 70/70/110 0.57, 120/120/120 0.
FLAT_MIN = 0.45
# Where each camera preset looks from (maths coordinates, z up): scene3dStage.ts CAMERA_DIRS, in its order of
# preference when two see a step equally well.
CAMERA_DIRS = {"iso": (0.7, -1.2, 0.62), "front": (0.2, -1.0, 0.25), "side": (1.0, 0.2, 0.25), "top": (0.0, -0.02, 1.0)}
# "iso" from the front-left corner instead (a scene's iso_from "left", pick_cameras): scene3dStage.ts ISO_LEFT.
ISO_LEFT = (-0.7, -1.2, 0.62)
CAMERA_LOW, CAMERA_BETTER = 0.6, 0.15  # a preset scoring under this is swapped for one at least this much better
DEG_SPAN, LEN_SHARE = 30.0, 0.5  # a param with no range explores value +- 30 degrees, or +- half its value
SIZES = {"box": 3, "sphere": 1, "cylinder": 2, "cone": 2, "torus": 2}
SAMPLED = ("graph", "surface", "curve", "psurface")
PALETTE = ("#1f6fd1", "#d0342c", "#2e9d4a", "#8a4fd0", "#e07b12", "#139aa8", "#8c5a3c", "#d04f9a")
_ID = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
_PARAM = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
_COLOR = re.compile(r"^#([0-9a-f]{6}|[0-9a-f]{3})$")
_AT = re.compile(r"\[?\s*(\d{1,3}):([0-5]\d)(?::([0-5]\d))?\s*\]?")
_FIGURE = re.compile(r"\b(graph|diagram|figure|plot|sketch|drawing|chart|curve)s?\b", re.I)
_ARROW = re.compile(r"\b(arrow|vector|axis|axes)s?\b", re.I)
_SPATIAL = re.compile(r"\b(axis|axes|vectors?|coordinates?)\b", re.I)
_EQUATION = re.compile(r"\b(equation|formula|function)s?\b", re.I)
_HEADING = re.compile(r"\b(heading|title)s?\b", re.I)
_GRAPH_EQ = re.compile(r"^(?:y|z|[fgh]\s*\(\s*x\s*(?:,\s*y\s*)?\))\s*=\s*(.+)$", re.I | re.S)
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

_flat = ask._flat
_num = ask._num
clock = ask.clock


class NoScene(LookupError):
    """No such piece, or not a board lecture (one plain line, HTTP 404)."""


# ---------- the answer's format ----------
_STR = {"type": "string"}
_NUM = {"type": "number"}
_STRS = {"type": "array", "items": {"type": "string"}}
_AXIS = {"type": "object", "properties": {"min": _NUM, "max": _NUM, "label": _STR}}
SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STR,
        "explain": _STR,
        "axes": {"type": "object", "properties": {"show": {"type": "boolean"}, "x": _AXIS, "y": _AXIS, "z": _AXIS}},
        "params": {"type": "array", "items": {"type": "object", "properties": {
            "name": _STR, "value": _NUM, "min": _NUM, "max": _NUM, "label": _STR,
            "unit": {"type": "string", "enum": list(UNITS)}}}},
        "defs": {"type": "array", "items": {"type": "object", "properties": {"name": _STR, "expr": _STR}}},
        "objects": {"type": "array", "items": {"type": "object", "properties": {
            "id": _STR, "type": {"type": "string", "enum": list(TYPES)},
            "points": {"type": "array", "items": _STRS},
            "center": _STRS, "size": _STRS, "rotation": _STRS,
            "expr": _STR, "x": _STR, "y": _STR, "z": _STR, "range": _STRS, "range2": _STRS,
            "color": _STR, "opacity": _NUM, "style": {"type": "string", "enum": list(STYLES)}, "label": _STR}}},
        "steps": {"type": "array", "items": {"type": "object", "properties": {
            "at": _STR, "cue": _STR, "over": _NUM, "caption": _STR, "show": _STRS, "hide": _STRS, "highlight": _STRS,
            "set": {"type": "array", "items": {"type": "object", "properties": {"param": _STR, "to": _NUM}}},
            "camera": {"type": "string", "enum": list(CAMERAS)}}}},
    },
}

SCENE_TASK = """This request is not a student's question. Replay Board can show a drawing from this lecture as an \
animated 3D model in place of the board, beside the video, which keeps playing. Design that 3D view for the drawing \
described below. Reply only with JSON in the given format. For this reply, ignore the answer style above: no \
markdown, no LaTeX, and no [m:ss] times or [#id] citations inside any text.

What to make:
- A faithful 3D model of what was drawn and of what the lecture teaches with it, in true geometry and proportions, \
not the sketch's perspective: a cube sketched in perspective is a cube with equal edges and right angles; a graph \
keeps its real intercepts, asymptotes and turning points; a vector keeps its direction. Take numbers from the board \
and the teacher; where the lecture gives none, choose clear values that respect every relation it states (lengths \
equal or not, angles of 90 degrees or not).
- Keep the board's orientation: what leans to the right on the board leans towards +x seen from the front, and \
what is on top stays on top. Pick the values that do this (with edge a along +x, a cell whose beta is 75 degrees \
leans right and is the same monoclinic cell as one of 105 degrees, which leans left).
- params: up to 8 numbers the steps can move: lengths, angles (unit "deg"), a turning or sweep angle, a point's \
place along a curve, a time, each with a short label (at most 32 characters). Write the geometry in terms of them, so that a step's "set" animates it: corners \
written with cos(theta*deg) and sin(theta*deg) turn when theta changes. "value" is where a param starts. "min" and \
"max" are the range a student may explore with a slider after the lecture (with every angle in its range at once, the shape must still exist): a sensible range around the values the \
lecture uses, not only those values (an angle e.g. 60 to 120, a length e.g. 1 to 6), holding "value" and every \
step's "to", where the model still makes sense.
- defs: up to 16 named values worked out from the params, so that a long formula is written once and is right \
once: [{"name": "cx", "expr": "c*cos(beta*deg)"}]. Each may use the params and the defs before it; every \
expression in the objects may use their names. Use them for the corners that several edges share. Build every point from the params and defs, so that the whole model follows any value a student sets: a prism's top is its base moved by the c edge's vector (cx, cy, cz), not lifted by the length c; a hexagon's corners come from the a and b edges and gamma.
- steps: 3 to 12 steps that follow how the teacher explains this drawing and the idea it illustrates, in the \
lecture's order. A step shows, hides or highlights objects, moves params ("set") and turns the camera; its caption \
says what it shows, in the teacher's own terms (plain text, at most 140 characters). Every step visibly changes \
what its caption talks about: it shows, moves or highlights the objects the caption names (the angle object \
labelled beta when the caption says beta), or turns the camera to where the change is seen; never a step that only \
repeats what is already on screen. After each step the model is \
a correct example of everything its caption names: a step that names a case sets every length and angle of that \
case (a monoclinic step makes a, b and c all different, alpha and gamma 90 and beta not 90, in that one step). If \
the teacher explains a case in parts, give each part its own step at its word. Every number in a caption is true \
of the model at that step: work it out from the expressions (on y = ln(x^2), x = e is at height 2), and never \
copy a number the board gives for something else. A step may show in 3D what the \
teacher only says in words about this drawing (a shape stretching, leaning or turning, a vector splitting into \
components, a region sweeping out a solid, a point sliding along a curve), but never content the lecture does not \
have.
- When this drawing is one case of a series that the lecture works through (types, cases, stages), the steps \
follow that series from where the teacher starts it, changing the one model case by case (stretching an edge, \
tilting an angle), a step for each case up to this drawing, at the moment the teacher names that case; then the \
drawing itself gets a step for each thing the teacher says about it (each length, angle, atom or part, at its \
word), not one step for all of it, and the model of the last case turns into the drawing where it sits in it (a \
cell that is a third of a hexagonal prism stays in place as one third of it, with its gamma corner at a corner of the hexagon, where the board marks 120 degrees: the centre of the hexagon is the cell's far corner). The params cover what changes along \
the series (each length and angle that differs from one case to the next).
- For a graph, animate what the board shows about it: how it is made from a simpler graph (|f(x)| folds the part \
below the x axis up, turning it over about the x axis; f(-x) mirrors it; 2*f(x) stretches it), a point moving \
along it, its intercepts and asymptotes; not only its parts appearing one by one.
- at and cue: when the lecture talks about this drawing or its idea, each step happens at the moment the teacher \
says it. "at" is the [m:ss] time of the transcript line where it is said, without the brackets ("3:07"), and \
"cue" the word or words (one to four, copied exactly from that line) said as the change happens ("tetragonal"): \
the step happens when the cue is said. Steps are in time order, at least a second apart. If the lecture never \
talks about this drawing, set every "at" and "cue" to "" and walk through the drawing in a sensible order instead.
- Before the first step, every param is at its value and only the objects that no step shows are visible. Keep the \
drawing's main frame visible from the start (shown by no step), so the view is never empty, and let the steps add \
details and changes as the teacher gets to them.
- over: the seconds a step's change takes (0.6 to 8); a slow sweep can run while the teacher talks.

The format:
- Coordinates: z is up, x to the right, y away from the viewer. Use a scale where the model is about 2 to 10 \
units across.
- objects: every field is present; unused ones are "" or []. ids are short lowercase names (edge_a, beta_angle).
  - point and label: points [[x, y, z]], one point (a label shows its "label" text there): several atoms are \
several point objects; segment and vector: points [from, to] (a vector has an arrow head at "to"); polygon: \
points, 3 to 12 corners of a flat face, drawn filled and see-through.
  - angle: points [vertex, end1, end2], exactly 3: the angle at the vertex between the edges vertex-end1 and \
vertex-end2 (0 to 180 degrees). The view draws its arc in the plane of the two edges (a square mark at 90 degrees) \
and shows its measured value next to its label. size [] (a radius of 0.3 of the shorter edge) or [radius]. Its \
label is the angle's name only ("beta", "theta", "angle ABC"), never its value.
  - box: center, size [width along x, depth along y, height along z]; sphere: size [radius]; cylinder and cone: \
size [radius, height], axis along z, the cone's tip up; torus: size [ring radius, tube radius], ring in the xy \
plane. rotation: [degrees about x, y, z], or [].
  - graph: expr f(x): the curve y = f(x) drawn in the plane z = 0, over range [from, to] in x ([] for the x axis' \
range). surface: expr f(x, y): the surface z = f(x, y) over the x and y axes' ranges. curve: x, y and z in t, \
over range [from, to] in t. psurface: x, y and z in u and v, over range (u) and range2 (v).
  - color "#rrggbb" (like the ink, and distinct for the parts the captions name), opacity 0.15 to 1, style \
"solid", "wire" (a wireframe; an angle: its arc only) or "dashed" (hidden edges, guides), label: a short name \
shown next to it ("" for none).
  - Every angle the teacher names or the drawing marks is an angle object at its vertex, between the two edges \
that make it (in a crystal cell, alpha is between b and c, beta between a and c, gamma between a and b), labelled \
with its name; never a curve. Write its three points with the same expressions or defs as those edges' ends, so \
that it follows them when the shape moves: its measured value is then the one the teacher says at every step (a \
120-degree angle measures 120, not its supplement). An angle the teacher marks on the board goes at the corner where it is marked, between two edges of the drawing from that corner (a hexagon's 120 degrees at a corner of the hexagon, with a and b along two of its sides from that corner), never at a centre the board does not mark.
  - Every length the teacher names (a, b, c, r, h...) is labelled with that name: the edge's own label, or a \
label object at the middle of the edge, a little off it. Keep labels apart: never two at one place. A label is \
true in every state it is shown in: when a step changes a graph or a length, its label keeps no formula or number \
that stops being true (give a name instead, or hide that label and show a new one in that step).
  - Draw every part the board shows (each edge, atom, line and mark), at its true place.
  - At most 40 objects: a polygon for a face is often clearer than many segments.
- Expressions are strings of numbers, + - * / ^ and parentheses, the functions sin cos tan asin acos atan sinh cosh \
tanh exp ln log10 log2 sqrt abs pow min max floor ceil sign, the constants pi, e and deg (deg = pi/180, so \
cos(beta*deg)), the params' and defs' names, and only the variables of that field: x in a graph's expr, x and y in \
a surface's, t in a curve's x, y and z, u and v in a psurface's, none anywhere else ("1.5" and "a*cos(beta*deg)" \
are fine). Write * for every product.
- axes: show true for graphs and surfaces, with labels; false for solids and diagrams. min and max always frame \
what is drawn (intercepts, asymptotes, the whole model). Shown axes are drawn already: never draw them again as \
segments or vectors.
- camera: "keep", "front" (looking along y at the xz plane), "top" (looking down at the xy plane, where graphs \
lie), "side" (looking along x at the yz plane) or "iso" (a three-quarter view). Turn it to where a change is \
seen best: a lean or a turn in the xz plane is seen from the front. Seen from the front, anything in the xy or yz \
plane is edge-on and collapses to a line: show angles in different planes from "iso".
- title: what the view shows (at most 80 characters). explain: one or two sentences on what it shows and what to \
watch (at most 300 characters).
- If the drawing cannot be shown in 3D (it is only writing), give objects [] and steps [], and say why in explain."""


# ---------- the drawing: the asked piece and the pieces that belong to it ----------
def _box(p: dict) -> tuple[float, float, float, float] | None:
    b = p.get("box")
    if not (isinstance(b, list) and len(b) >= 4 and all(_num(v) for v in b[:4])):
        return None
    x, y, w, h = (float(v) for v in b[:4])
    return (x, y, max(1.0, w), max(1.0, h))


def _area(b) -> float:
    return b[2] * b[3]


def _gap(a, b) -> float:
    """The distance between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return math.hypot(dx, dy)


def _inside(a, b, pad: float) -> float:
    """The share of box a inside box b grown by pad."""
    x0, y0 = max(a[0], b[0] - pad), max(a[1], b[1] - pad)
    x1, y1 = min(a[0] + a[2], b[0] + b[2] + pad), min(a[1] + a[3], b[1] + b[3] + pad)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / _area(a)


def _span(p: dict, clip0: float) -> tuple[float, float]:
    """When a piece is on the board: from when it is shown to when it is erased."""
    a = clip0 if p.get("pre_existing") else float(p.get("t_shown", p.get("t_seen", clip0)) or clip0)
    b = float(p["removed_at"]) if _num(p.get("removed_at")) else math.inf
    return a, b


def _together(p: dict, q: dict, clip0: float) -> bool:
    a, b = _span(p, clip0), _span(q, clip0)
    return max(a[0], b[0]) < min(a[1], b[1])


def _same_period(q: dict, anchor: dict, clip0: float) -> bool:
    """Written within a minute of the drawing, or while the teacher talked about it: later writing in the same
    place (test-full: sums written over an old graph minutes later) is not part of it."""
    def drawn(p: dict) -> float:
        return clip0 if p.get("pre_existing") else float(p.get("t_seen", clip0) or clip0)

    tq = drawn(q)
    if abs(tq - drawn(anchor)) <= 60:
        return True
    return any(a - 10 <= tq <= b + 10 for a, b in ask._merge(anchor.get("spoken") or []))


def _kind(p: dict) -> str:
    return _flat((p.get("read") or {}).get("kind")).lower()


def _text(p: dict) -> str:
    read = p.get("read") or {}
    return _flat(read.get("text")) or _flat(read.get("latex"))


def is_drawing(kind: str, text: str) -> bool:
    """Graphs, diagrams, sketches..., and arrows that are axes or vectors: drawn with their heads, axes and vectors
    are read as "arrow" ("x-axis and y-axis arrows", "red vector arrow labeled v"). A lone arrow is not one."""
    return bool(_FIGURE.search(kind) or (_ARROW.search(kind) and _SPATIAL.search(text)))


def is_figure(p: dict) -> bool:
    return is_drawing(_kind(p), _text(p))


def looks_like_graph_equation(s: Any) -> bool:
    """"y = |ln x|", "z = x^2 + y^2", "f(x) = e^{-x}" (LaTeX or plain), with x on the right: as the player's
    looksLikeGraphEquation (web/src/player/popoutable.ts)."""
    if not isinstance(s, str):
        return False
    flat = re.sub(r"\\(left|right|displaystyle|,|;|!|quad)", "", s.replace("$", ""))
    flat = re.sub(r"\s+", " ", flat).strip()
    m = _GRAPH_EQ.match(flat)
    if not m:
        return False
    rhs = m.group(1).split("=")[0]  # up to a second "=", as in y = |ln x| = |log_e x|
    return bool(re.search(r"x", re.sub(r"\\[a-z]+", " ", rhs, flags=re.I), re.I))


def can_popout(p: Any) -> bool:
    """The pieces the player offers a 3D view for (canPopout in popoutable.ts): drawings, and equations of a
    graph. Any other piece is refused before a model call is spent on it."""
    read = p.get("read") if isinstance(p, dict) else None
    if not isinstance(read, dict):
        return False
    kind = read.get("kind") if isinstance(read.get("kind"), str) else ""
    if is_drawing(kind, read.get("text") if isinstance(read.get("text"), str) else ""):
        return True
    if not _EQUATION.search(kind):
        return False
    return looks_like_graph_equation(read.get("latex") or "") or looks_like_graph_equation(read.get("text") or "")


def gather(replay: dict, piece_id: int) -> tuple[dict, dict, list[dict], list[dict]]:
    """(the asked piece, the drawing it belongs to, the drawing's pieces with the asked piece first, other writing
    next to it). A small piece inside a bigger drawing (a label, a mark) belongs to that drawing. The drawing's
    pieces are those on the board with it, written in the same period, and mostly inside its box or short labels
    touching it; never a whole board of writing."""
    _, clip0, _ = ask._video(replay)
    pieces = [p for p in (replay.get("board") or {}).get("pieces") or []
              if isinstance(p, dict) and isinstance(p.get("id"), int) and _box(p) is not None]
    asked = next((p for p in pieces if p["id"] == piece_id), None)
    if asked is None:
        raise NoScene("no such piece")
    ab = _box(asked)
    anchor = asked
    for q in pieces:  # the smallest drawing that holds the asked piece
        qb = _box(q)
        if (q is asked or not is_figure(q) or _area(qb) < 1.5 * _area(ab) or not _together(q, asked, clip0)
                or _inside(ab, qb, 4) < 0.6):
            continue
        if anchor is asked or _area(qb) < _area(_box(anchor)):
            anchor = q
    group = figure_group(pieces, anchor, clip0)
    if len(group) > 1:  # one drawing in several pieces (axes and a vector drawn one after another): the biggest
        anchor = max(group, key=lambda g: (_area(_box(g)), -g["id"]))
    nb = _union([_box(g) for g in group])
    side = min(nb[2], nb[3])
    pad, near_label = max(4.0, 0.06 * side), max(12.0, 0.2 * side)
    near_text = min(60.0, max(30.0, 0.6 * side))
    parts, nearby = [], []
    for q in pieces:
        if (q is asked or q is anchor or not _together(q, anchor, clip0) or not _same_period(q, anchor, clip0)
                or _HEADING.search(_kind(q))):
            continue
        qb = _box(q)
        inside, gap = _inside(qb, nb, pad), _gap(qb, nb)
        small = _area(qb) <= 0.25 * _area(nb)
        if _area(qb) <= _area(nb) and (inside >= 0.85 or (gap <= near_label and small
                                                          and (_kind(q) == "label" or len(_text(q)) <= 6))):
            parts.append((-inside, gap, q))
        elif gap <= near_text and len(_text(q)) <= 80 and _area(qb) <= 2 * _area(nb):
            nearby.append((gap, q))
    parts.sort(key=lambda t: (t[0], t[1], t[2]["id"]))
    nearby.sort(key=lambda t: (t[0], t[1]["id"]))
    members = [g for g in group if g is not asked and g is not anchor]
    drawing = [asked] + ([anchor] if anchor is not asked else []) + members + [
        q for _, _, q in parts if q not in members][:max(0, 10 - len(members))]
    return asked, anchor, drawing, [q for _, q in nearby[:6]]


GROUP_SHARE = 0.15  # figure pieces overlapping by this share of the smaller box are parts of one drawing
GROUP_WITHIN = 60.0  # s between when they were drawn, at most


def _union(boxes: list) -> tuple[float, float, float, float]:
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _overlap(a, b) -> float:
    """The overlap of two boxes as a share of the smaller one."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h) / max(1e-9, min(_area(a), _area(b)))


def figure_group(pieces: list[dict], anchor: dict, clip0: float) -> list[dict]:
    """The anchor and the figure pieces drawn with it as one drawing: on the board together, drawn within a minute
    of it, and overlapping it or another member (the z axis, the x and y axes and the vector of one 3D sketch are
    separate pieces once their arrowheads show). The anchor first; just the anchor when it is not a figure."""
    if not is_figure(anchor):
        return [anchor]

    def drawn(p: dict) -> float:
        return clip0 if p.get("pre_existing") else float(p.get("t_seen", clip0) or clip0)

    group, t0 = [anchor], drawn(anchor)
    grew = True
    while grew:
        grew = False
        for q in pieces:
            if any(q is g for g in group) or not is_figure(q) or not _together(q, anchor, clip0) \
                    or abs(drawn(q) - t0) > GROUP_WITHIN:
                continue
            if any(_overlap(_box(q), _box(g)) >= GROUP_SHARE for g in group):
                group.append(q)
                grew = True
    return group


def drawing_id(replay: dict, piece_id: int) -> int:
    """The drawing a piece opens: its anchor in gather(), followed until it is its own (a label in a diagram that
    sits in a bigger figure opens the figure). Every piece of one drawing gets the same scene."""
    seen = {piece_id}
    cur = piece_id
    for _ in range(4):
        nxt = gather(replay, cur)[1]["id"]
        if nxt in seen:
            break
        seen.add(nxt)
        cur = nxt
    return cur


def first_talked(replay: dict, pieces: list[dict]) -> float | None:
    """When the teacher first talks about any of these pieces (inside the analysed clip), or None."""
    _, clip0, clip1 = ask._video(replay)
    starts = [float(s["start"]) for p in pieces for s in p.get("spoken") or []
              if isinstance(s, dict) and _num(s.get("start"))]
    starts = [t for t in starts if clip0 - 1 <= t <= clip1 + 1]
    return min(max(clip0, min(starts)), clip1) if starts else None


def drawing_images(job_dir: Path, replay: dict, anchor: dict, pieces: list[dict]) -> list[tuple[str, np.ndarray, str]]:
    """(caption, image, format): the drawing's strokes on the board colour, and the sharp crop from the video."""
    out = []
    try:
        img = ask.compose_pieces(job_dir, replay, pieces, pad=12)
    except Exception:
        img = None
    if img is not None and img.size:
        out.append(("the drawing as captured from the board (its strokes on the board's colour)",
                    ask.fit_image(img, IMAGE_SIDE), ".png"))
    crop = job_dir / "crops" / f"{anchor['id']:03d}.jpg"
    if crop.is_file():
        try:
            img = imread(crop, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is not None and img.size:
            out.append(("a photo of the drawing from the video (other writing is faded, and the teacher may stand "
                        "in front of it)", ask.fit_image(img, IMAGE_SIDE), ".jpg"))
    return out


def drawing_text(replay: dict, asked: dict, anchor: dict, pieces: list[dict], nearby: list[dict]) -> str:
    """What the prompt says about the drawing: its pieces and the writing next to it, as in the lecture above, and
    what was said while it was talked about."""
    _, clip0, _ = ask._video(replay)
    index = ask.item_index(replay)

    def line(p: dict) -> str:
        it = index.get(f"#{p['id']}")
        if it is None:
            return f"#{p['id']} {ask._what(p.get('read'))}"
        return f"{it.ref} {ask._what(it.data.get('read'))} | {ask._board_when(it, clip0)}"

    out = ["# The drawing", "The drawing: " + line(anchor)]
    parts = [p for p in pieces if p is not anchor]
    if parts:
        out.append("Its parts on the board:")
        out += [f"- {line(p)}" for p in parts]
    seen, near = set(), []
    for q in nearby:
        it = index.get(f"#{q['id']}")
        ref = f"#{it.line}" if it is not None and it.line else f"#{q['id']}"
        if ref in seen:
            continue
        seen.add(ref)
        lit = index.get(ref)
        near.append(f"- {lit.ref} {ask._what(lit.data.get('read'))} | {ask._board_when(lit, clip0)}" if lit else
                    f"- {line(q)}")
    if near:
        out.append("Writing next to it on the board:")
        out += near
    spans = ask._merge([s for p in (asked, anchor) for s in p.get("spoken") or []])
    words = ask._words(replay)
    lines: list[str] = []
    for a, b in spans[:3]:
        lines += ask.speech_lines(words, a - 4, b + 4)
    if lines:
        out.append("Said while the drawing was talked about (all of the transcript is above):")
        out += lines[:40]
    else:
        out.append("The automatic analysis found no moment where the teacher talks about this drawing; check the "
                   "transcript above before you decide it is never talked about.")
    return "\n".join(out)


def scene_messages(job_dir: Path, replay: dict, piece_id: int, attempt: int = 1) -> tuple[list[dict], str, dict]:
    """(the messages, the prompt cache key, what was gathered: asked, anchor, pieces) for a drawing (piece_id is
    the drawing's own id: see drawing_id). The lecture goes first, word for word as for Ask Muse and the slides, so
    Meta's prompt cache holds it once for all of them."""
    asked, anchor, pieces, nearby = gather(replay, piece_id)
    system, key = context_messages(job_dir, replay)
    head = SCENE_TASK + "\n\n" + drawing_text(replay, asked, anchor, pieces, nearby)
    images = drawing_images(job_dir, replay, anchor, pieces)
    if images:
        head += "\n" + "\n".join(f"Image {i}: {cap}." for i, (cap, _, _) in enumerate(images, 1))
    if attempt > 1:  # a later try is a different request, so the disk cache does not hand back the same answer
        head += f"\n\n(Try {attempt}: reply with complete, valid JSON in the given format.)"
    content = [text_part(safe_text(head))]
    for i, (_, img, fmt) in enumerate(images, 1):
        content += [text_part(f"Image {i}:"), image_part(img, fmt=fmt, quality=90)]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    return msgs, key, {"asked": asked, "anchor": anchor, "pieces": pieces}


# ---------- checking the answer ----------
def _str(v: Any, n: int) -> str:
    """Plain text of at most n characters. A longer one is cut at the end of a word when that keeps over half of it
    ("position along x parabola" -> "position along x…", never "position along x parabo…": synth3d-saddle)."""
    text = _flat(safe_text(v)) if isinstance(v, str) else ""
    if len(text) <= n:
        return text
    head = text[:n - 1]
    if not text[n - 1].isspace():
        whole = head.rsplit(" ", 1)[0] if " " in head else ""
        if len(whole.rstrip()) > (n - 1) // 2:
            head = whole
    return head.rstrip(" ,;:-") + "…"


def _number(v: Any) -> float | None:
    try:
        return float(v) if _num(v) else None
    except OverflowError:  # an int too big for a float: JSON can carry one
        return None


def _clamp(v: float) -> float:
    return min(LIMIT, max(-LIMIT, v))


def _starts(clip: tuple[float, float]) -> tuple[float, float]:
    """(first, last) moment a step may start, on a 0.01 s grid: inside the clip, and the last early enough for its
    change to be over by the clip's end, since playback stops there (a change still running would stay half done)."""
    lo = math.ceil(clip[0] * 100 - 1e-6) / 100
    return lo, max(lo, math.floor((clip[1] - OVER_MIN) * 100 + 1e-6) / 100)


def _over(over: float, t: float, end: float) -> float:
    """A change's length, cut so it is over by `end` (the next step, or the clip's end), rounded down."""
    return max(OVER_MIN, math.floor(min(over, end - t) * 100 + 1e-6) / 100)


def _norm_id(v: Any, pattern: re.Pattern, n: int) -> str | None:
    """'Edge-A' -> 'edge_a'; None when nothing usable is left."""
    if not isinstance(v, str):
        return None
    s = re.sub(r"[^a-z0-9_]+", "_", v.strip().lower()).strip("_")[:n]
    return s if pattern.fullmatch(s) else None


def clean_axes(raw: Any, notes: list[str]) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {"show": raw.get("show") is True}
    for k in ("x", "y", "z"):
        a = raw.get(k) if isinstance(raw.get(k), dict) else {}
        lo, hi = _number(a.get("min")), _number(a.get("max"))
        if lo is not None and hi is not None:  # clamped first: a range beyond the limit must not turn upside down
            lo, hi = sorted((_clamp(lo), _clamp(hi)))
        if lo is None or hi is None or hi - lo < 1e-9:
            notes.append(f"axis {k}: no usable range, so -5 to 5")
            lo, hi = -5.0, 5.0
        out[k] = {"min": lo, "max": hi, "label": _str(a.get("label"), AXIS_LABEL_CHARS)}
    return out


def clean_params(raw: Any, notes: list[str], renames: dict[str, str] | None = None) -> list[dict]:
    """The params, checked. One named like a field's variable (a point's place along a curve called "t": test-full
    #3, rules 4) is renamed "p_t" when `renames` is given, which then maps the old name to the new one for every
    expression outside that variable's own fields; otherwise it is dropped."""
    out, seen = [], set()
    for p in raw if isinstance(raw, list) else []:
        if not isinstance(p, dict):
            continue
        name = _norm_id(p.get("name"), _PARAM, 16)
        if renames is not None and name in sx.VARIABLES and name not in renames:
            new = f"p_{name}"
            notes.append(f"param {name} renamed {new}: {name} is a variable of graphs, surfaces or curves")
            renames[name] = new
            name = new
        if name is None or name in sx.RESERVED or name in seen:
            notes.append(f"param {str(p.get('name'))[:20]!r} dropped: its name is not allowed or repeated")
            continue
        lo, hi, v = _number(p.get("min")), _number(p.get("max")), _number(p.get("value"))
        if lo is None or hi is None or v is None:
            notes.append(f"param {name} dropped: value, min or max is not a number")
            continue
        if len(out) == MAX_PARAMS:
            notes.append(f"param {name} dropped: at most {MAX_PARAMS}")
            continue
        unit = p.get("unit") if p.get("unit") in UNITS else ""
        v = _cut(v, f"param {name}: value", notes)
        lo, hi = _cut(lo, f"param {name}: min", notes), _cut(hi, f"param {name}: max", notes)
        lo, hi = fit_range(name, sorted((lo, hi)), [v], unit, notes)
        seen.add(name)
        out.append({"name": name, "value": v, "min": lo, "max": hi,
                    "label": _str(p.get("label"), PARAM_LABEL_CHARS) or name, "unit": unit})
    return out


def _round(v: float) -> float:
    return float(f"{v:.6g}")


def _cut(v: float, what: str, notes: list[str]) -> float:
    """v within +-LIMIT, never silently: a value cut there may leave a step moving nothing, then dropped."""
    c = _clamp(v)
    if c != v:
        notes.append(f"{what} {_round(v):g} cut to {_round(c):g} (at most {LIMIT:g} either way)")
    return c


def fit_range(name: str, rng: list[float] | tuple[float, float], values: list[float], unit: str,
              notes: list[str]) -> tuple[float, float]:
    """A param's [min, max]: the range a student explores with a slider. It holds every value the scene gives the
    param (its start, every step's "to"), so a step is never cut short of what its caption says (rules 5 cut the
    value to the range instead); one with no width gets a sensible one around its value (an angle +-30 degrees, a
    length +-half of it), so the slider moves."""
    lo, hi = float(rng[0]), float(rng[1])
    want_lo, want_hi = min([lo] + values), max([hi] + values)
    if (want_lo, want_hi) != (lo, hi):
        notes.append(f"param {name}: range widened from {_round(lo):g}..{_round(hi):g} to "
                     f"{_round(want_lo):g}..{_round(want_hi):g} to hold its values")
        lo, hi = want_lo, want_hi
    if hi - lo < 1e-9:
        mid = (lo + hi) / 2
        d = DEG_SPAN if unit == "deg" else (LEN_SHARE * abs(mid) if abs(mid) > 1e-6 else 1.0)
        lo, hi = _clamp(mid - d), _clamp(mid + d)
        notes.append(f"param {name}: no range to explore, so {_round(lo):g}..{_round(hi):g}")
    return lo, hi


def clean_defs(raw: Any, params: set[str], notes: list[str], renames: dict[str, str] | None = None) -> list[dict]:
    """Named values worked out from the params, in order: each may use the params and the defs before it (never a
    field's variable), and every object's expressions may use them. [{"name", "expr"}], the expr canonical."""
    out: list[dict] = []
    names = set(params)
    for d in raw if isinstance(raw, list) else []:
        if not isinstance(d, dict):
            continue
        name = _norm_id(d.get("name"), _PARAM, 16)
        if name is None or name in sx.RESERVED or name in names:
            notes.append(f"def {str(d.get('name'))[:20]!r} dropped: its name is not allowed or repeated")
            continue
        if len(out) == MAX_DEFS:
            notes.append(f"def {name} dropped: at most {MAX_DEFS}")
            continue
        try:
            expr = sx.canonical(d.get("expr"), names, renames)
        except sx.ExprError as e:
            notes.append(f"def {name} dropped: expression not allowed ({e})")
            continue
        names.add(name)
        out.append({"name": name, "expr": expr})
    return out


def with_defs(defs: list[tuple[str, tuple]], env: dict[str, float]) -> dict[str, float]:
    """The params of one state with every def's value added (defs parsed: (name, tree), in order)."""
    out = dict(env)
    for name, tree in defs:
        try:
            out[name] = float(np.asarray(sx.evaluate(tree, out), dtype=float))
        except (ValueError, TypeError, FloatingPointError, ZeroDivisionError):
            out[name] = math.nan
    return out


def _parsed_defs(scene: dict) -> list[tuple[str, tuple]]:
    names = {p["name"] for p in scene["params"]}
    out = []
    for d in scene.get("defs") or []:
        out.append((d["name"], sx.parse(d["expr"], names)))
        names.add(d["name"])
    return out


def _exprs(vals: Any, n: tuple[int, int] | int, names: set[str],
           renames: dict[str, str] | None = None) -> list[str] | None:
    """A list of expressions (no variables, only params), checked: None when it is not usable."""
    lo, hi = (n, n) if isinstance(n, int) else n
    if not isinstance(vals, list) or len(vals) < lo:
        return None
    try:
        return [sx.canonical(v, names, renames) for v in vals[:hi]]
    except sx.ExprError:
        return None


def split_points(o: Any, groups: dict[str, list[str]], notes: list[str]) -> list[Any]:
    """A point or label object holds one point. One given several (13 atoms in two points: test2 #7) becomes one
    object per point, `id`, `id_2`, `id_3`...; `groups` maps the id to them, so a step that names it names them
    all. The label stays on the first."""
    if not isinstance(o, dict) or o.get("type") not in ("point", "label"):
        return [o]
    pts = o.get("points") if isinstance(o.get("points"), list) else []
    oid = _norm_id(o.get("id"), _ID, 24)
    if len(pts) < 2 or oid is None:
        return [o]
    ids = [oid] + [f"{oid[:20]}_{k}" for k in range(2, len(pts) + 1)]
    groups[oid] = ids
    notes.append(f"object {oid}: {len(pts)} points made {len(pts)} {o['type']}s")
    keep_label = o.get("type") == "label"  # a label is its text: each copy keeps it
    return [{**o, "id": i, "points": [p], **({} if k == 0 or keep_label else {"label": ""})}
            for k, (i, p) in enumerate(zip(ids, pts))]


def clean_object(o: Any, index: int, params: set[str], notes: list[str],
                 renames: dict[str, str] | None = None) -> dict | None:
    """One object, flat with every field (unused ones empty), its expressions in canonical form; None when it
    cannot be drawn. `params`: the names its expressions may use besides their field's variables (the params and
    the defs). `renames`: params renamed (clean_params), each outside its own variable's fields."""
    renames = renames or {}
    if not isinstance(o, dict):
        return None
    oid = _norm_id(o.get("id"), _ID, 24)
    typ = o.get("type")
    if oid is None or typ not in TYPES:
        notes.append(f"object {str(o.get('id'))[:24]!r} dropped: bad id or type")
        return None
    out: dict[str, Any] = {"id": oid, "type": typ, "points": [], "center": [], "size": [], "rotation": [],
                           "expr": "", "x": "", "y": "", "z": "", "range": [], "range2": [], "color": "",
                           "opacity": 1.0, "style": "solid", "label": ""}

    def bad(why: str) -> None:
        notes.append(f"object {oid} dropped: {why}")

    try:
        if typ == "angle":  # [vertex, end1, end2], and [] (auto radius) or [radius]
            pts = o.get("points") if isinstance(o.get("points"), list) else []
            if len(pts) != 3:
                return bad(f"{len(pts)} points for an angle (its vertex and one point on each arm)")
            for p in pts:
                xyz = _exprs(p, 3, params, renames)
                if xyz is None:
                    return bad("a point is not 3 allowed expressions")
                out["points"].append(xyz)
            size = [s for s in o.get("size") or [] if not (isinstance(s, str) and not s.strip())] \
                if isinstance(o.get("size"), list) else []
            if len(size) > 1:
                notes.append(f"object {oid}: only its first size kept (an angle's radius)")
            if size:
                out["size"] = _exprs(size[:1], 1, params, renames)
                if out["size"] is None:
                    return bad("its radius is not an allowed expression")
        elif typ in POINTS:
            lo, hi = POINTS[typ]
            pts = o.get("points") if isinstance(o.get("points"), list) else []
            if len(pts) < lo or (typ == "polygon" and len(pts) > hi):
                return bad(f"{len(pts)} points for a {typ}")
            if len(pts) > hi:
                notes.append(f"object {oid}: only its first {hi} points kept")
            out["points"] = []
            for p in pts[:hi]:
                xyz = _exprs(p, 3, params, renames)
                if xyz is None:
                    return bad("a point is not 3 allowed expressions")
                out["points"].append(xyz)
        elif typ in SIZES:
            out["center"] = _exprs(o.get("center"), 3, params, renames)
            out["size"] = _exprs(o.get("size"), SIZES[typ], params, renames)
            if out["center"] is None or out["size"] is None:
                return bad("its center or size is not allowed expressions")
            rot = o.get("rotation")
            out["rotation"] = (_exprs(rot, 3, params, renames) or []) if rot else []
            if rot and not out["rotation"]:
                notes.append(f"object {oid}: its rotation is not 3 allowed expressions, so none")
        elif typ in ("graph", "surface"):
            own = {"x"} if typ == "graph" else {"x", "y"}
            out["expr"] = sx.canonical(o.get("expr"), params | own, {k: v for k, v in renames.items() if k not in own})
            if typ == "graph" and o.get("range"):
                out["range"] = _exprs(o.get("range"), 2, params, renames) or []
        else:  # curve, psurface
            own = {"t"} if typ == "curve" else {"u", "v"}
            mine = {k: v for k, v in renames.items() if k not in own}
            for k in ("x", "y", "z"):
                out[k] = sx.canonical(o.get(k), params | own, mine)
            out["range"] = _exprs(o.get("range"), 2, params, renames)
            if out["range"] is None:
                return bad("its range is not 2 allowed expressions")
            if typ == "psurface":
                out["range2"] = _exprs(o.get("range2"), 2, params, renames)
                if out["range2"] is None:
                    return bad("its range2 is not 2 allowed expressions")
    except sx.ExprError as e:
        return bad(f"expression not allowed ({e})")
    color = o.get("color").strip().lower() if isinstance(o.get("color"), str) else ""
    if _COLOR.fullmatch(color):
        out["color"] = color if len(color) == 7 else "#" + "".join(c * 2 for c in color[1:])
    else:
        out["color"] = PALETTE[index % len(PALETTE)]
    op = _number(o.get("opacity"))
    out["opacity"] = round(min(1.0, max(0.15, op)), 3) if op is not None else 1.0
    out["style"] = o.get("style") if o.get("style") in STYLES else "solid"
    out["label"] = _str(o.get("label"), OBJECT_LABEL_CHARS)
    if typ == "label" and not out["label"]:
        return bad("a label with no text")
    if typ == "angle":  # the view shows the measured value after the name: "beta = 75°" would show it twice
        name = _ANGLE_VALUE.sub("", out["label"]).strip()
        if name != out["label"]:
            notes.append(f"object {oid}: the value taken out of its label {out['label']!r}: the view shows the "
                         f"measured one")
            out["label"] = name
    return out


# A value at the end of an angle's label: after "=", ":" or "≈" (its unit optional), or on its own with a unit, also
# when it is the whole label ("90°" would show "90° 90°"); "angle 1" and "θ1" are names and stay.
_DEG = r"(?:[°º˚]|deg(?:rees?)?)"
_ANGLE_VALUE = re.compile(rf"(?:\s*[=:≈]\s*[-+]?\d+(?:\.\d+)?\s*{_DEG}?|(?:^|\s+)[-+]?\d+(?:\.\d+)?\s*{_DEG})\s*$",
                          re.I)


def parse_at(v: Any) -> float | None:
    """'3:07', '[3:07]' or '1:02:03' -> seconds; None for '' or anything else."""
    if not isinstance(v, str):
        return None
    m = _AT.fullmatch(v.strip())
    if not m:
        return None
    a, b, c = m.group(1), m.group(2), m.group(3)
    return float(int(a) * 3600 + int(b) * 60 + int(c)) if c is not None else float(int(a) * 60 + int(b))


def _tokens(text: Any) -> list[str]:
    return _TOKEN.findall(str(text).lower())


def find_cue(words: list[dict], cue: Any, t: float, before: float = CUE_BEFORE,
             after: float = CUE_AFTER) -> float | None:
    """When the teacher says the cue near t: the start of the first word of its first occurrence among the words
    spoken from t - before to t + after (t is a transcript line's [m:ss], cut to the second, and a line is about
    12 words). The words in order, or failing that the cue's first long word (4 letters or more) on its own, so
    "pushed over" still finds "pushed". None when it is not there."""
    want = _tokens(cue)
    if not want or not words:
        return None
    toks: list[tuple[str, float]] = []
    for w in words:
        s = w.get("s")
        if _num(s) and t - before <= float(s) <= t + after:
            toks += [(k, float(s)) for k in _tokens(w.get("w", ""))]
    seq = [k for k, _ in toks]
    for i in range(len(seq) - len(want) + 1):
        if seq[i:i + len(want)] == want:
            return toks[i][1]
    key = next((k for k in want if len(k) >= 4), None)
    if key is not None:
        for k, s in toks:
            if k == key:
                return s
    return None


def clean_steps(raw: Any, ids: list[str], params: list[dict], clip: tuple[float, float],
                notes: list[str], words: list[dict] | None = None,
                groups: dict[str, list[str]] | None = None,
                renames: dict[str, str] | None = None) -> tuple[list[dict], bool]:
    """(the steps, synced): known ids and params only, values within +-LIMIT; when synced, each at its cue word when
    that is found in `words` near its line's time, in time order, at least STEP_GAP apart, inside the clip, and each
    change over before the next step starts. `groups`: an object split in several (split_points) is named by
    its first id."""
    known = set(ids)
    pmap = {p["name"]: p for p in params}
    groups = groups or {}
    steps = []
    lines: dict[int, float | None] = {}  # each step's transcript line time (its cue word may come later in it)
    for s in raw if isinstance(raw, list) else []:
        if not isinstance(s, dict):
            continue

        def refs(key: str) -> list[str]:
            out = []
            for r in s.get(key) if isinstance(s.get(key), list) else []:
                rid = _norm_id(r, _ID, 24)
                for one in groups.get(rid, [rid]):
                    if one in known and one not in out:
                        out.append(one)
                if rid not in known and not any(one in known for one in groups.get(rid, [])):
                    notes.append(f"step: unknown object {str(r)[:24]!r} dropped from {key}")
            return out

        show, hide, highlight = refs("show"), refs("hide"), refs("highlight")
        hide = [h for h in hide if h not in show]
        sets: dict[str, float] = {}
        for st in s.get("set") if isinstance(s.get("set"), list) else []:
            name = _norm_id(st.get("param"), _PARAM, 16) if isinstance(st, dict) else None
            name = (renames or {}).get(name, name)
            to = _number(st.get("to")) if isinstance(st, dict) else None
            if name not in pmap or to is None:
                notes.append(f"step: set of unknown param or value {str(st)[:40]!r} dropped")
                continue
            # the param's range is widened to hold it (fit_range), not the value cut
            sets[name] = _cut(to, f"step: {name} set to", notes)
        over = _number(s.get("over"))
        t = line_t = parse_at(s.get("at"))
        cue = s.get("cue") if isinstance(s.get("cue"), str) else ""
        if t is not None and cue.strip() and words:
            at = find_cue(words, cue, t)
            if at is None:
                notes.append(f"step at {clock(t)}: its cue {cue.strip()[:30]!r} is not said there; the line's time kept")
            else:
                t = at
        step = {"at": "", "t": t,
                "over": min(OVER_MAX, max(OVER_MIN, over)) if over is not None else OVER_DEFAULT,
                "caption": _str(s.get("caption"), CAPTION_CHARS), "show": show, "hide": hide,
                "highlight": highlight, "set": [{"param": k, "to": v} for k, v in sets.items()],
                "camera": s.get("camera") if s.get("camera") in CAMERAS else "keep"}
        if not (step["caption"] or show or hide or highlight or step["set"] or step["camera"] != "keep"):
            notes.append("step dropped: it does nothing")
            continue
        steps.append(step)
        lines[id(step)] = line_t

    lo, hi = _starts(clip)
    timed = sum(1 for s in steps if s["t"] is not None)
    synced = bool(steps) and timed * 2 >= len(steps)
    if not synced:
        if timed:
            notes.append(f"only {timed} of {len(steps)} steps had a time: not synced")
        for s in steps:
            s["t"] = None
        if len(steps) > MAX_STEPS:
            notes.append(f"{len(steps) - MAX_STEPS} steps dropped: at most {MAX_STEPS}")
        return steps[:MAX_STEPS], False

    # An untimed step in a synced scene comes right after the step before it (or before the first timed one).
    first = next(s["t"] for s in steps if s["t"] is not None)
    prev = prev_line = None
    for i, s in enumerate(steps):
        if s["t"] is None:
            notes.append(f"step {i + 1} had no time: placed after the step before it")
            s["t"] = prev + STEP_GAP if prev is not None else first - STEP_GAP
            lines[id(s)] = prev_line if prev_line is not None else s["t"]
        prev, prev_line = s["t"], lines[id(s)]
    for s in steps:
        if not lo <= s["t"] <= hi:
            notes.append(f"step at {clock(s['t'])} moved into the clip {clock(lo)}-{clock(hi)}")
        s["t"] = min(hi, max(lo, s["t"]))
    # In the order of their lines (stable: steps of one line keep the answer's order); a cue word found further on
    # never jumps a step over the next one: the next is pushed after it below.
    steps.sort(key=lambda s: lines[id(s)])
    room = int((hi - lo) // STEP_GAP) + 1
    if len(steps) > min(MAX_STEPS, room):
        notes.append(f"{len(steps) - min(MAX_STEPS, room)} steps dropped: at most {min(MAX_STEPS, room)}")
        steps = steps[:min(MAX_STEPS, room)]
    for i in range(1, len(steps)):
        steps[i]["t"] = max(steps[i]["t"], steps[i - 1]["t"] + STEP_GAP)
    if steps and steps[-1]["t"] > hi:
        steps[-1]["t"] = hi
        for i in range(len(steps) - 2, -1, -1):
            steps[i]["t"] = min(steps[i]["t"], steps[i + 1]["t"] - STEP_GAP)
    for s in steps:
        s["t"] = round(s["t"], 2)
        s["at"] = clock(s["t"])
    for i, s in enumerate(steps):  # each change is over before the next step starts, the last by the clip's end
        s["over"] = _over(s["over"], s["t"], steps[i + 1]["t"] if i + 1 < len(steps) else clip[1])
    return steps, True


# ---------- the geometry, through every state ----------
def param_states(params: list[dict], steps: list[dict], finals: bool = False) -> list[dict[str, float]]:
    """The parameters at the start, after every step, and part-way through each change (a change moves every
    param of its step along a straight line, so these samples cover its path). `finals`: only the start and the
    state after each step that moves something (where an angle must be drawable)."""
    cur = {p["name"]: float(p["value"]) for p in params}
    out = [dict(cur)]
    for s in steps:
        if not s["set"]:
            continue
        before = dict(cur)
        for st in s["set"]:
            cur[st["param"]] = float(st["to"])
        if not finals:
            out += [{k: before[k] + (cur[k] - before[k]) * f for k in cur} for f in (0.25, 0.5, 0.75)]
        out.append(dict(cur))
    return out


def _rotation(deg: list[float]) -> np.ndarray:
    """three.js Euler(x, y, z, 'XYZ') in degrees: R = Rx Ry Rz."""
    a, b, c = (math.radians(v) for v in deg)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz


class Shape:
    """An object's expressions parsed once, evaluated per state."""

    def __init__(self, obj: dict, params: set[str], axes: dict):
        self.obj, self.axes = obj, axes
        typ = obj["type"]
        names = set(params)
        p = lambda s, extra=(): sx.parse(s, names | set(extra))  # noqa: E731
        self.points = [[p(v) for v in pt] for pt in obj["points"]]
        self.center = [p(v) for v in obj["center"]]
        self.size = [p(v) for v in obj["size"]]
        self.rotation = [p(v) for v in obj["rotation"]]
        self.range = [p(v) for v in obj["range"]]
        self.range2 = [p(v) for v in obj["range2"]]
        self.expr = p(obj["expr"], ("x",) if typ == "graph" else ("x", "y")) if typ in ("graph", "surface") else None
        var = ("t",) if typ == "curve" else ("u", "v")
        self.xyz = [p(obj[k], var) for k in ("x", "y", "z")] if typ in ("curve", "psurface") else None

    def _vals(self, trees: list, env: dict) -> list[float]:
        return [float(np.asarray(sx.evaluate(t, env), dtype=float)) for t in trees]

    def arms(self, env: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """An angle in one state: (vertex, arm 1, arm 2, radius), the arms as vectors from the vertex; None when it
        is degenerate there (an arm shorter than ARM_MIN, anything not finite or beyond LIMIT, a radius not above
        0): the player does not draw it in that state."""
        try:
            v, p1, p2 = (np.array(self._vals(pt, env), dtype=float) for pt in self.points)
            size = self._vals(self.size, env)
        except (ValueError, TypeError, FloatingPointError, ZeroDivisionError):
            return None
        if not all(np.all(np.isfinite(a)) and np.all(np.abs(a) <= LIMIT) for a in (v, p1, p2, np.array(size))):
            return None
        a, b = p1 - v, p2 - v
        la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if not (la >= ARM_MIN and lb >= ARM_MIN):
            return None
        r = size[0] if size else AUTO_RADIUS * min(la, lb)
        if not (math.isfinite(r) and r > 0):
            return None
        return v, a, b, r

    def angle_at(self, env: dict[str, float]) -> float | None:
        """An angle object's measured value in degrees (0 to 180) in one state, or None when it is degenerate."""
        arms = self.arms(env)
        if arms is None:
            return None
        _, a, b, _ = arms
        # atan2(|a x b|, a.b), as the player measures it (angle3d.ts): acos of the cosine loses the small angles
        # (under about 1e-6 degrees it gives 0) and the same near 180
        return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))

    def points_at(self, env: dict[str, float]) -> tuple[np.ndarray | None, int]:
        """(the points that frame it in this state, how many drawable samples it has); (None, 0) when a fixed
        value is not finite or a size is negative: then it cannot be drawn."""
        typ = self.obj["type"]
        fixed = [v for pt in self.points for v in self._vals(pt, env)] + self._vals(self.center, env) \
            + self._vals(self.size, env) + self._vals(self.rotation, env) + self._vals(self.range, env) \
            + self._vals(self.range2, env)
        if not all(math.isfinite(v) and abs(v) <= LIMIT for v in fixed):
            return None, 0
        if typ == "angle":  # framed by its vertex +- its radius; None when it is degenerate in this state
            arms = self.arms(env)
            if arms is None:
                return None, 0
            v, _, _, r = arms
            return np.array([v - r, v + r]), 1
        if typ in POINTS:
            pts = np.array([self._vals(pt, env) for pt in self.points], dtype=float)
            return pts, len(pts)
        if typ in SIZES:
            c = np.array(self._vals(self.center, env))
            s = self._vals(self.size, env)
            if any(v < -1e-9 for v in s):
                return None, 0
            if typ == "sphere":  # turning a sphere changes nothing
                return c + np.array([[-1, -1, -1], [1, 1, 1]], dtype=float) * s[0], 1
            if typ == "box":
                hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
            elif typ == "torus":
                hx = hy = s[0] + s[1]
                hz = s[1]
            else:  # cylinder, cone: axis along z
                hx = hy = s[0]
                hz = s[1] / 2
            local = np.array([[a, b, d] for a in (-hx, hx) for b in (-hy, hy) for d in (-hz, hz)])
            if self.rotation:
                local = local @ _rotation(self._vals(self.rotation, env)).T
            return c + local, 1
        ax = self.axes
        if typ == "graph":
            lo, hi = self._vals(self.range, env) if self.range else (ax["x"]["min"], ax["x"]["max"])
            xs = np.linspace(lo, hi, SAMPLES)
            ys = np.broadcast_to(np.asarray(sx.evaluate(self.expr, {**env, "x": xs}), dtype=float), xs.shape)
            pts = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
            vlo, vhi = ax["y"]["min"], ax["y"]["max"]
            col = 1
        elif typ == "surface":
            g = np.linspace(0, 1, GRID)
            X, Y = np.meshgrid(ax["x"]["min"] + g * (ax["x"]["max"] - ax["x"]["min"]),
                               ax["y"]["min"] + g * (ax["y"]["max"] - ax["y"]["min"]))
            Z = np.broadcast_to(np.asarray(sx.evaluate(self.expr, {**env, "x": X, "y": Y}), dtype=float), X.shape)
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            vlo, vhi = ax["z"]["min"], ax["z"]["max"]
            col = 2
        else:
            if typ == "curve":
                a, b = self._vals(self.range, env)
                var = {"t": np.linspace(a, b, SAMPLES)}
                shape = (SAMPLES,)
            else:
                (a, b), (c, d) = self._vals(self.range, env), self._vals(self.range2, env)
                U, V = np.meshgrid(np.linspace(a, b, GRID), np.linspace(c, d, GRID))
                var = {"u": U, "v": V}
                shape = U.shape
            cols = [np.broadcast_to(np.asarray(sx.evaluate(t, {**env, **var}), dtype=float), shape).ravel()
                    for t in self.xyz]
            pts = np.stack(cols, axis=1)
            vlo = vhi = None
            col = None
        ok = np.all(np.isfinite(pts), axis=1) & np.all(np.abs(pts) <= LIMIT, axis=1)
        drawable = int(ok.sum())
        if col is not None:  # graphs and surfaces are drawn inside their value axis' range
            span = vhi - vlo
            ok &= (pts[:, col] >= vlo - 0.05 * span) & (pts[:, col] <= vhi + 0.05 * span)
        return pts[ok], drawable


def _names(scene: dict) -> set[str]:
    """The names an object's expressions may use besides their field's variables: the params and the defs."""
    return {p["name"] for p in scene["params"]} | {d["name"] for d in scene.get("defs") or []}


def _envs(scene: dict, states: list[dict[str, float]]) -> list[dict[str, float]]:
    defs = _parsed_defs(scene)
    return [with_defs(defs, s) for s in states] if defs else states


def check_geometry(scene: dict, notes: list[str]) -> list[dict[str, float]]:
    """Drop the objects that cannot be drawn in some state (a fixed value not finite, a negative size) or in no
    state at all (a graph, curve or surface with no finite sample). Returns the states."""
    names = _names(scene)
    states = param_states(scene["params"], scene["steps"])
    envs = _envs(scene, states)
    finals = None
    keep = []
    for o in scene["objects"]:
        try:
            shape = Shape(o, names, scene["axes"])
            if o["type"] == "angle":
                # An angle only has to be drawable where the view rests: at the start and after each step. Part-way
                # through a change its arm may pass through zero (the player skips it for those frames).
                if finals is None:
                    finals = _envs(scene, param_states(scene["params"], scene["steps"], finals=True))
                if any(shape.arms(env) is None for env in finals):
                    notes.append(f"object {o['id']} dropped: the angle is degenerate (an arm of length 0, or a "
                                 f"value not finite) at the start or after a step")
                    continue
                keep.append(o)
                continue
            results = [shape.points_at(env) for env in envs]
        except (sx.ExprError, ValueError, TypeError, FloatingPointError) as e:
            notes.append(f"object {o['id']} dropped: could not be evaluated ({e})")
            continue
        if any(pts is None for pts, _ in results):
            notes.append(f"object {o['id']} dropped: not finite (or a negative size) in some state")
            continue
        if o["type"] in SAMPLED and not any(n >= 2 for _, n in results):
            notes.append(f"object {o['id']} dropped: nothing finite to draw")
            continue
        keep.append(o)
    scene["objects"] = keep
    return states


def compute_bounds(scene: dict, states: list[dict[str, float]] | None = None) -> list[list[float]]:
    """[[min x, y, z], [max x, y, z]] over every object in every state, and the axes when they are shown."""
    names = _names(scene)
    states = states if states is not None else param_states(scene["params"], scene["steps"])
    envs = _envs(scene, states)
    chunks = []
    for o in scene["objects"]:
        shape = Shape(o, names, scene["axes"])
        for env in envs:
            pts, _ = shape.points_at(env)
            if pts is not None and len(pts):
                chunks.append(pts)
    if scene["axes"]["show"]:
        ax = scene["axes"]
        chunks.append(np.array([[ax["x"]["min"], ax["y"]["min"], ax["z"]["min"]],
                                [ax["x"]["max"], ax["y"]["max"], ax["z"]["max"]]]))
    if not chunks:
        return [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    allp = np.concatenate(chunks)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    return [[round(float(v), 4) for v in lo], [round(float(v), 4) for v in hi]]


def clean_scene(raw: Any, *, piece: int, pieces: list[int], version: str, clip: tuple[float, float],
                talked: float | None = None, notes: list[str] | None = None,
                words: list[dict] | None = None) -> dict:
    """Muse Spark's answer -> a format-2 scene the player can draw safely. `talked`: when the drawing is first
    talked about (for the step made when no step is usable); `words`: the transcript's words, where each step's
    cue is looked up."""
    notes = notes if notes is not None else []
    raw = raw if isinstance(raw, dict) else {}
    axes = clean_axes(raw.get("axes"), notes)
    renames: dict[str, str] = {}
    params = clean_params(raw.get("params"), notes, renames)
    defs = clean_defs(raw.get("defs"), {p["name"] for p in params}, notes, renames)
    names = {p["name"] for p in params} | {d["name"] for d in defs}
    objects, seen = [], set()
    groups: dict[str, list[str]] = {}
    for i, o in enumerate(raw.get("objects") if isinstance(raw.get("objects"), list) else []):
        for part in split_points(o, groups, notes):
            obj = clean_object(part, i, names, notes, renames)
            if obj is None:
                continue
            if obj["id"] in seen:
                notes.append(f"object {obj['id']} dropped: its id is repeated")
                continue
            if len(objects) == MAX_OBJECTS:
                notes.append(f"object {obj['id']} dropped: at most {MAX_OBJECTS}")
                continue
            seen.add(obj["id"])
            objects.append(obj)
    scene: dict[str, Any] = {
        "format": FORMAT, "version": version, "piece": int(piece), "pieces": [int(p) for p in pieces],
        "title": _str(raw.get("title"), TITLE_CHARS) or "3D view",
        "explain": _str(raw.get("explain"), EXPLAIN_CHARS), "synced": False, "axes": axes, "params": params,
        "defs": defs, "objects": objects, "steps": [], "bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]}

    def steps_for(objs: list[dict], log: list[str]) -> tuple[list[dict], bool]:
        return clean_steps(raw.get("steps"), [o["id"] for o in objs], params, clip, log, words, groups, renames)

    scene["steps"], scene["synced"] = steps_for(objects, notes)
    drop_axis_copies(scene, notes)
    repeated = repeated_labels(scene)
    if repeated:
        for i, other in repeated.items():
            text = next(o["label"] for o in scene["objects"] if o["id"] == i)
            notes.append(f"object {i} dropped: its label {text!r} is said already by {other}, there whenever it is")
        scene["objects"] = [o for o in scene["objects"] if o["id"] not in repeated]
        scene["steps"], scene["synced"] = steps_for(scene["objects"], [])
    while True:
        count = len(scene["objects"])
        states = check_geometry(scene, notes)
        if len(scene["objects"]) == count:
            break
        # Steps may name objects that were just dropped; and with a step gone, one past the limit of 12 can come
        # back in, with states not checked yet: clean the steps again and check again, until nothing is dropped.
        checked = states
        scene["steps"], scene["synced"] = steps_for(scene["objects"], [])
        states = param_states(params, scene["steps"])
        if states == checked:  # every object left has passed these states already
            break
    if not scene["objects"]:
        scene["steps"], scene["synced"] = [], False
        if not scene["explain"]:
            scene["explain"] = "There is nothing in this drawing to show in 3D."
        scene["bounds"] = compute_bounds(scene, states)
        return scene
    link_captions(scene, notes)
    drop_filler(scene, notes)  # first: a hide that changes nothing would hold keep_base back
    keep_base(scene["steps"], scene["objects"], notes)
    if not scene["steps"]:
        # The drawing is there from the start (a step that showed it would leave the view empty until then); the
        # one step makes it glow when the teacher first talks about it.
        notes.append("no usable step: one step points at the whole drawing")
        lo, hi = _starts(clip)
        t = None if talked is None else round(min(hi, max(lo, talked)), 2)
        scene["steps"] = [{"at": clock(t) if t is not None else "", "t": t,
                           "over": OVER_DEFAULT if t is None else _over(OVER_DEFAULT, t, clip[1]),
                           "caption": _str(scene["explain"] or scene["title"], CAPTION_CHARS),
                           "show": [], "hide": [], "highlight": [o["id"] for o in scene["objects"]], "set": [],
                           "camera": "iso"}]
        scene["synced"] = t is not None
    fit_ranges(scene, notes)
    narrow_angle_ranges(scene, notes)
    nest_shared_angles(scene, notes)
    size_short_arms(scene, notes)
    pick_cameras(scene, notes)
    check_captions(scene, notes)
    degree_captions(scene, notes)
    # Dropping steps and sets that moved nothing leaves the same states, without repeats.
    scene["bounds"] = compute_bounds(scene, param_states(params, scene["steps"]))
    return scene


BASE_MIN = 3  # a view before the first step with fewer objects than this is too thin (test2 #7: one edge for 13 s)


def keep_base(steps: list[dict], objects: list[dict], notes: list[str]) -> None:
    """The view is never empty (or one lone line) before the first step: when fewer than BASE_MIN objects are there
    without a step showing them, the objects the first showing step brings are there from the start instead, and
    each step that showed one makes it glow. Only objects no step hides before showing them again: the base is what
    no step shows, so one shown again after a hide cannot be in it."""
    shown = {i for s in steps for i in s["show"]}
    base = [o for o in objects if o["id"] not in shown]
    if not objects or not shown or len(base) >= min(BASE_MIN, len(objects)):
        return
    k = next(i for i, s in enumerate(steps) if s["show"])

    def safe(oid: str) -> bool:  # every hide of it comes after its last show
        last_show = max(i for i, s in enumerate(steps) if oid in s["show"])
        return all(i > last_show for i, s in enumerate(steps) if oid in s["hide"])

    move = [i for i in steps[k]["show"] if safe(i)]
    if not move:
        return
    for s in steps:
        here = [i for i in s["show"] if i in move]
        if here:
            s["show"] = [i for i in s["show"] if i not in move]
            s["highlight"] = s["highlight"] + [i for i in here if i not in s["highlight"]]
    notes.append(f"{len(base)} objects shown before the first step: {', '.join(move)} there from the start")


def _constant_points(o: dict) -> list[list[float]] | None:
    """An object's points when they are plain numbers (no param moves them), else None."""
    try:
        return [[float(np.asarray(sx.evaluate(sx.parse(v, ()), {}), dtype=float)) for v in p] for p in o["points"]]
    except (sx.ExprError, ValueError, TypeError):
        return None


def drop_axis_copies(scene: dict, notes: list[str]) -> None:
    """With the axes shown, a segment or vector that lies along an axis over half its length or more, and that no
    step names, is that axis drawn a second time over the real one (test-full #3's x_axis and y_axis): dropped. One
    a step shows or highlights (an asymptote on the y axis) stays."""
    ax = scene["axes"]
    if not ax["show"]:
        return
    named = {i for s in scene["steps"] for k in ("show", "hide", "highlight") for i in s[k]}
    spans = [ax[k]["max"] - ax[k]["min"] for k in "xyz"]
    keep = []
    for o in scene["objects"]:
        pts = _constant_points(o) if o["type"] in ("segment", "vector") and o["id"] not in named else None
        along = None
        if pts:
            for a in range(3):
                if all(abs(p[b]) <= 0.01 * spans[b] + 1e-9 for p in pts for b in range(3) if b != a) \
                        and abs(pts[1][a] - pts[0][a]) >= 0.5 * spans[a]:
                    along = "xyz"[a]
        if along:
            notes.append(f"object {o['id']} dropped: it draws the {along} axis again")
            continue
        keep.append(o)
    scene["objects"] = keep


def size_short_arms(scene: dict, notes: list[str]) -> None:
    """An angle with the auto radius whose one arm is under half the other (a point one unit along an axis giving
    its direction: synth3d-vector3d's theta, 0.3 of a 1-unit arm beside a 5.4 long vector, a wedge nobody sees)
    gets a radius of its own: 0.15 of the longer arm, at most 0.9 of the shorter one, in every state the steps
    reach (the smallest), so its arc stays inside both arms and is never smaller than the auto one."""
    angles = [o for o in scene["objects"] if o["type"] == "angle" and not o["size"]]
    if not angles:
        return
    names = _names(scene)
    envs = _envs(scene, param_states(scene["params"], scene["steps"], finals=True))
    for o in angles:
        shape = Shape(o, names, scene["axes"])
        best, start = math.inf, None
        for env in envs:
            arms = shape.arms(env)
            if arms is None:
                continue
            _, a, b, auto = arms
            s, big = sorted((float(np.linalg.norm(a)), float(np.linalg.norm(b))))
            best = min(best, 0.9 * s, 0.15 * big)
            if start is None:
                start = (auto, s, big)
        if start is None or not math.isfinite(best) or best <= 1.2 * start[0]:
            continue
        r = float(f"{best:.3g}")
        o["size"] = [f"{r:g}"]
        notes.append(f"object {o['id']}: radius {r:g} (its arms are {start[1]:.3g} and {start[2]:.3g} long: "
                     f"{AUTO_RADIUS:g} of the shorter would hardly show)")


def _length_expr(p: list[str], v: list[str]) -> str:
    """The expression of |p - v| for two points given as expressions."""
    terms = []
    for a, b in zip(p, v):
        d = a if b == "0" else f"({a}) - ({b})"
        if d != "0":
            terms.append(f"({d})^2")
    return f"sqrt({' + '.join(terms) or '0'})"


def nest_shared_angles(scene: dict, notes: list[str]) -> None:
    """Angles with the auto radius that share their vertex in every state the steps reach (alpha, beta and gamma at
    one corner of test2 #4: three marks of radius 0.75 on top of each other, the labels pushed 30-60 px off them)
    get radii 1, 1.45, 1.9... times the group's auto radius (0.3 of the shortest arm of any of them), in object
    order, written as an expression of the arms, so the marks stay nested and inside the arms when a student moves
    the lengths."""
    angles = [o for o in scene["objects"] if o["type"] == "angle" and not o["size"]]
    if len(angles) < 2:
        return
    names = _names(scene)
    envs = _envs(scene, param_states(scene["params"], scene["steps"], finals=True))
    verts: dict[str, list] = {}
    for o in angles:
        shape = Shape(o, names, scene["axes"])
        verts[o["id"]] = [None if (arms := shape.arms(env)) is None else arms[0] for env in envs]

    def same(a: list, b: list) -> bool:
        both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        return bool(both) and all(float(np.linalg.norm(x - y)) <= 1e-6 * max(1.0, float(np.linalg.norm(x)))
                                  for x, y in both)

    groups: list[list[dict]] = []
    for o in angles:
        for g in groups:
            if same(verts[g[0]["id"]], verts[o["id"]]):
                g.append(o)
                break
        else:
            groups.append([o])
    for g in groups:
        if len(g) < 2:
            continue
        arms = list(dict.fromkeys(_length_expr(o["points"][k], o["points"][0]) for o in g for k in (1, 2)))
        for k, o in enumerate(g):
            f = AUTO_RADIUS * (1 + NEST * k)
            try:
                o["size"] = [sx.canonical(f"{f:g}*min({', '.join(arms)})", names)]
            except sx.ExprError:
                continue
        notes.append(f"angles {', '.join(o['id'] for o in g)} share a vertex: radii "
                     f"{', '.join(f'{1 + NEST * k:g}' for k in range(len(g)))} times {AUTO_RADIUS:g} of the "
                     f"shortest arm, so their marks do not lie on each other")


def narrow_angle_ranges(scene: dict, notes: list[str]) -> None:
    """Angle params a student explores together must make a shape that can exist: at the corners of their ranges'
    box (each one at its min or max, the other params as the lecture has them at the start and after each step),
    every angle object named like one of them ("beta" for param beta or beta_d) must measure that param's value,
    within ANGLE_TOL, and three edges from one corner must not lie nearly in one plane (FLAT_MIN). test2 #4 let
    alpha = beta = gamma = 120 (60..120 each): the cell is then flat, its def clamps a height to sqrt(0.2) and the
    labels read 119.9 while the sliders said 120. When a corner fails, every such range is narrowed towards the
    values the lecture uses (the widest that passes, whole degrees), never below them."""
    deg = [p for p in scene["params"] if p["unit"] == "deg"]
    if not deg:
        return
    names = _names(scene)
    pairs = []
    for o in scene["objects"]:
        if o["type"] != "angle":
            continue
        words = set(plain_words(o["label"]).split()) & set(_GREEK)
        for p in deg:
            if words & (set(re.split(r"[\s_]+", plain_words(p["name"]))) | set(plain_words(p["label"]).split())):
                pairs.append((Shape(o, names, scene["axes"]), p))
                break
    if not pairs:
        return
    group = list({p["name"]: p for _, p in pairs}.values())
    states = param_states(scene["params"], scene["steps"], finals=True)
    need = {p["name"]: (min(s[p["name"]] for s in states), max(s[p["name"]] for s in states)) for p in group}
    bases = list({tuple(sorted((k, v) for k, v in s.items() if k not in need)): s for s in states}.values())
    defs = _parsed_defs(scene)

    def ok(rng: dict[str, tuple[float, float]]) -> bool:
        for base in bases:
            for corner in itertools.product(*(rng[p["name"]] for p in group)):
                env = {**base, **dict(zip(need, corner))}
                env = with_defs(defs, env) if defs else env
                corners: dict[tuple, list[np.ndarray]] = {}
                for shape, p in pairs:
                    try:
                        arms = shape.arms(env)
                        a = shape.angle_at(env)
                    except (sx.ExprError, ValueError, TypeError, FloatingPointError, ZeroDivisionError):
                        arms = a = None
                    if arms is None or a is None or abs(a - env[p["name"]]) > ANGLE_TOL:
                        return False
                    dirs = corners.setdefault(tuple(np.round(arms[0], 6)), [])
                    for u in (_unit_vec(arms[1]), _unit_vec(arms[2])):
                        if u is not None and all(float(np.dot(u, w)) < 1 - 1e-9 for w in dirs):
                            dirs.append(u)
                # three edges from one corner (a cell's a, b, c) that lie nearly in one plane: a flat plate, the
                # limit past which no shape has those angles (alpha = beta = gamma = 120 on test2 #4)
                for dirs in corners.values():
                    if len(dirs) == 3 and float(np.linalg.det(np.array(dirs))) ** 2 < FLAT_MIN:
                        return False
        return True

    def at(f: float) -> dict[str, tuple[float, float]]:
        return {p["name"]: (need[p["name"]][0] - f * (need[p["name"]][0] - p["min"]),
                            need[p["name"]][1] + f * (p["max"] - need[p["name"]][1])) for p in group}

    if ok(at(1.0)):
        return
    if not ok(at(0.0)):
        notes.append(f"angles {', '.join(p['name'] for p in group)}: the lecture's own values do not make the "
                     f"shape at some corner of their ranges; ranges left as they are")
        return
    lo_f, hi_f = 0.0, 1.0
    for _ in range(14):
        mid = (lo_f + hi_f) / 2
        lo_f, hi_f = (mid, hi_f) if ok(at(mid)) else (lo_f, mid)
    rng = at(lo_f)
    for p in group:
        a, b = rng[p["name"]]
        a = min(need[p["name"]][0], float(math.ceil(a - 1e-9)))
        b = max(need[p["name"]][1], float(math.floor(b + 1e-9)))
        if (a, b) != (p["min"], p["max"]):
            notes.append(f"param {p['name']}: range narrowed from {_round(p['min']):g}..{_round(p['max']):g} to "
                         f"{_round(a):g}..{_round(b):g}: past it the angles together make no shape")
            p["min"], p["max"] = a, b


def _unit_vec(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if math.isfinite(n) and n > 1e-9 else None


def pick_cameras(scene: dict, notes: list[str]) -> None:
    """Each step looks from where what it points at is seen: the vectors and segments it shows or highlights as
    long as they are (the sine of their angle to the line of sight), the angles it shows or highlights face on (|their
    plane's normal . the line of sight|). synth3d-vector3d's v, the scene's main object, is (2, -3, 4): from "iso"
    the camera looked along it (24 degrees off), so it showed at 0.41 of its length and theta's wedge as a sliver;
    "side" shows it at 0.91. When the camera in effect scores under CAMERA_LOW and another preset scores at least
    CAMERA_BETTER more, the step turns to that one (the steps after it that keep the camera keep it)."""
    steps = scene["steps"]
    kinds = ("segment", "vector", "angle")
    if not steps or not any(o["type"] in kinds for o in scene["objects"]):
        return
    names = _names(scene)
    shapes = {o["id"]: (o["type"], Shape(o, names, scene["axes"])) for o in scene["objects"] if o["type"] in kinds}
    dirs = {k: np.array(v, dtype=float) / np.linalg.norm(v) for k, v in CAMERA_DIRS.items()}
    defs = _parsed_defs(scene)

    def seen(ids: Iterable[str], env: dict) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
        """The line directions and angle planes of these objects in this state, and the ids they come from."""
        lines, planes, what = [], [], []
        for i in ids:
            if i not in shapes:
                continue
            typ, shape = shapes[i]
            try:
                if typ == "angle":
                    arms = shape.arms(env)
                    u = None if arms is None else _unit_vec(np.cross(arms[1], arms[2]))
                    if u is not None:
                        planes.append(u)
                        what.append(i)
                else:
                    pts, _ = shape.points_at(env)
                    u = None if pts is None or len(pts) < 2 else _unit_vec(pts[-1] - pts[0])
                    if u is not None:
                        lines.append(u)
                        what.append(i)
            except (sx.ExprError, ValueError, TypeError, FloatingPointError, ZeroDivisionError):
                continue
        return lines, planes, what

    def scores_of(lines: list[np.ndarray], planes: list[np.ndarray], d: np.ndarray) -> list[float]:
        return [math.sqrt(max(0.0, 1.0 - float(np.dot(u, d)) ** 2)) for u in lines] + \
               [abs(float(np.dot(u, d))) for u in planes]

    def walk():
        """(step number, step, its state, objects visible after it, objects it shows, camera in effect) per step."""
        cur = {p["name"]: float(p["value"]) for p in scene["params"]}
        vis = _visible(scene)
        cam = next((s["camera"] for s in steps if s["camera"] != "keep"), "iso")
        for n, s in enumerate(steps, 1):
            cur.update({st["param"]: float(st["to"]) for st in s["set"]})
            env = with_defs(defs, dict(cur)) if defs else dict(cur)
            new = [i for i in s["show"] if i not in vis]
            vis = (vis | set(s["show"])) - set(s["hide"])
            cam = cam if s["camera"] == "keep" else s["camera"]
            yield n, s, env, vis, new, cam

    # "iso" looks from the front-right corner, the way a board drawing shows x coming towards us. A vector that
    # points into that corner is a stub from there (synth3d-vector3d: v = (2, -3, 4) at 0.41 of its length in steps
    # 3-5, which point at its components); from the front-left corner it is 0.85 and every axis 0.82. When the worst
    # seen line or angle on screen in the iso steps scores under CAMERA_LOW, and from the left at least CAMERA_BETTER
    # more, the scene's iso looks from the left (the player's iso_from).
    left = np.array(ISO_LEFT, dtype=float) / np.linalg.norm(ISO_LEFT)
    worst = {"right": 1.0, "left": 1.0}
    for _, _, env, vis, _, cam in walk():
        if cam == "iso":
            lines, planes, _ = seen(sorted(vis), env)
            for side, d in (("right", dirs["iso"]), ("left", left)):
                worst[side] = min([worst[side], *scores_of(lines, planes, d)])
    if worst["right"] < CAMERA_LOW and worst["left"] >= worst["right"] + CAMERA_BETTER:
        scene["iso_from"] = "left"
        dirs["iso"] = left
        notes.append(f"iso looks from the front-left: the worst line on screen is seen {worst['left']:.2f} face on, "
                     f"not {worst['right']:.2f}")
    chosen = next((s["camera"] for s in steps if s["camera"] != "keep"), "iso")
    for n, s, env, vis, new, _ in walk():
        chosen = chosen if s["camera"] == "keep" else s["camera"]  # the one in effect, with the turns made below
        lines, planes, what = seen([i for i in dict.fromkeys(new + s["highlight"]) if i in vis], env)
        if what:
            def score(k: str) -> float:
                vals = scores_of(lines, planes, dirs[k])
                return sum(vals) / len(vals)

            scores = {k: score(k) for k in CAMERA_DIRS}
            best = max(scores, key=lambda k: scores[k])
            if scores[chosen] < CAMERA_LOW and scores[best] >= scores[chosen] + CAMERA_BETTER:
                notes.append(f"step {n}: camera {best} instead of {chosen}: {', '.join(what)} seen "
                             f"{scores[best]:.2f} face on, not {scores[chosen]:.2f}")
                s["camera"] = chosen = best


def repeated_labels(scene: dict) -> dict[str, str]:
    """Label objects that only say again a name another object shows already: a "label" whose text is the label of
    an object of another type (an edge, a vector, an angle) that is visible in every state the label object is in
    (test2 #7, rules 6: lab_a, lab_b, lab_c beside vec_a, vec_b, vec_c put "a", "b", "c" twice on screen). Greek
    names match their letters. {label object id: the object that shows its name}."""
    shown = {i for s in scene["steps"] for i in s["show"]}
    vis = {o["id"] for o in scene["objects"] if o["id"] not in shown}
    states = [set(vis)]
    for s in scene["steps"]:
        vis = (vis | set(s["show"])) - set(s["hide"])
        states.append(set(vis))
    others: dict[str, list[str]] = {}
    for o in scene["objects"]:
        if o["type"] != "label" and o["label"]:
            others.setdefault(plain_words(o["label"]), []).append(o["id"])
    out = {}
    for o in scene["objects"]:
        if o["type"] != "label" or not plain_words(o["label"]):
            continue
        for other in others.get(plain_words(o["label"]), []):
            if all(other in st for st in states if o["id"] in st):
                out[o["id"]] = other
                break
    return out


_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "theta": "θ", "phi": "φ", "psi": "ψ",
          "omega": "ω", "lambda": "λ", "mu": "μ", "sigma": "σ", "rho": "ρ", "tau": "τ"}
_LETTERS = {**{v: k for k, v in _GREEK.items()}, "ς": "sigma", "ϕ": "phi", "ϑ": "theta", "µ": "mu"}
CAPTION_LINKS = 3  # objects a caption may bring forward in one step
_ANGLE_WORD = re.compile(r"(?<!\w)angles?(?!\w)")


def plain_words(s: str) -> str:
    """Lower case, Greek letters written as their names ("Angle β" -> "angle beta"), so that a caption and a label
    name one thing the same way whichever they use."""
    s = "".join(f" {_LETTERS[c]} " if c in _LETTERS else c for c in s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _visible(scene: dict) -> set[str]:
    """The objects visible before the first step: those no step shows (STATE RULES in scene3dState.ts)."""
    shown = {i for s in scene["steps"] for i in s["show"]}
    return {o["id"] for o in scene["objects"] if o["id"] not in shown}


def link_captions(scene: dict, notes: list[str]) -> None:
    """A step whose caption names an object by its label points at it (test2 #7 under rules 5: "Alpha and beta 90
    degrees, standing straight" pointed at nothing): visible and not highlighted, it is highlighted; hidden, it is
    shown. Whole words, any
    case, a Greek name as a word or a letter ("beta" or "β"); labels of 2 or more characters only, so "a", "b" and
    "c" never match the words a caption is made of. At most CAPTION_LINKS per step. One the step itself shows is
    visible then, so it is highlighted too. One the step hides is left as it is, and so is one of the base drawing
    that an earlier step hid: showing it again would take it out of the base (what no step shows), so it would be
    missing from the start."""
    base = _visible(scene)
    vis = set(base)
    named = []
    for o in scene["objects"]:
        word = plain_words(o["label"])
        if len(word) >= 2:
            named.append((o["id"], o["label"], re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)")))
    angles = [(o["id"], o["label"] or o["id"]) for o in scene["objects"] if o["type"] == "angle"]
    for n, s in enumerate(scene["steps"], 1):
        after = (vis | set(s["show"])) - set(s["hide"])
        caption = plain_words(s["caption"])
        hits = sorted((m.start(), k, oid, label) for k, (oid, label, rx) in enumerate(named)
                      if (m := rx.search(caption)))
        # "all angles 90 degrees" names every angle not on screen yet (test2 #4: 90 s of "all angles 90" with no
        # angle marked; they came at "Rhombohedral")
        if (m := _ANGLE_WORD.search(caption)):
            hits = sorted(hits + [(m.start(), len(named) + k, oid, f"{m.group(0)} ({label})")
                                  for k, (oid, label) in enumerate(angles) if oid not in after])
        added, seen = 0, set()
        for _, _, oid, label in hits:
            if added == CAPTION_LINKS:
                break
            if oid in seen:
                continue
            seen.add(oid)
            if oid in s["hide"]:
                continue
            if oid in after:
                if oid in s["highlight"]:
                    continue
                s["highlight"].append(oid)
                notes.append(f"step {n}: {oid} highlighted: its caption names {label!r}")
            elif oid in base:
                continue
            else:
                s["show"].append(oid)
                after.add(oid)
                notes.append(f"step {n}: {oid} shown: its caption names {label!r}")
            added += 1
        vis = after


def drop_filler(scene: dict, notes: list[str]) -> None:
    """A step must change what is on screen: a "set" that leaves a param where it is goes, and so does a step that
    then shows nothing new, hides nothing shown, highlights nothing, moves nothing and keeps the camera where it is
    (rules 5: test2 #4's "Leans over in one direction only" set all six params to the values they had, and test2
    #7's "Alpha and beta 90 degrees, standing straight" changed nothing at all). The camera before the first
    step is the first step's that names one (STATE RULES): when the step that named it goes, the first step left
    takes it over, so the view starts the same."""
    steps = scene["steps"]
    vis = _visible(scene)
    cur = {p["name"]: float(p["value"]) for p in scene["params"]}
    start_cam = next((s["camera"] for s in steps if s["camera"] != "keep"), "iso")
    cam = start_cam
    kept = []
    for n, s in enumerate(steps, 1):
        moves, still = [], []
        for st in s["set"]:
            was = cur[st["param"]]
            (still if abs(st["to"] - was) <= 1e-9 * max(1.0, abs(was)) else moves).append(st)
        if still:
            notes.append(f"step {n}: set of {', '.join(st['param'] for st in still)} removed: already there")
        s["set"] = moves
        changes = (any(i not in vis for i in s["show"]) or any(i in vis for i in s["hide"]) or s["highlight"]
                   or moves or s["camera"] not in ("keep", cam))
        vis = (vis | set(s["show"])) - set(s["hide"])
        cur.update({st["param"]: float(st["to"]) for st in moves})
        if s["camera"] != "keep":
            cam = s["camera"]
        if changes:
            kept.append(s)
        else:
            notes.append(f"step {n} dropped: it changes nothing on screen ({s['caption'][:40]!r})")
    if kept and next((s["camera"] for s in kept if s["camera"] != "keep"), "iso") != start_cam \
            and kept[0]["camera"] == "keep":
        kept[0]["camera"] = start_cam
    scene["steps"] = kept


def fit_ranges(scene: dict, notes: list[str]) -> None:
    """Every param's range holds its value and every step's "to" (fit_range)."""
    for p in scene["params"]:
        values = [float(p["value"])] + [float(st["to"]) for s in scene["steps"] for st in s["set"]
                                        if st["param"] == p["name"]]
        p["min"], p["max"] = fit_range(p["name"], (p["min"], p["max"]), values, p["unit"], notes)


def check_captions(scene: dict, notes: list[str]) -> None:
    """Logs (in the run log) a step whose caption names a param ("gamma") while nothing shown then carries that
    name in its label: the viewer reads about an angle the view does not point out (test2 #7 v3, varying from one
    call to the next). Nothing is changed."""
    terms = set()
    for p in scene["params"]:
        for s in (p["name"], p["label"]):
            s = s.strip().lower()
            if re.fullmatch(r"[a-z]{3,}", s):
                terms.add(s)
    if not terms or not scene["steps"]:
        return
    shown = {i for s in scene["steps"] for i in s["show"]}
    vis = {o["id"]: o["id"] not in shown for o in scene["objects"]}
    labels = {o["id"]: o["label"].lower() for o in scene["objects"]}
    for n, s in enumerate(scene["steps"], 1):
        vis.update({i: True for i in s["show"]})
        vis.update({i: False for i in s["hide"]})
        for term in sorted(terms & set(re.findall(r"[a-z]+", s["caption"].lower()))):
            if not any(vis[i] and (term in re.findall(r"[a-z]+", lab) or _GREEK.get(term, "\0") in lab)
                       for i, lab in labels.items()):
                notes.append(f"step {n}: the caption names {term}, but nothing shown is labelled {term}")


# What may stand between an angle's name and its value ("angles still 90", "beta is not 90", "alpha = 90"); any other
# word means the number is something else ("the angle between 2 lines", "angle theta at point 3, 2, 4").
_DEG_LINK = frozenset("is are = ≈ equals equal of at still not about now all both and stay stays remain remains "
                      "becomes become to exactly only just also each".split())
# The clause is about a ratio or radians, not degrees ("cos theta is 3 over root 29", "the cosine of theta is 0.6").
_DEG_RATIO = re.compile(r"(?<![^\W\d_])(?:cos|sin|tan|cot|sec|csc|cosine|sine|tangent|ratio|radians?|rad|pi|π)"
                        r"(?![^\W\d_])", re.I)
# A unit or a word right after the number ("90 degrees", "1.57 radians", "3 over", "3 cm", "2 times").
_DEG_UNIT = re.compile(r"\s*(?:[°º˚%/]|deg(?:rees?)?\b|rad(?:ians?)?\b|over\b|times\b|units?\b|percent\b|[ckm]?m\b|"
                       r"metres?\b|meters?\b|f(?:ee|oo)t\b|seconds?\b|minutes?\b|x\b)", re.I)
# What may follow an angle value: the phrase ends there ("alpha gamma 90 and beta not 90", "gamma 120 at the base").
_DEG_AFTER = frozenset("and but or so then now here too again each while at in on for with from".split())
# The clause counts angles: "the number of angles is 12", "how many angles... 12".
_DEG_COUNT = re.compile(r"\b(?:number of|how many|count)\b", re.I)
_DEG_TOKEN = re.compile(r"[^\W\d_]+|\d+(?:\.\d+)?|[=≈]|[^\w\s]")


def degree_captions(scene: dict, notes: list[str]) -> None:
    """A bare angle value in a caption gets its degree sign: test2 #7 said "angles 90", "angles still 90" and "alpha
    gamma 90 and beta not 90" (the teacher's words) under a view that shows "β 90°". Strict, since a wrong sign is
    worse than a missing one (two reviews turned counts, indices, coefficients, multiples of pi and radians into
    degrees with word rules alone: "angles at all 8 corners", "theta 1", "2 theta", "2 pi over 3"). The number:
    - is a value one of the scene's angle params takes (its start or a step's "to", within ANGLE_TOL), 10..360;
    - comes right after "angles", an angle param's name (2 letters or more) or its Greek letter, or after up to 3 of
      those names and _DEG_LINK words ("beta is not 90"), never right after the singular "angle" ("angle 1");
    - ends its phrase: the caption, a punctuation mark or a _DEG_AFTER word follows ("90 and", "120 at the base"),
      not a noun or a unit ("12 bonds", "90 degrees", "30 degs"), and not another value ("10 and 30": which is which);
    - is not glued to anything ("3D", "90s", "1:30", "90-degree"), and its clause names no ratio, pi or radians, nor
      a count ("the number of angles is 12").
    A caption the sign would take past CAPTION_CHARS is left as it is (the player would cut its last words)."""
    names = {"angles", "angle"}
    values = set()
    deg = {p["name"] for p in scene["params"] if p["unit"] == "deg"}
    for p in scene["params"]:
        if p["unit"] != "deg":
            continue
        values.add(float(p["value"]))
        for s in (p["name"], p["label"]):
            w = plain_words(s)
            if re.fullmatch(r"[a-z]{2,}", w):
                names.add(w)
                if w in _GREEK:
                    names.add(_GREEK[w])
            elif len(s.strip()) == 1 and s.strip() in _LETTERS:
                names.add(s.strip())
    values |= {float(st["to"]) for step in scene["steps"] for st in step.get("set", []) if st.get("param") in deg}
    if names == {"angles", "angle"}:
        return

    def is_value(v: float) -> bool:
        return 10 <= v <= 360 and any(abs(v - x) <= ANGLE_TOL for x in values)

    num = re.compile(r"[0-9]+(?:\.[0-9]+)?")
    for n, s in enumerate(scene["steps"], 1):
        text = s["caption"]
        toks = list(_DEG_TOKEN.finditer(text))
        words = [t.group(0).lower() for t in toks]
        ends = [k for k, w in enumerate(words) if w in ",;:.!?"]
        marks = []
        trigger, links, clause = None, 0, 0  # the name the chain started at, words since it, where the clause began
        for k, t in enumerate(toks):
            w = words[k]
            if w in names:
                if trigger is None:
                    trigger, links = t, 0
                else:
                    links += 1
                continue
            if trigger is not None and w in _DEG_LINK and links < 3:
                links += 1
                continue
            if trigger is not None and num.fullmatch(w):
                end = next((e for e in ends if e > k), len(toks))  # the clause's end (its punctuation mark)
                stop = toks[end].start() if end < len(toks) else len(text)
                after = words[k + 1] if k + 1 < len(toks) else ""
                ok = (is_value(float(w))
                      and not (links == 0 and trigger.group(0).lower() == "angle")
                      and t.start() > trigger.end()  # "θ90" is no value
                      and not re.match(r"[^\s,;.!?)]", text[t.end():t.end() + 1])  # glued: "3D", "90s", "1:30"
                      and not re.match(r"\.\w", text[t.end():t.end() + 2])
                      and not _DEG_UNIT.match(text, t.end())
                      and (after == "" or after in ",;.!?)" or after in _DEG_AFTER)
                      and not (after in ("and", "or", ",") and k + 2 < len(toks) and num.fullmatch(words[k + 2]))
                      and not _DEG_RATIO.search(text, clause, stop)
                      and not _DEG_COUNT.search(text, clause, t.start()))
                if ok:
                    marks.append(t.end())
            if w in ",;:.!?":
                clause = t.end()
            trigger = None
        if not marks:
            continue
        out, at = [], 0
        for m in marks:
            out += [text[at:m], "°"]
            at = m
        new = "".join(out) + text[at:]
        if len(new) > CAPTION_CHARS:
            notes.append(f"step {n}: no degree sign added: the caption would pass {CAPTION_CHARS} characters")
            continue
        s["caption"] = new
        notes.append(f"step {n}: degree sign added: {new[:60]!r}")


# ---------- making and keeping scenes ----------
def scene_version(raw: bytes) -> str:
    return hashlib.sha256(raw + b"|scene-rules-" + SCENE_RULES_VERSION.encode()).hexdigest()[:16]


def scene_path(job_dir: Path, piece_id: int) -> Path:
    return job_dir / POPOUT / f"{piece_id:03d}.v{FORMAT}.json"


def _load(path: Path) -> dict:
    for _ in range(20):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)  # Windows can refuse a read while the file is being replaced
    return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if not replace_file(tmp, path):
        tmp.unlink(missing_ok=True)
        raise OSError(f"could not save {path.name}: the file is in use")


def cached_scene(job_dir: Path, piece_id: int, version: str) -> dict | None:
    """The saved scene of a drawing when it is this lecture version's (old format-1 files are never read)."""
    data = _load(scene_path(job_dir, piece_id))
    if data.get("format") == FORMAT and data.get("version") == version and isinstance(data.get("objects"), list):
        return data
    return None


_locks: dict[str, threading.Lock] = {}
_failures: dict[str, int] = {}
_guard = threading.Lock()


def _key(job_dir: Path, piece_id: int) -> str:
    return f"{job_dir.resolve()}|{piece_id}"


def _piece_lock(key: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(key, threading.Lock())


def ask_scene(client: Any, job_dir: Path, replay: dict, piece_id: int, version: str, attempt: int = 1,
              reasoning: str = REASONING) -> tuple[dict, dict]:
    """One Muse Spark call (two when the first answer is cut short, not JSON, or has nothing drawable: the second
    is a different request, so the disk cache does not hand back the same answer). (scene, run log)."""
    _, clip0, clip1 = ask._video(replay)
    calls: list[dict] = []
    failed: list[str] = []
    for k in range(2):
        tries = attempt + k
        msgs, key, got = scene_messages(job_dir, replay, piece_id, tries)
        n0 = len(getattr(client, "calls", None) or [])
        t = time.perf_counter()
        reply = client.chat(msgs, schema=SCENE_SCHEMA, schema_name="scene_3d", reasoning=reasoning,
                            max_tokens=MAX_TOKENS, extra={"prompt_cache_key": key})
        recs = (getattr(client, "calls", None) or [])[n0:]
        usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
        finish = ((reply.get("choices") or [{}])[0] or {}).get("finish_reason")
        calls.append({"try": tries, "seconds": round(time.perf_counter() - t, 2), "usage": usage, "finish": finish,
                      "cached": bool(recs) and all(getattr(r, "cached", False) for r in recs),
                      "cost_usd": sum(float(r.cost_usd) for r in recs if getattr(r, "cost_usd", None) is not None)})
        try:
            if finish == "length":
                raise MetaAPIError("the answer was cut short")
            data = parse_json_object(chat_text(reply))
            notes: list[str] = []
            talked = first_talked(replay, [got["anchor"]])
            scene = clean_scene(data, piece=piece_id, pieces=[p["id"] for p in got["pieces"]], version=version,
                                clip=(clip0, clip1), talked=talked, notes=notes, words=ask._words(replay))
            if not scene["objects"] and isinstance(data.get("objects"), list) and data["objects"]:
                raise MetaAPIError("none of the 3D objects in the answer could be drawn")
        except MetaAPIError as e:
            failed.append(f"try {tries}: {e}")
            if k:
                raise
            continue
        log = {"version": version, "piece": piece_id, "pieces": scene["pieces"], "reasoning": reasoning,
               "calls": calls, "failed": failed, "repairs": notes, "answer": data}
        return scene, log
    raise MetaAPIError("no answer")  # not reached


def for_piece(scene: dict, piece_id: int) -> dict:
    """A drawing's scene as the answer for one of its pieces: "piece" is the asked one, first in "pieces"."""
    out = dict(scene)
    out["piece"] = int(piece_id)
    out["pieces"] = [int(piece_id)] + [p for p in scene.get("pieces") or [] if p != piece_id]
    return out


def refining(job_dir: Path, replay: dict) -> bool:
    """True while the job still works out the word times: replay.json is written first with rough (proportional)
    ones and again when the aligner is done. A scene made in between would be synced to the rough times, paid
    for, then made again by the new version."""
    if (replay.get("transcript") or {}).get("timing") != "proportional":
        return False
    st = _load(job_dir / "status.json")
    return st.get("state") in ("queued", "running")


def prepare(job_dir: Path, piece_id: int) -> tuple[dict, str, int]:
    """(the lecture, its version, the drawing the piece opens) for a 3D view; NoScene (404) or NotReady (409)
    when there is none to make now."""
    try:
        raw, replay = read_replay(job_dir)
    except NotReady:
        if not (job_dir / "replay.json").exists():
            raise NoScene("this lecture is not processed yet") from None
        raise NotReady("the lecture is being updated; try again in a moment") from None
    if replay.get("type") != "board":
        raise NoScene("3D views are made for board lectures only")
    piece = next((p for p in (replay.get("board") or {}).get("pieces") or []
                  if isinstance(p, dict) and p.get("id") == piece_id), None)
    if piece is None:
        raise NoScene("no such piece")
    if not can_popout(piece):
        raise NoScene("only drawings and the equations of graphs have a 3D view")
    if refining(job_dir, replay):
        raise NotReady("the word times are still being worked out: try again in a minute")
    return replay, scene_version(raw), drawing_id(replay, piece_id)


def make_scene(make_client: Callable[[], Any], job_dir: Path, piece_id: int, reasoning: str = REASONING) -> dict:
    """POST popout/{piece}: the 3D scene of the drawing the piece belongs to, from the job folder when it is there
    for this lecture version, else made by Muse Spark (one call per drawing at a time: a second request, from any
    of its pieces, waits for the first one's answer). NoScene (404), NotReady (409), MetaAPIError (502)."""
    replay, version, drawing = prepare(job_dir, piece_id)
    scene = cached_scene(job_dir, drawing, version)
    if scene is not None:
        return for_piece(scene, piece_id)
    key = _key(job_dir, drawing)
    with _piece_lock(key):
        scene = cached_scene(job_dir, drawing, version)  # made while this request waited
        if scene is not None:
            return for_piece(scene, piece_id)
        with _guard:
            attempt = 1 + _failures.get(key, 0)
        started = time.perf_counter()
        client = None
        try:
            client = make_client()
            scene, log = ask_scene(client, job_dir, replay, drawing, version, attempt, reasoning)
        except Exception:
            # The next request asks with new text: the disk cache may hold the answers that failed, and one the
            # checks tripped over (not a MetaAPIError) would otherwise come back on every request.
            with _guard:
                _failures[key] = _failures.get(key, 0) + 2
            raise
        finally:
            close = getattr(getattr(client, "http", None), "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
        _save(scene_path(job_dir, drawing), scene)
        with _guard:
            _failures.pop(key, None)
        costs = [c["cost_usd"] for c in log["calls"]]
        log.update(seconds=round(time.perf_counter() - started, 2), cost_usd=round(sum(costs), 6),
                   steps=len(scene["steps"]), objects=len(scene["objects"]), synced=scene["synced"],
                   asked=piece_id)
        try:
            _save(scene_path(job_dir, drawing).with_name(f"{drawing:03d}.v{FORMAT}-log.json"), log)
        except OSError:
            pass
        return for_piece(scene, piece_id)


_SCENE_FILE = re.compile(rf"^(\d{{3,}})\.v{FORMAT}\.json$")


def list_scenes(job_dir: Path) -> dict:
    """GET popouts: the drawings whose 3D view is made already and still this lecture version's and these rules'
    (the very files POST would hand back without a model call), so the board can mark them. Never makes anything.
    {"drawings": [{"drawing", "pieces", "title", "steps", "synced", "times"}]}, by drawing; "times" are the steps'
    times when synced, else []. A slide lecture, or one not processed yet, has none. NotReady while replay.json is
    being replaced.
    The board badges each listed drawing and a click asks POST popout/{drawing}, so a drawing is listed only when
    that request is answered from its file (prepare, then cached_scene): its piece is there and has a 3D view, it
    opens itself (drawing_id), and the word times are not being worked out. A file that cannot be read is left out:
    one broken file never breaks the list."""
    if not (job_dir / "replay.json").exists():
        return {"drawings": []}
    raw, replay = read_replay(job_dir)
    if replay.get("type") != "board" or refining(job_dir, replay):
        return {"drawings": []}
    version = scene_version(raw)
    board = replay.get("board") if isinstance(replay.get("board"), dict) else {}
    by_id = {p.get("id"): p for p in board.get("pieces") or [] if isinstance(p, dict)}
    found = []
    folder = job_dir / POPOUT
    for f in folder.iterdir() if folder.is_dir() else []:
        m = _SCENE_FILE.match(f.name)
        if not m:
            continue
        drawing = int(m.group(1))
        try:
            if not can_popout(by_id.get(drawing)) or drawing_id(replay, drawing) != drawing:
                continue
            scene = cached_scene(job_dir, drawing, version)
            if scene is not None:
                found.append(_made(drawing, scene))
        except Exception:
            continue
    found.sort(key=lambda d: d["drawing"])
    return {"drawings": found}


def _made(drawing: int, scene: dict) -> dict:
    """One drawing of list_scenes, read from its scene as the player reads it (cleanScene): a synced scene keeps the
    steps that have a time, in time order."""
    steps = [s for s in scene.get("steps") if isinstance(s, dict)] if isinstance(scene.get("steps"), list) else []
    times = sorted(float(s["t"]) for s in steps if _num(s.get("t")))
    synced = scene.get("synced") is True and bool(times)
    pieces = scene.get("pieces") if isinstance(scene.get("pieces"), list) else []
    pieces = [p for p in pieces if isinstance(p, int) and not isinstance(p, bool)]
    title = scene.get("title")
    return {"drawing": drawing, "pieces": pieces or [drawing], "title": title if isinstance(title, str) else "",
            "steps": len(times) if synced else len(steps), "synced": synced, "times": times if synced else []}


def plain(e: BaseException) -> str:
    """One line for the player."""
    if isinstance(e, MetaAPIError):
        text = str(e).splitlines()[0] if str(e) else "unknown error"
        return f"Muse Spark could not make the 3D view: {text}"
    if isinstance(e, (NoScene, NotReady)):
        return str(e.args[0]) if e.args else type(e).__name__
    return ask.plain(e).replace("ask-error.log", "popout-error.log")


# ---------- making scenes ahead of time: python -m app.popout <job> <piece> [<piece>...] ----------
COST_HINT = (0.03, 0.15)  # $ per drawing, measured (2026-11-02): a graph about 0.03, a seven-case series 0.07-0.14


def main(argv: list[str] | None = None) -> int:
    """Makes the 3D views of some drawings now, so they open at once in a demo. Shows what it may cost and asks
    first (--yes skips the question). One plain line per drawing; errors are one plain line."""
    import argparse
    import sys

    from .pipeline import JOBS

    ap = argparse.ArgumentParser(prog="python -m app.popout",
                                 description="Make the 3D views of board drawings ahead of time (Muse Spark).")
    ap.add_argument("job", help="the job's folder name in data/jobs (or a path to it)")
    ap.add_argument("pieces", type=int, nargs="+", help="board piece ids (a drawing, or a piece inside it)")
    ap.add_argument("--yes", action="store_true", help="do not ask before spending API credit")
    a = ap.parse_args(argv)
    job_dir = Path(a.job) if Path(a.job).is_dir() else JOBS / a.job
    if not job_dir.is_dir():
        print(f"error: no job folder {job_dir}", file=sys.stderr)
        return 1
    todo: dict[int, int] = {}
    try:
        for pid in a.pieces:
            replay, version, drawing = prepare(job_dir, pid)
            if cached_scene(job_dir, drawing, version) is not None:
                print(f"#{pid}: drawing #{drawing} already has its 3D view")
            elif drawing not in todo:
                todo[drawing] = pid
    except (NoScene, NotReady) as e:
        print(f"error: #{pid}: {plain(e)}", file=sys.stderr)
        return 1
    if not todo:
        return 0
    lo, hi = COST_HINT[0] * len(todo), COST_HINT[1] * len(todo)
    print(f"{len(todo)} 3D view(s) to make ({', '.join(f'#{d}' for d in todo)}): about ${lo:.2f} to ${hi:.2f} of "
          f"API credit and up to {2 * len(todo)} minutes.")
    if not a.yes:
        try:
            ok = input("Make them now? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            ok = False
        if not ok:
            print("nothing made")
            return 0
    from .core.meta_api import MetaClient

    spent, failed = 0.0, 0
    for drawing in todo:
        try:
            scene = make_scene(MetaClient, job_dir, drawing)
        except KeyboardInterrupt:
            print("stopped", file=sys.stderr)
            return 130
        except Exception as e:
            failed += 1
            print(f"error: #{drawing}: {plain(e)}", file=sys.stderr)
            continue
        log = _load(scene_path(job_dir, drawing).with_name(f"{drawing:03d}.v{FORMAT}-log.json"))
        cost = float(log.get("cost_usd") or 0.0)
        spent += cost
        print(f"#{drawing}: {scene['title']!r}, {len(scene['objects'])} objects, {len(scene['steps'])} steps"
              f"{' on the lecture times' if scene['synced'] else ' (not talked about)'}, "
              f"{log.get('seconds', '?')} s, ${cost:.3f}")
    print(f"spent ${spent:.3f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
