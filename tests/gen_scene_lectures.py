"""Generated board lectures about 3D figures, with narration and a known truth (for the 3D steps).

A camera looks at a whiteboard on a wall. A teacher draws one 3D-ish figure stroke by stroke and explains it in
spoken stages, the way a real lecture builds up a drawing. Three lectures, chosen to cover different objects:

  vector3d  axes, the vector v = (3, 2, 4), its components one by one, the angle theta with the x axis, |v| = sqrt 29
  cylinder  a cylinder of radius r and height h, V = pi r^2 h, doubling the height, then the radius
  saddle    z = x^2 - y^2: axes, the parabola along x (up), the one along y (down), the outline, the saddle point

Each lecture is planned from a script of beats (pure code, no rendering: `plan`), so the timeline can be tested
fast. Building it speaks every beat with Windows SAPI (System.Speech, the "Microsoft David" voice), plans again with
the real speech times (each drawing starts at the word that names it), renders the video and muxes the sound.
Next to each video, a truth JSON keeps the exact narration (with SAPI's word times), the stages in order (start,
end, cue words and when they are said, what is drawn, what is true of the figure then), every ink item's time and
box, and the 3D facts the drawing shows.

Usage (PowerShell, from the repo):
    .venv\\Scripts\\python -m tests.gen_scene_lectures                     # all three -> data\\synth3d\\
    .venv\\Scripts\\python -m tests.gen_scene_lectures saddle --force      # one, made again
    .venv\\Scripts\\python -m tests.gen_scene_lectures vector3d --stills 12 40   # PNG frames only (no audio)
Without --force, a lecture whose video is current but whose truth is older (TRUTH_VERSION) gets only its truth JSON
rewritten, from the cached speech, and only if the timeline comes out exactly as the video was made.
Then each one goes through the real pipeline:
    .venv\\Scripts\\python -m app.process data\\synth3d\\vector3d.mp4 --no-gestures --job data\\jobs\\synth3d-vector3d
"""
from __future__ import annotations

import argparse
import base64
import functools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synth3d"
SIZE = (1280, 720)
W, H = SIZE
FPS = 25
GEN_VERSION = 1                 # bump whenever the rendering or the scripts change (the video is made again)
TRUTH_VERSION = 3              # bump when only the truth JSON changes: it is rewritten, the video is kept
LECTURES = ("vector3d", "cylinder", "saddle")
# The piece to open "View in 3D" on in each processed job (data/jobs/synth3d-<name>, made 2026-11-02 by
# app.process), the other pieces that open the same scene, and a point on the piece's ink clear of other ink (as a
# fraction of the board) to hover. Piece ids change if a job is processed again: tests/test_gen_scene_lectures.py
# checks them against the job folder.
# The piece to hover for "View in 3D", the drawing (scene) it opens and the other pieces that open the same scene,
# as processed on 2026-11-03 (arrowheads captured, so a sketch can be several pieces; processing again renumbers).
OPEN_ON = {"vector3d": {"piece": 8, "drawing": 8, "same_scene": [6, 12, 13], "hover": [0.3638, 0.5644]},
           "cylinder": {"piece": 5, "drawing": 21, "same_scene": [7, 13, 21], "hover": [0.2288, 0.5671]},
           "saddle": {"piece": 22, "drawing": 22, "same_scene": [8, 11, 16], "hover": [0.3113, 0.6877]}}
VOICE = "Microsoft David Desktop"
VOICE_RATE = 0                  # SAPI rate, -10..10
AUDIO_RATE = 16000              # SAPI's word positions are right only at 16 kHz (at 22.05 kHz they run 1.38x fast)
MIN_DURATION, MAX_DURATION = 60.0, 100.0

BOARD = (70, 48, 1210, 560)     # white writing surface, video px: x in [70, 1210), y in [48, 560)
FRAME_PX = 14
NOISE_SIGMA = 2.5
WOBBLE_AMP, WOBBLE_PERIOD = 0.01, 3.7
LEAD_IN = 1.2                   # the first beat starts here
TAIL = 2.5                      # silence after the last beat
PEN_GAP = 0.35                  # pen lifted between two inks, at least
WALK_SPEED = 380.0              # px/s the plan allows for the teacher to walk between inks
PEN_SPEED = 240.0               # px/s along a stroke
WRITE_SPEED = 150.0             # px/s of text width
BODY_OFFSET = 150.0             # body centre - pen x while drawing (the teacher stands right of the pen)
STAND_X = 1236.0                # where the teacher stands to explain: right of the board, clear of the writing
COLORS = {"black": (32, 32, 38), "blue": (165, 72, 28), "red": (40, 38, 185), "green": (60, 130, 30),
          "purple": (140, 50, 120)}


# --------------------------------------------------------------------------------------
# Scripts: inks (what is drawn) and beats (what is said, and which ink is drawn at which word)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Ink:
    """One thing the teacher draws (strokes along paths) or writes (a line of handwriting, revealed left to right)."""
    id: str
    what: str                         # truth: what it is
    color: str = "black"
    paths: tuple = ()                 # strokes: point lists in video px, in pen order
    dashed: bool = False
    width: int = 5
    text: str = ""                    # writing: markup, see _parse_markup
    at: tuple = (0, 0)                # writing: (left, baseline)
    scale: float = 1.4
    figure: bool = True               # part of the 3D figure (False: a formula or note beside it)

    @property
    def kind(self) -> str:
        return "text" if self.text else "stroke"


@dataclass(frozen=True)
class Draw:
    ink: str
    word: str | None = None           # starts when this word of the beat is said (next occurrence); None: after the last
    offset: float = 0.0
    dur: float | None = None


@dataclass(frozen=True)
class Beat:
    stage: str
    say: str
    draws: tuple = ()
    cues: tuple = ()                  # words that mark the stage (each must be in `say`)
    true: str = ""                    # what is true of the figure when the stage ends
    stand: float | None = None        # where the teacher stands while not drawing (default STAND_X)
    pause: float = 1.2
    check: dict = field(default_factory=dict, hash=False)  # the numbers true of the figure when the stage ends


@dataclass(frozen=True)
class Lecture:
    name: str
    title: str
    inks: tuple
    beats: tuple
    facts: dict = field(default_factory=dict)

    @functools.cached_property
    def ink(self) -> dict:
        return {k.id: k for k in self.inks}


def _proj(o, ex, ey, ez):
    o, ex, ey, ez = (np.asarray(v, float) for v in (o, ex, ey, ez))
    return lambda x, y, z: o + x * ex + y * ey + z * ez


def _line(a, b, n: int = 2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def _head(tip, frm, size: float = 15.0, spread: float = 26.0):
    """Arrowhead at `tip` for a line coming from `frm`: one polyline left - tip - right."""
    tip, frm = np.asarray(tip, float), np.asarray(frm, float)
    d = frm - tip
    d /= np.linalg.norm(d) or 1.0
    out = []
    for s in (spread, -spread):
        a = math.radians(s)
        r = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
        out.append(tip + size * r)
    return np.array([out[0], tip, out[1]])


def _arrow(a, b, size: float = 15.0):
    return (_line(a, b), _head(b, a, size))


def _double_arrow(a, b, size: float = 12.0):
    return (_line(a, b), _head(b, a, size), _head(a, b, size))


def _ellipse(c, rx, ry, a0, a1, n: int = 64):
    """Points of an ellipse arc, angles in degrees, image coordinates (y down: 90 is the bottom)."""
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([c[0] + rx * np.cos(a), c[1] + ry * np.sin(a)], 1)


def _curve(P, f, u0, u1, n: int = 60):
    return np.array([P(*f(u)) for u in np.linspace(u0, u1, n)])


def _lecture_vector3d() -> Lecture:
    P = _proj((430, 300), (-55, 25), (40, 25), (0, -50))           # axonometric: x down-left, y down-right, z up
    O, X3, XY, V = P(0, 0, 0), P(3, 0, 0), P(3, 2, 0), P(3, 2, 4)
    ax_end = {"x": P(4.6, 0, 0), "y": P(0, 5.4, 0), "z": P(0, 0, 4.4)}
    # the x component as a dimension line just beside the x axis (inside the path's triangle), with end ticks
    nrm = np.array([25.0, 55.0]) / math.hypot(25, 55)
    off = 11 * nrm
    tick = 7 * nrm
    vdir = math.atan2(V[1] - O[1], V[0] - O[0])
    xdir = math.atan2(ax_end["x"][1] - O[1], ax_end["x"][0] - O[0])
    a0, a1 = math.degrees(xdir), math.degrees(vdir) % 360.0          # 156 -> 221: the arc between them
    r = 40.0
    bis = math.radians((a0 + a1) / 2)
    th = O + 62 * np.array([math.cos(bis), math.sin(bis)])
    mx = O + 0.75 * (X3 - O) - 28 * nrm
    my = (X3 + XY) / 2 + 26 * np.array([-0.53, 0.848])
    inks = (
        Ink("axis_x", "x axis (towards the viewer, drawn down-left) with arrowhead", paths=_arrow(O, ax_end["x"])),
        Ink("label_x", "label x", text="x", at=(int(ax_end["x"][0]) - 32, int(ax_end["x"][1]) + 30)),
        Ink("axis_y", "y axis (to the right) with arrowhead", paths=_arrow(O, ax_end["y"])),
        Ink("label_y", "label y", text="y", at=(int(ax_end["y"][0]) + 12, int(ax_end["y"][1]) + 14)),
        Ink("axis_z", "z axis (up) with arrowhead", paths=_arrow(O, ax_end["z"])),
        Ink("label_z", "label z", text="z", at=(int(ax_end["z"][0]) + 14, int(ax_end["z"][1]) + 18)),
        Ink("formula_v", "v = (3, 2, 4)", color="blue", text="v = (3, 2, 4)", at=(730, 130), scale=1.4,
            figure=False),
        Ink("vector_v", "vector v from the origin to (3, 2, 4), arrowhead at the tip", color="red", width=6,
            paths=_arrow(O, V, 17)),
        Ink("label_v", "label v at the tip", color="red", text="v", at=(int(V[0]) - 32, int(V[1]) + 8)),
        Ink("comp_x", "x component: 3 along the x axis (dimension line under it)", color="green", width=6,
            paths=(_line(O + off, X3 + off), _line(O + off - tick, O + off + tick),
                   _line(X3 + off - tick, X3 + off + tick))),
        Ink("label_3", "label 3", color="green", text="3", at=(int(mx[0]) - 10, int(mx[1]) + 12)),
        Ink("comp_y", "y component: 2 parallel to y, from (3,0,0) to (3,2,0), dashed", color="green", dashed=True,
            paths=(_line(X3, XY),)),
        Ink("label_2", "label 2", color="green", text="2", at=(int(my[0]) - 10, int(my[1]) + 12)),
        Ink("comp_z", "z component: up 4 from (3,2,0) to the tip of v, dashed", color="green", dashed=True,
            paths=(_line(XY, V),)),
        Ink("label_4", "label 4", color="green", text="4", at=(int(XY[0]) + 12, int(XY[1]) - 34)),
        Ink("arc_theta", "arc at the origin between the x axis and v", color="purple", width=4,
            paths=(_ellipse(O, r, r, a0, a1, 40),)),
        Ink("label_theta", "label theta on the arc", color="purple", text="{theta}",
            at=(int(th[0]) - 8, int(th[1]) + 12)),
        Ink("formula_len", "|v| = sqrt(29) = 5.39", color="blue", text="|v| = {sqrt:29} {approx} 5.39",
            at=(730, 235), scale=1.4, figure=False),
        Ink("formula_cos", "cos theta = 3/sqrt(29)", color="blue", text="cos {theta} = 3/{sqrt:29}", at=(730, 335),
            scale=1.4, figure=False),
        Ink("formula_theta", "theta = 56 degrees", color="blue", text="{theta} {approx} 56{deg}", at=(730, 435),
            scale=1.4, figure=False),
    )
    v = [3, 2, 4]
    length = math.sqrt(sum(c * c for c in v))
    theta = math.degrees(math.acos(v[0] / length))
    axes = {"origin": [0, 0, 0], "x": "towards the viewer", "y": "right", "z": "up"}
    vec = {"from": [0, 0, 0], "to": v, "length": round(length, 4)}
    angle = {"vertex": [0, 0, 0], "towards": [[1, 0, 0], v], "deg": round(theta, 2),
             "label_shows": f"θ {round(theta, 1):g}°"}         # as the 3D view labels an angle
    beats = (
        Beat("intro", "Today we look at a vector in three dimensions, and how to find its length and its angle.",
             cues=("vector", "three", "dimensions"), true="the board is empty", check={"drawn": []}),
        Beat("axes", "First, three axes. The z axis goes up, the x axis comes towards us, "
                     "and the y axis goes to the right.",
             draws=(Draw("axis_z", "z"), Draw("label_z"), Draw("axis_x", "x"), Draw("label_x"),
                    Draw("axis_y", "y"), Draw("label_y")),
             cues=("axes", "up", "towards", "right"),
             true="axes x (towards the viewer), y (right) and z (up) drawn from the origin, each labelled",
             check={"drawn": ["axes"], "axes": axes}),
        Beat("vector", "Our vector is v equals three, two, four. We draw it as an arrow from the origin to the "
                       "point three, two, four.",
             draws=(Draw("formula_v", "v"), Draw("vector_v", "arrow"), Draw("label_v")),
             cues=("vector", "arrow", "origin"),
             true="v = (3, 2, 4) written on the right; the arrow v from the origin to (3, 2, 4)",
             check={"drawn": ["axes", "v"], "v": vec}),
        Beat("x_component", "To reach its tip, first go three along x.",
             draws=(Draw("comp_x", "three"), Draw("label_3")), cues=("first", "three", "along"),
             true="the x component, 3 along the x axis, is marked", pause=0.9,
             check={"drawn": ["axes", "v", "x component"],
                    "x_component": {"from": [0, 0, 0], "to": [3, 0, 0], "length": 3, "label": "3"}}),
        Beat("y_component", "Then two along y.",
             draws=(Draw("comp_y", "two"), Draw("label_2")), cues=("then", "two", "along"),
             true="from (3, 0, 0), 2 parallel to the y axis to (3, 2, 0), dashed", pause=0.9,
             check={"drawn": ["axes", "v", "x component", "y component"],
                    "y_component": {"from": [3, 0, 0], "to": [3, 2, 0], "length": 2, "label": "2",
                                    "parallel_to": "y"}}),
        Beat("z_component", "Then up four in z, and we are at the tip of v.",
             draws=(Draw("comp_z", "up"), Draw("label_4")), cues=("up", "four", "tip"),
             true="from (3, 2, 0) up 4 to the tip of v, dashed: the three components make the path to the tip",
             check={"drawn": ["axes", "v", "x component", "y component", "z component"],
                    "z_component": {"from": [3, 2, 0], "to": v, "length": 4, "label": "4", "parallel_to": "z"},
                    "path_ends_at_tip_of_v": True}),
        Beat("angle", "Now the angle theta, between v and the x axis.",
             draws=(Draw("arc_theta", "angle"), Draw("label_theta")), cues=("angle", "theta"),
             true="an arc at the origin marks theta, the angle between v and the x axis",
             check={"drawn": ["axes", "v", "x component", "y component", "z component", "theta"], "theta": angle}),
        Beat("length", "The length of v is the square root of three squared plus two squared plus four squared. "
                       "That is the square root of twenty nine, about five point four.",
             draws=(Draw("formula_len", "That"),), cues=("length", "square", "root"),
             true="|v| = sqrt(29), about 5.39, written on the right",
             check={"length": round(length, 4), "length_squared": 29, "sum": "3^2 + 2^2 + 4^2"}),
        Beat("cos_theta", "So cos theta is three over root twenty nine, and theta is about fifty six degrees.",
             draws=(Draw("formula_cos", "cos"), Draw("formula_theta", "theta")), cues=("cos", "degrees"),
             true="cos theta = 3/sqrt(29) and theta = 56 degrees written on the right",
             check={"cos_theta": round(v[0] / length, 4), "theta_deg": round(theta, 2)}),
        Beat("summary", "So v has length root twenty nine, and it makes an angle of about fifty six degrees "
                        "with the x axis.",
             cues=("length", "angle", "degrees"), true="the whole figure and the three results are on the board",
             check={"length": round(length, 4), "theta_deg": round(theta, 2)}),
    )
    facts = {"vector": [3, 2, 4], "components": {"x": 3, "y": 2, "z": 4}, "length": round(math.sqrt(29), 4),
             "length_exact": "sqrt(29)", "cos_theta": round(3 / math.sqrt(29), 4),
             "theta_deg": round(math.degrees(math.acos(3 / math.sqrt(29))), 2),
             "theta_between": "v and the x axis",
             "projection": "axonometric: x drawn down-left, y down-right, z up; origin at video px "
                           f"{[round(float(c), 1) for c in O]}",
             "path_to_tip": [[0, 0, 0], [3, 0, 0], [3, 2, 0], [3, 2, 4]]}
    return Lecture("vector3d", "A vector in three dimensions", inks, beats, facts)


def _lecture_cylinder() -> Lecture:
    cx, top, bot, rx, ry = 360.0, 310.0, 470.0, 100.0, 28.0
    ext = 150.0                      # the top at twice the height
    big = 2 * rx
    inks = (
        Ink("top", "top circle of the cylinder (an ellipse)", paths=(_ellipse((cx, top), rx, ry, 180, 540, 90),)),
        Ink("sides", "the two vertical sides", paths=(_line((cx - rx, top), (cx - rx, bot)),
                                                      _line((cx + rx, top), (cx + rx, bot)))),
        Ink("bottom_front", "front half of the bottom circle", paths=(_ellipse((cx, bot), rx, ry, 180, 0, 50),)),
        Ink("bottom_back", "back half of the bottom circle, dashed", dashed=True,
            paths=(_ellipse((cx, bot), rx, ry, 180, 360, 50),)),
        Ink("radius_r", "radius r from the centre of the top", color="blue",
            paths=(_ellipse((cx, top), 3, 3, 0, 360, 12), _line((cx, top), (cx + rx, top)))),
        Ink("label_r", "label r", color="blue", text="r", at=(int(cx + 38), int(top - 9))),
        Ink("height_h", "height h: a double arrow beside the cylinder", color="blue",
            paths=_double_arrow((cx + rx + 36, top), (cx + rx + 36, bot))),
        Ink("label_h", "label h", color="blue", text="h", at=(int(cx + rx + 50), int((top + bot) / 2) + 10)),
        Ink("formula_V", "V = pi r^2 h", text="V = {pi}r{sup:2}h", at=(720, 150), scale=1.6, figure=False),
        Ink("ext_sides", "the sides continued up to twice the height, dashed", color="red", dashed=True,
            paths=(_line((cx - rx, top), (cx - rx, ext)), _line((cx + rx, top), (cx + rx, ext)))),
        Ink("ext_top", "the top at twice the height, dashed", color="red", dashed=True,
            paths=(_ellipse((cx, ext), rx, ry, 180, 540, 90),)),
        Ink("height_2h", "height 2h: a double arrow", color="red", paths=_double_arrow((cx + rx + 86, ext),
                                                                                    (cx + rx + 86, bot))),
        Ink("label_2h", "label 2h", color="red", text="2h", at=(int(cx + rx + 100), int((ext + bot) / 2) + 10)),
        Ink("formula_2h", "2h -> V x 2", color="red", text="2h {to} V {times} 2", at=(720, 270), scale=1.5,
            figure=False),
        Ink("big_base", "a base of twice the radius, dashed", color="green", dashed=True,
            paths=(_ellipse((cx, bot), big, 2 * ry, 180, 540, 110),)),
        Ink("radius_2r", "radius 2r", color="green", paths=(_line((cx, bot), (cx + big, bot)),)),
        Ink("label_2r", "label 2r", color="green", text="2r", at=(int(cx + big + 10), int(bot + 10))),
        Ink("formula_2r", "2r -> V x 4", color="green", text="2r {to} V {times} 4", at=(720, 380), scale=1.5,
            figure=False),
        Ink("formula_sq", "(2r)^2 = 4 r^2", color="green", text="(2r){sup:2} = 4r{sup:2}", at=(720, 480), scale=1.5,
            figure=False),
    )
    hr = round((bot - top) / rx, 2)  # the drawn height over the drawn radius
    cyl = ["top circle", "sides", "bottom"]

    def size(r: int, h: int) -> dict:  # factors over the first cylinder's r, h and V = pi r^2 h
        return {"r_factor": r, "h_factor": h, "V_factor": r * r * h}

    beats = (
        Beat("intro", "Today, the volume of a cylinder.", cues=("volume", "cylinder"), true="the board is empty",
             check={"drawn": []}),
        Beat("top", "Here is a cylinder. The top is a circle, which we see as an ellipse.",
             draws=(Draw("top", "top"),), cues=("top", "circle", "ellipse"), true="the top circle, seen as an ellipse",
             check={"drawn": ["top circle"]}),
        Beat("sides", "The sides go straight down, and the bottom is the same circle. "
                      "Its back is hidden, so we draw it dashed.",
             draws=(Draw("sides", "sides"), Draw("bottom_front", "bottom"), Draw("bottom_back", "dashed")),
             cues=("sides", "bottom", "dashed"), true="the whole cylinder: top, two sides, bottom (back half dashed)",
             check={"drawn": cyl, **size(1, 1), "drawn_h_over_r": hr}),
        Beat("radius", "The radius of the circle is r.", draws=(Draw("radius_r", "radius"), Draw("label_r")),
             cues=("radius",), true="the radius r is marked on the top circle", pause=0.9,
             check={"drawn": cyl + ["r"], **size(1, 1)}),
        Beat("height", "And the height of the cylinder is h.", draws=(Draw("height_h", "height"), Draw("label_h")),
             cues=("height",), true="the height h is marked beside the cylinder",
             check={"drawn": cyl + ["r", "h"], **size(1, 1)}),
        Beat("formula", "The volume is the area of the base, pi r squared, times the height. "
                        "So V equals pi r squared h.",
             draws=(Draw("formula_V", "So"),), cues=("volume", "area", "pi"),
             true="V = pi r^2 h written on the right", check={"V": "pi r^2 h", **size(1, 1)}),
        Beat("double_height", "Now, if we double the height, the volume doubles. Twice as tall holds twice as much.",
             draws=(Draw("ext_sides", "double"), Draw("ext_top"), Draw("height_2h"), Draw("label_2h"),
                    Draw("formula_2h")),
             cues=("double", "height", "doubles"),
             true="the cylinder is extended (dashed) to height 2h; 2h -> V x 2 written: the volume doubles",
             check={"drawn": cyl + ["r", "h", "cylinder of height 2h"], **size(1, 2)}),
        Beat("double_radius", "If we double the radius instead, the volume goes up four times.",
             draws=(Draw("big_base", "radius"), Draw("radius_2r"), Draw("label_2r"), Draw("formula_2r")),
             cues=("radius", "instead", "four"),
             true="a base of radius 2r is drawn (dashed); 2r -> V x 4 written: the volume is four times as big",
             check={"drawn": cyl + ["r", "h", "cylinder of height 2h", "base of radius 2r"],
                    **size(2, 1)}),    # "instead": the height is h again, so not V x 8
        Beat("why", "That is because the radius is squared. Two r, squared, is four r squared.",
             draws=(Draw("formula_sq", "Two"),), cues=("because", "squared"), true="(2r)^2 = 4r^2 written",
             check={**size(2, 1), "r_squared_factor": 4}),
        Beat("summary", "So the height counts once, but the radius counts twice.",
             cues=("once", "twice"), true="everything is on the board",
             check={"V_factor_when_h_doubles": 2, "V_factor_when_r_doubles": 4}),
    )
    facts = {"volume": "V = pi r^2 h", "double_height": {"h": "h -> 2h", "volume_factor": 2},
             "double_radius": {"r": "r -> 2r", "volume_factor": 4},
             "factors": "a stage's r_factor, h_factor and V_factor: its r, h and V over the first cylinder's "
                        "(the lecture gives no numbers, only ratios)",
             "drawing": {"centre_x_px": cx, "top_y_px": top, "bottom_y_px": bot, "radius_px": rx,
                         "height_px": bot - top, "twice_height_top_y_px": ext, "twice_radius_px": big}}
    return Lecture("cylinder", "The volume of a cylinder", inks, beats, facts)


def _lecture_saddle() -> Lecture:
    P = _proj((400, 300), (105, 0), (36, -28), (0, -70))          # oblique: x right, y back (up-right), z up
    a = 1.2
    O = P(0, 0, 0)
    ax_end = {"x": P(2.25, 0, 0), "y": P(0, 2.3, 0), "z": P(0, 0, 2.3)}
    para_x = _curve(P, lambda u: (u, 0, u * u), -a, a)
    para_y = _curve(P, lambda u: (0, u, -u * u), -a, a)
    outline = (_curve(P, lambda u: (-a, u, a * a - u * u), -a, a), _curve(P, lambda u: (u, a, u * u - a * a), -a, a),
               _curve(P, lambda u: (a, -u, a * a - u * u), -a, a), _curve(P, lambda u: (-u, -a, u * u - a * a), -a, a))
    inks = (
        Ink("formula_z", "z = x^2 - y^2", text="z = x{sup:2} - y{sup:2}", at=(730, 130), scale=1.6, figure=False),
        Ink("axis_x", "x axis (to the right)", paths=_arrow(O, ax_end["x"])),
        Ink("label_x", "label x", text="x", at=(int(ax_end["x"][0]) + 8, int(ax_end["x"][1]) + 30)),
        Ink("axis_y", "y axis (back into the board, drawn up-right)", paths=_arrow(O, ax_end["y"])),
        Ink("label_y", "label y", text="y", at=(int(ax_end["y"][0]) - 30, int(ax_end["y"][1]) - 8)),
        Ink("axis_z", "z axis (up)", paths=_arrow(O, ax_end["z"])),
        Ink("label_z", "label z", text="z", at=(int(ax_end["z"][0]) - 36, int(ax_end["z"][1]) + 20)),
        Ink("para_x", "the parabola along x (y = 0): z = x^2, curving up", color="red", width=6, paths=(para_x,)),
        Ink("note_x", "y = 0: z = x^2", color="red", text="y = 0:  z = x{sup:2}", at=(730, 240), scale=1.4,
            figure=False),
        Ink("para_y", "the parabola along y (x = 0): z = -y^2, curving down", color="blue", width=6, paths=(para_y,)),
        Ink("note_y", "x = 0: z = -y^2", color="blue", text="x = 0:  z = -y{sup:2}", at=(730, 335), scale=1.4,
            figure=False),
        Ink("outline", "the saddle's edges over the square -1.2..1.2 in x and y", width=4, paths=outline),
        Ink("saddle_dot", "the saddle point at the origin, circled", color="green", width=4,
            paths=(_ellipse(O, 4, 4, 0, 360, 14), _ellipse(O, 13, 13, -90, 270, 40))),
        Ink("label_saddle", "saddle point", color="green", text="saddle point", at=(730, 450), scale=1.4,
            figure=False),
        Ink("arrow_saddle", "arrow from the words to the saddle point", color="green", width=4,
            paths=_arrow((722, 438), O + np.array([17.0, 9.0]), 13)),
    )
    def z(x: float, y: float) -> float:
        return round(x * x - y * y, 4)

    def pts(*xy) -> list:
        return [[x, y, z(x, y)] for x, y in xy]

    axes = {"origin": [0, 0, 0], "x": "right", "y": "back into the board", "z": "up"}
    up_x = {"plane": "y = 0", "z": "x^2", "d2z_dx2": 2, "opens": "up", "points": pts((-a, 0), (0, 0), (a, 0))}
    down_y = {"plane": "x = 0", "z": "-y^2", "d2z_dy2": -2, "opens": "down", "points": pts((0, -a), (0, 0), (0, a))}
    hess = {"point": [0, 0, 0], "gradient": [0, 0], "hessian": [[2, 0], [0, -2]], "hessian_det": -4}
    beats = (
        Beat("intro", "Today, a surface with a saddle point.", cues=("surface", "saddle"), true="the board is empty",
             check={"drawn": []}),
        Beat("formula", "Our surface is z equals x squared minus y squared.", draws=(Draw("formula_z", "z"),),
             cues=("surface", "squared", "minus"), true="z = x^2 - y^2 written on the right",
             check={"drawn": [], "z": "x^2 - y^2"}),
        Beat("axes", "Let's draw the axes. x goes to the right, y goes back into the board, and z goes up.",
             draws=(Draw("axis_x", "x"), Draw("label_x"), Draw("axis_y", "y"), Draw("label_y"),
                    Draw("axis_z", "z"), Draw("label_z")),
             cues=("axes", "right", "back", "up"),
             true="axes x (right), y (back into the board, drawn up-right), z (up), labelled",
             check={"drawn": ["axes"], "axes": axes}),
        Beat("along_x", "Along x, where y is zero, z is x squared, so the surface curves up.",
             draws=(Draw("para_x", "Along"), Draw("note_x", "curves")), cues=("along", "curves", "up"),
             true="the red parabola z = x^2 in the plane y = 0, opening upwards",
             check={"drawn": ["axes", "curve along x"], "curve_along_x": up_x}),
        Beat("along_y", "Along y, where x is zero, z is minus y squared, so it curves down.",
             draws=(Draw("para_y", "Along"), Draw("note_y", "curves")), cues=("along", "curves", "down"),
             true="the blue parabola z = -y^2 in the plane x = 0, opening downwards",
             check={"drawn": ["axes", "curve along x", "curve along y"], "curve_along_y": down_y}),
        Beat("saddle_shape", "Put together, the surface bends up one way and down the other, like a horse saddle.",
             draws=(Draw("outline", "together"),), cues=("together", "saddle"),
             true="the edges of the surface over the square |x|, |y| <= 1.2 complete the saddle",
             check={"drawn": ["axes", "curve along x", "curve along y", "surface"],
                    "surface": {"z": "x^2 - y^2", "x_range": [-a, a], "y_range": [-a, a],
                                "z_range": [z(0, a), z(a, 0)], "corners": pts((a, a), (a, -a), (-a, a), (-a, -a))}}),
        Beat("saddle_point", "The middle, at the origin, is a saddle point.",
             draws=(Draw("saddle_dot", "middle"), Draw("label_saddle"), Draw("arrow_saddle")),
             cues=("middle", "origin", "point"), true="the origin is circled and labelled saddle point",
             check={"drawn": ["axes", "curve along x", "curve along y", "surface", "saddle point"],
                    "saddle_point": hess}),
        Beat("why", "It is a minimum along x, but a maximum along y.", cues=("minimum", "maximum"),
             true="nothing new is drawn: the red curve has its lowest point and the blue its highest at the origin",
             check={"minimum_along": "x", "d2z_dx2": 2, "maximum_along": "y", "d2z_dy2": -2}),
        Beat("summary", "So the origin is neither a maximum nor a minimum. It is a saddle point.",
             cues=("neither", "saddle"), true="everything is on the board",
             check={"saddle_point": hess}),
    )
    facts = {"surface": "z = x^2 - y^2", "domain_drawn": {"x": [-a, a], "y": [-a, a]},
             "along_x": {"plane": "y = 0", "curve": "z = x^2", "opens": "up"},
             "along_y": {"plane": "x = 0", "curve": "z = -y^2", "opens": "down"},
             "saddle_point": [0, 0, 0], "minimum_along": "x", "maximum_along": "y",
             "projection": "oblique: x right, y back into the board (drawn up-right), z up; origin at video px "
                           f"{[round(float(c), 1) for c in O]}"}
    return Lecture("saddle", "A saddle surface", inks, beats, facts)


@functools.lru_cache(maxsize=None)
def lecture(name: str) -> Lecture:
    makers = {"vector3d": _lecture_vector3d, "cylinder": _lecture_cylinder, "saddle": _lecture_saddle}
    if name not in makers:
        raise ValueError(f"unknown lecture {name!r}; expected one of {LECTURES}")
    return makers[name]()


# --------------------------------------------------------------------------------------
# Words and timing
# --------------------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'.]*[A-Za-z0-9]|[A-Za-z0-9]")


def words_of(text: str) -> list[tuple[str, int]]:
    """Whole words of `text` with their character positions (SAPI reports words by character position too)."""
    return [(m.group(0), m.start()) for m in _WORD.finditer(text)]


def has_word(text: str, word: str) -> bool:
    return any(w.lower() == word.lower() for w, _ in words_of(text))


def estimate_speech(text: str) -> dict:
    """A stand-in for SAPI's timing when nothing is spoken (fast tests): ~0.055 s a character plus pauses (fitted to
    the David voice at rate 0: within ~10% of its duration for these scripts)."""
    t, last, out = 0.12, 0, []
    for w, c in words_of(text):
        gap = text[last:c]
        t += 0.055 * len(gap) + (0.4 if "." in gap else 0.18 if "," in gap or ":" in gap else 0.0)
        out.append({"w": w, "c": c, "s": round(t, 3)})
        t += 0.055 * len(w) + 0.03
        last = c + len(w)
    for a, b in zip(out, out[1:]):
        a["e"] = round(b["s"] - 0.03, 3)
    if out:
        out[-1]["e"] = round(t, 3)
    return {"duration": round(t + 0.1, 3), "words": out}


def merge_sapi_words(text: str, events: list[dict], speech_end: float) -> list[dict]:
    """SAPI SpeakProgress events (c = character position, t = seconds) -> one entry per word of `text`.

    A word SAPI expands ("29" -> "twenty nine") sends several events with the same position: the first one counts.
    A word with no event gets the time of the one before it. Ends are the next word's start (the last: speech end).
    """
    first: dict[int, float] = {}
    for e in sorted(events, key=lambda e: (float(e["t"]), int(e["c"]))):
        first.setdefault(int(e["c"]), float(e["t"]))
    out, prev = [], 0.0
    for w, c in words_of(text):
        s = first.get(c)
        if s is None:  # SAPI may report a word joined with its neighbour: take the closest earlier position
            near = [t for cc, t in first.items() if cc <= c]
            s = max(near) if near else prev
        s = max(s, prev)
        out.append({"w": w, "c": c, "s": round(s, 3)})
        prev = s
    for a, b in zip(out, out[1:]):
        a["e"] = round(max(a["s"], b["s"] - 0.02), 3)
    if out:
        out[-1]["e"] = round(max(out[-1]["s"] + 0.1, speech_end), 3)
    return out


def _ink_start_x(ink: Ink) -> float:
    if ink.kind == "text":
        return float(ink.at[0])
    return float(np.asarray(ink.paths[0])[0][0])


def _ink_end_x(ink: Ink) -> float:
    if ink.kind == "text":
        return float(ink.at[0]) + _text_width_estimate(ink)
    return float(np.asarray(ink.paths[-1])[-1][0])


def _text_width_estimate(ink: Ink) -> float:
    plain = re.sub(r"\{(\w+)(?::([^}]*))?\}", lambda m: (m.group(2) or "") + "  ", ink.text)
    return 13.5 * ink.scale * len(plain)


def _path_len(p) -> float:
    p = np.asarray(p, float)
    return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0


def ink_duration(ink: Ink) -> float:
    if ink.kind == "text":
        return round(max(0.6, 0.3 + _text_width_estimate(ink) / WRITE_SPEED), 2)
    length = sum(_path_len(p) for p in ink.paths)
    return round(max(0.5, length / PEN_SPEED + 0.2 * (len(ink.paths) - 1)), 2)


def plan(lec: Lecture, speech: dict | None = None) -> dict:
    """The timeline: when each beat is said and each ink drawn. `speech` maps segment id -> {duration, words}
    (SAPI's, when built); without it the times are estimated (the same rules, so fast tests check the real logic).
    """
    t = LEAD_IN
    pen_free, pen_x = 0.0, None
    stand_prev = STAND_X
    segments, stages, inks = [], [], {}
    for i, beat in enumerate(lec.beats):
        sid = f"{lec.name}_{i:02d}"
        sp = (speech or {}).get(sid) or estimate_speech(beat.say)
        words = sp["words"]
        start = t
        seg_words = [{"w": w["w"], "c": w["c"], "s": round(start + w["s"], 3), "e": round(start + w["e"], 3)}
                     for w in words]
        cursor_c = -1
        drawn = []
        for d in beat.draws:
            ink = lec.ink[d.ink]
            if d.word is not None:
                cand = [w for w in seg_words if w["w"].lower() == d.word.lower() and w["c"] > cursor_c]
                if not cand:
                    raise ValueError(f"{lec.name}/{beat.stage}: {d.ink} waits for {d.word!r}, not in what is said")
                cursor_c = cand[0]["c"]
                anchor = cand[0]["s"] + d.offset
            else:
                anchor = start
            x0 = _ink_start_x(ink) + BODY_OFFSET
            if pen_x is None or anchor - pen_free > 2.5:   # the teacher was standing aside
                walk_from = stand_prev
                ready = max(start, pen_free) + abs(walk_from - x0) / WALK_SPEED + 0.4
            else:
                ready = pen_free + PEN_GAP + abs(pen_x - x0) / WALK_SPEED
            t0 = round(max(anchor, ready, pen_free + PEN_GAP), 3)
            t1 = round(t0 + (d.dur or ink_duration(ink)), 3)
            inks[d.ink] = {"t_start": t0, "t_end": t1, "stage": beat.stage}
            pen_free, pen_x = t1, _ink_end_x(ink) + BODY_OFFSET
            drawn.append(d.ink)
        end_speech = round(start + sp["duration"], 3)
        end = round(max(end_speech, pen_free) + beat.pause, 3)
        segments.append({"id": sid, "stage": beat.stage, "text": beat.say, "start": round(start, 3),
                         "end": end_speech, "words": [{k: w[k] for k in ("w", "s", "e")} for w in seg_words]})
        cue_times = []
        for cue in beat.cues:
            hit = next((w for w in seg_words if w["w"].lower() == cue.lower()), None)
            cue_times.append({"w": cue, "t": hit["s"] if hit else None})
        stages.append({"stage": beat.stage, "start": round(start, 3), "end": end, "say": beat.say,
                       "cues": cue_times, "draws": drawn, "true": beat.true,
                       "check": json.loads(json.dumps(beat.check)),
                       "stand": beat.stand if beat.stand is not None else STAND_X})
        stand_prev = beat.stand if beat.stand is not None else STAND_X
        t = end
    duration = round(t + TAIL, 3)
    return {"lecture": lec.name, "title": lec.title, "duration": duration, "segments": segments,
            "stages": stages, "inks": inks}


def check_truth(truth: dict) -> list[str]:
    """Problems with a plan or a truth JSON (empty when consistent): stage order, times inside the video, cue words
    in the narration and said inside their stage, every ink drawn once and inside its stage, one ink at a time."""
    bad = []
    dur = float(truth.get("duration", 0))
    if not (MIN_DURATION <= dur <= MAX_DURATION):
        bad.append(f"duration {dur} not in {MIN_DURATION}..{MAX_DURATION} s")
    stages = truth.get("stages") or []
    if not stages:
        bad.append("no stages")
    prev_end = 0.0
    names = [s["stage"] for s in stages]
    if len(set(names)) != len(names):
        bad.append("stage names repeat")
    full_text = " ".join(s["text"] for s in truth.get("segments") or [])
    for s in stages:
        if not (0.0 <= s["start"] < s["end"] <= dur):
            bad.append(f"stage {s['stage']}: {s['start']}..{s['end']} not inside 0..{dur}")
        if s["start"] < prev_end - 1e-6:
            bad.append(f"stage {s['stage']} starts before the one before it ends")
        prev_end = s["end"]
        if not s.get("cues"):
            bad.append(f"stage {s['stage']} has no cue words")
        if not isinstance(s.get("check"), dict) or not s["check"]:
            bad.append(f"stage {s['stage']} has no check (the numbers true of the figure when it ends)")
        for c in s.get("cues") or []:
            if not has_word(s["say"], c["w"]):
                bad.append(f"stage {s['stage']}: cue {c['w']!r} is not in what is said")
            if not has_word(full_text, c["w"]):
                bad.append(f"stage {s['stage']}: cue {c['w']!r} is not in the narration")
            if c.get("t") is None or not (s["start"] - 1e-6 <= c["t"] <= s["end"] + 1e-6):
                bad.append(f"stage {s['stage']}: cue {c['w']!r} said at {c.get('t')}, outside the stage")
    segs = truth.get("segments") or []
    for a, b in zip(segs, segs[1:]):
        if a["end"] > b["start"] + 1e-6:
            bad.append(f"narration {a['id']} runs into {b['id']}")
    for sg in segs:
        st = next((s for s in stages if s["stage"] == sg["stage"]), None)
        if st is None or not (st["start"] - 1e-6 <= sg["start"] and sg["end"] <= st["end"] + 1e-6):
            bad.append(f"narration {sg['id']} is outside its stage")
        prev = sg["start"] - 1e-6
        for w in sg["words"]:
            if not (prev <= w["s"] <= w["e"] + 1e-6 and w["e"] <= sg["end"] + 0.05):
                bad.append(f"narration {sg['id']}: word {w['w']!r} at {w['s']}..{w['e']} out of order")
                break
            prev = w["s"]
        if [w["w"] for w in sg["words"]] != [w for w, _ in words_of(sg["text"])]:
            bad.append(f"narration {sg['id']}: words do not match its text")
    inks = truth.get("inks") or {}
    drawn = [i for s in stages for i in s.get("draws") or []]
    if sorted(drawn) != sorted(inks):
        bad.append("the stages' draws and the inks do not match")
    if len(set(drawn)) != len(drawn):
        bad.append("an ink is drawn twice")
    for iid, k in inks.items():
        st = next((s for s in stages if s["stage"] == k["stage"]), None)
        if not (0.0 <= k["t_start"] < k["t_end"] <= dur):
            bad.append(f"ink {iid}: {k['t_start']}..{k['t_end']} not inside the video")
        if st is None or not (st["start"] - 1e-6 <= k["t_start"] and k["t_end"] <= st["end"] + 1e-6):
            bad.append(f"ink {iid} is drawn outside its stage")
    order = sorted(inks.items(), key=lambda kv: kv[1]["t_start"])
    for (ia, a), (ib, b) in zip(order, order[1:]):
        if a["t_end"] > b["t_start"] + 1e-6:
            bad.append(f"inks {ia} and {ib} are drawn at the same time")
    if inks and min(k["t_start"] for k in inks.values()) < 4.0:
        bad.append("drawing starts before 4 s (the pipeline counts the first seconds' writing as already there)")
    return bad


# --------------------------------------------------------------------------------------
# Handwriting: Hershey script plus a few hand-drawn glyphs
# --------------------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SLANT = 0.2                     # handwriting leans: 0.2 px right per px above the baseline
_MARKUP = re.compile(r"\{(\w+)(?::([^}]*))?\}|([^{]+)")
GLYPHS = ("theta", "pi", "sqrt", "approx", "to", "times", "deg", "sup")


def parse_markup(text: str) -> list[tuple[str, str]]:
    """'cos {theta} = 3/{sqrt:29}' -> [('t', 'cos '), ('theta', ''), ('t', ' = 3/'), ('sqrt', '29')]."""
    out = []
    for m in _MARKUP.finditer(text):
        if m.group(3) is not None:
            out.append(("t", m.group(3)))
        else:
            if m.group(1) not in GLYPHS:
                raise ValueError(f"unknown glyph {m.group(1)!r} in {text!r}")
            out.append((m.group(1), m.group(2) or ""))
    return out


def _thick(scale: float) -> int:
    return 4 if scale >= 1.45 else 3


def _render_text(text: str, scale: float, left: int, base: int, shape=(H, W)) -> np.ndarray:
    cov = np.zeros(shape, np.uint8)
    th = _thick(scale)
    h = 22.0 * scale
    x = float(left)

    def pl(pts, closed=False):
        p = np.round(np.asarray(pts, float) * 16).astype(np.int32)
        cv2.polylines(cov, [p], closed, 255, th, cv2.LINE_AA, 4)

    def put(s, sc, xx, yy):
        cv2.putText(cov, s, (int(round(xx)), int(round(yy))), _FONT, sc, 255, _thick(sc), cv2.LINE_AA)
        return cv2.getTextSize(s, _FONT, sc, _thick(sc))[0][0]

    for kind, val in parse_markup(text):
        if kind == "t":
            x += put(val, scale, x, base) + 2
        elif kind == "sup":
            x += put(val, scale * 0.62, x - 1, base - 0.6 * h) + 3
        elif kind == "theta":
            c = (x + 0.3 * h, base - 0.5 * h)
            pl(_ellipse(c, 0.27 * h, 0.5 * h, -100, 260, 40))
            pl([(x + 0.06 * h, base - 0.5 * h), (x + 0.54 * h, base - 0.5 * h)])
            x += 0.72 * h
        elif kind == "pi":
            pl([(x, base - 0.58 * h), (x + 0.88 * h, base - 0.64 * h)])
            pl([(x + 0.27 * h, base - 0.61 * h), (x + 0.22 * h, base - 0.3 * h), (x + 0.1 * h, base)])
            pl([(x + 0.62 * h, base - 0.63 * h), (x + 0.61 * h, base - 0.12 * h), (x + 0.72 * h, base)])
            x += 0.98 * h
        elif kind == "sqrt":
            wr = cv2.getTextSize(val, _FONT, scale, th)[0][0]
            top = base - 1.08 * h
            pl([(x, base - 0.36 * h), (x + 0.1 * h, base - 0.43 * h), (x + 0.28 * h, base + 0.06 * h),
                (x + 0.5 * h, top), (x + 0.62 * h + wr + 0.12 * h, top)])
            put(val, scale, x + 0.6 * h, base)
            x += 0.62 * h + wr + 0.2 * h
        elif kind == "approx":
            for yy in (base - 0.58 * h, base - 0.3 * h):
                u = np.linspace(0, 1, 16)
                pl(np.stack([x + 0.75 * h * u, yy - 0.09 * h * np.sin(2 * np.pi * u)], 1))
            x += 0.95 * h
        elif kind == "to":
            yy = base - 0.45 * h
            pl([(x, yy), (x + 0.95 * h, yy)])
            pl([(x + 0.75 * h, yy - 0.16 * h), (x + 0.95 * h, yy), (x + 0.75 * h, yy + 0.16 * h)])
            x += 1.15 * h
        elif kind == "times":
            pl([(x + 0.1 * h, base - 0.62 * h), (x + 0.55 * h, base - 0.18 * h)])
            pl([(x + 0.55 * h, base - 0.62 * h), (x + 0.1 * h, base - 0.18 * h)])
            x += 0.75 * h
        elif kind == "deg":
            pl(_ellipse((x + 0.17 * h, base - 0.95 * h), 0.12 * h, 0.12 * h, 0, 360, 16), closed=True)
            x += 0.4 * h
    m = np.float32([[1, -_SLANT, _SLANT * base], [0, 1, 0]])
    return cv2.warpAffine(cov, m, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------------------
# Inks: coverage and per-pixel reveal times
# --------------------------------------------------------------------------------------
def _dashes(p: np.ndarray, dash: float = 11.0, gap: float = 8.0) -> list[np.ndarray]:
    p = np.asarray(p, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out, a = [], 0.0
    while a < s[-1] - 1:
        b = min(a + dash, s[-1])
        u = np.linspace(a, b, max(2, int((b - a) / 2) + 2))
        out.append(np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1))
        a = b + gap
    return out


def _pen_track(ink: Ink) -> tuple[np.ndarray, np.ndarray]:
    """Pen points (N, 2) and their fraction of the ink's time, for a stroke ink: the paths in order, at even speed,
    with the pen moving twice as fast in the air between them."""
    pts, cost = [], []
    for i, p in enumerate(ink.paths):
        p = np.asarray(p, float)
        seg = np.hypot(*np.diff(p, axis=0).T)
        n = np.maximum(1, np.ceil(seg / 2.0)).astype(int)
        dense = [p[0]] + [p[k] + (p[k + 1] - p[k]) * (j / n[k]) for k in range(len(seg)) for j in range(1, n[k] + 1)]
        dense = np.array(dense)
        if pts:
            jump = float(np.hypot(*(dense[0] - pts[-1][-1])))
            cost.append(np.array([cost[-1][-1] + 0.5 * jump + 6.0]))
            pts.append(dense[:1])
            c = cost[-1][-1] + np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(dense, axis=0).T))])
            cost.append(c[1:])
            pts.append(dense[1:])
        else:
            cost.append(np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(dense, axis=0).T))]))
            pts.append(dense)
    P, C = np.concatenate(pts), np.concatenate(cost)
    return P, C / max(C[-1], 1e-9)


@dataclass
class InkPixels:
    ys: np.ndarray
    xs: np.ndarray
    a: np.ndarray            # coverage 0..1
    rt: np.ndarray           # reveal time per pixel
    color: np.ndarray        # BGR float32
    box: list                # [x, y, w, h] of the full ink (alpha >= 0.5)


def _ink_pixels(ink: Ink, t0: float, t1: float) -> InkPixels:
    if ink.kind == "text":
        cov = _render_text(ink.text, ink.scale, ink.at[0], ink.at[1])
        ys, xs = np.nonzero(cov)
        x0, x1 = xs.min(), xs.max() + 1
        rt = t0 + (xs - x0) / max(1, x1 - x0) * (t1 - t0)
    else:
        cov = np.zeros((H, W), np.uint8)
        for p in ink.paths:
            for d in (_dashes(p) if ink.dashed else [np.asarray(p, float)]):
                cv2.polylines(cov, [np.round(d * 16).astype(np.int32)], False, 255, ink.width, cv2.LINE_AA, 4)
        P, frac = _pen_track(ink)
        tmap = np.full((H, W), np.inf, np.float32)
        tt = t0 + frac * (t1 - t0)
        for k in range(len(P) - 1, 0, -1):  # earliest last, so each pixel keeps its first time
            a, b = P[k - 1], P[k]
            if np.hypot(*(b - a)) > 3.0:     # a pen-up jump between paths: nothing drawn
                continue
            cv2.line(tmap, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))),
                     float(tt[k - 1]), ink.width + 5, cv2.LINE_8)
        ys, xs = np.nonzero(cov)
        rt = tmap[ys, xs].astype(np.float64)
        rt[~np.isfinite(rt)] = t1
    m = cov >= 128
    bx, by, bw, bh = cv2.boundingRect(m.astype(np.uint8))
    return InkPixels(ys, xs, cov[ys, xs].astype(np.float32) / 255.0, rt,
                     np.array(COLORS[ink.color], np.float32), [int(bx), int(by), int(bw), int(bh)])


def pen_point(ink: Ink, t0: float, t1: float, t: float, box=None) -> tuple[float, float]:
    u = min(max((t - t0) / max(t1 - t0, 1e-9), 0.0), 1.0)
    if ink.kind == "text":
        x, y, w, h = box
        return x + w * u, y + h * (0.55 + 0.3 * math.sin(2 * math.pi * 2.4 * (t - t0)))
    P, frac = _pen_track_cached(ink)
    return float(np.interp(u, frac, P[:, 0])), float(np.interp(u, frac, P[:, 1]))


@functools.lru_cache(maxsize=256)
def _pen_track_cached(ink: Ink):
    return _pen_track(ink)


# --------------------------------------------------------------------------------------
# The teacher: where he stands, where his writing hand is, how he looks
# --------------------------------------------------------------------------------------
_RAISE = 0.45
_SHOULDER_Y = 318.0
_SHOULDER_DX = 66.0
_L1, _L2 = 124.0, 118.0          # upper arm, forearm (to the hand centre)
_MARKER = 22.0


class Motion:
    """Body x, facing and the writing hand over time, made once from the plan (so every frame agrees)."""

    def __init__(self, lec: Lecture, pl: dict, boxes: dict):
        self.lec, self.pl, self.boxes = lec, pl, boxes
        self.items = sorted(((v["t_start"], v["t_end"], lec.ink[k]) for k, v in pl["inks"].items()),
                            key=lambda r: r[0])
        self.dur = pl["duration"]
        self.dt = 0.02
        n = int(math.ceil(self.dur / self.dt)) + 2
        self.ts = np.arange(n) * self.dt
        xs = np.empty(n)
        x = STAND_X + 60.0      # he is already there, at the side, when the video starts
        for i, t in enumerate(self.ts):
            target = self._target(t)
            step = (target - x) * (1.0 - math.exp(-self.dt / 0.16))
            vmax = 460.0 * self.dt
            x += max(-vmax, min(vmax, step))
            xs[i] = x
        self.xs = xs
        self.vs = np.gradient(xs, self.dt)
        self.dist = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(xs)))])

    def _stand(self, t: float) -> float:
        for s in self.pl["stages"]:
            if s["start"] <= t < s["end"]:
                return float(s["stand"])
        return STAND_X

    def _target(self, t: float) -> float:
        for t0, t1, ink in self.items:
            if t0 <= t <= t1:
                return self.pen(ink, t0, t1, t)[0] + BODY_OFFSET
            if t < t0:
                lead = 0.9 + abs(self._stand(t) - (self.pen(ink, t0, t1, t0)[0] + BODY_OFFSET)) / WALK_SPEED
                prev_end = max([b for a, b, _ in self.items if b <= t] or [-1e9])
                if t0 - t <= lead or t - prev_end < 1.2:
                    nxt = self.pen(ink, t0, t1, t0)[0] + BODY_OFFSET
                    if t - prev_end < 1.2 and t0 - t > lead:   # just finished one: stay a moment
                        prev = [(a, b, k) for a, b, k in self.items if b <= t][-1]
                        return self.pen(prev[2], prev[0], prev[1], prev[1])[0] + BODY_OFFSET
                    return nxt
                break
        return self._stand(t)

    def pen(self, ink: Ink, t0: float, t1: float, t: float):
        return pen_point(ink, t0, t1, t, self.boxes.get(ink.id))

    def body_x(self, t: float) -> float:
        return float(np.interp(t, self.ts, self.xs))

    def speed(self, t: float) -> float:
        return float(np.interp(t, self.ts, self.vs))

    def walked(self, t: float) -> float:
        return float(np.interp(t, self.ts, self.dist))

    def hand(self, t: float):
        """(pen tip or None for an arm at rest, pen colour, writing now)."""
        rest = None
        for i, (t0, t1, ink) in enumerate(self.items):
            col = COLORS[ink.color]
            if t0 <= t <= t1:
                return self.pen(ink, t0, t1, t), col, True
            prev = self.items[i - 1] if i else None
            if t0 - _RAISE <= t < t0:
                p = self.pen(ink, t0, t1, t0)
                if prev is not None and t0 - prev[1] < 1.0:   # straight from the last ink to this one
                    q = self.pen(prev[2], prev[0], prev[1], prev[1])
                    s = _smooth((t - prev[1]) / max(t0 - prev[1], 1e-6))
                    return (q[0] + (p[0] - q[0]) * s, q[1] + (p[1] - q[1]) * s), col, False
                return ("raise", p, _smooth((t - (t0 - _RAISE)) / _RAISE)), col, False
            if t1 < t < t1 + _RAISE:
                nxt = self.items[i + 1] if i + 1 < len(self.items) else None
                if nxt is not None and nxt[0] - t1 < 1.0:
                    continue
                p = self.pen(ink, t0, t1, t1)
                return ("lower", p, _smooth((t - t1) / _RAISE)), col, False
            if t1 + _RAISE <= t:
                rest = col
        return None, rest or COLORS["black"], False

    def facing_board(self, t: float) -> bool:
        return any(t0 - 0.9 <= t <= t1 + 0.7 for t0, t1, _ in self.items)


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _elbow(s, hnd, l1, l2, bend: float):
    """Elbow of a two-link arm from shoulder s to hand h; bend picks the side (+1: elbow below the line)."""
    s, hnd = np.asarray(s, float), np.asarray(hnd, float)
    d = hnd - s
    dist = float(np.hypot(*d))
    dist_c = min(max(dist, abs(l1 - l2) + 1), l1 + l2 - 0.5)
    if dist > l1 + l2 - 0.5:
        hnd = s + d / dist * dist_c
        d = hnd - s
    a = (l1 * l1 - l2 * l2 + dist_c * dist_c) / (2 * dist_c)
    hh = math.sqrt(max(l1 * l1 - a * a, 0.0))
    u = d / (dist_c or 1.0)
    base = s + a * u
    perp = np.array([-u[1], u[0]])
    e1, e2 = base + hh * perp, base - hh * perp
    e = e1 if (e1[1] - e2[1]) * bend >= 0 else e2
    return e, hnd, dist


_SHIRT = (150, 104, 72)          # BGR: a mid blue shirt
_SHIRT_DARK = (124, 84, 58)
_TROUSERS = (58, 52, 50)
_BELT = (30, 40, 52)
_SKIN = (128, 160, 205)
_SKIN_DARK = (100, 128, 172)
_HAIR = (30, 38, 52)
_EYE = (40, 36, 34)
_LIP = (90, 100, 160)


def _teacher_parts(m: Motion, t: float):
    """[(colour, shaded, draw(canvas, ox, oy))] back to front, the region they cover, and the body centre."""
    cx = m.body_x(t)
    v = m.speed(t)
    walking = min(1.0, abs(v) / 140.0)
    phase = m.walked(t) / 38.0
    swing = 11.0 * math.sin(phase) * walking
    bob = 3.0 * abs(math.sin(phase)) * walking
    tip, pen_col, _writing = m.hand(t)
    board_side = m.facing_board(t)
    parts = []

    def P(pts, ox, oy):
        return np.round((np.asarray(pts, float) - (ox, oy)) * 16).astype(np.int32)

    def pt(x, y, ox, oy):
        return int(round((x - ox) * 16)), int(round((y - oy) * 16))

    def poly(pts):
        return lambda c, ox, oy: cv2.fillPoly(c, [P(pts, ox, oy)], 255, cv2.LINE_AA, 4)

    def ell(x, y, ax, ay, ang=0.0, a0=0, a1=360):
        return lambda c, ox, oy: cv2.ellipse(c, pt(x, y, ox, oy), (int(round(ax * 16)), int(round(ay * 16))),
                                             ang, a0, a1, 255, -1, cv2.LINE_AA, 4)

    def seg(x0, y0, x1, y1, th):
        return lambda c, ox, oy: cv2.line(c, pt(x0, y0, ox, oy), pt(x1, y1, ox, oy), 255, th, cv2.LINE_AA, 4)

    yb = H + 4
    y0 = bob
    # legs, then the shirt over them, the belt
    parts.append((_TROUSERS, True, poly([(cx - 70, 536 + y0), (cx - 3, 536 + y0), (cx - 8 + swing, yb),
                                         (cx - 64 + swing, yb)])))
    parts.append((_TROUSERS, True, poly([(cx + 3, 536 + y0), (cx + 70, 536 + y0), (cx + 64 - swing, yb),
                                         (cx + 8 - swing, yb)])))
    parts.append((_SHIRT, True, poly([(cx - 40, 294 + y0), (cx + 40, 294 + y0), (cx + 74, 302 + y0),
                                      (cx + 88, 322 + y0), (cx + 84, 430 + y0), (cx + 74, 540 + y0),
                                      (cx - 74, 540 + y0), (cx - 84, 430 + y0), (cx - 88, 322 + y0),
                                      (cx - 74, 302 + y0)])))
    parts.append((_BELT, True, poly([(cx - 74, 528 + y0), (cx + 74, 528 + y0), (cx + 74, 541 + y0),
                                     (cx - 74, 541 + y0)])))
    # arms: the writing one on the pen's side (the left of the picture: he stands right of what he writes)
    sl = (cx - _SHOULDER_DX, _SHOULDER_Y + y0)
    sr = (cx + _SHOULDER_DX, _SHOULDER_Y + y0)

    def hanging(s, side):
        e = (s[0] + side * 16, s[1] + 112)
        hnd = (s[0] + side * 20 + 0.08 * swing * side, s[1] + 215)
        return e, hnd

    def arm(s, e, hnd, front: bool):
        out = [(_SHIRT_DARK if not front else _SHIRT, True, seg(s[0], s[1], e[0], e[1], 30)),
               (_SHIRT, False, seg(e[0], e[1], e[0] + 0.62 * (hnd[0] - e[0]), e[1] + 0.62 * (hnd[1] - e[1]), 26)),
               (_SKIN, False, seg(e[0] + 0.55 * (hnd[0] - e[0]), e[1] + 0.55 * (hnd[1] - e[1]), hnd[0], hnd[1], 17)),
               (_SKIN, False, ell(hnd[0], hnd[1], 11.5, 13.0))]
        return out

    er, hr = hanging(sr, +1)
    parts += arm(sr, er, hr, False)
    if tip is None:
        el, hl = hanging(sl, -1)
        parts_arm = arm(sl, el, hl, True)
        marker = None
    else:
        if isinstance(tip, tuple) and len(tip) == 3 and isinstance(tip[0], str):
            how, p, s = tip
            el_r, hl_r = hanging(sl, -1)
            rest_tip = (hl_r[0] - 4, hl_r[1] + 8)
            if how == "raise":
                tip = (rest_tip[0] + (p[0] - rest_tip[0]) * s, rest_tip[1] + (p[1] - rest_tip[1]) * s)
            else:
                tip = (p[0] + (rest_tip[0] - p[0]) * s, p[1] + (rest_tip[1] - p[1]) * s)
        tx, ty = tip
        d = np.array([tx - sl[0], ty - sl[1]])
        u = d / (np.hypot(*d) or 1.0)
        want = np.array([tx, ty]) - (_MARKER + 9) * u
        e, hnd, _ = _elbow(sl, want, _L1, _L2, +1)
        parts_arm = arm(sl, e, hnd, True)
        marker = (hnd, np.array([tx, ty]) if np.hypot(*(np.array([tx, ty]) - hnd)) < _MARKER + 16 else
                  hnd + (_MARKER + 9) * u)
    # neck and head
    hy = 236 + y0
    parts.append((_SKIN_DARK, False, poly([(cx - 15, hy + 26), (cx + 15, hy + 26), (cx + 16, 300 + y0),
                                           (cx - 16, 300 + y0)])))
    parts.append((_SKIN, False, ell(cx - 34, hy + 6, 7, 12)))
    parts.append((_SKIN, False, ell(cx + 34, hy + 6, 7, 12)))
    parts.append((_SKIN, False, ell(cx, hy, 33, 42)))
    if board_side:   # the back of his head
        parts.append((_HAIR, False, ell(cx, hy - 6, 35, 38)))
        parts.append((_HAIR, False, ell(cx, hy + 18, 30, 22, 0, 0, 180)))
        parts.append((_SHIRT_DARK, False, poly([(cx - 34, 292 + y0), (cx + 34, 292 + y0), (cx + 30, 300 + y0),
                                                (cx - 30, 300 + y0)])))
    else:            # his face
        parts.append((_HAIR, False, ell(cx, hy - 16, 36, 30, 0, 180, 360)))
        parts.append((_HAIR, False, ell(cx - 30, hy - 8, 8, 18)))
        parts.append((_HAIR, False, ell(cx + 30, hy - 8, 8, 18)))
        parts.append((_EYE, False, ell(cx - 12, hy - 2, 4.0, 3.0)))
        parts.append((_EYE, False, ell(cx + 12, hy - 2, 4.0, 3.0)))
        parts.append((_HAIR, False, seg(cx - 19, hy - 11, cx - 6, hy - 12, 3)))
        parts.append((_HAIR, False, seg(cx + 6, hy - 12, cx + 19, hy - 11, 3)))
        parts.append((_SKIN_DARK, False, seg(cx + 1, hy - 1, cx + 3, hy + 13, 4)))
        parts.append((_LIP, False, seg(cx - 9, hy + 24, cx + 9, hy + 24, 3)))
        parts.append((_SKIN_DARK, False, poly([(cx - 15, 292 + y0), (cx + 15, 292 + y0), (cx, 314 + y0)])))
        parts.append(((235, 235, 235), False, seg(cx - 22, 294 + y0, cx - 4, 316 + y0, 4)))
        parts.append(((235, 235, 235), False, seg(cx + 22, 294 + y0, cx + 4, 316 + y0, 4)))
    parts += parts_arm
    if marker is not None:
        hnd, tipp = marker
        parts.append(((225, 225, 225), False, seg(hnd[0], hnd[1], tipp[0], tipp[1], 8)))
        parts.append((pen_col, False, seg(hnd[0] + 0.7 * (tipp[0] - hnd[0]), hnd[1] + 0.7 * (tipp[1] - hnd[1]),
                                          tipp[0], tipp[1], 6)))
    xs = [cx - 110, cx + 110]
    ys = [190.0]
    if marker is not None:
        xs += [marker[1][0] - 40, marker[1][0] + 40, marker[0][0] - 40, marker[0][0] + 40]
        ys += [marker[1][1] - 40, marker[0][1] - 40]
    x0 = max(int(math.floor(min(xs))) - 4, 0)
    x1 = min(int(math.ceil(max(xs))) + 4, W)
    y0r = max(int(math.floor(min(ys))) - 4, 0)
    return parts, (x0, y0r, x1, H), cx


def _draw_teacher(frame: np.ndarray, m: Motion, t: float) -> None:
    parts, (x0, y0, x1, y1), cx = _teacher_parts(m, t)
    if x1 <= x0:
        return
    R = frame[y0:y1, x0:x1]
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    light = (1.04 - 0.10 * (np.arange(y0, y1, dtype=np.float32) - 190.0) / 530.0)[:, None, None]
    for colour, shaded, draw in parts:
        canvas.fill(0)
        draw(canvas, x0, y0)
        bx, by, bw, bh = cv2.boundingRect(canvas)
        if bw == 0 or bh == 0:
            continue
        a = canvas[by:by + bh, bx:bx + bw].astype(np.float32) * (1.0 / 255.0)
        sub = R[by:by + bh, bx:bx + bw]
        c = np.asarray(colour, np.float32)[None, None, :] * light[by:by + bh]
        if shaded:
            xs = np.arange(x0 + bx, x0 + bx + bw, dtype=np.float32) - cx
            shade = 0.78 + 0.22 * np.cos(np.clip(xs / 95.0, -1, 1) * (np.pi / 2))
            c = shade[None, :, None] * c
        sub += a[..., None] * (c - sub)


# --------------------------------------------------------------------------------------
# The room and the camera
# --------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _background() -> np.ndarray:
    rng = np.random.default_rng(20261103)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.empty((H, W, 3), np.float32)
    wall = np.array([168, 184, 196], np.float32)
    grad = 1.04 - 0.08 * (yy / H) - 0.05 * (xx / W)
    low = cv2.resize(rng.normal(0, 1, (12, 20)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img[:] = wall * grad[..., None] + (2.5 * low + rng.normal(0, 1.2, (H, W)).astype(np.float32))[..., None]
    img[652:664] = (214, 216, 220)
    img[663:664] = (168, 170, 174)
    floor = np.array([78, 102, 132], np.float32)
    fy = np.arange(664, H, dtype=np.float32)
    img[664:] = floor * (0.92 + 0.1 * (fy - 664) / (H - 664))[:, None, None]
    img[664:] += rng.normal(0, 1.5, (H - 664, W)).astype(np.float32)[..., None]
    for y in (682, 704):
        img[y] *= 0.88
    fx0, fy0, fx1, fy1 = BOARD[0] - FRAME_PX, BOARD[1] - FRAME_PX, BOARD[2] + FRAME_PX, BOARD[3] + FRAME_PX
    sh = np.zeros((H, W), np.float32)
    sh[fy0 + 8:fy1 + 24, fx0 + 6:fx1 + 6] = 1.0
    img *= (1.0 - 0.2 * cv2.GaussianBlur(sh, (0, 0), 7))[..., None]
    img[fy0:fy1, fx0:fx1] = (176, 179, 178)
    img[fy0:fy0 + 3, fx0:fx1] = (216, 218, 217)
    img[fy0:fy1, fx0:fx0 + 2] = (206, 208, 207)
    img[fy1 - 3:fy1, fx0:fx1] = (126, 128, 128)
    img[fy0:fy1, fx1 - 2:fx1] = (140, 142, 141)
    bx0, by0, bx1, by1 = BOARD
    img[by0 - 1:by1 + 1, bx0 - 1:bx1 + 1] = (136, 139, 138)
    # pen tray with markers and an eraser
    img[fy1:fy1 + 12, fx0 + 30:fx1 - 30] = (164, 167, 166)
    img[fy1 + 10:fy1 + 12, fx0 + 30:fx1 - 30] = (112, 115, 114)
    img[fy1 + 12:fy1 + 18, fx0 + 34:fx1 - 26] *= 0.84
    for i, name in enumerate(("black", "blue", "red", "green")):
        x = 900 + 58 * i
        img[fy1 + 2:fy1 + 9, x:x + 42] = (210, 210, 210)
        img[fy1 + 2:fy1 + 9, x + 30:x + 42] = COLORS[name]
    img[fy1 - 6:fy1 + 9, 1150:1196] = (60, 60, 66)
    img[fy1 - 6:fy1 - 1, 1152:1194] = (120, 120, 126)
    # the white surface: a soft vignette, a broad glare from the lights, faint ghosts of old writing
    sy, sx = yy[by0:by1, bx0:bx1], xx[by0:by1, bx0:bx1]
    r2 = ((sx - (bx0 + bx1) / 2) / ((bx1 - bx0) / 2)) ** 2 + ((sy - (by0 + by1) / 2) / ((by1 - by0) / 2)) ** 2
    glare = 5.0 * np.exp(-(((sx - 820) / 260.0) ** 2 + ((sy - 150) / 120.0) ** 2))
    surf = np.array([238, 240, 238], np.float32) + (4.0 - 5.0 * r2 + glare)[..., None]
    ghost = np.zeros((by1 - by0, bx1 - bx0), np.float32)
    for gx, gy, gw in ((240, 120, 260), (700, 420, 300), (980, 300, 140)):
        cv2.putText(ghost, "~" * max(1, gw // 40), (gx - bx0, gy - by0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, 1.0,
                    5, cv2.LINE_AA)
    ghost = cv2.GaussianBlur(ghost, (0, 0), 6) * 7.0
    img[by0:by1, bx0:bx1] = surf - ghost[..., None]
    return img


def _noise_lut() -> np.ndarray:
    from statistics import NormalDist
    z = np.array([NormalDist().inv_cdf((k + 0.5) / 256) for k in range(256)], np.float64)
    return (z * (NOISE_SIGMA / z.std())).astype(np.float32).reshape(1, 256)


_NOISE_LUT = _noise_lut()


class Scene:
    """Everything one lecture's frames are made from."""

    def __init__(self, lec: Lecture, pl: dict):
        self.lec, self.pl = lec, pl
        self.pix = {k: _ink_pixels(lec.ink[k], v["t_start"], v["t_end"]) for k, v in pl["inks"].items()}
        self.order = sorted(self.pix, key=lambda k: pl["inks"][k]["t_start"])
        self.motion = Motion(lec, pl, {k: p.box for k, p in self.pix.items()})

    def frame(self, t: float, seed: int, i: int) -> np.ndarray:
        f = _background().copy()
        for k in self.order:
            p = self.pix[k]
            if p.rt.min() > t:
                break
            vis = p.rt <= t
            ys, xs = p.ys[vis], p.xs[vis]
            under = f[ys, xs]
            f[ys, xs] = under + p.a[vis][:, None] * (p.color - under)
        _draw_teacher(f, self.motion, t)
        f = cv2.GaussianBlur(f, (0, 0), 0.6)
        wob = 1.0 + WOBBLE_AMP * math.sin(2 * math.pi * t / WOBBLE_PERIOD)
        raw = np.random.default_rng([seed, i]).bytes(H * W * 3)
        noise = cv2.LUT(np.frombuffer(raw, np.uint8).reshape(H, W, 3), _NOISE_LUT)
        return cv2.convertScaleAbs(cv2.scaleAdd(f, wob, noise))

    def reach(self) -> float:
        """The longest shoulder-to-hand distance while drawing (the arm can reach L1 + L2)."""
        worst = 0.0
        for k, v in self.pl["inks"].items():
            for t in np.linspace(v["t_start"], v["t_end"], 25):
                cx = self.motion.body_x(t)
                tx, ty = self.motion.pen(self.lec.ink[k], v["t_start"], v["t_end"], t)
                worst = max(worst, math.hypot(tx - (cx - _SHOULDER_DX), ty - _SHOULDER_Y) - _MARKER - 9)
        return worst


# --------------------------------------------------------------------------------------
# Speech (Windows SAPI through Windows PowerShell 5.1)
# --------------------------------------------------------------------------------------
_TTS_PS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$job = Get-Content -Raw -Encoding UTF8 $env:RB_TTS_IN | ConvertFrom-Json
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo([int]$job.rate_hz,
  [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$out = @()
foreach ($s in $job.segments) {
  $syn = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $syn.SelectVoice([string]$job.voice)
  $syn.Rate = [int]$job.rate
  $events = New-Object System.Collections.ArrayList
  $h = Register-ObjectEvent -InputObject $syn -EventName SpeakProgress -MessageData $events -Action {
    [void]$Event.MessageData.Add(@{t = $EventArgs.AudioPosition.TotalSeconds; c = $EventArgs.CharacterPosition;
                                   w = $EventArgs.Text}) }
  $wav = Join-Path $env:RB_TTS_OUT ($s.id + '.wav')
  $syn.SetOutputToWaveFile($wav, $fmt)
  $syn.Speak([string]$s.text)
  $syn.SetOutputToNull()
  Start-Sleep -Milliseconds 60
  Unregister-Event -SourceIdentifier $h.Name
  $syn.Dispose()
  $out += @{id = $s.id; text = [string]$s.text; events = @($events)}
}
ConvertTo-Json -Depth 6 @{voice = [string]$job.voice; rate = [int]$job.rate; segments = $out} |
  Set-Content -Encoding UTF8 (Join-Path $env:RB_TTS_OUT 'tts.json')
"""


def _speech_end(pcm: np.ndarray, rate: int) -> float:
    hop = int(0.02 * rate)
    n = len(pcm) // hop
    if n == 0:
        return 0.0
    e = np.sqrt(np.mean(pcm[:n * hop].astype(np.float64).reshape(n, hop) ** 2, axis=1))
    loud = np.nonzero(e > 150)[0]
    return float((loud[-1] + 1) * hop / rate) if len(loud) else 0.0


def speak(lec: Lecture, tts_dir: Path, force: bool = False, cached_only: bool = False) -> dict:
    """Speak every beat with SAPI (cached in tts_dir). Returns {segment id: {duration, words, pcm}}.
    cached_only: never speak; a cache that does not match the script is an error."""
    tts_dir.mkdir(parents=True, exist_ok=True)
    want = [{"id": f"{lec.name}_{i:02d}", "text": b.say} for i, b in enumerate(lec.beats)]
    meta = tts_dir / "tts.json"
    cached = None
    if meta.is_file() and not force:
        try:
            cached = json.loads(meta.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            cached = None
    if not (cached and cached.get("voice") == VOICE and cached.get("rate") == VOICE_RATE
            and [{"id": s["id"], "text": s["text"]} for s in cached["segments"]] == want
            and all((tts_dir / f"{s['id']}.wav").is_file() for s in want)):
        if cached_only:
            raise RuntimeError(f"{lec.name}: the narration in {tts_dir} does not match the script; "
                               "make the lecture again with --force")
        jin = tts_dir / "tts_in.json"
        jin.write_text(json.dumps({"voice": VOICE, "rate": VOICE_RATE, "rate_hz": AUDIO_RATE, "segments": want}),
                       encoding="utf-8")
        ps = shutil.which("powershell.exe") or shutil.which("powershell")
        if not ps:
            raise RuntimeError("Windows PowerShell was not found (the narration uses Windows SAPI)")
        enc = base64.b64encode(_TTS_PS.encode("utf-16-le")).decode("ascii")
        env = dict(os.environ, RB_TTS_IN=str(jin), RB_TTS_OUT=str(tts_dir))
        res = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-EncodedCommand", enc], env=env,
                             capture_output=True, timeout=600)
        if res.returncode != 0 or not meta.is_file():
            err = (res.stderr or res.stdout).decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(f"text to speech failed: {err[-1] if err else res.returncode}")
        cached = json.loads(meta.read_text(encoding="utf-8-sig"))
    out = {}
    for s in cached["segments"]:
        with wave.open(str(tts_dir / f"{s['id']}.wav"), "rb") as wf:
            if wf.getframerate() != AUDIO_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise RuntimeError(f"{s['id']}.wav is not 16 kHz mono 16-bit")
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
        end = _speech_end(pcm, AUDIO_RATE)
        events = s.get("events") or []
        if isinstance(events, dict):
            events = [events]
        words = merge_sapi_words(s["text"], events, end)
        out[s["id"]] = {"duration": round(end, 3), "words": words, "pcm": pcm[:int(math.ceil(end * AUDIO_RATE))]}
    return out


def _mix_audio(pl: dict, speech: dict, seed: int) -> np.ndarray:
    n = int(round(pl["duration"] * AUDIO_RATE))
    rng = np.random.default_rng(seed + 7)
    room = rng.normal(0, 18.0, n)                                    # a quiet room
    room = np.convolve(room, np.ones(8) / 8, mode="same") * 2.0
    mix = room.astype(np.float64)
    for sg in pl["segments"]:
        pcm = speech[sg["id"]]["pcm"].astype(np.float64)
        a = int(round(sg["start"] * AUDIO_RATE))
        b = min(n, a + len(pcm))
        mix[a:b] += 0.9 * pcm[:b - a]
    return np.clip(np.round(mix), -32768, 32767).astype(np.int16)


# --------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------
def _paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"{name}.mp4", out_dir / f"{name}.json"


def truth_dict(lec: Lecture, pl: dict, scene: Scene | None = None, seed: int = 0) -> dict:
    bx0, by0, bx1, by1 = BOARD
    bw, bh = bx1 - bx0, by1 - by0

    def rel(b):
        return [round((b[0] - bx0) / bw, 4), round((b[1] - by0) / bh, 4), round(b[2] / bw, 4), round(b[3] / bh, 4)]

    inks = {}
    for k, v in pl["inks"].items():
        ink = lec.ink[k]
        d = {"what": ink.what, "figure": ink.figure, "color": ink.color, "kind": ink.kind, **v}
        if ink.kind == "text":
            d["text"] = ink.text
        if scene is not None:
            b = scene.pix[k].box
            d["box"], d["board_box"] = b, rel(b)
        inks[k] = d
    out = {
        "GEN_VERSION": GEN_VERSION, "TRUTH_VERSION": TRUTH_VERSION, "lecture": lec.name, "title": lec.title,
        "seed": seed,
        "video": f"{lec.name}.mp4", "size": list(SIZE), "fps": FPS, "duration": pl["duration"],
        "voice": VOICE, "voice_rate": VOICE_RATE,
        "board_corners": [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]],
        "facts": lec.facts,
        "narration_text": " ".join(s["text"] for s in pl["segments"]),
        "segments": pl["segments"], "stages": pl["stages"], "inks": inks,
        "teacher": {"stands_while_drawing": f"right of the pen, body centre = pen x + {BODY_OFFSET:.0f} px",
                    "stands_while_explaining_x": STAND_X},
        "pipeline": {"job": f"data/jobs/synth3d-{lec.name}",
                     "command": f".venv\\Scripts\\python -m app.process data\\synth3d\\{lec.name}.mp4 --no-gestures "
                                f"--job data\\jobs\\synth3d-{lec.name}"},
        "stage_check": "each stage's check: the numbers true of the figure when the stage ends (its 'true' in "
                       "words); a 3D view of the drawing should show them by then",
    }
    if lec.name in OPEN_ON:
        o = OPEN_ON[lec.name]
        out["pipeline"].update(open_piece=o["piece"], open_drawing=o["drawing"], same_scene_from=o["same_scene"],
                               hover_board_fraction=o["hover"],
                               open_note="piece ids of the job as processed on 2026-11-03; processing it again "
                                         "renumbers them")
    if scene is not None:
        fig = [scene.pix[k].box for k in pl["inks"] if lec.ink[k].figure]
        x0 = min(b[0] for b in fig)
        y0 = min(b[1] for b in fig)
        x1 = max(b[0] + b[2] for b in fig)
        y1 = max(b[1] + b[3] for b in fig)
        out["figure"] = {"box": [x0, y0, x1 - x0, y1 - y0], "board_box": rel([x0, y0, x1 - x0, y1 - y0]),
                         "inks": [k for k in pl["inks"] if lec.ink[k].figure]}
    return out


class _Encoder:
    def __init__(self, path: Path, wav: Path):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found on PATH")
        self.path, self.tmp = path, path.with_name(path.stem + ".partial.mp4")
        self.err = tempfile.TemporaryFile()
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", str(wav), "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(self.tmp)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).data)

    def close(self) -> None:
        self.proc.stdin.close()
        rc = self.proc.wait()
        self.err.seek(0)
        msg = self.err.read().decode("utf-8", "replace").strip().splitlines()
        self.err.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path.name}: {msg[-1] if msg else rc}")
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait()
        finally:
            self.err.close()
            self.tmp.unlink(missing_ok=True)


_VIDEO_KEYS = ("duration", "narration_text", "segments", "inks", "figure", "board_corners", "size", "fps")
_STAGE_KEYS = ("stage", "start", "end", "say", "cues", "draws", "stand")


def truth_differs(old: dict, new: dict) -> list[str]:
    """What of the video a new truth would describe differently from the old one (empty: only the words and
    numbers about it changed, so the video still matches)."""
    diff = [k for k in _VIDEO_KEYS if old.get(k) != new.get(k)]
    so, sn = old.get("stages") or [], new.get("stages") or []
    if len(so) != len(sn):
        diff.append("stages")
    else:
        diff += [f"stage {b['stage']}.{k}" for a, b in zip(so, sn) for k in _STAGE_KEYS if a.get(k) != b.get(k)]
    return diff


def refresh_truth(name: str, out_dir: Path, old: dict, log=print) -> Path:
    """Rewrite <name>.json from this generator, keeping the video: only when the narration, the timeline and the
    inks come out exactly as the video was made (from the cached speech); else the video must be made again."""
    lec = lecture(name)
    speech = speak(lec, out_dir / "tts" / name, cached_only=True)
    pl = plan(lec, speech)
    new = json.loads(json.dumps(truth_dict(lec, pl, Scene(lec, pl), int(old.get("seed", 0))), ensure_ascii=False))
    diff = truth_differs(old, new)
    if diff:
        raise RuntimeError(f"{name}: the video no longer matches the script ({diff[0]} changed); "
                           "make it again with --force")
    meta = _paths(out_dir, name)[1]
    tmp = meta.with_name(meta.name + ".partial")
    tmp.write_text(json.dumps(new, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, meta)
    log(f"{name}: truth rewritten (version {TRUTH_VERSION}), the video is kept")
    return meta


def build(name: str, out_dir: Path = OUT_DIR, seed: int = 0, force: bool = False, workers: int | None = None,
          log=print) -> tuple[Path, Path]:
    """Speak, plan, render and mux one lecture; writes out_dir/<name>.mp4 and <name>.json."""
    lec = lecture(name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video, meta = _paths(out_dir, name)
    if not force and video.is_file() and meta.is_file():
        try:
            old = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if isinstance(old, dict) and old.get("GEN_VERSION") == GEN_VERSION:
            if old.get("TRUTH_VERSION") != TRUTH_VERSION:
                refresh_truth(name, out_dir, old, log)
            else:
                log(f"{name}: up to date")
            return video, meta
    t_all = time.perf_counter()
    speech = speak(lec, out_dir / "tts" / name, force=force)
    pl = plan(lec, speech)
    bad = check_truth(pl)
    if bad:
        raise RuntimeError(f"{name}: the timeline is not consistent: {bad[0]}")
    scene = Scene(lec, pl)
    reach = scene.reach()
    if reach > _L1 + _L2:
        log(f"{name}: warning: the arm is {reach:.0f} px long at most while drawing (it reaches {_L1 + _L2:.0f})")
    wav = out_dir / "tts" / name / "mix.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(_mix_audio(pl, speech, seed).tobytes())
    n_frames = int(round(pl["duration"] * FPS))
    _background()
    enc = _Encoder(video, wav)
    workers = workers or max(2, min(6, (os.cpu_count() or 4) // 2))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            for i in range(n_frames):
                pending.append(pool.submit(scene.frame, i / FPS, seed, i))
                if len(pending) >= 2 * workers:
                    enc.write(pending.popleft().result())
            while pending:
                enc.write(pending.popleft().result())
        enc.close()
    except BaseException:
        enc.abort()
        raise
    truth = truth_dict(lec, pl, scene, seed)
    tmp = meta.with_name(meta.name + ".partial")
    tmp.write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, meta)
    log(f"{name}: {pl['duration']:.1f} s, {n_frames} frames, max arm reach {reach:.0f} px, "
        f"made in {time.perf_counter() - t_all:.0f} s -> {video}")
    return video, meta


def stills(name: str, times: list[float], out_dir: Path, seed: int = 0) -> list[Path]:
    """PNG frames at the given times, planned with estimated speech (quick look at the drawing, no audio)."""
    lec = lecture(name)
    pl = plan(lec)
    scene = Scene(lec, pl)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in times:
        img = scene.frame(t, seed, int(t * FPS))
        p = out_dir / f"{name}_{t:05.1f}.png"
        ok, buf = cv2.imencode(".png", img)
        p.write_bytes(buf.tobytes())
        paths.append(p)
    return paths


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate the 3D-figure board lectures (video + narration + truth).")
    ap.add_argument("lectures", nargs="*", default=list(LECTURES))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stills", nargs="*", type=float, default=None, help="only write PNG frames at these times")
    a = ap.parse_args()
    for name in a.lectures:
        if a.stills is not None:
            for p in stills(name, a.stills or [plan(lecture(name))["duration"] - 1.0], Path(a.out) / "stills", a.seed):
                print(p)
            continue
        build(name, Path(a.out), seed=a.seed, force=a.force)


if __name__ == "__main__":
    _main()
