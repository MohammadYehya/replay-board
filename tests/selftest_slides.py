"""Slide-analysis self-test on generated lectures (SPEC 5.4).

    python -m tests.selftest_slides            # all variants
    python -m tests.selftest_slides base -v    # one variant, verbose
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from app.config import DATA_DIR
from app.core.media import probe
from app.core.slides import SlideParams, analyse_slides
from tests import gen_slides as gen
from tests.selftest_board import Score

OUT = DATA_DIR / "selftest"
CASES = ("base", "dark", "fade", "pip")


def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / max(1e-6, aw * ah + bw * bh - inter)


def _time_overlap(a0, a1, b0, b1) -> float:
    """Overlap as a share of the shorter span."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-6, min(a1 - a0, b1 - b0))


def run_case(case: str, verbose: bool = False) -> Score:
    video = gen.make_slides_lecture(OUT, case)
    info = probe(video)
    p = SlideParams()
    k = p.width / gen.SIZE[0]  # video px -> working-copy px
    t0 = time.perf_counter()
    res = analyse_slides(info, OUT / f"work_{case}", params=p)
    sc = Score(case, seconds=time.perf_counter() - t0)
    if verbose:
        print("   live", res.live, "bg", res.background)
        for st in res.states:
            print(f"   state {st.id} {st.kind} {st.t_start:.2f}-{st.t_end:.2f} change@{st.t_change:.2f} blank={st.blank}")
        for g in res.gestures:
            print("   gesture", {k_: v for k_, v in g.items() if k_ != "path"})
        print("   stats", res.stats)

    # states
    truth = gen.states(case)
    if case != "fade":
        ok = len(res.states) == len(truth)
        detail = []
        for tr, st in zip(truth, res.states):
            good = st.kind == tr["kind"] and abs(st.t_start - tr["t_start"]) <= 0.75
            ok &= good
            detail.append(f"{tr['kind'][0]}{tr['t_start']:.1f}/{st.kind[0]}{st.t_start:.1f}{'' if good else '!'}")
        sc.check("states: right count, kind, time", ok, f"{len(res.states)}/{len(truth)}: " + " ".join(detail))

    # webcam
    wb = gen.webcam_box(case)
    if wb is not None:
        wbk = [v * k for v in wb]
        best = max((_iou(b, wbk) for b in res.live), default=0.0)
        sc.check("webcam box found (IoU >= 0.7)", best >= 0.7 and len(res.live) == 1, f"IoU {best:.2f}, {len(res.live)} live box(es)")

    # gestures
    want = gen.gestures(case)
    found = [g for g in res.gestures if g["type"] in ("hover", "sweep", "circle", "wiggle", "arrow")]
    if case == "pip":
        sc.check("no gestures", not [g for g in found if not (g["type"] == "hover" and g.get("parked"))], f"{len(found)} found")
    for w in want:
        hit = [g for g in found if g["type"] == w["type"] and g["tool"] == w["tool"]
               and _time_overlap(g["t_start"], g["t_end"], w["t_start"], w["t_end"]) >= 0.5]
        sc.check(f"{w['tool']} {w['type']} @{w['t_start']:.1f}", bool(hit),
                 f"found: {[(g['type'], g['tool'], g['t_start'], g['t_end']) for g in found if abs(g['t_start'] - w['t_start']) < 2]}")
    arrows_true = [w for w in want if w["type"] == "arrow"]
    false_arrows = [g for g in found if g["type"] == "arrow"
                    and not any(_time_overlap(g["t_start"], g["t_end"], w["t_start"], w["t_end"]) >= 0.5 for w in arrows_true)]
    sc.check("no travel flagged as arrow", not false_arrows, f"{[(g['t_start'], g['t_end']) for g in false_arrows]}")
    false_cw = [g for g in found if g["type"] in ("circle", "wiggle")
                and not any(w["type"] == g["type"] and _time_overlap(g["t_start"], g["t_end"], w["t_start"], w["t_end"]) >= 0.5 for w in want)]
    sc.check("no false circles or wiggles", not false_cw, f"{[(g['type'], g['t_start'], g['t_end']) for g in false_cw]}")

    # clean slides: no cursor left (looked for where the cursor really was, before/during/after)
    if case != "fade":
        worst = (0, None)
        for tr, st in zip(truth, res.states):
            if st.image is None:
                continue
            ref = cv2.resize(gen.clean_slide(tr["id"], case), (st.image.shape[1], st.image.shape[0]),
                             interpolation=cv2.INTER_AREA)
            d = cv2.absdiff(st.image, ref).max(axis=2) > 60
            spots = set()
            for t in np.arange(tr["t_start"] - 1.0, tr["t_end"] + 1.0, 0.25):
                ptr = gen.pointer(float(t), case)
                if ptr and ptr[0] == "cursor":
                    spots.add((int(ptr[1] * k) // 4 * 4, int(ptr[2] * k) // 4 * 4))
            for x, y in spots:
                n_px = int(d[max(0, y - 4):y + 20, max(0, x - 4):x + 14].sum())
                if n_px > worst[0]:
                    worst = (n_px, (tr["id"], x, y))
        sc.check("no cursor left in clean slides", worst[0] < 15, f"worst {worst[0]} px at {worst[1]}")

    speed = gen.DURATION / sc.seconds
    sc.check("speed >= 1x real time", speed >= 1.0, f"{speed:.1f}x")
    return sc


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verbose = "-v" in argv
    names = [a for a in argv if not a.startswith("-")] or list(CASES)
    total = ok = 0
    for name in names:
        sc = run_case(name, verbose)
        for check, good, detail in sc.checks:
            total += 1
            ok += good
            print(f"  [{'ok' if good else 'FAIL'}] {name:6} {check:34} {detail}")
        print(f"{name}: {'PASS' if sc.passed else 'FAIL'} ({sc.seconds:.1f}s)\n")
    print(f"slides self-test: {ok}/{total} checks passing")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
