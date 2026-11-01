"""Board-capture self-test on generated lectures (SPEC 5.4).

    python -m tests.selftest_board            # all variants, prints a score table
    python -m tests.selftest_board drift      # one variant
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import DATA_DIR
from app.core.board import BoardGeometry, BoardParams, capture_board
from app.core.media import probe
from tests import gen_board as gen

OUT = DATA_DIR / "selftest"

# variant name -> (video variant, analysis width, use the oracle person mask)
CASES = {
    "base": ("base", 800, False),
    "chalk": ("chalk", 800, False),
    "fade": ("fade", 800, False),
    "drift": ("drift", 800, False),
    "base@1200": ("base", 1200, False),
    "base+person": ("base", 800, True),
}


@dataclass
class Score:
    case: str
    checks: list = field(default_factory=list)  # (name, ok, detail)
    seconds: float = 0.0

    def check(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), detail))

    @property
    def passed(self):
        return all(ok for _, ok, _ in self.checks)


def _dilate(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def pixel_scores(pred: np.ndarray, truth: np.ndarray, tol: int = 2) -> tuple[float, float]:
    """Recall and precision with a `tol`-pixel tolerance."""
    recall = (truth & _dilate(pred, tol)).sum() / max(1, truth.sum())
    precision = (pred & _dilate(truth, tol)).sum() / max(1, pred.sum())
    return float(recall), float(precision)


def run_case(case: str, verbose: bool = False) -> Score:
    variant, width, oracle = CASES[case]
    video = gen.make_board_lecture(OUT, variant)
    info = probe(video)
    params = BoardParams(width=width)
    person = (lambda t: gen.teacher_mask(t, variant)) if oracle else None
    corners = gen.board_corners()
    t0 = time.perf_counter()
    res = capture_board(info, corners=corners, person_mask=person, params=params)
    sc = Score(case, seconds=time.perf_counter() - t0)
    geom = BoardGeometry(info, corners, width, params.decode_width)

    def to_board(mask):
        return geom.warp_mask(mask)

    items = {it["id"]: it for it in gen.ITEMS}
    item_ink = {i: _dilate(to_board(gen.item_ink(i)), 2) for i in items}
    events = [p for p in res.pieces if not p["pre_existing"]]
    if verbose:
        for p in res.pieces:
            print("   piece", p)
        print("   erases", res.erases, "stats", res.stats)

    # Match pieces to items by ink overlap.
    matched: dict[int, list] = {i: [] for i in items}
    extras = []
    for p in events:
        sel = res.add_piece == p["id"]
        ys, xs = np.divmod(res.add_idx[sel], res.width)
        best, frac = None, 0.0
        for i, ink in item_ink.items():
            f = ink[ys, xs].mean() if ys.size else 0.0
            if f > frac:
                best, frac = i, f
        if best is not None and frac >= 0.5:
            matched[best].append(p)
        else:
            extras.append(p)

    sc.check("no pre-existing pieces", not any(p["pre_existing"] for p in res.pieces),
             f"{sum(p['pre_existing'] for p in res.pieces)} pre-existing")
    for i, it in items.items():
        ps = matched[i]
        if len(ps) != 1:
            sc.check(f"item {i} found once", False, f"{len(ps)} pieces")
            continue
        p = ps[0]
        if i == gen.HIDDEN_ITEM:
            ok = p["revealed"] and gen.UNCOVER_T - 0.5 <= p["t_seen"] <= gen.UNCOVER_T + 3.0
            sc.check(f"item {i} hidden, shown on uncover", ok,
                     f"t_seen {p['t_seen']} uncover {gen.UNCOVER_T:.2f} revealed {p['revealed']} written_after {p['written_after']}")
        else:
            ok = it["t_start"] <= p["t_seen"] <= it["t_end"] + 1.5 and not p["revealed"]
            sc.check(f"item {i} on time", ok, f"t_seen {p['t_seen']} written {it['t_start']}-{it['t_end']} revealed {p['revealed']}")
    sc.check("no extra pieces", not extras, ", ".join(f"#{p['id']} t={p['t_seen']} box={p['box']}" for p in extras))

    er = res.erases
    e_ok = len(er) == 1 and gen.ERASE_T[0] - 0.5 <= er[0]["t"] <= gen.ERASE_T[1] + 1.0
    if e_ok:
        x, y, w, h = er[0]["box"]
        ys, xs = np.nonzero(item_ink[gen.ERASE_ITEM])
        e_ok = x <= xs.mean() <= x + w and y <= ys.mean() <= y + h
    sc.check("exactly one erase, of item 1", e_ok, f"{[(e['t'], e['box']) for e in er]}")
    gone = [p for p in matched[gen.ERASE_ITEM] if p["removed_at"] is not None]
    sc.check("item 1 marked removed", len(gone) == 1, f"removed_at {[p['removed_at'] for p in matched[gen.ERASE_ITEM]]}")

    for T in (32.0, gen.DURATION - 0.1):
        truth = to_board(gen.truth_ink(T, variant))
        rec, prec = pixel_scores(res.ink_at(T), truth)
        sc.check(f"pixels at {T:.1f}s", rec >= 0.85 and prec >= 0.9, f"recall {rec:.3f} precision {prec:.3f}")

    speed = gen.DURATION / sc.seconds
    sc.check("speed >= 2x real time", speed >= 2.0, f"{speed:.1f}x")
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
            print(f"  [{'ok' if good else 'FAIL'}] {name:12} {check:34} {detail}")
        print(f"{name}: {'PASS' if sc.passed else 'FAIL'} ({sc.seconds:.1f}s)\n")
    print(f"self-test: {ok}/{total} checks passing")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
