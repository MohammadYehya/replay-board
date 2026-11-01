"""Lines of writing: pieces that sit on one written line, for typeset mode and notes.

A teacher often writes one line in bursts with pauses ("#1. A = (1 +" ... "1)^1 = 2"), and board
capture rightly makes several pieces of it (each has its own time). Typeset, those fragments read
badly; grouped back into their line, Muse Spark can read and typeset the whole line at once.
"""
from __future__ import annotations

DRAWINGS = {"diagram", "graph", "arrow", "scribble", "underline", "box", "figure", "chart", "drawing",
            "plot", "sketch", "curve", "circle", "line", "shape", "picture", "bracket"}


def _end(p: dict) -> float:
    d = p.get("draw")
    if isinstance(d, (list, tuple)) and len(d) == 2 and d[1] is not None:
        return float(d[1])
    return float(p.get("t_shown", p.get("t_seen", 0.0)))


def same_line(a: dict, b: dict, max_dt: float = 90.0, gap_k: float = 0.9, min_gap: float = 25.0,
              max_ratio: float = 3.0) -> bool:
    """Side by side on one baseline, written close together in time, of comparable height (a fraction can make
    one piece of a line a few times taller than "#1.", but a block of several lines is not on any one line)."""
    if bool(a.get("pre_existing")) != bool(b.get("pre_existing")):
        return False
    if not a.get("pre_existing") and abs(a["t_seen"] - b["t_seen"]) > max_dt:
        return False
    ax, ay, aw, ah = a["box"]
    bx, by, bw, bh = b["box"]
    if max(ah, bh) > max_ratio * max(1, min(ah, bh)):
        return False
    overlap = min(ay + ah, by + bh) - max(ay, by)
    if overlap < 0.5 * min(ah, bh):
        return False
    gap = max(bx - (ax + aw), ax - (bx + bw), 0)
    return gap <= max(min_gap, gap_k * min(ah, bh))


def group_lines(pieces: list[dict], reads: dict, max_dt: float = 90.0) -> list[dict]:
    """Lines of 2+ textual pieces: [{id, pieces (left to right), box, t_complete, removed_at}].

    A piece taller than three typical lines is a block of its own (often everything already on the board
    when the recording starts) and is left out: it would chain the lines beside it into one."""
    text = [p for p in pieces
            if (reads.get(p["id"]) or {}).get("kind", "").lower() not in DRAWINGS
            and ((reads.get(p["id"]) or {}).get("text") or (reads.get(p["id"]) or {}).get("latex"))]
    if text:
        hs = sorted(p["box"][3] for p in text)
        text = [p for p in text if p["box"][3] <= 3 * hs[len(hs) // 2]]
    parent = {p["id"]: p["id"] for p in text}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(text):
        for b in text[i + 1:]:
            if same_line(a, b, max_dt):
                parent[find(a["id"])] = find(b["id"])
    groups: dict[int, list[dict]] = {}
    for p in text:
        groups.setdefault(find(p["id"]), []).append(p)
    lines = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda p: p["box"][0])
        x0 = min(p["box"][0] for p in members)
        y0 = min(p["box"][1] for p in members)
        x1 = max(p["box"][0] + p["box"][2] for p in members)
        y1 = max(p["box"][1] + p["box"][3] for p in members)
        removed = [p.get("removed_at") for p in members if p.get("removed_at") is not None]
        lines.append({
            "pieces": [p["id"] for p in members], "box": [x0, y0, x1 - x0, y1 - y0],
            "t_complete": round(max(_end(p) for p in members), 2),
            "removed_at": round(min(removed), 2) if removed else None,
            "pre_existing": bool(members[0].get("pre_existing")),
        })
    lines.sort(key=lambda ln: (ln["t_complete"], ln["box"][1]))
    for i, ln in enumerate(lines):
        ln["id"] = f"L{i + 1}"
    return lines
