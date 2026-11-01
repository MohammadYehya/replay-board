"""Pieces written on one line in bursts are grouped back into that line."""
from app.core.lines import group_lines
from app.notes import build_notes


def piece(i, box, t, text="x", kind="equation", **kw):
    return {"id": i, "box": box, "t_seen": t, "t_shown": t + 1, **kw}, {"kind": kind, "text": text, "latex": ""}


def test_groups_bursts_on_one_line_only():
    items = [
        piece(1, [550, 60, 20, 16], 180.0, "#1."),
        piece(2, [586, 62, 44, 12], 186.0, "A = (1+"),
        piece(3, [634, 60, 18, 19], 192.0, "1)^1 = 2"),
        piece(4, [550, 97, 28, 13], 211.0, "#2."),  # next line down
        piece(5, [700, 60, 30, 16], 400.0, "late"),  # same line, minutes later
        piece(6, [660, 60, 30, 16], 190.0, "graph", kind="graph"),  # a drawing
    ]
    pieces = [p for p, _ in items]
    reads = {p["id"]: r for p, r in items}
    lines = group_lines(pieces, reads)
    assert [ln["pieces"] for ln in lines] == [[1, 2, 3]]
    assert lines[0]["box"] == [550, 60, 102, 19] and lines[0]["t_complete"] == 193.0


def test_notes_use_the_line_reading():
    replay = {"type": "board", "chapters": [], "transcript": {"words": []},
              "board": {"pieces": [
                  {"id": 1, "t_seen": 180, "read": {"kind": "label", "text": "#1.", "latex": ""}, "spoken": []},
                  {"id": 2, "t_seen": 186, "read": {"kind": "equation", "text": "A = (1+", "latex": ""}, "spoken": []}],
                  "lines": [{"id": "L1", "pieces": [1, 2], "read": {"kind": "equation", "text": "#1. A = (1+1)^1 = 2",
                                                                      "latex": "A=(1+1)^1=2"}}]}}
    md = build_notes(replay)
    assert "$A=(1+1)^1=2$" in md and "A = (1+" not in md
