"""Study notes built from replay.json."""
from app.notes import build_notes


def test_board_notes_group_by_chapter_and_skip_unread():
    replay = {
        "type": "board", "subject": "Logs",
        "chapters": [{"t": 0.0, "title": "Intro"}, {"t": 60.0, "title": "Rules"}],
        "transcript": {"words": [{"w": "so", "s": 10.0, "e": 10.2}, {"w": "log", "s": 10.3, "e": 10.6}]},
        "board": {"pieces": [
            {"t_seen": 12.0, "read": {"kind": "equation", "text": "log 8 = 3", "latex": "\log_2 8 = 3"},
             "spoken": [{"start": 10.0, "end": 11.0}]},
            {"t_seen": 70.0, "read": {"kind": "graph", "text": "graph of y = ln x", "latex": ""},
             "spoken": [{"start": 71.0, "end": 73.0}]},
            {"t_seen": 75.0, "read": {"kind": "arrow", "text": "arrow pointing left", "latex": ""}, "spoken": []},
            {"t_seen": 80.0, "read": {"kind": "other", "text": "unreadable", "latex": ""}, "spoken": []},
            {"t_seen": 0.0, "pre_existing": True, "read": {"kind": "text", "text": "old", "latex": ""}, "spoken": []},
        ]},
    }
    md = build_notes(replay)
    assert md.startswith("# Logs")
    intro, rules = md.split("## Rules")
    assert "**0:10** $\log_2 8 = 3$" in intro and "so log" in intro
    assert "*graph of y = ln x*" in rules
    assert "unreadable" not in md and "old" not in md
    assert "arrow pointing left" not in md  # a drawing nobody talked about


def test_slide_notes_drop_repeats_from_build_steps():
    el = {"kind": "bullet", "text": "Interest on interest", "latex": "", "spoken": []}
    replay = {"type": "slides", "chapters": [], "transcript": {"words": []},
              "slides": {"states": [{"t_start": 0.0, "elements": [el]},
                                    {"t_start": 6.0, "elements": [el, {**el, "text": "Grows faster"}]}]}}
    md = build_notes(replay)
    assert md.count("Interest on interest") == 1 and "Grows faster" in md
