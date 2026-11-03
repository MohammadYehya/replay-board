"""Linking writing to speech, and chapters, for short and long lectures (app/core/understand.py link)."""
import re

from app.core import understand


class FakeClient:
    """Answers each window with chapters at the times it is given, and records the prompts."""

    def __init__(self, chapters_by_window):
        self.prompts = []
        self.chapters_by_window = chapters_by_window

    def ask(self, prompt, schema=None, reasoning=None):
        self.prompts.append(prompt)
        ids = re.findall(r'"id": "([^"]+)"', prompt)
        start = re.search(r"between (\d+) and", prompt)
        a = int(start.group(1)) if start else 0
        return {"elements": [{"id": i, "spoken": [], "redraw_at": -1} for i in ids],
                "chapters": self.chapters_by_window.get(a, [])}


def _elements(times):
    return [{"id": str(k), "kind": "text", "text": f"x{k}", "t_seen": t, "t_shown": t, "written_after": t - 1,
             "revealed": False, "removed_at": None} for k, t in enumerate(times)]


def _words(duration, step=5.0):
    return [{"w": "word", "s": float(t), "e": float(t) + 0.4} for t in range(0, int(duration), int(step))]


def test_a_short_lecture_asks_for_the_whole_lectures_chapters_as_before():
    # the prompt is unchanged for lectures up to 20 minutes, so answers cached before are still used
    c = FakeClient({0: [{"t": 0, "title": "Intro"}, {"t": 300, "title": "Cells"}]})
    out = understand.link(c, _elements([10, 200, 500]), _words(600), 600.0)
    assert len(c.prompts) == 1
    assert ("chapters: split the whole lecture into 3-10 topic chapters with short titles (first at 0).\n"
            in c.prompts[0])
    assert out["chapters"] == [{"t": 0.0, "title": "Intro"}, {"t": 300.0, "title": "Cells"}]
    assert set(out["elements"]) == {"0", "1", "2"}


def test_a_long_lecture_gets_chapters_from_every_window():
    # an hour: three 20-minute windows; before, only the first was asked, so chapters stopped at about 22 minutes
    c = FakeClient({0: [{"t": 0, "title": "Intro"}, {"t": 700, "title": "Lattices"},
                        {"t": 1300, "title": "past its end: dropped"}],
                    1200: [{"t": 1250, "title": "Symmetry"}, {"t": 1100, "title": "before its start: dropped"}],
                    2400: [{"t": 3000, "title": "Diffraction"}, {"t": "late", "title": "no time: dropped"}]})
    out = understand.link(c, _elements([30, 900, 1500, 2500, 3300]), _words(3600), 3600.0)
    assert len(c.prompts) == 3
    first = next(p for p in c.prompts if "up to 1200 s" in p)
    assert "split this part of the lecture (up to 1200 s) into 2-5 topic chapters" in first
    assert sum("topics that start in this part of the lecture (between" in p for p in c.prompts) == 2
    assert not any("split the whole lecture" in p for p in c.prompts)
    assert out["chapters"] == [{"t": 0.0, "title": "Intro"}, {"t": 700.0, "title": "Lattices"},
                               {"t": 1250.0, "title": "Symmetry"}, {"t": 3000.0, "title": "Diffraction"}]
    assert set(out["elements"]) == {"0", "1", "2", "3", "4"}


def test_malformed_chapters_are_left_out():
    c = FakeClient({0: [{"t": "soon", "title": "a"}, {"t": 5, "title": None}, "chapter", {"t": 9, "title": " "},
                        {"t": 12, "title": "Kept"}]})
    assert understand.link(c, _elements([1]), _words(60), 60.0)["chapters"] == [{"t": 12.0, "title": "Kept"}]


def test_a_long_stretch_of_talk_without_new_writing_still_gets_its_chapters():
    # 50 minutes, writing only in the first 10: the windows from 20 and 40 minutes are asked too (talk only)
    c = FakeClient({0: [{"t": 0, "title": "Intro"}], 1200: [{"t": 1500, "title": "Proof"}],
                    2400: [{"t": 2700, "title": "Examples"}]})
    out = understand.link(c, _elements([30, 300]), _words(3000), 3000.0)
    assert len(c.prompts) == 3
    assert [ch["title"] for ch in out["chapters"]] == ["Intro", "Proof", "Examples"]
    assert set(out["elements"]) == {"0", "1"}


def test_a_clip_cut_from_a_long_video_starts_its_chapters_where_the_clip_does():
    # --start 900 --end 1500: times are the video's, so the first window opens at 900, not 0
    c = FakeClient({})
    words = [w for w in _words(1500) if w["s"] >= 900]
    understand.link(c, _elements([930, 1100, 1300]), words, 1500.0)
    assert any("(first at 900)" in p for p in c.prompts) and not any("first at 0)" in p for p in c.prompts)
