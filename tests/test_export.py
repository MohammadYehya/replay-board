"""Slides export (app/export.py and its endpoints), offline: board pages (cuts, merging, focus, crop, fading,
items, what was said), slide pages, the cached deck, the key points call (a fake model) and the PowerPoint."""
import io
import json
import math
import warnings

import cv2
import numpy as np
import pytest

with warnings.catch_warnings():  # starlette's notice about httpx, at import time
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app import ask, export, server
from app.core.media import imread, imwrite
from app.core.meta_api import CallRecord, MetaAPIError

BG = "#f0f0e8"
BG_BGR = np.array([0xE8, 0xF0, 0xF0], np.float32)
INK = np.array([200, 40, 10], np.float32)


# ---------- lectures made up for the tests ----------
def piece(i, box, t, kind="text", text=None, latex="", spoken=(), **kw):
    """A piece the board starts drawing at t (its draw window), unless kw says otherwise."""
    p = {"id": i, "box": list(box), "image": f"pieces/{i:03d}.png", "draw": [t, t + 1.0], "t_seen": t + 0.3,
         "t_shown": t + 1.0, "written_after": t, "revealed": False, "removed_at": None, "pre_existing": False,
         "read": {"kind": kind, "text": f"item {i}" if text is None else text, "latex": latex},
         "spoken": [{"start": a, "end": b} for a, b in spoken], "redraw_at": None}
    p.update(kw)
    return p


def old(i, box, start=0.0, **kw):
    """Writing that was on the board before the recording."""
    return piece(i, box, start, pre_existing=True, t_seen=start, t_shown=start, **kw)


def board(pieces, chapters, erases=(), lines=(), start=0.0, end=180.0, size=(800, 400), words=None, subject="growth and logs"):
    if words is None:
        words = [{"w": f"w{i}.", "s": start + i + 0.1, "e": start + i + 0.5} for i in range(int(end - start))]
    return {
        "version": 1, "type": "board", "subject": subject,
        "video": {"file": "lecture.mp4", "duration": max(end, 200.0), "width": 1280, "height": 720,
                  "start": start, "end": end},
        "chapters": [{"t": t, "title": title} for t, title in chapters],
        "transcript": {"turns": [], "words": words, "timing": "mms"},
        "board": {"size": list(size), "background": BG, "pieces": pieces, "lines": list(lines),
                  "erases": [{"t": t, "box": list(b), "pixels": 100} for t, b in erases], "annotations": []},
    }


def solid(w, h):
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = INK.astype(np.uint8)
    img[..., 3] = 255
    return img


def write_job(d, replay, slide_size=(960, 540)):
    d.mkdir(parents=True, exist_ok=True)
    (d / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    if replay["type"] == "board":
        for p in replay["board"]["pieces"]:
            imwrite(d / p["image"], solid(p["box"][2], p["box"][3]))
    else:
        for st in replay["slides"]["states"]:
            if st.get("image"):
                img = np.full((slide_size[1], slide_size[0], 3), 255, np.uint8)
                cv2.putText(img, f"state {st['id']}", (60, slide_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
                imwrite(d / st["image"], img)
    return d


def segs(replay, d):
    return [(s["t0"], s["t1"]) for s in export.board_segments(replay, export.Pixels(d))]


# ---------- board pages ----------
def test_draw_start_follows_the_players_schedule():
    assert export.draw_start(old(1, (0, 0, 5, 5))) == -math.inf
    assert export.draw_start(piece(1, (0, 0, 5, 5), 10.0)) == 10.0  # its draw window
    no_window = piece(1, (0, 0, 5, 5), 10.0, draw=None, t_seen=12.0)
    assert export.draw_start(no_window) == pytest.approx(11.3)
    hidden = piece(1, (0, 0, 5, 5), 10.0, revealed=True, redraw_at=8.0, written_after=9.0, t_seen=14.0)
    assert export.draw_start(hidden) == 9.0  # redrawn when talked about, not before it can have been written
    assert export.draw_start({**hidden, "redraw_at": None}) == 14.0
    assert export.draw_start(piece(1, (0, 0, 5, 5), 2.0), clip0=5.0) == 5.0  # never before the clip


def test_pages_cut_at_chapters_and_big_erases_but_not_small_ones(tmp_path):
    pieces = [
        piece(1, (100, 100, 50, 30), 10),
        old(2, (700, 20, 20, 10)),
        piece(3, (300, 200, 100, 50), 70, removed_at=120.5),  # most of the ink, wiped at 120
        piece(4, (600, 300, 10, 10), 75, removed_at=90.4),  # a small fix at 90
        piece(5, (500, 100, 40, 40), 130),
    ]
    r = board(pieces, [(0, "Intro"), (60, "Growth")], erases=[(40, (5, 5, 5, 5)), (90, (600, 300, 10, 10)),
                                                            (120, (300, 200, 20, 20))])
    d = write_job(tmp_path / "lec", r)
    assert segs(r, d) == [(0, 60), (60, 120), (120, 180)]
    s = export.board_segments(r, export.Pixels(d))
    assert s[1]["before"] == pytest.approx(119.95) and s[0]["before"] is None and s[2]["before"] is None

    deck = export.build_deck(d, r, "v1")
    assert [p["title"] for p in deck["pages"]] == ["Intro", "Growth", "Growth (2)"]
    assert [p["chapter"] for p in deck["pages"]] == ["Intro", "Growth", "Growth"]
    assert [i["ref"] for i in deck["pages"][1]["items"]] == ["#3", "#4"]  # #4 was erased, but it was written here

    # The same erase, without the ink: still a cut when its box covers 8% of the board.
    r2 = board([piece(1, (100, 100, 50, 30), 10), piece(5, (500, 100, 40, 40), 130), piece(6, (50, 300, 20, 20), 70)],
               [(0, "Intro")], erases=[(120, (0, 0, 400, 70))])
    d2 = write_job(tmp_path / "area", r2)
    assert segs(r2, d2) == [(0, 120), (120, 180)]
    r2["board"]["erases"][0]["box"] = [0, 0, 300, 70]  # 6.6%: not a cut
    assert segs(r2, d2) == [(0, 180)]


def test_a_wipe_in_several_erases_is_one_cut(tmp_path):
    pieces = [piece(1, (100, 100, 60, 60), 10, removed_at=101.0), piece(2, (300, 100, 60, 60), 20, removed_at=103.5),
              piece(3, (500, 100, 60, 60), 120)]
    r = board(pieces, [(0, "Intro")], erases=[(100.8, (100, 100, 60, 60)), (103.3, (300, 100, 60, 60))])
    d = write_job(tmp_path / "lec", r)
    s = export.board_segments(r, export.Pixels(d))
    assert [(x["t0"], x["t1"]) for x in s] == [(0, 100.8), (100.8, 180)]
    assert s[0]["before"] == pytest.approx(100.75)  # the board before the wipe began, with both pieces
    deck = export.build_deck(d, r, "v1")
    assert [i["ref"] for i in deck["pages"][0]["items"]] == ["#1", "#2"]


def test_a_short_stretch_after_a_big_erase_joins_the_next_page(tmp_path):
    pieces = [piece(1, (100, 100, 200, 100), 10, removed_at=100.5), piece(2, (100, 100, 40, 40), 105),
              piece(3, (500, 100, 40, 40), 130)]
    r = board(pieces, [(0, "Intro"), (110, "Next")], erases=[(100, (100, 100, 200, 100))])
    d = write_job(tmp_path / "lec", r)
    s = export.board_segments(r, export.Pixels(d))
    # [100,110) is short; joined to the page before, that page would show the board after the wipe.
    assert [(x["t0"], x["t1"]) for x in s] == [(0, 100), (100, 180)] and s[0]["before"] == pytest.approx(99.95)
    deck = export.build_deck(d, r, "v1")
    assert [i["ref"] for i in deck["pages"][0]["items"]] == ["#1"]
    assert [p["title"] for p in deck["pages"]] == ["Intro", "Intro (2)"]


def test_short_and_empty_stretches_join_their_neighbour(tmp_path):
    chapters = [(0, "Intro"), (15, "Warm-up"), (50, "Growth"), (60, "Aside"), (100, "Logs")]
    pieces = [piece(1, (100, 100, 50, 30), 20), piece(2, (300, 100, 50, 30), 120)]
    r = board(pieces, chapters)
    d = write_job(tmp_path / "lec", r)
    # [0,15) is short: joins the next; [50,60) is short and [60,100) has nothing: they join the one before.
    assert segs(r, d) == [(0, 100), (100, 180)]
    deck = export.build_deck(d, r, "v1")
    assert [p["title"] for p in deck["pages"]] == ["Intro", "Logs"]

    # Writing from before the recording, talked about in [60,100): that stretch is a page of its own.
    r["board"]["pieces"].append(old(3, (600, 300, 50, 30), spoken=[(70, 80)]))
    write_job(d, r)
    assert segs(r, d) == [(0, 60), (60, 100), (100, 180)]
    # Talk that only grazes the start of a page (word times snap early) does not make it a page.
    r["board"]["pieces"][-1]["spoken"] = [{"start": 59.8, "end": 60.1}]
    assert segs(r, d) == [(0, 100), (100, 180)]


def test_a_clip_starting_late_and_writing_from_before_it(tmp_path):
    pieces = [
        old(1, (50, 50, 100, 40), start=380.0),  # never talked about: faded, never the focus
        old(2, (400, 50, 100, 40), start=380.0, spoken=[(385, 390)]),
        piece(3, (400, 200, 100, 40), 405),
    ]
    r = board(pieces, [(0, "Before the clip"), (400, "Inside")], start=380.0, end=480.0)
    d = write_job(tmp_path / "lec", r)
    assert segs(r, d) == [(380, 400), (400, 480)]
    deck = export.build_deck(d, r, "v1")
    p1, p2 = deck["pages"]
    assert (p1["title"], p2["title"]) == ("Before the clip", "Inside")
    assert [i["ref"] for i in p1["items"]] == ["#2"] and p1["items"][0]["t"] == 385.0
    assert [i["ref"] for i in p2["items"]] == ["#3"] and p2["items"][0]["t"] == 405.0
    assert p1["said"].startswith("w0.") and "w20." in p2["said"] and "w19." not in p2["said"]


def test_crop_is_padded_at_least_a_third_wide_and_widened_to_16_9_inside_the_board():
    W, H = 800, 400
    x, y, w, h = export.crop_box([[100, 100, 100, 100]], W, H)
    assert w / h == pytest.approx(16 / 9, abs=0.02)
    assert x <= 76 and y <= 76 and x + w >= 224 and y + h >= 224  # padded by 24 px
    assert 0 <= x and 0 <= y and x + w <= W and y + h <= H
    x, y, w, h = export.crop_box([[700, 300, 30, 20]], W, H)  # a short formula in a corner
    assert w >= 0.3 * W and x + w <= W and y + h <= H and w / h == pytest.approx(16 / 9, abs=0.02)
    x, y, w, h = export.crop_box([[10, 150, 400, 30]], W, H)  # a long line: taller toward 16:9
    assert h > 30 + 48 and w / h == pytest.approx(16 / 9, abs=0.02)
    assert export.crop_box([[0, 0, 700, 350]], W, H) is None  # most of the board: the whole board
    assert export.crop_box([[10, 10, 50, 50], [700, 300, 50, 50]], W, H) is None
    assert export.crop_box([], W, H) is None
    x, y, w, h = export.crop_box([[200, 200, 200, 200]], 1600, 800)  # padding scales with the board
    assert x <= 152 and y <= 152


def test_the_page_image_fades_the_rest_of_the_writing(tmp_path):
    pieces = [old(1, (350, 180, 60, 30)), piece(2, (20, 20, 40, 20), 10), piece(3, (720, 360, 60, 30), 20)]
    r = board(pieces, [(0, "Intro")], end=60.0)
    d = write_job(tmp_path / "lec", r)
    deck = export.build_deck(d, r, "v1")
    (page,) = deck["pages"]
    img = imread(d / page["image"], cv2.IMREAD_COLOR).astype(np.float32)
    assert page["image_size"] == [1600, 800] == [img.shape[1], img.shape[0]]  # the whole board, 2x
    focus = img[2 * 30, 2 * 40]  # inside piece 2
    faded = img[2 * 195, 2 * 380]  # inside piece 1, there before and never talked about
    blank = img[2 * 300, 2 * 200]
    assert np.abs(focus - INK).max() <= 3
    assert np.abs(faded - (export.FADE * INK + (1 - export.FADE) * BG_BGR)).max() <= 3
    assert np.abs(blank - BG_BGR).max() <= 2


def test_a_crop_page_and_partly_erased_writing(tmp_path):
    p = piece(1, (100, 100, 40, 20), 10)
    r = board([p, old(2, (600, 300, 50, 30))], [(0, "Intro")], end=60.0)
    d = write_job(tmp_path / "lec", r)
    gone = np.zeros((20, 40, 3), np.uint8)
    code = 30 * 10 + 1  # the right half leaves the board at 30 s: code = t * 10 + 1 = B | G << 8 | R << 16
    gone[:, 20:, 0], gone[:, 20:, 1] = code & 255, code >> 8
    imwrite(d / "pieces/001_gone.png", gone)
    r["board"]["pieces"][0]["gone"] = "pieces/001_gone.png"
    deck = export.build_deck(d, r, "v1")
    (page,) = deck["pages"]
    img = imread(d / page["image"], cv2.IMREAD_COLOR).astype(np.float32)
    h, w = img.shape[:2]
    assert w / h == pytest.approx(16 / 9, abs=0.03) and max(w, h) <= 1600 and w <= 3 * 0.3 * 800 + 3
    x, y, cw, ch = export.crop_box([[100, 100, 40, 20]], 800, 400)
    k = w / cw
    left = img[int((110 - y) * k), int((105 - x) * k)]
    right = img[int((110 - y) * k), int((135 - x) * k)]
    assert np.abs(left - INK).max() <= 6 and np.abs(right - BG_BGR).max() <= 6


def test_items_in_reading_order_with_lines_once(tmp_path):
    pieces = [
        piece(2, (60, 10, 30, 20), 5, text="1 +", spoken=[(4, 6)]),
        piece(3, (95, 12, 30, 18), 8, text="1 = 2", spoken=[(8.2, 9)]),
        piece(4, (10, 60, 50, 30), 20, kind="graph", text="graph of e^x"),
        piece(5, (300, 14, 40, 20), 25, kind="equation", text="x^2", latex="$x^2$"),
        piece(6, (200, 200, 40, 20), 30, kind="other", text=""),  # nothing readable: left out
        piece(7, (20, 300, 40, 20), 31, text="unreadable", kind="arrow", spoken=[(31, 34)]),  # a drawing counts
        piece(8, (400, 100, 30, 150), 32, text="tall a"),
        piece(9, (440, 100, 30, 150), 33, text="tall b"),
    ]
    lines = [{"id": "L1", "pieces": [2, 3], "box": [60, 10, 65, 22], "t_complete": 9,
              "read": {"kind": "equation", "text": "1 + 1 = 2", "latex": "1+1=2"}},
             {"id": "L2", "pieces": [8, 9], "box": [400, 100, 70, 150], "t_complete": 34,  # too tall to be a line
              "read": {"kind": "text", "text": "two columns", "latex": ""}}]
    r = board(pieces, [(0, "Intro")], lines=lines, end=60.0)
    d = write_job(tmp_path / "lec", r)
    (page,) = export.build_deck(d, r, "v1")["pages"]
    assert [i["ref"] for i in page["items"]] == ["#L1", "#5", "#4", "#8", "#9", "#7"]
    by = {i["ref"]: i for i in page["items"]}
    assert by["#L1"] == {"ref": "#L1", "kind": "equation", "text": "1 + 1 = 2", "latex": "1+1=2", "t": 5.0,
                         "drawing": False}  # talked about from 4 s, but drawn from 5 s
    assert by["#5"]["latex"] == "x^2" and by["#5"]["t"] == 25.0
    assert by["#4"]["drawing"] is True and by["#7"]["drawing"] is True and by["#7"]["text"] == ""


def test_what_was_said_is_cut_at_a_sentence():
    words = [{"w": w, "s": i * 0.3, "e": i * 0.3 + 0.2}
             for i, w in enumerate(("This is a fairly long sentence about growth that goes on. " * 80).split())]
    text = export.said_text(words, 0, 1e9)
    assert len(text) <= export.SAID_CHARS and text.endswith(" ...") and text[:-4].endswith(".")
    assert export.said_text(words, 0.3, 0.85) == "is a"  # from a, up to (not including) b
    assert export.said_text(words, 0.6, 0.6) == ""
    many = [{"w": "word", "s": i * 0.1, "e": i * 0.1} for i in range(1000)]
    cut = export.said_text(many, 0, 1e9)
    assert len(cut) <= export.SAID_CHARS and cut.endswith("word ...")


# ---------- slide pages ----------
def slides_replay():
    def el(i, kind, text, box, latex="", spoken=()):
        return {"id": i, "kind": kind, "text": text, "latex": latex, "box": list(box),
                "spoken": [{"start": a, "end": b} for a, b in spoken]}

    def state(i, kind, t0, t1, els, blank=False):
        return {"id": i, "kind": kind, "t_start": t0, "t_end": t1, "blank": blank,
                "image": f"slides/{i:03d}.png", "elements": els}

    return {
        "version": 1, "type": "slides", "subject": "compound interest",
        "video": {"file": "lecture.mp4", "duration": 40.0, "width": 1920, "height": 1080, "start": 0.0, "end": 40.0},
        "chapters": [],
        "transcript": {"words": [{"w": "hello.", "s": 0.5, "e": 0.9}, {"w": "interest", "s": 3.0, "e": 3.5},
                                 {"w": "grows.", "s": 12.5, "e": 13.0}, {"w": "bye.", "s": 39.0, "e": 39.5}]},
        "slides": {"size": [1920, 1080], "states": [
            state(0, "addition", 0.0, 2.0, [el("s0e1", "text", "Welcome", (10, 10, 100, 30))]),
            state(1, "slide", 2.0, 10.0, [el("s1e1", "title", "Compound interest", (10, 10, 200, 30)),
                                          el("s1e2", "bullet", "Interest on interest", (10, 60, 300, 20),
                                             spoken=[(4, 6)])]),
            state(2, "addition", 10.0, 20.0, [el("s2e1", "title", "Compound interest", (10, 10, 200, 30)),
                                              el("s2e2", "bullet", "Interest on interest", (10, 60, 300, 20)),
                                              el("s2e3", "chart", "", (10, 100, 300, 200)),
                                              el("s2e4", "equation", "A = P(1+r)^n", (10, 320, 200, 40),
                                                 "A=P(1+r)^n", [(12, 15)])]),
            state(3, "slide", 20.0, 21.0, [], blank=True),
            state(4, "slide", 21.0, 30.0, [el("s4e1", "title", "Towards e", (10, 10, 200, 30))]),
            state(5, "addition", 30.0, 40.0, [el("s5e1", "title", "Towards e", (10, 10, 200, 30)),
                                              el("s5e2", "equation", "e = 2.718", (10, 60, 200, 30), "e=2.718")]),
        ], "annotations": []},
    }


def test_slides_one_page_per_slide_with_its_build_steps(tmp_path):
    r = slides_replay()
    d = write_job(tmp_path / "lec", r, slide_size=(1920, 1080))
    assert [[s["id"] for s in g] for g in export.slide_groups(r)] == [[0], [1, 2], [4, 5]]
    deck = export.build_deck(d, r, "v1")
    pages = deck["pages"]
    assert [(p["t0"], p["t1"]) for p in pages] == [(0.0, 2.0), (2.0, 20.0), (21.0, 40.0)]
    assert [p["title"] for p in pages] == ["Slide 1", "Compound interest", "Towards e"]  # no chapters: the slide's own
    assert [p["image_size"] for p in pages] == [[1600, 900]] * 3  # the last state's image, at most 1600 px
    img = imread(d / pages[1]["image"], cv2.IMREAD_COLOR)
    ref = cv2.resize(imread(d / "slides/002.png", cv2.IMREAD_COLOR), (1600, 900), interpolation=cv2.INTER_AREA)
    assert np.abs(img.astype(int) - ref.astype(int)).mean() < 1  # state 2 (the build step), not state 1
    items = pages[1]["items"]
    assert [i["ref"] for i in items] == ["#s2e1", "#s2e2", "#s2e3", "#s2e4"]
    assert items[1]["t"] == 4.0  # talked about on the build step before: that time
    assert items[0]["t"] == 2.0 and items[3]["t"] == 12.0
    assert items[2]["drawing"] is True and items[3]["latex"] == "A=P(1+r)^n"
    assert pages[0]["said"] == "hello." and pages[1]["said"] == "interest grows." and pages[2]["said"] == "bye."


# ---------- the deck, cached ----------
def test_the_deck_is_cached_and_rebuilt_when_replay_json_changes(tmp_path, monkeypatch):
    r = board([piece(1, (100, 100, 50, 30), 10), piece(2, (300, 100, 50, 30), 70)], [(0, "Intro"), (60, "Growth")])
    d = write_job(tmp_path / "lec", r)
    deck, _ = export.load_deck(d)
    assert len(deck["pages"]) == 2 and (d / "export" / "deck.json").exists() and (d / "export/page_002.png").exists()
    real = export.build_deck
    monkeypatch.setattr(export, "build_deck", lambda *a: pytest.fail("rebuilt a deck that was up to date"))
    again, _ = export.load_deck(d)
    assert again == deck

    export._save_json(d / "export" / "points.json", {"version": deck["version"], "pages": {"p1": {"title": "T", "points": ["a"]}}})
    assert export.load_points(d, deck["version"]) == {"p1": {"title": "T", "points": ["a"]}}

    r["chapters"] = [{"t": 0, "title": "Intro"}]  # the pipeline rewrote replay.json
    (d / "replay.json").write_text(json.dumps(r), encoding="utf-8")
    monkeypatch.setattr(export, "build_deck", real)
    new, _ = export.load_deck(d)
    assert new["version"] != deck["version"] and len(new["pages"]) == 1
    assert not (d / "export/page_002.png").exists()  # the old deck's extra page is gone
    assert export.load_points(d, new["version"]) is None  # key points of the old deck are stale
    assert export.slides_data(d)["points_ready"] is False

    (d / "export/page_001.png").unlink()  # a missing image: built again
    export.load_deck(d)
    assert (d / "export/page_001.png").exists()


def test_no_replay_yet_is_not_ready(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(export.NotReady, match="not processed yet"):
        export.load_deck(tmp_path / "raw")
    (tmp_path / "raw" / "replay.json").write_text("{half", encoding="utf-8")
    with pytest.raises(export.NotReady, match="being updated"):
        export.load_deck(tmp_path / "raw")


def test_lecture_title_and_file_name(tmp_path):
    r = board([], [(0, "Intro")], subject="")
    d = write_job(tmp_path / "lec", r)
    (d / "status.json").write_text(json.dumps({"name": "Week 3 — logs.mp4"}), encoding="utf-8")
    assert export.lecture_title(d, r) == "Week 3 — logs"
    assert export.lecture_title(d, {**r, "subject": "growth"}) == "Growth"
    asc, uni = export.file_name('Logs: été / "growth"?')
    assert asc == "Logs ete growth - slides.pptx" and uni == "Logs été growth - slides.pptx"
    cd = export.content_disposition("Logs: été")
    assert cd.startswith('attachment; filename="Logs ete - slides.pptx"; filename*=UTF-8\'\'Logs%20%C3%A9t%C3%A9')


# ---------- key points ----------
class FakeClient:
    def __init__(self, answer=None, fail=None, cost=0.0123):
        self.answer, self.fail, self.cost = answer, fail, cost
        self.calls, self.seen = [], []

    def chat(self, messages, **kw):
        self.seen.append((messages, kw))
        if self.fail:
            raise MetaAPIError(self.fail)
        self.calls.append(CallRecord("/chat/completions", "muse-spark-1.3", 200, 1.0, False, 1, {}, self.cost, "k"))
        return {"choices": [{"message": {"content": json.dumps(self.answer)}}], "usage": {}}


def three_pages(d):
    pieces = [piece(1, (100, 100, 50, 30), 10, kind="equation", text="A = P(1+r)^n", latex="A=P(1+r)^n",
                    spoken=[(11, 14)]),
              piece(2, (300, 100, 50, 30), 70, kind="graph", text="graph of growth"),
              piece(3, (500, 100, 50, 30), 130, text="decay")]
    r = board(pieces, [(0, "Intro"), (60, "Growth"), (120, "Decay")])
    return write_job(d, r), r


ANSWER = {"pages": [
    {"id": "p1", "title": "**Compound interest** formula $A=P(1+r)^{n}$.",
     "points": ["- Money grows by $\\frac{1}{12}$ each month [#1] [0:11]", "", "Growth is $x^2$ then \\infty",
                "a " * 80, "five", "six"]},
    {"id": "p9", "title": "Unknown page", "points": ["ignored"]},
    {"id": "p1", "title": "A second p1", "points": ["ignored too"]},
    {"id": "3", "title": "Decay", "points": ["Things shrink by a share each step."]},
]}


def test_key_points_call_shape_and_parsing(tmp_path):
    d, r = three_pages(tmp_path / "lec")
    fake = FakeClient(ANSWER)
    data = export.slides_data(d, lambda: fake)
    assert len(fake.seen) == 1
    messages, kw = fake.seen[0]
    # The first message is Ask Muse's own, with the same cache key: the lecture is cached once for both.
    assert messages[0] == {"role": "system", "content": ask.SYSTEM + "\n\n" + ask.lecture_context(r)}
    assert kw["extra"]["prompt_cache_key"] == ask.prepare(d, r, {"question": "q"}).cache_key
    assert kw["reasoning"] == "low" and kw["schema_name"] == "slide_points"
    assert kw["schema"]["properties"]["pages"]["items"]["properties"].keys() == {"id", "title", "points"}
    task = messages[1]["content"]
    assert messages[1]["role"] == "user" and "Page p1 | 0:00-1:00" in task and 'chapter "Growth"' in task
    assert "#1 equation $A=P(1+r)^n$" in task and "#2 graph \"graph of growth\" (a drawing)" in task
    assert 'Said: "w0.' in task
    # The transcript writes "E dollars" for e: symbols are spelled as the board writes them (gest-test).
    assert 'the number e is "e", not "E"' in task

    assert data["points_ready"] is True
    p1, p2, p3 = data["pages"]
    assert p1["title"] == "Compound interest formula A=P(1+r)ⁿ"
    assert p1["points"][0] == "Money grows by 1/12 each month"
    assert p1["points"][1] == "Growth is x² then ∞"
    assert len(p1["points"]) == 4 and len(p1["points"][2]) <= export.POINT_CHARS and p1["points"][2].endswith("…")
    assert (p2["title"], p2["points"]) == ("Growth", None)  # missing from the answer: the chapter, no points
    assert (p3["title"], p3["points"]) == ("Decay", ["Things shrink by a share each step."])

    saved = json.loads((d / "export" / "points.json").read_text(encoding="utf-8"))
    assert saved["version"] == data["version"] and saved["cost_usd"] == 0.0123 and set(saved["pages"]) == {"p1", "p3"}
    export.slides_data(d, lambda: pytest.fail("asked the model again for points it already has"))


def test_points_of_an_older_deck_are_not_used(tmp_path):
    d, r = three_pages(tmp_path / "lec")
    export._save_json(d / "export" / "points.json", {"version": "old", "pages": {"p1": {"title": "Old", "points": ["x"]}}})
    data = export.slides_data(d)
    assert data["points_ready"] is False and data["pages"][0]["title"] == "Intro" and data["pages"][0]["points"] is None


def test_an_answer_without_points_is_a_model_error(tmp_path):
    d, r = three_pages(tmp_path / "lec")
    with pytest.raises(MetaAPIError, match="no key points"):
        export.slides_data(d, lambda: FakeClient({"pages": [{"id": "p7", "title": "x", "points": []}]}))
    assert not (d / "export" / "points.json").exists()


@pytest.mark.parametrize("raw, plain", [
    ("$A = P\\left(1+\\frac{r}{n}\\right)^{nt}$", "A = P(1+r/n)^nt"),
    ("\\sqrt{2} \\approx 1.41 and \\pi r^2", "√2 ≈ 1.41 and π r²"),
    ("as $n \\to \\infty$, $(1+1/n)^n \\to e$", "as n → ∞, (1+1/n)ⁿ → e"),
    ("**Bold** and `code` [#L3] at [1:02]", "Bold and code at"),
    ("1. a numbered point", "a numbered point"),
    ("\\alpha = \\beta \\neq 90^{\\circ}", "α = β ≠ 90°"),
    ("$5 earns the same as five separate $1 accounts", "$5 earns the same as five separate $1 accounts"),
    ("$1 grows to $(1+1/12)^{12} \\approx$ $2.61", "$1 grows to (1+1/12)¹² ≈ $2.61"),
])
def test_plain_text_from_the_model(raw, plain):
    assert export.plain_text(raw) == plain


# ---------- the endpoints ----------
@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JOBS", tmp_path)
    three_pages(tmp_path / "lec")

    def use(fake):
        monkeypatch.setattr(server, "_meta_client", fake if callable(fake) else (lambda: fake))
        return fake

    return TestClient(server.app), tmp_path / "lec", use


def test_slides_json_and_its_images(api):
    client, d, use = api
    use(lambda: pytest.fail("slides.json must not call the model"))
    r = client.get("/api/jobs/lec/slides.json")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    data = r.json()
    assert data["type"] == "board" and data["title"] == "Growth and logs" and data["points_ready"] is False
    assert [p["id"] for p in data["pages"]] == ["p1", "p2", "p3"]
    p = data["pages"][0]
    assert set(p) == {"id", "t0", "t1", "chapter", "title", "image", "image_size", "items", "said", "points"}
    assert p["image"] == "export/page_001.png" and p["points"] is None and p["items"][0]["ref"] == "#1"
    img = client.get(f"/api/jobs/lec/files/{p['image']}")
    assert img.status_code == 200 and img.headers["cache-control"] == "no-cache"
    assert img.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_slides_need_a_processed_lecture(api):
    client, d, use = api
    (d.parent / "raw").mkdir()
    for method, url in (("get", "slides.json"), ("post", "slides/points"), ("get", "slides.pptx")):
        r = getattr(client, method)(f"/api/jobs/raw/{url}")
        assert r.status_code == 409 and "not processed yet" in r.json()["detail"] and "\n" not in r.json()["detail"]
    assert client.get("/api/jobs/nope/slides.json").status_code == 404


def test_key_points_endpoint(api):
    client, d, use = api
    fake = use(FakeClient(ANSWER))
    r = client.post("/api/jobs/lec/slides/points", headers={"origin": "https://evil.example"})
    assert r.status_code == 403 and fake.seen == []

    r = client.post("/api/jobs/lec/slides/points", headers={"origin": "http://localhost:5173"})
    assert r.status_code == 200
    data = r.json()
    assert data["points_ready"] is True and data["pages"][0]["points"][1] == "Growth is x² then ∞"
    assert client.get("/api/jobs/lec/slides.json").json() == data
    client.post("/api/jobs/lec/slides/points")
    assert len(fake.seen) == 1  # saved: not asked twice


def test_a_model_failure_is_one_plain_line_and_the_deck_still_works(api):
    client, d, use = api
    use(FakeClient(fail="/chat/completions (muse-spark-1.3) failed: HTTP 500: busy\nmore"))
    r = client.post("/api/jobs/lec/slides/points")
    assert r.status_code == 502
    assert r.json()["detail"] == "Muse Spark could not write the key points: /chat/completions (muse-spark-1.3) failed: HTTP 500: busy"

    def no_key():
        raise MetaAPIError("MODEL_API_KEY is missing: put it in .env (see .env.example).")

    use(no_key)
    r = client.post("/api/jobs/lec/slides/points")
    assert r.status_code == 502 and "MODEL_API_KEY is missing" in r.json()["detail"]
    data = client.get("/api/jobs/lec/slides.json").json()
    assert data["points_ready"] is False and len(data["pages"]) == 3


def test_an_unexpected_failure_is_one_line_with_a_log(api, monkeypatch):
    client, d, use = api
    monkeypatch.setattr(export, "build_deck", lambda *a: 1 / 0)
    r = client.get("/api/jobs/lec/slides.json")
    assert r.status_code == 500 and "ZeroDivisionError" in r.json()["detail"] and "\n" not in r.json()["detail"]
    assert (d / "export-error.log").exists()


def pptx_of(content):
    from pptx import Presentation

    return Presentation(io.BytesIO(content))


def test_powerpoint_without_key_points(api):
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    client, d, use = api
    r = client.get("/api/jobs/lec/slides.pptx")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    assert r.headers["content-disposition"].startswith('attachment; filename="Growth and logs - slides.pptx"')
    assert (d / "export" / "slides.pptx").read_bytes() == r.content
    prs = pptx_of(r.content)
    assert prs.slide_width / prs.slide_height == pytest.approx(16 / 9, abs=0.01)
    slides = list(prs.slides)
    assert len(slides) == 4
    assert slides[0].shapes.title.text == "Growth and logs"
    texts = " ".join(sh.text_frame.text for sh in slides[0].shapes if sh.has_text_frame)
    assert "Made with Replay Board from the recorded lecture" in texts and "3:00" in texts
    assert [s.shapes.title.text for s in slides[1:]] == ["Intro", "Growth", "Decay"]
    pics = [sh for sh in slides[1].shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE]
    assert len(pics) == 1
    notes = slides[1].notes_slide.notes_text_frame.text
    assert notes.startswith("0:00–1:00 (Intro)") and "Key points" not in notes and "What was said:\nw0." in notes
    foot = " ".join(sh.text_frame.text for sh in slides[2].shapes if sh.has_text_frame)
    assert "1:00–2:00" in foot and "2 / 3" in foot


def test_powerpoint_with_key_points(api):
    from pptx.oxml.ns import qn

    client, d, use = api
    use(FakeClient(ANSWER))
    assert client.post("/api/jobs/lec/slides/points").status_code == 200
    slides = list(pptx_of(client.get("/api/jobs/lec/slides.pptx").content).slides)
    assert [s.shapes.title.text for s in slides[1:]] == ["Compound interest formula A=P(1+r)ⁿ", "Growth", "Decay"]
    box = next(sh for sh in slides[1].shapes if sh.has_text_frame and "Money grows" in sh.text_frame.text)
    paras = box.text_frame.paragraphs
    assert [p.text for p in paras][:2] == ["Money grows by 1/12 each month", "Growth is x² then ∞"]
    assert all(p._p.pPr.find(qn("a:buChar")) is not None for p in paras)  # real bullets
    notes = slides[1].notes_slide.notes_text_frame.text
    assert "Key points:\n- Money grows by 1/12 each month\n- Growth is x² then ∞" in notes
    assert "What was said:" in notes
    assert "Key points" not in slides[2].notes_slide.notes_text_frame.text


# ---------- review fixes ----------
def ink_pixels(img):
    return int((np.abs(img.astype(np.float32) - INK).max(axis=2) <= 6).sum())


def test_a_wipe_at_the_end_of_the_lecture_keeps_the_board_before_it(tmp_path):
    # The teacher wipes the whole board 10 s before the end: the last page must not show the clean board.
    pieces = [piece(1, (100, 100, 200, 100), 10, removed_at=171.0), piece(2, (400, 100, 200, 100), 60, removed_at=171.2)]
    r = board(pieces, [(0, "Intro")], erases=[(170.5, (0, 0, 800, 400))])
    d = write_job(tmp_path / "lec", r)
    (s,) = export.board_segments(r, export.Pixels(d))
    assert (s["t0"], s["t1"]) == (0, 180) and s["before"] == pytest.approx(170.45)
    (page,) = export.build_deck(d, r, "v1")["pages"]
    img = imread(d / page["image"], cv2.IMREAD_COLOR)
    assert ink_pixels(img) > 0.5 * img.shape[0] * img.shape[1] * (2 * 200 * 100) / (548 * 308)

    # The same with a little writing after the wipe (a short last stretch): still the board before the wipe.
    r["board"]["pieces"].append(piece(3, (300, 300, 40, 20), 174))
    write_job(d, r)
    (s,) = export.board_segments(r, export.Pixels(d))
    assert s["before"] == pytest.approx(170.45)
    (page,) = export.build_deck(d, r, "v2")["pages"]
    assert [i["ref"] for i in page["items"]] == ["#1", "#2", "#3"]

    # A wipe in the middle still ends its page, and the page after it shows the board at the end.
    r2 = board([piece(1, (100, 100, 200, 100), 10, removed_at=90.5), piece(2, (400, 100, 200, 100), 100)],
               [(0, "Intro")], erases=[(90, (0, 0, 800, 400))])
    d2 = write_job(tmp_path / "mid", r2)
    assert [(x["t0"], x["t1"], x["before"]) for x in export.board_segments(r2, export.Pixels(d2))] == [
        (0, 90, pytest.approx(89.95)), (90, 180, None)]


def test_a_line_started_before_the_page_seeks_to_its_part_drawn_in_the_page(tmp_path):
    pieces = [old(1, (100, 100, 50, 30)), piece(2, (160, 100, 50, 30), 70)]
    lines = [{"id": "L1", "pieces": [1, 2], "box": [100, 100, 110, 30], "t_complete": 71,
              "read": {"kind": "text", "text": "a b", "latex": ""}}]
    r = board(pieces, [(0, "Intro")], lines=lines)
    d = write_job(tmp_path / "lec", r)
    (page,) = export.build_deck(d, r, "v1")["pages"]
    assert [(i["ref"], i["t"]) for i in page["items"]] == [("#L1", 70.0)]  # not the clip start (0)
    r["board"]["pieces"][0]["spoken"] = [{"start": 30.0, "end": 34.0}]  # talked about: that time comes first
    (page,) = export.build_deck(d, r, "v2")["pages"]
    assert page["items"][0]["t"] == 30.0


def test_odd_boxes_and_readings_do_not_break_the_deck(tmp_path):
    pieces = [piece(1, (100, 100, 50, 30), 10), piece(2, (300, 100, 50, 30), 20)]
    r = board(pieces, [(0, "Intro")])
    d = write_job(tmp_path / "lec", r)
    r["board"]["pieces"].append({**piece(3, (0, 0, 1, 1), 30), "box": [None, 5, 10, 10]})
    r["board"]["pieces"][1]["read"] = "not an object"
    r["board"]["lines"] = [{"id": "L1", "pieces": [1, [2], {"x": 1}], "box": [None, 0, 10, 10], "read": {}},
                           {"id": "L2", "pieces": [1, "2"], "box": [100, 100, 50, 30],
                            "read": {"kind": "text", "text": "one", "latex": ""}}]
    (page,) = export.build_deck(d, r, "v1")["pages"]
    assert [i["ref"] for i in page["items"]] == ["#L2"]  # #3 has no box; #2 has no reading

    # A line the model could not read: its pieces are listed one by one, not dropped with it.
    r["board"]["pieces"][1]["read"] = {"kind": "text", "text": "two", "latex": ""}
    r["board"]["lines"] = [{"id": "L3", "pieces": [1, 2], "box": [100, 100, 250, 30],
                            "read": {"kind": "text", "text": "unreadable", "latex": ""}}]
    (page,) = export.build_deck(d, r, "v2")["pages"]
    assert [(i["ref"], i["text"]) for i in page["items"]] == [("#1", "item 1"), ("#2", "two")]


def test_one_chapter_and_no_speech(tmp_path):
    r = board([piece(1, (100, 100, 50, 30), 10), piece(2, (300, 100, 50, 30), 90)], [(0, "Only")], words=[])
    d = write_job(tmp_path / "lec", r)
    deck, _ = export.load_deck(d)
    (page,) = deck["pages"]
    assert (page["t0"], page["t1"], page["title"], page["said"]) == (0, 180, "Only", "")
    messages, _ = export.points_messages(d, r, deck)
    assert "Said: nothing" in messages[1]["content"] and "(no speech)" in messages[0]["content"]
    blob, _ = export.write_pptx(d)
    notes = list(pptx_of(blob).slides)[1].notes_slide.notes_text_frame.text
    assert notes.endswith("What was said:\n(nothing)")


def test_a_stale_page_image_that_cannot_be_deleted_does_not_fail_the_deck(tmp_path, monkeypatch):
    r = board([piece(1, (100, 100, 50, 30), 10), piece(2, (300, 100, 50, 30), 70)], [(0, "Intro"), (60, "Growth")])
    d = write_job(tmp_path / "lec", r)
    assert len(export.load_deck(d)[0]["pages"]) == 2
    r["chapters"] = [{"t": 0, "title": "Intro"}]
    (d / "replay.json").write_text(json.dumps(r), encoding="utf-8")
    real = type(d).unlink

    def busy(self, *a, **kw):  # Windows: the player is still downloading the old page
        if self.name == "page_002.png":
            raise PermissionError(13, "The process cannot access the file", str(self))
        return real(self, *a, **kw)

    monkeypatch.setattr(type(d), "unlink", busy)
    deck, _ = export.load_deck(d)
    assert len(deck["pages"]) == 1 and (d / "export/page_002.png").exists()


def test_control_characters_and_lone_surrogates_are_dropped_everywhere(api):
    client, d, use = api
    r = json.loads((d / "replay.json").read_text(encoding="utf-8"))
    # Readings can hold control characters (JSON escapes them); a model's answer can also hold a broken emoji escape.
    r["subject"] = "growth\x00 and logs"
    r["chapters"][0]["title"] = "Intro\x01 part"
    r["transcript"]["words"][0]["w"] = "w0.\x07"
    (d / "replay.json").write_text(json.dumps(r), encoding="utf-8")
    answer = {"pages": [{"id": "p1", "title": "Growth\x02 \udc00rate", "points": ["Bad\x0b char \ud83d here"]}]}
    use(FakeClient(answer))
    res = client.post("/api/jobs/lec/slides/points")
    assert res.status_code == 200
    data = res.json()
    assert data["title"] == "Growth and logs" and data["pages"][0]["chapter"] == "Intro part"
    assert data["pages"][0]["title"] == "Growth rate" and data["pages"][0]["points"] == ["Bad char here"]
    assert data["pages"][0]["said"].startswith("w0. w1.")
    assert data["pages"][1]["title"] == "Growth"
    pptx = client.get("/api/jobs/lec/slides.pptx")
    assert pptx.status_code == 200 and pptx_of(pptx.content).slides[1].shapes.title.text == "Growth rate"
    assert export.safe_text("a\x00b\tc\n\U0001f600￾") == "ab\tc\n\U0001f600"


def test_a_slide_lecture_with_nothing_to_show(tmp_path, monkeypatch):
    r = slides_replay()
    r["slides"]["states"] = [{**st, "blank": True} for st in r["slides"]["states"]]
    d = write_job(tmp_path / "lec", r)
    monkeypatch.setattr(server, "JOBS", tmp_path)
    monkeypatch.setattr(server, "_meta_client", lambda: pytest.fail("no pages: nothing for Muse Spark to write"))
    client = TestClient(server.app)
    res = client.post("/api/jobs/lec/slides/points")
    assert res.status_code == 200 and res.json()["pages"] == []
    pptx = client.get("/api/jobs/lec/slides.pptx")
    assert pptx.status_code == 200 and len(pptx_of(pptx.content).slides) == 1  # the title slide


def test_a_slide_without_an_image_has_no_picture(tmp_path, monkeypatch):
    r = slides_replay()
    d = write_job(tmp_path / "lec", r, slide_size=(1920, 1080))
    for name in ("slides/001.png", "slides/002.png"):  # the second page's states lost their images
        (d / name).unlink()
    deck, _ = export.load_deck(d)
    p1, p2, p3 = deck["pages"]
    assert (p2["image"], p2["image_size"]) == ("", None) and p1["image"] == "export/page_001.png"
    monkeypatch.setattr(export, "build_deck", lambda *a: pytest.fail("rebuilt a deck that was up to date"))
    assert export.load_deck(d)[0] == deck

    export._save_json(d / "export" / "points.json",
                      {"version": deck["version"], "pages": {"p2": {"title": "Interest", "points": ["One", "Two"]}}})
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    blob, _ = export.write_pptx(d)
    slide = list(pptx_of(blob).slides)[2]
    assert not [sh for sh in slide.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE]
    assert any(sh.has_text_frame and sh.text_frame.text == "One\nTwo" for sh in slide.shapes)


def test_a_powerpoint_open_in_powerpoint_does_not_hold_up_the_download(tmp_path, monkeypatch):
    r = board([piece(1, (100, 100, 50, 30), 10)], [(0, "Intro")])
    d = write_job(tmp_path / "lec", r)
    export.load_deck(d)
    tries = []

    def locked(tmp, path, n=40):  # Windows: export/slides.pptx is open in PowerPoint
        tries.append(n)
        return False

    monkeypatch.setattr(export, "replace_file", locked)
    blob, data = export.write_pptx(d)
    assert len(pptx_of(blob).slides) == 2 and tries == [2]
    assert not list((d / "export").glob("*.tmp"))


def test_a_deck_json_pointing_outside_its_pages_is_rebuilt(tmp_path):
    r = board([piece(1, (100, 100, 50, 30), 10)], [(0, "Intro")])
    d = write_job(tmp_path / "lec", r)
    deck, _ = export.load_deck(d)
    deck["pages"][0]["image"] = "replay.json"
    export._save_json(d / "export" / "deck.json", deck)
    again, _ = export.load_deck(d)
    assert again["pages"][0]["image"] == "export/page_001.png"


def test_key_points_are_sized_to_fit_their_box(api):
    from pptx.util import Emu

    client, d, use = api
    long = ["Monthly compounding of one dollar at one hundred percent gives about 2.61 by the end of the year!"] * 4
    assert export.points_size(long, 12.33, 1.0, 16) < 16
    assert export.points_size(["short"], 4.8, 5.5, 18) == 18
    r = json.loads((d / "replay.json").read_text(encoding="utf-8"))
    r["board"]["size"] = [800, 314]  # a wide board (2.5:1): the whole-board picture goes above the points
    for p in r["board"]["pieces"]:
        p["box"][1] = 50
    (d / "replay.json").write_text(json.dumps(r), encoding="utf-8")
    use(FakeClient({"pages": [{"id": f"p{i}", "title": "T", "points": long} for i in (1, 2, 3)]}))
    assert client.post("/api/jobs/lec/slides/points").status_code == 200
    prs = pptx_of(client.get("/api/jobs/lec/slides.pptx").content)
    footer_top = prs.slide_height - Emu(int(0.55 * 914400))
    for slide in list(prs.slides)[1:]:
        box = next(sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.startswith("Monthly"))
        size = box.text_frame.paragraphs[0].runs[0].font.size.pt
        assert box.top + box.height <= footer_top
        need = export.points_height(long, box.width / 914400, size)
        assert need <= box.height / 914400 + 1e-6

# ---------- stray marks (end-to-end review, real lectures) ----------
def test_a_stray_mark_does_not_make_the_slide_the_whole_board(tmp_path):
    # test-full: two small arrows from an old lecture "appear" as the camera settles (never talked about), far from
    # the formulas the page is about. They must not widen the crop to the whole board or clutter the items.
    pieces = [
        old(1, (20, 20, 300, 300)),  # the old lecture, faded
        piece(2, (40, 250, 12, 30), 2.3, kind="arrow", text="downward arrow"),
        piece(3, (600, 30, 70, 25), 24, kind="equation", text="A=P(1+r)^n", latex="A=P(1+r)^n", spoken=[(23, 31)]),
        piece(4, (730, 40, 40, 20), 46, kind="equation", text="x^300", latex="x^{300}", spoken=[(44, 52)]),
        piece(5, (690, 70, 20, 8), 47, kind="arrow", text="small arrow"),  # on the picture: listed, at full strength
        piece(6, (200, 330, 120, 60), 50, kind="other", text="person obscuring board, no legible writing"),
    ]
    r = board(pieces, [(0, "Intro")], end=60.0)
    d = write_job(tmp_path / "lec", r)
    (page,) = export.build_deck(d, r, "v1")["pages"]
    assert [i["ref"] for i in page["items"]] == ["#3", "#4", "#5"]
    assert page["items"][2]["drawing"] is True
    w, h = page["image_size"]
    assert w / h == pytest.approx(16 / 9, abs=0.03) and w <= 3 * 0.3 * 800 + 3  # a crop, not the whole board
    crop = export.crop_box([[600, 30, 70, 25], [730, 40, 40, 20]], 800, 400)
    img = imread(d / page["image"], cv2.IMREAD_COLOR).astype(np.float32)
    k = w / crop[2]
    assert np.abs(img[int((74 - crop[1]) * k), int((700 - crop[0]) * k)] - INK).max() <= 6  # piece 5, inside

    # Talked about in the page, the arrow is part of it again: listed, and the slide shows it (the whole board).
    r["board"]["pieces"][1]["spoken"] = [{"start": 2.0, "end": 6.0}]
    (page,) = export.build_deck(d, r, "v2")["pages"]
    assert [i["ref"] for i in page["items"]] == ["#3", "#4", "#5", "#2"] and page["image_size"] == [1600, 800]

    # A big drawing nobody talked about is not a stray mark: it still decides the crop.
    r["board"]["pieces"][1] = piece(2, (40, 150, 200, 200), 2.3, kind="graph", text="graph of growth")
    write_job(d, r)
    (page,) = export.build_deck(d, r, "v3")["pages"]
    assert "#2" in [i["ref"] for i in page["items"]] and page["image_size"] == [1600, 800]


def test_a_vague_reading_is_a_mark_not_writing():
    item = export._item("#9", {"kind": "other", "text": "small red marks", "latex": ""}, 3.0)
    assert item["drawing"] is True  # named, never typeset as if it were the writing
    assert export._item("#9", {"kind": "other", "text": "x+1", "latex": "x+1"}, 3.0)["drawing"] is False
    assert export._item("#9", {"kind": "other", "text": "", "latex": ""}, 3.0) is None
    assert export._item("#9", {"kind": "arrow", "text": "", "latex": ""}, 3.0)["drawing"] is True
