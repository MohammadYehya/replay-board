"""Ask Muse (app/ask.py and its endpoints), offline: the lecture context, attachments and their images,
the answer's citations, the event stream and the saved conversation. The model is a fake."""
import base64
import json
import math
import warnings

import cv2
import numpy as np
import pytest

with warnings.catch_warnings():  # starlette's notice about httpx, at import time
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app import ask, server
from app.core.media import imwrite
from app.core.meta_api import CallRecord, MetaAPIError

BG = "#f0f0e8"
BG_BGR = (0xe8, 0xf0, 0xf0)
INK = (200, 40, 10)


def ink(w, h):
    """A BGRA piece: ink in the middle, transparent around it."""
    img = np.zeros((h, w, 4), np.uint8)
    img[2:-2, 2:-2] = (*INK, 255)
    return img


def board_replay(start=0.0, end=60.0, duration=120.0, size=(400, 200)):
    def piece(i, box, t_seen, read, spoken=(), **kw):
        return {"id": i, "box": list(box), "image": f"pieces/{i:03d}.png", "t_seen": t_seen,
                "t_shown": kw.pop("t_shown", t_seen), "written_after": t_seen, "removed_at": None,
                "pre_existing": False, "read": read, "spoken": [{"start": a, "end": b} for a, b in spoken],
                "redraw_at": None, **kw}

    words = [{"w": f"w{i}", "s": start + i * 0.5, "e": start + i * 0.5 + 0.4} for i in range(int((end - start) * 2))]
    words[8]["w"] = "logarithm"
    return {
        "version": 1, "type": "board", "subject": "logs and growth",
        "video": {"file": "lecture.mp4", "duration": duration, "width": 1280, "height": 720, "start": start, "end": end},
        "chapters": [{"t": start, "title": "Intro"}, {"t": start + 30, "title": "Graphs"}],
        "transcript": {"turns": [], "words": words, "timing": "mms"},
        "board": {
            "size": list(size), "background": BG,
            "pieces": [
                piece(1, (10, 10, 40, 20), start, {"kind": "equation", "text": "y = ln x", "latex": "y=\\ln x"},
                      pre_existing=True),
                piece(2, (60, 10, 30, 20), start + 5, {"kind": "text", "text": "1 +", "latex": ""}, [(start + 4, start + 6)],
                      t_shown=start + 5.5),
                piece(3, (95, 12, 30, 18), start + 8, {"kind": "text", "text": "1 = 2", "latex": ""},
                      [(start + 8.2, start + 9)]),
                piece(4, (10, 60, 50, 30), start + 20, {"kind": "graph", "text": "graph of e^x", "latex": ""},
                      [(start + 20.5, start + 25)], removed_at=start + 30),
                piece(5, (200, 100, 40, 40), start + 50, {"kind": "equation", "text": "x^2", "latex": "x^2"},
                      revealed=True, redraw_at=start + 45),
            ],
            "lines": [{"id": "L1", "pieces": [2, 3], "box": [60, 10, 65, 22], "t_complete": start + 9,
                       "removed_at": None, "read": {"kind": "equation", "text": "1 + 1 = 2", "latex": "1+1=2"}}],
            "erases": [{"t": start + 30, "box": [10, 60, 50, 30]}], "annotations": [],
        },
    }


def slides_replay():
    def el(i, kind, text, box, latex="", spoken=()):
        return {"id": i, "kind": kind, "text": text, "latex": latex, "box": list(box),
                "spoken": [{"start": a, "end": b} for a, b in spoken]}

    return {
        "version": 1, "type": "slides", "subject": "compound interest",
        "video": {"file": "lecture.mp4", "duration": 30.0, "width": 960, "height": 540, "start": 0.0, "end": 30.0},
        "chapters": [{"t": 0.0, "title": "Interest"}],
        "transcript": {"words": [{"w": "interest", "s": 1.0, "e": 1.5}, {"w": "grows", "s": 12.5, "e": 13.0}]},
        "slides": {"size": [960, 540], "states": [
            {"id": 0, "kind": "slide", "t_start": 0.0, "t_end": 10.0, "image": "slides/000.png", "elements": [
                el("s0e1", "title", "Compound interest", (10, 10, 200, 30), spoken=[(1, 3)]),
                el("s0e2", "bullet", "Interest on interest", (10, 60, 300, 20))]},
            {"id": 1, "kind": "addition", "t_start": 10.0, "t_end": 30.0, "image": "slides/001.png", "elements": [
                el("s1e1", "title", "Compound interest", (10, 10, 200, 30)),
                el("s1e2", "bullet", "Interest on interest", (10, 60, 300, 20)),
                el("s1e3", "equation", "A = P(1+r)^n", (10, 100, 200, 40), "A=P(1+r)^n", [(12, 15)])]},
        ], "annotations": []},
    }


def write_job(d, replay):
    d.mkdir(parents=True, exist_ok=True)
    (d / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    if replay["type"] == "board":
        (d / "pieces").mkdir(exist_ok=True)
        for p in replay["board"]["pieces"]:
            imwrite(d / p["image"], ink(p["box"][2], p["box"][3]))
    else:
        (d / "slides").mkdir(exist_ok=True)
        for st in replay["slides"]["states"]:
            img = np.full((540, 960, 3), 255, np.uint8)
            cv2.putText(img, f"slide {st['id']}", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
            imwrite(d / st["image"], img)
    return d


def images(messages):
    """The decoded images in the last user message."""
    content = messages[-1]["content"]
    if isinstance(content, str):
        return []
    out = []
    for part in content:
        if part["type"] == "image_url":
            data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            out.append(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
    return out


def last_text(messages):
    content = messages[-1]["content"]
    return content if isinstance(content, str) else "\n".join(p["text"] for p in content if p["type"] == "text")


# ---------- the lecture context ----------
def test_context_is_deterministic_and_lists_items_and_transcript():
    r = board_replay()
    ctx = ask.lecture_context(r)
    assert ctx == ask.lecture_context(json.loads(json.dumps(r)))
    assert "Subject: logs and growth" in ctx and "Video: 2:00 long" in ctx
    assert "[0:00] Intro" in ctx and "[0:30] Graphs" in ctx
    assert "#L1 equation $1+1=2$ | written 0:05-0:09 | talked about 0:04-0:06, 0:08-0:09" in ctx
    assert "\n#2 " not in ctx and "\n#3 " not in ctx  # in line L1, so listed as the line
    assert "#1 equation $y=\\ln x$ | on the board from the start" in ctx
    assert '#4 graph "graph of e^x" | written 0:20 | erased 0:30 | talked about 0:20-0:25' in ctx
    assert "#5 equation $x^2$ | written by 0:50, hidden behind the teacher until then" in ctx
    assert ctx.index("#1 ") < ctx.index("#L1 ") < ctx.index("#4 ") < ctx.index("#5 ")  # in time order
    transcript = ctx.split("# Transcript\n", 1)[1].splitlines()
    assert transcript[0].startswith("[0:00] w0 w1") and len(transcript[0].split()) == 13  # 12 words a line
    assert "logarithm" in ctx and "Only 0:00 to 1:00 of the video was analysed" in ctx  # a clip of 0-60 s
    assert "Only" not in ask.lecture_context(board_replay(end=120.0))  # the whole video
    r["board"]["pieces"][2]["revealed"] = True  # #3, the second half of line L1
    assert "#L1 equation $1+1=2$ | written 0:05-0:09 | partly written while hidden behind the teacher" in         ask.lecture_context(r)


def test_context_breaks_transcript_on_pauses():
    words = [{"w": "a", "s": 0.0, "e": 0.3}, {"w": "b", "s": 0.4, "e": 0.6}, {"w": "c", "s": 5.0, "e": 5.2}]
    assert ask.speech_lines(words) == ["[0:00] a b", "[0:05] c"]
    assert ask.speech_lines(words, 0.35, 10) == ["[0:00] b", "[0:05] c"]


def test_context_for_slides_names_repeats_from_build_steps():
    ctx = ask.lecture_context(slides_replay())
    assert "Slide state 0, on screen 0:00-0:10:" in ctx
    assert "Slide state 1 (adds to the one before), on screen 0:10-0:30:" in ctx
    assert '#s0e1 title "Compound interest" | talked about 0:01-0:03' in ctx
    assert "#s1e1 same as #s0e1" in ctx and "#s1e2 same as #s0e2" in ctx
    assert "#s1e3 equation $A=P(1+r)^n$ | talked about 0:12-0:15" in ctx
    assert "[0:01] interest" in ctx


def test_context_of_a_clip_says_which_part_was_analysed():
    ctx = ask.lecture_context(board_replay(start=100.0, end=160.0, duration=200.0))
    assert "Only 1:40 to 2:40 of the video was analysed" in ctx
    assert "#1 equation $y=\\ln x$ | on the board from the start (1:40)" in ctx
    assert "[1:40] w0" in ctx  # the video's clock, not the clip's


def test_an_hour_long_lecture_stays_compact():
    rng = np.random.default_rng(0)
    words = [{"w": "word" + str(i % 7), "s": i * 0.4, "e": i * 0.4 + 0.3} for i in range(9000)]
    pieces = [{"id": i, "box": [0, 0, 10, 10], "image": "", "t_seen": i * 9.0, "t_shown": i * 9.0,
               "read": {"kind": "equation", "text": "f(x) = a x^2 + b x + c", "latex": "f(x)=ax^2+bx+c"},
               "spoken": [{"start": float(s), "end": float(s) + 5} for s in rng.uniform(0, 3600, 2)]}
              for i in range(400)]
    r = {"type": "board", "video": {"duration": 3600.0}, "transcript": {"words": words},
         "chapters": [{"t": i * 300.0, "title": f"Part {i}"} for i in range(12)],
         "board": {"size": [800, 400], "pieces": pieces}}
    ctx = ask.lecture_context(r)
    assert len(ctx) / 3.5 < 60_000  # well under 60k tokens


# ---------- the request ----------
@pytest.mark.parametrize("body,msg", [
    ([], "JSON object"),
    ({}, "type a question"),
    ({"question": "   "}, "type a question"),
    ({"question": "x" * 2001}, "too long"),
    ({"question": "q", "selection": {"kind": "piece"}}, "list"),
    ({"question": "q", "selection": [{"kind": "moment", "t": 1}] * 13}, "at most 12"),
    ({"question": "q", "selection": [{"kind": "slide", "id": 1}]}, "unknown kind"),
    ({"question": "q", "selection": [{"kind": "piece", "id": "L3"}]}, "number id"),
    ({"question": "q", "selection": [{"kind": "piece", "id": True}]}, "number id"),
    ({"question": "q", "selection": [{"kind": "line"}]}, "needs an id"),
    ({"question": "q", "selection": [{"kind": "span", "t0": 1}]}, "t0 and t1"),
    ({"question": "q", "selection": [{"kind": "moment", "t": float("nan")}]}, "needs t"),
    ({"question": "q", "selection": ["piece"]}, "object"),
    ({"question": "q", "history": "none"}, "history"),
    ({"question": "q", "history": [{"q": "a"}]}, "history entry"),
    ({"question": "q", "t": "now"}, "seconds"),
])
def test_bad_requests_are_one_plain_line(body, msg):
    with pytest.raises(ask.AskError) as e:
        ask.check_request(body)
    assert msg in str(e.value) and "\n" not in str(e.value)


def test_request_is_tidied():
    req = ask.check_request({
        "question": "  why?  ", "t": 12, "selection": [{"kind": "span", "t0": 9, "t1": 3, "label": "  so the  "},
                                                       {"kind": "piece", "id": "4"}],
        "history": [{"q": f"q{i}", "a": "a" * 5000} for i in range(9)]})
    assert req["question"] == "why?" and req["t"] == 12.0
    assert req["selection"] == [{"kind": "span", "t0": 3.0, "t1": 9.0, "label": "so the"}, {"kind": "piece", "id": 4}]
    assert [h["q"] for h in req["history"]] == ["q3", "q4", "q5", "q6", "q7", "q8"]
    assert all(len(h["a"]) == 2000 for h in req["history"])


# ---------- attachments ----------
def test_each_kind_of_attachment(tmp_path):
    r = board_replay()
    d = write_job(tmp_path / "job", r)
    sel = ask.check_request({"question": "q", "selection": [
        {"kind": "piece", "id": 4}, {"kind": "line", "id": "L1"}, {"kind": "moment", "t": 25.0},
        {"kind": "span", "t0": 4.0, "t1": 5.0}, {"kind": "piece", "id": 99}, {"kind": "line", "id": "L9"},
        {"kind": "element", "id": "s0e1"}, {"kind": "line", "id": "4"}, {"kind": "piece", "id": 4}]})["selection"]
    atts = ask.resolve_selection(d, r, sel)
    assert [a.item["kind"] for a in atts] == ["piece", "line", "moment", "span"]  # unknown ids and repeats dropped
    piece, line, moment, span = atts
    assert piece.item == {"kind": "piece", "id": 4, "label": "graph of e^x", "t": 20.5}
    assert "Piece of writing #4: graph \"graph of e^x\" | written 0:20 | erased 0:30" in piece.text
    assert 'Said then (0:20): "w41' in piece.text
    assert line.item == {"kind": "line", "id": "L1", "label": "1 + 1 = 2", "t": 4.0}
    assert "made of pieces #2, #3" in line.text
    assert moment.item["t"] == 25.0 and moment.item["label"] == "The board at 0:25"
    assert "On the board then: #L1, #1, #4." in moment.text  # #5 is not written yet
    assert "Speech from 0:05 to 0:35" in moment.text
    assert span.item == {"kind": "span", "t0": 4.0, "t1": 5.0, "label": "logarithm w9 w10", "t": 4.0}
    assert "[0:04] logarithm w9 w10" in span.text and span.image is None


def test_piece_and_line_images_are_drawn_on_the_board_colour(tmp_path):
    r = board_replay()
    d = write_job(tmp_path / "job", r)
    msgs = ask.prepare(d, r, {"question": "q", "selection": [{"kind": "piece", "id": 1},
                                                             {"kind": "line", "id": "L1"}]}).messages
    piece, line = images(msgs)
    # piece 1 is 40x20 plus 10 px padding, grown 4x for the model
    assert piece.shape[:2] == (160, 240)
    assert tuple(int(v) for v in piece[2, 2]) == BG_BGR
    assert np.abs(piece[80, 120].astype(int) - INK).max() <= 2
    # line L1: pieces 2 and 3 side by side with a gap between them
    inked = (np.abs(line.astype(int) - INK).max(axis=2) < 40).any(axis=0)
    runs = np.flatnonzero(np.diff(inked.astype(int)) != 0)
    assert len(runs) == 4  # two separate runs of ink


def test_moment_image_is_the_clean_board_at_that_time(tmp_path):
    r = board_replay()
    d = write_job(tmp_path / "job", r)

    def board_at(t):
        msgs = ask.prepare(d, r, {"question": "q", "selection": [{"kind": "moment", "t": t}]}).messages
        (img,) = images(msgs)
        assert img.shape[:2] == (512, 1024)  # 400x200 grown to the 1024 limit
        return lambda box: (np.abs(img[int(box[1] * 2.56) + 10, int(box[0] * 2.56) + 10].astype(int) - INK).max() < 40)

    at25 = board_at(25.0)
    assert at25((10, 10)) and at25((60, 10)) and at25((10, 60)) and not at25((200, 100))
    at55 = board_at(55.0)
    assert not at55((10, 60)) and at55((200, 100))  # #4 erased at 30, #5 shown at 50
    assert {p["id"] for p in ask.visible_pieces(r, 5.2)} == {1}  # #2 is seen at 5 but shown at 5.5


def test_at_most_six_images_and_none_over_1024(tmp_path):
    r = board_replay(size=(3000, 1500))
    d = write_job(tmp_path / "job", r)
    sel = ([{"kind": "piece", "id": i} for i in range(1, 6)] + [{"kind": "line", "id": "L1"}]
           + [{"kind": "moment", "t": t} for t in (1.0, 2.0, 3.0)])
    msgs = ask.prepare(d, r, {"question": "q", "selection": sel}).messages
    imgs = images(msgs)
    assert len(imgs) == 6 and all(max(i.shape[:2]) <= 1024 for i in imgs)
    text = last_text(msgs)
    assert "9. The board at [0:03]" in text and text.count("(image ") == 6
    # a moment past the limit still gets its text
    r2 = board_replay(size=(3000, 1500))
    msgs = ask.prepare(d, r2, {"question": "q", "selection": [{"kind": "moment", "t": 1.0}]}).messages
    assert images(msgs)[0].shape[:2] == (512, 1024)


def test_slide_element_and_moment(tmp_path):
    r = slides_replay()
    d = write_job(tmp_path / "job", r)
    p = ask.prepare(d, r, {"question": "q", "selection": [{"kind": "element", "id": "s1e3"},
                                                          {"kind": "moment", "t": 4.0},
                                                          {"kind": "piece", "id": 3}]})
    assert [s["kind"] for s in p.selection] == ["element", "moment"]
    assert p.selection[0] == {"kind": "element", "id": "s1e3", "label": "A = P(1+r)^n", "t": 12.0}
    crop, slide = images(p.messages)
    assert crop.shape[:2] == (163, 512)  # the 200x40 box padded by 16 px (10 at the left edge), grown to 512
    assert slide.shape[:2] == (540, 960)
    text = last_text(p.messages)
    assert "Slide element #s1e3 on slide state 1 (on screen 0:10-0:30): equation $A=P(1+r)^n$" in text
    assert "It shows slide state 0, with elements #s0e1 #s0e2." in text


def test_slide_crop_follows_an_image_of_another_size(tmp_path):
    r = slides_replay()  # boxes are in 960x540
    d = write_job(tmp_path / "job", r)
    big = np.full((1080, 1920, 3), 255, np.uint8)
    big[200:280, 20:420] = (0, 0, 255)  # element s1e3's box (10, 100, 200, 40), at twice the size
    imwrite(d / "slides/001.png", big)
    (crop,) = images(ask.prepare(d, r, {"question": "q", "selection": [{"kind": "element", "id": "s1e3"}]}).messages)
    assert crop.shape[:2] == (163, 512)  # the same framing as at 1x
    red = (crop[..., 2] > 200) & (crop[..., 0] < 60)
    assert red[60:100, 40:470].mean() > 0.95 and not red[:20].any() and not red[-20:].any()


def test_a_clip_seeks_on_the_videos_clock(tmp_path):
    r = board_replay(start=100.0, end=160.0, duration=200.0)
    d = write_job(tmp_path / "job", r)
    p = ask.prepare(d, r, {"question": "q", "t": 130.0, "selection": [{"kind": "piece", "id": 1},
                                                                       {"kind": "moment", "t": 125.0}]})
    assert p.selection[0]["t"] == 100.0  # pre-existing, never talked about: the clip's start
    assert "The student is at [2:10] in the video." in last_text(p.messages)
    assert "Speech from 1:45 to 2:15" in last_text(p.messages)
    items, times = ask.answer_refs("[#5] at [1:50], [0:30] and [3:30]", r)
    assert items[0]["t"] == 145.0 and times == [110.0, 30.0]  # 3:30 is past the video's end


def test_messages_put_the_lecture_first_and_history_after(tmp_path):
    r = board_replay()
    d = write_job(tmp_path / "job", r)
    hist = [{"q": f"q{i}", "a": f"a{i}"} for i in range(8)]
    p = ask.prepare(d, r, {"question": "What next?", "history": hist})
    m = p.messages
    assert m[0]["role"] == "system" and m[0]["content"].startswith(ask.SYSTEM) and "# Transcript" in m[0]["content"]
    assert [x["content"] for x in m[1:13:2]] == ["q2", "q3", "q4", "q5", "q6", "q7"]
    assert m[2]["role"] == "assistant" and m[-1] == {"role": "user", "content": "What next?"}
    # the cache key is the same for every question about this lecture
    p2 = ask.prepare(d, r, {"question": "Other", "t": 3})
    assert p.cache_key == p2.cache_key and p.cache_key.startswith("replay-board-job-")
    assert p2.messages[0] == m[0]


# ---------- citations ----------
def test_answer_refs_keep_only_real_items_and_times():
    r = board_replay()
    answer = ("See [#1] and [#L1], not [#99], [#L9] or [#s0e1]. At [0:05], [1:00:05], [9:99], [2:01] "
              "and [0:05] again, with [#1] again and [#4].")
    items, times = ask.answer_refs(answer, r)
    assert [i["ref"] for i in items] == ["#1", "#L1", "#4"]
    assert items[1] == {"ref": "#L1", "kind": "line", "id": "L1", "t": 4.0, "label": "1 + 1 = 2"}
    assert items[0]["t"] == 0.0 and items[0]["kind"] == "piece" and items[0]["id"] == 1
    assert times == [5.0]  # 1:00:05 and 2:01 are past the 2:00 video; 9:99 is not a time
    long = {**r, "video": {**r["video"], "duration": 4000.0}}
    assert ask.answer_refs("[1:00:05] and [61:30]", long)[1] == [3605.0, 3690.0]
    assert ask.answer_refs("[#s1e3] [#s0e9]", slides_replay())[0][0]["t"] == 12.0


def test_tidy_answer_splits_lists_and_ranges():
    assert ask.tidy_answer("see [#3, #4] and [#3,#L4; #s0e1]") == "see [#3] [#4] and [#3] [#L4] [#s0e1]"
    assert ask.tidy_answer("from [1:02-1:10] and [1:02 – 1:10:05]") == "from [1:02]–[1:10] and [1:02]–[1:10:05]"
    assert ask.tidy_answer("[#3] at [1:02]") == "[#3] at [1:02]"


def test_clock():
    assert [ask.clock(t) for t in (0, 59.99, 61.6, 3599.5, 3605)] == ["0:00", "0:59", "1:01", "59:59", "1:00:05"]


# ---------- events ----------
def parse_events(text):
    out = []
    assert text.endswith("\n\n")
    for block in text.split("\n\n")[:-1]:
        lines = block.split("\n")
        assert len(lines) == 2 and lines[0].startswith("event: ") and lines[1].startswith("data: ")
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def test_sse_framing():
    e = ask.sse("delta", {"text": "line one\nline two   é"})
    assert e.startswith("event: delta\ndata: ") and e.endswith("\n\n") and e.count("\n") == 3
    assert parse_events(e) == [("delta", {"text": "line one\nline two   é"})]


# ---------- the conversation file ----------
def test_chat_save_load_clear(tmp_path):
    assert ask.load_chat(tmp_path) == {"turns": []}
    ask.save_turn(tmp_path, {"id": "a", "q": "one"})
    ask.save_turn(tmp_path, {"id": "b", "q": "two"})
    assert [t["id"] for t in ask.load_chat(tmp_path)["turns"]] == ["a", "b"]
    assert not list(tmp_path.glob("*.tmp"))
    ask.clear_chat(tmp_path)
    assert ask.load_chat(tmp_path) == {"turns": []}
    (tmp_path / "chat.json").write_text("{broken", encoding="utf-8")
    assert ask.load_chat(tmp_path) == {"turns": []}
    ask.save_turn(tmp_path, {"id": "c"})  # a broken file is replaced, not a crash
    assert ask.load_chat(tmp_path)["turns"] == [{"id": "c"}]


# ---------- the endpoints ----------
class FakeClient:
    def __init__(self, chunks, cost=0.0123, cached=False, note="", fail=None):
        self.chunks, self.cost, self.cached, self.note, self.fail = chunks, cost, cached, note, fail
        self.calls, self.seen, self.before, self.closed = [], [], [], False

    def chat_stream(self, messages, **kw):
        self.seen.append((messages, kw))
        try:
            for c in self.chunks:
                yield c
        except GeneratorExit:
            self.closed = True
            raise
        if self.fail:
            raise MetaAPIError(self.fail)
        self.calls += self.before
        self.calls.append(CallRecord("/chat/completions", "muse-spark-1.3", 200, 0.5, self.cached, 1, {},
                                     self.cost, "k", self.note))


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JOBS", tmp_path)
    write_job(tmp_path / "lec", board_replay())

    def use(fake):
        """Answer with this fake client (or call this function to make the client)."""
        monkeypatch.setattr(server, "_meta_client", fake if callable(fake) else (lambda: fake))
        return fake

    return TestClient(server.app), tmp_path / "lec", use


@pytest.mark.parametrize("body", [
    b"not json", b"[]", b'{"question": ""}', b'{"question": "q", "selection": [{"kind": "piece"}]}',
    json.dumps({"question": "q", "selection": [{"kind": "moment", "t": 1}] * 13}).encode(),
])
def test_ask_rejects_bad_requests_before_streaming(api, body):
    client, d, use = api
    fake = use(FakeClient(["never"]))
    r = client.post("/api/jobs/lec/ask", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400 and isinstance(r.json()["detail"], str) and "\n" not in r.json()["detail"]
    assert fake.seen == []


def test_ask_needs_a_processed_lecture(api):
    client, d, use = api
    (d.parent / "raw").mkdir()
    r = client.post("/api/jobs/raw/ask", json={"question": "q"})
    assert r.status_code == 409 and "not processed yet" in r.json()["detail"]
    assert client.post("/api/jobs/nope/ask", json={"question": "q"}).status_code == 404


def test_a_whole_ask_streams_saves_and_lists_the_turn(api):
    client, d, use = api
    fake = use(FakeClient(["Look at [#L", "1] at [0:04]. ", "Not [#99]; $x^2$ is [#5]."]))
    body = {"question": "How are these connected?", "t": 12.5,
            "selection": [{"kind": "line", "id": "L1"}, {"kind": "piece", "id": 77}, {"kind": "moment", "t": 25}],
            "history": [{"q": "first?", "a": "first answer"}]}
    r = client.post("/api/jobs/lec/ask", json=body)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_events(r.text)
    kinds = [e for e, _ in events]
    assert kinds == ["delta", "delta", "delta", "done"]
    done = events[-1][1]
    assert done["answer"] == "".join(e["text"] for k, e in events if k == "delta")
    assert [i["ref"] for i in done["items"]] == ["#L1", "#5"]
    assert done["items"][1] == {"ref": "#5", "kind": "piece", "id": 5, "t": 45.0, "label": "x^2"}
    assert done["times"] == [4.0] and done["cost_usd"] == 0.0123 and done["cached"] is False
    assert [s["kind"] for s in done["selection"]] == ["line", "moment"]
    messages, kw = fake.seen[0]
    assert kw["reasoning"] == "low" and kw["max_tokens"] == 2000
    assert kw["extra"]["prompt_cache_key"].startswith("replay-board-lec-")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "Question: How are these connected?" in last_text(messages) and len(images(messages)) == 2

    turns = client.get("/api/jobs/lec/chat").json()["turns"]
    assert len(turns) == 1
    t = turns[0]
    assert t["id"] == done["id"] and t["q"] == body["question"] and t["answer"] == done["answer"]
    assert t["items"] == done["items"] and t["times"] == [4.0] and t["t"] == 12.5 and t["cost_usd"] == 0.0123
    assert t["selection"][0] == {"kind": "line", "id": "L1", "label": "1 + 1 = 2", "t": 4.0}
    assert isinstance(t["at"], float)

    assert client.delete("/api/jobs/lec/chat").json() == {"turns": []}
    assert client.get("/api/jobs/lec/chat").json() == {"turns": []}


def test_a_failed_answer_is_an_error_event_and_not_saved(api):
    client, d, use = api
    use(FakeClient(["Half an ans"], fail="the answer stopped halfway (network error: ReadError); please ask again"))
    events = parse_events(client.post("/api/jobs/lec/ask", json={"question": "q"}).text)
    assert [e for e, _ in events] == ["delta", "error"]
    assert events[1][1]["error"].startswith("Muse Spark could not answer: the answer stopped halfway")
    assert client.get("/api/jobs/lec/chat").json() == {"turns": []}

    def broken():
        raise MetaAPIError("MODEL_API_KEY is missing: put it in .env (see .env.example).")

    use(broken)
    events = parse_events(client.post("/api/jobs/lec/ask", json={"question": "q"}).text)
    assert events == [("error", {"error": "Muse Spark could not answer: MODEL_API_KEY is missing: put it in .env "
                                          "(see .env.example)."})]


def test_unexpected_failure_is_one_line_with_a_log(api):
    client, d, use = api

    class Boom(FakeClient):
        def chat_stream(self, messages, **kw):
            raise ZeroDivisionError("division by zero\nmore")
            yield ""

    use(Boom([]))
    events = parse_events(client.post("/api/jobs/lec/ask", json={"question": "q"}).text)
    assert events[0][0] == "error" and "ZeroDivisionError: division by zero" in events[0][1]["error"]
    assert "\n" not in events[0][1]["error"] and (d / "ask-error.log").exists()


def test_cached_and_cut_answers(api):
    client, d, use = api
    use(FakeClient(["All of it [#1]."], cost=0.0, cached=True))
    done = parse_events(client.post("/api/jobs/lec/ask", json={"question": "q"}).text)[-1][1]
    assert done["cached"] is True and done["cost_usd"] == 0.0

    use(FakeClient(["A long answer"], note="cut at max tokens; first text after 1.0 s"))
    events = parse_events(client.post("/api/jobs/lec/ask", json={"question": "q2"}).text)
    assert events[-2] == ("delta", {"text": ask.CUT_NOTE})
    assert events[-1][1]["answer"] == "A long answer" + ask.CUT_NOTE
    assert len(client.get("/api/jobs/lec/chat").json()["turns"]) == 2


def test_done_counts_every_call_of_the_answer(api):
    client, d, use = api
    fake = FakeClient(["Here [#1]."], cost=0.004, note="cut at max tokens")
    fake.calls = [CallRecord("/chat/completions", "m", 200, 1.0, False, 1, {}, 0.5, "old")]  # an earlier answer
    fake.before = [CallRecord("/chat/completions", "m", 200, 1.0, False, 1, {}, None, "k", "empty stream")]
    use(fake)
    events = parse_events(client.post("/api/jobs/lec/ask", json={"question": "q"}).text)
    done = events[-1][1]
    assert done["cost_usd"] == 0.004 and done["cached"] is False and done["saved"] is True
    assert done["answer"].endswith(ask.CUT_NOTE)


def test_a_failed_save_still_gives_the_answer(api, monkeypatch):
    client, d, use = api
    use(FakeClient(["Fine."]))

    def locked(job_dir, turn):
        raise OSError("could not save the conversation: chat.json is in use")

    monkeypatch.setattr(ask, "save_turn", locked)
    events = parse_events(client.post("/api/jobs/lec/ask", json={"question": "q"}).text)
    assert [e for e, _ in events] == ["delta", "done"]  # no error after done: the player would drop the answer
    assert events[-1][1]["saved"] is False and events[-1][1]["answer"] == "Fine."


def test_a_stopped_answer_closes_the_client_and_is_not_saved(tmp_path):
    r = board_replay()
    d = write_job(tmp_path / "job", r)

    class Http:
        closed = False

        def close(self):
            Http.closed = True

    fake = FakeClient(["one ", "two ", "three"])
    fake.http = Http()
    events = ask.answer_events(ask.prepare(d, r, {"question": "q"}), lambda: fake)
    assert next(events).startswith("event: delta")
    events.close()  # what happens when the student presses Stop and the connection drops
    assert Http.closed and fake.closed and ask.load_chat(d) == {"turns": []}


def test_the_server_closes_the_answer_when_the_student_stops_it():
    """Stop drops the connection: the server must close the answer's generator at once (so the model's stream is
    closed and the call logged), not leave it to the garbage collector."""
    import anyio

    closed = []

    def events():
        try:
            while True:  # an answer that would go on: only closing it ends it
                yield "event: delta\ndata: {\"text\": \"x\"}\n\n"
        finally:
            closed.append(True)

    async def main():
        first = anyio.Event()

        async def send(m):
            if m["type"] == "http.response.body":
                first.set()

        async def receive():
            await first.wait()
            return {"type": "http.disconnect"}

        resp = server.EventStream(events(), media_type="text/event-stream")
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}}  # what uvicorn says
        await resp(scope, receive, send)
        return list(closed)  # read before the response object (and its generator) goes away

    assert anyio.run(main) == [True]


def test_chat_endpoints_need_a_job(api):
    client, d, use = api
    assert client.get("/api/jobs/nope/chat").status_code == 404
    assert client.delete("/api/jobs/nope/chat").status_code == 404
    assert client.get("/api/jobs/lec/chat").json() == {"turns": []}


def test_isfinite_guard():
    assert not ask._num(math.inf) and not ask._num(True) and ask._num(3)


# ---------- fixes from review ----------
def test_empty_history_turns_are_left_out():
    req = ask.check_request({"question": "q", "history": [{"q": "a", "a": "  "}, {"q": "", "a": "b"},
                                                          {"q": "c", "a": "d"}]})
    assert req["history"] == [{"q": "c", "a": "d"}]


def test_a_moment_on_an_empty_board_sends_no_picture(tmp_path):
    r = board_replay()
    r["board"]["pieces"][0]["pre_existing"] = False  # nothing is on the board before 0:05
    r["board"]["pieces"][0]["t_shown"] = r["board"]["pieces"][0]["t_seen"] = 3.0
    d = write_job(tmp_path / "job", r)
    p = ask.prepare(d, r, {"question": "q", "selection": [{"kind": "moment", "t": 1.0}]})
    assert images(p.messages) == [] and "The board was empty then." in last_text(p.messages)


def test_system_file_errors_are_plain():
    msg = ask.plain(PermissionError(13, "Permission denied", r"C:\Users\someone\Replay Board\data\jobs\x\chat.json"))
    assert msg == "file problem: Permission denied (chat.json)"
    assert ask.plain(OSError("could not save the conversation: chat.json is in use")) == \
        "could not save the conversation: chat.json is in use"


def test_an_unreadable_conversation_is_not_overwritten(tmp_path, monkeypatch):
    ask.save_turn(tmp_path, {"id": "a"})
    real = ask.Path.read_text

    def locked(self, *a, **kw):
        if self.name == "chat.json":
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *a, **kw)

    monkeypatch.setattr(ask.time, "sleep", lambda s: None)
    monkeypatch.setattr(ask.Path, "read_text", locked)
    with pytest.raises(OSError):
        ask.save_turn(tmp_path, {"id": "b"})
    assert ask.load_chat(tmp_path) == {"turns": []}  # reading for the player is not strict
    monkeypatch.setattr(ask.Path, "read_text", real)
    assert [t["id"] for t in ask.load_chat(tmp_path)["turns"]] == ["a"]  # the old turn is still there


@pytest.mark.parametrize("origin,ok", [
    (None, True), ("http://testserver", True), ("http://localhost:5173", True), ("http://127.0.0.1:8000", True),
    ("http://[::1]:5173", True), ("http://192.168.1.20:5173", True), ("http://my-pc:5173", True),
    ("https://evil.example", False), ("null", False), ("http://localhost.evil.example", False),
    ("http://192.168.1.20.evil.example", False),
])
def test_ask_refuses_pages_from_other_sites(api, origin, ok):
    client, d, use = api
    fake = use(FakeClient(["Hi."]))
    headers = {"origin": origin} if origin else {}
    r = client.post("/api/jobs/lec/ask", json={"question": "q"}, headers=headers)
    assert (r.status_code == 200) is ok and (len(fake.seen) == 1) is ok
    if not ok:
        assert r.status_code == 403 and "\n" not in r.json()["detail"]
    assert (client.delete("/api/jobs/lec/chat", headers=headers).status_code == 200) is ok


def test_ask_refuses_a_huge_or_too_deep_body(api):
    client, d, use = api
    fake = use(FakeClient(["never"]))
    big = json.dumps({"question": "q", "history": [{"q": "q", "a": "a" * 1000}] * 2000}).encode()
    for body in (big, b"[" * 100_000):
        r = client.post("/api/jobs/lec/ask", content=body, headers={"content-type": "application/json"})
        assert r.status_code == 400 and "\n" not in r.json()["detail"]
    assert "too big" in client.post("/api/jobs/lec/ask", content=big).json()["detail"]
    assert fake.seen == []


def test_an_odd_replay_is_one_plain_line_not_a_crash(api):
    client, d, use = api
    fake = use(FakeClient(["never"]))
    r = board_replay()
    r["board"]["pieces"][0]["t_seen"] = "soon"
    (d / "replay.json").write_text(json.dumps(r), encoding="utf-8")
    resp = client.post("/api/jobs/lec/ask", json={"question": "q"})
    assert resp.status_code == 500 and "ValueError" in resp.json()["detail"] and "\n" not in resp.json()["detail"]
    assert (d / "ask-error.log").exists() and fake.seen == []


def test_job_ids_are_matched_whole(api):
    client, d, use = api
    with pytest.raises(server.HTTPException):
        server._job_dir("lec\n")
    assert server._job_dir("lec") == d


def test_popouts_refuse_pages_from_other_sites(api, monkeypatch):
    import app.popout as popout
    client, d, use = api
    made = []
    monkeypatch.setattr(popout, "make_scene", lambda *a, **k: made.append(a) or {"kind": "none"})
    r = client.post("/api/jobs/lec/popout/1", headers={"origin": "https://evil.example"})
    assert r.status_code == 403 and "\n" not in r.json()["detail"] and made == []
    assert client.post("/api/jobs/lec/popout/1").status_code == 200 and len(made) == 1


def test_dropped_connections_are_not_logged_as_tracebacks():
    import logging

    from app.server import quiet_dropped_connections

    def record(exc):
        return logging.LogRecord("asyncio", logging.ERROR, __file__, 1, "msg", None,
                                 (type(exc), exc, None) if exc else None)
    assert not quiet_dropped_connections(record(ConnectionResetError(10054, "reset")))
    assert not quiet_dropped_connections(record(ConnectionAbortedError()))
    assert quiet_dropped_connections(record(ValueError("real problem")))
    assert quiet_dropped_connections(record(None))
