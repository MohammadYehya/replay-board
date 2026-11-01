"""Unit tests for speech splitting, spoken forms and fallback timing (SPEC 5.3)."""
import numpy as np

from app.core import speech
from app.core.speech import RATE, _proportional, split_points, spoken_form


def test_short_audio_is_one_piece():
    assert split_points(np.zeros(RATE * 30, np.int16)) == [(0, RATE * 30)]


def test_split_cuts_at_quiet_point_within_last_30s():
    rng = np.random.default_rng(0)
    n = RATE * 1200  # 20 min
    pcm = (rng.normal(0, 3000, n)).astype(np.int16)
    quiet = int(520 * RATE)
    pcm[quiet: quiet + RATE] = 0  # one second of silence at 520 s
    pieces = split_points(pcm, max_len=540, search=30)
    assert pieces[0][0] == 0 and pieces[-1][1] == n
    assert all(b - a <= 540 * RATE for a, b in pieces)
    assert all(pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1))
    assert quiet <= pieces[0][1] <= quiet + RATE


def test_spoken_form():
    assert spoken_form("2a") == "two a"
    assert spoken_form("Hello,") == "hello"
    assert spoken_form("don’t") == "don't"
    assert spoken_form("=") == ""
    assert spoken_form("$5") == "five"


def test_proportional_fallback_covers_turn():
    turn = {"start": 10.0, "end": 12.0, "text": "so x equals two"}
    words = _proportional(turn, turn["text"].split())
    assert [w["w"] for w in words] == ["so", "x", "equals", "two"]
    assert words[0]["s"] == 10.0 and words[-1]["e"] == 12.0
    assert all(words[i]["e"] == words[i + 1]["s"] for i in range(3))
    # longer words get more time
    assert (words[2]["e"] - words[2]["s"]) > (words[1]["e"] - words[1]["s"])


def test_transcribe_offsets_pieces(monkeypatch):
    calls = []

    class Fake:
        def transcribe(self, wav):
            calls.append(len(wav))
            return {"turns": [{"startMs": 1000, "endMs": 2500, "transcript": "hello there"}]}

    monkeypatch.setattr(speech, "MAX_PIECE_S", 540.0)
    pcm = np.ones(RATE * 600, np.int16)
    turns = speech.transcribe(pcm, Fake())
    assert len(calls) == 2 and len(turns) == 2
    assert turns[0]["start"] == 1.0
    assert turns[1]["start"] > 500  # second piece shifted by its offset


def test_default_pieces_are_short_enough_to_run_in_parallel():
    pieces = split_points(np.ones(RATE * 600, np.int16))
    assert len(pieces) >= 4 and all(b - a <= speech.MAX_PIECE_S * RATE for a, b in pieces)
