"""python -m app.process: a bad argument is one plain line and a non-zero exit, not argparse's usage text."""
import pytest

from app import process


@pytest.mark.parametrize("argv, says", [
    (["clip.mp4", "--start", "abc"], "--start: 'abc' is not a number of seconds"),
    (["clip.mp4", "--end", "nan"], "--end: 'nan' is not a number of seconds"),
    (["clip.mp4", "--width", "wide"], "--width: 'wide' is not a width in pixels from 320 to 3840"),
    (["clip.mp4", "--width", "0"], "--width: '0' is not a width"),
    (["clip.mp4", "--width", "-5"], "--width: '-5' is not a width"),
    (["clip.mp4", "--width", "100000"], "--width: '100000' is not a width"),
    (["clip.mp4", "--width", "800.5"], "--width: '800.5' is not a width"),
    (["clip.mp4", "--bogus"], "unrecognized arguments: --bogus"),
    ([], "required: video"),
])
def test_a_bad_argument_is_one_plain_line(argv, says, capsys, monkeypatch):
    monkeypatch.setattr(process, "run", lambda *a, **k: pytest.fail("must not start processing"))
    assert process.main(argv) == 2
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: ") and says in lines[0], err


def test_good_arguments_reach_the_pipeline(capsys, monkeypatch, tmp_path):
    got = {}

    def run(video, job, start, end, **k):
        got.update(video=video, start=start, end=end, **k)
        return tmp_path / "replay.json"

    monkeypatch.setattr(process, "run", run)
    assert process.main(["clip.mp4", "--start", "12.5", "--end", "90", "--no-gestures", "--width", "640"]) == 0
    assert (got["video"], got["start"], got["end"], got["gestures"], got["width"]) == ("clip.mp4", 12.5, 90.0, False, 640)
    assert process.main(["clip.mp4"]) == 0 and got["width"] == 800
