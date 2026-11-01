"""status.json must survive a reader holding it open (Windows refuses to replace an open file)."""
import json
import threading
import time

from app.pipeline import Status, plain_error


def test_status_writes_while_a_reader_polls(tmp_path):
    st = Status(tmp_path)
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                with open(tmp_path / "status.json", "rb") as f:
                    f.read()
                    time.sleep(0.02)  # hold it open like a slow HTTP response
            except OSError:
                pass
            time.sleep(0.03)

    th = threading.Thread(target=reader)
    th.start()
    try:
        for i in range(25):
            st.start("capture")
            st.progress("capture", i / 25)
            st.done("capture", f"step {i}")
    finally:
        stop.set()
        th.join()
    data = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert data["stages"]["capture"]["state"] == "done"


def test_plain_error_is_one_line():
    try:
        raise ValueError("bad thing\nwith a second line")
    except ValueError as e:
        msg = plain_error(e)
    assert "\n" not in msg and "bad thing" in msg
    assert "file problem" in plain_error(PermissionError(13, "Access is denied", "C:/x/status.json"))
