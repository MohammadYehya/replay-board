"""Offline tests for the Meta API client: retries, resent uploads, cache, parsing, .env decoding."""
import json
from dataclasses import replace

import httpx
import numpy as np
import pytest

from app.config import load_settings, read_env_file
from app.core import meta_api
from app.core.meta_api import MetaAPIError, MetaClient, parse_json_object, parse_sam_output, strict_schema


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(meta_api.time, "sleep", lambda s: None)


def make_client(tmp_path, handler, cache=True):
    s = replace(load_settings(tmp_path / "none.env"), api_key="LLM_1_test", cache_dir=tmp_path / "cache")
    return MetaClient(s, use_cache=cache, transport=httpx.MockTransport(handler))


def chat_reply(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100,
                      "prompt_tokens_details": {"cached_tokens": 0}}}


def test_retry_then_success_and_cache(tmp_path):
    hits = []

    def handler(req):
        hits.append(req)
        if len(hits) < 3:
            return httpx.Response(503, headers={"Retry-After": "0"}, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=chat_reply('{"a": 1}'))

    c = make_client(tmp_path, handler)
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert c.ask("hi", schema=schema) == {"a": 1}
    assert len(hits) == 3 and c.calls[-1].attempts == 3
    assert c.calls[-1].cost_usd == pytest.approx((1000 * 1.25 + 100 * 4.25) / 1e6)
    body = json.loads(hits[-1].content)
    assert body["response_format"]["json_schema"]["strict"] is True
    # Second identical call comes from the disk cache.
    assert c.ask("hi", schema=schema) == {"a": 1}
    assert len(hits) == 3 and c.calls[-1].cached


def test_non_retryable_error_is_one_line(tmp_path):
    c = make_client(tmp_path, lambda req: httpx.Response(400, json={"error": {"message": "bad image"}}))
    with pytest.raises(MetaAPIError) as e:
        c.ask("hi")
    assert "HTTP 400" in str(e.value) and "bad image" in str(e.value) and "\n" not in str(e.value)


def test_retry_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(meta_api, "RETRY_BUDGET_S", 0.0)
    c = make_client(tmp_path, lambda req: httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(MetaAPIError, match="429"):
        c.ask("hi")


def test_transcribe_retry_resends_whole_upload(tmp_path):
    sizes = []

    def handler(req):
        body = req.read()
        sizes.append(len(body))
        assert b"RIFF" in body and b'"audioEncoding": "WAV"' in body
        if len(sizes) == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={"turns": [{"startMs": 0, "endMs": 900, "transcript": "so"}]})

    c = make_client(tmp_path, handler)
    from app.core.media import wav_bytes
    out = c.transcribe(wav_bytes(np.zeros(16000, np.int16)))
    assert out["turns"][0]["transcript"] == "so"
    assert len(sizes) == 2 and sizes[0] == sizes[1] > 32000


def test_reply_must_be_json_object(tmp_path):
    c = make_client(tmp_path, lambda req: httpx.Response(200, json=[1, 2]), cache=False)
    with pytest.raises(MetaAPIError, match="JSON object"):
        c.chat([{"role": "user", "content": "x"}])
    with pytest.raises(MetaAPIError):
        parse_json_object("[1]")
    assert parse_json_object('```json\n{"x": 2}\n```') == {"x": 2}


def test_strict_schema_requires_everything():
    s = strict_schema({"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}}}}})
    assert s["additionalProperties"] is False and s["required"] == ["a"]
    assert s["properties"]["a"]["required"] == ["b"] and s["properties"]["a"]["additionalProperties"] is False


def test_parse_sam_output_places_box_local_mask():
    text = "<0f>7<|box;x1=10;y1=20;x2=14;y2=24;w=64;h=48;c=0.9|><|mask;x=0;y=0;c=0.9;data=5,5,!!!!!(QO(0lu8?|>\n"
    objs = parse_sam_output(text, (48, 64))
    assert len(objs) == 1 and objs[0]["id"] == "7" and objs[0]["score"] == pytest.approx(0.9)
    m = objs[0]["mask"]
    assert m.shape == (48, 64) and m.sum() == 10
    ys, xs = np.nonzero(m)
    assert xs.min() >= 10 and xs.max() <= 14 and ys.min() >= 20 and ys.max() <= 24
    assert parse_sam_output("", (48, 64)) == []


@pytest.mark.parametrize("encoding,bom", [("utf-8", b""), ("utf-8", b"\xef\xbb\xbf"), ("utf-16", b"")])
def test_env_file_encodings(tmp_path, encoding, bom):
    p = tmp_path / ".env"
    p.write_bytes(bom + 'MODEL_API_KEY="LLM_1_abc"\n# c\nMODEL_SPARK=muse-spark-1.2\n'.encode(encoding))
    env = read_env_file(p)
    assert env["MODEL_API_KEY"] == "LLM_1_abc" and env["MODEL_SPARK"] == "muse-spark-1.2"


def test_timeout_is_retried_even_past_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(meta_api, "RETRY_BUDGET_S", 0.0)
    hits = []

    def handler(req):
        hits.append(1)
        if len(hits) == 1:
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(200, json=chat_reply("ok"))

    c = make_client(tmp_path, handler)
    assert c.ask("hi") == "ok" and len(hits) == 2


# ---------- streaming (Ask Muse) ----------
USAGE = {"prompt_tokens": 2000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 1000}}


def sse_body(*texts, usage=True, done=True, finish="stop"):
    """A streamed chat reply, as Meta sends it: a data line per chunk, usage in the last one, then [DONE]."""
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
    chunks += [{"choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]} for t in texts]
    last = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    if usage:
        last["usage"] = USAGE
    if finish:
        chunks.append(last)
    lines = [f"data: {json.dumps(c)}" for c in chunks] + (["data: [DONE]"] if done else [])
    return ("\n\n".join(lines) + "\n\n").encode()


def in_pieces(data: bytes, n: int = 7):
    """The body in small chunks that split events (and UTF-8 characters) anywhere."""
    for i in range(0, len(data), n):
        yield data[i:i + n]


def stream_reply(data, n=7):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=in_pieces(data, n))


def test_sse_data_joins_lines_and_skips_comments():
    lines = [": keep-alive", "event: x", "data: one", "data:two", "", "", "data: [DONE]"]
    assert list(meta_api.sse_data(lines)) == ["one\ntwo", "[DONE]"]


def test_chat_stream_yields_deltas_then_caches(tmp_path):
    hits = []

    def handler(req):
        hits.append(json.loads(req.content))
        if len(hits) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, json={"error": {"message": "busy"}})
        return stream_reply(sse_body("Hello ", "wörld", " [#3]."), n=5)

    c = make_client(tmp_path, handler)
    msgs = [{"role": "user", "content": "hi"}]
    out = list(c.chat_stream(msgs, reasoning="low", max_tokens=2000, extra={"prompt_cache_key": "k1"}))
    assert out == ["Hello ", "wörld", " [#3]."]
    body = hits[-1]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["reasoning_effort"] == "low" and body["max_completion_tokens"] == 2000 and body["prompt_cache_key"] == "k1"
    rec = c.calls[-1]
    assert rec.attempts == 2 and not rec.cached and rec.usage == USAGE
    assert rec.cost_usd == pytest.approx((1000 * 1.25 + 1000 * 0.15 + 100 * 4.25) / 1e6)
    assert "first text after" in rec.note and "cost is unknown" not in rec.note
    # Again: from the disk cache, in one piece, without a request.
    assert list(c.chat_stream(msgs, reasoning="low", max_tokens=2000, extra={"prompt_cache_key": "k1"})) == \
        ["Hello wörld [#3]."]
    assert len(hits) == 2 and c.calls[-1].cached and c.calls[-1].cost_usd == 0.0
    # The cached reply reads like any chat reply.
    cached = json.loads(c._cache_path(rec.key).read_text(encoding="utf-8"))
    assert meta_api.chat_text(cached) == "Hello wörld [#3]."


def test_chat_stream_without_cache_reads_nothing(tmp_path):
    hits = []

    def handler(req):
        hits.append(1)
        return stream_reply(sse_body("ok"))

    c = make_client(tmp_path, handler, cache=False)
    assert list(c.chat_stream([{"role": "user", "content": "x"}])) == ["ok"]
    assert list(c.chat_stream([{"role": "user", "content": "x"}])) == ["ok"]
    assert len(hits) == 2


def test_chat_stream_retries_a_break_before_the_first_text(tmp_path):
    hits = []

    def broken():
        yield b'data: {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}\n\n'
        raise httpx.ReadError("connection reset")

    def handler(req):
        hits.append(1)
        if len(hits) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=broken())
        if len(hits) == 2:
            raise httpx.ConnectError("refused", request=req)
        if len(hits) == 3:  # ends without a finish reason or [DONE]: not a whole answer
            return stream_reply(sse_body(finish=None, done=False))
        return stream_reply(sse_body("fine"))

    c = make_client(tmp_path, handler)
    assert list(c.chat_stream([{"role": "user", "content": "x"}])) == ["fine"]
    assert len(hits) == 4 and c.calls[-1].attempts == 4


def test_chat_stream_does_not_retry_after_text_was_shown(tmp_path):
    hits = []

    def broken():
        yield sse_body("Half an ", finish=None, done=False)
        raise httpx.ReadError("connection reset")

    def handler(req):
        hits.append(1)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=broken())

    c = make_client(tmp_path, handler)
    got = []
    with pytest.raises(MetaAPIError) as e:
        for d in c.chat_stream([{"role": "user", "content": "x"}]):
            got.append(d)
    assert got == ["Half an "] and len(hits) == 1
    assert "stopped halfway" in str(e.value) and "\n" not in str(e.value)
    assert not list((tmp_path / "cache").glob("*/*.json"))  # nothing cached


def test_chat_stream_errors(tmp_path):
    c = make_client(tmp_path, lambda req: httpx.Response(400, json={"error": {"message": "bad image"}}))
    with pytest.raises(MetaAPIError) as e:
        list(c.chat_stream([{"role": "user", "content": "x"}]))
    assert "HTTP 400" in str(e.value) and "bad image" in str(e.value)

    # An error inside the stream: retried when the code says so, otherwise raised at once.
    hits = []

    def handler(req):
        hits.append(1)
        if len(hits) == 1:
            return stream_reply(b'data: {"error": {"message": "overloaded", "code": 529}}\n\n')
        if len(hits) == 2:
            return stream_reply(b'data: {"error": {"message": "not allowed", "code": 403}}\n\n')
        return stream_reply(sse_body("never"))

    c = make_client(tmp_path / "b", handler)
    with pytest.raises(MetaAPIError, match="not allowed"):
        list(c.chat_stream([{"role": "user", "content": "x"}]))
    assert len(hits) == 2


def test_chat_stream_without_usage_or_cut_short(tmp_path):
    c = make_client(tmp_path, lambda req: stream_reply(sse_body("a", "b", usage=False, finish="length")))
    assert list(c.chat_stream([{"role": "user", "content": "x"}])) == ["a", "b"]
    rec = c.calls[-1]
    assert rec.cost_usd is None and "no usage" in rec.note and "cut at max tokens" in rec.note

    c = make_client(tmp_path / "b", lambda req: stream_reply(sse_body(finish="length")))
    with pytest.raises(MetaAPIError, match="empty answer.*thinking"):
        list(c.chat_stream([{"role": "user", "content": "x"}]))


def test_an_empty_stream_is_asked_again_without_streaming(tmp_path):
    hits = []

    def handler(req):
        body = json.loads(req.content)
        hits.append(body.get("stream", False))
        if body.get("stream"):  # as seen live with images: finish "stop" at once, no text, no usage
            return stream_reply(sse_body(usage=False))
        assert "stream_options" not in body and body["prompt_cache_key"] == "k"
        return httpx.Response(200, json=chat_reply("Here it is [#3]."))

    c = make_client(tmp_path, handler)
    msgs = [{"role": "user", "content": "x"}]
    assert list(c.chat_stream(msgs, extra={"prompt_cache_key": "k"})) == ["Here it is [#3]."]
    assert hits == [True, False]
    assert "empty stream" in c.calls[-2].note
    assert c.calls[-1].cost_usd == pytest.approx((1000 * 1.25 + 100 * 4.25) / 1e6) and not c.calls[-1].cached
    # The answer is cached under the streaming request too: asking again sends nothing.
    assert list(c.chat_stream(msgs, extra={"prompt_cache_key": "k"})) == ["Here it is [#3]."]
    assert hits == [True, False] and c.calls[-1].cached


def test_chat_stream_accepts_a_plain_json_reply(tmp_path):
    c = make_client(tmp_path, lambda req: httpx.Response(200, json=chat_reply("whole")))
    assert list(c.chat_stream([{"role": "user", "content": "x"}])) == ["whole"]
    assert c.calls[-1].cost_usd == pytest.approx((1000 * 1.25 + 100 * 4.25) / 1e6)


def test_chat_stream_stopped_by_the_reader_is_logged_and_not_cached(tmp_path):
    c = make_client(tmp_path, lambda req: stream_reply(sse_body("one ", "two ", "three")))
    gen = c.chat_stream([{"role": "user", "content": "x"}])
    assert next(gen) == "one "
    gen.close()  # the student pressed Stop
    rec = c.calls[-1]
    assert not rec.cached and rec.cost_usd is None and "stopped by the reader" in rec.note
    assert not list((tmp_path / "cache").glob("*/*.json"))


def test_a_cut_answer_says_so_again_from_the_cache(tmp_path):
    hits = []

    def handler(req):
        hits.append(1)
        return stream_reply(sse_body("a long ans", finish="length"))

    c = make_client(tmp_path, handler)
    msgs = [{"role": "user", "content": "x"}]
    assert list(c.chat_stream(msgs)) == ["a long ans"]
    assert "cut at max tokens" in c.calls[-1].note
    assert list(c.chat_stream(msgs)) == ["a long ans"]
    assert len(hits) == 1 and c.calls[-1].cached and c.calls[-1].note == "cut at max tokens"


def test_an_empty_stream_asked_again_and_cut(tmp_path):
    def handler(req):
        if json.loads(req.content).get("stream"):
            return stream_reply(sse_body(usage=False))
        reply = chat_reply("part of it")
        reply["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=reply)

    c = make_client(tmp_path, handler)
    assert list(c.chat_stream([{"role": "user", "content": "x"}])) == ["part of it"]
    assert c.calls[-1].note == "cut at max tokens" and c.calls[-1].cost_usd is not None


def test_a_drop_after_the_finish_is_a_whole_answer(tmp_path):
    """The last chunk (finish reason and usage) came, then the connection broke before [DONE]: the answer is whole."""
    hits = []

    def broken():
        yield sse_body("All ", "of it.", done=False)
        raise httpx.ReadError("connection reset")

    def handler(req):
        hits.append(1)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=broken())

    c = make_client(tmp_path, handler)
    assert list(c.chat_stream([{"role": "user", "content": "x"}])) == ["All ", "of it."]
    assert len(hits) == 1 and c.calls[-1].usage == USAGE and c.calls[-1].cost_usd is not None
    assert list(c.chat_stream([{"role": "user", "content": "x"}])) == ["All of it."]  # and it was cached
    assert len(hits) == 1


def test_a_cache_file_that_cannot_be_written_does_not_lose_the_answer(tmp_path, monkeypatch):
    """Windows refuses a replace while another call reads the file: the paid-for answer still comes back."""
    real = meta_api.Path.replace

    def locked(self, target):
        if str(self).endswith(".tmp"):
            raise PermissionError(13, "The process cannot access the file", str(target))
        return real(self, target)

    monkeypatch.setattr(meta_api.Path, "replace", locked)
    hits = []

    def handler(req):
        hits.append(1)
        if json.loads(req.content).get("stream"):
            return stream_reply(sse_body("streamed"))
        return httpx.Response(200, json=chat_reply("plain"))

    c = make_client(tmp_path, handler)
    msgs = [{"role": "user", "content": "x"}]
    assert list(c.chat_stream(msgs)) == ["streamed"]
    assert meta_api.chat_text(c.chat(msgs)) == "plain"
    assert not list((tmp_path / "cache").rglob("*.tmp")) and not list((tmp_path / "cache").rglob("??/*.json"))
    assert len(hits) == 2
    # Two identical calls write through their own temporary files.
    monkeypatch.setattr(meta_api.Path, "replace", real)
    paths = set()
    real_write = meta_api.Path.write_text

    def spy(self, *a, **kw):
        paths.add(self.name)
        return real_write(self, *a, **kw)

    monkeypatch.setattr(meta_api.Path, "write_text", spy)
    assert meta_api._write_cache(tmp_path / "x" / "k.json", {"a": 1}) and meta_api._write_cache(tmp_path / "x" / "k.json", {"a": 2})
    assert len({p for p in paths if p.endswith(".tmp")}) == 2
    assert json.loads((tmp_path / "x" / "k.json").read_text(encoding="utf-8")) == {"a": 2}
