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
