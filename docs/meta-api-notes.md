# Meta Model API: what we verified (M0, 1 Nov 2026)

Base URL `https://api.meta.ai/v1`, header `Authorization: Bearer $MODEL_API_KEY` (keys look like `LLM_<id>_<secret>`).
Docs: https://dev.meta.ai/docs. All calls go through `app/core/meta_api.py` (retries, disk cache, cost log).

## Smoke test (`python -m app.smoke`, one call each, 1080p board lecture)

| Call | Model id | Latency | Usage | Cost |
|---|---|---|---|---|
| Board or slides? (960 px frame, JSON schema, `reasoning_effort: low`) | `muse-spark-1.3` | 2.0 s | 686 in / 452 out (409 of them reasoning) | $0.0028 |
| 8 s of speech, WAV 16 kHz mono, `mode: ENDPOINTING` | `muse-voice-transcribe-1.0` | 4.1 s | 8 s audio | $0.0004 |
| "whiteboard" on a 960 px frame | `sam-3.1` | 1.8 s | 1 image | $0.0025 |

All three returned correct answers: `board / white`, the right sentence, and one whiteboard mask (score 0.94).

## Prices used for cost estimates
- Muse Spark (standard): $1.25 per M input tokens, $0.15 cached, $4.25 per M output tokens (reasoning tokens count as output).
- Muse Wonder 1.0: $4.00 / $0.20 / $20.00. Contributor models are cheaper but Meta may train on the content, so we never use them.
- Voice Transcribe: $0.18 per audio hour. SAM 3.1: $2.50 per 1,000 images; video $0.20 per 1,000 frames (50-frame minimum).

## Format fixes against SPEC section 4
- Model list from `GET /models`: `muse-spark-1.3`, `-1.2`, `-1.1` (+ `-contributor`), `muse-wonder-1.0`, `muse-voice-transcribe-1.0`, `sam-3.1`, `muse-image-1.0`. The id is `sam-3.1`; `sam-3-1` does not exist.
- Chat: `max_tokens` is deprecated, so we send `max_completion_tokens`. `reasoning_effort: "none"` is rejected (minimal/low/medium/high/xhigh are fine). Muse Spark is tuned for the default temperature, so we leave it unset.
- Chat image parts: `{"type":"image_url","image_url":{"url":"data:..."}}` (an object). On `/responses` it is a plain string.
- Spark grounding boxes are normalised to 0–1000 from the top-left, not pixels.
- Structured output: `strict` defaults to false, so we always send `strict: true` with `additionalProperties: false` and every property required.
- Transcribe: only WAV is accepted (not raw s16le), at most 10 min and 32 MB; `turns` is empty unless `mode` is `ENDPOINTING`. The reply is `{sessionId, transcript, audioDurationMs, turns:[{turnId, startMs, endMs, transcript}]}`, with no word times.
- SAM 3.1 on `/responses`: send `stream: false`; metadata values must be strings. The reply text is in `output[0].content[0].text`, and usage reports `images_processed`. Each object's mask is box-local (mask size = box size, inclusive corners); `meta-sam-parser` 0.1.0 parses it, and the mask payload keeps its leading encoding marker.
- Rate limits: 3,000 RPM / 4M TPM per team for Spark; Transcribe allows 256 concurrent streams.
