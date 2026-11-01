# Replay Board

**Rewatch a recorded lecture with a clean board beside it.** Upload a lecture video; Replay Board plays
the original, untouched, next to a virtual board that redraws the teacher's writing stroke by stroke,
in sync with their voice, including writing that was hidden behind the teacher while it was written.
Click any piece of writing to jump to the moment it was explained, or switch handwriting to typeset maths.
Slide lectures get the same treatment: clean slides, and the teacher's cursor or laser turned into
calm spotlight, circle, underline and arrow animations.

Built for the Meta Global AI Developer Hackathon. **Meta models do the understanding**; plain code does
the per-pixel measurement.

## How it works

```mermaid
flowchart LR
  V[Lecture video] --> C{Muse Spark<br/>board or slides?}
  C -->|board| S1[SAM 3.1<br/>board corners]
  C -->|board| S2[SAM 3.1 video<br/>person masks]
  S1 --> BC[Board capture<br/>code, per pixel]
  S2 --> BC
  BC --> R1[Muse Spark<br/>read each piece]
  C -->|slides| SA[Slide analysis<br/>code: states, clean slides,<br/>pointer track, gesture guesses]
  SA --> R2[Muse Spark<br/>read slides +<br/>gesture meaning]
  V --> T[Muse Voice Transcribe<br/>parallel pieces]
  T --> A[MMS forced aligner<br/>word times]
  R1 --> L[Muse Spark<br/>link writing to speech,<br/>redraw times, chapters]
  R2 --> L
  A --> L
  L --> J[(replay.json)]
  J --> P[Player<br/>video + board/slides<br/>on one clock]
```

| Step | Done by | What it decides |
|---|---|---|
| Board or slides? | **Muse Spark 1.3** on 4 frames | type, board colour, webcam corner, subject |
| Where the board is | **SAM 3.1** ("whiteboard", then "blackboard", ...) on 3 frames | mask → convex hull → 4 corners |
| Who is in front of the board | **SAM 3.1 video mode** ("person") on a 2 fps copy | occlusion mask for every sample |
| What is written, where and when | board capture (code) | pieces of writing, erases, hidden writing |
| What each piece says | **Muse Spark** on sharp crops from the video | kind, text, LaTeX |
| Slide changes, clean slides, pointer | slide analysis (code) | states, live regions, cursor/laser track, gesture guesses |
| What a pointing gesture means | **Muse Spark** with the path drawn on the slide + the words spoken | spotlight / circle / underline / arrow / pulse, target elements |
| Words | **Muse Voice Transcribe** in parallel pieces | turns |
| Word times | **MMS forced aligner** (Meta, torchaudio) | per-word start/end |
| When each element is talked about | **Muse Spark** with elements + timed transcript | spoken ranges, redraw time of hidden writing, chapters |

Every model call goes through one client (`app/core/meta_api.py`) with retries, byte uploads and a disk
cache keyed by model + request, so re-running a lecture costs nothing. Measured calls and prices are in
[docs/meta-api-notes.md](docs/meta-api-notes.md).

### Board capture: "writing stays put, the teacher moves"
Frames are flattened onto the board at 3 fps. Ink is what morphological closing removes; each pixel keeps
counters that only move while the board is visible there (SAM person mask, plus built-in detectors for
large dark things and too-thick strokes). A pixel is accepted after 1 s of ink and removed after 2 s of
clean board. New pixels are grouped into pieces of writing; removals into erases. Writing that appears
long after its spot was last seen blank was written while hidden, and is redrawn when it was talked about.
Camera drift of a pixel or two is tolerated rather than treated as erase-and-rewrite. The result is a
per-pixel replay log; the player gets per-piece images with drawing-order and erase-time maps.

### Self-tests on generated lectures
`tests/gen_board.py` and `tests/gen_slides.py` render lectures with known answers (a teacher who walks in
front of what he writes, an eraser, chalk, fade-ins, camera drift, a webcam corner, a parked cursor, a
laser...). The analysis is scored against them:

```text
python -m tests.selftest_board    ->  72/72 checks (6 variants)
python -m tests.selftest_slides   ->  47/47 checks (4 variants)
```

### What it costs and how long it takes
Measured on an 8-minute 1080p board lecture, nothing cached, on a 4-core laptop:

| | |
|---|---|
| Meta model calls | 19 (Muse Spark 9, SAM 3.1 7, Voice Transcribe 3) |
| API cost | $0.38 (SAM person masks $0.19) |
| First replay ready | 4.8 min |
| Word times refined (MMS) | 7.4 min (the replay is rewritten in place; the player picks it up) |
| Board capture alone | 4–12× real time, depending on what else the machine is doing |

Re-running a lecture is free: every model answer is cached.

## Run it (Windows, PowerShell)

Needs Python 3.10–3.13, ffmpeg on PATH, Node 20+, and a Meta Model API key.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env        # then put your MODEL_API_KEY in .env
cd web; npm install; npm run build; cd ..
```

```powershell
.venv\Scripts\python -m app.server
```
Open http://127.0.0.1:8000, drop a lecture video, and watch the stages; the player opens when it's ready.

Or from the command line (writes `data\jobs\<name>-<hash>\replay.json`):
```powershell
.venv\Scripts\python -m app.process my-lecture.mp4 [--start 60] [--end 600]
```

Tests:
```powershell
.venv\Scripts\python -m pytest -q             # unit tests
.venv\Scripts\python -m pytest -q -m selftest # generated-lecture self-tests (slow)
cd web; npm test
```

## Layout
```
app/            pipeline, CLI (app.process), server (app.server), smoke test (app.smoke)
app/core/       meta_api (one client), board, slides, sam, sam_codec, speech, understand, media
web/            player: React + Vite + TypeScript + KaTeX
tests/          generated-lecture self-tests and unit tests
data/           videos, job folders and the model cache (never committed)
```

## Also in the box
- **Study notes**: `GET /api/jobs/<id>/notes.md` (and the player's Notes button): every piece of writing or slide
  element by chapter, as LaTeX, with when it was explained and what was said.
- **3D pop-out** (stretch): for a drawn graph, surface or solid, Muse Spark returns a small JSON scene that the
  player renders with three.js. The scene is pure data: maths expressions are checked against a whitelist on
  the server and evaluated by the player's own parser, never as code.

## Limits
- The camera must stay still (a tripod or a fixed classroom camera). Panning or zooming cameras are not handled.
- MMS word alignment runs on the CPU at 1–3× real time; for long lectures the replay appears first with
  turn-level word times and is refined in place.
- Handwriting reading is as good as the frame: very small or blurry writing may be read partly.

## Privacy and video rules
The original video is never altered: no outlining, blurring or removing the teacher. Videos and anything
made from them stay in `data/`, which is not committed.
