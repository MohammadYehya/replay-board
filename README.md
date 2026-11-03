# Replay Board

**Rewatch a recorded lecture with a clean board beside it.** Upload a lecture video; Replay Board plays
the original, untouched, next to a virtual board that redraws the teacher's writing stroke by stroke,
in sync with their voice, including writing that was hidden behind the teacher while it was written.
Click any piece of writing to jump to the moment it was explained, or switch handwriting to typeset maths.
When the teacher points at the board (resting a finger on a line, tapping a result, sweeping under an
equation), the virtual board shows it as a calm spotlight, circle, underline or arrow on that writing.
Slide lectures get the same treatment: clean slides, and the teacher's cursor or laser turned into the
same animations. And you can **ask Muse** about the lecture: attach a formula from the first minute and a
result from the last, ask how they connect, and get an answer whose times and board items you can click.
Open a drawing **in 3D** and Muse Spark's model of it changes step by step as the teacher explains it, each
change at the word that names it, beside the video as it plays.
When you're done, **export the lecture as slides**: Muse Spark writes a deck that teaches the whole lecture
in order (typeset formulas, worked steps, tables, the teacher's own drawings, key takeaways and a formula
sheet, with what was said in the speaker notes), downloaded as a real PowerPoint.

Built for the Meta Global AI Developer Hackathon. **Meta models do the understanding**; plain code does
the per-pixel measurement.

## How it works

```mermaid
flowchart LR
  V[Lecture video] --> C{Muse Spark<br/>board or slides?}
  C -->|board| S1[SAM 3.1<br/>board corners]
  C -->|board| S2[SAM 3.1 video<br/>person masks]
  C -->|board| S3[SAM 3.1 video<br/>hand masks]
  S1 --> BC[Board capture<br/>code, per pixel]
  S2 --> BC
  BC --> R1[Muse Spark<br/>read each piece]
  S3 --> PT[Pointing samples<br/>code: hand stretched out,<br/>on writing, not writing]
  BC --> PT
  PT --> G1[Muse Spark<br/>gesture meaning]
  C -->|slides| SA[Slide analysis<br/>code: states, clean slides,<br/>pointer track, gesture guesses]
  SA --> R2[Muse Spark<br/>read slides +<br/>gesture meaning]
  V --> T[Muse Voice Transcribe<br/>parallel pieces]
  T --> A[MMS forced aligner<br/>word times]
  R1 --> L[Muse Spark<br/>link writing to speech,<br/>redraw times, chapters]
  G1 --> J
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
| Where the teacher's hand is | **SAM 3.1 video mode** ("hand") on a 3 fps copy | hand masks; code keeps moments where the hand is stretched out onto visible writing and no new ink follows |
| Slide changes, clean slides, pointer | slide analysis (code) | states, live regions, cursor/laser track, gesture guesses |
| What a pointing gesture means | **Muse Spark** with the path drawn on the slide (or on the video frame and the clean board) + the words spoken | spotlight / circle / underline / arrow / pulse, target elements, or "not pointing" |
| Questions about the lecture | **Muse Spark** with the whole lecture (board or slides, chapters, timed transcript) as a cached prefix, plus images of what the student attached | a streamed answer that cites clickable times and board items |
| 3D steps | **Muse Spark** with the drawing (redrawn and the sharp video crop), the writing around it and the timed transcript | a 3D scene as data (shapes, graphs, surfaces, angles, parameters and their ranges to explore) and up to 12 steps that change it, each with a caption and the teacher's cue word |
| Slides export | **Muse Spark** with the whole lecture: one call plans the deck, then one call per part writes it (4 at a time) | sections, which writing goes on which slide, then each slide's title, bullets, formulas, worked steps, tables, definitions, figures and notes |
| Words | **Muse Voice Transcribe** in parallel pieces | turns |
| Word times | **MMS forced aligner** (Meta, torchaudio) | per-word start/end |
| When each element is talked about | **Muse Spark** with elements + timed transcript | spoken ranges, redraw time of hidden writing, chapters |

Every model call goes through one client (`app/core/meta_api.py`) with retries, byte uploads and a disk
cache keyed by model + request, so re-running a lecture costs nothing. Measured calls and prices are in
[docs/meta-api-notes.md](docs/meta-api-notes.md).

### Board capture: "writing stays put, the teacher moves"
Frames are flattened onto the board at 3 fps. Ink is what morphological closing removes; each pixel keeps
counters that only move while the board is visible there (SAM person mask, plus built-in detectors for
large dark things and too-thick strokes; a thick shape that stays put for 1 s, clear of the teacher, is writing:
a filled arrowhead, a heavy underline, two strokes crossing). A pixel is accepted after 1 s of ink and removed after 2 s of
clean board. New pixels are grouped into pieces of writing; removals into erases. Writing that appears
long after its spot was last seen blank was written while hidden, and is redrawn when it was talked about.
Camera drift of a pixel or two is tolerated rather than treated as erase-and-rewrite. The result is a
per-pixel replay log; the player gets per-piece images with drawing-order and erase-time maps.

### Pointing on the board
SAM 3.1 tracks the teacher's hands at 3 fps. Code keeps a moment as pointing only when the hand is stretched
away from the body (not gesturing in front of the chest), its tip is on writing that is on the board at that
moment, and no new ink appears next to it in the following seconds (that would be writing). Nearby moments
become one gesture with a guessed shape (rest, sweep, circle). Muse Spark then sees the video frame and the
clean board with the hand's path drawn on both, plus the words spoken, and decides whether the teacher was
really pointing, at what, and how to show it. Most candidates are talking gestures and are dropped. When the
target is a block of several lines (often everything already on the board when the recording starts), Spark
gets that block zoomed in and says which line or term was meant; code fits the box to the ink. The
annotations are drawn on the virtual board only; the original video is never marked up.

### Self-tests on generated lectures
`tests/gen_board.py` and `tests/gen_slides.py` render lectures with known answers (a teacher who walks in
front of what he writes, an eraser, chalk, fade-ins, camera drift, filled arrowheads and crossing strokes, a
webcam corner, a parked cursor, a laser...). The analysis is scored against them:

```text
python -m tests.selftest_board    ->  136/136 checks (8 variants)
python -m tests.selftest_slides   ->  47/47 checks (4 variants)
```

### What it costs and how long it takes
Measured on an 8-minute 1080p board lecture, nothing cached, on a 4-core laptop:

| | |
|---|---|
| Meta model calls | 100 (Muse Spark 86, SAM 3.1 11, Voice Transcribe 3) |
| API cost | $1.17: base $0.39 (SAM person masks $0.19) + pointing gestures $0.78 (SAM hands $0.29, Muse Spark judging 74 candidates $0.49) |
| First replay ready | 5.0 min |
| Word times refined (MMS) | 5.8 min (the replay is rewritten in place; the player picks it up) |
| Board capture alone | 3–12× real time, depending on what else the machine is doing |

That is about $9 per hour of lecture; with `--no-gestures` (no pointing on the board) about $3 per hour.
Re-running a lecture is free: every model answer is cached.

## Run it (Windows, PowerShell)

Needs Python 3.10–3.13, ffmpeg on PATH, Node 22.18+ (its tests run TypeScript directly), and a Meta
Model API key.

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
.venv\Scripts\python -m app.process my-lecture.mp4 [--start 60] [--end 600] [--deck slides.pdf] [--no-gestures]
```
For a slide lecture you can add the PDF deck: Muse Spark then reads the crisp pages, and a page replaces the
video's slide wherever they match exactly (build steps and pen ink keep the video's image).

Tests:
```powershell
.venv\Scripts\python -m pytest -q             # unit tests
.venv\Scripts\python -m pytest -q -m selftest # generated-lecture self-tests (slow)
cd web; npm test
```

## Layout
```
app/            pipeline, CLI (app.process), server (app.server), slides export (export, pptx_deck), 3D steps
                (popout), smoke test
app/core/       meta_api (one client), board, slides, sam, sam_codec, speech, understand, media, texrender,
                scene_expr (the 3D maths grammar)
web/            player: React + Vite + TypeScript + KaTeX
tests/          generated-lecture self-tests and unit tests
data/           videos, job folders and the model cache (never committed)
```

## Also in the box
- **Ask Muse**: a drawer in the player (button or the A key). Attach anything from any point of the lecture:
  writing from the board (its tooltip's "Ask", or Shift+click), slide text, a stretch of the transcript (select
  it), "this moment", or an item from the searchable list of everything in the lecture, including writing that
  was later erased. Muse Spark gets the whole lecture every time (it fits easily in its context window), so a
  question can link the first minute to the last. The answer streams in; `[5:44]` times seek the video and item
  chips show the writing typeset, jump to it and outline it on the board. The conversation is kept per lecture.
  Cost on the 8-minute lecture: about $0.01 for a first question, $0.006 for a follow-up (Meta's prompt cache
  reuses the lecture), nothing for a repeated one (disk cache).
- **Slides export**: the player's Slides button turns the lecture into a deck that teaches it, in order, the way
  a teacher would prepare one. Muse Spark plans it from the whole lecture (sections, slides, which writing goes
  where), then writes the parts in parallel: titles that say what each slide teaches, bullets, typeset formulas,
  numbered worked steps, comparison tables, definition and key-idea boxes, and figures made from the teacher's
  own drawings (the ink lifted off the board onto a clean background, not a photo of it). Code checks every
  answer: each piece of writing that matters must land on a slide, the LaTeX must be valid, a formula made of
  plain numbers is worked out (a rounded result gets ≈, a false one is left out), and text must fit the layout.
  The **PowerPoint** (python-pptx, 16:9) has real slide titles, native bullets and tables, an agenda, key
  takeaways, a formula sheet, and speaker notes with what was said and when; equations are typeset by MathJax
  into sharp pictures that keep their LaTeX as alt text. The page previews the same slides, with **Present**
  (full screen) and **Save as PDF**. Cost on the 8-minute lecture: about $0.08 and 25 s, once; then it's cached.
- **Study notes**: the player's Notes page lists every line of writing (or slide element) by chapter, typeset with
  KaTeX, with what was said; each time links back to that moment in the player. Download as Markdown or print.
- **Typeset by line**: a line written in bursts (several pieces) is read by Muse Spark as one line and typeset whole
  once it is finished; until then its pieces draw one by one, in time.
- **3D steps**: a drawing or graph that has a 3D view gets a small "3D" badge on the board, and a "Watch in 3D"
  pill while the teacher explains it. Click it and the 3D view takes the board's place while the video keeps
  playing; the model changes as the teacher explains it: in a crystal lecture one cell turns from cubic into
  tetragonal, orthorhombic, rhombohedral and monoclinic, each at the word, with its angles marked and measured
  live ("β 75°"). Muse Spark writes the scene as pure data (shapes, graphs, surfaces, curves, angles,
  parameters, and steps that show, hide, highlight, move the camera or ease a parameter). The server checks
  every maths expression against a small grammar (the player compiles the same grammar with its own parser,
  never as code), evaluates every state of every step, moves each step onto the time of its cue word from the
  aligner, drops steps that change nothing, shows what a caption names, keeps angle marks apart, picks a camera
  that shows the step's subject (from the other front corner when a vector points straight at the usual one),
  adds the degree sign a caption leaves out, and narrows the sliders' ranges to shapes that can exist. A strip shows every
  value as it changes; **Try it** gives a slider per value so a student can explore ("Back to the lecture"
  returns). Step chips and the `[` `]` keys jump the video to a step; "Play the explanation" plays it with the
  lecture; a drawing never talked about gets steps you play yourself. One scene per drawing, made on the first
  open (30–70 s, about $0.03–0.10), then cached; `python -m app.popout <job> <piece>` makes them ahead of time.
  three.js is loaded only when a 3D view opens. `tests/gen_scene_lectures.py` makes narrated board lectures (a
  3D vector, a cylinder's volume, a saddle surface) with a truth file of what must hold at each spoken stage, to
  check the 3D steps end to end.

## Limits
- The camera must stay still (a tripod or a fixed classroom camera). Panning or zooming cameras are not handled.
- MMS word alignment runs on the CPU at 1–3× real time; for long lectures the replay appears first with
  turn-level word times and is refined in place.
- Handwriting reading is as good as the frame: very small or blurry writing may be read partly.

## Privacy and video rules
The original video is never altered: no outlining, blurring or removing the teacher. Videos and anything
made from them stay in `data/`, which is not committed.
