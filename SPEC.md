# Replay Board: build spec

## 1. The product

Upload a recorded lecture. Replay Board plays the **original video, untouched**, next to a **clean virtual board** that redraws the teacher's writing and pointing in sync with their voice, including writing that was hidden behind the teacher when it was written. Click anything on the board to jump to the moment it was explained, or switch handwriting to typeset maths.

It is for students rewatching recorded lectures, where the teacher blocks the board, the camera can't read it, or the cursor flies around the slides.

The entry is for the Meta Global AI Developer Hackathon:
- Judging: Impact 30%, Meta tooling 25%, Technical depth 20%, Demo 25%.
- Submission: a public repo, a 3-minute demo video, new work only.
- Deadline: **Wed 4 Nov 2026, 9 PM ET (Thu 5 Nov 02:00 Dublin)**. The owner wants to submit on Wednesday evening, Dublin time.

## 2. Hard requirements (from the owner, not negotiable)

- Recorded video upload only. No live mode. **No YouTube links or downloads.**
- **Never outline, blur, remove or change the teacher in the original video.** It plays untouched; everything is rebuilt beside it.
- Two lecture types:
  - (A) a filmed board, white or chalk;
  - (B) a screen recording of slides with voice-over. Optionally the PDF deck can be uploaded too.
- Writing hidden behind the teacher appears as soon as it becomes visible, and is redrawn at the time it was actually talked about.
- Fast: an hour-long lecture processes in minutes, not hours.
- **Meta models do the understanding.** Other tools may support them but never replace them.

## 3. Architecture: code measures, Meta models decide

Per-frame, pixel-level measurement is plain code, because it runs on thousands of frames and must be exact, fast and free. Every judgement about *meaning* goes to a Meta model.

| Step | Done by | Notes |
|---|---|---|
| Board or slides? | **Muse Spark**, 4 sampled frames | JSON: type, board colour, has webcam corner |
| Find the board | **SAM 3.1**: "whiteboard", then "blackboard", "chalkboard", "projector screen" | Mask, then the convex hull of its largest region, then 4 corners (approxPolyDP; minAreaRect fallback). Report the corners in video pixels. |
| Who is in front of the board | **SAM 3.1** "person" masks | Sample 1–2 fps, or use video mode if the API supports it, and dilate to cover motion between samples. This is the occlusion mask for board capture; keep the built-in detector (section 5.1) as a fallback. |
| What is written, where and when | Board capture (code, section 5.1) | |
| Read every piece of writing and every slide | **Muse Spark**, in parallel | Per element: kind, text, LaTeX, box |
| Slide changes, clean slides, pointer track | Slide analysis (code, section 5.2) | |
| What each pointing gesture means | **Muse Spark** | Input: the slide image with the pointer path drawn on it, the words spoken meanwhile, and the code's gesture guess as a hint. Output: spotlight / circle / underline / arrow / pulse / none, plus target element(s). |
| Words | **Muse Voice Transcribe** | Chunks of at most 9 min, sent in parallel |
| Word times | **MMS forced aligner** (Meta, via torchaudio) | Falls back to proportional timing |
| When each element is talked about | **Muse Spark** | Input: elements (text, when written/shown) plus transcript with word times. Output: spoken time ranges per element, the redraw time for hidden writing, chapter titles. |
| Stretch: 3D pop-out | **Muse Spark** returns a JSON scene, rendered with three.js | For drawn 3D shapes, surfaces and graphs |

Budget: $150 of API credits. Measure cost and latency per call in M0, and **cache every model answer on disk** (key = hash of model + prompt + inputs) so re-runs are free.

## 4. Meta Model API (verify against Meta's docs in M0)

- Base URL `https://api.meta.ai/v1`, authorised with `Authorization: Bearer $MODEL_API_KEY`. The key lives in `.env`.
- Model ids, overridable in `.env`: `muse-spark-1.3`, `muse-wonder-1.0`, `muse-voice-transcribe-1.0` and `sam-3.1` (some pages say `sam-3-1`; try both).
- **Chat completions** (`POST /chat/completions`):
  - images go as `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` content parts;
  - structured output uses `response_format: {type:"json_schema", json_schema:{name, schema, strict:true}}`;
  - strict schemas need `additionalProperties:false` and every property listed in `required`.
- **Voice Transcribe** (`POST /asr/transcribe`, multipart):
  - part `request` (JSON): `{model, audioEncoding:"WAV", mode:"ENDPOINTING"}`;
  - part `audio`: a mono 16-bit PCM WAV at 16 or 24 kHz, **at most 10 min and 32 MB**;
  - the reply has `turns[]` with `startMs`, `endMs` and `transcript`. Timing is per turn only: no word times.
- **SAM 3.1** (`POST /responses`):
  - `input:[{type:"message", role:"user", content:[{type:"input_text", text:"whiteboard"}, {type:"input_image", image_url:"data:..."}]}]`, with `metadata:{include_confidence:"true", mask_encoding:"one_bit"}`;
  - the output text uses special tokens; parse it with Meta's `meta-sam-parser` package (`image_segmentation_format().create_parser()`, `push`, `finish(CompletedOutcome())`, `decode_mask_to_raster`);
  - every object has a box plus a mask covering the box;
  - a box prompt can be built with `build_object_prompt`.
- Client rules:
  - one module for all model calls;
  - retry 408/429/5xx with exponential backoff, about 60 s in total, plus jitter, honouring `Retry-After`;
  - **upload bytes, not open file handles** (a retry must resend the whole upload);
  - check the answer is JSON and a dict.

## 5. Core algorithms

These were proven in a prototype on generated lectures and a real 8-minute 1080p lecture: 34 pieces of writing, exactly the 2 real erases, about 9x faster than real time. Implement them as described.

A note on units: `s` = analysis width / 800 for the board and / 960 for slides. Every pixel size below scales with `s` (areas with `s²`).

### 5.1 Board capture: "writing stays put"

Writing stays where it is; the teacher moves.

**Frames and ink**
- Sample at **3 fps**. Flatten the board to **800 px wide** with a perspective warp from the 4 corners (decode at 1280 px first). With no corners, use the whole frame scaled to 800 px wide.
- Decoding (also SAM's low-fps copies): on Windows, decode on the GPU (D3D11VA; `VIDEO_DECODE=auto`, `cpu` turns it off) only where it gives the very same pixels as software, because model answers are cached by the bytes sent. That means 8-bit 4:2:0 H.264, HEVC, VP9 or AV1, with no turn or flip. Once per file, 8 frames are decoded both ways and compared. Damaged data (`-xerror`) or any GPU failure switches that file to software, carrying on from the next frame.
- Board type: take the median brightness of the middle half of the board at 9 moments spread across the clip (seek to each; don't use the first frame). If it is above 110 it is a whiteboard, otherwise chalk.
- Grey image: on a whiteboard, `min(B,G,R)` (coloured markers come out dark); on chalk, `255 - max(B,G,R)`.
- `closed` = morphological closing of the grey image with an ellipse of 21 px.
- `contrast = closed - grey`. Set it to 0 in a border margin of 0.8% of the width.
- Ink: `contrast > 22`.
- Hysteresis: an already-accepted pixel stays ink while `contrast > 12`, so stroke edges don't flicker.

**What's in front (hidden pixels keep their state)**
- Primary: SAM 3.1 person mask, dilated.
- Fallback 1, large dark things: keep a per-pixel `bg_ref`, initialised to `closed` and then updated each step as `max(closed, bg_ref - 0.5)`. Where `bg_ref - closed > 45`, dilate by an ellipse of 15 px.
  - Not on a chalkboard where SAM looked (it is still used where a SAM chunk failed, and on chalkboards without a person mask). There the grey is `255 - max(B, G, R)`, so a dark shirt is brighter than the board: it lifts `bg_ref` as it passes, and the clean board behind it counts as "in front" until `bg_ref` sinks back, half a minute or more after the teacher left (measured on a real 27-min chalkboard lecture: writing shown a median 22 s late, 95 pieces wrongly tagged "written while hidden"; without it, 2).
- Fallback 2, too thick to be a pen stroke: where the distance transform of the ink exceeds `max_stroke/2` (max_stroke = 11 px), dilate by an ellipse of 31 px. Except thick ink that stays put: a thick pixel whose contrast against the clean board changes by at most 10 levels for 3 samples in a row is writing (a filled arrowhead, a heavy underline, strokes crossing), and it and thick pixels within 9 px of it leave this mask. Its count may only start while the whole thick shape is more than 6 px from the dilated person mask and the large-dark mask, so a still hand or sleeve that SAM missed never counts. Shapes wider than the closing (21 px) are not ink at all.
  - The replay shows ink this mask held back from when it would have been accepted as thin ink: a second count like `on`, in view of everything except this mask, and the pixel is shown from when it reached 3 (never later than its real acceptance, which still waits for the shape to stay put). That count restarts when the contrast of the pixel, or of any counted pixel within a 9 px ellipse around it (`jump_zone`: covers a replacing stroke up to 8 px wide), changes by more than 10 levels between samples (a stroke replacing a thing of another width or darkness changes only some of its rows). It is dropped for ink that was part of a too-thick thing whose middle turned into clean board without being accepted (a hand or sleeve SAM missed). Behind the teacher it waits, like `on`. Not caught: a thing SAM missed that is replaced by ink of exactly its width and darkness.

**Accepting and removing pixels**

Counters move only where the board is visible:
- `on` counts consecutive "ink" samples, `off` counts consecutive "clean" samples, and `last_clean` holds the time of the last clean sample.
- A pixel is **accepted** after 3 ink samples in a row (1 s), and **removed** after 6 clean samples in a row (2 s).

**Camera drift tolerance (essential on real video).** Real cameras drift by about 1 px, from a tripod settling or autofocus, and strokes are only 1–3 px wide. Without this, every stroke "erases and rewrites itself". So:
- an accepted pixel counts as ink if any faint ink (`contrast > 12`) is within a 2 px radius (5×5 ellipse);
- new ink within 2 px of accepted ink joins *that* ink's piece instead of starting a new one. To find the owner, max-dilate an image of piece ids.
- such ink is left out of the piece's image (the original marks stand for it) only when the ink beside it was accepted at least 2.5 samples before: a camera move takes 3 samples to re-accept ink. Ink beside ink accepted a sample or two before is the rest of fresh writing (a stroke beside an arrowhead the thick mask held back a sample longer).

A global image alignment was tried and did **not** help: the drift is not a pure shift, and the reference follows it. Don't rebuild that.

**Start of the clip**
- Ignore frames until the picture settles, meaning the middle's brightness is within 25 levels of the typical brightness (this covers fade-ins).
- After that, skip any sample that does not show the board: a title or end card, an advert, in an edited video. Its visible board's median grey (without the dilated person mask, when at least a tenth of the board is in view) is further than `max(40, typical / 2)` from the typical brightness. A skipped sample changes nothing: no counter moves, so a card is never writing, never erases the board and never gives the board its colour (the colour comes from the last sample that showed it). Measured margins: normal clips stay under half the threshold (an exposure swing reached 48 of 97); every card sample is far past it.
- Anything accepted in the first 2.5 s after that is **pre-existing**: show it, but it is not a timed event.
- Content never seen blank since the start that touches pre-existing content is pre-existing too.

**Pieces of writing (events)**
- Newly accepted pixels are grouped into connected components after a 13 px dilation.
- A component joins a piece updated within the last 3 s whose box, padded by 25 px, touches it. Otherwise it starts a new piece.
- A component under 25 px joins any touching piece, however old.
- Each piece records:
  - `t_seen` and `t_commit`, both taken from when **10% of its pixels** are shown in the replay (see Fallback 2), so one stray early pixel doesn't set the time, and `t_shown` from when 95% are;
  - `written_after` (the median `last_clean` of its pixels; the minimum across batches);
  - `revealed`, meaning it was written while hidden: at least 75% of its pixels were out of view for more than 10 s between when their spot was last seen blank and when they were in view as ink (before any wait for thick ink to stay put). The hand, arm or body covers a spot before the pen gets there, so a few seconds out of view say nothing about when it was written (measured: up to 8.3 s for writing seen as it was written, 16–17 s for writing done behind the teacher). A spot never seen blank since the clip started counts as not out of view: the ink may have been there all along. That hidden ink must also have come into view in one go (10–90% of it within 1 s), as when the teacher steps away: ink that comes into view bit by bit as the pen moves on was written in view, however long he stood on the spot first (measured: 0–0.67 s for writing done behind the teacher, 2–2.3 s for a 2.5 s line written with his body at the pen). The rule errs late, never early: hidden writing that was out of view under 10 s, came into view in parts, or sat behind the teacher since the clip started is shown when it comes into view, without the tag. A revealed piece also records `redraw_after`, the latest time any of its hidden ink's spots was seen blank: it is never redrawn before that.

**Not writing (drop it)**

These kinds of piece are dropped:
- under 25 px;
- more than 60% of its pixels removed within 6 s of being accepted (`min_life`);
- a long hairline: at most 1.5 px thick and at least 15× longer than thick;
- lying entirely inside the outer **3% band** of the board area (frame or wall edges).
- a typed caption burned into an edited video. Writing comes a stroke at a time; a caption comes all at once, on board that was seen blank a moment before, and goes within a minute.
  - A **burst** step is one that accepts at least 600 px (at 800) of ink that is not drift, was accepted on time (not after the wait for too-thick ink: typed text is thin) and whose spot was seen blank at most `accept_n` samples before. Only steps after the first `pre_window` seconds count. Measured: writing reaches at most 252 px a step on the self-test, 44 on the owner's clips and 214 on an edited lecture.
  - A piece is part of the burst if it got at least 150 px of that ink in that step. Captions got 655 px and more each; other pieces, including writing done as a caption came up, got at most 59.
  - The ink such a piece accepts within 2 s of the burst (not drift) is overlay, all of it, if at least half of it is gone within 60 s. Measured: captions 0.62–1.0. Some caption ink stays accepted where chalk lies under the text. Ink that came into view all at once and stays (a cut to another board, a banner leaving) measured 0.
  - A piece that is at least 60% overlay also loses the rest of its ink that is gone within 60 s.
  - Overlay ink is taken out of its piece before the piece is judged, so writing a caption joined keeps its own ink. It counts as short-lived for erases, so a caption leaving is not an erase.
  - A streamed partial leaves out window ink that may still turn out to be overlay, and its ready line stays before that ink. After ink that appears at once and stays, the stream lags about a minute; no step on the owner's clips comes near a burst.

When replaying, also leave out any mark that will be removed within 6 s, since the whole video is known by then. Leave out pre-existing marks lying entirely in the 3% band too (a ruler hanging on the board).

**Erases**
- Components of removed pixels, at least 60 px, merged with a recent erase within 2.5 s if they touch.
- Discard an erase if more than 60% of its pixels lived less than 6 s.

**Output**
- A per-step replay log of `(t, added pixel ids, their piece ids, their colours, removed pixel ids)`. The board at any time is rebuilt from it; ink colour is the original pushed away from the board: darkened ×0.72 on a whiteboard, and on a chalkboard lifted towards white by the same share (`255 - (255 - c) × 0.72`; darkening chalk drew it dim, close to the board's colour). An add's `t` is when it is shown (Fallback 2), never later than when it was accepted; lifetimes (`min_life`), removals, erases and drift go by acceptance time.
- Speed at 800 px: the prototype did 9–15x real time; the full capture measures 6–7.5x in software on 1080p lectures on a 4-core laptop (6.1–6.6x over a whole hour), and 8–9x with GPU decoding.

### 5.2 Slides: states, clean slides, pointer

**Working copy and webcam**
- First make one **working copy** with ffmpeg: cut to the requested part, `fps=12`, scaled to **960 px** wide, x264 crf 12 yuv444p. Every later pass reads that, so the original is decoded only once.
- **Webcam or embedded video ("live" regions):** at 2 fps, measure per pixel:
  - `busy`: the share of frame pairs changing by more than 12;
  - `flicker`: the share of *quiet* pairs (under 5% of the screen changed) changing by more than 6.
- A live region is:
  - a component with a busy core (`busy > 0.35`, closed 15 then opened 5) covering at least 0.4% of the screen; **or**
  - a flicker component covering at least 1% of the screen and filling at least 40% of its box. This catches a person who barely moves.
- Grow each box outward, by at most 15% of its size, to the **furthest** strip that still belongs. A strip belongs if more than 50% of it differs from the slide background by more than 25 levels, or more than 10% of it flickers more than 2%. Then ignore and blank the whole box.

**Slide states (4 fps)**
- A content change is a changed component (diff > 30, outside live regions) with an area of at least 1500 px or a side of at least 60 px. Cursor-sized changes don't count.
- A change counts once nothing has changed for 0.5 s.
- It is a **new slide** if at least 25% of the screen changed, or at least 25% of the old slide's *text pixels* changed (`|ref - medianBlur(ref,31)| > 40`). Otherwise it is an **addition** (a bullet appearing, pen ink).

**Clean slides**
- The per-pixel median of up to 11 samples spread across the settled time, built as you go so only one state's frames are in memory. Blank the webcam box with the background colour.
- A blank or black state (std < 4) is not saved for reading.

**Pointer (12 fps)**

Changes are measured against the clean slide: `max over channels |frame - clean| > 30`, outside live regions.

- **Laser** (any colour): inside a change component of at most 60 px, the pixels with max channel > 170, `max - min` > 120 and changed.
  - It must have at least `max(10, 15 s²)` px, a box of at most 30 px, fill at least 30% of that box, and an aspect ratio of at most 3.
  - Coloured ink next to a moving cursor is not a laser; that false detection once broke a sweep.
- **Cursor** candidates are changed components of 25–700 px with a side of at most 60 px.
  - Keep up to 6 learned cursor shapes (grey patch + mask, mask at least 20 px, std at least 40).
  - Appearance = mean |crop - shape| over the shape's mask, divided by 255.
  - Pick the nearest candidate with appearance ≤ 0.12. Otherwise accept the best (appearance + distance/900) only if it is within 60 px, or has appearance ≤ 0.35 and contrast ≥ 100.
  - If the cursor has been lost for 0.5 s (or never seen), re-acquire the most contrasty candidate with contrast ≥ 100. This fixed tracking that got stuck after a fade-in.
  - If nothing changed and the last box still matches (< 0.08), the cursor is parked there.
  - Take the **position from template matching** within the box ±8 px. Using the blob centre jumps 10 px when the cursor crosses light text.
  - During a transition there is no pointer.
- **Cursor parked in a clean slide:**
  - Find it with the learned shapes (masked SQDIFF, rms ≤ 0.08), only near places the cursor was seen: inside the state, plus the last sighting before it and the first after it.
  - Inpaint it out (Telea, radius 3), then re-track that state. Repeat up to 3 rounds.

**Gesture guesses (hints for Muse Spark)**

Split the track at gaps over 0.35 s or tool changes. Smooth with a 3-point average. Moving means faster than 55 px/s.

- **Hover:** resting for at least 0.6 s. It is *parked* if longer than 8 s, or if there's no content within 40 px.
- **Circle:** a window of at most 4 s where the heading turns at least 300° consistently: the sum of turns ≤ 1.35× the absolute sum, the box's short side ≥ 18 px, and the gap between start and end ≤ 0.6× the long side. Merge overlapping windows.
- **Wiggle:** 0.4–2 s, a box of 12–150 px, at least 3 reversals, path length at least 2.5× the box, speed at least 80 px/s. Trim the resting ends.
- The rest is split into straight pieces with RDP (8 px):
  - **sweep:** straightness ≥ 0.85, |dx| ≥ 2.5|dy|, at least 80 px, at most 700 px/s, at least 0.35 s;
  - **move:** straightness ≥ 0.75, at least 60 px;
  - an **arrow** is a move that starts after a rest of at least 0.25 s on content and ends in a (non-parked) hover. Other moves are just the pointer travelling.
- Merge consecutive sweeps on the same line.
- Underlines snap 4 px below the nearest text line.

### 5.3 Speech

- Audio: `ffmpeg ... -ac 1 -ar 16000 -af aresample=async=1:first_pts=0 -f s16le`. The `first_pts=0` keeps sample 0 at time 0 on the *video's* clock when the sound track starts late.
- Split into pieces of at most 540 s, cut at the quietest point (300 ms-smoothed 50 ms energy) within the last 30 s before each limit. Transcribe in parallel and add each piece's offset.
- Word times use torchaudio `MMS_FA`: `get_model()`, `get_tokenizer()`, `get_aligner()`, aligning each turn with 0.15 s of padding.
  - **torch/torchaudio must be below 2.9**: the docs say these tools were removed in 2.9.
  - Words are turned into spoken form: lowercase a–z and apostrophes, numbers spelled out with num2words ("2a" becomes "two a").
  - Symbols like "=" get the gap between their neighbours.
  - Fallback: share each turn's time by word length + 2.

### 5.4 Self-test with generated lectures (build it with the core; it caught most real bugs)

Generate videos with known answers, score the analysis against them, and keep the result at N/N passing.

- **Board lecture** (1280×720, 25 fps, 38 s): a whiteboard with a frame and a wall around it.
  - 5 items written progressively left to right: script-font text in 3 colours, plus an underline.
  - A teacher (body, head, an arm to the pen tip) who moves around. Item 3 is written while he stands in front of it and is uncovered at about 23.6 s. Item 1 is wiped off with an eraser at 33–35 s.
  - Noise σ 2.5 and a 1% brightness wobble.
  - Variants: chalk (ffmpeg `negate`), a fade-in from black, a camera drifting (+2/−1.5 px from 20 s, +2.5/−2 px from 28 s), and the same lecture analysed at 1200 px.
  - Checks:
    - every item is found, at the right time;
    - the hidden item is flagged and shown within a few seconds of being uncovered;
    - no extra items and exactly one erase;
    - pixel recall ≥ 0.85 and precision ≥ 0.9 against the true ink (shifted by the drift where it applies) at 32 s and at the end;
    - at least 2x real time.
- **Slides lecture** (40 s):
  - 3 slides with build steps (6 states, including a crossfade), a noisy webcam corner, and red pen ink;
  - a cursor doing a hover, a sweep, a circle, a wiggle and an A→B move with rests, parked for whole slides at times;
  - a laser doing a circle and a hover over coloured bars.
  - Variants: dark theme (`negate`), a fade-in (check gestures only), and a still slide with a picture-in-picture where the person barely moves (expect 1 state, the webcam found, no gestures).
  - Checks:
    - every state at the right time and of the right kind, with no extras;
    - the webcam box found (overlap ≥ 0.7);
    - each gesture found with the right type, tool and time (overlap ≥ 50%);
    - the A→B move flagged as an arrow, and no pointer travel flagged as one;
    - no false circles or wiggles;
    - no cursor left in clean slides;
    - at least 1x real time.

## 6. Output: `replay.json` (contract between backend and player)

```json
{
  "version": 1,
  "video": {"file": "lecture.mp4", "duration": 478.1, "width": 1920, "height": 1080},
  "type": "board",
  "board": {
    "size": [800, 322], "mode": "white", "corners": [[55,160],[1815,150],[1815,860],[55,865]],
    "background": "#f7f7f5",
    "pieces": [{
      "id": 12, "box": [430, 14, 61, 60],
      "image": "pieces/012.png", "order": "pieces/012_order.png",
      "t_seen": 63.0, "t_shown": 63.7, "written_after": 60.3, "revealed": false, "removed_at": null,
      "read": {"kind": "equation", "text": "A = (1 + 1/12)^12", "latex": "A=\\left(1+\\tfrac{1}{12}\\right)^{12}"},
      "spoken": [{"start": 61.2, "end": 70.4}],
      "redraw_at": 61.2
    }],
    "erases": [{"t": 142.0, "box": [527, 117, 26, 52]}]
  },
  "slides": {
    "size": [960, 540],
    "states": [{"id": 0, "kind": "slide", "t_start": 0.0, "t_end": 8.0, "image": "slides/000.png",
                "elements": [{"id": "s0e1", "kind": "heading", "text": "...", "latex": "", "box": [60, 40, 500, 60]}]}],
    "annotations": [{"type": "circle", "tool": "cursor", "t_start": 12.5, "t_end": 14.3,
                     "target": ["s1e3"], "box": [72, 224, 340, 69], "from": null, "to": null, "why": "..."}]
  },
  "transcript": {"turns": [{"start": 0.4, "end": 4.4, "text": "..."}],
                 "words": [{"w": "so", "s": 0.40, "e": 0.67}]},
  "chapters": [{"t": 0.0, "title": "Logs of negative numbers"}]
}
```

- `pieces/NNN.png` is an RGBA crop of the piece's strokes, made from the replay log, with a transparent background.
- `pieces/NNN_order.png` is an 8-bit drawing-order map: accept time first, then left to right. The player reveals pixels whose order is at or below the progress, which gives the stroke-by-stroke look. It steps one order level at a time and draws a level only once all of it is due, so no pixel is drawn more than one capture step (1/3 s) before the replay log shows it. In a revealed piece, ink that may be redrawn early (its spot was last seen blank before the redraw can start) comes first within each capture step.

### Streaming replay (board lectures)

While a board lecture is processed, `replay.json` is written in parts, so the player opens as soon as the board is found and plays the original video at once; the clean board fills in as capture goes (capture runs 4-9x real time). Measured at $0 on the 8-minute test lecture with saved answers: first partial after 7.8 s, first writing after 30 s (once the first SAM person chunk is back), then a partial about every 8 s, the whole board at 79 s, the final replay at 132 s. The final `replay.json` is exactly what it is without streaming. `STREAM_REPLAY=off` in `.env` turns it off (no partial writes, no `stream/` folder). Slide lectures never stream.

- **A partial replay** has the final shape plus `"partial": {"seq", "ready_until"}`:
  - `seq` starts at 1 and goes up with every write of `replay.json` during the job; `ready_until` is a time on the video's clock up to which the board is not expected to change.
  - The first partial is written right after the board is found, before capture: no pieces, `ready_until` = the clip's start. `video` (with `end` the clip's end), `type`, `board.size`, `board.corners` and `board.mode` are final from it; `board.background` may change.
  - `board.pieces`: the capture so far, only pieces seen by `ready_until` whose first ink is shown by then too. Ids are capture's own (stable). Pictures are `stream/<sha1, 12 hex>.png`, named by their bytes and never rewritten. `read` is a partial reading when there is one, else empty; `spoken` is `[]` and `redraw_at` null. `lines`, `annotations` and `chapters` are `[]`.
  - `transcript`: all of it with proportional word times once transcription is done (`timing` "none" when there was no speech), else empty with `timing` "proportional".
  - After capture, one more partial has the whole board (`ready_until` = the clip's end); once the read stage is done it gets those readings.
- **Ready until:** capture's time minus 7.67 s (an add can still be dropped as short-lived until `min_life` plus the removal wait after it was accepted), and never past thick ink not accepted yet (it may be shown from its due time) or young ink shown early. It only grows.
- **Checkpoints** in capture (`capture_board(checkpoint=...)`): every 30 s of video, at least 8 s of wall-clock time apart, and at most 10% of capture's time. Each finishes a copy of the loop's state; capture's result is the same with or without them (tested on the self-test clips and on 2 minutes of a real lecture).
- **A piece once shown stays:** a growing piece can fail capture's shape checks for a while or have its seen time move past `ready_until`; it stays as last shown until it is back, unless an erase after it was seen covers it. The final replay decides.
- **Partial reads:** pieces new in a partial are read in the background (two calls at once, up to 10 pieces a call) and the partial is written again with their readings. New pieces wait until there are 5, or the first has waited 20 s, because every call also sends the whole board. They are extra calls (about $0.01 a call); a failure leaves the pieces unread and never fails the job. When capture is done, reads not started are dropped (the read stage reads every piece).
- **One writer:** every write of `replay.json` takes one lock; after the final one nothing partial is written. If the job fails or is stopped (Ctrl+C) while the replay is partial, `replay.json` and `stream/` are taken away and the job reads as failed, as before. `stream/` is also removed once the job is done.
- **Re-runs:** a run in a job folder that already has a final `replay.json` (`python -m app.process` on the same video) does not stream: the old replay stays until the new one replaces it, also if the run fails.
- **Interrupted jobs:** status.json names the process running the job (`pid`). When the server starts, a job left queued, or running by a process that is gone, is marked failed ("processing was interrupted…") and its partial replay taken away; one stopped while refining word times is done with the replay it has.
- **status.json** gets `"replay": {"seq", "partial", "ready_until"}` after every write (partial, final and the refined rewrite); `GET /api/jobs/{id}` passes it through and `ready` still means a `replay.json` exists.
- **Gates while partial:** 3D views, writing or downloading the slide deck and the notes answer 409 "Available when processing finishes"; the 3D list is empty. Ask Muse answers from what there is. The player disables Notes, Slides and 3D, freezes the board at `ready_until` with a calm note when the playhead is past it, and shows how much is ready.

## 7. The app

- **Backend:** Python, FastAPI and uvicorn.
  - One job folder per upload under `data/jobs/<id>/` (gitignored), with a background worker and progress reported per stage.
  - Endpoints: `POST /api/jobs` (upload), `GET /api/jobs/{id}` (status), `GET /api/jobs/{id}/replay.json`, plus static assets.
  - CLI: `python -m app.process <video> [--deck slides.pdf]` writes the same job folder. It is the fastest way to iterate.
- **Player:** React, Vite and TypeScript, with KaTeX (and three.js only for the stretch goal).
  - Left: the original `<video>`. Right: the board canvas, or the slide plus an SVG animation layer.
  - One shared clock keeps the video, board and transcript in sync while scrubbing.
  - Click a piece or slide element to seek to its first `spoken` time.
  - A handwriting/typeset toggle draws KaTeX in the piece's box.
  - A transcript panel highlights the current word; clicking a word seeks.
  - Animations:
    - spotlight: a dimmed slide with a soft hole;
    - circle: an ellipse drawn with stroke-dashoffset;
    - underline and arrow: drawn progressively;
    - pulse: rings.
  - Hidden writing is drawn at `redraw_at`, with a small "was hidden" tag the first time. `redraw_at` is never before the piece's `redraw_after`; only what the camera showed in the first 6 s after the piece came into view, and of that only the order levels up to `redraw_to` (ink whose spot was last seen blank before the redraw can start), is redrawn early; ink it gained later or that was seen as it was written is drawn when the replay log shows it. The redraw takes `redraw_secs`, the time writing that ink takes at the speed measured on real lectures (88 px/s at 800 px wide), at least 1.2 s and at most 6 s; the writing speed never stretches it past 0.6 s after the camera shows it all.
- **Upload page:** drag and drop, the stages with their progress, then the player.

## 8. Milestones (each ends with something the owner can see working)

- **M0 (Thu):**
  - repo, git, `.gitignore`, `.env.example`, Python venv;
  - the Meta API client from section 4, with a disk cache;
  - a smoke test with the real key: one tiny call each to Muse Spark (with an image), Voice Transcribe and SAM 3.1. Report the model ids, latency, cost per call and any format fixes.
- **M1 (Fri): board lectures, backend.**
  - Board capture (5.1) with its generated-lecture self-test (5.4) passing first.
  - Then the full pipeline, `python -m app.process lecture.mp4`, which writes `replay.json`. It uses SAM for the corners and person masks, Muse Spark reading, Voice Transcribe plus MMS, and Muse Spark linking.
  - Test on the owner's real clip in `data/`.
- **M2 (Sat): player for board lectures.** Synced video and board, stroke-by-stroke redraw, click to jump, typeset toggle, transcript.
- **M3 (Sun): slides, end to end.**
  - Slide analysis (5.2) with its self-test, Muse Spark gesture meaning and slide reading, and the player's animation layer.
  - Optional: an uploaded PDF deck replaces the video frames.
- **M4 (Mon):**
  - 3 real lectures (a board, slides, a long one); fix what breaks;
  - speed (parallel calls, measured end to end) and the upload UX;
  - stretch: 3D pop-out.
- **M5 (Tue):**
  - features frozen at noon;
  - a README with an architecture diagram and how each Meta model is used;
  - a demo script for the 3-minute video, and the repo made public.
- **Wed:** the owner records the demo and submits.

## 9. Repo rules

- Never commit `.env`, uploaded videos, or anything made from videos the owner doesn't own. Test clips go in `data/`. **The demo uses the owner's own recordings.**
- Keep the self-test passing and add tests for new logic. Measure on real clips instead of guessing.
- Windows 11 dev machine: give commands for PowerShell; ffmpeg must be on PATH.

## 10. Gotchas already hit (don't relearn these)

- Camera drift of about 1 px makes strokes "erase and rewrite" themselves. Use the 2 px tolerance (5.1).
- Black first frames from a fade-in. Judge modes from several moments, and wait for the picture to settle.
- Edited videos (YouTube lectures) end on a black card: it was taken for the board, wiped every piece at the end and made the background black. Skip samples that don't show the board (5.1).
- On a chalkboard a dark shirt looks bright (the grey is `255 - max`), so a running-max reference is lifted by every pass of the teacher. Don't use the large-dark fallback there when SAM gives the person.
- On a chalkboard, typed white captions in an edited video look like chalk and became writing, and their leaving became erases. They come in one burst and go within a minute: drop them (5.1).
- Board frame and wall edges inside the analysed area turn into fake writing. Give the board corners and drop anything in the outer 3% band.
- A webcam person who barely moves. Use the flicker rule and the furthest-strip growth (5.2).
- A cursor parked for a whole slide gets baked into the clean slide. Remove it with the template, then re-track.
- Retries that resent an already-read file uploaded empty audio. Send bytes.
- `cv2.imread`/`imwrite` fail silently on non-ASCII Windows paths. Use `cv2.imencode`/`imdecode` with `Path.write_bytes`/`np.fromfile`.
- `.env` saved by PowerShell is UTF-16, and Notepad may add a BOM. Decode with `utf-8-sig`, then fall back to `utf-16`.
- A sound track that starts late puts words seconds off. Use `aresample first_pts=0`.
- GPU-decoded frames are not turned or flipped by ffmpeg's autorotate (a flip reports rotation 0), and the GPU hides damaged data differently from software. Gate on the display matrix and use `-xerror`.
- torchaudio 2.9 removed MMS_FA, so pin it below 2.9.
- Image, frame and audio handling must survive: rotated phone videos, odd sizes, variable frame rate, no audio track, clips under 3 s, `--start` past the end, `--end` before `--start`, and paths with spaces or accents. Give clear one-line errors, not tracebacks.
