# Demo video script (3 minutes)

Use **your own recordings only**: one board lecture (ideally where you step in front of what you write,
point at a result with your arm stretched out, and erase something) and one slide lecture with a cursor or laser. Process both before filming
(`python -m app.server`, drop the files), so the player opens instantly; show the upload with a short clip.
Record at 1920×1080, browser zoom 100%, dark theme. Close other tabs; hide the bookmarks bar.

| Time | On screen | Say (roughly) |
|---|---|---|
| 0:00–0:15 | A recorded board lecture where the teacher stands in front of the writing; then a slide recording with a cursor flying around. | "Rewatching a recorded lecture is hard: the teacher blocks the board, the camera can't read it, the cursor never stops moving." |
| 0:15–0:30 | Replay Board home page. Drag a short clip in. The stages tick: board found, person tracked, writing captured, transcript, reading, linking. | "Replay Board takes the recording as it is. Meta's models do the understanding: SAM 3.1 finds the board and tracks the teacher, Voice Transcribe and Meta's MMS aligner get every word and its time, Muse Spark reads the board and links it to what was said." |
| 0:30–0:55 | Player: original video left (untouched), clean board right, drawing stroke by stroke in sync. Scrub back and forth. | "The original video plays untouched. Beside it, a clean board redraws your writing as you write it, on the same clock as the video." |
| 0:55–1:10 | Seek to a moment where you stand in front of the writing: the board already shows it, with the "was hidden" tag. | "When the teacher stands in front of what they're writing, the camera can't see it. Replay Board draws it anyway, at the moment it was actually said." |
| 1:10–1:25 | Hover a piece (tooltip with typeset maths); click it: the video jumps to where it was explained. Toggle Typeset. | "Click any line to jump to where it was explained. Switch handwriting to typeset maths." |
| 1:25–1:35 | Transcript highlighting word by word; click a chapter. Partial erase visible on the board. | "Every word is timed; chapters come from the lecture itself. Erasing works too." |
| 1:35–1:45 | A moment where you point at a result on the board: the virtual board spotlights or underlines it while the video plays untouched. | "When the teacher points at the board, the clean board shows what they meant: SAM 3.1 follows the hand, Muse Spark decides if it was pointing and at what." |
| 1:45–2:15 | Press A: Ask Muse. Attach a formula from the start (board tooltip "Ask") and the result from the end (Browse lecture), ask "How are these connected?". The answer streams with typeset item chips and times; click a chip: the video jumps and the board outlines it. | "And you can ask about it. Muse Spark sees the whole lecture, so it can connect the first minute to the last, and every claim links back to the moment and the writing." |
| 2:15–2:35 | Slide lecture: clean slide (no cursor, webcam blanked); the cursor's wiggle becomes a calm circle; a sweep becomes an underline; an A→B move becomes an arrow; a rest becomes a spotlight. | "For slides, the restless cursor becomes calm, meaningful animations. Code tracks the pointer; Muse Spark looks at the path on the slide and what was being said, and decides what it meant." |
| 2:35–2:45 | Click "Slides": the lecture as a deck, a cropped board per slide with Muse Spark's key points. Press Present, arrow through two slides, Esc. Point at "Download PowerPoint" and "Save as PDF". | "And when you're done, the whole lecture becomes slides: the board as the teacher built it, key points from Muse Spark, ready to present, print or open in PowerPoint." |
| 2:45–2:55 | The architecture diagram from the README; the self-test result lines (72/72, 47/47). | "Under the hood: plain code measures every pixel, fast and exact; Meta models make every judgement about meaning. Generated lectures with known answers keep it honest." |
| 2:55–3:00 | Back to the board replay. | "Replay Board: every lecture, as if you had the best seat in the room." |

## Checklist before filming
- `.venv\Scripts\python -m pytest -q` and `.venv\Scripts\python -m tests.selftest_board` pass.
- `cd web; npm run build`, then use the built player at http://127.0.0.1:8000 (smoother than dev mode).
- The lecture list shows only your lectures (delete old test jobs in `data\jobs`).
- Pop-outs you will show are already opened once (cached), so they appear instantly.
- Open the Slides page of the lecture you film once before (the key points take about 6 s the first time), and
  check each slide's crop and title. The 3D pop-out and Notes page are no longer in the script; show them only if
  you have spare seconds.
- Decide on the Ask Muse question beforehand. A question asked before comes from the cache, free, but appears at
  once instead of streaming in; to show the streaming, ask it fresh on camera (about a cent).
