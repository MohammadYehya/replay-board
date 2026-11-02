# Demo video script (3 minutes)

Use **your own recordings only**: one board lecture (ideally where you step in front of what you write,
point at a result with your arm stretched out, erase something, and draw one shape or graph that you then explain
in a few spoken stages, e.g. "first a cube... now stretch it upwards... now tilt it", for the 3D view) and one
slide lecture with a cursor or laser. Process both before filming
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
| 1:45–2:00 | Hover your 3D-ish drawing, click "View in 3D" (made beforehand, so it opens at once). Click the first step chip: the video jumps there and the model changes at each word while the video plays beside it. Drag to turn it once. | "Drawings come alive in 3D: Muse Spark builds a model of what you drew and changes it step by step, at the exact words you explain it with." |
| 2:00–2:20 | Press A: Ask Muse. Attach a formula from the start (board tooltip "Ask") and the result from the end (Browse lecture), ask "How are these connected?". The answer streams with typeset item chips and times; click a chip: the video jumps and the board outlines it. | "And you can ask about it. Muse Spark sees the whole lecture, so it can connect the first minute to the last, and every claim links back to the moment and the writing." |
| 2:20–2:35 | Slide lecture: clean slide (no cursor, webcam blanked); the cursor's wiggle becomes a calm circle; a sweep becomes an underline; an A→B move becomes an arrow; a rest becomes a spotlight. | "For slides, the restless cursor becomes calm, meaningful animations. Code tracks the pointer; Muse Spark looks at the path on the slide and what was being said, and decides what it meant." |
| 2:35–2:45 | Click "Slides": Muse Spark's deck of the lecture. Scroll past the agenda to a formula slide, a worked example and the key takeaways. Click "Download PowerPoint" and show the file open in PowerPoint for two seconds (speaker notes visible). | "And when you're done, Muse Spark turns the whole lecture into a slide deck: typeset formulas, worked steps, the teacher's drawings and the key takeaways, as a real PowerPoint with what was said in the notes." |
| 2:45–2:55 | The architecture diagram from the README; the self-test result lines (72/72, 47/47). | "Under the hood: plain code measures every pixel, fast and exact; Meta models make every judgement about meaning. Generated lectures with known answers keep it honest." |
| 2:55–3:00 | Back to the board replay. | "Replay Board: every lecture, as if you had the best seat in the room." |

## Checklist before filming
- `.venv\Scripts\python -m pytest -q` and `.venv\Scripts\python -m tests.selftest_board` pass.
- `cd web; npm run build`, then use the built player at http://127.0.0.1:8000 (smoother than dev mode).
- The lecture list shows only your lectures (delete old test jobs in `data\jobs`).
- Make the 3D view you will show beforehand, so it opens at once: open it once in the player (the first time takes
  30–70 s), or run `.venv\Scripts\python -m app.popout <job> <piece>` (it shows the cost and asks first). Play
  through every step and check each change happens at the right word; if one is off, delete
  `data\jobs\<job>\popout\NNN.v2.json` and open it again for a new take (about $0.03–0.10).
- Open the Slides page of the lecture you film once before (Muse Spark writes the deck in about 25–30 s the first
  time, then it's cached) and read every slide. Download the PowerPoint and open it in PowerPoint once before
  filming: it was only checked in LibreOffice here. The Notes page is no longer in the script; show it only if you
  have spare seconds.
- Decide on the Ask Muse question beforehand. A question asked before comes from the cache, free, but appears at
  once instead of streaming in; to show the streaming, ask it fresh on camera (about a cent).
