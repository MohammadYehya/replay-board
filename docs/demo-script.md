# Demo video script (3 minutes)

Use **your own recordings only**: one board lecture (ideally where you step in front of what you write,
point at a result with your arm stretched out, and erase something) and one slide lecture with a cursor or laser. Process both before filming
(`python -m app.server`, drop the files), so the player opens instantly; show the upload with a short clip.
Record at 1920×1080, browser zoom 100%, dark theme. Close other tabs; hide the bookmarks bar.

| Time | On screen | Say (roughly) |
|---|---|---|
| 0:00–0:15 | A recorded board lecture where the teacher stands in front of the writing; then a slide recording with a cursor flying around. | "Rewatching a recorded lecture is hard: the teacher blocks the board, the camera can't read it, the cursor never stops moving." |
| 0:15–0:35 | Replay Board home page. Drag a short clip in. The stages tick: board found, person tracked, writing captured, transcript, reading, linking. | "Replay Board takes the recording as it is. Meta's models do the understanding: SAM 3.1 finds the board and tracks the teacher, Voice Transcribe and Meta's MMS aligner get every word and its time, Muse Spark reads the board and links it to what was said." |
| 0:35–1:05 | Player: original video left (untouched), clean board right, drawing stroke by stroke in sync. Scrub back and forth. | "The original video plays untouched. Beside it, a clean board redraws your writing as you write it, on the same clock as the video." |
| 1:05–1:25 | Seek to a moment where you stand in front of the writing: the board already shows it, with the "was hidden" tag. | "When the teacher stands in front of what they're writing, the camera can't see it. Replay Board draws it anyway, at the moment it was actually said." |
| 1:25–1:45 | Hover a piece (tooltip with typeset maths); click it: the video jumps to where it was explained. Toggle Typeset. | "Click any line to jump to where it was explained. Switch handwriting to typeset maths." |
| 1:45–1:55 | Transcript highlighting word by word; click a chapter. Partial erase visible on the board. | "Every word is timed; chapters come from the lecture itself. Erasing works too." |
| 1:55–2:05 | A moment where you point at a result on the board: the virtual board spotlights or underlines it while the video plays untouched. | "When the teacher points at the board, the clean board shows what they meant: SAM 3.1 follows the hand, Muse Spark decides if it was pointing and at what." |
| 2:05–2:30 | Slide lecture: clean slide (no cursor, webcam blanked); the cursor's wiggle becomes a calm circle; a sweep becomes an underline; an A→B move becomes an arrow; a rest becomes a spotlight. | "For slides, the restless cursor becomes calm, meaningful animations. Code tracks the pointer; Muse Spark looks at the path on the slide and what was being said, and decides what it meant." |
| 2:30–2:42 | Hover a drawn graph → "3D" → rotate the scene. Click "Notes": the notes page, typeset, by chapter; click a time and the player jumps there. | "Drawn graphs pop out in 3D. And you get study notes, organised by chapter, with what was said for each line, one click from the moment it was explained." |
| 2:42–2:55 | The architecture diagram from the README; the self-test result lines (72/72, 47/47). | "Under the hood: plain code measures every pixel, fast and exact; Meta models make every judgement about meaning. Generated lectures with known answers keep it honest." |
| 2:55–3:00 | Back to the board replay. | "Replay Board: every lecture, as if you had the best seat in the room." |

## Checklist before filming
- `.venv\Scripts\python -m pytest -q` and `.venv\Scripts\python -m tests.selftest_board` pass.
- `cd web; npm run build`, then use the built player at http://127.0.0.1:8000 (smoother than dev mode).
- The lecture list shows only your lectures (delete old test jobs in `data\jobs`).
- Pop-outs you will show are already opened once (cached), so they appear instantly.
