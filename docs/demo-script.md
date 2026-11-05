# Demo video: what to record, what to say, and the hand-off

A 3-minute video. You record three things: a short board lecture, screen clips of the app, and your narration.
Claude edits them into the final video: cuts, zooms, title and model cards, captions, clean sound and an end card.

**Timeline (deadline 02:00 UTC):**

| By (UTC) | You |
|---|---|
| 01:05 | Lecture recorded (recipe below) and copied to the PC |
| 01:12 | Lecture uploaded and processed; 3D view and Slides opened once (warm-up below) |
| 01:30 | Screen clips and narration recorded, files in the hand-off folder |
| 01:45 | Claude hands back the finished video |
| 01:55 | Uploaded to YouTube (unlisted), link pasted in the form, submitted |

**Running late?** Do the **fast lane**: shots 1, 3, 4, 5, 6 and 9 only. Claude makes shots 2, 11 and 12 without you,
for a video of about 2 minutes. That is better than a rushed 3.

Use only your own lecture, or a video whose licence allows reuse, with its credit in the end card. Never use
`data\test*.mp4`.

---

## 1. The lecture you'll show (record it first, about 3 minutes)

**Topic: "The volume of a cylinder".** Each beat makes one feature show up in the app.

**Camera:**
- A phone in landscape on a tripod or a stack of books, 1080p.
- The whole board in frame with a small margin.
- Don't touch the phone once it's recording. Before you start, long-press the screen to lock focus and exposure.
- Good light on the board and no glare. A dark marker (black or blue). Write big: letters at least 5 cm tall.

| Lecture time | Do | Say | Makes this work |
|---|---|---|---|
| 0:00 | Stand to the side. | "Today: the volume of a cylinder." | |
| 0:10 | On the left, write **A = πr²**. | "The area of a circle is pi r squared." | Ask Muse, first item |
| 0:25 | In the middle, draw a cylinder: an ellipse on top, two straight sides, and a curved bottom whose back half is dashed. | "Here's a cylinder." | 3D |
| 0:45 | Point at the drawing as you speak. | "First, the circle: the base. Now stretch it straight up to a height h. Now double the height: twice as tall holds twice as much." | 3D steps change at these words |
| 1:10 | Stand **between the camera and the right side of the board** and write **V = πr²h** with your body hiding it the whole time, for at least 10 seconds. Then step aside. | "So the volume is the base area times the height." | "was hidden" |
| 1:35 | Stretch your arm out fully and point at **πr²**. Hold it for 2 seconds. | "This part is just the base area." | Pointing |
| 1:50 | Write **V = 2πrh**, then erase it. | "Not this: that's the area of the side." | Erasing |
| 2:10 | Bottom right, write **r = 3, h = 5 → V = 45π ≈ 141**. | "With r 3 and h 5, the volume is 45 pi, about 141." | Ask Muse, second item; typeset |
| 2:40 | Step aside, face the camera. | "Base times height. That's the whole idea." | |

Copy the file to the PC.

## 2. Process it and warm up (while you set up recording)

1. In the server window: Ctrl+C, then `.venv\Scripts\python -m app.server`.
2. **Clear the lecture list.** Move every old job out of `data\jobs` into `data\jobs-archive`. They come from the test
   videos, which can't appear on camera.
3. Open http://127.0.0.1:8000, press Ctrl+F5, upload the lecture and let it finish. A 3-minute lecture costs about
   $0.45.
4. Open the lecture once. Hover the cylinder and click **3D** in its tooltip. It takes 30–70 s the first time.
   Play through the steps once.
5. Click **Slides** once. The deck takes about 30 s the first time. Click **Download PowerPoint** once.
6. Pick your Ask Muse question: **"How are these connected?"**, with A = πr² and the 45π result attached.

## 3. Recording the screen

- **Tool:** Xbox Game Bar is already in Windows and records just the browser window. Click Chrome, then press
  **Win+Alt+R** to start and again to stop. Files go to `Videos\Captures`.
- Before you start:
  - Chrome in full screen (**F11**), zoom at 100%, the bookmarks bar hidden.
  - Notifications off: Focus or Do not disturb.
- **One clip per shot, in order.** Don't rename them; Claude sorts them by time.
- At the start and end of each clip, keep still for 2 seconds.
- Move the mouse slowly. Hover for 1 second before each click. When you're not using the cursor, park it at the edge
  of the screen.
- Fluffed something? Pause 2 seconds and do it again in the same clip. Claude keeps the last good try.
- Record the screen **silently**. The narration is recorded separately (section 4), so you can concentrate on
  the mouse.

## 4. Recording the narration

- **Tool:** the phone's voice recorder, or the Sound Recorder app on Windows.
- **Room:** quiet, with soft furnishings if possible (a bedroom beats a kitchen).
- **Mic:** 15–20 cm from your mouth, a little to the side.
- **How:** read the lines below in order, as **one file**, with a 2-second pause between lines. Fluffed a line?
  Pause, then read the whole line again.
- **Pace:** calm, a bit slower than talking. Smile on the first and last lines; it can be heard.

## 5. Shots and lines

Shots marked **(Claude)** need nothing from you. Times are for the final cut.

| # | Final time | You record (silent screen clip) | You say (narration) | Claude adds |
|---|---|---|---|---|
| 1 | 0:00–0:10 | Nothing: Claude cuts it from your lecture file, while you stand in front of the board. | "Recorded lectures are hard to learn from. The teacher stands in front of the board, the camera can't read it, and finding one step means scrubbing through the whole thing." | Caption: "The board, hidden." |
| 2 | 0:10–0:15 | **(Claude)** | "This is Replay Board. It takes the recording as it is, and Meta's models do the understanding." | Title card |
| 3 | 0:15–0:38 | Home page → drag the lecture file in → **Upload and process** → watch the stages tick → when the player opens, press play and let the board fill in for about 20 s. Keep recording through the waiting. | "I drop in a lecture. SAM 3.1 finds the board and tracks the teacher. Muse Voice Transcribe and Meta's MMS aligner give every word its time. Muse Spark reads the writing and links it to what was said. And I don't wait: the lecture opens as soon as the board is found, and the clean board fills in as it goes." | The waiting sped up, with a "4×" tag; a model name pops up as each stage ticks |
| 4 | 0:38–0:55 | Open the finished lecture. Play from 0:10 for 15 s, then drag the timeline back and forth once. | "The original plays untouched. Beside it, a clean board redraws every stroke as it was written, on the same clock. Scrub anywhere, and the board is exactly as it was at that second." | Gentle zoom on the board |
| 5 | 0:55–1:10 | Seek to about 1:05 and play to about 1:25, through the moment you step aside. Then hover the "was hidden" tag. | "Here I'm standing right in front of what I'm writing. The camera never saw it. Replay Board draws it anyway, when I said it, and marks it as hidden." | Zoom on the tag |
| 6 | 1:10–1:25 | Hover V = πr²h (tooltip with typeset maths) → click it (the video jumps) → press **T** (typeset) → press **T** again. | "Every line is clickable: the video jumps to where it was explained. Press T, and the handwriting becomes typeset maths, read by Muse Spark." | Key hint "T" on screen |
| 7 | 1:25–1:35 | Seek to about 1:33 and play through the pointing. | "When I point, the clean board shows what I meant. SAM follows the hand; Muse Spark decides whether it was a point, and at what." | |
| 8 | 1:35–1:55 | Seek to about 0:40 → click **Watch in 3D** when it appears → let it play through the steps → click **Try it** and drag one slider for 2 s. | "Drawings come alive. Muse Spark turns my sketch into a 3D model, and it changes at the exact words I explain it with. Try it gives a slider for every value." | |
| 9 | 1:55–2:15 | Press **A** → attach A = πr² (its tooltip's **Ask**) and the 45π line (**Browse lecture**) → type "How are these connected?" → send → let the answer stream → click one item chip in it. | "And I can ask about it. Muse Spark sees the whole lecture, so it can connect the first minute to the last, and every claim links back to the moment and the writing." | Zoom on the answer |
| 10 | 2:15–2:30 | Click **Slides** → scroll slowly through four or five slides → click **Download PowerPoint**. | "When I'm done, Muse Spark turns the lecture into a deck that teaches it: typeset formulas, worked steps, my own drawings, and what I said in the speaker notes." | |
| 11 | 2:30–2:50 | **(Claude)** | "Under the hood, plain code measures every pixel, fast and exact, and Meta's models make every judgement about meaning. Generated lectures with known answers keep it honest: 136 out of 136 on boards, 47 out of 47 on slides." | "How it works" card: the four models and what each does; the self-test scores |
| 12 | 2:50–3:00 | **(Claude)** | "Replay Board. Every lecture, as if you had the best seat in the room." | End card: name, repo link, models, credits |

About 400 words of narration, so roughly 3 minutes at a calm pace.

## 6. Hand-off

Put everything in **`C:\Users\MYH\Documents\Replay Board\data\demo\raw\`**. The `data` folder is never committed.
- the screen clips (straight from `Videos\Captures`);
- the narration file;
- the lecture file (if it isn't already there).

Then tell Claude "files are in". You get back `data\demo\replay-board-demo.mp4` (1080p, YouTube loudness, captions
burned in), plus a `.srt` caption file to upload to YouTube as well.
