# Submission text

Paste-ready text for the hackathon form. Fill in the two links before submitting.

- **Repo:** https://github.com/MohammadYehya/replay-board
- **Demo video:** <your video URL>

## Name
Replay Board

## One line
Rewatch a recorded lecture with a clean board beside it: the teacher's writing redrawn stroke by stroke, in sync
with their voice, including what they hid while writing it.

## The problem
Recorded lectures are hard to rewatch. The teacher stands in front of what they are writing, the camera can't read
the board, the cursor flies around the slides, and finding "where did she derive that?" means scrubbing through an
hour of video. Students rewatch lectures to study, and the video is the worst possible format for that.

## What it does
Upload a lecture video. Replay Board plays the original, untouched, next to a virtual board that redraws the
writing stroke by stroke on the same clock as the video.
- Writing that was hidden behind the teacher appears anyway, at the moment it was talked about, with a small
  "was hidden" tag.
- Click any line to jump to where it was explained. Switch handwriting to typeset maths.
- Every word of the transcript is timed; chapters come from the lecture itself.
- When the teacher points at the board, the clean board shows it as a spotlight, circle, underline or arrow. On
  slide lectures, the restless cursor or laser becomes the same calm animations, on clean slides with the webcam
  blanked.
- **Ask Muse:** attach a formula from the first minute and a result from the last, ask how they connect, and get
  an answer whose times and board items you can click.
- **3D steps:** a drawing opens in 3D, and the model changes step by step as the teacher explains it, each change
  at the word that names it, beside the video as it plays. "Try it" gives a slider per value.
- **Slides export:** the whole lecture as a deck that teaches it, with typeset formulas, worked steps, the
  teacher's own drawings and what was said in the speaker notes, downloaded as a real PowerPoint.

The original video is never altered: no outlining, blurring or removing the teacher.

## How it uses Meta's models
Plain code measures every pixel (fast, exact and free on thousands of frames); Meta's models make every judgement
about meaning.
- **SAM 3.1** finds the board's corners, and in video mode tracks the teacher (the occlusion mask for board
  capture) and their hands (for pointing).
- **Muse Voice Transcribe** transcribes the audio in parallel pieces; **Meta's MMS forced aligner** gives every
  word its time.
- **Muse Spark 1.3** decides board or slides, reads every piece of writing and every slide (text and LaTeX),
  links each piece to the moment it was explained, sets redraw times for hidden writing, writes the chapters,
  judges what each pointing gesture meant, answers questions with the whole lecture in context, writes the 3D
  scenes and their steps, and writes the slide deck.

Every call goes through one client with retries and a disk cache, so re-running a lecture costs nothing. An
8-minute 1080p lecture costs $1.17 with pointing gestures ($0.39 without).

## How it's built
- **Backend:** Python, FastAPI, OpenCV, ffmpeg, torchaudio (MMS), python-pptx.
- **Player:** React, Vite, TypeScript, KaTeX and three.js.
- **Board capture:** "writing stays put, the teacher moves". Each pixel keeps counters that only move while the
  board is visible there. Ink is accepted after 1 s and removed after 2 s of clean board. Camera drift of a pixel
  or two is tolerated rather than treated as erase-and-rewrite, and thick ink that stays put is told apart from a
  hand or sleeve that SAM missed.
- **3D scenes are data, not code.** Muse Spark writes shapes, graphs, angles, parameters and steps. The server
  checks every maths expression against a small grammar and evaluates every state, and the player compiles the
  same grammar with its own parser.
- **Self-tests on generated lectures with known answers.** A teacher walks in front of what he writes, there are
  erasers, chalk, fade-ins, camera drift, filled arrowheads, a webcam corner, a parked cursor and a laser. The
  analysis scores 136/136 on board lectures and 47/47 on slide lectures, and these tests caught most real bugs
  before a real video did.

## Challenges
- Cameras drift by about a pixel, and strokes are only 1 to 3 px wide, so without tolerance every stroke
  "erased and rewrote itself".
- A filled arrowhead looks like an arm to a thickness test. The fix is that writing stays put and the teacher
  moves.
- Pointing is mostly talking with your hands. Code keeps only moments where the arm is stretched out onto visible
  writing with no new ink following; Muse Spark then drops most of those too.
- Long lectures: the replay appears first with turn-level word times, and the MMS word times refine it in place.

## What's next
- A browser extension front end for lecture platforms that allow it, on top of the same backend.
- Moving cameras (pan and zoom), which the board capture doesn't handle yet.
- Running the MMS aligner on a GPU, so an hour of lecture gets word times in minutes.
