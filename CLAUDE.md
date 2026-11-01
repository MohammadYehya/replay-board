# Replay Board: notes for Claude Code

Read `SPEC.md` first. It is the source of truth for what to build, the algorithms, the output format and the milestones.

## Context
- This is an entry for the Meta Global AI Developer Hackathon. Deadline: Wed 4 Nov 2026, 9 PM ET (Thu 02:00 Dublin).
- The owner builds through you on **Windows 11** (PowerShell), with Python 3.10–3.13, ffmpeg on PATH and Node 20+ for the player.
- The folder starts empty apart from these two files. The core algorithms in SPEC.md section 5 were already proven in a prototype. Implement them as written, with their self-test (5.4), before building on them.

## Layout to create
```
app/          backend: pipeline stages, FastAPI server, `python -m app.process <video>`
app/core/     board capture, slide analysis, speech/alignment, media (ffmpeg), meta_api client
web/          player: React + Vite + TypeScript
tests/        generated-lecture self-test and unit tests
data/         test clips and job folders (gitignored)
```

## Rules
- Call Meta models only through one client module, with retries and a disk cache. Model ids are configurable in `.env`.
- Secrets live in `.env`; never commit it. Videos, and anything made from them, live in `data/` (gitignored). Videos the owner doesn't own never go in the repo or the demo.
- Never alter the original video in the player: no outlining, blurring or removing the teacher.
- Keep the self-test passing and add tests with new logic. Measure on real clips; don't guess.
- Give commands for PowerShell. Errors for the owner are one plain line, not a traceback.
- Work milestone by milestone (SPEC.md section 8). At the end of each one:
  - commit;
  - tell the owner the one or two commands to see it working;
  - then continue unless told to stop.
- Ask before anything that would spend more than a few dollars of API credit or can't be undone.
- Write messages to the owner in plain, direct language.
