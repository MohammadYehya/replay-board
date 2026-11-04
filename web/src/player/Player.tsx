import { Suspense, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { getJob, getMadeScenes, getReplay, jobFileUrl, posterUrl } from '../api';
import { AskDrawer } from '../ask/AskDrawer';
import { SparkIcon } from '../ask/icons';
import { lectureIndex } from '../ask/items';
import { spanEntry, type SpanPick } from '../ask/selection';
import {
  addEntry, askStartsOpen, loadTray, rememberAskOpen, saveTray, session, takeAskRequest, TRAY_MAX, type TrayEntry,
} from '../ask/tray';
import { SlidesIcon } from '../slides/icons';
import type { AskRef, BoardLine, JobStatus, Piece, Replay, SlideElement } from '../types';
import { Board, type BoardHandle } from './Board';
import { boardSpriteKeys, preloadBoard } from './boardRenderer';
import { SpriteCache } from './sprites';
import {
  accepts, boardTime, catchUpText, failureOf, gateTitle, heldOf, jobEnded, knowsUpTo, pastReady, planPoll,
  PRELOAD_MS, readyFraction, readyUntil, replaySeq, reuseUnchanged, shouldWatch, stopAfterTaking, streamLine, WATCH_MS,
} from './stream';
import { Slides, type SlidesHandle } from './Slides';
import { drawable } from './slideTiming';
import {
  activeWord, chapterAt, formatTime, groupWords, isWrittenLine, lastStartedWord, seekTarget, turnsFromWords,
} from './timing';
import { fieldTakesKey } from './keys';
import { LazyView3D, View3DFallback } from './popoutLoader';
import { cleanMadeScenes, type MadeScene } from './popoutable';
import { Chapters, Transcript } from './Transcript';

const SKIP = 5;

function Shortcuts({ onClose, mode }: { onClose(): void; mode: 'board' | 'slides' | 'none' }) {
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest('.shortcuts, .keys-btn')) onClose();
    };
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [onClose]);
  const rows: [string[], string][] =
    mode === 'slides'
      ? [
          [['Space'], 'Play / pause'],
          [['←', '→'], 'Back / forward 5 s'],
          [['P'], "Show the teacher's pointer"],
          [['A'], 'Ask Muse about the lecture'],
          [['Click slide text'], 'Jump to where it is explained'],
          [['Shift+click slide text'], 'Attach it to a question'],
          [['Click slide'], 'Play / pause'],
          [['Click a word'], 'Jump to that word'],
        ]
      : [
          [['Space'], 'Play / pause'],
          [['←', '→'], 'Back / forward 5 s'],
          [['T'], 'Handwriting / typeset'],
          [['A'], 'Ask Muse about the lecture'],
          [['Click writing'], 'Jump to where it is explained'],
          [['Shift+click writing'], 'Attach it to a question'],
          [['Hover or tap a drawing'], 'View it in 3D, explained step by step'],
          [['[', ']'], 'Previous / next step in the 3D view'],
          [['Esc'], 'Back to the board from the 3D view'],
          [['Click board'], 'Play / pause'],
          [['Click a word'], 'Jump to that word'],
        ];
  return (
    <div className="shortcuts" role="dialog" aria-label="Keyboard shortcuts">
      <h3>Shortcuts</h3>
      <dl>
        {rows.map(([keys, what]) => (
          <div key={what + keys.join()} className="row">
            <dt>
              {keys.map((k) =>
                k.includes(' ') ? (
                  <span key={k} className="act">
                    {k}
                  </span>
                ) : (
                  <kbd key={k}>{k}</kbd>
                ),
              )}
            </dt>
            <dd>{what}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function Player({
  jobId,
  replay: first,
  name,
  running,
  status: firstStatus = null,
  startAt = null,
}: {
  jobId: string;
  replay: Replay;
  name?: string;
  /** Open at this moment (seconds on the video's clock), e.g. from a link in the notes. */
  startAt?: number | null;
  /** Whether the job was still running when the replay was loaded (false: its word times are final). */
  running?: boolean;
  /** The job's status when the replay was loaded (the player then watches the job itself). */
  status?: JobStatus | null;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const boardRef = useRef<BoardHandle>(null);
  const slidesRef = useRef<SlidesHandle>(null);
  // The replay is swapped for each newer version while it plays (a streaming lecture's board as it is captured, then
  // the final replay, then the one with refined word times); the video, play position, typeset mode, drawers and tray
  // are kept.
  const [replay, setReplay] = useState(first);
  const replayRef = useRef(replay);
  replayRef.current = replay;
  // The job as last seen: the player watches it while it runs.
  const [job, setJob] = useState<JobStatus | null>(firstStatus);
  const [watching, setWatching] = useState(() => shouldWatch(first, running));
  useEffect(() => {
    setReplay(first);
    setWatching(shouldWatch(first, running));
  }, [first]); // `running` describes the job when this replay was loaded
  const jobRunning = job ? !jobEnded(job) : running !== false;
  // Proportional word times are refined only while the job runs; a finished job keeps the ones it has.
  const refining = replay.transcript?.timing === 'proportional' && jobRunning;
  // Decoded board images, shared by every version of the board (preloaded before a version is swapped in).
  const [sprites] = useState(() => new SpriteCache());
  const clipStart = replay.video.start ?? 0;
  const clipEnd = replay.video.end ?? replay.video.duration;
  // A partial replay (the lecture still streams): its board covers only the part up to `ready`.
  const ready = readyUntil(replay);
  const streaming = ready != null;
  /** The job failed (or was interrupted) while the replay was partial: the board stops where it is, for good. */
  const failure = streaming ? failureOf(job, clipStart, clipEnd) : null;
  /** Notes, Slides and 3D need the final replay: their title while they wait for it (or will not get it), else null. */
  const gate = gateTitle(streaming, failure);
  const timeRef = useRef<number>(clipStart);
  /** The board's clock: the video's, held at `ready` while the playhead is past it. */
  const boardTimeRef = useRef<number>(clipStart);
  const [pastReadyNow, setPastReady] = useState(false);
  const [typeset, setTypeset] = useState(false);
  const [showPointer, setShowPointer] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [active, setActive] = useState(-1);
  const [upTo, setUpTo] = useState(-1);
  const [turnIdx, setTurnIdx] = useState(-1);
  const [chapter, setChapter] = useState(-1);
  const [showKeys, setShowKeys] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [popout, setPopout] = useState<Piece | null>(null);
  const popoutOpen = useRef(false);
  popoutOpen.current = !!popout;
  const [videoAspect, setVideoAspect] = useState(
    replay.video.width && replay.video.height ? replay.video.width / replay.video.height : 16 / 9,
  );

  // Ask Muse: the drawer (kept mounted once opened, so a conversation survives closing it) and the tray of things
  // attached to the next question, kept per lecture in sessionStorage. The drawer opens by itself only when the Notes
  // page's "Ask" sent the student here, or when it was left open before going to Slides or Notes: never after a
  // reload (pressing A to open it then closed it).
  const [askOpen, setAskOpen] = useState(() => askStartsOpen(session(), jobId));
  useEffect(() => takeAskRequest(session(), jobId), [jobId]);
  const [askMounted, setAskMounted] = useState(askOpen);
  const [askFocus, setAskFocus] = useState(0);
  const [tray, setTrayState] = useState<TrayEntry[]>(() => loadTray(session(), jobId));
  const trayRef = useRef(tray);
  const [askNotice, setAskNotice] = useState<{ text: string; key: number } | null>(null);
  const askBtnRef = useRef<HTMLButtonElement>(null);
  const askOpenRef = useRef(askOpen);
  askOpenRef.current = askOpen;
  const showKeysRef = useRef(showKeys);
  showKeysRef.current = showKeys;

  // Built from the transcript only (a streaming version keeps the transcript object when it did not change).
  const transcript = replay.transcript;
  const words = useMemo(
    () =>
      (transcript?.words ?? [])
        .filter((w) => w && Number.isFinite(w.s) && Number.isFinite(w.e))
        .slice()
        .sort((a, b) => a.s - b.s),
    [transcript],
  );
  const turns = useMemo(() => {
    const given = (transcript?.turns ?? []).slice().sort((a, b) => a.start - b.start);
    return given.length || !words.length ? given : turnsFromWords(words);
  }, [transcript, words]);
  const groups = useMemo(() => groupWords(turns, words), [turns, words]);
  const wordTurn = useMemo(() => {
    const a = new Int32Array(words.length).fill(-1);
    groups.forEach((g, ti) => g.forEach((i) => (a[i] = ti)));
    return a;
  }, [groups, words]);
  const chapters = useMemo(() => (replay.chapters ?? []).slice().sort((a, b) => a.t - b.t), [replay.chapters]);
  const board = replay.board;
  const slides = board ? undefined : replay.slides;
  const mode = board ? 'board' : slides ? 'slides' : 'none';
  /** The 3D view of a board drawing is open in place of the board. */
  const view3d = !!popout && !!board;

  // One watch loop while the job runs and its replay can still change: a streaming board lecture's partial versions
  // (status.json's replay.seq grows with each), then the final replay, then the rewrite with refined word times. Each
  // newer version's board images are loaded first, so the swap never blanks the board. An older server has no seq:
  // its one rewrite comes once the aligner is done.
  useEffect(() => {
    if (!watching) return;
    let stopped = false;
    let timer: number | undefined;
    const ctrl = new AbortController();
    // The status seq the replay held was fetched at (a partial replay carries its own).
    let seen = replaySeq(replayRef.current);
    const poll = async () => {
      try {
        const st = await getJob(jobId, ctrl.signal);
        if (stopped) return;
        setJob(st);
        const held = replayRef.current;
        const plan = planPoll(heldOf(held, seen), st);
        if (plan.fetch) {
          const next = await getReplay(jobId, ctrl.signal);
          if (stopped) return;
          if (accepts(held, next, { bySeq: plan.bySeq, ended: jobEnded(st) })) {
            if (next.board) await preloadBoard(sprites, jobId, next.board, PRELOAD_MS);
            if (stopped) return;
            const taken = reuseUnchanged(held, next);
            replayRef.current = taken;
            setReplay(taken);
            if (next.board) sprites.retain(boardSpriteKeys(next.board));
            if (stopAfterTaking(plan, next)) return setWatching(false);
          }
          if (plan.bySeq) seen = replaySeq(next) ?? st.replay?.seq ?? seen;
        }
        if (plan.stop) return setWatching(false);
      } catch (e) {
        if (stopped || (e as Error).name === 'AbortError') return;
        // the server may be restarting, or the file is being replaced: try again
      }
      timer = window.setTimeout(poll, WATCH_MS);
    };
    timer = window.setTimeout(poll, WATCH_MS);
    return () => {
      stopped = true;
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, [jobId, watching, sprites]);

  // The tab: the lecture's subject, also when a later version of the replay names it first.
  const subject = replay.subject?.trim();
  const title = subject ? subject.charAt(0).toUpperCase() + subject.slice(1) : '';
  useEffect(() => {
    if (title) document.title = `${title} · Replay Board`;
  }, [title]);

  // The poster: the video's own frame where it opens (the server clamps the time to the processed part, as
  // onLoadedMetadata does), until the video has a frame of its own. A move before then (a chapter clicked in the first
  // seconds of a long fragmented MP4, while Chrome still reads its fragments) shows the frame moved to, fetched first so
  // the one before stays until it is there: the opening frame stayed up for seconds while the board and transcript
  // were at 2:56. The clock below sees every move (no "seeking" event comes before the video has its metadata).
  const [poster, setPoster] = useState<string | undefined>(() => posterUrl(jobId, startAt));
  const hadFrame = useRef(false);
  /** The player has moved the video to startAt (onLoadedMetadata): before that its clock reads 0, not a move. */
  const startedAt = useRef(false);
  const posterAt = useRef(Math.min(Math.max(startAt ?? clipStart, clipStart), clipEnd));
  const posterTimer = useRef(0);
  const followPoster = useCallback(
    (now: number) => {
      const t = Math.min(Math.max(now, clipStart), clipEnd);
      if (hadFrame.current || Math.abs(t - posterAt.current) < 0.05) return;
      posterAt.current = t;
      window.clearTimeout(posterTimer.current);
      posterTimer.current = window.setTimeout(() => {
        const url = posterUrl(jobId, t);
        const show = (to: string | undefined) => {
          if (!hadFrame.current && posterAt.current === t) setPoster(to);
        };
        const img = new Image();
        img.onload = () => show(url);
        img.onerror = () => show(undefined); // black, rather than a frame from somewhere else
        img.src = url;
      }, 120); // a drag along the bar asks for the frame it stops at
    },
    [jobId, clipStart, clipEnd],
  );
  useEffect(() => () => window.clearTimeout(posterTimer.current), []);

  // One clock: the video's currentTime drives the board and the transcript, even while paused or scrubbing.
  useEffect(() => {
    let raf = 0;
    let prev = -1;
    const last = { active: -2, upTo: -2, turn: -2, chapter: -2, past: null as boolean | null };
    const tick = () => {
      const v = videoRef.current;
      if (v) {
        let t = v.currentTime;
        const dur = Number.isFinite(v.duration) ? v.duration : Infinity;
        // Stop once at the end of the processed part (only when playback runs into it, not on a seek).
        if (!v.paused && prev >= 0 && prev < clipEnd && t >= clipEnd && t - prev < 0.6 && clipEnd < dur - 0.3) {
          v.pause();
          v.currentTime = clipEnd;
          t = clipEnd;
          setToast('End of the processed part. Press play to keep watching the video.');
        }
        prev = t;
        timeRef.current = t;
        if (!hadFrame.current && (startAt == null || startedAt.current)) followPoster(t);
        // The board holds at `ready` past it (drawn as there, never blank) while the video plays on.
        const bt = boardTime(t, ready);
        boardTimeRef.current = bt;
        if (!popoutOpen.current) boardRef.current?.render(bt); // the 3D view reads timeRef itself
        const past = pastReady(t, ready);
        if (past !== last.past) setPastReady((last.past = past));
        slidesRef.current?.render(t);
        const up = lastStartedWord(words, t);
        const act = activeWord(words, t);
        let tu = up >= 0 ? wordTurn[up] : -1;
        if (!words.length) for (let i = 0; i < turns.length; i++) if (turns[i].start <= t) tu = i;
        const ch = chapterAt(chapters, t);
        if (act !== last.active) setActive((last.active = act));
        if (up !== last.upTo) setUpTo((last.upTo = up));
        if (tu !== last.turn) setTurnIdx((last.turn = tu));
        if (ch !== last.chapter) setChapter((last.chapter = ch));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [words, wordTurn, turns, chapters, clipEnd, followPoster, startAt, ready]);

  useEffect(() => {
    if (!toast) return;
    const id = window.setTimeout(() => setToast(null), 4500);
    return () => window.clearTimeout(id);
  }, [toast]);

  const seek = useCallback((t: number, play = false) => {
    const v = videoRef.current;
    if (!v) return;
    const dur = Number.isFinite(v.duration) ? v.duration : Infinity;
    v.currentTime = Math.max(0, Math.min(t, dur - 0.05));
    timeRef.current = v.currentTime;
    if (play) v.play().catch(() => {});
  }, []);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {});
    else v.pause();
  }, []);

  const seekPiece = useCallback((p: Piece) => seek(seekTarget(p, clipStart), true), [seek, clipStart]);
  const seekWord = useCallback((t: number) => seek(t), [seek]);
  const seekChapter = useCallback((t: number) => seek(t, true), [seek]);
  const seekAndPlay = useCallback((t: number) => seek(t, true), [seek]);
  // The 3D view takes the board's place while the video keeps playing: its steps follow the video's clock.
  const boardPanelRef = useRef<HTMLElement>(null);
  const openPopout = useCallback((p: Piece) => setPopout(p), []);
  // The drawings whose 3D view is made already (the board's "3D" badges and "Watch in 3D"): asked once per lecture,
  // and again when the view has made a new one. GET only: it never makes a view (that costs Muse Spark credit).
  const [made, setMade] = useState<MadeScene[]>([]);
  const [madeAsk, setMadeAsk] = useState(0);
  const madeRef = useRef(made);
  madeRef.current = made;
  const hasBoard = !!replay.board;
  // While the word times are refined (or the replay is partial) the server lists none (a view made then would
  // follow the old times): asked again once they are.
  useEffect(() => {
    if (!hasBoard || refining || streaming) return setMade([]);
    const ctrl = new AbortController();
    getMadeScenes(jobId, ctrl.signal).then((raw) => {
      if (!ctrl.signal.aborted) setMade(cleanMadeScenes(raw));
    });
    return () => ctrl.abort();
  }, [jobId, hasBoard, madeAsk, refining, streaming]);
  const onMade = useCallback((pieceId: number) => {
    if (!madeRef.current.some((m) => m.drawing === pieceId || m.pieces.includes(pieceId))) setMadeAsk((k) => k + 1);
  }, []);
  const closePopout = useCallback(() => {
    // the focus was in the 3D view (or nowhere): give it to the board panel, not to the page
    const a = document.activeElement as HTMLElement | null;
    const inside = !a || a === document.body || !!a.closest?.('.v3');
    setPopout(null);
    if (inside) requestAnimationFrame(() => boardPanelRef.current?.focus({ preventScroll: true }));
  }, []);

  // ---- Ask Muse ----
  const lecture = useMemo(
    () => lectureIndex(replay, { clipStart, isLine: (ln) => isWrittenLine(ln, replay.board?.size?.[0] ?? 960) }),
    [replay, clipStart],
  );
  useEffect(() => saveTray(session(), jobId, tray), [jobId, tray]);
  useEffect(() => {
    rememberAskOpen(jobId, askOpen);
    if (askOpen) setAskMounted(true);
  }, [askOpen, jobId]);
  useEffect(() => {
    if (!askNotice) return;
    const id = window.setTimeout(() => setAskNotice(null), 3200);
    return () => window.clearTimeout(id);
  }, [askNotice]);

  const setTray = useCallback((next: TrayEntry[]) => {
    trayRef.current = next;
    setTrayState(next);
  }, []);
  const openAsk = useCallback((focus: boolean) => {
    setAskOpen(true);
    if (focus) setAskFocus((k) => k + 1);
  }, []);
  const closeAsk = useCallback(() => {
    setAskOpen(false);
    if ((document.activeElement as HTMLElement | null)?.closest?.('.ask-drawer')) askBtnRef.current?.focus();
  }, []);
  /** Put something on the tray and show the drawer (without taking the focus from what the student clicked). */
  const attach = useCallback(
    (entry: TrayEntry) => {
      const r = addEntry(trayRef.current, entry);
      if (r.status === 'added') setTray(r.tray);
      setAskNotice({
        text:
          r.status === 'added'
            ? `Attached: ${entry.label}`
            : r.status === 'duplicate'
              ? 'That is already attached.'
              : `You can attach up to ${TRAY_MAX} things to one question.`,
        key: Date.now(),
      });
      openAsk(false);
    },
    [openAsk, setTray],
  );
  const askRef = useCallback(
    (ref: string, fallback: TrayEntry) => {
      const it = lecture.byRef.get(ref);
      attach(it ? { item: it.item, label: it.label, t: it.t } : fallback);
    },
    [lecture, attach],
  );
  const askPiece = useCallback(
    (p: Piece, line: BoardLine | null) =>
      line
        ? askRef(`#${line.id}`, { item: { kind: 'line', id: line.id }, label: line.read?.text || line.id, t: line.t_complete })
        : askRef(`#${p.id}`, { item: { kind: 'piece', id: p.id }, label: p.read?.text || `#${p.id}`, t: seekTarget(p, clipStart) }),
    [askRef, clipStart],
  );
  const askElement = useCallback(
    (el: SlideElement) =>
      askRef(`#${el.id}`, { item: { kind: 'element', id: el.id }, label: el.text || el.id, t: timeRef.current ?? 0 }),
    [askRef],
  );
  const askSpan = useCallback((s: SpanPick) => attach(spanEntry(s)), [attach]);
  const addMoment = useCallback(() => {
    const t = Math.round((timeRef.current ?? 0) * 10) / 10;
    attach({ item: { kind: 'moment', t }, label: 'Moment', t });
  }, [attach]);
  /** Show a board or slide item an answer cites: seek to it and light it up for a moment. */
  const showItem = useCallback(
    (r: AskRef) => {
      seek(r.t);
      if (r.kind === 'element') slidesRef.current?.highlight([String(r.id)]);
      else {
        setPopout(null); // the board item is shown on the board
        boardRef.current?.highlight([r.id]);
      }
    },
    [seek],
  );
  const seekOnly = useCallback((t: number) => seek(t), [seek]);

  // Keyboard: space, arrows, t (board), p (slides), a (Ask Muse). Ignored while typing in a field, and inside the
  // Ask Muse drawer (it has its own keys: Enter sends, Esc closes).
  const modeRef = useRef(mode);
  modeRef.current = mode;
  useEffect(() => {
    let swallowSpaceUp = false;
    const down = (e: KeyboardEvent) => {
      // The 3D view keeps the player's keys (it has no keys of its own); Esc goes back to the board, also from a
      // "Try it" slider (it keeps only the keys that move it: keys.ts).
      if (e.ctrlKey || e.metaKey || e.altKey || fieldTakesKey(e.target as HTMLInputElement | null, e.key)) return;
      if ((e.target as HTMLElement | null)?.closest?.('.ask-drawer')) return;
      const v = videoRef.current;
      if (!v) return;
      let handled = true;
      if (e.key === ' ' || e.code === 'Space') {
        if (!e.repeat) togglePlay();
        swallowSpaceUp = true;
      } else if (e.key === 'ArrowLeft') seek(v.currentTime - SKIP);
      else if (e.key === 'ArrowRight') seek(v.currentTime + SKIP);
      else if ((e.key === 't' || e.key === 'T') && modeRef.current === 'board') setTypeset((x) => !x);
      else if ((e.key === 'p' || e.key === 'P') && modeRef.current === 'slides') setShowPointer((x) => !x);
      else if (e.key === '?') setShowKeys((x) => !x);
      else if ((e.key === 'a' || e.key === 'A') && !e.shiftKey) {
        if (askOpenRef.current) closeAsk();
        else openAsk(true);
      } else if (e.key === 'Escape') {
        if (showKeysRef.current) setShowKeys(false);
        else if (askOpenRef.current) closeAsk();
        else if (popoutOpen.current) closePopout();
        else handled = false;
      } else handled = false;
      if (handled) {
        e.preventDefault();
        e.stopPropagation();
      }
    };
    const up = (e: KeyboardEvent) => {
      if ((e.key === ' ' || e.code === 'Space') && swallowSpaceUp) {
        swallowSpaceUp = false;
        e.preventDefault(); // stop a focused button from also "clicking" on space
      }
    };
    window.addEventListener('keydown', down, true);
    window.addEventListener('keyup', up, true);
    return () => {
      window.removeEventListener('keydown', down, true);
      window.removeEventListener('keyup', up, true);
    };
  }, [seek, togglePlay, openAsk, closeAsk, closePopout]);

  const onLoadedMetadata = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.videoWidth && v.videoHeight) setVideoAspect(v.videoWidth / v.videoHeight);
    if (clipStart > 0.05 && v.currentTime < clipStart) v.currentTime = clipStart;
    if (startAt != null && !startedAt.current) {
      startedAt.current = true;
      v.currentTime = Math.min(Math.max(startAt, clipStart), clipEnd);
    }
  };

  const viewSize = board?.size ?? slides?.size;
  const boardAspect = viewSize && viewSize[0] > 0 && viewSize[1] > 0 ? viewSize[0] / viewSize[1] : 16 / 9;
  const clipped = clipStart > 0.5 || clipEnd < replay.video.duration - 0.5;
  const pieces = board?.pieces ?? [];
  const hiddenCount = pieces.filter((p) => p.revealed && !p.pre_existing).length;
  const gestureCount = (board?.annotations ?? []).filter(drawable).length; // the hand pointing at writing, as drawn
  const slideStates = slides?.states ?? [];
  const slideCount = slideStates.filter((s) => s.kind !== 'addition').length || slideStates.length;
  const buildSteps = slideStates.length - slideCount;
  const annotationCount = slides?.annotations?.length ?? 0;
  const api = replay.stats?.api;
  const readyPct = Math.round(readyFraction(ready, clipStart, clipEnd) * 1000) / 10;
  // Past what is ready (or before anything is): the board says how far it is, or that it stops there.
  const catchUp =
    ready != null && board && (pastReadyNow || ready - clipStart < 1) ? catchUpText(ready, failure, clipStart) : null;
  /** "so far" while more of the board is coming. */
  const soFar = streaming && !failure ? ' so far' : '';
  const rowStyle = {
    '--cols': `${videoAspect.toFixed(4)}fr ${boardAspect.toFixed(4)}fr`,
    '--av': videoAspect.toFixed(4),
    '--ab': boardAspect.toFixed(4),
  } as CSSProperties;

  return (
    <div className={`player ${askOpen ? 'ask-open' : ''}`}>
      <header className="topbar">
        <a className="brand" href="#/" title="All lectures">
          <Logo />
          <span>Replay Board</span>
        </a>
        <div className="topbar-title">
          <span className="subject">{title || 'Lecture'}</span>
          {name && <span className="file-name">{name}</span>}
        </div>
        <div className="topbar-actions">
          <a className="ghost-btn" href="#/">
            All lectures
          </a>
          <a
            className={`ghost-btn notes-btn ${gate ? 'disabled' : ''}`}
            aria-label="Notes"
            {...(gate
              ? { role: 'link', 'aria-disabled': true, title: gate }
              : { href: `#/job/${encodeURIComponent(jobId)}/notes`, title: 'Study notes for this lecture, by chapter' })}
          >
            <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden>
              <path
                d="M4 1.8h5.2L12.5 5v9.2H4zM9 1.8V5.2h3.4M6 8h4.5M6 10.6h4.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinejoin="round"
              />
            </svg>
            <span className="notes-btn-label">Notes</span>
          </a>
          <a
            className={`ghost-btn slides-btn ${gate ? 'disabled' : ''}`}
            aria-label="Slides"
            {...(gate
              ? { role: 'link', 'aria-disabled': true, title: gate }
              : {
                  href: `#/job/${encodeURIComponent(jobId)}/slides`,
                  title: 'This lecture as a slide deck: present it, save it as a PDF or download a PowerPoint',
                })}
          >
            <SlidesIcon />
            <span className="slides-btn-label">Slides</span>
          </a>
          <button className={`ghost-btn keys-btn ${showKeys ? 'on' : ''}`} onClick={() => setShowKeys((x) => !x)}>
            <kbd>?</kbd> Shortcuts
          </button>
          <button
            ref={askBtnRef}
            className={`ghost-btn ask-btn ${askOpen ? 'on' : ''}`}
            aria-expanded={askOpen}
            aria-label={tray.length ? `Ask Muse, ${tray.length} attached` : 'Ask Muse'}
            title="Ask Muse Spark about this lecture (A)"
            onClick={() => (askOpen ? closeAsk() : openAsk(true))}
          >
            <SparkIcon size={13} />
            <span className="ask-btn-label">Ask Muse</span>
            {tray.length > 0 && (
              <span className="ask-btn-count" aria-hidden>
                {tray.length}
              </span>
            )}
          </button>
          {showKeys && <Shortcuts mode={mode} onClose={() => setShowKeys(false)} />}
        </div>
      </header>

      <main className="player-main" style={rowStyle}>
        <div className="media-row">
          <section className="panel video-panel">
            <header className="panel-head">
              <h2>Lecture</h2>
              <span className="panel-note">
                original video, untouched
                {clipped &&
                  ` · ${slides ? 'slides cover' : 'board covers'} ${formatTime(clipStart)}–${formatTime(clipEnd)}`}
              </span>
            </header>
            <div className="video-frame" style={{ aspectRatio: `${videoAspect}` }}>
              <video
                ref={videoRef}
                src={jobFileUrl(jobId, replay.video.file)}
                poster={poster}
                controls
                preload="auto"
                playsInline
                onLoadedMetadata={onLoadedMetadata}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onLoadedData={() => {
                  hadFrame.current = true;
                  setVideoError(null);
                }}
                onError={(e) => {
                  const code = e.currentTarget.error?.code;
                  setVideoError(
                    code === 4
                      ? 'This browser cannot play the video file (its format or codec). Try Chrome or Edge, or re-encode it to H.264 MP4.'
                      : 'The video could not be loaded. Is the server still running?',
                  );
                }}
              />
            </div>
            {ready != null && (
              <div className="stream-status">
                <div
                  className={`ready-bar ${failure ? 'failed' : ''}`}
                  role="progressbar"
                  aria-label="Board ready"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={readyPct}
                  title={`${failure ? 'The board stops at' : 'The board is ready up to'} ${formatTime(ready)}`}
                >
                  <div style={{ width: `${readyPct}%` }} />
                </div>
                {failure ? (
                  <p className="error-line stream-failed" role="alert">
                    {/* the job's whole line (with where its details are) is the tooltip */}
                    <span title={failure.error || undefined}>{streamLine(job, ready, clipStart, clipEnd)}</span>
                  </p>
                ) : (
                  <p className="stream-line">{streamLine(job, ready, clipStart, clipEnd)}</p>
                )}
              </div>
            )}
            {videoError && <p className="error-line">{videoError}</p>}
          </section>

          <section ref={boardPanelRef} className={`panel board-panel ${view3d ? 'has-3d' : ''}`} tabIndex={-1}>
            {/* The board stays mounted under the 3D view, so going back to it is instant. */}
            <div className="board-pane" hidden={view3d}>
            <header className="panel-head">
              <h2>{board || replay.type !== 'slides' ? 'Board' : 'Slides'}</h2>
              {board && (
                <>
                  <span className="panel-note">
                    {pieces.length} {pieces.length === 1 ? 'piece' : 'pieces'} of writing{soFar}
                    {hiddenCount > 0 && ` · ${hiddenCount} written while hidden`}
                    {gestureCount > 0 &&
                      ` · ${gestureCount} pointing ${gestureCount === 1 ? 'gesture' : 'gestures'}`}
                  </span>
                  <div className="segmented" role="group" aria-label="Board style">
                    <button aria-pressed={!typeset} className={!typeset ? 'on' : ''} onClick={() => setTypeset(false)}>
                      Handwriting
                    </button>
                    <button aria-pressed={typeset} className={typeset ? 'on' : ''} onClick={() => setTypeset(true)}>
                      Typeset
                    </button>
                  </div>
                </>
              )}
              {slides && (
                <>
                  <span className="panel-note">
                    {slideCount} {slideCount === 1 ? 'slide' : 'slides'} · {annotationCount}{' '}
                    {annotationCount === 1 ? 'annotation' : 'annotations'}
                  </span>
                  <button
                    className={`pointer-toggle ${showPointer ? 'on' : ''}`}
                    aria-pressed={showPointer}
                    title="Show where the teacher's pointer was (P)"
                    onClick={() => setShowPointer((x) => !x)}
                  >
                    <span className="dot" aria-hidden />
                    Pointer
                  </button>
                </>
              )}
            </header>
            {board ? (
              <Board
                ref={boardRef}
                jobId={jobId}
                board={board}
                typeset={typeset}
                timeRef={boardTimeRef}
                clipStart={clipStart}
                playing={playing}
                onSeekPiece={seekPiece}
                onTogglePlay={togglePlay}
                onPopout={openPopout}
                onAsk={askPiece}
                made={made}
                sprites={sprites}
                partial={streaming}
                catchUp={catchUp}
                catchUpFailed={!!failure}
                popoutLock={gate}
              />
            ) : slides ? (
              <Slides
                ref={slidesRef}
                jobId={jobId}
                slides={slides}
                timeRef={timeRef}
                clipStart={clipStart}
                playing={playing}
                showPointer={showPointer}
                onSeek={seekAndPlay}
                onTogglePlay={togglePlay}
                onAsk={askElement}
              />
            ) : (
              <div className="board-missing">
                {replay.type === 'slides' ? 'No slides were found in this recording.' : 'This replay has no board.'}
              </div>
            )}
            </div>
            {view3d && popout && board && (
              <Suspense fallback={<View3DFallback />}>
                <LazyView3D
                  key={popout.id}
                  jobId={jobId}
                  piece={popout}
                  boardColor={board.background ?? '#f4f4f2'}
                  aspect={boardAspect}
                  timeRef={timeRef}
                  clipStart={clipStart}
                  clipEnd={clipEnd}
                  playing={playing}
                  onSeek={seek}
                  onClose={closePopout}
                  onMade={onMade}
                />
              </Suspense>
            )}
          </section>
        </div>

        <div className="lower">
          <aside className="side">
            <Chapters chapters={chapters} current={chapter} onSeek={seekChapter}>
              <ul className="facts">
                {board && (
                  <li>
                    <b>{pieces.length}</b> pieces of writing{soFar}
                    {hiddenCount ? (
                      <>
                        , <b>{hiddenCount}</b> uncovered from behind the teacher
                      </>
                    ) : null}
                  </li>
                )}
                {slides && (
                  <li>
                    <b>{slideCount}</b> {slideCount === 1 ? 'slide' : 'slides'}
                    {buildSteps > 0 && (
                      <>
                        , <b>{buildSteps}</b> build {buildSteps === 1 ? 'step' : 'steps'}
                      </>
                    )}{' '}
                    · <b>{annotationCount}</b> pointer {annotationCount === 1 ? 'gesture' : 'gestures'} redrawn
                  </li>
                )}
                <li>
                  <b>{words.length}</b> words transcribed
                  {replay.stats?.seconds != null && !streaming && (
                    <>
                      {' '}
                      · processed in <b>{formatTime(replay.stats.seconds)}</b>
                    </>
                  )}
                  {api?.calls != null && !streaming && (
                    <>
                      {' '}
                      · <b>{api.calls}</b> Meta model calls
                    </>
                  )}
                </li>
              </ul>
            </Chapters>
          </aside>
          <Transcript
            turns={turns}
            words={words}
            groups={groups}
            wordTurn={wordTurn}
            active={active}
            upTo={upTo}
            currentTurn={turnIdx}
            timing={replay.transcript?.timing}
            refining={refining}
            pending={streaming && !failure}
            onSeek={seekWord}
            onAskSpan={askSpan}
          />
        </div>
      </main>

      {askMounted && (
        <AskDrawer
          jobId={jobId}
          open={askOpen}
          focusKey={askFocus}
          onClose={closeAsk}
          tray={tray}
          setTray={setTray}
          notice={askNotice}
          index={lecture}
          mode={mode}
          timeRef={timeRef}
          videoEnd={replay.video.duration}
          knows={ready != null ? knowsUpTo(ready) : null}
          onAddMoment={addMoment}
          onSeek={seekOnly}
          onShowItem={showItem}
        />
      )}

      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}

export function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" width="24" height="24" aria-hidden>
      {/* a board with two lines of chalk on it, and its ledge */}
      <rect x="2.5" y="5.5" width="27" height="18" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <path d="M7 11.5h11M7 16.5h6.5" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" />
      <path d="M1 27h30" stroke="currentColor" strokeWidth="1.7" />
    </svg>
  );
}
