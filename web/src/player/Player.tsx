import { Suspense, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { getJob, getReplay, jobFileUrl } from '../api';
import type { Piece, Replay } from '../types';
import { Board, type BoardHandle } from './Board';
import { Slides, type SlidesHandle } from './Slides';
import { drawable } from './slideTiming';
import { activeWord, chapterAt, formatTime, groupWords, lastStartedWord, seekTarget, turnsFromWords } from './timing';
import { LazyPopout, PopoutFallback } from './popoutLoader';
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
          [['Click slide text'], 'Jump to where it is explained'],
          [['Click slide'], 'Play / pause'],
          [['Click a word'], 'Jump to that word'],
        ]
      : [
          [['Space'], 'Play / pause'],
          [['←', '→'], 'Back / forward 5 s'],
          [['T'], 'Handwriting / typeset'],
          [['Click writing'], 'Jump to where it is explained'],
          [['Hover a graph'], 'View it in 3D (3D button)'],
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
  startAt = null,
}: {
  jobId: string;
  replay: Replay;
  name?: string;
  /** Open at this moment (seconds on the video's clock), e.g. from a link in the notes. */
  startAt?: number | null;
  /** Whether the job was still running when the replay was loaded (false: its word times are final). */
  running?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const boardRef = useRef<BoardHandle>(null);
  const slidesRef = useRef<SlidesHandle>(null);
  // The replay can be swapped for a rewritten one (refined word times) while it plays; the video is never reset.
  const [replay, setReplay] = useState(first);
  // Proportional word times are refined only while the job runs; a finished job keeps the ones it has.
  const refinable = (r: Replay) => r.transcript?.timing === 'proportional' && running !== false;
  const [refining, setRefining] = useState(() => refinable(first));
  useEffect(() => {
    setReplay(first);
    setRefining(refinable(first));
  }, [first]); // `running` describes the job when this replay was loaded
  const clipStart = replay.video.start ?? 0;
  const clipEnd = replay.video.end ?? replay.video.duration;
  const timeRef = useRef<number>(clipStart);
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

  const words = useMemo(
    () =>
      (replay.transcript?.words ?? [])
        .filter((w) => w && Number.isFinite(w.s) && Number.isFinite(w.e))
        .slice()
        .sort((a, b) => a.s - b.s),
    [replay],
  );
  const turns = useMemo(() => {
    const given = (replay.transcript?.turns ?? []).slice().sort((a, b) => a.start - b.start);
    return given.length || !words.length ? given : turnsFromWords(words);
  }, [replay, words]);
  const groups = useMemo(() => groupWords(turns, words), [turns, words]);
  const wordTurn = useMemo(() => {
    const a = new Int32Array(words.length).fill(-1);
    groups.forEach((g, ti) => g.forEach((i) => (a[i] = ti)));
    return a;
  }, [groups, words]);
  const chapters = useMemo(() => (replay.chapters ?? []).slice().sort((a, b) => a.t - b.t), [replay]);
  const board = replay.board;
  const slides = board ? undefined : replay.slides;
  const mode = board ? 'board' : slides ? 'slides' : 'none';

  // The first version of a replay has proportional word times while MMS still aligns them: watch the job and
  // swap in the rewritten replay (transcript, spoken ranges, chapters, redraw times) as soon as it lands.
  useEffect(() => {
    if (!refining) return;
    let stopped = false;
    let timer: number | undefined;
    const ctrl = new AbortController();
    const poll = async () => {
      try {
        const st = await getJob(jobId, ctrl.signal);
        if (stopped) return;
        const align = st.stages?.align?.state;
        const ended = st.state !== 'running' && st.state !== 'queued';
        if (ended || align === 'done' || align === 'skipped') {
          const next = await getReplay(jobId, ctrl.signal);
          if (stopped) return;
          // The aligner finishes a moment before the replay is rewritten: wait for the new file unless the job ended.
          if (next.transcript?.timing !== 'proportional' || ended) {
            setReplay(next);
            setRefining(false);
            return;
          }
        }
      } catch (e) {
        if (stopped || (e as Error).name === 'AbortError') return;
        // the server may be restarting, or the file is being replaced: try again
      }
      timer = window.setTimeout(poll, 4000);
    };
    timer = window.setTimeout(poll, 4000);
    return () => {
      stopped = true;
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, [jobId, refining]);

  // One clock: the video's currentTime drives the board and the transcript, even while paused or scrubbing.
  useEffect(() => {
    let raf = 0;
    let prev = -1;
    const last = { active: -2, upTo: -2, turn: -2, chapter: -2 };
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
        boardRef.current?.render(t);
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
  }, [words, wordTurn, turns, chapters, clipEnd]);

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
  // The 3D view is explored with the lecture paused; it does not resume by itself.
  const openPopout = useCallback((p: Piece) => {
    videoRef.current?.pause();
    setPopout(p);
  }, []);
  const closePopout = useCallback(() => setPopout(null), []);

  // Keyboard: space, arrows, t (board), p (slides). Ignored while typing in a field.
  const modeRef = useRef(mode);
  modeRef.current = mode;
  useEffect(() => {
    let swallowSpaceUp = false;
    const typing = (el: EventTarget | null) => {
      const h = el as HTMLElement | null;
      return !!h && (h.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(h.tagName));
    };
    const down = (e: KeyboardEvent) => {
      if (popoutOpen.current) return; // the 3D view has the keyboard (Esc closes it)
      if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return;
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
      else if (e.key === 'Escape') setShowKeys(false);
      else handled = false;
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
  }, [seek, togglePlay]);

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
  const startedAt = useRef(false);

  const viewSize = board?.size ?? slides?.size;
  const boardAspect = viewSize && viewSize[0] > 0 && viewSize[1] > 0 ? viewSize[0] / viewSize[1] : 16 / 9;
  const partial = clipStart > 0.5 || clipEnd < replay.video.duration - 0.5;
  const pieces = board?.pieces ?? [];
  const hiddenCount = pieces.filter((p) => p.revealed && !p.pre_existing).length;
  const gestureCount = (board?.annotations ?? []).filter(drawable).length; // the hand pointing at writing, as drawn
  const slideStates = slides?.states ?? [];
  const slideCount = slideStates.filter((s) => s.kind !== 'addition').length || slideStates.length;
  const buildSteps = slideStates.length - slideCount;
  const annotationCount = slides?.annotations?.length ?? 0;
  const api = replay.stats?.api;
  const subject = replay.subject?.trim();
  const rowStyle = {
    '--cols': `${videoAspect.toFixed(4)}fr ${boardAspect.toFixed(4)}fr`,
    '--av': videoAspect.toFixed(4),
    '--ab': boardAspect.toFixed(4),
  } as CSSProperties;

  return (
    <div className="player">
      <header className="topbar">
        <a className="brand" href="#/" title="All lectures">
          <Logo />
          <span>Replay Board</span>
        </a>
        <div className="topbar-title">
          <span className="subject">{subject ? subject.charAt(0).toUpperCase() + subject.slice(1) : 'Lecture'}</span>
          {name && <span className="file-name">{name}</span>}
        </div>
        <div className="topbar-actions">
          <a className="ghost-btn" href="#/">
            All lectures
          </a>
          <a
            className="ghost-btn notes-btn"
            href={`#/job/${encodeURIComponent(jobId)}/notes`}
            title="Study notes for this lecture, by chapter"
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
            Notes
          </a>
          <button className={`ghost-btn keys-btn ${showKeys ? 'on' : ''}`} onClick={() => setShowKeys((x) => !x)}>
            <kbd>?</kbd> Shortcuts
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
                {partial &&
                  ` · ${slides ? 'slides cover' : 'board covers'} ${formatTime(clipStart)}–${formatTime(clipEnd)}`}
              </span>
            </header>
            <div className="video-frame" style={{ aspectRatio: `${videoAspect}` }}>
              <video
                ref={videoRef}
                src={jobFileUrl(jobId, replay.video.file)}
                controls
                preload="auto"
                playsInline
                onLoadedMetadata={onLoadedMetadata}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onLoadedData={() => setVideoError(null)}
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
            {videoError && <p className="error-line">{videoError}</p>}
          </section>

          <section className="panel board-panel">
            <header className="panel-head">
              <h2>{board || replay.type !== 'slides' ? 'Board' : 'Slides'}</h2>
              {board && (
                <>
                  <span className="panel-note">
                    {pieces.length} {pieces.length === 1 ? 'piece' : 'pieces'} of writing
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
                timeRef={timeRef}
                clipStart={clipStart}
                playing={playing}
                onSeekPiece={seekPiece}
                onTogglePlay={togglePlay}
                onPopout={openPopout}
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
              />
            ) : (
              <div className="board-missing">
                {replay.type === 'slides' ? 'No slides were found in this recording.' : 'This replay has no board.'}
              </div>
            )}
          </section>
        </div>

        <div className="lower">
          <aside className="side">
            <Chapters chapters={chapters} current={chapter} onSeek={seekChapter}>
              <ul className="facts">
                {board && (
                  <li>
                    <b>{pieces.length}</b> pieces of writing
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
                  {replay.stats?.seconds != null && (
                    <>
                      {' '}
                      · processed in <b>{formatTime(replay.stats.seconds)}</b>
                    </>
                  )}
                  {api?.calls != null && (
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
            onSeek={seekWord}
          />
        </div>
      </main>

      {popout && board && (
        <Suspense fallback={<PopoutFallback />}>
          <LazyPopout
            jobId={jobId}
            piece={popout}
            boardColor={board.background ?? '#f4f4f2'}
            onClose={closePopout}
          />
        </Suspense>
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
    <svg className="logo" viewBox="0 0 32 32" width="22" height="22" aria-hidden>
      <rect x="2" y="5" width="28" height="19" rx="3.5" fill="none" stroke="currentColor" strokeWidth="2" />
      <path
        d="M7 18c3-6 5-6 7-2s4 4 7-4"
        fill="none"
        stroke="var(--accent)"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M12 28h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
