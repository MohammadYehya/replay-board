import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { jobFileUrl } from '../api';
import type { Piece, Replay } from '../types';
import { Board, type BoardHandle } from './Board';
import { activeWord, chapterAt, formatTime, groupWords, lastStartedWord, seekTarget, turnsFromWords } from './timing';
import { Chapters, Transcript } from './Transcript';

const SKIP = 5;

function Shortcuts({ onClose }: { onClose(): void }) {
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest('.shortcuts, .keys-btn')) onClose();
    };
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [onClose]);
  const rows: [string[], string][] = [
    [['Space'], 'Play / pause'],
    [['←', '→'], 'Back / forward 5 s'],
    [['T'], 'Handwriting / typeset'],
    [['Click writing'], 'Jump to where it is explained'],
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

export function Player({ jobId, replay, name }: { jobId: string; replay: Replay; name?: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const boardRef = useRef<BoardHandle>(null);
  const clipStart = replay.video.start ?? 0;
  const clipEnd = replay.video.end ?? replay.video.duration;
  const timeRef = useRef<number>(clipStart);
  const [typeset, setTypeset] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [active, setActive] = useState(-1);
  const [upTo, setUpTo] = useState(-1);
  const [turnIdx, setTurnIdx] = useState(-1);
  const [chapter, setChapter] = useState(-1);
  const [showKeys, setShowKeys] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
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

  // Keyboard: space, arrows, t. Ignored while typing in a field.
  useEffect(() => {
    let swallowSpaceUp = false;
    const typing = (el: EventTarget | null) => {
      const h = el as HTMLElement | null;
      return !!h && (h.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(h.tagName));
    };
    const down = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return;
      const v = videoRef.current;
      if (!v) return;
      let handled = true;
      if (e.key === ' ' || e.code === 'Space') {
        if (!e.repeat) togglePlay();
        swallowSpaceUp = true;
      } else if (e.key === 'ArrowLeft') seek(v.currentTime - SKIP);
      else if (e.key === 'ArrowRight') seek(v.currentTime + SKIP);
      else if (e.key === 't' || e.key === 'T') setTypeset((x) => !x);
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
  };

  const boardAspect = board ? board.size[0] / board.size[1] : 16 / 9;
  const partial = clipStart > 0.5 || clipEnd < replay.video.duration - 0.5;
  const pieces = board?.pieces ?? [];
  const hiddenCount = pieces.filter((p) => p.revealed && !p.pre_existing).length;
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
          <button className={`ghost-btn keys-btn ${showKeys ? 'on' : ''}`} onClick={() => setShowKeys((x) => !x)}>
            <kbd>?</kbd> Shortcuts
          </button>
          {showKeys && <Shortcuts onClose={() => setShowKeys(false)} />}
        </div>
      </header>

      <main className="player-main" style={rowStyle}>
        <div className="media-row">
          <section className="panel video-panel">
            <header className="panel-head">
              <h2>Lecture</h2>
              <span className="panel-note">
                original video, untouched
                {partial && ` · board covers ${formatTime(clipStart)}–${formatTime(clipEnd)}`}
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
              />
            ) : (
              <div className="board-missing">
                {replay.type === 'slides'
                  ? 'This is a slide recording. The slide view is not built yet.'
                  : 'This replay has no board.'}
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
            onSeek={seekWord}
          />
        </div>
      </main>

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
