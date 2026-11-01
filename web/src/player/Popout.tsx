// The 3D pop-out: a modal that asks the server for Muse Spark's scene of one board piece and shows it with
// three.js. This module (and three.js with it) is loaded on demand: see popoutLoader.tsx.
import { useEffect, useRef, useState } from 'react';
import { getPopout, jobFileUrl } from '../api';
import type { Piece, PopoutScene } from '../types';
import { mountStage, type Stage } from './popoutScene';

interface Props {
  jobId: string;
  piece: Piece;
  /** The board's colour, behind the thumbnail of the drawing. */
  boardColor: string;
  onClose(): void;
}

// One request per piece, shared: closing the modal while Muse Spark is still thinking and opening it again (or
// React running the effect twice in development) must not start a second model call. Answers are kept for the
// session; a failure is forgotten so "Try again" asks again.
const scenes = new Map<string, Promise<PopoutScene>>();
function sceneFor(jobId: string, pieceId: number): Promise<PopoutScene> {
  const key = `${jobId}/${pieceId}`;
  let p = scenes.get(key);
  if (!p) {
    p = getPopout(jobId, pieceId);
    scenes.set(key, p);
    p.catch(() => scenes.delete(key));
  }
  return p;
}

const cssVar = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

export default function Popout({ jobId, piece, boardColor, onClose }: Props) {
  const [scene, setScene] = useState<PopoutScene | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [slow, setSlow] = useState(false);
  const [legend, setLegend] = useState<Stage['legend']>([]);
  const [failed, setFailed] = useState<string | null>(null); // the scene came back but could not be drawn
  const [empty, setEmpty] = useState(false); // drawn, but nothing in it
  const hostRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Stage | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // Ask the server (it calls Muse Spark the first time, then answers from its cache).
  useEffect(() => {
    let alive = true;
    setScene(null);
    setError(null);
    setSlow(false);
    const slowTimer = window.setTimeout(() => setSlow(true), 2500);
    sceneFor(jobId, piece.id)
      .then(
        (s) => alive && setScene(s),
        (e: Error) => alive && setError(e.message || 'The 3D view could not be made.'),
      )
      .finally(() => window.clearTimeout(slowTimer));
    return () => {
      alive = false;
      window.clearTimeout(slowTimer);
    };
  }, [jobId, piece.id, attempt]);

  // Build the three.js stage once the scene is here; dispose every GPU resource when it goes.
  useEffect(() => {
    const host = hostRef.current;
    setFailed(null);
    setEmpty(false);
    setLegend([]);
    if (!scene || !host || scene.kind === 'none') return;
    let stage: Stage;
    try {
      stage = mountStage(host, scene, {
        accent: cssVar('--accent', '#7aa7ff'),
        reducedMotion: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
      });
    } catch (e) {
      host.replaceChildren(); // whatever was half built
      setFailed(
        /webgl/i.test((e as Error)?.message ?? '')
          ? 'This browser cannot show 3D views (WebGL is turned off or not available).'
          : 'This 3D view could not be drawn.',
      );
      return;
    }
    if (!stage.drawn) {
      stage.dispose();
      setEmpty(true);
      return;
    }
    stageRef.current = stage;
    setLegend(stage.legend);
    return () => {
      stage.dispose();
      stageRef.current = null;
    };
  }, [scene]);

  // Esc closes; focus moves into the dialog and back to where it was; the page behind does not scroll.
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    closeRef.current?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCloseRef.current();
      }
    };
    window.addEventListener('keydown', onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = overflow;
      before?.focus?.({ preventScroll: true });
    };
  }, []);

  const loading = !scene && !error;
  const none = !!scene && (scene.kind === 'none' || empty);
  const title = scene?.title?.trim() || piece.read?.text?.trim() || 'Drawing from the board';
  const live = !!scene && !none && !failed;

  return (
    <div
      className="popout-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="popout" role="dialog" aria-modal="true" aria-labelledby="popout-title" aria-busy={loading}>
        <header className="popout-head">
          {piece.image && (
            <div className="popout-thumb" style={{ background: boardColor }} aria-hidden>
              <img src={jobFileUrl(jobId, piece.image)} alt="" />
            </div>
          )}
          <div className="popout-titles">
            <h2 id="popout-title">{title}</h2>
            {scene?.explain && !none && <p className="popout-explain">{scene.explain}</p>}
            {live && (
              <p className="popout-credit">
                <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
                  <path d="M8 1l1.6 4.4L14 7l-4.4 1.6L8 13l-1.6-4.4L2 7l4.4-1.6z" fill="currentColor" />
                </svg>
                Made by Muse Spark from the board
              </p>
            )}
          </div>
          <button
            ref={closeRef}
            className="popout-close"
            onClick={onClose}
            aria-label="Close the 3D view"
            title="Close (Esc)"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
              <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </header>

        <div className="popout-stage">
          <div
            ref={hostRef}
            className="p3-host"
            role={live ? 'img' : undefined}
            aria-label={live ? `3D view you can turn: ${scene?.explain || title}` : undefined}
          />
          {loading && (
            <div className="popout-state" role="status">
              <span className="popout-spinner" aria-hidden />
              <p>Asking Muse Spark for a 3D view...</p>
              <small className={slow ? '' : 'hide'}>
                The first time takes a few seconds; after that it opens at once.
              </small>
            </div>
          )}
          {error && (
            <div className="popout-state error" role="alert">
              <p>{error}</p>
              <button className="ghost-btn" onClick={() => setAttempt((a) => a + 1)}>
                Try again
              </button>
            </div>
          )}
          {none && (
            <div className="popout-state">
              <svg viewBox="0 0 24 24" width="30" height="30" aria-hidden>
                <path
                  d="M12 2.8l8 4.6v9.2l-8 4.6-8-4.6V7.4z M4 7.4l8 4.6 8-4.6 M12 12v9.2"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinejoin="round"
                />
              </svg>
              <p>Nothing to show in 3D for this one.</p>
              {scene?.explain && <small>{scene.explain}</small>}
            </div>
          )}
          {failed && (
            <div className="popout-state error" role="alert">
              <p>{failed}</p>
            </div>
          )}
        </div>

        {live && (
          <footer className="popout-foot">
            <ul className="popout-legend">
              {legend.map((l, i) => (
                <li key={i}>
                  <span className="sw" style={{ background: l.color }} aria-hidden />
                  {l.label}
                </li>
              ))}
            </ul>
            <span className="popout-hint">Drag to turn · scroll to zoom · right-drag to move</span>
            <button className="ghost-btn" onClick={() => stageRef.current?.resetView()}>
              Reset view
            </button>
          </footer>
        )}
      </div>
    </div>
  );
}
