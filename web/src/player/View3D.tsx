// The 3D view, docked in place of the virtual board beside the original video (which keeps playing): Muse Spark's
// scene of one board drawing, with steps that play along on the video's clock (one clock: the video). Loaded on
// demand with three.js: see popoutLoader.tsx.
//
// Every value the scene's parameters take shows on the stage as it changes; "Try it" opens a slider per parameter, so
// a student can move one and see what it does (on top of the lecture: the video keeps playing, and the other values
// keep following it) until they go back to the lecture.
import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { getScene3D, jobFileUrl } from '../api';
import { SparkIcon } from '../ask/icons';
import type { Piece, Scene3D, Scene3DParam } from '../types';
import { fieldTakesKey } from './keys';
import { mountScene, type Insets, type Stage3D } from './scene3dStage';
import {
  beforeLine, chipCaption, drawingFrom, drawnWith, dwell, playFrom, sceneLoader, sceneState, seekTimeFor, shapeGuard,
  stepHeading, tryValue, type SceneState, type ShapeGuard,
} from './scene3dState';
import { displayName, formatValue, sliderStep, viewTitle } from './sceneText';
import { formatTime } from './timing';

interface Props {
  jobId: string;
  piece: Piece;
  /** The board's colour, behind the thumbnail of the drawing. */
  boardColor: string;
  /** The board's width / height: the stage takes a shape near it, kept between STAGE_ASPECT's bounds. */
  aspect: number;
  /** The video's clock (seconds), kept up to date by the player. */
  timeRef: RefObject<number>;
  /** The analysed part of the video. */
  clipStart: number;
  clipEnd: number;
  /** Whether the video is playing: going to a step keeps it as it is. */
  playing: boolean;
  /** Seek the video (and play it when asked). */
  onSeek(t: number, play: boolean): void;
  /** Back to the board. */
  onClose(): void;
  /** A scene is here for this drawing (made now or before): the board's list of made views may be out of date. */
  onMade?(pieceId: number): void;
}

/** The stage's width / height: the board's shape (2.1 to 3.2 wide on the lectures tried) left the model a strip, so
 *  it is kept between these (and at least 340 px tall: styles.css). */
export const STAGE_ASPECT = { min: 1.25, max: 1.7 } as const;

// One request per piece, shared for the session (see sceneLoader: only a usable scene is kept).
const sceneFor = sceneLoader((jobId, pieceId) => getScene3D(jobId, pieceId));

type Load =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'old' }
  | { status: 'ready'; scene: Scene3D };

const OLD_SERVER =
  'This server still makes the old kind of 3D view: restart `python -m app.server` with the latest code.';

/** Typing in a field, or in the Ask Muse drawer: the step keys are left alone (as the player's own keys are). A
 *  "Try it" slider keeps only the keys that move it, so [ ] still work from it (keys.ts). */
const typing = (el: EventTarget | null, key: string) => {
  const h = el as HTMLInputElement | null;
  return !!h && (fieldTakesKey(h, key) || !!h.closest?.('.ask-drawer'));
};

/** What is laid over the stage, measured: it is absolute (over the drawing) on a wide view, in the flow on a narrow
 *  one (where it covers nothing). */
const over = (el: HTMLElement | null) => !!el && getComputedStyle(el).position === 'absolute';

/** The elements that show one parameter's value: in the strip, and on its slider. */
interface ValueEls {
  chip: HTMLElement | null;
  value: HTMLElement | null;
  row: HTMLElement | null;
  slider: HTMLInputElement | null;
  sliderValue: HTMLElement | null;
}

export default function View3D(props: Props) {
  const { jobId, piece, boardColor, aspect, timeRef, clipStart, clipEnd, playing, onSeek, onClose, onMade } = props;
  const [load, setLoad] = useState<Load>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);
  const [waited, setWaited] = useState(0);
  const [failed, setFailed] = useState<string | null>(null);
  const [index, setIndex] = useState(-1);
  const [turned, setTurned] = useState(false);
  const [more, setMore] = useState(false);
  /** A scene that is not synced: the viewer asked to play its steps one after another. */
  const [auto, setAuto] = useState(false);
  /** "Try it": the sliders are open. */
  const [trying, setTrying] = useState(false);
  /** The viewer moved a slider: their values are on top of the lecture's until they go back to it. */
  const [mine, setMine] = useState(false);
  const regionRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const captionRef = useRef<HTMLDivElement>(null);
  const valuesRef = useRef<HTMLDivElement>(null);
  const chipsRef = useRef<HTMLOListElement>(null);
  const stageRef = useRef<Stage3D | null>(null);
  const indexRef = useRef(-1);
  const playingRef = useRef(playing);
  playingRef.current = playing;
  /** The viewer's own values ("Try it"), read by the stage on every frame. */
  const overridesRef = useRef<Record<string, number>>({});
  /** The scene's check that its angles make a shape that exists (null: nothing to check), and every value as last
   *  drawn with the viewer's values it was drawn with: a slider stops where the shape would stop existing with the
   *  other values as they are. */
  const guardRef = useRef<ShapeGuard | null>(null);
  const drawnRef = useRef<{ params: Record<string, number>; own: Record<string, number> }>({ params: {}, own: {} });
  const valueEls = useRef(new Map<string, ValueEls>()).current;
  /** What the values show now (name -> "text|changing|mine"), so a frame touches only what changed. */
  const painted = useRef(new Map<string, string>()).current;
  /** The parameter whose slider the viewer is pressing now (the lecture never moves it under their finger). */
  const heldRef = useRef<string | null>(null);
  /** A scene that is not synced: the step the viewer went to, and when (performance.now()). */
  const vstep = useRef({ index: -1, since: 0 });
  const reduced = useMemo(() => !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches, []);
  const onMadeRef = useRef(onMade);
  onMadeRef.current = onMade;

  // Ask the server (Muse Spark the first time, then its cache). The seconds waited show while it thinks.
  useEffect(() => {
    let alive = true;
    setLoad({ status: 'loading' });
    setWaited(0);
    const t0 = performance.now();
    const tick = window.setInterval(() => setWaited(Math.floor((performance.now() - t0) / 1000)), 1000);
    sceneFor(jobId, piece.id)
      .then(
        (r) => {
          if (!alive) return;
          if (r.kind === 'scene') {
            setLoad({ status: 'ready', scene: r.scene });
            if (r.scene.objects.length) onMadeRef.current?.(piece.id);
          } else if (r.kind === 'old') setLoad({ status: 'old' });
          else setLoad({ status: 'error', message: r.why });
        },
        (e: Error) => alive && setLoad({ status: 'error', message: e.message || 'The 3D view could not be made.' }),
      )
      .finally(() => window.clearInterval(tick));
    return () => {
      alive = false;
      window.clearInterval(tick);
    };
  }, [jobId, piece.id, attempt]);

  const scene = load.status === 'ready' ? load.scene : null;

  /** Shows every value as it is at this frame: the changing one stands out, the viewer's own are marked. Straight
   *  into the page (no re-render on every frame). */
  const paint = useCallback(
    (sc: Scene3D, st: SceneState) => {
      for (const p of sc.params) {
        const els = valueEls.get(p.name);
        if (!els) continue;
        const v = st.params[p.name];
        const text = formatValue(v, p.unit);
        const changing = st.changing.includes(p.name);
        const own = st.overridden.includes(p.name);
        const key = `${text}|${changing}|${own}`;
        if (painted.get(p.name) === key) continue;
        painted.set(p.name, key);
        if (els.value) els.value.textContent = text;
        if (els.sliderValue) els.sliderValue.textContent = text;
        for (const el of [els.chip, els.row]) {
          el?.classList.toggle('changing', changing);
          el?.classList.toggle('mine', own);
        }
        const s = els.slider;
        if (s) {
          // the slider follows the lecture until the viewer takes it (never while they hold it). It follows it with
          // the focus too: [ ] work from a slider and give the values back to the lecture (its thumb stayed where
          // the viewer left it, under the lecture's value), and the arrow keys then start from the value shown. The
          // viewer's own value follows too when the lecture moves another angle so far that theirs gives way.
          if (heldRef.current !== p.name && (!own || Math.abs(Number(s.value) - v) > 1e-9)) s.value = String(v);
          s.setAttribute('aria-valuetext', text);
        }
      }
    },
    [valueEls, painted],
  );

  // Build the three.js stage once the scene is here; every GPU resource goes with it.
  useEffect(() => {
    const host = hostRef.current;
    setFailed(null);
    setTurned(false);
    setAuto(false);
    indexRef.current = -1;
    setIndex(-1);
    overridesRef.current = {};
    setMine(false);
    painted.clear();
    guardRef.current = null;
    drawnRef.current = { params: {}, own: {} };
    if (!scene || !host || !scene.objects.length) return;
    const guard = shapeGuard(scene);
    guardRef.current = guard;
    vstep.current = { index: -1, since: performance.now() };
    let stage: Stage3D;
    try {
      stage = mountScene(host, scene, {
        accent: '#7aa7ff', // the model's first colour (black or grey ink), not the page's "now" yellow
        reducedMotion: reduced,
        // What covers the stage where it is laid over it (a wide view): the caption at the bottom, the values at the
        // top, the open sliders at the left. On a narrow view they are above and under the canvas and cover nothing.
        insets: (): Insets => {
          const out = { top: 0, right: 0, bottom: 0, left: 0 };
          const c = captionRef.current;
          if (c && over(c)) out.bottom = c.offsetHeight + 12;
          const v = valuesRef.current;
          if (v && over(v)) {
            if (v.classList.contains('open')) out.left = v.offsetLeft + v.offsetWidth + 6;
            else out.top = v.offsetTop + v.offsetHeight + 4;
          }
          return out;
        },
        insetElements: () => [captionRef.current, valuesRef.current],
        frame: (now) => {
          const opts = { reducedMotion: reduced, overrides: overridesRef.current, guard };
          const st = scene.synced
            ? sceneState(scene, { clock: timeRef.current ?? 0 }, opts)
            : sceneState(scene, { step: vstep.current.index, elapsed: (now - vstep.current.since) / 1000 }, opts);
          drawnRef.current = { params: st.params, own: opts.overrides };
          if (st.index !== indexRef.current) {
            indexRef.current = st.index;
            setIndex(st.index);
          }
          paint(scene, st);
          return st;
        },
        onTurn: setTurned,
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
    stageRef.current = stage;
    return () => {
      stage.dispose();
      stageRef.current = null;
    };
  }, [scene, timeRef, reduced, paint, painted]);

  // The view takes the keyboard focus when it opens (the button that opened it is gone); the player's keys still
  // work, and Esc goes back to the board (Player.tsx).
  useEffect(() => {
    regionRef.current?.focus({ preventScroll: true });
  }, []);

  // Keep the current step's chip in sight inside the strip (without scrolling the page).
  useEffect(() => {
    const ol = chipsRef.current;
    const li = ol?.children[Math.max(0, index)] as HTMLElement | undefined;
    if (!ol || !li) return;
    const left = li.offsetLeft;
    const right = left + li.offsetWidth;
    let to = ol.scrollLeft;
    if (left < ol.scrollLeft + 8) to = left - 8;
    else if (right > ol.scrollLeft + ol.clientWidth - 8) to = right - ol.clientWidth + 8;
    if (to !== ol.scrollLeft) ol.scrollTo({ left: Math.max(0, to), behavior: reduced ? 'auto' : 'smooth' });
  }, [index, reduced, scene]);

  /** Back to the lecture's values: the sliders follow it again. */
  const endOverrides = useCallback(() => {
    if (!Object.keys(overridesRef.current).length) return;
    overridesRef.current = {};
    setMine(false);
  }, []);
  /** A slider moved: that value is the viewer's from now on (on top of the lecture), until they go back. An angle
   *  that would make a shape that cannot exist with the other values as they are stops at the last one that can, on
   *  the slider's steps, and so does its slider (test2 #7: with gamma at 120, alpha goes as far as the cell lets it);
   *  the viewer's other angles stay where they are shown (tryValue). */
  const override = useCallback(
    (p: Scene3DParam, value: number) => {
      if (!Number.isFinite(value)) return;
      const grid = { min: Math.min(p.min, p.max), step: sliderStep(p.min, p.max, p.unit) };
      const own = overridesRef.current;
      const next = tryValue(guardRef.current, own, drawnWith(drawnRef.current, own), p.name, value, grid);
      const v = next[p.name];
      if (v !== value) {
        const s = valueEls.get(p.name)?.slider;
        if (s) s.value = String(v);
      }
      overridesRef.current = next;
      setMine(true);
    },
    [valueEls],
  );

  const n = scene?.steps.length ?? 0;
  /** Go to step i (-1: before the first). Synced: the video seeks there and keeps playing or stays paused (one
   *  clock; paused, it shows the step done). */
  const goTo = useCallback(
    (i: number) => {
      if (!scene) return;
      if (scene.synced) {
        const t = i < 0 ? playFrom(scene, clipStart) : seekTimeFor(scene, i, playingRef.current);
        if (t !== null) onSeek(t, playingRef.current);
        return;
      }
      const k = Math.max(-1, Math.min(n - 1, i));
      vstep.current = { index: k, since: performance.now() };
      indexRef.current = k;
      setIndex(k);
    },
    [scene, clipStart, onSeek, n],
  );
  /** A step the viewer picks by hand stops "Play the steps" and goes back to the lecture's values. */
  const pick = useCallback(
    (i: number) => {
      setAuto(false);
      endOverrides();
      goTo(i);
    },
    [goTo, endOverrides],
  );

  // "Play the steps" (a scene not tied to the video): each step stays for its change and time to read its caption.
  useEffect(() => {
    if (!auto || !scene || scene.synced) return;
    if (index >= n - 1) {
      const id = window.setTimeout(() => setAuto(false), (scene.steps[index] ? dwell(scene.steps[index]) : 0) * 1000);
      return () => window.clearTimeout(id);
    }
    const wait = index < 0 ? 0.3 : dwell(scene.steps[index]);
    const id = window.setTimeout(() => goTo(index + 1), wait * 1000);
    return () => window.clearTimeout(id);
  }, [auto, index, n, scene, goTo]);

  // [ and ]: the previous and next step (the player's own keys are untouched).
  useEffect(() => {
    if (!scene || !n) return;
    const down = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented || typing(e.target, e.key)) return;
      if (e.key !== '[' && e.key !== ']') return;
      e.preventDefault();
      const i = indexRef.current;
      if (e.key === '[' && i >= 0) pick(i - 1);
      else if (e.key === ']' && i < n - 1) pick(i + 1);
    };
    window.addEventListener('keydown', down);
    return () => window.removeEventListener('keydown', down);
  }, [scene, n, pick]);

  const loading = load.status === 'loading';
  const none = !!scene && !scene.objects.length;
  const live = !!scene && !none && !failed;
  const title = viewTitle(scene?.title, piece.read?.text);
  const step = scene && index >= 0 ? scene.steps[index] : null;
  const stageAspect = Math.min(STAGE_ASPECT.max, Math.max(STAGE_ASPECT.min, aspect || STAGE_ASPECT.max));
  const fromDrawing = scene
    ? drawingFrom(scene, (piece.spoken ?? []).map((s) => s.start), clipStart, clipEnd)
    : null;
  const params = live ? scene!.params : [];
  const els = (name: string) => {
    let e = valueEls.get(name);
    if (!e) valueEls.set(name, (e = { chip: null, value: null, row: null, slider: null, sliderValue: null }));
    return e;
  };
  /** Shown before the first frame paints them. */
  const startValue = (p: Scene3DParam) => formatValue(p.value, p.unit);

  return (
    <div className="v3" ref={regionRef} role="region" aria-label="3D view" tabIndex={-1} aria-busy={loading}>
      <header className="panel-head v3-bar">
        {piece.image && (
          <div className="v3-thumb" style={{ background: boardColor }} aria-hidden>
            <img src={jobFileUrl(jobId, piece.image)} alt="" />
          </div>
        )}
        <h2 className="v3-title" title={title}>
          <span className="v3-kicker">3D</span>
          <span className="v3-title-text">{title}</span>
        </h2>
        <button className="ghost-btn v3-back" onClick={onClose} title="Back to the board (Esc)">
          <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden>
            <path d="M9.5 3.5L5 8l4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Back to board
          <kbd>Esc</kbd>
        </button>
      </header>

      {scene?.explain && !none && (
        <div className={`v3-about ${more ? 'open' : ''}`}>
          <span className="v3-spark" title="Made by Muse Spark from the board and what was said">
            <SparkIcon size={12} />
          </span>
          <p className="v3-explain" title={more ? undefined : displayName(scene.explain)}>
            {displayName(scene.explain)}
          </p>
          <button className="v3-more" onClick={() => setMore((m) => !m)} aria-expanded={more}>
            {more ? 'Less' : 'More'}
          </button>
        </div>
      )}

      <div className="v3-stage" style={{ aspectRatio: `${stageAspect}` }}>
        <div
          ref={hostRef}
          className="p3-host"
          role={live ? 'img' : undefined}
          aria-label={live ? `3D view you can turn: ${displayName(scene?.explain || title)}` : undefined}
        />
        {loading && (
          <div className="popout-state" role="status">
            <span className="popout-spinner" aria-hidden />
            <p>
              Asking Muse Spark for a 3D explanation...
              {waited >= 2 && <span className="popout-waited"> {waited} s</span>}
            </p>
            <small className={waited >= 2 ? '' : 'hide'}>
              The first time can take a minute or two: Muse Spark reads the drawing and the whole lecture. After that it
              opens at once.
            </small>
          </div>
        )}
        {load.status === 'error' && (
          <div className="popout-state error" role="alert">
            <p>{load.message}</p>
            <button className="ghost-btn" onClick={() => setAttempt((a) => a + 1)}>
              Try again
            </button>
          </div>
        )}
        {load.status === 'old' && (
          <div className="popout-state error" role="alert">
            <p>{OLD_SERVER}</p>
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
            {scene?.explain && <small>{displayName(scene.explain)}</small>}
          </div>
        )}
        {failed && (
          <div className="popout-state error" role="alert">
            <p>{failed}</p>
          </div>
        )}
        {live && params.length > 0 && (
          <div
            ref={valuesRef}
            className={`v3-values ${trying ? 'open' : ''} ${mine ? 'has-mine' : ''}`}
            role="group"
            aria-label="Values in the drawing"
          >
            {!trying && (
            <div className="v3-vals">
              {params.map((p) => (
                  <span
                    key={p.name}
                    className="v3-val"
                    ref={(el) => {
                      els(p.name).chip = el;
                      if (!el) painted.delete(p.name);
                    }}
                  >
                    <span className="k">{displayName(p.label)}</span>
                    <span className="v" ref={(el) => void (els(p.name).value = el)}>
                      {startValue(p)}
                    </span>
                  </span>
                ))}
            </div>
            )}
            {trying && (
              <div className="v3-sliders" id={`v3-try-${piece.id}`}>
                <p className="v3-try-head">Move a value: the drawing follows. The video keeps playing.</p>
                {params.map((p) => {
                  const id = `v3-s-${piece.id}-${p.name}`;
                  return (
                    <div
                      key={p.name}
                      className="v3-slider"
                      ref={(el) => {
                        els(p.name).row = el;
                        if (!el) painted.delete(p.name);
                      }}
                    >
                      <label htmlFor={id} className="k">
                        {displayName(p.label)}
                      </label>
                      <span className="v" ref={(el) => void (els(p.name).sliderValue = el)} aria-hidden>
                        {startValue(p)}
                      </span>
                      <input
                        id={id}
                        type="range"
                        min={p.min}
                        max={p.max}
                        step={sliderStep(p.min, p.max, p.unit)}
                        defaultValue={p.value}
                        aria-valuetext={startValue(p)}
                        ref={(el) => {
                          els(p.name).slider = el;
                          if (el) painted.delete(p.name); // painted again: the slider starts at the value shown
                        }}
                        onInput={(e) => override(p, Number(e.currentTarget.value))}
                        onPointerDown={() => {
                          heldRef.current = p.name;
                          const up = () => {
                            heldRef.current = null;
                            window.removeEventListener('pointerup', up);
                            window.removeEventListener('pointercancel', up);
                          };
                          window.addEventListener('pointerup', up);
                          window.addEventListener('pointercancel', up);
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            )}
            {mine && (
              <div className="v3-mine" role="status">
                <span className="v3-mine-badge">Your values</span>
                <button className="v3-mine-back" onClick={endOverrides}>
                  Back to the lecture
                </button>
              </div>
            )}
          </div>
        )}
        {live && (
          <>
            <button
              className={`v3-reset ${turned ? 'on' : ''}`}
              onClick={() => {
                endOverrides();
                stageRef.current?.resetView();
              }}
              title={
                turned
                  ? 'You turned the view, so the steps no longer move the camera. Reset gives it back to them.'
                  : 'Back to the starting view'
              }
            >
              <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden>
                <path
                  d="M3.2 8a4.8 4.8 0 108.4-3.2M11.8 1.8v3.2H8.6"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Reset view
            </button>
            <div ref={captionRef} className="v3-caption" aria-live="polite">
              <div className="v3-cap-row">
                {step && <span className="v3-cap-head">{stepHeading(scene!, index)}</span>}
                <span className="v3-hint" aria-hidden>
                  Drag to turn · Ctrl+scroll to zoom · right-drag to move
                </span>
              </div>
              {step ? (
                step.caption &&
                (mine ? (
                  // The caption tells the lecture's values; with the viewer's own it may no longer be true.
                  <span className="v3-cap-text lecture">
                    <span className="v3-cap-from">In the lecture:</span> {displayName(step.caption)}
                  </span>
                ) : (
                  <span className="v3-cap-text">{displayName(step.caption)}</span>
                ))
              ) : (
                <span className="v3-cap-text muted">{beforeLine(scene!)}</span>
              )}
            </div>
          </>
        )}
      </div>

      {live && n > 0 && (
        <nav className="v3-steps" aria-label="Steps">
          <button
            className="v3-nav"
            onClick={() => pick(index - 1)}
            disabled={index < 0}
            aria-label="Previous step"
            title="Previous step ( [ )"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
              <path d="M10 3.5L5.5 8l4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <ol className="v3-chips" ref={chipsRef}>
            {scene!.steps.map((s, i) => (
              <li key={i}>
                <button
                  className={`v3-chip ${i === index ? 'on' : i < index ? 'done' : ''}`}
                  aria-current={i === index ? 'step' : undefined}
                  onClick={() => pick(i)}
                  title={displayName(s.caption) || `Step ${i + 1}`}
                >
                  <span className="n">{scene!.synced && s.t !== null ? formatTime(s.t) : i + 1}</span>
                  <span className="c">{chipCaption(displayName(s.caption)) || `Step ${i + 1}`}</span>
                </button>
              </li>
            ))}
          </ol>
          <button
            className="v3-nav"
            onClick={() => pick(index + 1)}
            disabled={index >= n - 1}
            aria-label="Next step"
            title="Next step ( ] )"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
              <path d="M6 3.5L10.5 8 6 12.5" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </nav>
      )}

      {live && (
        <footer className="v3-foot">
          {scene!.synced ? (
            <>
              <button
                className="v3-play"
                onClick={() => {
                  endOverrides();
                  onSeek(playFrom(scene!, clipStart) ?? clipStart, true);
                }}
              >
                <PlayIcon />
                Play the explanation
                <span className="v3-play-t">from {formatTime(playFrom(scene!, clipStart) ?? 0)}</span>
              </button>
              {fromDrawing !== null && (
                <button
                  className="v3-play second"
                  onClick={() => {
                    endOverrides();
                    onSeek(fromDrawing, true);
                  }}
                >
                  <PlayIcon />
                  This drawing
                  <span className="v3-play-t">from {formatTime(fromDrawing)}</span>
                </button>
              )}
            </>
          ) : (
            n > 0 && (
              <>
                <button
                  className="v3-play"
                  aria-pressed={auto}
                  onClick={() => {
                    endOverrides();
                    if (auto) return setAuto(false);
                    if (index >= n - 1) goTo(-1);
                    setAuto(true);
                  }}
                >
                  {auto ? <PauseIcon /> : <PlayIcon />}
                  {auto ? 'Pause the steps' : 'Play the steps'}
                </button>
                <span className="v3-note">Not tied to the video: the drawing is never talked about.</span>
              </>
            )
          )}
          {params.length > 0 && (
            <button
              className="v3-try"
              aria-pressed={trying}
              aria-controls={trying ? `v3-try-${piece.id}` : undefined}
              onClick={() => {
                painted.clear();
                setTrying((x) => !x);
              }}
              title={trying ? 'Hide the sliders' : 'Move the values in the drawing yourself and see what changes'}
            >
              <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
                <path d="M2 4.5h12M2 11.5h12" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                <circle cx="5.5" cy="4.5" r="1.9" fill="currentColor" />
                <circle cx="10.5" cy="11.5" r="1.9" fill="currentColor" />
              </svg>
              {trying ? 'Done trying' : 'Try it'}
            </button>
          )}
        </footer>
      )}
    </div>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
      <path d="M4 3l9 5-9 5z" fill="currentColor" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
      <path d="M4 3h3v10H4zM9 3h3v10H9z" fill="currentColor" />
    </svg>
  );
}
