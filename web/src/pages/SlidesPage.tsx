// The lecture as a slide deck: one slide per stretch of the lecture (between chapters and big erases on a board, one
// per slide in a slide lecture), each with the board or slide as it stood, Muse Spark's title and key points, and a
// link back into the player. Present shows one slide at a time, full screen; Save as PDF prints one slide per
// landscape page; the PowerPoint is made by the server (app/export.py).
import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react';
import { getSlides, getSlidesPptx, jobFileUrl, slidesPptxUrl, writeSlidePoints } from '../api';
import { SparkIcon } from '../ask/icons';
import { texHtml } from '../player/tex';
import {
  clampIndex, loadPrefs, playerAt, pointsBeside, presentKey, rangeLabel, savePrefs, sentence, typesetList,
  type SlidesPrefs, type SlidesView,
} from '../slides/deck';
import { PlayIcon, PresentIcon } from '../slides/icons';
import type { Deck, DeckPage } from '../types';

/** One key-points call per deck version at a time, even when the page mounts twice (React's StrictMode in dev). */
const pointsCalls = new Map<string, Promise<Deck>>();

/** Pictures failing together cause one fresh read of the deck, not one each. */
const REFRESH_GAP_MS = 4000;

function localStore(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** A slide's picture. One that cannot be loaded (the deck was rebuilt meanwhile, the file is gone) says so in a
 *  plain line instead of a broken-image icon. Keyed by its address, so a new deck version tries again. */
function DeckImage({ src, alt, size, onError }: { src: string; alt: string; size: [number, number] | null; onError(): void }) {
  const [broken, setBroken] = useState(false);
  if (broken) return <p className="deck-empty">The picture for this slide could not be loaded.</p>;
  return (
    <img
      className="deck-img"
      src={src}
      width={size?.[0]}
      height={size?.[1]}
      alt={alt}
      decoding="async"
      onError={() => {
        setBroken(true);
        onError();
      }}
    />
  );
}

/** The board (or slide) image of a page, or its items typeset. Items link into the player unless presenting. */
function SlideVisual({
  jobId,
  deck,
  page,
  view,
  links,
  onImageError,
}: {
  jobId: string;
  deck: Deck;
  page: DeckPage;
  view: SlidesView;
  links: boolean;
  onImageError(): void;
}) {
  if (view === 'typeset') {
    const entries = typesetList(page);
    if (!entries.length) return <p className="deck-empty">Talk only: nothing new was written.</p>;
    return (
      <ol className="deck-typeset">
        {entries.map((e, i) => {
          const html = e.tex ? texHtml(e.tex) : null;
          const body = html ? (
            <span className="deck-ts-tex" dangerouslySetInnerHTML={{ __html: html }} />
          ) : (
            <span className={e.drawing ? 'deck-ts-drawing' : 'deck-ts-text'}>{e.text}</span>
          );
          return (
            <li key={`${e.ref}-${i}`}>
              {links ? (
                <a className="deck-ts-item" href={playerAt(jobId, e.t)} title="Watch this explained">
                  {body}
                </a>
              ) : (
                <span className="deck-ts-item">{body}</span>
              )}
            </li>
          );
        })}
      </ol>
    );
  }
  if (!page.image) return <p className="deck-empty">No picture for this slide.</p>;
  const src = `${jobFileUrl(jobId, page.image)}${deck.version ? `?v=${encodeURIComponent(deck.version)}` : ''}`;
  const what = deck.type === 'slides' ? 'The slide' : 'The board';
  return (
    <DeckImage key={src} src={src} size={page.image_size} alt={`${what} for “${page.title}”`} onError={onImageError} />
  );
}

function Points({ page, writing }: { page: DeckPage; writing: boolean }) {
  if (page.points?.length)
    return (
      <ul className="deck-points">
        {page.points.map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ul>
    );
  if (writing)
    return (
      <p className="deck-points-wait">
        <SparkIcon size={12} />
        Muse Spark is writing the key points…
      </p>
    );
  return null;
}

function SlideCard({
  jobId,
  deck,
  page,
  index,
  view,
  said,
  writing,
  onPresent,
  onImageError,
}: {
  jobId: string;
  deck: Deck;
  page: DeckPage;
  index: number;
  view: SlidesView;
  said: boolean;
  writing: boolean;
  onPresent(index: number): void;
  onImageError(): void;
}) {
  const hasPoints = !!page.points?.length;
  const beside = view === 'image' && hasPoints && pointsBeside(page.image_size);
  const headId = `slide-title-${index + 1}`;
  return (
    <article className="deck-sheet" id={`slide-${index + 1}`} aria-labelledby={headId}>
      <div className={`deck-card ${hasPoints ? 'has-points' : ''}`}>
        <header className="deck-head">
          <span className="deck-num">
            <span className="sr-only">Slide </span>
            {index + 1}
          </span>
          <div className="deck-heading">
            {page.chapter && page.chapter !== page.title && <p className="deck-chapter">{page.chapter}</p>}
            <h2 id={headId}>{page.title}</h2>
          </div>
          <div className="deck-meta">
            <span className="deck-time">{rangeLabel(page.t0, page.t1)}</span>
            <a className="deck-play" href={playerAt(jobId, page.t0)} title="Watch this part of the lecture">
              <PlayIcon />
              Play
            </a>
            <button
              className="deck-from-here"
              onClick={() => onPresent(index)}
              title="Present from this slide"
              aria-label={`Present from slide ${index + 1}`}
            >
              <PresentIcon size={12} />
            </button>
          </div>
        </header>
        <div className={`deck-content ${beside ? 'beside' : 'under'}`}>
          <div className={`deck-visual ${view}`}>
            <SlideVisual jobId={jobId} deck={deck} page={page} view={view} links onImageError={onImageError} />
          </div>
          <Points page={page} writing={writing} />
        </div>
      </div>
      {said && page.said && (
        <>
          <details className="deck-said">
            <summary>What was said</summary>
            <p>{page.said}</p>
          </details>
          <div className="deck-said-print">
            <h3>What was said ({rangeLabel(page.t0, page.t1)})</h3>
            <p>{page.said}</p>
          </div>
        </>
      )}
    </article>
  );
}

/** One slide at a time over the whole screen. Keys: arrows, PageUp/PageDown, Space, Home/End; Esc leaves. */
function Presenter({
  jobId,
  deck,
  start,
  view,
  writing,
  onClose,
  onImageError,
}: {
  jobId: string;
  deck: Deck;
  start: number;
  view: SlidesView;
  writing: boolean;
  onClose(index: number): void;
  onImageError(): void;
}) {
  const count = deck.pages.length;
  const [shown, setIndex] = useState(() => clampIndex(start, count));
  // The deck can be swapped while presenting (key points landing, a rebuilt deck with fewer pages): stay on a page
  // that exists, or the overlay would vanish and leave the page locked under it.
  const index = clampIndex(shown, count);
  const at = useRef(index);
  at.current = index;
  const close = useRef(onClose);
  close.current = onClose;
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    box.current?.focus();
    document.body.classList.add('presenting');
    return () => document.body.classList.remove('presenting');
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      // Space and Enter on a focused button press it; only the slide itself takes them as "next".
      if ((e.key === ' ' || e.key === 'Enter') && (e.target as Element | null)?.closest?.('button, a')) return;
      const to = presentKey(e.key, e.shiftKey, at.current, count);
      if (to === null) return;
      e.preventDefault();
      if (to === 'exit') close.current(at.current);
      else {
        at.current = to; // two keys before the next render still move two slides
        setIndex(to);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [count]);

  // Esc in full screen is taken by the browser to leave full screen: leave present mode with it.
  useEffect(() => {
    let wasFull = !!document.fullscreenElement;
    const onChange = () => {
      if (document.fullscreenElement) wasFull = true;
      else if (wasFull) close.current(at.current);
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const page = deck.pages[index];
  if (!page) return null;
  const hasPoints = !!page.points?.length;
  const beside = view === 'image' && hasPoints && pointsBeside(page.image_size);
  return (
    <div className="present" role="dialog" aria-modal="true" aria-label={`Presenting ${deck.title}`} tabIndex={-1} ref={box}>
      <div className="present-stage">
        <h2 className="present-title">{page.title}</h2>
        <div className={`present-body ${beside ? 'beside' : 'under'} ${view}`}>
          <div className={`present-visual ${view}`}>
            <SlideVisual jobId={jobId} deck={deck} page={page} view={view} links={false} onImageError={onImageError} />
          </div>
          <Points page={page} writing={writing} />
        </div>
      </div>
      <footer className="present-bar">
        <span className="present-count" aria-live="polite" aria-atomic="true">
          <span className="sr-only">Slide </span>
          {index + 1} / {count}
          <span className="sr-only">: {page.title}</span>
        </span>
        <span className="present-name" aria-hidden>
          {page.title}
        </span>
        <span className="present-time">{rangeLabel(page.t0, page.t1)}</span>
        <div className="present-nav">
          <button
            className="present-btn"
            onClick={() => setIndex(clampIndex(index - 1, count))}
            disabled={index === 0}
            aria-label="Previous slide"
            title="Previous slide (←)"
          >
            ‹
          </button>
          <button
            className="present-btn"
            onClick={() => setIndex(clampIndex(index + 1, count))}
            disabled={index === count - 1}
            aria-label="Next slide"
            title="Next slide (→ or Space)"
          >
            ›
          </button>
          <button className="present-btn exit" onClick={() => close.current(at.current)} title="Leave (Esc)">
            Exit
          </button>
        </div>
      </footer>
    </div>
  );
}

type PointsState = { state: 'idle' | 'writing' } | { state: 'failed'; error: string };
type PptxState = { state: 'idle' | 'making' } | { state: 'failed'; error: string };

export function SlidesPage({ jobId }: { jobId: string }) {
  const [deck, setDeck] = useState<Deck | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [points, setPoints] = useState<PointsState>({ state: 'idle' });
  const [prefs, setPrefs] = useState<SlidesPrefs>(() => loadPrefs(localStore(), jobId));
  const [presenting, setPresenting] = useState<number | null>(null);
  const [pptx, setPptx] = useState<PptxState>({ state: 'idle' });
  const [attempt, setAttempt] = useState(0);
  const presentBtn = useRef<HTMLButtonElement>(null);
  const fromHeader = useRef(true);
  const alive = useRef(true);
  const triedFor = useRef<string | null>(null);
  const deckRef = useRef<Deck | null>(null);
  deckRef.current = deck;
  const lastRefresh = useRef(0);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    const ac = new AbortController();
    getSlides(jobId, ac.signal)
      .then(setDeck)
      .catch((e) => {
        if (!ac.signal.aborted) setError(sentence(String((e as Error).message || e)));
      });
    return () => ac.abort();
  }, [jobId, attempt]);

  const tryAgain = () => {
    setError(null);
    setAttempt((n) => n + 1);
  };

  /** A picture would not load: the deck was probably rebuilt since it was read (replay.json rewritten when the word
   *  times landed). Read it again, at most once every few seconds however many pictures fail together; a new
   *  version brings new picture addresses, the same version changes nothing (so this cannot loop). */
  const onImageError = useCallback(() => {
    const now = Date.now();
    if (!deckRef.current || now - lastRefresh.current < REFRESH_GAP_MS) return;
    lastRefresh.current = now;
    getSlides(jobId)
      .then((fresh) => {
        if (alive.current && fresh.version !== deckRef.current?.version) setDeck(fresh);
      })
      .catch(() => {
        /* the cards already say the picture could not be loaded */
      });
  }, [jobId]);

  useEffect(() => {
    if (deck) document.title = `Slides · ${capital(deck.title)} · Replay Board`;
  }, [deck]);

  const update = (next: Partial<SlidesPrefs>) => {
    const v = { ...prefs, ...next };
    setPrefs(v);
    savePrefs(localStore(), jobId, v);
  };

  /** Ask Muse Spark for the titles and key points. The deck stays usable whatever happens. */
  const writePoints = useCallback(
    (version: string) => {
      const key = `${jobId}\n${version}`;
      let call = pointsCalls.get(key);
      if (!call) {
        call = writeSlidePoints(jobId);
        pointsCalls.set(key, call);
        const forget = () => pointsCalls.delete(key);
        call.then(forget, forget);
      }
      setPoints({ state: 'writing' });
      call.then(
        (d) => {
          if (!alive.current) return;
          setDeck(d);
          setPoints(
            d.points_ready || !d.pages.length
              ? { state: 'idle' }
              : { state: 'failed', error: 'Muse Spark did not write the key points this time.' },
          );
        },
        (e) => {
          if (alive.current) setPoints({ state: 'failed', error: sentence(String((e as Error).message || e)) });
        },
      );
    },
    [jobId],
  );

  // On first open, write the key points once, automatically.
  useEffect(() => {
    if (!deck || deck.points_ready || !deck.pages.length || triedFor.current === deck.version) return;
    triedFor.current = deck.version;
    writePoints(deck.version);
  }, [deck, writePoints]);

  const present = (index: number, header = false) => {
    fromHeader.current = header;
    setPresenting(index);
    const root = document.documentElement;
    if (root.requestFullscreen && !document.fullscreenElement) root.requestFullscreen().catch(() => {});
  };

  // Focus goes back where presenting started: the Present button (always in view, the bar is sticky), or the
  // "present from here" button of the slide that was showing, which is scrolled into view.
  const stopPresenting = useCallback((index: number) => {
    setPresenting(null);
    if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {});
    requestAnimationFrame(() => {
      const sheet = document.getElementById(`slide-${index + 1}`);
      sheet?.scrollIntoView({ block: 'start' });
      const back = fromHeader.current ? presentBtn.current : sheet?.querySelector<HTMLElement>('.deck-from-here');
      (back ?? presentBtn.current)?.focus({ preventScroll: true });
    });
  }, []);

  /** The PowerPoint, fetched so that a failure is one plain line here. A modified click (new tab, save link as)
   *  is left to the browser. */
  const downloadPptx = async (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    const url = e.currentTarget.href;
    if (!deck || pptx.state === 'making') return;
    setPptx({ state: 'making' });
    try {
      const { blob, name } = await getSlidesPptx(url, deck.title);
      const href = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = href;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(href), 60_000);
      if (alive.current) setPptx({ state: 'idle' });
    } catch (err) {
      if (alive.current) setPptx({ state: 'failed', error: sentence(String((err as Error).message || err)) });
    }
  };

  const view: SlidesView = deck?.type === 'board' ? prefs.view : 'image';
  const writing = points.state === 'writing';
  const title = deck ? capital(deck.title) : '';
  const pages = deck?.pages ?? [];
  const anySaid = pages.some((p) => p.said);
  // The overlay is up. A deck re-read with no pages takes it down, and the page must not stay inert under nothing.
  const showing = !!deck && presenting !== null && pages.length > 0;
  const pptxNote = deck && !deck.points_ready ? ' (key points are not written yet: titles only)' : '';

  return (
    <div className="slides-page">
      {/* Landscape pages while this page is open (the notes print portrait). */}
      <style>{'@page { size: landscape; margin: 10mm; }'}</style>
      {/* While presenting, the page under the overlay can be neither focused nor read: the dialog is modal. */}
      <header className="notes-bar slides-bar" inert={showing}>
        <a className="ghost-btn" href={`#/job/${encodeURIComponent(jobId)}`}>
          ← Back to the lecture
        </a>
        <div className="notes-actions slides-actions">
          <button
            ref={presentBtn}
            className="ghost-btn present-start"
            onClick={() => present(0, true)}
            disabled={!pages.length}
            title="Show one slide at a time, full screen"
          >
            <PresentIcon />
            Present
          </button>
          <button className="ghost-btn" onClick={() => window.print()} disabled={!pages.length} title="One slide per landscape page">
            Save as PDF
          </button>
          {deck && pages.length > 0 ? (
            <a
              className="ghost-btn"
              href={slidesPptxUrl(jobId, deck.version, deck.points_ready)}
              download
              onClick={downloadPptx}
              aria-busy={pptx.state === 'making'}
              aria-disabled={pptx.state === 'making'}
              title={`Download the slides as a PowerPoint file${pptxNote}`}
            >
              {pptx.state === 'making' ? 'Making the PowerPoint…' : 'Download PowerPoint'}
            </a>
          ) : (
            <button className="ghost-btn" disabled>
              Download PowerPoint
            </button>
          )}
        </div>
        {/* In the sticky bar, next to the button that failed, wherever the page is scrolled. */}
        {pptx.state === 'failed' && (
          <p className="slides-bar-error" role="alert">
            {pptx.error}
          </p>
        )}
      </header>
      <main className="slides-main" inert={showing}>
        {error && (
          <p className="error-line slides-load-error" role="alert">
            <span>{error}</span>
            <button className="ghost-btn" onClick={tryAgain}>
              Try again
            </button>
          </p>
        )}
        {!deck && !error && <p className="muted">Making the slides…</p>}
        {deck && (
          <>
            <div className="slides-intro">
              <h1 className="notes-title">{title}</h1>
              <p className="notes-sub screen-only">
                {pages.length} slide{pages.length === 1 ? '' : 's'} made by Replay Board from what the teacher{' '}
                {deck.type === 'slides' ? 'showed' : 'wrote'}. Titles and key points by Muse Spark. Play opens the
                lecture at that slide.
              </p>
              <p className="notes-sub print-only">
                Made with Replay Board from the recorded lecture · {pages.length} slide{pages.length === 1 ? '' : 's'}
                {pages.length > 0 && ` · ${rangeLabel(pages[0].t0, pages[pages.length - 1].t1)}`}
              </p>
              <div className="slides-tools" role="group" aria-label="Slide options">
                {deck.type === 'board' && (
                  <div className="seg" role="group" aria-label="Show the board as">
                    <button
                      className={view === 'image' ? 'on' : ''}
                      aria-pressed={view === 'image'}
                      onClick={() => update({ view: 'image' })}
                    >
                      Board
                    </button>
                    <button
                      className={view === 'typeset' ? 'on' : ''}
                      aria-pressed={view === 'typeset'}
                      onClick={() => update({ view: 'typeset' })}
                    >
                      Typeset
                    </button>
                  </div>
                )}
                {anySaid && (
                  <label className="said-toggle">
                    <input type="checkbox" checked={prefs.said} onChange={(e) => update({ said: e.target.checked })} />
                    Include what was said
                  </label>
                )}
              </div>
              {points.state === 'failed' && (
                <p className="slides-points-error" role="alert">
                  <span>{points.error}</span>
                  <button className="ghost-btn" onClick={() => deck && writePoints(deck.version)}>
                    Retry
                  </button>
                </p>
              )}
              <p className="sr-only" role="status">
                {writing ? 'Muse Spark is writing the key points.' : ''}
              </p>
            </div>
            {pages.length ? (
              pages.map((p, i) => (
                <SlideCard
                  key={p.id}
                  jobId={jobId}
                  deck={deck}
                  page={p}
                  index={i}
                  view={view}
                  said={prefs.said}
                  writing={writing}
                  onPresent={present}
                  onImageError={onImageError}
                />
              ))
            ) : (
              <p className="muted">This lecture has no slides: nothing was written or shown.</p>
            )}
          </>
        )}
      </main>
      {showing && deck && presenting !== null && (
        <Presenter
          jobId={jobId}
          deck={deck}
          start={presenting}
          view={view}
          writing={writing}
          onClose={stopPresenting}
          onImageError={onImageError}
        />
      )}
    </div>
  );
}
