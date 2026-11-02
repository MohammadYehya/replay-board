// The lecture as a deck: Muse Spark writes the slides a good teacher would have prepared, covering the whole lecture
// in order (app/export.py), and this page previews them as 16:9 slides, presents them full screen, prints them one per
// landscape page and downloads the same deck as a PowerPoint (app/pptx_deck.py). Opening the page starts the writing
// once; while Muse Spark writes, the page shows how far it is and the slides finished so far.
import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react';
import { getSlides, getSlidesPptx, slidesPptxUrl, writeSlides } from '../api';
import { SparkIcon } from '../ask/icons';
import { joinCall, runFlow, sleep } from '../slides/flow';
import { durationLabel, playerAt, progressShare, rangeLabel, sentence, stepLine } from '../slides/deck';
import { PlayIcon, PresentIcon } from '../slides/icons';
import { Presenter } from '../slides/Presenter';
import { SlideFrame } from '../slides/SlideFrame';
import type { Deck, DeckSlide, SlidesStatus } from '../types';

/** One request to write a lecture's slides at a time, even when the page mounts twice (React's StrictMode in dev)
 *  or Try again is pressed while one is on its way. The server joins a second tab's request to the same run. */
const writeCalls = new Map<string, Promise<SlidesStatus>>();

/** Pictures failing together cause one fresh read of the deck, not one each. */
const REFRESH_GAP_MS = 4000;

const aborted = (e: unknown) => (e as Error | null)?.name === 'AbortError';
const line = (e: unknown) => sentence(String((e as Error | null)?.message || e)) || 'Something went wrong.';

type PptxState = { state: 'idle' | 'making' } | { state: 'failed'; error: string };

function SpeakerNotes({ slide }: { slide: DeckSlide }) {
  if (!slide.notes && !slide.said) return null;
  const range = rangeLabel(slide.t0, slide.t1);
  return (
    <details className="sl-notes">
      <summary>Speaker notes</summary>
      {slide.notes && <p className="sl-notes-text">{slide.notes}</p>}
      {slide.said && (
        <>
          <h3>What was said{range ? ` (${range})` : ''}</h3>
          <p className="sl-said">{slide.said}</p>
        </>
      )}
    </details>
  );
}

function SlideCard({
  jobId,
  deck,
  slide,
  index,
  canPresent,
  onPresent,
  onImageError,
}: {
  jobId: string;
  deck: Deck;
  slide: DeckSlide;
  index: number;
  canPresent: boolean;
  onPresent(index: number): void;
  onImageError(): void;
}) {
  const headingId = `slide-${index + 1}-title`;
  const range = rangeLabel(slide.t0, slide.t1);
  return (
    <article className="sl-card" id={`slide-${index + 1}`} aria-labelledby={headingId}>
      <SlideFrame jobId={jobId} deck={deck} slide={slide} index={index} headingId={headingId} onImageError={onImageError} />
      <div className="sl-under">
        <span className="sl-under-num">Slide {index + 1}</span>
        {range && <span className="sl-under-time">{range}</span>}
        <span className="sl-under-gap" />
        {slide.t !== null && (
          <a className="sl-play" href={playerAt(jobId, slide.t)} title="Watch this part of the lecture">
            <PlayIcon />
            Play
          </a>
        )}
        <button
          className="sl-from-here"
          onClick={() => onPresent(index)}
          disabled={!canPresent}
          title="Present from this slide"
          aria-label={`Present from slide ${index + 1}`}
        >
          <PresentIcon size={12} />
        </button>
      </div>
      <SpeakerNotes slide={slide} />
    </article>
  );
}

/** How far Muse Spark is: the step, a calm bar, and that the page may be left meanwhile. */
function Progress({ status }: { status: SlidesStatus | null }) {
  const share = status ? progressShare(status) : null;
  const text = status ? stepLine(status) : 'Opening the slides…';
  return (
    <div className="slides-progress">
      <p className="slides-progress-step" role="status">
        <SparkIcon size={14} />
        <span>{text}</span>
      </p>
      <div
        className={`slides-progress-bar${share === null ? ' indeterminate' : ''}`}
        role="progressbar"
        aria-label="Writing the slides"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={share === null ? undefined : Math.round(share * 100)}
      >
        <span style={share === null ? undefined : { width: `${Math.max(4, share * 100)}%` }} />
      </div>
      {status && (
        <p className="slides-progress-note">
          Muse Spark is reading the whole lecture to write its slides. This takes a minute or two; you can leave this
          page meanwhile.
        </p>
      )}
    </div>
  );
}

export function SlidesPage({ jobId }: { jobId: string }) {
  const [status, setStatus] = useState<SlidesStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** Each start over of the flow; `write` asks Muse Spark again first (Try again after a failure). */
  const [run, setRun] = useState({ n: 0, write: false });
  const [presenting, setPresenting] = useState<number | null>(null);
  const [pptx, setPptx] = useState<PptxState>({ state: 'idle' });
  const presentBtn = useRef<HTMLButtonElement>(null);
  const fromHeader = useRef(true);
  const alive = useRef(true);
  const statusRef = useRef<SlidesStatus | null>(null);
  statusRef.current = status;
  const lastRefresh = useRef(0);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  // Read, write once if nothing is written, poll while writing; one run per (re)start.
  useEffect(() => {
    const ac = new AbortController();
    let live = true;
    runFlow(
      {
        get: () => getSlides(jobId, ac.signal),
        // Not tied to this run: the server starts writing either way, and a remount joins the same request.
        write: () => joinCall(writeCalls, jobId, () => writeSlides(jobId)),
        wait: (ms) => sleep(ms, ac.signal),
        show: (s) => {
          setStatus(s);
          setError(null);
        },
        alive: () => live,
      },
      { write: run.write },
    ).catch((e) => {
      if (live && !aborted(e)) setError(line(e));
    });
    return () => {
      live = false;
      ac.abort();
    };
  }, [jobId, run]);

  const startOver = (write: boolean) => {
    setError(null);
    setRun((r) => ({ n: r.n + 1, write }));
  };

  /** A figure would not load: the deck was probably rewritten since it was read. Read it again, at most once every
   *  few seconds however many figures fail together; a new version brings new addresses, the same version changes
   *  nothing (so this cannot loop). */
  const onImageError = useCallback(() => {
    const now = Date.now();
    const cur = statusRef.current;
    if (cur?.state !== 'ready' || now - lastRefresh.current < REFRESH_GAP_MS) return;
    lastRefresh.current = now;
    getSlides(jobId)
      .then((fresh) => {
        if (alive.current && fresh.state === 'ready' && fresh.deck?.version !== statusRef.current?.deck?.version)
          setStatus(fresh);
      })
      .catch(() => {
        /* the frames already say the figure could not be loaded */
      });
  }, [jobId]);

  const deck = status?.deck ?? null;
  const ready = status?.state === 'ready' && !!deck;
  const slides = deck?.slides ?? [];

  useEffect(() => {
    if (deck) document.title = `Slides · ${deck.title} · Replay Board`;
  }, [deck]);

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
      const card = document.getElementById(`slide-${index + 1}`);
      card?.scrollIntoView({ block: 'start' });
      const back = fromHeader.current ? presentBtn.current : card?.querySelector<HTMLElement>('.sl-from-here');
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
      if (alive.current) setPptx({ state: 'failed', error: line(err) });
    }
  };

  // The overlay is up. A deck re-read with no slides takes it down, and the page must not stay inert under nothing.
  const showing = ready && presenting !== null && slides.length > 0;
  const canPresent = ready && slides.length > 0;
  const writing = !error && (!status || status.state === 'none' || status.state === 'writing');
  const failed = !error && status?.state === 'failed';
  const meta = deck
    ? [
        ready ? `${slides.length} slide${slides.length === 1 ? '' : 's'}` : '',
        deck.lecture !== deck.title ? deck.lecture : '',
        durationLabel(deck.duration) ? `${durationLabel(deck.duration)} of lecture` : '',
      ].filter(Boolean)
    : [];

  return (
    <div className="slides-page">
      {/* While this page is open, a printed page is one slide: 960 x 540 pt, the PowerPoint's slide size, like a
          PowerPoint saved as PDF (the notes print portrait). */}
      <style>{'@page { size: 960pt 540pt; margin: 0; }'}</style>
      {/* While presenting, the page under the overlay can be neither focused nor read: the dialog is modal. */}
      <header className="notes-bar slides-bar" inert={showing}>
        <a className="ghost-btn" href={`#/job/${encodeURIComponent(jobId)}`}>
          ← Back to the lecture
        </a>
        <p className="slides-bar-name" title={deck?.lecture}>
          {deck?.lecture ?? ''}
        </p>
        <div className="slides-actions">
          <button
            ref={presentBtn}
            className="ghost-btn"
            onClick={() => present(0, true)}
            disabled={!canPresent}
            title="Show one slide at a time, full screen"
          >
            <PresentIcon />
            Present
          </button>
          <button
            className="ghost-btn"
            onClick={() => window.print()}
            disabled={!canPresent}
            title="One slide per landscape page"
          >
            Save as PDF
          </button>
          {ready && deck ? (
            <a
              className="primary-btn slides-download"
              href={slidesPptxUrl(jobId, deck.version)}
              download
              onClick={downloadPptx}
              aria-busy={pptx.state === 'making'}
              aria-disabled={pptx.state === 'making'}
              title="Download the slides as a PowerPoint file"
            >
              {pptx.state === 'making' ? 'Making the PowerPoint…' : 'Download PowerPoint'}
            </a>
          ) : (
            <button className="primary-btn slides-download" disabled>
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
        <div className="slides-intro">
          <h1 className="notes-title">{deck?.title ?? 'Slides'}</h1>
          {deck?.subtitle && <p className="slides-subtitle">{deck.subtitle}</p>}
          {meta.length > 0 && <p className="slides-meta">{meta.join(' · ')}</p>}
          {ready && (
            <p className="slides-how">
              Written by Muse Spark from the whole lecture. Play opens the lecture at that moment; the PowerPoint has the
              same slides, with the speaker notes.
            </p>
          )}
          {deck && deck.degraded.length > 0 && (
            <div className="slides-degraded">
              {deck.degraded.map((d, i) => (
                <p key={i}>{d}</p>
              ))}
            </div>
          )}
        </div>
        {error && (
          <p className="error-line slides-load-error" role="alert">
            <span>{error}</span>
            <button className="ghost-btn" onClick={() => startOver(false)}>
              Try again
            </button>
          </p>
        )}
        {failed && status && (
          <p className="error-line slides-load-error" role="alert">
            <span>{status.error}</span>
            <button className="ghost-btn" onClick={() => startOver(true)}>
              Try again
            </button>
          </p>
        )}
        {writing && <Progress status={status} />}
        {deck && slides.length > 0 && (
          <div className="sl-list">
            {slides.map((s, i) => (
              <SlideCard
                key={s.id}
                jobId={jobId}
                deck={deck}
                slide={s}
                index={i}
                canPresent={canPresent}
                onPresent={present}
                onImageError={onImageError}
              />
            ))}
          </div>
        )}
        {ready && !slides.length && <p className="muted">This lecture has no slides: nothing was said or shown.</p>}
      </main>
      {showing && deck && presenting !== null && (
        <Presenter jobId={jobId} deck={deck} start={presenting} onClose={stopPresenting} onImageError={onImageError} />
      )}
    </div>
  );
}
