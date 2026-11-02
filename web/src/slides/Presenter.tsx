// Present mode: one slide at a time over the whole screen, the 16:9 frame fitted to it. Keys: arrows,
// PageUp/PageDown, Space/Shift+Space, Home/End; Esc leaves (and so does leaving full screen).
import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react';
import type { Deck } from '../types';
import { clampIndex, presentKey, rangeLabel } from './deck';
import { SlideFrame } from './SlideFrame';

export function Presenter({
  jobId,
  deck,
  start,
  onClose,
  onImageError,
}: {
  jobId: string;
  deck: Deck;
  start: number;
  onClose(index: number): void;
  onImageError?(): void;
}) {
  const count = deck.slides.length;
  const [shown, setIndex] = useState(() => clampIndex(start, count));
  // The deck can be swapped while presenting (a rewritten deck with fewer slides): stay on a slide that exists, or
  // the overlay would vanish and leave the page locked under it.
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
      if ((e.key === ' ' || e.key === 'Enter') && (e.target as Element | null)?.closest?.('button, a, summary')) return;
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

  // Focus stays in the dialog: Tab from its last control comes back to its first (the page under it is inert too).
  const trap = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Tab' || !box.current) return;
    const stops = Array.from(box.current.querySelectorAll<HTMLElement>('button:not(:disabled), a[href]'));
    if (!stops.length) return;
    const first = stops[0];
    const last = stops[stops.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || active === box.current)) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  };

  const slide = deck.slides[index];
  if (!slide) return null;
  const range = rangeLabel(slide.t0, slide.t1);
  return (
    <div
      className="present"
      role="dialog"
      aria-modal="true"
      aria-label={`Presenting ${deck.title}`}
      tabIndex={-1}
      ref={box}
      onKeyDown={trap}
    >
      <div className="present-stage">
        <SlideFrame key={slide.id} jobId={jobId} deck={deck} slide={slide} index={index} onImageError={onImageError} />
      </div>
      <footer className="present-bar">
        <span className="present-count" aria-live="polite" aria-atomic="true">
          <span className="sr-only">Slide </span>
          {index + 1} / {count}
          <span className="sr-only">: {slide.title}</span>
        </span>
        <span className="present-name" aria-hidden="true">
          {slide.title}
        </span>
        {range && <span className="present-time">{range}</span>}
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
