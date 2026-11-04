import { Fragment, memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { SparkIcon } from '../ask/icons';
import { spanFromTurns, spanFromWords, type SpanPick } from '../ask/selection';
import type { Chapter, Turn, Word } from '../types';
import { formatTime } from './timing';

/** True while the student has text selected: the click that ends a selection is not a click on a word. */
const selecting = () => {
  const sel = window.getSelection();
  return !!sel && !sel.isCollapsed;
};

type TurnState = 'past' | 'current' | 'future';
const EMPTY: number[] = [];

/** One paragraph of the transcript. It re-renders only when its turn becomes past/current/future: the word
 *  highlight ("now", "said") is set on the spans directly, so a long turn is not rebuilt for every word. */
const TurnBlock = memo(function TurnBlock(props: {
  index: number;
  turn: Turn;
  words: Word[];
  wordIdx: number[];
  state: TurnState;
  onSeek(t: number): void;
}) {
  const { index, turn, words, wordIdx, state, onSeek } = props;
  return (
    <div className={`turn ${state}`} data-turn={index}>
      <button className="turn-time" onClick={() => onSeek(turn.start)} title="Play from here">
        {formatTime(turn.start)}
      </button>
      {wordIdx.length ? (
        <p
          className="turn-text"
          onClick={(e) => {
            if (selecting()) return;
            const i = Number((e.target as HTMLElement).dataset?.i);
            if (Number.isInteger(i) && words[i]) onSeek(words[i].s);
          }}
        >
          {wordIdx.map((i, k) => (
            <Fragment key={i}>
              {k > 0 ? ' ' : null}
              <span data-i={i} className="w">
                {words[i].w}
              </span>
            </Fragment>
          ))}
        </p>
      ) : (
        <p className="turn-text plain" onClick={() => !selecting() && onSeek(turn.start)}>
          {turn.text}
        </p>
      )}
    </div>
  );
});

export function Transcript(props: {
  turns: Turn[];
  words: Word[];
  groups: number[][];
  wordTurn: Int32Array;
  active: number;
  upTo: number;
  currentTurn: number;
  timing?: string;
  /** True while the backend still refines the word times (the player swaps in the new ones when they land). */
  refining?: boolean;
  /** The lecture is still processed (a streaming replay): no transcript yet is not "no speech". */
  pending?: boolean;
  onSeek(t: number): void;
  /** Ask Muse about the selected words: a small "Ask about this" button appears by a selection. */
  onAskSpan?(span: SpanPick): void;
}) {
  const { turns, words, groups, wordTurn, active, upTo, currentTurn, timing, refining, pending, onSeek, onAskSpan } = props;
  const boxRef = useRef<HTMLDivElement>(null);
  const [detached, setDetached] = useState(false);
  const idleTimer = useRef<number | undefined>(undefined);

  // Selected words -> the stretch of speech they cover, with a button to ask about it beside the selection.
  const [askSel, setAskSel] = useState<{ left: number; top: number; pick: SpanPick } | null>(null);
  const pointerDown = useRef(false);
  const readSelection = useCallback(() => {
    const box = boxRef.current;
    const sel = window.getSelection();
    if (!onAskSpan || !box || !sel || sel.isCollapsed || !sel.rangeCount) return setAskSel(null);
    const range = sel.getRangeAt(0);
    if (!box.contains(range.commonAncestorContainer)) return setAskSel(null);
    const turnOf = (n: Node) =>
      (n instanceof Element ? n : n.parentElement)?.closest<HTMLElement>('[data-turn]') ?? null;
    const a = turnOf(range.startContainer);
    const b = turnOf(range.endContainer);
    const from = a ? Number(a.dataset.turn) : 0;
    const to = b ? Number(b.dataset.turn) : turns.length - 1;
    // a word counts when some of it is selected (not when the selection only touches its edge)
    const covers = (el: Element) => {
      const r = document.createRange();
      r.selectNodeContents(el);
      // END_TO_START: the selection's start against the word's end; START_TO_END: its end against the word's start
      return range.compareBoundaryPoints(Range.END_TO_START, r) < 0 && range.compareBoundaryPoints(Range.START_TO_END, r) > 0;
    };
    const idx: number[] = [];
    const plain: number[] = [];
    for (let ti = from; ti <= to; ti++) {
      const el = box.querySelector(`[data-turn="${ti}"] .turn-text`);
      if (!el) continue;
      const spans = el.querySelectorAll<HTMLElement>('[data-i]');
      if (spans.length) spans.forEach((sp) => covers(sp) && idx.push(Number(sp.dataset.i)));
      else if (covers(el)) plain.push(ti);
    }
    const text = sel.toString();
    const pick =
      spanFromWords(words, idx, text) ?? (plain.length ? spanFromTurns(turns, plain[0], plain[plain.length - 1], text) : null);
    if (!pick) return setAskSel(null);
    const rects = range.getClientRects();
    const last = rects.length ? rects[rects.length - 1] : range.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();
    const left = Math.min(Math.max(4, last.right - boxRect.left - 70), Math.max(4, box.clientWidth - 150));
    setAskSel({ left, top: last.bottom - boxRect.top + box.scrollTop + 6, pick });
  }, [onAskSpan, words, turns]);

  useEffect(() => {
    if (!onAskSpan) return;
    let timer: number | undefined;
    const onChange = () => {
      const sel = window.getSelection();
      const box = boxRef.current;
      if (!sel || sel.isCollapsed || !box || !sel.anchorNode || !box.contains(sel.anchorNode)) {
        setAskSel(null);
        return;
      }
      // a selection made by touch (handles dragged) or the keyboard has no mouseup: read it once it settles
      window.clearTimeout(timer);
      timer = window.setTimeout(() => !pointerDown.current && readSelection(), 300);
    };
    // the selection is final once the button is up (even when the drag ended outside the transcript). A long press
    // on a phone ends in pointercancel, not pointerup (the browser takes over to select): without it the flag would
    // stay set and the button would never show on touch.
    const onUp = () => {
      if (!pointerDown.current) return;
      pointerDown.current = false;
      window.setTimeout(readSelection, 0);
    };
    document.addEventListener('selectionchange', onChange);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener('selectionchange', onChange);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, [onAskSpan, readSelection]);

  // Word spans by global index, collected after the words render (the highlight below writes to them).
  const spans = useRef<HTMLElement[]>([]);
  const marked = useRef({ active: -1, turn: -1, upTo: -1 });
  useLayoutEffect(() => {
    const arr: HTMLElement[] = [];
    boxRef.current?.querySelectorAll<HTMLElement>('[data-i]').forEach((el) => (arr[Number(el.dataset.i)] = el));
    spans.current = arr;
    marked.current = { active: -1, turn: -1, upTo: -1 };
  }, [words, groups, turns]);

  // "now" on the word being spoken; "said" on the words of the current turn already spoken.
  useLayoutEffect(() => {
    const el = spans.current;
    const m = marked.current;
    if (m.active !== active) {
      el[m.active]?.classList.remove('now');
      el[active]?.classList.add('now');
      m.active = active;
    }
    if (m.turn !== currentTurn) {
      for (const i of groups[m.turn] ?? []) el[i]?.classList.remove('said');
      m.turn = currentTurn;
      m.upTo = -2;
    }
    const up = upTo >= 0 && wordTurn[upTo] === currentTurn ? upTo : -1;
    if (m.upTo !== up) {
      for (const i of groups[currentTurn] ?? []) el[i]?.classList.toggle('said', i <= up);
      m.upTo = up;
    }
  }, [active, upTo, currentTurn, groups, wordTurn, words, turns]);

  const scrollToCurrent = useCallback(
    (force: boolean) => {
      const box = boxRef.current;
      if (!box) return;
      const i = active >= 0 ? active : upTo;
      let el: HTMLElement | null = i >= 0 ? box.querySelector(`[data-i="${i}"]`) : null;
      if (!el && currentTurn >= 0) el = box.querySelector(`[data-turn="${currentTurn}"]`);
      if (!el) return;
      const boxRect = box.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      const rel = r.top - boxRect.top;
      if (!force && rel > box.clientHeight * 0.15 && r.bottom - boxRect.top < box.clientHeight * 0.75) return;
      box.scrollTo({ top: box.scrollTop + rel - box.clientHeight * 0.33, behavior: 'smooth' });
    },
    [active, upTo, currentTurn],
  );

  useEffect(() => {
    if (!detached) scrollToCurrent(false);
  }, [scrollToCurrent, detached]);

  const userScrolled = () => {
    setDetached(true);
    window.clearTimeout(idleTimer.current);
    idleTimer.current = window.setTimeout(() => setDetached(false), 4000);
  };
  useEffect(() => () => window.clearTimeout(idleTimer.current), []);

  const reattach = () => {
    window.clearTimeout(idleTimer.current);
    setDetached(false);
    scrollToCurrent(true);
  };

  return (
    <section className="panel transcript-panel">
      <header className="panel-head">
        <h2>Transcript</h2>
        {pending ? null : refining && words.length > 0 ? (
          <span className="panel-note refining" role="status">
            refining word times…
          </span>
        ) : (
          timing &&
          timing !== 'none' &&
          words.length > 0 && (
            <span className="panel-note">{timing === 'mms' ? 'word timing: MMS aligner' : `timing: ${timing}`}</span>
          )
        )}
      </header>
      <div
        ref={boxRef}
        className="transcript"
        onWheel={userScrolled}
        onTouchMove={userScrolled}
        onPointerDown={(e) => {
          pointerDown.current = true;
          if (e.target === e.currentTarget) userScrolled(); // grabbing the scrollbar
        }}
        onKeyDown={(e) => {
          if (['PageUp', 'PageDown', 'Home', 'End', 'ArrowUp', 'ArrowDown'].includes(e.key)) userScrolled();
        }}
        tabIndex={-1}
      >
        {turns.length === 0 && (
          <p className="empty">
            {pending ? 'The transcript appears here once the voice is transcribed.' : 'No speech was transcribed for this lecture.'}
          </p>
        )}
        {askSel && onAskSpan && (
          <button
            type="button"
            className="ask-sel-btn"
            style={{ left: askSel.left, top: askSel.top }}
            onMouseDown={(e) => e.preventDefault()} // keep the selection
            onClick={() => {
              onAskSpan(askSel.pick);
              window.getSelection()?.removeAllRanges();
              setAskSel(null);
            }}
            title={`Ask Muse about what was said ${formatTime(askSel.pick.t0)}–${formatTime(askSel.pick.t1)}`}
          >
            <SparkIcon size={12} />
            Ask about this
          </button>
        )}
        {turns.map((turn, ti) => {
          const state: TurnState = ti < currentTurn ? 'past' : ti === currentTurn ? 'current' : 'future';
          return (
            <TurnBlock
              key={ti}
              index={ti}
              turn={turn}
              words={words}
              wordIdx={groups[ti] ?? EMPTY}
              state={state}
              onSeek={onSeek}
            />
          );
        })}
      </div>
      {detached && (
        <button className="follow-btn" onClick={reattach}>
          Back to now
        </button>
      )}
    </section>
  );
}

export function Chapters(props: {
  chapters: Chapter[];
  current: number;
  onSeek(t: number): void;
  children?: React.ReactNode; // footer (facts about the replay)
}) {
  const { chapters, current, onSeek, children } = props;
  const listRef = useRef<HTMLOListElement>(null);

  // Keep the current chapter visible inside the (scrollable) list, without moving the page.
  useEffect(() => {
    const list = listRef.current;
    const el = list?.querySelector<HTMLElement>('button.on');
    if (!list || !el) return;
    const top = el.offsetTop - list.offsetTop;
    if (top < list.scrollTop || top + el.offsetHeight > list.scrollTop + list.clientHeight)
      list.scrollTo({ top: top - list.clientHeight / 3, behavior: 'smooth' });
  }, [current]);

  return (
    <section className="panel chapters-panel">
      {chapters.length > 0 && (
        <>
          <header className="panel-head">
            <h2>Chapters</h2>
          </header>
          <ol className="chapters" ref={listRef}>
            {chapters.map((c, i) => (
              <li key={i}>
                <button className={i === current ? 'on' : ''} onClick={() => onSeek(c.t)}>
                  <span className="ch-time">{formatTime(c.t)}</span>
                  <span className="ch-title">{c.title}</span>
                </button>
              </li>
            ))}
          </ol>
        </>
      )}
      {children}
    </section>
  );
}
