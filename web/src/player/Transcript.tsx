import { Fragment, memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { Chapter, Turn, Word } from '../types';
import { formatTime } from './timing';

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
        <p className="turn-text plain" onClick={() => onSeek(turn.start)}>
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
  onSeek(t: number): void;
}) {
  const { turns, words, groups, wordTurn, active, upTo, currentTurn, timing, refining, onSeek } = props;
  const boxRef = useRef<HTMLDivElement>(null);
  const [detached, setDetached] = useState(false);
  const idleTimer = useRef<number | undefined>(undefined);

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
        {refining && words.length > 0 ? (
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
          if (e.target === e.currentTarget) userScrolled(); // grabbing the scrollbar
        }}
        onKeyDown={(e) => {
          if (['PageUp', 'PageDown', 'Home', 'End', 'ArrowUp', 'ArrowDown'].includes(e.key)) userScrolled();
        }}
        tabIndex={-1}
      >
        {turns.length === 0 && <p className="empty">No speech was transcribed for this lecture.</p>}
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
