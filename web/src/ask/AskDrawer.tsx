// Ask Muse: a drawer beside the lecture where a student asks Muse Spark about it. Muse Spark gets the whole lecture
// with every question; the student can attach writing, slide text, speech and moments from anywhere in it (the
// tray), and the answer cites times and board items that seek the video and light up the item on the board.
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { askMuse, clearChat, getChat } from '../api';
import { texHtml } from '../player/tex';
import { formatTime } from '../player/timing';
import type { AskRef, AskTurn } from '../types';
import {
  askRefOf, entryLatex, entryTime, filterGroups, savedEntry, type LectureIndex, type LectureItem,
} from './items';
import { parseMarkdown } from './markdown';
import { costLabel, historyFor, QUESTION_MAX, requestSelection } from './protocol';
import { renderBlocks, type RenderContext } from './render';
import { hasEntry, itemKey, removeEntry, TRAY_MAX, type TrayEntry } from './tray';
import { KindIcon, SparkIcon } from './icons';

type Status = 'streaming' | 'done' | 'error' | 'stopped';

interface UiTurn {
  key: string;
  id: string;
  q: string;
  entries: TrayEntry[];
  answer: string;
  items: AskRef[];
  times: number[];
  t: number | null;
  cost: number | null;
  cached: boolean;
  /** False when the server answered but could not save the turn. */
  saved: boolean;
  status: Status;
  error?: string;
}

interface Props {
  jobId: string;
  open: boolean;
  /** Bumped when the drawer is opened from its button or the A key: the question box takes the focus. */
  focusKey: number;
  onClose(): void;
  tray: TrayEntry[];
  setTray(next: TrayEntry[]): void;
  notice: { text: string; key: number } | null;
  index: LectureIndex;
  mode: 'board' | 'slides' | 'none';
  timeRef: RefObject<number>;
  /** The video's length: a cited time after it is not a link. */
  videoEnd: number;
  /** While the lecture is still processed: how far Muse Spark knows it ("Knows the lecture up to 3:20"). */
  knows?: string | null;
  onAddMoment(): void;
  onSeek(t: number): void;
  /** Seek to a board or slide item and light it up. */
  onShowItem(ref: AskRef): void;
}

const KIND_NAMES: Record<string, string> = {
  piece: 'Writing',
  line: 'Line of writing',
  element: 'Slide text',
  span: 'Speech',
  moment: 'Moment',
};

/** A chip for an attached item: label and time; clicking it shows the item in the lecture. Writing with a formula is
 *  typeset as Browse shows it (the board's raw text, like "A=P(1+r)^n", reads badly next to it). */
function Chip(props: { entry: TrayEntry; index: LectureIndex; onOpen(e: TrayEntry): void; onRemove?(e: TrayEntry): void }) {
  const { entry, index, onOpen, onRemove } = props;
  const kind = entry.item.kind;
  const what = KIND_NAMES[kind] ?? 'Item';
  const time = formatTime(entryTime(entry, index)); // where a click jumps (openEntry)
  const latex = entryLatex(entry, index);
  const html = latex ? texHtml(latex) : null;
  return (
    <span className={`ask-chip kind-${kind}`}>
      <button
        type="button"
        className="ask-chip-main"
        onClick={() => onOpen(entry)}
        title={`${what}: ${entry.label} · ${time}`}
      >
        <KindIcon kind={kind} />
        {html ? (
          <span className="ask-chip-label ask-chip-math" dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <span className="ask-chip-label">{entry.label}</span>
        )}
        {/* a saved moment is labelled "The board at 3:20" already */}
        {!entry.label.endsWith(time) && <span className="ask-chip-time">{time}</span>}
      </button>
      {onRemove && (
        <button
          type="button"
          className="ask-chip-x"
          aria-label={`Remove ${entry.label}`}
          title="Remove"
          onClick={() => onRemove(entry)}
        >
          <svg viewBox="0 0 12 12" width="9" height="9" aria-hidden>
            <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      )}
    </span>
  );
}

/** The answer, rendered from its Markdown subset; refs and times become buttons. */
const AnswerBody = memo(function AnswerBody(props: {
  turn: UiTurn;
  index: LectureIndex;
  videoEnd: number;
  onSeek(t: number): void;
  onShowItem(ref: AskRef): void;
}) {
  const { turn, index, videoEnd, onSeek, onShowItem } = props;
  const streaming = turn.status === 'streaming';
  const blocks = useMemo(() => parseMarkdown(turn.answer, { streaming }), [turn.answer, streaming]);
  const ctx = useMemo<RenderContext>(() => {
    // A finished answer cites only what the server checked; while it streams, the lecture itself says what exists.
    const final = turn.status === 'done';
    const byRef = new Map(turn.items.map((r) => [r.ref, r]));
    // The lecture's own entry gives the chip its formula and where to jump (when the board shows the item).
    const find = (ref: string): (AskRef & { latex?: string }) | null => {
      const it = index.byRef.get(ref);
      if (final) {
        const r = byRef.get(ref);
        return r ? { ...r, t: it?.t ?? r.t, latex: it?.latex } : null;
      }
      return it ? { ...askRefOf(it), latex: it.latex } : null;
    };
    return {
      timeOk: (s) => (final ? turn.times.some((x) => Math.abs(x - s) < 0.75) : s >= 0 && s <= videoEnd + 0.5),
      refInfo: find,
      onTime: onSeek,
      onRef: (ref) => {
        const r = find(ref);
        if (r) onShowItem(r);
      },
      clock: formatTime,
    };
  }, [turn.status, turn.items, turn.times, index, videoEnd, onSeek, onShowItem]);
  return <div className="md">{renderBlocks(blocks, ctx)}</div>;
});

function Turn(props: {
  turn: UiTurn;
  index: LectureIndex;
  videoEnd: number;
  busy: boolean;
  onSeek(t: number): void;
  onShowItem(ref: AskRef): void;
  onOpenEntry(e: TrayEntry): void;
  onRetry(t: UiTurn): void;
}) {
  const { turn, index, videoEnd, busy, onSeek, onShowItem, onOpenEntry, onRetry } = props;
  const cost = costLabel(turn.cost, turn.cached);
  const attached = turn.entries.length;
  return (
    <li className={`ask-turn ${turn.status}`}>
      <div className="ask-q">
        {attached > 0 && (
          <div className="ask-q-items" aria-label="Attached">
            {turn.entries.map((e) => (
              <Chip key={itemKey(e.item)} entry={e} index={index} onOpen={onOpenEntry} />
            ))}
          </div>
        )}
        <p className="ask-q-text">{turn.q}</p>
        {turn.t != null && (
          <button type="button" className="ask-q-at" onClick={() => onSeek(turn.t!)} title="Go back to where you asked">
            asked at {formatTime(turn.t)}
          </button>
        )}
      </div>
      <div className="ask-a" aria-live={turn.status === 'streaming' ? 'polite' : undefined} aria-busy={turn.status === 'streaming'}>
        <div className="ask-a-head">
          <SparkIcon size={12} />
          Muse Spark
          {turn.status === 'streaming' && <span className="ask-typing" aria-label="answering" />}
        </div>
        {turn.answer ? (
          <AnswerBody turn={turn} index={index} videoEnd={videoEnd} onSeek={onSeek} onShowItem={onShowItem} />
        ) : (
          turn.status === 'streaming' && <p className="ask-wait">Reading the lecture…</p>
        )}
        {turn.status === 'error' && <p className="error-line ask-error">{turn.error}</p>}
        {turn.status === 'stopped' && <p className="ask-stopped">Stopped.</p>}
        {(turn.status === 'error' || turn.status === 'stopped') && (
          <button type="button" className="ask-retry" disabled={busy} onClick={() => onRetry(turn)}>
            Ask again
          </button>
        )}
        {turn.status === 'done' && (attached > 0 || cost || !turn.saved) && (
          <p className="ask-foot">
            {[
              attached > 0 && (
                <span key="n">{attached === 1 ? 'With 1 attached item' : `With ${attached} attached items`}</span>
              ),
              cost && (
                <span key="c" title="Cost of this answer">
                  {cost}
                </span>
              ),
              !turn.saved && (
                <span key="s" title="The server could not save this answer: it will be gone after a reload">
                  not saved
                </span>
              ),
            ]
              .filter(Boolean)
              .flatMap((x, i) => (i ? [<span key={`d${i}`} aria-hidden> · </span>, x] : [x]))}
          </p>
        )}
      </div>
    </li>
  );
}

const BrowseRow = memo(function BrowseRow(props: {
  it: LectureItem;
  attached: boolean;
  onToggle(it: LectureItem): void;
  onShow(it: LectureItem): void;
}) {
  const { it, attached, onToggle, onShow } = props;
  const html = it.latex ? texHtml(it.latex) : null;
  return (
    <li className={`ask-row ${attached ? 'on' : ''}`}>
      <button type="button" className="ask-row-time" onClick={() => onShow(it)} title="Show it in the lecture">
        {formatTime(it.t)}
      </button>
      <div className="ask-row-body">
        {html ? (
          <span className="ask-row-tex" dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <span className="ask-row-text">{it.text || it.label}</span>
        )}
        {(it.erasedAt != null || it.preExisting) && (
          <span className="ask-row-tags">
            {it.preExisting && <span className="ask-tag">already on the board</span>}
            {it.erasedAt != null && <span className="ask-tag erased">erased at {formatTime(it.erasedAt)}</span>}
          </span>
        )}
      </div>
      <button
        type="button"
        className="ask-row-add"
        aria-pressed={attached}
        aria-label={attached ? `Remove ${it.label}` : `Attach ${it.label}`}
        title={attached ? 'Attached (click to remove)' : 'Attach to your question'}
        onClick={() => onToggle(it)}
      >
        {attached ? (
          <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden>
            <path d="M3.5 8.4l3 3 6-6.6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden>
            <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        )}
      </button>
    </li>
  );
});

function Browse(props: {
  index: LectureIndex;
  mode: 'board' | 'slides' | 'none';
  tray: TrayEntry[];
  onToggle(it: LectureItem): void;
  onShow(it: LectureItem): void;
  onDone(): void;
}) {
  const { index, mode, tray, onToggle, onShow, onDone } = props;
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => inputRef.current?.focus(), []);
  const groups = useMemo(() => filterGroups(index.groups, query), [index, query]);
  const attached = useMemo(() => new Set(tray.map((e) => itemKey(e.item))), [tray]);
  const what = mode === 'slides' ? 'slide' : 'board';
  return (
    <div className="ask-browse">
      <div className="ask-browse-head">
        <input
          ref={inputRef}
          type="search"
          className="ask-search"
          placeholder={`Search the ${what}${mode === 'slides' ? 's' : ''}…`}
          aria-label={`Search everything on the ${what}`}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="button" className="ghost-btn" onClick={onDone}>
          Done
        </button>
      </div>
      <p className="ask-browse-note">
        {mode === 'slides'
          ? 'Every piece of slide text, by chapter.'
          : 'Everything written on the board, by chapter, including what was erased.'}{' '}
        Press + to attach it.
      </p>
      {groups.length === 0 && (
        <p className="ask-empty-line">{index.count ? 'Nothing matches that.' : `Nothing was found on the ${what}.`}</p>
      )}
      {groups.map((g) => (
        <section key={`${g.t}-${g.title}`} className="ask-group">
          <h4>
            <span>{g.title}</span>
            <span className="ask-group-time">{formatTime(g.t)}</span>
          </h4>
          {g.items.length ? (
            <ul className="ask-rows">
              {g.items.map((it) => (
                <BrowseRow
                  key={it.ref}
                  it={it}
                  attached={attached.has(itemKey(it.item))}
                  onToggle={onToggle}
                  onShow={onShow}
                />
              ))}
            </ul>
          ) : (
            <p className="ask-empty-line">Talk only: nothing new was written.</p>
          )}
        </section>
      ))}
    </div>
  );
}

const fromSaved = (t: AskTurn, index: LectureIndex): UiTurn => ({
  key: `saved-${t.id || Math.random().toString(36).slice(2)}`,
  id: t.id,
  q: t.q,
  entries: t.selection.map((i) => savedEntry(i, index)),
  answer: t.answer,
  items: t.items,
  times: t.times,
  t: t.t,
  cost: t.cost_usd,
  cached: !!t.cached,
  saved: true,
  status: 'done',
});

// Memoised: the player re-renders on every spoken word; the drawer only needs to when its props change.
export const AskDrawer = memo(function AskDrawer(props: Props) {
  const {
    jobId, open, focusKey, onClose, tray, setTray, notice, index, mode, timeRef, videoEnd, knows, onAddMoment, onSeek,
    onShowItem,
  } = props;
  const [turns, setTurns] = useState<UiTurn[]>([]);
  const turnsRef = useRef(turns);
  turnsRef.current = turns;
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [view, setView] = useState<'chat' | 'browse'>('chat');
  const [confirmClear, setConfirmClear] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const stick = useRef(true);
  const busy = turns.some((t) => t.status === 'streaming');

  // The saved conversation, once, when the drawer first opens.
  useEffect(() => {
    if (!open || loaded) return;
    const ac = new AbortController();
    getChat(jobId, ac.signal)
      .then((saved) => {
        setTurns((now) => [...saved.map((t) => fromSaved(t, index)), ...now]);
        setLoaded(true);
      })
      .catch((e) => {
        if ((e as Error).name === 'AbortError') return;
        setLoadError('The earlier conversation could not be loaded.');
        setLoaded(true);
      });
    return () => ac.abort();
  }, [open, loaded, jobId, index]);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Focus the question box only when the drawer was opened from its button or the A key (focusKey bumped). Opening it
  // by attaching something (Shift+click, Ask in a tooltip) leaves the focus where the student was, so Space still
  // plays the video instead of typing in the box.
  const focusedKey = useRef(0);
  useEffect(() => {
    if (!open || !focusKey || focusKey === focusedKey.current) return;
    focusedKey.current = focusKey;
    textRef.current?.focus();
  }, [open, focusKey]);

  // Follow the answer as it streams, unless the student scrolled up to read.
  useLayoutEffect(() => {
    const el = bodyRef.current;
    if (el && stick.current && view === 'chat') el.scrollTop = el.scrollHeight;
  }, [turns, view, open]);
  const onScroll = () => {
    const el = bodyRef.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const patch = useCallback((key: string, fn: (t: UiTurn) => Partial<UiTurn>) => {
    setTurns((ts) => ts.map((t) => (t.key === key ? { ...t, ...fn(t) } : t)));
  }, []);

  const send = useCallback(
    async (question: string, entries: TrayEntry[], fromTray: boolean) => {
      const q = question.trim();
      if (!q || abortRef.current) return;
      if (q.length > QUESTION_MAX) {
        setFormError(`Questions can be up to ${QUESTION_MAX} characters; this one has ${q.length}.`);
        return;
      }
      setFormError(null);
      setView('chat');
      const history = historyFor(turnsRef.current.map((t) => ({ q: t.q, answer: t.answer, ok: t.status === 'done' })));
      const key = `local-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const t = timeRef.current ?? null;
      setTurns((ts) => [
        ...ts,
        { key, id: '', q, entries, answer: '', items: [], times: [], t, cost: null, cached: false, saved: true, status: 'streaming' },
      ]);
      stick.current = true;
      if (fromTray) {
        setInput('');
        setTray([]);
      }
      const ac = new AbortController();
      abortRef.current = ac;
      // Text arrives in many small pieces: add them to the answer at most once per frame.
      let pending = '';
      let timer = 0;
      const flush = () => {
        timer = 0;
        const add = pending;
        pending = '';
        if (add) patch(key, (x) => ({ answer: x.answer + add }));
      };
      try {
        const done = await askMuse(
          jobId,
          { question: q, selection: requestSelection(entries), history, t },
          {
            signal: ac.signal,
            onDelta: (d) => {
              pending += d;
              if (!timer) timer = window.setTimeout(flush, 40);
            },
          },
        );
        window.clearTimeout(timer);
        flush();
        patch(key, (x) => ({
          id: done.id,
          answer: done.answer || x.answer,
          items: done.items,
          times: done.times,
          cost: done.cost_usd,
          cached: done.cached,
          saved: done.saved,
          status: 'done',
        }));
      } catch (e) {
        window.clearTimeout(timer);
        flush();
        if ((e as Error).name === 'AbortError') patch(key, () => ({ status: 'stopped' }));
        else patch(key, () => ({ status: 'error', error: (e as Error).message || 'Muse Spark could not answer.' }));
      } finally {
        if (abortRef.current === ac) abortRef.current = null;
      }
    },
    [jobId, patch, setTray, timeRef],
  );

  const submit = () => send(input, tray, true);
  const stop = () => abortRef.current?.abort();
  const retry = useCallback(
    (turn: UiTurn) => {
      setTurns((ts) => ts.filter((t) => t.key !== turn.key));
      void send(turn.q, turn.entries, false);
    },
    [send],
  );

  const startOver = async () => {
    setConfirmClear(false);
    abortRef.current?.abort();
    try {
      await clearChat(jobId);
      setTurns([]);
      setLoadError(null);
    } catch (e) {
      setLoadError((e as Error).message);
    }
    textRef.current?.focus();
  };

  const openEntry = useCallback(
    (e: TrayEntry) => {
      const it = e.item;
      if (it.kind === 'piece' || it.kind === 'line' || it.kind === 'element') {
        const found = index.byRef.get(`#${it.id}`);
        onShowItem(found ? askRefOf(found) : { ref: `#${it.id}`, kind: it.kind, id: it.id, t: e.t, label: e.label });
      } else onSeek(e.t);
    },
    [index, onSeek, onShowItem],
  );
  const removeFromTray = useCallback((e: TrayEntry) => setTray(removeEntry(tray, itemKey(e.item))), [tray, setTray]);

  const trayRef = useRef(tray);
  trayRef.current = tray;
  const [browseNote, setBrowseNote] = useState<string | null>(null);
  const toggleItem = useCallback(
    (it: LectureItem) => {
      const now = trayRef.current;
      if (hasEntry(now, it.item)) {
        setTray(removeEntry(now, itemKey(it.item)));
        setBrowseNote(null);
      } else if (now.length >= TRAY_MAX) setBrowseNote(`You can attach up to ${TRAY_MAX} things to one question.`);
      else {
        setTray([...now, { item: it.item, label: it.label, t: it.t }]);
        setBrowseNote(null);
      }
    },
    [setTray],
  );
  const showItem = useCallback((it: LectureItem) => onShowItem(askRefOf(it)), [onShowItem]);
  useEffect(() => {
    if (view === 'chat') setBrowseNote(null);
  }, [view]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'Escape') return;
    e.preventDefault();
    e.stopPropagation();
    if (confirmClear) setConfirmClear(false);
    else if (view === 'browse') {
      setView('chat');
      textRef.current?.focus();
    } else onClose();
  };

  const onTextKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (!busy) submit();
    }
  };

  // The question box grows with what is typed, up to a few lines.
  useLayoutEffect(() => {
    const el = textRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight + 2, 168)}px`;
  }, [input, open]);

  const suggestions = tray.length
    ? ['Explain this', ...(tray.length >= 2 ? ['How are these connected?'] : []), 'Give me a practice question']
    : ['Summarise the lecture so far', 'What should I review?'];
  const what = mode === 'slides' ? 'slides' : 'board';
  const left = QUESTION_MAX - input.trim().length;

  return (
    <aside className="ask-drawer" aria-label="Ask Muse" hidden={!open} onKeyDown={onKeyDown}>
      <header className="ask-head">
        <div className="ask-title">
          <SparkIcon size={15} />
          <h2>Ask Muse</h2>
          <span className={`ask-sub ${knows ? 'partial' : ''}`}>{knows || 'Muse Spark sees the whole lecture'}</span>
        </div>
        <div className="ask-head-actions">
          {turns.length > 0 && (
            <button
              type="button"
              className="ghost-btn ask-new"
              onClick={() => setConfirmClear(true)}
              disabled={confirmClear}
              title="Clear this conversation and start a new one"
            >
              New conversation
            </button>
          )}
          <button type="button" className="ask-close" aria-label="Close Ask Muse (Esc)" title="Close (Esc)" onClick={onClose}>
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
              <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </header>
      {confirmClear && (
        <div className="ask-confirm" role="alertdialog" aria-label="Start a new conversation?">
          <span>Clear this conversation? The saved questions and answers are deleted.</span>
          <span className="ask-confirm-actions">
            <button type="button" className="ask-danger" onClick={startOver}>
              Clear it
            </button>
            {/* the focus starts on the safe choice: Enter right after "New conversation" deletes nothing */}
            <button type="button" className="ghost-btn" onClick={() => setConfirmClear(false)} autoFocus>
              Keep it
            </button>
          </span>
        </div>
      )}

      <div className="ask-body" ref={bodyRef} onScroll={onScroll}>
        {view === 'browse' ? (
          <Browse
            index={index}
            mode={mode}
            tray={tray}
            onToggle={toggleItem}
            onShow={showItem}
            onDone={() => {
              setView('chat');
              textRef.current?.focus();
            }}
          />
        ) : (
          <>
            {loadError && <p className="error-line ask-load-error">{loadError}</p>}
            {turns.length === 0 ? (
              <div className="ask-intro">
                {knows ? (
                  <p>
                    Ask anything about this lecture. It is still being processed: Muse Spark reads the {what} as far as
                    it is ready and every word transcribed so far.
                  </p>
                ) : (
                  <p>
                    Ask anything about this lecture. Muse Spark reads all of it, the {what} and every word said, so you
                    can ask about the start and the end in one question.
                  </p>
                )}
                <p className="muted">
                  {mode === 'slides'
                    ? 'To ask about something on a slide, Shift+click it, or select words in the transcript and press Ask about this.'
                    : 'To ask about some writing, hover it on the board and press Ask (or Shift+click it), or select words in the transcript and press Ask about this.'}
                </p>
              </div>
            ) : (
              <ol className="ask-turns">
                {turns.map((t) => (
                  <Turn
                    key={t.key}
                    turn={t}
                    index={index}
                    videoEnd={videoEnd}
                    busy={busy}
                    onSeek={onSeek}
                    onShowItem={onShowItem}
                    onOpenEntry={openEntry}
                    onRetry={retry}
                  />
                ))}
              </ol>
            )}
          </>
        )}
      </div>

      <div className="ask-compose">
        {(notice || browseNote) && (
          <p key={notice?.key} className="ask-notice" role="status">
            {browseNote ?? notice?.text}
          </p>
        )}
        {tray.length > 0 && (
          <div className="ask-tray">
            <div className="ask-tray-head">
              <span>
                Attached <span className="muted">{tray.length}/{TRAY_MAX}</span>
              </span>
              <button type="button" className="ask-link" onClick={() => setTray([])}>
                Clear
              </button>
            </div>
            <div className="ask-chips">
              {tray.map((e) => (
                <Chip key={itemKey(e.item)} entry={e} index={index} onOpen={openEntry} onRemove={removeFromTray} />
              ))}
            </div>
          </div>
        )}
        <div className="ask-quick">
          <button type="button" className="ask-pill" onClick={onAddMoment} title="Attach the board and what is said around now">
            <KindIcon kind="moment" />
            This moment
          </button>
          <button
            type="button"
            className={`ask-pill ${view === 'browse' ? 'on' : ''}`}
            aria-pressed={view === 'browse'}
            onClick={() => setView((v) => (v === 'browse' ? 'chat' : 'browse'))}
            title={`Find anything on the ${what} and attach it`}
          >
            <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
              <circle cx="7" cy="7" r="4.4" fill="none" stroke="currentColor" strokeWidth="1.5" />
              <path d="M10.3 10.3l3.2 3.2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            Browse lecture
          </button>
        </div>
        {!busy && !input.trim() && (
          <div className="ask-suggest" aria-label="Suggested questions">
            {suggestions.map((s) => (
              <button key={s} type="button" className="ask-suggestion" onClick={() => send(s, tray, true)}>
                {s}
              </button>
            ))}
          </div>
        )}
        <form
          className="ask-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!busy) submit();
          }}
        >
          <textarea
            ref={textRef}
            className="ask-input"
            rows={1}
            value={input}
            placeholder={tray.length ? 'Ask about what you attached…' : 'Ask about the lecture…'}
            aria-label="Your question"
            onChange={(e) => {
              setInput(e.target.value);
              if (formError) setFormError(null);
            }}
            onKeyDown={onTextKey}
          />
          {busy ? (
            <button type="button" className="ask-send stop" onClick={stop} aria-label="Stop the answer">
              <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
                <rect x="3.5" y="3.5" width="9" height="9" rx="1.5" fill="currentColor" />
              </svg>
              Stop
            </button>
          ) : (
            <button type="submit" className="ask-send" disabled={!input.trim()} aria-label="Send the question">
              <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden>
                <path d="M2.5 8h10M8.5 3.5L13 8l-4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Send
            </button>
          )}
        </form>
        {formError ? (
          <p className="error-line ask-form-error">{formError}</p>
        ) : (
          left < 200 && <p className="ask-count">{left >= 0 ? `${left} characters left` : `${-left} characters too long`}</p>
        )}
      </div>
    </aside>
  );
});
