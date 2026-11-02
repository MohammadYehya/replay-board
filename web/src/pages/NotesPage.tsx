// Study notes for one lecture: every line of writing (or slide element) by chapter, typeset with KaTeX, with the
// moment it was explained (a link back into the player at that time) and what was said then. "Ask" on a line opens
// the player there with that line attached to an Ask Muse question.
import { useCallback, useEffect, useRef, useState } from 'react';
import { getReplay, notesUrl } from '../api';
import { SparkIcon } from '../ask/icons';
import { lectureIndex, matchNote, type LectureIndex } from '../ask/items';
import { addEntry, cleanItem, loadTray, openKey, saveTray, session, shortLabel, type TrayEntry } from '../ask/tray';
import { texHtml } from '../player/tex';
import { formatTime, isWrittenLine } from '../player/timing';
import { SlidesIcon } from '../slides/icons';

interface NoteItem {
  t: number;
  latex: string;
  text: string;
  drawing: boolean;
  said: string;
  /** The item to attach when asking Muse about this line: {kind: piece | line | element, id}. */
  ref?: unknown;
}
interface NotesData {
  title: string;
  chapters: { t: number; title: string; items: NoteItem[] }[];
}

const playerAt = (jobId: string, t: number) => `#/job/${encodeURIComponent(jobId)}?t=${Math.max(0, Math.floor(t))}`;

function Item({ jobId, it, onAsk }: { jobId: string; it: NoteItem; onAsk(it: NoteItem): void }) {
  const html = it.latex ? texHtml(it.latex) : null;
  return (
    <li className="note-item">
      <a className="note-time" href={playerAt(jobId, it.t - 1)} title="Watch this moment">
        {formatTime(it.t)}
      </a>
      <div className="note-body">
        {html ? (
          <span className="note-tex" dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <span className={it.drawing ? 'note-drawing' : 'note-text'}>{it.text || it.latex}</span>
        )}
        {it.said && <p className="note-said">“{it.said}”</p>}
      </div>
      <button className="note-ask" onClick={() => onAsk(it)} title="Ask Muse about this, in the player">
        <SparkIcon size={11} />
        Ask
      </button>
    </li>
  );
}

export function NotesPage({ jobId }: { jobId: string }) {
  const [data, setData] = useState<NotesData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    fetch(`/api/jobs/${encodeURIComponent(jobId)}/notes.json`, { signal: ac.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error(r.status === 404 ? 'This lecture is not processed yet.' : `Error ${r.status}`);
        return (await r.json()) as NotesData;
      })
      .then(setData)
      .catch((e) => {
        if (!ac.signal.aborted) setError(String(e.message || e));
      });
    return () => ac.abort();
  }, [jobId]);

  useEffect(() => {
    if (data) document.title = `Notes · ${data.title} · Replay Board`;
  }, [data]);

  // "Ask": put the board or slide item behind the note on the lecture's Ask Muse tray, and open the player there
  // with the drawer open. The server names the item (ref); a server without refs makes us find it in the replay,
  // fetched once, on the first Ask.
  const indexRef = useRef<Promise<LectureIndex | null> | null>(null);
  const ask = useCallback(
    async (it: NoteItem) => {
      const given = cleanItem(it.ref);
      let entry: TrayEntry;
      if (given) entry = { item: given, label: shortLabel(it.text || it.latex) || 'Writing', t: it.t };
      else {
        indexRef.current ??= getReplay(jobId)
          .then((r) =>
            lectureIndex(r, { isLine: (ln) => isWrittenLine(ln, r.board?.size?.[0] ?? 960) }),
          )
          .catch(() => null);
        const index = await indexRef.current;
        const found = index ? matchNote(index, it) : null;
        entry = found
          ? { item: found.item, label: found.label, t: found.t }
          : { item: { kind: 'moment', t: it.t }, label: 'Moment', t: it.t };
      }
      const store = session();
      saveTray(store, jobId, addEntry(loadTray(store, jobId), entry).tray);
      try {
        store?.setItem(openKey(jobId), '1');
      } catch {
        /* storage blocked: the player opens without the drawer */
      }
      window.location.hash = playerAt(jobId, entry.t - 1);
    },
    [jobId],
  );

  const title = data ? data.title.charAt(0).toUpperCase() + data.title.slice(1) : '';
  return (
    <div className="notes-page">
      <header className="notes-bar">
        <a className="ghost-btn" href={`#/job/${encodeURIComponent(jobId)}`}>
          ← Back to the lecture
        </a>
        <div className="notes-actions">
          <a className="ghost-btn" href={`#/job/${encodeURIComponent(jobId)}/slides`} title="This lecture as a slide deck">
            <SlidesIcon />
            Slides
          </a>
          <a className="ghost-btn" href={notesUrl(jobId)} download={`${jobId}-notes.md`}>
            Download Markdown
          </a>
          <button className="ghost-btn" onClick={() => window.print()}>
            Print
          </button>
        </div>
      </header>
      <main className="notes-main">
        {error && <p className="error-line">{error}</p>}
        {!data && !error && <p className="muted">Loading the notes…</p>}
        {data && (
          <>
            <h1 className="notes-title">{title}</h1>
            <p className="notes-sub">
              Notes made by Replay Board from the board and what was said. Click a time to watch that moment, or Ask
              to ask Muse Spark about a line.
            </p>
            {data.chapters.map((ch) => (
              <section key={ch.t} className="notes-chapter">
                <h2>
                  <a href={playerAt(jobId, ch.t)}>{ch.title}</a>
                  <span className="notes-ch-time">{formatTime(ch.t)}</span>
                </h2>
                {ch.items.length ? (
                  <ul className="note-list">
                    {ch.items.map((it, i) => (
                      <Item key={i} jobId={jobId} it={it} onAsk={ask} />
                    ))}
                  </ul>
                ) : (
                  <p className="muted note-empty">Talk only: nothing new was written.</p>
                )}
              </section>
            ))}
          </>
        )}
      </main>
    </div>
  );
}
