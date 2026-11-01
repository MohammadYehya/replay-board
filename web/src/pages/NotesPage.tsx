// Study notes for one lecture: every line of writing (or slide element) by chapter, typeset with KaTeX, with the
// moment it was explained (a link back into the player at that time) and what was said then.
import { useEffect, useState } from 'react';
import { notesUrl } from '../api';
import { texHtml } from '../player/tex';
import { formatTime } from '../player/timing';

interface NoteItem {
  t: number;
  latex: string;
  text: string;
  drawing: boolean;
  said: string;
}
interface NotesData {
  title: string;
  chapters: { t: number; title: string; items: NoteItem[] }[];
}

const playerAt = (jobId: string, t: number) => `#/job/${encodeURIComponent(jobId)}?t=${Math.max(0, Math.floor(t))}`;

function Item({ jobId, it }: { jobId: string; it: NoteItem }) {
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

  const title = data ? data.title.charAt(0).toUpperCase() + data.title.slice(1) : '';
  return (
    <div className="notes-page">
      <header className="notes-bar">
        <a className="ghost-btn" href={`#/job/${encodeURIComponent(jobId)}`}>
          ← Back to the lecture
        </a>
        <div className="notes-actions">
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
              Notes made by Replay Board from the board and what was said. Click a time to watch that moment.
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
                      <Item key={i} jobId={jobId} it={it} />
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
