import { useEffect, useRef, useState } from 'react';
import { listJobs, uploadVideo } from '../api';
import { Logo } from '../player/Player';
import type { JobSummary } from '../types';

const VIDEO_EXT = ['.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi'];

const isVideo = (f: File) => f.type.startsWith('video/') || VIDEO_EXT.some((x) => f.name.toLowerCase().endsWith(x));

function mb(bytes: number) {
  if (bytes <= 0) return '0 MB';
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(0.1, bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;
}

/** "90", "1:30" or "1:02:03" -> seconds; "" -> undefined; junk -> NaN. */
function parseTime(s: string): number | undefined {
  const t = s.trim();
  if (!t) return undefined;
  if (!/^\d+(:\d{1,2}){0,2}(\.\d+)?$/.test(t)) return NaN;
  return t.split(':').reduce((acc, part) => acc * 60 + Number(part), 0);
}

function JobList() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    const ctrl = new AbortController();
    const load = async () => {
      try {
        const j = await listJobs(ctrl.signal);
        if (stopped) return;
        setJobs(j);
        setError(null);
      } catch (e) {
        if (stopped || (e as Error).name === 'AbortError') return;
        setError((e as Error).message);
      }
      timer = window.setTimeout(load, 4000);
    };
    load();
    return () => {
      stopped = true;
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, []);

  return (
    <section className="panel jobs-panel">
      <header className="panel-head">
        <h2>Your lectures</h2>
      </header>
      {error && <p className="error-line">{error}</p>}
      {jobs && jobs.length === 0 && <p className="muted pad">Nothing yet. Upload a lecture to get started.</p>}
      {jobs && jobs.length > 0 && (
        <ul className="jobs">
          {jobs.map((j) => {
            const label =
              j.state === 'done' && j.ready
                ? 'Ready'
                : j.state === 'failed'
                  ? 'Failed'
                  : j.state === 'running'
                    ? 'Processing'
                    : j.state === 'queued'
                      ? 'Queued'
                      : j.ready
                        ? 'Ready'
                        : j.state;
            const cls = label === 'Ready' ? 'ok' : label === 'Failed' ? 'bad' : label === 'Processing' ? 'busy' : 'idle';
            const sub = j.detail || (j.name && j.name !== j.id ? j.id : '');
            const showId = !!sub;
            return (
              <li key={j.id}>
                <a href={`#/job/${encodeURIComponent(j.id)}`} className={showId ? '' : 'single'}>
                  <span className="job-name">{j.name || j.id}</span>
                  {showId && <span className="job-id">{sub}</span>}
                  <span className={`pill ${cls}`}>{label}</span>
                </a>
                {j.state === 'failed' && j.error && (
                  <p className="job-error" title={j.error}>
                    {j.error}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

export function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [deck, setDeck] = useState<File | null>(null);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => abortRef.current?.(), []);

  const choose = (f: File | null | undefined) => {
    if (!f) return;
    if (!isVideo(f)) {
      setError(`“${f.name}” is not a video. Use ${VIDEO_EXT.join(', ')}.`);
      return;
    }
    setError(null);
    setFile(f);
  };

  const s0 = parseTime(start);
  const s1 = parseTime(end);
  const rangeError =
    (s0 !== undefined && Number.isNaN(s0)) || (s1 !== undefined && Number.isNaN(s1))
      ? 'Times look like 90 or 1:30.'
      : s0 !== undefined && s1 !== undefined && s1 <= s0
        ? 'The end must be after the start.'
        : null;

  const submit = async () => {
    if (!file || rangeError || progress != null) return;
    setError(null);
    setProgress(0);
    const job = uploadVideo(file, { start: s0, end: s1, deck, onProgress: setProgress });
    abortRef.current = job.abort;
    try {
      const id = await job.promise;
      window.location.hash = `#/job/${encodeURIComponent(id)}`;
    } catch (e) {
      setError((e as Error).message);
      setProgress(null);
    } finally {
      abortRef.current = null;
    }
  };

  const uploading = progress != null;
  const uploadingRef = useRef(uploading);
  uploadingRef.current = uploading;
  const chooseRef = useRef(choose);
  chooseRef.current = choose;

  // A video dropped anywhere on the page is taken (a near miss must not open the file in the tab instead).
  useEffect(() => {
    let depth = 0;
    // Drops onto a file field (the slide deck) are left to the field itself.
    const carriesFiles = (e: DragEvent) => !!e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files');
    const hasFiles = (e: DragEvent) =>
      carriesFiles(e) && !(e.target instanceof HTMLInputElement && e.target.type === 'file');
    const enter = (e: DragEvent) => {
      if (!carriesFiles(e)) return;
      depth++;
      if (!uploadingRef.current) setDragging(true);
    };
    const over = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = uploadingRef.current ? 'none' : 'copy';
    };
    const leave = (e: DragEvent) => {
      if (!carriesFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (!depth) setDragging(false);
    };
    const drop = (e: DragEvent) => {
      depth = 0;
      setDragging(false);
      if (!hasFiles(e)) return;
      e.preventDefault();
      if (!uploadingRef.current) chooseRef.current(e.dataTransfer?.files?.[0]);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragover', over);
    window.addEventListener('dragleave', leave);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragover', over);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('drop', drop);
    };
  }, []);

  return (
    <div className="page">
      <header className="topbar">
        <a className="brand" href="#/">
          <Logo />
          <span>Replay Board</span>
        </a>
      </header>
      <main className="narrow home">
        <section className="hero">
          <h1>Rewatch the lecture. See the whole board.</h1>
          <p>
            Upload a recorded lecture. Replay Board plays the original video untouched, next to a clean board that
            redraws every stroke in time with the voice, including writing the teacher was standing in front of.
          </p>
        </section>

        <section className={`panel drop ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}>
          <input
            ref={inputRef}
            type="file"
            accept={['video/*', ...VIDEO_EXT].join(',')}
            hidden
            onChange={(e) => {
              choose(e.target.files?.[0]);
              e.target.value = '';
            }}
          />
          {!file ? (
            <button className="drop-target" onClick={() => inputRef.current?.click()}>
              <svg viewBox="0 0 48 48" width="44" height="44" aria-hidden>
                <rect x="5" y="9" width="38" height="26" rx="4" fill="none" stroke="currentColor" strokeWidth="2.5" />
                <path d="M24 30V16m0 0l-6 6m6-6l6 6" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M17 41h14" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
              <span className="drop-title">Drop a lecture video here</span>
              <span className="drop-sub">or click to choose a file · MP4, MOV, MKV, WebM</span>
            </button>
          ) : (
            <div className="chosen">
              <div className="file-row">
                <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden>
                  <rect x="3" y="5" width="18" height="14" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
                  <path d="M10 9.5v5l4.5-2.5z" fill="var(--accent)" />
                </svg>
                <div className="file-meta">
                  <span className="file-title">{file.name}</span>
                  <span className="muted">{mb(file.size)}</span>
                </div>
                {!uploading && (
                  <button className="ghost-btn" onClick={() => setFile(null)}>
                    Change
                  </button>
                )}
              </div>

              <div className="options">
                <label>
                  <span>Process from</span>
                  <input
                    value={start}
                    onChange={(e) => setStart(e.target.value)}
                    placeholder="start"
                    disabled={uploading}
                    inputMode="decimal"
                  />
                </label>
                <label>
                  <span>to</span>
                  <input
                    value={end}
                    onChange={(e) => setEnd(e.target.value)}
                    placeholder="end"
                    disabled={uploading}
                    inputMode="decimal"
                  />
                </label>
                <label className="deck">
                  <span>Slide deck (optional)</span>
                  <input
                    type="file"
                    accept="application/pdf,.pdf"
                    disabled={uploading}
                    onChange={(e) => setDeck(e.target.files?.[0] ?? null)}
                  />
                </label>
              </div>
              <p className="hint">Leave the times empty to process the whole video. Times look like 90 or 1:30.</p>
              {rangeError && <p className="error-line">{rangeError}</p>}

              {uploading ? (
                <div className="upload-progress">
                  <div className="bar big">
                    <div style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
                  </div>
                  <div className="upload-line">
                    <span>
                      {progress! < 1
                        ? `Uploading ${Math.round(progress! * 100)}% · ${mb(file.size * progress!)} of ${mb(file.size)}`
                        : 'Uploaded. Starting…'}
                    </span>
                    {progress! < 1 && (
                      <button className="ghost-btn" onClick={() => abortRef.current?.()}>
                        Cancel
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <button className="primary-btn" onClick={submit} disabled={!!rangeError}>
                  Upload and process
                </button>
              )}
            </div>
          )}
          {error && <p className="error-line">{error}</p>}
        </section>

        <JobList />
      </main>
    </div>
  );
}
