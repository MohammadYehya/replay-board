import { useEffect, useState } from 'react';
import { getJob, getReplay } from '../api';
import { Logo, Player } from '../player/Player';
import { formatTime } from '../player/timing';
import type { JobStatus, Replay, StageStatus } from '../types';

// Stage keys written by app/pipeline.py, in pipeline order, with who does the work.
const STAGES: Record<string, [string, string]> = {
  probe: ['Read the video file', 'ffmpeg'],
  classify: ['Board or slides?', 'Muse Spark'],
  board: ['Find the board', 'SAM 3.1'],
  people: ['Find the teacher', 'SAM 3.1'],
  capture: ['Capture the writing', 'stroke tracker'],
  speech: ['Transcribe the voice', 'Voice Transcribe + MMS'],
  read: ['Read the writing', 'Muse Spark'],
  link: ['Link writing to speech', 'Muse Spark'],
  write: ['Write the replay', ''],
};

function stageList(stages: Record<string, StageStatus> | undefined): [string, StageStatus][] {
  const s = stages ?? {};
  const known = Object.keys(STAGES).filter((k) => k in s);
  const extra = Object.keys(s).filter((k) => !(k in STAGES));
  return [...known, ...extra].map((k) => [k, s[k]]);
}

const title = (k: string) => STAGES[k]?.[0] ?? k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, ' ');

function StageIcon({ state, failed }: { state: string; failed: boolean }) {
  if (state === 'done')
    return (
      <svg viewBox="0 0 20 20" className="st-icon done" aria-label="done">
        <circle cx="10" cy="10" r="9" />
        <path d="M6 10.3l2.6 2.6L14.2 7" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  if (state === 'running' && failed)
    return (
      <svg viewBox="0 0 20 20" className="st-icon failed" aria-label="stopped">
        <circle cx="10" cy="10" r="9" />
        <path d="M7 7l6 6M13 7l-6 6" fill="none" strokeWidth="2" strokeLinecap="round" />
      </svg>
    );
  if (state === 'running') return <span className="st-icon spinner" aria-label="running" />;
  if (state === 'skipped')
    return (
      <svg viewBox="0 0 20 20" className="st-icon skipped" aria-label="skipped">
        <circle cx="10" cy="10" r="9" />
        <path d="M6.5 10h7" fill="none" strokeWidth="2" strokeLinecap="round" />
      </svg>
    );
  return (
    <svg viewBox="0 0 20 20" className="st-icon pending" aria-label="waiting">
      <circle cx="10" cy="10" r="8" />
    </svg>
  );
}

function JobProgress({ status, error, missing }: { status: JobStatus | null; error: string | null; missing: boolean }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(id);
  }, []);
  const failed = status?.state === 'failed';
  const stages = stageList(status?.stages);
  const doneCount = stages.reduce(
    (acc, [, s]) => acc + (s.state === 'done' || s.state === 'skipped' ? 1 : s.state === 'running' ? (s.progress ?? 0) : 0),
    0,
  );
  const overall = stages.length ? doneCount / stages.length : 0;
  const elapsed =
    status?.seconds != null && (status.state === 'done' || failed)
      ? status.seconds
      : status?.started
        ? Math.max(0, now - status.started)
        : null;
  let stateLabel = 'Processing';
  if (missing) stateLabel = 'Lecture not found';
  else if (status == null) stateLabel = error ? 'Cannot reach the server' : 'Connecting…';
  else if (status.state === 'queued') stateLabel = 'Waiting in the queue';
  else if (failed) stateLabel = 'Failed';
  else if (status.state === 'done') stateLabel = 'Done';

  return (
    <div className="page">
      <header className="topbar">
        <a className="brand" href="#/">
          <Logo />
          <span>Replay Board</span>
        </a>
        <div className="topbar-title">
          <span className="subject">{status?.name ?? status?.id ?? 'Lecture'}</span>
        </div>
        <div className="topbar-actions">
          <a className="ghost-btn" href="#/">
            All lectures
          </a>
        </div>
      </header>
      <main className="narrow">
        <section className="panel job-card">
          <div className="job-head">
            <div>
              <h1>{stateLabel}</h1>
              {!missing && (status || !error) && (
                <p className="muted">
                  {failed
                    ? 'Processing stopped.'
                    : 'Meta models read the board and the voice; the replay opens as soon as it is ready.'}
                </p>
              )}
            </div>
            {elapsed != null && <div className="elapsed">{formatTime(elapsed)}</div>}
          </div>
          {!failed && !missing && status && (
            <div className="bar big">
              <div style={{ width: `${Math.round(overall * 100)}%` }} />
            </div>
          )}
          {failed && <p className="error-line">{status?.error || 'Processing failed.'}</p>}
          {error && <p className="error-line">{error}</p>}
          {stages.length > 0 ? (
            <ol className="stages">
              {stages.map(([key, s]) => {
                const by = STAGES[key]?.[1];
                const pct = s.state === 'done' ? 1 : Math.max(0, Math.min(1, s.progress ?? 0));
                return (
                  <li key={key} className={`stage ${s.state} ${failed && s.state === 'running' ? 'stopped' : ''}`}>
                    <StageIcon state={s.state} failed={failed} />
                    <div className="stage-body">
                      <div className="stage-line">
                        <span className="stage-name">{title(key)}</span>
                        {by && <span className="stage-by">{by}</span>}
                        <span className="stage-secs">
                          {s.seconds != null ? `${s.seconds < 10 ? s.seconds.toFixed(1) : Math.round(s.seconds)} s` : ''}
                        </span>
                      </div>
                      {(s.state === 'running' || s.state === 'done') && (
                        <div className="bar">
                          <div style={{ width: `${Math.round(pct * 100)}%` }} />
                        </div>
                      )}
                      {s.note && <div className="stage-note">{s.note}</div>}
                    </div>
                  </li>
                );
              })}
            </ol>
          ) : (
            status && !failed && <p className="muted">Starting…</p>
          )}
          {(failed || missing) && (
            <p>
              <a className="link" href="#/">
                Back to all lectures
              </a>
            </p>
          )}
        </section>
      </main>
    </div>
  );
}

export function JobPage({ jobId }: { jobId: string }) {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    setStatus(null);
    setReplay(null);
    setError(null);
    setMissing(false);
    let stopped = false;
    let timer: number | undefined;
    let gotStatus = false;
    let afterFail = 0;
    const ctrl = new AbortController();
    const poll = async () => {
      let wait = 1000;
      try {
        const st = await getJob(jobId, ctrl.signal);
        if (stopped) return;
        gotStatus = true;
        setStatus(st);
        setError(null);
        if (st.state === 'failed') {
          // Stages running in parallel may still finish and update the list; look a few more times, then stop.
          if (++afterFail > 4) return;
          wait = 2000;
        }
        if (st.ready && (st.state === 'done' || st.state === 'unknown')) {
          const r = await getReplay(jobId, ctrl.signal);
          if (!stopped) setReplay(r);
          return;
        }
        if (st.state === 'done') {
          setError('Processing finished but no replay was written.');
          return;
        }
      } catch (e) {
        if (stopped || (e as Error).name === 'AbortError') return;
        const msg = (e as Error).message || 'Something went wrong.';
        if (!gotStatus && /no such job|not found/i.test(msg)) {
          setMissing(true);
          setError(`There is no lecture called “${jobId}”.`);
          return;
        }
        setError(msg); // keep polling: the server may be restarting
      }
      timer = window.setTimeout(poll, wait);
    };
    poll();
    return () => {
      stopped = true;
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, [jobId]);

  if (replay) return <Player key={jobId} jobId={jobId} replay={replay} name={status?.name} />;
  return <JobProgress status={status} error={error} missing={missing} />;
}
