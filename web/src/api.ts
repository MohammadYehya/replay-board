import type { JobStatus, JobSummary, PopoutScene, Replay } from './types';

/** URL of a file inside a job folder (video, piece images). Each path segment is encoded. */
export function jobFileUrl(jobId: string, path: string): string {
  const clean = path.replace(/\\/g, '/').split('/').filter(Boolean).map(encodeURIComponent).join('/');
  return `/api/jobs/${encodeURIComponent(jobId)}/files/${clean}`;
}

/** One plain line for the owner, never a stack trace. */
async function errorLine(r: Response): Promise<string> {
  try {
    const body = await r.json();
    if (body && typeof body.detail === 'string') return body.detail;
  } catch {
    /* not JSON */
  }
  if (r.status === 404) return 'Not found.';
  return `The server answered ${r.status} ${r.statusText}.`.trim();
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  let r: Response;
  try {
    r = await fetch(url, { signal, cache: 'no-store' });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new Error('Cannot reach the Replay Board server. Is `python -m app.server` running?');
  }
  if (!r.ok) throw new Error(await errorLine(r));
  return (await r.json()) as T;
}

export const listJobs = (signal?: AbortSignal) => getJson<JobSummary[]>('/api/jobs', signal);
export const getJob = (id: string, signal?: AbortSignal) =>
  getJson<JobStatus>(`/api/jobs/${encodeURIComponent(id)}`, signal);
/** The replay, or an error while it cannot be read whole (the server answers {} while the file is replaced). */
export async function getReplay(id: string, signal?: AbortSignal): Promise<Replay> {
  const r = await getJson<Replay>(`/api/jobs/${encodeURIComponent(id)}/replay.json`, signal);
  if (!r || typeof r !== 'object' || !r.video || typeof r.video.file !== 'string') {
    throw new Error('The replay is being written. Trying again…');
  }
  return r;
}

/** Study notes for a job (Markdown), opened in a new tab. */
export const notesUrl = (id: string) => `/api/jobs/${encodeURIComponent(id)}/notes.md`;

/** The 3D scene Muse Spark makes for one board piece. The first call for a piece takes a few seconds (and a
 *  model call); the server caches the answer, so later ones are instant. */
export async function getPopout(jobId: string, pieceId: number, signal?: AbortSignal): Promise<PopoutScene> {
  let r: Response;
  try {
    r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/popout/${pieceId}`, { method: 'POST', signal });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new Error('Cannot reach the Replay Board server. Is `python -m app.server` running?');
  }
  if (!r.ok) throw new Error(await errorLine(r));
  const scene = (await r.json()) as PopoutScene;
  if (!scene || typeof scene !== 'object' || typeof scene.kind !== 'string')
    throw new Error('The 3D view came back empty.');
  return scene;
}

export interface UploadOptions {
  start?: number;
  end?: number;
  deck?: File | null;
  onProgress?: (fraction: number) => void;
}

/** Upload a video with XMLHttpRequest so we can show upload progress. Resolves to the new job id. */
export function uploadVideo(file: File, opts: UploadOptions = {}): { promise: Promise<string>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<string>((resolve, reject) => {
    const form = new FormData();
    form.append('file', file, file.name);
    if (opts.start != null) form.append('start', String(opts.start));
    if (opts.end != null) form.append('end', String(opts.end));
    if (opts.deck) form.append('deck', opts.deck, opts.deck.name);
    xhr.open('POST', '/api/jobs');
    xhr.responseType = 'json';
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && opts.onProgress) opts.onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      const body = xhr.response as { id?: string; detail?: unknown } | null;
      if (xhr.status >= 200 && xhr.status < 300 && body && typeof body.id === 'string') {
        opts.onProgress?.(1);
        resolve(body.id);
      } else if (body && typeof body.detail === 'string') {
        reject(new Error(body.detail));
      } else {
        reject(new Error(`Upload failed (server answered ${xhr.status}).`));
      }
    };
    xhr.onerror = () => reject(new Error('Upload failed: cannot reach the Replay Board server.'));
    xhr.onabort = () => reject(new Error('Upload cancelled.'));
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}
