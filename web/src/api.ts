import type { JobStatus, JobSummary, Replay } from './types';

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
export const getReplay = (id: string, signal?: AbortSignal) =>
  getJson<Replay>(`/api/jobs/${encodeURIComponent(id)}/replay.json`, signal);

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
