import { cleanChat, readAnswer } from './ask/protocol';
import { cleanDeck, pptxFileName } from './slides/deck';
import type { AskDone, AskRequest, AskTurn, Deck, JobStatus, JobSummary, PopoutScene, Replay } from './types';

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

const SERVER_DOWN = 'Cannot reach the Replay Board server. Is `python -m app.server` running?';
const NO_ASK = 'This server has no Ask Muse yet: restart `python -m app.server` with the latest code.';

/**
 * Ask Muse Spark a question about the lecture (POST /api/jobs/{id}/ask). The answer streams in: each piece of text
 * goes to onDelta as it arrives, and the promise resolves with the whole answer, its refs, times and cost.
 * Every failure is one plain line (an aborted request rejects with an AbortError).
 */
export async function askMuse(
  jobId: string,
  req: AskRequest,
  opts: { signal?: AbortSignal; onDelta?: (text: string) => void } = {},
): Promise<AskDone> {
  let r: Response;
  try {
    r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(req),
      signal: opts.signal,
      cache: 'no-store',
    });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new Error(SERVER_DOWN);
  }
  if (!r.ok) {
    const line = await errorLine(r);
    if ((r.status === 404 || r.status === 405) && /^(not found\.?|method not allowed)$/i.test(line)) throw new Error(NO_ASK);
    throw new Error(line);
  }
  if (!r.body || !(r.headers.get('content-type') ?? '').includes('text/event-stream')) {
    throw new Error('The server did not stream an answer. Is it up to date?');
  }
  return readAnswer(r.body, opts.onDelta ?? (() => {}));
}

/** The saved conversation about a lecture (oldest first); none when the server has no Ask Muse yet. */
export async function getChat(jobId: string, signal?: AbortSignal): Promise<AskTurn[]> {
  let r: Response;
  try {
    r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/chat`, { signal, cache: 'no-store' });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new Error(SERVER_DOWN);
  }
  if (r.status === 404 || r.status === 405) return [];
  if (!r.ok) throw new Error(await errorLine(r));
  return cleanChat(await r.json());
}

/** Forget the conversation about a lecture. */
export async function clearChat(jobId: string): Promise<void> {
  let r: Response;
  try {
    r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/chat`, { method: 'DELETE' });
  } catch {
    throw new Error(SERVER_DOWN);
  }
  if (!r.ok && r.status !== 404) throw new Error(await errorLine(r));
}

// ---------- slides export ----------

const NO_SLIDES = 'This server cannot make slides yet: restart `python -m app.server` with the latest code.';

/** The deck from GET slides.json or POST slides/points, or one plain line. A server without the slides routes answers
 *  the player's own index.html to the GET (and 405 to the POST): that is "restart the server", not a JSON error. */
async function slidesError(r: Response): Promise<Error> {
  const line = await errorLine(r);
  if ((r.status === 404 || r.status === 405) && /^(not found\.?|method not allowed)$/i.test(line)) return new Error(NO_SLIDES);
  return new Error(line);
}

async function readDeck(r: Response): Promise<Deck> {
  if (!r.ok) throw await slidesError(r);
  if (!(r.headers.get('content-type') ?? '').includes('json')) throw new Error(NO_SLIDES);
  let body: unknown;
  try {
    body = await r.json();
  } catch {
    throw new Error('The slides came back damaged. Try reloading the page.');
  }
  // An older server without a built player answers every unknown address with {"message": "player not built..."}.
  const o = body as { pages?: unknown; message?: unknown } | null;
  if (o && typeof o === 'object' && !Array.isArray(o.pages) && typeof o.message === 'string') throw new Error(NO_SLIDES);
  return cleanDeck(body);
}

/** The lecture's slide deck (made by code on the server, no model call). Key points may not be written yet. */
export async function getSlides(jobId: string, signal?: AbortSignal): Promise<Deck> {
  let r: Response;
  try {
    r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/slides.json`, { signal, cache: 'no-store' });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new Error(SERVER_DOWN);
  }
  return readDeck(r);
}

/** Ask Muse Spark for each slide's title and key points (one call per deck, cached by the server); the deck with
 *  them filled in. */
export async function writeSlidePoints(jobId: string, signal?: AbortSignal): Promise<Deck> {
  let r: Response;
  try {
    r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/slides/points`, { method: 'POST', signal, cache: 'no-store' });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new Error(SERVER_DOWN);
  }
  return readDeck(r);
}

/** The deck as a PowerPoint file. The version makes a new download after the key points land. */
export const slidesPptxUrl = (jobId: string, version = '', points = false) =>
  `/api/jobs/${encodeURIComponent(jobId)}/slides.pptx` +
  (version ? `?v=${encodeURIComponent(version)}${points ? '&points=1' : ''}` : '');

/** The PowerPoint and its file name, or one plain line. Fetched rather than followed as a plain link, so that a
 *  failure on the server (or an older server answering with the player's page) is a line on the page, not a broken
 *  or bogus download. */
export async function getSlidesPptx(url: string, title: string, signal?: AbortSignal): Promise<{ blob: Blob; name: string }> {
  let r: Response;
  try {
    r = await fetch(url, { signal, cache: 'no-store' });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new Error(SERVER_DOWN);
  }
  if (!r.ok) throw await slidesError(r);
  const type = (r.headers.get('content-type') ?? '').toLowerCase();
  if (type.includes('text/html') || type.includes('json')) throw new Error(NO_SLIDES);
  let blob: Blob;
  try {
    blob = await r.blob();
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new Error('The PowerPoint download broke off. Try again.');
  }
  if (!blob.size) throw new Error('The PowerPoint came back empty. Try again.');
  return { blob, name: pptxFileName(r.headers.get('content-disposition'), title) };
}
