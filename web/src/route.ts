// The player's pages, from the address bar's hash. No DOM, so `npm test` can run it with plain Node.

export type Route =
  | { page: 'home' }
  | { page: 'job'; id: string; t: number | null }
  | { page: 'notes'; id: string }
  | { page: 'slides'; id: string };

// #/job/<id>, #/job/<id>?t=<seconds> (open the player at a moment), #/job/<id>/notes, #/job/<id>/slides
export function parseRoute(hash: string): Route {
  const m = hash.match(/^#\/job\/([^/?#]+)(\/notes|\/slides)?\/?(?:\?(.*))?$/);
  if (m) {
    try {
      const id = decodeURIComponent(m[1]);
      if (m[2] === '/notes') return { page: 'notes', id };
      if (m[2] === '/slides') return { page: 'slides', id };
      // No "t" (or an empty one) is no moment: Number(null) and Number('') would both be 0, the clip's start.
      const q = new URLSearchParams(m[3] ?? '').get('t');
      const t = q === null || !q.trim() ? NaN : Number(q);
      return { page: 'job', id, t: Number.isFinite(t) && t >= 0 ? t : null };
    } catch {
      return { page: 'home' };
    }
  }
  return { page: 'home' };
}
