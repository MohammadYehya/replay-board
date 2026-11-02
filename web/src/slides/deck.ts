// The slide deck made from a lecture (GET /api/jobs/{id}/slides.json, app/export.py): checking what the server
// sends, the typeset list of a slide's items, present-mode keys, and the page's toggles kept per lecture.
// No DOM and no runtime imports beyond plain modules, so `npm test` can run it with plain Node.
import type { Deck, DeckItem, DeckPage } from '../types';
import { formatTime } from '../player/timing.ts';

/** Kinds the server flags as drawings ({drawing: true}); an item of one of these kinds is a drawing either way. */
export const DECK_DRAWING_KINDS = new Set([
  'graph', 'diagram', 'arrow', 'scribble', 'underline', 'box', 'figure', 'chart', 'drawing',
]);

/** At most this many key points are shown per slide (Muse Spark is asked for 2 to 4). */
export const POINTS_MAX = 6;

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const str = (v: unknown, fallback = '') => (typeof v === 'string' ? v : fallback);

/** A path inside the job folder ("export/page_001.png"), or '' when it is not one. */
export function cleanImagePath(v: unknown): string {
  if (typeof v !== 'string') return '';
  const p = v.trim().replace(/\\/g, '/');
  if (!p || p.startsWith('/') || /^[a-z][a-z0-9+.-]*:/i.test(p)) return '';
  const parts = p.split('/');
  if (parts.some((s) => s === '..' || s === '.' || s === '')) return '';
  return p;
}

function cleanItem(x: unknown, fallbackT: number): DeckItem | null {
  if (!x || typeof x !== 'object') return null;
  const o = x as Record<string, unknown>;
  const kind = str(o.kind).trim().toLowerCase();
  const item: DeckItem = {
    ref: str(o.ref),
    kind,
    text: str(o.text).trim(),
    latex: str(o.latex).trim(),
    t: finite(o.t) && o.t >= 0 ? o.t : fallbackT,
  };
  if (o.drawing === true || DECK_DRAWING_KINDS.has(kind)) item.drawing = true;
  if (!item.ref && !item.text && !item.latex && !item.kind) return null;
  return item;
}

function cleanPoints(v: unknown): string[] | null {
  if (!Array.isArray(v)) return null;
  return v
    .filter((p): p is string => typeof p === 'string')
    .map((p) => p.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .slice(0, POINTS_MAX);
}

function cleanPage(x: unknown, i: number): DeckPage | null {
  if (!x || typeof x !== 'object') return null;
  const o = x as Record<string, unknown>;
  let t0 = finite(o.t0) && o.t0 >= 0 ? o.t0 : 0;
  let t1 = finite(o.t1) && o.t1 >= 0 ? o.t1 : t0;
  if (t1 < t0) [t0, t1] = [t1, t0];
  const chapter = str(o.chapter).trim();
  const size = Array.isArray(o.image_size) ? o.image_size : null;
  const imageSize: [number, number] | null =
    size && size.length >= 2 && finite(size[0]) && finite(size[1]) && size[0] > 0 && size[1] > 0
      ? [size[0], size[1]]
      : null;
  return {
    id: str(o.id).trim() || `p${i + 1}`,
    t0,
    t1,
    chapter,
    title: str(o.title).trim() || chapter || `Slide ${i + 1}`,
    image: cleanImagePath(o.image),
    image_size: imageSize,
    items: Array.isArray(o.items)
      ? o.items.map((it) => cleanItem(it, t0)).filter((it): it is DeckItem => it !== null)
      : [],
    said: str(o.said).trim(),
    points: cleanPoints(o.points),
  };
}

/** The deck as the page uses it, from whatever the server sent; throws one plain line when it is not a deck. */
export function cleanDeck(x: unknown): Deck {
  if (!x || typeof x !== 'object' || !Array.isArray((x as { pages?: unknown }).pages)) {
    throw new Error('The slides came back empty. Try reloading the page.');
  }
  const o = x as Record<string, unknown>;
  const pages = (o.pages as unknown[]).map(cleanPage).filter((p): p is DeckPage => p !== null);
  // Ids key the slide cards: two pages with the same id would make React mix them up.
  const seen = new Set<string>();
  for (const p of pages) {
    let id = p.id;
    for (let n = 2; seen.has(id); n++) id = `${p.id}-${n}`;
    seen.add(id);
    p.id = id;
  }
  const type = str(o.type) === 'slides' ? 'slides' : 'board';
  return {
    type,
    title: str(o.title).trim() || 'Lecture',
    version: str(o.version),
    points_ready: o.points_ready === true,
    pages,
  };
}

/** One plain line as the page shows it: the server's lines start in lower case ("this lecture is not processed
 *  yet, ..."), and a line never spreads over several. */
export function sentence(s: string): string {
  const t = s.replace(/\s+/g, ' ').trim();
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : t;
}

const UNSAFE_NAME = /[\x00-\x1f<>:"/\\|?*]+/g;

/** The PowerPoint's file name: the server's (Content-Disposition, the UTF-8 `filename*` first), else
 *  "<lecture title> - slides.pptx"; never a path, never without ".pptx". */
export function pptxFileName(disposition: string | null, title: string): string {
  let name = '';
  if (disposition) {
    const star = disposition.match(/filename\*\s*=\s*[^']*'[^']*'([^;]+)/i);
    if (star) {
      try {
        name = decodeURIComponent(star[1].trim().replace(/^"|"$/g, ''));
      } catch {
        name = '';
      }
    }
    if (!name) {
      const plain = disposition.match(/filename\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^;]+))/i);
      if (plain) name = plain[1] !== undefined ? plain[1].replace(/\\(.)/g, '$1') : (plain[2] ?? '');
    }
  }
  const tidy = (s: string) => s.replace(UNSAFE_NAME, ' ').replace(/\s+/g, ' ').trim().replace(/^[.\s-]+|[.\s]+$/g, '');
  name = tidy(name);
  if (!name) name = `${tidy(title).slice(0, 80).trim() || 'lecture'} - slides.pptx`;
  return /\.pptx$/i.test(name) ? name : `${name}.pptx`;
}

/** "1:23–4:56" */
export const rangeLabel = (t0: number, t1: number) => `${formatTime(t0)}–${formatTime(t1)}`;

/** The player at a moment (whole seconds on the video's clock). */
export const playerAt = (jobId: string, t: number) =>
  `#/job/${encodeURIComponent(jobId)}?t=${Math.max(0, Math.floor(finite(t) ? t : 0))}`;

const kindName = (k: string, fallback: string) => (k ? k.charAt(0).toUpperCase() + k.slice(1) : fallback);
const readable = (s: string) => (s && s.toLowerCase() !== 'unreadable' ? s : '');

/** A drawing named in plain text: "Graph: y = x² against x", or just "Diagram". */
export function drawingName(it: DeckItem): string {
  const name = kindName(it.kind, 'Drawing');
  const what = readable(it.text);
  return what && what.toLowerCase() !== it.kind ? `${name}: ${what}` : name;
}

export interface TypesetEntry {
  ref: string;
  /** Where the player should seek to see it explained. */
  t: number;
  /** LaTeX to typeset; null for plain text and drawings. */
  tex: string | null;
  /** Plain text: what it says, the drawing's name, or the LaTeX when KaTeX cannot typeset it. */
  text: string;
  drawing: boolean;
}

/** A slide's items for the Typeset view, in the server's (reading) order: writing typeset, drawings named. */
export function typesetList(page: Pick<DeckPage, 'items'>): TypesetEntry[] {
  return page.items.map((it) => {
    if (it.drawing) return { ref: it.ref, t: it.t, tex: null, text: drawingName(it), drawing: true };
    const latex = it.latex.trim();
    const text = readable(it.text) || latex || kindName(it.kind, 'Writing');
    return { ref: it.ref, t: it.t, tex: latex || null, text, drawing: false };
  });
}

/** Points beside the image for a tall image, under it for a wide one. */
export const pointsBeside = (size: [number, number] | null) => !!size && size[0] / size[1] < 1.45;

// ---------- present mode ----------

export const clampIndex = (i: number, count: number) => (count <= 0 ? 0 : Math.min(Math.max(0, Math.round(i)), count - 1));

/**
 * What a key does in present mode: the slide index to show, 'exit', or null (not ours: let the browser have it).
 * Next: → ↓ PageDown Space N; previous: ← ↑ PageUp Backspace P (Shift+Space too); Home / End; Esc leaves.
 */
export function presentKey(key: string, shift: boolean, index: number, count: number): number | 'exit' | null {
  switch (key) {
    case 'Escape':
    case 'Esc':
      return 'exit';
    case 'ArrowRight':
    case 'ArrowDown':
    case 'PageDown':
    case 'n':
    case 'N':
      return clampIndex(index + 1, count);
    case ' ':
    case 'Spacebar':
      return clampIndex(index + (shift ? -1 : 1), count);
    case 'ArrowLeft':
    case 'ArrowUp':
    case 'PageUp':
    case 'Backspace':
    case 'p':
    case 'P':
      return clampIndex(index - 1, count);
    case 'Home':
      return 0;
    case 'End':
      return clampIndex(count - 1, count);
    default:
      return null;
  }
}

// ---------- the page's toggles, kept per lecture ----------

export type SlidesView = 'image' | 'typeset';

export interface SlidesPrefs {
  view: SlidesView;
  /** Include what was said (on the page and when printing). */
  said: boolean;
}

export const DEFAULT_PREFS: SlidesPrefs = { view: 'image', said: false };
export const prefsKey = (jobId: string) => `replay-board:slides:${jobId}`;

type Store = Pick<Storage, 'getItem' | 'setItem'>;

/** The toggles saved for this lecture; the defaults when there are none, storage is blocked, or it holds junk. */
export function loadPrefs(store: Store | null, jobId: string): SlidesPrefs {
  try {
    const raw = store?.getItem(prefsKey(jobId));
    if (!raw) return { ...DEFAULT_PREFS };
    const o = JSON.parse(raw) as Record<string, unknown>;
    return {
      view: o && o.view === 'typeset' ? 'typeset' : 'image',
      said: !!o && o.said === true,
    };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

export function savePrefs(store: Store | null, jobId: string, prefs: SlidesPrefs): void {
  try {
    store?.setItem(prefsKey(jobId), JSON.stringify({ view: prefs.view, said: prefs.said }));
  } catch {
    /* storage full or blocked: the toggles just are not remembered */
  }
}
