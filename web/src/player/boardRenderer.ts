// Draws the virtual board for a time t on a <canvas>, and drives the HTML overlays (typeset pieces,
// "was hidden" tags) from the same clock. Pure canvas drawing: the original video is never touched.
import { jobFileUrl } from '../api';
import type { BoardData, BoardLine, Piece } from '../types';
import { FrameCache, goneCount, loadSprite, type Sprite } from './sprites';
import { alphaAt, isWrittenLine, progressAt, schedule, stepFor, tagAt, talkingAt, typesetIds, type Schedule } from './timing';

interface Entry {
  piece: Piece;
  sched: Schedule;
  sprite: Sprite | null;
  typeset: boolean;
  area: number;
  p: number; // progress at the last render
  a: number; // alpha at the last render
  /** Bigger typeset pieces whose box holds this one and whose reading repeats it (the reader saw it twice). */
  coveredBy: Entry[];
  hide: number; // 0..1, how far a covering piece has replaced this one in typeset mode (last render)
  lineHide: number; // 0..1, how far its finished line has replaced it in typeset mode (last render)
}

/** A line written in bursts (several pieces), typeset as one expression once all of it is written. */
interface LineEntry {
  line: BoardLine;
  key: number; // negative id of its typeset element
  members: Entry[];
  ok: boolean; // its typesetting fits legibly
  w: number; // 0..1 shown (last render)
}

const LINE_FADE = 0.4;

/** Typeset-layer id of a line: negative, so it never clashes with a piece id. */
export const lineKey = (i: number) => -(i + 1);

/** Reading reduced to its symbols, for "does this reading repeat that one" checks. */
const readingKey = (p: Piece) =>
  (p.read?.latex?.trim() || p.read?.text?.trim() || '')
    .replace(/\\(left|right|text|mathrm)(?![a-zA-Z])/g, '')
    .replace(/[\s{}$\\]/g, '')
    .toLowerCase();

function overlapShare(a: Piece['box'], b: Piece['box']): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? (w * h) / Math.max(1, a[2] * a[3]) : 0;
}

function parseColour(c: string | undefined): [number, number, number] | null {
  if (!c) return null;
  const hex = c.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    const h = hex[1].length === 3 ? hex[1].replace(/./g, (x) => x + x) : hex[1];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  const rgb = c.match(/rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)/i);
  return rgb ? [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])] : null;
}

export function boardBackground(board: BoardData): { css: string; light: boolean } {
  const fallback = board.mode === 'chalk' ? '#26302c' : '#f4f4f1';
  const rgb = parseColour(board.background) ?? parseColour(fallback)!;
  const lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
  return { css: parseColour(board.background) ? board.background! : fallback, light: lum > 120 };
}

export class BoardRenderer {
  readonly W: number;
  readonly H: number;
  readonly bg: string;
  readonly light: boolean;
  readonly entries: Entry[];
  readonly byId = new Map<number, Entry>();
  /** Pieces that may be typeset (the typeset layer can still veto one it cannot fit legibly). */
  readonly typesetIds: Set<number>;
  /** Elements the renderer updates every frame; filled by the React layer. */
  typesetEls = new Map<number, HTMLElement>();
  tagEls = new Map<number, HTMLElement>();
  readonly lines: LineEntry[] = [];
  private lineByKey = new Map<number, LineEntry>();

  private cache = new FrameCache();
  private ctx: CanvasRenderingContext2D;
  private scale = 1; // canvas pixels per board pixel
  private cssScale = 1; // css pixels per board pixel
  private lastT = NaN;
  private dirty = true;
  private hoverId: number | null = null;
  private typesetMode = false;
  private dead = false;
  private styleMemo = new WeakMap<HTMLElement, string>();

  private canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement, board: BoardData, jobId: string) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('Canvas is not available in this browser.');
    this.ctx = ctx;
    [this.W, this.H] = board.size;
    const bg = boardBackground(board);
    this.bg = bg.css;
    this.light = bg.light;
    this.typesetIds = typesetIds(board.pieces, this.W, this.H);
    this.entries = board.pieces.map((piece) => ({
      piece,
      sched: schedule(piece),
      sprite: null,
      typeset: this.typesetIds.has(piece.id),
      area: Math.max(1, piece.box[2] * piece.box[3]),
      p: 0,
      a: 0,
      coveredBy: [],
      hide: 0,
      lineHide: 0,
    }));
    for (const e of this.entries) {
      if (!e.typeset) continue;
      const key = readingKey(e.piece);
      if (!key) continue;
      for (const o of this.entries) {
        if (o === e || !o.typeset || o.area < e.area || (o.area === e.area && o.piece.id < e.piece.id)) continue;
        if (overlapShare(e.piece.box, o.piece.box) >= 0.6 && readingKey(o.piece).includes(key)) e.coveredBy.push(o);
      }
    }
    // Pre-existing writing underneath, then in the order it is drawn.
    this.entries.sort((a, b) => a.sched.start - b.sched.start);
    for (const e of this.entries) this.byId.set(e.piece.id, e);
    (board.lines ?? []).forEach((line, i) => {
      const members = line.pieces.map((id) => this.byId.get(id)).filter((m): m is Entry => !!m);
      if (members.length < 2 || !(line.read?.latex?.trim() || line.read?.text?.trim()) || !isWrittenLine(line, this.W)) return;
      const le: LineEntry = { line, key: lineKey(i), members, ok: true, w: 0 };
      this.lines.push(le);
      this.lineByKey.set(le.key, le);
    });
    for (const e of this.entries) {
      const p = e.piece;
      const file = (f: string | null | undefined) => (f ? jobFileUrl(jobId, f) : null);
      loadSprite(p.id, jobFileUrl(jobId, p.image), file(p.order), this.light, file(p.gone))
        .then((s) => {
          if (this.dead) return;
          e.sprite = s;
          this.dirty = true;
          this.onSpriteLoaded?.(p.id, s);
        })
        .catch(() => {
          /* a missing piece image just isn't drawn */
        });
    }
  }

  onSpriteLoaded?: (id: number, s: Sprite) => void;

  destroy() {
    this.dead = true;
  }

  /** Match the canvas resolution to its on-screen size. */
  resize(cssWidth: number, dpr: number) {
    if (cssWidth <= 0) return;
    this.cssScale = cssWidth / this.W;
    const scale = Math.min(4, Math.max(1, (cssWidth * dpr) / this.W));
    const w = Math.round(this.W * scale);
    const h = Math.round(this.H * scale);
    if (w !== this.canvas.width || h !== this.canvas.height) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.scale = w / this.W;
    this.dirty = true;
  }

  setHover(id: number | null) {
    if (id !== this.hoverId) {
      this.hoverId = id;
      this.dirty = true;
    }
  }

  setTypeset(on: boolean) {
    if (on !== this.typesetMode) {
      this.typesetMode = on;
      this.dirty = true;
    }
  }

  invalidate() {
    this.dirty = true;
  }

  /** The typeset layer reports when a reading cannot be shown legibly in its box: keep the handwriting. */
  setTypesetOk(id: number, ok: boolean) {
    const le = this.lineByKey.get(id);
    if (le) {
      if (le.ok !== ok) {
        le.ok = ok;
        this.dirty = true;
      }
      return;
    }
    const e = this.byId.get(id);
    if (!e) return;
    const v = ok && this.typesetIds.has(id);
    if (v !== e.typeset) {
      e.typeset = v;
      this.dirty = true;
    }
  }

  /** Same pieces, new timing (a replay rewritten with refined word times): reschedule them, keep the sprites. */
  setPieces(pieces: Piece[]) {
    let changed = false;
    for (const p of pieces) {
      const e = this.byId.get(p.id);
      if (!e || e.piece === p) continue;
      e.piece = p;
      e.sched = schedule(p);
      changed = true;
    }
    if (changed) {
      this.entries.sort((a, b) => a.sched.start - b.sched.start);
      this.dirty = true;
    }
  }

  inkOf(id: number): string | null {
    return this.byId.get(id)?.sprite?.ink ?? null;
  }

  isVisible(id: number, t: number): boolean {
    const e = this.byId.get(id);
    return !!e && progressAt(e.sched, t) > 0 && alphaAt(e.sched, t) >= 0.05;
  }

  private setStyle(el: HTMLElement, opacity: number, clip: string) {
    const key = `${opacity.toFixed(3)}|${clip}`;
    if (this.styleMemo.get(el) === key) return;
    this.styleMemo.set(el, key);
    el.style.opacity = opacity.toFixed(3);
    el.style.clipPath = clip;
    el.style.visibility = opacity > 0 ? 'visible' : 'hidden';
  }

  render(t: number) {
    if (!this.dirty && t === this.lastT) return;
    this.dirty = false;
    this.lastT = t;
    const { ctx, scale } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = this.bg;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    for (const e of this.entries) {
      e.p = progressAt(e.sched, t);
      e.a = alphaAt(e.sched, t);
    }
    // In typeset mode a repeated reading fades out as the piece that repeats it is drawn.
    for (const e of this.entries) {
      e.hide = 0;
      e.lineHide = 0;
      if (!this.typesetMode || !e.typeset) continue;
      for (const o of e.coveredBy) if (o.typeset) e.hide = Math.max(e.hide, o.p * o.a);
    }
    // A finished line replaces its fragments: shown once every piece of it is fully drawn (fading in over
    // LINE_FADE after the last one finishes) and faded with the first of them to be erased.
    for (const le of this.lines) {
      le.w = 0;
      if (!this.typesetMode || !le.ok || le.members.some((m) => m.p < 1)) continue;
      let since = Infinity;
      let alpha = 1;
      for (const m of le.members) {
        if (m.sched.end !== -Infinity) since = Math.min(since, t - m.sched.end);
        alpha = Math.min(alpha, m.a);
      }
      le.w = Math.min(1, since / LINE_FADE) * alpha;
      for (const m of le.members) m.lineHide = Math.max(m.lineHide, le.w);
    }

    // Highlighter behind whatever is being talked about right now.
    for (const e of this.entries) {
      if (e.p <= 0 || e.a <= 0) continue;
      const k = talkingAt(e.piece, t) * e.a * Math.min(1, e.p * 2);
      if (k < 0.01) continue;
      const [x, y, w, h] = e.piece.box;
      ctx.fillStyle = this.light ? `rgba(255, 207, 51, ${0.34 * k})` : `rgba(255, 214, 92, ${0.2 * k})`;
      ctx.beginPath();
      ctx.roundRect(x - 4, y - 3, w + 8, h + 6, 5);
      ctx.fill();
    }

    const hover = this.hoverId != null ? this.byId.get(this.hoverId) : undefined;
    if (hover && hover.p > 0 && hover.a > 0) {
      const [x, y, w, h] = hover.piece.box;
      ctx.fillStyle = this.light ? 'rgba(76, 141, 255, 0.13)' : 'rgba(122, 167, 255, 0.16)';
      ctx.beginPath();
      ctx.roundRect(x - 5, y - 4, w + 10, h + 8, 6);
      ctx.fill();
    }

    for (const e of this.entries) {
      if (e.p <= 0 || e.a <= 0 || !e.sprite) continue;
      if (this.typesetMode && e.typeset) continue;
      if (e.lineHide > 0.98) continue;
      const step = stepFor(e.p);
      if (!step) continue;
      const [x, y, w, h] = e.piece.box;
      ctx.globalAlpha = e.a * (1 - e.lineHide);
      ctx.drawImage(this.cache.frame(e.sprite, step, goneCount(e.sprite, t)), x, y, w || e.sprite.w, h || e.sprite.h);
    }
    ctx.globalAlpha = 1;

    if (hover && hover.p > 0 && hover.a > 0) {
      const [x, y, w, h] = hover.piece.box;
      ctx.strokeStyle = this.light ? 'rgba(37, 99, 235, 0.85)' : 'rgba(122, 167, 255, 0.9)';
      ctx.lineWidth = 1.5 / this.cssScale;
      ctx.beginPath();
      ctx.roundRect(x - 5, y - 4, w + 10, h + 8, 6);
      ctx.stroke();
    }

    for (const [id, el] of this.typesetEls) {
      const le = this.lineByKey.get(id);
      if (le) {
        this.setStyle(el, le.w, 'none');
        continue;
      }
      const e = this.byId.get(id);
      if (!e) continue;
      const vis = e.p > 0 && e.typeset ? e.a * (1 - e.hide) * (1 - e.lineHide) : 0;
      this.setStyle(el, vis, e.p >= 1 ? 'none' : `inset(-20% ${((1 - e.p) * 100).toFixed(1)}% -20% -5%)`);
    }
    for (const [id, el] of this.tagEls) {
      const e = this.byId.get(id);
      if (!e) continue;
      this.setStyle(el, tagAt(e.sched, t) * e.a, 'none');
    }
  }

  /** The visible piece whose ink is nearest to board point (bx, by), within a few pixels. */
  hitTest(bx: number, by: number, t: number): Piece | null {
    const R = Math.max(4, 9 / this.cssScale);
    let best: Entry | null = null;
    let bestD = Infinity;
    for (const e of this.entries) {
      if (progressAt(e.sched, t) <= 0 || alphaAt(e.sched, t) < 0.05) continue;
      if (this.typesetMode && e.typeset && e.hide > 0.95) continue;
      const [x, y, w, h] = e.piece.box;
      if (bx < x - R || bx > x + w + R || by < y - R || by > y + h + R) continue;
      let d: number;
      if (this.typesetMode && (e.typeset || e.lineHide > 0.5)) {
        const dx = Math.max(x - bx, 0, bx - (x + w));
        const dy = Math.max(y - by, 0, by - (y + h));
        d = Math.hypot(dx, dy);
      } else if (e.sprite) {
        const s = e.sprite;
        d = nearestInk(s, ((bx - x) * s.w) / (w || s.w), ((by - y) * s.h) / (h || s.h), R, t);
      } else continue;
      if (d > R) continue;
      if (d < bestD - 0.75 || (Math.abs(d - bestD) <= 0.75 && best && e.area < best.area)) {
        best = e;
        bestD = d;
      }
    }
    return best?.piece ?? null;
  }
}

function nearestInk(s: Sprite, lx: number, ly: number, R: number, t: number): number {
  const x0 = Math.max(0, Math.floor(lx - R));
  const x1 = Math.min(s.w - 1, Math.ceil(lx + R));
  const y0 = Math.max(0, Math.floor(ly - R));
  const y1 = Math.min(s.h - 1, Math.ceil(ly + R));
  let best = Infinity;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if (s.rgba[(y * s.w + x) * 4 + 3] < 64) continue;
      if (s.goneAt && s.goneAt[y * s.w + x] <= t) continue; // already erased
      const d = Math.hypot(x - lx, y - ly);
      if (d < best) best = d;
    }
  }
  return best;
}
