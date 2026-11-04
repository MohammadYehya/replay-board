// Decodes each piece's PNG and drawing-order map once, then builds partially drawn frames on demand.
// Frames are cached per quantised progress step (timing.STEPS) in one shared LRU so long lectures stay bounded.
import { STEPS, thresholdFor } from './timing';

export interface Sprite {
  id: number;
  w: number;
  h: number;
  /** The whole piece, ready for drawImage. */
  full: HTMLCanvasElement;
  rgba: Uint8ClampedArray;
  /** Pixel indices sorted by drawing order. */
  sorted: Uint32Array;
  /** upto[T] = how many of `sorted` have order 1..T (T = 0..255). */
  upto: Uint32Array;
  /** Representative ink colour, css string. */
  ink: string;
  /** Vertical centre of the ink, 0 (top) .. 1 (bottom). */
  cy: number;
  /** When each pixel is erased (Infinity = never), or null when nothing of it is ever erased. */
  goneAt: Float32Array | null;
  /** Distinct erase times of this piece, ascending. */
  goneTimes: number[];
}

/** How many of the piece's partial erases have happened by t. */
export function goneCount(s: Sprite, t: number): number {
  let n = 0;
  while (n < s.goneTimes.length && s.goneTimes[n] <= t) n++;
  return n;
}

async function decode(url: string): Promise<ImageData> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const bmp = await createImageBitmap(await r.blob(), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('no 2d context');
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return ctx.getImageData(0, 0, c.width, c.height);
}

function toCanvas(img: ImageData): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = img.width;
  c.height = img.height;
  c.getContext('2d')?.putImageData(img, 0, 0);
  return c;
}

/** Mean colour of the strongest half of the ink: darkest on a light board, brightest on a dark one. */
function inkColour(rgba: Uint8ClampedArray, lightBoard: boolean): string {
  const lum: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i + 3] < 128) continue;
    lum.push(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]);
    idx.push(i);
  }
  if (!idx.length) return lightBoard ? '#2a2f3a' : '#f2f2ee';
  const order = lum.map((_, k) => k).sort((a, b) => (lightBoard ? lum[a] - lum[b] : lum[b] - lum[a]));
  const n = Math.max(1, Math.floor(order.length / 2));
  let r = 0, g = 0, b = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[order[k]];
    r += rgba[i];
    g += rgba[i + 1];
    b += rgba[i + 2];
  }
  const f = lightBoard ? 0.85 : 1; // a touch darker on white boards so typeset text reads well
  const c = (v: number) => Math.round(Math.min(255, (v / n) * f));
  return `rgb(${c(r)}, ${c(g)}, ${c(b)})`;
}

export async function loadSprite(
  id: number, imageUrl: string, orderUrl: string | null, lightBoard: boolean, goneUrl: string | null = null,
): Promise<Sprite> {
  const [img, ord, gone] = await Promise.all([
    decode(imageUrl),
    orderUrl ? decode(orderUrl).catch(() => null) : Promise.resolve(null),
    goneUrl ? decode(goneUrl).catch(() => null) : Promise.resolve(null),
  ]);
  const { width: w, height: h, data: rgba } = img;
  const n = w * h;
  const order = new Uint8Array(n);
  const sameSize = ord && ord.width === w && ord.height === h;
  for (let i = 0; i < n; i++) {
    if (rgba[i * 4 + 3] === 0) continue;
    // Missing/odd order map: fall back to a left-to-right sweep. Ink with order 0 is drawn last.
    const o = sameSize ? ord!.data[i * 4] : 1 + Math.floor(((i % w) / Math.max(1, w - 1)) * 254);
    order[i] = o === 0 ? 255 : o;
  }
  const counts = new Uint32Array(256);
  for (let i = 0; i < n; i++) if (order[i]) counts[order[i]]++;
  const upto = new Uint32Array(256);
  for (let t = 1; t < 256; t++) upto[t] = upto[t - 1] + counts[t];
  const pos = new Uint32Array(256);
  for (let t = 1; t < 256; t++) pos[t] = upto[t - 1];
  const sorted = new Uint32Array(upto[255]);
  for (let i = 0; i < n; i++) if (order[i]) sorted[pos[order[i]]++] = i;
  // Weighted by contrast squared, so faint ghost marks barely count.
  let sy = 0;
  let sw = 0;
  for (let i = 0; i < n; i++) {
    const a = rgba[i * 4 + 3];
    if (a < 64) continue;
    const lum = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
    const c = (lightBoard ? 255 - lum : lum) * (a / 255);
    sy += c * c * (Math.floor(i / w) + 0.5);
    sw += c * c;
  }
  const cy = sw ? sy / sw / h : 0.5;
  // Partly erased pieces: decode when each pixel leaves the board.
  let goneAt: Float32Array | null = null;
  const times = new Set<number>();
  if (gone && gone.width === w && gone.height === h) {
    goneAt = new Float32Array(n).fill(Infinity);
    for (let i = 0; i < n; i++) {
      if (!order[i]) continue;
      const code = (gone.data[i * 4] << 16) | (gone.data[i * 4 + 1] << 8) | gone.data[i * 4 + 2];
      if (!code) continue;
      const t = (code - 1) / 10;
      goneAt[i] = t;
      times.add(t);
    }
    if (!times.size) goneAt = null;
  }
  const goneTimes = [...times].sort((a, b) => a - b);
  return { id, w, h, full: toCanvas(img), rgba, sorted, upto, ink: inkColour(rgba, lightBoard), cy, goneAt, goneTimes };
}

/** Decoded sprites shared by every board renderer of one lecture (player/stream.ts spriteKey): a replay swapped in
 *  while the lecture streams rebuilds the renderer, which takes the sprites it had at once instead of fetching and
 *  decoding them again (the board would blank and refill on every swap). */
export class SpriteCache {
  private done = new Map<string, Sprite>();
  private pending = new Map<string, Promise<Sprite>>();

  get(key: string): Sprite | undefined {
    return this.done.get(key);
  }

  /** The sprite, loaded once however many ask for it. A failure is not kept (the next ask tries again). */
  load(key: string, make: () => Promise<Sprite>): Promise<Sprite> {
    const hit = this.done.get(key);
    if (hit) return Promise.resolve(hit);
    const going = this.pending.get(key);
    if (going) return going;
    const p = make().then(
      (s) => {
        if (this.pending.get(key) === p) {
          this.pending.delete(key);
          this.done.set(key, s);
        }
        return s;
      },
      (e) => {
        if (this.pending.get(key) === p) this.pending.delete(key);
        throw e;
      },
    );
    this.pending.set(key, p);
    return p;
  }

  /** Forget every sprite but these (a piece whose image grew has a new one; the old one is not needed again). */
  retain(keys: ReadonlySet<string>) {
    for (const k of [...this.done.keys()]) if (!keys.has(k)) this.done.delete(k);
    for (const k of [...this.pending.keys()]) if (!keys.has(k)) this.pending.delete(k);
  }

  get size() {
    return this.done.size;
  }
}

/** Shared LRU of partial frames, bounded by pixel bytes. */
export class FrameCache {
  private map = new Map<string, HTMLCanvasElement>();
  private bytes = 0;
  private budget: number;

  constructor(budgetBytes = 96 * 1024 * 1024) {
    this.budget = budgetBytes;
  }

  /** The sprite drawn up to `step` (1..STEPS), without the pixels erased by its first `gone` erases.
   *  STEPS with nothing erased returns the full piece. */
  frame(s: Sprite, step: number, gone = 0): HTMLCanvasElement {
    if (step >= STEPS && gone === 0) return s.full;
    const key = `${s.id}:${step}:${gone}`;
    const hit = this.map.get(key);
    if (hit) {
      this.map.delete(key);
      this.map.set(key, hit);
      return hit;
    }
    const img = new ImageData(s.w, s.h);
    const out = img.data;
    const count = s.upto[thresholdFor(Math.min(step, STEPS))];
    const cut = gone > 0 && s.goneAt ? s.goneTimes[gone - 1] : -Infinity;
    for (let k = 0; k < count; k++) {
      if (s.goneAt && s.goneAt[s.sorted[k]] <= cut) continue;
      const p = s.sorted[k] * 4;
      out[p] = s.rgba[p];
      out[p + 1] = s.rgba[p + 1];
      out[p + 2] = s.rgba[p + 2];
      out[p + 3] = s.rgba[p + 3];
    }
    const c = toCanvas(img);
    this.map.set(key, c);
    this.bytes += s.w * s.h * 4;
    while (this.bytes > this.budget && this.map.size > 1) {
      const [oldKey, old] = this.map.entries().next().value as [string, HTMLCanvasElement];
      this.map.delete(oldKey);
      this.bytes -= old.width * old.height * 4;
    }
    return c;
  }
}
