// Fitting a slide's text: the largest size from `max` down to `min` points at which a box does not spill, like the
// PowerPoint. Every box waiting is fitted in one batch: each step sets the size of every box still searching, then
// reads them all, so the page lays out once per step for the whole deck. Fitted one box at a time, a 76-slide deck
// laid the page out 231 times on opening (8 s of a frozen page, measured); in one batch it takes about 6.
// No DOM, so `npm test` can run it with plain Node: the slide frames pass in what to set and what to measure.

export interface FitBox {
  max: number;
  min: number;
  /** Sets the text size, in points. */
  set(size: number): void;
  /** Whether the content spills at the size set last. */
  spills(): boolean;
  /** False once the box is gone from the page. */
  live?(): boolean;
}

/** Fits the boxes together (a binary search each, in step); returns how many measuring steps it took. */
export function fitAll(boxes: FitBox[]): number {
  if (!boxes.length) return 0;
  for (const b of boxes) b.set(b.max);
  let steps = 1;
  const searching = boxes
    .filter((b) => b.spills())
    .map((b) => ({ b, lo: b.min, hi: b.max - 1, best: b.min, mid: b.min }));
  let open = searching.filter((o) => o.lo <= o.hi);
  while (open.length) {
    for (const o of open) {
      o.mid = Math.floor((o.lo + o.hi) / 2);
      o.b.set(o.mid);
    }
    steps++;
    for (const o of open) {
      if (o.b.spills()) o.hi = o.mid - 1;
      else {
        o.best = o.mid;
        o.lo = o.mid + 1;
      }
    }
    open = open.filter((o) => o.lo <= o.hi);
  }
  for (const o of searching) o.b.set(o.best);
  return steps;
}

/** Boxes waiting to be fitted. A box added while the page renders is fitted before the next paint, with every other
 *  box added meanwhile; one added after a resize or when fonts arrive (`settle`) waits until things are quiet for a
 *  moment, so dragging a window's edge does not fit the deck at every frame. */
export class FitQueue {
  settleMs: number;
  private waiting = new Set<FitBox>();
  private soon = false;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(settleMs = 150) {
    this.settleMs = settleMs;
  }

  add(box: FitBox, settle = false): void {
    this.waiting.add(box);
    if (this.soon) return;
    if (!settle) {
      this.soon = true;
      queueMicrotask(() => this.flush());
      return;
    }
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), this.settleMs);
  }

  delete(box: FitBox): void {
    this.waiting.delete(box);
  }

  /** Fits every box waiting now; returns the measuring steps it took. */
  flush(): number {
    this.soon = false;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    const boxes = [...this.waiting].filter((b) => b.live?.() ?? true);
    this.waiting.clear();
    return fitAll(boxes);
  }
}
