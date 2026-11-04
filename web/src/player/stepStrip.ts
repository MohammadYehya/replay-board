// The 3D view's strip of step chips: it scrolls in whole chips. A scroll position always starts at a chip, a chip the
// strip's edge would cut is hidden (a gap, never a sliver), and there is room after the last chip for the strip to
// end on whole chips too. No DOM here, so `npm test` can run it with plain Node.
//
// Positions are the chips' offsetLeft and width inside the strip (its padding included, as the DOM gives them), the
// strip's padding-left (`pad`, where the first chip sits) and its clientWidth (`view`).

export interface StripLayout {
  lefts: number[];
  widths: number[];
  pad: number;
  view: number;
}

const EPS = 0.5;

/** The scroll position that starts at chip j (as chip 0 sits at scroll 0). */
export const startOf = (s: StripLayout, j: number) => Math.max(0, s.lefts[j] - s.pad);

/** Whether chip i is whole in the strip at scroll position `scroll`. */
export function isWhole(s: StripLayout, i: number, scroll: number): boolean {
  const left = s.lefts[i];
  return left >= scroll - EPS && left + s.widths[i] <= scroll + s.view + EPS;
}

/** The chips the strip's edges cut at `scroll`: they are hidden, so no sliver of a chip shows next to the arrows. */
export function cutChips(s: StripLayout, scroll: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < s.lefts.length; i++) {
    const left = s.lefts[i];
    const right = left + s.widths[i];
    if (right > scroll + EPS && left < scroll + s.view - EPS && !isWhole(s, i, scroll)) out.push(i);
  }
  return out;
}

/** Where the strip scrolls to show chip i whole (-1, before the first step: the first chip): of the chip starts it
 *  can scroll to (up to `max`) from which chip i is whole, the nearest to where it is. So it stays put while the chip
 *  is whole, brings a chip cut on the left to the left edge, and one cut on the right just in. */
export function scrollFor(s: StripLayout, i: number, scroll: number, max = Infinity): number {
  const n = s.lefts.length;
  if (!n) return 0;
  const k = Math.max(0, Math.min(n - 1, i));
  let best = startOf(s, k); // a chip wider than the strip: at least its start shows
  let dist = Infinity;
  for (let j = 0; j <= k; j++) {
    const at = startOf(s, j);
    if (at > max + EPS || !isWhole(s, k, at)) continue;
    if (Math.abs(at - scroll) < dist) {
      best = at;
      dist = Math.abs(at - scroll);
    }
  }
  return best;
}

/** How much room to leave after the last chip so the strip can scroll to the chip start that shows the last chips
 *  whole (the browser stops scrolling at the end of the content, which is rarely a chip start). `end` is where the
 *  content ends now (the last chip's right edge plus the strip's padding-right). */
export function endRoom(s: StripLayout, end: number): number {
  const n = s.lefts.length;
  if (!n) return 0;
  const last = n - 1;
  let j = last;
  while (j > 0 && isWhole(s, last, startOf(s, j - 1))) j--;
  const reach = Math.max(0, end - s.view); // the furthest the strip scrolls now
  return startOf(s, j) > reach + EPS ? Math.ceil(startOf(s, j) - reach) : 0;
}
