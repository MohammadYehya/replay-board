// Tooltip helpers shared by the board and the slide view.
import { texHtml } from './tex';
import { cleanLatex } from './texClean';
import { clauses } from './texLayout';

export const kindLabel = (k: string | undefined, fallback = 'Writing') =>
  k ? k.charAt(0).toUpperCase() + k.slice(1) : fallback;

/** True when the plain text only repeats the LaTeX, e.g. "A = P(1+r)^n" and "$A=P(1+r)^n$". */
export function sameReading(text: string, latex: string) {
  const norm = (s: string) =>
    s
      .replace(/\\(left|right|tfrac|dfrac|frac|quad|qquad|cdot|,|;|!)/g, '')
      .replace(/[\s{}\\$]/g, '')
      .toLowerCase();
  return norm(text) === norm(cleanLatex(latex));
}

/** The reading for the tooltip: short LaTeX on one display line; long LaTeX as a paragraph of its clauses that
 *  wraps between clauses and after relations, so nothing is cut off at the tooltip's edge. */
export function tipTex(latex: string): string | null {
  if (!latex) return null;
  const clean = cleanLatex(latex);
  if (clean.length <= 34) return texHtml(clean, true);
  const parts = clauses(clean).map((c) => texHtml(c));
  if (!parts.length || parts.some((x) => x == null)) return texHtml(clean, true);
  return `<div class="tip-line">${parts.join(' ')}</div>`;
}
