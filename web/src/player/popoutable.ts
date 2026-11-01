// Which board pieces get a "View in 3D" button: drawings (graphs, diagrams, sketches...) and equations that
// define a graph, y = f(x) or z = f(x, y). Slides never do.
import type { Piece } from '../types';

const DRAWING = /\b(graph|diagram|figure|plot|sketch|drawing|chart|curve)s?\b/i;
const EQUATION = /\b(equation|formula|function)s?\b/i;

/** "y = |ln x|", "z = x^2 + y^2", "f(x) = e^{-x}", "f(x, y) = ..." (LaTeX or plain), with x on the right. */
export function looksLikeGraphEquation(s: string): boolean {
  const flat = s
    .replace(/\$/g, '')
    .replace(/\\(left|right|displaystyle|,|;|!|quad)/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  const m = /^(?:y|z|[fgh]\s*\(\s*x\s*(?:,\s*y\s*)?\))\s*=\s*(.+)$/i.exec(flat);
  if (!m) return false;
  // the right-hand side (up to a second "=", as in y = |ln x| = |log_e x|) must use x
  const rhs = m[1].split('=')[0];
  return /x/i.test(rhs.replace(/\\[a-z]+/gi, ' '));
}

export function canPopout(p: Piece | null | undefined): boolean {
  const read = p?.read;
  if (!read) return false;
  const kind = read.kind ?? '';
  if (DRAWING.test(kind)) return true;
  if (!EQUATION.test(kind)) return false;
  return looksLikeGraphEquation(read.latex ?? '') || looksLikeGraphEquation(read.text ?? '');
}
