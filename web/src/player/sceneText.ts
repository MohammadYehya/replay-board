// How the 3D view shows the words and numbers of a scene, with no three.js and no DOM (so it runs in Node tests):
// - displayName(): every piece of scene text the player shows (labels, parameter and axis names, the title, captions,
//   step chips) reads "beta" as β and "75 degrees" as 75°, as the teacher writes them on the board;
// - formatValue(): a parameter's or an angle's value as the view prints it.

/** Greek letters written out as words, and the letter each one becomes. */
const GREEK: Record<string, string> = {
  alpha: 'α',
  beta: 'β',
  gamma: 'γ',
  delta: 'δ',
  theta: 'θ',
  phi: 'φ',
  psi: 'ψ',
  omega: 'ω',
  lambda: 'λ',
  mu: 'μ',
  sigma: 'σ',
  rho: 'ρ',
  tau: 'τ',
};

// Whole words only: "alphabet", "beta_arc" and "mu2" stay as they are (letters, digits and _ join a word).
const GREEK_WORD = new RegExp(`(?<![A-Za-z0-9_])(${Object.keys(GREEK).join('|')})(?![A-Za-z0-9_])`, 'gi');
// "75 degrees", "90 degree", "12.5 Degrees": the number and the degree sign.
const DEGREES = /(\d+(?:\.\d+)?)\s+degrees?(?![A-Za-z0-9_])/gi;

/** Scene text as the view shows it: Greek names as letters, "<number> degrees" as "<number>°". */
export function displayName(s: string): string {
  if (!s) return '';
  return s.replace(DEGREES, '$1°').replace(GREEK_WORD, (w) => GREEK[w.toLowerCase()] ?? w);
}

/** Words the model writes where the board writes a formula, in an object's label or the title only (captions are
 *  sentences): "z equals minus y squared" -> "z = −y²", "x squared minus y squared" -> "x² − y²". A minus after "="
 *  or at the start is a sign, glued to what follows; any other is an operation, with spaces. */
export function formulaText(s: string): string {
  if (!s) return '';
  return s
    .replace(/(\S)\s+squared(?![A-Za-z0-9_-])/gi, '$1²')
    .replace(/(\S)\s+cubed(?![A-Za-z0-9_-])/gi, '$1³')
    .replace(/(?<![A-Za-z0-9_-])equals(?![A-Za-z0-9_-])/gi, '=')
    .replace(/(^|[=(]\s*)minus\s+/gi, '$1−')
    .replace(/(?<![A-Za-z0-9_])minus\s+/gi, '− ');
}

/** An object's label or the view's title as the view shows it: scene text (displayName) with formulas as symbols. */
export function labelText(s: string): string {
  return formulaText(displayName(s));
}

/** The 3D view's title: the scene's, shown as scene text; before there is one, the board's own reading of the drawing
 *  as it is (the lecture's own text is never changed). */
export function viewTitle(sceneTitle: string | undefined, boardText: string | undefined): string {
  if (sceneTitle) return labelText(sceneTitle);
  return boardText?.trim() || 'Drawing from the board';
}

/** A number rounded to `digits` decimals, without trailing zeros ("75", "2.5", "-0.25"); never "-0". */
export function roundText(v: number, digits: number): string {
  if (!Number.isFinite(v)) return '–';
  const s = v.toFixed(digits);
  const t = s.includes('.') ? s.replace(/\.?0+$/, '') : s;
  return t === '-0' ? '0' : t;
}

/** An angle's measured value: one decimal, a trailing ".0" dropped, and the degree sign ("75°", "109.5°"). */
export function formatDegrees(v: number): string {
  return `${roundText(v, 1)}°`;
}

/** A parameter's value as the view prints it: degrees with one decimal and "°", anything else with at most two
 *  decimals ("3", "2.75"). */
export function formatValue(v: number, unit: '' | 'deg'): string {
  return unit === 'deg' ? formatDegrees(v) : roundText(v, Math.abs(v) >= 100 ? 0 : 2);
}

/** An angle's label: its name and its value now ("β 75°"), or the value alone when it has no name. */
export function angleLabel(name: string, degrees: number): string {
  const n = displayName(name).trim();
  return n ? `${n} ${formatDegrees(degrees)}` : formatDegrees(degrees);
}

/** The step of a "Try it" slider over [min, max]: whole degrees for an angle (a tenth for a range under 10 degrees),
 *  else a round step of about a hundredth of the range. */
export function sliderStep(min: number, max: number, unit: '' | 'deg'): number {
  const span = Math.abs(max - min);
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  if (unit === 'deg') return span >= 10 ? 1 : 0.1;
  return 10 ** Math.floor(Math.log10(span / 100));
}
