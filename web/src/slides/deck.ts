// The lecture deck Muse Spark writes (GET /api/jobs/{id}/slides.json, app/export.py; format 2 in types.ts):
// checking what the server sends, inline maths, the content layout, present-mode keys and the PowerPoint's name.
// No DOM and no runtime imports beyond plain modules, so `npm test` can run it with plain Node.
import type {
  BulletItem, BulletsBlock, CalloutBlock, CalloutStyle, Deck, DeckBlock, DeckSlide, EquationBlock, FigureBlock,
  SlideKind, SlidesState, SlidesStatus, StepItem, StepsBlock, TableBlock,
} from '../types';
import { formatTime } from '../player/timing.ts';

/** What one deck may hold. The server keeps to these; the page enforces them again so junk cannot break it. */
export const LIMITS = {
  title: 70,
  subtitle: 110,
  section: 70,
  bullet: 120,
  bullets: 6,
  label: 40,
  /** A steps block's label (the server allows it more than an equation's). */
  stepsLabel: 60,
  latex: 200,
  stepText: 90,
  steps: 6,
  cell: 40,
  columns: 6,
  rows: 8,
  caption: 160,
  calloutTitle: 70,
  calloutText: 280,
  refs: 12,
  /** Blocks on a content, agenda or summary slide. */
  blocks: 3,
  /** Formulas on one formula-sheet slide. */
  formulas: 8,
  notes: 1200,
  said: 3000,
  degraded: 8,
  slides: 200,
} as const;

const SLIDE_KINDS: readonly SlideKind[] = ['title', 'agenda', 'section', 'content', 'summary', 'formulas'];
const CALLOUT_STYLES: readonly CalloutStyle[] = ['definition', 'key', 'note'];
const STATES: readonly SlidesState[] = ['none', 'writing', 'ready', 'failed'];

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const time = (v: unknown): number | null => (finite(v) && v >= 0 ? v : null);
const str = (v: unknown) => (typeof v === 'string' ? v : '');
const obj = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;

/** One line, spaces tidied. */
export const oneLine = (v: unknown) => str(v).replace(/\s+/g, ' ').trim();

/** Plain text that keeps its paragraphs (speaker notes, what was said): spaces tidied, at most `max` characters. */
function prose(v: unknown, max: number): string {
  const t = str(v)
    .replace(/[ \t\f\v\r]+/g, ' ')
    .replace(/ ?\n ?/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return t.length > max ? cutPlain(t, max) : t;
}

/** At most `max` characters, cut at a word when one is near, with "…". */
function cutPlain(s: string, max: number): string {
  if (s.length <= max) return s;
  const head = s.slice(0, max - 1);
  const space = head.lastIndexOf(' ');
  return `${(space > max * 0.6 ? head.slice(0, space) : head).replace(/[\s,;:.-]+$/, '')}…`;
}

// ---------- inline maths ----------

export type InlinePart = { math: false; text: string } | { math: true; tex: string };

/** Index of the "$" closing inline maths opened at i, or -1. Like Pandoc (and app/export.py): a non-space must follow
 *  the opening "$", and the next unescaped "$" must close it: a non-space before it and no digit after it. */
function inlineEnd(s: string, i: number): number {
  const first = s[i + 1];
  if (first === undefined || first === '$' || /\s/.test(first)) return -1;
  let j = i + 1;
  while (j < s.length) {
    if (s[j] === '\\') {
      j += 2;
      continue;
    }
    if (s[j] === '$') return !/\s/.test(s[j - 1]) && !/[0-9]/.test(s[j + 1] ?? '') ? j : -1;
    j++;
  }
  return -1;
}

/**
 * Text with inline maths ("the slope $m$ of $y = mx + c$") as text and maths parts, read as the server writes it
 * (app/export.py): "costs $5 and $10" stays text, "\$" is a dollar sign, "$$...$$" is maths, and so is "$ x^2 $"
 * (LaTeX with spaces inside the dollars). An unclosed "$" is text.
 */
export function splitInline(s: string): InlinePart[] {
  const out: InlinePart[] = [];
  let text = '';
  const flush = () => {
    if (text) out.push({ math: false, text });
    text = '';
  };
  const maths = (tex: string) => {
    if (!tex.trim()) return;
    flush();
    out.push({ math: true, tex: tex.trim() });
  };
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === '\\' && s[i + 1] === '$') {
      text += '$';
      i += 2;
      continue;
    }
    if (c === '$') {
      if (s.startsWith('$$', i)) {
        const j = s.indexOf('$$', i + 2);
        if (j > i + 2) {
          maths(s.slice(i + 2, j));
          i = j + 2;
          continue;
        }
      }
      let j = inlineEnd(s, i);
      if (j < 0) {
        const k = s.indexOf('$', i + 1);
        const inner = k > i + 1 ? s.slice(i + 1, k) : '';
        j = inner.trim() && /[\\^_{}]/.test(inner) ? k : -1;
      }
      if (j > 0) {
        maths(s.slice(i + 1, j));
        i = j + 1;
        continue;
      }
    }
    text += c;
    i++;
  }
  flush();
  return out;
}

/** The text as it reads without maths typesetting: the LaTeX of each maths part stays as written. */
export const inlinePlain = (s: string) =>
  splitInline(s)
    .map((p) => (p.math ? p.tex : p.text))
    .join('');

/** Commands that show nothing of their own (delimiter sizes, fonts, accents, spacing) and escaped braces. */
const SILENT_TEX =
  /\\(?:left|right|[bB]igg?[lr]?|displaystyle|textstyle|scriptstyle|mathrm|mathit|mathbf|mathsf|mathtt|mathcal|mathbb|mathfrak|boldsymbol|operatorname|text|textrm|textit|textbf|quad|qquad|vec|hat|bar|overline|underline|tilde|dot|ddot|limits|nolimits)(?![A-Za-z])|\\[{}]/g;

/**
 * About how many characters inline maths shows once typeset. The server's limits count a formula as its Unicode
 * rendering ("$\alpha^{2}$" is 2; app/export.py shown_len), so this never counts more than that: a command is one
 * character; the commands above, braces, ^, _ and spaces none. LaTeX longer than a display formula counts in full.
 */
export function mathShown(tex: string): number {
  if (tex.length > LIMITS.latex) return tex.length;
  return tex
    .replace(SILENT_TEX, '')
    .replace(/\\[A-Za-z]+/g, 'x')
    .replace(/\\./g, 'x')
    .replace(/[{}^_\s]/g, '').length;
}

const partShown = (p: InlinePart) => (p.math ? mathShown(p.tex) : p.text.length);

/** Characters text with inline maths shows: its text, "\$" as one, each formula as mathShown. */
export const shownLength = (s: string) => splitInline(s).reduce((n, p) => n + partShown(p), 0);

/** At most `max` characters as shown (shownLength, as the server counts them) of text with inline maths: never cut
 *  inside a "$...$", "…" when cut. */
export function cutInline(s: string, max: number): string {
  const parts = splitInline(s);
  if (parts.reduce((n, p) => n + partShown(p), 0) <= max) return s;
  let out = '';
  let used = 0;
  for (const p of parts) {
    const n = partShown(p);
    if (used + n <= max - 1) {
      out += p.math ? `$${p.tex}$` : p.text.replace(/\$/g, '\\$');
      used += n;
      continue;
    }
    if (!p.math) out += cutPlain(p.text, max - used).replace(/…$/, '').replace(/\$/g, '\\$');
    break;
  }
  out = out.replace(/[\s,;:.\\-]+$/, ''); // a cut "\$" leaves no lone backslash
  // Nothing fits before a long formula: the words without typesetting, rather than a bare "…".
  return out ? `${out}…` : cutPlain(inlinePlain(s), max);
}

const inlineText = (v: unknown, max: number) => cutInline(oneLine(v), max);
const plainText = (v: unknown, max: number) => cutPlain(oneLine(v), max);

// ---------- cleaning what the server sends ----------

/** A figure's picture: "export/fig_NNN.png" inside the job folder, or '' when it is anything else. */
export function cleanFigurePath(v: unknown): string {
  if (typeof v !== 'string') return '';
  const p = v.trim().replace(/\\/g, '/');
  return /^export\/fig_\d{3,6}\.png$/.test(p) ? p : '';
}

/** LaTeX for a display formula: one line, '' when empty or too long to be one formula. */
function cleanLatex(v: unknown): string {
  const t = oneLine(v);
  return t.length <= LIMITS.latex ? t : '';
}

function cleanRefs(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  const refs = v.map(oneLine).filter((r) => /^#[A-Za-z0-9]{1,16}$/.test(r));
  return [...new Set(refs)].slice(0, LIMITS.refs);
}

function cleanBullets(o: Record<string, unknown>): BulletsBlock | null {
  if (!Array.isArray(o.items)) return null;
  const items: BulletItem[] = [];
  for (const x of o.items) {
    const it = typeof x === 'string' ? { text: x } : obj(x);
    if (!it) continue;
    const text = inlineText(it.text, LIMITS.bullet);
    if (text) items.push({ text, level: it.level === 1 ? 1 : 0 });
  }
  if (!items.length) return null;
  // A list cannot start indented.
  items[0].level = 0;
  return { type: 'bullets', items: items.slice(0, LIMITS.bullets) };
}

function cleanEquation(o: Record<string, unknown>): EquationBlock | null {
  const latex = cleanLatex(o.latex);
  if (!latex) return null;
  return { type: 'equation', latex, label: plainText(o.label, LIMITS.label), refs: cleanRefs(o.refs) };
}

function cleanSteps(o: Record<string, unknown>): StepsBlock | null {
  if (!Array.isArray(o.items)) return null;
  const items: StepItem[] = [];
  for (const x of o.items) {
    const it = typeof x === 'string' ? { text: x } : obj(x);
    if (!it) continue;
    const step = { latex: cleanLatex(it.latex), text: inlineText(it.text, LIMITS.stepText) };
    if (step.latex || step.text) items.push(step);
  }
  if (!items.length) return null;
  return { type: 'steps', label: plainText(o.label, LIMITS.stepsLabel), items: items.slice(0, LIMITS.steps) };
}

function cleanSize(v: unknown): [number, number] | null {
  if (!Array.isArray(v) || v.length < 2) return null;
  const [w, h] = v;
  return finite(w) && finite(h) && w > 0 && h > 0 ? [w, h] : null;
}

function cleanFigure(o: Record<string, unknown>): FigureBlock {
  return {
    type: 'figure',
    refs: cleanRefs(o.refs),
    caption: plainText(o.caption, LIMITS.caption),
    image: cleanFigurePath(o.image),
    size: cleanSize(o.size),
  };
}

function cleanTable(o: Record<string, unknown>): TableBlock | null {
  const cell = (v: unknown) => inlineText(typeof v === 'number' && Number.isFinite(v) ? String(v) : v, LIMITS.cell);
  const header = Array.isArray(o.header) ? o.header.map(cell) : [];
  const rows = (Array.isArray(o.rows) ? o.rows : [])
    .filter((r): r is unknown[] => Array.isArray(r))
    .map((r) => r.map(cell));
  const columns = Math.min(LIMITS.columns, Math.max(header.length, ...rows.map((r) => r.length), 0));
  if (!columns) return null;
  const fit = (r: string[]) => Array.from({ length: columns }, (_, i) => r[i] ?? '');
  const body = rows.map(fit).filter((r) => r.some(Boolean)).slice(0, LIMITS.rows);
  const head = header.some(Boolean) ? fit(header) : [];
  if (!body.length && !head.length) return null;
  return { type: 'table', header: head, rows: body, caption: plainText(o.caption, LIMITS.caption) };
}

function cleanCallout(o: Record<string, unknown>): CalloutBlock | null {
  const style = CALLOUT_STYLES.find((s) => s === o.style) ?? 'note';
  const block: CalloutBlock = {
    type: 'callout',
    style,
    title: plainText(o.title, LIMITS.calloutTitle),
    text: inlineText(o.text, LIMITS.calloutText),
    latex: cleanLatex(o.latex),
  };
  return block.title || block.text || block.latex ? block : null;
}

/** One block, or null for junk and block types this page does not know (a newer server's). */
export function cleanBlock(x: unknown): DeckBlock | null {
  const o = obj(x);
  if (!o) return null;
  switch (o.type) {
    case 'bullets':
      return cleanBullets(o);
    case 'equation':
      return cleanEquation(o);
    case 'steps':
      return cleanSteps(o);
    case 'figure':
      return cleanFigure(o);
    case 'table':
      return cleanTable(o);
    case 'callout':
      return cleanCallout(o);
    default:
      return null;
  }
}

/** The blocks a slide of this kind may show: none on title and section slides, only formulas on a formula sheet,
 *  else at most 3 with at most one figure, one table and one steps block (the first of each). */
export function slideBlocks(kind: SlideKind, raw: unknown): DeckBlock[] {
  if (kind === 'title' || kind === 'section' || !Array.isArray(raw)) return [];
  const blocks = raw.map(cleanBlock).filter((b): b is DeckBlock => b !== null);
  if (kind === 'formulas') return blocks.filter((b) => b.type === 'equation').slice(0, LIMITS.formulas);
  const once = new Set<string>();
  const out: DeckBlock[] = [];
  for (const b of blocks) {
    if (b.type === 'figure' || b.type === 'table' || b.type === 'steps') {
      if (once.has(b.type)) continue;
      once.add(b.type);
    }
    out.push(b);
    if (out.length === LIMITS.blocks) break;
  }
  return out;
}

function cleanSlide(x: unknown, i: number): DeckSlide | null {
  const o = obj(x);
  if (!o) return null;
  const kind = SLIDE_KINDS.find((k) => k === o.kind) ?? 'content';
  let t0 = time(o.t0);
  let t1 = time(o.t1);
  if (t0 !== null && t1 !== null && t1 < t0) [t0, t1] = [t1, t0];
  if (t0 === null) t1 = null;
  const section = plainText(o.section, LIMITS.section);
  const blocks = slideBlocks(kind, o.blocks);
  const fallback =
    kind === 'agenda' ? 'Agenda' : kind === 'summary' ? 'Key takeaways' : kind === 'formulas' ? 'Formula sheet' : '';
  return {
    id: oneLine(o.id).slice(0, 40) || `s${i + 1}`,
    kind,
    title: plainText(o.title, LIMITS.title) || fallback || section || `Slide ${i + 1}`,
    section: kind === 'content' || kind === 'section' ? section : '',
    t0,
    t1,
    // No Play when the server says none (a summary of the whole lecture); missing or junk: the slide's start.
    t: o.t === null ? null : (time(o.t) ?? t0),
    blocks,
    notes: prose(o.notes, LIMITS.notes),
    said: prose(o.said, LIMITS.said),
  };
}

const OLD_SERVER = 'This server still makes the old slides: restart `python -m app.server` with the latest code.';
const NOT_A_DECK = 'The slides came back damaged. Try reloading the page.';

/** The deck as the page uses it, from whatever the server sent; throws one plain line when it is not a deck. */
export function cleanDeck(x: unknown): Deck {
  const o = obj(x);
  if (o && !Array.isArray(o.slides) && Array.isArray(o.pages)) throw new Error(OLD_SERVER);
  if (!o || !Array.isArray(o.slides)) throw new Error(NOT_A_DECK);
  const slides = o.slides
    .slice(0, LIMITS.slides)
    .map(cleanSlide)
    .filter((s): s is DeckSlide => s !== null);
  // Ids key the slide frames: two slides with the same id would make React mix them up.
  const seen = new Set<string>();
  for (const s of slides) {
    let id = s.id;
    for (let n = 2; seen.has(id); n++) id = `${s.id}-${n}`;
    seen.add(id);
    s.id = id;
  }
  let clip: [number, number] | null = null;
  if (Array.isArray(o.clip) && o.clip.length >= 2) {
    const [a, b] = [time(o.clip[0]), time(o.clip[1])];
    if (a !== null && b !== null) clip = a <= b ? [a, b] : [b, a];
  }
  const ends = slides.map((s) => s.t1).filter((t): t is number => t !== null);
  const duration = time(o.duration) ?? (clip ? clip[1] - clip[0] : ends.length ? Math.max(...ends) : 0);
  const lecture = plainText(o.lecture, 120);
  const title = plainText(o.title, LIMITS.title) || lecture || 'Lecture';
  return {
    format: 2,
    version: oneLine(o.version),
    kind: o.kind === 'slides' ? 'slides' : 'board',
    title,
    subtitle: plainText(o.subtitle, LIMITS.subtitle),
    lecture: lecture || title,
    duration,
    clip: clip ?? [0, duration],
    slides,
    degraded: (Array.isArray(o.degraded) ? o.degraded : [])
      .map((d) => sentence(oneLine(d)))
      .filter(Boolean)
      .slice(0, LIMITS.degraded),
  };
}

const whole = (v: unknown) => (finite(v) && v >= 0 ? Math.floor(v) : 0);

/** The slides' state from GET slides.json or POST slides/write; throws one plain line when it is not one. */
export function cleanStatus(x: unknown): SlidesStatus {
  const o = obj(x);
  if (o && !('state' in o) && Array.isArray(o.pages)) throw new Error(OLD_SERVER);
  const state = STATES.find((s) => s === o?.state);
  if (!o || !state) throw new Error(NOT_A_DECK);
  let deck: Deck | null = null;
  if (o.deck != null) {
    try {
      deck = cleanDeck(o.deck);
    } catch (e) {
      // Slides finished so far are a bonus while writing; a finished deck that cannot be read is an error.
      if (state === 'ready') throw e;
    }
  }
  if (state === 'ready' && !deck) throw new Error(NOT_A_DECK);
  const total = whole(o.total);
  const error = state === 'failed' ? sentence(oneLine(o.error)) || 'Muse Spark could not write the slides.' : null;
  return { state, step: oneLine(o.step).slice(0, 120), done: total ? Math.min(whole(o.done), total) : whole(o.done), total, error, deck };
}

/** One plain line as the page shows it: the server's lines start in lower case ("this lecture is not processed
 *  yet, ..."), and a line never spreads over several. */
export function sentence(s: string): string {
  const t = s.replace(/\s+/g, ' ').trim();
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : t;
}

/** What the page says while the slides are being written: "Muse Spark is planning the slides…",
 *  "Writing part 2 of 5…". */
export function stepLine(s: Pick<SlidesStatus, 'state' | 'step'>): string {
  const step = s.step.replace(/[.…\s]+$/, '').trim();
  if (!step) return s.state === 'none' ? 'Muse Spark is starting on the slides…' : 'Muse Spark is writing the slides…';
  if (/^planning\b/i.test(step)) return `Muse Spark is ${step.charAt(0).toLowerCase()}${step.slice(1)}…`;
  return `${sentence(step)}…`;
}

/** How far the writing is, 0 to 1, or null when the server has not said (an indeterminate bar). */
export const progressShare = (s: Pick<SlidesStatus, 'done' | 'total'>) =>
  s.total > 0 ? Math.min(1, Math.max(0, s.done / s.total)) : null;

// ---------- what the frames show ----------

/** "1:23–4:56", or '' when the slide covers no part of the lecture. */
export const rangeLabel = (t0: number | null, t1: number | null) =>
  t0 === null ? '' : t1 === null || t1 <= t0 ? formatTime(t0) : `${formatTime(t0)}–${formatTime(t1)}`;

/** The player at a moment (whole seconds on the video's clock). */
export const playerAt = (jobId: string, t: number) =>
  `#/job/${encodeURIComponent(jobId)}?t=${Math.max(0, Math.floor(finite(t) ? t : 0))}`;

// The layout rules below follow the PowerPoint (app/pptx_deck.py), so the preview looks like the file it downloads.

export interface DeckSection {
  title: string;
  t0: number | null;
  t1: number | null;
  /** Index of its first slide. */
  first: number;
}

/** The deck's sections in order, from its section and content slides (a run of slides with the same section is one
 *  section; a section slide names its own). */
export function deckSections(slides: DeckSlide[]): DeckSection[] {
  const out: DeckSection[] = [];
  slides.forEach((s, i) => {
    if (s.kind !== 'content' && s.kind !== 'section') return;
    const name = s.section || (s.kind === 'section' ? s.title : '');
    if (!name) return;
    if (!out.length || out[out.length - 1].title !== name) out.push({ title: name, t0: null, t1: null, first: i });
    const cur = out[out.length - 1];
    if (s.t0 !== null) cur.t0 = cur.t0 === null ? s.t0 : Math.min(cur.t0, s.t0);
    if (s.t1 !== null) cur.t1 = cur.t1 === null ? s.t1 : Math.max(cur.t1, s.t1);
  });
  return out;
}

/** "Part 2" on a section slide, and the part of the lecture it covers (the section's when the slide has none). */
export function sectionFacts(slides: DeckSlide[], slide: DeckSlide): { part: number; range: string } {
  const name = slide.title || slide.section;
  const key = slide.section || name;
  const sections = deckSections(slides);
  const k = sections.findIndex((s) => s.title === key || s.title === name);
  const sec = k >= 0 ? sections[k] : null;
  return { part: k + 1, range: rangeLabel(slide.t0, slide.t1) || (sec ? rangeLabel(sec.t0, sec.t1) : '') };
}

export interface AgendaEntry {
  text: string;
  /** The part of the lecture it covers ('' when it is not one of the deck's sections). */
  range: string;
}

export interface AgendaLayout {
  entries: AgendaEntry[];
  columns: 1 | 2;
  /** Entries per column. */
  rows: number;
  /** Row height and the largest text size, in points. */
  rowHeight: number;
  size: number;
}

/** The agenda: the deck's sections with their times; a deck without sections lists what the slide lists. Two
 *  columns past 7 entries; rows share the body's height, at most 52 pt each. */
export function agendaLayout(slide: DeckSlide, slides: DeckSlide[]): AgendaLayout {
  const sections = deckSections(slides);
  const entries: AgendaEntry[] = sections.length
    ? sections.map((s) => ({ text: s.title, range: rangeLabel(s.t0, s.t1) }))
    : slide.blocks.flatMap((b) => (b.type === 'bullets' ? b.items.map((it) => ({ text: it.text, range: '' })) : []));
  const columns = entries.length > 7 ? 2 : 1;
  const rows = Math.max(1, Math.ceil(entries.length / columns));
  return { entries, columns, rows, rowHeight: Math.min(52, BODY_HEIGHT / rows), size: columns === 1 ? 22 : 18 };
}

/** The body of a content slide in points: below the title, above the footer. */
export const BODY_HEIGHT = 360;

export interface ContentLayout {
  /** stack: one column (a very wide figure goes in it, in its place); split: two columns; figure: a figure alone. */
  mode: 'stack' | 'split' | 'figure';
  main: DeckBlock[];
  /** The figure of a split or of a figure alone. */
  figure: FigureBlock | null;
  /** The table of a split without a figure (sideTable). */
  table: TableBlock | null;
  /** The side column's share of the width in a split. */
  side: number;
  /** The side column (the figure's or the table's) is the left one: its block came first. */
  figureFirst: boolean;
}

/** A table this tall and this narrow sits beside a slide's other blocks when the slide has no figure; any other
 *  table is stacked with them (app/pptx_deck.py, side_table). */
export const SIDE_TABLE = { rows: 4, columns: 3 } as const;

/** The table that goes in a column of its own beside the other blocks, or null. */
export function sideTable(blocks: DeckBlock[]): TableBlock | null {
  if (blocks.length < 2 || blocks.some((b) => b.type === 'figure')) return null;
  const t = blocks.find((b): b is TableBlock => b.type === 'table');
  if (!t) return null;
  const columns = t.header.length || Math.max(0, ...t.rows.map((r) => r.length));
  return t.rows.length >= SIDE_TABLE.rows && columns <= SIDE_TABLE.columns ? t : null;
}

/** A content slide: its blocks stacked in one column, or two columns when there is a figure: the figure in its own
 *  column on the side where it comes in the list, wider for a wide figure; a very wide figure stays in the stack. A
 *  tall, narrow table without a figure takes half the width beside the other blocks. */
export function contentLayout(blocks: DeckBlock[]): ContentLayout {
  const figure = (blocks.find((b) => b.type === 'figure') as FigureBlock | undefined) ?? null;
  const rest = blocks.filter((b) => b !== figure);
  const stack: ContentLayout = { mode: 'stack', main: blocks, figure: null, table: null, side: 0, figureFirst: false };
  const table = sideTable(blocks);
  if (table) {
    const main = blocks.filter((b) => b !== table);
    return { mode: 'split', main, figure: null, table, side: 0.5, figureFirst: blocks.indexOf(table) === 0 };
  }
  if (!figure) return stack;
  if (!rest.length) return { mode: 'figure', main: [], figure, table: null, side: 1, figureFirst: true };
  const aspect = figure.size ? figure.size[0] / figure.size[1] : 1;
  if (aspect >= 2) return stack;
  const side = aspect >= 1.3 ? 0.56 : aspect >= 0.85 ? 0.5 : 0.38;
  return { mode: 'split', main: rest, figure, table: null, side, figureFirst: blocks.indexOf(figure) === 0 };
}

/** The blocks as the frame draws them: formulas that follow one another share a row (side by side when each fits
 *  its half, as in the PowerPoint), and a summary's takeaways are numbered rows. */
export type FrameBlock = DeckBlock | { type: 'equations'; items: EquationBlock[] };

export function frameBlocks(kind: SlideKind, blocks: DeckBlock[]): FrameBlock[] {
  const out: FrameBlock[] = [];
  for (const b of blocks) {
    const shown: DeckBlock =
      kind === 'summary' && b.type === 'bullets'
        ? { type: 'steps', label: '', items: b.items.map((it) => ({ latex: '', text: it.text })) }
        : b;
    const last = out[out.length - 1];
    if (shown.type === 'equation' && last && (last.type === 'equation' || last.type === 'equations')) {
      out[out.length - 1] = { type: 'equations', items: [...(last.type === 'equations' ? last.items : [last]), shown] };
    } else out.push(shown);
  }
  return out;
}

const PLAIN_CMDS: Record<string, string> = {
  to: '→', rightarrow: '→', times: '×', cdot: '·', approx: '≈', neq: '≠', ne: '≠', le: '≤', leq: '≤', ge: '≥',
  geq: '≥', pm: '±', div: '÷',
};
const PLAIN_MATH = new RegExp(
  `^(?:\\s|\\d|[.,=+\\-<>()]|\\\\\\$|\\\\%|\\{,\\}|\\\\(?:${Object.keys(PLAIN_CMDS).join('|')})(?![A-Za-z]))+$`,
);

/** A formula of numbers only ("1331", "\$2.25", "1{,}000", "50\%", "1000 \to 1100") as text, else '': a step like
 *  that is set in the body font, like the words beside it (app/pptx_deck.py, plain_math). */
export function plainMath(latex: string): string {
  let s = latex.replace(/\s+/g, ' ').trim();
  if (!/\d/.test(s) || !PLAIN_MATH.test(s)) return '';
  s = s.replace(/\{,\}/g, ',').replace(/\\\$/g, '$').replace(/\\%/g, '%');
  s = s.replace(/\\([A-Za-z]+)/g, (_, c: string) => ` ${PLAIN_CMDS[c]} `);
  s = s.replace(/(?<=[\d)%])\s*-\s*/g, ' − ').replace(/-/g, '−'); // a minus between numbers, else a sign
  s = s.replace(/\s*([=<>+→≈≠≤≥×÷±·])\s*/g, ' $1 ');
  return s.replace(/\s+/g, ' ').trim();
}

/** Formula sheets: one column for one or two formulas, two for more; cards share the body's height, at most
 *  150 pt each, 18 pt apart. */
export function formulaGrid(n: number): { columns: 1 | 2; rows: number; cardHeight: number } {
  const columns = n <= 2 ? 1 : 2;
  const rows = Math.max(1, Math.ceil(n / columns));
  return { columns, rows, cardHeight: Math.min(150, (BODY_HEIGHT - 18 * (rows - 1)) / rows) };
}

/** "7:58" for the title slide. */
export const durationLabel = (seconds: number) => (seconds > 0 ? formatTime(seconds) : '');

/** The title slide's line about the lecture: "Board lecture · 7:58 · made with Replay Board from the recording". */
export const coverFacts = (deck: Pick<Deck, 'kind' | 'duration'>) =>
  [deck.kind === 'slides' ? 'Slide lecture' : 'Board lecture', durationLabel(deck.duration), 'made with Replay Board from the recording']
    .filter(Boolean)
    .join(' · ');

/** The section shown above a content slide's title, when the deck has at least two sections. */
export const eyebrow = (slide: DeckSlide, sectionCount: number) =>
  slide.kind === 'content' && sectionCount >= 2 ? slide.section : '';

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

// ---------- the PowerPoint download ----------

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
