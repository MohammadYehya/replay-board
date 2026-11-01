// Line layouts for a typeset reading. A teacher writes "x = -1, log(-(-1))" and then "= log 1" and "= 0"
// underneath; KaTeX left to wrap on its own breaks wherever the box runs out ("log(− −" / "1) = ..."). So we
// offer a few layouts that break only at top-level separators and relations, and the typeset layer keeps the
// one that fits the piece's box at the largest size. Pure string work: no DOM, so `npm test` can run it.

/** One line of a layout. `cont` lines start with a relation that continues the chain on the line above. */
export interface Line {
  tex: string;
  cont: boolean;
}
export type Layout = Line[];

interface Mark {
  kind: 'sep' | 'rel';
  start: number;
  end: number;
  keep: string; // punctuation kept at the end of the line before a separator
}

const REL_CMDS = new Set([
  'approx', 'equiv', 'neq', 'ne', 'le', 'leq', 'ge', 'geq', 'Rightarrow', 'implies', 'iff', 'Leftrightarrow',
]);
const SEP_CMDS = new Set(['\\', 'quad', 'qquad', 'newline']);
const isLetter = (c: string) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
const isDigit = (c: string) => c >= '0' && c <= '9';

/** Separators and relations that sit outside every group, bracket, \left..\right and environment. */
function marks(s: string): Mark[] {
  const out: Mark[] = [];
  const n = s.length;
  let depth = 0;
  const readCmd = (j: number): [string, number] => {
    let k = j + 1;
    if (k < n && isLetter(s[k])) {
      while (k < n && isLetter(s[k])) k++;
      return [s.slice(j + 1, k), k];
    }
    return [s.slice(j + 1, k + 1), Math.min(n, k + 1)];
  };
  const skipDelim = (j: number): number => {
    while (j < n && s[j] === ' ') j++;
    if (j >= n) return j;
    return s[j] === '\\' ? readCmd(j)[1] : j + 1;
  };
  let i = 0;
  while (i < n) {
    const c = s[i];
    if (c === '\\') {
      const [name, next] = readCmd(i);
      if (name === 'left') {
        depth++;
        i = skipDelim(next);
        continue;
      }
      if (name === 'right') {
        depth = Math.max(0, depth - 1);
        i = skipDelim(next);
        continue;
      }
      if (name === 'begin' || name === '{') depth++;
      else if (name === 'end' || name === '}') depth = Math.max(0, depth - 1);
      else if (depth === 0 && SEP_CMDS.has(name)) out.push({ kind: 'sep', start: i, end: next, keep: '' });
      else if (depth === 0 && REL_CMDS.has(name)) out.push({ kind: 'rel', start: i, end: next, keep: '' });
      i = next;
      continue;
    }
    if (c === '{' || c === '(' || c === '[') depth++;
    else if (c === '}' || c === ')' || c === ']') depth = Math.max(0, depth - 1);
    else if (depth === 0) {
      if (c === '=') out.push({ kind: 'rel', start: i, end: i + 1, keep: '' });
      else if (c === ';') out.push({ kind: 'sep', start: i, end: i + 1, keep: ';' });
      else if (c === ',') {
        // "1,000" is a number, not a list.
        const prev = s.slice(0, i).trimEnd().slice(-1);
        const nxt = s.slice(i + 1).trimStart().charAt(0);
        if (!(isDigit(prev) && isDigit(nxt))) out.push({ kind: 'sep', start: i, end: i + 1, keep: ',' });
      } else if (c === '.') {
        // "3. \quad ..." ends a sentence; "2.718" and "..." do not.
        const nxt = s.charAt(i + 1);
        if (nxt === ' ' || nxt === '\\') out.push({ kind: 'sep', start: i, end: i + 1, keep: '.' });
      }
    }
    i++;
  }
  return out;
}

const junk = (t: string) => !t || /^[\s,.;]*$/.test(t);

/** Split at top-level commas, sentence stops, \quad and \\ (the punctuation stays on its line). */
export function clauses(s: string): string[] {
  const out: string[] = [];
  let from = 0;
  for (const m of marks(s)) {
    if (m.kind !== 'sep') continue;
    out.push((s.slice(from, m.start) + m.keep).trim());
    from = m.end;
  }
  out.push(s.slice(from).trim());
  return out.filter((t) => !junk(t));
}

/** Split a chain "a = b = c" into "a = b" / "= c": a new line before every top-level relation but the first. */
export function relationLines(seg: string): Line[] {
  const rels = marks(seg).filter((m) => m.kind === 'rel');
  const cuts: number[] = [];
  for (let k = 1; k < rels.length; k++) if (!junk(seg.slice(rels[k].end))) cuts.push(rels[k].start);
  const lines: Line[] = [];
  let from = 0;
  for (const c of cuts) {
    lines.push({ tex: seg.slice(from, c).trim(), cont: lines.length > 0 });
    from = c;
  }
  lines.push({ tex: seg.slice(from).trim(), cont: lines.length > 0 });
  return lines.filter((l) => !junk(l.tex));
}

/** Distinct candidate layouts, the single line first. */
export function texLayouts(latex: string): Layout[] {
  const s = latex.trim();
  if (!s) return [];
  const seen = new Set<string>();
  const out: Layout[] = [];
  const add = (l: Layout) => {
    if (!l.length) return;
    const key = l.map((x) => (x.cont ? '+' : '|') + x.tex).join('\n');
    if (seen.has(key)) return;
    seen.add(key);
    out.push(l);
  };
  add([{ tex: s, cont: false }]);
  const cl = clauses(s);
  if (cl.length > 1) add(cl.map((tex) => ({ tex, cont: false })));
  add(relationLines(s));
  add(cl.flatMap(relationLines));
  return out;
}

/** Plain readings sometimes describe the marks ("16/2/15 with underline", "circled 1!"); keep only the writing. */
export function writtenText(text: string): string {
  return text
    .trim()
    .replace(/^(circled|boxed|underlined)\s+/i, '')
    .replace(/\s*[(,]?\s*(with (an? )?(double )?underline|underlined( twice)?|circled|boxed|crossed out)\)?\s*$/i, '')
    .trim();
}
