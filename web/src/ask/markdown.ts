// The small Markdown subset Muse Spark answers in, parsed into a tree the player renders itself (ask/render.ts):
// paragraphs, "- " and "1. " lists (one level of nesting), **bold**, *italic*, `code` (and ``` code blocks, which
// models write now and then), maths in $...$, $$...$$, \(...\) or \[...\], times [m:ss] / [h:mm:ss] on the video's
// clock, and refs [#12], [#L3], [#s0e1].
// Nothing in the answer is ever treated as HTML: every character the model writes ends up in a text node, except
// the LaTeX, which only KaTeX reads. While an answer streams in, half-written maths is kept aside as "pending" text
// instead of flashing as a broken formula.
// No DOM and no imports, so `npm test` can run it with plain Node.

export type Inline =
  | { t: 'text'; v: string }
  | { t: 'br' }
  | { t: 'b'; c: Inline[] }
  | { t: 'i'; c: Inline[] }
  | { t: 'code'; v: string }
  | { t: 'math'; tex: string; display: boolean }
  | { t: 'time'; s: number; raw: string }
  | { t: 'ref'; ref: string; raw: string }
  /** Maths still being written (streaming only): shown dimmed, as plain text. */
  | { t: 'pending'; v: string };

export interface ListItem {
  c: Inline[];
  sub: Block[];
}

export type Block =
  | { t: 'p'; c: Inline[] }
  | { t: 'h'; c: Inline[] }
  | { t: 'ul'; items: ListItem[] }
  | { t: 'ol'; start: number; items: ListItem[] }
  | { t: 'math'; tex: string }
  /** A fenced code block (``` or ~~~): shown as it is, nothing inside is parsed. */
  | { t: 'pre'; v: string }
  | { t: 'pending'; v: string };

export interface ParseOptions {
  /** The text may end mid-way (an answer still streaming in). */
  streaming?: boolean;
}

/** Seconds for "m:ss" or "h:mm:ss", or null. */
export function parseClock(s: string): number | null {
  const m = /^(\d{1,3}):([0-5]\d)(?::([0-5]\d))?$/.exec(s.trim());
  if (!m) return null;
  return m[3] != null ? Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]) : Number(m[1]) * 60 + Number(m[2]);
}

const REF = /^#(?:\d{1,6}|L\d{1,6}|s\d{1,5}e\d{1,5})$/;
const CLOCK = String.raw`\d{1,3}:[0-5]\d(?::[0-5]\d)?`;
/** [1:02], [1:02:03], [1:02–1:30] (a range: two links). */
const TIME_RE = new RegExp(String.raw`^\[(${CLOCK})(?:\s*(?:-|–|—|to)\s*(${CLOCK}))?\]`);
/** [#12], [#L3], [#s0e1], or several: [#12, #L3]. */
const REFS_RE = /^\[(#[A-Za-z0-9]+(?:\s*[,;]\s*#[A-Za-z0-9]+)*)\]/;
const ESCAPABLE = '\\`*_$[]#{}()!.-+';

const isSpace = (c: string | undefined) => c === undefined || /\s/.test(c);
const isWordChar = (c: string | undefined) => c !== undefined && /[\p{L}\p{N}]/u.test(c);

/** The code span opened by the run of backticks at i: it ends at the next run of exactly as many ("``a`b``" is
 *  the code "a`b"). Null when it is never closed. */
function codeSpan(s: string, i: number): { code: string; end: number; run: number } | null {
  let k = i;
  while (s[k] === '`') k++;
  const run = k - i;
  for (let j = k; j < s.length; ) {
    const q = s.indexOf('`', j);
    if (q < 0) break;
    let e = q;
    while (s[e] === '`') e++;
    if (e - q === run) {
      let code = s.slice(k, q);
      if (code.length > 2 && code.startsWith(' ') && code.endsWith(' ') && code.trim()) code = code.slice(1, -1);
      return code ? { code, end: e, run } : null;
    }
    j = e;
  }
  return null;
}

/** Index of the "$" closing inline maths opened at i, or -1 (pandoc's rule: "$5 and $10" is not maths). The next
 *  unescaped "$" must close it: a formula never holds a bare "$", so in "Cost is $3, not $x$" the "$3" is money. */
function inlineMathEnd(s: string, i: number): number {
  if (isSpace(s[i + 1]) || s[i + 1] === '$') return -1;
  for (let j = i + 1; j < s.length; j++) {
    const c = s[j];
    if (c === '\\') {
      j++;
      continue;
    }
    if (c === '\n' && s[j + 1] === '\n') return -1;
    if (c === '$') return !isSpace(s[j - 1]) && !/\d/.test(s[j + 1] ?? '') ? j : -1;
  }
  return -1;
}

/** Index of the closing delimiter `d` after `from` that ends emphasis, or -1. */
function emphasisEnd(s: string, from: number, d: string): number {
  for (let j = from; j < s.length; j++) {
    const c = s[j];
    if (c === '\\') {
      j++;
      continue;
    }
    if (c === '`') {
      const cs = codeSpan(s, j);
      if (cs) j = cs.end - 1;
      else while (s[j + 1] === '`') j++;
      continue;
    }
    if (c === '$') {
      const k = s[j + 1] === '$' ? s.indexOf('$$', j + 2) : inlineMathEnd(s, j);
      if (k > 0) {
        j = s[j + 1] === '$' ? k + 1 : k;
        continue;
      }
    }
    if (s.startsWith(d, j) && !isSpace(s[j - 1])) {
      if (d === '*' && s[j + 1] === '*') {
        // "**" inside "*...*" is bold: skip over it whole
        const k = emphasisEnd(s, j + 2, '**');
        if (k > 0) {
          j = k + 1;
          continue;
        }
      }
      if (d === '_' && isWordChar(s[j + 1])) continue; // snake_case
      return j;
    }
  }
  return -1;
}

/** Looks like the start of a formula (not a price): for showing half-written maths as pending. */
const looksLikeTex = (tail: string) => tail.length <= 3 || /[\\^_{}=]/.test(tail);

export function parseInline(src: string, opts: ParseOptions = {}): Inline[] {
  const out: Inline[] = [];
  let buf = '';
  const flush = () => {
    if (buf) out.push({ t: 'text', v: buf });
    buf = '';
  };
  const s = src;
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === '\\') {
      const nx = s[i + 1];
      if (nx === '(' || nx === '[') {
        const close = nx === '(' ? '\\)' : '\\]';
        const j = s.indexOf(close, i + 2);
        if (j >= 0) {
          flush();
          out.push({ t: 'math', tex: s.slice(i + 2, j).trim(), display: nx === '[' });
          i = j + 2;
          continue;
        }
        if (opts.streaming) {
          flush();
          out.push({ t: 'pending', v: s.slice(i) });
          return out;
        }
      }
      if (nx !== undefined && ESCAPABLE.includes(nx)) {
        buf += nx;
        i += 2;
        continue;
      }
      buf += c;
      i++;
      continue;
    }
    if (c === '`') {
      const cs = codeSpan(s, i);
      if (cs) {
        flush();
        out.push({ t: 'code', v: cs.code });
        i = cs.end;
        continue;
      }
      // never closed: the backticks are text (a whole run, so "``" cannot close a later "`")
      let k = i;
      while (s[k] === '`') k++;
      buf += s.slice(i, k);
      i = k;
      continue;
    }
    if (c === '$') {
      if (s[i + 1] === '$') {
        const j = s.indexOf('$$', i + 2);
        if (j >= 0 && s.slice(i + 2, j).trim()) {
          flush();
          out.push({ t: 'math', tex: s.slice(i + 2, j).trim(), display: true });
          i = j + 2;
          continue;
        }
        if (j < 0 && opts.streaming) {
          flush();
          out.push({ t: 'pending', v: s.slice(i) });
          return out;
        }
        buf += '$$';
        i += 2;
        continue;
      }
      const j = inlineMathEnd(s, i);
      if (j > i + 1) {
        flush();
        out.push({ t: 'math', tex: s.slice(i + 1, j), display: false });
        i = j + 1;
        continue;
      }
      const tail = s.slice(i + 1);
      if (opts.streaming && !isSpace(s[i + 1]) && !tail.includes('\n') && looksLikeTex(tail)) {
        flush();
        out.push({ t: 'pending', v: s.slice(i) });
        return out;
      }
      buf += c;
      i++;
      continue;
    }
    if (c === '[') {
      const rest = s.slice(i);
      const tm = TIME_RE.exec(rest);
      if (tm) {
        const a = parseClock(tm[1]);
        const b = tm[2] ? parseClock(tm[2]) : null;
        if (a != null) {
          flush();
          out.push({ t: 'time', s: a, raw: tm[1] });
          if (b != null) {
            out.push({ t: 'text', v: '–' });
            out.push({ t: 'time', s: b, raw: tm[2] });
          }
          i += tm[0].length;
          continue;
        }
      }
      const rm = REFS_RE.exec(rest);
      if (rm) {
        const refs = rm[1].split(/\s*[,;]\s*/);
        if (refs.every((r) => REF.test(r))) {
          flush();
          refs.forEach((r, k) => {
            if (k) out.push({ t: 'text', v: ' ' });
            out.push({ t: 'ref', ref: r, raw: `[${r}]` });
          });
          i += rm[0].length;
          continue;
        }
      }
      buf += c;
      i++;
      continue;
    }
    if (c === '*' || c === '_') {
      const strong = s[i + 1] === c;
      const d = strong ? c + c : c;
      const from = i + d.length;
      const okOpen = !isSpace(s[from]) && !(c === '_' && isWordChar(s[i - 1]));
      if (okOpen) {
        const j = emphasisEnd(s, from, d);
        if (j > from) {
          flush();
          const inner = parseInline(s.slice(from, j), { streaming: false });
          out.push(strong ? { t: 'b', c: inner } : { t: 'i', c: inner });
          i = j + d.length;
          continue;
        }
        // still streaming: the closing marker may be on its way; show the words without the marker
        if (opts.streaming && !s.slice(from).includes('\n')) {
          flush();
          const inner = parseInline(s.slice(from), opts);
          out.push(strong ? { t: 'b', c: inner } : { t: 'i', c: inner });
          return out;
        }
      }
      buf += d;
      i += d.length;
      continue;
    }
    if (c === '\n') {
      flush();
      out.push({ t: 'br' });
      i++;
      continue;
    }
    buf += c;
    i++;
  }
  flush();
  // a trailing line break adds nothing
  while (out.length && out[out.length - 1].t === 'br') out.pop();
  return out;
}

interface ListLine {
  type: 'ul' | 'ol';
  indent: number;
  num: number;
  text: string;
}

function listLine(line: string): ListLine | null {
  let m = /^(\s*)[-*+•]\s+(.*)$/.exec(line);
  if (m) return { type: 'ul', indent: m[1].replace(/\t/g, '    ').length, num: 0, text: m[2] };
  m = /^(\s*)(\d{1,4})[.)]\s+(.*)$/.exec(line);
  if (m) return { type: 'ol', indent: m[1].replace(/\t/g, '    ').length, num: Number(m[2]), text: m[3] };
  return null;
}

const blank = (l: string) => !l.trim();
const indentOf = (l: string) => (/^\s*/.exec(l)?.[0] ?? '').replace(/\t/g, '    ').length;
const HEADING = /^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$/;
const MATH_OPEN = /^\s*(\$\$|\\\[)(.*)$/;
/** ```python or ~~~ opens a code block (a ``` fence's info string has no backticks, or it is inline code). */
const FENCE_OPEN = /^\s{0,3}(`{3,}|~{3,})([^`]*)$/;
const FENCE_CLOSE = /^\s{0,3}(`{3,}|~{3,})\s*$/;

export function parseMarkdown(src: string, opts: ParseOptions = {}): Block[] {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  return parseLines(lines, opts, true);
}

/** `atEnd`: these lines run to the end of the answer (only there can something be half-written). */
function parseLines(lines: string[], opts: ParseOptions, atEnd: boolean): Block[] {
  const out: Block[] = [];
  const n = lines.length;
  let i = 0;
  const done: ParseOptions = { ...opts, streaming: false };
  /** Options for text that ends before line `to`: streaming tolerance only for the text the answer ends with. */
  const upTo = (to: number): ParseOptions => (atEnd && lines.slice(to).every(blank) ? opts : done);
  const startsBlock = (l: string) => {
    if (HEADING.test(l) || FENCE_OPEN.test(l)) return true;
    const ll = listLine(l);
    if (ll && ll.indent <= 3) return true;
    const mm = MATH_OPEN.exec(l);
    return !!mm && !mm[2].includes(mm[1] === '$$' ? '$$' : '\\]');
  };
  while (i < n) {
    const line = lines[i];
    if (blank(line)) {
      i++;
      continue;
    }
    // a fenced code block, up to a fence of the same kind at least as long (or the end: it may still be streaming)
    const fm = FENCE_OPEN.exec(line);
    if (fm) {
      const body: string[] = [];
      let j = i + 1;
      for (; j < n; j++) {
        const cm = FENCE_CLOSE.exec(lines[j]);
        if (cm && cm[1][0] === fm[1][0] && cm[1].length >= fm[1].length) break;
        body.push(lines[j]);
      }
      out.push({ t: 'pre', v: body.join('\n') });
      i = j + 1;
      continue;
    }
    // $$ ... $$ (or \[ ... \]) over several lines
    const mm = MATH_OPEN.exec(line);
    if (mm) {
      const close = mm[1] === '$$' ? '$$' : '\\]';
      const k = mm[2].indexOf(close);
      if (k >= 0 && !mm[2].slice(k + close.length).trim()) {
        const tex = mm[2].slice(0, k).trim();
        if (tex) out.push({ t: 'math', tex });
        i++;
        continue;
      }
      if (k < 0) {
        const buf = [mm[2]];
        let j = i + 1;
        let after: string | null = null;
        for (; j < n; j++) {
          const kk = lines[j].indexOf(close);
          if (kk >= 0) {
            buf.push(lines[j].slice(0, kk));
            after = lines[j].slice(kk + close.length).trim();
            break;
          }
          buf.push(lines[j]);
        }
        if (j < n) {
          const tex = buf.join('\n').trim();
          if (tex) out.push({ t: 'math', tex });
          if (after) out.push({ t: 'p', c: parseInline(after, upTo(j + 1)) });
          i = j + 1;
          continue;
        }
        if (opts.streaming && atEnd) {
          out.push({ t: 'pending', v: lines.slice(i).join('\n') });
          return out;
        }
        // never closed: plain text below
      }
    }
    const hm = HEADING.exec(line);
    if (hm) {
      out.push({ t: 'h', c: parseInline(hm[1], upTo(i + 1)) });
      i++;
      continue;
    }
    const first = listLine(line);
    if (first && first.indent <= 3) {
      const items: ListItem[] = [];
      const base = first.indent;
      while (i < n) {
        const ll = listLine(lines[i]);
        if (!ll || ll.type !== first.type || ll.indent > base + 1 || ll.indent < base) break;
        // the item: its first line, its continuation lines, and anything indented under it (a nested list)
        const own = [ll.text];
        const nested: string[] = [];
        let j = i + 1;
        for (; j < n; j++) {
          const l = lines[j];
          if (blank(l)) {
            // a blank line ends the item unless more of it (indented) or the next item follows
            let k = j + 1;
            while (k < n && blank(lines[k])) k++;
            if (k < n && indentOf(lines[k]) > base + 1) {
              nested.push('');
              continue;
            }
            break;
          }
          const sub = listLine(l);
          if (indentOf(l) > base + 1) {
            if (sub || nested.length) nested.push(l);
            else own.push(l.trim());
            continue;
          }
          if (sub || startsBlock(l)) break;
          if (nested.length) break;
          own.push(l.trim()); // a lazy continuation line
        }
        const tail = upTo(j);
        const subBlocks = nested.length ? parseLines(dedent(nested), opts, tail === opts) : [];
        items.push({ c: parseInline(own.join('\n'), nested.length ? done : tail), sub: subBlocks });
        i = j;
        // blank lines between items of the same list
        let k = i;
        while (k < n && blank(lines[k])) k++;
        const next = k < n ? listLine(lines[k]) : null;
        if (next && next.type === first.type && next.indent >= base && next.indent <= base + 1) i = k;
        else break;
      }
      out.push(first.type === 'ul' ? { t: 'ul', items } : { t: 'ol', start: first.num, items });
      continue;
    }
    // a paragraph: up to a blank line or the start of another block
    const buf = [line];
    i++;
    while (i < n && !blank(lines[i]) && !startsBlock(lines[i])) buf.push(lines[i++]);
    out.push({ t: 'p', c: parseInline(buf.join('\n'), upTo(i)) });
  }
  return out;
}

function dedent(lines: string[]): string[] {
  const ind = Math.min(...lines.filter((l) => !blank(l)).map(indentOf));
  return lines.map((l) => {
    let k = 0;
    let w = 0;
    while (k < l.length && w < ind && (l[k] === ' ' || l[k] === '\t')) {
      w += l[k] === '\t' ? 4 : 1;
      k++;
    }
    return l.slice(k);
  });
}

/** Every time cited in a parsed answer, in order (for tests and for checking the server's list). */
export function citedTimes(blocks: Block[]): number[] {
  const out: number[] = [];
  const walk = (xs: Inline[]) => {
    for (const x of xs) {
      if (x.t === 'time') out.push(x.s);
      else if (x.t === 'b' || x.t === 'i') walk(x.c);
    }
  };
  const block = (b: Block) => {
    if (b.t === 'p' || b.t === 'h') walk(b.c);
    else if (b.t === 'ul' || b.t === 'ol')
      for (const it of b.items) {
        walk(it.c);
        it.sub.forEach(block);
      }
  };
  blocks.forEach(block);
  return out;
}
