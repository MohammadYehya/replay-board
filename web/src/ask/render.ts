// Renders a parsed answer (ask/markdown.ts) as React elements. Text always goes into text nodes (React escapes it);
// the only raw HTML is KaTeX's own output for the maths, made with trust off, so a link or a script in the
// answer can never become one. Times become buttons that seek the video, refs become chips that show the item.
// Written with createElement (no JSX) so `npm test` can render it with plain Node.
import { createElement as h, type ReactNode } from 'react';
import katex from 'katex';
import type { Block, Inline, ListItem } from './markdown.ts';

export interface RenderContext {
  /** Whether a cited time is a moment of the video (it becomes a link; otherwise it stays text). */
  timeOk(s: number): boolean;
  /** The chip for a ref, or null when the answer cites something that is not in the lecture (it stays text). With
   *  LaTeX the chip shows the item typeset, else its label. */
  refInfo(ref: string): { label: string; t: number; latex?: string } | null;
  onTime(s: number): void;
  onRef(ref: string): void;
  clock(s: number): string;
}

const cache = new Map<string, string>();

/** KaTeX HTML for a formula ('' if KaTeX fails outright; a parse error is drawn by KaTeX in red instead). */
export function mathHtml(tex: string, display: boolean): string {
  const key = `${display ? 'D' : 'I'}${tex}`;
  let html = cache.get(key);
  if (html === undefined) {
    try {
      html = katex.renderToString(tex, {
        throwOnError: false,
        displayMode: display,
        output: 'html',
        strict: 'ignore',
        trust: false,
        maxSize: 30,
        maxExpand: 300,
      });
    } catch {
      html = '';
    }
    if (cache.size > 500) cache.clear();
    cache.set(key, html);
  }
  return html;
}

function inline(x: Inline, ctx: RenderContext, key: string): ReactNode {
  switch (x.t) {
    case 'text':
      return x.v;
    case 'br':
      return h('br', { key });
    case 'b':
      return h('strong', { key }, ...inlines(x.c, ctx, key));
    case 'i':
      return h('em', { key }, ...inlines(x.c, ctx, key));
    case 'code':
      return h('code', { key, className: 'md-code' }, x.v);
    case 'math': {
      const html = mathHtml(x.tex, x.display);
      if (!html) return h('code', { key, className: 'md-code' }, x.tex);
      return h('span', { key, className: x.display ? 'md-dmath' : 'md-math', dangerouslySetInnerHTML: { __html: html } });
    }
    case 'time':
      if (!ctx.timeOk(x.s)) return `[${x.raw}]`;
      return h(
        'button',
        { key, type: 'button', className: 'md-time', title: `Go to ${x.raw} in the lecture`, onClick: () => ctx.onTime(x.s) },
        x.raw,
      );
    case 'ref': {
      const info = ctx.refInfo(x.ref);
      if (!info) return x.raw;
      return h(
        'button',
        {
          key,
          type: 'button',
          className: 'md-ref',
          'data-ref': x.ref,
          title: `${info.label} · ${ctx.clock(info.t)}`,
          'aria-label': `Show ${info.label} on the board, at ${ctx.clock(info.t)}`,
          onClick: () => ctx.onRef(x.ref),
        },
        chipLabel(info),
      );
    }
    case 'pending':
      return h('span', { key, className: 'md-pending' }, x.v);
  }
}

/** A ref chip's face: the item's formula typeset when it has one (the board's raw text, like "A=P(1+r)^n", reads
 *  badly next to typeset maths), else its plain label. */
function chipLabel(info: { label: string; latex?: string }): ReactNode {
  const tex = (info.latex ?? '').trim().replace(/^\$+|\$+$/g, '').trim();
  const html = tex ? mathHtml(tex, false) : '';
  if (html) return h('span', { className: 'md-ref-label md-ref-math', dangerouslySetInnerHTML: { __html: html } });
  return h('span', { className: 'md-ref-label' }, info.label);
}

const isDisplay = (x: Inline | undefined) => x?.t === 'math' && x.display;

// $$...$$ inside a paragraph or a list item is drawn as a block of its own, so the line breaks around it would add
// an empty line under the formula: they are left out.
const inlines = (xs: Inline[], ctx: RenderContext, key: string) =>
  xs.flatMap((x, k) =>
    x.t === 'br' && (isDisplay(xs[k - 1]) || isDisplay(xs[k + 1])) ? [] : [inline(x, ctx, `${key}.${k}`)],
  );

function item(it: ListItem, ctx: RenderContext, key: string): ReactNode {
  return h('li', { key }, ...inlines(it.c, ctx, key), ...renderBlocks(it.sub, ctx, `${key}s`));
}

function block(b: Block, ctx: RenderContext, key: string): ReactNode {
  switch (b.t) {
    case 'p':
      return h('p', { key, className: 'md-p' }, ...inlines(b.c, ctx, key));
    case 'h':
      return h('p', { key, className: 'md-p md-h' }, h('strong', null, ...inlines(b.c, ctx, key)));
    case 'ul':
      return h('ul', { key, className: 'md-list' }, ...b.items.map((it, k) => item(it, ctx, `${key}.${k}`)));
    case 'ol':
      return h(
        'ol',
        { key, className: 'md-list', start: b.start },
        ...b.items.map((it, k) => item(it, ctx, `${key}.${k}`)),
      );
    case 'math': {
      const html = mathHtml(b.tex, true);
      if (!html) return h('pre', { key, className: 'md-code' }, b.tex);
      return h('div', { key, className: 'md-dmath', dangerouslySetInnerHTML: { __html: html } });
    }
    case 'pre':
      return h('pre', { key, className: 'md-code md-pre' }, h('code', null, b.v));
    case 'pending':
      return h('p', { key, className: 'md-p md-pending' }, b.v);
  }
}

export function renderBlocks(blocks: Block[], ctx: RenderContext, key = 'b'): ReactNode[] {
  return blocks.map((b, k) => block(b, ctx, `${key}${k}`));
}
