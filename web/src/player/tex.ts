import katex from 'katex';
import { cleanLatex } from './texClean';

const memo = new Map<string, string | null>();

/** KaTeX HTML for a LaTeX string, or null when KaTeX cannot parse it (the caller shows plain text). */
export function texHtml(latex: string, display = false): string | null {
  const key = `${display ? 'D' : 'I'}${latex}`;
  let html = memo.get(key);
  if (html === undefined) {
    try {
      html = katex.renderToString(cleanLatex(latex), {
        throwOnError: true,
        displayMode: display,
        output: 'html',
        strict: 'ignore',
      });
    } catch {
      html = null;
    }
    memo.set(key, html);
  }
  return html;
}
