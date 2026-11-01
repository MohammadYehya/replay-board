/** Readings sometimes wrap LaTeX in $...$ or use a bare $ for money; KaTeX wants neither. */
export function cleanLatex(s: string): string {
  let t = s.trim();
  const m =
    t.match(/^\$\$([\s\S]*)\$\$$/) ??
    t.match(/^\$([\s\S]*)\$$/) ??
    t.match(/^\\\(([\s\S]*)\\\)$/) ??
    t.match(/^\\\[([\s\S]*)\\\]$/);
  if (m) t = m[1].trim();
  return t.replace(/(?<!\\)\$/g, '\\$');
}
