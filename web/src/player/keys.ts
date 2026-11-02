// Which keys a focused field keeps for itself. The player's keys (space, arrows, Esc...) and the 3D view's [ ] are
// left to a field that takes text; a "Try it" slider keeps only the keys that move it, so Esc still goes back to the
// board and [ ] still go through the steps while it has the focus. No DOM: it runs in Node tests.

/** The keys a range input moves its value with. */
const SLIDER_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']);

/** The parts of a focused element that say which keys it takes. */
export interface KeyTarget {
  tagName?: string;
  type?: string;
  isContentEditable?: boolean;
}

/** Whether the focused element `el` keeps `key` (a KeyboardEvent.key) for itself. */
export function fieldTakesKey(el: KeyTarget | null | undefined, key: string): boolean {
  if (!el) return false;
  if (el.isContentEditable) return true;
  const tag = (el.tagName ?? '').toUpperCase();
  if (tag === 'INPUT' && (el.type ?? '').toLowerCase() === 'range') return SLIDER_KEYS.has(key);
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}
