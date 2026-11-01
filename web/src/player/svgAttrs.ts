// Attribute writes for the layers the shared clock animates: render(t) runs every frame, so each attribute is
// written only when its value changes.
const attrMemo = new WeakMap<Element, Map<string, string>>();

/** Write an attribute only when its value changes: the clock calls render every frame. */
export function setAttr(el: Element | null | undefined, name: string, value: string) {
  if (!el) return;
  let m = attrMemo.get(el);
  if (!m) attrMemo.set(el, (m = new Map()));
  if (m.get(name) === value) return;
  m.set(name, value);
  el.setAttribute(name, value);
}

/** An opacity: exactly 0 or 1 at the ends, else three decimals. */
export const op = (v: number) => (v <= 0.001 ? '0' : v >= 0.999 ? '1' : v.toFixed(3));
/** A coordinate, to a tenth of a pixel. */
export const n1 = (v: number) => (Math.round(v * 10) / 10).toString();
