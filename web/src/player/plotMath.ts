// Pure helpers for the 3D pop-out: sampling y = f(x) into drawable segments, axis ticks and ink colours.
// No three.js here, so they can be unit-tested in Node.

export type Pt = [number, number];

/** Sample y = f(x) over [xmin, xmax] into continuous segments inside the y window.
 *  Segments end where the curve leaves the window (found by bisection, so an asymptote such as ln|x| at 0 runs
 *  right to the edge), where f is undefined (NaN / ±Infinity), and at jumps (floor, tan's poles) that stay big
 *  however closely they are bisected. Steep parts are refined until neighbouring points are close on screen. */
export function sampleSegments(
  f: (x: number) => number,
  xmin: number,
  xmax: number,
  ymin: number,
  ymax: number,
  n = 1200,
): Pt[][] {
  const xr = xmax - xmin;
  const yr = ymax - ymin;
  if (!(xr > 0) || !(yr > 0) || !Number.isFinite(xr) || !Number.isFinite(yr)) return [];
  const m = 0.02 * yr;
  const lo = ymin - m;
  const hi = ymax + m;
  const inside = (v: number) => Number.isFinite(v) && v >= lo && v <= hi;
  const steep = 0.02 * yr; // a step this big between two neighbouring samples is checked for a jump
  const maxStep = 1 / 350; // in units of the window: refine until points are this close
  const segs: Pt[][] = [];
  let cur: Pt[] | null = null;
  let budget = 30000; // refined points in all; a wild function (sin(1/x)) stays drawable

  /** Last x on the inside going from a (inside) towards b (outside or undefined). */
  const edge = (a: number, b: number): Pt => {
    let fa = f(a);
    for (let k = 0; k < 40; k++) {
      const mid = (a + b) / 2;
      const fm = f(mid);
      if (inside(fm)) {
        a = mid;
        fa = fm;
      } else b = mid;
    }
    return [a, fa];
  };

  /** Points between two inside samples so that the step is small on screen (none for a smooth, flat part). */
  const refine = (xa: number, ya: number, xb: number, yb: number, depth: number, out: Pt[]) => {
    const d = Math.hypot((xb - xa) / xr, (yb - ya) / yr);
    if (d <= maxStep || depth > 12 || budget <= 0) return;
    budget--;
    const xm = (xa + xb) / 2;
    const ym = f(xm);
    if (!inside(ym)) return; // a spike out of the window between samples: rare, left as a straight step
    refine(xa, ya, xm, ym, depth + 1, out);
    out.push([xm, ym]);
    refine(xm, ym, xb, yb, depth + 1, out);
  };

  /** A big step between two inside samples: a real discontinuity, a dip outside the window, or just steep. */
  const split = (a: number, b: number): { kind: 'jump'; l: Pt; r: Pt } | { kind: 'out'; mid: number } | null => {
    let fl = f(a);
    let fr = f(b);
    for (let k = 0; k < 48; k++) {
      const mid = (a + b) / 2;
      const fm = f(mid);
      if (!inside(fm)) return { kind: 'out', mid };
      if (Math.abs(fm - fl) >= Math.abs(fr - fm)) {
        b = mid;
        fr = fm;
      } else {
        a = mid;
        fl = fm;
      }
    }
    // bisected down to neighbouring floats: a continuous function has closed the gap, a jump has not
    return Math.abs(fr - fl) > 1e-4 * yr ? { kind: 'jump', l: [a, fl], r: [b, fr] } : null;
  };

  const close = () => {
    // a lone point (a tread that starts exactly at the end of the range) is not a segment
    if (cur && cur.length >= 2 && cur.some(([x, y]) => x !== cur![0][0] || y !== cur![0][1])) segs.push(cur);
    cur = null;
  };

  let px = xmin;
  let pv = f(xmin);
  if (inside(pv)) cur = [[px, pv]];
  for (let i = 1; i <= n; i++) {
    const x = xmin + (xr * i) / n;
    const v = f(x);
    if (inside(v)) {
      if (!cur) {
        // coming back into the window (or into the domain)
        cur = [edge(x, px)];
        refine(cur[0][0], cur[0][1], x, v, 0, cur);
      } else if (Math.abs(v - pv) > steep) {
        const s = split(px, x);
        if (s && s.kind === 'jump') {
          refine(px, pv, s.l[0], s.l[1], 0, cur);
          cur.push(s.l);
          close();
          cur = [s.r];
          refine(s.r[0], s.r[1], x, v, 0, cur);
        } else if (s && s.kind === 'out') {
          cur.push(edge(px, s.mid));
          close();
          cur = [edge(x, s.mid)];
          refine(cur[0][0], cur[0][1], x, v, 0, cur);
        } else refine(px, pv, x, v, 0, cur);
      } else refine(px, pv, x, v, 0, cur);
      cur.push([x, v]);
    } else if (cur) {
      cur.push(edge(px, x));
      close();
    }
    px = x;
    pv = v;
  }
  close();
  return segs;
}

/** A tick step of 1, 2 or 5 times a power of ten, giving about `target` ticks over the range. */
export function niceStep(range: number, target = 5): number {
  if (!(range > 0) || !Number.isFinite(range)) return 1;
  const raw = range / Math.max(1, target);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const r = raw / p;
  return (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * p;
}

/** Tick values inside [min, max], multiples of a nice step. */
export function niceTicks(min: number, max: number, target = 5): { step: number; ticks: number[] } {
  const step = niceStep(max - min, target);
  const ticks: number[] = [];
  const first = Math.ceil(min / step - 1e-9);
  for (let k = first; k * step <= max + step * 1e-9 && ticks.length < 50; k++) {
    const v = k * step;
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return { step, ticks };
}

/** A tick label with as many decimals as the step needs (no trailing zeros), and a real minus sign. */
export function formatTick(v: number, step: number): string {
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  let s = v.toFixed(decimals);
  if (s.includes('.')) s = s.replace(/\.?0+$/, '');
  if (s === '-0') s = '0';
  return s.replace('-', '−');
}

/** The given window, unless it shows less than a fifth of the finite values: then one fitted to them. */
export function fitRange(values: ArrayLike<number>, min: number, max: number): { min: number; max: number } {
  const finite: number[] = [];
  let shown = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    finite.push(v);
    if (v >= min && v <= max) shown++;
  }
  if (!finite.length || shown >= 0.2 * finite.length) return { min, max };
  finite.sort((a, b) => a - b);
  const q = (p: number) => finite[Math.min(finite.length - 1, Math.max(0, Math.round(p * (finite.length - 1))))];
  let lo = q(0.03);
  let hi = q(0.97);
  if (!(hi > lo)) {
    lo -= 1;
    hi += 1;
  }
  const pad = 0.08 * (hi - lo);
  return { min: lo - pad, max: hi + pad };
}

// ---------- colours ----------

function parseHex(c: string | undefined | null): [number, number, number] | null {
  if (!c) return null;
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(c.trim());
  if (!m) return null;
  let h = m[1];
  if (h.length === 3) h = h.replace(/./g, (d) => d + d);
  return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
}

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  h /= 6;
  return [h, s, l];
}

function hslToHex(h: number, s: number, l: number): string {
  const f = (n: number) => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    const v = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(v * 255)
      .toString(16)
      .padStart(2, '0');
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

/** An ink colour that reads on the dark 3D stage: board ink is often dark (it was drawn on a whiteboard), so its
 *  hue is kept and it is lightened; black, grey or unparsable ink becomes the fallback (the theme accent). */
export function stageInk(c: string | undefined | null, fallback: string): string {
  const rgb = parseHex(c);
  if (!rgb) return fallback;
  const [h, s, l] = rgbToHsl(...rgb);
  if (s < 0.12) return l > 0.78 ? hslToHex(h, s, l) : fallback; // white chalk stays white; black/grey -> accent
  return hslToHex(h, Math.max(s, 0.5), Math.min(0.8, Math.max(l, 0.62)));
}
