// The teacher's pointing gestures, re-drawn as calm annotations on SVG: a spotlight (everything else dims around a
// soft hole), a circle drawn like a pen loop, an underline drawn left to right, an arrow whose head pops in at the
// end, and pulse rings. Shared by the slide view (mouse cursor, laser pointer) and the board view (the teacher's
// hand). It renders inside the parent's <svg>, in the coordinates of the slide or board image, and moves only with
// the shared clock: the parent calls render(t) every frame, never CSS timers, so scrubbing lands on exactly the
// right frame. The original video is never touched.
import {
  forwardRef, memo, useId, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState, type ReactNode,
  type RefObject,
} from 'react';
import type { Box, SlideAnnotation } from '../types';
import {
  annotationAlpha, annotationTiming, arrowEnds, arrowGeometry, arrowHeadAt, circlePath, drawable, drawProgress, kindOf,
  pulseRect, pulseRings, spotlightHole, strokeOf, underlinePath, underlineY, unitOf, PULSE_RINGS, type AnnotationTiming,
  type ArrowBoxes, type Kind, type Pt, type Rect,
} from './slideTiming';
import { n1, op, setAttr } from './svgAttrs';

export interface AnnotationLayerHandle {
  render(t: number): void;
}

/** The spotlight's veil over everything outside its hole. */
export interface Veil {
  colour: string;
  opacity: number;
}

/** Slides: a black veil at half strength. */
export const DEFAULT_VEIL: Veil = { colour: '#000', opacity: 0.5 };

export interface AnnotationLayerProps {
  annotations: readonly SlideAnnotation[];
  /** Size of the image the annotations' coordinates are in (slide or board pixels). */
  size: [number, number];
  /** The shared clock: the frame for its time is drawn at once after every React render. */
  timeRef: RefObject<number>;
  /** Colour of an annotation's lines and rings. Keep it stable (module level or useCallback). */
  colourOf(a: SlideAnnotation): string;
  /** Line width in image pixels (default strokeOf(size)). */
  stroke?: number;
  veil?: Veil;
  /** When what an annotation points at goes away (its slide changes, its writing is erased): it starts fading
   *  there, even inside its hold. Keep it stable. */
  endOf?(a: SlideAnnotation): number | null | undefined;
  /** Box of one of an annotation's targets, so an arrow points at it instead of covering it. Keep it stable. */
  boxOf?(a: SlideAnnotation, id: string): Box | null | undefined;
  /** The board: an arrow's end boxes and the writing it keeps off, in place of boxOf. Keep it stable. */
  arrowBoxesOf?(a: SlideAnnotation): ArrowBoxes;
  /** Drawn above the spotlight's veil and below the annotations (the slide view's hover box). */
  children?: ReactNode;
}

interface Prepared {
  a: SlideAnnotation;
  kind: Kind;
  tm: AnnotationTiming;
  colour: string;
  d?: string; // circle, underline, arrow shaft
  head?: string; // arrowhead wings
  tip?: Pt;
  hole?: Rect; // spotlight
}

interface AnnDom {
  g: SVGGElement;
  paths: SVGPathElement[]; // progressively drawn strokes (glow + line)
  head: SVGGElement | null;
  rings: SVGRectElement[];
}

function prepare(
  annotations: readonly SlideAnnotation[],
  size: [number, number],
  colourOf: AnnotationLayerProps['colourOf'],
  endOf?: AnnotationLayerProps['endOf'],
  boxOf?: AnnotationLayerProps['boxOf'],
  arrowBoxesOf?: AnnotationLayerProps['arrowBoxesOf'],
): Prepared[] {
  const out: Prepared[] = [];
  for (const a of annotations) {
    if (!drawable(a)) continue;
    const kind = kindOf(a);
    const p: Prepared = { a, kind, tm: annotationTiming(a, endOf?.(a)), colour: colourOf(a) };
    if (kind === 'circle') p.d = circlePath(a.box, size, a.path);
    else if (kind === 'underline') p.d = underlinePath(a.box, size, underlineY(a, size));
    else if (kind === 'spotlight') p.hole = spotlightHole(a.box, size);
    else if (kind === 'arrow') {
      const box = (id: string | undefined) => (id && boxOf ? (boxOf(a, id) ?? null) : null);
      const tg = a.target ?? [];
      const ends: ArrowBoxes = arrowBoxesOf?.(a) ?? { from: tg.length >= 2 ? box(tg[0]) : null, to: box(tg[tg.length - 1]) };
      const [from, to] = arrowEnds(a, ends.from, ends.to, 8 * unitOf(size));
      const g = arrowGeometry(from, to, size, ends.from, ends.to, ends.obstacles);
      p.d = g.d;
      p.head = g.head;
      p.tip = g.tip;
    }
    out.push(p);
  }
  return out;
}

/** Sets the attributes of the elements React built for a time t, and only the ones that changed. */
class AnnotationAnimator {
  prepared: Prepared[] = [];
  size: [number, number] = [960, 540];
  stroke = 3.4;
  dirty = true;

  private anns = new Map<number, AnnDom>();
  private holes = new Map<number, SVGRectElement>();
  private dim: SVGGElement | null = null;
  private lastT = NaN;

  /** Find the elements React rendered (called after every React render of the layer). */
  collect(root: Element) {
    this.anns.clear();
    root.querySelectorAll<SVGGElement>('g[data-ann]').forEach((g) => {
      this.anns.set(Number(g.dataset.ann), {
        g,
        paths: Array.from(g.querySelectorAll<SVGPathElement>('path[data-draw]')),
        head: g.querySelector<SVGGElement>('g[data-head]'),
        rings: Array.from(g.querySelectorAll<SVGRectElement>('rect[data-ring]')),
      });
    });
    this.holes.clear();
    root.querySelectorAll<SVGRectElement>('rect[data-hole]').forEach((el) => this.holes.set(Number(el.dataset.hole), el));
    this.dim = root.querySelector<SVGGElement>('g[data-dim]');
    this.dirty = true;
  }

  render(t: number) {
    if (!this.dirty && t === this.lastT) return;
    this.dirty = false;
    this.lastT = t;
    const { size } = this;
    const sw = this.stroke;
    let spotMax = 0;
    const spots: [number, number][] = [];
    this.prepared.forEach((p, i) => {
      const alpha = annotationAlpha(p.tm, t);
      if (p.kind === 'spotlight') {
        if (alpha > 0.001) {
          spots.push([i, alpha]);
          spotMax = Math.max(spotMax, alpha);
        } else setAttr(this.holes.get(i), 'opacity', '0');
        return;
      }
      const dom = this.anns.get(i);
      if (!dom) return;
      if (alpha <= 0.001) {
        setAttr(dom.g, 'display', 'none');
        return;
      }
      if (p.kind === 'pulse') {
        setAttr(dom.g, 'display', 'inline');
        setAttr(dom.g, 'opacity', op(alpha));
        pulseRings(p.tm, t).forEach((r, j) => {
          const el = dom.rings[j];
          if (!el) return;
          if (r.alpha <= 0.002) {
            setAttr(el, 'opacity', '0');
            return;
          }
          const rc = pulseRect(p.a.box, size, r.k);
          setAttr(el, 'x', n1(rc.x));
          setAttr(el, 'y', n1(rc.y));
          setAttr(el, 'width', n1(rc.w));
          setAttr(el, 'height', n1(rc.h));
          setAttr(el, 'rx', n1(rc.r));
          setAttr(el, 'stroke-width', (sw * (1 - 0.45 * r.k)).toFixed(2));
          setAttr(el, 'opacity', op(r.alpha));
        });
        return;
      }
      const progress = drawProgress(p.tm, t);
      if (progress <= 0.0005) {
        setAttr(dom.g, 'display', 'none'); // nothing drawn yet: a round cap alone would show as a dot
        return;
      }
      setAttr(dom.g, 'display', 'inline');
      setAttr(dom.g, 'opacity', op(alpha));
      const off = (1 - progress).toFixed(4);
      for (const path of dom.paths) setAttr(path, 'stroke-dashoffset', off);
      if (dom.head && p.tip) {
        const k = arrowHeadAt(p.tm, t);
        const s = 0.55 + 0.45 * k;
        const [x, y] = p.tip;
        setAttr(dom.head, 'opacity', op(k));
        setAttr(dom.head, 'transform', `translate(${n1(x)} ${n1(y)}) scale(${s.toFixed(3)}) translate(${n1(-x)} ${n1(-y)})`);
      }
    });

    // Spotlight: one veil with a soft hole per active spotlight (a fading one closes its hole).
    if (this.dim) {
      if (spotMax <= 0.001) setAttr(this.dim, 'display', 'none');
      else {
        setAttr(this.dim, 'display', 'inline');
        setAttr(this.dim, 'opacity', op(spotMax));
      }
    }
    for (const [i, a] of spots) setAttr(this.holes.get(i), 'opacity', op(a / spotMax));
  }
}

/** One annotation's SVG, built once; AnnotationAnimator animates it. */
const AnnotationShape = memo(function AnnotationShape(props: {
  p: Prepared;
  i: number;
  sw: number;
  size: [number, number];
}) {
  const { p, i, sw, size } = props;
  if (p.kind === 'spotlight') return null; // drawn by the shared veil
  const common = { fill: 'none', stroke: p.colour, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  if (p.kind === 'pulse') {
    const base = pulseRect(p.a.box, size, 0); // a faint glow where the rings start
    return (
      <g data-ann={i} display="none" opacity={0}>
        <rect x={base.x} y={base.y} width={base.w} height={base.h} rx={base.r} fill={p.colour} fillOpacity={0.12} />
        {Array.from({ length: PULSE_RINGS }, (_, j) => (
          <rect key={j} data-ring={j} {...common} strokeWidth={sw} opacity={0} />
        ))}
      </g>
    );
  }
  // circle, underline, arrow: a soft glow under a crisp line, both drawn with stroke-dashoffset (pathLength 1)
  const drawn = { pathLength: 1, strokeDasharray: '1 1', strokeDashoffset: 1 };
  return (
    <g data-ann={i} display="none" opacity={0}>
      <path data-draw="" d={p.d} {...common} {...drawn} strokeWidth={sw * 2.8} strokeOpacity={0.2} />
      <path data-draw="" d={p.d} {...common} {...drawn} strokeWidth={sw} />
      {p.head && (
        <g data-head="" opacity={0}>
          <path d={p.head} {...common} strokeWidth={sw * 2.8} strokeOpacity={0.2} />
          <path d={p.head} {...common} strokeWidth={sw} />
        </g>
      )}
    </g>
  );
});

/** The annotation layer, to be placed inside an <svg> whose viewBox is the image's size. */
export const AnnotationLayer = memo(
  forwardRef<AnnotationLayerHandle, AnnotationLayerProps>(function AnnotationLayer(props, ref) {
    const { annotations, size, timeRef, colourOf, stroke, veil = DEFAULT_VEIL, endOf, boxOf, arrowBoxesOf, children } = props;
    const [W, H] = size;
    const u = unitOf(size);
    const sw = stroke ?? strokeOf(size);
    const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '');
    const blurId = `rb-blur-${uid}`;
    const maskId = `rb-spot-${uid}`;
    const rootRef = useRef<SVGGElement>(null);
    const [anim] = useState(() => new AnnotationAnimator());

    const prepared = useMemo(
      () => prepare(annotations, size, colourOf, endOf, boxOf, arrowBoxesOf),
      [annotations, size, colourOf, endOf, boxOf, arrowBoxesOf],
    );
    anim.prepared = prepared;
    anim.size = size;
    anim.stroke = sw;

    // After every render: hand the animator the elements React made, and draw the current frame at once.
    useLayoutEffect(() => {
      const root = rootRef.current;
      if (!root) return;
      anim.collect(root);
      anim.render(timeRef.current ?? 0);
    });

    useImperativeHandle(ref, () => ({ render: (t: number) => anim.render(t) }), [anim]);

    const spots = prepared.map((p, i) => [p, i] as const).filter(([p]) => p.kind === 'spotlight');

    return (
      <g ref={rootRef}>
        {spots.length > 0 && (
          <>
            <defs>
              <filter id={blurId} filterUnits="userSpaceOnUse" x={0} y={0} width={W} height={H}>
                <feGaussianBlur stdDeviation={9 * u} />
              </filter>
              <mask id={maskId} maskUnits="userSpaceOnUse" x={0} y={0} width={W} height={H}>
                <rect width={W} height={H} fill="#fff" />
                {spots.map(([p, i]) => (
                  <rect
                    key={i}
                    data-hole={i}
                    x={p.hole!.x}
                    y={p.hole!.y}
                    width={p.hole!.w}
                    height={p.hole!.h}
                    rx={p.hole!.r}
                    fill="#000"
                    opacity={0}
                    filter={`url(#${blurId})`}
                  />
                ))}
              </mask>
            </defs>
            <g data-dim="" display="none" opacity={0}>
              <rect width={W} height={H} fill={veil.colour} fillOpacity={veil.opacity} mask={`url(#${maskId})`} />
            </g>
          </>
        )}

        {children}

        {prepared.map((p, i) => (
          <AnnotationShape key={i} p={p} i={i} sw={sw} size={size} />
        ))}
      </g>
    );
  }),
);
