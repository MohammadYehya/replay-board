// The slide view: the clean slide of the state shown at t (crossfading through each change), an SVG layer that
// re-draws the teacher's pointer gestures as calm annotations, and invisible hit boxes on the slide's elements.
// Everything moves with the shared clock (render(t) is called every frame by the player), never with CSS
// timers, so scrubbing lands on exactly the right frame. The original video is never touched.
import {
  forwardRef, memo, useCallback, useEffect, useId, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState,
  type RefObject,
} from 'react';
import { jobFileUrl } from '../api';
import type { PointerSample, SlideAnnotation, SlideElement, SlidesData, SlideState } from '../types';
import {
  annotationAlpha, annotationTiming, arrowEnds, arrowGeometry, arrowHeadAt, circlePath, crossfadeAt, drawProgress,
  elementSeekTime, isLightColour, kindOf, pickElement, pointerAt, pulseRect, pulseRings, slideEndAt, sortStates, spotlightHole,
  stateIndexAt, strokeOf, toolColour, underlinePath, underlineY, unitOf, PULSE_RINGS, type AnnotationTiming, type Kind, type Pt,
  type Rect,
} from './slideTiming';
import { formatTime, talkingAt } from './timing';
import { kindLabel, sameReading, tipTex } from './tip';

export interface SlidesHandle {
  render(t: number): void;
}

interface Props {
  jobId: string;
  slides: SlidesData;
  timeRef: RefObject<number>;
  clipStart: number;
  playing: boolean;
  showPointer: boolean;
  /** Seek to t and play. */
  onSeek(t: number): void;
  onTogglePlay(): void;
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

const EMPTY: SlideElement[] = [];
const TALK_FILL_LIGHT = 'rgb(255, 207, 51)';
const TALK_FILL_DARK = 'rgb(255, 214, 92)';

/** Write an attribute only when its value changes: the clock calls render every frame. */
const attrMemo = new WeakMap<Element, Map<string, string>>();
function setAttr(el: Element | null | undefined, name: string, value: string) {
  if (!el) return;
  let m = attrMemo.get(el);
  if (!m) attrMemo.set(el, (m = new Map()));
  if (m.get(name) === value) return;
  m.set(name, value);
  el.setAttribute(name, value);
}
const op = (v: number) => (v <= 0.001 ? '0' : v >= 0.999 ? '1' : v.toFixed(3));
const n1 = (v: number) => (Math.round(v * 10) / 10).toString();

function prepare(slides: SlidesData, states: SlideState[]): Prepared[] {
  const size = slides.size;
  const indexOf = new Map(states.map((s, i) => [s.id, i]));
  const out: Prepared[] = [];
  for (const a of slides.annotations ?? []) {
    if (!a || !Array.isArray(a.box) || a.box.length < 4 || !Number.isFinite(a.t_start)) continue;
    const kind = kindOf(a);
    const si = a.state != null ? (indexOf.get(a.state) ?? -1) : -1;
    const st = si >= 0 ? states[si] : undefined;
    const p: Prepared = { a, kind, tm: annotationTiming(a, slideEndAt(states, si)), colour: toolColour(a.tool) };
    if (kind === 'circle') p.d = circlePath(a.box, size, a.path);
    else if (kind === 'underline') p.d = underlinePath(a.box, size, underlineY(a, size));
    else if (kind === 'spotlight') p.hole = spotlightHole(a.box, size);
    else if (kind === 'arrow') {
      const els = st?.elements ?? [];
      const boxOf = (id: string | undefined) => (id ? (els.find((e) => e.id === id)?.box ?? null) : null);
      const tg = a.target ?? [];
      const fromBox = tg.length >= 2 ? boxOf(tg[0]) : null;
      const toBox = boxOf(tg[tg.length - 1]);
      const [from, to] = arrowEnds(a, fromBox, toBox, 8 * unitOf(size));
      const g = arrowGeometry(from, to, size, fromBox, toBox);
      p.d = g.d;
      p.head = g.head;
      p.tip = g.tip;
    }
    out.push(p);
  }
  return out;
}

/** Drives the slide images and the SVG layer for a time t. React builds the elements; this only sets
 *  attributes on them, and only the ones that changed. */
class SlideAnimator {
  states: SlideState[] = [];
  prepared: Prepared[] = [];
  samples: PointerSample[] = [];
  size: [number, number] = [960, 540];
  light = true;
  showPointer = false;
  /** The elements of the state React is showing, by id (for the "talked about" highlight). */
  elements = new Map<string, SlideElement>();
  onState?: (i: number) => void;
  dirty = true;

  private imgs = new Map<number, HTMLImageElement>();
  private anns = new Map<number, AnnDom>();
  private holes = new Map<number, SVGRectElement>();
  private dim: SVGGElement | null = null;
  private talk = new Map<string, SVGRectElement>();
  private pointer: SVGGElement | null = null;
  private pointerFill: SVGElement[] = [];
  private idx = -2;
  private lastT = NaN;
  private shownImgs = new Set<number>();

  /** Find the elements React rendered (called after every React render of the view). */
  collect(root: HTMLElement) {
    this.imgs.clear();
    root.querySelectorAll<HTMLImageElement>('img[data-state]').forEach((el) => this.imgs.set(Number(el.dataset.state), el));
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
    this.talk.clear();
    root.querySelectorAll<SVGRectElement>('rect[data-talk]').forEach((el) => this.talk.set(el.dataset.talk ?? '', el));
    this.pointer = root.querySelector<SVGGElement>('g[data-pointer]');
    this.pointerFill = this.pointer ? Array.from(this.pointer.querySelectorAll<SVGElement>('[data-fill]')) : [];
    this.shownImgs = new Set(this.imgs.keys()); // re-assert every image's opacity on the next frame
    this.dirty = true;
  }

  render(t: number) {
    if (!this.dirty && t === this.lastT) return;
    this.dirty = false;
    this.lastT = t;
    const { states, size } = this;

    // Slide image: the current state, crossfading from the previous one during its change window.
    const idx = stateIndexAt(states, t);
    if (idx !== this.idx) {
      this.idx = idx;
      this.onState?.(idx);
    }
    const want = new Map<number, number>();
    if (idx >= 0) {
      const { prev, mix } = crossfadeAt(states, idx, t);
      const hasNew = !!states[idx].image;
      if (hasNew) want.set(idx, mix);
      if (prev >= 0 && states[prev].image) want.set(prev, hasNew ? 1 : 1 - mix);
    }
    for (const i of this.shownImgs) if (!want.has(i)) setAttr(this.imgs.get(i), 'style', 'opacity:0');
    for (const [i, o] of want) setAttr(this.imgs.get(i), 'style', `opacity:${op(o)}`);
    this.shownImgs = new Set(want.keys());

    // Annotations.
    const sw = strokeOf(size);
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

    // Spotlight: one dimming layer with a soft hole per active spotlight (a fading one closes its hole).
    if (this.dim) {
      if (spotMax <= 0.001) setAttr(this.dim, 'display', 'none');
      else {
        setAttr(this.dim, 'display', 'inline');
        setAttr(this.dim, 'opacity', op(spotMax));
      }
    }
    for (const [i, a] of spots) setAttr(this.holes.get(i), 'opacity', op(a / spotMax));

    // Highlighter behind what is being talked about.
    const strength = this.light ? 0.42 : 0.24;
    for (const [id, rect] of this.talk) {
      const el = this.elements.get(id);
      setAttr(rect, 'opacity', op(el ? talkingAt(el, t) * strength : 0));
    }

    // The teacher's pointer, when asked for.
    if (this.pointer) {
      const d = this.showPointer ? pointerAt(this.samples, t) : null;
      if (!d || d.alpha <= 0.01) setAttr(this.pointer, 'display', 'none');
      else {
        setAttr(this.pointer, 'display', 'inline');
        setAttr(this.pointer, 'opacity', op(0.8 * d.alpha));
        setAttr(this.pointer, 'transform', `translate(${n1(d.x)} ${n1(d.y)})`);
        const c = toolColour(d.tool);
        for (const el of this.pointerFill) setAttr(el, 'fill', c);
      }
    }
  }
}

/** One annotation's SVG, built once; SlideAnimator animates it. */
const AnnotationShape = memo(function AnnotationShape(props: {
  p: Prepared;
  i: number;
  sw: number;
  size: [number, number];
}) {
  const { p, i, sw, size } = props;
  if (p.kind === 'spotlight') return null; // drawn by the shared dimming layer
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

// Memoised: the player re-renders on every spoken word, the slides only need to when their props change.
export const Slides = memo(
  forwardRef<SlidesHandle, Props>(function Slides(props, ref) {
    const { jobId, slides, timeRef, clipStart, playing, showPointer, onSeek, onTogglePlay } = props;
    const [W, H] = slides.size;
    const u = unitOf(slides.size);
    const sw = strokeOf(slides.size);
    const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '');
    const blurId = `rb-blur-${uid}`;
    const maskId = `rb-spot-${uid}`;
    const stageRef = useRef<HTMLDivElement>(null);
    const clipRef = useRef<HTMLDivElement>(null);
    const tipRef = useRef<HTMLDivElement>(null);
    const pointerRef = useRef({ x: 0, y: 0 }); // last mouse position in stage CSS pixels
    const [anim] = useState(() => new SlideAnimator());

    const states = useMemo(() => sortStates(slides.states ?? []), [slides]);
    const prepared = useMemo(() => prepare(slides, states), [slides, states]);
    const samples = useMemo(
      () => (slides.pointer ?? []).filter((s) => Array.isArray(s) && Number.isFinite(s[0])).slice().sort((a, b) => a[0] - b[0]),
      [slides],
    );
    const light = isLightColour(slides.background);
    const background = slides.background || (light ? '#ffffff' : '#111111');

    const [stateIdx, setStateIdx] = useState(() => stateIndexAt(states, timeRef.current ?? 0));
    const state: SlideState | undefined = stateIdx >= 0 ? states[stateIdx] : undefined;
    const elements = state?.elements ?? EMPTY;
    const [hover, setHover] = useState<SlideElement | null>(null);
    const [cssW, setCssW] = useState(0);
    const [flash, setFlash] = useState<{ key: number; play: boolean } | null>(null);
    const [tipPos, setTipPos] = useState<{ left: number; top: number; below: boolean } | null>(null);

    anim.states = states;
    anim.prepared = prepared;
    anim.samples = samples;
    anim.size = slides.size;
    anim.light = light;
    anim.onState = setStateIdx;
    if (anim.showPointer !== showPointer) {
      anim.showPointer = showPointer;
      anim.dirty = true;
    }

    // After every render: hand the animator the elements React made, and draw the current frame at once.
    useLayoutEffect(() => {
      const root = stageRef.current;
      if (!root) return;
      anim.elements = new Map(elements.map((e) => [e.id, e]));
      anim.collect(root);
      anim.render(timeRef.current ?? 0);
    });

    // A new state, or a refreshed replay: keep the hovered element only if it is still on the slide.
    useEffect(() => {
      setHover((h) => (h ? (elements.find((e) => e.id === h.id) ?? null) : null));
    }, [elements]);

    // Preload and decode every slide image, so a change never shows a half-loaded picture.
    const imageKey = states.map((s) => s.image ?? '').join('|');
    useEffect(() => {
      const keep: HTMLImageElement[] = [];
      for (const path of imageKey.split('|')) {
        if (!path) continue;
        const im = new Image();
        im.src = jobFileUrl(jobId, path);
        im.decode?.().catch(() => {});
        keep.push(im);
      }
      return () => {
        keep.length = 0;
      };
    }, [imageKey, jobId]);

    useEffect(() => {
      const stage = stageRef.current;
      if (!stage) return;
      const ro = new ResizeObserver(() => setCssW(stage.clientWidth));
      ro.observe(stage);
      return () => ro.disconnect();
    }, []);

    useImperativeHandle(ref, () => ({ render: (t: number) => anim.render(t) }), [anim]);

    const pick = useCallback(
      (clientX: number, clientY: number): SlideElement | null => {
        const r = clipRef.current?.getBoundingClientRect();
        if (!r || !r.width || !r.height) return null;
        const x = ((clientX - r.left) / r.width) * W;
        const y = ((clientY - r.top) / r.height) * H;
        return pickElement(elements, x, y, 3 * u);
      },
      [elements, W, H, u],
    );

    // Tooltip above (or below) the hovered element, kept inside the stage horizontally.
    useLayoutEffect(() => {
      const tip = tipRef.current;
      if (!hover || !tip || !cssW) {
        setTipPos(null);
        return;
      }
      const k = cssW / W;
      const [x, y, w, h] = hover.box;
      const tw = tip.offsetWidth;
      const th = tip.offsetHeight;
      const stageH = H * k;
      const clampX = (v: number) => Math.min(Math.max(8, v), Math.max(8, cssW - tw - 8));
      const aboveTop = y * k - th - 10;
      const belowTop = (y + h) * k + 10;
      if (aboveTop >= -40) setTipPos({ left: clampX((x + w / 2) * k - tw / 2), top: aboveTop, below: false });
      else if (belowTop + th <= stageH + 120) setTipPos({ left: clampX((x + w / 2) * k - tw / 2), top: belowTop, below: true });
      else {
        const p = pointerRef.current;
        const top = Math.min(Math.max(8, p.y + 18), Math.max(8, stageH - th - 8));
        setTipPos({ left: clampX(p.x - tw / 2), top, below: true });
      }
    }, [hover, cssW, W, H]);

    const onClick = (e: React.MouseEvent) => {
      const el = pick(e.clientX, e.clientY);
      if (el) {
        onSeek(elementSeekTime(el, state, clipStart));
        return;
      }
      setFlash({ key: Date.now(), play: !playing });
      onTogglePlay();
    };

    const spots = prepared.map((p, i) => [p, i] as const).filter(([p]) => p.kind === 'spotlight');
    const tipLatex = hover?.latex?.trim() ?? '';
    const tipHtml = useMemo(() => tipTex(tipLatex), [tipLatex]);
    const rawText = hover?.text?.trim() ?? '';
    const tipText = tipHtml && sameReading(rawText, tipLatex) ? '' : rawText || (tipHtml ? '' : tipLatex);
    const spokenHover = !!hover?.spoken?.some((s) => Number.isFinite(s.start));
    const tipTime = hover ? elementSeekTime(hover, state, clipStart) : 0;
    const hoverStroke = light ? 'rgba(37, 99, 235, 0.85)' : 'rgba(122, 167, 255, 0.9)';
    const hoverFill = light ? 'rgba(76, 141, 255, 0.10)' : 'rgba(122, 167, 255, 0.16)';

    return (
      <div
        ref={stageRef}
        className={`slide-stage ${hover ? 'is-hovering' : ''}`}
        style={{ aspectRatio: `${W} / ${H}` }}
        onPointerMove={(e) => {
          const r = stageRef.current?.getBoundingClientRect();
          if (r) pointerRef.current = { x: e.clientX - r.left, y: e.clientY - r.top };
          const el = pick(e.clientX, e.clientY);
          setHover((h) => (h?.id === el?.id ? h : el));
        }}
        onPointerLeave={() => setHover(null)}
        onClick={onClick}
      >
        <div ref={clipRef} className="slide-clip" style={{ background }}>
          {states.map((s, i) =>
            s.image ? (
              <img
                key={`${i}:${s.image}`}
                data-state={i}
                className="slide-img"
                src={jobFileUrl(jobId, s.image)}
                alt=""
                draggable={false}
                decoding="sync"
              />
            ) : null,
          )}

          <svg className="slide-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
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

            <g className={`talk ${light ? 'on-light' : 'on-dark'}`}>
              {elements.map((el) => {
                const [x, y, w, h] = el.box;
                return (
                  <rect
                    key={el.id}
                    data-talk={el.id}
                    x={x - 4 * u}
                    y={y - 3 * u}
                    width={w + 8 * u}
                    height={h + 6 * u}
                    rx={5 * u}
                    fill={light ? TALK_FILL_LIGHT : TALK_FILL_DARK}
                    opacity={0}
                  />
                );
              })}
            </g>

            {spots.length > 0 && (
              <g data-dim="" display="none" opacity={0}>
                <rect width={W} height={H} fill="#000" fillOpacity={0.5} mask={`url(#${maskId})`} />
              </g>
            )}

            {hover && (
              <rect
                x={hover.box[0] - 5 * u}
                y={hover.box[1] - 4 * u}
                width={hover.box[2] + 10 * u}
                height={hover.box[3] + 8 * u}
                rx={6 * u}
                fill={hoverFill}
                stroke={hoverStroke}
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
              />
            )}

            {prepared.map((p, i) => (
              <AnnotationShape key={i} p={p} i={i} sw={sw} size={slides.size} />
            ))}

            <g data-pointer="" display="none" opacity={0}>
              <circle data-fill="" r={10 * u} opacity={0.22} />
              <circle data-fill="" r={4.5 * u} stroke="#fff" strokeOpacity={0.85} strokeWidth={1.3 * u} />
            </g>
          </svg>
        </div>

        {hover && (
          <div
            ref={tipRef}
            className={`board-tip ${tipPos?.below ? 'below' : ''}`}
            style={{ left: tipPos?.left ?? 0, top: tipPos?.top ?? 0, visibility: tipPos ? 'visible' : 'hidden' }}
          >
            <div className="tip-head">
              <span className="tip-kind">{kindLabel(hover.kind, 'Text')}</span>
            </div>
            {tipHtml && <div className="tip-tex" dangerouslySetInnerHTML={{ __html: tipHtml }} />}
            {tipText && <div className={tipHtml ? 'tip-text muted' : 'tip-text'}>{tipText}</div>}
            <div className="tip-foot">
              <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
                <path d="M4 3l9 5-9 5z" fill="currentColor" />
              </svg>
              {spokenHover
                ? `Click to hear it explained · ${formatTime(tipTime)}`
                : `Click to play this slide from ${formatTime(tipTime)}`}
            </div>
          </div>
        )}

        {flash && (
          <div key={flash.key} className="board-flash" aria-hidden>
            {flash.play ? (
              <svg viewBox="0 0 24 24" width="28" height="28">
                <path d="M8 5l11 7-11 7z" fill="currentColor" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" width="28" height="28">
                <path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor" />
              </svg>
            )}
          </div>
        )}

        {states.length === 0 && <div className="board-empty">No slides were found in this video.</div>}
      </div>
    );
  }),
);
