// The slide view: the clean slide of the state shown at t (crossfading through each change), an SVG layer that
// re-draws the teacher's pointer gestures as calm annotations (AnnotationLayer), and invisible hit boxes on the
// slide's elements. Everything moves with the shared clock (render(t) is called every frame by the player), never
// with CSS timers, so scrubbing lands on exactly the right frame. The original video is never touched.
import {
  forwardRef, memo, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState,
  type RefObject,
} from 'react';
import { jobFileUrl } from '../api';
import type { Box, PointerSample, SlideAnnotation, SlideElement, SlidesData, SlideState } from '../types';
import { AnnotationLayer, type AnnotationLayerHandle } from './AnnotationLayer';
import {
  crossfadeAt, elementSeekTime, isLightColour, pickElement, pointerAt, slideEndAt, sortStates, stateIndexAt, toolColour,
  unitOf,
} from './slideTiming';
import { n1, op, setAttr } from './svgAttrs';
import { formatTime, talkingAt } from './timing';
import { kindLabel, sameReading, tipTex } from './tip';

export interface SlidesHandle {
  render(t: number): void;
  /** Outline these slide elements ("s0e1") for a moment: an answer of Ask Muse pointing at them. */
  highlight(ids: string[]): void;
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
  /** Attach a slide element to an Ask Muse question (Shift+click). */
  onAsk?(el: SlideElement): void;
}

const HIGHLIGHT_MS = 2400;

const EMPTY: SlideElement[] = [];
const NO_ANNOTATIONS: SlideAnnotation[] = [];
const TALK_FILL_LIGHT = 'rgb(255, 207, 51)';
const TALK_FILL_DARK = 'rgb(255, 214, 92)';
/** Amber for the mouse cursor, red for the laser pointer. */
const slideColour = (a: SlideAnnotation) => toolColour(a.tool);

/** Drives the slide images, the highlighter and the pointer dot for a time t (the annotations have their own
 *  layer). React builds the elements; this only sets attributes on them, and only the ones that changed. */
class SlideAnimator {
  states: SlideState[] = [];
  samples: PointerSample[] = [];
  light = true;
  showPointer = false;
  /** The elements of the state React is showing, by id (for the "talked about" highlight). */
  elements = new Map<string, SlideElement>();
  onState?: (i: number) => void;
  dirty = true;

  private imgs = new Map<number, HTMLImageElement>();
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
    const { states } = this;

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

// Memoised: the player re-renders on every spoken word, the slides only need to when their props change.
export const Slides = memo(
  forwardRef<SlidesHandle, Props>(function Slides(props, ref) {
    const { jobId, slides, timeRef, clipStart, playing, showPointer, onSeek, onTogglePlay, onAsk } = props;
    const [W, H] = slides.size;
    const u = unitOf(slides.size);
    const stageRef = useRef<HTMLDivElement>(null);
    const clipRef = useRef<HTMLDivElement>(null);
    const tipRef = useRef<HTMLDivElement>(null);
    const annRef = useRef<AnnotationLayerHandle>(null);
    const pointerRef = useRef({ x: 0, y: 0 }); // last mouse position in stage CSS pixels
    const [anim] = useState(() => new SlideAnimator());

    const states = useMemo(() => sortStates(slides.states ?? []), [slides]);
    const annotations = slides.annotations ?? NO_ANNOTATIONS;
    // An annotation never outlives its slide, and an arrow points at its slide's elements.
    const stateIndex = useMemo(() => new Map(states.map((s, i) => [s.id, i])), [states]);
    const annSlide = useCallback(
      (a: SlideAnnotation) => (a.state != null ? (stateIndex.get(a.state) ?? -1) : -1),
      [stateIndex],
    );
    const annEnd = useCallback((a: SlideAnnotation) => slideEndAt(states, annSlide(a)), [states, annSlide]);
    const annBox = useCallback(
      (a: SlideAnnotation, id: string) => states[annSlide(a)]?.elements?.find((e) => e.id === id)?.box ?? null,
      [states, annSlide],
    );
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
    const [lit, setLit] = useState<{ key: number; boxes: Box[] } | null>(null);
    const litTimer = useRef<number | undefined>(undefined);
    useEffect(() => () => window.clearTimeout(litTimer.current), []);

    anim.states = states;
    anim.samples = samples;
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

    useImperativeHandle(
      ref,
      () => ({
        render(t: number) {
          anim.render(t);
          annRef.current?.render(t);
        },
        highlight(ids: string[]) {
          const boxes: Box[] = [];
          for (const id of ids) {
            for (const st of states) {
              const el = st.elements?.find((e) => e.id === id);
              if (el) {
                boxes.push(el.box);
                break;
              }
            }
          }
          window.clearTimeout(litTimer.current);
          if (!boxes.length) return setLit(null);
          setLit({ key: Date.now(), boxes });
          litTimer.current = window.setTimeout(() => setLit(null), HIGHLIGHT_MS);
        },
      }),
      [anim, states],
    );

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
      if (el && e.shiftKey && onAsk) {
        onAsk(el); // Shift+click: attach it to a question
        return;
      }
      if (el) {
        onSeek(elementSeekTime(el, state, clipStart));
        return;
      }
      setFlash({ key: Date.now(), play: !playing });
      onTogglePlay();
    };

    const tipLatex = hover?.latex?.trim() ?? '';
    const tipHtml = useMemo(() => tipTex(tipLatex), [tipLatex]);
    const rawText = hover?.text?.trim() ?? '';
    const tipText = tipHtml && sameReading(rawText, tipLatex) ? '' : rawText || (tipHtml ? '' : tipLatex);
    const spokenHover = !!hover?.spoken?.some((s) => Number.isFinite(s.start));
    const tipTime = hover ? elementSeekTime(hover, state, clipStart) : 0;
    const hoverStroke = light ? 'rgba(27, 24, 16, 0.78)' : 'rgba(237, 231, 216, 0.85)';
    const hoverFill = light ? 'rgba(27, 24, 16, 0.06)' : 'rgba(237, 231, 216, 0.09)';

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

            <AnnotationLayer
              ref={annRef}
              annotations={annotations}
              size={slides.size}
              timeRef={timeRef}
              colourOf={slideColour}
              endOf={annEnd}
              boxOf={annBox}
            >
              {/* the hover box sits above the spotlight's veil and below the annotations */}
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
            </AnnotationLayer>

            <g data-pointer="" display="none" opacity={0}>
              <circle data-fill="" r={10 * u} opacity={0.22} />
              <circle data-fill="" r={4.5 * u} stroke="#fff" strokeOpacity={0.85} strokeWidth={1.3 * u} />
            </g>
          </svg>

          {lit && (
            <div key={lit.key} className="ask-lit-layer" aria-hidden>
              {lit.boxes.map(([x, y, w, h], i) => (
                <div
                  key={i}
                  className="ask-lit"
                  style={{
                    left: `${((x - 6 * u) / W) * 100}%`,
                    top: `${((y - 5 * u) / H) * 100}%`,
                    width: `${((w + 12 * u) / W) * 100}%`,
                    height: `${((h + 10 * u) / H) * 100}%`,
                  }}
                />
              ))}
            </div>
          )}
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
