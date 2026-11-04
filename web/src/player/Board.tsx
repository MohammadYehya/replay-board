import {
  forwardRef, memo, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState,
  type CSSProperties, type RefObject,
} from 'react';
import type { BoardData, BoardLine, Box, Piece, SlideAnnotation } from '../types';
import { AnnotationLayer, type AnnotationLayerHandle, type Veil } from './AnnotationLayer';
import { BoardRenderer, boardBackground, lineKey } from './boardRenderer';
import { boardAnnotations, boardArrowBoxes, targetGoneAt, toolColour } from './slideTiming';
import { texHtml } from './tex';
import { cleanLatex } from './texClean';
import { texLayouts, writtenText } from './texLayout';
import { formatTime, isWrittenLine, schedule, seekTarget, typesetIds } from './timing';
import { badgePlaces, boxOverlap, canPopout, drawingOnBoard, explainedAt, type BadgeSpot, type MadeScene } from './popoutable';
import { preloadView3D } from './popoutLoader';
import { labelText } from './sceneText';
import { SliceQueue } from './sliceQueue';
import type { SpriteCache } from './sprites';
import { kindLabel, sameReading, tipTex } from './tip';

export interface BoardHandle {
  render(t: number): void;
  /** Outline these pieces (numbers) or lines ("L3") for a moment: an answer of Ask Muse pointing at them. */
  highlight(ids: (number | string)[]): void;
}

interface Props {
  jobId: string;
  board: BoardData;
  typeset: boolean;
  timeRef: RefObject<number>;
  clipStart: number;
  playing: boolean;
  onSeekPiece(p: Piece): void;
  onTogglePlay(): void;
  /** Open the 3D view (in place of the board) for a drawing or a y = f(x) equation (board lectures only). */
  onPopout?(p: Piece): void;
  /** Attach a piece to an Ask Muse question: its whole line when that line is finished by now (else line is null). */
  onAsk?(p: Piece, line: BoardLine | null): void;
  /** Drawings whose 3D view is made already: a "3D" badge on each (once it is on the board), and "Watch in 3D" while
   *  the lecture explains it. */
  made?: MadeScene[];
  /** The lecture's decoded sprites, shared by every renderer of it: a replay swapped in while the lecture streams
   *  is drawn whole at once instead of blanking while its images load again. */
  sprites?: SpriteCache;
  /** The replay is partial (the lecture is still processed): an empty board is not "no writing" yet. */
  partial?: boolean;
  /** A calm line over the board while the playhead is past what is ready ("Board catching up: ready up to 3:20"). */
  catchUp?: string | null;
  /** The job failed: the line says the board stops there (no spinner, the error colour). */
  catchUpFailed?: boolean;
  /** While set, the 3D buttons and badges are shown disabled with this title (the replay is not final yet). */
  popoutLock?: string | null;
}

const HIGHLIGHT_MS = 2400;
/** The "3D" badge's and the "Watch in 3D" pill's sizes (CSS pixels, measured at 1366 px; a little smaller on a phone)
 *  and their gap to the drawing: the board finds each a place this big (styles.css .b3d). */
const PILL_PX: [number, number] = [104, 23];
const BADGE_PX: [number, number] = [46, 20];
const BADGE_GAP_PX = 3;

/** The typeset readings waiting to be fitted (by their inner element), fitted a slice at a time between frames. */
const fitQueue = new SliceQueue<HTMLElement>();

/** Top of a text block th high in a box h high, centred on the ink's centre cy (0..1) and kept inside the box. */
const placeTop = (cy: number, h: number, th: number) => Math.min(Math.max(0, cy * h - th / 2), Math.max(0, h - th));

/** Largest font size (board px, at most maxFont) at which the content fits the piece's box. */
function fitText(inner: HTMLElement, w: number, h: number, maxFont: number): number {
  let lo = 2;
  let hi = Math.max(6, Math.min(h * 1.25, maxFont));
  let best = lo;
  for (let i = 0; i < 13; i++) {
    const mid = (lo + hi) / 2;
    inner.style.fontSize = `${mid}px`;
    if (inner.scrollWidth <= w + 0.5 && inner.scrollHeight <= h + 0.5) {
      best = mid;
      lo = mid;
    } else hi = mid;
  }
  inner.style.fontSize = `${best}px`;
  return best;
}

interface Candidate {
  html: string;
  className: string;
  lines: number;
  penalty: number; // multiplies the fitted size when choosing between candidates
}

/** The spotlight's veil: a soft dark veil on a light board (black would look like the lights went out), the
 *  slides' half-black on chalk. */
const VEIL_LIGHT: Veil = { colour: '#1b1810', opacity: 0.3 }; // warm ink, like the page
const VEIL_DARK: Veil = { colour: '#000', opacity: 0.5 };

/** When a piece starts showing on the board (an arrow keeps off what is there by then). */
const pieceStart = (p: Piece) => schedule(p).start;

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

/** Ways to typeset a reading: LaTeX broken only between clauses and before relations, KaTeX's own wrapping as a
 *  last resort, and the plain text when KaTeX can't parse the LaTeX. */
function candidatesFor(latex: string, text: string): Candidate[] {
  const out: Candidate[] = [];
  const clean = latex ? cleanLatex(latex) : '';
  for (const layout of texLayouts(clean)) {
    const parts = layout.map((l) => texHtml(l.tex));
    if (parts.some((p) => p == null)) continue;
    out.push({
      html: parts.map((p, k) => `<div class="ts-line${layout[k].cont ? ' cont' : ''}">${p}</div>`).join(''),
      className: 'ts-inner',
      lines: layout.length,
      penalty: 1 - 0.03 * (layout.length - 1),
    });
  }
  const whole = clean ? texHtml(clean) : null;
  if (whole) out.push({ html: `<div class="ts-line">${whole}</div>`, className: 'ts-inner wrap', lines: 1, penalty: 0.78 });
  if (!out.length && text) out.push({ html: escapeHtml(text), className: 'ts-inner wrap ts-text', lines: 1, penalty: 1 });
  return out;
}

/** Indent each continuation line ("= ...") so its relation sits under the first relation of the line above. */
function alignContinuations(inner: HTMLElement) {
  const size = parseFloat(inner.style.fontSize) || 16;
  let anchor = 0;
  for (const line of Array.from(inner.children) as HTMLElement[]) {
    if (!line.classList.contains('cont')) {
      line.style.paddingLeft = '';
      const rel = line.querySelector<HTMLElement>('.base > .mrel');
      anchor = 0;
      for (let el: HTMLElement | null = rel; el && el !== line; el = el.offsetParent as HTMLElement | null) {
        anchor += el.offsetLeft;
      }
      if (!rel) anchor = 0;
    } else line.style.paddingLeft = `${(anchor / size).toFixed(3)}em`;
  }
}

const TypesetPiece = memo(function TypesetPiece(props: {
  piece: Piece;
  ink: string;
  cy: number;
  maxFont: number;
  minFont: number;
  fontsVersion: number;
  register(id: number, el: HTMLElement | null): void;
  onFit(id: number, legible: boolean): void;
}) {
  const { piece, ink, cy, maxFont, minFont, fontsVersion, register, onFit } = props;
  const innerRef = useRef<HTMLDivElement>(null);
  const [x, y, w, h] = piece.box;
  const latex = piece.read?.latex?.trim() ?? '';
  const text = writtenText(piece.read?.text ?? '') || latex;
  const candidates = useMemo(() => candidatesFor(latex, text), [latex, text]);
  const registerEl = useCallback((el: HTMLDivElement | null) => register(piece.id, el), [register, piece.id]);
  // The fitted text block's height, and where the ink's centre is: the block is placed again when the centre arrives
  // (with the piece's image) without fitting it again.
  const blockH = useRef(0);
  const cyRef = useRef(cy);
  cyRef.current = cy;
  const fitted = useRef(false);
  useLayoutEffect(() => {
    const inner = innerRef.current;
    if (!inner) return;
    // Not legible until fitted: the handwriting stands in meanwhile (typeset mode is off when the player opens).
    if (!fitted.current) onFit(piece.id, false);
    // Fitting forces a few layouts per reading: it runs in slices between frames (fitQueue), not in the commit that
    // first shows the player.
    fitQueue.add(inner, () => fit(inner));
    return () => fitQueue.cancel(inner);
  }, [piece.id, w, h, candidates, maxFont, minFont, fontsVersion, onFit]);
  useLayoutEffect(() => {
    const inner = innerRef.current;
    if (inner && blockH.current) inner.style.top = `${placeTop(cy, h, blockH.current)}px`;
  }, [cy, h]);
  function fit(inner: HTMLDivElement) {
    // The inner element is filled here, not by React: try each layout and keep the one that fits largest.
    // Unwrapped layouts scale linearly with the font size, so one measurement at a reference size is enough;
    // only wrapping text needs the (costlier) search.
    const REF = 40;
    const cap = Math.max(6, Math.min(h * 1.25, maxFont));
    const load = (c: Candidate) => {
      inner.className = c.className;
      inner.innerHTML = c.html;
      inner.style.fontSize = `${REF}px`;
      alignContinuations(inner);
    };
    const wraps = (c: Candidate) => c.className.includes('wrap');
    const sizeOf = (c: Candidate): number => {
      load(c);
      if (wraps(c)) return fitText(inner, w, h, maxFont);
      inner.style.width = 'max-content';
      const cw = inner.offsetWidth;
      const ch = inner.offsetHeight;
      inner.style.width = `${w}px`;
      if (!cw || !ch) return 0;
      return Math.min(cap, REF * Math.min(w / cw, h / ch) * 0.98);
    };
    let best: Candidate | null = null;
    let bestSize = 0;
    let bestScore = -1;
    for (const c of candidates) {
      if (wraps(c) && best && cap * c.penalty <= bestScore + 0.01) continue; // cannot win
      const size = sizeOf(c);
      const score = size * c.penalty;
      if (score > bestScore + 0.01) {
        best = c;
        bestSize = size;
        bestScore = score;
      }
    }
    let size = 0;
    if (best) {
      load(best);
      if (wraps(best)) size = fitText(inner, w, h, maxFont);
      else {
        size = bestSize;
        inner.style.fontSize = `${size}px`;
      }
    }
    // Centre the text block on where the ink actually is (a box can hold more than its main line).
    blockH.current = inner.offsetHeight;
    inner.style.top = `${placeTop(cyRef.current, h, blockH.current)}px`;
    fitted.current = true;
    // Too small to read: keep the handwriting for this piece instead.
    onFit(piece.id, size >= minFont);
  }
  // Opacity, visibility and the left-to-right reveal are set every frame by BoardRenderer.
  return (
    <div className="ts-piece" ref={registerEl} style={{ left: x, top: y, width: w, height: h, color: ink }}>
      <div ref={innerRef} className="ts-inner" style={{ width: w }} />
    </div>
  );
});

// Memoised: the player re-renders on every spoken word, the board only needs to when its props change.
export const Board = memo(
  forwardRef<BoardHandle, Props>(function Board(props, ref) {
    const {
      jobId, board, typeset, timeRef, clipStart, playing, onSeekPiece, onTogglePlay, onPopout, onAsk, made, sprites,
      partial, catchUp, catchUpFailed, popoutLock,
    } = props;
    const [W, H] = board.size;
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const stageRef = useRef<HTMLDivElement>(null);
    const tipRef = useRef<HTMLDivElement>(null);
    const rendererRef = useRef<BoardRenderer | null>(null);
    const typesetEls = useRef(new Map<number, HTMLElement>()).current;
    const tagEls = useRef(new Map<number, HTMLElement>()).current;
    const hoverRef = useRef<Piece | null>(null);
    const pointerRef = useRef({ x: 0, y: 0 }); // last pointer position in stage CSS pixels
    /** The last press was a finger (or a pen), and the piece whose tooltip was open then: a touch screen has no
     *  hover, so a first tap on a piece opens its tooltip (Ask, View in 3D) and a second one plays it. */
    const touchRef = useRef<{ touch: boolean; open: number | null }>({ touch: false, open: null });
    const [cssW, setCssW] = useState(0);
    const [hover, setHover] = useState<Piece | null>(null);
    const [inks, setInks] = useState<Record<number, { ink: string; cy: number }>>({});
    const [fontsVersion, setFontsVersion] = useState(0);
    const [flash, setFlash] = useState<{ key: number; play: boolean } | null>(null);
    const [tipPos, setTipPos] = useState<{ left: number; top: number; below: boolean } | null>(null);
    const [lit, setLit] = useState<{ key: number; boxes: Box[] } | null>(null);
    const { css: background, light } = useMemo(() => boardBackground(board), [board]);
    const defaultInk = light ? '#1f2633' : '#f1f1ea';

    // The teacher's hand pointing at writing, re-drawn like the pointer gestures on slides.
    const annRef = useRef<AnnotationLayerHandle>(null);
    const annotations = useMemo(() => boardAnnotations(board), [board]);
    const annColour = useCallback((a: SlideAnnotation) => toolColour(a.tool, light), [light]);
    // An annotation fades with its writing when that is erased; an arrow points at its pieces' lines and keeps off
    // the other writing on the board.
    const writtenLines = useMemo(() => (board.lines ?? []).filter((ln) => isWrittenLine(ln, W)), [board, W]);
    const annEnd = useCallback((a: SlideAnnotation) => targetGoneAt(a, board.pieces), [board]);
    const annArrow = useCallback(
      (a: SlideAnnotation) => boardArrowBoxes(a, board.pieces, writtenLines, board.size, pieceStart),
      [board, writtenLines],
    );

    const typesetPieces = useMemo(() => {
      const ids = typesetIds(board.pieces, W, H);
      return board.pieces.filter((p) => ids.has(p.id));
    }, [board, W, H]);
    // Lines written in bursts are typeset whole once finished; the renderer decides when (keys < 0).
    const typesetLines = useMemo(
      () =>
        (board.lines ?? [])
          .map((ln, i) => ({ ln, key: lineKey(i) }))
          .filter(
            ({ ln }) => ln.pieces.length >= 2 && (ln.read?.latex?.trim() || ln.read?.text?.trim()) && isWrittenLine(ln, W),
          )
          .map(({ ln, key }) => ({
            first: ln.pieces[0],
            piece: { id: key, box: ln.box, image: '', t_seen: ln.t_complete, t_shown: ln.t_complete, read: ln.read } as Piece,
          })),
      [board, W],
    );
    const hiddenPieces = useMemo(() => board.pieces.filter((p) => p.revealed && !p.pre_existing), [board]);
    const typesetRef = useRef(typeset);
    typesetRef.current = typeset;
    // Which typeset readings fit their box legibly (measured by the typeset layer, possibly before the renderer exists).
    const legible = useRef(new Map<number, boolean>()).current;
    const onFit = useCallback(
      (id: number, ok: boolean) => {
        legible.set(id, ok);
        rendererRef.current?.setTypesetOk(id, ok);
      },
      [legible],
    );

    // The renderer (with its frame cache) is rebuilt only when the board's writing changes: a new version of a
    // streaming replay (more pieces), or the final one. It takes the sprites it had from the shared cache, so the
    // board never blanks. A replay rewritten with refined word times keeps the same pieces and only moves when they
    // are talked about and redrawn: that is handed to the live renderer below instead.
    const latestBoard = useRef(board);
    latestBoard.current = board;
    const structure = useMemo(
      () =>
        [jobId, W, H, board.mode ?? '', board.background ?? '']
          .concat(board.pieces.map((p) => `${p.id}:${p.image}:${p.order ?? ''}:${p.gone ?? ''}:${p.box.join(',')}`))
          .concat((board.lines ?? []).map((ln) => `${ln.id}:${ln.pieces.join(',')}:${ln.box.join(',')}`))
          .join('|'),
      [board, jobId, W, H],
    );
    const spritesRef = useRef(sprites);
    spritesRef.current = sprites;

    useLayoutEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const board = latestBoard.current;
      const r = new BoardRenderer(canvas, board, jobId, {
        sprites: spritesRef.current,
        // the same ink as before (a sprite from the cache) changes nothing
        onSpriteLoaded: (id, s) =>
          setInks((m) => (m[id]?.ink === s.ink && m[id]?.cy === s.cy ? m : { ...m, [id]: { ink: s.ink, cy: s.cy } })),
      });
      r.typesetEls = typesetEls;
      r.tagEls = tagEls;
      r.setTypeset(typesetRef.current);
      for (const [id, ok] of legible) r.setTypesetOk(id, ok);
      rendererRef.current = r;
      const w = stageRef.current?.clientWidth ?? 0;
      if (w) r.resize(w, window.devicePixelRatio || 1);
      r.render(timeRef.current ?? 0);
      return () => {
        r.destroy();
        rendererRef.current = null;
      };
    }, [structure, typesetEls, tagEls, timeRef, legible]); // `structure` stands for board and jobId

    // New timing for the same pieces (spoken ranges, redraw times): reschedule without reloading anything.
    useEffect(() => {
      rendererRef.current?.setPieces(board.pieces);
      rendererRef.current?.render(timeRef.current ?? 0);
      const h = hoverRef.current;
      const fresh = h ? board.pieces.find((p) => p.id === h.id) : undefined;
      if (h && fresh && fresh !== h) {
        hoverRef.current = fresh;
        setHover(fresh);
      }
    }, [board, timeRef]);

    useEffect(() => {
      rendererRef.current?.setTypeset(typeset);
    }, [typeset]);

    useEffect(() => {
      const stage = stageRef.current;
      if (!stage) return;
      const ro = new ResizeObserver(() => {
        const w = stage.clientWidth;
        setCssW(w);
        rendererRef.current?.resize(w, window.devicePixelRatio || 1);
        rendererRef.current?.render(timeRef.current ?? 0); // redraw at once so a resized canvas never shows blank
      });
      ro.observe(stage);
      return () => ro.disconnect();
    }, [timeRef]);

    useEffect(() => {
      const fonts = document.fonts;
      if (!fonts) return;
      // Re-fit the typeset readings once KaTeX's fonts arrive; bursts of font loads are coalesced into one pass.
      let timer: number | undefined;
      let alive = true;
      const bump = () => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => alive && setFontsVersion((v) => v + 1), 250);
      };
      fonts.ready.then(bump);
      fonts.addEventListener('loadingdone', bump);
      return () => {
        alive = false;
        window.clearTimeout(timer);
        fonts.removeEventListener('loadingdone', bump);
      };
    }, []);

    const leaveTimer = useRef<number | undefined>(undefined);
    const cancelLeave = useCallback(() => {
      window.clearTimeout(leaveTimer.current);
      leaveTimer.current = undefined;
    }, []);
    useEffect(() => cancelLeave, [cancelLeave]);

    const changeHover = useCallback((p: Piece | null) => {
      if (hoverRef.current?.id === p?.id) return;
      hoverRef.current = p;
      rendererRef.current?.setHover(p?.id ?? null);
      setHover(p);
    }, []);

    // A tooltip with buttons (Ask, 3D) stays a moment after the pointer leaves its piece, so the pointer can reach it.
    const onPopoutRef = useRef(onPopout);
    onPopoutRef.current = onPopout;
    const onAskRef = useRef(onAsk);
    onAskRef.current = onAsk;
    const hoverTo = (p: Piece | null) => {
      const h = hoverRef.current;
      if (!p && h && (onAskRef.current || (onPopoutRef.current && canPopout(h)))) {
        if (leaveTimer.current === undefined)
          leaveTimer.current = window.setTimeout(() => {
            leaveTimer.current = undefined;
            changeHover(null);
          }, 450);
        return;
      }
      cancelLeave();
      changeHover(p);
    };

    // The drawings with a made 3D view, and their badges (shown, and turned into "Watch in 3D", frame by frame).
    const madeHere = useMemo(
      () =>
        (made ?? [])
          .map((m) => ({ m, p: board.pieces.find((p) => p.id === m.drawing) }))
          .filter((x): x is { m: MadeScene; p: Piece } => !!x.p),
      [made, board],
    );
    const madeRef = useRef(madeHere);
    madeRef.current = madeHere;
    // Where each badge goes: beside its drawing where it covers the least writing, the small badge and the wider
    // "Watch in 3D" pill each (popoutable.ts badgePlaces). The drawing's own writing (its labels too) counts double:
    // the pill shows while the lecture explains it. Worked out again when the board's size changes and as the
    // pieces' images (their ink) arrive.
    const spots = useMemo(() => {
      const r = rendererRef.current;
      const k = cssW > 0 ? W / cssW : 1; // board pixels per CSS pixel
      const px = (s: [number, number]): [number, number] => [s[0] * k, s[1] * k];
      const out = new Map<number, ReturnType<typeof badgePlaces>>();
      const placed: Box[] = []; // badges never sit on each other (two drawings side by side on a phone)
      for (const { m, p } of madeHere) {
        const own = new Set([p.id, ...m.pieces]);
        const ink = (b: Box) =>
          (r ? r.inkIn(...b) + r.inkIn(...b, own) : 0) + 50 * placed.reduce((n, q) => n + boxOverlap(b, q), 0);
        const spot = badgePlaces(p.box, [W, H], px(BADGE_PX), px(PILL_PX), BADGE_GAP_PX * k, ink);
        out.set(p.id, spot);
        placed.push(spot.badge.box, spot.pill.box);
      }
      return out;
    }, [madeHere, cssW, W, H, inks]); // `inks` changes as the pieces' images (their ink) arrive
    const badgeEls = useRef(new Map<number, HTMLElement>()).current;
    const paintBadges = useCallback(
      (t: number) => {
        const r = rendererRef.current;
        if (!r || !badgeEls.size) return;
        const made = new Map(madeRef.current.map((x) => [x.m.drawing, x.m]));
        const seen = (pid: number) => r.isVisible(pid, t);
        const now = new Set(
          explainedAt(madeRef.current.map((x) => x.m), t, (m) => drawingOnBoard(m, seen)).map((m) => m.drawing),
        );
        for (const [id, el] of badgeEls) {
          const m = made.get(id);
          const state = !(m ? drawingOnBoard(m, seen) : seen(id)) ? 'off' : now.has(id) ? 'pill' : 'badge';
          if (el.dataset.state !== state) el.dataset.state = state;
        }
      },
      [badgeEls],
    );
    useEffect(() => paintBadges(timeRef.current ?? 0), [madeHere, paintBadges, timeRef]);

    const litTimer = useRef<number | undefined>(undefined);
    useEffect(() => () => window.clearTimeout(litTimer.current), []);
    useImperativeHandle(
      ref,
      () => ({
        render(t: number) {
          annRef.current?.render(t);
          const r = rendererRef.current;
          if (!r) return;
          const h = hoverRef.current;
          if (h && !r.isVisible(h.id, t)) changeHover(null);
          r.render(t);
          paintBadges(t);
        },
        highlight(ids: (number | string)[]) {
          const b = latestBoard.current;
          const boxes: Box[] = [];
          for (const id of ids) {
            const box =
              typeof id === 'number' || /^\d+$/.test(id)
                ? b.pieces.find((p) => p.id === Number(id))?.box
                : (b.lines ?? []).find((ln) => ln.id === id)?.box;
            if (box) boxes.push(box);
          }
          window.clearTimeout(litTimer.current);
          if (!boxes.length) return setLit(null);
          setLit({ key: Date.now(), boxes });
          litTimer.current = window.setTimeout(() => setLit(null), HIGHLIGHT_MS);
        },
      }),
      [changeHover, paintBadges],
    );

    /** The finished line a piece belongs to at time t (its tooltip reads, and Ask attaches, the whole line). */
    const lineOf = useCallback(
      (p: Piece, t: number): BoardLine | null =>
        (board.lines ?? []).find(
          (ln) =>
            ln.pieces.includes(p.id) &&
            !!(ln.read?.latex?.trim() || ln.read?.text?.trim()) &&
            isWrittenLine(ln, W) &&
            t >= ln.t_complete,
        ) ?? null,
      [board, W],
    );

    const register = useCallback(
      (id: number, el: HTMLElement | null) => {
        if (el) typesetEls.set(id, el);
        else typesetEls.delete(id);
        rendererRef.current?.invalidate();
      },
      [typesetEls],
    );

    const pick = (clientX: number, clientY: number): Piece | null => {
      const canvas = canvasRef.current;
      const r = rendererRef.current;
      if (!canvas || !r) return null;
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      const bx = ((clientX - rect.left) / rect.width) * W;
      const by = ((clientY - rect.top) / rect.height) * H;
      return r.hitTest(bx, by, timeRef.current ?? 0);
    };

    // Place the tooltip above (or below) the hovered piece, kept inside the stage horizontally.
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
        // A piece as tall as the board: put the tip by the pointer, inside the board.
        const p = pointerRef.current;
        const top = Math.min(Math.max(8, p.y + 18), Math.max(8, stageH - th - 8));
        setTipPos({ left: clampX(p.x - tw / 2), top, below: true });
      }
    }, [hover, cssW, W, H, typeset]);

    const onClick = (e: React.MouseEvent) => {
      if (tipRef.current?.contains(e.target as Node)) {
        // a click on the tooltip itself (not its buttons) does what clicking the piece does
        const h = hoverRef.current;
        if (h && e.shiftKey && onAsk) onAsk(h, lineOf(h, timeRef.current ?? 0));
        else if (h) onSeekPiece(h);
        return;
      }
      const p = pick(e.clientX, e.clientY);
      const touch = touchRef.current;
      if (touch.touch && p && touch.open !== p.id && (onAsk || (onPopout && canPopout(p)))) {
        cancelLeave();
        changeHover(p);
        return;
      }
      if (touch.touch && !p && hoverRef.current) {
        changeHover(null); // a tap beside the writing closes its tooltip first
        return;
      }
      if (p && e.shiftKey && onAsk) {
        onAsk(p, lineOf(p, timeRef.current ?? 0)); // Shift+click: attach it to a question
        return;
      }
      if (p) {
        onSeekPiece(p);
        return;
      }
      setFlash({ key: Date.now(), play: !playing });
      onTogglePlay();
    };

    const tipTime = hover ? seekTarget(hover, clipStart) : 0;
    // A fragment of a line that is finished by now reads as its whole line ("#1." -> "A = (1+1)^1 = 2").
    const hoverLine = hover ? lineOf(hover, timeRef.current ?? 0) : null;
    const tipRead = hoverLine?.read ?? hover?.read;
    const tipLatex = tipRead?.latex?.trim() ?? '';
    const tipHtml = useMemo(() => tipTex(tipLatex), [tipLatex]);
    const rawText = tipRead?.text?.trim() ?? '';
    const tipText = tipHtml && sameReading(rawText, tipLatex) ? '' : rawText || (tipHtml ? '' : tipLatex);
    const noSpoken = hover ? !(hover.spoken && hover.spoken.length) : false;
    const can3d = !!onPopout && canPopout(hover);
    const locked3d = popoutLock || undefined;
    useEffect(() => {
      if (can3d && !locked3d) preloadView3D();
    }, [can3d, locked3d]);
    const open3d = (e: React.MouseEvent) => {
      e.stopPropagation();
      const h = hoverRef.current;
      if (!h || !onPopout || locked3d) return;
      cancelLeave();
      changeHover(null);
      onPopout(h);
    };
    const ask = (e: React.MouseEvent) => {
      e.stopPropagation();
      const h = hoverRef.current;
      if (h && onAsk) onAsk(h, lineOf(h, timeRef.current ?? 0));
    };
    const interactive = can3d || !!onAsk;

    return (
      <div
        ref={stageRef}
        className={`board-stage ${hover ? 'is-hovering' : ''}`}
        style={{ aspectRatio: `${W} / ${H}`, background }}
        onPointerMove={(e) => {
          const r = stageRef.current?.getBoundingClientRect();
          if (r) pointerRef.current = { x: e.clientX - r.left, y: e.clientY - r.top };
          if (tipRef.current?.contains(e.target as Node) || (e.target as HTMLElement).closest?.('.b3d')) {
            cancelLeave(); // on the tooltip (or a 3D badge): keep it
            return;
          }
          hoverTo(pick(e.clientX, e.clientY));
        }}
        onPointerDown={(e) => {
          touchRef.current = { touch: e.pointerType === 'touch' || e.pointerType === 'pen', open: hoverRef.current?.id ?? null };
        }}
        onPointerLeave={(e) => {
          // a finger leaves when it lifts: the tooltip it opened stays until the next tap
          if (e.pointerType !== 'touch' && e.pointerType !== 'pen') hoverTo(null);
        }}
        onClick={onClick}
      >
        <canvas ref={canvasRef} className="board-canvas" />

        {/* The clip box has the stage's own size, so the scaled board-pixel layer never widens the page. */}
        <div className="ts-clip">
          <div
            className={`ts-layer ${typeset ? 'on' : ''}`}
            style={{ width: W, height: H, transform: `scale(${cssW ? cssW / W : 0})` }}
            aria-hidden={!typeset}
          >
            {typesetLines.map(({ first, piece }) => (
              <TypesetPiece
                key={piece.id}
                piece={piece}
                ink={inks[first]?.ink ?? defaultInk}
                cy={0.5}
                maxFont={W * 0.019}
                minFont={W * 0.0085}
                fontsVersion={fontsVersion}
                register={register}
                onFit={onFit}
              />
            ))}
            {typesetPieces.map((p) => (
              <TypesetPiece
                key={p.id}
                piece={p}
                ink={inks[p.id]?.ink ?? defaultInk}
                cy={inks[p.id]?.cy ?? 0.5}
                maxFont={W * 0.019} // about the height of handwriting on a filmed board
                minFont={W * 0.0085}
                fontsVersion={fontsVersion}
                register={register}
                onFit={onFit}
              />
            ))}
          </div>
        </div>

        {/* Pointing gestures, over the handwriting and the typeset layer alike, in board pixels. */}
        {annotations.length > 0 && (
          <svg className="ann-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
            <AnnotationLayer
              ref={annRef}
              annotations={annotations}
              size={board.size}
              timeRef={timeRef}
              colourOf={annColour}
              veil={light ? VEIL_LIGHT : VEIL_DARK}
              endOf={annEnd}
              arrowBoxesOf={annArrow}
            />
          </svg>
        )}

        <div className="tag-layer" aria-hidden>
          {hiddenPieces.map((p) => {
            const [x, y, w, h] = p.box;
            const nearTop = y < H * 0.12;
            const rightEdge = x + w > W * 0.85;
            return (
              <div
                key={p.id}
                ref={(el) => {
                  if (el) tagEls.set(p.id, el);
                  else tagEls.delete(p.id);
                }}
                className={`hidden-tag ${nearTop ? 'below' : 'above'} ${rightEdge ? 'from-right' : ''}`}
                style={{
                  left: `${((rightEdge ? x + w : x) / W) * 100}%`,
                  top: `${((nearTop ? y + h : y) / H) * 100}%`,
                }}
              >
                <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden>
                  <path
                    d="M2 2l12 12M6.6 6.7a2 2 0 002.7 2.7M4.2 4.4C2.8 5.3 1.8 6.6 1.3 8c1.1 3 3.8 5 6.7 5 1.3 0 2.5-.4 3.6-1M7 3.1c.3 0 .7-.1 1-.1 2.9 0 5.6 2 6.7 5-.3.9-.8 1.7-1.4 2.4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
                was hidden
              </div>
            );
          })}
        </div>

        {onPopout && madeHere.length > 0 && (
          <div className="b3d-layer">
            {madeHere.map(({ m, p }) => {
              const spot = spots.get(p.id);
              if (!spot) return null;
              /** A spot as CSS: the edge it keeps (its right edge, or its left one beside the drawing) and its top. */
              const place = ({ box: [x, y, w], align }: BadgeSpot) => ({
                left: `${((align === 'right' ? x + w : x) / W) * 100}%`,
                top: `${(y / H) * 100}%`,
                shift: align === 'right' ? 'translateX(-100%)' : 'none',
              });
              const b = place(spot.badge);
              const pl = place(spot.pill);
              const name = labelText(m.title) || 'this drawing';
              return (
                <button
                  key={p.id}
                  ref={(el) => {
                    if (el) badgeEls.set(p.id, el);
                    else badgeEls.delete(p.id);
                  }}
                  className="b3d"
                  data-state="off"
                  data-long={spot.long ? undefined : 'no'}
                  style={
                    {
                      '--b3d-left': b.left,
                      '--b3d-top': b.top,
                      '--b3d-shift': b.shift,
                      '--b3d-pill-left': pl.left,
                      '--b3d-pill-top': pl.top,
                      '--b3d-pill-shift': pl.shift,
                    } as CSSProperties
                  }
                  aria-label={`View in 3D: ${name}`}
                  aria-disabled={locked3d ? true : undefined}
                  title={locked3d ?? `${name}: open its 3D view${m.steps ? `, ${m.steps} steps` : ''}`}
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (locked3d) return;
                    cancelLeave();
                    changeHover(null);
                    onPopout(p);
                  }}
                >
                  <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
                    <path
                      d="M8 1.6l5.6 3.2v6.4L8 14.4l-5.6-3.2V4.8zM2.4 4.8L8 8l5.6-3.2M8 8v6.4"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.4"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span className="b3d-short">3D</span>
                  <span className="b3d-long">Watch in 3D</span>
                </button>
              );
            })}
          </div>
        )}

        {hover && (
          <div
            ref={tipRef}
            className={`board-tip ${tipPos?.below ? 'below' : ''} ${interactive ? 'interactive' : ''}`}
            style={{ left: tipPos?.left ?? 0, top: tipPos?.top ?? 0, visibility: tipPos ? 'visible' : 'hidden' }}
          >
            <div className="tip-head">
              <span className="tip-kind">{kindLabel(tipRead?.kind)}</span>
              {hover.revealed && !hover.pre_existing && <span className="tip-flag">was hidden</span>}
              {(onAsk || can3d) && (
                <span className="tip-btns">
                  {onAsk && (
                    <button
                      className="tip-ask"
                      onClick={ask}
                      title={`Ask Muse about ${hoverLine ? 'this line' : 'this'} (Shift+click)`}
                    >
                      Ask
                    </button>
                  )}
                  {can3d && (
                    <button
                      className="tip-3d"
                      onClick={open3d}
                      aria-disabled={locked3d ? true : undefined}
                      title={locked3d ?? 'View in 3D, explained step by step (made by Muse Spark)'}
                    >
                      3D
                    </button>
                  )}
                </span>
              )}
            </div>
            {tipHtml && <div className="tip-tex" dangerouslySetInnerHTML={{ __html: tipHtml }} />}
            {tipText && <div className={tipHtml ? 'tip-text muted' : 'tip-text'}>{tipText}</div>}
            <div className="tip-foot">
              <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden>
                <path d="M4 3l9 5-9 5z" fill="currentColor" />
              </svg>
              {noSpoken && hover.pre_existing
                ? `Already on the board · click to play from ${formatTime(tipTime)}`
                : `Click to hear it explained · ${formatTime(tipTime)}`}
            </div>
            {can3d && (
              <button
                className="tip-view3d"
                onClick={open3d}
                aria-disabled={locked3d ? true : undefined}
                title={locked3d}
              >
                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
                  <path
                    d="M8 1.6l5.6 3.2v6.4L8 14.4l-5.6-3.2V4.8zM2.4 4.8L8 8l5.6-3.2M8 8v6.4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.3"
                    strokeLinejoin="round"
                  />
                </svg>
                View in 3D
                <small>Muse Spark</small>
              </button>
            )}
          </div>
        )}

        {lit && (
          <div key={lit.key} className="ask-lit-layer" aria-hidden>
            {lit.boxes.map(([x, y, w, h], i) => {
              const pad = Math.max(4, W * 0.006);
              return (
                <div
                  key={i}
                  className="ask-lit"
                  style={{
                    left: `${((x - pad) / W) * 100}%`,
                    top: `${((y - pad) / H) * 100}%`,
                    width: `${((w + 2 * pad) / W) * 100}%`,
                    height: `${((h + 2 * pad) / H) * 100}%`,
                  }}
                />
              );
            })}
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

        {catchUp && (
          <div className={`board-catchup ${catchUpFailed ? 'failed' : ''}`} role="status">
            {catchUp}
          </div>
        )}

        {board.pieces.length === 0 && !partial && <div className="board-empty">No writing was found on this board.</div>}
      </div>
    );
  }),
);
