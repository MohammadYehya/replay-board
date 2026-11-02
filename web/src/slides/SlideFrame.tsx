// One slide of the lecture deck as a 16:9 frame, drawn in HTML to look like the PowerPoint (app/pptx_deck.py): a white
// 960 x 540 pt slide, slate text, a blue accent, Calibri. Sizes are points of that slide and scale with the frame
// (container query units), so the slide sorter, Present mode and the printed PDF all use this one component. Like the
// PowerPoint, text starts at its full size and shrinks until the slide's content fits.
import { Fragment, useLayoutEffect, useRef, useState, type CSSProperties, type RefObject } from 'react';
import { jobFileUrl } from '../api';
import { texHtml } from '../player/tex';
import type {
  BulletItem, BulletsBlock, CalloutBlock, Deck, DeckBlock, DeckSlide, EquationBlock, FigureBlock, StepsBlock, TableBlock,
} from '../types';
import {
  agendaLayout, contentLayout, coverFacts, deckSections, eyebrow, formulaGrid, frameBlocks, plainMath, rangeLabel,
  sectionFacts, splitInline, type FrameBlock,
} from './deck';
import { FitQueue, type FitBox } from './fit';

/** Text with inline maths ($...$) typeset by KaTeX; maths KaTeX cannot read stays as written. */
export function Rich({ text }: { text: string }) {
  return (
    <>
      {splitInline(text).map((p, i) => {
        if (!p.math) return <Fragment key={i}>{p.text}</Fragment>;
        const html = texHtml(p.tex);
        return html ? (
          <span key={i} className="sl-m" dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <span key={i} className="sl-m-raw">
            {p.tex}
          </span>
        );
      })}
    </>
  );
}

/** A display formula. It never spills out of its column: the slide's text shrinks first, then it scrolls. */
function Tex({ latex, className = '' }: { latex: string; className?: string }) {
  const html = texHtml(latex, true);
  return html ? (
    <div className={`sl-tex sl-scroll ${className}`} dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <div className={`sl-tex sl-scroll ${className}`}>
      <code className="sl-tex-raw">{latex}</code>
    </div>
  );
}

function Bullets({ block }: { block: BulletsBlock }) {
  // Second-level points go in a list under the point before them.
  const groups: { item: BulletItem; subs: BulletItem[] }[] = [];
  for (const it of block.items) {
    if (it.level === 1 && groups.length) groups[groups.length - 1].subs.push(it);
    else groups.push({ item: it, subs: [] });
  }
  return (
    <ul className="sl-bullets">
      {groups.map((g, i) => (
        <li key={i}>
          <Rich text={g.item.text} />
          {g.subs.length > 0 && (
            <ul>
              {g.subs.map((s, j) => (
                <li key={j}>
                  <Rich text={s.text} />
                </li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ul>
  );
}

function Equation({ block }: { block: EquationBlock }) {
  return (
    <div className="sl-eq">
      {block.label && (
        <p className="sl-label">
          <Rich text={block.label} />
        </p>
      )}
      <Tex latex={block.latex} />
    </div>
  );
}

/** A formula-sheet card: its formula as large as fits the card (28 pt at most), like the PowerPoint's. */
function FormulaCard({ block, fitKey }: { block: EquationBlock; fitKey: string }) {
  const card = useRef<HTMLDivElement>(null);
  useFit(card, fitKey, 28, 10);
  return (
    <div className="sl-eq sl-card-eq" ref={card} style={startSize(28)}>
      {block.label && (
        <p className="sl-label">
          <Rich text={block.label} />
        </p>
      )}
      <Tex latex={block.latex} />
    </div>
  );
}

function Steps({ block }: { block: StepsBlock }) {
  // Formulas and words side by side when the steps have both; else one column.
  const grid = block.items.some((it) => it.latex) && block.items.some((it) => it.text);
  return (
    <div className="sl-steps">
      {block.label && (
        <p className="sl-steps-label">
          <Rich text={block.label} />
        </p>
      )}
      <ol className={grid ? 'grid' : ''}>
        {block.items.map((it, i) => (
          <li key={i}>
            <span className="sl-step-n" aria-hidden="true">
              {i + 1}
            </span>
            {it.latex &&
              (plainMath(it.latex) ? (
                // a step that is just a number: in the body font, like the text beside it
                <p className={grid && it.text ? 'sl-step-eq sl-step-num' : 'sl-step-eq sl-step-num wide'}>
                  {plainMath(it.latex)}
                </p>
              ) : (
                <Tex latex={it.latex} className={grid && it.text ? 'sl-step-eq' : 'sl-step-eq wide'} />
              ))}
            {it.text && (
              <p className={grid && it.latex ? 'sl-step-text' : 'sl-step-text wide'}>
                <Rich text={it.text} />
              </p>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

/** A drawing from the board or a slide. One that cannot be loaded says so in a plain line; keyed by its address,
 *  so a new deck version tries again. */
function Figure({
  jobId,
  version,
  block,
  onImageError,
}: {
  jobId: string;
  version: string;
  block: FigureBlock;
  onImageError?(): void;
}) {
  const src = block.image ? `${jobFileUrl(jobId, block.image)}${version ? `?v=${encodeURIComponent(version)}` : ''}` : '';
  const [broken, setBroken] = useState('');
  const failed = !src || broken === src;
  return (
    <figure className="sl-figure">
      {failed ? (
        <p className="sl-missing">The figure could not be loaded.</p>
      ) : (
        <img
          src={src}
          alt={block.caption || 'A figure from the lecture'}
          width={block.size?.[0]}
          height={block.size?.[1]}
          decoding="async"
          onError={() => {
            setBroken(src);
            onImageError?.();
          }}
        />
      )}
      {block.caption && (
        <figcaption>
          <Rich text={block.caption} />
        </figcaption>
      )}
    </figure>
  );
}

function Table({ block }: { block: TableBlock }) {
  return (
    <div className="sl-table-wrap sl-scroll">
      <table className="sl-table">
        {block.caption && (
          <caption>
            <Rich text={block.caption} />
          </caption>
        )}
        {block.header.length > 0 && (
          <thead>
            <tr>
              {block.header.map((h, i) => (
                <th key={i} scope="col">
                  <Rich text={h} />
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {block.rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j}>
                  <Rich text={c} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const CALLOUT_TAG = { definition: 'Definition', key: 'Key idea', note: 'Note' } as const;

function Callout({ block }: { block: CalloutBlock }) {
  return (
    <div className={`sl-callout ${block.style}`} role="note">
      <p className="sl-callout-tag">{CALLOUT_TAG[block.style]}</p>
      {block.title && (
        <p className="sl-callout-title">
          <Rich text={block.title} />
        </p>
      )}
      {block.text && (
        <p className="sl-callout-text">
          <Rich text={block.text} />
        </p>
      )}
      {block.latex && <Tex latex={block.latex} className="sl-callout-eq" />}
    </div>
  );
}

function Block({
  block,
  jobId,
  version,
  onImageError,
}: {
  block: FrameBlock;
  jobId: string;
  version: string;
  onImageError?(): void;
}) {
  switch (block.type) {
    case 'bullets':
      return <Bullets block={block} />;
    case 'equation':
      return <Equation block={block} />;
    case 'equations':
      // formulas that follow one another: side by side on cards of one height when each fits half the row
      return (
        <div className="sl-eq-row">
          {block.items.map((b, k) => (
            <Equation key={k} block={b} />
          ))}
        </div>
      );
    case 'steps':
      return <Steps block={block} />;
    case 'figure':
      return <Figure jobId={jobId} version={version} block={block} onImageError={onImageError} />;
    case 'table':
      return <Table block={block} />;
    case 'callout':
      return <Callout block={block} />;
  }
}

// ---------- fitting the text ----------

/** A body spills when its content is taller than it, or a formula or table is wider than its column. */
const bodySpills = (el: HTMLElement) =>
  el.scrollHeight > el.clientHeight + 1 ||
  Array.from(el.querySelectorAll<HTMLElement>('.sl-scroll')).some((c) => c.scrollWidth > c.clientWidth + 1);

/** A title box holds its text at the bottom: it spills when the text is taller than the box. */
const titleSpills = (el: HTMLElement) => {
  const text = el.firstElementChild as HTMLElement | null;
  return !!text && text.offsetHeight > el.clientHeight + 1;
};

/** A fitted box starts at its largest size, so a box that fits at once is never laid out twice. */
const startSize = (s: number) => ({ '--s': s }) as CSSProperties;

/** Every fitted box of every frame on the page, fitted together (fit.ts). */
const fitQueue = new FitQueue();

/**
 * The largest text size from `max` down to `min` points at which the element does not spill: set as its --s, before
 * the frame is first painted. The text scales with the frame, so the size found holds at any frame size; it is found
 * again when the element's size changes (a narrow frame grows downwards) or the fonts arrive.
 */
function useFit(
  ref: RefObject<HTMLElement | null>,
  key: string,
  max: number,
  min: number,
  spills: (el: HTMLElement) => boolean = bodySpills,
) {
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    let live = true;
    const box: FitBox = {
      max,
      min,
      // Unchanged sizes are not set again: each change makes the browser lay the slide out anew.
      set: (s) => {
        if (el.style.getPropertyValue('--s') !== String(s)) el.style.setProperty('--s', String(s));
      },
      spills: () => spills(el),
      live: () => live && el.isConnected,
    };
    fitQueue.add(box);
    // Figures change nothing (their box is set by the layout), but KaTeX's fonts load late.
    const sizeOf = () => `${el.clientWidth}x${el.clientHeight}`;
    let size = sizeOf();
    const ro =
      typeof ResizeObserver === 'function'
        ? new ResizeObserver(() => {
            const now = sizeOf();
            if (now === size) return;
            size = now;
            fitQueue.add(box, true);
          })
        : null;
    ro?.observe(el);
    document.fonts?.ready.then(
      () => live && fitQueue.add(box, true),
      () => {},
    );
    return () => {
      live = false;
      ro?.disconnect();
      fitQueue.delete(box);
    };
  }, [ref, key, max, min, spills]);
}

// ---------- the frame ----------

function Footer({ deck, slide, index }: { deck: Deck; slide: DeckSlide; index: number }) {
  const range = rangeLabel(slide.t0, slide.t1);
  return (
    <footer className="sl-foot">
      <span className="sl-foot-name">
        {/* the deck's title: the job's name can be the subject of writing left from an earlier lesson */}
        <span className="sl-foot-lecture">{deck.title}</span>
        {range ? <span className="sl-foot-time">{`  ·  ${range}`}</span> : null}
      </span>
      <span className="sl-num">
        <span className="sr-only">Slide </span>
        {index + 1}
      </span>
    </footer>
  );
}

export interface SlideFrameProps {
  jobId: string;
  deck: Deck;
  slide: DeckSlide;
  index: number;
  /** The id of the slide's heading (the sorter's articles are labelled by it); none in Present mode. */
  headingId?: string;
  onImageError?(): void;
}

/** One slide as a 16:9 frame. */
export function SlideFrame({ jobId, deck, slide, index, headingId, onImageError }: SlideFrameProps) {
  const body = useRef<HTMLDivElement>(null);
  const titleBox = useRef<HTMLDivElement>(null);
  const key = `${deck.version}|${slide.id}|${slide.title}|${slide.blocks.length}|${deck.slides.length}`;
  const kind = slide.kind;
  const big = kind === 'title' || kind === 'section';
  const titleMax = big ? 44 : 32;
  useFit(titleBox, key, titleMax, big ? (kind === 'title' ? 30 : 32) : 20, titleSpills);

  let max = 24;
  let min = 10;
  let content = null;
  if (kind === 'agenda') {
    const lay = agendaLayout(slide, deck.slides);
    max = lay.size;
    min = 12;
    content = (
      <ol
        className={`sl-agenda cols-${lay.columns}`}
        style={{ '--rows': lay.rows, '--row-h': lay.rowHeight } as CSSProperties}
      >
        {lay.entries.map((e, k) => (
          <li key={k} className={k % lay.rows < lay.rows - 1 && k < lay.entries.length - 1 ? 'ruled' : ''}>
            <span className="sl-agenda-n" aria-hidden="true">
              {String(k + 1).padStart(2, '0')}
            </span>
            <span className="sl-agenda-text">
              <Rich text={e.text} />
            </span>
            {e.range && <span className="sl-agenda-time">{e.range}</span>}
          </li>
        ))}
      </ol>
    );
  } else if (kind === 'formulas') {
    const eqs = slide.blocks.filter((b): b is EquationBlock => b.type === 'equation');
    const grid = formulaGrid(eqs.length);
    // The cards have set sizes and fit their own formulas.
    max = 24;
    min = 24;
    content = (
      <div
        className={`sl-formulas cols-${grid.columns}`}
        style={{ '--rows': grid.rows, '--card-h': grid.cardHeight } as CSSProperties}
      >
        {eqs.map((b, k) => (
          <FormulaCard key={k} block={b} fitKey={`${key}|${k}`} />
        ))}
      </div>
    );
  } else if (!big) {
    const lay = contentLayout(slide.blocks);
    const blocks = (list: DeckBlock[]) =>
      frameBlocks(kind, list).map((b, k) => (
        <Block key={k} block={b} jobId={jobId} version={deck.version} onImageError={onImageError} />
      ));
    const aside = lay.figure ?? lay.table;
    if (lay.mode === 'split' && aside) {
      const side = <div className={lay.table ? 'sl-side sl-side-table' : 'sl-side'}>{blocks([aside])}</div>;
      const col = <div className="sl-col">{blocks(lay.main)}</div>;
      const share = `${Math.round(lay.side * 100)}%`;
      content = (
        <div
          className={lay.table ? 'sl-split with-table' : 'sl-split'}
          style={
            {
              gridTemplateColumns: lay.figureFirst
                ? `minmax(0, ${share}) minmax(0, 1fr)`
                : `minmax(0, 1fr) minmax(0, ${share})`,
            } as CSSProperties
          }
        >
          {lay.figureFirst ? side : col}
          {lay.figureFirst ? col : side}
        </div>
      );
    } else if (lay.mode === 'figure' && lay.figure) {
      content = <div className="sl-side sl-alone">{blocks([lay.figure])}</div>;
    } else {
      const withFigure = lay.main.some((b) => b.type === 'figure');
      content = <div className={withFigure ? 'sl-col with-figure' : 'sl-col'}>{blocks(lay.main)}</div>;
    }
  }
  useFit(body, key, max, min);

  if (kind === 'title') {
    return (
      <div className="sl-frame">
        <div className="sl sl-k-title">
          <div className="sl-cover-top">
            <div className="sl-cover-bar" aria-hidden="true" />
            <div className="sl-title-box sl-cover-box" ref={titleBox} style={startSize(titleMax)}>
              <h2 className="sl-title" id={headingId}>
                <Rich text={slide.title} />
              </h2>
            </div>
          </div>
          {deck.subtitle ? (
            <>
              <p className="sl-subtitle">{deck.subtitle}</p>
              <div className="sl-cover-rule" aria-hidden="true" />
              <p className="sl-cover-facts">{coverFacts(deck)}</p>
            </>
          ) : (
            <p className="sl-cover-facts alone">{coverFacts(deck)}</p>
          )}
        </div>
      </div>
    );
  }

  if (kind === 'section') {
    const facts = sectionFacts(deck.slides, slide);
    return (
      <div className="sl-frame">
        <div className="sl sl-k-section">
          <div className="sl-part-wrap">
            {facts.part > 0 && <p className="sl-part">Part {facts.part}</p>}
            <div className="sl-title-box sl-part-box" ref={titleBox} style={startSize(titleMax)}>
              <h2 className="sl-title" id={headingId}>
                <Rich text={slide.title} />
              </h2>
            </div>
            {facts.range && <p className="sl-part-range">{facts.range}</p>}
          </div>
        </div>
      </div>
    );
  }

  const above = eyebrow(slide, deckSections(deck.slides).length);
  return (
    <div className="sl-frame">
      <div className={`sl sl-k-${kind}`}>
        <header className="sl-head">
          <p className="sl-eyebrow">{above}</p>
          <div className="sl-title-box" ref={titleBox} style={startSize(titleMax)}>
            <h2 className="sl-title" id={headingId}>
              <Rich text={slide.title} />
            </h2>
          </div>
          <div className="sl-bar" aria-hidden="true" />
        </header>
        <div className="sl-body" ref={body} style={startSize(max)}>
          {content}
        </div>
        <Footer deck={deck} slide={slide} index={index} />
      </div>
    </div>
  );
}
