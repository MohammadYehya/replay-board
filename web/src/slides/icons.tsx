// Icons for the slides export.

const line = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.3, strokeLinecap: 'round', strokeLinejoin: 'round' } as const;

/** A presentation screen on a stand: the Slides page. */
export function SlidesIcon({ size = 13 }: { size?: number }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden>
      <path d="M1.8 2.6h12.4M2.8 2.6v7.6h10.4V2.6M8 10.2v2.4M5.6 14.2L8 12.6l2.4 1.6M5.2 7.6l2-2.2 1.7 1.5 2.1-2.4" {...line} />
    </svg>
  );
}

/** Full screen: Present. */
export function PresentIcon({ size = 13 }: { size?: number }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden>
      <path d="M6.2 4.6v6.8l5.4-3.4z" fill="currentColor" />
      <path d="M1.8 5V2.6a.8.8 0 01.8-.8H5M11 1.8h2.4a.8.8 0 01.8.8V5M14.2 11v2.4a.8.8 0 01-.8.8H11M5 14.2H2.6a.8.8 0 01-.8-.8V11" {...line} />
    </svg>
  );
}

export function PlayIcon({ size = 10 }: { size?: number }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden>
      <path d="M4 2.4v11.2L13.2 8z" fill="currentColor" />
    </svg>
  );
}
