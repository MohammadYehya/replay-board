// Small icons for Ask Muse: the Muse Spark sparkle, and one per kind of attached item.

export function SparkIcon({ size = 14 }: { size?: number }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden className="spark-icon">
      <path d="M8 1.2l1.7 4.6 4.6 1.7-4.6 1.7L8 13.8 6.3 9.2 1.7 7.5l4.6-1.7z" fill="currentColor" />
      <path d="M13.2 11.2l.6 1.6 1.6.6-1.6.6-.6 1.6-.6-1.6-1.6-.6 1.6-.6z" fill="currentColor" opacity="0.7" />
    </svg>
  );
}

export function KindIcon({ kind }: { kind: string }) {
  const common = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.4, strokeLinecap: 'round', strokeLinejoin: 'round' } as const;
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden className="kind-icon">
      {kind === 'span' ? (
        <path d="M3 6.5v3M5.7 4.5v7M8.4 6v4M11.1 3.5v9M13.6 6.8v2.4" {...common} />
      ) : kind === 'moment' ? (
        <>
          <circle cx="8" cy="8" r="5.6" {...common} />
          <path d="M8 4.9v3.3l2.2 1.4" {...common} />
        </>
      ) : kind === 'element' ? (
        <>
          <rect x="2" y="3" width="12" height="9" rx="1.6" {...common} />
          <path d="M4.6 6h6.8M4.6 8.6h4.4" {...common} />
        </>
      ) : (
        <path d="M3 13l.9-3.2L10.6 3a1.5 1.5 0 012.2 2.1L6.1 11.9z M9.4 4.3l2.2 2.2" {...common} />
      )}
    </svg>
  );
}
