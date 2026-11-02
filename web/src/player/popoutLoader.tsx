// Loads the 3D view (View3D.tsx with three.js) only when it is needed, so the player's main bundle stays small.
// Hovering a drawing that has a 3D view starts the download, so the click usually finds it ready.
import { lazy, type ComponentProps, type ComponentType } from 'react';

const load = () => import('./View3D');

let started = false;
export function preloadView3D() {
  if (started) return;
  started = true;
  load().catch(() => {
    started = false; // try again on the next hover or click
  });
}

function View3DUnavailable({ onClose }: { onClose(): void }) {
  return (
    <div className="v3" role="region" aria-label="3D view">
      <div className="v3-stage v3-small">
        <div className="popout-state error" role="alert">
          <p>The 3D viewer could not be loaded. Is the server still running?</p>
          <button className="ghost-btn" onClick={onClose}>
            Back to board
          </button>
        </div>
      </div>
    </div>
  );
}

type View3DProps = ComponentProps<(typeof import('./View3D'))['default']>;

export const LazyView3D = lazy<ComponentType<View3DProps>>(() =>
  load().catch(() => ({ default: View3DUnavailable as ComponentType<View3DProps> })),
);

/** Shown for the moment the chunk takes to arrive. */
export function View3DFallback() {
  return (
    <div className="v3" role="region" aria-label="3D view" aria-busy="true">
      <div className="v3-stage v3-small">
        <div className="popout-state" role="status">
          <span className="popout-spinner" aria-hidden />
          <p>Opening the 3D view...</p>
        </div>
      </div>
    </div>
  );
}
