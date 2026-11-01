// Loads the 3D pop-out (Popout.tsx with three.js) only when it is needed, so the player's main bundle stays small.
// Hovering a piece that has a 3D view starts the download, so the click usually finds it ready.
import { lazy, type ComponentProps, type ComponentType } from 'react';

const load = () => import('./Popout');

let started = false;
export function preloadPopout() {
  if (started) return;
  started = true;
  load().catch(() => {
    started = false; // try again on the next hover or click
  });
}

function PopoutUnavailable({ onClose }: { onClose(): void }) {
  return (
    <div className="popout-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="popout small" role="alertdialog" aria-label="3D view">
        <div className="popout-state error">
          <p>The 3D viewer could not be loaded. Is the server still running?</p>
          <button className="ghost-btn" onClick={onClose} autoFocus>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

type PopoutProps = ComponentProps<(typeof import('./Popout'))['default']>;

export const LazyPopout = lazy<ComponentType<PopoutProps>>(() => load().catch(() => ({ default: PopoutUnavailable })));

/** Shown for the moment the chunk takes to arrive. */
export function PopoutFallback() {
  return (
    <div className="popout-backdrop">
      <div className="popout" role="dialog" aria-busy="true" aria-label="3D view">
        <div className="popout-stage">
          <div className="popout-state" role="status">
            <span className="popout-spinner" aria-hidden />
            <p>Opening the 3D view...</p>
          </div>
        </div>
      </div>
    </div>
  );
}
