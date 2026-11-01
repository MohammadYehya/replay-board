/*
 * Replay Board player (React + Vite + TypeScript + KaTeX).
 *
 *   Develop:  .venv\Scripts\python -m app.server      (backend on http://127.0.0.1:8000)
 *             cd web; npm install; npm run dev         (player on http://localhost:5173, /api is proxied)
 *   Build:    cd web; npm run build                    (tsc strict + vite build -> web/dist, served by app.server at /)
 *   Test:     cd web; npm test                         (timing rules, plain Node)
 *
 * Routes (hash): #/ upload + lecture list, #/job/<id> progress then player.
 * The video plays untouched; the board is redrawn on a canvas beside it from replay.json (SPEC.md section 6).
 * The video's currentTime is the one clock: a requestAnimationFrame loop renders board + transcript from it.
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'katex/dist/katex.min.css';
import './styles.css';
import { App } from './App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
