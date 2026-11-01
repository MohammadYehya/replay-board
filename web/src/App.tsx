import { useEffect, useState } from 'react';
import { HomePage } from './pages/HomePage';
import { JobPage } from './pages/JobPage';
import { NotesPage } from './pages/NotesPage';

type Route = { page: 'home' } | { page: 'job'; id: string; t: number | null } | { page: 'notes'; id: string };

// #/job/<id>, #/job/<id>?t=<seconds> (open the player at a moment), #/job/<id>/notes
function parse(hash: string): Route {
  const m = hash.match(/^#\/job\/([^/?#]+)(\/notes)?(?:\?(.*))?$/);
  if (m) {
    try {
      const id = decodeURIComponent(m[1]);
      if (m[2]) return { page: 'notes', id };
      const t = Number(new URLSearchParams(m[3] ?? '').get('t'));
      return { page: 'job', id, t: m[3] && Number.isFinite(t) && t >= 0 ? t : null };
    } catch {
      return { page: 'home' };
    }
  }
  return { page: 'home' };
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => {
      setRoute(parse(window.location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  useEffect(() => {
    document.title = route.page === 'home' ? 'Replay Board' : `${route.id} · Replay Board`;
  }, [route]);
  // A file dropped outside the upload page's handler must never replace the app with the file itself.
  useEffect(() => {
    const block = (e: DragEvent) => {
      if (e.target instanceof HTMLInputElement && e.target.type === 'file') return;
      if (e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files')) e.preventDefault();
    };
    window.addEventListener('dragover', block);
    window.addEventListener('drop', block);
    return () => {
      window.removeEventListener('dragover', block);
      window.removeEventListener('drop', block);
    };
  }, []);
  if (route.page === 'notes') return <NotesPage key={`n-${route.id}`} jobId={route.id} />;
  return route.page === 'job' ? <JobPage key={route.id} jobId={route.id} startAt={route.t} /> : <HomePage />;
}
