import { useEffect, useState } from 'react';
import { HomePage } from './pages/HomePage';
import { JobPage } from './pages/JobPage';

type Route = { page: 'home' } | { page: 'job'; id: string };

function parse(hash: string): Route {
  const m = hash.match(/^#\/job\/([^/?#]+)/);
  if (m) {
    try {
      return { page: 'job', id: decodeURIComponent(m[1]) };
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
    document.title = route.page === 'job' ? `${route.id} · Replay Board` : 'Replay Board';
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
  return route.page === 'job' ? <JobPage key={route.id} jobId={route.id} /> : <HomePage />;
}
