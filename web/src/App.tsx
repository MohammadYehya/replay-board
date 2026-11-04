import { useEffect, useState } from 'react';
import { HomePage } from './pages/HomePage';
import { JobPage } from './pages/JobPage';
import { NotesPage } from './pages/NotesPage';
import { SlidesPage } from './pages/SlidesPage';
import { parseRoute, type Route } from './route';

export function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  useEffect(() => {
    const on = () => {
      setRoute(parseRoute(window.location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  // The tab names the lecture once its page knows the name (JobPage, NotesPage, SlidesPage), never the job's folder
  // id ("zylinder-woche-3-entwurf-84fec595"); until then, the app's.
  const pageKey = route.page === 'home' ? 'home' : `${route.page}/${route.id}`;
  useEffect(() => {
    document.title = 'Replay Board';
  }, [pageKey]);
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
  if (route.page === 'slides') return <SlidesPage key={`s-${route.id}`} jobId={route.id} />;
  return route.page === 'job' ? <JobPage key={route.id} jobId={route.id} startAt={route.t} /> : <HomePage />;
}
