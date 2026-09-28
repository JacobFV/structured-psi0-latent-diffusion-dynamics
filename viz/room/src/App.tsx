import { lazy, Suspense, useEffect, useState, type ComponentType } from 'react';
import Ticker from './components/Ticker';
import { Loading } from './components/ui';
import { forceFixture, useMeta } from './lib/api';
import { go, href, useHashView } from './lib/url';

const views: { id: string; key: string; title: string; load: () => Promise<{ default: ComponentType }> }[] = [
  { id: 'board', key: '1', title: 'Board', load: () => import('./views/Board') },
  { id: 'live', key: '2', title: 'Live ops', load: () => import('./views/LiveOps') },
  { id: 'results', key: '3', title: 'Results', load: () => import('./views/Results') },
  { id: 'theatre', key: '4', title: 'Theatre', load: () => import('./views/Theatre') },
  { id: 'edits', key: '5', title: 'Edits', load: () => import('./views/Edits') },
  { id: 'training', key: '6', title: 'Training', load: () => import('./views/Training') },
  { id: 'robustness', key: '7', title: 'Robustness', load: () => import('./views/Robustness') },
  { id: 'physics', key: '8', title: 'Physics', load: () => import('./views/Physics') },
  { id: 'psi0', key: '9', title: 'Ψ₀', load: () => import('./views/Psi0') },
  { id: 'knowledge', key: '0', title: 'Knowledge', load: () => import('./views/Knowledge') },
];
// not numbered: the claims/caveats page the board drills into
const extra = [{ id: 'overview', key: '', title: 'Claims', load: () => import('./views/Overview') }];
const all = [...views, ...extra];
const lazyViews = Object.fromEntries(all.map((v) => [v.id, lazy(v.load)]));

type Theme = 'light' | 'dark';
function initialTheme(): Theme {
  try {
    const s = localStorage.getItem('rrp-room-theme');
    if (s === 'light' || s === 'dark') return s;
  } catch { /* storage may be unavailable */ }
  return 'dark'; // dark-first terminal
}

export default function App() {
  const view = useHashView('board');
  const current = all.find((v) => v.id === view) || views[0];
  const View = lazyViews[current.id];
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const meta = useMeta();
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('rrp-room-theme', theme); } catch { /* ignore */ }
  }, [theme]);
  useEffect(() => { document.title = `rrp · ${current.title}`; }, [current.title]);
  // keyboard: 1–0 switch views, / focuses the page's first filter/search, [ ] step replays (theatre handles those)
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.ctrlKey || e.metaKey || e.altKey || ['INPUT', 'SELECT', 'TEXTAREA'].includes(t?.tagName)) return;
      const v = views.find((x) => x.key === e.key);
      if (v) { e.preventDefault(); go(v.id); return; }
      if (e.key === '/') {
        const el = document.querySelector<HTMLElement>('main input[type=search], main .filters select, main select');
        if (el) { e.preventDefault(); el.focus(); }
      }
    };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, []);
  const dataMode = meta === undefined ? '…' : forceFixture() ? 'FIXTURES (URL)' : meta === null ? 'STATIC SNAPSHOT' : meta.exporter_available ? 'API' : 'FIXTURES (no exporter)';
  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href={href('board')} style={{ color: 'inherit', textDecoration: 'none' }}>RRP▮ROOM</a>
        <nav aria-label="Views">
          {views.map((v) => (
            <a key={v.id} href={href(v.id)} aria-current={v.id === current.id ? 'page' : undefined} title={`${v.title} (key ${v.key})`}>
              <kbd>{v.key}</kbd>{v.title}
            </a>
          ))}
        </nav>
        <div className="right">
          <span title={meta ? `exporter ${meta.exporter} · cache ${meta.cache_s}s / live ${meta.live_cache_s}s` : ''}>{dataMode}</span>
          <span>{window.location.host} · read-only</span>
          <span title="keys: 1–0 views · / filter · [ ] previous/next replay · space play">⌨ 1–0 / [ ]</span>
          <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle colour theme">{theme === 'dark' ? 'LIGHT' : 'DARK'}</button>
        </div>
      </header>
      <Ticker />
      <main className="main" id="main">
        <Suspense fallback={<div className="view"><Loading what={current.title} /></div>}>
          <div className="view"><View key={current.id} /></div>
        </Suspense>
      </main>
    </div>
  );
}
