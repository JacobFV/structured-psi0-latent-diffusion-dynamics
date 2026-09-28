import { lazy, Suspense, useEffect, useState, type ComponentType } from 'react';
import { Loading } from './components/ui';
import { forceFixture, useMeta } from './lib/api';
import { href, useHashView } from './lib/url';

const views: { id: string; title: string; load: () => Promise<{ default: ComponentType }> }[] = [
  { id: 'overview', title: 'Overview', load: () => import('./views/Overview') },
  { id: 'live', title: 'Live ops', load: () => import('./views/LiveOps') },
  { id: 'results', title: 'Results matrix', load: () => import('./views/Results') },
  { id: 'theatre', title: 'Episode theatre', load: () => import('./views/Theatre') },
  { id: 'edits', title: 'Causal edits', load: () => import('./views/Edits') },
  { id: 'training', title: 'Training', load: () => import('./views/Training') },
  { id: 'robustness', title: 'Robustness', load: () => import('./views/Robustness') },
  { id: 'physics', title: 'Physics credibility', load: () => import('./views/Physics') },
  { id: 'psi0', title: 'Ψ₀ line', load: () => import('./views/Psi0') },
  { id: 'knowledge', title: 'Knowledge', load: () => import('./views/Knowledge') },
];
const lazyViews = Object.fromEntries(views.map((v) => [v.id, lazy(v.load)]));

type Theme = 'light' | 'dark';
function initialTheme(): Theme {
  try {
    const s = localStorage.getItem('rrp-room-theme');
    if (s === 'light' || s === 'dark') return s;
  } catch { /* storage may be unavailable */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const view = useHashView('overview');
  const current = views.find((v) => v.id === view) || views[0];
  const View = lazyViews[current.id];
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const meta = useMeta();
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('rrp-room-theme', theme); } catch { /* ignore */ }
  }, [theme]);
  useEffect(() => { document.title = `rrp room · ${current.title}`; }, [current.title]);
  const dataMode = meta === undefined ? '…' : forceFixture() ? 'FIXTURES (forced by URL)' : meta === null ? 'static snapshot'
    : meta.exporter_available ? 'live API' : 'FIXTURES (exporter not built yet)';
  return (
    <div className="shell">
      <nav className="nav" aria-label="Views">
        <div className="brand"><span className="mark">R</span>rrp room</div>
        {views.map((v, i) => (
          <a key={v.id} href={href(v.id)} aria-current={v.id === current.id ? 'page' : undefined}>
            <span className="idx">{i + 1}</span>{v.title}
          </a>
        ))}
        <div className="foot">
          <div>data: <b className={dataMode.startsWith('FIXTURES') ? 'badge src t-fixture' : ''}>{dataMode}</b></div>
          {meta && <div title={meta.exporter}>exporter <code>rrp.viz.export</code> · cache {meta.cache_s}s / live {meta.live_cache_s}s</div>}
          <div>served at <code>{window.location.host}</code> · read-only</div>
          <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle colour theme">
            {theme === 'dark' ? 'Light theme' : 'Dark theme'}
          </button>
        </div>
      </nav>
      <main className="main" id="main">
        <Suspense fallback={<Loading what={current.title} />}>
          <View key={current.id} />
        </Suspense>
      </main>
    </div>
  );
}
