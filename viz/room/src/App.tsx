import { lazy, Suspense, useEffect, useState, type ComponentType } from 'react';
import { SidebarContext } from './components/shell';
import Ticker from './components/Ticker';
import { Loading } from './components/ui';
import { forceFixture, useMeta } from './lib/api';
import { go, href, useHashView } from './lib/url';

// IBM-2 shell: left sidebar (brand, view picker, that view's controls), one workspace, no top tabs.
const views: { id: string; key: string; title: string; load: () => Promise<{ default: ComponentType }> }[] = [
  { id: 'board', key: '1', title: 'Board', load: () => import('./views/Board') },
  { id: 'evidence', key: '2', title: 'Evidence', load: () => import('./views/Evidence') },
  { id: 'runs', key: '3', title: 'Runs', load: () => import('./views/RunHistory') },
  { id: 'ops', key: '4', title: 'Ops', load: () => import('./views/Ops') },
  { id: 'library', key: '5', title: 'Library', load: () => import('./views/Library') },
];
// old addresses stay valid: each maps to the consolidated view that now holds it (as a lens)
const aliases: Record<string, string> = {
  results: 'evidence', edits: 'evidence', robustness: 'evidence', physics: 'evidence', psi0: 'evidence',
  theatre: 'runs', live: 'ops', training: 'ops', knowledge: 'library', overview: 'library',
};
const lazyViews = Object.fromEntries(views.map((v) => [v.id, lazy(v.load)]));

type Theme = 'light' | 'dark';
function initialTheme(): Theme {
  try {
    const s = localStorage.getItem('rrp-room-theme');
    if (s === 'light' || s === 'dark') return s;
  } catch { /* storage may be unavailable */ }
  return 'dark';
}

export default function App() {
  const raw = useHashView('board');
  const view = aliases[raw] ?? raw;
  const current = views.find((v) => v.id === view) || views[0];
  const View = lazyViews[current.id];
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [menu, setMenu] = useState(false);
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  const meta = useMeta();
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('rrp-room-theme', theme); } catch { /* ignore */ }
  }, [theme]);
  useEffect(() => { document.title = `rrp · ${current.title}`; setMenu(false); }, [current.title]);
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.key === 'Escape') setMenu(false);
      if (e.ctrlKey || e.metaKey || e.altKey || ['INPUT', 'SELECT', 'TEXTAREA'].includes(t?.tagName)) return;
      const v = views.find((x) => x.key === e.key);
      if (v) { e.preventDefault(); go(v.id); return; }
      if (e.key === '/') {
        const el = document.querySelector<HTMLElement>('.app-sidebar input[type=search], .app-sidebar select:not(.view-select)');
        if (el) { e.preventDefault(); setMenu(true); el.focus(); }
      }
    };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, []);
  const dataMode = meta === undefined ? '…' : forceFixture() ? 'FIXTURES (URL)' : meta === null ? 'static snapshot' : meta.exporter_available ? 'live API' : 'FIXTURES (no exporter)';
  return (
    <SidebarContext.Provider value={{ element: slot, closeMenu: () => setMenu(false) }}>
      <div className="app-shell" data-menu={menu ? 'open' : 'closed'}>
        <a className="skip-link" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus(); }}>Skip to content</a>
        <header className="mobile-bar">
          <button aria-label="Open controls" aria-expanded={menu} aria-controls="app-sidebar" onClick={() => setMenu(true)}>☰</button>
          <strong>rrp room</strong><span>{current.title}</span>
        </header>
        <button className="sidebar-backdrop" aria-label="Close controls" tabIndex={-1} onClick={() => setMenu(false)} />
        <aside id="app-sidebar" className="app-sidebar">
          <div className="sidebar-brand">
            <a href={href('board')} className="brand-mark">rrp</a>
            <select className="view-select" aria-label="View" value={current.id} onChange={(e) => go(e.target.value)}>
              {views.map((v) => <option key={v.id} value={v.id}>{v.key} · {v.title}</option>)}
            </select>
            <button className="sidebar-close" aria-label="Close controls" onClick={() => setMenu(false)}>×</button>
          </div>
          <section className="sidebar-controls" ref={setSlot} aria-label={`${current.title} controls`} />
          <footer className="sidebar-foot">
            <span title={meta ? `exporter ${meta.exporter}; cache ${meta.cache_s}s, live ${meta.live_cache_s}s` : ''}>data: {dataMode}</span>
            <span>{typeof window !== 'undefined' ? window.location.host : ''} · read-only</span>
            <span title="1–5 views · / filter · [ ] previous/next run · space play">keys 1–5 · / · [ ]</span>
            <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? 'light theme' : 'dark theme'}</button>
          </footer>
        </aside>
        <main id="main" tabIndex={-1} className={`workspace workspace-${current.id}`}>
          <Ticker />
          <Suspense fallback={<div className="view"><Loading what={current.title} /></div>}>
            <div className="view"><View key={current.id} /></div>
          </Suspense>
        </main>
      </div>
    </SidebarContext.Provider>
  );
}
