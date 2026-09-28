/** Overview (IBM-2's first view): the one-screen board, and in the same view a compact reader for decisions, roadmap,
 * backlog, strategy, STATUS, claims and docs, picked from one sidebar list (no separate Library). */
import { lazy, Suspense, type ComponentType } from 'react';
import { SideGroup, SidebarControls } from '../components/shell';
import { Loading } from '../components/ui';
import { href, readParam, useHashView } from '../lib/url';
import type { LensDef } from './Lens';

export const OVERVIEW: LensDef[] = [
  { id: 'overview', title: 'Board', load: () => import('./Board') },
  { id: 'knowledge', title: 'Knowledge', load: () => import('./Knowledge') },
  { id: 'claims', title: 'Claims and caveats', load: () => import('./Overview') },
];
declare global { var __RRP_LENS__: Record<string, ComponentType> | undefined }
const cache = new Map<string, ComponentType>();
function comp(l: LensDef) {
  if (!cache.has(l.id)) cache.set(l.id, lazy(l.load));
  return globalThis.__RRP_LENS__?.[l.id] ?? cache.get(l.id)!;
}
const ITEMS: [string, string, Record<string, string>?][] = [
  ['overview', 'Board'], ['knowledge', 'Decisions', { tab: 'decisions' }], ['knowledge', 'Roadmap', { tab: 'roadmap' }],
  ['knowledge', 'Backlog', { tab: 'backlog' }], ['knowledge', 'Strategy', { tab: 'strategy' }], ['knowledge', 'STATUS', { tab: 'status' }],
  ['knowledge', 'D ↔ P crosswalk', { tab: 'crosswalk' }], ['claims', 'Claims and caveats'], ['knowledge', 'Docs', { tab: 'docs' }],
];
export default function OverviewView() {
  const raw = useHashView('overview');
  const lens = OVERVIEW.find((l) => l.id === raw) || OVERVIEW[0];
  const tab = readParam('tab') || 'decisions';
  const View = comp(lens);
  return (
    <>
      <SidebarControls>
        <SideGroup title="Overview">
          <div className="side-list">
            {ITEMS.map(([id, label, q]) => {
              const on = id === lens.id && (!q || q.tab === tab);
              return <button key={label} aria-pressed={on} onClick={() => { window.location.hash = href(id, q).slice(1); }}>{label}</button>;
            })}
          </div>
        </SideGroup>
      </SidebarControls>
      <Suspense fallback={<Loading what={lens.title} />}><View key={lens.id} /></Suspense>
    </>
  );
}
