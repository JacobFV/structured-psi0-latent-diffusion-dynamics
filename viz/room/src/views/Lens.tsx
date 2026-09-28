/** Consolidated views (owner: five sidebar entries). A lens view shows one of its lenses; the lens picker is in the sidebar. */
import { lazy, Suspense, type ComponentType } from 'react';
import { SideGroup, SidebarControls } from '../components/shell';
import { Loading } from '../components/ui';
import { go, useHashView, useUrlState } from '../lib/url';

export type LensDef = { id: string; title: string; load: () => Promise<{ default: ComponentType }> };
declare global { var __RRP_LENS__: Record<string, ComponentType> | undefined }
const cache = new Map<string, ComponentType>();
function lazyOf(l: LensDef) {
  if (!cache.has(l.id)) cache.set(l.id, lazy(l.load));
  return cache.get(l.id)!;
}

export function useShowData() {
  return useUrlState('data', '0')[0] === '1';
}
export function LensView({ lenses, fallback, dataToggle = true }: { lenses: LensDef[]; fallback: string; dataToggle?: boolean }) {
  const raw = useHashView(fallback);
  const [data, setData] = useUrlState('data', '0');
  const cur = lenses.find((l) => l.id === raw) || lenses.find((l) => l.id === fallback) || lenses[0];
  // render check: lens modules pre-registered so SSR renders the lens instead of the Suspense fallback
  const View = globalThis.__RRP_LENS__?.[cur.id] ?? lazyOf(cur);
  return (
    <>
      <SidebarControls>
        <SideGroup title="Lens">
          <div className="side-list">
            {lenses.map((l) => <button key={l.id} aria-pressed={l.id === cur.id} onClick={() => go(l.id)}>{l.title}</button>)}
          </div>
          {dataToggle && <label className="check"><input type="checkbox" checked={data === '1'} onChange={(e) => setData(e.target.checked ? '1' : '0')} /> show data tables</label>}
        </SideGroup>
      </SidebarControls>
      <Suspense fallback={<Loading what={cur.title} />}>
        <View key={cur.id} />
      </Suspense>
    </>
  );
}
