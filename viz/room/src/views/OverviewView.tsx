import { LensView, type LensDef } from './Lens';

export const OVERVIEW: LensDef[] = [
  { id: 'overview', title: 'Board', load: () => import('./Board') },
  { id: 'knowledge', title: 'Decisions, roadmap, docs', load: () => import('./Knowledge') },
  { id: 'claims', title: 'Claims and caveats', load: () => import('./Overview') },
];
export default function OverviewView() {
  return <LensView lenses={OVERVIEW} fallback="overview" dataToggle={false} />;
}
