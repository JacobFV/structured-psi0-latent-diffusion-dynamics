import { LensView, type LensDef } from './Lens';

export const LIBRARY: LensDef[] = [
  { id: 'knowledge', title: 'Decisions, roadmap, docs', load: () => import('./Knowledge') },
  { id: 'overview', title: 'Claims and caveats', load: () => import('./Overview') },
];
export default function Library() {
  return <LensView lenses={LIBRARY} fallback="knowledge" />;
}
