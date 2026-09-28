import { LensView, type LensDef } from './Lens';

export const OPS: LensDef[] = [
  { id: 'live', title: 'Peer, leases, DAGs', load: () => import('./LiveOps') },
  { id: 'training', title: 'Training health', load: () => import('./Training') },
];
export default function Ops() {
  return <LensView lenses={OPS} fallback="live" />;
}
