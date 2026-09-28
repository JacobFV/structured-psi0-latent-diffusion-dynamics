import { LensView, type LensDef } from './Lens';

export const TRAINING: LensDef[] = [
  { id: 'training', title: 'Training curves', load: () => import('./Training') },
  { id: 'live', title: 'Peer, leases, DAGs', load: () => import('./LiveOps') },
];
export default function TrainingView() {
  return <LensView lenses={TRAINING} fallback="training" />;
}
