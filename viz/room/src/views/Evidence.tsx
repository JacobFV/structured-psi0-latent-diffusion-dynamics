import { LensView, type LensDef } from './Lens';

export const EVIDENCE: LensDef[] = [
  { id: 'results', title: 'Success matrix', load: () => import('./EvidenceMatrix') },
  { id: 'radar', title: 'Route radar', load: () => import('./Radar') },
  { id: 'edits', title: 'Causal effects', load: () => import('./Edits') },
  { id: 'robustness', title: 'Robustness break-points', load: () => import('./Robustness') },
  { id: 'physics', title: 'Physics gates', load: () => import('./Physics') },
  { id: 'psi0', title: 'Ψ₀ line', load: () => import('./Psi0') },
];
export default function Evaluations() {
  return <LensView lenses={EVIDENCE} fallback="results" />;
}
