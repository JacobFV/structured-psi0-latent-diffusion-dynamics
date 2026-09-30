import { LensView, type LensDef } from './Lens';

export const EVIDENCE: LensDef[] = [
  { id: 'results', title: 'Success', load: () => import('./EvidenceMatrix') },
  { id: 'radar', title: 'Route radar', load: () => import('./Radar') },
  { id: 'edits', title: 'Causal effects', load: () => import('./Edits') },
  { id: 'factors', title: 'Relation factors', load: () => import('./FactorsView') },
];
export default function Evaluations() {
  return <LensView lenses={EVIDENCE} fallback="results" />;
}
