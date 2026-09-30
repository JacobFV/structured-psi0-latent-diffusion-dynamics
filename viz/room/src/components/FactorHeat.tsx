/**
 * Relation-factor attention inspection (D-144 R21, docs/relations.md 10, viz/CONTRACT.md): per-factor per-head logit
 * maps at a chosen recorded step (`rrp-viz/factormap/v1`, `rrp.viz.record.record_factor_maps`), the factor's source
 * badge (AGENTS.md: privileged / estimated / public sources marked unmistakably, never colour alone) and control.
 */
import { useMemo, useState } from 'react';
import { divColor } from './charts';
import { SourceBadge } from './ui';
import { nearestFactorStep, type FactorMap, type FactorProvenance } from '../lib/replay';

function badgeLabel(p: FactorProvenance | undefined): string {
  if (!p) return 'unknown source';
  if (p.privileged) return `PRIVILEGED (${p.source}) · not deployable`;
  if (p.source === 'probe') return 'ESTIMATED (probe)';
  if (p.source?.startsWith('estimator:')) return `ESTIMATED (${p.source})`;
  return `PUBLIC (${p.source})`;
}

/** One factor's [H,Q,K] map: H small-multiple grids, one head each, diverging colour scaled by the map's own max |value|. */
function HeadGrids({ mat }: { mat: number[][][] }) {
  const amax = useMemo(() => {
    let m = 1e-9;
    for (const h of mat) for (const row of h) for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  }, [mat]);
  const Q = mat[0]?.length || 0, K = mat[0]?.[0]?.length || 0;
  const cell = Math.max(2, Math.min(14, Math.floor(240 / Math.max(Q, K, 1))));
  return (
    <div className="row" style={{ flexWrap: 'wrap', gap: 12 }}>
      {mat.map((head, h) => (
        <div key={h}>
          <div className="muted small mono" style={{ marginBottom: 2 }}>head {h}</div>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${K}, ${cell}px)`, gridAutoRows: `${cell}px`, border: '1px solid var(--line)' }}>
            {head.flatMap((row, q) => row.map((v, k) => (
              <div key={`${q}-${k}`} title={`q${q} × k${k} = ${v.toFixed(4)}`}
                   style={{ background: divColor((v / amax) * 0.5), width: cell, height: cell }} />
            )))}
          </div>
        </div>
      ))}
      <div className="muted small" style={{ alignSelf: 'end' }}>Q × K, {Q}×{K} · scaled to ± {amax.toFixed(3)} (this map's max |logit|)</div>
    </div>
  );
}

export function FactorMapPanel({ map, t }: { map: FactorMap; t: number }) {
  const i = nearestFactorStep(map, t);
  const step = i >= 0 ? map.steps[i] : null;
  const names = step ? Object.keys(step.factors).sort() : [];
  const [pick, setPick] = useState<string>('');
  const name = names.includes(pick) ? pick : names[0] || '';
  const prov = useMemo(() => Object.fromEntries(map.provenance.map((p) => [p.name, p])), [map.provenance]);
  if (!step || !names.length) return <p className="muted small">no factor maps recorded near t={t.toFixed(2)}s</p>;
  const p = prov[name];
  return (
    <div>
      <div className="row small" style={{ marginBottom: 4, flexWrap: 'wrap', gap: 6 }}>
        <label>factor <select value={name} onChange={(e) => setPick(e.target.value)}>{names.map((n) => <option key={n} value={n}>{n}</option>)}</select></label>
        <SourceBadge label={badgeLabel(p)} title={p ? `${p.op ?? ''} / ${p.form ?? ''}${p.control ? ` · control ${p.control}` : ''}` : undefined} />
        <span className="mono muted small">site {step.site} · recorded step t={step.t.toFixed(3)}s{Math.abs(step.t - t) > 1e-6 ? ` (nearest to cursor ${t.toFixed(2)}s)` : ''}</span>
      </div>
      {name ? <HeadGrids mat={step.factors[name]} /> : null}
    </div>
  );
}
