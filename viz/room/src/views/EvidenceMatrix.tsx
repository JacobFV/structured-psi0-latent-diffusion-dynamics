/**
 * Success matrix (Evidence lens): the curated result sets only, as heatmaps with CI whiskers, plus the grasp v1 → v2
 * delta small multiple. Curated = the decision-referenced comparison files named on each panel. Raw rows (every result
 * found) only behind the sidebar "show data" toggle.
 */
import { ModeBadge } from '../components/board';
import { Heat, type KN } from '../components/matrix';
import { useDoc, type Envelope } from '../lib/api';
import { arr, num, rows, str, type Row } from '../lib/format';
import { href, useUrlState } from '../lib/url';
import Results from './Results';

type Cells = Map<string, KN>;
function add(m: Cells, key: string, k: number, n: number, tip: string, link: string) {
  const x = m.get(key) || { k: 0, n: 0, tip, link };
  x.k += k; x.n += n;
  m.set(key, x);
}
const shortLine = (l: string) => l.replace(/\s*\(.*\)$/, '').replace(/, seed /, ' s').replace(/^BC direct1701 final.*/, 'BC direct').replace(/^frozen sem/, 'frozen');

export function armSets(rs: Row[]) {
  const v2: Cells = new Map(), v1: Cells = new Map();
  const bodies = new Set<string>(), lines: string[] = [];
  for (const r of rs) {
    const kp = arr(r.key_path).map(str);
    if (kp.length !== 3 || !str(r.source_file).endsWith('compare_gc2_final.json') || !/^grasp_v[12]$/.test(kp[2])) continue;
    const k = num(r.k), n = num(r.n);
    if (k === null || !n) continue;
    bodies.add(kp[1]);
    if (!lines.includes(kp[0])) lines.push(kp[0]);
    add(kp[2] === 'grasp_v2' ? v2 : v1, `${kp[1]}|${kp[0]}`, k, n, `${kp[0]} · ${kp[2]} · compare_gc2_final.json (${str(r.decision)})`, href('results', { q: 'compare_gc2_final', data: '1' }));
  }
  return { v1, v2, bodies: [...bodies].sort(), lines };
}
export function leggedSet(rs: Row[]) {
  const m: Cells = new Map();
  const bodies = new Set<string>();
  for (const r of rs) {
    if (!/summary_contact_v2\.json$/.test(str(r.source_file))) continue;
    const kp = arr(r.key_path).map(str);
    const k = num(r.k), n = num(r.n);
    if (k === null || !n || kp.length !== 3) continue;
    const col = kp[1] === 'r2_final' ? `${kp[2]} (R2)` : kp[1] === 'references' ? kp[2] : null;
    if (!col) continue;
    bodies.add(kp[0]);
    add(m, `${kp[0]}|${col}`, k, n, `${kp.join(' / ')} · summary_contact_v2.json (${str(r.decision)})`, href('results', { q: 'summary_contact_v2', data: '1' }));
  }
  return { m, bodies: [...bodies], cols: ['semfix (R2)', 'nosem (R2)', 'bc', 'teacher'] };
}

export default function EvidenceMatrix() {
  const { result } = useDoc<Envelope>('results');
  const [data] = useUrlState('data', '0');
  const rs = result.status === 'ok' ? rows(result.data.rows) : [];
  const arm = armSets(rs), leg = leggedSet(rs);
  return (
    <div className="ev-grid">
      {leg.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Legged · contact v2 · success<ModeBadge result={result} /><span className="meta" title="R2 = deployable route, r2_final checkpoints, 3 seeds × 30 episodes; bc and teacher 30 episodes; D-124">summary_contact_v2 · D-124</span></header>
          <Heat rows={leg.bodies} cols={leg.cols} cell={(r, c) => leg.m.get(`${r}|${c}`) ?? null} />
        </section>
      )}
      {arm.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Arm · grasp_v2 · success<ModeBadge result={result} /><span className="meta" title="deployable success per body and lineage, 2 training seeds, 90 episodes per cell; D-127">compare_gc2_final · D-127</span></header>
          <Heat rows={arm.bodies} cols={arm.lines} colLabel={shortLine} cell={(r, c) => arm.v2.get(`${r}|${c}`) ?? null} />
        </section>
      )}
      {arm.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Arm · grasp v1 → v2 · Δ success<ModeBadge result={result} /><span className="meta" title="v2 − v1 on the same body, lineage and episodes; ˙ = CI includes 0">realistic grasp physics</span></header>
          <Heat rows={arm.bodies} cols={arm.lines} colLabel={shortLine} cell={(r, c) => arm.v2.get(`${r}|${c}`) ?? null} base={(r, c) => arm.v1.get(`${r}|${c}`) ?? null} />
        </section>
      )}
      {!leg.bodies.length && !arm.bodies.length && result.status === 'ok' && <p className="side-note">no curated comparison rows in /api/results</p>}
      {data === '1' && <section className="ev-panel wide"><header>All result rows (raw)</header><Results /></section>}
    </div>
  );
}
