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
export function v6Set(rs: Row[]) {
  const m: Cells = new Map();
  const bodies: string[] = [], lines: string[] = [];
  for (const r of rs) {
    const kp = arr(r.key_path).map(str);
    if (kp.length !== 3 || kp[0] !== 'v6' || !/compare_v6\.json$/.test(str(r.source_file)) || kp[2] === 'all' || !/^(semfix|nosem)_s/.test(kp[1])) continue;
    const k = num(r.k), n = num(r.n);
    if (k === null || !n) continue;
    if (!bodies.includes(kp[2])) bodies.push(kp[2]);
    if (!lines.includes(kp[1])) lines.push(kp[1]);
    add(m, `${kp[2]}|${kp[1]}`, k, n, `${kp[1]} · grasp_v2.1 · compare_v6.json (${str(r.decision) || 'D-134'})`, href('results', { q: 'compare_v6', data: '1' }));
  }
  return { m, bodies, lines: lines.sort() };
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

/** go2 contact v1 → v2 (the only body with matched v1 rows): v1 = legged_fixrep_compare_go2.json (D-090), v2 = legged8_compare_go2.json (D-113). */
const V1 = 'artifacts/runs/legged_fixrep_compare_go2.json', V2 = 'research/tracks/legged8/legged8_compare_go2.json';
function go2Contact(res: Row[], eds: Row[]) {
  const sem = (v: string) => (/nosem/.test(v) ? 'nosem' : /fixsem|semfix/.test(v) ? 'semantic' : null);
  const halt: { variant: string; seed: string; v1?: Row; v2?: Row }[] = [];
  for (const [file, ver] of [[V1, 'v1'], [V2, 'v2']] as const) {
    for (const r of eds) {
      if (str(r.source_file) !== file || str(r.edit) !== 'ctx_halt' || !/Δforward/.test(str(r.metric))) continue;
      const v = sem(str(r.variant));
      if (!v) continue;
      const seed = r.seed === null || r.seed === undefined ? 'pooled' : `s${str(r.seed)}`;
      let x = halt.find((h) => h.variant === v && h.seed === seed);
      if (!x) halt.push((x = { variant: v, seed }));
      x[ver] = r;
    }
  }
  const succ = new Map<string, { k: number; n: number }>();
  for (const r of res) {
    const f = str(r.source_file);
    if ((f !== V1 && f !== V2) || str(r.metric) !== 'success' || arr(r.key_path).map(str)[4] !== 'r2_final') continue;
    const v = sem(str(r.variant));
    if (!v) continue;
    const key = `${v}|${f === V1 ? 'v1' : 'v2'}`;
    const x = succ.get(key) || { k: 0, n: 0 };
    x.k += num(r.k) || 0; x.n += num(r.n) || 0;
    succ.set(key, x);
  }
  return { halt: halt.filter((h) => h.v1 && h.v2).sort((a, b) => a.variant.localeCompare(b.variant) || a.seed.localeCompare(b.seed)), succ };
}
function Go2Contact({ res, eds, result }: { res: Row[]; eds: Row[]; result: Parameters<typeof ModeBadge>[0]['result'] }) {
  const { halt, succ } = go2Contact(res, eds);
  if (!halt.length) return null;
  const vals = halt.flatMap((h) => [h.v1, h.v2].flatMap((r) => [num(r!.effect)!, num(arr(r!.ci)[0])!, num(arr(r!.ci)[1])!]));
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const W = 420, left = 110, right = 16, H = halt.length * 16 + 14;
  const x = (v: number) => left + ((v - lo) / (hi - lo || 1)) * (W - left - right);
  const tone = (v: string) => (v === 'nosem' ? 'var(--muted)' : 'var(--up)');
  return (
    <section className="ev-panel">
      <header>go2 · contact v1 → v2<ModeBadge result={result} /><span className="meta" title="go2 only; anymal_c/t1 have no v1 match (hexapod6 has v1 only). v1: legged_fixrep_compare_go2.json (D-090); v2: legged8_compare_go2.json (D-113)">go2 only · D-090 → D-113</span></header>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="halt effect contact v1 vs v2">
        <line x1={x(0)} x2={x(0)} y1={0} y2={H - 12} stroke="var(--ink-2)" strokeDasharray="2 2" />
        <text x={left} y={H - 2} fontSize={10} fill="var(--muted)">{lo.toFixed(2)} m</text>
        <text x={W - right} y={H - 2} fontSize={10} fill="var(--muted)" textAnchor="end">{hi.toFixed(2)} m · halt Δforward</text>
        {halt.map((h, i) => {
          const y = i * 16 + 8, a = num(h.v1!.effect)!, b = num(h.v2!.effect)!;
          return (
            <g key={`${h.variant}${h.seed}`}>
              <title>{`${h.variant} ${h.seed}: contact v1 ${a} [${arr(h.v1!.ci).join(', ')}] → v2 ${b} [${arr(h.v2!.ci).join(', ')}]`}</title>
              <text x={left - 6} y={y + 4} fontSize={11} textAnchor="end" fill="var(--ink-2)">{h.variant} {h.seed}</text>
              <line x1={x(a)} x2={x(b)} y1={y} y2={y} stroke={tone(h.variant)} strokeWidth={1.2} />
              <circle cx={x(a)} cy={y} r={3.5} fill="var(--surface)" stroke={tone(h.variant)} strokeWidth={1.5} />
              <circle cx={x(b)} cy={y} r={4} fill={tone(h.variant)} />
            </g>
          );
        })}
      </svg>
      <div className="legend small"><span>○ contact v1 → ● contact v2</span>
        {['semantic', 'nosem'].map((v) => { const a = succ.get(`${v}|v1`), b = succ.get(`${v}|v2`); return a && b ? <span key={v} title="R2 (r2_final) success pooled over 3 seeds">{v} R2 success {a.k}/{a.n} → {b.k}/{b.n}</span> : null; })}
      </div>
    </section>
  );
}

export default function EvidenceMatrix() {
  const { result } = useDoc<Envelope>('results');
  const edits = useDoc<Envelope>('edits');
  const [data] = useUrlState('data', '0');
  const rs = result.status === 'ok' ? rows(result.data.rows) : [];
  const arm = armSets(rs), leg = leggedSet(rs), a6 = v6Set(rs);
  return (
    <div className="ev-grid">
      {leg.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Legged · contact v2 · success<ModeBadge result={result} /><span className="meta" title="R2 = deployable route, r2_final checkpoints, 3 seeds × 30 episodes; bc and teacher 30 episodes; D-124">summary_contact_v2 · D-124</span></header>
          <Heat rows={leg.bodies} cols={leg.cols} cell={(r, c) => leg.m.get(`${r}|${c}`) ?? null} />
        </section>
      )}
      {a6.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Arm v6 · grasp_v2.1 · success<ModeBadge result={result} /><span className="meta" title="teacher v2 + grasp_v2.1 + v6 BC expert; semfix 425/480 vs nosem 314/480 pooled; 2 seeds per variant; D-134">compare_v6 · D-134 · current</span></header>
          <Heat rows={a6.bodies} cols={a6.lines} colLabel={(c) => c.replace('_s', ' s')} cell={(r, c) => a6.m.get(`${r}|${c}`) ?? null} />
        </section>
      )}
      {arm.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Arm v1 · grasp_v2 · success<ModeBadge result={result} /><span className="meta" title="deployable success per body and lineage, 2 training seeds, 90 episodes per cell; D-127">compare_gc2_final · D-127</span></header>
          <Heat rows={arm.bodies} cols={arm.lines} colLabel={shortLine} cell={(r, c) => arm.v2.get(`${r}|${c}`) ?? null} />
        </section>
      )}
      {arm.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Arm v1 · grasp v1 → v2 · Δ success<ModeBadge result={result} /><span className="meta" title="v2 − v1 on the same body, lineage and episodes; ˙ = CI includes 0">realistic grasp physics</span></header>
          <Heat rows={arm.bodies} cols={arm.lines} colLabel={shortLine} cell={(r, c) => arm.v2.get(`${r}|${c}`) ?? null} base={(r, c) => arm.v1.get(`${r}|${c}`) ?? null} />
        </section>
      )}
      <Go2Contact res={rs} eds={edits.result.status === 'ok' ? rows(edits.result.data.rows) : []} result={edits.result} />
      {!leg.bodies.length && !arm.bodies.length && result.status === 'ok' && <p className="side-note">no curated comparison rows in /api/results</p>}
      {data === '1' && <section className="ev-panel wide"><header>All result rows (raw)</header><Results /></section>}
    </div>
  );
}
