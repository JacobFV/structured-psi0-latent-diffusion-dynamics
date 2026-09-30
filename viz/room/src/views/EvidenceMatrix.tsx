/** Evaluations → Success: the current curated results only (legged contact v2 D-124, arm v6 D-134, sealed-target transfer
 * D-135/D-136), as heatmaps with CI whiskers. */
import { Cap, ModeBadge } from '../components/board';
import { Heat, type KN } from '../components/matrix';
import { useDoc, type Envelope } from '../lib/api';
import { arr, num, rows, str, type Row } from '../lib/format';
import { href, useUrlState } from '../lib/url';

type Cells = Map<string, KN>;
function add(m: Cells, key: string, k: number, n: number, tip: string, link: string) {
  const x = m.get(key) || { k: 0, n: 0, tip, link };
  x.k += k; x.n += n;
  m.set(key, x);
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
/** Sealed target bodies (D-135 targets_v6.json + D-136 d136_compare.json): k/200 pooled over 2 seeds per method × budget. */
export const TARGETS = ['xarm7_pg2', 'xarm7_tf3', 'panda_tf3'];
export function transferSet(rs: Row[], variant: 'semfix' | 'nosem') {
  const m = new Map<string, { k: number; n: number; sealed: boolean; tip: string }>();
  const put = (key: string, k: number, n: number, sealed: boolean, tip: string) => {
    const x = m.get(key) || { k: 0, n: 0, sealed: true, tip };
    x.k += k; x.n += n; x.sealed = x.sealed && sealed;
    m.set(key, x);
  };
  for (const r of rs) {
    const f = str(r.source_file), kp = arr(r.key_path).map(str);
    const k = num(r.k), n = num(r.n);
    if (k === null || !n) continue;
    if (/ladder\/armv6\/targets_v6\.json$/.test(f) && kp.length === 4) {
      const [who, lin, tgt, cell] = kp;
      if (!TARGETS.includes(tgt)) continue;
      if (who === 'latent' && lin.startsWith(`${variant}_s`)) {
        const mm = /^(zero_shot|flow_sft|system0_refit)(?:_b(\d+))?$/.exec(cell);
        if (mm) put(`${tgt}|${mm[1] === 'zero_shot' ? 'latent zero-shot' : mm[1] === 'flow_sft' ? 'flow SFT' : 'refit'}|${mm[2] || '0'}`, k, n, true, 'targets_v6.json · D-135');
      } else if (who === 'bc') {
        const mm = /^(zero_shot|bc_sft)(?:_b(\d+))?$/.exec(cell);
        if (mm) put(`${tgt}|${mm[1] === 'zero_shot' ? 'BC zero-shot' : 'BC SFT'}|${mm[2] || '0'}`, k, n, true, 'targets_v6.json · D-135');
      }
    } else if (/armdiag\/d136_compare\.json$/.test(f) && kp.length === 3 && kp[0] === 'pooled' && kp[2] === 'joint_adapt') {
      const mm = new RegExp(`^${variant}/(xarm7_(?:pg2|tf3))/b(\\d+)$`).exec(kp[1]);
      if (mm) put(`${mm[1]}|joint (D-136)|${mm[2]}`, k, n, true, 'd136_compare.json · D-136 · method added AFTER D-135 (same sealed scenes)');
    }
  }
  return m;
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

function TransferPanel({ rs, result }: { rs: Row[]; result: Parameters<typeof ModeBadge>[0]['result'] }) {
  const [variant, setVariant] = useUrlState('tv', 'semfix');
  const [budget, setBudget] = useUrlState('tb', '100');
  const m = transferSet(rs, variant === 'nosem' ? 'nosem' : 'semfix');
  if (!m.size) return null;
  const cols: [string, string][] = [['latent zero-shot', '0'], ['flow SFT', budget], ['refit', budget], ['joint (D-136)', budget], ['BC zero-shot', '0'], ['BC SFT', budget]];
  const W = 180, H = 70, xs = [5, 20, 100], X = (b: number) => 22 + ((Math.log(b) - Math.log(5)) / (Math.log(100) - Math.log(5))) * (W - 30), Y = (v: number) => H - 12 - v * (H - 18);
  const curves: [string, string][] = [['BC SFT', 'var(--ink-2)'], ['joint (D-136)', 'var(--s1)'], ['refit', 'var(--s2)']];
  return (
    <section className="ev-panel wide">
      <header>Cross-body transfer · sealed targets<ModeBadge result={result} />
        <span className="meta" title="k/200 pooled over 2 seeds, each sealed cell run once. joint = joint flow + system-0 adaptation, update-matched to BC SFT: a method added AFTER D-135 (D-136), evaluated on the same sealed scenes. Not run on panda_tf3.">
          <select value={variant} onChange={(e) => setVariant(e.target.value)} aria-label="latent variant"><option value="semfix">semfix v6</option><option value="nosem">nosem v6</option></select>{' '}
          <select value={budget} onChange={(e) => setBudget(e.target.value)} aria-label="adaptation budget">{['5', '20', '100'].map((b) => <option key={b} value={b}>budget {b} demos</option>)}</select>
          {' '}D-135 · D-136
        </span>
      </header>
      <div className="row" style={{ alignItems: 'flex-start', gap: 16 }}>
        <Heat showK rows={TARGETS} cols={cols.map(([c, b]) => `${c}${b !== '0' ? ` b${b}` : ''}`)}
          colLabel={(c) => c.replace('(D-136)', '†')}
          cell={(r, c) => { const [name, b] = cols.find(([n, bb]) => `${n}${bb !== '0' ? ` b${bb}` : ''}` === c)!; const x = m.get(`${r}|${name}|${b}`); return x ? { k: x.k, n: x.n, tip: `${x.tip} · sealed run`, link: href('results', { q: name.startsWith('joint') ? 'd136_compare' : 'targets_v6', data: '1' }) } : null; }} />
        <div style={{ display: 'grid', gap: 2 }}>
          {TARGETS.map((t) => (
            <svg key={t} width={W} height={H} role="img" aria-label={`budget curves ${t}`}>
              <text x={22} y={9} fill="var(--ink-2)">{t} · success vs demos</text>
              <line x1={22} x2={W - 8} y1={Y(0)} y2={Y(0)} stroke="var(--grid)" />
              {xs.map((b) => <text key={b} x={X(b)} y={H - 1} textAnchor="middle" fill="var(--muted)">{b}</text>)}
              {curves.map(([name, col]) => {
                const pts = xs.map((b) => { const x = m.get(`${t}|${name}|${b}`); return x && x.n ? [X(b), Y(x.k / x.n), x.k, x.n] as const : null; });
                const ok = pts.filter(Boolean) as (readonly [number, number, number, number])[];
                if (!ok.length) return null;
                return (
                  <g key={name}>
                    <polyline points={ok.map((p) => `${p[0]},${p[1]}`).join(' ')} fill="none" stroke={col} strokeWidth={1.5} />
                    {ok.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={2.5} fill={col}><title>{`${t} · ${name} · b${xs[pts.indexOf(p)]}: ${p[2]}/${p[3]}`}</title></circle>)}
                  </g>
                );
              })}
            </svg>
          ))}
          <div className="legend">{curves.map(([n, c]) => <span key={n}><i className="sw" style={{ background: c }} />{n}</span>)}</div>
        </div>
      </div>
      <Cap>heatmap: target × method at budget b, cell = successes / 200 sealed scenes, hatched = not run · curves: success vs demos (5/20/100) · † joint: method added after D-135 (D-136)</Cap>
    </section>
  );
}

export default function EvidenceMatrix() {
  const { result } = useDoc<Envelope>('results');
  const rs = result.status === 'ok' ? rows(result.data.rows) : [];
  const leg = leggedSet(rs), a6 = v6Set(rs);
  return (
    <div className="ev-grid">
      {leg.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Legged · contact v2 · success<ModeBadge result={result} /><span className="meta" title="R2 = deployable route, r2_final checkpoints, 3 seeds × 30 episodes; bc and teacher 30 episodes">summary_contact_v2 · D-124</span></header>
          <Heat rows={leg.bodies} cols={leg.cols} cell={(r, c) => leg.m.get(`${r}|${c}`) ?? null} />
          <Cap>rows: body · cols: route (latent R2: 3 seeds × 30; bc, teacher: 30) · cell: success % (hover k/n) · whisker: Wilson 95% CI</Cap>
        </section>
      )}
      {a6.bodies.length > 0 && (
        <section className="ev-panel">
          <header>Arm v6 · grasp_v2.1 · success<ModeBadge result={result} /><span className="meta" title="teacher v2 + grasp_v2.1 + v6 BC expert; semfix 425/480 vs nosem 314/480 pooled">compare_v6 · D-134</span></header>
          <Heat rows={a6.bodies} cols={a6.lines} colLabel={(c) => c.replace('_s', ' s')} cell={(r, c) => a6.m.get(`${r}|${c}`) ?? null} />
          <Cap>rows: arm body · cols: lineage × seed · cell: success % of 90 (panda, parm6) or 30 episodes · whisker: Wilson 95% CI</Cap>
        </section>
      )}
      <TransferPanel rs={rs} result={result} />
      {!leg.bodies.length && !a6.bodies.length && result.status === 'ok' && <p className="side-note">no curated comparison rows in /api/results</p>}
    </div>
  );
}
