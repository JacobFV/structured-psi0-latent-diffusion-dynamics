/**
 * Route radar (Evaluations lens), after IBM-2's radar: one radar per robot, one polygon per route, from curated evidence only.
 * Axes and normalisation (0 = floor, 1 = best route on that robot unless stated):
 *   competence       nominal success rate (robustness sweep, same seeds per route)        raw rate
 *   robustness       pooled perturbed success rate (same sweep)                           raw rate
 *   controllability  halt edit: forward travel removed, −Δforward (legged8 compare)       ÷ best route
 *   smoothness       1 / nominal joint jerk RMS                                            best route ÷ route
 *   energy           cost of transport per route: not in the curated evidence              always missing
 * A missing value is a gap: no point and no polygon edge through that axis.
 */
import { ModeBadge } from '../components/board';
import { useDoc, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, num, rows, str } from '../lib/format';
import { href } from '../lib/url';

type Val = { r: number | null; raw: string; src: string };
const AXES = ['competence', 'robustness', 'controllability', 'smoothness', 'energy'] as const;
const COLORS: Record<string, string> = { teacher: 'var(--warning)', bc: 'var(--ink-2)', semfix: 'var(--s1)', frozen_sem: 'var(--s7)', nosem: 'var(--s2)' };

function radarData(rb: Envelope | null, ed: Envelope | null) {
  const reps = rb ? rows(rb.reports).filter((r) => /^(teacher|bc|semfix|nosem|frozen_sem)$/.test(str(r.route))) : [];
  const edits = ed ? rows(ed.rows) : [];
  const robots = [...new Set(reps.map((r) => str(r.robot)))];
  return robots.map((robot) => {
    const rs = reps.filter((r) => str(r.robot) === robot);
    const vals: Record<string, Record<string, Val>> = {};
    const halt = (route: string) => edits.find((e) => str(e.body) === robot && /legged8_compare/.test(str(e.source_file)) && str(e.edit) === 'ctx_halt' && str(e.variant) === route && (e.seed === null || e.seed === undefined));
    const jerks = rs.map((r) => num(isObj(r.nominal) && isObj(r.nominal.motion) ? r.nominal.motion.joint_jerk_rms : null)).filter((v): v is number => v !== null && v > 0);
    const halts = rs.map((r) => num(halt(str(r.route))?.effect)).filter((v): v is number => v !== null).map((v) => Math.max(0, -v));
    const bestHalt = Math.max(1e-9, ...halts);
    for (const r of rs) {
      const route = str(r.route);
      const nom = isObj(r.nominal) ? r.nominal : {}, pool = isObj(r.pooled_perturbed) ? r.pooled_perturbed : {};
      const jerk = num(isObj(nom.motion) ? nom.motion.joint_jerk_rms : null);
      const h = halt(route);
      const he = num(h?.effect);
      vals[route] = {
        competence: { r: num(nom.rate), raw: `${str(nom.k)}/${str(nom.n)}`, src: `${str(r.decision)} robustness nominal` },
        robustness: { r: num(pool.rate), raw: `${str(pool.success)}/${str(pool.n)}`, src: `${str(r.decision)} pooled perturbed` },
        controllability: he !== null ? { r: Math.max(0, -he) / bestHalt, raw: `halt Δforward ${fmtNum(he)} m [${arr(h?.ci).map((x) => fmtNum(x)).join(', ')}]`, src: str(h?.source_file) } : { r: null, raw: 'no halt-edit row for this route', src: '' },
        smoothness: jerk && jerks.length ? { r: Math.min(...jerks) / jerk, raw: `jerk RMS ${fmtNum(jerk)}`, src: 'nominal motion' } : { r: null, raw: 'not recorded', src: '' },
        energy: { r: null, raw: 'cost of transport per route is not in the curated evidence', src: '' },
      };
    }
    return { robot, routes: rs.map((r) => str(r.route)), vals, decision: [...new Set(rs.map((r) => str(r.decision)))].join(', ') };
  });
}

function RadarSvg({ routes, vals }: { routes: string[]; vals: Record<string, Record<string, Val>> }) {
  const S = 220, c = S / 2, R = 78;
  const ang = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / AXES.length;
  const pt = (i: number, r: number) => [c + Math.cos(ang(i)) * R * r, c + Math.sin(ang(i)) * R * r];
  return (
    <svg width={S} height={S} role="img" aria-label="route radar">
      {[0.25, 0.5, 0.75, 1].map((g) => <polygon key={g} points={AXES.map((_, i) => pt(i, g).join(',')).join(' ')} fill="none" stroke="var(--grid)" />)}
      {AXES.map((a, i) => {
        const [x, y] = pt(i, 1.2);
        const missing = routes.every((rt) => vals[rt]?.[a]?.r === null || vals[rt]?.[a]?.r === undefined);
        return (
          <g key={a}>
            <line x1={c} y1={c} x2={pt(i, 1)[0]} y2={pt(i, 1)[1]} stroke="var(--grid)" strokeDasharray={missing ? '2 3' : undefined} />
            <text x={x} y={y + 3} textAnchor="middle" fill={missing ? 'var(--muted)' : 'var(--ink-2)'}>{a}{missing ? ' (missing)' : ''}</text>
          </g>
        );
      })}
      {routes.map((rt) => {
        const col = COLORS[rt] || 'var(--s3)';
        // polygon edges only between consecutive present axes
        const segs: string[] = [];
        AXES.forEach((a, i) => {
          const b = AXES[(i + 1) % AXES.length];
          const va = vals[rt]?.[a]?.r, vb = vals[rt]?.[b]?.r;
          if (va != null && vb != null) segs.push(`M${pt(i, va).join(',')}L${pt((i + 1) % AXES.length, vb).join(',')}`);
        });
        return (
          <g key={rt}>
            <path d={segs.join('')} fill="none" stroke={col} strokeWidth={1.8} />
            {AXES.map((a, i) => {
              const v = vals[rt]?.[a];
              if (v?.r == null) return null;
              const [x, y] = pt(i, Math.max(0, Math.min(1, v.r)));
              return <circle key={a} cx={x} cy={y} r={3} fill={col}><title>{`${rt} · ${a}: ${fmtNum(v.r)} (${v.raw}; ${v.src})`}</title></circle>;
            })}
          </g>
        );
      })}
    </svg>
  );
}

export default function Radar() {
  const rb = useDoc<Envelope>('robustness');
  const ed = useDoc<Envelope>('edits');
  const data = radarData(rb.result.status === 'ok' ? rb.result.data : null, ed.result.status === 'ok' ? ed.result.data : null);
  if (!data.length) return <p className="side-note">no curated robustness reports in /api/robustness</p>;
  return (
    <div className="ev-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
      {data.map((d) => (
        <section key={d.robot} className="ev-panel">
          <header>{d.robot}<ModeBadge result={rb.result} /><span className="meta" title="axes and normalisation: hover points; energy is not in the curated evidence">{d.decision}</span></header>
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <RadarSvg routes={d.routes} vals={d.vals} />
            <div className="legend" style={{ display: 'grid', gap: 2 }}>
              {d.routes.map((rt) => <a key={rt} href={href('robustness', { robot: d.robot })} style={{ color: 'var(--ink-2)' }}><i className="sw" style={{ background: COLORS[rt] || 'var(--s3)' }} />{rt}</a>)}
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}
