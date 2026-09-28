import { Forest, Lines, type ForestRow } from '../components/charts';
import { Card, DataTable, Did, Gate, ModeBanner, PageHead, Provenance, Select } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, num, pick, rows, sortNatural, str, uniq, wilson, type Row } from '../lib/format';
import { seriesColor } from '../lib/labels';
import { useUrlState } from '../lib/url';

export default function Robustness() {
  const { result, reload, busy } = useDoc<Envelope>('robustness');
  return (
    <>
      <PageHead
        title="Robustness"
        sub="Perturbation sweeps per robot: success (or motion quality) against each factor's level for every route, with 95% CIs and the recorded break-points; route comparisons and variant-level paired differences (D-108 / D-112 / D-114)."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="robustness (/api/robustness)">{(d) => <RobustBody d={d} />}</Gate>
      <Provenance result={result} />
    </>
  );
}

function ciPair(r: Row, key = 'ci'): [number | null, number | null] {
  const c = r[key];
  if (Array.isArray(c)) return [num(c[0]), num(c[1])];
  return [num(pick(r, `${key}_lo`)), num(pick(r, `${key}_hi`))];
}

function RobustBody({ d }: { d: Envelope }) {
  const levels = rows(pick(d, 'levels', 'sweeps'));
  const reports = rows(pick(d, 'reports'));
  const comps = rows(pick(d, 'comparisons'));
  const vdiffs = rows(pick(d, 'variant_diffs'));
  const robots = uniq(levels.map((r) => str(pick(r, 'robot', 'body')))).filter(Boolean).sort(sortNatural);
  const [robot, setRobot] = useUrlState('robot', robots[0] || '');
  const [metric, setMetric] = useUrlState('m', 'rate');
  const [hidden, setHidden] = useUrlState('hide', '');
  const rl = levels.filter((r) => !robot || str(pick(r, 'robot', 'body')) === robot);
  const routes = uniq(rl.map((r) => str(r.route))).sort(sortNatural);
  const allRoutes = uniq(levels.map((r) => str(r.route))).sort(sortNatural);
  const hide = new Set(hidden.split(',').filter(Boolean));
  const factors = uniq(rl.map((r) => str(r.factor))).sort(sortNatural);
  const motionKeys = uniq(rl.flatMap((r) => (isObj(r.motion) ? Object.keys(r.motion) : []))).sort();
  const metricOpts = ['rate', 'drop', ...motionKeys.map((k) => `motion.${k}`)];
  const valueOf = (r: Row): [number | null, number | null, number | null] => {
    if (metric === 'rate') {
      const k = num(r.k), n = num(r.n);
      const rate = num(r.rate) ?? (k !== null && n ? k / n : null);
      let [lo, hi] = ciPair(r);
      if ((lo === null || hi === null) && k !== null && n) [lo, hi] = wilson(k, n);
      return [rate, lo, hi];
    }
    if (metric === 'drop') return [num(r.drop), null, null];
    const key = metric.slice('motion.'.length);
    return [isObj(r.motion) ? num(r.motion[key]) : null, null, null];
  };
  const rep = (route: string) => reports.find((x) => str(x.route) === route && str(pick(x, 'robot', 'body')) === robot);
  return (
    <div className="stack">
      <div className="card" style={{ padding: '10px 14px' }}>
        <div className="filters" style={{ marginBottom: 6 }}>
          <Select label="robot" value={robot} options={robots} onChange={setRobot} all={false} />
          <Select label="y value" value={metric} options={metricOpts} onChange={setMetric} all={false} />
        </div>
        <div className="legend">
          {routes.map((r) => (
            <button key={r} className="ghost small" aria-pressed={!hide.has(r)} style={{ opacity: hide.has(r) ? 0.4 : 1 }}
              onClick={() => { const n = new Set(hide); if (n.has(r)) n.delete(r); else n.add(r); setHidden([...n].join(',')); }}>
              <i className="sw" style={{ background: seriesColor(r, allRoutes) }} />{r}
            </button>
          ))}
          <span className="muted">click a route to hide it · dashed verticals: recorded break-points (low / high) · bands: 95% CI</span>
        </div>
      </div>
      {factors.length === 0 ? <p className="muted">No sweep levels for this robot.</p> : (
        <div className="grid g2">
          {factors.map((f) => {
            const fr = rl.filter((r) => str(r.factor) === f && !hide.has(str(r.route)));
            const lvls = uniq(fr.map((r) => str(r.level)));
            const numeric = lvls.every((l) => num(l) !== null);
            const lv = numeric ? lvls.map(Number).sort((a, b) => a - b).map(String) : lvls.sort(sortNatural);
            const rts = uniq(fr.map((r) => str(r.route))).sort(sortNatural);
            const data = lv.map((l) => {
              const o: Row = { level: numeric ? Number(l) : l };
              for (const rt of rts) {
                const e = fr.find((r) => str(r.level) === l && str(r.route) === rt);
                if (!e) continue;
                const [v, lo, hi] = valueOf(e);
                o[rt] = v; o[`${rt}__lo`] = lo; o[`${rt}__hi`] = hi;
              }
              return o;
            });
            const bps = rts.flatMap((rt) => {
              const b = rep(rt)?.break_points;
              const x = isObj(b) && isObj(b[f]) ? b[f] : null;
              if (!x) return [];
              return (['low', 'high'] as const).filter((k) => num(x[k]) !== null).map((k) => ({ x: num(x[k])!, label: `${rt} ${k}`, color: seriesColor(rt, allRoutes) }));
            });
            return (
              <Card key={f} title={f} hint={`${lv.length} levels · ${rts.length} routes`}>
                <Lines
                  data={data} xKey="level" xType={numeric ? 'number' : 'category'} height={240} yDomain={metric === 'rate' ? [0, 1] : undefined} xLabel={`${f} level`}
                  series={rts.map((rt) => ({ key: rt, label: rt, color: seriesColor(rt, allRoutes), band: metric === 'rate' ? [`${rt}__lo`, `${rt}__hi`] as [string, string] : undefined, dots: true }))}
                  refs={numeric ? bps : []}
                />
                <p className="small muted" style={{ margin: '4px 0 0' }}>
                  Break-points: {bps.length ? bps.map((b) => `${b.label} @ ${fmtNum(b.x)}`).join(' · ') : 'none recorded within the swept range'}
                </p>
              </Card>
            );
          })}
        </div>
      )}
      <Card title="Reports" hint="nominal and pooled-perturbed success per route and robot">
        <DataTable rows={reports.map((r) => {
          const nom = isObj(r.nominal) ? r.nominal : {}, pool = isObj(r.pooled_perturbed) ? r.pooled_perturbed : {};
          return {
            robot: pick(r, 'robot', 'body'), route: r.route, 'nominal k/n': `${str(nom.k)}/${str(nom.n)}`, nominal_rate: nom.rate, nominal_ci: nom.ci,
            'perturbed k/n': `${str(pool.success)}/${str(pool.n)}`, perturbed_rate: pool.rate, perturbed_ci: pool.wilson95, lost: pool.lost, gained: pool.gained,
            sources: arr(nom.sources).map(str).join(', '), decision: r.decision, source_file: r.source_file,
          };
        })} />
      </Card>
      <Card title="Route comparisons" hint="difference in level-mean success (A − B), recorded CI and p">
        <Forest xLabel="Δ level-mean success (A − B)" rows={comps.filter((c) => !robot || str(c.robot) === robot).map((c, i): ForestRow => {
          const [lo, hi] = ciPair(c, 'diff_level_mean_ci');
          return { key: `c${i}`, label: `${str(c.a)} vs ${str(c.b)} · ${str(c.robot)}`, effect: num(c.diff_level_mean), lo, hi, p: num(c.p_level_mean), n: num(c.n_seeds),
            tone: 'var(--s1)', note: `drop Δ ${fmtNum(c.diff_drop)} (p ${fmtNum(c.p_drop)}) · ${str(c.metric)} · ${str(c.decision)}` };
        })} />
        <DataTable rows={comps.map((c) => ({ robot: c.robot, a: c.a, b: c.b, metric: c.metric, n_seeds: c.n_seeds, n_levels: c.n_levels, level_mean_a: c.level_mean_a, level_mean_b: c.level_mean_b, diff: c.diff_level_mean, ci: c.diff_level_mean_ci, p: c.p_level_mean, diff_drop: c.diff_drop, p_drop: c.p_drop, decision: c.decision }))} tall />
      </Card>
      <Card title="Variant-level paired differences" hint="per training seed (D-108 / D-112)">
        <VariantDiffs vd={vdiffs} />
      </Card>
      {arr(d.notes).length > 0 && <ul className="small muted">{arr(d.notes).map((n, i) => <li key={i}>{str(n)}</li>)}</ul>}
    </div>
  );
}

function VariantDiffs({ vd }: { vd: Row[] }) {
  if (!vd.length) return <p className="muted small">No variant-level differences in the document.</p>;
  const out: ForestRow[] = [];
  const table: Row[] = [];
  vd.forEach((v, vi) => {
    const metrics = isObj(v.metrics) ? v.metrics : {};
    for (const [m, body] of Object.entries(metrics)) {
      const per = isObj(body) && isObj(body.per_training_seed) ? body.per_training_seed : {};
      for (const [seed, c] of Object.entries(per)) {
        if (!isObj(c)) continue;
        const [lo, hi] = ciPair(c, 'diff_level_mean_ci');
        out.push({ key: `${vi}-${m}-${seed}`, label: `${str(v.a)} − ${str(v.b)} · ${str(v.robot)} · ${m} · t${seed}`, effect: num(c.diff_level_mean), lo, hi, p: num(c.p_level_mean), n: num(c.n_seeds), tone: 'var(--s3)' });
        table.push({ robot: v.robot, metric: m, training_seed: seed, a: c.a, b: c.b, diff: c.diff_level_mean, ci: c.diff_level_mean_ci, p: c.p_level_mean, diff_drop: c.diff_drop, p_drop: c.p_drop });
      }
      if (isObj(body)) {
        const rest = Object.fromEntries(Object.entries(body).filter(([k]) => k !== 'per_training_seed'));
        if (Object.keys(rest).length) table.push({ robot: v.robot, metric: m, training_seed: 'pooled', ...rest });
      }
    }
  });
  return (
    <>
      {v0decisions(vd)}
      {out.length ? <Forest rows={out} xLabel="Δ level-mean success (A − B)" /> : null}
      <DataTable rows={table} tall />
    </>
  );
}
function v0decisions(vd: Row[]) {
  const ds = uniq(vd.flatMap((v) => [str(v.decision), ...arr(v.decisions).map(str)]).filter(Boolean));
  return ds.length ? <div className="row small">{ds.map((x) => <Did key={x} id={x} />)}</div> : null;
}
