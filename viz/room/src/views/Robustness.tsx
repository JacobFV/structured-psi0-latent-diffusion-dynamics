import { Forest, Lines, type ForestRow } from '../components/charts';
import { Card, DataTable, Gate, ModeBanner, PageHead, Provenance, Select } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { num, pick, rows, sortNatural, str, uniq, wilson, type Row } from '../lib/format';
import { seriesColor } from '../lib/labels';
import { useUrlState } from '../lib/url';

export default function Robustness() {
  const { result, reload, busy } = useDoc<Envelope>('robustness');
  return (
    <>
      <PageHead
        title="Robustness"
        sub="Perturbation sweeps (route × factor × level) with 95% CIs, break-points, motion-quality medians and variant-level paired differences (D-108 / D-112)."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="robustness (/api/robustness)">
        {(d) => <RobustBody d={d} />}
      </Gate>
      <Provenance result={result} />
    </>
  );
}

function RobustBody({ d }: { d: Envelope }) {
  const sweep = rows(pick(d, 'sweeps', 'sweep', 'rows'));
  const bps = rows(pick(d, 'break_points', 'breakpoints'));
  const mq = rows(pick(d, 'motion_quality', 'motion'));
  const diffs = rows(pick(d, 'variant_diffs', 'paired_diffs', 'diffs'));
  const [family, setFamily] = useUrlState('family', '');
  const [body, setBody] = useUrlState('body', '');
  const opt = (k: string) => uniq(sweep.map((r) => str(r[k]))).filter(Boolean).sort(sortNatural);
  const shown = sweep.filter((r) => (!family || str(r.family) === family) && (!body || str(r.body) === body));
  const factors = uniq(shown.map((r) => str(r.factor))).sort(sortNatural);
  const routes = uniq(sweep.map((r) => str(pick(r, 'route', 'variant')))).sort(sortNatural);
  return (
    <div className="stack">
      <div className="filters">
        {opt('family').length > 0 && <Select label="family" value={family} options={opt('family')} onChange={setFamily} />}
        {opt('body').length > 0 && <Select label="body" value={body} options={opt('body')} onChange={setBody} />}
        <span className="legend">{routes.map((r) => <span key={r}><i className="sw" style={{ background: seriesColor(r, routes) }} />{r}</span>)}</span>
      </div>
      {factors.length === 0 ? <p className="muted">No sweep rows in the document.</p> : (
        <div className="grid g2">
          {factors.map((f) => {
            const fr = shown.filter((r) => str(r.factor) === f);
            const levels = uniq(fr.map((r) => str(r.level)));
            const numeric = levels.every((l) => num(l) !== null);
            const lv = numeric ? levels.map(Number).sort((a, b) => a - b).map(String) : levels.sort(sortNatural);
            const rts = uniq(fr.map((r) => str(pick(r, 'route', 'variant')))).sort(sortNatural);
            const data = lv.map((l) => {
              const o: Row = { level: numeric ? Number(l) : l };
              for (const rt of rts) {
                const e = fr.find((r) => str(r.level) === l && str(pick(r, 'route', 'variant')) === rt);
                if (!e) continue;
                const k = num(e.k), n = num(e.n);
                const rate = num(e.rate) ?? (k !== null && n ? k / n : null);
                let lo = num(e.ci_lo), hi = num(e.ci_hi);
                if ((lo === null || hi === null) && k !== null && n) [lo, hi] = wilson(k, n);
                o[rt] = rate; o[`${rt}__lo`] = lo; o[`${rt}__hi`] = hi;
              }
              return o;
            });
            const fbps = bps.filter((b) => str(b.factor) === f && (!family || !b.family || str(b.family) === family) && (!body || !b.body || str(b.body) === body));
            return (
              <Card key={f} title={f} hint={`${lv.length} levels · ${rts.length} routes`}>
                <Lines
                  data={data} xKey="level" xType={numeric ? 'number' : 'category'} height={240} yDomain={[0, 1]} xLabel={`${f} level`}
                  series={rts.map((rt) => ({ key: rt, label: rt, color: seriesColor(rt, routes), band: [`${rt}__lo`, `${rt}__hi`], dots: true }))}
                  refs={fbps.map((b) => ({ x: num(pick(b, 'level', 'break_level', 'value')) ?? undefined, label: `break ${str(pick(b, 'route', 'variant'))}`, color: seriesColor(str(pick(b, 'route', 'variant')), routes) })).filter((r) => r.x !== undefined)}
                />
                {fbps.length > 0 && <p className="small muted" style={{ margin: '4px 0 0' }}>Break-points (dashed): {fbps.map((b) => `${str(pick(b, 'route', 'variant'))} @ ${str(pick(b, 'level', 'break_level', 'value'))}${b.criterion ? ` (${str(b.criterion)})` : ''}`).join(' · ')}</p>}
              </Card>
            );
          })}
        </div>
      )}
      <Card title="Break-points" hint={`${bps.length}`}><DataTable rows={bps} empty="No break-points in the document." /></Card>
      <Card title="Motion quality (medians)" hint={`${mq.length}`}><DataTable rows={mq} tall empty="No motion-quality rows in the document." /></Card>
      <Card title="Variant-level paired differences" hint="D-108 / D-112">
        {diffs.length ? (
          <Forest xLabel="paired difference (B − A)" rows={diffs.map((r, i): ForestRow => {
            const ci = Array.isArray(r.ci) ? r.ci : [pick(r, 'ci_lo'), pick(r, 'ci_hi')];
            return {
              key: `${i}`, label: `${str(pick(r, 'comparison', 'pair')) || `${str(pick(r, 'variant_a', 'a'))} → ${str(pick(r, 'variant_b', 'b'))}`}${r.factor ? ` · ${str(r.factor)}` : ''}`,
              effect: num(pick(r, 'diff', 'delta', 'effect')), lo: num(ci[0]), hi: num(ci[1]), p: num(pick(r, 'p', 'permutation_p', 'p_value')), n: num(pick(r, 'n_pairs', 'n')),
              note: [str(r.decision), str(r.caveat)].filter(Boolean).join(' · '),
            };
          })} />
        ) : <p className="muted small">No paired differences in the document.</p>}
        {diffs.length > 0 && <DataTable rows={diffs} />}
      </Card>
    </div>
  );
}
