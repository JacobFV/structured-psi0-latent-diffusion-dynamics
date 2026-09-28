import { Card, DataTable, Gate, ModeBanner, PageHead, Provenance, Status } from '../components/ui';
import Sections from '../components/Sections';
import { useDoc, type Envelope } from '../lib/api';
import { fmtNum, num, pick, rows, str, uniq, type Row } from '../lib/format';
import { seriesColor, stateTone } from '../lib/labels';

function MetricBars({ rs, metric, label }: { rs: Row[]; metric: RegExp; label: string }) {
  const key = Object.keys(rs[0] || {}).find((k) => metric.test(k));
  if (!key) return null;
  const vals = rs.map((r) => ({ name: str(pick(r, 'tracker', 'body', 'name', 'key')), v: num(r[key]), ver: str(pick(r, 'contact_version', 'version')) })).filter((x) => x.v !== null);
  if (!vals.length) return null;
  const max = Math.max(...vals.map((x) => Math.abs(x.v!)));
  const names = uniq(vals.map((x) => x.ver || 'value'));
  return (
    <figure style={{ margin: '0 0 12px' }}>
      <div className="chart-title"><b>{label}</b><span className="muted">({key})</span></div>
      <div style={{ display: 'grid', gap: 3 }}>
        {vals.map((x, i) => (
          <div key={i} style={{ display: 'grid', gridTemplateColumns: '200px 1fr 70px', gap: 8, alignItems: 'center', fontSize: 12 }}>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={x.name}>{x.name}{x.ver ? ` · ${x.ver}` : ''}</span>
            <div style={{ height: 10, background: 'var(--surface-3)', borderRadius: 3 }}>
              <div style={{ width: `${(Math.abs(x.v!) / (max || 1)) * 100}%`, height: '100%', borderRadius: 3, background: seriesColor(x.ver || 'value', names) }} />
            </div>
            <span className="num" style={{ textAlign: 'right' }}>{fmtNum(x.v)}</span>
          </div>
        ))}
      </div>
    </figure>
  );
}

export default function Physics() {
  const { result, reload, busy } = useDoc<Envelope>('physics');
  return (
    <>
      <PageHead
        title="Physics credibility"
        sub="Whether the simulator can be trusted for each claim: contact v2 slip and cost of transport per tracker, grasp-rig results, tracker validation and gate backfill verdicts (D-093..D-114)."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="physics (/api/physics)">
        {(d) => {
          const gates = rows(pick(d, 'gates', 'gate_backfill', 'verdicts'));
          const counts = gates.reduce<Record<string, number>>((a, g) => { const t = stateTone(pick(g, 'verdict', 'state', 'status')); a[t] = (a[t] || 0) + 1; return a; }, {});
          return (
            <>
              {gates.length > 0 && (
                <div className="row" style={{ marginBottom: 12 }}>
                  <b>Gate verdicts:</b>
                  {Object.entries(counts).map(([t, n]) => <Status key={t} state={t === 'good' ? 'pass' : t === 'critical' ? 'fail' : t}>{n} {t === 'good' ? 'pass' : t === 'critical' ? 'fail' : t}</Status>)}
                </div>
              )}
              <Sections d={d} specs={[
                { keys: ['contact_v2', 'contact', 'slip_cot'], title: 'Contact v2: slip and cost of transport', hint: 'per tracker',
                  render: (rs) => (
                    <>
                      <div className="grid g2"><MetricBars rs={rs} metric={/slip/i} label="slip" /><MetricBars rs={rs} metric={/cot|cost/i} label="cost of transport" /></div>
                      <DataTable rows={rs} tall />
                    </>
                  ) },
                { keys: ['grasp_rig', 'grasp'], title: 'Grasp rig', hint: 'grasp contact versions' },
                { keys: ['tracker_validation', 'trackers'], title: 'Tracker validation' },
                { keys: ['actuator', 'actuator_limits'], title: 'Actuator limits' },
                { keys: ['gates', 'gate_backfill', 'verdicts'], title: 'Gate backfill verdicts', render: (rs) => <DataTable rows={rs} tall /> },
              ]} />
              {gates.length === 0 && <Card title="Gates"><p className="muted small">No gate verdicts in the document.</p></Card>}
            </>
          );
        }}
      </Gate>
      <Provenance result={result} />
    </>
  );
}
