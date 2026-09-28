import { useState } from 'react';
import { Card, DataTable, Did, Gate, KV, ModeBanner, PageHead, Provenance, RawDoc, Select, Status, TablesBrowser } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, num, pick, rows, sortNatural, str, uniq, type Row } from '../lib/format';
import { stateTone } from '../lib/labels';
import { useUrlState } from '../lib/url';

export default function Physics() {
  const { result, reload, busy } = useDoc<Envelope>('physics');
  return (
    <>
      <PageHead
        title="Physics credibility"
        sub="Whether the simulator and trackers can carry each claim: tracker validation and contact-v2 gates (slip ratio, duty, clearance, cost of transport per mode), grasp and dataset gates with their criteria, and the gate backfill tables (D-093..D-118)."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="physics (/api/physics)">{(d) => <PhysicsBody d={d} />}</Gate>
      <Provenance result={result} />
    </>
  );
}

function PhysicsBody({ d }: { d: Envelope }) {
  const trackers = rows(pick(d, 'trackers'));
  const gates = rows(pick(d, 'gates'));
  const dsg = rows(pick(d, 'dataset_gates'));
  const tables = rows(pick(d, 'gate_tables'));
  const counts = gates.reduce<Record<string, number>>((a, g) => { const v = str(g.verdict) || 'unknown'; a[v] = (a[v] || 0) + 1; return a; }, {});
  const tPass = trackers.filter((t) => t.passed === true).length;
  const cPass = trackers.filter((t) => isObj(t.contact_gate) && t.contact_gate.passed === true).length;
  return (
    <div className="stack">
      <div className="grid g3">
        <div className="card stat"><div className="k">Gate verdicts</div><div className="v row">{Object.entries(counts).map(([v, n]) => <Status key={v} state={v}>{n} {v}</Status>)}</div><div className="s">{gates.length} gate reports</div></div>
        <div className="card stat"><div className="k">Trackers passing the tracking gate</div><div className="v num">{tPass} / {trackers.length}</div><div className="s">{trackers.filter((t) => t.synthetic === true).length} synthetic (marked)</div></div>
        <div className="card stat"><div className="k">Trackers passing the contact gate</div><div className="v num">{cPass} / {trackers.filter((t) => isObj(t.contact_gate)).length}</div><div className="s">slip, stepping and clearance (contact v2)</div></div>
      </div>
      <Card title="Tracker validation" hint="learned tracker vs its targets, per body and version"><Trackers trackers={trackers} /></Card>
      <Card title="Gates" hint="each criterion with its value and threshold"><Gates gates={gates} /></Card>
      <Card title="Dataset gates" hint="per-body eligibility">
        <DataTable tall rows={dsg.map((g) => ({ body: g.body, passed: isObj(g.gate) ? (g.gate.passed ? 'pass' : 'fail') : undefined, ...(isObj(g.gate) ? Object.fromEntries(Object.entries(g.gate).filter(([k]) => k !== 'passed')) : {}), actuator_limits: g.actuator_limits, source_file: g.source_file }))} />
      </Card>
      <Card title="Gate backfill and summary tables" hint={`${tables.length} tables`}><TablesBrowser tables={tables} param="gtable" /></Card>
      {arr(d.notes).length > 0 && <ul className="small muted">{arr(d.notes).map((n, i) => <li key={i}>{str(n)}</li>)}</ul>}
      {d.documents !== undefined && <Card title="Other physics documents" hint={`${str(d.n_documents)}`}><RawDoc data={{ documents: d.documents } as Row} /></Card>}
    </div>
  );
}

function Trackers({ trackers }: { trackers: Row[] }) {
  const bodies = uniq(trackers.map((t) => str(t.body))).sort(sortNatural);
  const [body, setBody] = useUrlState('tbody', '');
  const [mode, setMode] = useUrlState('tmode', 'forward');
  const [sel, setSel] = useState('');
  const shown = trackers.filter((t) => !body || str(t.body) === body);
  const modes = uniq(trackers.flatMap((t) => (isObj(t.modes) ? Object.keys(t.modes) : []))).sort();
  const slipMax = Math.max(0.01, ...shown.map((t) => num(isObj(t.contact_gate) ? t.contact_gate.slip_ratio : null) || 0));
  const cur = trackers.find((t) => `${str(t.tracker_version)}@${str(t.source_file)}` === sel);
  return (
    <>
      <div className="filters">
        <Select label="body" value={body} options={bodies} onChange={setBody} />
        <Select label="mode for the metrics columns" value={mode} options={modes} onChange={setMode} all={false} />
      </div>
      <div className="table-wrap tall">
        <table className="t">
          <thead><tr>
            <th>body · tracker</th><th>tracking gate</th><th>contact gate</th><th>slip ratio (contact gate)</th>
            <th className="n">no-fall</th><th className="n">fwd ratio</th><th className="n">turn ratio</th>
            <th className="n">{mode} CoT</th><th className="n">{mode} slip m/s</th><th className="n">{mode} duty</th><th className="n">{mode} apex m</th>
          </tr></thead>
          <tbody>
            {shown.map((t, i) => {
              const g = isObj(t.gate) ? t.gate : {}, c = isObj(t.contact_gate) ? t.contact_gate : null;
              const m = isObj(t.modes) && isObj(t.modes[mode]) ? t.modes[mode] : {};
              const slip = c ? num(c.slip_ratio) : null;
              const key = `${str(t.tracker_version)}@${str(t.source_file)}`;
              return (
                <tr key={i} className={sel === key ? 'sel' : ''} onClick={() => setSel(sel === key ? '' : key)} style={{ cursor: 'pointer' }}>
                  <td><b style={{ fontWeight: 600 }}>{str(t.body)}</b>{t.synthetic ? <span className="badge src t-mock" style={{ marginLeft: 4 }}>SYNTHETIC</span> : null}<div className="small muted mono">{str(t.tracker_version)}</div></td>
                  <td><Status state={t.passed === true ? 'pass' : t.passed === false ? 'fail' : 'unknown'}>{t.passed === true ? 'pass' : t.passed === false ? 'fail' : '—'}</Status></td>
                  <td>{c ? <Status state={c.passed ? 'pass' : 'fail'}>{c.passed ? 'pass' : `fail${[c.slip_ok === false && 'slip', c.stepping_ok === false && 'stepping', c.clearance_ok === false && 'clearance'].filter(Boolean).length ? ` (${[c.slip_ok === false && 'slip', c.stepping_ok === false && 'stepping', c.clearance_ok === false && 'clearance'].filter(Boolean).join(', ')})` : ''}`}</Status> : <span className="muted">not run</span>}</td>
                  <td>{slip !== null ? <div style={{ display: 'grid', gridTemplateColumns: '90px 44px', gap: 6, alignItems: 'center' }}><div className="cell-bar"><i style={{ width: `${(slip / slipMax) * 100}%`, background: c?.slip_ok ? 'var(--good)' : 'var(--critical)' }} /></div><span className="num small">{fmtNum(slip)}</span></div> : '—'}</td>
                  <td className="n">{fmtNum(g.no_fall_rate)}</td><td className="n">{fmtNum(g.forward_ratio)}</td><td className="n">{fmtNum(g.turn_ratio)}</td>
                  <td className="n">{fmtNum(m.cot)}</td><td className="n">{fmtNum(m.slip_mps)}</td><td className="n">{m.duty_min != null ? `${fmtNum(m.duty_min)}–${fmtNum(m.duty_max)}` : '—'}</td><td className="n">{fmtNum(m.swing_apex_m)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">Slip bar: green when the contact gate's slip check passed, red when it failed. Click a row for all fields.</p>
      {cur && <KV data={cur} />}
    </>
  );
}

function Gates({ gates }: { gates: Row[] }) {
  if (!gates.length) return <p className="muted small">No gates in the document.</p>;
  return (
    <div className="stack" style={{ gap: 8 }}>
      {gates.map((g, i) => {
        const subj = isObj(g.subject) ? g.subject : {};
        const crit = rows(g.criteria);
        return (
          <details key={i} open={stateTone(g.verdict) === 'critical'}>
            <summary className="row" style={{ cursor: 'pointer' }}>
              <Status state={g.verdict}>{str(g.verdict)}</Status>
              <b style={{ fontWeight: 600 }}>{str(g.gate)}</b>
              <span className="small muted">{str(subj.name)}{subj.n != null ? ` · n ${fmtNum(subj.n)}` : ''}{arr(subj.grasp).length ? ` · ${arr(subj.grasp).map(str).join(', ')}` : ''}</span>
              <span className="small muted">{str(g.version)}</span>
              {g.decision ? <Did id={g.decision} /> : null}
              <span className="small muted">{crit.filter((c) => c.status === 'pass').length}/{crit.length} criteria pass</span>
            </summary>
            <div style={{ padding: '6px 0 6px 18px' }}>
              <DataTable rows={crit.map((c) => ({ name: c.name, status: c.status, value: c.value, threshold: c.threshold, note: c.note }))} />
              <div className="small muted">bodies: {arr(subj.bodies).map(str).join(', ') || '—'} · <code>{str(g.source_file)}</code></div>
            </div>
          </details>
        );
      })}
    </div>
  );
}
