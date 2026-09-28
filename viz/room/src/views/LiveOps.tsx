import { useMemo, useState } from 'react';
import { Lines } from '../components/charts';
import type { ReactNode } from 'react';
import { Card, cellValue, DataTable, Gate, KV, ModeBanner, PageHead, Provenance, Stat, Status } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { ageSeconds, ago, arr, fmtBytes, fmtNum, fmtTime, isObj, num, pick, rows, str, timeOf, uniq, type Row } from '../lib/format';
import { stateTone } from '../lib/labels';
import { useUrlState } from '../lib/url';

/** One level of flattening: {declared: {mem_gb: 3}} -> {"declared.mem_gb": 3}. */
function flat(r: Row, prefix = '', depth = 0): Row {
  const out: Row = {};
  for (const [k, v] of Object.entries(r)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (isObj(v) && depth < 2) Object.assign(out, flat(v, key, depth + 1));
    else out[key] = v;
  }
  return out;
}
function findKey(keys: string[], ...res: RegExp[]) {
  for (const re of res) {
    const k = keys.find((x) => re.test(x));
    if (k) return k;
  }
  return undefined;
}

const VITALS: { id: string; title: string; unit: string; res: RegExp[] }[] = [
  { id: 'gpu', title: 'GPU utilisation', unit: '%', res: [/gpu.*util/i] },
  { id: 'temp', title: 'Temperatures', unit: '°C', res: [/gpu.*temp/i, /cpu.*temp/i, /temp/i] },
  { id: 'mem', title: 'Memory available', unit: 'GB', res: [/mem.*avail/i, /avail.*mem/i, /available/i] },
  { id: 'psi', title: 'Pressure stall (PSI)', unit: '%', res: [/psi/i] },
];

export default function LiveOps() {
  const live = useDoc<Envelope>('live', 10_000);
  const dags = useDoc<Envelope>('dags', 10_000);
  return (
    <>
      <PageHead
        title="Live ops"
        sub="Peer vitals, admission, leases (declared vs measured), watchdog events and run-DAG progress. Read-only: the room never starts or stops anything. Polls every 10 s."
        right={<StaleFlag result={live.result} />}
      />
      <ModeBanner result={live.result} reload={live.reload} busy={live.busy} />
      <Gate result={live.result} what="live ops (/api/live)">
        {(d) => <LiveBody d={d} />}
      </Gate>
      <h2 style={{ fontSize: 16, margin: '22px 0 8px' }}>Run DAGs</h2>
      <ModeBanner result={dags.result} reload={dags.reload} busy={dags.busy} />
      <Gate result={dags.result} what="run DAGs (/api/dags)">
        {(d) => <Dags d={d} />}
      </Gate>
      <Provenance result={live.result} />
      <Provenance result={dags.result} />
    </>
  );
}

function StaleFlag({ result }: { result: ReturnType<typeof useDoc<Envelope>>['result'] }) {
  if (result.status !== 'ok') return null;
  const age = ageSeconds(result.data.generated_at);
  const stale = result.data.stale === true || result.mode === 'stale' || (result.mode !== 'fixture' && age !== null && age > 60);
  return (
    <span className={`badge ${stale ? 'caveat' : ''}`} style={{ fontSize: 13, padding: '3px 10px' }} title="stale if the live document is older than 60 s">
      <Status state={stale ? 'stale' : 'ok'}>{stale ? `STALE · ${ago(result.data.generated_at)}` : `fresh · ${ago(result.data.generated_at)}`}</Status>
    </span>
  );
}

function LiveBody({ d }: { d: Envelope }) {
  const peer = (isObj(d.peer) ? d.peer : isObj(d.node) ? d.node : {}) as Row;
  const pf = flat(peer);
  const keys = Object.keys(pf);
  const admission = pick(peer, 'admission') ?? pick(d, 'admission');
  const admState = isObj(admission) ? pick(admission, 'state', 'status') : admission;
  const admReason = isObj(admission) ? pick(admission, 'reason', 'reasons') : pick(peer, 'admission_reason');
  const samples = rows(pick(d, 'watchdog', 'watchdog_samples', 'samples')).map((r) => flat(r));
  const leases = rows(pick(d, 'leases', 'active_leases'));
  const host = isObj(d.host) ? flat(d.host) : null;
  const stat = (label: string, ...res: RegExp[]) => {
    const k = findKey(keys, ...res);
    return { label, key: k, v: k ? pf[k] : undefined };
  };
  const tiles = [
    stat('GPU util %', /gpu.*util/i),
    stat('GPU temp °C', /gpu.*temp/i),
    stat('CPU temp °C', /cpu.*temp/i),
    stat('Memory available', /mem.*avail/i, /avail/i),
    stat('PSI (memory, some avg10)', /psi.*mem.*some.*10/i, /psi.*mem/i, /psi/i),
    stat('Project memory', /project.*mem/i),
    stat('Disk free', /disk.*free/i),
  ];
  return (
    <div className="stack">
      <div className="grid g4">
        <Stat k="Admission" v={<Status state={admState}>{str(admState) || '—'}</Status>} s={str(admReason) || 'no reason recorded'} />
        {tiles.map((t) => (
          <Stat key={t.label} k={t.label} v={t.v === undefined ? <span className="muted">—</span> : fmtVital(t.key || '', t.v)} s={t.key ? <code>{t.key}</code> : 'not in document'} />
        ))}
      </div>
      <Card title="Peer vitals timeline" hint={`${samples.length} watchdog samples (the document carries the last 200)`}>
        <Vitals samples={samples} />
      </Card>
      <Card title="Leases" hint={`${leases.length} active · declared vs measured memory, throttle (memory.high) events`}>
        <Leases leases={leases} />
      </Card>
      <div className="grid g2">
        <Card title="Watchdog events" hint="samples whose level is not ok">
          <WatchdogEvents samples={samples} />
        </Card>
        <Card title="Host basics">
          {host ? <KV data={host} /> : <p className="muted small">No host section in the document.</p>}
        </Card>
      </div>
      <Card title="Peer (all fields)"><KV data={pf} /></Card>
    </div>
  );
}

function fmtVital(key: string, v: unknown) {
  const n = num(v);
  if (n === null) return str(v);
  if (/bytes/i.test(key)) return fmtBytes(n);
  if (/_mb$/i.test(key)) return `${fmtNum(n / 1024)} GB`;
  if (/_gb$|gib$/i.test(key)) return `${fmtNum(n)} GB`;
  return fmtNum(n);
}

function Vitals({ samples }: { samples: Row[] }) {
  const tKey = samples.length ? findKey(Object.keys(samples[0]), /^t$/, /^ts$/, /^time/, /timestamp/, /^at$/) : undefined;
  const data = useMemo(() => samples.map((s) => ({ ...s, __t: tKey ? timeOf(s[tKey]) : null })).filter((s) => s.__t !== null), [samples, tKey]);
  if (!samples.length) return <p className="muted small">No watchdog samples in the document.</p>;
  if (!tKey) return <p className="muted small">Samples have no time field (looked for t / ts / time / timestamp).</p>;
  const numericKeys = uniq(samples.flatMap((s) => Object.keys(s).filter((k) => typeof s[k] === 'number' && k !== tKey)));
  const fmtClock = (v: number) => new Date(v).toISOString().slice(11, 19);
  const colors = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s7)'];
  const panels = VITALS.map((p) => {
    const ks = uniq(p.res.flatMap((re) => numericKeys.filter((k) => re.test(k)))).slice(0, 4);
    return { ...p, ks };
  }).filter((p) => p.ks.length);
  if (!panels.length) return <p className="muted small">No numeric vitals in the samples (fields: {numericKeys.join(', ') || 'none'}).</p>;
  const warn = data.filter((s) => { const lv = str(pick(s, 'level')); return lv && stateTone(lv) !== 'good' && !/^ok|normal|info$/i.test(lv); });
  return (
    <div className="grid g2">
      {panels.map((p) => (
        <Lines
          key={p.id} title={<b>{p.title}</b>} right={<span className="muted small">{p.unit}</span>} data={data} xKey="__t" syncId="vitals" height={170}
          xFormat={fmtClock} series={p.ks.map((k, i) => ({ key: k, label: k, color: colors[i] }))}
          refs={warn.slice(-20).map((w) => ({ x: w.__t as number, color: 'var(--serious)' }))}
        />
      ))}
    </div>
  );
}

function Leases({ leases }: { leases: Row[] }) {
  const [sel, setSel] = useUrlState('lease', '');
  if (!leases.length) return <p className="muted small">No active leases in the document.</p>;
  const flatRows = leases.map((r) => flat(r));
  const keys = uniq(flatRows.flatMap((r) => Object.keys(r)));
  const dKey = findKey(keys, /declared.*mem/i, /mem.*declared/i, /^mem(ory)?(_gb|_mb)?$/i);
  const cKey = findKey(keys, /measured.*(current|now)/i, /current.*mem/i, /mem.*current/i, /^measured.*mem/i);
  const pKey = findKey(keys, /measured.*peak/i, /peak/i);
  const eKey = findKey(keys, /memory\.?high/i, /throttl/i, /high_events/i);
  const shaped = flatRows.map((r) => {
    const dv = dKey ? num(r[dKey]) : null, cv = cKey ? num(r[cKey]) : null, pv = pKey ? num(r[pKey]) : null;
    const lead: Row = {
      id: pick(r, 'id', 'lease', 'lease_id'), label: pick(r, 'label', 'name'), workstream: pick(r, 'workstream', 'track'),
    };
    return {
      ...lead,
      'measured / declared': dv && (cv !== null || pv !== null)
        ? <LeaseBar declared={dv} current={cv} peak={pv} />
        : '—',
      throttle: eKey ? r[eKey] : undefined,
      age: pick(r, 'age_s', 'age') != null ? `${fmtNum(num(pick(r, 'age_s', 'age')))} s` : undefined,
      ...Object.fromEntries(Object.entries(r).filter(([k]) => !['id', 'label', 'workstream', 'name', 'lease', 'lease_id', 'track'].includes(k))),
    } as Row;
  });
  return (
    <>
      <p className="muted small" style={{ marginTop: 0 }}>
        Bar = measured current memory (fill) against the declared memory (full width); the tick marks the measured peak. Columns used:
        declared <code>{dKey || '—'}</code>, current <code>{cKey || '—'}</code>, peak <code>{pKey || '—'}</code>, throttle <code>{eKey || '—'}</code>.
      </p>
      <LeaseTable rows={shaped} sel={sel} setSel={setSel} />
    </>
  );
}
function LeaseTable({ rows: rs, sel, setSel }: { rows: Row[]; sel: string; setSel: (s: string) => void }) {
  const cols = Object.keys(rs[0] || {}).slice(0, 14);
  return (
    <div className="table-wrap tall">
      <table className="t">
        <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rs.map((r, i) => (
            <tr key={i} className={str(r.id) === sel ? 'sel' : ''} onClick={() => setSel(str(r.id))} style={{ cursor: 'pointer' }}>
              {cols.map((c) => {
                const v = r[c];
                const node = v !== null && typeof v === 'object' && '$$typeof' in (v as object) ? (v as ReactNode) : cellValue(c, v);
                return <td key={c} className={typeof v === 'number' ? 'n' : ''}>{node}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function LeaseBar({ declared, current, peak }: { declared: number; current: number | null; peak: number | null }) {
  const f = (v: number) => `${Math.min(100, (v / declared) * 100)}%`;
  const over = (peak ?? current ?? 0) > declared;
  return (
    <div style={{ display: 'grid', gap: 2, minWidth: 140 }} title={`current ${fmtNum(current)} · peak ${fmtNum(peak)} · declared ${fmtNum(declared)}`}>
      <div className="cell-bar">
        {current !== null && <i style={{ width: f(current), background: over ? 'var(--critical)' : 'var(--accent)' }} />}
        {peak !== null && <b style={{ left: f(peak) }} />}
      </div>
      <span className="small num muted">{fmtNum(current)} / {fmtNum(declared)}{peak !== null ? ` · peak ${fmtNum(peak)}` : ''}{over ? ' · OVER' : ''}</span>
    </div>
  );
}

function WatchdogEvents({ samples }: { samples: Row[] }) {
  const events = samples.filter((s) => {
    const lv = str(pick(s, 'level'));
    return lv && !/^(ok|normal|info|green)$/i.test(lv);
  });
  if (!samples.length) return <p className="muted small">No watchdog samples in the document.</p>;
  if (!events.length) return <p className="muted small">All {samples.length} samples are at level ok.</p>;
  return (
    <DataTable
      tall
      rows={events.slice().reverse().map((s) => ({
        time: fmtTime(timeOf(pick(s, 't', 'ts', 'time', 'timestamp'))), level: pick(s, 'level'),
        reasons: arr(pick(s, 'reasons')).map(str).join('; ') || str(pick(s, 'reasons', 'reason')),
        action: pick(s, 'action', 'shed', 'killed'),
      }))}
    />
  );
}

function Dags({ d }: { d: Envelope }) {
  const list = rows(pick(d, 'dags', 'ledgers'));
  const [open, setOpen] = useUrlState('dag', '');
  const [ws, setWs] = useState('');
  if (!list.length) return <p className="muted small">No run-DAG ledgers in the document.</p>;
  const wsOf = (g: Row) => str(pick(g, 'workstream', 'track')) || str(pick(g, 'name', 'dag')).split(/[_/.-]/)[0] || 'other';
  const workstreams = uniq(list.map(wsOf)).sort();
  const shown = list.filter((g) => !ws || wsOf(g) === ws);
  return (
    <div className="stack">
      <div className="filters">
        <label>workstream
          <select value={ws} onChange={(e) => setWs(e.target.value)}>
            <option value="">all ({workstreams.length})</option>
            {workstreams.map((w) => <option key={w}>{w}</option>)}
          </select>
        </label>
        <span className="legend">
          {(['good', 'active', 'neutral', 'serious', 'critical'] as const).map((t) => (
            <span key={t}><i className="sw" style={{ background: t === 'active' ? 'var(--accent)' : t === 'neutral' ? 'var(--axis)' : `var(--${t})` }} />
              {{ good: 'completed', active: 'running', neutral: 'pending / planned', serious: 'blocked / held', critical: 'failed' }[t]}</span>
          ))}
        </span>
      </div>
      {workstreams.filter((w) => !ws || w === ws).map((w) => (
        <Card key={w} title={w} hint={`${shown.filter((g) => wsOf(g) === w).length} DAG(s)`}>
          <div className="stack" style={{ gap: 10 }}>
            {shown.filter((g) => wsOf(g) === w).map((g, i) => {
              const name = str(pick(g, 'name', 'dag')) || `dag ${i}`;
              const where = str(pick(g, 'host', 'location', 'copy'));
              const nodes = rows(pick(g, 'nodes'));
              const counts = isObj(g.counts) ? g.counts : null;
              const eta = pick(g, 'eta', 'stated_eta');
              const key = `${name}@${where}`;
              return (
                <div key={key}>
                  <div className="row" style={{ marginBottom: 4 }}>
                    <b style={{ fontWeight: 600 }}>{name}</b>
                    {where && <span className="badge">{where}</span>}
                    <span className="muted small num">{counts ? Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(' · ') : `${nodes.length} nodes`}</span>
                    {eta ? <span className="small">ETA (stated in notes): {str(eta)}</span> : <span className="muted small">no stated ETA</span>}
                    <button className="ghost small" onClick={() => setOpen(open === key ? '' : key)}>{open === key ? 'hide nodes' : 'nodes'}</button>
                  </div>
                  <div className="bars" role="img" aria-label={`${name}: ${nodes.length} nodes by state`}>
                    {nodes.map((n, j) => (
                      <i key={j} className={stateTone(pick(n, 'state'))} style={{ flex: 1 }} title={`${str(pick(n, 'id'))} · ${str(pick(n, 'stage'))} · ${str(pick(n, 'state'))}${pick(n, 'rc') != null ? ` · rc ${str(pick(n, 'rc'))}` : ''}`} />
                    ))}
                  </div>
                  {open === key && <div style={{ marginTop: 8 }}><DataTable rows={nodes} tall /></div>}
                </div>
              );
            })}
          </div>
        </Card>
      ))}
    </div>
  );
}
