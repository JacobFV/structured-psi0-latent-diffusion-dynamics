import { SideGroup, SidebarControls } from '../components/shell';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Lines } from '../components/charts';
import { Card, DataTable, Gate, KV, ModeBanner, PageHead, Provenance, Stat, Status } from '../components/ui';
import { useDoc, type DocResult, type Envelope } from '../lib/api';
import { ageSeconds, ago, arr, fmtBytes, fmtNum, fmtTime, isObj, num, pick, rows, str, timeOf, uniq, type Row } from '../lib/format';
import { stateTone } from '../lib/labels';
import { useUrlState } from '../lib/url';

const GB = 1024 ** 3;
function psiAvg10(s: unknown, which: 'some' | 'full' = 'some') {
  const m = new RegExp(`${which} avg10=([0-9.]+)`).exec(str(s));
  return m ? Number(m[1]) : null;
}
function gpuUtil(node: Row) {
  const g = arr(node.gpu)[0];
  return num(pick(g, 'util_pct', 'utilization', 'util'));
}

export default function LiveOps() {
  const live = useDoc<Envelope>('live', 10_000);
  const dags = useDoc<Envelope>('dags', 30_000);
  return (
    <>
      <PageHead
        title="Live ops"
        sub="Peer vitals, admission, leases (declared vs measured), broker and watchdog events, and run-DAG progress per workstream. Read-only: the room never starts or stops anything. Live polls every 10 s (one bounded ssh read of the peer per export)."
        right={<StaleFlag result={live.result} />}
      />
      <ModeBanner result={live.result} reload={live.reload} busy={live.busy} />
      <Gate result={live.result} what="live ops (/api/live)">{(d) => <LiveBody d={d} />}</Gate>
      <h2 className="sect">Run DAGs</h2>
      <ModeBanner result={dags.result} reload={dags.reload} busy={dags.busy} />
      <Gate result={dags.result} what="run DAGs (/api/dags)">{(d) => <Dags d={d} />}</Gate>
      <Provenance result={live.result} />
      <Provenance result={dags.result} />
    </>
  );
}

function StaleFlag({ result }: { result: DocResult<Envelope> }) {
  const [, tick] = useState(0);
  useEffect(() => { const id = window.setInterval(() => tick((x) => x + 1), 5000); return () => window.clearInterval(id); }, []);
  if (result.status !== 'ok') return null;
  const d = result.data;
  const readAt = pick(d, 'peer_read_at', 'generated_at');
  const age = ageSeconds(readAt);
  const stale = d.stale === true || result.mode === 'stale' || (result.mode !== 'fixture' && age !== null && age > 60);
  return (
    <span className={`badge ${stale ? 'caveat' : ''}`} style={{ fontSize: 13, padding: '3px 10px' }} title="stale when the last good peer read is older than 60 s">
      <Status state={stale ? 'stale' : 'ok'}>{stale ? `STALE · peer read ${readAt ? ago(readAt) : 'never'}` : `live · peer read ${ago(readAt)}`}</Status>
    </span>
  );
}

function LiveBody({ d }: { d: Envelope }) {
  const node = (isObj(d.node) ? d.node : isObj(d.peer) ? d.peer : {}) as Row;
  const adm = isObj(node.admission) ? node.admission : {};
  const stopped = adm.stopped;
  const wl = isObj(node.watchdog_last) ? node.watchdog_last : null;
  const samples = rows(pick(d, 'watchdog', 'watchdog_samples'));
  const leases = rows(pick(d, 'leases'));
  const events = rows(pick(d, 'broker_events'));
  const host = isObj(d.host) ? d.host : null;
  const limits = isObj(node.limits) ? node.limits : null;
  const memAvail = num(node.memory_available), memTotal = num(node.mem_total);
  const gpuHist = useGpuHistory(d);
  return (
    <div className="stack">
      {arr(d.peer_errors).length > 0 && <div className="state error">Peer read errors: {arr(d.peer_errors).map(str).join('; ')}</div>}
      {d.last_error ? <div className="state error">Last error: {str(d.last_error)}</div> : null}
      <div className="grid g4">
        <Stat k={`Admission (${str(d.peer) || 'peer'})`}
          v={<Status state={stopped === true ? 'blocked' : stopped === false ? 'ok' : 'unknown'}>{stopped === true ? 'STOPPED' : stopped === false ? 'open' : '—'}</Status>}
          s={str(adm.reason) || (stopped === false ? 'admitting new leases' : 'no reason recorded')} />
        <Stat k="GPU utilisation" v={<span className="num">{gpuUtil(node) !== null ? `${gpuUtil(node)}%` : '—'}</span>}
          s={arr(node.gpu).map((g) => `${str(pick(g, 'name'))} · ${str(pick(g, 'temp_c'))} °C · ${str(pick(g, 'power_w'))} W`).join(' · ') || 'not in document'} />
        <Stat k="Temperatures" v={<span className="num">{fmtNum(node.gpu_temp_c)} / {fmtNum(node.cpu_temp_c)} °C</span>} s="GPU / CPU" />
        <Stat k="Memory available" v={<span className="num">{memAvail !== null ? fmtBytes(memAvail) : '—'}</span>}
          s={memTotal ? `of ${fmtBytes(memTotal)} · project ${fmtBytes(node.project_memory)}` : ''} />
        <Stat k="PSI memory (avg10)" v={<span className="num">{fmtNum(psiAvg10(node.psi_memory))} / {fmtNum(psiAvg10(node.psi_memory, 'full'))}</span>} s={`some / full · cpu some ${fmtNum(psiAvg10(node.psi_cpu))}`} />
        <Stat k="Load average" v={<span className="num">{str(node.loadavg).split(' ').slice(0, 3).join(' ') || '—'}</span>} s={limits ? `limit ${fmtNum(limits.cpu_cores)} cores` : ''} />
        <Stat k="Disk free" v={<span className="num">{fmtBytes(node.disk_free)}</span>} s={`home ${fmtBytes(node.home_free_bytes)} · shm ${fmtBytes(node.shm_free_bytes)}`} />
        <Stat k="Watchdog" v={<Status state={wl ? pick(wl, 'level') : 'unknown'}>{wl ? str(pick(wl, 'level')) : '—'}</Status>}
          s={`heartbeat ${node.watchdog_heartbeat_age_s != null ? `${fmtNum(node.watchdog_heartbeat_age_s)} s ago` : '—'}${wl && arr(wl.reasons).length ? ` · ${arr(wl.reasons).map(str).join('; ')}` : ''}`} />
      </div>
      <Card title="Peer vitals timeline" hint={`${samples.length} watchdog samples (last 200) · dashed lines mark samples whose level is not ok`}>
        <Vitals samples={samples} gpuHist={gpuHist} />
      </Card>
      <Card title="Leases" hint={`${leases.length} active${d.n_leases_total != null ? ` · ${fmtNum(d.n_leases_total)} total in the broker` : ''} · declared vs measured memory, memory.high throttle and OOM events`}>
        <Leases leases={leases} />
      </Card>
      <div className="grid g2">
        <Card title="Broker events" hint="lease, admission, revoke and throttle events (most recent first)">
          <BrokerEvents events={events} />
        </Card>
        <Card title="Watchdog events" hint="samples whose level is not ok">
          <WatchdogEvents samples={samples} />
        </Card>
      </div>
      <div className="grid g2">
        <Card title="Peer limits">{limits ? <KV data={Object.fromEntries(Object.entries(limits).map(([k, v]) => [k, /bytes/.test(k) ? fmtBytes(v) : v]))} /> : <p className="muted small">Not in the document.</p>}</Card>
        <Card title="Host basics" hint={str(pick(host, 'gpu'))}>{host ? <KV data={host} /> : <p className="muted small">No host section in the document.</p>}</Card>
      </div>
      {arr(d.notes).length > 0 && <ul className="small muted">{arr(d.notes).map((n, i) => <li key={i}>{str(n)}</li>)}</ul>}
    </div>
  );
}

/** GPU utilisation is only a point reading in each live document; this keeps the readings this browser has seen. */
function useGpuHistory(d: Envelope) {
  const hist = useRef<{ t: number; util: number }[]>([]);
  const node = isObj(d.node) ? d.node : {};
  const t = timeOf(pick(d, 'peer_read_at', 'generated_at'));
  const u = gpuUtil(node as Row);
  if (t !== null && u !== null && !hist.current.some((h) => h.t === t)) {
    hist.current = [...hist.current, { t, util: u }].slice(-360);
  }
  return hist.current;
}

const PANELS: { id: string; title: string; unit: string; keys: RegExp[]; scale?: number }[] = [
  { id: 'temp', title: 'Temperatures', unit: '°C', keys: [/^gpu_temp_c$/, /^thermal_c$/, /cpu.*temp/] },
  { id: 'mem', title: 'Memory', unit: 'GB', keys: [/^memory_available$/, /^project_memory$/, /^project_memory_current$/, /^swap_free$/], scale: GB },
  { id: 'psi', title: 'PSI (full avg10)', unit: '%', keys: [/^psi/] },
  { id: 'cpu', title: 'CPU cores', unit: 'cores', keys: [/^idle_cores$/, /^project_cpu_cores$/] },
  { id: 'gpumem', title: 'Project GPU memory', unit: 'GB', keys: [/^project_gpu_bytes$/], scale: GB },
];
const COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s7)'];

function Vitals({ samples, gpuHist }: { samples: Row[]; gpuHist: { t: number; util: number }[] }) {
  const data = useMemo(() => samples.map((s) => {
    const o: Row = { __t: timeOf(pick(s, 't', 'ts', 'time')) };
    for (const [k, v] of Object.entries(s)) if (typeof v === 'number') o[k] = v;
    return o;
  }).filter((s) => s.__t !== null), [samples]);
  const fmtClock = (v: number) => new Date(v).toTimeString().slice(0, 8);
  const numericKeys = uniq(data.flatMap((s) => Object.keys(s).filter((k) => k !== '__t' && k !== 't')));
  const warn = samples.filter((s) => !/^(ok|normal)$/i.test(str(s.level))).map((s) => timeOf(s.t)).filter((x): x is number => x !== null);
  return (
    <div className="grid g3">
      {PANELS.map((p) => {
        const ks = uniq(p.keys.flatMap((re) => numericKeys.filter((k) => re.test(k)))).slice(0, 4);
        if (!ks.length) return <div key={p.id} className="muted small">{p.title}: not in the samples.</div>;
        const scaled = p.scale ? data.map((s) => Object.fromEntries(Object.entries(s).map(([k, v]) => [k, k !== '__t' && typeof v === 'number' ? v / p.scale! : v]))) : data;
        return (
          <Lines key={p.id} title={<b>{p.title}</b>} right={<span className="muted small">{p.unit}</span>} data={scaled} xKey="__t" syncId="vitals" height={170}
            xFormat={fmtClock} series={ks.map((k, i) => ({ key: k, label: k, color: COLORS[i] }))}
            refs={warn.slice(-12).map((x) => ({ x, color: 'var(--serious)' }))} />
        );
      })}
      <Lines title={<b>GPU utilisation</b>} right={<span className="muted small">% · readings seen by this page ({gpuHist.length})</span>}
        data={gpuHist.map((h) => ({ __t: h.t, util: h.util }))} xKey="__t" height={170} xFormat={fmtClock} yDomain={[0, 100]}
        series={[{ key: 'util', label: 'util_pct', color: COLORS[0], dots: gpuHist.length < 20 }]} />
    </div>
  );
}

function bytesOrNum(v: unknown) {
  const n = num(v);
  return n !== null && n > 1e6 ? fmtBytes(n) : fmtNum(v);
}

function Leases({ leases }: { leases: Row[] }) {
  const [sel, setSel] = useUrlState('lease', '');
  if (!leases.length) return <p className="muted small">No active leases in the document.</p>;
  const cur = leases.find((l) => str(l.id) === sel);
  return (
    <>
      <div className="table-wrap tall">
        <table className="t">
          <thead><tr>
            <th>label</th><th>workstream · DAG node</th><th>memory: measured / declared</th><th className="n">memory.high</th><th className="n">throttle ev.</th><th className="n">OOM</th>
            <th className="n">CPU</th><th>GPU</th><th className="n">age</th><th className="n">heartbeat</th>
          </tr></thead>
          <tbody>
            {leases.map((l) => {
              const dec = isObj(l.declared) ? l.declared : {}, mea = isObj(l.measured) ? l.measured : {};
              const hi = num(mea.memory_high_events) || 0, oom = (num(mea.oom_events) || 0) + (num(mea.oom_kill_events) || 0);
              return (
                <tr key={str(l.id)} className={str(l.id) === sel ? 'sel' : ''} onClick={() => setSel(str(l.id) === sel ? '' : str(l.id))} style={{ cursor: 'pointer' }}>
                  <td><b style={{ fontWeight: 600 }}>{str(l.label)}</b><div className="small muted mono">{str(l.id)}</div></td>
                  <td>{str(l.workstream) || <span className="muted">—</span>}<div className="small muted">{[l.dag, l.dag_node].map(str).filter(Boolean).join(' · ')}</div></td>
                  <td><LeaseBar declared={num(dec.memory_bytes)} current={num(mea.memory_current)} peak={num(mea.memory_peak)} high={num(mea.memory_high)} /></td>
                  <td className="n">{bytesOrNum(mea.memory_high)}</td>
                  <td className="n">{hi ? <Status state="throttled">{hi}</Status> : '0'}</td>
                  <td className="n">{oom ? <Status state="failed">{oom}</Status> : '0'}</td>
                  <td className="n">{fmtNum(dec.cpu_cores)}</td>
                  <td>{dec.gpu ? `yes · ${fmtBytes(dec.gpu_memory_bytes)}` : 'no'}</td>
                  <td className="n">{l.age_s != null ? `${fmtNum(Math.round((num(l.age_s) || 0) / 60))} min` : '—'}</td>
                  <td className="n">{l.heartbeat_at != null ? ago(l.heartbeat_at) : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small">Bar: measured current memory (fill) against the declared memory (full width); black tick = measured peak; orange tick = memory.high (throttle threshold). Click a row for all fields.</p>
      {cur && <KV data={cur} />}
    </>
  );
}
function LeaseBar({ declared, current, peak, high }: { declared: number | null; current: number | null; peak: number | null; high: number | null }) {
  if (!declared) return <span className="muted">no declared memory</span>;
  const f = (v: number) => `${Math.min(100, (v / declared) * 100)}%`;
  const over = (peak ?? current ?? 0) > declared;
  return (
    <div style={{ display: 'grid', gap: 2, minWidth: 170 }}>
      <div className="cell-bar">
        {current !== null && <i style={{ width: f(current), background: over ? 'var(--critical)' : 'var(--accent)' }} />}
        {high !== null && <b style={{ left: f(high), background: 'var(--serious)' }} />}
        {peak !== null && <b style={{ left: f(peak) }} />}
      </div>
      <span className="small num muted">{fmtBytes(current)} / {fmtBytes(declared)} · peak {fmtBytes(peak)}{over ? ' · OVER' : ''}</span>
    </div>
  );
}

function BrokerEvents({ events }: { events: Row[] }) {
  const [kind, setKind] = useState('');
  if (!events.length) return <p className="muted small">No broker events in the document.</p>;
  const kinds = uniq(events.map((e) => str(e.kind))).sort();
  const shown = events.filter((e) => !kind || str(e.kind) === kind).slice().reverse();
  return (
    <>
      <div className="filters" style={{ marginBottom: 6 }}>
        <label>kind<select value={kind} onChange={(e) => setKind(e.target.value)}><option value="">all ({events.length})</option>{kinds.map((k) => <option key={k}>{k}</option>)}</select></label>
      </div>
      <div className="table-wrap tall">
        <table className="t">
          <thead><tr><th>time</th><th>kind</th><th>lease / label</th><th>reason · resources</th></tr></thead>
          <tbody>
            {shown.map((e, i) => {
              const k = str(e.kind);
              const tone = /stopped|revoke|throttl|kill|oom/.test(k) ? 'critical' : /released/.test(k) ? 'neutral' : /acquired|resumed|started/.test(k) ? 'good' : 'warning';
              return (
                <tr key={i}>
                  <td className="num small">{fmtTime(timeOf(e.t)).slice(11)}</td>
                  <td><span className={`status ${tone}`}><i />{k}</span></td>
                  <td className="small">{str(e.label) || <span className="mono muted">{str(e.lease_id)}</span>}</td>
                  <td className="small">{str(e.reason)}{e.mem != null ? ` mem ${fmtBytes(e.mem)}` : ''}{e.cpu != null ? ` · cpu ${fmtNum(e.cpu)}` : ''}{e.gpu ? ' · gpu' : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

function WatchdogEvents({ samples }: { samples: Row[] }) {
  const events = samples.filter((s) => { const lv = str(pick(s, 'level')); return lv && !/^(ok|normal|info)$/i.test(lv); });
  const throttled = samples.filter((s) => arr(s.throttled_leases).length || s.gpu_thermal_throttle === true);
  if (!samples.length) return <p className="muted small">No watchdog samples in the document.</p>;
  if (!events.length && !throttled.length) return <p className="muted small">All {samples.length} samples are at level ok, with no throttled leases and no GPU thermal throttle.</p>;
  return (
    <DataTable tall rows={[...events, ...throttled.filter((t) => !events.includes(t))].reverse().map((s) => ({
      time: fmtTime(timeOf(s.t)), level: s.level, reasons: arr(s.reasons).map(str).join('; '),
      throttled_leases: arr(s.throttled_leases).map(str).join(', ') || undefined, gpu_thermal_throttle: s.gpu_thermal_throttle || undefined,
    }))} />
  );
}

function Dags({ d }: { d: Envelope }) {
  const list = rows(pick(d, 'dags'));
  const [open, setOpen] = useUrlState('dag', '');
  const [ws, setWs] = useUrlState('ws', '');
  const [hideDone, setHideDone] = useUrlState('active', '0');
  if (!list.length) return <p className="muted small">No run-DAG ledgers in the document.</p>;
  const wsOf = (g: Row) => str(pick(g, 'workstream', 'track')) || 'other';
  const workstreams = uniq(list.map(wsOf)).sort();
  const shown = list.filter((g) => (!ws || wsOf(g) === ws) && (hideDone !== '1' || g.complete !== true));
  const legend: [string, string][] = [['good', 'completed'], ['active', 'running'], ['neutral', 'pending / planned'], ['serious', 'blocked / held'], ['critical', 'failed']];
  return (
    <div className="stack">
      <SidebarControls>
        <SideGroup title="Run DAGs">
          <div className="filters">
            <label>workstream<select value={ws} onChange={(e) => setWs(e.target.value)}><option value="">all ({workstreams.length})</option>{workstreams.map((w) => <option key={w}>{w}</option>)}</select></label>
            <label className="check"><input type="checkbox" checked={hideDone === '1'} onChange={(e) => setHideDone(e.target.checked ? '1' : '0')} /> only incomplete</label>
            <span className="legend">{legend.map(([t, l]) => <span key={t}><i className="sw" style={{ background: t === 'active' ? 'var(--accent)' : t === 'neutral' ? 'var(--axis)' : `var(--${t})` }} />{l}</span>)}</span>
          </div>
        </SideGroup>
      </SidebarControls>
      {arr(d.notes).length > 0 && <ul className="small muted" style={{ margin: 0 }}>{arr(d.notes).map((n, i) => <li key={i}>{str(n)}</li>)}</ul>}
      {workstreams.filter((w) => shown.some((g) => wsOf(g) === w)).map((w) => {
        const gs = shown.filter((g) => wsOf(g) === w);
        return (
          <Card key={w} title={w} hint={`${gs.length} DAG(s)`}>
            <div className="stack" style={{ gap: 12 }}>
              {gs.map((g, i) => {
                const name = str(pick(g, 'dag', 'name')) || `dag ${i}`;
                const where = str(pick(g, 'location', 'host'));
                const nodes = rows(pick(g, 'nodes'));
                const counts = isObj(g.counts) ? g.counts : null;
                const etas = arr(pick(g, 'eta_statements', 'eta'));
                const key = `${name}@${where}`;
                const stages = uniq(nodes.map((n) => str(n.stage)));
                return (
                  <div key={key}>
                    <div className="row" style={{ marginBottom: 4 }}>
                      <b style={{ fontWeight: 600 }}>{name}</b>
                      {where && <span className="badge">{where}</span>}
                      {g.complete === true && <Status state="completed">complete</Status>}
                      <span className="muted small num">{counts ? Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(' · ') : `${nodes.length} nodes`}</span>
                      <span className="small muted">updated {ago(pick(g, 'updated', 'mtime'))}</span>
                      <span className="spacer" />
                      <button className="ghost small" onClick={() => setOpen(open === key ? '' : key)}>{open === key ? 'hide nodes' : `${nodes.length} nodes`}</button>
                    </div>
                    <div className="bars" role="img" aria-label={`${name}: ${nodes.length} nodes by state`}>
                      {nodes.map((n, j) => (
                        <i key={j} className={stateTone(n.state)} style={{ flex: 1 }}
                          title={`${str(n.id)} · ${str(n.stage)} · ${str(n.state)}${n.rc != null ? ` · rc ${str(n.rc)}` : ''}${n.gate_verdict ? ` · gate ${str(n.gate_verdict)}` : ''}${n.caveat ? ` · ${str(n.caveat)}` : ''}`} />
                      ))}
                    </div>
                    <div className="small muted" style={{ marginTop: 2 }}>
                      {etas.length ? <>ETA as stated in track notes: {etas.map((e) => (isObj(e) ? `${str(pick(e, 'text', 'eta'))} (${str(pick(e, 'source_file', 'source'))})` : str(e))).join(' · ')}</> : 'no ETA stated in track notes (none is estimated here)'}
                      {stages.length > 1 && <> · stages: {stages.join(' → ')}</>}
                    </div>
                    {open === key && <div style={{ marginTop: 8 }}><DataTable tall rows={nodes.map((n) => ({ id: n.id, stage: n.stage, state: n.state, placement: n.placement, lease: n.lease, rc: n.rc, attempts: n.attempts, gate_verdict: n.gate_verdict, caveat: n.caveat, started: n.started, ended: n.ended, out: n.out }))} /></div>}
                  </div>
                );
              })}
            </div>
          </Card>
        );
      })}
    </div>
  );
}
