/** Ops folded into Training as charts: peer vitals, leases (memory bars), run-DAG progress, broker events. */
import { Lines } from '../components/charts';
import { ModeBadge } from '../components/board';
import { useDoc, type DocResult, type Envelope } from '../lib/api';
import { arr, fmtBytes, fmtNum, isObj, num, rows, str, timeOf } from '../lib/format';
import { href } from '../lib/url';

const GB = 1024 ** 3;
const ok = (r: DocResult<Envelope>) => (r.status === 'ok' ? r.data : null);

export default function OpsCharts() {
  const live = useDoc<Envelope>('live', 10_000);
  const dags = useDoc<Envelope>('dags', 30_000);
  const l = ok(live.result), g = ok(dags.result);
  const wd = l ? rows(l.watchdog) : [];
  const data = wd.map((s) => ({
    t: timeOf(s.t), temp: num(s.gpu_temp_c), cpu: num(s.thermal_c), mem: num(s.memory_available) !== null ? num(s.memory_available)! / GB : null,
    proj: num(s.project_memory) !== null ? num(s.project_memory)! / GB : null, psi: num(s.psi_full_avg10), cores: num(s.project_cpu_cores),
  })).filter((d) => d.t !== null);
  const clock = (v: number) => new Date(v).toTimeString().slice(0, 5);
  const node = l && isObj(l.node) ? l.node : null;
  const adm = node && isObj(node.admission) ? node.admission : null;
  const leases = l ? rows(l.leases) : [];
  const act = g ? rows(g.dags).filter((d) => d.complete === false) : [];
  const done = g ? rows(g.dags).filter((d) => d.complete === true).slice(0, 6) : [];
  const events = l ? rows(l.broker_events) : [];
  const t0 = Math.min(...events.map((e) => timeOf(e.t) || Infinity)), t1 = Math.max(...events.map((e) => timeOf(e.t) || 0));
  return (
    <div className="ev-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', marginBottom: 8 }}>
      <section className="ev-panel">
        <header>Peer vitals<ModeBadge result={live.result} /><span className="meta">{adm ? (adm.stopped ? `admission STOPPED ${str(adm.reason)}` : 'admitting') : ''} · {wd.length} samples</span></header>
        {data.length ? (
          <div className="grid g2">
            <Lines title={<b>temperature °C</b>} data={data} xKey="t" xFormat={clock} height={110} syncId="ops" series={[{ key: 'temp', label: 'GPU', color: 'var(--s2)' }, { key: 'cpu', label: 'CPU', color: 'var(--s1)' }]} />
            <Lines title={<b>memory GB</b>} data={data} xKey="t" xFormat={clock} height={110} syncId="ops" series={[{ key: 'mem', label: 'available', color: 'var(--s3)' }, { key: 'proj', label: 'project', color: 'var(--s7)' }]} />
            <Lines title={<b>PSI full avg10</b>} data={data} xKey="t" xFormat={clock} height={110} syncId="ops" series={[{ key: 'psi', label: 'psi', color: 'var(--s8)' }]} />
            <Lines title={<b>project CPU cores</b>} data={data} xKey="t" xFormat={clock} height={110} syncId="ops" series={[{ key: 'cores', label: 'cores', color: 'var(--s1)' }]} />
          </div>
        ) : <p className="side-note">no watchdog samples</p>}
      </section>
      <section className="ev-panel">
        <header>Leases · memory<ModeBadge result={live.result} /><span className="meta">bar = current / declared · tick = peak · orange = memory.high</span></header>
        <div style={{ display: 'grid', gap: 3 }}>
          {leases.map((x) => {
            const dec = isObj(x.declared) ? x.declared : {}, m = isObj(x.measured) ? x.measured : {};
            const d = num(dec.memory_bytes) || 1, c = num(m.memory_current) || 0, p = num(m.memory_peak), h = num(m.memory_high);
            const thr = num(m.memory_high_events) || 0;
            const f = (v: number) => `${Math.min(100, (v / d) * 100)}%`;
            return (
              <a key={str(x.id)} href={href('live', { lease: str(x.id), data: '1' })} style={{ display: 'grid', gridTemplateColumns: '150px 1fr 70px', gap: 6, alignItems: 'center', color: 'var(--ink)', fontSize: 11 }}
                title={`${str(x.label)} · ${str(x.workstream)} ${str(x.dag_node)} · current ${fmtBytes(c)} / declared ${fmtBytes(d)} · peak ${fmtBytes(p)} · memory.high events ${thr}`}>
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{str(x.label)}</span>
                <span className="ibar" style={{ height: 8 }}>
                  <i style={{ width: f(c), background: thr ? 'var(--critical)' : 'var(--accent)' }} />
                  {h !== null && <u style={{ left: f(h) }} />}
                  {p !== null && <b style={{ left: f(p) }} />}
                </span>
                <span className="mono" style={{ fontSize: 10 }}>{(c / GB).toFixed(1)}/{(d / GB).toFixed(0)}G{dec.gpu ? ' gpu' : ''}</span>
              </a>
            );
          })}
          {!leases.length && <p className="side-note">no active leases</p>}
        </div>
      </section>
      <section className="ev-panel">
        <header>Run DAGs<ModeBadge result={dags.result} /><span className="meta">done · running · failed · planned; ETA only where recorded</span></header>
        <div className="dagbars">
          {[...act, ...done].map((d) => {
            const c = isObj(d.counts) ? d.counts : {};
            const total = Object.values(c).reduce<number>((a, v) => a + (num(v) || 0), 0) || 1;
            const eta = arr(d.eta_statements).map((e) => str(isObj(e) ? e.text ?? e.eta : e)).join('; ');
            return (
              <a key={`${str(d.dag)}@${str(d.location)}`} href={href('live', { ws: str(d.track), data: '1' })} title={`${str(d.track)} · ${Object.entries(c).map(([k, v]) => `${k} ${v}`).join(', ')}${eta ? ` · ETA ${eta}` : ''}`}>
                <span>{str(d.dag)}</span>
                <span className="bar">{['completed', 'running', 'failed'].map((k) => <i key={k} style={{ width: `${((num(c[k]) || 0) / total) * 100}%`, background: k === 'completed' ? 'var(--good)' : k === 'running' ? 'var(--accent)' : 'var(--critical)' }} />)}</span>
                <b>{fmtNum(c.completed ?? 0)}/{total}{eta ? ` · ${eta}` : ''}</b>
              </a>
            );
          })}
        </div>
      </section>
      <section className="ev-panel">
        <header>Broker events<ModeBadge result={live.result} /><span className="meta">{events.length} recent · hover a mark</span></header>
        {events.length ? (
          <svg width="100%" height={46} viewBox={`0 0 400 46`} role="img" aria-label="broker events over time">
            {[['acquired', 6], ['released|expired', 20], ['stopped|revoke|throttl|kill|oom', 34]].map(([re, y]) => (
              <g key={String(re)}>
                <text x={0} y={Number(y) + 3} fill="var(--muted)">{String(re).split('|')[0]}</text>
                {events.filter((e) => new RegExp(String(re)).test(str(e.kind))).map((e, i) => {
                  const x = 80 + (((timeOf(e.t) || t0) - t0) / (t1 - t0 || 1)) * 310;
                  return <rect key={i} x={x} y={Number(y) - 4} width={2} height={8} fill={/stopped|revoke|kill|oom/.test(str(e.kind)) ? 'var(--critical)' : /acquired/.test(str(e.kind)) ? 'var(--good)' : 'var(--axis)'}><title>{`${new Date(timeOf(e.t) || 0).toTimeString().slice(0, 8)} ${str(e.kind)} ${str(e.label) || str(e.lease_id)} ${str(e.reason)}`}</title></rect>;
                })}
              </g>
            ))}
          </svg>
        ) : <p className="side-note">no broker events</p>}
      </section>
    </div>
  );
}
