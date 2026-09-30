/** Peer vitals over the watchdog window (leases and DAG progress are on the Board). */
import { Lines } from '../components/charts';
import { ModeBadge } from '../components/board';
import { useDoc, type DocResult, type Envelope } from '../lib/api';
import { isObj, num, rows, str, timeOf } from '../lib/format';

const GB = 1024 ** 3;
const ok = (r: DocResult<Envelope>) => (r.status === 'ok' ? r.data : null);

export default function OpsCharts() {
  const live = useDoc<Envelope>('live', 10_000);
  const l = ok(live.result);
  const wd = l ? rows(l.watchdog) : [];
  const data = wd.map((s) => ({
    t: timeOf(s.t), temp: num(s.gpu_temp_c), cpu: num(s.thermal_c), mem: num(s.memory_available) !== null ? num(s.memory_available)! / GB : null,
    proj: num(s.project_memory) !== null ? num(s.project_memory)! / GB : null, psi: num(s.psi_full_avg10), cores: num(s.project_cpu_cores),
  })).filter((d) => d.t !== null);
  const clock = (v: number) => new Date(v).toTimeString().slice(0, 5);
  const node = l && isObj(l.node) ? l.node : null;
  const adm = node && isObj(node.admission) ? node.admission : null;
  return (
    <div className="ev-grid" style={{ gridTemplateColumns: '1fr' }}>
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
        <p className="fig-cap">x: time of the peer watchdog samples (last 200) · y: GPU and CPU temperature (°C), memory available and project memory (GB), full memory pressure-stall (avg10 %), project CPU cores</p>
      </section>
    </div>
  );
}
