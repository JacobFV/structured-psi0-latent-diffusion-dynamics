/**
 * Live ticker: real events only, taken from the exporter documents (broker events and non-ok watchdog samples from /api/live,
 * run-DAG node starts/ends from /api/dags, latest decisions from /api/overview). Newest first; pauses on hover; static under
 * prefers-reduced-motion. If there is nothing, it says so.
 */
import { useDoc, type Envelope } from '../lib/api';
import { arr, isObj, rows, str, timeOf, type Row } from '../lib/format';
import { href } from '../lib/url';

type Ev = { t: number; cls: 'active' | 'failed' | 'pending' | 'commit' | 'info'; text: string; link?: string };

function brokerEvents(live: Envelope): Ev[] {
  return rows(live.broker_events).map((e) => {
    const k = str(e.kind);
    const cls: Ev['cls'] = /stopped|revoke|kill|oom|throttl|fail/.test(k) ? 'failed' : /acquired|resumed|started/.test(k) ? 'active' : /expired|queued|wait/.test(k) ? 'pending' : 'commit';
    const who = str(e.label) || str(e.lease_id);
    return { t: timeOf(e.t) ?? 0, cls, text: `${k.replace(/_/g, ' ').toUpperCase()} ${who}${e.reason ? ` · ${str(e.reason)}` : ''}${e.mem ? ` · ${(Number(e.mem) / 1024 ** 3).toFixed(1)}G` : ''}`, link: href('live') };
  });
}
function watchdogEvents(live: Envelope): Ev[] {
  return rows(live.watchdog).filter((s) => !/^(ok|normal)$/i.test(str(s.level))).map((s) => ({
    t: timeOf(s.t) ?? 0, cls: 'failed' as const, text: `WATCHDOG ${str(s.level).toUpperCase()} ${arr(s.reasons).map(str).join('; ')}`, link: href('live'),
  }));
}
function dagEvents(dags: Envelope): Ev[] {
  const out: Ev[] = [];
  for (const g of rows(dags.dags)) {
    const dag = str(pick2(g, 'dag', 'name'));
    for (const n of rows(g.nodes)) {
      const st = str(n.state);
      const ended = timeOf(n.ended), started = timeOf(n.started);
      if (ended) out.push({ t: ended, cls: /fail/.test(st) ? 'failed' : 'active', text: `${/fail/.test(st) ? '✗' : '✓'} ${dag}/${str(n.id)} ${st}${n.rc != null ? ` rc=${str(n.rc)}` : ''}`, link: href('live', { tab: 'dags' }) });
      else if (started && /run/.test(st)) out.push({ t: started, cls: 'pending', text: `▶ ${dag}/${str(n.id)} running`, link: href('live', { tab: 'dags' }) });
    }
  }
  return out;
}
function pick2(o: Row, a: string, b: string) { return o[a] ?? o[b]; }
function decisionEvents(ov: Envelope): Ev[] {
  return rows(ov.latest_decisions ?? ov.decisions).map((d) => ({
    t: timeOf(d.date) ?? 0, cls: 'info' as const, text: `${str(d.id)} ${str(d.title)}`, link: href('knowledge', { tab: 'decisions', d: str(d.id) }),
  }));
}

export default function Ticker() {
  const live = useDoc<Envelope>('live', 10_000);
  const dags = useDoc<Envelope>('dags', 30_000);
  const ov = useDoc<Envelope>('overview', 60_000);
  const evs: Ev[] = [];
  if (live.result.status === 'ok') evs.push(...brokerEvents(live.result.data), ...watchdogEvents(live.result.data));
  if (dags.result.status === 'ok') evs.push(...dagEvents(dags.result.data));
  if (ov.result.status === 'ok') evs.push(...decisionEvents(ov.result.data));
  evs.sort((a, b) => b.t - a.t);
  const shown = evs.slice(0, 60);
  const liveOk = live.result.status === 'ok';
  const stale = liveOk && (live.result.status === 'ok' && (live.result.data.stale === true || live.result.mode !== 'live'));
  const mode = live.result.status === 'ok' ? live.result.mode : null;
  const label = !liveOk ? 'NO FEED' : mode === 'fixture' ? 'FIXTURE' : mode === 'snapshot' ? 'SNAPSHOT' : stale ? 'STALE' : 'LIVE';
  const item = (e: Ev, i: number, dup = false) => (
    <a key={`${dup ? 'd' : ''}${i}`} className={`tk tk-${e.cls}`} href={e.link} style={{ color: undefined }} aria-hidden={dup || undefined} tabIndex={dup ? -1 : undefined}>
      <time>{e.t ? new Date(e.t).toTimeString().slice(0, 5) : ''}</time>{e.text}
    </a>
  );
  return (
    <div className="ticker" role="marquee" aria-label="latest events">
      <span className={`ticker-label ${label === 'LIVE' ? '' : 'stale'}`} title={isObj(live.result) && live.result.status === 'ok' ? `peer read ${str(live.result.data.peer_read_at)}` : 'the live document is not available'}>{label}</span>
      {shown.length === 0 ? <span style={{ paddingLeft: 12 }}>no events in the exporter documents yet</span> : (
        <div className="ticker-track" style={{ ['--ticker-s' as string]: `${Math.max(40, shown.length * 5)}s` }}>
          {shown.map((e, i) => item(e, i))}
          {shown.map((e, i) => item(e, i, true))}
        </div>
      )}
    </div>
  );
}
