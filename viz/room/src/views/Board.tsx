/**
 * Landing board: everything at once, dense, every panel drills into its full view. Every figure comes from an exporter
 * document; aggregations (Σk/Σn, medians) are named where they are shown. Missing inputs render as "no data".
 */
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Delta, IBar, Kpi, ModeBadge, Panel, RateBar, Spark, Warn } from '../components/board';
import { divColor, seqColor, seqInk } from '../components/charts';
import { fetchTrainingSeries, useDoc, type DocResult, type Envelope } from '../lib/api';
import { ago, arr, fmtBytes, fmtNum, isObj, num, pick, rows, sortNatural, str, timeOf, uniq, wilson, type Row } from '../lib/format';
import { stateTone } from '../lib/labels';
import { href, useUrlState } from '../lib/url';

const GB = 1024 ** 3;
const ok = (r: DocResult<Envelope>) => (r.status === 'ok' ? r.data : null);
function psi(s: unknown, which = 'some') {
  const m = new RegExp(`${which} avg10=([0-9.]+)`).exec(str(s));
  return m ? Number(m[1]) : null;
}
function median(xs: number[]) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}

export default function Board() {
  const live = useDoc<Envelope>('live', 10_000);
  const dags = useDoc<Envelope>('dags', 30_000);
  const results = useDoc<Envelope>('results');
  const edits = useDoc<Envelope>('edits');
  const training = useDoc<Envelope>('training');
  const robust = useDoc<Envelope>('robustness');
  const physics = useDoc<Envelope>('physics');
  const psi0 = useDoc<Envelope>('psi0');
  const overview = useDoc<Envelope>('overview');
  const knowledge = useDoc<Envelope>('knowledge');
  const replays = useDoc<Envelope>('replays');
  const videos = useDoc<Envelope>('videos');
  return (
    <div className="board">
      <Kpis live={live.result} dags={dags.result} results={results.result} edits={edits.result} psi0={psi0.result} overview={overview.result}
        knowledge={knowledge.result} physics={physics.result} replays={replays.result} />
      <div className="board-grid">
        <VitalsPanel r={live.result} />
        <LeasesPanel r={live.result} />
        <HeatPanel r={results.result} />
        <DagPanel r={dags.result} />
        <LineagePanel r={results.result} />
        <EditsPanel r={edits.result} />
        <TrainingPanel r={training.result} />
        <RobustPanel r={robust.result} />
        <Psi0Panel r={psi0.result} />
        <GatesPanel r={physics.result} />
        <DecisionsPanel r={overview.result} />
        <CaveatsPanel r={overview.result} />
        <MediaPanel replays={replays.result} videos={videos.result} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ KPI strip */
function Kpis(p: Record<'live' | 'dags' | 'results' | 'edits' | 'psi0' | 'overview' | 'knowledge' | 'physics' | 'replays', DocResult<Envelope>>) {
  const live = ok(p.live), node = live && isObj(live.node) ? live.node : null;
  const wd = live ? rows(live.watchdog) : [];
  const gpu = node ? num(pick(arr(node.gpu)[0], 'util_pct')) : null;
  const gtemp = node ? num(node.gpu_temp_c) : null;
  const mem = node ? num(node.memory_available) : null;
  const memTot = node ? num(node.mem_total) : null;
  const psiFull = node ? psi(node.psi_memory, 'full') : null;
  const adm = node && isObj(node.admission) ? node.admission : null;
  const leases = live ? rows(live.leases) : [];
  const throttled = leases.filter((l) => isObj(l.measured) && (num(l.measured.memory_high_events) || 0) > 0).length;
  const tiles: ReactNode[] = [];
  const L = href('live');
  tiles.push(<Kpi key="gpu" label="peer GPU util" value={gpu === null ? '—' : `${gpu}%`} tone={gpu === null ? 'empty' : ''} href={L}
    sub={gtemp !== null ? `${gtemp.toFixed(0)}°C gpu · ${fmtNum(node?.cpu_temp_c)}°C cpu` : 'no reading'} spark={wd.map((s) => num(s.gpu_temp_c))} title="spark: GPU temperature from the watchdog samples" />);
  tiles.push(<Kpi key="mem" label="peer mem avail" value={mem === null ? '—' : `${(mem / GB).toFixed(1)}G`} href={L}
    tone={mem === null ? 'empty' : memTot && mem / memTot < 0.15 ? 'bad' : ''} sub={memTot ? `of ${(memTot / GB).toFixed(0)}G · proj ${fmtBytes(node?.project_memory)}` : ''}
    spark={wd.map((s) => { const v = num(s.memory_available); return v === null ? null : v / GB; })} />);
  tiles.push(<Kpi key="psi" label="PSI mem full/some" value={psiFull === null ? '—' : `${psiFull.toFixed(1)}`} href={L}
    tone={psiFull === null ? 'empty' : psiFull > 10 ? 'bad' : psiFull > 2 ? 'warn' : 'good'} sub={`some ${fmtNum(psi(node?.psi_memory))} · cpu ${fmtNum(psi(node?.psi_cpu))}`} spark={wd.map((s) => num(s.psi_full_avg10))} />);
  tiles.push(<Kpi key="adm" label="admission" value={adm ? (adm.stopped ? 'STOPPED' : 'OPEN') : '—'} tone={!adm ? 'empty' : adm.stopped ? 'bad' : 'good'} href={L} sub={str(adm?.reason) || (node ? `watchdog ${str(pick(node.watchdog_last, 'level'))}` : '')} />);
  tiles.push(<Kpi key="leases" label="leases running" value={live ? leases.length : '—'} tone={live ? '' : 'empty'} href={L}
    sub={live ? `${throttled} throttled · ${fmtNum(live.n_leases_total)} total · queue n/a` : ''} />);
  const dg = ok(p.dags);
  if (dg) {
    const active = rows(dg.dags).filter((g) => g.complete === false);
    const byWs = new Map<string, { done: number; total: number; fail: number }>();
    for (const g of active) {
      const ws = str(pick(g, 'track', 'workstream'));
      const c = isObj(g.counts) ? g.counts : {};
      const x = byWs.get(ws) || { done: 0, total: 0, fail: 0 };
      x.done += num(c.completed) || 0; x.fail += num(c.failed) || 0; x.total += rows(g.nodes).length;
      byWs.set(ws, x);
    }
    for (const [ws, x] of byWs) tiles.push(<Kpi key={`dag-${ws}`} label={`DAG ${ws}`} value={`${x.done}/${x.total}`} tone={x.fail ? 'bad' : ''} href={href('live', { tab: 'dags', ws })} sub={`${x.fail} failed · ${((x.done / (x.total || 1)) * 100).toFixed(0)}% done`} />);
  } else tiles.push(<Kpi key="dag" label="DAGs" value="—" tone="empty" />);
  const res = ok(p.results);
  if (res) {
    const lin = armLineages(rows(res.rows));
    const sum = (re: RegExp) => lin.filter((l) => re.test(l.label)).reduce((acc, l) => ({ k: acc.k + l.k, n: acc.n + l.n }), { k: 0, n: 0 });
    const sf = sum(/^semfix/i), ns = sum(/^nosem/i);
    tiles.push(<Kpi key="arm" label="arm gv2 semfix | nosem" href={href('results', { mode: 'rate', metric: 'grasp_v2', r1: 'body_or_hint', c: 'source_file' })}
      value={sf.n || ns.n ? <>{sf.n ? ((sf.k / sf.n) * 100).toFixed(0) : '—'}<span style={{ color: 'var(--muted)' }}>|</span>{ns.n ? ((ns.k / ns.n) * 100).toFixed(0) : '—'}%</> : '—'}
      tone={sf.n && ns.n ? (sf.k / sf.n > ns.k / ns.n ? 'good' : 'bad') : 'empty'}
      sub={sf.n || ns.n ? `${sf.k}/${sf.n} vs ${ns.k}/${ns.n} · compare_gc2_final` : 'no grasp_v2 compare rows'}
      title="Σk/Σn over the 'semfix, seed *' and 'nosem, seed *' lineages of artifacts/runs/armexpert_gc2eval/compare_gc2_final.json (all bodies); D-121 marks the grasp_v2 re-eval INTERIM" />);
  }
  const ed = ok(p.edits);
  if (ed) {
    const halt = rows(ed.rows).filter((r) => ['halt', 'ctx_halt'].includes(str(r.edit)) && str(r.role) === 'edit' && num(r.effect) !== null && str(r.variant));
    for (const body of uniq(halt.map((r) => str(r.body))).filter(Boolean).sort(sortNatural)) {
      const hb = halt.filter((r) => str(r.body) === body);
      const metric = mode(hb.map((r) => str(r.metric)));
      const pick1 = (sem: boolean) => hb.filter((r) => str(r.metric) === metric && /nosem/.test(str(r.variant)) !== sem).map((r) => num(r.effect)!);
      const sem = pick1(true), nos = pick1(false);
      if (!sem.length && !nos.length) continue;
      tiles.push(<Kpi key={`halt-${body}`} label={`halt · ${body}`} href={href('edits', { body, edit: hb[0] ? str(hb[0].edit) : 'halt' })}
        value={<><Delta v={median(sem)} /><span style={{ color: 'var(--muted)', fontSize: 12 }}> | </span><span style={{ fontSize: 13 }}><Delta v={median(nos)} /></span></>}
        sub={`sem | nosem median ${metric} · n ${sem.length}|${nos.length}`}
        title={`halt / ctx_halt effects on ${metric} for ${body}: median over rows of semantic variants (fixsem/semfix/sem) vs nosem`} />);
    }
  }
  const ps = ok(p.psi0);
  if (ps) {
    const s2 = rows(ps.runs).filter((r) => str(r.run).startsWith('step2'));
    for (const r of s2) {
      const k = num(r.k), n = num(r.n);
      tiles.push(<Kpi key={str(r.run)} label={`Ψ₀ s2 ${str(r.run).replace(/^step2_\w+?_/, '')}`} value={k !== null && n ? `${k}/${n}` : '—'} href={href('psi0')}
        tone={k === null || !n ? 'empty' : k / n >= 0.8 ? 'good' : k / n < 0.3 ? 'bad' : 'warn'} sub={`${str(r.task).replace(/^G1Wholebody/, '').replace(/-v0$/, '')}${r.interim ? ' · interim' : ''}`} />);
    }
  }
  const ph = ok(p.physics);
  if (ph) {
    const g = rows(ph.gates);
    const pass = g.filter((x) => stateTone(x.verdict) === 'good').length, fail = g.filter((x) => stateTone(x.verdict) === 'critical').length;
    tiles.push(<Kpi key="gates" label="gates pass/fail" value={<>{pass}<span style={{ color: 'var(--muted)' }}>/</span><span style={{ color: fail ? 'var(--critical)' : undefined }}>{fail}</span></>} href={href('physics')} sub={`${g.length} gate reports`} />);
  }
  const kn = ok(p.knowledge), ov = ok(p.overview);
  if (kn || ov) {
    const ds = kn ? rows(kn.decisions) : rows(ov?.latest_decisions);
    const today = ds.filter((d) => str(d.date) === new Date().toISOString().slice(0, 10)).length;
    tiles.push(<Kpi key="dec" label="decisions" value={kn ? fmtNum(kn.n_decisions ?? ds.length) : ds.length} href={href('knowledge')} sub={`latest ${str(ds[ds.length - 1]?.id ?? rows(ov?.latest_decisions)[0]?.id)} · ${today} today`} />);
  }
  if (ov) {
    const open = rows(ov.open);
    tiles.push(<Kpi key="open" label="roadmap open" value={open.length} href={href('knowledge', { tab: 'roadmap' })} sub={`${open.filter((o) => /run/.test(str(o.status))).length} running`} />);
    const claims = rows(ov.claims);
    tiles.push(<Kpi key="claims" label="claims est./all" value={`${claims.filter((c) => str(c.status) === 'established').length}/${claims.length}`} href={href('overview')} sub={`${arr(ov.caveats).length} caveats`} />);
  }
  const rp = ok(p.replays);
  tiles.push(<Kpi key="rep" label="replays" value={rp ? rows(rp.replays).length : '—'} tone={rp && rows(rp.replays).length ? '' : 'empty'} href={href('theatre')} sub={rp && !rows(rp.replays).length ? 'none recorded yet' : 'recorded episodes'} />);
  return <div className="kpis">{tiles}</div>;
}
/** Rows of the grasp_v2 lineage comparison: [lineage label, body, 'grasp_v2'] in compare_gc2_final.json, summed per label. */
function armLineages(rs: Row[]) {
  const by = new Map<string, { label: string; k: number; n: number; bodies: number }>();
  for (const r of rs) {
    const kp = arr(r.key_path);
    if (str(r.metric) !== 'grasp_v2' || kp.length !== 3 || !str(r.source_file).endsWith('compare_gc2_final.json')) continue;
    const label = str(kp[0]);
    const x = by.get(label) || { label, k: 0, n: 0, bodies: 0 };
    x.k += num(r.k) || 0; x.n += num(r.n) || 0; x.bodies++;
    by.set(label, x);
  }
  return [...by.values()];
}
function mode(xs: string[]) {
  const c = new Map<string, number>();
  xs.forEach((x) => c.set(x, (c.get(x) || 0) + 1));
  return [...c.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? '';
}

/* ------------------------------------------------------------------ live panels */
function VitalsPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const wd = d ? rows(d.watchdog) : [];
  const metrics: [string, string, (v: number) => string, number?][] = [
    ['gpu_temp_c', 'GPU °C', (v) => v.toFixed(0)], ['thermal_c', 'CPU °C', (v) => v.toFixed(0)],
    ['memory_available', 'mem avail', (v) => `${(v / GB).toFixed(1)}G`], ['project_memory', 'proj mem', (v) => `${(v / GB).toFixed(1)}G`],
    ['psi_full_avg10', 'PSI full', (v) => v.toFixed(1)], ['project_cpu_cores', 'proj CPU', (v) => v.toFixed(1)], ['idle_cores', 'idle cores', (v) => v.toFixed(1)],
    ['project_gpu_bytes', 'proj GPU mem', (v) => `${(v / GB).toFixed(1)}G`],
  ];
  const t0 = timeOf(wd[0]?.t), t1 = timeOf(wd[wd.length - 1]?.t);
  return (
    <Panel title="Peer vitals" href={href('live')} result={r} span={2} meta={t0 && t1 ? `${wd.length} samples · ${Math.round((t1 - t0) / 60000)} min · read ${ago(d?.peer_read_at)}` : ''}>
      <table className="bt">
        <thead><tr><th>metric</th><th>trend</th><th className="n">last</th><th className="n">min</th><th className="n">max</th><th className="n">Δ</th></tr></thead>
        <tbody>
          {metrics.map(([k, label, f]) => {
            const vs = wd.map((s) => num(s[k]));
            const fin = vs.filter((v): v is number => v !== null);
            if (!fin.length) return <tr key={k}><td>{label}</td><td colSpan={5} className="muted">not in samples</td></tr>;
            const last = fin[fin.length - 1], first = fin[0];
            return (
              <tr key={k}>
                <td>{label}</td>
                <td><Spark values={vs} width={150} height={14} /></td>
                <td className="n">{f(last)}</td><td className="n">{f(Math.min(...fin))}</td><td className="n">{f(Math.max(...fin))}</td>
                <td className="n"><Delta v={first ? (last - first) / Math.abs(first) : null} pct lowerIsBetter={/temp|psi|proj|thermal/.test(k)} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="board-note">Δ = change over the sample window (%). Levels not ok: {wd.filter((s) => !/^ok$/i.test(str(s.level))).length}.</p>
    </Panel>
  );
}

function LeasesPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const ls = d ? rows(d.leases) : [];
  return (
    <Panel title="Leases" href={href('live')} result={r} span={2} meta={d ? `${ls.length} running · ${fmtNum(d.n_leases_total)} total` : ''}>
      {!ls.length ? <p className="board-note">no active leases</p> : (
        <table className="bt">
          <thead><tr><th>lease</th><th>ws · node</th><th>mem cur / decl (▮peak ▮high)</th><th className="n">GB</th><th className="n">thr</th><th>gpu</th><th className="n">age</th></tr></thead>
          <tbody>
            {ls.map((l) => {
              const dec = isObj(l.declared) ? l.declared : {}, m = isObj(l.measured) ? l.measured : {};
              const cur = num(m.memory_current), dm = num(dec.memory_bytes), pk = num(m.memory_peak), hi = num(m.memory_high);
              const thr = num(m.memory_high_events) || 0;
              return (
                <tr key={str(l.id)} className="click" onClick={() => { window.location.hash = href('live', { lease: str(l.id) }).slice(1); }}>
                  <td className="name" title={str(l.id)}>{str(l.label)}</td>
                  <td className="name muted">{[l.workstream, l.dag_node].map(str).filter(Boolean).join(' · ') || '—'}</td>
                  <td><IBar value={cur} max={dm} peak={pk} mark={hi} width={120} color={pk && dm && pk > dm ? 'var(--critical)' : undefined} title={`cur ${fmtBytes(cur)} · peak ${fmtBytes(pk)} · high ${fmtBytes(hi)} · declared ${fmtBytes(dm)}`} /></td>
                  <td className="n">{cur !== null ? (cur / GB).toFixed(1) : '—'}/{dm !== null ? (dm / GB).toFixed(0) : '—'}</td>
                  <td className="n" style={{ color: thr ? 'var(--critical)' : undefined }}>{thr}</td>
                  <td>{dec.gpu ? 'gpu' : ''}</td>
                  <td className="n">{l.age_s != null ? `${Math.round((num(l.age_s) || 0) / 60)}m` : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

function DagPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const gs = d ? rows(d.dags) : [];
  const sorted = [...gs].sort((a, b) => Number(a.complete === true) - Number(b.complete === true) || (timeOf(b.updated) ?? 0) - (timeOf(a.updated) ?? 0));
  const order = ['completed', 'running', 'pending', 'planned', 'blocked', 'failed'];
  const toneOf = (s: string) => ({ completed: 'var(--good)', running: 'var(--accent)', failed: 'var(--critical)', blocked: 'var(--serious)' } as Record<string, string>)[s] || 'var(--axis)';
  return (
    <Panel title="Run DAGs" href={href('live', { tab: 'dags' })} result={r} span={2} meta={d ? `${gs.filter((g) => g.complete === false).length} active · ${gs.length} ledgers` : ''}>
      <table className="bt">
        <thead><tr><th>ws</th><th>dag</th><th>nodes by state</th><th className="n">done</th><th className="n">upd</th></tr></thead>
        <tbody>
          {sorted.map((g, i) => {
            const c = isObj(g.counts) ? g.counts : {};
            const total = Object.values(c).reduce<number>((a, v) => a + (num(v) || 0), 0) || rows(g.nodes).length;
            const states = Object.keys(c).sort((a, b) => (order.indexOf(a) + 99) % 99 - (order.indexOf(b) + 99) % 99);
            return (
              <tr key={i} className="click" onClick={() => { window.location.hash = href('live', { tab: 'dags', ws: str(g.track) }).slice(1); }}>
                <td className="muted">{str(pick(g, 'track', 'workstream'))}</td>
                <td className="name">{str(pick(g, 'dag', 'name'))}</td>
                <td style={{ width: '40%' }}>
                  <div style={{ display: 'flex', height: 8, gap: 1, background: 'var(--surface-3)' }} title={Object.entries(c).map(([k, v]) => `${k} ${v}`).join(' · ')}>
                    {states.map((s) => <i key={s} style={{ flex: num(c[s]) || 0, background: toneOf(s) }} />)}
                  </div>
                </td>
                <td className="n" style={{ color: num(c.failed) ? 'var(--critical)' : undefined }}>{fmtNum(c.completed ?? 0)}/{total}</td>
                <td className="n muted">{ago(g.updated).replace(' ago', '')}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="board-note"><i style={{ color: 'var(--good)' }}>■</i> completed <i style={{ color: 'var(--accent)' }}>■</i> running <i style={{ color: 'var(--axis)' }}>■</i> planned/pending <i style={{ color: 'var(--critical)' }}>■</i> failed</p>
    </Panel>
  );
}

/* ------------------------------------------------------------------ results wall */
function HeatPanel({ r }: { r: DocResult<Envelope> }) {
  const [base, setBase] = useUrlState('bb', '');
  const [fam, setFam] = useUrlState('bf', 'arm');
  const d = ok(r);
  const data = useMemo(() => {
    if (!d) return null;
    const rs = rows(d.rows).filter((x) => str(x.metric) === 'success' && (!fam || str(x.family) === fam));
    const key = (x: Row) => `${str(x.task) || '·'} · ${str(x.body) || (x.body_hint ? `${str(x.body_hint)}*` : '?')}`.replace(/^· · /, '');
    const cells = new Map<string, { k: number; n: number; rows: number; interim: boolean; caveat: boolean }>();
    for (const x of rs) {
      const k = num(x.k), n = num(x.n);
      if (k === null || !n) continue;
      const id = `${key(x)}|${str(x.route) || '?'}`;
      const c = cells.get(id) || { k: 0, n: 0, rows: 0, interim: false, caveat: false };
      c.k += k; c.n += n; c.rows++; c.interim ||= !!x.interim; c.caveat ||= !!x.caveat;
      cells.set(id, c);
    }
    const rowKeys = uniq([...cells.keys()].map((k) => k.split('|')[0])).sort(sortNatural);
    const cnt = new Map<string, number>();
    for (const k of cells.keys()) { const c = k.split('|')[1]; cnt.set(c, (cnt.get(c) || 0) + 1); }
    const cols = [...cnt.entries()].sort((a, b) => b[1] - a[1]).map((x) => x[0]).slice(0, 10);
    return { cells, rowKeys: rowKeys.slice(0, 26), cols, total: rowKeys.length, fams: uniq(rows(d.rows).map((x) => str(x.family))).filter(Boolean).sort() };
  }, [d, fam]);
  const b = data && data.cols.includes(base) ? base : '';
  return (
    <Panel title="Results wall · success" href={href('results', { r1: 'task', r2: 'body_or_hint', c: 'route', agg: 'pool', f_family: fam || undefined })} result={r} span={2} rows2
      meta={<>
        <select value={fam} onChange={(e) => setFam(e.target.value)} onClick={(e) => e.stopPropagation()} aria-label="family">{(data?.fams || []).map((f) => <option key={f}>{f}</option>)}<option value="">all</option></select>{' '}
        <select value={b} onChange={(e) => setBase(e.target.value)} aria-label="baseline route"><option value="">no baseline</option>{(data?.cols || []).map((c) => <option key={c} value={c}>Δ vs {c}</option>)}</select>
      </>}>
      {data && (
        <>
          <div className="wall" style={{ gridTemplateColumns: `minmax(90px, 1.4fr) repeat(${data.cols.length}, minmax(34px, 1fr))` }}>
            <div className="h">task · body \ route</div>
            {data.cols.map((c) => <div key={c} className="h" title={c}>{c === '?' ? '(none)' : c}</div>)}
            {data.rowKeys.map((rk) => (
              <Row2 key={rk} rk={rk} cols={data.cols} cells={data.cells} base={b} fam={fam} />
            ))}
          </div>
          <p className="board-note">
            % success, Σk/Σn pooled per cell over its rows (seeds, variants, versions, conditions): a coarse map; drill in for the rows. * body from a path hint.
            {b ? ' Colour = Δ vs the baseline route (blue ▲ better, red ▼ worse).' : ' Colour = rate.'} Striped = includes interim; ⚠ = includes caveats.
            {data.total > data.rowKeys.length ? ` Showing ${data.rowKeys.length} of ${data.total} rows.` : ''}
          </p>
        </>
      )}
    </Panel>
  );
}
function Row2({ rk, cols, cells, base, fam }: { rk: string; cols: string[]; cells: Map<string, { k: number; n: number; rows: number; interim: boolean; caveat: boolean }>; base: string; fam: string }) {
  const parts = rk.split(' · ');
  const [task, body] = parts.length > 1 ? parts : ['·', parts[0]];
  const bc = base ? cells.get(`${rk}|${base}`) : undefined;
  return (
    <>
      <div className="h" title={rk}>{rk}</div>
      {cols.map((c) => {
        const x = cells.get(`${rk}|${c}`);
        if (!x) return <div key={c} className="c e" />;
        const rate = x.k / x.n;
        const [lo, hi] = wilson(x.k, x.n);
        const dlt = bc && c !== base ? rate - bc.k / bc.n : null;
        const bg = dlt !== null ? divColor(dlt) : seqColor(rate);
        const ink = dlt !== null ? (Math.abs(dlt) > 0.3 ? '#fff' : 'var(--ink)') : seqInk(rate);
        return (
          <div key={c} className="c" style={{ background: bg, color: ink, backgroundImage: x.interim ? 'repeating-linear-gradient(135deg, rgba(255,255,255,.22) 0 2px, transparent 2px 5px)' : undefined }}
            title={`${rk} × ${c}\n${x.k}/${x.n} = ${(rate * 100).toFixed(1)}% [${(lo * 100).toFixed(0)}, ${(hi * 100).toFixed(0)}] Wilson\n${x.rows} rows pooled${x.interim ? ' · includes interim' : ''}${x.caveat ? ' · includes caveats' : ''}${bc ? `\nbaseline ${base}: ${bc.k}/${bc.n}` : ''}`}
            onClick={() => { window.location.hash = href('results', { f_family: fam || undefined, f_route: c, f_body_or_hint: body.endsWith('*') ? `${body.slice(0, -1)} (hint)` : body, f_task: task === '·' ? undefined : task, agg: 'pool' }).slice(1); }}>
            <b>{dlt !== null ? <>{dlt > 0 ? '▲' : dlt < 0 ? '▼' : '■'}{Math.abs(dlt * 100).toFixed(0)}</> : (rate * 100).toFixed(0)}{x.caveat ? '⚠' : ''}</b>
          </div>
        );
      })}
    </>
  );
}

function LineagePanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const lin = d ? armLineages(rows(d.rows)) : [];
  return (
    <Panel title="Arm lineages · grasp_v2" href={href('results', { mode: 'rate', metric: 'grasp_v2', r1: 'body_or_hint', c: 'source_file' })} result={r} meta="compare_gc2_final · INTERIM (D-121)">
      {!lin.length ? <p className="board-note">no grasp_v2 lineage rows</p> : (
        <table className="bt">
          <thead><tr><th>lineage (as labelled)</th><th>success · 95% CI</th><th className="n">Σk/Σn</th><th className="n">%</th></tr></thead>
          <tbody>
            {lin.map((l) => {
              const [lo, hi] = wilson(l.k, l.n);
              return (
                <tr key={l.label}>
                  <td className="name" title={l.label}>{l.label}</td>
                  <td><RateBar rate={l.n ? l.k / l.n : null} lo={lo} hi={hi} width={90} /></td>
                  <td className="n">{l.k}/{l.n}</td>
                  <td className="n">{l.n ? ((l.k / l.n) * 100).toFixed(0) : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <p className="board-note">Summed over the {lin[0]?.bodies ?? 0} bodies of each lineage in the file; Wilson CI computed here.</p>
    </Panel>
  );
}

/* ------------------------------------------------------------------ edits mini forest */
const EDIT_ORDER = ['halt', 'ctx_halt', 'mirror_active', 'mirror_goal', 'probe_goal_mirror', 'z_turn', 'mirror_inactive', 'ctx_mirror_inactive', 'rand_norm_8'];
function EditsPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const groups = useMemo(() => {
    if (!d) return [];
    const rs = rows(d.rows).filter((x) => str(x.kind) !== 'permutation' && num(x.effect) !== null && EDIT_ORDER.includes(str(x.edit)));
    const out: { body: string; edit: string; metric: string; pts: { e: number; lo: number | null; hi: number | null; seed: string; variant: string }[]; control: boolean; variant: string }[] = [];
    for (const body of uniq(rs.map((x) => str(x.body))).filter(Boolean).sort(sortNatural)) {
      for (const edit of EDIT_ORDER) {
        const g0 = rs.filter((x) => str(x.body) === body && str(x.edit) === edit);
        if (!g0.length) continue;
        const metric = mode(g0.map((x) => str(x.metric)));
        for (const variant of uniq(g0.map((x) => str(x.variant))).sort(sortNatural)) {
          const g = g0.filter((x) => str(x.variant) === variant && str(x.metric) === metric);
          if (!g.length) continue;
          const pts = g.map((x) => ({ e: num(x.effect)!, lo: Array.isArray(x.ci) ? num(x.ci[0]) : null, hi: Array.isArray(x.ci) ? num(x.ci[1]) : null, seed: str(x.seed), variant }));
          out.push({ body, edit, metric, pts, control: g.some((x) => str(x.role) === 'control'), variant: variant || '—' });
        }
      }
    }
    return out;
  }, [d]);
  return (
    <Panel title="Causal edits · mini forest" href={href('edits')} result={r} span={2} rows2 meta={d ? `${rows(d.rows).length} effect rows` : ''}>
      <table className="bt">
        <thead><tr><th>body</th><th>edit</th><th>variant</th><th>metric</th><th>per seed (dot) with CI · ○ pooled · | 0</th><th className="n">med</th><th className="n">CI≠0</th></tr></thead>
        <tbody>
          {groups.map((g) => {
            const vals = g.pts.flatMap((p) => [p.e, p.lo ?? p.e, p.hi ?? p.e]);
            const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
            const w = 170, h = 14, x = (v: number) => 3 + ((v - lo) / (hi - lo || 1)) * (w - 6);
            const excl = g.pts.filter((p) => p.lo !== null && p.hi !== null && (p.lo > 0 || p.hi < 0)).length;
            return (
              <tr key={`${g.body}-${g.edit}-${g.variant}`} className="click" onClick={() => { window.location.hash = href('edits', { body: g.body, edit: g.edit, metric: g.metric }).slice(1); }}>
                <td>{g.body}</td>
                <td style={{ color: g.control ? 'var(--muted)' : undefined }}>{g.edit}{g.control ? ' (ctl)' : ''}</td>
                <td className={/nosem/.test(g.variant) ? 'muted' : ''}>{g.variant}</td>
                <td className="muted name" style={{ maxWidth: 90 }}>{g.metric}</td>
                <td>
                  <svg width={w} height={h} role="img" aria-label={`${g.pts.length} effects`}>
                    <line x1={x(0)} x2={x(0)} y1={0} y2={h} stroke="var(--ink-2)" />
                    {g.pts.map((p, i) => {
                      const y = h / 2;
                      const col = g.control ? 'var(--muted)' : p.lo !== null && p.hi !== null && (p.lo > 0 || p.hi < 0) ? (p.e > 0 ? 'var(--up)' : 'var(--down)') : 'var(--ink-2)';
                      return (
                        <g key={i}>
                          <title>{`${p.variant} seed ${p.seed || '?'}: ${fmtNum(p.e)} [${fmtNum(p.lo)}, ${fmtNum(p.hi)}]`}</title>
                          {p.lo !== null && p.hi !== null && <line x1={x(p.lo)} x2={x(p.hi)} y1={y} y2={y} stroke={col} strokeOpacity={0.5} />}
                          {p.seed ? <circle cx={x(p.e)} cy={y} r={2} fill={col} /> : <circle cx={x(p.e)} cy={y} r={3.2} fill="none" stroke={col} strokeWidth={1.2} />}
                        </g>
                      );
                    })}
                  </svg>
                </td>
                <td className="n"><Delta v={median(g.pts.map((p) => p.e))} /></td>
                <td className="n">{excl}/{g.pts.length}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="board-note">One row per body · edit · variant (the edit's most common metric). Axis per row, always including 0. Filled dot = one seed; ring = recorded pooled-over-seeds effect. Blue/red = CI excludes 0; grey = CI includes 0, or a control.</p>
    </Panel>
  );
}

/* ------------------------------------------------------------------ training sparklines */
function TrainingPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const recent = useMemo(() => {
    if (!d) return [];
    const withT = rows(d.runs).map((x) => ({ x, t: num(pick(x.last, 't')) })).filter((y) => y.t !== null && y.t > 1.5e9);
    return withT.sort((a, b) => b.t! - a.t!).slice(0, 10).map((y) => ({ ...y.x, __t: y.t }) as Row);
  }, [d]);
  const [series, setSeries] = useState<Record<string, Row>>({});
  useEffect(() => {
    let live = true;
    for (const x of recent) {
      const id = str(x.id);
      if (Array.isArray(x.step)) { setSeries((s) => ({ ...s, [id]: x })); continue; }
      fetchTrainingSeries(id).then((res) => { if (live && res.status === 'ok') setSeries((s) => ({ ...s, [id]: res.data as Row })); });
    }
    return () => { live = false; };
  }, [recent]);
  return (
    <Panel title="Training · most recent logs" href={href('training')} result={r} span={2} meta={d ? `${rows(d.runs).length} runs` : ''}>
      {!recent.length ? <p className="board-note">no run records a wall-clock time</p> : (
        <table className="bt">
          <thead><tr><th>run</th><th>kind</th><th>loss</th><th className="n">last</th><th>grad norm</th><th className="n">step</th><th className="n">logged</th></tr></thead>
          <tbody>
            {recent.map((x) => {
              const s = series[str(x.id)];
              const losses = s && isObj(s.losses) ? s.losses : {};
              const lk = ['loss', 'flow', 'total', 'q_loss', 'bc'].find((k) => Array.isArray(losses[k])) || Object.keys(losses)[0];
              const lv = lk ? arr(losses[lk]).map(num) : [];
              const gn = s ? arr(s.grad_norm).map(num) : [];
              return (
                <tr key={str(x.id)} className="click" onClick={() => { window.location.hash = href('training', { runs: str(x.id) }).slice(1); }}>
                  <td className="name" title={str(x.run)}>{str(x.run).replace(/^artifacts\/runs\//, '')}</td>
                  <td className="muted">{str(x.kind)}</td>
                  <td>{s ? <Spark values={lv} width={90} height={14} /> : <span className="muted">…</span>}</td>
                  <td className="n">{lk ? <span title={lk}>{fmtNum(lv[lv.length - 1])}</span> : '—'}</td>
                  <td>{gn.length ? <Spark values={gn} width={70} height={14} color="var(--s2)" /> : <span className="muted">—</span>}</td>
                  <td className="n">{fmtNum(typeof x.last_step === 'number' ? x.last_step : pick(x.last, 'step', 'update'))}</td>
                  <td className="n muted">{ago((x as Row).__t).replace(' ago', '')}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ robustness strip */
function RobustPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const reps = d ? rows(d.reports) : [];
  const factors = uniq(reps.flatMap((x) => (isObj(x.break_points) ? Object.keys(x.break_points) : [])));
  return (
    <Panel title="Robustness · break-points" href={href('robustness')} result={r} span={2} meta={d ? `${reps.length} route × robot reports` : ''}>
      {!reps.length ? <p className="board-note">no reports</p> : (
        <>
          <div className="strip" style={{ gridTemplateColumns: `minmax(110px, 1.5fr) 64px 64px repeat(${factors.length}, minmax(38px, 1fr))` }}>
            <div className="h" style={{ color: 'var(--muted)' }}>robot · route</div><div style={{ color: 'var(--muted)' }}>nominal</div><div style={{ color: 'var(--muted)' }}>perturbed</div>
            {factors.map((f) => <div key={f} className="bp" style={{ color: 'var(--muted)' }} title={f}>{f}</div>)}
            {reps.map((x, i) => {
              const nom = isObj(x.nominal) ? x.nominal : {}, pool = isObj(x.pooled_perturbed) ? x.pooled_perturbed : {};
              const bps = isObj(x.break_points) ? x.break_points : {};
              const robot = str(pick(x, 'robot', 'body'));
              return (
                <RowFrag key={i}>
                  <div title={str(x.source_file)} style={{ cursor: 'pointer' }} onClick={() => { window.location.hash = href('robustness', { robot }).slice(1); }}>{robot} · <b>{str(x.route)}</b></div>
                  <div><RateBar rate={num(nom.rate)} lo={Array.isArray(nom.ci) ? num(nom.ci[0]) : null} hi={Array.isArray(nom.ci) ? num(nom.ci[1]) : null} width={56} /></div>
                  <div><RateBar rate={num(pool.rate)} lo={Array.isArray(pool.wilson95) ? num(pool.wilson95[0]) : null} hi={Array.isArray(pool.wilson95) ? num(pool.wilson95[1]) : null} width={56} /></div>
                  {factors.map((f) => {
                    const b = isObj(bps[f]) ? bps[f] : null;
                    const lo = b ? num(b.low) : null, hi = b ? num(b.high) : null;
                    const txt = [lo !== null ? `≤${fmtNum(lo)}` : '', hi !== null ? `≥${fmtNum(hi)}` : ''].filter(Boolean).join(' ');
                    return <div key={f} className="bp" style={{ background: txt ? 'color-mix(in srgb, var(--critical) 22%, var(--surface))' : undefined }} title={b ? `${f}: breaks ${txt || 'nowhere within the swept range'}` : `${f}: not swept`}>{b ? txt || '·' : ''}</div>;
                  })}
                </RowFrag>
              );
            })}
          </div>
          <p className="board-note">Cell = recorded break level (≤low / ≥high); · = no break within the swept range; blank = not swept. Bars: success with 95% CI.</p>
        </>
      )}
    </Panel>
  );
}
function RowFrag({ children }: { children: ReactNode }) { return <>{children}</>; }

/* ------------------------------------------------------------------ Ψ₀, gates, decisions, caveats, media */
function Psi0Panel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const runs = d ? rows(d.runs).filter((x) => num(x.n)) : [];
  runs.sort((a, b) => Number(str(b.run).startsWith('step2')) - Number(str(a.run).startsWith('step2')));
  return (
    <Panel title="Ψ₀ line" href={href('psi0')} result={r} meta={d ? `${rows(d.p_decisions).length} P-decisions` : ''}>
      <table className="bt">
        <tbody>
          {runs.slice(0, 14).map((x) => {
            const k = num(x.k)!, n = num(x.n)!;
            const ci = Array.isArray(x.ci) ? [num(x.ci[0]), num(x.ci[1])] : wilson(k, n);
            return (
              <tr key={str(x.run)} title={`${str(x.task)} · ${str(x.interim_reason)}`}>
                <td className="name" style={{ maxWidth: 130 }}>{str(x.run).replace(/^psi0rel_/, 'rel ').replace(/^step2_/, 's2 ')}</td>
                <td><RateBar rate={k / n} lo={ci[0]} hi={ci[1]} width={60} /></td>
                <td className="n">{k}/{n}</td>
                <td>{x.interim ? <span className="warn-glyph" title={str(x.interim_reason) || 'interim'}>◐</span> : null}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}
function GatesPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const gates = d ? rows(d.gates) : [];
  const tr = d ? rows(d.trackers) : [];
  const bodies = uniq(tr.map((t) => str(t.body))).sort(sortNatural);
  return (
    <Panel title="Physics gates" href={href('physics')} result={r} meta={d ? `${gates.length} gates · ${tr.length} trackers` : ''}>
      <table className="bt">
        <tbody>
          {gates.map((g, i) => (
            <tr key={i} title={rows(g.criteria).map((c) => `${str(c.name)}: ${str(c.status)} (${fmtNum(c.value)} ${str(c.threshold)})`).join('\n')}>
              <td><span className={`status ${stateTone(g.verdict)}`}><i />{str(g.verdict)}</span></td>
              <td className="name">{str(g.gate)}</td>
              <td className="n">{rows(g.criteria).filter((c) => c.status === 'pass').length}/{rows(g.criteria).length}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginTop: 4 }}>
        {bodies.map((b) => {
          const ts = tr.filter((t) => str(t.body) === b);
          const pass = ts.filter((t) => t.passed === true).length, cp = ts.filter((t) => isObj(t.contact_gate) && t.contact_gate.passed === true).length;
          return <span key={b} className="mbadge" title={`${ts.length} tracker versions · tracking gate pass ${pass} · contact gate pass ${cp}`}>{b} {pass}/{ts.length}{cp ? ` c${cp}` : ''}</span>;
        })}
      </div>
    </Panel>
  );
}
function DecisionsPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const ds = d ? rows(d.latest_decisions) : [];
  return (
    <Panel title="Latest decisions" href={href('knowledge', { tab: 'decisions' })} result={r} meta={d ? `STATUS ${str(d.status_updated)}` : ''}>
      <table className="bt">
        <tbody>
          {ds.map((x) => (
            <tr key={str(x.id)} className="click" onClick={() => { window.location.hash = href('knowledge', { tab: 'decisions', d: str(x.id) }).slice(1); }}>
              <td className="n muted">{str(x.date).slice(5)}</td>
              <td className="mono">{str(x.id)}</td>
              <td className="name" style={{ maxWidth: 240 }} title={str(x.title)}>{str(x.title)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
function CaveatsPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const cs = d ? arr(d.caveats) : [];
  const claims = d ? rows(d.claims) : [];
  return (
    <Panel title="Claims & caveats" href={href('overview')} result={r} meta={d ? `${claims.length} claims · ${cs.length} caveats` : ''}>
      <table className="bt">
        <tbody>
          {claims.map((c, i) => (
            <tr key={`c${i}`} title={`${str(c.title)}: ${str(c.text)}`}>
              <td><span className={`status ${stateTone(c.status === 'established' ? 'ok' : c.status)}`}><i />{str(c.status).slice(0, 5)}</span></td>
              <td className="name" style={{ maxWidth: 250 }}>{str(c.title)}</td>
              <td>{c.interim ? <Warn text="interim" /> : null}{c.caveat ? <Warn text={c.caveat} /> : null}</td>
            </tr>
          ))}
          {cs.map((c, i) => (
            <tr key={`v${i}`} title={str(pick(c, 'text') ?? c)}>
              <td><Warn text={pick(c, 'text') ?? c} /></td>
              <td className="name" style={{ maxWidth: 250, color: 'var(--ink-2)' }} colSpan={2}>{str(pick(c, 'text') ?? c)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
function MediaPanel({ replays, videos }: { replays: DocResult<Envelope>; videos: DocResult<Envelope> }) {
  const rp = ok(replays), vd = ok(videos);
  const rs = rp ? rows(rp.replays) : [];
  const vs = vd ? rows(vd.videos) : [];
  const bySrc = new Map<string, number>();
  vs.forEach((v) => { const s = str(v.source) || '?'; bySrc.set(s, (bySrc.get(s) || 0) + 1); });
  return (
    <Panel title="Theatre" href={href('theatre')} meta={<><ModeBadge result={replays} /> <ModeBadge result={videos} /></>}>
      <div className="row" style={{ gap: 12, marginBottom: 3 }}>
        <span><b className="num" style={{ fontSize: 16 }}>{rs.length}</b> <span className="muted">replays</span></span>
        <span><b className="num" style={{ fontSize: 16 }}>{vs.length}</b> <span className="muted">videos</span></span>
      </div>
      {rs.length ? (
        <table className="bt"><tbody>
          {rs.slice(0, 8).map((x) => (
            <tr key={str(x.id)} className="click" onClick={() => { window.location.hash = href('theatre', { a: str(x.id) }).slice(1); }}>
              <td className="name">{[x.body, x.route, x.condition].map(str).filter(Boolean).join(' · ')}</td>
              <td><span className={`status ${x.success ? 'good' : x.success === false ? 'critical' : 'neutral'}`}><i /></span></td>
            </tr>
          ))}
        </tbody></table>
      ) : <p className="board-note">no replays recorded yet{rp && arr(rp.missing).length ? ` · ${arr(rp.missing).map(str).join('; ')}` : ''}</p>}
      <table className="bt"><tbody>
        {[...bySrc.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([s, n]) => <tr key={s}><td className="name">{s}</td><td className="n">{n}</td></tr>)}
      </tbody></table>
    </Panel>
  );
}
