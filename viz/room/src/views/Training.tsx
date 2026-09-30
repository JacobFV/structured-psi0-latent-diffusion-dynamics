import OpsCharts from './OpsCharts';
import { SidebarControls } from '../components/shell';
import { useEffect, useMemo, useState } from 'react';
import { Lines } from '../components/charts';
import { Card, Caveat, DataTable, Did, ErrorState, Gate, Loading, ModeBanner, NoData, Provenance, SourceBadge, Status } from '../components/ui';
import { fetchTrainingSeries, useDoc, type DocResult, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, num, pick, rows, sortNatural, str, uniq, type Row } from '../lib/format';
import { SERIES } from '../lib/labels';
import { useUrlState } from '../lib/url';

type Run = { name: string; kind: string; step: number[]; series: Record<string, (number | null)[]>; raw: Row };

function toRun(r: Row): Run {
  const step = arr(pick(r, 'step', 'steps')).map((x) => num(x) ?? NaN);
  const series: Record<string, (number | null)[]> = {};
  const add = (k: string, v: unknown) => { if (Array.isArray(v) && v.length === step.length) series[k] = v.map((x) => num(x)); };
  const losses = pick(r, 'losses', 'loss');
  if (isObj(losses)) for (const [k, v] of Object.entries(losses)) add(`loss:${k}`, v);
  else add('loss:loss', losses);
  for (const k of ['grad_norm', 'clip_scale', 'lr', 'alpha']) add(k, r[k]);
  const probes = pick(r, 'probes');
  if (isObj(probes)) for (const [k, v] of Object.entries(probes)) add(`probe:${k}`, v);
  const extra = pick(r, 'metrics', 'series');
  if (isObj(extra)) for (const [k, v] of Object.entries(extra)) add(`metric:${k}`, v);
  return { name: str(pick(r, 'id', 'run', 'name')), kind: str(pick(r, 'kind', 'type')), step, series, raw: r };
}

export default function Training() {
  const { result, reload, busy } = useDoc<Envelope>('training');
  return (
    <>
      <OpsCharts />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="training (/api/training)">
        {(d) => (
          <>
            <TrainingBody index={rows(pick(d, 'runs', 'series', 'curves'))} />
            <GradHealth rs={rows(pick(d, 'grad_health'))} />
          </>
        )}
      </Gate>
      <Provenance result={result} />
    </>
  );
}

/** Loads the selected runs' series (index rows may already embed them, as fixtures do). */
function useSeries(index: Row[], ids: string[]) {
  const [state, setState] = useState<Record<string, DocResult<Record<string, unknown>>>>({});
  useEffect(() => {
    let live = true;
    for (const id of ids) {
      const row = index.find((r) => str(r.id) === id || str(r.run) === id);
      if (!row) continue;
      if (Array.isArray(row.step)) { setState((s) => ({ ...s, [id]: { status: 'ok', data: row, mode: 'fixture', fetchedAt: Date.now() } })); continue; }
      setState((s) => (s[id] ? s : { ...s, [id]: { status: 'loading' } }));
      fetchTrainingSeries(str(row.id)).then((r) => live && setState((s) => ({ ...s, [id]: r })));
    }
    return () => { live = false; };
  }, [index, ids.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps
  return state;
}

function TrainingBody({ index }: { index: Row[] }) {
  const idOf = (r: Row) => str(pick(r, 'id', 'run', 'name'));
  const [kind, setKind] = useUrlState('kind', '');
  const [sel, setSel] = useUrlState('runs', '');
  const [log, setLog] = useUrlState('log', '1');
  const [gn, setGn] = useUrlState('gn', '0');
  const [q, setQ] = useState('');
  const kinds = uniq(index.map((r) => str(r.kind))).filter(Boolean).sort();
  const firstWithGrad = index.find((r) => r.grad_norm_key) || index[0];
  const selected = (sel || (firstWithGrad ? idOf(firstWithGrad) : '')).split(',').filter(Boolean);
  const series = useSeries(index, selected);
  const list = index.filter((r) => (!kind || str(r.kind) === kind) && (gn !== '1' || r.grad_norm_key || Array.isArray(r.grad_norm)) && (!q || JSON.stringify([r.run, r.id, r.body]).toLowerCase().includes(q.toLowerCase())));
  const toggle = (name: string, multi: boolean) => {
    const next = multi ? (selected.includes(name) ? selected.filter((x) => x !== name) : [...selected, name].slice(-4)) : [name];
    setSel(next.join(','));
  };
  if (!index.length) return <p className="muted">The training document has no runs.</p>;
  const chosen: Run[] = [];
  const pending: string[] = [];
  const problems: DocResult<Record<string, unknown>>[] = [];
  for (const id of selected) {
    const r = series[id];
    const row = index.find((x) => idOf(x) === id);
    if (!r || r.status === 'loading') pending.push(id);
    else if (r.status === 'ok') chosen.push(toRun({ ...(row || {}), ...r.data, id }));
    else problems.push(r);
  }
  return (
    <div>
      <SidebarControls>
      <section className="card">
        <header><h2>Runs</h2><span className="hint">{list.length} of {index.length} · shift-click compares up to 4</span></header>
        <div className="body">
          <div className="filters" style={{ marginBottom: 8 }}>
            <label>kind<select value={kind} onChange={(e) => setKind(e.target.value)}><option value="">all ({kinds.length})</option>{kinds.map((k) => <option key={k}>{k}</option>)}</select></label>
            <label>search<input type="search" value={q} onChange={(e) => setQ(e.target.value)} /></label>
            <label className="small" style={{ display: 'flex', gap: 4, alignItems: 'center' }}><input type="checkbox" checked={gn === '1'} onChange={(e) => setGn(e.target.checked ? '1' : '0')} /> has grad norm</label>
          </div>
          <div className="side-list" style={{ maxHeight: 640, overflow: 'auto' }}>
            {list.slice().sort((a, b) => sortNatural(str(a.run), str(b.run))).map((r) => (
              <button key={idOf(r)} className="side-line" aria-pressed={selected.includes(idOf(r))} onClick={(e) => toggle(idOf(r), e.shiftKey || e.metaKey || e.ctrlKey)}
                title={`${str(r.run || r.id)}\n${str(r.kind) || 'kind ?'} · ${fmtNum(pick(r, 'n_points'))} points${r.stride && r.stride !== 1 ? ` (stride ${str(r.stride)})` : ''} · last step ${fmtNum(typeof r.last_step === 'number' ? r.last_step : pick(r.last, 'step', 'update'))}${r.grad_norm_key ? ' · grad norm' : ''}${r.clip_scale_key ? ' · clip scale' : ''}${r.has_alpha ? ' · α' : ''}${r.location && r.location !== 'repo' ? ` · ${str(r.location)}` : ''}\n● logged in the last 24 h · ○ older · – no wall-clock time recorded`}>
                <span className="nm">{str(r.run || r.id).replace(/^artifacts\/runs\//, '')}</span>
                <span className="mt">{str(r.body) || str(r.kind) || '?'}</span>
                <span className="gl">{(() => { const t = num(pick(r.last, 't')); return t === null || t < 1.5e9 ? '–' : Date.now() / 1000 - t < 86400 ? '●' : '○'; })()}</span>
              </button>
            ))}
          </div>
        </div>
      </section>
      </SidebarControls>
      <div className="stack" style={{ minWidth: 0 }}>
        <div className="row">
          <label className="small"><input type="checkbox" checked={log === '1'} onChange={(e) => setLog(e.target.checked ? '1' : '0')} /> log scale for losses and grad norm (non-positive values are dropped, not clamped)</label>
        </div>
        {chosen.map((r, i) => <RunHeader key={r.name} r={r} color={SERIES[i]} />)}
        {pending.length > 0 && <Loading what={`series for ${pending.join(', ')}`} />}
        {problems.map((p, i) => p.status === 'missing' ? <NoData key={i} expected={p.expected} /> : p.status === 'error' ? <ErrorState key={i} message={p.message} /> : null)}
        {chosen.length > 0 && <RunCharts runs={chosen} log={log === '1'} />}
        {!selected.length && <p className="muted">Select a run.</p>}
      </div>
    </div>
  );
}

function GradHealth({ rs }: { rs: Row[] }) {
  if (!rs.length) return null;
  const runs = rs.flatMap((g) => Object.entries(isObj(g.doc) ? g.doc : {}).filter(([, v]) => isObj(v)).map(([name, v]) => ({ name, v: v as Row, src: str(g.source_file), decision: str(g.decision) })));
  const gmax = Math.max(1e-9, ...runs.flatMap((r) => arr(r.v.by_third).filter(isObj).map((t) => num(t.median_grad_norm) || 0)));
  return (
    <section className="ev-panel" style={{ border: 'var(--seam)', marginTop: 8 }}>
      <header>Gradient health (D-085)<span className="meta">median grad norm per training third (bars) · mean update scale per third (dots, 0–1)</span></header>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 6 }}>
        {runs.map((r) => {
          const th = arr(r.v.by_third).filter(isObj);
          return (
            <div key={r.name} title={`${r.src}${r.decision ? ` · ${r.decision}` : ''}`} style={{ display: 'grid', gridTemplateColumns: '110px 1fr', gap: 6, alignItems: 'end', fontSize: 10 }}>
              <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.name}</span>
              <svg width={96} height={28} role="img" aria-label={`${r.name} grad health`}>
                {th.map((t, i) => {
                  const g = num(t.median_grad_norm) || 0, u = num(t.mean_update_scale);
                  const h = (g / gmax) * 24;
                  return (
                    <g key={i}>
                      <title>{`steps ${str(t.steps)}: median grad norm ${fmtNum(g)} · mean update scale ${fmtNum(u)}`}</title>
                      <rect x={i * 32} y={27 - h} width={20} height={h} fill="var(--s1)" />
                      {u !== null && <circle cx={i * 32 + 26} cy={27 - u * 24} r={2.5} fill="var(--s2)" />}
                    </g>
                  );
                })}
              </svg>
            </div>
          );
        })}
      </div>
      <p className="fig-cap">one mini chart per run · bars: median gradient norm in each training third (scaled to the largest across runs) · dots: mean update (clip) scale in that third, 0–1</p>
    </section>
  );
}

function RunHeader({ r, color }: { r: Run; color: string }) {
  const gate = pick(r.raw, 'gate_state', 'gate');
  const refs = arr(pick(r.raw, 'decision_refs', 'decisions')).map(str);
  const dec = str(pick(r.raw, 'decision'));
  return (
    <div className="label-banner" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="row">
        <b style={{ wordBreak: 'break-all' }}>{r.name}</b>
        <span className="badge">{r.kind || 'kind ?'}</span>
        {pick(r.raw, 'source_label') ? <SourceBadge label={pick(r.raw, 'source_label')} /> : null}
        {gate != null && <Status state={isObj(gate) ? pick(gate, 'state', 'status') : gate}>gate: {isObj(gate) ? str(pick(gate, 'state', 'status')) : str(gate)}</Status>}
        {[dec, ...refs].filter(Boolean).map((d) => <Did key={d} id={d} />)}
        {pick(r.raw, 'interim') ? <span className="badge interim">interim</span> : null}
        <Caveat text={pick(r.raw, 'caveat')} />
      </div>
      <div className="kv">
        {['run', 'source_file', 'location', 'step_key', 'n_records', 'n_points', 'stride', 'ckpt', 'ckpt_sha', 'body', 'variant', 'seed'].filter((k) => r.raw[k] != null).map((k) => <span key={k}>{k} <b className="mono">{str(r.raw[k])}</b></span>)}
      </div>
    </div>
  );
}

function RunCharts({ runs, log }: { runs: Run[]; log: boolean }) {
  const keys = uniq(runs.flatMap((r) => Object.keys(r.series)));
  const lossKeys = keys.filter((k) => k.startsWith('loss:'));
  const probeKeys = keys.filter((k) => k.startsWith('probe:') || k.startsWith('metric:'));
  const multi = runs.length > 1;
  // merge runs on step for shared charts
  const merged = useMemo(() => {
    const bySstep = new Map<number, Row>();
    runs.forEach((r, ri) => r.step.forEach((s, i) => {
      if (!Number.isFinite(s)) return;
      const row = bySstep.get(s) || { step: s };
      for (const [k, v] of Object.entries(r.series)) row[`${ri}|${k}`] = v[i];
      bySstep.set(s, row);
    }));
    return Array.from(bySstep.values()).sort((a, b) => (a.step as number) - (b.step as number));
  }, [runs]);
  const series = (k: string) => runs.map((r, ri) => ({ key: `${ri}|${k}`, label: multi ? r.name : k.replace(/^\w+:/, ''), color: SERIES[ri % SERIES.length] })).filter((_, ri) => runs[ri].series[k]);
  // clip-active shading (clip_scale < 1) for the first run
  const shades = useMemo(() => {
    const r = runs[0];
    const cs = r.series.clip_scale;
    if (!cs) return [];
    const out: { x1: number; x2: number }[] = [];
    let s: number | null = null;
    cs.forEach((v, i) => {
      const on = v !== null && v < 0.999;
      if (on && s === null) s = r.step[i];
      if (!on && s !== null) { out.push({ x1: s, x2: r.step[i] }); s = null; }
    });
    if (s !== null) out.push({ x1: s, x2: r.step[r.step.length - 1] });
    return out.slice(0, 400);
  }, [runs]);
  const dagger = rows(pick(runs[0].raw, 'dagger_rounds', 'rounds')).map((d) => ({ x: num(pick(d, 'step', 'start_step')) ?? NaN, label: `DAgger ${str(pick(d, 'round', 'id'))}` })).filter((d) => Number.isFinite(d.x));
  const gates = rows(pick(runs[0].raw, 'gate_events', 'events')).map((d) => ({ x: num(pick(d, 'step')) ?? NaN, label: str(pick(d, 'event', 'state', 'label')) })).filter((d) => Number.isFinite(d.x));
  const refs = [...dagger, ...gates].map((r) => ({ ...r, color: 'var(--s7)' }));
  return (
    <>
      <Card title="Logged series" hint={multi ? 'losses and counters as logged · one colour per run' : 'losses and counters as logged · one panel each'}>
        {lossKeys.length ? (
          <div className="grid g2">
            {lossKeys.map((k) => (
              <Lines key={k} title={<b>{k.slice(5)}</b>} data={merged} xKey="step" series={series(k)} logY={log} height={200} syncId="train" refs={refs} xLabel="step" />
            ))}
          </div>
        ) : <p className="muted small">No loss series recorded for the selected run(s).</p>}
        <p className="fig-cap">one chart per logged series · x: training step · y: the logged value as recorded (log scale when chosen and all values are positive) · colour: run · violet dashed verticals: DAgger rounds / gate events</p>
      </Card>
      <Card title="Gradient health" hint="grad norm (clip-active steps shaded, clip_scale < 1) and clip scale on its own axis — never a dual axis">
        <div className="grid g2">
          {keys.includes('grad_norm') ? <Lines title={<b>grad norm</b>} right={shades.length ? <span className="small muted">{shades.length} clip-active interval(s) shaded{multi ? ' (first run)' : ''}</span> : undefined}
            data={merged} xKey="step" series={series('grad_norm')} logY={log} height={200} syncId="train" shades={shades} xLabel="step" /> : <p className="muted small">grad_norm not recorded.</p>}
          <p className="fig-cap" style={{ gridColumn: '1 / -1', order: 9 }}>x: training step · left: gradient norm (orange bands: steps where clip scale &lt; 1) · right: clip scale (1 = no clipping) · colour: run</p>
          {keys.includes('clip_scale') ? <Lines title={<b>clip scale</b>} data={merged} xKey="step" series={series('clip_scale')} height={200} syncId="train" yDomain={[0, 'auto']} refs={[{ y: 1, label: 'no clip' }]} xLabel="step" /> : <p className="muted small">clip_scale not recorded.</p>}
        </div>
      </Card>
      <Card title="Schedules">
        <div className="grid g2">
          {keys.includes('lr') ? <Lines title={<b>learning rate</b>} data={merged} xKey="step" series={series('lr')} height={180} syncId="train" xLabel="step" /> : <p className="muted small">lr not recorded.</p>}
          {keys.includes('alpha') ? <Lines title={<b>reward-schedule α</b>} data={merged} xKey="step" series={series('alpha')} height={180} syncId="train" xLabel="step" /> : <p className="muted small">α not recorded (only reward-scheduled runs have it).</p>}
        </div>
      </Card>
      {probeKeys.length > 0 && (
        <Card title="Probes and metrics" hint="diagnostics recorded during training">
          <div className="grid g2">
            {probeKeys.map((k) => <Lines key={k} title={<b>{k}</b>} data={merged} xKey="step" series={series(k)} height={180} syncId="train" xLabel="step" />)}
          </div>
        </Card>
      )}
      {(dagger.length > 0 || gates.length > 0) && (
        <Card title="Rounds and gate events"><DataTable rows={[...dagger, ...gates].map((r) => ({ step: r.x, event: r.label }))} /></Card>
      )}
    </>
  );
}
