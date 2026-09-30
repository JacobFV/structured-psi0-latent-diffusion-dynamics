/**
 * Board (focused on NOW, D-140 one repo + D-144 relation factors): one screen at 1440×900.
 *   headline strip: D-134…D-141 compressed to one row + the D-140 pause status
 *   policy × env × task matrix (recorded `rrp matrix`), relation-factor registry + catalog waves, factor sets in runs,
 *   curriculum scheduler (schedule.jsonl; honest empty state until R11 writes one), run DAGs, decisions, leases.
 * Older result panels live in Overview → Results board. Every figure is captioned; the render check asserts the budget.
 */
import type { ReactNode } from 'react';
import { Cap, ModeBadge } from '../components/board';
import { useDoc, type DocResult, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, newcombe, num, rows, str, type Row } from '../lib/format';
import { href } from '../lib/url';

export const VIEWPORT_H = 900;
export const TICKER_H = 20;
export const PAD = 8;
export const KPI_H = 50;
export const ROWS = 3;
export const CELL_H = Math.floor((VIEWPORT_H - TICKER_H - 2 * PAD - KPI_H) / ROWS);
export const HEAD_H = 18;
export const CAP_H = 40;
const LINE = 13;

const ok = (r: DocResult<Envelope>) => (r.status === 'ok' ? r.data : null);
const GB = 1024 ** 3;

/* ---------- shared helpers (arm v6 D-134, transfer D-135/136, legged halt D-124) */
function lineages(res: Envelope | null) {
  const by = new Map<string, { k: number; n: number }>();
  if (res) for (const r of rows(res.rows)) {
    const kp = arr(r.key_path).map(str);
    if (str(r.metric) !== 'grasp_v2' || kp.length !== 3 || !str(r.source_file).endsWith('compare_gc2_final.json')) continue;
    const x = by.get(kp[0]) || { k: 0, n: 0 };
    x.k += num(r.k) || 0; x.n += num(r.n) || 0;
    by.set(kp[0], x);
  }
  const sum = (re: RegExp) => [...by.entries()].filter(([l]) => re.test(l)).reduce((a, [, v]) => ({ k: a.k + v.k, n: a.n + v.n }), { k: 0, n: 0 });
  return { semfix: sum(/^semfix/i), nosem: sum(/^nosem/i) };
}
function v6(res: Envelope | null) {
  const cells = new Map<string, { k: number; n: number }>();
  const bodies: string[] = [];
  const tot = { semfix: { k: 0, n: 0 }, nosem: { k: 0, n: 0 } };
  if (res) for (const r of rows(res.rows)) {
    const kp = arr(r.key_path).map(str);
    if (kp.length !== 3 || kp[0] !== 'v6' || !/compare_v6\.json$/.test(str(r.source_file))) continue;
    const lin = /^semfix_s/.test(kp[1]) ? 'semfix' : /^nosem_s/.test(kp[1]) ? 'nosem' : null;
    if (!lin) continue;
    const k = num(r.k) || 0, n = num(r.n) || 0;
    if (kp[2] === 'all') { tot[lin].k += k; tot[lin].n += n; continue; }
    if (!bodies.includes(kp[2])) bodies.push(kp[2]);
    const x = cells.get(`${kp[2]}|${lin}`) || { k: 0, n: 0 };
    x.k += k; x.n += n;
    cells.set(`${kp[2]}|${lin}`, x);
  }
  return { cells, bodies, tot };
}
function transfer(res: Envelope | null) {
  const z = () => ({ k: 0, n: 0 });
  const t = { joint: z(), bcSft: z(), gripLatent: z(), gripBc: z() };
  if (res) for (const r of rows(res.rows)) {
    const f = str(r.source_file), kp = arr(r.key_path).map(str), k = num(r.k) || 0, n = num(r.n) || 0;
    if (/armdiag\/d136_compare\.json$/.test(f) && kp.length === 3 && kp[0] === 'pooled' && /^semfix\/xarm7_(pg2|tf3)\/b\d+$/.test(kp[1])) {
      if (kp[2] === 'joint_adapt') { t.joint.k += k; t.joint.n += n; }
      if (kp[2] === 'bc_sft') { t.bcSft.k += k; t.bcSft.n += n; }
    }
    if (/ladder\/armv6\/targets_v6\.json$/.test(f) && kp.length === 4 && kp[2] === 'panda_tf3' && kp[3] === 'zero_shot') {
      if (kp[0] === 'latent' && kp[1].startsWith('semfix_s')) { t.gripLatent.k += k; t.gripLatent.n += n; }
      if (kp[0] === 'bc') { t.gripBc.k += k; t.gripBc.n += n; }
    }
  }
  return t;
}
function haltDiffs(ed: Envelope | null) {
  if (!ed) return [];
  const rs = rows(ed.rows);
  return ['anymal_c', 'go2'].flatMap((b) => {
    const d = rs.find((r) => str(r.body) === b && /summary_contact_v2/.test(str(r.source_file)) && /pooled_diff/.test(str(r.metric)));
    return d ? [{ body: b, v: num(d.effect)!, lo: num(arr(d.ci)[0])!, hi: num(arr(d.ci)[1])!, row: d }] : [];
  });
}
function P({ title, link, result, meta, children, cls = '' }: { title: string; link: string; result?: DocResult<Envelope>; meta?: ReactNode; children: ReactNode; cls?: string }) {
  return (
    <section className={`bp ${cls}`}>
      <header><a href={link}>{title}</a>{meta !== undefined && <span className="meta">{meta}</span>}{result && <ModeBadge result={result} />}</header>
      <div className="bb">{children}</div>
    </section>
  );
}

const activeDags = (g: Envelope | null) => (g ? rows(g.dags).filter((d) => d.complete === false) : []);
function LeasePanel({ r }: { r: DocResult<Envelope> }) {
  const l = ok(r);
  const ls = l ? rows(l.leases) : [];
  return (
    <P title="Leases · memory" link={href('training')} result={r} meta={`${ls.length}`}>
      {ls.map((x) => {
        const dec = isObj(x.declared) ? x.declared : {}, m = isObj(x.measured) ? x.measured : {};
        const dm = num(dec.memory_bytes) || 1, c = num(m.memory_current) || 0, pk = num(m.memory_peak), hi = num(m.memory_high), thr = num(m.memory_high_events) || 0;
        const f = (v: number) => `${Math.min(100, (v / dm) * 100)}%`;
        return (
          <div key={str(x.id)} className="barrow" title={`${str(x.label)} · ${str(x.workstream)} ${str(x.dag_node)} · ${(c / GB).toFixed(1)}/${(dm / GB).toFixed(0)} GB · peak ${fmtNum(pk !== null ? pk / GB : null)} GB · memory.high events ${thr}`}>
            <span>{str(x.label)}</span>
            <span className="bar"><i style={{ width: f(c), background: thr ? 'var(--critical)' : 'var(--accent)' }} />{hi !== null && <u style={{ left: f(hi) }} />}{pk !== null && <em style={{ left: f(pk) }} />}</span>
            <b>{(c / GB).toFixed(1)}/{(dm / GB).toFixed(0)}G</b>
          </div>
        );
      })}
      <Cap>bar: current / declared memory · black tick: peak · orange: memory.high · red: throttled</Cap>
    </P>
  );
}
function DagPanel({ r }: { r: DocResult<Envelope> }) {
  const g = ok(r);
  const all = g ? rows(g.dags) : [];
  const shown = [...activeDags(g), ...all.filter((d) => d.complete === true)].slice(0, 10);
  return (
    <P title="Run DAGs" link={href('training')} result={r} meta={`${activeDags(g).length} active`}>
      {shown.map((d) => {
        const c = isObj(d.counts) ? d.counts : {};
        const total = Object.values(c).reduce<number>((a, v) => a + (num(v) || 0), 0) || 1;
        const eta = arr(d.eta_statements).map((e) => str(isObj(e) ? e.text ?? e.eta : e)).join('; ');
        return (
          <div key={`${str(d.dag)}${str(d.location)}`} className="barrow" title={`${str(d.track)} · ${Object.entries(c).map(([k, v]) => `${k} ${v}`).join(', ')}${eta ? ` · ETA ${eta}` : ' · no ETA recorded'}`}>
            <span>{str(d.dag)}</span>
            <span className="bar">{['completed', 'running', 'failed'].map((k) => <i key={k} style={{ position: 'relative', display: 'inline-block', width: `${((num(c[k]) || 0) / total) * 100}%`, background: k === 'completed' ? 'var(--good)' : k === 'running' ? 'var(--accent)' : 'var(--critical)' }} />)}</span>
            <b>{fmtNum(c.completed ?? 0)}/{total}{eta ? ` ${eta}` : ''}</b>
          </div>
        );
      })}
      <Cap>bar: nodes completed (green), running (blue), failed (red) · done/total · ETA only if stated</Cap>
    </P>
  );
}
function DecisionPanel({ r, wide }: { r: DocResult<Envelope>; wide?: boolean }) {
  const o = ok(r);
  const ds = o ? rows(o.latest_decisions) : [];
  return (
    <P title="Latest decisions" link={href('board')} result={r} cls={wide ? 'span4' : 'span2'}>
      {ds.slice(0, 15).map((x) => (
        <div key={str(x.id)} className="barrow dline" title={`${str(x.date)} · ${str(x.title)} (research/decisions.md)`}>
          <span className="mono">{str(x.id)}</span><span>{str(x.title)}</span>
        </div>
      ))}
    </P>
  );
}

const FORM_COLOR: Record<string, string> = { bias: 'var(--s1)', aug: 'var(--s3)', gate: 'var(--s4)', mask: 'var(--s8)', message: 'var(--s7)', embed: 'var(--s5)', readout: 'var(--s2)' };
const SRC_TONE: Record<string, string> = { learned: 'var(--tone-learned)', bc: 'var(--tone-bc)', scripted_teacher: 'var(--tone-teacher)', oracle: 'var(--tone-oracle)', third_party: 'var(--ink-2)' };

function matrixShape(m: Envelope | null) {
  const rs = m ? rows(m.rows) : [];
  const pols = [...new Set(rs.map((r) => str(r.policy)))];
  const cols = [...new Set(rs.map((r) => `${str(r.env_id)}|${str(r.task)}`))];
  const envs = m && isObj(m.envs) ? Object.keys(m.envs) : [];
  const evaluated = new Set(rs.map((r) => str(r.env_id)));
  return { rs, pols, cols, envs, missing: envs.filter((e) => !evaluated.has(e)) };
}
function factorShape(f: Envelope | null) {
  const fs = f ? rows(f.factors) : [];
  const fams = [...new Set(fs.map((x) => str(x.family)))].sort();
  return { fs, fams, cat: f ? rows(f.catalog) : [], presets: f && isObj(f.presets) ? f.presets : {}, runs: f ? rows(f.runs) : [], sched: f ? rows(f.schedules) : [] };
}

/** Content heights (px); the render check asserts HEAD_H + need ≤ cells × CELL_H. */
export function panelNeeds(docs: Record<string, Envelope | null>) {
  const m = matrixShape(docs.matrix ?? null), f = factorShape(docs.factors ?? null);
  return {
    matrix: { need: 16 + m.pols.length * 17 + 14 + CAP_H + 14, cells: 2 },
    factors: { need: f.fams.length * 16 + 22 + 16 + CAP_H, cells: 1 },
    scheduler: { need: 150 + CAP_H, cells: 1 },
    dags: { need: 10 * LINE + 4 + CAP_H, cells: 1 },
    decisions: { need: 15 * LINE + 4, cells: 1 },  // spans 2 columns
    leases: { need: (docs.live ? rows(docs.live.leases).length : 0) * LINE + 4 + CAP_H, cells: 1 },
  };
}

export default function Board() {
  const d = {
    matrix: useDoc<Envelope>('matrix'), factors: useDoc<Envelope>('factors'), results: useDoc<Envelope>('results'), edits: useDoc<Envelope>('edits'),
    psi0: useDoc<Envelope>('psi0'), live: useDoc<Envelope>('live', 10_000), dags: useDoc<Envelope>('dags', 30_000), overview: useDoc<Envelope>('overview'),
  };
  const hasSchedule = factorShape(ok(d.factors.result)).sched.some((x) => rows(x.records).length);
  return (
    <div className="board5" style={{ ['--cell-h' as string]: `${CELL_H}px`, ['--kpi-h' as string]: `${KPI_H}px` }}>
      <Headlines d={d} />
      <div className="b5grid">
        <MatrixPanel r={d.matrix.result} />
        <FactorsPanel r={d.factors.result} />
        <DagPanel r={d.dags.result} />
        <LeasePanel r={d.live.result} />
        <DecisionPanel r={d.overview.result} wide={!hasSchedule} />
        {hasSchedule && <SchedulerPanel r={d.factors.result} />}
      </div>
    </div>
  );
}

/* ---------- headline strip: D-134…D-141 in one row + status */
function H({ k, v, s, tone, link, tip }: { k: string; v: ReactNode; s?: ReactNode; tone?: string; link: string; tip: string }) {
  return <a className={`k5 ${tone || ''}`} href={link} title={tip}><span>{k}</span><b>{v}</b><small>{s}</small></a>;
}
function Headlines({ d }: { d: Record<string, { result: DocResult<Envelope> }> }) {
  const res = ok(d.results.result);
  const V = v6(res).tot, L = lineages(res), T = transfer(res), halts = haltDiffs(ok(d.edits.result));
  const diff = (a: { k: number; n: number }, b: { k: number; n: number }) => {
    if (!a.n || !b.n) return null;
    const [lo, hi] = newcombe(b.k, b.n, a.k, a.n);
    return { v: Math.round((a.k / a.n - b.k / b.n) * 100), lo: Math.round(lo * 100), hi: Math.round(hi * 100) };
  };
  const arm = diff(V.semfix, V.nosem), na = diff(T.joint, T.bcSft), ng = diff(T.gripLatent, T.gripBc);
  const s2 = ok(d.psi0.result) ? rows(ok(d.psi0.result)!.runs).filter((r) => str(r.run).startsWith('step2') && !/structured/.test(str(r.run))) : [];
  const fmtD = (x: ReturnType<typeof diff>) => (x ? `${x.v > 0 ? '+' : ''}${x.v} [${x.lo}, ${x.hi}]` : '—');
  return (
    <div className="k5strip">
      <H k="arm v6 sf − ns · D-134" v={fmtD(arm)} s={`${V.semfix.k}/${V.semfix.n} vs ${V.nosem.k}/${V.nosem.n} (v1 ${L.semfix.k} vs ${L.nosem.k})`} tone="good" link={href('results')} tip="in-distribution arm bodies, grasp_v2.1, success difference in points with Newcombe 95% CI (compare_v6.json)" />
      <H k="new arm latent − BC · D-136" v={fmtD(na)} s={`${T.joint.k}/${T.joint.n} vs ${T.bcSft.k}/${T.bcSft.n} · NOT supported`} tone="bad" link={href('results', { tb: '100' })} tip="sealed xarm7 targets: joint adaptation (added after D-135) vs BC SFT at equal data and updates; worse in 12/12 cells" />
      <H k="new gripper zs · D-135" v={fmtD(ng)} s={`${T.gripLatent.k}/${T.gripLatent.n} vs BC ${T.gripBc.k}/${T.gripBc.n}`} tone="bad" link={href('results')} tip="panda_tf3 zero-shot, latent semfix v6 vs BC" />
      {halts.length ? <H k="legged halt Δ · D-124" v={halts.map((h) => h.v.toFixed(2)).join(' | ')} s={halts.map((h) => h.body).join(' | ') + ' (m)'} link={href('edits')} tip="semantic − nosem forward travel after a halt edit, contact v2" /> : null}
      <H k="Ψ₀ step 2 · D-141" v={s2.map((r) => `${str(r.k)}/${str(r.n)}`).join(' · ') || '—'} s="+ structured 0/20 (D-141): integration bug" link={href('board')} tip={`${s2.map((r) => `${str(r.run)} ${str(r.k)}/${str(r.n)}`).join(' · ')} (run summaries) · structured 0/20 from the D-141 write-up (the exported run summaries still hold an interim 0/6): a confirmed integration bug, not evidence against structure`} />
      {!rows(ok(d.factors.result)?.schedules).some((x) => rows(x.records).length) && <H k="curriculum (R11)" v="no runs" s="schedule.jsonl not written yet" link={href('factors')} tip="the curriculum scheduler (docs/relations.md 5.5) exports <run>/schedule.jsonl; no run has written one yet" />}
      <H k="status · D-140 / D-144" v="paused" s="one-repo refactor · relation factors" tone="warn" link={href('board')} tip="D-140: experiments wound down for the one-repo refactor (resume steps in track notes). D-144: relation-factor registry and scheduler under construction." />
    </div>
  );
}

/* ---------- policy × env × task matrix */
function MatrixPanel({ r }: { r: DocResult<Envelope> }) {
  const m = matrixShape(ok(r));
  if (!m.rs.length) return <P title="Policy × env × task" link={href('matrix')} result={r} cls="span2 row2"><p className="board-note">no recorded `rrp matrix --out` output</p><Cap>expected artifacts/runs/**/matrix*.jsonl</Cap></P>;
  const src = new Map(m.rs.map((x) => [str(x.policy), str(x.source)]));
  const envs = [...new Set(m.cols.map((c) => c.split('|')[0]))];
  const cell = (p: string, c: string) => m.rs.find((x) => str(x.policy) === p && `${str(x.env_id)}|${str(x.task)}` === c);
  const acc = m.rs.filter((x) => x.status === 'accepted').length;
  return (
    <P title="Policy × env × task · negotiation" link={href('matrix')} result={r} cls="span2 row2" meta={`${acc} accepted · ${m.rs.length - acc} n/a`}>
      <div className="pm" style={{ gridTemplateColumns: `150px repeat(${m.cols.length}, minmax(22px, 1fr))` }}>
        <span />
        {envs.map((e) => <span key={e} className="pm-env" style={{ gridColumn: `span ${m.cols.filter((c) => c.startsWith(`${e}|`)).length}` }}>{e}</span>)}
        <span />
        {m.cols.map((c) => <span key={c} className="pm-task" title={c.replace('|', ' · ')}>{c.split('|')[1].replace(/^cw\//, '')}</span>)}
        {m.pols.map((p) => (
          <FragRow key={p} cells={[
            <span key="l" className="pm-pol" title={`${p} · source ${src.get(p)}`}><i style={{ background: SRC_TONE[src.get(p) || ''] || 'var(--axis)' }} />{p}</span>,
            ...m.cols.map((c) => {
              const x = cell(p, c);
              if (!x) return <span key={c} className="pm-c e" title="not in the recorded matrix" />;
              const accd = x.status === 'accepted';
              const s = isObj(x.summary) ? x.summary : null;
              return <a key={c} className={`pm-c ${accd ? 'ok' : 'na'}`} href={href('matrix', { p, c })} title={`${p} × ${c.replace('|', ' · ')} (${str(x.body)})\n${accd ? 'accepted' : 'n/a'}${s ? ` · ${str(s.successes)}/${str(s.attempted)} success` : ''}\n${arr(x.reasons).map(str).join('\n')}\n${str(x.source_file)}`}>{accd ? (s ? `${str(s.successes)}` : '✓') : '·'}</a>;
            }),
          ]} />
        ))}
      </div>
      {m.missing.length > 0 && <p className="board-note">registered envs with no recorded matrix row: {m.missing.join(', ')}</p>}
      <Cap>rows: policy (tag = source) · cols: env · task · cell: ✓ accepted, · n/a (hover: reasons) · hatched: not in the recorded matrix</Cap>
    </P>
  );
}
function FragRow({ cells }: { cells: ReactNode[] }) { return <>{cells}</>; }

/* ---------- relation-factor registry + catalog */
function FactorsPanel({ r }: { r: DocResult<Envelope> }) {
  const f = factorShape(ok(r));
  const waves = ['W1', 'W2', 'P', 'X', 'M'];
  const wcol: Record<string, string> = { W1: 'var(--good)', W2: 'var(--accent)', P: 'var(--axis)', X: 'var(--surface-3)', M: 'var(--muted)' };
  return (
    <P title="Relation factors · registry" link={href('factors')} result={r} cls="span2" meta={`${f.fs.length} declared · ${f.fs.filter((x) => x.status === 'implemented').length} implemented`}>
      {f.fams.map((fam) => (
        <div key={fam} className="fr">
          <span className="fr-l">{fam}</span>
          <span className="fr-sq">
            {f.fs.filter((x) => x.family === fam).map((x) => (
              <a key={str(x.name)} href={href('factors', { f: str(x.name) })} className={`fsq ${x.status === 'planned' ? 'planned' : ''}`} style={{ background: x.status === 'planned' ? 'transparent' : FORM_COLOR[str(x.form)] || 'var(--axis)', borderColor: FORM_COLOR[str(x.form)] || 'var(--axis)' }}
                title={`${str(x.name)} v${str(x.version)} · ${str(x.status)}\nfield ${str(x.field)} (${str(x.field_prov) || '—'}) × op ${str(x.op)} × form ${str(x.form)} · sources ${arr(x.sources).join(', ')}${arr(x.gates).length ? ` · gates ${arr(x.gates).join(', ')}` : ''}\n${str(x.doc)}`}>
                {x.field_prov === 'privileged' ? <i className="pv" /> : x.field_prov === 'estimated' ? <i className="es" /> : null}
              </a>
            ))}
          </span>
        </div>
      ))}
      <div className="fr">
        <span className="fr-l">catalog</span>
        <span className="wbar">{waves.map((w) => { const n = f.cat.filter((c) => c.status === w).length; return n ? <i key={w} style={{ flex: n, background: wcol[w] }} title={`${w}: ${n} candidate families`}>{w} {n}</i> : null; })}</span>
      </div>
      <Cap>square: registered factor, colour = form (hover: field × op), hollow = planned, dot = estimated/privileged field · bar: catalog families by wave</Cap>
    </P>
  );
}

/* ---------- curriculum scheduler */
function SchedulerPanel({ r }: { r: DocResult<Envelope> }) {
  const f = factorShape(ok(r));
  const s = f.sched[0];
  const recs = s ? rows(s.records) : [];
  if (!recs.length) return null;
  // generic reader: per record {step, shares|share{f}, full_world, levels{f}, competence{f}, interference{f}, reasons[]}
  const steps = recs.map((x, i) => num(x.step) ?? i);
  const shareOf = (x: Row) => (isObj(x.shares) ? x.shares : isObj(x.share) ? x.share : {}) as Record<string, unknown>;
  const facs = [...new Set(recs.flatMap((x) => Object.keys(shareOf(x))))].slice(0, 8);
  const W = 560, H = 120, x0 = Math.min(...steps), x1 = Math.max(...steps);
  const X = (v: number) => 30 + ((v - x0) / (x1 - x0 || 1)) * (W - 40), Y = (v: number) => H - 18 - v * (H - 24);
  let base = recs.map(() => 0);
  const areas = facs.map((fac, k) => {
    const top = recs.map((x, i) => base[i] + (num(shareOf(x)[fac]) || 0));
    const d = `M${steps.map((st, i) => `${X(st)},${Y(top[i])}`).join('L')}L${[...steps].reverse().map((st, j) => `${X(st)},${Y(base[recs.length - 1 - j])}`).join('L')}Z`;
    base = top;
    return <path key={fac} d={d} fill={`var(--s${(k % 8) + 1})`} opacity={0.55}><title>{fac}</title></path>;
  });
  const fw = recs.map((x) => num(x.full_world ?? x.full_world_share));
  const steer = rows(s.steer);
  return (
    <P title="Curriculum scheduler" link={href('factors', { view: 'schedule' })} result={r} cls="span2" meta={`${str(s.run).split('/').slice(-2).join('/')} · ${recs.length} intervals · ${steer.length} steers`}>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="per-factor share over steps">
        {areas}
        {fw.some((v) => v !== null) && <polyline points={steps.map((st, i) => (fw[i] === null ? '' : `${X(st)},${Y(fw[i]!)}`)).filter(Boolean).join(' ')} fill="none" stroke="var(--ink)" strokeWidth={1.5} />}
        {steer.map((e, i) => { const st = num(e.step); return st === null ? null : <line key={i} x1={X(st)} x2={X(st)} y1={6} y2={H - 18} stroke="var(--critical)" strokeDasharray="2 2"><title>{`${str(e.op ?? e.raw)} · ${str(e.reason)} · ${str(e.author)}`}</title></line>; })}
        <text x={30} y={H - 4} fill="var(--muted)">step {x0}</text><text x={W - 10} y={H - 4} fill="var(--muted)" textAnchor="end">{x1}</text>
      </svg>
      <div className="legend">{facs.map((fac, k) => <span key={fac}><i className="sw" style={{ background: `var(--s${(k % 8) + 1})` }} />{fac}</span>)}<span>— full world</span></div>
      <Cap>x: step · area: per-factor share · line: full-world share · red: steers (hover)</Cap>
    </P>
  );
}
