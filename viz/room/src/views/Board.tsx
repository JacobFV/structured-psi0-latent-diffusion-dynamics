/**
 * Board v5 (after IBM-2's overview dashboard): one dense screen at 1440×900. The compact declared radar in the centre,
 * a KPI strip on top, and small visual panels around it; one-line labels, evidence on hover, every panel drills to its
 * view. Layout has a pixel budget (asserted by the render check): each panel's content must fit its fixed cell.
 */
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Cap, ModeBadge, Spark } from '../components/board';
import { seqColor, seqInk } from '../components/charts';
import RadarChart, { type RadarDoc } from '../components/RadarChart';
import { fetchTrainingSeries, useDoc, type DocResult, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, newcombe, num, rows, str, wilson, type Row } from '../lib/format';
import { href } from '../lib/url';

/* ---------- layout budget (px) at 1440×900 */
export const VIEWPORT_H = 900;
export const TICKER_H = 20;
export const PAD = 8;
export const KPI_H = 50;
export const ROWS = 3;
export const CELL_H = Math.floor((VIEWPORT_H - TICKER_H - 2 * PAD - KPI_H) / ROWS); // 271
export const HEAD_H = 18;
export const CAP_H = 40; // up to three 10 px caption lines under each figure (narrow cells wrap)
const LINE = 13;

const ok = (r: DocResult<Envelope>) => (r.status === 'ok' ? r.data : null);
const GB = 1024 ** 3;
const pct = (v: number) => `${Math.round(v * 100)}`;

/* ---------- data helpers (shared with the budget check) */
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
/** Arm v6 (D-134, compare_v6.json): per lineage (semfix/nosem × 2 seeds) and body; 'all' rows give the pooled totals. */
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
/** Cross-body transfer (D-135/D-136): new arm (sealed xarm7, joint adaptation vs BC SFT, update-matched) and new gripper
 * on a known arm (panda_tf3 zero-shot, latent semfix vs BC). k/n pooled over seeds, budgets and targets. */
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
function leggedCells(res: Envelope | null) {
  const m = new Map<string, { k: number; n: number }>();
  const bodies: string[] = [];
  if (res) for (const r of rows(res.rows)) {
    if (!/summary_contact_v2\.json$/.test(str(r.source_file))) continue;
    const kp = arr(r.key_path).map(str);
    const col = kp[1] === 'r2_final' ? kp[2] : kp[1] === 'references' ? kp[2] : null;
    if (!col || kp.length !== 3 || !num(r.n)) continue;
    if (!bodies.includes(kp[0])) bodies.push(kp[0]);
    const x = m.get(`${kp[0]}|${col}`) || { k: 0, n: 0 };
    x.k += num(r.k) || 0; x.n += num(r.n) || 0;
    m.set(`${kp[0]}|${col}`, x);
  }
  return { m, bodies };
}
function armCells(res: Envelope | null) {
  const m = new Map<string, { k: number; n: number }>();
  const bodies: string[] = [];
  if (res) for (const r of rows(res.rows)) {
    const kp = arr(r.key_path).map(str);
    if (kp.length !== 3 || !str(r.source_file).endsWith('compare_gc2_final.json') || kp[2] !== 'grasp_v2') continue;
    const col = /^semfix/.test(kp[0]) ? 'semfix' : /^nosem/.test(kp[0]) ? 'nosem' : /^frozen/.test(kp[0]) ? 'frozen' : /^BC/.test(kp[0]) ? 'bc' : null;
    if (!col) continue;
    if (!bodies.includes(kp[1])) bodies.push(kp[1]);
    const x = m.get(`${kp[1]}|${col}`) || { k: 0, n: 0 };
    x.k += num(r.k) || 0; x.n += num(r.n) || 0;
    m.set(`${kp[1]}|${col}`, x);
  }
  return { m, bodies };
}
const activeDags = (g: Envelope | null) => (g ? rows(g.dags).filter((d) => d.complete === false) : []);
const recentRuns = (t: Envelope | null) => (t ? rows(t.runs).map((x) => ({ x, t: num(isObj(x.last) ? x.last.t : null) })).filter((y) => y.t !== null && y.t > 1.5e9).sort((a, b) => b.t! - a.t!).slice(0, 8).map((y) => y.x) : []);

/** Content height each panel needs (px); the render check asserts HEAD_H + need ≤ CELL_H (radar: ≤ 2 cells). */
export function panelNeeds(docs: Record<string, Envelope | null>) {
  const lc = leggedCells(docs.results ?? null), ac = armCells(docs.results ?? null), a6 = v6(docs.results ?? null);
  const reps = docs.robustness ? rows(docs.robustness.reports) : [];
  return {
    radar: { need: Math.min(2 * CELL_H - HEAD_H - 30 - CAP_H, 520) + 20 + CAP_H, cells: 2 },
    claim: { need: CAP_H + (haltDiffs(docs.edits ?? null).length + 3) * 15 + 2 * 12 + 10, cells: 1 },  // legged rows + 3 arm rows
    success: { need: CAP_H + (lc.bodies.length + Math.max(ac.bodies.length, a6.bodies.length) + 2) * 17 + 8, cells: 1 },
    robustness: { need: CAP_H + (reps.length + 1) * LINE + 6, cells: 1 },
    psi0: { need: CAP_H + (docs.psi0 ? rows(docs.psi0.runs).filter((r) => num(r.n) && (str(r.run).startsWith('step2') || !r.interim)).length : 0) * LINE + 4, cells: 1 },
    leases: { need: CAP_H + (docs.live ? rows(docs.live.leases).length : 0) * LINE + 4, cells: 1 },
    dags: { need: CAP_H + Math.min(10, activeDags(docs.dags ?? null).length + 4) * LINE + 4, cells: 1 },
    training: { need: CAP_H + recentRuns(docs.training ?? null).length * LINE + LINE + 4, cells: 1 },
    decisions: { need: Math.min(15, docs.overview ? rows(docs.overview.latest_decisions).length : 0) * LINE + 4, cells: 1 },
  };
}

function P({ title, link, result, meta, children, cls = '' }: { title: string; link: string; result?: DocResult<Envelope>; meta?: ReactNode; children: ReactNode; cls?: string }) {
  return (
    <section className={`bp ${cls}`}>
      <header><a href={link}>{title}</a>{meta !== undefined && <span className="meta">{meta}</span>}{result && <ModeBadge result={result} />}</header>
      <div className="bb">{children}</div>
    </section>
  );
}

export default function Board() {
  const d = {
    radar: useDoc<Envelope>('radar'), edits: useDoc<Envelope>('edits'), results: useDoc<Envelope>('results'), robustness: useDoc<Envelope>('robustness'),
    psi0: useDoc<Envelope>('psi0'), live: useDoc<Envelope>('live', 10_000), dags: useDoc<Envelope>('dags', 30_000), training: useDoc<Envelope>('training'),
    overview: useDoc<Envelope>('overview'), knowledge: useDoc<Envelope>('knowledge'),
  };
  return (
    <div className="board5" style={{ ['--cell-h' as string]: `${CELL_H}px`, ['--kpi-h' as string]: `${KPI_H}px` }}>
      <Kpis d={d} />
      <div className="b5grid">
        <RadarPanel r={d.radar.result} />
        <ClaimPanel ed={d.edits.result} res={d.results.result} />
        <SuccessPanel res={d.results.result} />
        <RobustPanel r={d.robustness.result} />
        <Psi0Panel r={d.psi0.result} />
        <LeasePanel r={d.live.result} />
        <DagPanel r={d.dags.result} />
        <TrainPanel r={d.training.result} />
        <DecisionPanel r={d.overview.result} />
      </div>
    </div>
  );
}

/* ---------- KPI strip */
function Kpi({ k, v, s, spark, tone, link, tip }: { k: string; v: ReactNode; s?: ReactNode; spark?: (number | null)[]; tone?: string; link: string; tip?: string }) {
  return (
    <a className={`k5 ${tone || ''}`} href={link} title={tip}>
      <span>{k}</span><b>{v}</b>
      {spark && spark.filter((x) => x !== null).length > 1 ? <Spark values={spark} width={84} height={12} /> : <small>{s}</small>}
    </a>
  );
}
function Kpis({ d }: { d: Record<string, { result: DocResult<Envelope> }> }) {
  const l = ok(d.live.result), node = l && isObj(l.node) ? l.node : null;
  const wd = l ? rows(l.watchdog) : [];
  const adm = node && isObj(node.admission) ? node.admission : null;
  const L = lineages(ok(d.results.result));
  const V = v6(ok(d.results.result)).tot;
  const halts = haltDiffs(ok(d.edits.result));
  const s2 = ok(d.psi0.result) ? rows(ok(d.psi0.result)!.runs).filter((r) => str(r.run).startsWith('step2')) : [];
  const road = ok(d.knowledge.result) ? rows(ok(d.knowledge.result)!.roadmap) : [];
  const st = (r: Row) => str(r.status).toLowerCase();
  const col = (s: string) => (/done/.test(s) ? 'var(--good)' : /run/.test(s) ? 'var(--accent)' : /block/.test(s) ? 'var(--critical)' : /open|conditional|queued/.test(s) ? 'var(--warning)' : 'var(--axis)');
  const gpu = node ? num(arr(node.gpu).map((g) => (isObj(g) ? g.util_pct : null))[0]) : null;
  const mem = node ? num(node.memory_available) : null;
  return (
    <div className="k5strip">
      <Kpi k="GPU · temp" v={gpu !== null ? `${gpu}% · ${fmtNum(node?.gpu_temp_c)}°` : '—'} spark={wd.map((s) => num(s.gpu_temp_c))} link={href('training')} tip="peer GPU utilisation and temperature (spark: watchdog GPU temp)" />
      <Kpi k="mem avail" v={mem !== null ? `${(mem / GB).toFixed(0)}G` : '—'} spark={wd.map((s) => { const v = num(s.memory_available); return v === null ? null : v / GB; })} link={href('training')} />
      <Kpi k="PSI full" v={fmtNum(wd.length ? num(wd[wd.length - 1].psi_full_avg10) : null)} spark={wd.map((s) => num(s.psi_full_avg10))} link={href('training')} />
      <Kpi k="jobs · admission" v={`${l ? rows(l.leases).length : '—'} · ${adm ? (adm.stopped ? 'STOP' : 'open') : '—'}`} tone={adm?.stopped ? 'bad' : ''} s={l ? `${fmtNum(l.n_leases_total)} leases total` : ''} link={href('training')} />
      <Kpi k="arm v6 sf | ns" v={V.semfix.n ? `${pct(V.semfix.k / V.semfix.n)}|${pct(V.nosem.k / V.nosem.n)}%` : '—'} tone="good" s={`${V.semfix.k}/${V.semfix.n} vs ${V.nosem.k}/${V.nosem.n} · v1 ${L.semfix.k} vs ${L.nosem.k}`} link={href('results')} tip={`arm v6 (grasp_v2.1, compare_v6.json, D-134): semfix ${V.semfix.k}/${V.semfix.n} vs nosem ${V.nosem.k}/${V.nosem.n}. v1 under grasp_v2 (compare_gc2_final.json, D-127): ${L.semfix.k}/${L.semfix.n} vs ${L.nosem.k}/${L.nosem.n}`} />
      {halts.map((h) => <Kpi key={h.body} k={`halt Δ ${h.body}`} v={`${h.v.toFixed(2)}m`} s={`[${h.lo.toFixed(2)}, ${h.hi.toFixed(2)}]`} link={href('edits', { body: h.body })} tip={`semantic − nosem forward travel after a halt edit, contact v2 (${str(h.row.decision)}, ${str(h.row.source_file)})`} />)}
      <Kpi k="Ψ₀ step 2" v={s2.map((r) => `${str(r.k)}/${str(r.n)}`).join(' · ') || '—'} s={s2.map((r) => str(r.run).replace(/^step2_\w+?_/, '')).join(' · ')} link={href('psi0')} tip="released · direct · structured (structured is interim)" />
      <a className="k5" href={href('knowledge', { tab: 'roadmap' })} title="roadmap items by status (hover a square)">
        <span>open · {road.filter((r) => !/done/.test(st(r))).length}/{road.length}</span>
        <span style={{ display: 'flex', flexWrap: 'wrap', gap: 1, width: 120 }}>{road.map((r, i) => <i key={i} title={`#${str(r.n)} ${str(r.status)}: ${str(r.question)}`} style={{ width: 7, height: 7, background: col(st(r)) }} />)}</span>
      </a>
    </div>
  );
}

/* ---------- panels */
function RadarPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r) as (RadarDoc & Envelope) | null;
  return (
    <P title="Routes · declared radar" link={href('radar')} result={r} cls="span2 row2" meta={d ? `${d.axes?.length ?? 0} axes · 1 = reference` : ''}>
      {d && d.axes?.length ? <><RadarChart radar={d} size={Math.min(2 * CELL_H - HEAD_H - 30 - CAP_H, 520)} compact /><Cap>spokes: {d.axes.length} declared axes (arm/legged success, edit control, robustness, smoothness, latency, transfer) · radius: 0 = axis floor, ring 1 = reference route (BC or teacher, per axis) · lines: routes; gaps = no evidence; whiskers = seed range or 95% CI</Cap></> : <p className="board-note">no radar axes</p>}
    </P>
  );
}
function ClaimPanel({ ed, res }: { ed: DocResult<Envelope>; res: DocResult<Envelope> }) {
  const halts = haltDiffs(ok(ed));
  const L = lineages(ok(res));
  const V = v6(ok(res)).tot;
  const W = 290;
  const legged = halts.length ? (
    <Mini items={halts.map((h) => ({ label: `halt Δ ${h.body}`, v: h.v, lo: h.lo, hi: h.hi, tip: `${h.v} m [${h.lo}, ${h.hi}] · ${str(h.row.decision)} ${str(h.row.source_file)}` }))} lo={-0.8} hi={0.2} fmt={(v) => v.toFixed(2)} W={W} unit="m" />
  ) : null;
  let arm: ReactNode = null;
  const T = transfer(ok(res));
  const armItems: { label: string; v: number; lo: number; hi: number; tip: string; bad?: boolean }[] = [];
  const rowOf = (label: string, a: { k: number; n: number }, b: { k: number; n: number }, src: string) => {
    if (!a.n || !b.n) return;
    const dv = a.k / a.n - b.k / b.n;
    const [lo, hi] = newcombe(b.k, b.n, a.k, a.n);
    armItems.push({ label, v: dv * 100, lo: lo * 100, hi: hi * 100, tip: `${a.k}/${a.n} vs ${b.k}/${b.n}, Newcombe 95% · ${src}`, bad: hi < 0 });
  };
  rowOf('arm v6 sf − ns', V.semfix, V.nosem, 'in-distribution arm bodies · grasp_v2.1 · compare_v6.json · D-134 (v1: +' + (L.semfix.n ? Math.round((L.semfix.k / L.semfix.n - L.nosem.k / L.nosem.n) * 100) : '?') + ' pts, D-127)');
  rowOf('new arm: latent − BC', T.joint, T.bcSft, 'sealed xarm7_pg2+tf3 · semfix joint adaptation (added after D-135) vs BC SFT, update-matched · worse in 12/12 cells · D-136');
  rowOf('new gripper zs: latent − BC', T.gripLatent, T.gripBc, 'panda_tf3 zero-shot · semfix v6 vs BC · targets_v6.json · D-135');
  if (armItems.length) arm = <Mini items={armItems} lo={-40} hi={60} fmt={(v) => `${v > 0 ? '+' : ''}${v.toFixed(0)}`} W={W} unit="pts" />;
  return <P title="Claims" link={href('results')} result={ed} meta={T.joint.n ? 'new-arm transfer: NOT supported' : undefined}>{legged}{arm}<Cap>top: legged forward travel after a halt edit, semantic − nosem (m, contact v2) · bottom: success difference in points (arm v6 in-distribution; new arm and new gripper vs BC) · dot = estimate, bar = 95% CI, red = CI below 0</Cap></P>;
}
function Mini({ items, lo, hi, fmt, W, unit }: { items: { label: string; v: number; lo: number; hi: number; tip: string; bad?: boolean }[]; lo: number; hi: number; fmt: (v: number) => string; W: number; unit: string }) {
  const left = 118, right = 36, H = items.length * 15 + 12;
  const x = (v: number) => left + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (W - left - right);
  return (
    <svg width={W} height={H} role="img" aria-label={items.map((i) => `${i.label} ${fmt(i.v)}`).join('; ')}>
      <line x1={x(0)} x2={x(0)} y1={0} y2={H - 11} stroke="var(--ink-2)" strokeDasharray="2 2" />
      <text x={left} y={H - 1} fill="var(--muted)">{fmt(lo)}</text>
      <text x={W - right} y={H - 1} fill="var(--muted)" textAnchor="end">{fmt(hi)} {unit}</text>
      {items.map((it, i) => (
        <g key={it.label}>
          <title>{it.tip}</title>
          <text x={left - 4} y={i * 15 + 10} textAnchor="end" fill="var(--ink-2)">{it.label}</text>
          <line x1={x(it.lo)} x2={x(it.hi)} y1={i * 15 + 7} y2={i * 15 + 7} stroke={it.bad ? 'var(--down)' : 'var(--up)'} strokeWidth={2.5} opacity={0.5} />
          <circle cx={x(it.v)} cy={i * 15 + 7} r={3.5} fill={it.bad ? 'var(--down)' : 'var(--up)'} />
          <text x={W - right + 4} y={i * 15 + 10} fill="var(--ink)" fontFamily="var(--mono)">{fmt(it.v)}</text>
        </g>
      ))}
    </svg>
  );
}
function SuccessPanel({ res }: { res: DocResult<Envelope> }) {
  const lc = leggedCells(ok(res)), ac = armCells(ok(res)), a6 = v6(ok(res));
  for (const [key, val] of ac.m) a6.cells.set(`${key.split('|')[0]}|${key.split('|')[1]} v1`, val);
  const cell = (x: { k: number; n: number } | undefined, tip: string) => {
    if (!x || !x.n) return <i className="c e" />;
    const p = x.k / x.n;
    const [lo, hi] = wilson(x.k, x.n);
    return <i className="c" style={{ background: seqColor(p), color: seqInk(p) }} title={`${tip}: ${x.k}/${x.n} [${pct(lo)}, ${pct(hi)}]`}>{pct(p)}</i>;
  };
  const block = (title: string, bodies: string[], cols: string[], m: Map<string, { k: number; n: number }>, src: string) => bodies.length ? (
    <div className="mh" style={{ gridTemplateColumns: `72px repeat(${cols.length}, 1fr)` }}>
      <span className="h">{title}</span>{cols.map((c) => <span key={c} className="h">{c}</span>)}
      {bodies.map((b) => <FragRow key={b} cells={[<span key="l" className="r">{b.replace(/_(pg2|tf3)$/, '')}</span>, ...cols.map((c) => <span key={c}>{cell(m.get(`${b}|${c}`), `${b} ${c} · ${src}`)}</span>)]} />)}
    </div>
  ) : null;
  return (
    <P title="Success · curated" link={href('results')} result={res}>
      {block('legged cv2', lc.bodies, ['semfix', 'nosem', 'bc', 'teacher'], lc.m, 'summary_contact_v2 D-124')}
      {a6.bodies.length
        ? block('arm v6 | v1', a6.bodies, ['semfix', 'nosem', 'semfix v1', 'nosem v1', 'bc v1'], a6.cells, 'v6: compare_v6 D-134 (grasp_v2.1) · v1: compare_gc2_final D-127 (grasp_v2)')
        : block('arm gv2', ac.bodies, ['semfix', 'nosem', 'frozen', 'bc'], ac.m, 'compare_gc2_final D-127')}
      <Cap>rows: body · cols: route (legged: contact v2, R2 final) / lineage (arm: v6 grasp_v2.1, then v1 grasp_v2) · cell: success % of pooled episodes (hover k/n) · colour: success rate</Cap>
    </P>
  );
}
function FragRow({ cells }: { cells: ReactNode[] }) { return <>{cells}</>; }
function RobustPanel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const reps = d ? rows(d.reports) : [];
  const factors = [...new Set(reps.flatMap((x) => (isObj(x.break_points) ? Object.keys(x.break_points) : [])))].slice(0, 9);
  return (
    <P title="Robustness · break-points" link={href('robustness')} result={r}>
      <div className="strip5" style={{ gridTemplateColumns: `92px 30px repeat(${factors.length}, 1fr)` }}>
        <span className="h">robot·route</span><span className="h">nom</span>{factors.map((f) => <span key={f} className="h" title={f}>{f.slice(0, 5)}</span>)}
        {reps.map((x, i) => {
          const nom = isObj(x.nominal) ? x.nominal : {};
          const bps = isObj(x.break_points) ? x.break_points : {};
          return (
            <FragRow key={i} cells={[
              <span key="n" className="r" title={str(x.source_file)}>{str(x.robot).replace(/_(pg2|tf3)$/, '').slice(0, 7)}·{str(x.route)}</span>,
              <span key="nm" className="mono" style={{ background: seqColor(num(nom.rate) || 0), color: seqInk(num(nom.rate) || 0) }}>{pct(num(nom.rate) || 0)}</span>,
              ...factors.map((f) => {
                const b = isObj(bps[f]) ? bps[f] : null;
                const lo = b ? num(b.low) : null, hi = b ? num(b.high) : null;
                const has = lo !== null || hi !== null;
                return <span key={f} className="mono" title={b ? `${f}: ${has ? `breaks ${lo !== null ? `≤${lo}` : ''} ${hi !== null ? `≥${hi}` : ''}` : 'no break in range'} (${str(x.decision)})` : `${f}: not swept`} style={{ background: has ? 'color-mix(in srgb, var(--critical) 30%, var(--surface))' : b ? 'var(--surface-2)' : 'transparent' }}>{has ? '×' : b ? '·' : ''}</span>;
              }),
            ]} />
          );
        })}
      </div>
      <Cap>rows: robot · route · nom: unperturbed success % (colour) · cols: perturbation factor · × = success breaks within the swept range, · = no break, blank = not swept</Cap>
    </P>
  );
}
function Psi0Panel({ r }: { r: DocResult<Envelope> }) {
  const d = ok(r);
  const runs = d ? rows(d.runs).filter((x) => num(x.n) && (str(x.run).startsWith('step2') || !x.interim)) : [];
  runs.sort((a, b) => Number(str(b.run).startsWith('step2')) - Number(str(a.run).startsWith('step2')));
  return (
    <P title="Ψ₀ · step 2 and reproduction" link={href('psi0')} result={r}>
      {runs.map((x) => {
        const k = num(x.k)!, n = num(x.n)!;
        const [lo, hi] = Array.isArray(x.ci) ? [num(x.ci[0])!, num(x.ci[1])!] : wilson(k, n);
        return (
          <div key={str(x.run)} className="barrow" title={`${str(x.task)} · ${k}/${n} [${pct(lo)}, ${pct(hi)}]${x.interim ? ` · interim: ${str(x.interim_reason)}` : ''}`}>
            <span>{str(x.run).replace(/^psi0rel_/, 'rel ').replace(/^step2_tabletop_/, 's2 ')}</span>
            <span className="bar"><i style={{ width: `${(k / n) * 100}%`, background: x.interim ? 'var(--axis)' : 'var(--s1)' }} /><s style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} /></span>
            <b>{k}/{n}</b>
          </div>
        );
      })}
      <Cap>rows: Ψ₀ run (s2 = step 2, rel = released checkpoint) · bar: success rate, line = 95% CI · k/n episodes; grey = interim</Cap>
    </P>
  );
}
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
      <Cap>rows: active lease · bar: current memory / declared memory · black tick: peak · orange tick: memory.high throttle line · red fill: throttled</Cap>
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
      <Cap>rows: run DAG (active first) · bar: share of nodes completed (green), running (blue), failed (red) · done/total nodes · ETA only where a track note states one</Cap>
    </P>
  );
}
function TrainPanel({ r }: { r: DocResult<Envelope> }) {
  const t = ok(r);
  const runs = useMemo(() => recentRuns(t), [t]);
  const [series, setSeries] = useState<Record<string, Row>>({});
  useEffect(() => {
    let live = true;
    for (const x of runs) {
      if (Array.isArray(x.step)) { setSeries((s) => ({ ...s, [str(x.id)]: x })); continue; }
      fetchTrainingSeries(str(x.id)).then((res) => { if (live && res.status === 'ok') setSeries((s) => ({ ...s, [str(x.id)]: res.data as Row })); });
    }
    return () => { live = false; };
  }, [runs]);
  return (
    <P title="Training · latest runs" link={href('training')} result={r}>
      <div className="barrow h"><span>run</span><span>loss</span><b>grad</b></div>
      {runs.map((x) => {
        const s = series[str(x.id)];
        const losses = s && isObj(s.losses) ? s.losses : {};
        const lk = ['loss', 'flow', 'total', 'q_loss'].find((k) => Array.isArray(losses[k])) || Object.keys(losses)[0];
        return (
          <a key={str(x.id)} className="barrow" href={href('training', { runs: str(x.id) })} title={`${str(x.run)} · ${str(x.kind)} · last ${new Date((num(isObj(x.last) ? x.last.t : null) || 0) * 1000).toISOString().slice(5, 16)}`}>
            <span>{str(x.run).replace(/^artifacts\/runs\//, '')}</span>
            <span>{lk ? <Spark values={arr(losses[lk]).map(num)} width={90} height={11} /> : null}</span>
            <b>{s && Array.isArray(s.grad_norm) ? <Spark values={arr(s.grad_norm).map(num)} width={44} height={11} color="var(--s2)" /> : '—'}</b>
          </a>
        );
      })}
      <Cap>rows: 8 most recently logged training runs · loss: main logged loss over steps · grad: gradient norm over steps (each line scaled to its own range)</Cap>
    </P>
  );
}
function DecisionPanel({ r }: { r: DocResult<Envelope> }) {
  const o = ok(r);
  const ds = o ? rows(o.latest_decisions) : [];
  return (
    <P title="Latest decisions" link={href('knowledge')} result={r}>
      {ds.slice(0, 15).map((x) => (
        <a key={str(x.id)} className="barrow dline" href={href('knowledge', { tab: 'decisions', d: str(x.id) })} title={str(x.title)}>
          <span className="mono">{str(x.id)}</span><span>{str(x.title)}</span>
        </a>
      ))}
    </P>
  );
}
