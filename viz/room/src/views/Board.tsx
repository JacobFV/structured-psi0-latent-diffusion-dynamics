/**
 * Investor board (v3): only what matters, each with one visual and one big number with its CI, each linked to its evidence.
 *   (a) the central claim: semantic packet → causal task control (legged halt under contact v2; arm semfix vs nosem, grasp_v2)
 *   (b) competence against baselines (latent route vs BC vs teacher, arm and legged)
 *   (c) what is open or blocked
 *   (d) a one-line live status
 * Every figure is read from an exporter document; the rows used are named under each number. A section whose inputs are
 * missing is hidden here (the detail views say "no data" with the path).
 */
import type { ReactNode } from 'react';
import { ModeBadge } from '../components/board';
import { SideGroup, SidebarControls } from '../components/shell';
import { useDoc, type DocResult, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, newcombe, num, pick, rows, str, timeOf, wilson, type Row } from '../lib/format';
import { href } from '../lib/url';

const ok = (r: DocResult<Envelope>) => (r.status === 'ok' ? r.data : null);

/** Horizontal dot + CI whisker chart with a zero (or reference) line. */
function Whiskers({ items, lo, hi, ref0 = 0, fmt = (v: number) => fmtNum(v), width = 300, unit }: {
  items: { label: string; v: number; lo: number; hi: number; tone?: string; link?: string; sub?: string }[];
  lo: number; hi: number; ref0?: number | null; fmt?: (v: number) => string; width?: number; unit?: string;
}) {
  const left = 92, right = 64, rowH = 17, h = items.length * rowH + 14;
  const x = (v: number) => left + ((v - lo) / (hi - lo || 1)) * (width - left - right);
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${h}`} role="img" aria-label={items.map((i) => `${i.label} ${fmt(i.v)}`).join(', ')} style={{ maxWidth: width * 1.6 }}>
      {ref0 !== null && ref0 >= lo && ref0 <= hi && <line x1={x(ref0)} x2={x(ref0)} y1={0} y2={h - 12} stroke="var(--ink-2)" strokeDasharray="2 2" />}
      <text x={left} y={h - 2} fontSize={9} fill="var(--muted)">{fmt(lo)}</text>
      <text x={width - right} y={h - 2} fontSize={9} fill="var(--muted)" textAnchor="end">{fmt(hi)}{unit ? ` ${unit}` : ''}</text>
      {items.map((it, i) => {
        const y = i * rowH + 9;
        const col = it.tone || 'var(--ink)';
        const g = (
          <g key={it.label}>
            <title>{`${it.label}: ${fmt(it.v)} [${fmt(it.lo)}, ${fmt(it.hi)}]${it.sub ? ` · ${it.sub}` : ''}`}</title>
            <text x={left - 6} y={y + 3.5} fontSize={11} fill="var(--ink-2)" textAnchor="end">{it.label}</text>
            <line x1={x(it.lo)} x2={x(it.hi)} y1={y} y2={y} stroke={col} strokeWidth={2} strokeLinecap="round" opacity={0.55} />
            <circle cx={x(it.v)} cy={y} r={4} fill={col} />
            <text x={width - right + 6} y={y + 3.5} fontSize={11} fill="var(--ink)" fontFamily="var(--mono)">{fmt(it.v)}</text>
          </g>
        );
        return it.link ? <a key={it.label} href={it.link}>{g}</a> : g;
      })}
    </svg>
  );
}
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

function Claim({ q, result, children, tone }: { q: ReactNode; result: DocResult<Envelope>; children: ReactNode; tone?: 'good' | 'bad' }) {
  return (
    <section className={`claim ${tone || ''}`}>
      <div className="q">{q}<ModeBadge result={result} /></div>
      {children}
    </section>
  );
}

/* (a1) legged halt effect under contact v2 ------------------------------------------------------ */
function leggedHalt(ed: Envelope | null) {
  if (!ed) return [];
  const rs = rows(ed.rows);
  const out: { body: string; diff: Row; sem?: Row; nosem?: Row }[] = [];
  for (const body of ['anymal_c', 'go2']) {
    // the recorded pooled contrast (fixsem/semfix − nosem) from the contact-v2 summary
    const diff = rs.find((r) => str(r.body) === body && /summary_contact_v2/.test(str(r.source_file)) && /pooled_diff/.test(str(r.metric)));
    const cmp = rs.filter((r) => str(r.body) === body && /legged8_compare/.test(str(r.source_file)) && str(r.edit) === 'ctx_halt');
    if (!diff) continue;
    out.push({ body, diff, sem: cmp.find((r) => /sem/.test(str(r.variant)) && !/nosem/.test(str(r.variant))), nosem: cmp.find((r) => str(r.variant) === 'nosem') });
  }
  return out;
}
/* (a2) arm semfix vs nosem under grasp_v2 ------------------------------------------------------- */
function armLineages(res: Envelope | null) {
  if (!res) return [];
  const by = new Map<string, { label: string; k: number; n: number; bodies: number }>();
  for (const r of rows(res.rows)) {
    const kp = arr(r.key_path);
    if (str(r.metric) !== 'grasp_v2' || kp.length !== 3 || !str(r.source_file).endsWith('compare_gc2_final.json')) continue;
    const label = str(kp[0]);
    const x = by.get(label) || { label, k: 0, n: 0, bodies: 0 };
    x.k += num(r.k) || 0; x.n += num(r.n) || 0; x.bodies++;
    by.set(label, x);
  }
  return [...by.values()];
}
function sumOf(lin: ReturnType<typeof armLineages>, re: RegExp) {
  const xs = lin.filter((l) => re.test(l.label));
  return { k: xs.reduce((a, l) => a + l.k, 0), n: xs.reduce((a, l) => a + l.n, 0), labels: xs.map((l) => l.label), bodies: xs[0]?.bodies ?? 0 };
}

export default function Board() {
  const edits = useDoc<Envelope>('edits');
  const results = useDoc<Envelope>('results');
  const robust = useDoc<Envelope>('robustness');
  const overview = useDoc<Envelope>('overview');
  const knowledge = useDoc<Envelope>('knowledge');
  const psi0 = useDoc<Envelope>('psi0');
  const live = useDoc<Envelope>('live', 10_000);
  const dags = useDoc<Envelope>('dags', 30_000);

  const halt = leggedHalt(ok(edits.result));
  const lin = armLineages(ok(results.result));
  const sf = sumOf(lin, /^semfix/i), ns = sumOf(lin, /^nosem/i), bc = sumOf(lin, /^BC/i), fz = sumOf(lin, /^frozen/i);
  const q = 'compare_gc2_final';
  const armLink = href('results', { q, metric: 'grasp_v2', r1: 'body_or_hint', c: 'source_file', mode: 'rate' });

  return (
    <>
      <SidebarControls>
        <SideGroup title="This board">
          <p className="side-note" style={{ padding: 0 }}>
            The few numbers that carry the project, each with its interval and a link to the rows behind it. Everything else is in the
            views below the picker.
          </p>
        </SideGroup>
        <SideGroup title="Drill down">
          <div className="side-list">
            {[['runs', 'Run history: 160 recorded episodes'], ['results', 'Results matrix'], ['edits', 'Causal edits'], ['robustness', 'Robustness'], ['live', 'Live ops'], ['knowledge', 'Decisions & roadmap']].map(([v, t]) => (
              <button key={v} onClick={() => { window.location.hash = href(v).slice(1); }}>{t}</button>
            ))}
          </div>
        </SideGroup>
      </SidebarControls>
      <div className="inv">
        <LiveLine live={live.result} dags={dags.result} overview={overview.result} />
        {(halt.length > 0 || sf.n > 0) && (
          <div className="inv-row">
            {halt.length > 0 && (
              <Claim result={edits.result} q="Central claim · legged: a halt packet stops the robot (contact v2)">
                {(() => {
                  const main = halt[0];
                  const [lo, hi] = [num(arr(main.diff.ci)[0]), num(arr(main.diff.ci)[1])];
                  return (
                    <>
                      <a className="big" href={href('edits', { body: main.body, edit: 'halt' })}>{fmtNum(num(main.diff.effect))} m</a>
                      <div className="ci">95% CI [{fmtNum(lo)}, {fmtNum(hi)}] · {main.body} · semantic − nosem forward travel after a halt edit</div>
                    </>
                  );
                })()}
                <Whiskers unit="m" lo={Math.min(-0.8, ...halt.flatMap((h) => [num(arr(h.diff.ci)[0]) ?? 0, num(arr(h.sem?.ci)[0]) ?? 0]))} hi={Math.max(0.5, ...halt.flatMap((h) => [num(arr(h.nosem?.ci)[1]) ?? 0]))}
                  items={halt.flatMap((h) => [
                    ...(h.sem ? [{ label: `${h.body} ${str(h.sem.variant)}`, v: num(h.sem.effect)!, lo: num(arr(h.sem.ci)[0])!, hi: num(arr(h.sem.ci)[1])!, tone: 'var(--up)', link: href('edits', { body: h.body, edit: 'ctx_halt' }) }] : []),
                    ...(h.nosem ? [{ label: `${h.body} nosem`, v: num(h.nosem.effect)!, lo: num(arr(h.nosem.ci)[0])!, hi: num(arr(h.nosem.ci)[1])!, tone: 'var(--muted)', link: href('edits', { body: h.body, edit: 'ctx_halt' }) }] : []),
                    { label: `${h.body} Δ`, v: num(h.diff.effect)!, lo: num(arr(h.diff.ci)[0])!, hi: num(arr(h.diff.ci)[1])!, tone: 'var(--ink)', link: href('edits', { body: h.body, edit: 'halt' }) },
                  ])} />
                <div className="what">With the semantic packet, halting the context makes the robot travel less (negative); without it (nosem), it does not.</div>
                <div className="src">Δ = recorded pooled contrast, {halt.map((h) => `${h.body} ${str(h.diff.decision) || ''}`).join(', ')} (<a href={href('knowledge', { tab: 'docs', doc: 'research/tracks/legged8.md' })}>summary_contact_v2.json</a>); per-variant rows from legged8_compare_*.json.</div>
              </Claim>
            )}
            {sf.n > 0 && ns.n > 0 && (() => {
              const a = sf.k / sf.n, b = ns.k / ns.n;
              const [dlo, dhi] = newcombe(ns.k, ns.n, sf.k, sf.n);
              const items = [['semfix', sf, 'var(--up)'], ['frozen sem', fz, 'var(--s7)'], ['BC direct', bc, 'var(--ink-2)'], ['nosem', ns, 'var(--muted)']] as const;
              return (
                <Claim result={results.result} q="Central claim · arm: semantic supervision under realistic grasp (grasp_v2)">
                  <a className="big" href={armLink}>{sf.k} vs {ns.k} <span style={{ fontSize: 'var(--fs-l)', color: 'var(--muted)' }}>of {sf.n}</span></a>
                  <div className="ci">semfix {pct(a)} vs nosem {pct(b)} · difference {(100 * (a - b)).toFixed(0)} pts, 95% CI [{(100 * dlo).toFixed(0)}, {(100 * dhi).toFixed(0)}] (Newcombe)</div>
                  <Whiskers fmt={pct} lo={0} hi={1} ref0={null}
                    items={items.filter(([, s]) => s.n).map(([label, s, tone]) => { const [l, h] = wilson(s.k, s.n); return { label, v: s.k / s.n, lo: l, hi: h, tone, link: armLink, sub: `${s.k}/${s.n} · ${s.labels.join(', ')}` }; })} />
                  <div className="what">Deployable pick-and-place success pooled over {sf.bodies} arm bodies and 2 training seeds (BC: one seed).</div>
                  <div className="src">Σk/Σn over the lineage rows of <a href={armLink}>armexpert_gc2eval/compare_gc2_final.json</a> (D-121, D-127); Wilson intervals computed here.</div>
                </Claim>
              );
            })()}
          </div>
        )}
        <Baselines robust={robust.result} />
        <OpenBlocked overview={overview.result} knowledge={knowledge.result} psi0={psi0.result} />
      </div>
    </>
  );
}

/* (b) competence vs baselines: nominal (unperturbed) success from the robustness reports, same seeds per robot ------ */
function Baselines({ robust }: { robust: DocResult<Envelope> }) {
  const d = ok(robust);
  if (!d) return null;
  const reps = rows(d.reports).filter((r) => /^(teacher|bc|semfix|nosem|frozen_sem)$/.test(str(r.route)));
  const robots = [...new Set(reps.map((r) => str(r.robot)))];
  if (!robots.length) return null;
  const tone = (route: string) => (route === 'teacher' ? 'var(--warning)' : route === 'bc' ? 'var(--ink-2)' : /nosem/.test(route) ? 'var(--muted)' : 'var(--up)');
  return (
    <div className="inv-row">
      {robots.map((robot) => {
        const rs = reps.filter((r) => str(r.robot) === robot);
        const lat = rs.filter((r) => /sem/.test(str(r.route)) && !/nosem/.test(str(r.route)));
        const best = lat.map((r) => isObj(r.nominal) ? r.nominal : {}).map((n) => ({ k: num(n.k) || 0, n: num(n.n) || 0 }))[0];
        const bcr = rs.find((r) => str(r.route) === 'bc');
        const bcn = bcr && isObj(bcr.nominal) ? bcr.nominal : null;
        const link = href('robustness', { robot });
        return (
          <Claim key={robot} result={robust} q={`Competence · ${robot}: latent route vs BC vs scripted teacher`}>
            {best && bcn ? (
              <a className="big" href={link}>{best.k}/{best.n} <span style={{ fontSize: 'var(--fs-l)', color: 'var(--muted)' }}>vs BC {str(bcn.k)}/{str(bcn.n)}</span></a>
            ) : null}
            <Whiskers fmt={pct} lo={0} hi={1} ref0={null}
              items={rs.map((r) => {
                const n = isObj(r.nominal) ? r.nominal : {};
                const k = num(n.k) || 0, nn = num(n.n) || 0;
                const [l, h] = Array.isArray(n.ci) ? [num(n.ci[0])!, num(n.ci[1])!] : wilson(k, nn);
                return { label: str(r.route), v: nn ? k / nn : 0, lo: l, hi: h, tone: tone(str(r.route)), link, sub: `${k}/${nn} · ${arr(n.sources).map(str).join(', ')}` };
              })} />
            <div className="src">Nominal (unperturbed) episodes of the robustness sweep, same seeds for every route ({[...new Set(rs.map((r) => str(r.decision)))].join(', ')}). Teacher = scripted, privileged: an upper reference, not a competitor.</div>
          </Claim>
        );
      })}
    </div>
  );
}

/* (c) open or blocked ---------------------------------------------------------------------------------------------- */
function OpenBlocked({ overview, knowledge, psi0 }: { overview: DocResult<Envelope>; knowledge: DocResult<Envelope>; psi0: DocResult<Envelope> }) {
  const kn = ok(knowledge) || ok(overview);
  const road = kn ? rows(kn.roadmap ?? kn.open) : [];
  const pick1 = (re: RegExp) => road.find((r) => re.test(str(r.question)));
  const items: { st: string; text: string; link: string; why?: string }[] = [];
  const add = (r: Row | undefined, why: string) => {
    if (r && !items.some((i) => i.text === str(r.question))) items.push({ st: str(r.status), text: str(r.question), link: href('knowledge', { tab: 'roadmap', q: `#${str(r.n)}` }), why });
  };
  add(pick1(/humanoid/i), 'humanoid trackers do not pass the gates yet: humanoid tasks are untestable');
  add(pick1(/sealed target bodies|held-out/i), 'held-out (sealed) bodies not yet tested');
  add(pick1(/Legged held-out/i), 'legged held-out transfer not started');
  road.filter((r) => /block/i.test(str(r.status))).forEach((r) => add(r, 'blocked'));
  const ps = ok(psi0);
  const s2 = ps ? rows(ps.runs).find((r) => str(r.run) === 'step2_tabletop_structured') : undefined;
  if (s2) items.push({ st: s2.interim ? 'interim' : 'recorded', text: `Ψ₀ step 2 · structured (ours): ${str(s2.k)}/${str(s2.n)} so far`, link: href('psi0'), why: str(s2.interim_reason) });
  if (!items.length) return null;
  return (
    <div className="inv-row">
      <Claim result={knowledge.status === 'ok' ? knowledge : overview} q="Open or blocked">
        <div className="openlist">
          {items.slice(0, 6).map((i) => (
            <a key={i.text} href={i.link} title={i.why}>
              <span className={`st status ${/block|fail/.test(i.st) ? 'critical' : /interim|run/.test(i.st) ? 'active' : 'neutral'}`}><i />{i.st}</span>
              <span>{i.text}</span>
            </a>
          ))}
        </div>
        <div className="src">From docs/experiments_roadmap.md (status as written) and the Ψ₀ run summaries.</div>
      </Claim>
    </div>
  );
}

/* (d) one-line live status ----------------------------------------------------------------------------------------- */
function LiveLine({ live, dags, overview }: { live: DocResult<Envelope>; dags: DocResult<Envelope>; overview: DocResult<Envelope> }) {
  const l = ok(live), g = ok(dags), o = ok(overview);
  if (!l && !g && !o) return null;
  const node = l && isObj(l.node) ? l.node : null;
  const adm = node && isObj(node.admission) ? node.admission : null;
  const ws = o ? rows(o.workstreams).filter((w) => /run|implement/.test(str(w.state))) : [];
  const active = g ? rows(g.dags).filter((d) => d.complete === false) : [];
  return (
    <div className="statusline">
      <ModeBadge result={live} />
      {adm && <span>peer <b style={{ color: adm.stopped ? 'var(--critical)' : 'var(--good)' }}>{adm.stopped ? 'ADMISSION STOPPED' : 'admitting'}</b></span>}
      {l && <a href={href('live')}><b>{rows(l.leases).length}</b> jobs running</a>}
      {ws.length > 0 && <span>running: {ws.map((w) => <a key={str(w.id)} href={href('knowledge', { tab: 'strategy' })} title={str(w.workstream)} style={{ marginRight: 6 }}><b>{str(w.id)}</b> {str(w.workstream).split(/[,:(]/)[0].slice(0, 28)}</a>)}</span>}
      {active.map((d) => {
        const c = isObj(d.counts) ? d.counts : {};
        const total = Object.values(c).reduce<number>((a, v) => a + (num(v) || 0), 0);
        const eta = arr(d.eta_statements).map((e) => str(pick(e, 'text', 'eta') ?? e)).join('; ');
        return <a key={str(d.dag)} href={href('live', { ws: str(d.track) })}>{str(d.dag)} <b>{fmtNum(c.completed ?? 0)}/{total}</b>{eta ? ` · ETA ${eta}` : ''}</a>;
      })}
      {active.length > 0 && active.every((d) => !arr(d.eta_statements).length) && <span style={{ color: 'var(--muted)' }}>no ETA recorded</span>}
      {l && <span style={{ marginLeft: 'auto', color: 'var(--muted)' }}>peer read {l.peer_read_at ? new Date(timeOf(l.peer_read_at) || 0).toTimeString().slice(0, 5) : '—'}</span>}
    </div>
  );
}
