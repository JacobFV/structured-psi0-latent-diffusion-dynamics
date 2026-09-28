/**
 * Board (v3, one screen): four tiles (claim, competence, open, live), each one chart and one big number with its CI.
 * Explanations live in hover titles and the evidence links. Layout has a pixel budget (TILE_H at 1440×900) that the render
 * check asserts, so the board never scrolls at that size. Every figure comes from an exporter document.
 */
import type { ReactNode } from 'react';
import { ModeBadge, Spark } from '../components/board';
import { useDoc, type DocResult, type Envelope } from '../lib/api';
import { arr, fmtNum, isObj, newcombe, num, rows, str, timeOf, wilson, type Row } from '../lib/format';
import { href } from '../lib/url';

/* ---------- layout budget (px) */
export const VIEWPORT_H = 900;
export const TICKER_H = 20;
export const PAD = 8;
export const TILE_H = Math.floor((VIEWPORT_H - TICKER_H - 2 * PAD) / 2); // two rows of tiles
export const TILE_CHROME = 22 /* header */ + 40 /* big number */ + 16 /* ci line */ + 14 /* evidence link */ + 12 /* padding */;
const ROW = 17, AXIS = 14;

const ok = (r: DocResult<Envelope>) => (r.status === 'ok' ? r.data : null);
const pct = (v: number) => `${Math.round(v * 100)}%`;

/* ---------- data selection (shared by the tiles and the budget check) */
function lineages(res: Envelope | null) {
  const by = new Map<string, { k: number; n: number }>();
  if (res) for (const r of rows(res.rows)) {
    const kp = arr(r.key_path);
    if (str(r.metric) !== 'grasp_v2' || kp.length !== 3 || !str(r.source_file).endsWith('compare_gc2_final.json')) continue;
    const x = by.get(str(kp[0])) || { k: 0, n: 0 };
    x.k += num(r.k) || 0; x.n += num(r.n) || 0;
    by.set(str(kp[0]), x);
  }
  const sum = (re: RegExp) => [...by.entries()].filter(([l]) => re.test(l)).reduce((a, [, v]) => ({ k: a.k + v.k, n: a.n + v.n }), { k: 0, n: 0 });
  return { semfix: sum(/^semfix/i), nosem: sum(/^nosem/i), bc: sum(/^BC/i), frozen: sum(/^frozen/i) };
}
function haltRows(ed: Envelope | null) {
  if (!ed) return [];
  const rs = rows(ed.rows);
  return ['anymal_c', 'go2'].flatMap((body) => {
    const diff = rs.find((r) => str(r.body) === body && /summary_contact_v2/.test(str(r.source_file)) && /pooled_diff/.test(str(r.metric)));
    return diff ? [{ body, diff }] : [];
  });
}
function competence(rb: Envelope | null) {
  const reps = rb ? rows(rb.reports).filter((r) => /^(teacher|bc|semfix|nosem|frozen_sem)$/.test(str(r.route))) : [];
  const robots = [...new Set(reps.map((r) => str(r.robot)))];
  const routes = ['teacher', 'bc', 'semfix', 'frozen_sem', 'nosem'].filter((x) => reps.some((r) => str(r.route) === x));
  return { reps, robots, routes };
}
function roadmap(kn: Envelope | null) {
  return kn ? rows(kn.roadmap) : [];
}
function activeDags(g: Envelope | null) {
  return g ? rows(g.dags).filter((d) => d.complete === false) : [];
}
/** Pixel height each tile's chart needs; the render check asserts TILE_CHROME + need ≤ TILE_H. */
export function tileNeeds(docs: Record<string, Envelope | null>) {
  const h = haltRows(docs.edits ?? null).length;
  const c = competence(docs.robustness ?? null);
  const road = roadmap(docs.knowledge ?? null);
  const dags = activeDags(docs.dags ?? null).slice(0, 5);
  return {
    claim: (2 + h) * ROW + 2 * AXIS + 8,
    competence: c.routes.length * ROW + AXIS + 16,
    open: Math.ceil(Math.max(1, road.length) / 18) * 14 + 3 * 16 + 8,
    live: Math.min(5, dags.length) * 16 + 34 + 16,
  };
}

/* ---------- tile */
function Tile({ title, result, big, ci, link, linkText, tone, tip, children }: {
  title: string; result: DocResult<Envelope>; big: ReactNode; ci?: ReactNode; link: string; linkText: string; tone?: 'good' | 'bad'; tip?: string; children: ReactNode;
}) {
  return (
    <section className={`btile ${tone || ''}`} title={tip}>
      <header><span>{title}</span><ModeBadge result={result} /></header>
      <a className="big" href={link}>{big}</a>
      <div className="ci">{ci}</div>
      <div className="chart">{children}</div>
      <a className="ev" href={link}>{linkText} ›</a>
    </section>
  );
}

/** Dot + whisker rows on one axis. */
function Dots({ items, lo, hi, fmt, zero }: { items: { label: string; v: number; lo: number; hi: number; tone: string; tip: string; link?: string }[]; lo: number; hi: number; fmt: (v: number) => string; zero?: boolean }) {
  const W = 420, left = 104, right = 50, H = items.length * ROW + AXIS;
  const x = (v: number) => left + ((v - lo) / (hi - lo || 1)) * (W - left - right);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="xMinYMin meet" role="img" aria-label={items.map((i) => `${i.label} ${fmt(i.v)}`).join('; ')}>
      {zero && lo < 0 && hi > 0 && <line x1={x(0)} x2={x(0)} y1={0} y2={H - AXIS + 2} stroke="var(--ink-2)" strokeDasharray="2 2" />}
      <text x={left} y={H - 2} fontSize={10} fill="var(--muted)">{fmt(lo)}</text>
      <text x={W - right} y={H - 2} fontSize={10} fill="var(--muted)" textAnchor="end">{fmt(hi)}</text>
      {items.map((it, i) => {
        const y = i * ROW + 9;
        const g = (
          <g key={it.label}>
            <title>{it.tip}</title>
            <text x={left - 6} y={y + 4} fontSize={11} fill="var(--ink-2)" textAnchor="end">{it.label}</text>
            <line x1={x(it.lo)} x2={x(it.hi)} y1={y} y2={y} stroke={it.tone} strokeWidth={2.5} strokeLinecap="round" opacity={0.5} />
            <circle cx={x(it.v)} cy={y} r={4.5} fill={it.tone} />
            <text x={W - right + 6} y={y + 4} fontSize={11} fill="var(--ink)" fontFamily="var(--mono)">{fmt(it.v)}</text>
          </g>
        );
        return it.link ? <a key={it.label} href={it.link}>{g}</a> : g;
      })}
    </svg>
  );
}

export default function Board() {
  const edits = useDoc<Envelope>('edits');
  const results = useDoc<Envelope>('results');
  const robust = useDoc<Envelope>('robustness');
  const knowledge = useDoc<Envelope>('knowledge');
  const psi0 = useDoc<Envelope>('psi0');
  const live = useDoc<Envelope>('live', 10_000);
  const dags = useDoc<Envelope>('dags', 30_000);
  return (
    <div className="board1" style={{ ['--tile-h' as string]: `${TILE_H}px` }}>
      <ClaimTile edits={edits.result} results={results.result} />
      <CompetenceTile robust={robust.result} results={results.result} />
      <OpenTile knowledge={knowledge.result} psi0={psi0.result} />
      <LiveTile live={live.result} dags={dags.result} />
    </div>
  );
}

/* ---------- 1 · claim */
function ClaimTile({ edits, results }: { edits: DocResult<Envelope>; results: DocResult<Envelope> }) {
  const L = lineages(ok(results));
  const halt = haltRows(ok(edits));
  const { semfix: s, nosem: n } = L;
  if (!s.n && !halt.length) return null;
  const d = s.n && n.n ? s.k / s.n - n.k / n.n : null;
  const [dlo, dhi] = s.n && n.n ? newcombe(n.k, n.n, s.k, s.n) : [null, null];
  const armLink = href('results', { q: 'compare_gc2_final', metric: 'grasp_v2' });
  const arm = [['semfix', s, 'var(--up)'], ['nosem', n, 'var(--muted)']] as const;
  const hs = halt.map((h) => ({ body: h.body, v: num(h.diff.effect)!, lo: num(arr(h.diff.ci)[0])!, hi: num(arr(h.diff.ci)[1])!, dec: str(h.diff.decision) }));
  return (
    <Tile title="Semantic packet → causal task control" result={s.n ? results : edits} link={armLink} linkText="evidence: grasp_v2 rows, halt contrasts"
      big={d !== null ? <>+{Math.round(d * 100)} pts</> : `${fmtNum(hs[0]?.v)} m`}
      ci={d !== null ? <>arm semfix {s.k}/{s.n} vs nosem {n.k}/{n.n} · 95% CI [{Math.round(dlo! * 100)}, {Math.round(dhi! * 100)}]</> : null}
      tone="good"
      tip="Arm: deployable pick-and-place success under grasp_v2, pooled over bodies and 2 seeds (compare_gc2_final.json; D-121/D-127; Newcombe CI). Legged: recorded pooled contrast of forward travel after a halt context edit, semantic − nosem, under contact v2 (summary_contact_v2.json; D-124).">
      {s.n > 0 && <Dots fmt={pct} lo={0} hi={1} items={arm.map(([label, x, tone]) => { const [lo, hi] = wilson(x.k, x.n); return { label: `arm ${label}`, v: x.k / x.n, lo, hi, tone, tip: `${x.k}/${x.n} grasp_v2 successes`, link: armLink }; })} />}
      {hs.length > 0 && <Dots zero fmt={(v) => `${v.toFixed(2)} m`} lo={Math.min(-0.8, ...hs.map((h) => h.lo))} hi={0.2}
        items={hs.map((h) => ({ label: `halt Δ ${h.body}`, v: h.v, lo: h.lo, hi: h.hi, tone: 'var(--up)', tip: `${h.body}: semantic − nosem forward travel after halt, ${h.v} m [${h.lo}, ${h.hi}] (${h.dec})`, link: href('edits', { body: h.body, edit: 'halt' }) }))} />}
    </Tile>
  );
}

/* ---------- 2 · competence vs baselines */
function CompetenceTile({ robust, results }: { robust: DocResult<Envelope>; results: DocResult<Envelope> }) {
  const c = competence(ok(robust));
  const L = lineages(ok(results));
  if (!c.robots.length) return null;
  const tone = (r: string) => (r === 'teacher' ? 'var(--warning)' : r === 'bc' ? 'var(--ink-2)' : r === 'nosem' ? 'var(--muted)' : 'var(--up)');
  const W = 420, left = 76, colW = (W - left) / c.robots.length, H = c.routes.length * ROW + AXIS;
  const lat = L.semfix, bc = L.bc;
  return (
    <Tile title="Competence vs baselines" result={robust} link={href('robustness')} linkText="evidence: nominal episodes of the robustness sweep"
      big={lat.n && bc.n ? <>{pct(lat.k / lat.n)} <span className="vs">vs BC {pct(bc.k / bc.n)}</span></> : '—'}
      ci={lat.n && bc.n ? <>arm grasp_v2 latent (semfix) {lat.k}/{lat.n} vs BC direct {bc.k}/{bc.n}</> : null}
      tip="Big number: arm grasp_v2 success, latent route (semfix, 2 seeds) vs plain BC (1 seed), compare_gc2_final.json. Chart: nominal success per robot and route, same 20 seeds per route (D-108/D-114); the scripted teacher is privileged, an upper reference.">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="xMinYMin meet" role="img" aria-label="nominal success per robot and route">
        {c.robots.map((rb, j) => <text key={rb} x={left + j * colW + colW / 2} y={H - 2} fontSize={10.5} textAnchor="middle" fill="var(--muted)">{rb}</text>)}
        {c.routes.map((rt, i) => {
          const y = i * ROW + 9;
          return (
            <g key={rt}>
              <text x={left - 6} y={y + 4} fontSize={11} textAnchor="end" fill="var(--ink-2)">{rt}</text>
              {c.robots.map((rb, j) => {
                const r = c.reps.find((x) => str(x.robot) === rb && str(x.route) === rt);
                const nm = r && isObj(r.nominal) ? r.nominal : null;
                if (!nm) return null;
                const k = num(nm.k) || 0, n = num(nm.n) || 0;
                const [lo, hi] = Array.isArray(nm.ci) ? [num(nm.ci[0])!, num(nm.ci[1])!] : wilson(k, n);
                const x = (v: number) => left + j * colW + 6 + v * (colW - 12);
                return (
                  <a key={rb} href={href('robustness', { robot: rb })}>
                    <title>{`${rb} ${rt}: ${k}/${n} nominal [${pct(lo)}, ${pct(hi)}] (${str(r!.decision)})`}</title>
                    <line x1={left + j * colW + 6} x2={left + (j + 1) * colW - 6} y1={y} y2={y} stroke="var(--grid)" />
                    <line x1={x(lo)} x2={x(hi)} y1={y} y2={y} stroke={tone(rt)} strokeWidth={2.5} opacity={0.5} />
                    <circle cx={x(n ? k / n : 0)} cy={y} r={4.5} fill={tone(rt)} />
                  </a>
                );
              })}
            </g>
          );
        })}
      </svg>
    </Tile>
  );
}

/* ---------- 3 · open or blocked */
function OpenTile({ knowledge, psi0 }: { knowledge: DocResult<Envelope>; psi0: DocResult<Envelope> }) {
  const road = roadmap(ok(knowledge));
  if (!road.length) return null;
  const st = (r: Row) => str(r.status).toLowerCase();
  const col = (s: string) => (/done|complete/.test(s) ? 'var(--good)' : /run/.test(s) ? 'var(--accent)' : /block/.test(s) ? 'var(--critical)' : /open|conditional|queued/.test(s) ? 'var(--warning)' : 'var(--axis)');
  const notDone = road.filter((r) => !/done|complete/.test(st(r)));
  const find = (re: RegExp) => road.find((r) => re.test(str(r.question)));
  const s2 = ok(psi0) ? rows(ok(psi0)!.runs).find((r) => str(r.run) === 'step2_tabletop_structured') : undefined;
  const keys: { label: string; r?: Row; tip: string; link: string; s: string }[] = [
    { label: 'humanoid trackers pass gates?', r: find(/humanoid/i), tip: '', link: '', s: '' },
    { label: 'held-out bodies tested?', r: find(/sealed target bodies|held-out/i), tip: '', link: '', s: '' },
  ].filter((k) => k.r).map((k) => ({ ...k, s: str(k.r!.status), tip: str(k.r!.question), link: href('knowledge', { tab: 'roadmap', q: `#${str(k.r!.n)}` }) }));
  if (s2) keys.push({ label: `Ψ₀ structured ${str(s2.k)}/${str(s2.n)}`, tip: str(s2.interim_reason) || 'Ψ₀ step 2, our structure', link: href('psi0'), s: s2.interim ? 'interim' : 'recorded' });
  const per = 18;
  return (
    <Tile title="Open or blocked" result={knowledge} link={href('knowledge', { tab: 'roadmap' })} linkText="roadmap"
      big={<>{notDone.length}<span className="vs"> of {road.length} open</span></>}
      ci={<>{road.filter((r) => /run/.test(st(r))).length} running · {road.filter((r) => /block/.test(st(r))).length} blocked · {road.filter((r) => /done/.test(st(r))).length} done</>}
      tip="Roadmap items from docs/experiments_roadmap.md, one square each, coloured by recorded status; hover a square for the question.">
      <svg viewBox={`0 0 420 ${Math.ceil(road.length / per) * 14}`} width="100%" height={Math.ceil(road.length / per) * 14} preserveAspectRatio="xMinYMin meet" role="img" aria-label="roadmap status">
        {road.map((r, i) => (
          <a key={i} href={href('knowledge', { tab: 'roadmap', q: `#${str(r.n)}` })}>
            <rect x={(i % per) * 23} y={Math.floor(i / per) * 14} width={21} height={12} fill={col(st(r))}><title>{`#${str(r.n)} ${str(r.status)}: ${str(r.question)}`}</title></rect>
          </a>
        ))}
      </svg>
      <div className="keys">
        {keys.map((k) => <a key={k.label} href={k.link} title={k.tip}><i style={{ background: col(k.s.toLowerCase()) }} />{k.label} <b>{k.s}</b></a>)}
      </div>
    </Tile>
  );
}

/* ---------- 4 · live */
function LiveTile({ live, dags }: { live: DocResult<Envelope>; dags: DocResult<Envelope> }) {
  const l = ok(live), g = ok(dags);
  if (!l && !g) return null;
  const node = l && isObj(l.node) ? l.node : null;
  const adm = node && isObj(node.admission) ? node.admission : null;
  const act = activeDags(g).slice(0, 5);
  const mem = l ? rows(l.watchdog).map((s) => { const v = num(s.memory_available); return v === null ? null : v / 1024 ** 3; }) : [];
  return (
    <Tile title="Running now" result={live} link={href('live')} linkText="ops"
      big={<>{l ? rows(l.leases).length : '—'}<span className="vs"> jobs</span></>}
      ci={<>{adm ? (adm.stopped ? 'admission STOPPED' : 'peer admitting') : ''}{l?.peer_read_at ? ` · read ${new Date(timeOf(l.peer_read_at) || 0).toTimeString().slice(0, 5)}` : ''}</>}
      tone={adm?.stopped ? 'bad' : undefined}
      tip="Active run DAGs (done / total nodes; ETA only where a track note states one) and peer memory available over the watchdog window.">
      <div className="dagbars">
        {act.map((d) => {
          const c = isObj(d.counts) ? d.counts : {};
          const total = Object.values(c).reduce<number>((a, v) => a + (num(v) || 0), 0) || 1;
          const done = num(c.completed) || 0, run = num(c.running) || 0, fail = num(c.failed) || 0;
          const eta = arr(d.eta_statements).map((e) => str(isObj(e) ? e.text ?? e.eta : e)).join('; ');
          return (
            <a key={str(d.dag)} href={href('live', { ws: str(d.track) })} title={`${str(d.dag)}: ${Object.entries(c).map(([k, v]) => `${k} ${v}`).join(', ')}${eta ? ` · ETA ${eta}` : ' · no ETA recorded'}`}>
              <span>{str(d.dag)}</span>
              <span className="bar"><i style={{ width: `${(done / total) * 100}%`, background: 'var(--good)' }} /><i style={{ width: `${(run / total) * 100}%`, background: 'var(--accent)' }} /><i style={{ width: `${(fail / total) * 100}%`, background: 'var(--critical)' }} /></span>
              <b>{done}/{total}{eta ? ` · ${eta}` : ''}</b>
            </a>
          );
        })}
        {!act.length && <span className="muted">no active DAG</span>}
      </div>
      <div className="memline" title="peer memory available (GB), last watchdog samples"><span>mem avail</span><Spark values={mem} width={220} height={18} /><b>{fmtNum(mem[mem.length - 1])} GB</b></div>
    </Tile>
  );
}
