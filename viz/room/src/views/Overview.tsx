import Markdown from '../components/Markdown';
import { Card, Caveat, DataTable, Did, Gate, Interim, ModeBanner, PageHead, Provenance, SourceBadge, Stat, Status } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { arr, decisionIds, fmtNum, isObj, pick, rows, str, text, uniq, type Row } from '../lib/format';
import { href } from '../lib/url';

function refsOf(o: unknown): string[] {
  const direct = pick(o, 'decisions', 'decision', 'decision_refs', 'refs_d');
  const list = Array.isArray(direct) ? direct.map(str) : direct ? [str(direct)] : [];
  return uniq(list.length ? list : decisionIds(text(o)));
}
function Refs({ o, max = 6 }: { o: unknown; max?: number }) {
  const r = refsOf(o);
  return (
    <>
      {r.slice(0, max).map((d) => <a key={d} className="did" href={href('knowledge', { tab: 'decisions', d })} style={{ marginRight: 3 }}>{d}</a>)}
      {r.length > max && <span className="muted small">+{r.length - max}</span>}
    </>
  );
}
function Src({ o }: { o: Row }) {
  const f = str(o.source_file);
  if (!f) return null;
  const line = o.line != null ? `:${str(o.line)}` : '';
  return f.endsWith('.md')
    ? <a className="small muted" href={href('knowledge', { tab: 'docs', doc: f })}>{f}{line}</a>
    : <code className="small muted">{f}{line}</code>;
}

export default function Overview() {
  const { result, reload, busy } = useDoc<Envelope>('overview');
  return (
    <>
      <PageHead
        title="Overview"
        sub="What is established (with its decision), what is open (the roadmap), the caveats that qualify it, key numbers quoted in STATUS and the latest decisions. Parsed from STATUS.md, the roadmap and research/decisions.md."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="overview (/api/overview)">
        {(d) => {
          const claims = rows(pick(d, 'claims', 'established'));
          const open = rows(pick(d, 'open'));
          const caveats = arr(pick(d, 'caveats'));
          const numbers = rows(pick(d, 'key_numbers', 'numbers'));
          const decisions = rows(pick(d, 'latest_decisions', 'decisions'));
          const ws = rows(pick(d, 'workstreams'));
          const summary = isObj(d.results_summary) ? d.results_summary : null;
          const current = str(pick(d, 'current_state_markdown', 'summary'));
          const byStatus = (s: string) => claims.filter((c) => str(c.status || 'established') === s);
          const statuses = uniq(claims.map((c) => str(c.status || 'established')));
          const openRunning = open.filter((o) => /run/i.test(str(o.status))).length;
          const openDone = open.filter((o) => /done/i.test(str(o.status))).length;
          return (
            <div className="stack">
              <div className="grid g4">
                <Stat k="Claims in STATUS" v={<span className="num">{claims.length}</span>} s={statuses.map((s) => `${s} ${byStatus(s).length}`).join(' · ')} />
                <Stat k="Roadmap items" v={<span className="num">{open.length}</span>} s={`${openRunning} running · ${openDone} done`} />
                <Stat k="Result rows catalogued" v={<span className="num">{fmtNum(pick(summary, 'n_rows'))}</span>} s={summary ? `${fmtNum(pick(summary, 'n_files'))} files · ${fmtNum(pick(summary, 'n_interim'))} interim · ${fmtNum(pick(summary, 'n_with_caveat'))} with caveat` : 'no summary'} />
                <Stat k="STATUS updated" v={<span style={{ fontSize: 16 }}>{str(pick(d, 'status_updated')) || '—'}</span>} s={<a href={href('knowledge', { tab: 'status' })}>read STATUS →</a>} />
              </div>
              {current && <Card title="Current state" hint="STATUS.md"><Markdown source={current} /></Card>}
              <div className="grid g2">
                <Card title="Claims" hint={`${claims.length} · by status, each with its decisions`}>
                  <div className="stack" style={{ gap: 10 }}>
                    {statuses.map((s) => (
                      <div key={s}>
                        <div className="row" style={{ marginBottom: 4 }}><Status state={s === 'established' ? 'ok' : s}>{s}</Status><span className="muted small">{byStatus(s).length}</span></div>
                        <ul style={{ margin: 0, paddingLeft: 18, display: 'grid', gap: 8 }}>
                          {byStatus(s).map((c, i) => (
                            <li key={i}>
                              <b style={{ fontWeight: 600 }}>{str(c.title)}</b>{c.title ? ': ' : ''}<span className="ink2">{str(c.text)}</span>{' '}
                              <Refs o={c} /> <Interim on={c.interim} /> <Caveat text={c.caveat} /> {c.source_label ? <SourceBadge label={c.source_label} /> : null}
                              <div><Src o={c} /></div>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                    {!claims.length && <p className="muted small">No claims in the document.</p>}
                  </div>
                </Card>
                <div className="stack">
                  <Card title="Caveats" hint={`${caveats.length} · shown, never hidden`}>
                    <ul style={{ margin: 0, paddingLeft: 18, display: 'grid', gap: 8 }}>
                      {caveats.map((c, i) => <li key={i}>{text(c)} <Refs o={c} />{isObj(c) && <div><Src o={c} /></div>}</li>)}
                    </ul>
                    {!caveats.length && <p className="muted small">No caveats in the document.</p>}
                  </Card>
                  <Card title="Latest decisions" right={<a href={href('knowledge', { tab: 'decisions' })}>all →</a>}>
                    <div className="dlist">
                      {decisions.slice(0, 15).map((r, i) => (
                        <a key={i} className="d" href={href('knowledge', { tab: 'decisions', d: str(r.id) })} style={{ color: 'inherit', textDecoration: 'none' }}>
                          <span className="date">{str(r.date)}</span><span><Did id={r.id} /></span><span>{str(r.title)}</span>
                        </a>
                      ))}
                    </div>
                  </Card>
                </div>
              </div>
              <Card title="Open: roadmap" hint={`${open.length} items from docs/experiments_roadmap.md`} right={<a href={href('knowledge', { tab: 'roadmap' })}>roadmap →</a>}>
                <DataTable tall rows={open.map((o) => ({ '#': o.n, section: o.section, question: o.question, status: o.status, detail: o.status_text, depends_on: o.depends_on, cost: o.cost, decisions: arr(o.decisions).join(' ') }))} />
              </Card>
              <Card title="Key numbers quoted in STATUS" hint="k/n as written, with the claim they support">
                <DataTable tall rows={numbers.map((n) => ({
                  value: str(pick(n, 'text', 'value')), k: n.k, n: n.n, rate: typeof n.k === 'number' && typeof n.n === 'number' && n.n ? Number(((n.k as number) / (n.n as number)).toFixed(3)) : undefined,
                  claim: str(n.claim), context: str(n.context), decisions: arr(n.decisions).join(' '), interim: n.interim, source: `${str(n.source_file)}:${str(n.line)}`,
                }))} />
              </Card>
              <Card title="Workstreams" hint="STATUS table">
                <DataTable rows={ws.map((w) => ({ id: w.id, workstream: w.workstream, state: w.state, status: pick(w, 'status (decisions)', 'status'), where: w.where, decisions: arr(w.decisions).join(' ') }))} />
              </Card>
            </div>
          );
        }}
      </Gate>
      <Provenance result={result} />
    </>
  );
}
