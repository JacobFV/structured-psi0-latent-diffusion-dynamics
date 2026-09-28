import { useEffect, useMemo, useState } from 'react';
import Markdown from '../components/Markdown';
import { Card, DataTable, Did, Gate, Loading, ModeBanner, NoData, ErrorState, PageHead, Provenance } from '../components/ui';
import { SideGroup, SidebarControls } from '../components/shell';
import { fetchMarkdown, getMeta, useDoc, type DocResult, type Envelope } from '../lib/api';
import { ago, arr, decisionIds, pick, rows, sortNatural, str, uniq, type Row } from '../lib/format';
import { href, useUrlState } from '../lib/url';

type Tab = 'decisions' | 'crosswalk' | 'roadmap' | 'backlog' | 'strategy' | 'status' | 'docs';

export default function Knowledge() {
  const { result, reload, busy } = useDoc<Envelope>('knowledge');
  const psi0 = useDoc<Envelope>('psi0');
  const [tab, setTab] = useUrlState('tab', 'decisions');
  return (
    <>
      <PageHead title="Knowledge" sub="Decisions timeline with the D ↔ P crosswalk, roadmap, backlog, strategy workstreams, STATUS and a reader for the allowlisted markdown documents." />
      <SidebarControls>
        <SideGroup title="Section">
          <div className="side-list">
            {([['decisions', 'Decisions timeline'], ['crosswalk', 'D ↔ P crosswalk'], ['roadmap', 'Roadmap'], ['backlog', 'Backlog'], ['strategy', 'Strategy workstreams'], ['status', 'STATUS'], ['docs', 'Docs reader']] as [Tab, string][]).map(([id, label]) => (
              <button key={id} aria-pressed={tab === id} onClick={() => setTab(id)}>{label}</button>
            ))}
          </div>
        </SideGroup>
      </SidebarControls>
      {tab === 'docs' ? <DocsReader knowledge={result} /> : (
        <>
          <ModeBanner result={result} reload={reload} busy={busy} />
          <Gate result={result} what="knowledge (/api/knowledge)">
            {(d) => {
              if (tab === 'decisions') return <Decisions d={d} psi0={psi0.result} />;
              if (tab === 'crosswalk') return <Crosswalk d={d} psi0={psi0.result} />;
              if (tab === 'roadmap') return <ListTab rs={rows(pick(d, 'roadmap', 'roadmap_items')).map(flatCw)} what="roadmap items" />;
              if (tab === 'backlog') return <ListTab rs={rows(pick(d, 'backlog', 'backlog_items')).map(flatCw)} what="backlog items" />;
              if (tab === 'strategy') return <Strategy rs={rows(pick(d, 'workstreams', 'strategy', 'strategy_workstreams'))} />;
              const status = str(pick(d, 'status_markdown', 'status', 'STATUS'));
              const evidence = str(pick(d, 'evidence_matrix_markdown'));
              return (
                <div className="stack">
                  {rows(pick(d, 'status_table')).length > 0 && <Card title="Workstream status table"><DataTable rows={rows(pick(d, 'status_table')).map(flatCw)} /></Card>}
                  {status ? <Card title="STATUS.md"><Markdown source={status} /></Card> : <NoData expected="knowledge.status_markdown (STATUS.md)" />}
                  {evidence && <Card title="Evidence matrix"><Markdown source={evidence} /></Card>}
                </div>
              );
            }}
          </Gate>
          <Provenance result={result} />
        </>
      )}
    </>
  );
}

function crosswalkRows(d: Row, psi0: DocResult<Envelope>): Row[] {
  const k = rows(pick(d, 'crosswalk', 'd_p_crosswalk'));
  if (k.length) return k;
  return psi0.status === 'ok' ? rows(pick(psi0.data, 'crosswalk', 'd_p_crosswalk')) : [];
}

function Decisions({ d, psi0 }: { d: Row; psi0: DocResult<Envelope> }) {
  const all = rows(pick(d, 'decisions', 'decision_entries'));
  const [sel, setSel] = useUrlState('d', '');
  const [q, setQ] = useUrlState('q', '');
  const cw = crosswalkRows(d, psi0);
  const shown = all.filter((r) => !q || `${str(r.id)} ${str(r.title)} ${str(r.body)}`.toLowerCase().includes(q.toLowerCase()));
  const byDate = useMemo(() => {
    const m = new Map<string, number>();
    for (const r of all) { const k = str(r.date).slice(0, 10); m.set(k, (m.get(k) || 0) + 1); }
    return Array.from(m.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [all]);
  const cur = all.find((r) => str(r.id) === sel);
  const maxDay = Math.max(1, ...byDate.map((x) => x[1]));
  if (!all.length) return <NoData expected="knowledge.decisions (parsed from research/decisions.md)" />;
  const linked = cur ? cw.filter((r) => Object.values(r).some((v) => (Array.isArray(v) ? v.map(str) : str(v).split(/[\s,;]+/)).includes(str(cur.id)))) : [];
  return (
    <div className="theatre" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.2fr)' }}>
      <div className="stack">
        <Card title="Decisions per day" hint={`${all.length} decisions`}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 60 }} role="img" aria-label="decisions per day">
            {byDate.map(([day, n]) => (
              <div key={day} title={`${day}: ${n}`} onClick={() => setQ(day)} style={{ flex: 1, height: `${(n / maxDay) * 100}%`, minHeight: 2, background: 'var(--s1)', borderRadius: '2px 2px 0 0', cursor: 'pointer' }} />
            ))}
          </div>
          <div className="row small muted"><span>{byDate[0]?.[0]}</span><span className="spacer" /><span>{byDate[byDate.length - 1]?.[0]}</span></div>
        </Card>
        <Card title="Timeline" right={<input type="search" placeholder="search id, title, text, date" value={q} onChange={(e) => setQ(e.target.value)} />}>
          <div className="dlist" style={{ maxHeight: 640, overflow: 'auto' }}>
            {shown.slice().reverse().map((r) => (
              <div key={str(r.id)} className={`d ${str(r.id) === sel ? 'sel' : ''}`} onClick={() => setSel(str(r.id))}>
                <span className="date">{str(r.date)}</span>
                <span><Did id={r.id} /></span>
                <span>{str(r.title)}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <div className="stack" style={{ alignContent: 'start' }}>
        {cur ? (
          <Card title={<><Did id={cur.id} /> {str(cur.title)}</>} hint={str(cur.date)}>
            <Markdown source={str(pick(cur, 'body', 'markdown', 'text'))} />
            {decisionIds(str(cur.body)).filter((x) => x !== str(cur.id)).length > 0 && (
              <p className="small">Mentions: {decisionIds(str(cur.body)).filter((x) => x !== str(cur.id)).map((x) => <a key={x} className="did" href={href('knowledge', { tab: 'decisions', d: x })} style={{ marginRight: 4 }}>{x}</a>)}</p>
            )}
            {arr(cur.refs_p).length > 0 && <p className="small">psi1z: {arr(cur.refs_p).map((x) => <Did key={str(x)} id={x} />)}</p>}
            {arr(cur.workstreams).length > 0 && <p className="small">workstreams: {arr(cur.workstreams).map(str).join(', ')}</p>}
            {arr(cur.paths).length > 0 && <p className="small">paths: {arr(cur.paths).map((x) => str(x).endsWith('.md') ? <a key={str(x)} href={href('knowledge', { tab: 'docs', doc: str(x) })} style={{ marginRight: 6 }}>{str(x)}</a> : <code key={str(x)} style={{ marginRight: 6 }}>{str(x)}</code>)}</p>}
            <p className="small muted">{str(cur.source_file)}:{str(cur.line)}{cur.line_end ? `–${str(cur.line_end)}` : ''}</p>
            {linked.length > 0 && <><h3 style={{ fontSize: 12 }}>Crosswalk</h3><DataTable rows={linked.map(flatCw)} /></>}
          </Card>
        ) : <div className="state">Select a decision.</div>}
      </div>
    </div>
  );
}

function Crosswalk({ d, psi0 }: { d: Row; psi0: DocResult<Envelope> }) {
  const cw = crosswalkRows(d, psi0);
  if (!cw.length) return <NoData expected="knowledge.crosswalk or psi0.crosswalk (D ↔ P)" />;
  return <Card title="D ↔ P crosswalk" hint={`${cw.length} links between rrp decisions (D) and psi1z decisions (P)`}><DataTable rows={cw.map(flatCw)} tall /></Card>;
}

function flatCw(r: Row): Row {
  return Object.fromEntries(Object.entries(r).map(([k, v]) => [k, Array.isArray(v) ? v.map(str).join(' ') : v]));
}

function ListTab({ rs, what }: { rs: Row[]; what: string }) {
  const [q, setQ] = useUrlState('q', '');
  const qq = q.startsWith('#') ? q.slice(1) : '';
  const shown = rs.filter((r) => (qq ? str(r.n) === qq : !q || JSON.stringify(r).toLowerCase().includes(q.toLowerCase())));
  if (!rs.length) return <NoData expected={`knowledge: ${what}`} />;
  const statuses = uniq(rs.map((r) => str(pick(r, 'status', 'state')))).filter(Boolean).sort(sortNatural);
  return (
    <Card title={what} hint={`${shown.length} of ${rs.length}${statuses.length ? ` · ${statuses.map((s) => `${s} ${rs.filter((r) => str(pick(r, 'status', 'state')) === s).length}`).join(' · ')}` : ''}`}
      right={<input type="search" placeholder="filter" value={q} onChange={(e) => setQ(e.target.value)} />}>
      <DataTable rows={shown} tall max={1000} />
    </Card>
  );
}

function Strategy({ rs }: { rs: Row[] }) {
  if (!rs.length) return <NoData expected="knowledge.workstreams (docs/strategy.md)" />;
  return (
    <div className="stack">
      {rs.map((w, i) => (
        <Card key={i} title={<>{str(w.id)} · {str(w.workstream)}</>} hint={[w.owner ? `owner ${str(w.owner)}` : '', w.depends_on ? `depends on ${str(w.depends_on)}` : ''].filter(Boolean).join(' · ')}
          right={arr(w.decisions).map((x) => <Did key={str(x)} id={x} />)}>
          <p style={{ marginTop: 0 }}><b>Status:</b> {str(w.status)}</p>
          {w.detail_markdown ? <details><summary className="small">scope and detail</summary><Markdown source={str(w.detail_markdown)} /></details> : null}
        </Card>
      ))}
    </div>
  );
}

function DocsReader({ knowledge }: { knowledge: DocResult<Envelope> }) {
  const [doc, setDoc] = useUrlState('doc', 'STATUS.md');
  const [list, setList] = useState<string[] | null>(null);
  const [res, setRes] = useState<DocResult<{ markdown: string; modified?: string; sources?: string[] }>>({ status: 'loading' });
  const [q, setQ] = useState('');
  useEffect(() => {
    let live = true;
    (async () => {
      const fromKnowledge = knowledge.status === 'ok' ? arr(pick(knowledge.data, 'docs', 'doc_list')).map((x) => str(typeof x === 'object' ? pick(x, 'path') : x)).filter(Boolean) : [];
      const meta = await getMeta();
      let fromApi: string[] = [];
      if (meta) {
        try { const r = await fetch('/api/doclist'); if (r.ok) fromApi = (await r.json()).docs || []; } catch { /* keep the knowledge list */ }
      }
      if (live) setList(uniq([...fromApi, ...fromKnowledge]).sort(sortNatural));
    })();
    return () => { live = false; };
  }, [knowledge]);
  useEffect(() => {
    let live = true;
    setRes({ status: 'loading' });
    fetchMarkdown(doc).then((r) => live && setRes(r));
    return () => { live = false; };
  }, [doc]);
  const shown = (list || []).filter((p) => !q || p.toLowerCase().includes(q.toLowerCase()));
  return (
    <div>
      <SidebarControls>
      <section className="card">
        <header><h2>Documents</h2><span className="hint">{list ? `${shown.length} of ${list.length}` : '…'}</span></header>
        <div className="body">
          <input type="search" placeholder="filter" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: '100%', marginBottom: 8 }} />
          <div className="picker" style={{ maxHeight: 680 }}>
            {list === null ? <Loading what="document list" /> : shown.map((p) => (
              <button key={p} aria-pressed={p === doc} onClick={() => setDoc(p)}><span className="small" style={{ wordBreak: 'break-all' }}>{p}</span></button>
            ))}
          </div>
        </div>
      </section>
      </SidebarControls>
      <section className="card">
        <header><h2 style={{ wordBreak: 'break-all' }}>{doc}</h2>{res.status === 'ok' && res.data.modified && <span className="hint">modified {ago(res.data.modified)}</span>}</header>
        <div className="body">
          {res.status === 'loading' && <Loading what={doc} />}
          {res.status === 'missing' && <NoData expected={res.expected} detail={res.detail} />}
          {res.status === 'error' && <ErrorState message={res.message} />}
          {res.status === 'ok' && <Markdown source={res.data.markdown} />}
        </div>
      </section>
    </div>
  );
}
