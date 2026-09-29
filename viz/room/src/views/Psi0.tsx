import { useShowData } from './Lens';
import { RateBar } from '../components/board';
import Markdown from '../components/Markdown';
import { Card, DataTable, Did, Gate, ModeBanner, PageHead, Provenance, TablesBrowser } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { arr, fmtNum, num, pick, rows, str, wilson, type Row } from '../lib/format';
import { href } from '../lib/url';

export default function Psi0() {
  const { result, reload, busy } = useDoc<Envelope>('psi0');
  return (
    <>
      <PageHead
        title="Ψ₀ line"
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="Ψ₀ line (/api/psi0)">{(d) => <Psi0Body d={d} />}</Gate>
      <Provenance result={result} />
    </>
  );
}

function RunsTable({ runs }: { runs: Row[] }) {
  if (!runs.length) return <p className="muted small">No run summaries in the document.</p>;
  return (
    <div className="table-wrap tall">
      <table className="t">
        <thead><tr><th>run</th><th>task</th><th>level</th><th className="n">k/n</th><th>rate · 95% CI</th><th>source</th><th>notes</th></tr></thead>
        <tbody>
          {runs.map((r, i) => {
            const k = num(r.k), n = num(r.n);
            const rate = num(r.rate) ?? (k !== null && n ? k / n : null);
            const ci = Array.isArray(r.ci) ? [num(r.ci[0]), num(r.ci[1])] : k !== null && n ? wilson(k, n) : [null, null];
            return (
              <tr key={i}>
                <td className="mono small">{str(r.run)}</td>
                <td>{str(r.task)}</td>
                <td>{str(r.level) || '—'}</td>
                <td className="n">{k ?? '?'}/{n ?? '?'}</td>
                <td style={{ minWidth: 170 }}>
                  <div className="cell-bar"><i style={{ width: `${(rate ?? 0) * 100}%` }} />{ci[0] !== null && <b style={{ left: `${(ci[0] as number) * 100}%`, background: 'var(--ink-2)' }} />}{ci[1] !== null && <b style={{ left: `${(ci[1] as number) * 100}%`, background: 'var(--ink-2)' }} />}</div>
                  <span className="small num">{rate === null ? '—' : fmtNum(rate)} [{fmtNum(ci[0])}, {fmtNum(ci[1])}]</span>
                </td>
                <td className="small">{str(r.source_label || r.source) || <span className="muted">not recorded</span>}{r.copied_from ? <div className="muted mono">{str(r.copied_from)}</div> : null}</td>
                <td className="small">
                  {r.interim ? <span className="badge interim" title={str(r.interim_reason)}>interim</span> : null}
                  {r.infra_failure_run ? <span className="badge caveat">infra failure</span> : null}
                  {r.caveat ? <span className="badge caveat">{str(r.caveat)}</span> : null}
                  <div className="muted">{str(r.interim_reason)}{r.count_from ? ` · counted from ${str(r.count_from)}` : ''}</div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Psi0Body({ d }: { d: Envelope }) {
  const showData = useShowData();
  const runs = rows(pick(d, 'runs'));
  const pd = rows(pick(d, 'p_decisions'));
  const cw = rows(pick(d, 'crosswalk'));
  const w10 = rows(pick(d, 'rrp_w10_decisions'));
  const missing = arr(d.missing).map(str);
  if (!showData) return <Psi0Bars runs={runs} />;
  return (
    <div className="stack">
      {arr(d.notes).length > 0 && <ul className="small muted" style={{ margin: 0 }}>{arr(d.notes).map((n, i) => <li key={i}>{str(n)}</li>)}</ul>}
      {missing.length > 0 && <div className="state"><h3>Missing inputs</h3><ul className="small">{missing.map((m) => <li key={m}>{m}</li>)}</ul></div>}
      {<Card title="Reproduction and step-2 runs" hint="released-checkpoint and our Ψ₀ runs; counts as recorded"><RunsTable runs={runs} /></Card>}
      {(
        <div className="grid g2">
          <Card title="P-decisions (appendix P, research/decisions.md)">
            <div className="dlist">
              {pd.slice().reverse().map((r, i) => (
                <details key={i} style={{ borderBottom: '1px solid var(--border)', padding: '6px 4px' }}>
                  <summary className="row" style={{ cursor: 'pointer' }}><span className="date">{str(r.date)}</span><Did id={r.id} /><span>{str(r.title)}</span>
                    {arr(r.refs_d).map((x) => <a key={str(x)} className="did" href={href('knowledge', { tab: 'decisions', d: str(x) })}>{str(x)}</a>)}</summary>
                  <Markdown source={str(pick(r, 'body', 'markdown'))} />
                </details>
              ))}
            </div>
          </Card>
          <Card title="rrp W10 decisions" hint="D-entries that reference the Ψ₀ line">
            <DataTable rows={w10.map((r) => ({ id: r.id, date: r.date, title: r.title, refs_p: arr(r.refs_p).join(' ') }))} tall />
          </Card>
        </div>
      )}
      {(
        <Card title="D ↔ P crosswalk" hint="rrp D-numbers cited in each P-decision heading">
          <DataTable rows={cw.map((r) => ({ rrp: arr(r.rrp).join(' '), psi1z: arr(r.psi1z).join(' '), topic: r.topic, source: `${str(r.source_file)}:${str(r.line)}` }))} />
          <div style={{ height: 12 }} />
          <TablesBrowser tables={rows(pick(d, 'p_to_d_table'))} param="ptable" />
        </Card>
      )}
      {(
        <>
          <Card title="research/tracks/psi0.md">{str(d.notes_markdown) ? <Markdown source={str(d.notes_markdown)} /> : <p className="muted">Not in the document.</p>}</Card>
          <Card title="Tables in the notes"><TablesBrowser tables={rows(pick(d, 'notes_tables'))} param="ntable" /></Card>
        </>
      )}
    </div>
  );
}

function Psi0Bars({ runs }: { runs: Row[] }) {
  const rs = runs.filter((r) => num(r.n)).sort((a, b) => Number(str(b.run).startsWith('step2')) - Number(str(a.run).startsWith('step2')));
  return (
    <section className="ev-panel" style={{ borderLeft: 'var(--seam)', borderTop: 'var(--seam)' }}>
      <header>Ψ₀ runs · success with 95% CI<span className="meta">released checkpoints (rel) and step 2 (s2) · ◐ interim</span></header>
      <div style={{ display: 'grid', gridTemplateColumns: '200px 240px 60px 20px', gap: '3px 10px', alignItems: 'center', fontSize: 11 }}>
        {rs.map((r) => {
          const k = num(r.k)!, n = num(r.n)!;
          const ci = Array.isArray(r.ci) ? [num(r.ci[0]), num(r.ci[1])] : wilson(k, n);
          return (
            <FragRow key={str(r.run)} cells={[
              <span key="a" title={`${str(r.task)} · ${str(r.interim_reason)}`} className="mono">{str(r.run).replace(/^psi0rel_/, 'rel ').replace(/^step2_/, 's2 ')}</span>,
              <RateBar key="b" rate={k / n} lo={ci[0]} hi={ci[1]} width={230} />,
              <span key="c" className="mono">{k}/{n}</span>,
              <span key="d" className="warn-glyph" title={str(r.interim_reason)}>{r.interim ? '◐' : ''}</span>,
            ]} />
          );
        })}
      </div>
      <p className="fig-cap">rows: Ψ₀ run (rel = released checkpoint reproduction, s2 = step 2) · bar: success rate, line: 95% CI · k/n episodes · ◐ interim (hover: reason)</p>
    </section>
  );
}
function FragRow({ cells }: { cells: React.ReactNode[] }) { return <>{cells}</>; }
