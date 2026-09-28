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
        sub="Reproduction of Ψ₀ with the released checkpoints and the step-2 runs (psi1z), the psi1z P-decisions and their crosswalk to rrp D-decisions. Reads ~/work/psi1z and the local copies of its run summaries."
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
  const runs = rows(pick(d, 'runs'));
  const pd = rows(pick(d, 'p_decisions'));
  const cw = rows(pick(d, 'crosswalk'));
  const w10 = rows(pick(d, 'rrp_w10_decisions'));
  const missing = arr(d.missing).map(str);
  return (
    <div className="stack">
      {arr(d.notes).length > 0 && <ul className="small muted" style={{ margin: 0 }}>{arr(d.notes).map((n, i) => <li key={i}>{str(n)}</li>)}</ul>}
      {missing.length > 0 && <div className="state"><h3>Missing inputs</h3><ul className="small">{missing.map((m) => <li key={m}>{m}</li>)}</ul></div>}
      {<Card title="Reproduction and step-2 runs" hint="released-checkpoint and psi1z runs; counts as recorded"><RunsTable runs={runs} /></Card>}
      {(
        <div className="grid g2">
          <Card title="psi1z P-decisions">
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
        <Card title="D ↔ P crosswalk" hint="docs/related_repos.md and the psi1z decision table">
          <DataTable rows={cw.map((r) => ({ rrp: arr(r.rrp).join(' '), psi1z: arr(r.psi1z).join(' '), topic: r.topic, source: `${str(r.source_file)}:${str(r.line)}` }))} />
          <div style={{ height: 12 }} />
          <TablesBrowser tables={rows(pick(d, 'p_to_d_table'))} param="ptable" />
        </Card>
      )}
      {(
        <>
          <Card title="psi1z research/notes.md">{str(d.notes_markdown) ? <Markdown source={str(d.notes_markdown)} /> : <p className="muted">Not in the document.</p>}</Card>
          <Card title="Tables in the notes"><TablesBrowser tables={rows(pick(d, 'notes_tables'))} param="ntable" /></Card>
        </>
      )}
      {<Card title="psi1z README.md">{str(d.readme_markdown) ? <Markdown source={str(d.readme_markdown)} /> : <p className="muted">Not in the document.</p>}</Card>}
    </div>
  );
}
