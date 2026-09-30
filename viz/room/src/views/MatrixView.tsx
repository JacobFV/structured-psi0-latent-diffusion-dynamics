/** Evaluations → Policy × env × task: the recorded `rrp matrix` rows with every declared reason, plus the registries. */
import { Cap, ModeBadge } from '../components/board';
import { DataTable, Gate } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { arr, isObj, rows, str } from '../lib/format';
import { useShowData } from './Lens';

export default function MatrixView() {
  const { result } = useDoc<Envelope>('matrix');
  const show = useShowData();
  return (
    <Gate result={result} what="policy × env × task matrix (/api/matrix)">
      {(d) => {
        const rs = rows(d.rows);
        const pols = [...new Set(rs.map((r) => str(r.policy)))];
        const cols = [...new Set(rs.map((r) => `${str(r.env_id)} · ${str(r.task)}`))];
        const reasonCount = new Map<string, number>();
        rs.forEach((r) => arr(r.reasons).forEach((x) => reasonCount.set(str(x), (reasonCount.get(str(x)) || 0) + 1)));
        return (
          <div className="ev-grid" style={{ gridTemplateColumns: '1fr' }}>
            <section className="ev-panel">
              <header>Policy × env × task · negotiation<ModeBadge result={result} /><span className="meta">{arr(d.files).map((f) => str(isObj(f) ? f.file : f)).filter((v, i, a) => a.indexOf(v) === i).join(', ')}</span></header>
              <div className="table-wrap">
                <table className="bt">
                  <thead><tr><th>policy</th><th>source</th>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                  <tbody>
                    {pols.map((p) => (
                      <tr key={p}>
                        <td>{p}</td><td className="muted">{str(rs.find((r) => str(r.policy) === p)?.source)}</td>
                        {cols.map((c) => {
                          const r = rs.find((x) => str(x.policy) === p && `${str(x.env_id)} · ${str(x.task)}` === c);
                          if (!r) return <td key={c} className="muted">—</td>;
                          return <td key={c} style={{ color: r.status === 'accepted' ? 'var(--good)' : 'var(--muted)', whiteSpace: 'normal', minWidth: 120 }} title={arr(r.reasons).map(str).join('\n')}>
                            {r.status === 'accepted' ? '✓ accepted' : `n/a: ${arr(r.reasons).map(str)[0] || ''}${arr(r.reasons).length > 1 ? ` (+${arr(r.reasons).length - 1})` : ''}`}
                          </td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Cap>rows: registered policy and its source · cols: env · task pairs in the recorded matrix · cell: accepted, or n/a with the first declared negotiate() reason (hover: all reasons) · — : pair not in the recorded matrix</Cap>
            </section>
            <section className="ev-panel">
              <header>Decline reasons<span className="meta">{reasonCount.size} distinct</span></header>
              {[...reasonCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20).map(([reason, n]) => (
                <div key={reason} className="barrow" style={{ gridTemplateColumns: 'minmax(0, 3fr) 1fr auto' }}><span title={reason}>{reason}</span><span className="bar"><i style={{ width: `${(n / rs.length) * 100}%`, background: 'var(--axis)' }} /></span><b>{n}</b></div>
              ))}
              <Cap>rows: a declared reason for n/a · bar and number: how many matrix cells cite it (of {rs.length})</Cap>
            </section>
            <section className="ev-panel">
              <header>Registries<span className="meta">read from src (envs/base.py ENVS, policies/base.py POLICIES)</span></header>
              <div className="row small"><b>envs</b> {Object.keys(isObj(d.envs) ? d.envs : {}).join(' · ')}</div>
              <div className="row small"><b>policies</b> {Object.keys(isObj(d.policies) ? d.policies : {}).join(' · ')}</div>
            </section>
            {show && <section className="ev-panel"><header>Rows</header><DataTable rows={rs} tall /></section>}
          </div>
        );
      }}
    </Gate>
  );
}
