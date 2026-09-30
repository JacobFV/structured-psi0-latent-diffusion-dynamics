/** Evaluations → Relation factors: registry by operator × form, candidate catalog by section and wave. */
import { Cap, ModeBadge } from '../components/board';
import { Gate } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { fmtNum, rows, str } from '../lib/format';

export default function FactorsView() {
  const { result } = useDoc<Envelope>('factors');
  return (
    <Gate result={result} what="relation factors (/api/factors)">
      {(d) => {
        const fs = rows(d.factors), cat = rows(d.catalog);
        const ops = [...new Set(fs.map((x) => str(x.op)))].sort(), forms = [...new Set(fs.map((x) => str(x.form)))].sort();
        const sections = [...new Set(cat.map((c) => str(c.section)))];
        const waves = ['W1', 'W2', 'P', 'X', 'M'];
        const wcol: Record<string, string> = { W1: 'var(--good)', W2: 'var(--accent)', P: 'var(--axis)', X: 'var(--surface-3)', M: 'var(--muted)' };
        return (
          <div className="ev-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))' }}>
            <section className="ev-panel">
              <header>Registry · operator × form<ModeBadge result={result} /><span className="meta">{fs.length} factors</span></header>
              <div className="pm" style={{ gridTemplateColumns: `90px repeat(${forms.length}, 1fr)` }}>
                <span />{forms.map((f) => <span key={f} className="pm-task">{f}</span>)}
                {ops.map((o) => [
                  <span key={`${o}l`} className="pm-pol">{o}</span>,
                  ...forms.map((f) => { const xs = fs.filter((x) => str(x.op) === o && str(x.form) === f); return <span key={`${o}${f}`} className={`pm-c ${xs.length ? 'ok' : 'e'}`} title={xs.map((x) => `${str(x.name)} (${str(x.status)}, field ${str(x.field)} ${str(x.field_prov)})`).join('\n')}>{xs.length || ''}</span>; }),
                ])}
              </div>
              <Cap>rows: operator · cols: form · cell: number of registered factors (hover: names, field, provenance)</Cap>
            </section>
            <section className="ev-panel">
              <header>Candidate catalog · status by section<span className="meta">research/relations_catalog.md</span></header>
              {sections.map((sec) => {
                const xs = cat.filter((c) => str(c.section) === sec);
                return (
                  <div key={sec} className="fr" style={{ gridTemplateColumns: '220px 1fr' }}>
                    <span className="fr-l" title={sec}>{sec.slice(0, 34)}</span>
                    <span className="fr-sq">{xs.map((c, i) => <i key={i} className="fsq" style={{ background: wcol[str(c.status)] || 'var(--axis)', borderColor: 'var(--axis)' }} title={`${str(c.family)} · ${str(c.status_text)}\n${str(c.candidates)}\nenvs: ${str(c.envs)}`} />)}</span>
                  </div>
                );
              })}
              <div className="legend">{waves.map((w) => <span key={w}><i className="sw" style={{ background: wcol[w] }} />{w} {cat.filter((c) => c.status === w).length}</span>)}</div>
              <Cap>rows: catalog section · square: candidate relation family, colour = wave (W1 first, W2 next, P planned, X out of scope, M meta)</Cap>
            </section>
            <section className="ev-panel">
              <header>Runs · schedules<span className="meta">{rows(d.runs).length} run{rows(d.runs).length === 1 ? '' : 's'} with a factor set</span></header>
              {rows(d.schedules).length ? rows(d.schedules).map((s) => (
                <div key={str(s.run)} className="small" style={{ marginBottom: 8 }}>
                  <div>{str(s.run)} · step {s.step == null ? '—' : str(s.step)} · {rows(s.records).length} intervals · {rows(s.steer).length} steers{s.has_competence === false ? <span className="muted"> · no competence signals yet (R11 placeholder)</span> : null}</div>
                  {rows(s.competence_by_depth).length > 0 && (
                    <table className="bt">
                      <thead><tr><th>factor</th><th>depth</th><th>share</th><th>competence</th><th>plateau</th><th>interference</th></tr></thead>
                      <tbody>
                        {rows(s.competence_by_depth).map((c) => (
                          <tr key={str(c.factor)}>
                            <td className="mono">{str(c.factor)}</td>
                            <td className="n">{c.depth == null ? '—' : str(c.depth)}</td>
                            <td className="n">{fmtNum(c.share)}</td>
                            <td className="n">{c.competence == null ? <span className="muted">—</span> : fmtNum(c.competence)}</td>
                            <td className="n">{fmtNum(c.plateau)}</td>
                            <td className="n">{fmtNum(c.interference)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )) : <p className="side-note">no run has written schedule.jsonl yet</p>}
              <Cap>competence by composition depth: the latest decision line per run (docs/relations.md 5.5) · depth = the factor's current level k · competence / plateau / interference: null until R11 fills in the scheduler's real signals (today's foundation placeholder always leaves them empty) — never fabricated</Cap>
            </section>
          </div>
        );
      }}
    </Gate>
  );
}
