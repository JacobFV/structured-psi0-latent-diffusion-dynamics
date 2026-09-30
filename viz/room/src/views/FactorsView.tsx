/** Evaluations → Relation factors: registry, presets, fields (provenance), candidate catalog, runs with factor sets, schedules. */
import { Cap, ModeBadge } from '../components/board';
import { DataTable, Gate } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { arr, isObj, rows, str } from '../lib/format';

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
              <Cap>rows: operator · cols: form · cell: number of registered factors with that operator and form (hover: names, status, field and its provenance) · hatched: none</Cap>
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
              <Cap>rows: catalog section · square: one candidate relation family, colour = status (W1 first wave: envs can label now; W2 needs a new scene part or label; P planned; X out of scope; M meta) · hover: candidates, decomposition status, envs</Cap>
            </section>
            <section className="ev-panel wide">
              <header>Registered factors<span className="meta">field provenance: public / estimated / privileged</span></header>
              <DataTable tall rows={fs.map((x) => ({ name: x.name, v: x.version, status: x.status, field: x.field, prov: x.field_prov, op: x.op, form: x.form, sources: arr(x.sources).join(', '), gates: arr(x.gates).join(', '), readout: x.readout, doc: x.doc }))} />
              <Cap>one row per registered factor (rrp factors list): field × operator × form, declared sources and gates; prov = provenance of the field it reads</Cap>
            </section>
            <section className="ev-panel">
              <header>Presets</header>
              {Object.entries(isObj(d.presets) ? d.presets : {}).map(([n, items]) => <div key={n} className="small"><b>{n}</b> ({arr(items).length}): {arr(items).map((i) => (typeof i === 'string' ? i : JSON.stringify(i))).join(', ')}</div>)}
            </section>
            <section className="ev-panel">
              <header>Runs with a factor set · schedules</header>
              {rows(d.runs).length ? <DataTable rows={rows(d.runs).map((r) => ({ run: r.run, factors: JSON.stringify(r.factors), file: r.file }))} /> : <p className="side-note">no run records versions["factors"] yet</p>}
              {rows(d.schedules).length ? rows(d.schedules).map((s) => <div key={str(s.run)} className="small">{str(s.run)} · {rows(s.records).length} intervals · {rows(s.steer).length} steers</div>) : <p className="side-note">no run has written schedule.jsonl yet</p>}
            </section>
          </div>
        );
      }}
    </Gate>
  );
}
