import { Card, DataTable, Did, Gate, ModeBanner, PageHead, Provenance } from '../components/ui';
import Sections from '../components/Sections';
import Markdown from '../components/Markdown';
import { useDoc, type Envelope } from '../lib/api';
import { pick, str } from '../lib/format';
import { href } from '../lib/url';

export default function Psi0() {
  const { result, reload, busy } = useDoc<Envelope>('psi0');
  return (
    <>
      <PageHead
        title="Ψ₀ line"
        sub="Reproduction of Ψ₀, the step-2 results and the psi1z line: its P-decisions and their crosswalk to our D-decisions (reads ~/work/psi1z and its local results copies)."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="Ψ₀ line (/api/psi0)">
        {(d) => (
          <Sections d={d} specs={[
            { keys: ['summary', 'status'], title: 'Summary' },
            { keys: ['reproduction', 'repro', 'reproduction_table'], title: 'Reproduction table' },
            { keys: ['step2', 'step_2', 'step2_results'], title: 'Step-2 results' },
            { keys: ['p_decisions', 'decisions', 'psi1z_decisions'], title: 'psi1z P-decisions', render: (rs) => (
              <div className="dlist">
                {rs.map((r, i) => (
                  <details key={i} className="d" style={{ display: 'block' }}>
                    <summary className="row"><span className="date">{str(pick(r, 'date'))}</span><Did id={pick(r, 'id')} /><span>{str(pick(r, 'title'))}</span></summary>
                    {pick(r, 'body', 'markdown') ? <Markdown source={str(pick(r, 'body', 'markdown'))} /> : null}
                  </details>
                ))}
              </div>
            ) },
            { keys: ['crosswalk', 'd_p_crosswalk'], title: 'D ↔ P crosswalk', render: (rs) => (
              <DataTable rows={rs} />
            ) },
          ]} />
        )}
      </Gate>
      <Card title="Related" hint="knowledge view">
        <a href={href('knowledge', { tab: 'crosswalk' })}>decision crosswalk in Knowledge →</a>
      </Card>
      <Provenance result={result} />
    </>
  );
}
