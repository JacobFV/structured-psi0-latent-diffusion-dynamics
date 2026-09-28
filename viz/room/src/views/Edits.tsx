import { Forest, type ForestRow } from '../components/charts';
import { Card, DataTable, Did, Gate, ModeBanner, PageHead, Provenance, Select } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { fmtNum, num, pick, rows, sortNatural, str, uniq, type Row } from '../lib/format';
import { seriesColor } from '../lib/labels';
import { useUrlState } from '../lib/url';

function isControl(r: Row) {
  const c = r.control;
  if (c === true) return true;
  if (typeof c === 'string' && /^(true|yes|control)$/i.test(c)) return true;
  const e = str(r.edit).toLowerCase();
  return /control|sham|mirror_inactive|placebo|noop/.test(e);
}
function ciOf(r: Row): [number | null, number | null] {
  const ci = r.ci;
  if (Array.isArray(ci)) return [num(ci[0]), num(ci[1])];
  return [num(pick(r, 'ci_lo', 'lo')), num(pick(r, 'ci_hi', 'hi'))];
}
function pOf(r: Row) {
  return num(pick(r, 'permutation_p', 'perm_p', 'p', 'p_value'));
}

export default function Edits() {
  const { result, reload, busy } = useDoc<Envelope>('edits');
  return (
    <>
      <PageHead
        title="Causal edits"
        sub="Effect of editing the runtime representation (context halt, mirror, z-halt/turn, goal, rebind…) on behaviour, per body, variant and seed, with controls and permutation tests. Probes are diagnostics; these edits and rollouts are the evidence of causal use."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="edits (/api/edits)">
        {(d) => <EditsBody all={rows(pick(d, 'rows', 'edits', 'effects'))} />}
      </Gate>
      <Provenance result={result} />
    </>
  );
}

function EditsBody({ all }: { all: Row[] }) {
  const [body, setBody] = useUrlState('body', '');
  const [variant, setVariant] = useUrlState('variant', '');
  const [edit, setEdit] = useUrlState('edit', '');
  const [alpha, setAlpha] = useUrlState('alpha', '0.05');
  const opt = (k: string) => uniq(all.map((r) => str(r[k]))).filter(Boolean).sort(sortNatural);
  const shown = all.filter((r) => (!body || str(r.body) === body) && (!variant || str(r.variant) === variant) && (!edit || str(r.edit) === edit));
  const groups = uniq(shown.map((r) => `${str(r.body)} · ${str(r.variant)}`)).sort(sortNatural);
  const edits = opt('edit');
  const a = Number(alpha) || 0.05;
  if (!all.length) return <p className="muted">The edits document has no rows.</p>;
  const sig = shown.filter((r) => !isControl(r) && (pOf(r) ?? 1) < a).length;
  const ctlSig = shown.filter((r) => isControl(r) && (pOf(r) ?? 1) < a).length;
  return (
    <div className="stack">
      <div className="filters">
        <Select label="body" value={body} options={opt('body')} onChange={setBody} />
        <Select label="variant" value={variant} options={opt('variant')} onChange={setVariant} />
        <Select label="edit" value={edit} options={edits} onChange={setEdit} />
        <label>α (display only)
          <select value={alpha} onChange={(e) => setAlpha(e.target.value)}>{['0.05', '0.01', '0.001'].map((x) => <option key={x}>{x}</option>)}</select>
        </label>
        <span className="legend">
          <span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--s1)" /></svg> edit</span>
          <span><svg width="12" height="12"><rect x="1.5" y="1.5" width="9" height="9" fill="var(--surface)" stroke="var(--muted)" strokeWidth="2" /></svg> control</span>
          <span className="muted">bars: 95% CI · p: permutation test (as recorded)</span>
        </span>
      </div>
      <div className="grid g3">
        <div className="card stat"><div className="k">effects shown</div><div className="v num">{shown.length}</div><div className="s">{groups.length} body · variant groups</div></div>
        <div className="card stat"><div className="k">edits with p &lt; {a}</div><div className="v num">{sig}</div><div className="s">of {shown.filter((r) => !isControl(r)).length} non-control rows (uncorrected)</div></div>
        <div className="card stat"><div className="k">controls with p &lt; {a}</div><div className="v num">{ctlSig}</div><div className="s">controls should not move behaviour</div></div>
      </div>
      {groups.map((g) => {
        const rs = shown.filter((r) => `${str(r.body)} · ${str(r.variant)}` === g)
          .sort((x, y) => sortNatural(str(x.edit), str(y.edit)) || sortNatural(str(x.seed), str(y.seed)) || Number(isControl(x)) - Number(isControl(y)));
        const frs: ForestRow[] = rs.map((r, i) => {
          const [lo, hi] = ciOf(r);
          return {
            key: `${str(r.edit)} · seed ${str(r.seed)} · ${i}`,
            label: `${str(r.edit)}${isControl(r) ? ' (control)' : ''} · s${str(r.seed)}`,
            effect: num(pick(r, 'effect', 'delta', 'diff')), lo, hi, control: isControl(r), p: pOf(r), n: num(pick(r, 'n_pairs', 'n')),
            tone: seriesColor(str(r.edit), edits),
            note: [str(r.metric), r.decision ? `decision ${str(r.decision)}` : '', typeof r.control === 'number' ? `control effect ${fmtNum(r.control)}` : typeof r.control === 'string' && !isControl(r) ? `control: ${r.control}` : '', str(r.caveat)].filter(Boolean).join(' · '),
          };
        });
        const decisions = uniq(rs.map((r) => str(r.decision)).filter(Boolean));
        return (
          <Card key={g} title={g} hint={`${rs.length} effects`} right={decisions.map((d) => <Did key={d} id={d} />)}>
            <Forest rows={frs} xLabel={`effect (${uniq(rs.map((r) => str(r.metric)).filter(Boolean)).join(', ') || 'as recorded'})`} />
          </Card>
        );
      })}
      <Card title="All effect rows" hint={`${shown.length}`}>
        <DataTable rows={shown} tall />
      </Card>
    </div>
  );
}
