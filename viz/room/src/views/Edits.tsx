import { useShowData } from './Lens';
import { SideGroup, SidebarControls } from '../components/shell';
import { Forest, type ForestRow } from '../components/charts';
import { Card, DataTable, Did, Gate, ModeBanner, PageHead, Provenance, Select } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { arr, num, pick, rows, sortNatural, str, uniq, type Row } from '../lib/format';
import { seriesColor } from '../lib/labels';
import { useUrlState } from '../lib/url';

function isControl(r: Row) {
  if (r.role !== undefined && r.role !== null) return str(r.role) === 'control';
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
  return num(pick(r, 'permutation_p', 'p_two_sided', 'p_one_sided', 'perm_p', 'p', 'p_value'));
}

export default function Edits() {
  const { result, reload, busy } = useDoc<Envelope>('edits');
  return (
    <>
      <PageHead
        title="Causal edits"
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="edits (/api/edits)">
        {(d) => <EditsBody all={rows(pick(d, 'rows', 'edits', 'effects'))} />}
      </Gate>
      <Provenance result={result} />
    </>
  );
}

function EditsBody({ all: raw }: { all: Row[] }) {
  const showData = useShowData();
  const all = raw.filter((r) => str(r.kind) !== 'permutation');
  const perms = raw.filter((r) => str(r.kind) === 'permutation');
  const opt = (k: string, rs = all) => uniq(rs.map((r) => str(r[k]))).filter(Boolean).sort(sortNatural);
  const [body, setBody] = useUrlState('body', opt('body')[0] || '');
  const [variant, setVariant] = useUrlState('variant', '');
  const [edit, setEdit] = useUrlState('edit', '');
  // default lens: the body's most common metric (usually forward travel), so one forest per variant, not dozens
  const bodyRows0 = all.filter((r) => !body || str(r.body) === body);
  const counts = new Map<string, number>();
  bodyRows0.forEach((r) => counts.set(str(r.metric), (counts.get(str(r.metric)) || 0) + 1));
  const topMetric = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? '';
  const [metric, setMetric] = useUrlState('metric', topMetric);
  const [seed, setSeed] = useUrlState('seed', '');
  const [cop, setCop] = useUrlState('cop', '');
  const [alpha, setAlpha] = useUrlState('alpha', '0.05');
  const [cap, setCap] = useUrlState('cap', '60');
  const shown = all.filter((r) => (!body || str(r.body) === body) && (!variant || str(r.variant) === variant) && (!edit || str(r.edit) === edit)
    && (!metric || str(r.metric) === metric) && (!seed || str(r.seed) === seed) && (!cop || str(r.context_or_packet) === cop));
  const groups = uniq(shown.map((r) => `${str(r.body) || '∅'} · ${str(r.variant) || '∅'} · ${str(r.metric) || '∅'}`)).sort(sortNatural);
  const edits = opt('edit');
  const a = Number(alpha) || 0.05;
  const limit = Number(cap) || 60;
  if (!raw.length) return <p className="muted">The edits document has no rows.</p>;
  const nonCtl = shown.filter((r) => !isControl(r));
  const sig = nonCtl.filter((r) => (pOf(r) ?? 1) < a).length;
  const excl = nonCtl.filter((r) => { const [lo, hi] = ciOf(r); return lo !== null && hi !== null && (lo > 0 || hi < 0); }).length;
  const ctl = shown.filter((r) => isControl(r));
  const ctlExcl = ctl.filter((r) => { const [lo, hi] = ciOf(r); return lo !== null && hi !== null && (lo > 0 || hi < 0); }).length;
  const bodyRows = all.filter((r) => !body || str(r.body) === body);
  return (
    <div className="stack">
      <SidebarControls>
        <SideGroup title="Filter effects">      <div>
        <div className="filters" style={{ marginBottom: 0 }}>
          <Select label="body" value={body} options={opt('body')} onChange={setBody} />
          <Select label="variant" value={variant} options={opt('variant', bodyRows)} onChange={setVariant} />
          <Select label="metric" value={metric} options={opt('metric', bodyRows)} onChange={setMetric} />
          <Select label="edit" value={edit} options={opt('edit', bodyRows)} onChange={setEdit} width={180} />
          <Select label="seed" value={seed} options={opt('seed', bodyRows)} onChange={setSeed} />
          <Select label="context / packet" value={cop} options={opt('context_or_packet', bodyRows)} onChange={setCop} />
          <label>α (display only)
            <select value={alpha} onChange={(e) => setAlpha(e.target.value)}>{['0.05', '0.01', '0.001'].map((x) => <option key={x}>{x}</option>)}</select>
          </label>
          <label>rows per plot
            <select value={cap} onChange={(e) => setCap(e.target.value)}>{['30', '60', '150', '400'].map((x) => <option key={x}>{x}</option>)}</select>
          </label>
        </div>
        <div className="legend" style={{ marginTop: 8 }}>
          <span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--s1)" /></svg> edit (colour = edit)</span>
          <span><svg width="12" height="12"><rect x="1.5" y="1.5" width="9" height="9" fill="var(--surface)" stroke="var(--muted)" strokeWidth="2" /></svg> control (inactive / random / irrelevant edit)</span>
          <span className="muted">bars: recorded 95% interval · effect = mean paired difference (edited − unedited)</span>
        </div>
      </div>
        </SideGroup>
      </SidebarControls>
      {showData && <div className="grid g4">
        <div className="card stat"><div className="k">effects shown</div><div className="v num">{shown.length}</div><div className="s">{groups.length} body · variant · metric groups</div></div>
        <div className="card stat"><div className="k">edits whose CI excludes 0</div><div className="v num">{excl}</div><div className="s">of {nonCtl.length} edit / difference rows</div></div>
        <div className="card stat"><div className="k">controls whose CI excludes 0</div><div className="v num">{ctlExcl}</div><div className="s">of {ctl.length} control rows (should stay near 0)</div></div>
        <div className="card stat"><div className="k">rows with p &lt; {a}</div><div className="v num">{sig}</div><div className="s">uncorrected; p is recorded only where a test was run</div></div>
      </div>}
      {groups.map((g) => {
        const rs = shown.filter((r) => `${str(r.body) || '∅'} · ${str(r.variant) || '∅'} · ${str(r.metric) || '∅'}` === g)
          .sort((x, y) => sortNatural(str(x.edit), str(y.edit)) || sortNatural(str(x.seed), str(y.seed)) || Number(isControl(x)) - Number(isControl(y)));
        const frs: ForestRow[] = rs.slice(0, limit).map((r, i) => {
          const [lo, hi] = ciOf(r);
          return {
            key: `${str(r.edit)} · seed ${str(r.seed)} · ${i}`,
            label: `${str(r.edit)}${str(r.role) === 'difference' ? ' (edit − control)' : ''} · s${str(r.seed) || '?'}${r.context_or_packet ? ` · ${str(r.context_or_packet)}` : ''}`,
            effect: num(pick(r, 'effect', 'delta', 'diff')), lo, hi, control: isControl(r), p: pOf(r), n: num(pick(r, 'n_pairs', 'n')),
            tone: str(r.role) === 'difference' ? 'var(--s7)' : seriesColor(str(r.edit), edits),
            note: [r.decision ? `decision ${str(r.decision)}` : '', `${str(r.location)}:${str(r.source_file)}`, str(r.caveat)].filter(Boolean).join(' · '),
          };
        });
        const decisions = uniq(rs.flatMap((r) => [str(r.decision), ...arr(r.decisions).map(str)]).filter(Boolean)).slice(0, 8);
        return (
          <Card key={g} title={g} hint={`${rs.length} effects${rs.length > limit ? ` (first ${limit} drawn; raise “rows per plot” or filter)` : ''}`} right={decisions.map((d) => <Did key={d} id={d} />)}>
            <Forest rows={frs} xLabel={`effect on ${g.split(' · ')[2]}`} />
          </Card>
        );
      })}
      {showData && <>
      <Card title="Permutation tests" hint={`${perms.length} recorded exact tests (label permutations)`}>
        <DataTable rows={perms.map((r) => ({ body: r.body, variant: r.variant, edit: r.edit, metric: r.metric, direction: r.direction, p_one_sided: r.p_one_sided, p_two_sided: r.p_two_sided, n_perm: r.n_perm, every_seed_ordered: r.every_seed_ordered, effect: r.effect, decision: r.decision, source_file: r.source_file }))} tall empty="No permutation rows in the document." />
      </Card>
      <Card title="All effect rows (filtered)" hint={`${shown.length}`}>
        <DataTable rows={shown.map((r) => ({ body: r.body, variant: r.variant, seed: r.seed, edit: r.edit, role: r.role, metric: r.metric, effect: r.effect, ci: r.ci, n_pairs: r.n_pairs, p_two_sided: r.p_two_sided, context_or_packet: r.context_or_packet, decision: r.decision, location: r.location, source_file: r.source_file }))} tall />
      </Card>
      </>}
    </div>
  );
}
