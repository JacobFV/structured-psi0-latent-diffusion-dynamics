import { SideGroup, SidebarControls } from '../components/shell';
import { useMemo, useState } from 'react';
import { divColor, DivLegend, seqColor, seqInk, SeqLegend } from '../components/charts';
import { Card, Caveat, DataTable, Did, Gate, Interim, ModeBanner, PageHead, Provenance, Seg, Select, SourceBadge, TablesBrowser } from '../components/ui';
import { useDoc, type Envelope } from '../lib/api';
import { arr, fmtNum, newcombe, num, pct, pick, rows, sortNatural, str, uniq, wilson, type Row } from '../lib/format';
import { sourceStyle } from '../lib/labels';
import { href, useUrlState, writeParams } from '../lib/url';

type Entry = Row & { __rate: number | null; __lo: number | null; __hi: number | null; __k: number | null; __n: number | null };

const BASE_DIMS = ['family', 'task', 'body', 'body_or_hint', 'route', 'variant', 'seed', 'condition', 'metric', 'source_kind', 'location', 'ci_method'];

function normalise(r: Row): Entry {
  const k = num(r.k), n = num(r.n);
  let rate = num(r.rate);
  if (rate === null && k !== null && n) rate = k / n;
  let lo = num(r.ci_lo), hi = num(r.ci_hi);
  if ((lo === null || hi === null) && Array.isArray(r.ci)) { lo = num(r.ci[0]); hi = num(r.ci[1]); }
  const body_or_hint = r.body ? str(r.body) : r.body_hint ? `${str(r.body_hint)} (hint)` : null;
  return { ...r, body_or_hint, source_kind: r.source_label ? sourceStyle(r.source_label).kind : null, __rate: rate, __lo: lo, __hi: hi, __k: k, __n: n };
}
function keyOf(r: Row, dims: string[]) {
  return dims.map((d) => str(r[d]) || '∅').join(' · ');
}

export default function Results() {
  const { result, reload, busy } = useDoc<Envelope>('results');
  return (
    <>
      <PageHead
        title="Results matrix"
        sub="Every normalised result row (summary.json, compare tables, gate reports) as a heatmap of rate with its 95% CI. Filter by family, body, route, variant, seed and physics version; delta mode compares two values of one dimension on otherwise-identical rows."
      />
      <ModeBanner result={result} reload={reload} busy={busy} />
      <Gate result={result} what="results (/api/results)">
        {(d) => (
          <>
            <Matrix all={rows(pick(d, 'rows', 'results')).map(normalise)} />
            {arr(d.notes).length > 0 && <ul className="small muted">{arr(d.notes).map((n, i) => <li key={i}>{str(n)}</li>)}</ul>}
            <div style={{ height: 14 }} />
            <Card title="Compare and gate tables" hint="every markdown results table found (compare tables, gate reports), as written">
              <TablesBrowser tables={rows(pick(d, 'tables'))} />
            </Card>
          </>
        )}
      </Gate>
      <Provenance result={result} />
    </>
  );
}

function Matrix({ all }: { all: Entry[] }) {
  const versionDims = useMemo(() => uniq(all.flatMap((r) => Object.keys(r).filter((k) => /_version$/.test(k)))).sort(), [all]);
  const dims = [...BASE_DIMS, ...versionDims];
  const [rowA, setRowA] = useUrlState('r1', 'family');
  const [rowB, setRowB] = useUrlState('r2', 'body_or_hint');
  const [metric, setMetric] = useUrlState('metric', 'success');
  const [agg, setAgg] = useUrlState('agg', 'maxn');
  const [col, setCol] = useUrlState('c', 'route');
  const [mode, setMode] = useUrlState('mode', 'rate');
  const [ddim, setDdim] = useUrlState('dd', versionDims.find((v) => /grasp/.test(v)) || versionDims[0] || 'variant');
  const [da, setDa] = useUrlState('da', '');
  const [db, setDb] = useUrlState('db', '');
  const [interim, setInterim] = useUrlState('interim', 'show');
  const [sel, setSel] = useUrlState('cell', '');
  const [q, setQ] = useUrlState('q', '');
  const filterDims = ['family', 'task', 'body_or_hint', 'route', 'variant', 'seed', 'condition', 'source_kind', 'location', ...versionDims];
  const [filters, setFilters] = useState<Record<string, string>>(() => {
    const q = new URLSearchParams(window.location.hash.split('?')[1] || '');
    return Object.fromEntries(filterDims.map((f) => [f, q.get(`f_${f}`) || '']));
  });
  const setFilter = (f: string, v: string) => {
    setFilters((x) => ({ ...x, [f]: v }));
    writeParams({ [`f_${f}`]: v || undefined });
  };
  const options = (f: string) => uniq(all.map((r) => str(r[f]))).filter((x) => x !== '').sort(sortNatural);

  const ql = q.toLowerCase();
  const filtered = all.filter((r) => (!ql || `${str(r.source_file)} ${str(r.source_label)} ${arr(r.key_path).map(str).join('/')}`.toLowerCase().includes(ql)) && (!metric || str(r.metric) === metric) && filterDims.every((f) => !filters[f] || str(r[f]) === filters[f]) && (interim === 'show' || !r.interim));
  const rowDims = [rowA, rowB].filter((x, i, a) => x && a.indexOf(x) === i);
  const rowKeys = uniq(filtered.map((r) => keyOf(r, rowDims))).sort(sortNatural);
  const colKeys = uniq(filtered.map((r) => keyOf(r, [col]))).sort(sortNatural);

  // delta pairs: A and B differ only in `ddim`
  const ident = dims.filter((x) => x !== ddim && x !== 'source_kind');
  const dvals = options(ddim);
  const A = da || dvals[0] || '', B = db || dvals[1] || '';
  const pairs = useMemo(() => {
    if (mode !== 'delta') return [];
    const bIdx = new Map<string, Entry[]>();
    for (const r of filtered) if (str(r[ddim]) === B) {
      const k = keyOf(r, ident);
      bIdx.set(k, [...(bIdx.get(k) || []), r]);
    }
    const out: { a: Entry; b: Entry; d: number | null; lo: number | null; hi: number | null }[] = [];
    for (const a of filtered) if (str(a[ddim]) === A) {
      for (const b of bIdx.get(keyOf(a, ident)) || []) {
        const d = a.__rate !== null && b.__rate !== null ? b.__rate - a.__rate : null;
        const ci = a.__k !== null && a.__n && b.__k !== null && b.__n ? newcombe(a.__k, a.__n, b.__k, b.__n) : [null, null];
        out.push({ a, b, d, lo: ci[0], hi: ci[1] });
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, filtered.length, ddim, A, B, JSON.stringify(filters), interim, metric]);

  const cells = new Map<string, Entry[]>();
  const pairCells = new Map<string, typeof pairs>();
  if (mode === 'delta') {
    for (const p of pairs) {
      const k = `${keyOf(p.a, rowDims)}|${keyOf(p.a, [col])}`;
      pairCells.set(k, [...(pairCells.get(k) || []), p]);
    }
  } else {
    for (const r of filtered) {
      const k = `${keyOf(r, rowDims)}|${keyOf(r, [col])}`;
      cells.set(k, [...(cells.get(k) || []), r]);
    }
  }
  const shownRows = mode === 'delta' ? rowKeys.filter((rk) => colKeys.some((ck) => pairCells.has(`${rk}|${ck}`))) : rowKeys;
  const shownCols = mode === 'delta' ? colKeys.filter((ck) => rowKeys.some((rk) => pairCells.has(`${rk}|${ck}`))) : colKeys;
  const selEntries = mode === 'delta' ? [] : cells.get(sel) || [];
  const selPairs = mode === 'delta' ? pairCells.get(sel) || [] : [];

  if (!all.length) return <p className="muted">The results document has no rows.</p>;
  return (
    <div className="stack">
      <SidebarControls>
        <SideGroup title="Matrix">      <div>
        <div className="filters" style={{ marginBottom: 6 }}>
          <label>search<input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="file, label, key path" /></label>
          <Seg value={mode as 'rate' | 'delta'} onChange={(v) => setMode(v)} options={[{ id: 'rate', label: 'Rates' }, { id: 'delta', label: 'Delta A → B' }]} label="mode" />
          <Select label="rows" value={rowA} options={dims} onChange={setRowA} all={false} />
          <Select label="then" value={rowB} options={['', ...dims]} onChange={setRowB} all={false} />
          <Select label="columns" value={col} options={dims} onChange={setCol} all={false} />
          <label>metric
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              <option value="">any metric (mixes metrics)</option>
              {options('metric').map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
          {mode !== 'delta' && (
            <label>cell with several rows
              <select value={agg} onChange={(e) => setAgg(e.target.value)}>
                <option value="maxn">show the row with the largest n</option>
                <option value="pool">pool k/n across its rows</option>
              </select>
            </label>
          )}
          <label>interim
            <select value={interim} onChange={(e) => setInterim(e.target.value)}><option value="show">show (striped)</option><option value="hide">hide</option></select>
          </label>
          {mode === 'delta' && (
            <>
              <span className="sep" />
              <Select label="delta dimension" value={ddim} options={dims.filter((x) => x !== 'source_kind')} onChange={(v) => { setDdim(v); setDa(''); setDb(''); }} all={false} />
              <Select label="A (from)" value={A} options={dvals} onChange={setDa} all={false} />
              <Select label="B (to)" value={B} options={dvals} onChange={setDb} all={false} />
            </>
          )}
        </div>
        <div className="filters" style={{ marginBottom: 0 }}>
          {filterDims.map((f) => <Select key={f} label={f} value={filters[f] || ''} options={options(f)} onChange={(v) => setFilter(f, v)} width={130} />)}
        </div>
      </div>
        </SideGroup>
      </SidebarControls>
      <Card
        title={mode === 'delta' ? `Δ ${ddim}: ${A || '?'} → ${B || '?'}` : 'Rate with 95% CI'}
        hint={mode === 'delta'
          ? `${pairs.length} matched pairs (identical on every other dimension). CI: Newcombe hybrid score, computed here from k/n.`
          : `${filtered.length} of ${all.length} rows. ${agg === 'pool' ? 'Cells with several rows pool their k/n (Wilson CI): pooling mixes conditions and runs, read with care.' : 'A cell holding several rows shows the one with the largest n'}; ×k counts the rows. Click a cell for all rows.`}
        right={mode === 'delta' ? <DivLegend /> : <SeqLegend />}
      >
        {shownRows.length === 0 ? <p className="muted">{mode === 'delta' ? 'No matched A/B pairs under these filters.' : 'No rows under these filters.'}</p> : (
          <div className="table-wrap" style={{ maxHeight: 640 }}>
            <table className="heat">
              <thead>
                <tr>
                  <th>{rowDims.join(' · ')} \ {col}</th>
                  {shownCols.map((c) => <th key={c} className="col" title={c}>{c}</th>)}
                </tr>
              </thead>
              <tbody>
                {shownRows.map((rk) => (
                  <tr key={rk}>
                    <th title={rk}>{rk}</th>
                    {shownCols.map((ck) => {
                      const k = `${rk}|${ck}`;
                      if (mode === 'delta') {
                        const ps = pairCells.get(k);
                        if (!ps?.length) return <td key={ck} className="empty" />;
                        const best = ps.reduce((x, y) => (Math.min(y.a.__n || 0, y.b.__n || 0) > Math.min(x.a.__n || 0, x.b.__n || 0) ? y : x));
                        const bg = best.d === null ? 'var(--surface-3)' : divColor(best.d);
                        const sig = best.lo !== null && best.hi !== null && (best.lo > 0 || best.hi < 0);
                        return (
                          <td key={ck} className={sel === k ? 'sel' : ''} style={{ background: bg, color: Math.abs(best.d || 0) > 0.3 ? '#fff' : 'var(--ink)' }}
                            onClick={() => setSel(sel === k ? '' : k)}
                            title={`${rk} × ${ck}\n${A}: ${pct(best.a.__rate)} (${best.a.__k}/${best.a.__n}) → ${B}: ${pct(best.b.__rate)} (${best.b.__k}/${best.b.__n})\nΔ ${fmtNum(best.d)} [${fmtNum(best.lo)}, ${fmtNum(best.hi)}]`}>
                            {ps.length > 1 && <span className="more">×{ps.length}</span>}
                            <div className="r">{best.d === null ? '—' : `${best.d > 0 ? '▲' : best.d < 0 ? '▼' : '■'}${Math.abs(best.d * 100).toFixed(0)}`}{sig ? '*' : ''}</div>
                            <div className="ci">{best.lo !== null ? `[${(best.lo * 100).toFixed(0)}, ${(best.hi! * 100).toFixed(0)}]` : 'no k/n'}</div>
                          </td>
                        );
                      }
                      const es = cells.get(k);
                      if (!es?.length) return <td key={ck} className="empty" />;
                      let best = es.reduce((x, y) => ((y.__n || 0) > (x.__n || 0) ? y : x));
                      if (agg === 'pool' && es.length > 1) {
                        const kk = es.reduce((a, e) => a + (e.__k || 0), 0), nn = es.reduce((a, e) => a + (e.__n || 0), 0);
                        const [plo, phi] = nn ? wilson(kk, nn) : [null, null];
                        best = { ...best, __k: kk, __n: nn, __rate: nn ? kk / nn : null, __lo: plo, __hi: phi, interim: es.some((e) => e.interim), caveat: es.some((e) => e.caveat) ? 'pooled rows include caveats' : null, source_label: `pooled over ${es.length} rows` };
                      }
                      const rate = best.__rate;
                      let lo = best.__lo, hi = best.__hi;
                      if ((lo === null || hi === null) && best.__k !== null && best.__n) [lo, hi] = wilson(best.__k, best.__n);
                      return (
                        <td key={ck} className={`${sel === k ? 'sel' : ''} ${best.interim ? 'interim' : ''}`}
                          style={{ background: rate === null ? 'var(--surface-3)' : seqColor(rate), color: rate === null ? 'var(--ink)' : seqInk(rate) }}
                          onClick={() => setSel(sel === k ? '' : k)}
                          title={`${rk} × ${ck}\n${pct(rate)} (${best.__k ?? '?'}/${best.__n ?? '?'}) CI [${pct(lo)}, ${pct(hi)}]\n${str(best.source_label)}${best.interim ? '\nINTERIM' : ''}${best.caveat ? `\ncaveat: ${str(best.caveat)}` : ''}`}>
                          {es.length > 1 && <span className="more">×{es.length}</span>}
                          <div className="r">{rate === null ? '—' : (rate * 100).toFixed(0)}{best.caveat ? '⚠' : ''}</div>
                          <div className="ci">{lo !== null && hi !== null ? `${(lo * 100).toFixed(0)}–${(hi * 100).toFixed(0)}` : ''}{best.__n !== null ? ` n${best.__n}` : ''}</div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="muted small" style={{ marginBottom: 0 }}>
          Values are percentages. {mode === 'delta' ? '* = the 95% CI excludes 0.' : 'Striped = interim; ⚠ = has a caveat (hover or click).'} Rows come only from the document; empty cells mean no row exists.
        </p>
      </Card>
      {sel && mode !== 'delta' && (
        <Card title={`Cell: ${sel.replace('|', ' × ')}`} hint={`${selEntries.length} row(s)`} right={<button className="ghost" onClick={() => setSel('')}>close</button>}>
          <EntryList entries={selEntries} />
        </Card>
      )}
      {sel && mode === 'delta' && (
        <Card title={`Pairs: ${sel.replace('|', ' × ')}`} hint={`${selPairs.length} pair(s)`} right={<button className="ghost" onClick={() => setSel('')}>close</button>}>
          <DataTable rows={selPairs.map((p) => ({
            seed: p.a.seed, variant: p.a.variant, metric: p.a.metric,
            [`A ${A}`]: `${pct(p.a.__rate)} (${p.a.__k ?? '?'}/${p.a.__n ?? '?'})`, [`B ${B}`]: `${pct(p.b.__rate)} (${p.b.__k ?? '?'}/${p.b.__n ?? '?'})`,
            delta: p.d, ci: p.lo !== null ? [p.lo, p.hi] : null,
            'A source': p.a.source_label, 'B source': p.b.source_label, 'A decision': p.a.decision, 'B decision': p.b.decision,
            caveat: [str(p.a.caveat), str(p.b.caveat)].filter(Boolean).join(' | ') || undefined,
          }))} />
        </Card>
      )}
      <Card title="All filtered rows" hint={`${filtered.length}`}>
        <DataTable tall rows={filtered.map(({ __rate, __lo, __hi, __k, __n, ...r }) => r)}
          columns={['family', 'task', 'body', 'body_hint', 'route', 'variant', 'seed', 'eval_seeds', 'condition', ...versionDims, 'metric', 'k', 'n', 'rate', 'ci_lo', 'ci_hi', 'ci_method', 'source_label', 'decision', 'interim', 'interim_reason', 'caveat', 'location', 'source_file']} />
      </Card>
    </div>
  );
}

function EntryList({ entries }: { entries: Entry[] }) {
  return (
    <div className="stack" style={{ gap: 8 }}>
      {entries.map((e, i) => (
        <div key={i} className="row" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 6 }}>
          <b className="num" style={{ minWidth: 64 }}>{pct(e.__rate, 1)}</b>
          <span className="num muted">{e.__k ?? '?'}/{e.__n ?? '?'} · CI [{pct(e.__lo, 1)}, {pct(e.__hi, 1)}]</span>
          <span>{['family', 'task', 'body_or_hint', 'route', 'variant', 'seed', 'condition', 'metric'].map((k) => str(e[k])).filter(Boolean).join(' · ')}</span>
          {e.eval_seeds ? <span className="small muted">eval seeds {str(e.eval_seeds)}</span> : null}
          <span className="small muted">{str(e.ci_method)}</span>
          <SourceBadge label={e.source_label} />
          <Did id={e.decision} />
          <Interim on={e.interim} />
          <Caveat text={e.caveat} />
          <code className="small muted">{str(e.location)}:{str(e.source_file)}{Array.isArray(e.key_path) ? ` → ${e.key_path.map(str).join('.')}` : ''}</code>
          <a className="small" href={href('theatre', { task: str(e.task), body: str(e.body), route: str(e.route) })}>replays →</a>
        </div>
      ))}
    </div>
  );
}
