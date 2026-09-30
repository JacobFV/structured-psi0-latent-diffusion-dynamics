import { useMemo, useState, type ReactNode } from 'react';
import type { DocResult, Envelope } from '../lib/api';
import { ago, fmtNum, fmtTime, num, shortSha, sortNatural, str, timeOf, type Row } from '../lib/format';
import { sourceStyle, stateTone } from '../lib/labels';

export function Card({ title, hint, right, children, flush, className }: {
  title?: ReactNode; hint?: ReactNode; right?: ReactNode; children: ReactNode; flush?: boolean; className?: string;
}) {
  return (
    <section className={`card ${className || ''}`}>
      {(title || right) && (
        <header>
          {title && <h2>{title}</h2>}
          {hint && <span className="hint">{hint}</span>}
          {right && <div className="right">{right}</div>}
        </header>
      )}
      <div className={`body ${flush ? 'flush' : ''}`}>{children}</div>
    </section>
  );
}

export function PageHead({ title, sub, right }: { title: string; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {sub && <div className="sub">{sub}</div>}
      </div>
      {right && <div className="right">{right}</div>}
    </div>
  );
}

export function SourceBadge({ label, title }: { label: unknown; title?: string }) {
  const s = sourceStyle(label);
  return (
    <span className={`badge src t-${s.tone}`} title={title || s.detail || s.label}>
      {s.label}
    </span>
  );
}
export function Status({ state, children }: { state: unknown; children?: ReactNode }) {
  const tone = stateTone(state);
  return (
    <span className={`status ${tone}`}>
      <i aria-hidden />
      {children ?? str(state) ?? '—'}
    </span>
  );
}
export function Did({ id }: { id: unknown }) {
  const s = str(id);
  if (!s) return null;
  return <span className="did" title={`decision ${s}`}>{s}</span>;
}
export function Caveat({ text }: { text: unknown }) {
  const s = str(text);
  if (!s) return null;
  return <span className="badge caveat" title={s}>⚠ {s.length > 60 ? `${s.slice(0, 57)}…` : s}</span>;
}
export function Loading({ what }: { what?: string }) {
  return (
    <div aria-busy="true" aria-live="polite">
      <div className="skeleton" />
      <p className="muted small">Loading {what || 'data'}…</p>
    </div>
  );
}
export function NoData({ expected, detail, what }: { expected: string; detail?: string; what?: string }) {
  return (
    <div className="state" role="status">
      <h3>No data{what ? ` for ${what}` : ''}</h3>
      <div>Expected source: <code>{expected}</code></div>
      {detail && <pre className="small muted" style={{ whiteSpace: 'pre-wrap', margin: '6px 0 0' }}>{detail}</pre>}
      <div className="small muted" style={{ marginTop: 6 }}>Nothing is shown in place of missing data.</div>
    </div>
  );
}
export function ErrorState({ message }: { message: string }) {
  return (
    <div className="state error" role="alert">
      <h3>Could not load</h3>
      <code>{message}</code>
    </div>
  );
}

/** Renders loading / missing / error explicitly; children only get real data. */
export function Gate<T>({ result, what, children }: { result: DocResult<T>; what?: string; children: (data: T) => ReactNode }) {
  if (result.status === 'loading') return <Loading what={what} />;
  if (result.status === 'missing') return <NoData expected={result.expected} detail={result.detail} what={what} />;
  if (result.status === 'error') return <ErrorState message={result.message} />;
  return <>{children(result.data)}</>;
}

export function ModeBanner({ result, reload, busy }: { result: DocResult<Envelope>; reload?: () => void; busy?: boolean }) {
  if (result.status !== 'ok') return null;
  const d = result.data;
  const gen = str(d.generated_at);
  const stale = result.mode === 'stale' || d.stale === true;
  const cls = result.mode === 'fixture' ? 'fixture' : result.mode === 'snapshot' ? 'snapshot' : stale ? 'stale' : '';
  return (
    <div className={`mode-banner ${cls}`} role="status">
      {result.mode === 'fixture' && (
        <>
          <span className="badge src t-fixture">FIXTURE</span>
          <span>
            Synthetic example data from <code>{result.sourceFile}</code>, shaped like the contract. <b>Not a result.</b> It is shown only
            because the exporter module does not exist yet (or <code>fixture=1</code> is in the URL).
          </span>
        </>
      )}
      {result.mode === 'snapshot' && (
        <>
          <span className="badge">SNAPSHOT</span>
          <span>Snapshot as of <b>{gen ? fmtTime(timeOf(gen)) : 'unknown time'}</b> ({ago(gen)}). Static build: nothing here is live.</span>
        </>
      )}
      {(result.mode === 'live' || result.mode === 'stale') && (
        <>
          <Status state={stale ? 'stale' : 'ok'}>{stale ? 'STALE' : 'live export'}</Status>
          <span className="ink2">
            generated {gen ? ago(gen) : '—'} · git <code>{shortSha(d.git_sha)}</code>
            {result.fileMtime && <> · file written {ago(result.fileMtime)}</>}
          </span>
          {result.exportError && (
            <details>
              <summary className="small">latest export failed; showing the previous file</summary>
              <pre className="small" style={{ whiteSpace: 'pre-wrap' }}>{result.exportError}</pre>
            </details>
          )}
        </>
      )}
      {reload && (
        <button className="ghost" onClick={reload} disabled={busy} style={{ marginLeft: 'auto' }} title="Re-fetch (the API still honours its cache)">
          {busy ? 'refreshing…' : 'refresh'}
        </button>
      )}
    </div>
  );
}

export function Provenance({ result }: { result: DocResult<Envelope> }) {
  if (result.status !== 'ok') return null;
  const d = result.data;
  const sources = Array.isArray(d.sources) ? d.sources.map(str) : [];
  return (
    <footer className="prov">
      <div>
        schema <code>{str(d.schema) || '—'}</code> · generated <span className="num">{fmtTime(timeOf(d.generated_at))}</span> · git{' '}
        <code>{shortSha(d.git_sha)}</code> · served from <code>{result.sourceFile || '—'}</code>
      </div>
      {sources.length > 0 && (
        <details>
          <summary>{sources.length} source file{sources.length === 1 ? '' : 's'}</summary>
          <ul>{sources.slice(0, 400).map((s) => <li key={s}><code>{s}</code></li>)}</ul>
        </details>
      )}
    </footer>
  );
}

export function Select({ label, value, options, onChange, all = true, width }: {
  label: string; value: string; options: string[]; onChange: (v: string) => void; all?: boolean; width?: number;
}) {
  return (
    <label>
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)} style={width ? { width } : undefined}>
        {all && <option value="">all ({options.length})</option>}
        {options.map((o) => <option key={o} value={o}>{o || '(none)'}</option>)}
      </select>
    </label>
  );
}
/** Column rendering hints shared by generic tables. */
export function cellValue(key: string, v: unknown): ReactNode {
  if (v === null || v === undefined || v === '') return <span className="muted">—</span>;
  const k = key.toLowerCase();
  if (k === 'source_label' || k === 'source') return <SourceBadge label={v} />;
  if (k === 'state' || k === 'status' || k === 'verdict' || k === 'gate' || k === 'level') return <Status state={v} />;
  if (k === 'decision' || k === 'id' && /^[DP]-\d+/.test(str(v))) return <Did id={v} />;
  if (k === 'caveat') return <Caveat text={v} />;
  if (k === 'interim') return v ? 'interim' : <span className="muted">no</span>;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'number') return fmtNum(v);
  if (Array.isArray(v) && v.length === 2 && v.every((x) => typeof x === 'number')) return `[${fmtNum(v[0])}, ${fmtNum(v[1])}]`;
  if (typeof v === 'object') return <code className="small">{JSON.stringify(v).slice(0, 200)}</code>;
  return str(v);
}

export function DataTable({ rows, columns, max = 500, onRow, selected, tall, empty }: {
  rows: Row[]; columns?: string[]; max?: number; onRow?: (r: Row) => void; selected?: (r: Row) => boolean; tall?: boolean; empty?: string;
}) {
  const cols = useMemo(() => {
    if (columns) return columns;
    const seen: string[] = [];
    for (const r of rows.slice(0, 200)) for (const k of Object.keys(r)) if (!seen.includes(k)) seen.push(k);
    return seen.slice(0, 18);
  }, [rows, columns]);
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const sorted = useMemo(() => {
    if (!sort) return rows;
    return [...rows].sort((a, b) => {
      const x = a[sort.key], y = b[sort.key];
      const nx = num(x), ny = num(y);
      if (nx !== null && ny !== null) return (nx - ny) * sort.dir;
      return sortNatural(str(x), str(y)) * sort.dir;
    });
  }, [rows, sort]);
  if (!rows.length) return <p className="muted small">{empty || 'No rows.'}</p>;
  const numeric = new Set(cols.filter((c) => rows.slice(0, 50).every((r) => r[c] == null || typeof r[c] === 'number')));
  return (
    <div className={`table-wrap ${tall ? 'tall' : ''}`}>
      <table className="t">
        <thead>
          <tr>
            {cols.map((c) => (
              <th
                key={c}
                className={`sortable ${numeric.has(c) ? 'n' : ''}`}
                onClick={() => setSort((s) => (s?.key === c ? (s.dir === 1 ? { key: c, dir: -1 } : null) : { key: c, dir: 1 }))}
                aria-sort={sort?.key === c ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}
              >
                {c}{sort?.key === c ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.slice(0, max).map((r, i) => (
            <tr key={i} onClick={onRow ? () => onRow(r) : undefined} className={selected?.(r) ? 'sel' : ''} style={onRow ? { cursor: 'pointer' } : undefined}>
              {cols.map((c) => (
                <td key={c} className={numeric.has(c) ? 'n' : typeof r[c] === 'string' && str(r[c]).length > 60 ? 'wrap' : ''}>{cellValue(c, r[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > max && <p className="muted small">Showing {max} of {rows.length} rows.</p>}
    </div>
  );
}

