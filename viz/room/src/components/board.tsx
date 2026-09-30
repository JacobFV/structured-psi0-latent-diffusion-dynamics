/** Mode badge (live / stale / snapshot / fixture / no data) and figure caption. */
import type { ReactNode } from 'react';
import type { DocResult } from '../lib/api';
import { ageSeconds, str } from '../lib/format';

/** live / stale / snapshot / fixture / no data: shown on every panel. */
export function ModeBadge({ result }: { result: DocResult<unknown> | null | undefined }) {
  if (!result || result.status === 'loading') return <span className="mbadge nodata">…</span>;
  if (result.status === 'missing') return <span className="mbadge nodata" title={`expected ${result.expected}`}>NO DATA</span>;
  if (result.status === 'error') return <span className="mbadge stale" title={result.message}>ERROR</span>;
  const d = result.data as { generated_at?: string; stale?: boolean } | undefined;
  const age = ageSeconds(d?.generated_at);
  if (result.mode === 'fixture') return <span className="mbadge fixture" title="synthetic fixture data, not a result">FIXTURE</span>;
  if (result.mode === 'snapshot') return <span className="mbadge snapshot" title={`snapshot generated ${str(d?.generated_at)}`}>SNAPSHOT</span>;
  if (result.mode === 'stale' || d?.stale === true) return <span className="mbadge stale" title={result.exportError || 'stale'}>STALE</span>;
  return <span className="mbadge live" title={`exported ${age !== null ? `${Math.round(age)} s ago` : '—'}`}>LIVE</span>;
}

/** Figure caption (owner: every figure says what its axes/rows/columns/colour encode and the unit/denominator). */
export function Cap({ children }: { children: ReactNode }) {
  return <p className="fig-cap">{children}</p>;
}
