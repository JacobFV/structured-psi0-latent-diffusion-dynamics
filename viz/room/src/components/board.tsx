/** Dense "terminal" building blocks: seam panels, KPI tiles, sparklines, inline bars, delta glyphs, mode badges. */
import type { ReactNode } from 'react';
import type { DocResult } from '../lib/api';
import { ageSeconds, fmtNum, str } from '../lib/format';

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

export function Panel({ title, href, meta, result, span, rows2, children, className = '' }: {
  title: string; href?: string; meta?: ReactNode; result?: DocResult<unknown> | null; span?: 2 | 3 | 4; rows2?: boolean; children: ReactNode; className?: string;
}) {
  return (
    <section className={`board-panel ${span ? `span-${span}` : ''} ${rows2 ? 'row-2' : ''} ${className}`}>
      <header>
        {href ? <a className="title" href={href} title="open the full view">{title} ›</a> : <span className="title">{title}</span>}
        {result !== undefined && <ModeBadge result={result} />}
        {meta !== undefined && <span className="meta">{meta}</span>}
      </header>
      <div className="board-body">
        {result && result.status === 'missing' ? <p className="board-note">no data · expected <code>{result.expected}</code></p>
          : result && result.status === 'error' ? <p className="board-note">error: {result.message}</p>
            : result && result.status === 'loading' ? <p className="board-note">loading…</p> : children}
      </div>
    </section>
  );
}

export type KpiTone = 'good' | 'bad' | 'warn' | 'empty' | '';
export function Kpi({ label, value, sub, tone = '', href, title, spark }: {
  label: string; value: ReactNode; sub?: ReactNode; tone?: KpiTone; href?: string; title?: string; spark?: (number | null)[];
}) {
  const body = (
    <>
      <span>{label}</span>
      <b>{value}</b>
      {spark && spark.filter((v) => v !== null).length > 1 ? <Spark values={spark} width={100} height={14} /> : null}
      {sub !== undefined && <small>{sub}</small>}
    </>
  );
  return href ? <a className={`kpi ${tone}`} href={href} title={title}>{body}</a> : <div className={`kpi ${tone}`} title={title}>{body}</div>;
}

/** Plain sparkline of recorded values; nulls break the line; nothing is smoothed. */
export function Spark({ values, width = 90, height = 18, color, band }: {
  values: (number | null | undefined)[]; width?: number; height?: number; color?: string; band?: [number, number];
}) {
  const finite = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (finite.length < 2) return <svg className="spark empty" width={width} height={height} aria-label="not enough points" />;
  let lo = Math.min(...finite), hi = Math.max(...finite);
  if (band) { lo = Math.min(lo, band[0]); hi = Math.max(hi, band[1]); }
  const range = hi - lo || 1;
  const x = (i: number) => (i / (values.length - 1)) * width;
  const y = (v: number) => height - 1.5 - ((v - lo) / range) * (height - 3);
  let d = '', pen = false;
  values.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) { pen = false; return; }
    d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  });
  const last = [...values].reverse().find((v): v is number => v != null && Number.isFinite(v))!;
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${finite.length} points, last ${fmtNum(last)}`}>
      <path d={d} style={color ? { stroke: color } : undefined} />
      <circle cx={x(values.lastIndexOf(last))} cy={y(last)} r={1.6} fill={color || 'var(--ink)'} />
    </svg>
  );
}

/** Inline bar: value against a full scale, optional peak tick and threshold (e.g. memory.high) tick. */
export function IBar({ value, max, peak, mark, color, title, width = 80 }: {
  value: number | null; max: number | null; peak?: number | null; mark?: number | null; color?: string; title?: string; width?: number;
}) {
  if (!max) return <span className="ibar" style={{ width }} title="no scale recorded" />;
  const f = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
  return (
    <div className="ibar" style={{ width }} title={title}>
      {value !== null && <i style={{ width: f(value), background: color }} />}
      {mark != null && <u style={{ left: f(mark) }} />}
      {peak != null && <b style={{ left: f(peak) }} />}
    </div>
  );
}
/** Rate in [0,1] with a CI whisker. */
export function RateBar({ rate, lo, hi, width = 70 }: { rate: number | null; lo?: number | null; hi?: number | null; width?: number }) {
  return (
    <div className="ibar whisker" style={{ width }} title={`${fmtNum(rate)} [${fmtNum(lo)}, ${fmtNum(hi)}]`}>
      {rate !== null && <i style={{ width: `${rate * 100}%` }} />}
      {lo != null && hi != null && <s style={{ left: `${lo * 100}%`, width: `${Math.max(0.5, (hi - lo) * 100)}%` }} />}
    </div>
  );
}
/** ▲ / ▼ with a CVD-safe diverging pair (blue up, red down). `good` flips the meaning when lower is better. */
export function Delta({ v, digits = 2, pct, lowerIsBetter }: { v: number | null; digits?: number; pct?: boolean; lowerIsBetter?: boolean }) {
  if (v === null || !Number.isFinite(v)) return <span className="flat">—</span>;
  const up = v > 0;
  const cls = v === 0 ? 'flat' : (up !== !!lowerIsBetter) ? 'up' : 'down';
  const txt = pct ? `${(Math.abs(v) * 100).toFixed(0)}` : Math.abs(v).toFixed(digits);
  return <span className={`${cls} num`}>{v === 0 ? '■' : up ? '▲' : '▼'}{txt}</span>;
}
export function Warn({ text }: { text: unknown }) {
  const s = str(text);
  return s ? <span className="warn-glyph" title={s} aria-label={`caveat: ${s}`}>⚠</span> : null;
}
