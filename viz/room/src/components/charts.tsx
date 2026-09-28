import { useState, type ReactNode } from 'react';
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { fmtNum } from '../lib/format';

export type Series = { key: string; label: string; color: string; band?: [string, string]; dashed?: boolean; dots?: boolean };

/**
 * Single-axis line chart (dataviz rule: never two y-scales). Missing values break lines; nothing is interpolated.
 * Optional CI bands per series (range areas), shaded x-intervals (e.g. "clip active") and reference lines.
 */
export function Lines({
  data, xKey, series, height = 220, logY, shades = [], refs = [], syncId, yDomain, xLabel, yLabel, xType = 'number', title, right, xFormat,
}: {
  data: Record<string, unknown>[]; xKey: string; series: Series[]; height?: number; logY?: boolean;
  shades?: { x1: number; x2: number; label?: string; color?: string }[];
  refs?: { x?: number; y?: number; label?: string; color?: string }[];
  syncId?: string; yDomain?: [number | 'auto' | 'dataMin' | 'dataMax', number | 'auto' | 'dataMin' | 'dataMax'];
  xLabel?: string; yLabel?: string; xType?: 'number' | 'category'; title?: ReactNode; right?: ReactNode;
  xFormat?: (v: number) => string;
}) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const safe = logY ? data.map((d) => {
    const o: Record<string, unknown> = { ...d };
    for (const s of series) if (typeof o[s.key] === 'number' && (o[s.key] as number) <= 0) o[s.key] = null;
    return o;
  }) : data;
  return (
    <figure style={{ margin: 0 }}>
      {(title || right) && <div className="chart-title">{title}<span className="spacer" />{right}</div>}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={safe} syncId={syncId} margin={{ top: 6, right: 12, bottom: xLabel ? 16 : 2, left: 4 }}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey={xKey} type={xType} domain={xType === 'number' ? ['dataMin', 'dataMax'] : undefined} tickFormatter={xFormat || ((v) => fmtNum(v))}
            label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -8, className: 'recharts-text' } : undefined}
            allowDuplicatedCategory={false}
          />
          <YAxis
            scale={logY ? 'log' : 'auto'} domain={yDomain || (logY ? ['auto', 'auto'] : ['auto', 'auto'])} allowDataOverflow={!!logY}
            width={54} tickFormatter={(v) => fmtNum(v)}
            label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', className: 'recharts-text' } : undefined}
          />
          <Tooltip
            formatter={(v: unknown, name: unknown) => [Array.isArray(v) ? `[${fmtNum(v[0])}, ${fmtNum(v[1])}]` : fmtNum(v), String(name)]}
            labelFormatter={(l: unknown) => `${xLabel || xKey} ${xFormat ? xFormat(Number(l)) : fmtNum(l)}`}
          />
          {series.length > 1 && (
            <Legend
              onClick={(e: { dataKey?: unknown }) => {
                const k = String(e.dataKey);
                setHidden((h) => { const n = new Set(h); if (n.has(k)) n.delete(k); else n.add(k); return n; });
              }}
              wrapperStyle={{ cursor: 'pointer', fontSize: 12 }}
            />
          )}
          {shades.map((s, i) => (
            <ReferenceArea key={`s${i}`} x1={s.x1} x2={s.x2} fill={s.color || 'var(--serious)'} fillOpacity={0.14} strokeOpacity={0} ifOverflow="hidden" />
          ))}
          {series.filter((s) => s.band).map((s) => (
            <Area
              key={`${s.key}-band`} dataKey={(d: Record<string, unknown>) => {
                const lo = d[s.band![0]], hi = d[s.band![1]];
                return typeof lo === 'number' && typeof hi === 'number' ? [lo, hi] : null;
              }}
              name={`${s.label} 95% CI`} stroke="none" fill={s.color} fillOpacity={0.14} legendType="none" isAnimationActive={false}
              hide={hidden.has(s.key)} connectNulls={false}
            />
          ))}
          {series.map((s) => (
            <Line
              key={s.key} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} dot={s.dots ? { r: 3 } : false}
              strokeDasharray={s.dashed ? '5 4' : undefined} isAnimationActive={false} connectNulls={false} hide={hidden.has(s.key)}
              activeDot={{ r: 4 }}
            />
          ))}
          {refs.map((r, i) => r.x !== undefined ? (
            <ReferenceLine key={`r${i}`} x={r.x} stroke={r.color || 'var(--ink-2)'} strokeDasharray="4 3" label={r.label ? { value: r.label, position: 'top', className: 'recharts-text' } : undefined} />
          ) : (
            <ReferenceLine key={`r${i}`} y={r.y} stroke={r.color || 'var(--ink-2)'} strokeDasharray="4 3" label={r.label ? { value: r.label, position: 'right', className: 'recharts-text' } : undefined} />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </figure>
  );
}

/** Sequential colour for a rate in [0, 1] (one hue, light → dark). */
export function seqColor(v: number) {
  const steps = 8;
  const i = Math.max(0, Math.min(steps - 1, Math.floor(v * steps - 1e-9)));
  return `var(--seq-${i})`;
}
export function seqInk(v: number) {
  return v >= 0.5 ? '#ffffff' : 'var(--ink)';
}
/** Diverging colour for a delta in [-1, 1]: red ← grey → blue. */
export function divColor(d: number) {
  const a = Math.min(1, Math.abs(d) / 0.5);
  const pole = d < 0 ? 'var(--div-neg)' : 'var(--div-pos)';
  return `color-mix(in srgb, ${pole} ${Math.round(a * 100)}%, var(--div-mid))`;
}
export function SeqLegend({ label = 'rate' }: { label?: string }) {
  return (
    <span className="legend">
      {label} 0
      <span className="ramp">{Array.from({ length: 8 }, (_, i) => <i key={i} style={{ background: `var(--seq-${i})` }} />)}</span>
      1
    </span>
  );
}
export function DivLegend() {
  return (
    <span className="legend">
      −0.5
      <span className="ramp">{[-0.5, -0.35, -0.2, -0.07, 0.07, 0.2, 0.35, 0.5].map((d) => <i key={d} style={{ background: divColor(d) }} />)}</span>
      +0.5 (B − A)
    </span>
  );
}

export type ForestRow = { key: string; label: ReactNode; group?: string; effect: number | null; lo: number | null; hi: number | null; control?: boolean; p?: number | null; n?: number | null; tone?: string; note?: string };

/** Forest plot: effect ± 95% CI per row, zero line, controls drawn hollow and grey. */
export function Forest({ rows, xLabel = 'effect', width = 640 }: { rows: ForestRow[]; xLabel?: string; width?: number }) {
  const [tip, setTip] = useState<{ x: number; y: number; r: ForestRow } | null>(null);
  const vals = rows.flatMap((r) => [r.lo, r.hi, r.effect]).filter((v): v is number => v !== null && Number.isFinite(v));
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  if (!(hi > lo)) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.06;
  lo -= pad; hi += pad;
  const left = 250, right = 120, rowH = 22, top = 8, h = top + rows.length * rowH + 30;
  const x = (v: number) => left + ((v - lo) / (hi - lo)) * (width - left - right);
  const ticks = niceTicks(lo, hi, 6);
  return (
    <>
      <svg className="plot" width={width} height={h} viewBox={`0 0 ${width} ${h}`} role="img" aria-label={`forest plot of ${rows.length} effects`}>
        <g className="grid">{ticks.map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={top} y2={h - 26} />)}</g>
        <line x1={x(0)} x2={x(0)} y1={top} y2={h - 26} stroke="var(--ink-2)" strokeWidth={1} />
        {ticks.map((t) => <text key={t} x={x(t)} y={h - 12} textAnchor="middle">{fmtNum(t)}</text>)}
        <text className="lab" x={(left + width - right) / 2} y={h} textAnchor="middle">{xLabel}</text>
        <text className="lab" x={width - right + 8} y={top + 10} fontWeight={600}>effect [CI] · p</text>
        {rows.map((r, i) => {
          const y = top + i * rowH + rowH / 2 + 8;
          const color = r.control ? 'var(--muted)' : r.tone || 'var(--s1)';
          return (
            <g key={r.key} className="forest-row"
              onPointerMove={(e) => setTip({ x: e.clientX, y: e.clientY, r })} onPointerLeave={() => setTip(null)}>
              <rect className="hit" x={0} y={y - rowH / 2} width={width} height={rowH} fill="transparent" />
              <text className="lab" x={left - 10} y={y + 4} textAnchor="end">{typeof r.label === 'string' ? r.label.slice(0, 44) : r.label}</text>
              {r.lo !== null && r.hi !== null && (
                <line x1={x(r.lo)} x2={x(r.hi)} y1={y} y2={y} stroke={color} strokeWidth={2} strokeLinecap="round" />
              )}
              {r.effect !== null && (
                r.control
                  ? <rect x={x(r.effect) - 4.5} y={y - 4.5} width={9} height={9} fill="var(--surface)" stroke={color} strokeWidth={2} />
                  : <circle cx={x(r.effect)} cy={y} r={5} fill={color} stroke="var(--surface)" strokeWidth={2} />
              )}
              <text x={width - right + 8} y={y + 4}>
                {r.effect === null ? '—' : fmtNum(r.effect)}{r.lo !== null && r.hi !== null ? ` [${fmtNum(r.lo)}, ${fmtNum(r.hi)}]` : ''}
                {r.p != null ? ` · ${r.p < 0.001 ? '<0.001' : fmtNum(r.p)}` : ''}
              </text>
            </g>
          );
        })}
      </svg>
      {tip && (
        <div className="tip" style={{ left: tip.x + 12, top: tip.y + 12 }}>
          <div><b>{tip.r.key}</b>{tip.r.control ? ' · control' : ''}</div>
          <div className="num">effect {fmtNum(tip.r.effect)} {tip.r.lo !== null ? `[${fmtNum(tip.r.lo)}, ${fmtNum(tip.r.hi)}]` : ''}</div>
          {tip.r.n != null && <div className="num">n pairs {fmtNum(tip.r.n)}</div>}
          {tip.r.p != null && <div className="num">permutation p {fmtNum(tip.r.p)}</div>}
          {tip.r.note && <div className="muted">{tip.r.note}</div>}
        </div>
      )}
    </>
  );
}

export function niceTicks(lo: number, hi: number, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const step = 10 ** Math.floor(Math.log10(raw));
  const nice = [1, 2, 2.5, 5, 10].map((m) => m * step).find((s) => s >= raw) || raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / nice) * nice; v <= hi + 1e-12; v += nice) out.push(Number(v.toPrecision(10)));
  return out;
}
