/**
 * Multi-axis route radar (after IBM-2's radar-chart.tsx), drawn from /api/radar (rrp-viz/radar/v1). Radii use the declared
 * floor-to-reference normalization (0 = floor, 1 = reference). Missing values are gaps: no point and no polygon edge through
 * that axis. Whiskers show each value's spread. Hover shows value, r, spread and evidence.
 */
import { useState } from 'react';
import { fmtNum } from '../lib/format';

export type RadarValue = { value?: number; r?: number; drawn?: number; missing?: string; missing_r?: string; spread?: { x: number[]; r?: number[]; drawn?: number[]; meaning?: string };
  evidence?: string[]; decisions?: string[]; sha1?: string[]; k?: number; n?: number; source_labels?: string[] };
export type RadarAxis = { id: string; label: string; metric: string; direction: 'min' | 'max'; protocol: string; decision: string;
  floor: { value: number | null; meaning?: string; reference_multiple?: number }; reference: { series?: string; value?: number; resolved_value?: number | null; meaning?: string }; series: Record<string, RadarValue> };
export type RadarDoc = { normalization: { clamp: [number, number]; rule: string }; series: { id: string; label: string }[]; axes: RadarAxis[] };

export const SERIES_COLOR: Record<string, string> = { teacher: 'var(--s6)', bc: 'var(--ink-2)', semfix_v6: 'var(--s1)', nosem_v6: 'var(--s2)', semfix: 'var(--s5)', nosem: 'var(--s8)', latent_jointfix: 'var(--s3)' };

export function radarEdges(present: boolean[]) {
  const n = present.length;
  const out: [number, number][] = [];
  present.forEach((here, i) => { const j = (i + 1) % n; if (n > 1 && here && present[j]) out.push([i, j]); });
  return out;
}

export default function RadarChart({ radar, size = 300, compact = false }: { radar: RadarDoc; size?: number; compact?: boolean }) {
  const [focus, setFocus] = useState<string | null>(null);
  const n = radar.axes.length;
  const max = radar.normalization.clamp[1];
  const c = size / 2, outer = size / 2 - (compact ? 34 : 92);
  const ang = (i: number) => -Math.PI / 2 + (2 * Math.PI * i) / n;
  const pt = (i: number, r: number) => [c + Math.cos(ang(i)) * (r / max) * outer, c + Math.sin(ang(i)) * (r / max) * outer];
  return (
    <figure className={`radar ${compact ? 'compact' : ''}`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="route comparison across declared axes">
        {[0.5, 1, max].map((ring) => (
          <polygon key={ring} points={radar.axes.map((_, i) => pt(i, ring).join(',')).join(' ')} fill="none" stroke={ring === 1 ? 'var(--axis)' : 'var(--grid)'} strokeDasharray={ring === 1 ? undefined : '2 3'} />
        ))}
        {radar.axes.map((a, i) => {
          const [x, y] = pt(i, max);
          const [lx, ly] = pt(i, max + (compact ? 0.16 : 0.22));
          const anyValue = Object.values(a.series).some((v) => v.drawn != null);
          return (
            <g key={a.id}>
              <line x1={c} y1={c} x2={x} y2={y} stroke="var(--grid)" />
              <text x={lx} y={ly} textAnchor={Math.abs(lx - c) < 8 ? 'middle' : lx < c ? 'end' : 'start'} dominantBaseline="middle" fill={anyValue ? 'var(--ink-2)' : 'var(--muted)'}>
                <title>{`${a.label} · ${a.metric} ${a.direction === 'min' ? '↓ lower is better' : '↑'}\n${a.protocol}\nfloor ${a.floor.value} (${a.floor.meaning ?? ''}) · reference ${a.reference.series ?? a.reference.meaning ?? ''} = ${fmtNum(a.reference.resolved_value)}\n${a.decision}`}</title>
                {compact ? a.label.replace(/^(arm|legged) /, '').slice(0, 16) : a.label}
              </text>
            </g>
          );
        })}
        {radar.series.map((s) => {
          const vals = radar.axes.map((a) => a.series[s.id]);
          const present = vals.map((v) => v?.drawn != null);
          const col = SERIES_COLOR[s.id] || 'var(--s3)';
          const dim = focus && focus !== s.id;
          return (
            <g key={s.id} opacity={dim ? 0.15 : 1}>
              {present.every(Boolean) && <polygon points={vals.map((v, i) => pt(i, v!.drawn!).join(',')).join(' ')} fill={col} fillOpacity={0.08} stroke="none" />}
              {radarEdges(present).map(([i, j]) => {
                const [x1, y1] = pt(i, vals[i]!.drawn!), [x2, y2] = pt(j, vals[j]!.drawn!);
                return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={1.6} />;
              })}
              {vals.map((v, i) => {
                if (!v?.spread?.drawn) return null;
                const [x1, y1] = pt(i, v.spread.drawn[0]), [x2, y2] = pt(i, v.spread.drawn[1]);
                return <line key={`w${i}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={3} strokeOpacity={0.35} strokeLinecap="round"><title>{`${s.label} · ${radar.axes[i].label}: ${v.spread.meaning ?? 'spread'} ${v.spread.x.map((x) => fmtNum(x)).join('–')}`}</title></line>;
              })}
              {vals.map((v, i) => {
                if (v?.drawn == null) return null;
                const [x, y] = pt(i, v.drawn);
                const clamped = v.r != null && v.r !== v.drawn;
                return (
                  <circle key={i} cx={x} cy={y} r={compact ? 2.4 : 3.4} fill={clamped ? 'var(--surface)' : col} stroke={col} strokeWidth={1.2}>
                    <title>{`${s.label} · ${radar.axes[i].label}: ${fmtNum(v.value)}${v.k != null ? ` (${v.k}/${v.n})` : ''} · r ${fmtNum(v.r)}${clamped ? ' (clamped for drawing)' : ''}\n${(v.evidence || []).join('\n')}\n${(v.decisions || []).join(' ')}`}</title>
                  </circle>
                );
              })}
            </g>
          );
        })}
      </svg>
      <figcaption className="legend" style={{ justifyContent: 'center' }}>
        {radar.series.map((s) => {
          const miss = radar.axes.filter((a) => a.series[s.id]?.drawn == null);
          return (
            <button key={s.id} className="ghost" onMouseEnter={() => setFocus(s.id)} onMouseLeave={() => setFocus(null)} onFocus={() => setFocus(s.id)} onBlur={() => setFocus(null)}
              title={miss.map((a) => `${a.label}: ${a.series[s.id]?.missing ?? a.series[s.id]?.missing_r ?? 'missing'}`).join('\n')} style={{ padding: 0, fontSize: 10 }}>
              <i className="sw" style={{ background: SERIES_COLOR[s.id] || 'var(--s3)' }} />{compact ? s.id : s.label}
              {miss.length ? <small className="muted"> · {miss.length === n ? 'no data' : `${miss.length}/${n} gaps`}</small> : null}
            </button>
          );
        })}
      </figcaption>
    </figure>
  );
}
