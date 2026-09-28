/**
 * Synchronized timeline panels for one or two replays. Every panel shares the playback clock: a cursor at the current
 * time, click/drag to scrub, edit-active intervals shaded. Missing signals say so; nothing is interpolated.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { fmtNum, str } from '../lib/format';
import { seriesColor } from '../lib/labels';
import { frameAt, relTimes, runs, segments, type Clock, type Replay } from '../lib/replay';
import { PcaPlot } from './Stage';

export type Side = { replay: Replay; times: number[]; color: string; tag: string };

/** The recorder's own description of a signal (meta.signal_notes); privileged signals are flagged. */
function noteOf(sides: Side[], key?: string) {
  if (!key) return null;
  const notes = sides[0]?.replay.meta.signal_notes as Record<string, string> | undefined;
  const n = notes?.[key];
  if (!n) return null;
  return (
    <span className="small muted" title={n} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 520 }}>
      {/privileged/i.test(n) && <span className="badge src t-oracle" style={{ marginRight: 4 }}>PRIVILEGED · display only</span>}
      {n}
    </span>
  );
}

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(200, el.clientWidth)));
    ro.observe(el);
    setW(Math.max(200, el.clientWidth));
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function Track({ title, value, t, duration, onSeek, sides, height = 60, children, legend }: {
  title: ReactNode; value?: ReactNode; t: number; duration: number; onSeek: (t: number) => void; sides: Side[]; height?: number;
  children: (x: (t: number) => number, w: number, h: number) => ReactNode; legend?: ReactNode;
}) {
  const [ref, w] = useWidth();
  const pad = 4;
  const x = (tt: number) => pad + (duration > 0 ? (tt / duration) * (w - 2 * pad) : 0);
  const drag = useRef(false);
  const seekAt = (clientX: number, el: SVGSVGElement) => {
    const r = el.getBoundingClientRect();
    onSeek(Math.max(0, Math.min(duration, ((clientX - r.left - pad) / (r.width - 2 * pad)) * duration)));
  };
  return (
    <div className="tl-panel">
      <div className="h"><b>{title}</b>{legend}<span className="v">{value}</span></div>
      <div ref={ref}>
        <svg width={w} height={height} viewBox={`0 0 ${w} ${height}`}
          onPointerDown={(e) => { drag.current = true; (e.target as Element).setPointerCapture?.(e.pointerId); seekAt(e.clientX, e.currentTarget); }}
          onPointerMove={(e) => { if (drag.current) seekAt(e.clientX, e.currentTarget); }}
          onPointerUp={() => { drag.current = false; }}>
          {sides.map((s, si) => {
            const ea = s.replay.signals.edit_active;
            if (!ea) return null;
            return runs(ea, (v) => !!v).map(([a, b]) => (
              <rect key={`${si}-${a}`} x={x(s.times[a])} width={Math.max(1, x(s.times[Math.min(b, s.times.length - 1)]) - x(s.times[a]))}
                y={sides.length > 1 ? (si * height) / 2 : 0} height={sides.length > 1 ? height / 2 : height}
                fill="var(--s2)" opacity={0.1} />
            ));
          })}
          {children(x, w, height)}
          <line className="cursor" x1={x(t)} x2={x(t)} y1={0} y2={height} stroke="var(--ink)" strokeWidth={1.2} />
        </svg>
      </div>
    </div>
  );
}

function linePath(times: number[], vals: (number | null | undefined)[], x: (t: number) => number, y: (v: number) => number, maxPts = 1200) {
  const step = Math.max(1, Math.floor(vals.length / maxPts));
  let d = '', pen = false;
  for (let i = 0; i < vals.length; i += step) {
    const v = vals[i];
    if (v === null || v === undefined || !Number.isFinite(v)) { pen = false; continue; }
    d += `${pen ? 'L' : 'M'}${x(times[i]).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}
function extent(xs: (number | null | undefined)[]) {
  let lo = Infinity, hi = -Infinity;
  for (const v of xs) if (v !== null && v !== undefined && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!Number.isFinite(lo)) return [0, 1];
  if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
  return [lo, hi];
}

type Val = number | null | undefined | (number | null)[];
function ScalarTrack({ title, get, sides, t, duration, onSeek, unit, height = 56, noteKey, names }: {
  title: string; get: (r: Replay) => Val[] | undefined; sides: Side[]; t: number; duration: number;
  onSeek: (t: number) => void; unit?: string; height?: number; noteKey?: string; names?: (r: Replay) => string[] | undefined;
}) {
  const data = sides.map((s) => get(s.replay));
  if (data.every((d) => !d)) return <Missing title={title} />;
  const multi = data.some((d) => d?.some((v) => Array.isArray(v)));
  const cols = multi ? Math.max(...data.flatMap((d) => (d || []).map((v) => (Array.isArray(v) ? v.length : 0)))) : 1;
  const colOf = (d: Val[], j: number) => d.map((v) => (Array.isArray(v) ? v[j] ?? null : multi ? null : v ?? null));
  const flatVals = data.flatMap((d) => (d ? Array.from({ length: cols }, (_, j) => colOf(d, j)).flat() : []));
  const [lo, hi] = extent(flatVals);
  const h = multi ? Math.max(height, 70) : height;
  const y = (v: number) => h - 4 - ((v - lo) / (hi - lo)) * (h - 8);
  const cur = sides.map((s, i) => { const d = data[i]; return d ? d[frameAt(s.times, t)] : undefined; });
  const colors = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)', 'var(--s7)', 'var(--s8)'];
  const nm = names?.(sides[0].replay);
  const fmtCur = (v: Val) => (Array.isArray(v) ? `[${v.map((x) => fmtNum(x)).join(', ')}]` : fmtNum(v));
  return (
    <Track title={title} t={t} duration={duration} onSeek={onSeek} sides={sides} height={h}
      legend={<>{noteOf(sides, noteKey)}{multi && <span className="legend small">{Array.from({ length: Math.min(cols, 8) }, (_, j) => <span key={j}><i className="sw" style={{ background: colors[j] }} />{nm?.[j] || `#${j}`}</span>)}</span>}</>}
      value={<>{sides.map((s, i) => <span key={i} style={{ marginLeft: 8 }}>{sides.length > 1 ? `${s.tag} ` : ''}{data[i] ? fmtCur(cur[i]) : 'n/a'}</span>)}{unit ? ` ${unit}` : ''} <span className="muted">[{fmtNum(lo)}, {fmtNum(hi)}]</span></>}>
      {(x) => sides.map((s, i) => data[i] && Array.from({ length: Math.min(cols, 8) }, (_, j) => (
        <path key={`${i}-${j}`} d={linePath(s.times, colOf(data[i]!, j), x, y)} fill="none" stroke={multi ? colors[j] : s.color} strokeWidth={1.5} strokeDasharray={i === 1 ? '5 3' : undefined} opacity={multi && i === 1 ? 0.7 : 1} />
      )))}
    </Track>
  );
}

function Missing({ title, why }: { title: string; why?: string }) {
  return (
    <div className="tl-panel">
      <div className="h"><b>{title}</b></div>
      <div className="missing">{why || 'not recorded in this replay (omitted, not faked)'}</div>
    </div>
  );
}

function RasterTrack({ title, get, names, sides, t, duration, onSeek, prob, noteKey }: {
  title: string; get: (r: Replay) => unknown[] | undefined; names?: (r: Replay) => string[] | undefined; sides: Side[]; t: number; duration: number;
  onSeek: (t: number) => void; prob?: boolean; noteKey?: string;
}) {
  const data = sides.map((s) => get(s.replay) as (unknown[] | null)[] | undefined);
  if (data.every((d) => !d)) return <Missing title={title} />;
  const rowsN = sides.map((_, i) => Math.max(0, ...(data[i] || []).map((f) => (Array.isArray(f) ? f.length : 0))));
  const rowH = 9;
  const total = rowsN.reduce((a, b) => a + b, 0);
  const height = Math.max(20, total * rowH + (sides.length - 1) * 6);
  const labels = sides.map((s, i) => names?.(s.replay) || Array.from({ length: rowsN[i] }, (_, k) => `#${k}`));
  const curFlags = sides.map((s, i) => { const f = data[i]?.[frameAt(s.times, t)]; return Array.isArray(f) ? f : []; });
  return (
    <Track title={title} t={t} duration={duration} onSeek={onSeek} sides={sides} height={height} legend={noteOf(sides, noteKey)}
      value={<span className="small">{sides.map((s, i) => (
        <span key={i} style={{ marginLeft: 8 }}>{sides.length > 1 ? `${s.tag}: ` : ''}
          {labels[i].filter((_, k) => (prob ? Number(curFlags[i][k]) > 0.5 : !!curFlags[i][k])).join(', ') || 'none'}</span>
      ))}</span>}>
      {(x) => {
        let y0 = 0;
        return sides.map((s, si) => {
          const d = data[si];
          const off = y0;
          y0 += rowsN[si] * rowH + 6;
          if (!d) return null;
          const els: ReactNode[] = [];
          for (let k = 0; k < rowsN[si]; k++) {
            if (prob) {
              const step = Math.max(1, Math.floor(d.length / 600));
              for (let i = 0; i < d.length; i += step) {
                const v = Number((d[i] as unknown[] | null)?.[k]);
                if (!Number.isFinite(v) || v <= 0.02) continue;
                const t1 = s.times[Math.min(i + step, s.times.length - 1)];
                els.push(<rect key={`${k}-${i}`} x={x(s.times[i])} width={Math.max(1, x(t1) - x(s.times[i]))} y={off + k * rowH} height={rowH - 2} fill={s.color} opacity={Math.min(1, v)} />);
              }
            } else {
              for (const [a, b] of runs(d, (f) => Array.isArray(f) && !!f[k])) {
                els.push(<rect key={`${k}-${a}`} x={x(s.times[a])} width={Math.max(1.5, x(s.times[Math.min(b, s.times.length - 1)]) - x(s.times[a]))} y={off + k * rowH} height={rowH - 2} rx={1.5} fill={s.color} />);
              }
            }
            els.push(<text key={`l${k}`} x={4} y={off + k * rowH + 7} fontSize={8} fill="var(--muted)">{labels[si][k]}</text>);
          }
          return <g key={si}>{els}</g>;
        });
      }}
    </Track>
  );
}

function CategoryTrack({ title, get, sides, t, duration, onSeek, noteKey }: {
  title: string; get: (r: Replay) => unknown[] | undefined; sides: Side[]; t: number; duration: number; onSeek: (t: number) => void; noteKey?: string;
}) {
  const data = sides.map((s) => get(s.replay));
  if (data.every((d) => !d)) return <Missing title={title} />;
  const laneH = 18;
  const height = sides.length * (laneH + 4);
  const cats = Array.from(new Set(data.flatMap((d) => (d || []).map((v) => str(v))))).filter(Boolean);
  const cur = sides.map((s, i) => str(data[i]?.[frameAt(s.times, t)]));
  return (
    <Track title={title} t={t} duration={duration} onSeek={onSeek} sides={sides} height={height} legend={noteOf(sides, noteKey)}
      value={sides.map((s, i) => <span key={i} style={{ marginLeft: 8 }}>{sides.length > 1 ? `${s.tag}: ` : ''}{cur[i] || '—'}</span>)}>
      {(x) => sides.map((s, si) => {
        const d = data[si];
        if (!d) return null;
        return segments(d.map((v) => str(v))).map((seg) => {
          const x0 = x(s.times[seg.s]), x1 = x(s.times[Math.min(seg.e, s.times.length - 1)]);
          return (
            <g key={`${si}-${seg.s}`}>
              <rect x={x0} width={Math.max(1, x1 - x0)} y={si * (laneH + 4)} height={laneH} fill={seg.v ? seriesColor(seg.v, cats) : 'transparent'} opacity={0.75} />
              {x1 - x0 > 40 && <text x={x0 + 3} y={si * (laneH + 4) + 12} fontSize={10} fill="#fff">{seg.v}</text>}
            </g>
          );
        });
      })}
    </Track>
  );
}

function EventsTrack({ sides, t, duration, onSeek }: { sides: Side[]; t: number; duration: number; onSeek: (t: number) => void }) {
  const ev = sides.map((s) => {
    const t0 = s.replay.frames.t[0] || 0;
    return [
      ...(s.replay.signals.task_events || []).map((e) => ({ t: e.t - t0, text: `${e.event}: ${e.status}`, kind: 'event', status: e.status })),
      ...(s.replay.annotations || []).map((a) => ({ t: a.t - t0, text: a.text, kind: 'note', status: '' })),
    ];
  });
  if (ev.every((e) => !e.length)) return <Missing title="Task events & annotations" />;
  const laneH = 26;
  const recent = sides.map((_, i) => ev[i].filter((e) => e.t <= t).slice(-1)[0]);
  return (
    <Track title="Task events & annotations" t={t} duration={duration} onSeek={onSeek} sides={sides} height={sides.length * laneH}
      value={recent.map((r, i) => <span key={i} style={{ marginLeft: 8 }}>{sides.length > 1 ? `${sides[i].tag}: ` : ''}{r ? r.text : '—'}</span>)}>
      {(x) => ev.map((es, si) => es.map((e, k) => {
        const ok = /success|done|complete|achiev|true/i.test(e.status);
        const bad = /fail|false|drop|violat/i.test(e.status);
        return (
          <g key={`${si}-${k}`}>
            <title>{`${fmtNum(e.t)} s · ${e.text}`}</title>
            <line x1={x(e.t)} x2={x(e.t)} y1={si * laneH + 2} y2={si * laneH + laneH - 4} stroke={e.kind === 'note' ? 'var(--muted)' : sides[si].color} strokeWidth={1.5} />
            {e.kind === 'note'
              ? <rect x={x(e.t) - 3} y={si * laneH + 2} width={6} height={6} fill="var(--muted)" />
              : <circle cx={x(e.t)} cy={si * laneH + 6} r={4} fill={ok ? 'var(--good)' : bad ? 'var(--critical)' : 'var(--warning)'} />}
          </g>
        );
      }))}
    </Track>
  );
}

function JointTracks({ sides, t, duration, onSeek }: { sides: Side[]; t: number; duration: number; onSeek: (t: number) => void }) {
  const primary = sides[0].replay;
  const tgt = primary.signals.joint_target, pos = primary.signals.joint_pos;
  const [showAll, setShowAll] = useState(false);
  if (!tgt && !pos) return <Missing title="Joint targets vs positions" />;
  const width = (xs?: (number[] | null)[]) => Math.max(0, ...(xs || []).map((f) => (f ? f.length : 0)));
  const pn = (primary.meta.joint_names as string[] | undefined) || Array.from({ length: width(pos) }, (_, j) => `joint ${j}`);
  const tn = (primary.meta.joint_target_names as string[] | undefined) || (width(tgt) === width(pos) ? pn : Array.from({ length: width(tgt) }, (_, j) => `target ${j}`));
  // rows: every measured joint with its same-named target (if any), then targets with no measured joint
  // names shared -> pair by name; no shared name -> pair by column order and say so on every row (e.g. r0_joint1 ↔ arm[0])
  const shared = pn.some((n) => tn.includes(n));
  const rowsJ: { name: string; p: number; tIdx: number }[] = shared
    ? pn.map((name, j) => ({ name, p: j, tIdx: tn.indexOf(name) }))
    : pn.map((name, j) => ({ name: j < tn.length ? `${name} ↔ ${tn[j]}` : name, p: j, tIdx: j < tn.length ? j : -1 }));
  if (shared) tn.forEach((name, k) => { if (!pn.includes(name)) rowsJ.push({ name, p: -1, tIdx: k }); });
  else for (let k = pn.length; k < tn.length; k++) rowsJ.push({ name: tn[k], p: -1, tIdx: k });
  const shown = showAll ? rowsJ : rowsJ.slice(0, 8);
  return (
    <div className="tl-panel">
      <div className="h">
        <b>Joint targets (dashed) vs positions (solid)</b>
        {noteOf(sides, 'joint_target')}
        <span className="muted small">{sides.length > 1 ? `replay ${sides[0].tag} only` : ''}{!shared && tgt && pos ? ' · target and position names differ: paired by column order (shown as position ↔ target)' : ''}</span>
        {rowsJ.length > 8 && <button className="ghost small" onClick={() => setShowAll(!showAll)} style={{ marginLeft: 'auto' }}>{showAll ? 'fewer' : `all ${rowsJ.length}`}</button>}
      </div>
      <div style={{ display: 'grid', gap: 2 }}>
        {shown.map((r, j) => (
          <JointRow key={j} name={r.name} tgt={r.tIdx >= 0 ? tgt?.map((f) => f?.[r.tIdx] ?? null) : undefined} pos={r.p >= 0 ? pos?.map((f) => f?.[r.p] ?? null) : undefined}
            side={sides[0]} t={t} duration={duration} onSeek={onSeek} />
        ))}
      </div>
    </div>
  );
}
function JointRow({ name, tgt, pos, side, t, duration, onSeek }: {
  name: string; tgt?: (number | null)[]; pos?: (number | null)[]; side: Side; t: number; duration: number; onSeek: (t: number) => void;
}) {
  const [ref, w] = useWidth();
  const h = 30;
  const [lo, hi] = extent([...(tgt || []), ...(pos || [])]);
  const x = (tt: number) => 96 + (duration > 0 ? (tt / duration) * (w - 100) : 0);
  const y = (v: number) => h - 3 - ((v - lo) / (hi - lo)) * (h - 6);
  const f = frameAt(side.times, t);
  return (
    <div ref={ref}>
      <svg width={w} height={h} onPointerDown={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        onSeek(Math.max(0, Math.min(duration, ((e.clientX - r.left - 96) / (r.width - 100)) * duration)));
      }}>
        <text x={0} y={12} fontSize={10.5} fill="var(--ink-2)"><title>{name}</title>{name.length > 15 ? `${name.slice(0, 14)}…` : name}</text>
        <text x={0} y={25} fontSize={9.5} fill="var(--muted)" className="num">{fmtNum(pos?.[f])} / {fmtNum(tgt?.[f])}</text>
        <line x1={96} x2={w} y1={h - 1} y2={h - 1} stroke="var(--grid)" />
        {tgt && <path d={linePath(side.times, tgt, x, y)} fill="none" stroke="var(--s2)" strokeWidth={1.3} strokeDasharray="4 3" />}
        {pos && <path d={linePath(side.times, pos, x, y)} fill="none" stroke="var(--s1)" strokeWidth={1.5} />}
        <line x1={x(t)} x2={x(t)} y1={0} y2={h} stroke="var(--ink)" />
      </svg>
    </div>
  );
}

function ProbeTracks({ sides, t, duration, onSeek }: { sides: Side[]; t: number; duration: number; onSeek: (t: number) => void }) {
  const keys = Array.from(new Set(sides.flatMap((s) => Object.keys(s.replay.signals.probe || {}))));
  if (!keys.length) return <Missing title="Probe readouts" why="no probe signals in this replay (probes are diagnostics, not evidence of use)" />;
  return (
    <>
      {keys.map((k) => {
        const sample = sides.map((s) => s.replay.signals.probe?.[k]).find(Boolean) || [];
        const first = sample.find((v) => v !== null && v !== undefined);
        const title = `Probe · ${k}`;
        if (Array.isArray(first)) {
          if (k === 'goal' && first.length === 2) {
            return (
              <div key={k} className="tl-panel" style={{ padding: 0, border: 0 }}>
                <ScalarTrack title={`${title} · x`} get={(r) => (r.signals.probe?.[k] as (number[] | null)[] | undefined)?.map((v) => v?.[0] ?? null)} sides={sides} t={t} duration={duration} onSeek={onSeek} height={40} />
                <ScalarTrack title={`${title} · y`} get={(r) => (r.signals.probe?.[k] as (number[] | null)[] | undefined)?.map((v) => v?.[1] ?? null)} sides={sides} t={t} duration={duration} onSeek={onSeek} height={40} />
              </div>
            );
          }
          return <RasterTrack key={k} title={`${title} (per slot)`} get={(r) => r.signals.probe?.[k]} sides={sides} t={t} duration={duration} onSeek={onSeek} prob
            names={(r) => (k === 'contact' && r.meta.contact_bodies?.length === first.length ? r.meta.contact_bodies : undefined)} noteKey="probe" />;
        }
        if (typeof first === 'number' && !/subtask|index|slot/.test(k)) return <ScalarTrack key={k} title={title} get={(r) => r.signals.probe?.[k] as (number | null)[] | undefined} sides={sides} t={t} duration={duration} onSeek={onSeek} noteKey="probe" />;
        return <CategoryTrack key={k} title={title} get={(r) => r.signals.probe?.[k]} sides={sides} t={t} duration={duration} onSeek={onSeek} noteKey="probe" />;
      })}
    </>
  );
}

export default function Timelines({ sides, clock, t, duration }: { sides: Side[]; clock: Clock; t: number; duration: number }) {
  const onSeek = (x: number) => clock.set(x);
  const p = { sides, t, duration, onSeek };
  const pcaSeries = useMemo(() => {
    const basisKey = (r: Replay) => JSON.stringify(r.meta.packet_pca_basis?.explained_variance_ratio ?? null) + str(r.meta.packet_pca_basis?.n_fit);
    const withPca = sides.filter((s) => s.replay.signals.packet_pca);
    const same = withPca.length < 2 || basisKey(withPca[0].replay) === basisKey(withPca[1].replay);
    return {
      same,
      series: (same ? withPca : withPca.slice(0, 1)).map((s) => ({
        label: s.tag, color: s.color, pts: s.replay.signals.packet_pca!, times: relTimes(s.replay), edit: s.replay.signals.edit_active,
      })),
    };
  }, [sides]);
  const pm = sides[0].replay.meta.packet_pca as { fit_on?: string; explained_variance?: number[] } | undefined;
  const ev = (pm?.explained_variance || sides[0].replay.meta.packet_pca_basis?.explained_variance_ratio) as number[] | undefined;
  return (
    <div className="tl">
      <ScalarTrack title="Forward progress" get={(r) => r.signals.forward_progress} noteKey="forward_progress" {...p} />
      <CategoryTrack title="Phase" get={(r) => r.signals.phase} noteKey="phase" {...p} />
      <EventsTrack {...p} />
      <CategoryTrack title="Edit active" get={(r) => r.signals.edit_active?.map((v) => (v ? 'edit on' : v === false ? 'off' : null))} noteKey="edit_active" {...p} />
      <RasterTrack title="Contacts (feet / fingers)" get={(r) => r.signals.contacts} names={(r) => r.meta.contact_bodies} noteKey="contacts" {...p} />
      <ProbeTracks {...p} />
      <div className="tl-panel">
        <div className="h">
          <b>Packet PCA trajectory (3D)</b>
          {noteOf(sides, 'packet_pca')}
          <span className="muted small" title={pm?.fit_on}>
            {ev ? `explained variance ${ev.map((v) => fmtNum(v)).join(' / ')}` : ''}{' '}
            {!pcaSeries.same && '· bases differ between A and B: only A is drawn'}
          </span>
        </div>
        {pcaSeries.series.length
          ? <PcaPlot series={pcaSeries.series} clock={clock} />
          : <div className="missing">no packet_pca in this replay (e.g. teacher or BC routes have no packets)</div>}
        {pcaSeries.series.length > 0 && <div className="legend small">
          {pcaSeries.series.map((s) => <span key={s.label}><i className="sw" style={{ background: s.color }} />{s.label}</span>)}
          <span><i className="sw" style={{ background: 'var(--s2)' }} />edit-active frames</span>
          <span className="muted">axes: PC1 red, PC2 green, PC3 blue · drag to orbit</span>
        </div>}
      </div>
      <JointTracks {...p} />
      <ScalarTrack title="Object height (object_pose z)" get={(r) => r.signals.object_pose?.map((v) => (v && v.length >= 3 ? v[2] : null))} unit="m" noteKey="object_pose" {...p} />
      <ScalarTrack title="Penetration" get={(r) => r.signals.penetration_mm as Val[] | undefined} unit="mm" noteKey="penetration_mm" {...p} />
      <ScalarTrack title="Slip" get={(r) => r.signals.slip as Val[] | undefined} noteKey="slip" names={(r) => r.meta.contact_bodies} {...p} />
    </div>
  );
}
