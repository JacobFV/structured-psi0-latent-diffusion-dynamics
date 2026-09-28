/**
 * Run-history visualizations (v3). Each reads only recorded replay fields; a panel is offered only when its inputs exist
 * (see `available*` helpers). Derived quantities (finite-difference velocity, PCA-space speed, duty factor, calibration
 * bins) are computed here and labelled as computed. All share the playback clock (cursor + click-to-seek).
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { fmtNum, isObj, num, str } from '../lib/format';
import { seriesColor } from '../lib/labels';
import { frameAt, runs, segments, type Clock, type Replay } from '../lib/replay';
import { linePath, Missing, ScalarTrack, Track, useWidth, type Side } from './Timelines';

type P = { sides: Side[]; t: number; duration: number; onSeek: (t: number) => void };
const numArr = (x: unknown): (number | null)[] | null => (Array.isArray(x) ? x.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : null)) : null);

/* ---------------------------------------------------------------- top-down trajectory map */
function xyPath(r: Replay, which: 'base' | 'object'): (number[] | null)[] {
  if (which === 'object') {
    if (r.signals.object_pose) return r.signals.object_pose.map((p) => (p && p.length >= 2 ? [p[0], p[1]] : null));
    const ob = r.meta.object_body ? r.bodies.indexOf(r.meta.object_body) : -1;
    return ob >= 0 ? r.frames.body_pos.map((f) => (f[ob] ? [f[ob][0], f[ob][1]] : null)) : [];
  }
  const bb = r.meta.base_body ? r.bodies.indexOf(r.meta.base_body) : -1;
  return bb >= 0 ? r.frames.body_pos.map((f) => (f[bb] ? [f[bb][0], f[bb][1]] : null)) : [];
}
function waypoints(r: Replay): [string, number[]][] {
  const w = r.meta.waypoints;
  if (Array.isArray(w)) return w.filter(Array.isArray).map((p, i) => [`w${i}`, p as number[]]);
  if (isObj(w)) return Object.entries(w).filter(([, p]) => Array.isArray(p)).map(([k, p]) => [k, p as number[]]);
  return [];
}
function editFrame(r: Replay, times: number[]) {
  const ea = r.signals.edit_active;
  const i = ea ? ea.findIndex((v) => v === true) : -1;
  if (i >= 0) return i;
  const te = num(r.meta.edit_onset_t) ?? num(r.meta.t_edit);
  return te !== null ? frameAt(times, te) : -1;
}
export function hasMap(r: Replay) {
  const b = xyPath(r, 'base'), o = xyPath(r, 'object');
  const moves = (p: (number[] | null)[]) => {
    const f = p.filter(Boolean) as number[][];
    return f.length > 1 && Math.hypot(f[0][0] - f[f.length - 1][0], f[0][1] - f[f.length - 1][1]) > 0.02;
  };
  return moves(b) || moves(o);
}
export function TopDownMap({ sides, t, onSeek }: P) {
  const [ref, w] = useWidth();
  const layers = sides.map((s) => ({ s, base: xyPath(s.replay, 'base'), obj: xyPath(s.replay, 'object'), wp: waypoints(s.replay), ef: editFrame(s.replay, s.times) }));
  const pts = layers.flatMap((l) => [...l.base, ...l.obj, ...l.wp.map((x) => x[1])]).filter(Boolean) as number[][];
  if (!pts.length) return <Missing title="Top-down map" />;
  let x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
  let y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
  const span = Math.max(x1 - x0, y1 - y0, 0.2) * 1.12;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  x0 = cx - span / 2; x1 = cx + span / 2; y0 = cy - span / 2; y1 = cy + span / 2;
  const H = Math.min(420, Math.max(220, w * 0.55)), W = w;
  const s = Math.min(W, H) / span;
  const X = (x: number) => W / 2 + (x - cx) * s, Y = (y: number) => H / 2 - (y - cy) * s;
  const path = (p: (number[] | null)[]) => {
    let d = '', pen = false;
    p.forEach((q) => { if (!q) { pen = false; return; } d += `${pen ? 'L' : 'M'}${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`; pen = true; });
    return d;
  };
  const gridStep = [0.05, 0.1, 0.25, 0.5, 1, 2, 5].find((g) => span / g <= 12) || 10;
  const gx: number[] = [];
  for (let v = Math.ceil(x0 / gridStep) * gridStep; v <= x1; v += gridStep) gx.push(v);
  const gy: number[] = [];
  for (let v = Math.ceil(y0 / gridStep) * gridStep; v <= y1; v += gridStep) gy.push(v);
  return (
    <div ref={ref}>
      <svg width={W} height={H} role="img" aria-label="top-down trajectories" onPointerDown={(e) => {
        // seek to the base frame nearest the click
        const r = e.currentTarget.getBoundingClientRect();
        const px = e.clientX - r.left, py = e.clientY - r.top;
        const L = layers[0];
        const p = L.base.length ? L.base : L.obj;
        let best = -1, bd = 1e9;
        p.forEach((q, i) => { if (q) { const d = Math.hypot(X(q[0]) - px, Y(q[1]) - py); if (d < bd) { bd = d; best = i; } } });
        if (best >= 0 && bd < 30) onSeek(L.s.times[best]);
      }}>
        {gx.map((v) => <line key={`x${v}`} x1={X(v)} x2={X(v)} y1={0} y2={H} stroke="var(--grid)" />)}
        {gy.map((v) => <line key={`y${v}`} x1={0} x2={W} y1={Y(v)} y2={Y(v)} stroke="var(--grid)" />)}
        <text x={4} y={H - 4} fontSize={10} fill="var(--muted)">grid {gridStep} m · x →, y ↑ (world)</text>
        {layers.map((l, li) => {
          const f = frameAt(l.s.times, t);
          const dash = li === 1 ? '5 3' : undefined;
          const bp = l.base[f], op = l.obj[f];
          const fell = l.s.replay.meta.fell === true;
          const lastBase = [...l.base].reverse().find(Boolean);
          const ep = l.ef >= 0 ? l.base[l.ef] || l.obj[l.ef] : null;
          return (
            <g key={li}>
              {l.wp.map(([k, p]) => (
                <g key={k}><title>{`waypoint ${k} (${fmtNum(p[0])}, ${fmtNum(p[1])})`}</title>
                  <rect x={X(p[0]) - 5} y={Y(p[1]) - 5} width={10} height={10} fill="none" stroke="var(--s4)" strokeWidth={1.5} transform={`rotate(45 ${X(p[0])} ${Y(p[1])})`} />
                  <text x={X(p[0]) + 8} y={Y(p[1]) + 4} fontSize={10} fill="var(--s4)">{k}</text></g>
              ))}
              {Array.isArray(l.s.replay.signals.contact_pos) && (l.s.replay.signals.contact_pos as unknown[][]).map((fr, fi) => (Array.isArray(fr) ? fr : []).map((cp, ci) => (Array.isArray(cp) && cp.length >= 2 && fi % 2 === 0 ? <circle key={`${fi}-${ci}`} cx={X(cp[0] as number)} cy={Y(cp[1] as number)} r={1.4} fill={seriesColor(String(ci), ['0', '1', '2', '3', '4', '5'])} opacity={0.35} /> : null)))}
              {l.base.length > 0 && <path d={path(l.base)} fill="none" stroke={l.s.color} strokeWidth={1.8} strokeDasharray={dash} opacity={0.9} />}
              {l.obj.length > 0 && <path d={path(l.obj)} fill="none" stroke="var(--s3)" strokeWidth={1.5} strokeDasharray={li === 1 ? '2 3' : undefined} />}
              {ep && <g><title>{`edit onset${l.s.replay.meta.edit ? ` (${str(l.s.replay.meta.edit)})` : ''} at ${fmtNum(l.s.times[l.ef])} s`}</title>
                <circle cx={X(ep[0])} cy={Y(ep[1])} r={6} fill="none" stroke="var(--s2)" strokeWidth={2} /><text x={X(ep[0]) + 8} y={Y(ep[1]) - 6} fontSize={10} fill="var(--s2)">edit</text></g>}
              {fell && lastBase && <g><title>fell (recorded meta.fell)</title><text x={X(lastBase[0])} y={Y(lastBase[1]) + 4} fontSize={12} textAnchor="middle" fill="var(--critical)">✕</text></g>}
              {bp && <circle cx={X(bp[0])} cy={Y(bp[1])} r={4.5} fill={l.s.color} stroke="var(--surface)" strokeWidth={1.5} />}
              {op && <rect x={X(op[0]) - 3.5} y={Y(op[1]) - 3.5} width={7} height={7} fill="var(--s3)" stroke="var(--surface)" />}
            </g>
          );
        })}
      </svg>
      <div className="legend small">
        {sides.map((s) => <span key={s.tag}><i className="sw" style={{ background: s.color }} />{sides.length > 1 ? `${s.tag} ` : ''}base path</span>)}
        <span><i className="sw" style={{ background: 'var(--s3)' }} />object</span>
        {sides.some((s) => s.replay.signals.contact_pos) && <span>· contact points (privileged)</span>}
        <span style={{ color: 'var(--s4)' }}>◇ waypoint</span><span style={{ color: 'var(--s2)' }}>○ edit onset</span><span style={{ color: 'var(--critical)' }}>✕ fell</span>
        <span className="muted">click the map to seek</span>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- gait / contact diagram */
export function GaitDiagram({ sides, t, duration, onSeek }: P) {
  const data = sides.map((s) => s.replay.signals.contacts as (boolean[] | number[] | null)[] | undefined);
  if (data.every((d) => !d)) return <Missing title="Contact / gait diagram" />;
  const force = sides.map((s) => s.replay.signals.contact_force as (number[] | null)[] | undefined);
  const rowH = 12;
  const rowsN = sides.map((_, i) => Math.max(0, ...(data[i] || []).map((f) => (Array.isArray(f) ? f.length : 0))));
  const H = rowsN.reduce((a, b) => a + b * rowH + 8, 0);
  return (
    <Track title="" t={t} duration={duration} onSeek={onSeek} sides={sides} height={H}
      legend={<span className="small muted">bars = in contact{force.some(Boolean) ? '; shade = normal force' : ''} · duty factor = share of frames in contact (computed here)</span>}>
      {(x, w) => {
        let off = 0;
        return sides.map((s, si) => {
          const d = data[si];
          const y0 = off;
          off += rowsN[si] * rowH + 8;
          if (!d) return null;
          const names = s.replay.meta.contact_bodies || [];
          const fmax = Math.max(1e-9, ...(force[si] || []).flatMap((f) => (f || []).map((v) => Math.abs(v || 0))));
          return (
            <g key={si}>
              {Array.from({ length: rowsN[si] }, (_, k) => {
                const on = d.filter((f) => Array.isArray(f) && !!f[k]).length;
                const duty = d.length ? on / d.length : 0;
                return (
                  <g key={k}>
                    <rect x={0} y={y0 + k * rowH} width={w} height={rowH - 2} fill="var(--surface-2)" />
                    {runs(d, (f) => Array.isArray(f) && !!f[k]).map(([a, b]) => {
                      const fv = force[si]?.slice(a, b).map((f) => Math.abs(f?.[k] || 0));
                      const op = fv && fv.length ? 0.35 + 0.65 * (Math.max(...fv) / fmax) : 1;
                      return <rect key={a} x={x(s.times[a])} width={Math.max(1.5, x(s.times[Math.min(b, s.times.length - 1)]) - x(s.times[a]))} y={y0 + k * rowH} height={rowH - 2} fill={s.color} opacity={op} />;
                    })}
                    <text x={3} y={y0 + k * rowH + 8.5} fontSize={9} fill="var(--ink)" style={{ paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}>
                      {sides.length > 1 ? `${s.tag} ` : ''}{(names[k] || `#${k}`).replace(/^r\d_/, '')} · {(duty * 100).toFixed(0)}%
                    </text>
                  </g>
                );
              })}
            </g>
          );
        });
      }}
    </Track>
  );
}

/* ---------------------------------------------------------------- joint raster heatmap (canvas) */
type JMode = 'pos' | 'target' | 'error' | 'vel';
function jointMatrix(r: Replay, mode: JMode): { names: string[]; rows: (number | null)[][] } | null {
  const pos = r.signals.joint_pos, tgt = r.signals.joint_target;
  const pn = (r.meta.joint_names as string[] | undefined) || [];
  const tn = (r.meta.joint_target_names as string[] | undefined) || pn;
  const width = (xs?: (number[] | null)[]) => Math.max(0, ...(xs || []).map((f) => (f ? f.length : 0)));
  const col = (xs: (number[] | null)[] | undefined, j: number) => (xs || []).map((f) => (f && typeof f[j] === 'number' ? f[j] : null));
  if (mode === 'pos' && pos) return { names: Array.from({ length: width(pos) }, (_, j) => pn[j] || `q${j}`), rows: Array.from({ length: width(pos) }, (_, j) => col(pos, j)) };
  if (mode === 'target' && tgt) return { names: Array.from({ length: width(tgt) }, (_, j) => tn[j] || `u${j}`), rows: Array.from({ length: width(tgt) }, (_, j) => col(tgt, j)) };
  if (mode === 'vel') {
    const jv = r.signals.joint_vel as (number[] | null)[] | undefined;
    if (jv) return { names: Array.from({ length: width(jv) }, (_, j) => pn[j] || `q${j}`), rows: Array.from({ length: width(jv) }, (_, j) => col(jv, j)) };
    return null;
  }
  if (mode === 'error' && pos && tgt) {
    const shared = pn.some((n) => tn.includes(n));
    const m = Math.min(width(pos), width(tgt));
    const pairs = shared ? pn.map((n, j) => [j, tn.indexOf(n)]).filter(([, k]) => k >= 0) : Array.from({ length: m }, (_, j) => [j, j]);
    return {
      names: pairs.map(([j, k]) => (shared ? pn[j] : `${pn[j] || j}↔${tn[k] || k}`)),
      rows: pairs.map(([j, k]) => pos.map((f, i) => { const a = f?.[j], b = tgt[i]?.[k]; return typeof a === 'number' && typeof b === 'number' ? b - a : null; })),
    };
  }
  return null;
}
export function JointHeatmap({ sides, t, duration, onSeek }: P) {
  const r = sides[0].replay;
  const modes = (['pos', 'target', 'error', 'vel'] as JMode[]).filter((m) => jointMatrix(r, m));
  const [mode, setMode] = useState<JMode>(modes[0] || 'pos');
  const mat = useMemo(() => jointMatrix(r, mode), [r, mode]);
  const cv = useRef<HTMLCanvasElement>(null);
  const [ref, w] = useWidth();
  const rowH = 9, left = 96;
  useEffect(() => {
    const c = cv.current;
    if (!c || !mat) return;
    const n = mat.rows[0]?.length || 0;
    c.width = Math.max(1, Math.min(n, 1200));
    c.height = mat.rows.length;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(c.width, c.height);
    const diverging = mode === 'error' || mode === 'vel';
    mat.rows.forEach((row, j) => {
      const fin = row.filter((v): v is number => v !== null);
      const lo = Math.min(...fin), hi = Math.max(...fin), amax = Math.max(1e-9, ...fin.map(Math.abs));
      for (let px = 0; px < c.width; px++) {
        const v = row[Math.floor((px / c.width) * n)];
        const o = (j * c.width + px) * 4;
        if (v === null || v === undefined) { img.data[o + 3] = 0; continue; }
        let rr: number, gg: number, bb: number;
        if (diverging) {
          const a = Math.max(-1, Math.min(1, v / amax));
          // blue (+) ↔ grey ↔ red (−), CVD-safe pair
          const g0 = [128, 128, 124];
          const pole = a >= 0 ? [57, 135, 229] : [230, 103, 103];
          const k = Math.abs(a);
          [rr, gg, bb] = [g0[0] + (pole[0] - g0[0]) * k, g0[1] + (pole[1] - g0[1]) * k, g0[2] + (pole[2] - g0[2]) * k];
        } else {
          const k = hi > lo ? (v - lo) / (hi - lo) : 0.5;
          // sequential blue ramp
          [rr, gg, bb] = [205 - 192 * k, 226 - 172 * k, 251 - 144 * k];
        }
        img.data[o] = rr; img.data[o + 1] = gg; img.data[o + 2] = bb; img.data[o + 3] = 255;
      }
    });
    ctx.putImageData(img, 0, 0);
  }, [mat, mode]);
  if (!mat) return <Missing title="Joint heatmap" />;
  const H = mat.rows.length * rowH;
  const plotW = Math.max(50, w - left);
  const x = (tt: number) => left + (duration > 0 ? (tt / duration) * plotW : 0);
  const end = sides[0].times[sides[0].times.length - 1] || 0;
  return (
    <div ref={ref}>
      <div className="row small" style={{ marginBottom: 3 }}>
        <label>show <select value={mode} onChange={(e) => setMode(e.target.value as JMode)}>
          {modes.map((m) => <option key={m} value={m}>{{ pos: 'measured position', target: 'target', error: 'tracking error (target − position)', vel: 'velocity (recorded)' }[m]}</option>)}
        </select></label>
        <span className="muted">{mode === 'error' || mode === 'vel' ? 'blue + / red −, scaled per joint' : 'light → dark = low → high, scaled per joint'} · {sides.length > 1 ? 'replay A' : ''}</span>
      </div>
      <div style={{ position: 'relative', height: H }} onPointerDown={(e) => {
        const r0 = e.currentTarget.getBoundingClientRect();
        const px = e.clientX - r0.left - left;
        if (px >= 0) onSeek(Math.max(0, Math.min(duration, (px / plotW) * duration)));
      }}>
        {mat.names.map((n, j) => <div key={j} className="mono" title={n} style={{ position: 'absolute', left: 0, top: j * rowH, width: left - 4, height: rowH, fontSize: 8.5, lineHeight: `${rowH}px`, overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis', color: 'var(--ink-2)' }}>{n}</div>)}
        <canvas ref={cv} className="heat" style={{ position: 'absolute', left, top: 0, width: duration > 0 ? (end / duration) * plotW : plotW, height: H }} />
        <div style={{ position: 'absolute', left: x(t), top: 0, bottom: 0, width: 1.5, background: 'var(--ink)' }} />
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- phase portrait */
export function PhasePortrait({ sides, t }: P) {
  const r = sides[0].replay;
  const pos = r.signals.joint_pos;
  const names = (r.meta.joint_names as string[] | undefined) || [];
  const nj = Math.max(0, ...(pos || []).map((f) => (f ? f.length : 0)));
  const [j, setJ] = useState(0);
  const [ref, w] = useWidth();
  if (!pos || !nj) return <Missing title="Phase portrait" />;
  const recorded = r.signals.joint_vel as (number[] | null)[] | undefined;
  const times = sides[0].times;
  const q = pos.map((f) => (f && typeof f[j] === 'number' ? f[j] : null));
  const v = recorded
    ? recorded.map((f) => (f && typeof f[j] === 'number' ? f[j] : null))
    : q.map((x, i) => { const a = q[i - 1], b = q[i + 1]; const dt = (times[i + 1] ?? NaN) - (times[i - 1] ?? NaN); return a != null && b != null && dt > 0 ? (b - a) / dt : null; });
  const qs = q.filter((x): x is number => x !== null), vs = v.filter((x): x is number => x !== null);
  const H = 200, W = Math.min(w, 420);
  const [q0, q1] = [Math.min(...qs), Math.max(...qs)], [v0, v1] = [Math.min(...vs), Math.max(...vs)];
  const X = (x: number) => 30 + ((x - q0) / (q1 - q0 || 1)) * (W - 40), Y = (y: number) => H - 18 - ((y - v0) / (v1 - v0 || 1)) * (H - 28);
  let d = '', pen = false;
  q.forEach((x, i) => { const y = v[i]; if (x === null || y === null) { pen = false; return; } d += `${pen ? 'L' : 'M'}${X(x).toFixed(1)},${Y(y).toFixed(1)}`; pen = true; });
  const f = frameAt(times, t);
  return (
    <div ref={ref} className="row" style={{ alignItems: 'flex-start', gap: 12 }}>
      <svg width={W} height={H} role="img" aria-label={`phase portrait of ${names[j] || `joint ${j}`}`}>
        {v0 < 0 && v1 > 0 && <line x1={30} x2={W - 10} y1={Y(0)} y2={Y(0)} stroke="var(--grid)" />}
        <path d={d} fill="none" stroke="var(--s1)" strokeWidth={1.2} opacity={0.85} />
        {q[f] != null && v[f] != null && <circle cx={X(q[f]!)} cy={Y(v[f]!)} r={4.5} fill="var(--s2)" stroke="var(--surface)" />}
        <text x={30} y={H - 4} fontSize={10} fill="var(--muted)">position {fmtNum(q0)} … {fmtNum(q1)}</text>
        <text x={2} y={10} fontSize={10} fill="var(--muted)">vel {fmtNum(v1)}</text>
      </svg>
      <div className="small" style={{ display: 'grid', gap: 4 }}>
        <label>joint <select value={j} onChange={(e) => setJ(Number(e.target.value))}>{Array.from({ length: nj }, (_, k) => <option key={k} value={k}>{names[k] || `joint ${k}`}</option>)}</select></label>
        <span className="muted">{recorded ? 'velocity: recorded joint_vel' : 'velocity: central finite difference of recorded positions (computed here; joint_vel not recorded)'}</span>
        <span className="mono">now q {fmtNum(q[f])} · v {fmtNum(v[f])}</span>
        {sides.length > 1 && <span className="muted">replay A</span>}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- packet views */
export function pcaSpeed(r: Replay, times: number[]) {
  const p = r.signals.packet_pca;
  if (!p) return undefined;
  return p.map((x, i) => {
    const y = p[i - 1];
    const dt = times[i] - times[i - 1];
    return x && y && dt > 0 ? Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]) / dt : null;
  });
}
function packetShape(r: Replay) {
  const sh = r.meta.packet_shape as number[] | undefined;
  return Array.isArray(sh) && sh.length === 3 ? sh : null;
}
function flatDeep(v: unknown): number[] | null {
  if (!Array.isArray(v)) return null;
  const out: number[] = [];
  const walk = (x: unknown) => { if (Array.isArray(x)) x.forEach(walk); else out.push(typeof x === 'number' ? x : NaN); };
  walk(v);
  return out;
}
export function PacketHeatmap({ sides, t, duration, onSeek, signal = 'packet_z', label }: P & { signal?: string; label?: string }) {
  const r = sides[0].replay;
  const raw = r.signals[signal] as unknown[] | undefined;
  const z = useMemo(() => raw?.map(flatDeep), [raw]);
  const cv = useRef<HTMLCanvasElement>(null);
  const [ref, w] = useWidth();
  const sh = packetShape(r);
  const D = Math.max(0, ...(z || []).map((f) => (f ? f.length : 0)));
  useEffect(() => {
    const c = cv.current;
    if (!c || !z || !D) return;
    c.width = Math.min(z.length, 1200); c.height = D;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(c.width, c.height);
    const amax = Math.max(1e-9, ...z.flatMap((f) => (f || []).filter(Number.isFinite).map((v) => Math.abs(v))));
    for (let px = 0; px < c.width; px++) {
      const f = z[Math.floor((px / c.width) * z.length)];
      for (let d = 0; d < D; d++) {
        const o = (d * c.width + px) * 4;
        const v = f?.[d];
        if (typeof v !== 'number' || !Number.isFinite(v)) { img.data[o + 3] = 0; continue; }
        const k = Math.min(1, Math.abs(v) / amax);
        img.data[o] = 205 - 192 * k; img.data[o + 1] = 226 - 172 * k; img.data[o + 2] = 251 - 144 * k; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [z, D]);
  if (!z || !D) return null;
  const H = Math.min(260, Math.max(80, D * 2));
  const x = (tt: number) => (duration > 0 ? (tt / duration) * w : 0);
  const norm = z.map((f) => (f ? Math.sqrt(f.filter(Number.isFinite).reduce((a, v) => a + v * v, 0)) : null));
  const events = signal === 'packet_z' && Array.isArray(r.signals.packet_events) ? (r.signals.packet_events as { t: number; edit?: unknown }[]) : [];
  const t0 = r.frames.t[0] || 0;
  return (
    <div ref={ref}>
      <div style={{ position: 'relative', height: H }} onPointerDown={(e) => { const r0 = e.currentTarget.getBoundingClientRect(); onSeek(((e.clientX - r0.left) / r0.width) * duration); }}>
        <canvas ref={cv} className="heat" style={{ position: 'absolute', inset: 0, width: '100%', height: H }} />
        {events.map((e, i) => <div key={i} title={`new packet at ${fmtNum(e.t - t0)} s${e.edit ? ' (edited)' : ''}`} style={{ position: 'absolute', left: x(e.t - t0), top: 0, height: 4, width: 1.5, background: e.edit ? 'var(--s2)' : 'var(--ink-2)' }} />)}
        <div style={{ position: 'absolute', left: x(t), top: 0, bottom: 0, width: 1.5, background: 'var(--ink)' }} />
      </div>
      <p className="small muted" style={{ margin: '2px 0' }}>{label || '|z|'} over time ({D} values{sh && signal === 'packet_z' ? ` = ${sh[0]} knots × ${sh[1]} assemblies × ${sh[2]} dims` : ''}; rows in recorded order){events.length ? ` · ticks = new packets (${events.length})` : ''}</p>
      {signal === 'packet_z' && <ScalarTrack title="Packet norm ‖z‖ of the recorded dims (computed here)" get={(rr) => (rr === r ? norm : undefined)} {...{ sides: sides.slice(0, 1), t, duration, onSeek }} height={44} />}
    </div>
  );
}

/* ---------------------------------------------------------------- probe vs truth + calibration */
export function hasProbeTruth(r: Replay) {
  return isObj(r.signals.probe_truth);
}
function toNum(xs: unknown[]): (number | (number | null)[] | null)[] {
  const one = (v: unknown) => (typeof v === 'boolean' ? (v ? 1 : 0) : typeof v === 'number' && Number.isFinite(v) ? v : null);
  return xs.map((v) => (Array.isArray(v) ? v.map(one) : one(v)));
}
export function ProbeTruth({ sides, t, duration, onSeek }: P) {
  const r = sides[0].replay;
  const pt = r.signals.probe_truth as Record<string, unknown[]> | undefined;
  const pr = r.signals.probe as Record<string, unknown[]> | undefined;
  if (!pt || !pr) return null;
  const keys = Object.keys(pt).filter((k) => pr[k]);
  const truthReplay = { ...r } as Replay;
  return (
    <div>
      {keys.map((k) => {
        const pred = pr[k], truth = pt[k];
        const flat = (xs: unknown[]) => xs.map((v) => (Array.isArray(v) ? v.map(Number) : [Number(v)]));
        const P0 = flat(pred), T0 = flat(truth);
        const binary = T0.every((row) => row.every((v) => v === 0 || v === 1 || Number.isNaN(v)));
        // calibration: bin predicted probability, observed frequency of truth (binary keys only)
        const bins = Array.from({ length: 10 }, () => ({ n: 0, pos: 0 }));
        if (binary) P0.forEach((row, i) => row.forEach((p, j) => { const tv = T0[i]?.[j]; if (!Number.isFinite(p) || !Number.isFinite(tv)) return; const b = Math.min(9, Math.max(0, Math.floor(p * 10))); bins[b].n++; bins[b].pos += tv; }));
        return (
          <div key={k} style={{ marginBottom: 6 }}>
            <ScalarTrack title={`${k}: probe (solid) vs truth (dashed, privileged)`} sides={[sides[0], { ...sides[0], replay: truthReplay, tag: 'truth', color: 'var(--s2)' }]}
              get={(rr) => (rr === r ? toNum(pred) : rr === truthReplay ? toNum(truth) : undefined)} t={t} duration={duration} onSeek={onSeek} />
            {binary && (
              <div className="row small" style={{ gap: 2, alignItems: 'flex-end' }} title="calibration: observed frequency of the truth in each predicted-probability decile (computed here)">
                <span className="muted" style={{ width: 90 }}>calibration</span>
                {bins.map((b, i) => (
                  <div key={i} style={{ width: 22, height: 26, background: 'var(--surface-3)', position: 'relative' }} title={`p∈[${i / 10}, ${(i + 1) / 10}): ${b.n ? `${((b.pos / b.n) * 100).toFixed(0)}% true of ${b.n}` : 'no frames'}`}>
                    {b.n > 0 && <i style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: `${(b.pos / b.n) * 100}%`, background: 'var(--s1)' }} />}
                    <i style={{ position: 'absolute', left: 0, right: 0, bottom: `${(i + 0.5) * 10}%`, height: 1, background: 'var(--ink-2)' }} />
                  </div>
                ))}
                <span className="muted">bar = observed · tick = ideal</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- task-event Gantt */
export function EventGantt({ sides, t, duration, onSeek }: P) {
  const all = sides.map((s) => {
    const t0 = s.replay.frames.t[0] || 0;
    const end = s.times[s.times.length - 1] || 0;
    const ev = s.replay.signals.task_events || [];
    const names = [...new Set(ev.map((e) => e.event))];
    return { s, names, spans: names.map((n) => {
      const xs = ev.filter((e) => e.event === n).map((e) => ({ t: e.t - t0, status: e.status }));
      return xs.map((e, i) => ({ a: e.t, b: xs[i + 1]?.t ?? end, status: e.status }));
    }) };
  });
  if (all.every((a) => !a.names.length)) return <Missing title="Task events" />;
  const rowH = 14;
  const H = all.reduce((a, x) => a + x.names.length * rowH + 6, 0);
  const statuses = [...new Set(all.flatMap((a) => a.spans.flat().map((s) => s.status)))];
  const col = (st: string) => (/succe|done|complete|achiev/.test(st) ? 'var(--good)' : /fail|abort|violat/.test(st) ? 'var(--critical)' : /active|run/.test(st) ? 'var(--accent)' : /pend|wait/.test(st) ? 'var(--axis)' : seriesColor(st, statuses));
  return (
    <Track title="" t={t} duration={duration} onSeek={onSeek} sides={sides} height={H}
      legend={<span className="legend small">{statuses.map((s) => <span key={s}><i className="sw" style={{ background: col(s) }} />{s}</span>)}</span>}>
      {(x) => {
        let off = 0;
        return all.map((g, gi) => {
          const y0 = off;
          off += g.names.length * rowH + 6;
          return g.names.map((n, k) => (
            <g key={`${gi}-${n}`}>
              {g.spans[k].map((sp, i) => (
                <rect key={i} x={x(sp.a)} width={Math.max(1, x(sp.b) - x(sp.a))} y={y0 + k * rowH} height={rowH - 2} fill={col(sp.status)} opacity={0.85}>
                  <title>{`${n}: ${sp.status} ${fmtNum(sp.a)}–${fmtNum(sp.b)} s`}</title>
                </rect>
              ))}
              <text x={3} y={y0 + k * rowH + 9.5} fontSize={9.5} fill="#fff" style={{ paintOrder: 'stroke', stroke: 'rgba(0,0,0,.55)', strokeWidth: 2.5 }}>{sides.length > 1 ? `${g.s.tag} ` : ''}{n}</text>
            </g>
          ));
        });
      }}
    </Track>
  );
}

/* ---------------------------------------------------------------- edit timeline & compare differences */
function diffSeries(a: Side, b: Side, get: (r: Replay) => (number | null)[] | undefined) {
  const va = get(a.replay), vb = get(b.replay);
  if (!va || !vb) return undefined;
  return a.times.map((tt, i) => {
    const x = va[i], y = vb[frameAt(b.times, tt)];
    return x != null && y != null && tt <= (b.times[b.times.length - 1] ?? 0) ? y - x : null;
  });
}
export const DIFFS: [string, string, (r: Replay) => (number | null)[] | undefined][] = [
  ['forward progress', 'm', (r) => numArr(r.signals.forward_progress) ?? undefined],
  ['object height', 'm', (r) => r.signals.object_pose?.map((p) => (p && p.length >= 3 ? p[2] : null))],
  ['base x', 'm', (r) => { const i = r.meta.base_body ? r.bodies.indexOf(r.meta.base_body) : -1; return i >= 0 ? r.frames.body_pos.map((f) => f[i]?.[0] ?? null) : undefined; }],
];
export function EditPanel({ sides, t, duration, onSeek }: P) {
  const edited = sides.filter((s) => s.replay.signals.edit_active || s.replay.meta.t_edit != null);
  return (
    <div>
      {edited.map((s) => {
        const ef = editFrame(s.replay, s.times);
        return (
          <div key={s.tag} className="small" style={{ marginBottom: 3 }}>
            <b style={{ color: s.color }}>{sides.length > 1 ? s.tag : ''}</b> edit <b>{str(s.replay.meta.edit) || 'context/packet edit'}</b>
            {ef >= 0 ? <> from <b className="mono">{fmtNum(s.times[ef])} s</b></> : ' (no onset recorded)'}
            {s.replay.meta.edit_row ? <span className="muted"> · eval row: <code>{JSON.stringify(s.replay.meta.edit_row).slice(0, 180)}</code></span> : null}
          </div>
        );
      })}
      {sides.some((s) => s.replay.signals.edit_dz_norm) && <ScalarTrack title="|edited − unedited packet| (same flow noise)" noteKey="edit_dz_norm" get={(rr) => rr.signals.edit_dz_norm as (number | null)[] | undefined} sides={sides} t={t} duration={duration} onSeek={onSeek} />}
      {sides.length > 1 && DIFFS.map(([label, unit, get]) => {
        const d = diffSeries(sides[0], sides[1], get);
        if (!d || d.every((v) => v === null)) return null;
        return <ScalarTrack key={label} title={`B − A · ${label} (computed here, same clock)`} unit={unit} sides={[{ ...sides[0], color: 'var(--s7)' }]} get={(r) => (r === sides[0].replay ? d : undefined)} t={t} duration={duration} onSeek={onSeek} />;
      })}
      {sides.length < 2 && <p className="small muted">Pick a comparison run (B) in the sidebar, e.g. the unedited episode on the same seed, to see B − A difference traces.</p>}
    </div>
  );
}
export function hasEdit(r: Replay) {
  return !!r.signals.edit_active || r.meta.t_edit != null;
}

/* ---------------------------------------------------------------- phase + progress lanes */
export function PhaseLanes({ sides, t, duration, onSeek }: P) {
  const data = sides.map((s) => s.replay.signals.phase);
  if (data.every((d) => !d)) return null;
  const laneH = 16;
  const cats = [...new Set(data.flatMap((d) => (d || []).map((v) => str(v))))].filter(Boolean);
  return (
    <Track title="" t={t} duration={duration} onSeek={onSeek} sides={sides} height={sides.length * (laneH + 3)}>
      {(x) => sides.map((s, si) => (data[si] ? segments(data[si]!.map((v) => str(v))).map((seg) => {
        const x0 = x(s.times[seg.s]), x1 = x(s.times[Math.min(seg.e, s.times.length - 1)]);
        return (
          <g key={`${si}-${seg.s}`}>
            <rect x={x0} width={Math.max(1, x1 - x0)} y={si * (laneH + 3)} height={laneH} fill={seg.v ? seriesColor(seg.v, cats) : 'transparent'} opacity={0.8}><title>{seg.v}</title></rect>
            {x1 - x0 > 44 && <text x={x0 + 3} y={si * (laneH + 3) + 11.5} fontSize={10} fill="#fff">{seg.v}</text>}
          </g>
        );
      }) : null))}
    </Track>
  );
}
export { linePath, Missing };
export type { Clock };
