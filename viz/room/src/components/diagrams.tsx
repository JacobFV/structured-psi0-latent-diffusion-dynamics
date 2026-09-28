/**
 * Run diagrams (after IBM-2's encoding-flow viewer): a pipeline flow with live mini-plots at the shared cursor, a
 * morphology stick diagram from recorded body positions, and the packet structure (knots × assemblies with probe heads).
 * Everything drawn is a recorded field of the replay; what is inferred (limb grouping from body names) is labelled.
 */
import type { ReactNode } from 'react';
import { fmtNum, isObj, num, str } from '../lib/format';
import { frameAt, type Replay } from '../lib/replay';
import { useWidth } from './Timelines';

const at = <T,>(xs: T[] | undefined, f: number): T | undefined => (xs ? xs[Math.min(f, xs.length - 1)] : undefined);
function lastDefined<T>(xs: (T | null | undefined)[] | undefined, f: number): T | undefined {
  if (!xs) return undefined;
  for (let i = Math.min(f, xs.length - 1); i >= 0; i--) if (xs[i] !== null && xs[i] !== undefined) return xs[i] as T;
  return undefined;
}
function editTarget(r: Replay): 'context' | 'packet' | null {
  const e = str(r.meta.edit);
  if (!e && !r.signals.edit_active) return null;
  return /^z[_-]|packet/i.test(e) || /packet/.test(str(r.meta.context_or_packet)) ? 'packet' : 'context';
}

/* ---------------------------------------------------------------- mini widgets */
function MiniBars({ values, w = 96, h = 30, signed = true }: { values: (number | null)[]; w?: number; h?: number; signed?: boolean }) {
  const fin = values.map((v) => (v === null || !Number.isFinite(v) ? 0 : v));
  const m = Math.max(1e-9, ...fin.map(Math.abs));
  const bw = w / Math.max(1, fin.length);
  return (
    <svg width={w} height={h} role="img" aria-label="current values">
      {signed && <line x1={0} x2={w} y1={h / 2} y2={h / 2} stroke="var(--grid)" />}
      {fin.map((v, i) => {
        const hh = (Math.abs(v) / m) * (signed ? h / 2 - 1 : h - 1);
        const y = signed ? (v >= 0 ? h / 2 - hh : h / 2) : h - hh;
        return <rect key={i} x={i * bw + 0.5} y={y} width={Math.max(1, bw - 1)} height={hh} fill={v >= 0 ? 'var(--up)' : 'var(--down)'} />;
      })}
    </svg>
  );
}
function MiniGrid({ cells, rows, cols, w = 96, h = 30, hi }: { cells: (number | null)[]; rows: number; cols: number; w?: number; h?: number; hi?: boolean }) {
  const m = Math.max(1e-9, ...cells.map((v) => Math.abs(v ?? 0)));
  const cw = w / cols, ch = h / rows;
  return (
    <svg width={w} height={h} role="img" aria-label={`${rows}×${cols} grid`}>
      {cells.map((v, i) => {
        const k = v === null ? 0 : Math.abs(v) / m;
        return <rect key={i} x={(i % cols) * cw} y={Math.floor(i / cols) * ch} width={cw - 0.5} height={ch - 0.5} fill={v === null ? 'var(--surface-3)' : `color-mix(in srgb, var(--s1) ${Math.round(15 + 85 * k)}%, var(--surface))`} stroke={hi ? 'var(--s2)' : 'none'} />;
      })}
    </svg>
  );
}
function MiniPath({ pts, cur, w = 96, h = 30 }: { pts: (number[] | null)[]; cur: number; w?: number; h?: number }) {
  const f = pts.filter(Boolean) as number[][];
  if (f.length < 2) return null;
  const xs = f.map((p) => p[0]), ys = f.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const s = Math.min((w - 4) / (x1 - x0 || 1), (h - 4) / (y1 - y0 || 1));
  const X = (x: number) => 2 + (x - x0) * s, Y = (y: number) => h - 2 - (y - y0) * s;
  const d = f.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
  const c = lastDefined(pts, cur);
  return (
    <svg width={w} height={h} role="img" aria-label="top-down path">
      <path d={d} fill="none" stroke="var(--ink-2)" strokeWidth={1} />
      {c && <circle cx={X(c[0])} cy={Y(c[1])} r={2.5} fill="var(--s2)" />}
    </svg>
  );
}

/* ---------------------------------------------------------------- (i) pipeline flow */
type Stage = { id: string; title: string; sub: string; body: ReactNode; edit?: boolean; tip: string };
export function hasPipeline(r: Replay) {
  return !!(r.signals.joint_target || r.signals.packet_pca || r.signals.packet_z);
}
export function PipelineFlow({ r, times, t }: { r: Replay; times: number[]; t: number }) {
  const [ref, w] = useWidth();
  const f = frameAt(times, t);
  const m = r.meta;
  const latent = !!(r.signals.packet_pca || r.signals.packet_z || m.packet_pca_basis);
  const editOn = !!at(r.signals.edit_active, f);
  const target = editTarget(r);
  const ev = (r.signals.task_events || []).filter((e) => e.t - (r.frames.t[0] || 0) <= t);
  const lastStatus = new Map<string, string>();
  ev.forEach((e) => lastStatus.set(e.event, e.status));
  const shape = Array.isArray(m.packet_shape) ? (m.packet_shape as number[]) : null;
  const pn = lastDefined(r.signals.packet_norm as (number[][] | null)[] | undefined, f);
  const pz = lastDefined(r.signals.packet_z as (number[] | null)[] | undefined, f);
  const pca = lastDefined(r.signals.packet_pca, f);
  const tgt = at(r.signals.joint_target, f) || null;
  const pos = at(r.signals.joint_pos, f) || null;
  const bb = m.base_body ? r.bodies.indexOf(m.base_body) : -1;
  const path = r.signals.object_pose && m.family !== 'legged' ? r.signals.object_pose.map((p) => (p ? [p[0], p[1]] : null)) : bb >= 0 ? r.frames.body_pos.map((fr) => (fr[bb] ? [fr[bb][0], fr[bb][1]] : null)) : [];
  const nEvents = Array.isArray(r.signals.packet_events) ? (r.signals.packet_events as { t: number }[]).filter((e) => e.t - (r.frames.t[0] || 0) <= t).length : null;
  const stages: Stage[] = [
    { id: 'ctx', title: 'task context', sub: [...lastStatus.entries()].slice(-2).map(([k, v]) => `${k}:${v}`).join(' · ') || str(m.task),
      body: <div className="fd-events">{[...lastStatus.entries()].slice(0, 4).map(([k, v]) => <span key={k} className={`status ${/succe/.test(v) ? 'good' : /fail/.test(v) ? 'critical' : /active/.test(v) ? 'active' : 'neutral'}`}><i />{k}</span>)}</div>,
      edit: editOn && target === 'context', tip: 'public task context: task events and their status at the cursor' },
  ];
  if (latent) {
    stages.push({ id: 'sys1', title: 'system i (flow)', sub: nEvents !== null ? `${nEvents} packets so far` : 'generates packets', body: <span className="mono">{str(m.route)}</span>, tip: str(m.source_label) });
    stages.push({
      id: 'z', title: 'packet z', sub: shape ? `${shape[0]} knots × ${shape[1]} assemblies` : pca ? 'PCA of the packet' : 'not recorded',
      body: pn && shape ? <MiniGrid cells={pn.flat()} rows={shape[0]} cols={shape[1]} hi={editOn && target === 'packet'} />
        : pz && shape ? <MiniGrid cells={pz.slice(0, shape[0] * shape[1] * shape[2]).filter((_, i) => i % shape[2] === 0)} rows={shape[0]} cols={shape[1]} />
          : pca ? <MiniBars values={pca} /> : <span className="muted">—</span>,
      edit: editOn && target === 'packet', tip: 'the packet system 0 executes at the cursor (norm per knot × assembly, or its PCA coordinates)',
    });
    stages.push({ id: 'sys0', title: 'system 0', sub: 'packet → joint targets', body: <span className="mono">{Object.keys(isObj(m.ckpt_sha) ? m.ckpt_sha : {}).join(' + ') || '—'}</span>, tip: 'the deployable controller that executes the packet' });
  } else {
    stages.push({ id: 'pol', title: /teacher/i.test(str(m.route) + str(m.source_label)) ? 'scripted teacher' : 'policy', sub: str(m.route), body: <span className="mono">{str(m.variant)}</span>, tip: `${str(m.source_label)} (no packet: this route has no system i / packet)` });
  }
  stages.push({ id: 'tgt', title: 'joint targets', sub: tgt ? `${tgt.length} commands` : 'not recorded', body: tgt ? <MiniBars values={tgt.map((v, i) => (pos && typeof pos[i] === 'number' ? v - pos[i] : v))} /> : <span className="muted">—</span>, tip: 'current commands minus measured positions (tracking error) per actuator' });
  if (m.tracker) stages.push({ id: 'trk', title: 'tracker', sub: 'targets → torques', body: <span className="mono" title={str(m.tracker)}>{str(m.tracker).split(':').slice(-3, -1).join(':')}</span>, tip: str(m.tracker) });
  stages.push({ id: 'robot', title: 'robot', sub: `${str(m.body)}${m.fell === true ? ' · fell' : ''}`, body: path.length ? <MiniPath pts={path} cur={f} /> : <span className="muted">—</span>, tip: 'top-down path of the base (legged) or the object (arm) with the current position' });

  const gap = 26, bw = Math.max(92, Math.min(150, (w - gap * (stages.length - 1)) / stages.length));
  return (
    <div ref={ref} className="flowd">
      <div className="fd-row" style={{ gridTemplateColumns: stages.map(() => `${bw}px`).join(` ${gap}px `) }}>
        {stages.map((s, i) => (
          <FragStage key={s.id} s={s} arrow={i < stages.length - 1} animate={!!r.signals.joint_target} />
        ))}
      </div>
      {target && <div className="fd-note"><span className="fd-edit">edit</span> {str(m.edit) || 'edit'} enters at the {target === 'packet' ? 'packet' : 'task context'}{editOn ? ' · active now' : m.edit_onset_t != null ? ` · from ${fmtNum(num(m.edit_onset_t))} s` : ''}</div>}
    </div>
  );
}
function FragStage({ s, arrow, animate }: { s: Stage; arrow: boolean; animate: boolean }) {
  return (
    <>
      <div className={`fd-stage ${s.edit ? 'edit' : ''}`} title={s.tip}>
        <b>{s.title}</b>
        <div className="fd-body">{s.body}</div>
        <small>{s.sub}</small>
        {s.edit && <span className="fd-edit">edit ↓</span>}
      </div>
      {arrow && (
        <svg className="fd-arrow" width="26" height="12" viewBox="0 0 26 12" aria-hidden>
          <line x1={0} x2={20} y1={6} y2={6} className={animate ? 'flowing' : ''} />
          <path d="M20,2 L26,6 L20,10 Z" />
        </svg>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- (ii) morphology */
function limbOf(name: string) {
  const n = name.replace(/^r\d_/, '');
  const m = /^(LF|RF|LH|RH|FL|FR|RL|RR|L|R|left|right)[_]/i.exec(n) || /^(leg\d+|arm\d*|link|wrist|finger)/i.exec(n);
  return m ? m[1].toLowerCase() : 'trunk';
}
export function hasMorphology(r: Replay) {
  return r.bodies.filter((b) => b !== 'world').length >= 3;
}
export function Morphology({ r, times, t }: { r: Replay; times: number[]; t: number }) {
  const [ref, w] = useWidth();
  const f = frameAt(times, t);
  const P = r.frames.body_pos[f] || [];
  const names = r.bodies;
  const skip = new Set(['world', r.meta.object_body, 'target_zone', 'cube']);
  const idx = names.map((n, i) => [n, i] as const).filter(([n]) => !skip.has(n));
  const bb = r.meta.base_body ? names.indexOf(r.meta.base_body) : idx[0]?.[1] ?? -1;
  const pts = idx.map(([, i]) => P[i]).filter(Boolean);
  if (pts.length < 3) return null;
  const view = r.meta.family === 'legged' ? ([0, 2] as const) : ([0, 2] as const); // side view x–z
  const xs = pts.map((p) => p[view[0]]), zs = pts.map((p) => p[view[1]]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), z0 = Math.min(...zs), z1 = Math.max(...zs);
  const H = 200, W = Math.min(w, 420);
  const s = Math.min((W - 30) / (x1 - x0 || 1), (H - 30) / (z1 - z0 || 1));
  const X = (x: number) => 15 + (x - x0) * s, Y = (z: number) => H - 15 - (z - z0) * s;
  // edges: consecutive bodies of the same limb (MuJoCo body order is depth-first), each limb's first body to the base
  const edges: [number, number][] = [];
  const lastOfLimb = new Map<string, number>();
  for (const [n, i] of idx) {
    if (i === bb) continue;
    const limb = limbOf(n);
    const prev = lastOfLimb.get(limb);
    edges.push([prev ?? bb, i]);
    lastOfLimb.set(limb, i);
  }
  const cb = (r.meta.contact_bodies || []) as string[];
  const flags = at(r.signals.contacts, f) as (boolean | number)[] | undefined;
  const force = at(r.signals.contact_force as (number[] | null)[] | undefined, f) || null;
  const torque = at(r.signals.joint_torque as (number[] | null)[] | undefined, f) || null;
  const tnames = (r.meta.joint_torque_names as string[] | undefined) || [];
  const limbTorque = new Map<string, number>();
  if (torque) tnames.forEach((n, k) => { const l = limbOf(n); limbTorque.set(l, (limbTorque.get(l) || 0) + Math.abs(torque[k] || 0)); });
  const tmax = Math.max(1e-9, ...limbTorque.values());
  return (
    <div ref={ref} className="row" style={{ alignItems: 'flex-start', gap: 12 }}>
      <svg width={W} height={H} role="img" aria-label="morphology, side view">
        <line x1={0} x2={W} y1={Y(0)} y2={Y(0)} stroke="var(--grid)" />
        {edges.map(([a, b], k) => {
          const pa = P[a], pb = P[b];
          if (!pa || !pb) return null;
          const lt = limbTorque.get(limbOf(names[b]));
          const col = lt !== undefined ? `color-mix(in srgb, var(--s2) ${Math.round(20 + 80 * (lt / tmax))}%, var(--ink-2))` : 'var(--ink-2)';
          return <line key={k} x1={X(pa[view[0]])} y1={Y(pa[view[1]])} x2={X(pb[view[0]])} y2={Y(pb[view[1]])} stroke={col} strokeWidth={2} />;
        })}
        {idx.map(([n, i]) => {
          const p = P[i];
          if (!p) return null;
          const c = cb.indexOf(n);
          const on = c >= 0 && flags ? !!flags[c] : false;
          const fz = c >= 0 && force ? force[c] : null;
          return (
            <g key={n}>
              <title>{`${n}${c >= 0 ? ` · contact ${on ? 'yes' : 'no'}${fz !== null && fz !== undefined ? ` · ${fmtNum(fz)} N` : ''}` : ''}`}</title>
              <circle cx={X(p[view[0]])} cy={Y(p[view[1]])} r={i === bb ? 5 : c >= 0 ? 4 : 2.5} fill={on ? 'var(--s2)' : i === bb ? 'var(--ink)' : 'var(--surface)'} stroke="var(--ink-2)" strokeWidth={1} />
            </g>
          );
        })}
      </svg>
      <div className="small" style={{ display: 'grid', gap: 3, maxWidth: 220 }}>
        <span><i className="sw" style={{ background: 'var(--s2)' }} /> contact body in contact{force ? ' (hover: normal force)' : ''}</span>
        {torque && <span><i className="sw" style={{ background: 'var(--s2)', opacity: 0.6 }} /> limb shade = Σ|torque| of its actuators</span>}
        <span className="muted">side view (x–z) of recorded body positions; limbs grouped from body names and order (inferred)</span>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- (iv) packet structure */
export function hasPacketStructure(r: Replay) {
  return Array.isArray(r.meta.packet_shape) && !!(r.signals.packet_norm || r.signals.packet_z);
}
export function PacketStructure({ r, times, t }: { r: Replay; times: number[]; t: number }) {
  const f = frameAt(times, t);
  const shape = r.meta.packet_shape as number[];
  const [K, A] = shape;
  const pn = lastDefined(r.signals.packet_norm as (number[][] | null)[] | undefined, f);
  const pz = lastDefined(r.signals.packet_z as (number[] | null)[] | undefined, f);
  const val = (k: number, a: number) => {
    if (pn && pn[k] && typeof pn[k][a] === 'number') return pn[k][a];
    if (pz) { const base = (k * A + a) * shape[2]; const xs = pz.slice(base, base + shape[2]); return Math.sqrt(xs.reduce((s, v) => s + v * v, 0)); }
    return null;
  };
  const all: number[] = [];
  for (let k = 0; k < K; k++) for (let a = 0; a < A; a++) { const v = val(k, a); if (v !== null) all.push(v); }
  const m = Math.max(1e-9, ...all);
  const probe = r.signals.probe || {};
  const heads = Object.keys(probe);
  const aNames = (r.meta.assemblies as string[] | undefined) || (r.meta.family === 'legged' ? [...((r.meta.contact_bodies as string[]) || []).map((n) => n.replace(/^r\d_/, '')), 'body'] : []);
  const cell = 26;
  const W = 90 + A * cell + 180, H = 18 + K * cell + 8;
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg width={W} height={H} role="img" aria-label={`packet ${K} knots by ${A} assemblies`}>
        {Array.from({ length: A }, (_, a) => <text key={a} x={90 + a * cell + cell / 2} y={12} textAnchor="middle" fill="var(--muted)">{(aNames[a] || `a${a}`).slice(0, 5)}</text>)}
        {Array.from({ length: K }, (_, k) => (
          <g key={k}>
            <text x={84} y={18 + k * cell + cell / 2 + 3} textAnchor="end" fill="var(--muted)">knot {k}</text>
            {Array.from({ length: A }, (_, a) => {
              const v = val(k, a);
              return <rect key={a} x={90 + a * cell} y={18 + k * cell} width={cell - 2} height={cell - 2} fill={v === null ? 'var(--surface-3)' : `color-mix(in srgb, var(--s1) ${Math.round(12 + 88 * (v / m))}%, var(--surface))`}><title>{`knot ${k} · ${aNames[a] || `assembly ${a}`}: ‖z‖ ${fmtNum(v)}`}</title></rect>;
            })}
          </g>
        ))}
        {heads.map((h, i) => {
          const pv = lastDefined(probe[h] as unknown[], f);
          const y = 18 + i * 16 + 8;
          const txt = Array.isArray(pv) ? pv.map((x) => (typeof x === 'boolean' ? (x ? '1' : '0') : fmtNum(x))).join(' ') : typeof pv === 'boolean' ? (pv ? 'yes' : 'no') : fmtNum(pv);
          return (
            <g key={h}>
              <line x1={90 + A * cell} x2={90 + A * cell + 14} y1={y - 3} y2={y - 3} stroke="var(--axis)" />
              <text x={90 + A * cell + 18} y={y} fill="var(--ink-2)">{h} head: {txt.slice(0, 26)}</text>
            </g>
          );
        })}
      </svg>
      <div className="small muted">cell = ‖z‖ of one knot × assembly at the cursor{pn ? ' (full dz, recorded packet_norm)' : ' (first recorded dims of packet_z)'} · probe heads read the same packet (diagnostics)</div>
    </div>
  );
}
