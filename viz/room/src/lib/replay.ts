/** rrp-viz/replay/v1 types (viz/CONTRACT.md) and a shared playback clock. */
import { useEffect, useState } from 'react';

export type Geom = {
  name: string; body: string; type: 'plane' | 'box' | 'sphere' | 'capsule' | 'cylinder' | 'ellipsoid' | 'mesh';
  size: number[]; rgba: number[]; pos?: number[]; quat?: number[]; mesh?: { vertices: number[]; faces: number[]; name?: string; source_faces?: number };
};
export type Replay = {
  schema: string; id: string;
  meta: {
    family?: string; task?: string; body?: string; route?: string; source_label?: string; ckpt_sha?: string | null; variant?: string;
    seed?: number | string; condition?: string; success?: boolean | null; failure_stage?: string | null;
    physics?: Record<string, unknown>; decision_refs?: string[]; caveat?: string; contact_bodies?: string[]; base_body?: string;
    object_body?: string; joint_names?: string[]; packet_pca_basis?: Record<string, unknown>; packet_pca?: Record<string, unknown>;
    fixture?: boolean; [k: string]: unknown;
  };
  fps: number; n_frames: number; geoms: Geom[]; bodies: string[];
  frames: { t: number[]; body_pos: number[][][]; body_quat: number[][][] };
  signals: {
    joint_target?: (number[] | null)[]; joint_pos?: (number[] | null)[]; contacts?: (boolean[] | number[] | null)[];
    packet_pca?: (number[] | null)[];
    probe?: Record<string, unknown[]>;
    phase?: (string | number | null)[]; task_events?: { t: number; event: string; status: string }[];
    edit_active?: (boolean | null)[]; forward_progress?: (number | null)[]; object_pose?: (number[] | null)[];
    penetration_mm?: (number | null)[]; slip?: (number | null)[]; [k: string]: unknown;
  };
  annotations?: { t: number; text: string }[];
};

/** Times relative to the first frame. */
export function relTimes(r: Replay) {
  const t = r.frames.t;
  const t0 = t.length ? t[0] : 0;
  return t.map((x) => x - t0);
}
/** Index of the last frame with time ≤ t (binary search). */
export function frameAt(times: number[], t: number) {
  let lo = 0, hi = times.length - 1;
  if (hi < 0) return 0;
  if (t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}
/** Runs of truthy values as [startIndex, endIndexExclusive]. */
export function runs<T>(xs: T[], on: (x: T) => boolean) {
  const out: [number, number][] = [];
  let s = -1;
  xs.forEach((x, i) => {
    if (on(x)) { if (s < 0) s = i; } else if (s >= 0) { out.push([s, i]); s = -1; }
  });
  if (s >= 0) out.push([s, xs.length]);
  return out;
}
/** Runs of equal categorical values. */
export function segments<T>(xs: T[]) {
  const out: { v: T; s: number; e: number }[] = [];
  xs.forEach((x, i) => {
    const last = out[out.length - 1];
    if (last && JSON.stringify(last.v) === JSON.stringify(x)) last.e = i + 1;
    else out.push({ v: x, s: i, e: i + 1 });
  });
  return out;
}

type Listener = () => void;
/** Playback clock shared by the 3D stages and the timelines; the 3D stages read it every animation frame. */
export class Clock {
  t = 0;
  playing = false;
  speed = 1;
  loop = true;
  duration = 0;
  private listeners = new Set<Listener>();
  private raf = 0;
  private last = 0;
  subscribe(fn: Listener) { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; }
  private emit() { this.listeners.forEach((f) => f()); }
  set(t: number) {
    this.t = Math.max(0, Math.min(this.duration, t));
    this.emit();
  }
  setDuration(d: number) { this.duration = Math.max(0, d); if (this.t > this.duration) this.t = this.duration; this.emit(); }
  play() {
    if (this.playing) return;
    if (this.t >= this.duration) this.t = 0;
    this.playing = true;
    this.last = performance.now();
    const step = (now: number) => {
      if (!this.playing) return;
      const dt = Math.min(0.1, (now - this.last) / 1000);
      this.last = now;
      let t = this.t + dt * this.speed;
      if (t >= this.duration) {
        if (this.loop && this.duration > 0) t = t % this.duration;
        else { t = this.duration; this.playing = false; }
      }
      this.t = t;
      this.emit();
      if (this.playing) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
    this.emit();
  }
  pause() { this.playing = false; cancelAnimationFrame(this.raf); this.emit(); }
  toggle() { if (this.playing) this.pause(); else this.play(); }
  dispose() { this.pause(); this.listeners.clear(); }
}

/** React view of the clock, throttled (timelines do not need 60 Hz). */
export function useClock(clock: Clock, hz = 24) {
  const [snap, setSnap] = useState({ t: clock.t, playing: clock.playing, speed: clock.speed, loop: clock.loop, duration: clock.duration });
  useEffect(() => {
    let last = 0, timer = 0;
    const read = () => setSnap({ t: clock.t, playing: clock.playing, speed: clock.speed, loop: clock.loop, duration: clock.duration });
    const unsub = clock.subscribe(() => {
      const now = performance.now();
      if (!clock.playing || now - last > 1000 / hz) { last = now; read(); }
      else { window.clearTimeout(timer); timer = window.setTimeout(read, 1000 / hz); }
    });
    return () => { unsub(); window.clearTimeout(timer); };
  }, [clock, hz]);
  return snap;
}
