/**
 * three.js replay stage. Geometry comes only from the replay's `geoms` (MuJoCo size semantics, z-up, local pose in the
 * body frame); bodies are posed from recorded `body_pos`/`body_quat`. Contact markers need `meta.contact_bodies`;
 * without it no marker is drawn (the raster still shows the flags). Trails show recorded positions only.
 * Renders on demand (clock ticks, orbit, resize), not continuously.
 */
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { frameAt, relTimes, type Clock, type Geom, type Replay } from '../lib/replay';

function geomObject(g: Geom): THREE.Mesh | null {
  const s = g.size || [];
  let geo: THREE.BufferGeometry | null = null;
  let alongZ = false;
  switch (g.type) {
    case 'plane': geo = new THREE.PlaneGeometry(2 * (s[0] || 4), 2 * (s[1] || 4)); break;
    case 'sphere': geo = new THREE.SphereGeometry(s[0] || 0.01, 24, 16); break;
    case 'capsule': geo = new THREE.CapsuleGeometry(s[0] || 0.01, 2 * (s[1] || 0), 6, 16); alongZ = true; break;
    case 'cylinder': geo = new THREE.CylinderGeometry(s[0] || 0.01, s[0] || 0.01, 2 * (s[1] || 0.01), 24); alongZ = true; break;
    case 'ellipsoid': geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(s[0] || 0.01, s[1] || 0.01, s[2] || 0.01); break;
    case 'box': geo = new THREE.BoxGeometry(2 * (s[0] || 0.01), 2 * (s[1] || 0.01), 2 * (s[2] || 0.01)); break;
    case 'mesh': {
      if (!g.mesh?.vertices?.length || !g.mesh.faces?.length) return null;
      geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(g.mesh.vertices, 3));
      geo.setIndex(g.mesh.faces);
      geo.computeVertexNormals();
      break;
    }
    default: return null;
  }
  if (alongZ) geo.rotateX(Math.PI / 2);
  const [r = 0.6, gg = 0.6, b = 0.6, a = 1] = g.rgba || [];
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(r, gg, b), transparent: a < 1, opacity: a, roughness: 0.7, metalness: 0.05,
    side: g.type === 'plane' ? THREE.DoubleSide : THREE.FrontSide, flatShading: g.type === 'mesh',
  });
  const mesh = new THREE.Mesh(geo, mat);
  if (g.pos) mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
  if (g.quat) mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  mesh.castShadow = g.type !== 'plane';
  mesh.receiveShadow = true;
  mesh.name = g.name;
  return mesh;
}

function cssColor(name: string, fallback: string) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export type StageOptions = { follow: boolean; contacts: boolean; trails: boolean; ghost: 'A' | 'B' | null };

export default function Stage({ replay, clock, options, height = 440, tint }: {
  replay: Replay; clock: Clock; options: StageOptions; height?: number | string; tint?: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const opts = useRef(options);
  opts.current = options;
  const [err, setErr] = useState<string | null>(null);
  const api = useRef<{ render: () => void } | null>(null);

  useEffect(() => { api.current?.render(); }, [options]);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
    } catch (e) {
      setErr(`WebGL is unavailable (${String(e)}). Use the video fallback or the timelines.`);
      return;
    }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const bg = () => new THREE.Color(cssColor('--scene-bg', '#e9ecef'));
    scene.background = bg();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444450, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2.5, -2, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    const sc = sun.shadow.camera as THREE.OrthographicCamera;
    sc.left = -3; sc.right = 3; sc.top = 3; sc.bottom = -3; sc.near = 0.1; sc.far = 20;
    scene.add(sun, sun.target);

    const bodyIndex = new Map(replay.bodies.map((b, i) => [b, i]));
    const groups = replay.bodies.map((name) => {
      const g = new THREE.Group();
      g.name = name;
      scene.add(g);
      return g;
    });
    const worldGroup = new THREE.Group();
    scene.add(worldGroup);
    let hasPlane = false;
    for (const geom of replay.geoms) {
      const m = geomObject(geom);
      if (!m) continue;
      if (geom.type === 'plane') hasPlane = true;
      const bi = bodyIndex.get(geom.body);
      (bi === undefined ? worldGroup : groups[bi]).add(m);
    }
    if (!hasPlane) {
      const grid = new THREE.GridHelper(4, 40, 0x888888, 0xbbbbbb);
      grid.rotateX(Math.PI / 2);
      (grid.material as THREE.Material).opacity = 0.35;
      (grid.material as THREE.Material).transparent = true;
      scene.add(grid);
    }

    const times = relTimes(replay);
    const n = replay.frames.t.length;
    const posOf = (f: number, b: number) => replay.frames.body_pos[f]?.[b];

    // contact markers (need meta.contact_bodies)
    const cb = (replay.meta.contact_bodies || []).map((name) => bodyIndex.get(name));
    const contactColor = new THREE.Color(cssColor('--s2', '#eb6834'));
    const markers = cb.map(() => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), new THREE.MeshBasicMaterial({ color: contactColor, transparent: true, opacity: 0.9, depthTest: false }));
      m.renderOrder = 10;
      m.visible = false;
      scene.add(m);
      return m;
    });

    // base body: meta.base_body, else the first non-world body that moves
    let base = replay.meta.base_body ? bodyIndex.get(replay.meta.base_body) : undefined;
    if (base === undefined) {
      base = replay.bodies.findIndex((b, i) => b !== 'world' && n > 1 && (() => {
        const a = posOf(0, i), z = posOf(n - 1, i);
        return a && z && Math.hypot(a[0] - z[0], a[1] - z[1], a[2] - z[2]) > 1e-4;
      })());
      if (base < 0) base = replay.bodies.findIndex((b) => b !== 'world');
    }
    const objBody = replay.meta.object_body ? bodyIndex.get(replay.meta.object_body) : undefined;
    const objectPath: (number[] | null)[] = replay.signals.object_pose
      ? replay.signals.object_pose.map((p) => (p && p.length >= 3 ? p.slice(0, 3) : null))
      : objBody !== undefined ? replay.frames.body_pos.map((f) => f[objBody] || null) : [];
    const basePath: (number[] | null)[] = base !== undefined && base >= 0 ? replay.frames.body_pos.map((f) => f[base!] || null) : [];
    function trail(path: (number[] | null)[], color: string) {
      const pts = path.map((p) => (p ? new THREE.Vector3(p[0], p[1], p[2]) : new THREE.Vector3(NaN, NaN, NaN)));
      const valid = pts.filter((p) => Number.isFinite(p.x));
      if (valid.length < 2) return null;
      const full = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map((p) => (Number.isFinite(p.x) ? p : valid[0]))),
        new THREE.LineBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity: 0.22 }));
      const past = new THREE.Line(full.geometry.clone(), new THREE.LineBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity: 0.95 }));
      const ghost = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), new THREE.MeshBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity: 0.5 }));
      scene.add(full, past, ghost);
      return { full, past, ghost, pts };
    }
    const trails = [trail(objectPath, cssColor('--s3', '#1baf7a')), trail(basePath, cssColor('--s7', '#9085e9'))].filter(Boolean) as NonNullable<ReturnType<typeof trail>>[];

    // camera framing from all recorded positions
    const box = new THREE.Box3();
    for (let f = 0; f < n; f += Math.max(1, Math.floor(n / 50))) for (const p of replay.frames.body_pos[f] || []) box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
    if (box.isEmpty()) box.setFromCenterAndSize(new THREE.Vector3(0, 0, 0.3), new THREE.Vector3(1, 1, 1));
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(0.4, box.getSize(new THREE.Vector3()).length() * 0.6);
    camera.position.set(center.x + radius * 1.4, center.y - radius * 1.6, center.z + radius * 0.9);
    controls.target.copy(center);
    controls.update();
    let lastFollow: THREE.Vector3 | null = null;

    let frame = -1;
    let dirty = true;
    function pose(f: number) {
      const P = replay.frames.body_pos[f], Q = replay.frames.body_quat[f];
      if (!P || !Q) return;
      groups.forEach((g, i) => {
        const p = P[i], q = Q[i];
        if (p) g.position.set(p[0], p[1], p[2]);
        if (q) g.quaternion.set(q[1], q[2], q[3], q[0]);
      });
      const flags = replay.signals.contacts?.[f];
      markers.forEach((m, k) => {
        const bi = cb[k];
        const on = !!flags?.[k];
        m.visible = opts.current.contacts && on && bi !== undefined;
        if (m.visible && bi !== undefined) { const p = P[bi]; if (p) m.position.set(p[0], p[1], p[2]); }
      });
      for (const tr of trails) {
        const start = Math.max(0, f - 45);
        tr.past.geometry.setDrawRange(start, f - start + 1);
        tr.full.visible = tr.past.visible = opts.current.trails;
        const p = tr.pts[f];
        tr.ghost.visible = opts.current.trails && Number.isFinite(p?.x);
        if (tr.ghost.visible) tr.ghost.position.copy(p);
      }
      if (opts.current.follow && base !== undefined && base >= 0 && P[base]) {
        const bp = new THREE.Vector3(P[base][0], P[base][1], P[base][2]);
        if (lastFollow) {
          const d = bp.clone().sub(lastFollow);
          camera.position.add(d);
          controls.target.add(d);
        } else controls.target.copy(bp);
        lastFollow = bp;
        sun.position.set(bp.x + 2.5, bp.y - 2, bp.z + 5);
        sun.target.position.copy(bp);
        controls.update();
      } else lastFollow = null;
    }
    function size() {
      const w = el!.clientWidth || 600, h = el!.clientHeight || 400;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      dirty = true;
    }
    size();
    const ro = new ResizeObserver(size);
    ro.observe(el);
    let raf = 0;
    const loop = () => {
      const f = frameAt(times, clock.t);
      if (f !== frame) { frame = f; pose(f); dirty = true; }
      if (dirty) { renderer.render(scene, camera); dirty = false; }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    controls.addEventListener('change', () => { dirty = true; });
    const mo = new MutationObserver(() => { scene.background = bg(); dirty = true; });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    api.current = { render: () => { if (frame >= 0) pose(frame); dirty = true; } };
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      mo.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose?.();
      });
      renderer.dispose();
      renderer.domElement.remove();
      api.current = null;
    };
  }, [replay, clock]);

  if (err) return <div className="state error" style={{ height }}>{err}</div>;
  return <div ref={host} style={{ width: '100%', height, outline: tint ? `2px solid ${tint}` : undefined, outlineOffset: -2 }} />;
}

/** Small orbitable 3D plot of the packet PCA trajectory with the current point. Edit-active frames drawn in orange. */
export function PcaPlot({ series, clock, height = 240 }: {
  series: { label: string; color: string; pts: (number[] | null)[]; times: number[]; edit?: (boolean | null)[] }[]; clock: Clock; height?: number;
}) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' }); } catch { return; }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const bg = () => new THREE.Color(cssColor('--surface', '#fcfcfb'));
    scene.background = bg();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    const all = series.flatMap((s) => s.pts.filter((p): p is number[] => !!p && p.length === 3));
    const box = new THREE.Box3();
    all.forEach((p) => box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2])));
    if (box.isEmpty()) box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    const c = box.getCenter(new THREE.Vector3());
    const sz = box.getSize(new THREE.Vector3());
    const s = 1 / Math.max(1e-6, sz.x, sz.y, sz.z);
    const root = new THREE.Group();
    root.scale.setScalar(s * 2);
    root.position.copy(c.clone().multiplyScalar(-s * 2));
    scene.add(root);
    const axes = new THREE.AxesHelper(1.2);
    axes.position.set(-1.1, -1.1, -1.1);
    scene.add(axes);
    const box3 = new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)), new THREE.Color(cssColor('--axis', '#c3c2b7')));
    scene.add(box3);
    const editColor = new THREE.Color(cssColor('--s2', '#eb6834'));
    const heads: { mesh: THREE.Mesh; s: (typeof series)[number] }[] = [];
    for (const ser of series) {
      const pts: THREE.Vector3[] = [];
      const cols: number[] = [];
      const base = new THREE.Color(ser.color);
      ser.pts.forEach((p, i) => {
        if (!p || p.length !== 3) return;
        pts.push(new THREE.Vector3(p[0], p[1], p[2]));
        const col = ser.edit?.[i] ? editColor : base;
        cols.push(col.r, col.g, col.b);
      });
      if (pts.length < 2) continue;
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
      root.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true })));
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.035 / (s * 2), 16, 12), new THREE.MeshBasicMaterial({ color: base }));
      root.add(head);
      heads.push({ mesh: head, s: ser });
    }
    camera.position.set(2.6, 2.2, 2.4);
    controls.target.set(0, 0, 0);
    controls.update();
    let dirty = true;
    let lastT = -1;
    const size = () => {
      const w = el.clientWidth || 300, h = el.clientHeight || 240;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      dirty = true;
    };
    size();
    const ro = new ResizeObserver(size);
    ro.observe(el);
    controls.addEventListener('change', () => { dirty = true; });
    const mo = new MutationObserver(() => { scene.background = bg(); dirty = true; });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    let raf = 0;
    const loop = () => {
      if (clock.t !== lastT) {
        lastT = clock.t;
        for (const h of heads) {
          const f = frameAt(h.s.times, clock.t);
          let p = h.s.pts[f];
          for (let k = f; !p && k >= 0; k--) p = h.s.pts[k];
          h.mesh.visible = !!p;
          if (p) h.mesh.position.set(p[0], p[1], p[2]);
        }
        dirty = true;
      }
      if (dirty) { renderer.render(scene, camera); dirty = false; }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      mo.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | undefined;
        mat?.dispose?.();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [series, clock]);
  return <div ref={host} style={{ width: '100%', height }} />;
}
