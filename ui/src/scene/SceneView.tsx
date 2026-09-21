import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { useWB } from "../common/store";
import { geomMesh } from "./geometry";
import type { Poses, Scene } from "../transport/responses";

interface Ctx3 {
  renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; orbit: OrbitControls;
  gizmo: TransformControls; bodies: Map<number, THREE.Group>; marker: THREE.Mesh; ghost: THREE.Mesh;
  render: () => void;
}

declare global {
  interface Window { __rrpScene?: { projectBody: (name: string) => { x: number; y: number } | null; rendered: number } }
}

function applyPoses(c: Ctx3, poses: Poses | null) {
  if (!poses) return;
  c.bodies.forEach((grp, id) => {
    const p = poses.xpos[id];
    const q = poses.xquat[id];
    if (p) grp.position.set(p[0], p[1], p[2]);
    if (q) grp.quaternion.set(q[1], q[2], q[3], q[0]);
  });
}

/** Live three.js view built from /scene geoms, posed from snapshot/state_update body poses. */
export function SceneView() {
  const { scene, poses, selection, setSelection, gizmoMode, goal, setGoal, teleportDraft, setTeleportDraft, setStreamMode, toast } = useWB();
  const host = useRef<HTMLDivElement>(null);
  const ctx = useRef<Ctx3 | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const sceneRef = useRef<Scene | null>(null);

  // build once per scene description
  useEffect(() => {
    const el = host.current;
    if (!el || !scene) return;
    sceneRef.current = scene;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: false, powerPreference: "low-power" });
    } catch (e) {
      setFailed(String(e));
      toast({ level: "warn", title: "WebGL unavailable; switched to server stream mode", message: String(e) });
      setStreamMode(true);
      return;
    }
    renderer.setPixelRatio(1);
    const w = el.clientWidth || 640, h = el.clientHeight || 480;
    renderer.setSize(w, h);
    renderer.domElement.setAttribute("aria-label", "3D scene view: drag to orbit, wheel to zoom, click a body to select");
    renderer.domElement.setAttribute("data-testid", "scene-canvas");
    el.appendChild(renderer.domElement);
    const s3 = new THREE.Scene();
    s3.background = new THREE.Color(0x1b1f27);
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(1.25, -1.0, 0.85);
    s3.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.4));
    const dl = new THREE.DirectionalLight(0xffffff, 1.6);
    dl.position.set(1, -1, 2);
    s3.add(dl);
    const bodies = new Map<number, THREE.Group>();
    for (const b of scene.bodies) {
      const grp = new THREE.Group();
      grp.name = b.name;
      grp.userData.bodyId = b.id;
      grp.userData.bodyName = b.name;
      s3.add(grp);
      bodies.set(b.id, grp);
    }
    for (const g of scene.geoms) {
      const m = geomMesh(g);
      if (!m) continue;
      m.userData.bodyId = g.body;
      bodies.get(g.body)?.add(m);
    }
    s3.add(new THREE.AxesHelper(0.15));
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0x22c55e, wireframe: true }));
    marker.name = "__goal_marker";
    s3.add(marker);
    const ghost = new THREE.Mesh(new THREE.BoxGeometry(0.044, 0.044, 0.044),
      new THREE.MeshBasicMaterial({ color: 0xef4444, wireframe: true }));
    ghost.name = "__teleport_ghost";
    ghost.visible = false;
    s3.add(ghost);

    let rendered = 0;
    let raf = 0;
    const render = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => { raf = 0; renderer.render(s3, camera); rendered += 1; if (window.__rrpScene) window.__rrpScene.rendered = rendered; });
    };
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(0.3, 0, 0.1);
    orbit.update();
    orbit.addEventListener("change", render);
    const gizmo = new TransformControls(camera, renderer.domElement);
    gizmo.setSize(0.7);
    gizmo.addEventListener("dragging-changed", (ev) => { orbit.enabled = !(ev as unknown as { value: boolean }).value; });
    gizmo.addEventListener("change", render);
    s3.add(gizmo.getHelper());
    const c: Ctx3 = { renderer, scene: s3, camera, orbit, gizmo, bodies, marker, ghost, render };
    ctx.current = c;

    // click-to-select (ignore drags)
    const ray = new THREE.Raycaster();
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; };
    const onUp = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4 || gizmo.dragging) { down = null; return; }
      down = null;
      const r = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
      const hits = ray.intersectObjects([...bodies.values()], true).filter((h) => h.object.userData.bodyId !== undefined && h.object.userData.bodyId !== 0);
      if (hits.length) {
        const bid = hits[0].object.userData.bodyId as number;
        const name = sceneRef.current?.bodies.find((b) => b.id === bid)?.name;
        if (name) setSelection({ kind: "body", id: name });
      }
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);
    const ro = new ResizeObserver(() => {
      const W = el.clientWidth, H = el.clientHeight;
      if (W > 0 && H > 0) { renderer.setSize(W, H); camera.aspect = W / H; camera.updateProjectionMatrix(); render(); }
    });
    ro.observe(el);
    window.__rrpScene = {
      rendered: 0,
      projectBody: (name: string) => {
        const grp = [...bodies.values()].find((g) => g.userData.bodyName === name);
        if (!grp) return null;
        const box = new THREE.Box3().setFromObject(grp);
        const v = (box.isEmpty() ? grp.getWorldPosition(new THREE.Vector3()) : box.getCenter(new THREE.Vector3())).project(camera);
        const r = renderer.domElement.getBoundingClientRect();
        return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height };
      },
    };
    render();
    return () => {
      ro.disconnect();
      cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      gizmo.detach(); gizmo.dispose(); orbit.dispose();
      s3.traverse((o) => { const m = o as THREE.Mesh; if (m.geometry) m.geometry.dispose(); });
      renderer.dispose();
      renderer.domElement.remove();
      ctx.current = null;
      delete window.__rrpScene;
    };
  }, [scene, setSelection, setStreamMode, toast]);

  // poses
  useEffect(() => { const c = ctx.current; if (c) { applyPoses(c, poses); c.render(); } }, [poses]);

  // selection highlight
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    const selName = selection?.kind === "body" || selection?.kind === "object" ? selection.id : null;
    c.bodies.forEach((grp) => {
      const on = grp.userData.bodyName === selName;
      grp.traverse((o) => {
        const m = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
        if (m && "emissive" in m) m.emissive.setHex(on ? 0x2f6fff : 0x000000);
      });
    });
    c.render();
  }, [selection, scene]);

  // gizmo: goal marker (user command target) vs teleport ghost (simulator intervention)
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    c.marker.position.set(goal[0], goal[1], goal[2]);
    c.ghost.visible = gizmoMode === "teleport" && !!teleportDraft;
    if (teleportDraft) c.ghost.position.set(...teleportDraft.pos);
    const target = gizmoMode === "goal" ? c.marker : (teleportDraft ? c.ghost : null);
    if (target) { if (c.gizmo.object !== target) c.gizmo.attach(target); } else c.gizmo.detach();
    c.render();
  }, [goal, gizmoMode, teleportDraft, scene]);

  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    const onEnd = () => {
      const p = c.gizmo.object?.position;
      if (!p) return;
      const v: [number, number, number] = [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)];
      if (c.gizmo.object === c.marker) setGoal(v);
      else if (teleportDraft) setTeleportDraft({ ...teleportDraft, pos: v });
    };
    c.gizmo.addEventListener("mouseUp", onEnd);
    return () => c.gizmo.removeEventListener("mouseUp", onEnd);
  }, [setGoal, setTeleportDraft, teleportDraft, scene]);

  return (
    <div className="scene-host" ref={host} data-testid="scene-3d">
      {!scene && <div className="scene-empty">No session. Create one from the toolbar.</div>}
      {failed && <div className="scene-empty">3D view unavailable: {failed}</div>}
    </div>
  );
}

/** Low-load stream mode: server-rendered JPEG frames polled at ≤ 4 fps. */
export function StreamView() {
  const { api, sessionId } = useWB();
  const [n, setN] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [fps, setFps] = useState(2);
  useEffect(() => {
    if (!sessionId) return;
    const iv = window.setInterval(() => setN((x) => x + 1), 1000 / Math.min(4, Math.max(0.5, fps)));
    return () => window.clearInterval(iv);
  }, [sessionId, fps]);
  if (!sessionId) return <div className="scene-empty">No session.</div>;
  return (
    <div className="stream-host" data-testid="stream-view">
      <img alt="server-rendered frame (stream mode)" src={api.frameUrl(sessionId, n)}
        onError={() => setErr("frame endpoint failed (server renderer unavailable?)")} onLoad={() => setErr(null)} />
      <div className="stream-meta">
        <span className="pill">STREAM MODE (server-rendered, low load)</span>
        <label>fps <input type="number" min={0.5} max={4} step={0.5} value={fps} onChange={(e) => setFps(Number(e.target.value))} aria-label="stream frames per second (max 4)" /></label>
        {err && <span className="err">{err}</span>}
      </div>
    </div>
  );
}
