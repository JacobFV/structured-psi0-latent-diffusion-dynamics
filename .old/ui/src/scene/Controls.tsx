import { useEffect, useMemo, useState } from "react";
import { useWB } from "../common/store";
import { fmt, sourceStyle } from "../common/labels";
import type { CommandGroup } from "../transport/responses";

export function SessionBar() {
  const { caps, createSession, snapshot, sessionId, busy } = useWB();
  const [robot, setRobot] = useState("");
  const [task, setTask] = useState("pick_place");
  const [seed, setSeed] = useState(1);
  useEffect(() => { if (caps && !robot) setRobot(caps.robots[0] ?? ""); }, [caps, robot]);
  return (
    <form className="session-bar" onSubmit={(e) => { e.preventDefault(); void createSession(robot, task, seed); }} aria-label="Create session">
      <label>Robot <select aria-label="robot" value={robot} onChange={(e) => setRobot(e.target.value)}>
        {caps?.robots.map((r) => <option key={r} value={r}>{r}</option>)}
      </select></label>
      <label>Task <select aria-label="task" value={task} onChange={(e) => setTask(e.target.value)}>
        {caps?.tasks.map((t) => <option key={t} value={t}>{t}</option>)}
      </select></label>
      <label>Seed <input aria-label="seed" type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ width: 64 }} /></label>
      <button type="submit" disabled={!robot || busy > 0}>Create session</button>
      {sessionId && snapshot && <span className="muted" data-testid="session-id">session {sessionId} · {snapshot.robot}/{snapshot.task} · seed {snapshot.seed}</span>}
    </form>
  );
}

export function ModeBadge() {
  const { snapshot } = useWB();
  if (!snapshot) return null;
  const src = snapshot.mode === "learned" ? `learned:${snapshot.policy ?? ""}` : snapshot.mode;
  const st = sourceStyle(src);
  return (
    <span className="mode-badge" data-testid="mode-badge" data-mode={snapshot.mode} style={{ background: st.color, color: st.fg }}
      title={snapshot.control_label}>
      {st.label}
      <span className="mode-badge-server"> · server: {snapshot.control_label}</span>
    </span>
  );
}

export function RunControls() {
  const { snapshot, command, caps, rebaseSeq, streamMode, setStreamMode } = useWB();
  const [n, setN] = useState(10);
  const [mode, setMode] = useState("user");
  const [policy, setPolicy] = useState("");
  if (!snapshot) return null;
  const policies = caps?.policies ?? [];
  return (
    <div className="run-controls" role="group" aria-label="Simulation controls">
      <button onClick={() => void command({ type: "step", n })} title="Step the backend simulator (shortcut: .)">Step</button>
      <input aria-label="steps per click" type="number" min={1} max={2000} value={n} onChange={(e) => setN(Number(e.target.value))} style={{ width: 60 }} />
      {snapshot.running
        ? <button onClick={() => void command({ type: "pause" })}>Pause</button>
        : <button onClick={() => void command({ type: "run" })}>Run</button>}
      <button onClick={() => { if (confirm("Reset the session to its initial snapshot? The recorded timeline is cleared (export first if needed).")) { rebaseSeq(); void command({ type: "reset" }); } }}>Reset…</button>
      <span className="sep" />
      <label>Mode <select aria-label="control mode" value={mode} onChange={(e) => setMode(e.target.value)}>
        {(caps?.modes ?? []).filter((m) => m !== "debug").map((m) => <option key={m} value={m}>{m === "learned" && !policies.length ? "learned (no policy registered)" : m}</option>)}
      </select></label>
      {mode === "learned" && (
        <select aria-label="policy" value={policy} onChange={(e) => setPolicy(e.target.value)}>
          <option value="">— select policy —</option>
          {policies.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      )}
      <button onClick={() => void command({ type: "set_mode", mode, policy: mode === "learned" ? policy || null : null }, "Set mode")}>Apply mode</button>
      <span className="sep" />
      <label className="toggle"><input type="checkbox" checked={streamMode} onChange={(e) => setStreamMode(e.target.checked)} /> Low-load stream mode</label>
      <span className="muted" data-testid="sim-time">t={fmt(snapshot.time, 2)}s · seq {snapshot.seq} · queued chunks {snapshot.executor.queued}</span>
    </div>
  );
}

function GroupSliders({ group, current }: { group: CommandGroup; current: number[] }) {
  const { command } = useWB();
  const [vals, setVals] = useState<number[]>(current);
  const [dirty, setDirty] = useState(false);
  useEffect(() => { if (!dirty) setVals(current); }, [current, dirty]);
  return (
    <fieldset className="group-sliders" data-testid={`group-${group.name}`}>
      <legend>{group.name} <span className="muted">({group.width} × {group.units}; bounds from controller contract)</span></legend>
      {vals.map((v, i) => (
        <div key={i} className="slider-row">
          <label htmlFor={`sl-${group.name}-${i}`}>{group.name}[{i}]</label>
          <input id={`sl-${group.name}-${i}`} type="range" min={group.lower[i]} max={group.upper[i]} step={(group.upper[i] - group.lower[i]) / 200}
            value={v} onChange={(e) => { const nv = [...vals]; nv[i] = Number(e.target.value); setVals(nv); setDirty(true); }} />
          <input aria-label={`${group.name}[${i}] value`} type="number" step={0.01} value={Number(v.toFixed(4))}
            onChange={(e) => { const nv = [...vals]; nv[i] = Number(e.target.value); setVals(nv); setDirty(true); }} style={{ width: 72 }} />
          <span className="muted small">[{fmt(group.lower[i], 3)}, {fmt(group.upper[i], 3)}]</span>
        </div>
      ))}
      <button onClick={async () => { const r = await command({ type: "joint_target", group: group.name, values: vals }, `Joint target (${group.name})`); if (r) setDirty(false); }}>
        Send {group.name} joint target
      </button>
      {dirty && <button onClick={() => { setDirty(false); setVals(current); }}>Revert</button>}
    </fieldset>
  );
}

export function JointControls() {
  const { snapshot } = useWB();
  if (!snapshot) return null;
  const r = snapshot.robots[0];
  if (!r) return null;
  return (
    <div className="joint-controls">
      <p className="muted small">User commands are validated against controller limits by the backend and switch the mode to USER.</p>
      {r.controller.groups.map((g) => (
        <GroupSliders key={g.name} group={g} current={r.controller.current_targets?.[g.name] ?? g.lower.map((lo, i) => (lo + g.upper[i]) / 2)} />
      ))}
    </div>
  );
}

function Vec3Input({ label, value, onChange }: { label: string; value: [number, number, number]; onChange: (v: [number, number, number]) => void }) {
  return (
    <span className="vec3">
      {(["x", "y", "z"] as const).map((ax, i) => (
        <label key={ax}>{ax}<input aria-label={`${label} ${ax}`} type="number" step={0.01} value={value[i]}
          onChange={(e) => { const v = [...value] as [number, number, number]; v[i] = Number(e.target.value); onChange(v); }} style={{ width: 70 }} /></label>
      ))}
    </span>
  );
}

export function TargetEditor() {
  const { snapshot, scene, command, gizmoMode, setGizmoMode, goal, setGoal, teleportDraft, setTeleportDraft, selection } = useWB();
  const [yaw, setYaw] = useState(0);
  const objects = scene?.objects ?? [];
  const objBody = teleportDraft?.body ?? (selection?.kind === "object" || selection?.kind === "body" ? objects.find((o) => o.sim_body === selection.id)?.sim_body : undefined) ?? objects[0]?.sim_body;
  const est = useMemo(() => {
    const o = objects.find((x) => x.sim_body === objBody);
    return o ? snapshot?.observation.objects.find((x) => x.descriptor === o.descriptor)?.position_estimate ?? null : null;
  }, [objects, objBody, snapshot]);
  if (!snapshot) return null;
  return (
    <div className="target-editor">
      <div role="radiogroup" aria-label="gizmo mode" className="mode-switch">
        <label className={gizmoMode === "goal" ? "on" : ""}><input type="radio" name="gizmo" checked={gizmoMode === "goal"} onChange={() => setGizmoMode("goal")} /> Edit target goal (user command)</label>
        <label className={`danger ${gizmoMode === "teleport" ? "on" : ""}`}><input type="radio" name="gizmo" checked={gizmoMode === "teleport"}
          onChange={() => { setGizmoMode("teleport"); if (objBody) setTeleportDraft({ body: objBody, pos: (est?.slice(0, 3) as [number, number, number]) ?? [0.4, 0, 0.05] }); }} /> Teleport object (contaminates evaluation)</label>
      </div>
      {gizmoMode === "goal" ? (
        <div className="ee-target">
          <p className="muted small">End-effector goal (world frame, m). Drag the green gizmo in the scene or type values; routed via IK → validated joint targets.</p>
          <Vec3Input label="ee target" value={goal} onChange={setGoal} />
          <label>yaw <input aria-label="ee target yaw" type="number" step={0.1} value={yaw} onChange={(e) => setYaw(Number(e.target.value))} style={{ width: 60 }} /></label>
          <button onClick={() => void command({ type: "ee_target", pos: goal, yaw }, "End-effector target")}>Send EE target</button>
        </div>
      ) : (
        <div className="teleport">
          <p className="warn-text small">Teleporting moves the object physically in the simulator. It is an intervention, is logged, and marks this run CONTAMINATED for autonomous evaluation.</p>
          <label>Object <select aria-label="teleport object" value={teleportDraft?.body ?? objBody ?? ""}
            onChange={(e) => setTeleportDraft({ body: e.target.value, pos: teleportDraft?.pos ?? [0.4, 0, 0.05] })}>
            {objects.map((o) => <option key={o.sim_body} value={o.sim_body}>{o.sim_body} ({o.descriptor})</option>)}
          </select></label>
          <Vec3Input label="teleport position" value={teleportDraft?.pos ?? [0.4, 0, 0.05]} onChange={(v) => setTeleportDraft({ body: teleportDraft?.body ?? objBody ?? "", pos: v })} />
          <button className="danger" onClick={() => {
            const d = teleportDraft ?? (objBody ? { body: objBody, pos: [0.4, 0, 0.05] as [number, number, number] } : null);
            if (!d) return;
            if (confirm(`Teleport ${d.body} to [${d.pos.join(", ")}]?\n\nThis is a simulator intervention: the run will be marked CONTAMINATED and excluded from autonomous evaluation.`)) {
              void command({ type: "teleport", body: d.body, pos: d.pos }, "Teleport");
            }
          }}>Teleport…</button>
        </div>
      )}
    </div>
  );
}

export function BodyList() {
  const { scene, selection, setSelection } = useWB();
  if (!scene) return null;
  return (
    <label className="body-list">Select body <select aria-label="select body" value={selection?.kind === "body" ? selection.id : ""}
      onChange={(e) => e.target.value && setSelection({ kind: "body", id: e.target.value })}>
      <option value="">—</option>
      {scene.bodies.filter((b) => b.id !== 0).map((b) => <option key={b.id} value={b.name}>{b.name}</option>)}
    </select></label>
  );
}
