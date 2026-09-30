import { useEffect, useState } from "react";
import { useWB } from "../common/store";
import { fmt, sourceStyle } from "../common/labels";
import { bodyInfo, bodyOfJoint, jointByAddress, observedObject } from "../common/model";
import { EventInspector } from "../graph/EventInspector";
import type { ProbeQuery } from "../transport/types";
import type { ProbeResult } from "../transport/responses";

const TABS = ["Selection", "Morphology", "Observation", "Packet", "Probes", "Routing", "Log", "Privileged"] as const;
type Tab = (typeof TABS)[number];

export function Inspector() {
  const { selection } = useWB();
  const [tab, setTab] = useState<Tab>("Selection");
  return (
    <div className="inspector">
      <div role="tablist" aria-label="Inspector" className="tabs">
        {TABS.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>
      <div role="tabpanel" className="tab-body" aria-label={tab}>
        {tab === "Selection" && (selection?.kind === "event" ? <EventInspector eventId={selection.id} /> : <SelectionView />)}
        {tab === "Morphology" && <Morphology />}
        {tab === "Observation" && <Observation />}
        {tab === "Packet" && <PacketPanel />}
        {tab === "Probes" && <Probes />}
        {tab === "Routing" && <Routing />}
        {tab === "Log" && <Log />}
        {tab === "Privileged" && <Privileged />}
      </div>
    </div>
  );
}

function JointDetail({ address }: { address: string }) {
  const { snapshot, scene } = useWB();
  const j = snapshot && jointByAddress(snapshot, address);
  if (!snapshot || !j) return <div className="muted">unknown joint {address}</div>;
  const q = j.obsIndex >= 0 ? snapshot.observation.joints.qpos[j.obsIndex] : null;
  return (
    <div data-testid="joint-detail">
      <h4>Joint <code>{j.joint.name}</code></h4>
      <div className="kv">
        <span>address</span><span>{j.joint.address}</span>
        <span>type</span><span>{j.joint.type}</span>
        <span>range</span><span>{fmt(j.joint.range)}</span>
        <span>measured q</span><span data-testid="joint-q">{fmt(q, 4)}</span>
        <span>mimic of</span><span>{j.joint.mimic_of ?? "— (independent)"}</span>
        <span>body</span><span>{scene ? bodyOfJoint(scene, j.joint.name) ?? "—" : "—"}</span>
        <span>robot</span><span>{j.robot.name}</span>
      </div>
    </div>
  );
}

function SelectionView() {
  const { selection, snapshot, scene, poses, setSelection } = useWB();
  if (!snapshot || !scene) return <div className="muted">No session.</div>;
  if (!selection) return <div className="muted">Select a body in the scene (click or the body list), a node in the morphology tree, or an event in the graph.</div>;
  if (selection.kind === "joint") return <JointDetail address={selection.id} />;
  if (selection.kind === "assembly") {
    const a = snapshot.robots.flatMap((r) => r.assemblies.map((x) => ({ r, x }))).find((z) => z.x.id === selection.id);
    const ents = Object.entries(snapshot.robots.find((r) => r === a?.r)?.manipulators ?? {}).filter(([, asm]) => asm === selection.id).map(([e]) => e);
    return a ? (
      <div data-testid="assembly-detail"><h4>Assembly <code>{a.x.id}</code> ({a.x.kind})</h4>
        <div className="kv"><span>capabilities</span><span>{a.x.capabilities.join(", ")}</span>
          <span>frame site</span><span>{a.x.frame_site}</span>
          <span>task entities</span><span>{ents.join(", ") || "—"}</span>
          <span>controller owner</span><span>{ents.map((e) => `${e}: ${snapshot.runtime.owners[e] ?? "free"}`).join("; ") || "—"}</span>
          <span>members</span><span>{a.x.members.length} links</span></div></div>
    ) : <div className="muted">unknown assembly</div>;
  }
  if (selection.kind === "event" || selection.kind === "entity") return <div>{selection.kind} {selection.id}</div>;
  const info = bodyInfo(scene, snapshot, selection.id);
  if (!info) return <div className="muted">unknown body {selection.id}</div>;
  const pos = poses?.xpos[info.id];
  const obs = info.object ? observedObject(snapshot, info.object.descriptor) : null;
  return (
    <div data-testid="body-detail">
      <h4>Body <code data-testid="selected-body">{info.name}</code></h4>
      <div className="kv">
        <span>world position (render pose)</span><span data-testid="body-pos">{fmt(pos, 4)}</span>
        <span>joints</span><span>{info.joints.length ? info.joints.map((j) => (
          <button key={j.joint.address} className="link" onClick={() => setSelection({ kind: "joint", id: j.joint.address })}>{j.joint.name}</button>)) : "—"}</span>
        <span>assemblies</span><span>{info.assemblies.map((a) => (
          <button key={a.assembly.id} className="link" onClick={() => setSelection({ kind: "assembly", id: a.assembly.id })}>{a.assembly.id}</button>)) }{!info.assemblies.length && "—"}</span>
      </div>
      {info.object && (
        <div className="obj-detail"><h5>Declared object: {info.object.descriptor} ({info.object.kind})</h5>
          {obs ? (
            <div className="kv"><span>visible</span><span>{String(obs.visible)}</span>
              <span>estimate (public)</span><span>{fmt(obs.position_estimate, 4)}</span>
              <span>std</span><span>{obs.position_cov_diag ? fmt(obs.position_cov_diag.map(Math.sqrt), 4) : "unknown"}</span>
              <span>slot</span><span>{obs.slot}</span></div>
          ) : <span className="muted">no public estimate for this object</span>}
        </div>
      )}
      <p className="muted small">Render pose comes from the simulator pose stream for display; policy inputs are the public estimates.</p>
    </div>
  );
}

function Morphology() {
  const { snapshot, setSelection, selection } = useWB();
  if (!snapshot) return <div className="muted">No session.</div>;
  return (
    <div className="morph" data-testid="morphology">
      {snapshot.robots.map((r) => {
        const inAsm = new Set(r.assemblies.flatMap((a) => a.members));
        return (
          <details key={r.index} open>
            <summary><strong>{r.name}</strong> <span className="muted">({r.family}{r.synthetic ? ", synthetic" : ""}, spec {r.spec_hash})</span></summary>
            <div className="kv small">
              <span>independent controls</span><span data-testid="independent-controls">{r.independent_controls}</span>
              <span>generalized coordinates</span><span data-testid="generalized-coordinates">{r.generalized_coordinates}</span>
              <span>controller</span><span>{r.controller.id} <code className="small">{r.controller.version}</code></span>
              <span>lineage</span><span>{r.lineage.join(" › ")}</span>
            </div>
            <h5>Controller groups</h5>
            <ul>{r.controller.groups.map((g) => <li key={g.name}><code>{g.name}</code> width {g.width} [{g.units}] lower {fmt(g.lower, 2)} upper {fmt(g.upper, 2)}</li>)}</ul>
            <h5>Assemblies</h5>
            <ul className="tree">
              {r.assemblies.map((a) => (
                <li key={a.id}>
                  <button className={`link ${selection?.kind === "assembly" && selection.id === a.id ? "sel" : ""}`} onClick={() => setSelection({ kind: "assembly", id: a.id })}>{a.id}</button>
                  <span className="muted"> {a.kind} · {a.capabilities.join("/")}</span>
                  <ul>{a.members.map((m) => {
                    const js = r.joints.filter((j) => j.address.split(":")[0] === m);
                    return <li key={m}>link <code>{m}</code>{js.map((j) => (
                      <button key={j.address} className={`link ${selection?.kind === "joint" && selection.id === j.address ? "sel" : ""}`}
                        data-testid={`morph-joint-${j.name}-${a.id}`} onClick={() => setSelection({ kind: "joint", id: j.address })}>
                        {j.name}{j.mimic_of ? " (mimic)" : ""}</button>))}</li>;
                  })}</ul>
                </li>
              ))}
            </ul>
            <h5>Joints</h5>
            <table className="tbl"><thead><tr><th>joint</th><th>type</th><th>range</th><th>q</th><th>mimic of</th></tr></thead><tbody>
              {r.joints.map((j) => {
                const ref = jointByAddress(snapshot, j.address);
                const q = ref && ref.obsIndex >= 0 ? snapshot.observation.joints.qpos[ref.obsIndex] : null;
                return (
                  <tr key={j.address} className={selection?.kind === "joint" && selection.id === j.address ? "sel" : ""}>
                    <td><button className="link" data-testid={`morph-joint-${j.name}`} onClick={() => setSelection({ kind: "joint", id: j.address })}>{j.name}</button>
                      {!inAsm.has(j.address.split(":")[0]) && <span className="muted small"> (no assembly)</span>}</td>
                    <td>{j.type}</td><td>{fmt(j.range, 2)}</td><td data-testid={`q-${j.name}`}>{fmt(q, 4)}</td><td>{j.mimic_of ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody></table>
          </details>
        );
      })}
    </div>
  );
}

function Observation() {
  const { snapshot } = useWB();
  if (!snapshot) return <div className="muted">No session.</div>;
  const o = snapshot.observation;
  return (
    <div data-testid="observation">
      <p className="pill" style={{ background: "#0e7490" }}>PUBLIC ESTIMATES (what policies may see)</p>
      <h5>Objects</h5>
      <table className="tbl"><thead><tr><th>slot</th><th>descriptor</th><th>visible</th><th>estimate</th><th>std</th><th>t</th></tr></thead><tbody>
        {o.objects.map((x) => (
          <tr key={x.slot}><td>{x.slot}</td><td>{x.descriptor}</td><td>{String(x.visible)}</td><td>{x.position_estimate ? fmt(x.position_estimate, 3) : "unknown"}</td>
            <td>{x.position_cov_diag ? fmt(x.position_cov_diag.map(Math.sqrt), 4) : "unknown"}</td><td>{fmt(x.timestamp, 2)}</td></tr>
        ))}
      </tbody></table>
      <h5>Predicate estimates</h5>
      <table className="tbl"><thead><tr><th>predicate</th><th>value</th><th>known</th><th>conf</th><th>estimator</th></tr></thead><tbody>
        {o.predicates.map((p, i) => (
          <tr key={i}><td><code>{p.predicate}({p.args.join(", ")})</code></td><td>{p.known ? String(p.value) : "UNKNOWN"}</td>
            <td>{p.known ? "known" : "unknown"}</td><td>{fmt(p.confidence, 2)}</td><td className="small">{p.estimator}</td></tr>
        ))}
      </tbody></table>
      <h5>Sensor channels</h5>
      <ul>{o.channels.map((c) => <li key={c.name}><code>{c.name}</code> {fmt(c.values, 3)}</li>)}</ul>
      <h5>Measured joints</h5>
      <table className="tbl" data-testid="obs-joints"><tbody>
        {o.joints.addresses.map((a, i) => <tr key={a}><td><code>{a}</code></td><td data-testid={`obsq-${i}`}>{fmt(o.joints.qpos[i], 4)}</td></tr>)}
      </tbody></table>
    </div>
  );
}

const QUERIES: ProbeQuery[] = ["visible", "looking_at", "focused_on", "acting_on", "held_by", "manipulator_tasks",
  "relative_pose", "distance", "active_frame", "contact_mode", "desired_delta", "feasibility", "uncertainty", "object_qa"];

function ProbeView({ r }: { r: ProbeResult }) {
  return (
    <div className="probe-result" data-testid={`probe-result-${r.query}`}>
      <div className="probe-head"><code>{r.query}</code>{r.subject && <> subject <code>{r.subject}</code></>}{r.object && <> object <code>{r.object}</code></>}
        <span className="pill" style={{ background: r.source.startsWith("learned") ? "#3b82f6" : "#0e7490" }}>source: {r.source}</span>
        {r.null === true && <span className="pill null" data-testid="probe-null">NULL answer</span>}
        {r.multiple && <span className="pill">MULTIPLE answers</span>}
        {r.unknown_reason && <span className="pill unknown" data-testid="probe-unknown">UNKNOWN: {r.unknown_reason}</span>}
      </div>
      {r.note && <div className="muted small">{r.note}</div>}
      {r.answers.length ? (
        <ul>{r.answers.map((a, i) => (
          <li key={i}>{Object.entries(a).map(([k, v]) => (
            <span key={k} className="ans-kv"><b>{k}</b>=<span className={v === null || (Array.isArray(v) && !v.length) ? "nullval" : ""}>
              {v === null ? "null" : Array.isArray(v) && !v.length ? "[] (none)" : typeof v === "object" ? JSON.stringify(v) : String(v)}</span></span>
          ))}</li>
        ))}</ul>
      ) : <div className="muted">no answers</div>}
    </div>
  );
}

function Probes() {
  const { runProbe, probeResults, snapshot } = useWB();
  const [query, setQuery] = useState<ProbeQuery>("held_by");
  const [subject, setSubject] = useState("");
  const [object, setObject] = useState("");
  const [question, setQuestion] = useState("");
  const ents = snapshot?.graph.document.entity_declarations ?? [];
  return (
    <div data-testid="probes">
      <form className="row wrap" onSubmit={(e) => { e.preventDefault(); void runProbe({ query, subject: subject || null, object: object || null, question: question || null }); }}>
        <label>query <select aria-label="probe query" value={query} onChange={(e) => setQuery(e.target.value as ProbeQuery)}>{QUERIES.map((q) => <option key={q}>{q}</option>)}</select></label>
        <label>subject <input aria-label="probe subject" list="probe-ents" value={subject} onChange={(e) => setSubject(e.target.value)} style={{ width: 90 }} /></label>
        <label>object <input aria-label="probe object" list="probe-ents" value={object} onChange={(e) => setObject(e.target.value)} style={{ width: 90 }} /></label>
        {query === "object_qa" && <label>question <input aria-label="probe question" value={question} onChange={(e) => setQuestion(e.target.value)} /></label>}
        <datalist id="probe-ents">{ents.map((e) => <option key={e.id} value={e.id} />)}</datalist>
        <button type="submit">Run probe</button>
        <button type="button" onClick={() => QUERIES.filter((q) => q !== "object_qa").forEach((q) => void runProbe({ query: q }))}>Run all (no args)</button>
      </form>
      <p className="muted small">Probe answers are diagnostics with declared sources; they do not change execution.</p>
      {probeResults.map((r, i) => <ProbeView key={i} r={r} />)}
    </div>
  );
}

/** Who is bound to act for each event, and who currently owns each manipulator's controller. */
function Routing() {
  const { snapshot } = useWB();
  if (!snapshot) return <div className="muted">No session.</div>;
  const doc = snapshot.graph.document;
  const manips = doc.entity_declarations.filter((e) => e.type === "manipulator");
  const bound = snapshot.robots.flatMap((r) => Object.entries(r.manipulators ?? {}).map(([ent, asm]) => ({ ent, asm, robot: r.name })));
  return (
    <div data-testid="routing">
      <h5>Controller ownership (runtime, authoritative)</h5>
      <table className="tbl" data-testid="owners"><thead><tr><th>entity</th><th>owned by event</th></tr></thead><tbody>
        {manips.map((m) => <tr key={m.id} data-testid={`owner-${m.id}`}><td>{m.id}</td><td>{snapshot.runtime.owners[m.id] ?? "free"}</td></tr>)}
        {Object.entries(snapshot.runtime.owners).filter(([e]) => !manips.some((m) => m.id === e)).map(([e, ev]) => <tr key={e}><td>{e}</td><td>{ev}</td></tr>)}
      </tbody></table>
      <h5>Actor routing (from role bindings)</h5>
      <table className="tbl" data-testid="actor-routing"><thead><tr><th>event</th><th>actors</th><th>exclusive control</th><th>status</th></tr></thead><tbody>
        {doc.events.map((e) => (
          <tr key={e.id} data-testid={`route-${e.id}`}><td>{e.id}</td>
            <td>{e.roles.filter((r) => r.role === "actor" || r.role === "cooperating_actor").sort((a, b) => a.ordinal - b.ordinal)
              .map((r) => `${r.role}[${r.ordinal}]=${r.binding.kind === "entity" ? r.binding.entity.id : "output"}`).join(", ") || "—"}</td>
            <td>{e.resources.filter((r) => r.mode === "exclusive_control").map((r) => r.entity.id).join(", ") || "—"}</td>
            <td>{snapshot.runtime.events[e.id]?.status}</td></tr>
        ))}
      </tbody></table>
      <h5>Task manipulator → robot assembly (public binding)</h5>
      <ul>{bound.map((b) => <li key={b.ent}><code>{b.ent}</code> → {b.robot} / {b.asm}</li>)}
        {manips.filter((m) => !bound.some((b) => b.ent === m.id)).map((m) => <li key={m.id}><code>{m.id}</code> → <span className="warn-text">no physical assembly bound in this scene</span></li>)}</ul>
    </div>
  );
}

function Log() {
  const { log, staleDropped, snapshotsReceived } = useWB();
  return (
    <div data-testid="event-log">
      <div className="muted small">snapshots received: <span data-testid="snapshots-received">{snapshotsReceived}</span> · stale messages dropped: {staleDropped}</div>
      <table className="tbl"><thead><tr><th>seq</th><th>kind</th><th>source</th><th>graph v</th><th>summary</th></tr></thead><tbody>
        {[...log].reverse().map((l, i) => {
          const st = sourceStyle(l.source);
          return <tr key={i} data-kind={l.kind}><td>{l.seq}</td><td>{l.kind}</td><td><span className="src-chip" style={{ background: st.color, color: st.fg }}>{l.source ?? "—"}</span></td><td>{l.graph_version ?? ""}</td><td>{l.summary}</td></tr>;
        })}
      </tbody></table>
    </div>
  );
}

function Privileged() {
  const { privileged, setPrivileged, privSnapshot } = useWB();
  const ov = privSnapshot?.privileged_overlay;
  return (
    <div data-testid="privileged">
      <label className="toggle priv-toggle"><input type="checkbox" checked={privileged} onChange={(e) => setPrivileged(e.target.checked)} />
        Show privileged simulator truth overlay (display only)</label>
      {privileged && (
        <div className="privileged-box">
          <div className="priv-banner" data-testid="priv-banner">⚠ {ov?.label ?? "PRIVILEGED SIMULATOR TRUTH (display only; never a policy input)"}</div>
          {ov ? (<>
            <h5>Object poses (truth)</h5>
            <ul>{Object.entries(ov.object_poses).map(([k, v]) => <li key={k}><code>{k}</code> {fmt(v, 3)}</li>)}</ul>
            <h5>held_by (truth)</h5><pre>{JSON.stringify(ov.held_by)}</pre>
            <h5>Predicates (truth)</h5><pre>{JSON.stringify(ov.predicates, null, 1)}</pre>
            <h5>Event completion (truth)</h5><pre>{JSON.stringify(ov.event_completion_truth, null, 1)}</pre>
          </>) : <div className="muted">loading…</div>}
        </div>
      )}
    </div>
  );
}


/** R38: probes of the EXACT received controller packet z (system i -> system 0), not internal diagnostics. */
function PacketPanel() {
  const { api, sessionId, command, reportError } = useWB();
  const [view, setView] = useState<Record<string, any> | null>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [joint, setJoint] = useState(2);
  const [offset, setOffset] = useState(0.12);
  useEffect(() => {
    if (!sessionId) return;
    let alive = true;
    const tick = async () => {
      try { const v = await api.packet(sessionId); if (alive) setView(v); } catch (e) { reportError(e, "packet"); }
    };
    void tick();
    const h = setInterval(() => void tick(), 1000);
    return () => { alive = false; clearInterval(h); };
  }, [api, sessionId, reportError]);
  if (!view) return <div className="muted">loading…</div>;
  if (!view.available) return <div data-testid="packet-panel" className="muted">No latent packet held by system 0 ({view.reason}). Select a latent policy in learned mode.</div>;
  return (
    <div data-testid="packet-panel">
      <div className="packet-banner" data-testid="packet-banner">{view.label}</div>
      <table className="kv"><tbody>
        <tr><td>source obs</td><td><code>{view.observation_id}</code> ({view.source}, {view.policy_version})</td></tr>
        <tr><td>latent space</td><td><code>{view.latent_space_version}</code> / realizer <code>{view.realizer_compat_version}</code></td></tr>
        <tr><td>shape [K,M,D]</td><td>{JSON.stringify(view.shape)} knots {JSON.stringify(view.knot_times)} s</td></tr>
        <tr><td>age / phase</td><td>{fmt(view.age_s, 2)} s — {view.valid ? "valid" : "EXPIRED"} (until {fmt(view.valid_until, 2)})</td></tr>
        <tr><td>graph / runtime</td><td>v{view.graph_version} / r{view.runtime_version}</td></tr>
        <tr><td>system 0</td><td>ticks {view.system0.ticks}, packets {view.system0.packets}, rejected {view.system0.rejected}, fallback holds {view.system0.fallback_holds}</td></tr>
        <tr><td>frozen</td><td>{view.frozen ? "YES (diagnostic)" : "no"}</td></tr>
      </tbody></table>
      {view.probes && (<>
        <h5>Per manipulator (from received z)</h5>
        <ul>{Object.entries(view.probes.per_assembly as Record<string, { subtask: string; subtask_p: number }>).map(([h, a]) =>
          <li key={h}><code>{h}</code>: subtask <b>{a.subtask}</b> (p={a.subtask_p})</li>)}</ul>
        <h5>Per entity handle (opaque registry)</h5>
        <table className="kv"><thead><tr><th>handle</th><th>visible</th><th>focused</th><th>held_by m0</th><th>acting_on m0</th><th>rel pos to TCP (m) ± std</th><th>desired Δ (m)</th></tr></thead>
          <tbody>{Object.entries(view.probes.per_entity as Record<string, any>).map(([h, e]) => ( // eslint-disable-line @typescript-eslint/no-explicit-any
            <tr key={h}><td><code>{h}</code></td><td>{e.visible[0]}</td><td>{e.focused_on[0]}</td><td>{e.held_by[0]}</td><td>{e.acting_on[0]}</td>
              <td>{fmt(e.rel_pos_m, 2)} ± {fmt(e.rel_pos_std_m, 2)}</td><td>{fmt(e.desired_delta_m, 2)}</td></tr>))}</tbody></table>
      </>)}
      <div className="row wrap">
        <button onClick={() => void command({ type: view.frozen ? "latent_unfreeze" : "latent_freeze" })}>{view.frozen ? "Unfreeze latent" : "Freeze latent (diagnostic)"}</button>
        <label>joint # <input aria-label="disturb joint" type="number" min={1} value={joint} onChange={(e) => setJoint(Number(e.target.value))} style={{ width: 50 }} /></label>
        <label>offset rad <input aria-label="disturb offset" type="number" step={0.02} value={offset} onChange={(e) => setOffset(Number(e.target.value))} style={{ width: 70 }} /></label>
        <button onClick={() => void command({ type: "disturb_joint", n: joint, yaw: offset })}>Disturb joint (contaminates eval)</button>
      </div>
      <p className="muted small">Packet probes read only z + opaque handles; they never alter control. Privileged truth is in the Privileged tab.</p>
    </div>
  );
}
