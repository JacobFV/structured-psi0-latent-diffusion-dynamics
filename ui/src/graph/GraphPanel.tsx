import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow, Background, Controls, Handle, Position, MarkerType, useNodesState,
  type Connection, type Edge, type Node, type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useWB } from "../common/store";
import { statusColor } from "../common/labels";
import type { Binding, EventDef, Receipt, RuntimeEvent, Snapshot } from "../transport/responses";

export function bindingText(b: Binding): string {
  return b.kind === "entity" ? b.entity.id : `${b.event_id}#${b.attempt}.${b.output_name}`;
}

interface EventNodeData extends Record<string, unknown> {
  ev: EventDef; rt?: RuntimeEvent; priority?: number; receipts: Receipt[]; success: boolean;
}

function EventNode({ data, selected }: NodeProps<Node<EventNodeData>>) {
  const { ev, rt, priority, receipts, success } = data;
  const roles = [...ev.roles].sort((a, b) => a.role.localeCompare(b.role) || a.ordinal - b.ordinal);
  const rej = rt?.rejections ?? [];
  return (
    <div className={`event-node ${selected ? "selected" : ""}`} data-testid={`event-node-${ev.id}`} data-status={rt?.status}
      style={{ borderColor: statusColor(rt?.status) }}>
      <Handle type="target" position={Position.Left} aria-label={`${ev.id} input handle`} />
      <div className="en-head">
        <strong>{ev.id}</strong>
        <span className="en-op">{ev.operator}</span>
        <span className="en-status" style={{ background: statusColor(rt?.status) }} data-testid={`status-${ev.id}`}>{rt?.status ?? "?"}</span>
      </div>
      <ul className="en-roles">
        {roles.map((r) => (
          <li key={`${r.role}-${r.ordinal}`} className={r.binding.kind === "event_output" ? "out-binding" : ""}>
            {r.role}[{r.ordinal}] → {bindingText(r.binding)}
          </li>
        ))}
      </ul>
      <div className="en-meta">
        {rt?.reason && <span className="en-reason" data-testid={`reason-${ev.id}`}>reason: {rt.reason}</span>}
        {rej.length > 0 && <span className="en-rej" data-testid={`rejections-${ev.id}`}>rejected ×{rej.length} ({rej[rej.length - 1].reason})</span>}
        {priority !== undefined && <span>prio {priority}</span>}
        {success && <span className="en-success">success event</span>}
        {rt && <span>attempt {rt.attempt}</span>}
      </div>
      {receipts.length > 0 && (
        <div className="en-receipts">
          {receipts.map((r, i) => <span key={i} className={r.valid ? "rc-ok" : "rc-bad"}>{r.valid ? "✓" : "✗"} {r.output_name} v{r.version}</span>)}
        </div>
      )}
      <Handle type="source" position={Position.Right} aria-label={`${ev.id} output handle`} />
    </div>
  );
}

const nodeTypes = { event: EventNode };

function autoLayout(events: EventDef[]): Record<string, [number, number]> {
  const depth: Record<string, number> = {};
  const byId = Object.fromEntries(events.map((e) => [e.id, e]));
  const d = (id: string, seen = new Set<string>()): number => {
    if (depth[id] !== undefined) return depth[id];
    if (seen.has(id) || !byId[id]) return 0;
    seen.add(id);
    const e = byId[id];
    const deps = [...e.requires_completed, ...e.requires_active,
      ...e.roles.flatMap((r) => (r.binding.kind === "event_output" ? [r.binding.event_id] : []))];
    depth[id] = deps.length ? 1 + Math.max(...deps.map((x) => d(x, seen))) : 0;
    return depth[id];
  };
  events.forEach((e) => d(e.id));
  const rows: Record<number, number> = {};
  const out: Record<string, [number, number]> = {};
  for (const e of events) {
    const k = depth[e.id] ?? 0;
    rows[k] = (rows[k] ?? 0) + 1;
    out[e.id] = [30 + k * 290, 20 + (rows[k] - 1) * 190];
  }
  return out;
}

function buildEdges(snap: Snapshot): Edge[] {
  const edges: Edge[] = [];
  for (const e of snap.graph.document.events) {
    for (const s of e.requires_completed) edges.push({
      id: `c:${s}->${e.id}`, source: s, target: e.id, label: "requires completed", data: { mode: "completed" },
      markerEnd: { type: MarkerType.ArrowClosed }, style: { strokeWidth: 2 }, className: "edge-completed",
    });
    for (const s of e.requires_active) edges.push({
      id: `a:${s}->${e.id}`, source: s, target: e.id, label: "requires active", data: { mode: "active" },
      markerEnd: { type: MarkerType.ArrowClosed }, style: { strokeWidth: 2, strokeDasharray: "8 5" }, className: "edge-active",
    });
    for (const r of e.roles) if (r.binding.kind === "event_output") edges.push({
      id: `o:${r.binding.event_id}->${e.id}:${r.role}${r.ordinal}`, source: r.binding.event_id, target: e.id,
      label: `${r.binding.output_name} → ${r.role}[${r.ordinal}]`, deletable: false, data: { mode: "output" },
      style: { strokeWidth: 2, strokeDasharray: "2 4", stroke: "#a78bfa" }, className: "edge-output",
    });
  }
  return edges;
}

export function GraphPanel() {
  const { snapshot, graphEdit, saveLayout, selection, setSelection, editBase } = useWB();
  const [newEdgeMode, setNewEdgeMode] = useState<"completed" | "active">("completed");
  const [showAdd, setShowAdd] = useState<"" | "event" | "entity">("");
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<EventNodeData>>([]);
  const localPos = useRef<Record<string, { x: number; y: number }>>({});
  const [selectedEdge, setSelectedEdge] = useState<Edge | null>(null);

  const doc = snapshot?.graph.document;
  useEffect(() => {
    if (!snapshot || !doc) { setNodes([]); return; }
    const auto = autoLayout(doc.events);
    const lay = snapshot.graph.layout ?? {};
    setNodes((prev) => doc.events.map((ev) => {
      const old = prev.find((n) => n.id === ev.id);
      const p = localPos.current[ev.id] ?? (lay[ev.id] ? { x: lay[ev.id][0], y: lay[ev.id][1] } : { x: auto[ev.id][0], y: auto[ev.id][1] });
      return {
        id: ev.id, type: "event", position: old?.dragging ? old.position : p, deletable: false,
        selected: selection?.kind === "event" && selection.id === ev.id,
        data: {
          ev, rt: snapshot.runtime.events[ev.id], priority: snapshot.graph.priorities[ev.id],
          receipts: snapshot.runtime.receipts.filter((r) => r.event_id === ev.id), success: doc.success_events.includes(ev.id),
        },
      };
    }));
  }, [snapshot, doc, setNodes, selection]);

  const edges = useMemo(() => (snapshot ? buildEdges(snapshot) : []).map((e) => ({ ...e, selected: selectedEdge?.id === e.id })), [snapshot, selectedEdge]);

  const onConnect = useCallback((c: Connection) => {
    if (!c.source || !c.target) return;
    void graphEdit([{ op: "add_dependency", src: c.source, dst: c.target, mode: newEdgeMode }], `Add dependency ${c.source} → ${c.target}`);
  }, [graphEdit, newEdgeMode]);

  const removeEdge = useCallback((e: Edge) => {
    const mode = (e.data as { mode?: string } | undefined)?.mode;
    if (mode !== "completed" && mode !== "active") return;
    void graphEdit([{ op: "remove_dependency", src: e.source, dst: e.target, mode }], `Remove dependency ${e.source} → ${e.target}`);
    setSelectedEdge(null);
  }, [graphEdit]);

  if (!snapshot || !doc) return <div className="panel-empty">Event graph appears once a session exists.</div>;
  const stale = editBase !== null && editBase !== snapshot.graph.version;
  return (
    <div className="graph-panel">
      <div className="graph-toolbar" role="toolbar" aria-label="Graph editing">
        <span data-testid="graph-version">graph v{snapshot.graph.version}</span>
        <span className={stale ? "stale" : "muted"} data-testid="edit-base">edits based on v{editBase ?? "?"}{stale ? " (stale — another edit was committed)" : ""}</span>
        <label>new edge: <select aria-label="new dependency mode" value={newEdgeMode} onChange={(e) => setNewEdgeMode(e.target.value as "completed" | "active")}>
          <option value="completed">requires completed (solid)</option>
          <option value="active">requires active (dashed)</option>
        </select></label>
        <button onClick={() => setShowAdd(showAdd === "event" ? "" : "event")}>Add event…</button>
        <button onClick={() => setShowAdd(showAdd === "entity" ? "" : "entity")}>Add entity…</button>
        {selectedEdge && (selectedEdge.data as { mode?: string })?.mode !== "output" && (
          <button onClick={() => removeEdge(selectedEdge)}>Remove dependency {selectedEdge.source}→{selectedEdge.target}</button>
        )}
        <DependencyForm mode={newEdgeMode} />
      </div>
      {showAdd === "event" && <AddEventForm onDone={() => setShowAdd("")} />}
      {showAdd === "entity" && <AddEntityForm onDone={() => setShowAdd("")} />}
      <div className="graph-canvas" data-testid="graph-canvas">
        <ReactFlow
          nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onConnect={onConnect}
          onNodeClick={(_, n) => { setSelection({ kind: "event", id: n.id }); setSelectedEdge(null); }}
          onEdgeClick={(_, e) => setSelectedEdge(e)}
          onEdgesDelete={(es) => es.forEach(removeEdge)}
          onNodeDragStop={(_, n) => {
            localPos.current[n.id] = n.position;
            void saveLayout({ [n.id]: [Math.round(n.position.x), Math.round(n.position.y)] });   // layout only; no graph version change
          }}
          deleteKeyCode={["Delete", "Backspace"]} fitView fitViewOptions={{ padding: 0.2 }} minZoom={0.2} maxZoom={2}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={20} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <div className="graph-legend small">
        <span><i className="lg lg-solid" /> requires completed</span>
        <span><i className="lg lg-dashed" /> requires active</span>
        <span><i className="lg lg-dotted" /> output binding</span>
        <span className="muted">Drag between handles to add a dependency; select an edge + Delete to remove. Dragging nodes changes layout only.</span>
      </div>
    </div>
  );
}

/** Keyboard-accessible alternative to dragging between handles. */
function DependencyForm({ mode }: { mode: "completed" | "active" }) {
  const { snapshot, graphEdit } = useWB();
  const ids = snapshot?.graph.document.events.map((e) => e.id) ?? [];
  const [src, setSrc] = useState("");
  const [dst, setDst] = useState("");
  return (
    <form className="dep-form" onSubmit={(e) => { e.preventDefault(); if (src && dst) void graphEdit([{ op: "add_dependency", src, dst, mode }], `Add dependency ${src} → ${dst}`); }}>
      <select aria-label="dependency source (prerequisite)" value={src} onChange={(e) => setSrc(e.target.value)}>
        <option value="">prerequisite…</option>{ids.map((i) => <option key={i}>{i}</option>)}
      </select>
      <span>→</span>
      <select aria-label="dependency target (dependent)" value={dst} onChange={(e) => setDst(e.target.value)}>
        <option value="">dependent…</option>{ids.map((i) => <option key={i}>{i}</option>)}
      </select>
      <button type="submit" disabled={!src || !dst}>Connect</button>
    </form>
  );
}

const ROLES = ["actor", "patient", "instrument", "target", "source", "destination", "reference", "support", "cooperating_actor"];

function AddEventForm({ onDone }: { onDone: () => void }) {
  const { snapshot, graphEdit } = useWB();
  const doc = snapshot!.graph.document;
  const manips = doc.entity_declarations.filter((e) => e.type === "manipulator");
  const [id, setId] = useState("");
  const [operator, setOperator] = useState("inspect");
  const [actor, setActor] = useState(manips[0]?.id ?? "");
  const [patient, setPatient] = useState(doc.entity_declarations.find((e) => e.type === "object")?.id ?? "");
  const [claim, setClaim] = useState(false);
  const [requires, setRequires] = useState<string[]>([]);
  const [output, setOutput] = useState("");
  const [outputType, setOutputType] = useState("frame_estimate");
  const submit = async () => {
    const ref = (x: string) => ({ id: x, version: 0 });
    const roles = [];
    if (actor) roles.push({ role: "actor", ordinal: 0, binding: { kind: "entity", entity: ref(actor) } });
    if (patient) roles.push({ role: "patient", ordinal: 0, binding: { kind: "entity", entity: ref(patient) } });
    const ev = {
      id, operator, roles, requires_completed: requires, requires_active: [], preconditions: [], invariants: [],
      desired_effects: [], completion: [], resources: claim && actor ? [{ entity: ref(actor), mode: "exclusive_control" }] : [],
      produces: output ? [{ name: output, type: outputType, availability: "on_success" }] : [], frame_binding: null,
      timeout_seconds: 20, recovery: { max_attempts: 2, on_failure: "fail" },
    };
    const r = await graphEdit([{ op: "add_event", event: ev }], `Add event ${id}`);
    if (r) onDone();
  };
  return (
    <form className="add-form" aria-label="Add event" onSubmit={(e) => { e.preventDefault(); void submit(); }}>
      <label>id <input aria-label="new event id" required pattern="[A-Za-z0-9_]+" value={id} onChange={(e) => setId(e.target.value)} /></label>
      <label>operator <input aria-label="new event operator" required value={operator} onChange={(e) => setOperator(e.target.value)} /></label>
      <label>actor[0] <select aria-label="new event actor" value={actor} onChange={(e) => setActor(e.target.value)}>
        <option value="">(none)</option>{doc.entity_declarations.map((e) => <option key={e.id} value={e.id}>{e.id} ({e.type})</option>)}
      </select></label>
      <label>patient[0] <select aria-label="new event patient" value={patient} onChange={(e) => setPatient(e.target.value)}>
        <option value="">(none)</option>{doc.entity_declarations.map((e) => <option key={e.id} value={e.id}>{e.id} ({e.type})</option>)}
      </select></label>
      <label className="toggle"><input type="checkbox" checked={claim} onChange={(e) => setClaim(e.target.checked)} /> claim exclusive control of actor</label>
      <fieldset><legend>requires completed</legend>
        {doc.events.map((e) => (
          <label key={e.id} className="toggle"><input type="checkbox" checked={requires.includes(e.id)}
            onChange={(x) => setRequires(x.target.checked ? [...requires, e.id] : requires.filter((r) => r !== e.id))} /> {e.id}</label>
        ))}
      </fieldset>
      <label>produces output <input aria-label="new event output name" placeholder="(none)" value={output} onChange={(e) => setOutput(e.target.value)} /></label>
      {output && <select aria-label="new event output type" value={outputType} onChange={(e) => setOutputType(e.target.value)}>
        {["frame_estimate", "contact_anchor", "alignment_receipt", "completion_receipt"].map((t) => <option key={t}>{t}</option>)}
      </select>}
      <button type="submit">Add event</button>
      <button type="button" onClick={onDone}>Cancel</button>
    </form>
  );
}

function AddEntityForm({ onDone }: { onDone: () => void }) {
  const { graphEdit } = useWB();
  const [id, setId] = useState("");
  const [type, setType] = useState("manipulator");
  const [descriptor, setDescriptor] = useState("");
  return (
    <form className="add-form" aria-label="Add entity" onSubmit={async (e) => {
      e.preventDefault();
      const r = await graphEdit([{ op: "add_entity", entity: { id, type, descriptor: descriptor || id } }], `Add entity ${id}`);
      if (r) onDone();
    }}>
      <label>id <input aria-label="new entity id" required value={id} onChange={(e) => setId(e.target.value)} /></label>
      <label>type <select aria-label="new entity type" value={type} onChange={(e) => setType(e.target.value)}>
        {["manipulator", "object", "feature", "sensor", "body"].map((t) => <option key={t}>{t}</option>)}
      </select></label>
      <label>descriptor <input aria-label="new entity descriptor" value={descriptor} onChange={(e) => setDescriptor(e.target.value)} /></label>
      <button type="submit">Add entity</button>
      <button type="button" onClick={onDone}>Cancel</button>
    </form>
  );
}

export { ROLES };
