import { useState } from "react";
import { useWB } from "../common/store";
import { fmt, statusColor } from "../common/labels";
import { ROLES, bindingText } from "./GraphPanel";
import type { Condition } from "../transport/responses";

function Cond({ c }: { c: Condition }) {
  return (
    <li><code>{c.predicate}({c.arguments.map(bindingText).join(", ")})</code> {c.comparison} {fmt(c.value)}
      <span className="muted small"> [{c.source}{c.persistence_seconds ? `, hold ${c.persistence_seconds}s` : ""}]</span></li>
  );
}

function CondList({ title, list }: { title: string; list: Condition[] }) {
  return (
    <div className="cond-list"><h5>{title}</h5>
      {list.length ? <ul>{list.map((c, i) => <Cond key={i} c={c} />)}</ul> : <span className="muted small">none</span>}
    </div>
  );
}

export function EventInspector({ eventId }: { eventId: string }) {
  const { snapshot, graphEdit, command, editBase, setEditBase, toast } = useWB();
  const [prio, setPrio] = useState(0);
  const [role, setRole] = useState("actor");
  const [ordinal, setOrdinal] = useState(0);
  const [bkind, setBkind] = useState<"entity" | "event_output">("entity");
  const [entity, setEntity] = useState("");
  const [producer, setProducer] = useState("");
  const [outputName, setOutputName] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [inspectReceipt, setInspectReceipt] = useState<number | null>(null);
  if (!snapshot) return null;
  const doc = snapshot.graph.document;
  const ev = doc.events.find((e) => e.id === eventId);
  if (!ev) return <div className="muted">Event {eventId} no longer exists (graph v{snapshot.graph.version}).</div>;
  const rt = snapshot.runtime.events[eventId];
  const receipts = snapshot.runtime.receipts.filter((r) => r.event_id === eventId);
  const producers = doc.events.filter((e) => e.produces.length && e.id !== eventId);
  const prod = doc.events.find((e) => e.id === producer);
  const stale = editBase !== null && editBase !== snapshot.graph.version;

  const request = async () => {
    const r = await command<{ accepted: boolean; reason_code: string | null; detail: string | null }>(
      { type: "request_event", event_id: eventId, expected_version: snapshot.graph.version }, `Request ${eventId}`);
    if (!r) return;
    if (r.accepted) toast({ level: "info", title: `Execution of ${eventId} accepted`, message: "guards, prerequisites and ownership checked by the backend" });
    else toast({ level: "error", title: `Execution request for ${eventId} rejected`, code: r.reason_code ?? undefined, message: r.detail ?? "" });
  };

  return (
    <div className="event-inspector" data-testid="event-inspector">
      <h4>Event <code>{ev.id}</code> <span className="muted">operator</span> <code>{ev.operator}</code>
        <span className="en-status" style={{ background: statusColor(rt?.status) }}>{rt?.status}</span></h4>
      <div className="kv">
        <span>reason</span><span data-testid="event-reason">{rt?.reason ?? "—"}</span>
        <span>attempt</span><span>{rt?.attempt}</span>
        <span>priority</span><span>{snapshot.graph.priorities[eventId] ?? "default"}</span>
        <span>requires completed</span><span>{ev.requires_completed.join(", ") || "—"}</span>
        <span>requires active</span><span>{ev.requires_active.join(", ") || "—"}</span>
        <span>resources</span><span>{ev.resources.map((r) => `${r.entity.id}:${r.mode}`).join(", ") || "—"}</span>
        <span>timeout / recovery</span><span>{ev.timeout_seconds}s · {ev.recovery.max_attempts}× {ev.recovery.on_failure}</span>
      </div>
      <h5>Ordered roles</h5>
      <table className="tbl"><thead><tr><th>role[ordinal]</th><th>binding</th><th /></tr></thead><tbody>
        {[...ev.roles].sort((a, b) => a.role.localeCompare(b.role) || a.ordinal - b.ordinal).map((r) => (
          <tr key={`${r.role}${r.ordinal}`} data-testid={`role-${r.role}-${r.ordinal}`}>
            <td>{r.role}[{r.ordinal}]</td><td>{r.binding.kind === "entity" ? `entity ${r.binding.entity.id} v${r.binding.entity.version}` : `output ${bindingText(r.binding)}`}</td>
            <td><button className="small" onClick={() => void graphEdit([{ op: "unbind_role", event_id: eventId, role: r.role, ordinal: r.ordinal }], `Unbind ${r.role}[${r.ordinal}]`)}>unbind</button></td>
          </tr>
        ))}
      </tbody></table>
      <CondList title="Preconditions" list={ev.preconditions} />
      <CondList title="Invariants" list={ev.invariants} />
      <CondList title="Desired effects (targets, not facts)" list={ev.desired_effects} />
      <CondList title="Completion" list={ev.completion} />
      <h5>Produces</h5>
      {ev.produces.length ? <ul>{ev.produces.map((o) => <li key={o.name}><code>{o.name}</code> : {o.type} ({o.availability})</li>)}</ul> : <span className="muted small">no outputs</span>}
      <h5>Receipts</h5>
      {receipts.length ? (
        <ul className="receipts">{receipts.map((r, i) => (
          <li key={i} className={r.valid ? "rc-ok" : "rc-bad"}>
            {r.valid ? "valid" : `INVALID (${r.invalid_reason})`} · {r.output_name}:{r.type} · attempt {r.attempt} · v{r.version}
            <button className="small" onClick={() => setInspectReceipt(inspectReceipt === i ? null : i)}>inspect</button>
            {inspectReceipt === i && <pre className="receipt-value">{JSON.stringify(r.value, null, 1)}</pre>}
          </li>
        ))}</ul>
      ) : <span className="muted small">no receipts yet</span>}
      {rt?.rejections?.length ? (
        <><h5>Recent rejections</h5><ul data-testid="event-rejections">{rt.rejections.map((r, i) => <li key={i}><code>{r.reason}</code> {r.detail ? `— ${r.detail}` : ""}</li>)}</ul></>
      ) : null}

      <h5>Execution</h5>
      <div className="row">
        <button onClick={() => void request()}>Request execution</button>
        <button onClick={() => void command({ type: "cancel_event", event_id: eventId }, `Cancel ${eventId}`)}>Cancel event</button>
        <button className="debug" onClick={() => {
          if (confirm(`DEBUG OVERRIDE: force ${eventId} to succeeded?\n\nThis is NOT execution. It contaminates the run and is excluded from evaluation metrics.`))
            void command({ type: "debug_force_success", event_id: eventId, confirm: true }, "Debug override");
        }}>DEBUG force success…</button>
      </div>

      <h5>Edit <span className={stale ? "stale" : "muted"}>(based on graph v{editBase}{stale ? `; server is v${snapshot.graph.version}` : ""})</span>
        {stale && <button className="small" onClick={() => setEditBase(snapshot.graph.version)}>Rebase to v{snapshot.graph.version}</button>}</h5>
      <form className="row" onSubmit={(e) => { e.preventDefault(); void graphEdit([{ op: "set_priority", event_id: eventId, priority: prio }], `Set priority of ${eventId}`); }}>
        <label>priority <input aria-label="priority" type="number" value={prio} onChange={(e) => setPrio(Number(e.target.value))} style={{ width: 60 }} /></label>
        <button type="submit">Set priority</button>
      </form>
      <form className="bind-form" aria-label="Bind role slot" onSubmit={(e) => {
        e.preventDefault();
        const binding = bkind === "entity" ? { kind: "entity", entity: { id: entity, version: 0 } }
          : { kind: "event_output", event_id: producer, attempt, output_name: outputName };
        void graphEdit([{ op: "bind_role", event_id: eventId, role, ordinal, binding }], `Bind ${role}[${ordinal}]`);
      }}>
        <label>role <select aria-label="bind role" value={role} onChange={(e) => setRole(e.target.value)}>{ROLES.map((r) => <option key={r}>{r}</option>)}</select></label>
        <label>ordinal <input aria-label="bind ordinal" type="number" min={0} value={ordinal} onChange={(e) => setOrdinal(Number(e.target.value))} style={{ width: 48 }} /></label>
        <label>source <select aria-label="binding kind" value={bkind} onChange={(e) => setBkind(e.target.value as "entity" | "event_output")}>
          <option value="entity">entity</option><option value="event_output">event output</option>
        </select></label>
        {bkind === "entity" ? (
          <select aria-label="bind entity" value={entity} onChange={(e) => setEntity(e.target.value)} required>
            <option value="">entity…</option>
            {doc.entity_declarations.map((d) => <option key={d.id} value={d.id}>{d.id} ({d.type})</option>)}
          </select>
        ) : (
          <>
            <select aria-label="producer event" value={producer} onChange={(e) => { setProducer(e.target.value); setOutputName(""); setAttempt(snapshot.runtime.events[e.target.value]?.attempt ?? 0); }} required>
              <option value="">producer…</option>{producers.map((p) => <option key={p.id}>{p.id}</option>)}
            </select>
            <select aria-label="producer output" value={outputName} onChange={(e) => setOutputName(e.target.value)} required>
              <option value="">output…</option>{prod?.produces.map((o) => <option key={o.name}>{o.name}</option>)}
            </select>
            <label>attempt <input aria-label="producer attempt" type="number" min={0} value={attempt} onChange={(e) => setAttempt(Number(e.target.value))} style={{ width: 48 }} /></label>
          </>
        )}
        <button type="submit">Bind</button>
        <p className="muted small">Rebinding actor[·] to an entity moves this event's exclusive-control claims to that entity (backend rule).</p>
      </form>
      <div className="row">
        <button onClick={() => { if (confirm(`Remove event ${eventId} from the graph?`)) void graphEdit([{ op: "remove_event", event_id: eventId }], `Remove ${eventId}`); }}>Remove event…</button>
      </div>
    </div>
  );
}
