// Response/message shapes returned by src/rrp/service/{app,sessions,probes}.py.
// These are plain dict views on the backend (not pydantic models), so they are mirrored by hand;
// request shapes are generated into ./types.ts by scripts/export_ui_types.py.
import type { MessageKind } from "./types";

export interface Capabilities {
  robots: string[];
  tasks: string[];
  modes: string[];
  policies: string[];
  message_kinds: string[];
  render: { stream_mode: boolean; host_gpu: boolean };
  labels: Record<string, string>;
}

export interface EntityRef { id: string; version: number }
export type Binding =
  | { kind: "entity"; entity: EntityRef }
  | { kind: "event_output"; event_id: string; attempt: number; output_name: string };
export interface RoleSlot { role: string; ordinal: number; binding: Binding }
export interface Condition {
  predicate: string;
  arguments: Binding[];
  comparison: string;
  value: unknown;
  source: string;
  persistence_seconds: number;
}
export interface OutputDecl { name: string; type: string; availability: string }
export interface EventDef {
  id: string;
  operator: string;
  roles: RoleSlot[];
  requires_completed: string[];
  requires_active: string[];
  preconditions: Condition[];
  invariants: Condition[];
  desired_effects: Condition[];
  completion: Condition[];
  resources: { entity: EntityRef; mode: string }[];
  produces: OutputDecl[];
  frame_binding: Binding | null;
  timeout_seconds: number;
  recovery: { max_attempts: number; on_failure: string };
}
export interface EntityDecl { id: string; type: string; descriptor: string }
export interface TaskDocument {
  schema_version: string;
  task_id: string;
  graph_version: number;
  entity_declarations: EntityDecl[];
  events: EventDef[];
  success_events: string[];
}
export interface Receipt {
  event_id: string; attempt: number; output_name: string; type: string; version: number;
  valid: boolean; value: unknown; invalid_reason: string | null;
}
export interface RuntimeEvent {
  status: string; attempt: number; reason: string | null;
  rejections: { reason: string; detail?: string | null; [k: string]: unknown }[];
}
export interface Poses { xpos: number[][]; xquat: number[][]; time: number }
export interface CommandGroup { name: string; width: number; units: string; lower: number[]; upper: number[] }
export interface RobotSummary {
  index: number; name: string; family: string; spec_hash: string; synthetic: boolean;
  joints: { address: string; name: string; type: string; range: number[] | null; mimic_of: string | null }[];
  assemblies: { id: string; kind: string; capabilities: string[]; frame_site: string; members: string[] }[];
  controller: { id: string; version: string; groups: CommandGroup[]; current_targets?: Record<string, number[]> };
  manipulators?: Record<string, string>;
  independent_controls: number; generalized_coordinates: number; lineage: string[];
}
export interface ObjectDescriptor {
  slot: number; descriptor: string; bbox_xyxy: number[] | null; position_estimate: number[] | null;
  position_cov_diag: number[] | null; visible: boolean; bound_entity: string | null; timestamp: number;
}
export interface PredicateEstimate {
  predicate: string; args: string[]; value: unknown; known: boolean; confidence: number; estimator: string;
  timestamp: number;
}
export interface Snapshot {
  session_id: string; robot: string; task: string; seed: number; seq: number;
  mode: string; policy: string | null; running: boolean; contaminated: boolean; control_label: string;
  time: number; dt: number;
  graph: {
    document: TaskDocument; version: number; priorities: Record<string, number>;
    layout: Record<string, number[]>;
    history: { request_id: string; version: number; provenance: string }[];
  };
  runtime: {
    version: number; events: Record<string, RuntimeEvent>; receipts: Receipt[];
    owners: Record<string, string>; succeeded: boolean;
  };
  observation: {
    objects: ObjectDescriptor[]; predicates: PredicateEstimate[];
    channels: { name: string; values: number[] }[];
    joints: { addresses: string[]; qpos: number[] };
  };
  robots: RobotSummary[];
  poses: Poses;
  executor: { queued: number; meta: unknown; log: unknown[] };
  interventions: Record<string, unknown>[];
  privileged_overlay?: {
    label: string; object_poses: Record<string, number[]>; held_by: Record<string, string[]>;
    predicates: Record<string, unknown>; event_completion_truth: Record<string, unknown>;
  };
}
export interface SceneGeom {
  id: number; name: string; type: number; body: number; size: number[]; pos: number[]; quat: number[];
  rgba: number[];
}
export interface SceneBody { id: number; name: string; parent: number; joints?: string[] }
export interface Scene {
  geoms: SceneGeom[]; bodies: SceneBody[];
  objects?: { sim_body: string; descriptor: string; kind: string }[];
}
export interface WsMessage {
  kind: MessageKind; seq: number; session_id: string; graph_version?: number; runtime_version?: number;
  source?: string; t?: number; payload: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}
export interface StepRecord {
  seq: number; t: number; source: string; command: Record<string, number[]> | null; graph_version: number;
  runtime_version: number; statuses: Record<string, string>; rejected: string | null; contaminated: boolean;
}
export interface Episode {
  session_id: string; robot: string; task: string; seed: number; contaminated: boolean; steps: StepRecord[];
  graph_history: unknown[]; interventions: unknown[]; reproduce: string;
}
export interface ProbeResult {
  query: string; subject: string | null; object: string | null; source: string; t: number;
  answers: Record<string, unknown>[]; null?: boolean; multiple?: boolean; unknown_reason?: string; note?: string;
}
export interface ReplayResult { steps: number; final_state_match: boolean; max_abs_qpos_deviation: number; note: string }
export interface ResourcesView {
  totals?: {
    cpu_cores: number; memory_bytes: number; gpu: number; leases: number;
    limits: { cpu_cores: number; memory_bytes: number; gpu_slots: number; disk_bytes: number };
    admission_stopped: boolean;
  };
  watchdog?: { level: string; reasons: string[]; t: number } | null;
  error?: string; read_only: boolean;
}
