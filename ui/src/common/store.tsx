import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Api, ApiError } from "../transport/api";
import { SessionSocket, type ConnState } from "../transport/socket";
import type { Command, ProbeRequest } from "../transport/types";
import type { Capabilities, Episode, Poses, ProbeResult, ReplayResult, Scene, Snapshot, WsMessage } from "../transport/responses";

export type Selection =
  | { kind: "body"; id: string }
  | { kind: "joint"; id: string }        // robot-local joint address, e.g. "r0/0:j0"
  | { kind: "assembly"; id: string }
  | { kind: "object"; id: string }       // sim body name of a declared object
  | { kind: "event"; id: string }
  | { kind: "entity"; id: string }
  | null;

export interface Toast {
  id: number; level: "error" | "warn" | "info"; title: string; code?: string; message?: string;
  action?: { label: string; fn: () => void };
}

export interface LogEntry { seq: number; kind: string; source?: string; graph_version?: number; summary: string; at: number }

export type GizmoMode = "goal" | "teleport";

function rid(prefix: string) {
  const r = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : Math.random().toString(36).slice(2);
  return `${prefix}-${r}`;
}

function summarize(m: WsMessage): string {
  const p = m.payload ?? {};
  switch (m.kind) {
    case "graph_committed": return `graph v${p.graph_version} committed (${p.request_id}); queue_dropped=${p.queue_dropped}`;
    case "command_rejected": return `rejected: ${p.reason}${p.message ? ` — ${p.message}` : ""}`;
    case "event_request": return `request ${p.event_id}: ${p.accepted ? "accepted" : `rejected (${p.reason_code})`}`;
    case "mode_changed": return `mode → ${p.label}`;
    case "intervention": return `INTERVENTION ${p.kind} ${p.body ?? ""} (contaminates evaluation)`;
    case "debug_override": return `DEBUG override on ${p.event_id} (excluded from evaluation)`;
    case "probe_result": return `probe ${p.query} (${p.source})`;
    case "session_reset": return `session reset (seed ${p.seed})`;
    case "event_cancelled": return `cancelled ${p.event_id}`;
    default: return JSON.stringify(p).slice(0, 120);
  }
}

export function useWorkbenchState() {
  const params = useMemo(() => new URLSearchParams(location.search), []);
  const token = params.get("token") ?? "";
  const api = useMemo(() => new Api(token), [token]);

  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(params.get("session"));
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [poses, setPoses] = useState<Poses | null>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [conn, setConn] = useState<ConnState>("closed");
  const [connects, setConnects] = useState(0);
  const [snapshotsReceived, setSnapshotsReceived] = useState(0);
  const [staleDropped, setStaleDropped] = useState(0);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [selection, setSelection] = useState<Selection>(null);
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [replayResult, setReplayResult] = useState<ReplayResult | null>(null);
  const [probeResults, setProbeResults] = useState<ProbeResult[]>([]);
  const [privileged, setPrivileged] = useState(false);
  const [privSnapshot, setPrivSnapshot] = useState<Snapshot | null>(null);
  const [streamMode, setStreamMode] = useState(params.get("lowload") === "1");
  const [editBase, setEditBase] = useState<number | null>(null);
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>("goal");
  const [goal, setGoal] = useState<[number, number, number]>([0.35, 0.0, 0.2]);
  const [teleportDraft, setTeleportDraft] = useState<{ body: string; pos: [number, number, number] } | null>(null);
  const [busy, setBusy] = useState(0);

  const sidRef = useRef(sessionId);
  sidRef.current = sessionId;
  const snapSeqRef = useRef(0);
  const lastHttpReject = useRef<{ reason: string; at: number } | null>(null);
  const sockRef = useRef<SessionSocket | null>(null);
  const toastId = useRef(1);

  const toast = useCallback((t: Omit<Toast, "id">) => {
    const id = toastId.current++;
    setToasts((ts) => [...ts.slice(-5), { ...t, id }]);
    if (t.level === "info") window.setTimeout(() => setToasts((ts) => ts.filter((x) => x.id !== id)), 4000);
  }, []);
  const dismissToast = useCallback((id: number) => setToasts((ts) => ts.filter((x) => x.id !== id)), []);

  // ---------------------------------------------------------------- snapshot refresh (coalesced)
  const inflight = useRef(false);
  const pending = useRef(false);
  const lastRefresh = useRef(0);
  const refreshTimer = useRef<number | undefined>(undefined);
  const refresh = useCallback(async (): Promise<Snapshot | null> => {
    const sid = sidRef.current;
    if (!sid) return null;
    if (inflight.current) { pending.current = true; return null; }
    inflight.current = true;
    try {
      const s = await api.snapshot(sid);
      if (sidRef.current !== sid) return null;
      if (s.seq >= snapSeqRef.current) {
        snapSeqRef.current = s.seq;
        setSnapshot(s);
        setPoses(s.poses);
      }
      lastRefresh.current = performance.now();
      return s;
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        toast({ level: "error", title: "Session no longer exists", code: e.code, message: e.message });
        setSessionId(null);
      }
      return null;
    } finally {
      inflight.current = false;
      if (pending.current) { pending.current = false; window.setTimeout(() => void refresh(), 50); }
    }
  }, [api, toast]);

  const scheduleRefresh = useCallback((minIntervalMs = 0) => {
    const since = performance.now() - lastRefresh.current;
    if (since >= minIntervalMs) { void refresh(); return; }
    if (refreshTimer.current !== undefined) return;
    refreshTimer.current = window.setTimeout(() => { refreshTimer.current = undefined; void refresh(); }, minIntervalMs - since);
  }, [refresh]);

  const reportError = useCallback((e: unknown, what: string) => {
    if (e instanceof ApiError) {
      lastHttpReject.current = { reason: e.code, at: performance.now() };
      const action = e.status === 409 ? {
        label: "Refresh & rebase",
        fn: () => { void refresh().then((s) => { if (s) setEditBase(s.graph.version); }); },
      } : undefined;
      toast({
        level: "error", title: `${what} rejected (HTTP ${e.status})`, code: e.code,
        message: e.status === 409 ? `${e.message}. Your edit was based on a stale graph version; nothing was changed.` : e.message,
        action,
      });
    } else {
      toast({ level: "error", title: `${what} failed`, message: String(e) });
    }
  }, [toast, refresh]);

  // ---------------------------------------------------------------- WebSocket
  const onMessage = useCallback((m: WsMessage) => {
    if (m.session_id !== sidRef.current) return;
    if (m.kind === "session_snapshot") {
      snapSeqRef.current = m.seq;
      setSnapshot(m.payload as Snapshot);
      setPoses((m.payload as Snapshot).poses);
      setSnapshotsReceived((n) => n + 1);
      setLog((l) => [...l.slice(-199), { seq: m.seq, kind: m.kind, summary: "authoritative snapshot (rehydrated)", at: Date.now() }]);
      return;
    }
    if (m.kind === "session_reset") {
      // the backend restarts sequence numbering on reset; rebase our high-water mark first
      snapSeqRef.current = m.seq;
      setLog((l) => [...l.slice(-199), { seq: m.seq, kind: m.kind, source: m.source, graph_version: m.graph_version, summary: summarize(m), at: Date.now() }]);
      void refresh().then((s) => { if (s) setEditBase(s.graph.version); });
      return;
    }
    if (m.seq < snapSeqRef.current) { setStaleDropped((n) => n + 1); return; }   // older than our snapshot
    if (m.kind === "state_update") {
      const p = m.payload;
      setPoses(p.poses);
      setSnapshot((s) => s ? {
        ...s, seq: m.seq, mode: p.mode, control_label: p.label, time: p.poses?.time ?? s.time,
        executor: { ...s.executor, queued: p.queued },
        runtime: { ...s.runtime, events: Object.fromEntries(Object.entries(s.runtime.events).map(([k, v]) => [k, { ...v, status: p.statuses?.[k] ?? v.status }])) },
      } : s);
      scheduleRefresh(400);
      return;
    }
    setLog((l) => [...l.slice(-199), { seq: m.seq, kind: m.kind, source: m.source, graph_version: m.graph_version, summary: summarize(m), at: Date.now() }]);
    if (m.kind === "command_rejected") {
      const r = lastHttpReject.current;
      const reason = m.payload?.reason as string;
      const dup = r && (r.reason === reason || (reason === "version_conflict" && r.reason === "version_conflict")) && performance.now() - r.at < 3000;
      if (!dup) toast({ level: "error", title: "Command rejected by backend", code: reason, message: m.payload?.message ?? "" });
    }
    scheduleRefresh(0);
  }, [refresh, scheduleRefresh, toast]);

  useEffect(() => {
    if (!sessionId || !token) return;
    const sock = new SessionSocket(sessionId, token, onMessage, (st, info) => {
      setConn(st);
      setConnects(info.connects);
      if (st === "open" && info.connects > 1) void refresh();   // rehydrate from GET snapshot as well
    });
    sockRef.current = sock;
    sock.start();
    return () => { sock.stop(); sockRef.current = null; };
  }, [sessionId, token, onMessage, refresh]);

  const reconnectWs = useCallback(() => sockRef.current?.reconnect(), []);
  /** After an explicit reset the backend's sequence restarts at 0. */
  const rebaseSeq = useCallback(() => { snapSeqRef.current = 0; }, []);

  // ---------------------------------------------------------------- session lifecycle
  useEffect(() => {
    api.capabilities().then(setCaps).catch((e) => reportError(e, "Load capabilities"));
  }, [api, reportError]);

  useEffect(() => {
    if (!sessionId) { setSnapshot(null); setScene(null); return; }
    const u = new URL(location.href);
    u.searchParams.set("session", sessionId);
    history.replaceState(null, "", u.toString());
    let alive = true;
    api.scene(sessionId).then((sc) => alive && setScene(sc)).catch((e) => reportError(e, "Load scene"));
    api.snapshot(sessionId).then((s) => {
      if (!alive) return;
      snapSeqRef.current = s.seq; setSnapshot(s); setPoses(s.poses); setEditBase(s.graph.version);
    }).catch((e) => reportError(e, "Load snapshot"));
    return () => { alive = false; };
  }, [api, sessionId, reportError]);

  const createSession = useCallback(async (robot: string, task: string, seed: number) => {
    try {
      setBusy((b) => b + 1);
      const r = await api.createSession({ robot, task, seed });
      snapSeqRef.current = 0;
      setEpisode(null); setReplayResult(null); setProbeResults([]); setLog([]); setSelection(null);
      setSessionId(r.session_id);
      toast({ level: "info", title: `Session ${r.session_id} created`, message: `${robot} / ${task} / seed ${seed}` });
    } catch (e) { reportError(e, "Create session"); } finally { setBusy((b) => b - 1); }
  }, [api, reportError, toast]);

  // ---------------------------------------------------------------- commands
  const command = useCallback(async <T = Record<string, unknown>,>(cmd: Command, what?: string): Promise<T | null> => {
    const sid = sidRef.current;
    if (!sid) return null;
    try {
      setBusy((b) => b + 1);
      const r = await api.command<T>(sid, cmd);
      scheduleRefresh(0);
      return r;
    } catch (e) {
      reportError(e, what ?? `Command ${cmd.type}`);
      return null;
    } finally { setBusy((b) => b - 1); }
  }, [api, reportError, scheduleRefresh]);

  const graphEdit = useCallback(async (operations: Record<string, unknown>[], what = "Graph edit") => {
    const sid = sidRef.current;
    if (!sid || !snapshot) return null;
    const expected = editBase ?? snapshot.graph.version;
    try {
      setBusy((b) => b + 1);
      const r = await api.graphEdit(sid, { expected_version: expected, request_id: rid("edit"), operations });
      setEditBase(r.graph_version as number);
      toast({ level: "info", title: `${what}: graph v${r.graph_version} committed`, message: "unsent policy chunks dropped (queue_dropped)" });
      scheduleRefresh(0);
      return r;
    } catch (e) {
      reportError(e, what);
      scheduleRefresh(0);
      return null;
    } finally { setBusy((b) => b - 1); }
  }, [api, snapshot, editBase, reportError, scheduleRefresh, toast]);

  const saveLayout = useCallback(async (positions: Record<string, number[]>) => {
    const sid = sidRef.current;
    if (!sid) return;
    try { await api.layout(sid, { positions }); } catch (e) { reportError(e, "Save layout"); }
  }, [api, reportError]);

  const runProbe = useCallback(async (req: ProbeRequest) => {
    const sid = sidRef.current;
    if (!sid) return null;
    try {
      const r = await api.probe(sid, req);
      setProbeResults((p) => [r, ...p].slice(0, 20));
      return r;
    } catch (e) { reportError(e, `Probe ${req.query}`); return null; }
  }, [api, reportError]);

  const loadEpisode = useCallback(async () => {
    const sid = sidRef.current;
    if (!sid) return null;
    try { const e = await api.episode(sid); setEpisode(e); return e; } catch (e) { reportError(e, "Load episode"); return null; }
  }, [api, reportError]);

  const replay = useCallback(async () => {
    const r = await command<ReplayResult>({ type: "replay" }, "Physics replay");
    if (r) setReplayResult(r);
    return r;
  }, [command]);

  // privileged overlay: separate fetch, separate state; never merged into the observation view
  useEffect(() => {
    if (!privileged || !sessionId) { setPrivSnapshot(null); return; }
    let alive = true;
    const tick = () => api.snapshot(sessionId, true).then((s) => alive && setPrivSnapshot(s)).catch(() => undefined);
    void tick();
    const iv = window.setInterval(tick, 1500);
    return () => { alive = false; window.clearInterval(iv); };
  }, [privileged, sessionId, api]);

  return {
    token, api, caps, sessionId, setSessionId, snapshot, poses, scene, conn, connects, snapshotsReceived, staleDropped,
    log, toasts, toast, dismissToast, selection, setSelection, episode, loadEpisode, replayResult, replay,
    probeResults, runProbe, privileged, setPrivileged, privSnapshot, streamMode, setStreamMode, editBase, setEditBase,
    gizmoMode, setGizmoMode, goal, setGoal, teleportDraft, setTeleportDraft, busy, refresh, reconnectWs, rebaseSeq,
    createSession, command, graphEdit, saveLayout, reportError,
  };
}

export type WB = ReturnType<typeof useWorkbenchState>;
const Ctx = createContext<WB | null>(null);

export function WorkbenchProvider({ children }: { children: ReactNode }) {
  const wb = useWorkbenchState();
  return <Ctx.Provider value={wb}>{children}</Ctx.Provider>;
}

export function useWB(): WB {
  const v = useContext(Ctx);
  if (!v) throw new Error("useWB outside provider");
  return v;
}
