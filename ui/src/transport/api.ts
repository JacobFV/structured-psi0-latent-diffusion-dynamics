import type { Command, CreateSession, GraphEditRequest, LayoutRequest, ProbeRequest } from "./types";
import type {
  Capabilities, Episode, ProbeResult, ReplayResult, ResourcesView, Scene, Snapshot,
} from "./responses";

/** A backend rejection: HTTP 4xx/5xx with the service's {code, message, context} body. */
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public context: unknown = null) {
    super(message);
  }
}

export class Api {
  constructor(private token: string, private base = "") {}

  private async req<T>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = {};
    if (method !== "GET") {
      headers["content-type"] = "application/json";
      headers["x-rrp-token"] = this.token;
    }
    let res: Response;
    try {
      res = await fetch(this.base + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
    } catch (e) {
      throw new ApiError(0, "network_error", String(e));
    }
    const text = await res.text();
    let data: any = null; // eslint-disable-line @typescript-eslint/no-explicit-any
    try { data = text ? JSON.parse(text) : null; } catch { data = { message: text }; }
    if (!res.ok) {
      // FastAPI HTTPException wraps our dict under "detail"; direct JSONResponse bodies are flat.
      const d = data && typeof data.detail === "object" && data.detail !== null ? data.detail : data;
      const code = (d && (d.code as string)) || (Array.isArray(data?.detail) ? "validation_error" : `http_${res.status}`);
      const msg = (d && (d.message as string)) || (Array.isArray(data?.detail)
        ? data.detail.map((x: { msg: string; loc: unknown[] }) => `${x.loc?.join(".")}: ${x.msg}`).join("; ")
        : res.statusText);
      throw new ApiError(res.status, code, msg, d?.context ?? null);
    }
    return data as T;
  }

  capabilities() { return this.req<Capabilities>("GET", "/api/capabilities"); }
  sessions() { return this.req<{ sessions: { id: string; robot: string; task: string; mode: string }[] }>("GET", "/api/sessions"); }
  createSession(body: CreateSession) { return this.req<{ session_id: string; snapshot: Snapshot }>("POST", "/api/sessions", body); }
  snapshot(sid: string, privileged = false) {
    return this.req<Snapshot>("GET", `/api/sessions/${sid}/snapshot${privileged ? "?privileged=true" : ""}`);
  }
  scene(sid: string) { return this.req<Scene>("GET", `/api/sessions/${sid}/scene`); }
  command<T = Record<string, unknown>>(sid: string, cmd: Command) { return this.req<T>("POST", `/api/sessions/${sid}/commands`, cmd); }
  graphEdit(sid: string, body: GraphEditRequest) { return this.req<Record<string, unknown>>("POST", `/api/sessions/${sid}/graph`, body); }
  layout(sid: string, body: LayoutRequest) { return this.req<{ graph_version: number }>("PUT", `/api/sessions/${sid}/graph/layout`, body); }
  probe(sid: string, body: ProbeRequest) { return this.req<ProbeResult>("POST", `/api/sessions/${sid}/probes`, body); }
  episode(sid: string) { return this.req<Episode>("GET", `/api/sessions/${sid}/episode`); }
  replay(sid: string) { return this.command<ReplayResult>(sid, { type: "replay" }); }
  packet(sid: string) { return this.req<Record<string, any>>("GET", `/api/sessions/${sid}/packet`); } // eslint-disable-line @typescript-eslint/no-explicit-any
  resources() { return this.req<ResourcesView>("GET", "/api/resources"); }
  frameUrl(sid: string, n: number) { return `${this.base}/api/sessions/${sid}/frame?n=${n}`; }
}
