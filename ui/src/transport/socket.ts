import type { WsMessage } from "./responses";

export type ConnState = "connecting" | "open" | "closed";

/**
 * Session WebSocket. The server sends `session_snapshot` first, then sequenced messages.
 * This client never sends commands over the socket and keeps no outgoing queue, so a
 * reconnect can never replay stale commands; it only rehydrates from the new snapshot.
 */
export class SessionSocket {
  private ws: WebSocket | null = null;
  private stopped = false;
  private retry = 0;
  private timer: number | undefined;
  public connects = 0;

  constructor(
    private sessionId: string,
    private token: string,
    private onMessage: (m: WsMessage) => void,
    private onState: (s: ConnState, info: { connects: number }) => void,
  ) {}

  start() { this.stopped = false; this.open(); }

  private open() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws/sessions/${this.sessionId}?token=${encodeURIComponent(this.token)}`;
    this.onState("connecting", { connects: this.connects });
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => { this.retry = 0; this.connects += 1; this.onState("open", { connects: this.connects }); };
    ws.onmessage = (ev) => {
      try { this.onMessage(JSON.parse(ev.data as string) as WsMessage); } catch { /* ignore malformed */ }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.onState("closed", { connects: this.connects });
      if (!this.stopped) {
        const delay = Math.min(5000, 300 * 2 ** this.retry++);
        this.timer = window.setTimeout(() => this.open(), delay);
      }
    };
  }

  /** Deliberately drop the connection and reconnect (used to test/force rehydration). */
  reconnect() {
    window.clearTimeout(this.timer);
    const old = this.ws;
    this.ws = null;
    if (old) { old.onclose = null; old.close(); }
    this.onState("closed", { connects: this.connects });
    this.timer = window.setTimeout(() => { if (!this.stopped) this.open(); }, 150);
  }

  stop() {
    this.stopped = true;
    window.clearTimeout(this.timer);
    const old = this.ws;
    this.ws = null;
    if (old) { old.onclose = null; old.close(); }
  }
}
