import { useEffect, useState } from "react";
import { useWB } from "../common/store";
import type { ResourcesView } from "../transport/responses";

const gb = (b: number) => (b / 2 ** 30).toFixed(1);

/** Read-only view of the project resource broker (never mutates leases). */
export function Resources() {
  const { api } = useWB();
  const [r, setR] = useState<ResourcesView | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = () => api.resources().then((x) => alive && setR(x)).catch(() => undefined);
    void tick();
    const iv = window.setInterval(tick, 5000);
    return () => { alive = false; window.clearInterval(iv); };
  }, [api]);
  const t = r?.totals;
  return (
    <div className="resources" data-testid="resources">
      <strong>Resources</strong> <span className="muted small">(read-only broker view)</span>
      {r?.error && <div className="err small">{r.error}</div>}
      {t && (
        <div className="kv small">
          <span>leased CPU</span><span>{t.cpu_cores.toFixed(1)} / {t.limits.cpu_cores.toFixed(1)} cores</span>
          <span>leased memory</span><span>{gb(t.memory_bytes)} / {gb(t.limits.memory_bytes)} GiB</span>
          <span>leases</span><span>{t.leases}{t.admission_stopped ? " · ADMISSION STOPPED" : ""}</span>
          <span>watchdog</span><span className={r?.watchdog?.level === "ok" ? "ok-text" : "err"}>{r?.watchdog?.level ?? "unknown"}{r?.watchdog?.reasons?.length ? `: ${r.watchdog.reasons.join(", ")}` : ""}</span>
        </div>
      )}
    </div>
  );
}
