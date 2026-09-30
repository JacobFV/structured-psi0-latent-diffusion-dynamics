import { useEffect, useMemo } from "react";
import { useWB } from "../common/store";
import { SOURCE_STYLES, fmt, sourceStyle } from "../common/labels";

/** Recorded timeline (backend episode record), export and physics replay. */
export function Playback() {
  const { sessionId, snapshot, episode, loadEpisode, replay, replayResult, api, reportError } = useWB();
  const seq = snapshot?.seq ?? 0;
  const running = snapshot?.running ?? false;
  useEffect(() => {
    if (!sessionId || running) return;
    const t = window.setTimeout(() => void loadEpisode(), 150);
    return () => window.clearTimeout(t);
  }, [sessionId, seq, running, loadEpisode]);
  useEffect(() => {   // while running, refresh the recorded timeline at a low fixed rate
    if (!sessionId || !running) return;
    const iv = window.setInterval(() => void loadEpisode(), 2000);
    return () => window.clearInterval(iv);
  }, [sessionId, running, loadEpisode]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const s of episode?.steps ?? []) c[s.source] = (c[s.source] ?? 0) + 1;
    return c;
  }, [episode]);
  const steps = episode?.steps ?? [];
  const shown = steps.slice(-300);
  const download = async () => {
    if (!sessionId) return;
    try {
      const ep = await api.episode(sessionId);
      const blob = new Blob([JSON.stringify(ep, null, 1)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `episode-${sessionId}.json`;
      document.body.appendChild(a); a.click(); a.remove();
      window.setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (e) { reportError(e, "Export episode"); }
  };
  return (
    <div className="playback">
      <div className="pb-head">
        <strong>Timeline</strong>
        <span data-testid="timeline-count">{steps.length} recorded steps</span>
        {episode?.contaminated && <span className="pill contaminated">CONTAMINATED</span>}
        {Object.entries(counts).map(([k, v]) => { const st = sourceStyle(k); return <span key={k} className="src-chip" data-testid={`count-${k}`} style={{ background: st.color, color: st.fg }}>{st.label}: {v}</span>; })}
        <button onClick={() => void download()} disabled={!sessionId}>Export episode</button>
        <button onClick={() => void replay()} disabled={!sessionId}>Physics replay</button>
        {replayResult && (
          <span data-testid="replay-result" className={replayResult.final_state_match ? "ok-text" : "err"}>
            replay {replayResult.steps} steps: final_state_match={String(replayResult.final_state_match)} (max |Δq| {fmt(replayResult.max_abs_qpos_deviation, 8)}){replayResult.note ? ` — ${replayResult.note}` : ""}
          </span>
        )}
      </div>
      <div className="timeline" role="list" aria-label="recorded steps (colored by control source)" data-testid="timeline">
        {shown.map((s) => {
          const st = sourceStyle(s.source);
          return <span key={s.seq} role="listitem" className={`tick ${s.rejected ? "rej" : ""} ${s.contaminated ? "cont" : ""}`}
            style={{ background: st.color }} title={`seq ${s.seq} t=${s.t.toFixed(2)} source=${s.source} graph v${s.graph_version}${s.rejected ? ` REJECTED ${s.rejected}` : ""}`} />;
        })}
      </div>
      <div className="legend" data-testid="source-legend">
        {(["scripted_teacher", "learned", "user", "debug", "hold"] as const).map((k) => (
          <span key={k} className="src-chip" data-testid={`legend-${k}`} style={{ background: SOURCE_STYLES[k].color, color: SOURCE_STYLES[k].fg }}>{SOURCE_STYLES[k].label}</span>
        ))}
      </div>
    </div>
  );
}
