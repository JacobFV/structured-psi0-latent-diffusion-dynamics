import { useEffect } from "react";
import { WorkbenchProvider, useWB } from "./common/store";
import { Toasts } from "./common/Toasts";
import { SceneView, StreamView } from "./scene/SceneView";
import { BodyList, JointControls, ModeBadge, RunControls, SessionBar, TargetEditor } from "./scene/Controls";
import { GraphPanel } from "./graph/GraphPanel";
import { Inspector } from "./inspector/Inspector";
import { Playback } from "./playback/Playback";
import { Resources } from "./resources/Resources";

function ConnBadge() {
  const { conn, connects, reconnectWs, sessionId } = useWB();
  if (!sessionId) return null;
  return (
    <span className="conn">
      <span className={`dot ${conn}`} aria-hidden /> <span data-testid="ws-state">{conn}</span>
      <span className="muted small"> (connects: <span data-testid="ws-connects">{connects}</span>)</span>
      <button className="small" onClick={reconnectWs} title="Drop the WebSocket and rehydrate from an authoritative snapshot">Reconnect WS</button>
    </span>
  );
}

function Shell() {
  const { token, snapshot, streamMode, sessionId, command } = useWB();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || t.isContentEditable) return;
      if (e.key === "." && sessionId) { e.preventDefault(); void command({ type: "step", n: 1 }); }   // non-destructive only
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sessionId, command]);
  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">RRP workbench</span>
        <SessionBar />
        <ModeBadge />
        {snapshot?.contaminated && <span className="pill contaminated" data-testid="contaminated-badge">CONTAMINATED — excluded from autonomous evaluation</span>}
        <ConnBadge />
      </header>
      {!token && <div className="banner err">No token in URL. Open the page as <code>/?token=$(cat ops/workbench-token)</code>; mutations will be rejected.</div>}
      <RunControls />
      <main className="main">
        <section className="scene-col" aria-label="Robot scene">
          <div className="scene-wrap">{streamMode ? <StreamView /> : <SceneView />}</div>
          <div className="scene-tools">
            <BodyList />
            <details open><summary>Joint targets</summary><JointControls /></details>
            <details open><summary>Target / object gizmo</summary><TargetEditor /></details>
          </div>
        </section>
        <section className="graph-col" aria-label="Event graph"><GraphPanel /></section>
        <section className="insp-col" aria-label="Inspector"><Inspector /></section>
      </main>
      <footer className="bottom">
        <Playback />
        <Resources />
      </footer>
      <Toasts />
    </div>
  );
}

export function App() {
  return <WorkbenchProvider><Shell /></WorkbenchProvider>;
}
