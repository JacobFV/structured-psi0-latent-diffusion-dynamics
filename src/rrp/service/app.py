"""Loopback-only workbench service. Mutations need the per-process token; WebSocket checks
origin and token. Robot/task selectors are fixed registry keys, never paths."""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import JSONResponse, Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from rrp.contracts.errors import RRPError, VersionConflict, ControllerRejection
from rrp.tasks.interventions import EditRejected
from .schemas import CreateSession, Command, GraphEditRequest, LayoutRequest, ProbeRequest, public_message_kinds
from .sessions import WorkbenchSession, robot_registry, task_registry, MODES
from . import probes as probe_mod

MAX_BODY = 256 * 1024
REPO = Path(__file__).resolve().parents[3]


class State:
    def __init__(self, token: str, port: int):
        self.token = token
        self.port = port
        self.sessions: dict[str, WorkbenchSession] = {}
        self.runners: dict[str, threading.Thread] = {}
        self.policies: dict = {}
        self.lock = threading.Lock()


def create_app(token: str | None = None, port: int = 8765, allowed_origins: list[str] | None = None,
               max_sessions: int = 4) -> FastAPI:
    st = State(token or secrets.token_urlsafe(24), port)
    origins = set(allowed_origins or [f"http://127.0.0.1:{port}", f"http://localhost:{port}"])
    app = FastAPI(title="rrp workbench", version="1.0")
    app.state.rrp = st

    @app.middleware("http")
    async def limits(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_BODY:
            return JSONResponse({"code": "payload_too_large"}, status_code=413)
        if request.method in ("POST", "PUT") and request.url.path.startswith("/api/"):
            if "application/json" not in request.headers.get("content-type", ""):
                return JSONResponse({"code": "unsupported_media_type"}, status_code=415)
        return await call_next(request)

    def auth(request: Request):
        tok = request.headers.get("x-rrp-token") or request.headers.get("authorization", "").removeprefix("Bearer ")
        if not tok or not secrets.compare_digest(tok, st.token):
            raise HTTPException(status_code=401, detail={"code": "unauthorized"})
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            raise HTTPException(status_code=403, detail={"code": "bad_origin"})

    def sess(sid: str) -> WorkbenchSession:
        s = st.sessions.get(sid)
        if s is None:
            raise HTTPException(404, detail={"code": "unknown_session"})
        return s

    def rrp_error(e: Exception, status=400):
        code = getattr(e, "code", "error")
        return JSONResponse({"code": code, "message": str(e), "context": getattr(e, "context", {})}, status_code=status)

    @app.get("/health")
    def health():
        return {"ok": True, "sessions": len(st.sessions), "schema_version": "1.0"}

    @app.get("/api/capabilities")
    def capabilities():
        return {"robots": sorted(robot_registry()), "tasks": sorted(task_registry()), "modes": list(MODES),
                "policies": sorted(st.policies), "message_kinds": list(public_message_kinds()),
                "render": {"stream_mode": True, "host_gpu": False},
                "labels": {"scripted_teacher": "privileged scripted teacher", "learned": "learned policy",
                           "user": "user teleoperation", "debug": "debug override (excluded from evaluation)"}}

    @app.get("/api/robots")
    def robots():
        return {"robots": sorted(robot_registry())}

    @app.post("/api/sessions", dependencies=[Depends(auth)])
    def create_session(req: CreateSession):
        if len(st.sessions) >= max_sessions:
            return JSONResponse({"code": "too_many_sessions"}, status_code=429)
        try:
            s = WorkbenchSession(req.robot, req.task, req.seed)
        except RRPError as e:
            return rrp_error(e)
        st.sessions[s.id] = s
        return {"session_id": s.id, "snapshot": s.snapshot_view()}

    @app.get("/api/sessions")
    def list_sessions():
        return {"sessions": [dict(id=k, robot=v.robot_key, task=v.task, mode=v.mode) for k, v in st.sessions.items()]}

    @app.get("/api/sessions/{sid}/snapshot")
    def snapshot(sid: str, privileged: bool = False):
        return sess(sid).snapshot_view(privileged=privileged)

    @app.get("/api/sessions/{sid}/scene")
    def scene(sid: str):
        from .sessions import scene_geometry
        return scene_geometry(sess(sid).sim.model)

    @app.get("/api/sessions/{sid}/frame")
    def frame(sid: str):
        data = sess(sid).render_frame()
        return Response(content=data, media_type="image/jpeg")

    @app.post("/api/sessions/{sid}/commands", dependencies=[Depends(auth)])
    def command(sid: str, cmd: Command):
        s = sess(sid)
        try:
            if cmd.type == "step":
                return {"steps": s.step(cmd.n)}
            if cmd.type == "run":
                _start_runner(st, s)
                return {"running": True}
            if cmd.type == "pause":
                s.running = False
                return {"running": False}
            if cmd.type == "reset":
                s.running = False
                s.reset(cmd.seed)
                return {"snapshot": s.snapshot_view()}
            if cmd.type == "joint_target":
                s.user_joint_target(cmd.group or "arm", cmd.values or [])
                return {"accepted": True, "mode": s.mode}
            if cmd.type == "ee_target":
                s.user_ee_target(cmd.pos, cmd.yaw)
                return {"accepted": True, "mode": s.mode}
            if cmd.type == "teleport":
                s.teleport(cmd.body, cmd.pos)
                return {"accepted": True, "contaminated": True}
            if cmd.type == "request_event":
                return s.request_event(cmd.event_id, cmd.expected_version if cmd.expected_version is not None
                                       else -1)
            if cmd.type == "cancel_event":
                s.cancel_event(cmd.event_id)
                return {"accepted": True}
            if cmd.type == "set_mode":
                pol = None
                if cmd.mode == "learned":
                    loader = st.policies.get(cmd.policy or "")
                    if loader is None:
                        raise RRPError(f"unknown policy {cmd.policy}", code="unknown_policy")
                    pol = loader()
                s.set_mode(cmd.mode, pol, cmd.policy)
                return {"mode": s.mode, "label": s.control_label()}
            if cmd.type == "debug_force_success":
                s.debug_force_success(cmd.event_id, cmd.confirm)
                return {"accepted": True, "excluded_from_evaluation": True}
            if cmd.type == "replay":
                return s.physics_replay()
        except ControllerRejection as e:
            s.publish("command_rejected", dict(reason=e.code, message=str(e)), source="user")
            return rrp_error(e, 422)
        except RRPError as e:
            return rrp_error(e, 422)
        except (KeyError, ValueError) as e:
            return rrp_error(e, 422)
        return rrp_error(ValueError("unhandled command"), 400)

    @app.get("/api/sessions/{sid}/graph")
    def graph(sid: str):
        v = sess(sid).snapshot_view()
        return {"graph": v["graph"], "runtime": v["runtime"]}

    @app.post("/api/sessions/{sid}/graph", dependencies=[Depends(auth)])
    def graph_edit(sid: str, req: GraphEditRequest):
        s = sess(sid)
        try:
            return s.apply_graph_edit(req.operations, req.expected_version, req.request_id)
        except VersionConflict as e:
            return rrp_error(e, 409)
        except EditRejected as e:
            return rrp_error(e, 422)

    @app.put("/api/sessions/{sid}/graph/layout", dependencies=[Depends(auth)])
    def layout(sid: str, req: LayoutRequest):
        s = sess(sid)
        s.layout.update({k: v[:2] for k, v in req.positions.items()})   # presentation only; no graph version change
        return {"layout": s.layout, "graph_version": s.sim.runtime.graph_version}

    @app.post("/api/sessions/{sid}/probes", dependencies=[Depends(auth)])
    def probe(sid: str, req: ProbeRequest):
        s = sess(sid)
        res = probe_mod.run_probe(s, req)
        s.publish("probe_result", res, source="user")
        return res

    @app.get("/api/sessions/{sid}/episode")
    def episode(sid: str):
        return sess(sid).export_episode()

    @app.get("/api/runs")
    def runs():
        reg = REPO / "research" / "registry.jsonl"
        rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()] if reg.exists() else []
        return {"runs": rows[-200:]}

    @app.get("/api/resources")
    def resources():
        try:
            from rrp.ops.runtime import make_broker
            br, _ = make_broker(require_watchdog=False)
            t = br.totals()
            stj = json.loads((br.state_dir / "state.json").read_text())
            return {"totals": t, "watchdog": stj.get("watchdog_last"), "read_only": True}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "read_only": True}

    @app.websocket("/ws/sessions/{sid}")
    async def ws(websocket: WebSocket, sid: str):
        origin = websocket.headers.get("origin")
        tok = websocket.query_params.get("token", "")
        if (origin and origin not in origins) or not secrets.compare_digest(tok, st.token):
            await websocket.close(code=4403)
            return
        s = st.sessions.get(sid)
        if s is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)

        def listener(msg):
            def put():
                if q.full() and msg["kind"] == "state_update":
                    return                     # drop render/state frames, never control messages
                if q.full():
                    try:
                        # evict one state frame to make room for a control message
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(msg)
            loop.call_soon_threadsafe(put)

        s.listeners.append(listener)
        try:
            # rehydrate from an authoritative snapshot, then monotonic updates
            await websocket.send_text(json.dumps(dict(kind="session_snapshot", seq=s.seq, session_id=s.id,
                                                      payload=s.snapshot_view()), default=str))
            while True:
                msg = await q.get()
                await websocket.send_text(json.dumps(msg, default=str))
        except WebSocketDisconnect:
            pass
        finally:
            if listener in s.listeners:
                s.listeners.remove(listener)
            # declared disconnect behaviour: pause autonomous execution (hold), never replay stale chunks
            if not s.listeners:
                s.running = False
                s.sim.executor.invalidate("client_disconnect", float(s.sim.data.time))

    ui_dist = REPO / "ui" / "dist"
    if ui_dist.exists():
        app.mount("/assets", StaticFiles(directory=ui_dist / "assets"), name="assets")

        @app.get("/")
        def index():
            html = (ui_dist / "index.html").read_text()
            return HTMLResponse(html)

    return app


def _start_runner(st: State, s: WorkbenchSession, rate_hz: float = 20.0):
    if s.running:
        return
    s.running = True

    def loop():
        period = 1.0 / rate_hz
        while s.running:
            t0 = time.monotonic()
            try:
                s.step(1)
            except Exception as e:  # noqa: BLE001
                s.running = False
                s.publish("command_rejected", dict(reason="runner_error", message=str(e)))
                break
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)

    th = threading.Thread(target=loop, daemon=True, name=f"runner-{s.id}")
    st.runners[s.id] = th
    th.start()


def create_test_app():
    return create_app(token="test-token")


def serve(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing to bind a non-loopback address; use an SSH tunnel for remote access")
    token = secrets.token_urlsafe(24)
    tok_path = REPO / "ops" / "workbench-token"
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(tok_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    app = create_app(token=token, port=port)
    try:
        from rrp.policy.registry import register_workbench_policies
        register_workbench_policies(app.state.rrp.policies)
    except ImportError:
        pass
    print(f"workbench: http://{host}:{port}/?token=<see {tok_path}>", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", ws_max_size=MAX_BODY)
