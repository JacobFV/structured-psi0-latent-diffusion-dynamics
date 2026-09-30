"""`simple`: Ψ₀'s SIMPLE benchmark (G1 + Dex3, MuJoCo physics + Isaac Sim 5.1 rendering) as an rrp `Env` (D-140; the
env side of psi1z's closed_loop/eval_loop).

SIMPLE needs its own venv (Isaac Sim 5.1, torch 2.7), so the simulator runs in a worker process
(`rrp.envs.simple.worker`, started with that venv's python, the aarch64 preloads and the compat layer) and this class
talks to it over a 127.0.0.1 connection with a random auth key. Nothing here imports SIMPLE, Isaac or torch.

- reset(seed): seed = SIMPLE eval config index (0..9 per DR level); `split="train"` resets to a recorded training
  episode's scene instead (replay / labels).
- action space `psi0`: the 36-d Ψ₀ command (`rrp.bodies.g1_simple` layout) at 50 Hz, submitted as chunks
  (`submit_chunk`, capability `chunk_executor`); the upstream agent inside the worker turns each row into WBC/AMO
  targets. `step(None)` executes the next queued row. The observation's `chunk_request` channel is 1 exactly when the
  agent's queue is empty (the upstream agent would query its server now).
- observation: head RGB image, the 43-d `joint_qpos` as measured node state (SIMPLE exposes no joint velocities: qvel
  is zeros and must not be read), the instruction, and `psi0_state` (the 32-d state the upstream agent sends: 28
  hand/arm joints + last COMMANDED torso rpy/height; public command history).
- truth(): palms, pelvis, object poses, hand-object contacts, contact points, reward, success (privileged: labels and
  judging only). Success = SIMPLE's `_success`.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from rrp.envs.simple.compat import DEFAULT_RENDER_PROFILE, RENDER_PROFILES, ext_dir, psi_home  # noqa: F401  (stdlib only)
from rrp.envs.simple.worker import LEVELS  # stdlib/numpy only

CONTROL_HZ = 50.0
EXEC_HORIZON = 24          # rows the upstream server returns per query (and the agent executes)


def worker_env(authkey: bytes, render_profile: str = DEFAULT_RENDER_PROFILE) -> dict:
    """Environment of the worker process. `render_profile` (a key of `RENDER_PROFILES`) is set here, overriding any
    RRP_SIMPLE_RENDER of the calling shell: the profile of a run is the kwarg, never an ambient variable."""
    if render_profile not in RENDER_PROFILES:
        raise ValueError(f"unknown render_profile {render_profile!r}; known: {sorted(RENDER_PROFILES)}")
    ext = ext_dir()
    carb = ext / "venvs/simple/lib/python3.11/site-packages/isaacsim/kit/libcarb.so"
    src = str(Path(__file__).resolve().parents[3])
    return dict(os.environ, RRP_SIMPLE_COMPAT="1", RRP_PSI0_EXT=str(ext), RRP_SIMPLE_RENDER=render_profile,
                OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
                # aarch64: libcarb needs static TLS at process start (after torch it fails); libgomp: Isaac 5.1 startup check
                LD_PRELOAD=f"/lib/aarch64-linux-gnu/libgomp.so.1:{carb}",
                PYTHONPATH=os.pathsep.join([src] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
                RRP_SIMPLE_AUTHKEY=authkey.hex())


class SimpleEnv:
    def __init__(self, task: str, *, level: int = 0, sim_mode: str = "mujoco_isaac", render: bool = True,
                 instruction: str | None = None, split: str = "eval", log: str | Path | None = None,
                 worker_cmd: list[str] | None = None, start_timeout: float = 1800.0,
                 render_profile: str = DEFAULT_RENDER_PROFILE):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level}")
        if split not in ("eval", "train"):
            raise ValueError(f"split must be 'eval' or 'train', got {split!r}")
        key = secrets.token_bytes(32)
        env = worker_env(key, render_profile)          # validates render_profile before anything is started
        from multiprocessing.connection import Listener
        self.split = split
        self._listener = Listener(("127.0.0.1", 0), authkey=key)
        addr = "%s:%d" % self._listener.address
        cmd = worker_cmd or [str(ext_dir() / "venvs/simple/bin/python"), "-m", "rrp.envs.simple.worker"]
        self._log = open(log, "a") if log else subprocess.DEVNULL
        self._proc = subprocess.Popen(cmd + ["--address", addr], env=env, start_new_session=True,
                                      cwd=str(ext_dir() / "psi0/third_party/SIMPLE") if worker_cmd is None else None,
                                      stdout=self._log, stderr=subprocess.STDOUT)
        self._conn = self._accept(start_timeout)
        self.info = self._call("init", task=task, level=level, sim_mode=sim_mode, render=render, instruction=instruction)
        self.joint_names = list(self.info["joint_names"])
        self._pending, self.last_rejected = None, None
        self._obs = None
        self.spec = self._make_spec(task)

    # ------------------------------------------------------------------ Env
    def reset(self, seed: int | None = None):
        self._pending, self.last_rejected = None, None
        self._obs = self._call("reset", episode=int(seed or 0), split=self.split)
        return self.observe()

    def observe(self):
        return _observation(self._obs, self.spec, self.joint_names)

    def step(self, command=None):
        """command None: execute the next queued row. A NativeCommand with group `psi0` is a one-row chunk (it is
        queued at the agent's next query, like any submitted chunk). A step whose agent needs rows and has none is
        rejected ("chunk_required"), never silently held."""
        from rrp.envs.base import StepResult
        if command is not None:
            rows = np.asarray(command.groups["psi0"], np.float32).reshape(1, -1)
            self._pending = rows if self._pending is None else np.concatenate([self._pending, rows])
        executed = self._pending
        try:
            self._obs = self._call("step", rows=self._take_pending())
            self.last_rejected = None
        except ChunkRequired:
            self.last_rejected = "chunk_required"
        return StepResult(observation=self.observe(), qpos=np.asarray(self._obs["joint_qpos"]), time=float(self._obs["time"]),
                          rejected=self.last_rejected, source="simple_agent",
                          command=None if executed is None else {"psi0_chunk": executed.tolist()})

    def submit_chunk(self, chunk, robot: int = 0, execute_prefix: int | None = None):
        """ActionChunk with group `psi0` [H, 36] (denormalized Ψ₀ units); the first `execute_prefix` rows (default
        EXEC_HORIZON, as the upstream server returns them) are queued in the agent at its next query."""
        g = {c.group: c for c in chunk.command_groups}["psi0"]
        rows = np.asarray(g.values, np.float32)[: execute_prefix or EXEC_HORIZON]
        if rows.ndim != 2 or rows.shape[1] != 36:
            raise ValueError(f"psi0 chunk must be [H, 36], got {rows.shape}")
        self._pending = rows

    def truth(self) -> dict:
        return self._call("truth")

    def state_view(self):
        """`rrp.envs.base.StateView` (capability "privileged_truth"; D-144 R8), built from `truth()`. Labels only,
        never a policy input."""
        return simple_state_view(self.truth())

    def render(self, camera: str | None = None, *, width: int = 320, height: int = 240) -> np.ndarray:
        return self._call("render")

    def close(self) -> None:
        try:
            self._call("close")
        except Exception:  # noqa: BLE001
            pass
        for f in (self._conn.close, self._listener.close):
            try:
                f()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(self._proc.pid, 9)

    # ------------------------------------------------------------------ internals
    def _take_pending(self):
        p, self._pending = self._pending, None
        return p

    def _accept(self, timeout):
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("c", self._listener.accept()), daemon=True)
        t.start()
        t0 = time.time()
        while t.is_alive() and time.time() - t0 < timeout:
            if self._proc.poll() is not None:
                raise RuntimeError(f"SIMPLE worker exited with {self._proc.returncode} before connecting")
            t.join(1.0)
        if "c" not in box:
            os.killpg(self._proc.pid, 9)
            raise TimeoutError("SIMPLE worker did not connect")
        return box["c"]

    def _call(self, method, **kw):
        self._conn.send((method, kw))
        r = self._conn.recv()
        if r[0] == "ok":
            return r[1]
        if r[1] == "chunk_required":
            raise ChunkRequired(r[2])
        raise RuntimeError(f"SIMPLE worker {method} failed: {r[1]}\n{r[2]}")

    def _make_spec(self, task):
        return env_spec(task=task, provenance=dict(level=self.info["level"], sim_mode=self.info["sim_mode"],
                                                   render=self.info["render"], agent=self.info["agent"],
                                                   upstream=self.info["upstream"], uid_fixes=self.info["uid_fixes"]))

class ChunkRequired(Exception):
    pass


def env_spec(*, task: str, body: str | list[str] = "g1_simple", provenance: dict | None = None):
    """The static EnvSpec of `simple` (no worker, no Isaac): what `rrp matrix` negotiates against. A running
    SimpleEnv's spec is the same plus the worker's provenance (level, render profile, agent, upstream revisions)."""
    from rrp.bodies import g1_simple as G
    from rrp.envs.base import ActionSpace, BodyInfo, EnvSpec
    if body not in ("g1_simple", ["g1_simple"]):
        raise ValueError(f"simple has one body, g1_simple; got {body!r}")
    name = task.split("/", 1)[1] if task.startswith("simple/") else task
    return EnvSpec(env_id="simple", backend="isaac_simple", task=f"simple/{name}",
                   bodies=[BodyInfo(robot=0, family="humanoid", key="g1_simple", robot_spec_hash=G.spec_hash())],
                   control_hz=CONTROL_HZ,
                   action_spaces=[ActionSpace(group="psi0", kind="psi0", width=G.ACTION_DIM, rate_hz=CONTROL_HZ,
                                              units="rad|m|m/s|flag (rrp.bodies.g1_simple layout)")],
                   capabilities=["privileged_truth", "render", "chunk_executor", "images", "language", "proprio"],
                   frame={"units": "m", "up": "+z"}, provenance=provenance or {})


def make_env(*, task: str, body: str | list[str] = "g1_simple", seed: int = 0, **kw) -> SimpleEnv:
    """Registry factory (`make_env("simple", task="simple/<Task>", body="g1_simple", seed=<eval config>)`), reset to
    `seed`. kw: level, sim_mode, render, instruction, split, log, render_profile (a key of `RENDER_PROFILES`; default
    `DEFAULT_RENDER_PROFILE`; `--env-kw render_profile=pt4_iso65`)."""
    if body not in ("g1_simple", ["g1_simple"]):
        raise ValueError(f"simple has one body, g1_simple; got {body!r}")
    name = task.split("/", 1)[1] if task.startswith("simple/") else task
    from rrp.tasks.spec import SIMPLE_TASKS
    if name not in SIMPLE_TASKS:
        raise KeyError(f"unknown SIMPLE task {task!r}; known: {sorted(SIMPLE_TASKS)}")
    env = SimpleEnv(name, **kw)
    env.reset(seed)
    return env


def _observation(o: dict, spec, joint_names):
    from rrp.core.observation import ImageObs, NodeState, PolicyObservation, SensorChannel
    t = float(o["time"])
    q = np.asarray(o["joint_qpos"], np.float64)
    img = np.asarray(o["image"])
    ones = lambda n: np.ones(n, bool)  # noqa: E731
    return PolicyObservation(
        observation_id=f"simple:{o['step']}", sensor_time=t, robot_spec_hash=spec.bodies[0].robot_spec_hash,
        sensor_images=[ImageObs(camera="head_stereo_left", height=img.shape[0], width=img.shape[1], encoding="rgb8",
                                pixels=img, timestamp=t)],
        measured_node_state=NodeState(joint_addresses=[f"g1_simple/{n}" for n in joint_names], qpos=q,
                                      qvel=np.zeros_like(q), qpos_mask=ones(len(q)), timestamp=t),
        declared_sensor_channels=[
            SensorChannel(name="psi0_state", kind="command_history", values=np.asarray(o["psi0_state"], np.float64),
                          mask=ones(32), timestamp=t),
            SensorChannel(name="chunk_request", kind="flag", values=np.asarray([float(o["chunk_request"])]),
                          mask=ones(1), timestamp=t)],
        instruction=o["instruction"])


# ------------------------------------------------------------------ StateView (D-144 R8): a pure function of the
# worker's truth() dict (rrp.envs.simple.worker._sim_truth), so it is testable on a recorded dict with no Isaac / SIMPLE
# venv. SIMPLE reports no contact normal (only the boolean per hand x object pair and a mean contact point), so
# ContactState.normal is zeros (unknown) rather than fabricated; gravity is standard (SIMPLE does not report it).
SIMPLE_GRAVITY = np.array([0.0, 0.0, -9.81])


def _simple_entities(truth: dict) -> list:
    from rrp.envs.base import EntityState
    out = []
    pelvis = truth.get("pelvis")
    if pelvis is not None:
        p = np.asarray(pelvis, np.float64)
        out.append(EntityState(id="pelvis", kind="body", name="pelvis", pos=p[:3],
                               quat=p[3:7] if p.shape[0] >= 7 else None, vel=None, extent=None, mass=None,
                               friction=None, material=None, parent=None, assembly="g1_simple", body=0, visible=True,
                               attrs=None))
    for side, pos in (truth.get("palm") or {}).items():
        p = np.asarray(pos, np.float64)
        out.append(EntityState(id=f"palm:{side}", kind="body", name=f"{side}_palm", pos=p[:3], quat=None, vel=None,
                               extent=None, mass=None, friction=None, material=None, parent="pelvis",
                               assembly="g1_simple", body=0, visible=True, attrs={"side": side}))
    target = truth.get("target_name")
    for name, pose in (truth.get("objects") or {}).items():
        p = np.asarray(pose, np.float64)
        out.append(EntityState(id=f"object:{name}", kind="object", name=name, pos=p[:3],
                               quat=p[3:7] if p.shape[0] >= 7 else None, vel=None, extent=None, mass=None,
                               friction=None, material=None, parent=None, assembly=None, body=None, visible=True,
                               attrs={"is_target": name == target}))
    return out


def _simple_contacts(truth: dict, t: float) -> list:
    from rrp.envs.base import ContactState
    cpt = truth.get("contact_point") or {}
    out = []
    for key, on in (truth.get("contact") or {}).items():
        if not on:
            continue
        side, obj = key.split(":", 1)
        raw = cpt.get(side)
        pos = np.asarray(raw, np.float64) if raw is not None else np.full(3, np.nan)
        out.append(ContactState(a=f"palm:{side}", b=f"object:{obj}", pos=pos, normal=np.zeros(3), force=None, time=t))
    return out


class SimpleStateView:
    """`rrp.envs.base.StateView` over one SIMPLE `truth()` dict. Labels only, never a policy input."""

    def __init__(self, truth: dict):
        self._truth = truth
        self.caps = frozenset({"poses", "contacts"})
        self.time = float(truth.get("step", 0)) / CONTROL_HZ
        self.gravity = SIMPLE_GRAVITY
        self._entities = _simple_entities(truth)
        self._contacts = _simple_contacts(truth, self.time)

    def entities(self) -> list:
        return list(self._entities)

    def contacts(self) -> list:
        return list(self._contacts)

    def joints(self):
        from rrp.envs.base import CapabilityError
        raise CapabilityError("simple StateView has no joints (SIMPLE exposes no joint velocities / axes)")

    def camera(self, name: str):
        from rrp.envs.base import CapabilityError
        raise CapabilityError("simple StateView has no camera (intrinsics/extrinsics not in truth())")

    def ui_tree(self):
        from rrp.envs.base import CapabilityError
        raise CapabilityError("simple StateView has no ui_tree")

    def token_entity(self, token_set: str, slot) -> str | None:
        if token_set == "pelvis":
            return "pelvis" if self._truth.get("pelvis") is not None else None
        if token_set == "palm" and slot in (self._truth.get("palm") or {}):
            return f"palm:{slot}"
        if token_set == "objects" and slot in (self._truth.get("objects") or {}):
            return f"object:{slot}"
        return None


def simple_state_view(truth: dict) -> SimpleStateView:
    """Pure builder: `truth` is exactly what `rrp.envs.simple.worker.Worker.truth()` (`_sim_truth()`) returns (or a
    recorded / fixture copy of it), so this is unit-testable with no Isaac / SIMPLE venv."""
    return SimpleStateView(truth)
