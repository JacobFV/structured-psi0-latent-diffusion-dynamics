"""SIMPLE side of `SimpleEnv` (runs in the SIMPLE venv: Isaac Sim 5.1, Python 3.11). Moved from psi1z `eval_loop`,
`closed_loop` and the simulator logging of `replay_labels` (D-140).

One SIMPLE task env + the UNMODIFIED upstream Ψ₀ agent (`psi0` on the MP/AMO path, `psi0_decoupled_wbc` on the teleop
path) whose HTTP action client is replaced by `ChunkClient`: when the agent's command queue runs empty it asks for rows,
and those rows come from the harness-side policy through `SimpleEnv.submit_chunk`. So state construction (STATE_SLICES +
last torso command), command conversion, stand warm-up, WBC/AMO and stabilization are exactly upstream's.

Protocol: the parent (`SimpleEnv`) listens on 127.0.0.1 with a random auth key; this process connects back and serves
`(method, kwargs)` requests (`init`, `reset`, `step`, `truth`, `render`, `close`) with `("ok", result)` or
`("err", code, traceback)`. Messages are plain dicts of numpy arrays: nothing from `rrp.core` is needed in this venv.

Per-step simulator truth (palms, pelvis, object poses, hand-object contacts, hand-target contact points, reward, success)
is kept for `truth()`: ANALYSIS AND LABELS ONLY, never part of the observation the policy receives.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# upstream simple/baselines/psi0*.py STATE_SLICES (joint_qpos ranges, in order) -> the first 28 state dims
STATE_SLICES = ((29, 32), (34, 36), (32, 34), (36, 43), (15, 22), (22, 29))
LEVELS = (0, 1, 2)


class NeedChunk(Exception):
    """The agent's queue is empty and no rows were submitted (the harness must submit a chunk first)."""


class ChunkClient:
    """Stands in for upstream `HttpActionClient`. Returns the submitted rows once; checks that the state the agent sends
    equals the state the policy was shown (`expect`), so the policy input and the upstream agent never drift apart."""

    def __init__(self):
        self.rows = None
        self.expect = None
        self.queries = 0

    def query_action(self, observations, instruction, state_dict, *a, **k):
        st = np.asarray(state_dict["states"], np.float32).reshape(-1)
        if self.rows is None:
            raise NeedChunk()
        if self.expect is not None and not np.allclose(st, self.expect, atol=1e-6):
            raise RuntimeError(f"agent state {st.tolist()} != state shown to the policy {self.expect.tolist()}")
        rows, self.rows, self.expect = self.rows, None, None
        self.queries += 1
        return (rows,)


def state32(joint_qpos, last_torso_rpyh) -> np.ndarray:
    """The 32-d Ψ₀ state exactly as the upstream agents build it (28 hand/arm joints + last commanded torso rpy/height)."""
    q = np.asarray(joint_qpos, np.float32)
    return np.concatenate([q[s:e] for s, e in STATE_SLICES] + [np.asarray(last_torso_rpyh, np.float32)]).astype(np.float32)


def palm_ids(model) -> dict:
    import mujoco
    out = {}
    for side in ("left", "right"):
        for name in (f"{side}_hand_palm_link", f"{side}_wrist_yaw_link"):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i >= 0:
                out[side] = i
                break
    return out


def subtree_geoms(model, body_ids) -> set:
    ids = set(body_ids)
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if model.body_parentid[b] in ids and b not in ids:
                ids.add(b); changed = True
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in ids}


class Worker:
    def __init__(self, task: str, level: int = 0, sim_mode: str = "mujoco_isaac", render: bool = True,
                 success_criteria: float | None = None, instruction: str | None = None, data_root: str | None = None):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level}")
        import gymnasium as gym
        import simple.envs  # noqa: F401
        from gymnasium.wrappers import TimeLimit
        from rrp.envs.simple.compat import fix_task_uid, psi_home
        fix_task_uid(gym.spec(f"simple/{task}").kwargs["task"])
        self.task_name, self.level, self.sim_mode, self.instruction_override = task, level, sim_mode, instruction
        self.mp = task.endswith("MP-v0")
        self.data_root = Path(data_root or psi_home() / "data")
        sc = success_criteria if success_criteria is not None else (0.9 if self.mp else 0.7)
        if self.mp:
            self.sonic, self.control_dt = None, 0.0
            raw = gym.make(f"simple/{task}", sim_mode=sim_mode, render_hz=50, headless=True, success_criteria=sc)
        else:
            from simple.cli.eval_decoupled_wbc import _make_sonic_config
            self.sonic = _make_sonic_config()
            self.control_dt = 4 * self.sonic["SIMULATE_DT"]
            raw = gym.make(f"simple/{task}", sim_mode=sim_mode, render_hz=50, headless=True, success_criteria=sc,
                           sonic_config=self.sonic)
        self.raw, self.sim = raw, raw.unwrapped
        self.task = self.sim.task
        self.max_steps = self.task.metadata.get("max_episode_steps")
        self.env = TimeLimit(raw, max_episode_steps=self.max_steps) if self.max_steps else raw
        if not render:      # labels need no images: software EGL without a GPU is ~100x slower (replay_labels, P-005)
            blank = {k: np.zeros(sp.shape, sp.dtype) for k, sp in self.sim.observation_space.spaces.items() if k != "joint_qpos"}
            self.sim._render_frame = lambda: dict(blank)
        self.client = ChunkClient()
        if self.mp:
            from simple.baselines.psi0 import Psi0Agent
            self.agent = Psi0Agent(self.task.robot, "127.0.0.1", 0, client=self.client)
        else:
            from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent
            self.agent = Psi0DecoupledWbcAgent(self.task.robot, "127.0.0.1", 0, sonic_config=self.sonic)
            self.agent.client = self.client
        self._eval_ds = None
        self.obs = self.info = None

    # ------------------------------------------------------------------ requests
    def init_info(self) -> dict:
        from rrp.envs.simple.compat import UID_FIXES, _PATCH_LOG, ext_dir, render_profile
        return dict(task=self.task_name, level=self.level, sim_mode=self.sim_mode, mp=self.mp, max_steps=self.max_steps,
                    agent=type(self.agent).__name__, joint_names=list(self.task.robot.joint_names),
                    render=render_profile(), uid_fixes=list(UID_FIXES),
                    compat_log=list(_PATCH_LOG), upstream=dict(psi0=_git(ext_dir() / "psi0"), simple=_git(ext_dir() / "psi0/third_party/SIMPLE")))

    def reset(self, episode: int, split: str = "eval") -> dict:
        """split eval: SIMPLE eval config `episode` of simple-eval/<task>/dr-level-<L> (10 per level);
        split train: the recorded training episode's scene config (replay / label generation)."""
        import torch
        conf = self._episode_config(episode, split)
        obs, info = self.env.reset(options={"state_dict": conf})
        n = 0
        if not self.mp:
            self.agent._wbc_policy.lower_body_policy.use_policy_action = True
        while not self.mp and not self.task.robot.stabilized and n < 300:
            ts = time.monotonic()
            obs, *_, info = self.env.step(self.agent.get_stabilize_action(obs))
            self.sim.update_viewer(); self.sim.update_reward()
            dt = self.control_dt - (time.monotonic() - ts)
            if dt > 0:
                time.sleep(dt)
            n += 1
        if not self.mp:
            self.agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)
        self.agent.reset()
        self.client.__init__()
        self.obs, self.info = obs, info
        instr = self.task.instruction
        try:
            tname = self.task.target.name if hasattr(self.task.target, "name") else None
        except Exception:  # noqa: BLE001
            tname = None
        if self.instruction_override:
            instr = self.instruction_override.format(tname) if "{}" in self.instruction_override else self.instruction_override
        self.instruction, self.target_name = instr, tname
        m = self.sim.mujoco.mjModel
        self.palms = palm_ids(m)
        self.objs = [k for k in info if k != "proprio"]
        ids = {k: int(self.sim.mujoco.mj_objects[k].id) for k in self.objs}
        self.hand_g = {s: subtree_geoms(m, [b]) for s, b in self.palms.items()}
        self.obj_g = {k: subtree_geoms(m, [b]) for k, b in ids.items()}
        self.steps, self.stabilize_steps, self.done, self.terminated, self.truncated = 0, n, False, False, False
        self.reward, self.max_reward, self.first_contact = 0.0, 0.0, {}
        self._truth = self._sim_truth()
        return self._pack()

    def step(self, rows=None) -> dict:
        if self.done:
            raise RuntimeError("episode is over; reset first")
        if rows is not None:
            self.client.rows = np.asarray(rows, np.float32)
            self.client.expect = self._state()
        act = self.agent.get_action(self.obs, info=self.info, instruction=self.instruction)
        if self.client.rows is not None:
            raise RuntimeError("a chunk was submitted but the agent did not ask for one (queue not empty)")
        self.obs, self.reward, term, trunc, self.info = self.env.step(act)
        self.steps += 1
        self.terminated, self.truncated = bool(term), bool(trunc)
        self.done = self.terminated or self.truncated
        self.max_reward = max(self.max_reward, float(self.reward))
        self._truth = self._sim_truth()
        return self._pack()

    def truth(self) -> dict:
        return self._truth

    def render(self) -> np.ndarray:
        return np.asarray(self.obs["head_stereo_left"])

    def close(self) -> None:
        self.raw.close()

    # ------------------------------------------------------------------ internals
    def _episode_config(self, episode: int, split: str) -> dict:
        if split == "eval":
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from simple.datasets.lerobot import get_episode_lerobot
            if self._eval_ds is None:
                self._eval_ds = LeRobotDataset(repo_id=f"simple/{self.task_name}",
                                               root=str(self.data_root / f"simple-eval/{self.task_name}/dr-level-{self.level}"))
            if not 0 <= episode < self._eval_ds.num_episodes:
                raise ValueError(f"eval config {episode} not in [0, {self._eval_ds.num_episodes})")
            return get_episode_lerobot(self._eval_ds, episode)[0]
        if split == "train":
            root = self.data_root / f"simple/{self.task_name}"
            meta = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
            return json.loads(meta[episode]["environment_config"])
        raise ValueError(split)

    def _needs_chunk(self) -> bool:
        """True when the agent's next get_action will query (MP agent: after its 60-step stand warm-up)."""
        a = self.agent
        return len(a._action_queue) == 0 and not (self.mp and a._global_step_idx == 0 and not a._skip_stand_warmup)

    def _state(self) -> np.ndarray:
        return state32(self.obs["joint_qpos"], self.agent._last_cmd_torso_rpyh)

    def _pack(self) -> dict:
        return dict(joint_qpos=np.asarray(self.obs["joint_qpos"], np.float32), image=np.asarray(self.obs["head_stereo_left"]),
                    instruction=self.instruction, psi0_state=self._state(), chunk_request=self._needs_chunk(),
                    step=self.steps, time=self.steps * 0.02, done=self.done, stabilize_steps=self.stabilize_steps)

    def _sim_truth(self) -> dict:
        d = self.sim.mujoco.mjData
        con = {(s, k): False for s in self.palms for k in self.objs}
        cpos = {s: [] for s in self.palms}
        for ci in range(d.ncon):
            g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
            for s in self.palms:
                for k in self.objs:
                    if (g1 in self.hand_g[s] and g2 in self.obj_g[k]) or (g2 in self.hand_g[s] and g1 in self.obj_g[k]):
                        con[(s, k)] = True
                        if k == "target":
                            cpos[s].append(np.asarray(d.contact[ci].pos, np.float32).copy())
        for (s, k), c in con.items():
            if c and f"{s}:{k}" not in self.first_contact:
                self.first_contact[f"{s}:{k}"] = self.steps
        return dict(source="privileged:sim", step=self.steps,
                    palm={s: d.xpos[b].astype(np.float32).copy() for s, b in self.palms.items()},
                    pelvis=np.concatenate([d.qpos[:3], d.qpos[3:7]]).astype(np.float32),
                    objects={k: np.asarray(self.info[k], np.float32) for k in self.objs},
                    contact={f"{s}:{k}": v for (s, k), v in con.items()},
                    contact_point={s: (np.mean(p, 0) if p else np.full(3, np.nan, np.float32)) for s, p in cpos.items()},
                    reward=float(self.reward), max_reward=self.max_reward, first_contact=dict(self.first_contact),
                    success=bool(self.sim._success), terminated=self.terminated, truncated=self.truncated,
                    target_name=self.target_name)


def _git(path: Path) -> dict:
    import subprocess
    try:
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return dict(sha=sha or None, dirty=dirty)
    except Exception:  # noqa: BLE001
        return dict(sha=None, dirty=None)


def serve(conn, factory) -> None:
    """Request loop. `factory(**init_kwargs)` builds the backend (the real Worker, or a test double)."""
    w = None
    while True:
        try:
            method, kw = conn.recv()
        except EOFError:
            break
        try:
            if method == "init":
                w = factory(**kw)
                res = w.init_info()
            elif method in ("reset", "step", "truth", "render"):
                res = getattr(w, method)(**kw)
            elif method == "close":
                if w is not None:
                    w.close()
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown request {method}")
            conn.send(("ok", res))
        except NeedChunk:
            conn.send(("err", "chunk_required", "the agent asked for a chunk and none was submitted"))
        except Exception as e:  # noqa: BLE001
            conn.send(("err", type(e).__name__, traceback.format_exc()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True, help="host:port of the parent's listener (127.0.0.1 only)")
    a = ap.parse_args()
    host, port = a.address.rsplit(":", 1)
    if host not in ("127.0.0.1", "localhost"):
        raise SystemExit("local connections only")
    from multiprocessing.connection import Client
    from rrp.envs.simple import compat
    compat.install()
    conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ.pop("RRP_SIMPLE_AUTHKEY")))
    serve(conn, Worker)
    os._exit(0)          # Isaac Sim's atexit handlers can hang (as upstream's eval_loop did)


if __name__ == "__main__":
    sys.exit(main())
