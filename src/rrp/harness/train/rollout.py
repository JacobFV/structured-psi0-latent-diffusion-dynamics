"""Shared online-rollout machinery for adaptation (GRPO / EXPO-inspired).

* PolicyAdapter: featurize sessions exactly like rrp.policy.runner.LearnedPolicy, decode latents with
  the FROZEN codec (or identity for direct normalized actions), and build versioned ActionChunks.
* SDEPolicy: stochastic flow-SDE sampler that records, per chunk, the full latent path, old log-prob,
  time grid, variance schedule, valid mask, behavior version and observation provenance.
* drive(): episodes on harness.rollout (same termination rules as the arm eval) that can start from restored
  snapshots and pause at a registered public event boundary.

Rewards are computed from the PRIVILEGED simulator success evaluator (allowed as a training reward in
simulation only; labelled `privileged_sim_success`). Policies never see it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from rrp.core.action import ActionChunk, GroupCommand
from rrp.policies.features.featurizer import featurizer_for
from rrp.policies.nets.batch import collate_inputs
from rrp.harness.train.flow_sde import SDEConfig, sample_sde

REWARD_LABEL = "privileged_sim_success"   # simulator-truth evaluator used as reward (sim training only)


class PolicyAdapter:
    def __init__(self, model, codec, device, *, execute_prefix: int = 8, version: str = "policy@0"):
        self.model = model
        self.codec = codec
        self.device = device
        self.execute_prefix = execute_prefix
        self.version = version
        self._feat: dict = {}
        self.prev: dict = {}            # id(session) -> last executed normalized row (policy input feature)
        self.calls = 0                  # per-sample chunk generations
        self.batched_calls = 0
        self.velocity_evals = 0         # per-sample velocity-field evaluations (policy forwards)
        self.infer_s = 0.0

    def featurizer(self, s):
        k = id(s)
        if k not in self._feat:
            self._feat[k] = featurizer_for(s)
        return self._feat[k]

    def forget(self, sessions):
        for s in sessions:
            self._feat.pop(id(s), None)
            self.prev.pop(id(s), None)

    def copy_prev(self, src, dst):
        if id(src) in self.prev:
            self.prev[id(dst)] = self.prev[id(src)].copy()

    def featurize(self, sessions):
        feats, obs = [], []
        for s in sessions:
            o = s.observe()
            obs.append(o)
            feats.append(self.featurizer(s)(o, self.prev.get(id(s))))
        return feats, obs

    def latent_valid(self, batch, H):
        d = self.model.cfg.latent_dim
        B, N = batch.node_mask.shape
        return batch.node_mask[:, None, :, None].expand(B, H, N, d).clone()

    @torch.no_grad()
    def decode(self, z, batch):
        if self.codec is not None:
            a, _ = self.codec.decode(z, batch.node_feats, batch.node_mask)
            return a
        return z[..., 0]

    def make_chunk(self, s, pi, obs, a_norm: np.ndarray, *, policy_version: str, source: str = "learned"):
        """a_norm [rows, n] normalized actions (rows are all executed) -> ActionChunk + native rows."""
        f = self.featurizer(s)
        a_norm = np.clip(a_norm, -6, 6)
        groups_seq = f.aspace.denormalize(a_norm, pi.q0)
        gnames = sorted(set(f.aspace.node_group), key=f.aspace.node_group.index)
        cg = []
        for g in gnames:
            vals = np.array([[row[g][c] for c in range(len(row[g]))] for row in groups_seq], float)
            cg.append(GroupCommand(group=g, values=vals, mask=np.ones_like(vals, bool)))
        rt = s.runtime
        ch = ActionChunk(observation_id=obs.observation_id, graph_version=rt.graph_version,
                         runtime_version=rt.runtime_version, robot_spec_hash=pi.meta["spec_hash"],
                         controller_version=s.controller_version(), policy_version=policy_version,
                         codec_version=self.codec.cfg.version if self.codec else None, start_time=float(s.data.time),
                         dt=s.dt, horizon=a_norm.shape[0], command_groups=cg, sampling_seed=None, source=source)
        return ch, groups_seq


class SDEPolicy(PolicyAdapter):
    """Stochastic flow-SDE actor. `last_records[i]` describes the chunk returned for sessions[i]."""

    def __init__(self, model, codec, device, sde: SDEConfig, *, execute_prefix=8, version="policy@0", seed=0):
        super().__init__(model, codec, device, execute_prefix=execute_prefix, version=version)
        self.sde = sde.validate()
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.last_records: list[dict] = []

    @torch.no_grad()
    def chunks(self, sessions):
        t0 = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        feats, obs = self.featurize(sessions)
        batch = collate_inputs(feats).to(self.device)
        H = self.model.cfg.horizon
        cache = self.model.prepare(batch)
        valid = self.latent_valid(batch, H)
        noise = torch.randn(valid.shape, generator=self.gen, device=self.device, dtype=cache.ctx.dtype)
        path = sample_sde(lambda z, t: self.model.velocity(z, t, cache), noise, valid, self.sde,
                          behavior_version=self.version, generator=self.gen)
        a = self.decode(path.action, batch).float().cpu().numpy()
        if was_training:
            self.model.train()
        out, self.last_records = [], []
        P = min(self.execute_prefix, H)
        for i, (s, pi, o) in enumerate(zip(sessions, feats, obs)):
            n = pi.act_node_feats.shape[0]
            ai = a[i, :, :n]
            ch, _ = self.make_chunk(s, pi, o, ai[:P], policy_version=self.version)
            self.prev[id(s)] = np.clip(ai[P - 1], -6, 6)
            out.append(ch)
            self.last_records.append(dict(pi=pi, path=path.select(slice(i, i + 1)).to("cpu"),
                                          observation_id=o.observation_id, behavior_version=self.version,
                                          sim_time=float(s.data.time)))
        self.calls += len(sessions)
        self.batched_calls += 1
        self.velocity_evals += len(sessions) * self.sde.nfe
        self.infer_s += time.perf_counter() - t0
        return out


# ------------------------------------------------------------------ episode driver
@dataclass
class EpisodeState:
    session: object
    seed: int
    max_steps: int                   # remaining step allowance for THIS drive call
    tag: dict = field(default_factory=dict)
    steps: int = 0                   # steps executed in this drive call (new control transitions)
    done: bool = False
    paused: bool = False
    outcome: str | None = None
    fell: bool = False
    calls: int = 0
    rejections: int = 0
    cmd_rejections: int = 0
    chunks: list = field(default_factory=list)   # records from policy.last_records (+ step index)
    note: str = ""


def cube_fell(s) -> bool:
    try:
        return bool(s.data.xpos[s.model.body("cube").id][2] < -0.05)
    except KeyError:
        return False


class _DrivePolicy:
    """Policy (rrp.policies.base) over a chunk actor (PolicyAdapter / SDEPolicy): a new chunk for every episode whose
    executor queue is empty (Act.chunk, executed `execute_prefix` rows), the actor's per-chunk record attached."""

    def __init__(self, policy, states):
        from rrp.policies.base import PolicyInfo, Requirements
        self.policy, self.states = policy, states
        self.info = PolicyInfo("drive", "learned", str(getattr(policy, "version", "")),
                               Requirements(frozenset({"joint_position", "gripper"}), observations=frozenset()))

    def reset(self, spec, task, seeds, *, envs=None):
        pass

    def act(self, obs):
        from rrp.policies.base import Act
        out = {i: Act(None) for i in obs}
        need = [i for i in sorted(obs) if not self.states[i].session.executor.queue]
        if need:
            chunks = self.policy.chunks([self.states[i].session for i in need])
            recs = getattr(self.policy, "last_records", [None] * len(need))
            for i, ch, rec in zip(need, chunks, recs):
                self.states[i].calls += 1
                out[i] = Act(None, chunk=ch, info=dict(execute_prefix=self.policy.execute_prefix, record=rec))
        return out


class _DriveHook:
    """on_step: per-state bookkeeping of drive(): chunk records (accepted submissions), step and rejection counts, the
    caller's on_step observer, the state's own step allowance ("timeout") and the pause boundary (stop_fn)."""

    def __init__(self, states, stop_fn, on_step):
        self.states, self.stop_fn, self.observer = states, stop_fn, on_step

    def on_step(self, i, env, act, r):
        from rrp.tasks.spec import Judgement
        st = self.states[i]
        if act.chunk is not None and not act.info.get("chunk_rejected"):
            rec = act.info.get("record")
            if rec is not None:
                st.chunks.append(dict(rec, step=st.steps, executed_rows=act.chunk.horizon))
        elif act.chunk is not None:
            st.rejections += 1
        st.steps += 1
        if r.rejected:
            st.cmd_rejections += 1
        if self.observer is not None:
            self.observer(st, r)
        s = st.session
        if cube_fell(s) or s.runtime.succeeded():
            return None                                  # the task judge ends it
        if st.steps >= st.max_steps:
            return Judgement(True, "timeout", "timeout")
        if self.stop_fn is not None and self.stop_fn(st):
            s.executor.invalidate("branch_point", float(s.data.time))
            return Judgement(True, "timeout", "paused")
        return None


def drive(policy, states: list[EpisodeState], *, stop_fn=None, on_step=None) -> list[EpisodeState]:
    """Advance all active states through harness.rollout until done/paused. `stop_fn(state)` (checked after each
    step) pauses a state at a boundary; the executor queue is dropped so the next chunk is fresh. Termination: the
    object dropped (failure), the public task graph completed (done; finalize() judges privileged success), the
    state's step allowance, or the pause. A crash of the actor ends the running episodes as "crash"."""
    import math
    from rrp.harness.rollout import rollout
    from rrp.tasks.spec import get_task
    active = [st for st in states if not (st.done or st.paused)]
    if not active:
        return states
    eps = rollout(lambda i: active[i].session, _DrivePolicy(policy, active), get_task("pick_place"),
                  list(range(len(active))), batch=len(active), max_seconds=math.inf,
                  hooks=[_DriveHook(active, stop_fn, on_step)])
    for st, ep in zip(active, eps):
        if ep.failure_reason == "paused":
            st.paused = True
        elif ep.outcome == "crash":
            st.done, st.outcome, st.note = True, "crash", ep.metrics.get("note", "")
        elif ep.failure_reason == "dropped_off_table":
            st.done, st.outcome, st.fell = True, "failure", True
        elif ep.failure_reason == "timeout" and not st.session.runtime.succeeded():
            st.done, st.outcome = True, "timeout"
        else:
            st.done = True
    return states


def finalize(st: EpisodeState) -> dict:
    """Terminal outcome + reward from the privileged simulator evaluator (training reward only)."""
    s = st.session
    priv = bool(s.privileged_success()) if st.outcome != "crash" else False
    if st.outcome is None or (st.outcome == "timeout" and priv):
        st.outcome = "success" if priv else "failure"
    grasped = s.runtime.status("grasp") == "succeeded" if "grasp" in s.runtime.instances else False
    return dict(privileged_success=priv, public_success=bool(s.runtime.succeeded()), outcome=st.outcome,
                grasp_public=bool(grasped), reward_label=REWARD_LABEL, cube_zone_xy=cube_zone_xy(s))


def cube_zone_xy(s) -> float | None:
    """PRIVILEGED simulator truth (reward shaping only): final cube-to-target-zone horizontal distance."""
    try:
        c = s.data.xpos[s.model.body("cube").id]
        z = s.data.xpos[s.model.body("target_zone").id]
    except KeyError:
        return None
    return float(np.linalg.norm(c[:2] - z[:2]))


def event_boundary(event: str = "grasp"):
    """Public branch point: the named runtime event has status 'succeeded' (public runtime only)."""
    def fn(st: EpisodeState) -> bool:
        rt = st.session.runtime
        return event in rt.instances and rt.status(event) == "succeeded"
    return fn


def feasible(session) -> bool:
    from rrp.policies.teachers.arm import PickPlaceTeacher
    return bool(PickPlaceTeacher(session).feasibility()["feasible"])


def teacher_prefix(policy, st: EpisodeState, boundary, max_steps: int) -> bool:
    """Execute the SCRIPTED TEACHER (source=scripted_teacher, privileged planner) until the public
    boundary fires. Used only for labelled "suffix adaptation from teacher prefix" experiments; the
    learned policy's previous-action feature is set exactly as in teacher data collection."""
    from rrp.policies.teachers.arm import PickPlaceTeacher
    s = st.session
    t = PickPlaceTeacher(s)
    f = policy.featurizer(s)
    while st.steps < max_steps:
        pi_q0 = f(s.observe(), policy.prev.get(id(s))).q0
        c = t.act()
        policy.prev[id(s)] = f.aspace.normalize([c.groups], pi_q0)[0]
        s.step(c)
        st.steps += 1
        st.tag["teacher_steps"] = st.tag.get("teacher_steps", 0) + 1
        if cube_fell(s) or s.runtime.succeeded():
            st.done = True
            return False
        if boundary(st):
            s.executor.invalidate("branch_point", float(s.data.time))
            return True
    st.done, st.outcome = True, "timeout"
    return False
