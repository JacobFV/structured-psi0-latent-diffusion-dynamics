"""Shared online-rollout machinery for adaptation (GRPO / EXPO-inspired).

* PolicyAdapter: featurize sessions exactly like rrp.policy.runner.LearnedPolicy, decode latents with
  the FROZEN codec (or identity for direct normalized actions), and build versioned ActionChunks.
* SDEPolicy: stochastic flow-SDE sampler that records, per chunk, the full latent path, old log-prob,
  time grid, variance schedule, valid mask, behavior version and observation provenance.
* drive(): lock-step batched episode loop (same termination rules as rrp.evaluation.runner.evaluate)
  that can start from restored snapshots and pause at a registered public event boundary.

Rewards are computed from the PRIVILEGED simulator success evaluator (allowed as a training reward in
simulation only; labelled `privileged_sim_success`). Policies never see it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from rrp.contracts.action import ActionChunk, GroupCommand
from rrp.contracts.errors import StaleActionError
from rrp.data.collect import featurizer_for
from rrp.model.batch import collate_inputs
from rrp.learning.flow_sde import SDEConfig, sample_sde

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


def drive(policy, states: list[EpisodeState], *, stop_fn=None, on_step=None) -> list[EpisodeState]:
    """Advance all states in lock step until done/paused. `stop_fn(state)` (checked after each step)
    pauses a state at a boundary; the executor queue is dropped so the next chunk is fresh."""
    while True:
        active = [st for st in states if not (st.done or st.paused)]
        if not active:
            return states
        need = [st for st in active if not st.session.executor.queue]
        if need:
            try:
                chunks = policy.chunks([st.session for st in need])
            except Exception as e:  # noqa: BLE001 - recorded, never hidden
                for st in need:
                    st.done, st.outcome, st.note = True, "crash", repr(e)[:200]
                continue
            recs = getattr(policy, "last_records", [None] * len(need))
            for st, ch, rec in zip(need, chunks, recs):
                st.calls += 1
                try:
                    st.session.submit_chunk(ch, execute_prefix=policy.execute_prefix)
                    if rec is not None:
                        rec = dict(rec, step=st.steps, executed_rows=len(st.session.executor.queue))
                        st.chunks.append(rec)
                except StaleActionError:
                    st.rejections += 1
        for st in active:
            if st.done:
                continue
            s = st.session
            try:
                r = s.step(None)
            except FloatingPointError as e:
                st.done, st.outcome, st.note = True, "crash", str(e)
                continue
            st.steps += 1
            if r.rejected:
                st.cmd_rejections += 1
            if on_step is not None:
                on_step(st, r)
            if cube_fell(s):
                st.done, st.outcome, st.fell = True, "failure", True
            elif s.runtime.succeeded():
                st.done = True
            elif st.steps >= st.max_steps:
                st.done, st.outcome = True, "timeout"
            elif stop_fn is not None and stop_fn(st):
                st.paused = True
                s.executor.invalidate("branch_point", float(s.data.time))


def finalize(st: EpisodeState) -> dict:
    """Terminal outcome + reward from the privileged simulator evaluator (training reward only)."""
    s = st.session
    priv = bool(s.privileged_success()) if st.outcome != "crash" else False
    if st.outcome is None or (st.outcome == "timeout" and priv):
        st.outcome = "success" if priv else "failure"
    grasped = s.runtime.status("grasp") == "succeeded" if "grasp" in s.runtime.instances else False
    return dict(privileged_success=priv, public_success=bool(s.runtime.succeeded()), outcome=st.outcome,
                grasp_public=bool(grasped), reward_label=REWARD_LABEL)


def event_boundary(event: str = "grasp"):
    """Public branch point: the named runtime event has status 'succeeded' (public runtime only)."""
    def fn(st: EpisodeState) -> bool:
        rt = st.session.runtime
        return event in rt.instances and rt.status(event) == "succeeded"
    return fn


def feasible(session) -> bool:
    from rrp.control.teachers import PickPlaceTeacher
    return bool(PickPlaceTeacher(session).feasibility()["feasible"])


def teacher_prefix(policy, st: EpisodeState, boundary, max_steps: int) -> bool:
    """Execute the SCRIPTED TEACHER (source=scripted_teacher, privileged planner) until the public
    boundary fires. Used only for labelled "suffix adaptation from teacher prefix" experiments; the
    learned policy's previous-action feature is set exactly as in teacher data collection."""
    from rrp.control.teachers import PickPlaceTeacher
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
