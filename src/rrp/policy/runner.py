"""Learned policy adapter: Policy.prepare/sample -> versioned ActionChunk -> controller queue.

Every chunk is labelled source="learned" with policy/codec/controller/graph versions; the
executor rejects it if any version became stale.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from rrp.contracts.action import ActionChunk, GroupCommand
from rrp.data.collect import featurizer_for
from rrp.learning.checkpoint import load_checkpoint
from rrp.model.batch import collate_inputs
from rrp.model.codec import ActionCodec, CodecConfig
from rrp.model.flow import FlowPolicy, PolicyConfig


@dataclass
class PolicyStats:
    calls: int = 0
    nfe: int = 0
    prepare_s: float = 0.0
    sample_s: float = 0.0
    latencies: list = field(default_factory=list)


class LearnedPolicy:
    def __init__(self, model: FlowPolicy, codec: ActionCodec | None, device, *, nfe: int = 8,
                 execute_prefix: int = 8, name: str = "policy", seed: int = 0):
        self.model = model.eval()
        self.codec = codec.eval() if codec is not None else None
        self.device = device
        self.nfe = nfe
        self.execute_prefix = execute_prefix
        self.name = name
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.stats = PolicyStats()
        self._feat = {}
        self._prev = {}

    @classmethod
    def from_checkpoint(cls, path, device="cpu", **kw):
        st = load_checkpoint(path, map_location=device)
        cfg = st["config"]
        model = FlowPolicy(PolicyConfig(**cfg["policy"])).to(device)
        model.load_state_dict(st["model"])
        codec = None
        if cfg.get("codec_checkpoint"):
            cs = load_checkpoint(cfg["codec_checkpoint"], map_location=device)
            codec = ActionCodec(CodecConfig(**cs["config"]["codec"])).to(device)
            codec.load_state_dict(cs["model"])
        return cls(model, codec, device, name=cfg["policy"].get("name", "policy"), **kw)

    def featurizer(self, session):
        # stored ON the session: id()-keyed caches go stale when sessions are garbage-collected
        # and a new session (possibly another robot) reuses the id (bug found 2026-09-21)
        f = getattr(session, "_rrp_featurizer", None)
        if f is None:
            f = featurizer_for(session)
            session._rrp_featurizer = f
            session._rrp_prev_action = None
        return f

    @torch.no_grad()
    def chunks(self, sessions: list) -> list[ActionChunk]:
        t0 = time.perf_counter()
        feats, obs_list = [], []
        for s in sessions:
            f = self.featurizer(s)
            obs = s.observe()
            obs_list.append(obs)
            feats.append(f(obs, getattr(s, "_rrp_prev_action", None)))
        batch = collate_inputs(feats).to(self.device)
        H = self.model.cfg.horizon
        t1 = time.perf_counter()
        cache = self.model.prepare(batch)
        z = self.model.sample(cache, H, nfe=self.nfe, generator=self.gen)
        if self.codec is not None:
            a, _ = self.codec.decode(z, batch.node_feats, batch.node_mask)
        else:
            a = z[..., 0]
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        a = a.float().cpu().numpy()
        out = []
        for i, (s, pi, obs) in enumerate(zip(sessions, feats, obs_list)):
            f = self.featurizer(s)
            n = pi.act_node_feats.shape[0]
            ai = np.clip(a[i, :, :n], -6, 6)
            groups_seq = f.aspace.denormalize(ai, pi.q0)
            s._rrp_prev_action = ai[min(self.execute_prefix, H) - 1]
            gnames = sorted(set(f.aspace.node_group), key=f.aspace.node_group.index)
            cg = []
            for g in gnames:
                vals = np.array([[row[g][c] for c in range(len(row[g]))] for row in groups_seq], float)
                cg.append(GroupCommand(group=g, values=vals, mask=np.ones_like(vals, bool)))
            rt = s.runtime
            out.append(ActionChunk(observation_id=obs.observation_id, graph_version=rt.graph_version,
                                   runtime_version=rt.runtime_version, robot_spec_hash=pi.meta["spec_hash"],
                                   controller_version=s.controller_version(), policy_version=self.name,
                                   codec_version=self.codec.cfg.version if self.codec else None,
                                   start_time=float(s.data.time), dt=s.dt, horizon=H, command_groups=cg,
                                   sampling_seed=None, source="learned"))
        self.stats.calls += len(sessions)
        self.stats.nfe += self.nfe
        self.stats.prepare_s += t2 - t1
        self.stats.latencies.append((time.perf_counter() - t0) / 1.0)
        return out

    def act(self, session):
        """Workbench hook: submit one chunk for a single session."""
        ch = self.chunks([session])[0]
        session.submit_chunk(ch, execute_prefix=self.execute_prefix)
