"""System i (latent path): observation -> cached typed context -> flow sampling -> LatentActionChunk (R38).
The packet is the ONLY thing handed to system 0."""
from __future__ import annotations

import time

import numpy as np
import torch

from rrp.core.latent_action import LatentActionChunk, AssemblyHandle, EntityHandle
from rrp.policies.features.featurizer import cached_featurizer
from rrp.policies.features.multi import assembly_handles, multi_featurizer
from rrp.policies.nets.checkpoint import load_checkpoint
from rrp.policies.nets.batch import collate_inputs
from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
from rrp.policies.nets.latent_batch import assembly_batch


class LatentPolicy:
    def __init__(self, model: FlowPolicy, *, knot_times, latent_space_version, realizer_compat_version, device,
                 nfe=8, validity_s=0.8, name="latent_policy", seed=0):
        self.model = model.eval()
        self.knot_times = list(knot_times)
        self.lsv, self.rcv = latent_space_version, realizer_compat_version
        self.device, self.nfe, self.validity = device, nfe, validity_s
        self.name = name
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.calls = 0
        self.latencies = []

    @classmethod
    def from_checkpoint(cls, path, device="cpu", **kw):
        st = load_checkpoint(path, map_location=device)
        rep = load_checkpoint(st["config"]["representation"], map_location="cpu")
        # training snapshots (policy_last.pt) carry no result: versions come from the representation
        res = (st.get("extra") or {}).get("result") or rep["extra"]["result"]
        from rrp.policies.nets.semantic_latent import LatentConfig
        lcfg = LatentConfig(**rep["config"]["latent"])
        pc = PolicyConfig(**dict(st["config"]["policy"], horizon=lcfg.knots, latent_dim=lcfg.dz, aux=False))
        m = FlowPolicy(pc).to(device)
        m.load_state_dict(st["model"])
        from rrp.policies.system0 import bundle_versions, is_fingerprinted
        from rrp.core.errors import ControllerRejection
        lsv, rcv = bundle_versions(lcfg.version(), rep["model"]["E"], rep["model"]["R"])
        legacy = not is_fingerprinted(res["latent_space_version"])
        if not legacy and (res["latent_space_version"], res["realizer_compat_version"]) != (lsv, rcv):
            raise ControllerRejection(f"generator {path} was trained against latent space {res['latent_space_version']}, "
                                      f"but {st['config']['representation']} is now {lsv}", code="latent_space_mismatch")
        pol = cls(m, knot_times=lcfg.knot_times, latent_space_version=lsv, realizer_compat_version=rcv, device=device,
                  name=st["config"].get("name", "latent_policy"), **kw)
        pol.legacy_unfingerprinted = legacy       # trained before D-038: bound to the representation FILE, not weights
        return pol

    def featurizer(self, s):
        return cached_featurizer(s)             # W4 dedup (rrp.features.featurizer)

    @torch.no_grad()
    def packets(self, sessions, noise_keys=None) -> list[LatentActionChunk]:
        """noise_keys: optional per-session integer seeds for the initial flow noise (paired interventions:
        the same key gives the same noise whatever else is in the batch)."""
        t0 = time.perf_counter()
        feats, obs = [], []
        for s in sessions:
            o = s.observe()
            obs.append(o)
            feats.append(self.featurizer(s)(o))
        b = assembly_batch(collate_inputs(feats).to(self.device))
        cache = self.model.prepare(b)
        noise = None
        if noise_keys is not None:
            K, N, D = len(self.knot_times), b.node_mask.shape[1], self.model.cfg.latent_dim
            noise = torch.stack([torch.randn((K, N, D), generator=torch.Generator().manual_seed(int(k) % (2 ** 63)))
                                 for k in noise_keys]).to(self.device, cache.ctx.dtype)
        ns = getattr(self, "noise_scale", 1.0)          # sampling temperature of the initial flow noise (1 = standard)
        if noise is None and ns != 1.0:
            noise = ns * torch.randn((b.node_mask.shape[0], len(self.knot_times), cache.node_mask.shape[1],
                                      self.model.cfg.latent_dim), generator=self.gen, device=cache.ctx.device,
                                     dtype=cache.ctx.dtype)
        z = self.model.sample(cache, len(self.knot_times), nfe=self.nfe, generator=self.gen, noise=noise)
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        z = z.float().cpu().numpy()
        out = []
        for i, (s, o, pi) in enumerate(zip(sessions, obs, feats)):
            f = self.featurizer(s)
            M = int(b.node_mask[i].sum())
            gasms = [a for a in f.spec.assemblies if a.kind in ("gripper", "hand")][:M]
            now = float(s.data.time)
            out.append(LatentActionChunk(
                latent_space_version=self.lsv, realizer_compat_version=self.rcv, z=z[i][:, :M].astype(np.float32),
                knot_times=self.knot_times,
                assemblies=[AssemblyHandle(handle=f"asm:{f.spec.spec_hash}:{a.frame.link}", robot_index=0) for a in gasms],
                assembly_mask=[True] * M,
                entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in o.object_descriptors],
                observation_id=o.observation_id, graph_version=s.runtime.graph_version,
                runtime_version=s.runtime.runtime_version, robot_spec_hash=f.spec.spec_hash, generated_at=time.time(),
                valid_from=now, valid_until=now + self.validity, source="learned", policy_version=self.name,
                sampling=dict(nfe=self.nfe, sampler="euler")))
        self.calls += len(sessions)
        self.latencies.append(time.perf_counter() - t0)
        return out


class DualLatentPolicy(LatentPolicy):
    """System i for multi-robot scenes: one packet over all controllable grasping assemblies."""

    def featurizer(self, s):
        return multi_featurizer(s)

    @torch.no_grad()
    def packets(self, sessions, noise_keys=None) -> list[LatentActionChunk]:
        t0 = time.perf_counter()
        feats, obs = [], []
        for s in sessions:
            o = s.observe()
            obs.append(o)
            feats.append(self.featurizer(s)(o))
        b = assembly_batch(collate_inputs(feats).to(self.device))
        cache = self.model.prepare(b)
        noise = None
        if noise_keys is not None:          # paired interventions: same key -> same initial flow noise
            K, N, D = len(self.knot_times), b.node_mask.shape[1], self.model.cfg.latent_dim
            noise = torch.stack([torch.randn((K, N, D), generator=torch.Generator().manual_seed(int(k) % (2 ** 63)))
                                 for k in noise_keys]).to(self.device, cache.ctx.dtype)
        z = self.model.sample(cache, len(self.knot_times), nfe=self.nfe, generator=self.gen, noise=noise)
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        z = z.float().cpu().numpy()
        out = []
        for i, (s, o) in enumerate(zip(sessions, obs)):
            f = self.featurizer(s)
            hs, mask = assembly_handles(f, z.shape[2])
            zi = z[i].astype(np.float32)
            zi[:, ~np.array(mask)] = 0.0
            now = float(s.data.time)
            out.append(LatentActionChunk(
                latent_space_version=self.lsv, realizer_compat_version=self.rcv, z=zi, knot_times=self.knot_times,
                assemblies=hs, assembly_mask=mask,
                entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in o.object_descriptors],
                observation_id=o.observation_id, graph_version=s.runtime.graph_version,
                runtime_version=s.runtime.runtime_version, robot_spec_hash=f.spec_hash,
                generated_at=time.time(), valid_from=now, valid_until=now + self.validity, source="learned",
                policy_version=self.name, sampling=dict(nfe=self.nfe, sampler="euler")))
        self.calls += len(sessions)
        self.latencies.append(time.perf_counter() - t0)
        return out


class LatentStackPolicy:
    """Policy adapter (rrp.policies.base) over system i (LatentPolicy / DualLatentPolicy) + one system 0 per episode.

    Every `replan_ticks` ticks (and whenever system 0 holds no packet) system i emits a packet for the episodes that need
    one; the packet (after the optional `packet_hook(i, packet) -> packet` intervention) is offered to that episode's
    system 0 (a rejected packet is counted by system 0 and the declared fallback holds); then every episode's system 0
    realizes its held packet from fresh local state (Act.command, None = hold). The same schedule as
    harness.eval.latent_eval.evaluate_latent / dual_latent_eval.evaluate_dual_latent.

    Other bodies (e.g. the ComputerWorld pointer, rrp.policies.pointer) pass `make_s0(env) -> system 0` and their
    `requires`; the env clock is `env.data.time` / `env.runtime.graph_version` (MuJoCo) or `env.time` / 0."""

    def __init__(self, system_i: LatentPolicy, realizer, *, replan_ticks: int = 8, device: str = "cpu",
                 name: str = "latent", version: str | None = None, variant: str | None = None, source: str = "learned",
                 bodies=None, tasks=None, privileged: bool = False, make_s0=None, requires=None):
        from rrp.policies.base import PolicyInfo, Requirements
        self.sys_i, self.realizer, self.replan, self.device = system_i, realizer, int(replan_ticks), device
        self.dual = isinstance(system_i, DualLatentPolicy)
        self.make_s0 = make_s0
        self.packet_hook = None
        self.info = PolicyInfo(name, source, version or f"{system_i.lsv}|{system_i.rcv}", requires or Requirements(
            frozenset({"joint_position", "gripper"}), observations=frozenset({"proprio", "object_descriptors", "task_graph"}),
            body_families=frozenset({"arm", "dual_arm"}), bodies=frozenset(bodies) if bodies else None,
            tasks=frozenset(tasks) if tasks else None, privileged=privileged), variant)

    def reset(self, spec, task, seeds, *, envs=None):
        from rrp.policies.system0 import DualLatentSystem0, LatentSystem0
        cls = DualLatentSystem0 if self.dual else LatentSystem0
        self.envs = list(envs)
        if hasattr(self.sys_i, "reset"):
            self.sys_i.reset(self.envs)
        self.s0 = [self.make_s0(e) if self.make_s0 else
                   cls(self.realizer, self.sys_i.featurizer(e), latent_space_version=self.sys_i.lsv,
                       realizer_compat_version=self.sys_i.rcv, device=self.device) for e in self.envs]
        self.t = [0] * len(self.envs)

    def act(self, obs):
        from rrp.core.errors import ControllerRejection, StaleActionError
        from rrp.policies.base import Act
        idx = sorted(obs)
        need = [i for i in idx if self.t[i] % self.replan == 0 or self.s0[i].packet is None]
        emitted = {}
        if need:
            for i, p in zip(need, self.sys_i.packets([self.envs[i] for i in need])):
                if self.packet_hook is not None:
                    p = self.packet_hook(i, p)
                emitted[i] = p
                try:
                    self.s0[i].receive(p, **env_clock(self.envs[i]))
                except (ControllerRejection, StaleActionError):
                    pass
        out = {}
        for i in idx:
            e = self.envs[i]
            cv = e.controller_version() if hasattr(e, "controller_version") else None
            out[i] = Act(self.s0[i].tick(e, cv), packet=emitted.get(i))
            self.t[i] += 1
        return out


def env_clock(e) -> dict:
    """receive() kwargs: MuJoCo sessions carry data.time and a task-graph runtime; other envs (ComputerWorld) have
    `time` and no graph (version 0)."""
    if hasattr(e, "data"):
        return dict(now=float(e.data.time), graph_version=e.runtime.graph_version)
    return dict(now=float(e.time), graph_version=0)


def make_latent(*, flow: str, representation: str | None = None, device: str = "cpu", replan_ticks: int = 8,
                dual: bool = False, name: str = "latent", variant: str | None = None, **kw) -> LatentStackPolicy:
    """flow: system i checkpoint; representation: the frozen bundle whose realizer is system 0 (default: the flow's)."""
    from rrp.policies.bundles import load_representation
    cls = DualLatentPolicy if dual else LatentPolicy
    si = cls.from_checkpoint(flow, device=device, **kw)
    rep = representation or load_checkpoint(flow, map_location="cpu")["config"]["representation"]
    _, _, R, _, _ = load_representation(rep, device)
    return LatentStackPolicy(si, R, replan_ticks=replan_ticks, device=device, name=name, variant=variant,
                             version=f"learned:{flow}")
