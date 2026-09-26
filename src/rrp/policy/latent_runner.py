"""System i (latent path): observation -> cached typed context -> flow sampling -> LatentActionChunk (R38).
The packet is the ONLY thing handed to system 0."""
from __future__ import annotations

import time

import numpy as np
import torch

from rrp.contracts.latent_action import LatentActionChunk, AssemblyHandle, EntityHandle
from rrp.data.collect import featurizer_for
from rrp.learning.checkpoint import load_checkpoint
from rrp.model.batch import collate_inputs
from rrp.model.flow import FlowPolicy, PolicyConfig
from rrp.model.latent_batch import assembly_batch


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
        from rrp.model.semantic_latent import LatentConfig
        lcfg = LatentConfig(**rep["config"]["latent"])
        pc = PolicyConfig(**dict(st["config"]["policy"], horizon=lcfg.knots, latent_dim=lcfg.dz, aux=False))
        m = FlowPolicy(pc).to(device)
        m.load_state_dict(st["model"])
        from rrp.control.latent_realizer import bundle_versions, is_fingerprinted
        from rrp.contracts.errors import ControllerRejection
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
        f = getattr(s, "_rrp_featurizer", None)
        if f is None:
            f = s._rrp_featurizer = featurizer_for(s)
        return f

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
