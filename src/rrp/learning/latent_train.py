"""Training for the controller-facing semantic latent (R38).

Stage A (representation, `train_representation`): target encoder E, system-0 realizer R and packet probes P jointly:
    z ~ E(context_t, task_t, morphology, demonstrated a[t:t+H])            (reparameterized)
    L_real  = || R(z, knot_times, phase=j*dt, state_{t+j}, local_{t+j}) - a*_{t+j} ||^2      (j ~ U{0..J})
    L_sem   = probe_loss(P(z, queries), privileged/public labels at t)       (semantic_weight; 0 => latent_nosem)
    L       = L_real + semantic_weight * L_sem + beta * KL(q(z) || N(0, I))
state_{t+j} comes from the stored rollout at t+j, INCLUDING DART execution-noise (off-nominal) episodes; labels
a*_{t+j} are the teacher's corrective 1-step commands. For latent_nosem a separate probe is afterwards trained on
the frozen (detached) z for MEASUREMENT only (`fit_probes_on_frozen`).

Stage B (system i, `train_latent_flow`): FlowPolicy over (knots x assemblies) generating z with the frozen E mean
as the clean target; semantic loss through the FROZEN probe on z_hat_clean = z_tau + (1 - tau) v (gradients flow
into the flow model; probe/encoder parameters are frozen, not detached).
"""
from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rrp.learning.checkpoint import save_checkpoint, load_checkpoint
from rrp.learning.packed import PackedChunkDataset
from rrp.model.batch import Batch
from rrp.model.flow import FlowPolicy, PolicyConfig, interpolate_target, masked_mse
from rrp.model.latent_probes import PacketProbe, probe_loss, probe_metrics
from rrp.model.semantic_latent import LatentConfig, TargetEncoder, assembly_tokens
from rrp.control.latent_realizer import LatentRealizer, REALIZER_RECURRENT_STATE
from rrp.ops.jobs import CheckpointSignal


def _dev():
    if torch.cuda.is_available():
        from rrp.ops.gpu import apply_cap
        apply_cap()
        return torch.device("cuda")
    return torch.device("cpu")


class LatentData:
    """Row i = (episode, t). Realizer targets use row i+j of the same episode (stride-1 packing required)."""

    def __init__(self, packed_dir: Path):
        self.ds = PackedChunkDataset(packed_dir)
        if self.ds.meta["stride"] != 1:
            raise ValueError("latent training needs stride-1 packing (state at t+j)")
        self.ep = np.asarray(self.ds.arr["ep_idx"])
        self.t = np.asarray(self.ds.arr["t"])
        self.n = len(self.ds)

    def sample(self, B: int, rng: random.Random, max_j: int, idx_pool=None):
        pool = idx_pool if idx_pool is not None else range(self.n)
        sel = np.array(sorted(rng.sample(pool, B)) if not isinstance(pool, range) else
                       sorted(rng.sample(range(self.n), B)))
        j = np.array([rng.randint(0, max_j) for _ in range(B)])
        tgt = np.minimum(sel + j, self.n - 1)
        same = self.ep[tgt] == self.ep[sel]
        tgt = np.where(same, tgt, sel)            # phase beyond episode end -> j=0
        j = np.where(same, j, 0)
        return sel, tgt, j

    def fetch(self, sel, tgt, dev):
        batch, a, v, lab, eff = self.ds.collate(sel)
        A = self.ds.arr
        order = np.argsort(tgt)
        inv = np.argsort(order)
        ts = tgt[order]
        nodes = np.asarray(A["node"][ts]).astype(np.float32)[inv]
        nn_ = np.asarray(A["n_nodes"][ts])[inv]
        a1 = np.asarray(A["a"][ts][:, 0]).astype(np.float32)[inv]           # 1-step teacher command at t+j
        v1 = np.asarray(A["valid"][ts][:, 0])[inv]
        loc = np.asarray(A["local"][ts]).astype(np.float32)[inv]
        N = batch.node_feats.shape[1]
        lab["subtask"] = torch.from_numpy(np.asarray(A["subtask"][sel]).astype(np.int64))
        r = dict(node=torch.from_numpy(nodes[:, :N]), node_mask=torch.from_numpy(np.arange(N)[None] < nn_[:, None]),
                 a1=torch.from_numpy(a1[:, :N]), v1=torch.from_numpy(v1[:, :N]), local=torch.from_numpy(loc))
        mv = lambda x: x.to(dev, non_blocking=True)
        return batch.to(dev), mv(a[..., 0]), mv(v), {k: mv(x) for k, x in lab.items()}, {k: mv(x) for k, x in r.items()}


def representation_step(E, R, P, data_b, cfg: LatentConfig, train=True):
    batch, a, v, lab, r, j = data_b
    af, am, ai = assembly_tokens(batch)
    mu, logvar = E(batch, a, v, af, am, ai)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if train else mu
    kt = torch.tensor(cfg.knot_times, dtype=z.dtype, device=z.device)
    phase = torch.as_tensor(j * cfg.control_dt, dtype=z.dtype, device=z.device)
    pred = R(z, am, kt, phase, r["node"], r["node_mask"], r["local"])
    m = (r["v1"] & r["node_mask"]).float()
    l_real = ((pred - r["a1"]) ** 2 * m).sum() / m.sum().clamp(min=1)
    mm = am[:, None, :, None].float().expand_as(mu)
    kl = (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar) * mm).sum() / mm.sum().clamp(min=1)
    smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
    out = P(z, am, batch.bank_tokens["scene"].shape[1])
    l_sem, logs = probe_loss(out, lab, smask)
    loss = l_real + cfg.semantic_weight * l_sem + cfg.beta_kl * kl
    logs.update(real=float(l_real.detach()), kl=float(kl.detach()), sem=float(l_sem.detach()))
    return loss, logs, (z, am, out, lab, smask)


def train_representation(cfg_json: dict, out_dir: Path) -> dict:
    dev = _dev()
    sig = CheckpointSignal()
    cfg = LatentConfig(**cfg_json["latent"])
    seed = cfg_json.get("seed", 0)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = LatentData(Path(cfg_json["packed_dir"]))
    E, R = TargetEncoder(cfg).to(dev), LatentRealizer(cfg.dz, layers=cfg.realizer_layers).to(dev)
    P = PacketProbe(cfg.dz, cfg.knots).to(dev)
    params = list(E.parameters()) + list(R.parameters()) + list(P.parameters())
    opt = torch.optim.AdamW(params, lr=cfg_json.get("lr", 3e-4), weight_decay=1e-4)
    steps = cfg_json["steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg_json.get("lr", 3e-4), total_steps=steps, pct_start=0.05)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train_log.jsonl", "a")
    t0 = time.time()
    step = 0
    last = out_dir / "rep_last.pt"
    if last.exists():
        st = load_checkpoint(last, map_location=dev)
        E.load_state_dict(st["model"]["E"]); R.load_state_dict(st["model"]["R"]); P.load_state_dict(st["model"]["P"])
        opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["extra"]["sched"]); step = st["step"]
    B = cfg_json.get("batch_size", 128)
    while step < steps and not sig.requested:
        sel, tgt, j = data.sample(B, rng, cfg.max_phase_ticks)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        loss, logs, _ = representation_step(E, R, P, (batch, a, v, lab, r, torch.as_tensor(j, device=dev)), cfg)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), gn=float(gn), **logs)) + "\n")
            log.flush()
        if step % 2000 == 0 or sig.requested:
            save_checkpoint(last, model=_bundle(E, R, P), optimizer=opt, step=step, versions=dict(latent=cfg.version()),
                            config=cfg_json, extra=dict(sched=sched.state_dict()))
    res = dict(steps=step, wall_s=time.time() - t0, interrupted=sig.requested, latent_space_version=cfg.version(),
               realizer_compat_version=f"rz-{cfg.version()}-{REALIZER_RECURRENT_STATE}",
               eval=evaluate_representation(E, R, P, data, cfg, dev) if not sig.requested else None)
    name = "representation.pt" if not sig.requested else "representation_interrupted.pt"
    save_checkpoint(out_dir / name, model=_bundle(E, R, P), optimizer=None, step=step,
                    versions=dict(latent=cfg.version()), config=cfg_json, extra=dict(result=res))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _bundle(E, R, P):
    class _B:
        def state_dict(self):
            return dict(E=E.state_dict(), R=R.state_dict(), P=P.state_dict())
    return _B()


@torch.no_grad()
def evaluate_representation(E, R, P, data, cfg, dev, n_batches=30, seed=99) -> dict:
    """Encoded-target quality: realization error vs baselines, packet-probe metrics with shuffled-z control."""
    E.eval(); R.eval(); P.eval()
    rng = random.Random(seed)
    agg, agg_sh, real, hold = {}, {}, [], []
    for _ in range(n_batches):
        sel, tgt, j = data.sample(128, rng, cfg.max_phase_ticks)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        _, _, (z, am, out, lab2, smask) = representation_step(E, R, P, (batch, a, v, lab, r, torch.as_tensor(j, device=dev)),
                                                              cfg, train=False)
        kt = torch.tensor(cfg.knot_times, device=dev)
        pred = R(z, am, kt, torch.as_tensor(j * cfg.control_dt, dtype=z.dtype, device=dev), r["node"], r["node_mask"], r["local"])
        m = (r["v1"] & r["node_mask"]).float()
        real.append(float(((pred - r["a1"]) ** 2 * m).sum() / m.sum()))
        hold.append(float(((r["a1"]) ** 2 * m).sum() / m.sum()))
        for k, (x, n) in probe_metrics(out, lab2, smask).items():
            s_, n_ = agg.get(k, (0, 0)); agg[k] = (s_ + x, n_ + n)
        zs = z[torch.randperm(z.shape[0], device=dev)]
        for k, (x, n) in probe_metrics(P(zs, am, smask.shape[1]), lab2, smask).items():
            s_, n_ = agg_sh.get(k, (0, 0)); agg_sh[k] = (s_ + x, n_ + n)
    E.train(); R.train(); P.train()
    f = lambda d: {k: (x / n if n else None) for k, (x, n) in d.items()}
    return dict(realize_mse=float(np.mean(real)), zero_action_mse=float(np.mean(hold)), probes=f(agg),
                probes_shuffled_z=f(agg_sh))
