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
        if "held_m" in A:          # multi-assembly pack (rrp.learning.dual_latent)
            S = lab["held"].shape[1]
            for k in ("held_m", "contact_m", "rel_tcp_m"):
                x = np.asarray(A[k][sel])[:, :S]
                lab[k] = torch.from_numpy(x.astype(np.float32) if x.dtype == np.float16 else x)
            lab["subtask_m"] = torch.from_numpy(np.asarray(A["subtask_m"][sel]).astype(np.int64))
            na = np.asarray(A["node_asm"][ts])[inv][:, :N].astype(np.int64)            # slot of each node at t+j
            locm = np.asarray(A["local_m"][ts]).astype(np.float32)[inv]                 # [B,M,4] at t+j
            r["node_asm"] = torch.from_numpy(na)
            r["local"] = torch.from_numpy(np.take_along_axis(locm, na[..., None].clip(0, locm.shape[1] - 1), 1))  # per node
        mv = lambda x: x.to(dev, non_blocking=True)
        return batch.to(dev), mv(a[..., 0]), mv(v), {k: mv(x) for k, x in lab.items()}, {k: mv(x) for k, x in r.items()}


def _prefetch(data, B, rng, max_j, dev, depth: int = 4):
    """Background thread: sample + collate on CPU ahead of the GPU step (same rng sequence as the serial loop)."""
    import queue
    import threading
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def work():
        while not stop.is_set():
            sel, tgt, j = data.sample(B, rng, max_j)
            item = (data.fetch(sel, tgt, "cpu"), j)
            while not stop.is_set():
                try:
                    q.put(item, timeout=1)
                    break
                except queue.Full:
                    pass

    th = threading.Thread(target=work, daemon=True)
    th.start()
    mv = lambda x: x.to(dev, non_blocking=True)
    try:
        while True:
            (batch, a, v, lab, r), j = q.get()
            yield batch.to(dev), mv(a), mv(v), {k: mv(x) for k, x in lab.items()}, {k: mv(x) for k, x in r.items()}, j
    finally:
        stop.set()


def representation_step(E, R, P, data_b, cfg: LatentConfig, train=True):
    batch, a, v, lab, r, j = data_b
    af, am, ai = assembly_tokens(batch)
    mu, logvar = E(batch, a, v, af, am, ai)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if train else mu
    kt = torch.tensor(cfg.knot_times, dtype=z.dtype, device=z.device)
    phase = torch.as_tensor(j * cfg.control_dt, dtype=z.dtype, device=z.device)
    pred = R(z, am, kt, phase, r["node"], r["node_mask"], r["local"], node_asm=r.get("node_asm"))
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
    P = PacketProbe(cfg.dz, cfg.knots, **cfg_json.get("probe", {})).to(dev)
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
    feed = _prefetch(data, B, rng, cfg.max_phase_ticks, dev) if cfg_json.get("prefetch") else None
    while step < steps and not sig.requested:
        if feed is not None:
            batch, a, v, lab, r, j = next(feed)
        else:
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
        pred = R(z, am, kt, torch.as_tensor(j * cfg.control_dt, dtype=z.dtype, device=dev), r["node"], r["node_mask"], r["local"],
                 node_asm=r.get("node_asm"))
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


def load_representation(path: Path, dev):
    st = load_checkpoint(path, map_location=dev)
    cfg = LatentConfig(**st["config"]["latent"])
    E, R, P = TargetEncoder(cfg).to(dev), LatentRealizer(cfg.dz, layers=cfg.realizer_layers).to(dev), \
        PacketProbe(cfg.dz, cfg.knots, **st["config"].get("probe", {})).to(dev)
    E.load_state_dict(st["model"]["E"]); R.load_state_dict(st["model"]["R"]); P.load_state_dict(st["model"]["P"])
    for m in (E, R, P):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)    # frozen parameters; gradients still flow THROUGH P to its input z
    res = st["extra"]["result"]
    return cfg, E, R, P, res


def train_latent_flow(cfg_json: dict, out_dir: Path) -> dict:
    """Stage B: system-i flow generating z (knots x assemblies) toward the frozen encoder mean."""
    from rrp.model.latent_batch import assembly_batch
    dev = _dev()
    sig = CheckpointSignal()
    seed = cfg_json.get("seed", 0)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    lcfg, E, R, P, rep_res = load_representation(Path(cfg_json["representation"]), dev)
    data = LatentData(Path(cfg_json["packed_dir"]))
    pcfg = PolicyConfig(**dict(cfg_json["policy"], horizon=lcfg.knots, latent_dim=lcfg.dz, aux=False))
    model = FlowPolicy(pcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_json.get("lr", 3e-4), weight_decay=1e-4)
    steps = cfg_json["steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg_json.get("lr", 3e-4), total_steps=steps, pct_start=0.05)
    w_sem = cfg_json.get("packet_semantic_weight", 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    B = cfg_json.get("batch_size", 128)
    gen = torch.Generator(device=dev).manual_seed(seed)
    step, t0 = 0, time.time()
    last = out_dir / "policy_last.pt"      # full resume state (model incl. norm buffers, optimizer, scheduler, RNG)
    if last.exists():
        st = load_checkpoint(last, map_location=dev)
        if st["config"] != cfg_json:
            raise ValueError(f"{last} was written by a different config; use a fresh out_dir")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["extra"]["sched"]); step = st["step"]
        rng.setstate(st["extra"]["rng_py"]); gen.set_state(st["extra"]["gen"].cpu())
        print(f"resumed {last} at step {step}", flush=True)
    elif cfg_json.get("normalize_target", False):
        mean, std = latent_target_stats(E, data, dev, seed=seed + 17)
        model.set_target_norm(mean, std)
        (out_dir / "target_norm.json").write_text(json.dumps(dict(mean=mean.tolist(), std=std.tolist()), indent=0))
    log = open(out_dir / "train_log.jsonl", "a")

    def snapshot():
        save_checkpoint(last, model=model, optimizer=opt, step=step, versions=dict(latent=rep_res["latent_space_version"],
                        policy=pcfg.name), config=cfg_json,
                        extra=dict(sched=sched.state_dict(), rng_py=rng.getstate(), gen=gen.get_state()))

    feed = _prefetch(data, B, rng, 0, dev) if cfg_json.get("prefetch") else None   # resume: rng runs ahead by <= depth
    while step < steps and not sig.requested:
        if feed is not None:
            batch, a, v, lab, r, j = next(feed)
        else:
            sel, tgt, j = data.sample(B, rng, 0)
            batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        with torch.no_grad():
            af, am, ai = assembly_tokens(batch)
            z_target, _ = E(batch, a, v, af, am, ai)             # clean target = frozen posterior mean
        ab = assembly_batch(batch)
        smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
        S = batch.bank_tokens["scene"].shape[1]
        pl_fn = (lambda zc: probe_loss(P(zc, am, S), lab, smask)) if w_sem > 0 else None
        valid = am[:, None, :].expand(-1, lcfg.knots, -1)
        loss, logs = model.loss(ab, z_target, valid, None, generator=gen, packet_loss_fn=pl_fn, packet_weight=w_sem,
                                packet_tau_min=cfg_json.get("packet_tau_min", 0.0))
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), gn=float(gn), **logs)) + "\n")
            log.flush()
        if step % cfg_json.get("snapshot_every", 1000) == 0 or sig.requested:
            snapshot()
    res = dict(steps=step, wall_s=time.time() - t0, interrupted=sig.requested,
               latent_space_version=rep_res["latent_space_version"],
               realizer_compat_version=rep_res["realizer_compat_version"],
               eval=None if sig.requested else evaluate_generated(model, E, R, P, data, lcfg, dev))
    save_checkpoint(out_dir / ("policy.pt" if not sig.requested else "policy_interrupted.pt"), model=model, optimizer=None,
                    step=step, versions=dict(latent=rep_res["latent_space_version"], policy=pcfg.name),
                    config=cfg_json, extra=dict(result=res))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


@torch.no_grad()
def latent_target_stats(E, data, dev, n_batches: int = 20, seed: int = 0):
    """Per-dim mean/std of the frozen encoder mean over valid (knot, assembly) entries of the training data."""
    rng = random.Random(seed)
    zs = []
    for _ in range(n_batches):
        sel, tgt, j = data.sample(128, rng, 0)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        af, am, ai = assembly_tokens(batch)
        mu, _ = E(batch, a, v, af, am, ai)
        zs.append(mu[am[:, None, :].expand(-1, mu.shape[1], -1)].float())
    z = torch.cat(zs)
    return z.mean(0), z.std(0)


@torch.no_grad()
def evaluate_generated(model, E, R, P, data, lcfg, dev, n_batches=20, seed=7, nfe=8) -> dict:
    """Packet-probe semantics on (a) oracle encoder targets, (b) one-step clean estimates at tau=0.5, (c) FREE
    samples from pure noise with permissible observations only; plus shuffled-z control. Disjoint RNG stream."""
    from rrp.model.latent_batch import assembly_batch
    model.eval()
    rng = random.Random(seed)
    aggs = {k: {} for k in ("oracle", "one_step", "free", "free_shuffled")}
    zdist = []
    for _ in range(n_batches):
        sel, tgt, j = data.sample(128, rng, 0)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        af, am, ai = assembly_tokens(batch)
        zt, _ = E(batch, a, v, af, am, ai)
        ab = assembly_batch(batch)
        cache = model.prepare(ab)
        eps = torch.randn_like(zt)
        tau = torch.full((zt.shape[0],), 0.5, device=dev)
        z_tau, _ = interpolate_target(eps, model.normalize(zt), tau)
        vv = model.velocity(z_tau, tau, cache)
        z1 = model.denormalize(z_tau + 0.5 * vv)
        zf = model.sample(cache, lcfg.knots, nfe=nfe)
        smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
        S = smask.shape[1]
        for name, z in (("oracle", zt), ("one_step", z1), ("free", zf),
                        ("free_shuffled", zf[torch.randperm(zf.shape[0], device=dev)])):
            for k, (x, n) in probe_metrics(P(z, am, S), lab, smask).items():
                s_, n_ = aggs[name].get(k, (0, 0)); aggs[name][k] = (s_ + x, n_ + n)
        zdist.append(float(((zf - zt) ** 2).mean()))
    model.train()
    f = lambda d: {k: (x / n if n else None) for k, (x, n) in d.items()}
    return dict(free_vs_oracle_mse=float(np.mean(zdist)), **{k: f(v) for k, v in aggs.items()})


def fit_probes_on_frozen(rep_path: Path, packed_dir: Path, out_path: Path, steps: int = 6000, seed: int = 5,
                         metadata_only: bool = False) -> dict:
    """MEASUREMENT probe: a fresh PacketProbe trained on DETACHED z from the frozen encoder (identical procedure
    for latent_sem and latent_nosem). metadata_only=True trains the no-latent control probe."""
    dev = _dev()
    lcfg, E, R, _, rep_res = load_representation(rep_path, dev)
    data = LatentData(packed_dir)
    pk = load_checkpoint(rep_path, map_location="cpu")["config"].get("probe", {})
    P = PacketProbe(lcfg.dz, lcfg.knots, metadata_only=metadata_only, seed=seed, **pk).to(dev)
    opt = torch.optim.AdamW(P.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = random.Random(seed)
    t0 = time.time()
    for step in range(steps):
        sel, tgt, j = data.sample(128, rng, 0)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        with torch.no_grad():
            af, am, ai = assembly_tokens(batch)
            mu, _ = E(batch, a, v, af, am, ai)
        smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
        loss, _ = probe_loss(P(mu.detach(), am, smask.shape[1]), lab, smask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # held-out evaluation on a disjoint RNG stream: real z vs shuffled z
    P.eval()
    agg, sh = {}, {}
    rng2 = random.Random(seed + 1000)
    with torch.no_grad():
        for _ in range(30):
            sel, tgt, j = data.sample(128, rng2, 0)
            batch, a, v, lab, r = data.fetch(sel, tgt, dev)
            af, am, ai = assembly_tokens(batch)
            mu, _ = E(batch, a, v, af, am, ai)
            smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
            for d_, z in ((agg, mu), (sh, mu[torch.randperm(mu.shape[0], device=dev)])):
                for k, (x, n) in probe_metrics(P(z, am, smask.shape[1]), lab, smask).items():
                    s_, n_ = d_.get(k, (0, 0)); d_[k] = (s_ + x, n_ + n)
    f = lambda d: {k: (x / n if n else None) for k, (x, n) in d.items()}
    res = dict(steps=steps, wall_s=time.time() - t0, metadata_only=metadata_only, encoded_target=f(agg),
               encoded_target_shuffled=f(sh), latent_space_version=rep_res["latent_space_version"])
    torch.save(dict(state=P.state_dict(), cfg=dict(dz=lcfg.dz, knots=lcfg.knots, metadata_only=metadata_only,
                                                     seed=seed, **pk), result=res), out_path)
    out_path.with_suffix(".json").write_text(json.dumps(res, indent=1))
    return res


def sft_latent_flow(flow_ckpt: Path, target_packed_dir: Path, budget: int, *, seed: int, out_dir: Path, steps: int,
                    lr: float = 1e-4, packet_semantic_weight: float | None = None) -> dict:
    """Supervised new-body adaptation of SYSTEM I only: latent meaning (encoder, probes) and system 0 frozen.
    Episodes chosen by nested budgets over the target demo pool (fixed permutation per seed)."""
    from rrp.evaluation.adaptation import nested_budget_indices
    from rrp.model.latent_batch import assembly_batch
    dev = _dev()
    st = load_checkpoint(flow_ckpt, map_location=dev)
    cfgj = st["config"]
    lcfg, E, R, P, rep_res = load_representation(Path(cfgj["representation"]), dev)
    pcfg = PolicyConfig(**dict(cfgj["policy"], horizon=lcfg.knots, latent_dim=lcfg.dz, aux=False))
    model = FlowPolicy(pcfg).to(dev)
    model.load_state_dict(st["model"])
    data = LatentData(target_packed_dir)
    eps = sorted(set(data.ep.tolist()))
    chosen = [eps[i] for i in nested_budget_indices(len(eps), [budget], seed)[budget]]
    pool = [int(i) for i in np.nonzero(np.isin(data.ep, chosen))[0]]
    transitions = len(pool)
    w_sem = cfgj.get("packet_semantic_weight", 0.0) if packet_semantic_weight is None else packet_semantic_weight
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    B = min(128, max(8, len(pool)))
    for step in range(steps):
        sel = np.array(sorted(rng.sample(pool, B) if len(pool) >= B else [rng.choice(pool) for _ in range(B)]))
        batch, a, v, lab, r = data.fetch(sel, sel, dev)
        with torch.no_grad():
            af, am, ai = assembly_tokens(batch)
            zt, _ = E(batch, a, v, af, am, ai)
        ab = assembly_batch(batch)
        smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
        S = smask.shape[1]
        fn = (lambda zc: probe_loss(P(zc, am, S), lab, smask)) if w_sem > 0 else None
        loss, _ = model.loss(ab, zt, am[:, None, :].expand(-1, lcfg.knots, -1), None, packet_loss_fn=fn,
                             packet_weight=w_sem)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    res = dict(budget=budget, seed=seed, demo_episodes=len(chosen), demo_control_transitions=transitions,
               optimizer_updates=steps, wall_s=time.time() - t0, changed_modules="system_i_flow_only",
               frozen=["target_encoder", "packet_probes", "system0_realizer"],
               latent_space_version=rep_res["latent_space_version"], source_checkpoint=str(flow_ckpt))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_dir / "policy.pt", model=model, optimizer=None, step=steps,
                    versions=dict(st["versions"], adapted=True), config=cfgj, extra=dict(result=dict(st["extra"]["result"], sft=res)))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res
