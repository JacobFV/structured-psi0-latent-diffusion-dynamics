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

    def __init__(self, packed_dir: Path, zero_prev_action: bool = False):
        self.ds = PackedChunkDataset(packed_dir, zero_prev_action=zero_prev_action)
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
        if self.ds.zero_prev_action:                    # bug B-1 fix (see rrp.learning.packed.PREV_ACTION_COL)
            from rrp.learning.packed import PREV_ACTION_COL
            nodes[..., PREV_ACTION_COL] = 0
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
        from rrp.model.binding_aug import goal_effect_from_batch
        lab["goal_effect"] = goal_effect_from_batch(batch)          # task-goal label (public spec + estimates)
        mv = lambda x: x.to(dev, non_blocking=True)
        return batch.to(dev), mv(a[..., 0]), mv(v), {k: mv(x) for k, x in lab.items()}, {k: mv(x) for k, x in r.items()}


_PF_DATA = None


def _pf_fetch(sel, tgt):
    return _PF_DATA.fetch(sel, tgt, "cpu")


def _prefetch(data, B, rng, max_j, dev, depth: int = 6, workers: int = 3):
    """Collate batches on CPU ahead of the GPU step in forked worker processes (the packed arrays are memory-mapped,
    so workers share the page cache). Batch order = the serial rng sequence (sampling happens here, in order)."""
    import collections
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    global _PF_DATA
    _PF_DATA = data
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"))
    pend = collections.deque()
    mv = lambda x: x.to(dev, non_blocking=True)
    try:
        while True:
            while len(pend) < depth:
                sel, tgt, j = data.sample(B, rng, max_j)
                pend.append((ex.submit(_pf_fetch, sel, tgt), j))
            fut, j = pend.popleft()
            batch, a, v, lab, r = fut.result()
            yield batch.to(dev), mv(a), mv(v), {k: mv(x) for k, x in lab.items()}, {k: mv(x) for k, x in r.items()}, j
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def representation_step(E, R, P, data_b, cfg: LatentConfig, train=True):
    """With cfg.binding_cf > 0 (training only) a fraction of the batch is appended as counterfactual-binding copies
    (model/binding_aug.py): same trajectory, body and scene tokens, task roles rebound to another slot, `focus`
    labels follow the binding. Realizer loss uses FACTUAL rows only; probe and KL terms use all rows."""
    batch, a, v, lab, r, j = data_b
    nf, cf = batch.B, None
    if train and cfg.binding_cf > 0:
        from rrp.model.binding_aug import augment
        batch, a, v, lab, nf, cf = augment(batch, a, v, lab, cfg.binding_cf)
    af, am, ai = assembly_tokens(batch)
    mu, logvar = E(batch, a, v, af, am, ai)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if train else mu
    kt = torch.tensor(cfg.knot_times, dtype=z.dtype, device=z.device)
    phase = torch.as_tensor(j * cfg.control_dt, dtype=z.dtype, device=z.device)
    pred = R(z[:nf], am[:nf], kt, phase, r["node"], r["node_mask"], r["local"], node_asm=r.get("node_asm"))
    m = (r["v1"] & r["node_mask"]).float()
    l_real = ((pred - r["a1"]) ** 2 * m).sum() / m.sum().clamp(min=1)
    mm = am[:, None, :, None].float().expand_as(mu)
    kl = (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar) * mm).sum() / mm.sum().clamp(min=1)
    smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
    out = P(z, am, batch.bank_tokens["scene"].shape[1])
    l_sem, logs = probe_loss(out, lab, smask)
    loss = l_real + cfg.semantic_weight * l_sem + cfg.beta_kl * kl
    if cf is not None:
        d = (mu[:nf][cf["pick"]] - mu[nf:]).flatten(1).norm(dim=1) / mu[:nf][cf["pick"]].flatten(1).norm(dim=1).clamp(min=1e-3)
        logs.update(cf_rel_dist=float(d.mean().detach()), n_cf=int(len(cf["pick"])))
        if cfg.binding_contrast > 0:
            l_con = F.relu(0.25 - d).mean()
            loss = loss + cfg.binding_contrast * l_con
            logs.update(contrast=float(l_con.detach()))
    logs.update(real=float(l_real.detach()), kl=float(kl.detach()), sem=float(l_sem.detach()))
    return loss, logs, (z[:nf], am[:nf], {k: x[:nf] for k, x in out.items()}, {k: x[:nf] for k, x in lab.items()},
                        smask[:nf])


@torch.no_grad()
def binding_cf_metrics(E, P, batch, a, v, lab, gen=None) -> dict:
    """Counterfactual-binding response of the encoded packet (raw sums): relative z change, probe metrics on the
    counterfactual rows against binding-following labels, and `focus_follows` = the probe's focus flips from the
    rebound-away slot to the newly bound slot (among swaps that change the focus label)."""
    from rrp.model.binding_aug import augment
    b2, a2, v2, l2, nf, info = augment(batch, a, v, lab, 1.0, gen)
    if info is None:
        return {}
    af, am, ai = assembly_tokens(b2)
    mu, _ = E(b2, a2, v2, af, am, ai)
    S = b2.bank_tokens["scene"].shape[1]
    out = P(mu[nf:], am[nf:], S)
    lc = {k: x[nf:] for k, x in l2.items()}
    smask = b2.bank_mask["scene"][nf:] & lc["slot_valid"].bool()
    res = {f"cf_{k}": x for k, x in probe_metrics(out, lc, smask).items()}
    zf = mu[:nf][info["pick"]]
    rd = (zf - mu[nf:]).flatten(1).norm(dim=1) / zf.flatten(1).norm(dim=1).clamp(min=1e-3)
    res["cf_rel_z_dist"] = (float(rd.sum()), int(len(rd)))
    fo = out["focused_on"][..., 0] > 0
    b = torch.arange(len(info["pick"]), device=fo.device)
    lf = lab["focus"][info["pick"]].bool()
    chg = lf[b, info["src"]] != lf[b, info["dst"]]
    follows = (fo[b, info["src"]] == lc["focus"].bool()[b, info["src"]]) & (fo[b, info["dst"]] == lc["focus"].bool()[b, info["dst"]])
    res["cf_focus_follows"] = (int((follows & chg).sum()), int(chg.sum()))
    return res


def train_representation(cfg_json: dict, out_dir: Path) -> dict:
    dev = _dev()
    sig = CheckpointSignal()
    cfg = LatentConfig(**cfg_json["latent"])
    seed = cfg_json.get("seed", 0)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = LatentData(Path(cfg_json["packed_dir"]), zero_prev_action=cfg_json.get("zero_prev_action", False))
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
    agg, agg_sh, agg_cf, real, hold = {}, {}, {}, [], []
    gcf = torch.Generator().manual_seed(seed)
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
        for k, (x, n) in binding_cf_metrics(E, P, batch, a, v, lab, gcf).items():
            s_, n_ = agg_cf.get(k, (0, 0)); agg_cf[k] = (s_ + x, n_ + n)
    E.train(); R.train(); P.train()
    f = lambda d: {k: (x / n if n else None) for k, (x, n) in d.items()}
    return dict(realize_mse=float(np.mean(real)), zero_action_mse=float(np.mean(hold)), probes=f(agg),
                probes_shuffled_z=f(agg_sh), binding_counterfactual=f(agg_cf))


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
    from rrp.control.latent_realizer import bundle_versions
    lsv, rcv = bundle_versions(cfg.version(), st["model"]["E"], st["model"]["R"])
    R.bundle_versions = (lsv, rcv)
    res = dict(st["extra"]["result"], config_latent_space_version=st["extra"]["result"]["latent_space_version"],
               latent_space_version=lsv, realizer_compat_version=rcv)
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
    data = LatentData(Path(cfg_json["packed_dir"]), zero_prev_action=cfg_json.get("zero_prev_action", False))
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
    elif cfg_json.get("init_from"):            # warm start (e.g. bug B-1 fine-tune): weights incl. target-norm buffers
        model.load_state_dict(load_checkpoint(Path(cfg_json["init_from"]), map_location=dev)["model"])
        print(f"initialized from {cfg_json['init_from']}", flush=True)
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
                         metadata_only: bool = False, binding_cf: float = 0.0) -> dict:
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
    gcf = torch.Generator().manual_seed(seed)
    for step in range(steps):
        sel, tgt, j = data.sample(128, rng, 0)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        if binding_cf > 0:        # probe also sees counterfactual-binding packets (labels follow the binding)
            from rrp.model.binding_aug import augment
            batch, a, v, lab, _, _ = augment(batch, a, v, lab, binding_cf, gcf)
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
    agg, sh, cfm = {}, {}, {}
    rng2 = random.Random(seed + 1000)
    gcf2 = torch.Generator().manual_seed(seed + 1000)
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
            for k, (x, n) in binding_cf_metrics(E, P, batch, a, v, lab, gcf2).items():
                s_, n_ = cfm.get(k, (0, 0)); cfm[k] = (s_ + x, n_ + n)
    f = lambda d: {k: (x / n if n else None) for k, (x, n) in d.items()}
    res = dict(steps=steps, wall_s=time.time() - t0, metadata_only=metadata_only, binding_cf=binding_cf,
               encoded_target=f(agg), encoded_target_shuffled=f(sh), binding_counterfactual=f(cfm),
               latent_space_version=rep_res["latent_space_version"])
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


def refit_realizer(cfg_json: dict, out_dir: Path) -> dict:
    """Re-train ONLY system 0 (the realizer R) on a FROZEN Stage-A encoder E (same latent space, same probes), with the
    same objective as Stage A (L_real with reparameterized z ~ q(z|context, demonstrated chunk), phases j <= max).
    Motivated by bug B-1 (config "zero_prev_action": true) and by closed-loop robustness variants. The latent space
    version is unchanged (E identical), so flows trained on it stay valid; the realizer compat version changes.
    cfg: representation, packed_dir, steps, batch_size, lr, seed, name, zero_prev_action, init ("fresh"|"old")."""
    dev = _dev()
    sig = CheckpointSignal()
    seed = cfg_json.get("seed", 0)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rep_path = Path(cfg_json["representation"])
    st0 = load_checkpoint(rep_path, map_location=dev)
    lcfg, E, R_old, P, rep_res = load_representation(rep_path, dev)
    data = LatentData(Path(cfg_json["packed_dir"]), zero_prev_action=cfg_json.get("zero_prev_action", False))
    R = LatentRealizer(lcfg.dz, layers=lcfg.realizer_layers).to(dev)
    if cfg_json.get("init", "fresh") == "old":
        R.load_state_dict(R_old.state_dict())
    for p_ in R.parameters():
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(R.parameters(), lr=cfg_json.get("lr", 3e-4), weight_decay=1e-4)
    steps = cfg_json["steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg_json.get("lr", 3e-4), total_steps=steps, pct_start=0.05)
    out_dir.mkdir(parents=True, exist_ok=True)
    last = out_dir / "rz_last.pt"
    step = 0
    if last.exists():
        st = load_checkpoint(last, map_location=dev)
        R.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["extra"]["sched"])
        step = st["step"]; rng.setstate(st["extra"]["rng_py"])
    log = open(out_dir / "train_log.jsonl", "a")
    B = cfg_json.get("batch_size", 128)
    t0 = time.time()
    kt = torch.tensor(lcfg.knot_times, device=dev)
    dag = _load_dagger(cfg_json.get("dagger") or [], dev)
    Bd = int(round(B * cfg_json.get("dagger_frac", 0.5))) if dag else 0
    drng = np.random.default_rng(seed + 11)
    nw = cfg_json.get("prefetch_workers", 3)

    def _serial():
        while True:
            sel, tgt, j = data.sample(B - Bd, rng, lcfg.max_phase_ticks)
            yield (*data.fetch(sel, tgt, dev), j)
    feed = _prefetch(data, B - Bd, rng, lcfg.max_phase_ticks, dev, workers=nw) if nw > 0 else _serial()
    while step < steps and not sig.requested:
        batch, a, v, lab, r, j = next(feed)
        j = torch.as_tensor(j, device=dev)
        with torch.no_grad():
            af, am, ai = assembly_tokens(batch)
            mu, logvar = E(batch, a, v, af, am, ai)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        phase = torch.as_tensor(j * lcfg.control_dt, dtype=z.dtype, device=dev)
        pred = R(z, am, kt, phase, r["node"], r["node_mask"], r["local"], node_asm=r.get("node_asm"))
        m = (r["v1"] & r["node_mask"]).float()
        se, cnt = ((pred - r["a1"]) ** 2 * m).sum(), m.sum()
        l_pack = float((se / cnt.clamp(min=1)).detach())
        l_dag = None
        if Bd:
            idx = torch.from_numpy(drng.integers(0, dag["n"], Bd)).to(dev)
            rp = dag["rp"][idx]
            mu_d, lv_d = dag["mu"][rp].float(), dag["lv"][rp].float()
            zd = mu_d + torch.randn_like(mu_d) * (0.5 * lv_d).exp()
            Md = zd.shape[2]
            amd = torch.ones(Bd, Md, dtype=torch.bool, device=dev)
            nmask = torch.arange(dag["node"].shape[1], device=dev)[None] < dag["n_nodes"][idx][:, None]
            pd = R(zd, amd, kt, dag["j"][idx].float() * lcfg.control_dt, dag["node"][idx].float(), nmask,
                   dag["local"][idx].float())
            md = nmask.float()
            sed = ((pd - dag["a1"][idx]) ** 2 * md).sum()
            l_dag = float((sed / md.sum().clamp(min=1)).detach())
            se, cnt = se + sed, cnt + md.sum()
        loss = se / cnt.clamp(min=1)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, real=float(loss.detach()), pack=l_pack, dagger=l_dag,
                                      gn=float(gn))) + "\n")
            log.flush()
        if step % 2000 == 0 or sig.requested:
            save_checkpoint(last, model=R, optimizer=opt, step=step, versions=dict(latent=rep_res["latent_space_version"]),
                            config=cfg_json, extra=dict(sched=sched.state_dict(), rng_py=rng.getstate()))
    name = cfg_json.get("name", out_dir.name)
    res = dict(rep_res, steps_refit=step, wall_s=time.time() - t0, interrupted=sig.requested,
               realizer_compat_version=__import__("rrp.control.latent_realizer", fromlist=["x"]).bundle_versions(
                   lcfg.version(), E.state_dict(), R.state_dict())[1], refit_name=name,
               refit_of=str(rep_path), zero_prev_action=cfg_json.get("zero_prev_action", False),
               eval=None if sig.requested else evaluate_representation(E, R, P, data, lcfg, dev))
    E.eval(); R.eval(); P.eval()
    save_checkpoint(out_dir / ("representation.pt" if not sig.requested else "representation_interrupted.pt"),
                    model=_bundle(E, R, P), optimizer=None, step=step, versions=dict(latent=rep_res["latent_space_version"]),
                    config=dict(st0["config"], refit=cfg_json), extra=dict(result=res))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _load_dagger(paths, dev):
    """System-0 DAgger buffers written by rrp.evaluation.ladder.save_dagger (single-assembly bodies; M = 1)."""
    if not paths:
        return None
    parts = [np.load(p) for p in paths]
    off, rp = 0, []
    for q in parts:
        rp.append(q["rp"].astype(np.int64) + off)
        off += len(q["mu"])
    Mmax = max(q["mu"].shape[2] for q in parts)
    if any(q["mu"].shape[2] != Mmax for q in parts):
        raise ValueError("mixed assembly counts in DAgger buffers")
    t = lambda k, dt=None: torch.from_numpy(np.concatenate([q[k] for q in parts]).astype(dt) if dt else
                                            np.concatenate([q[k] for q in parts])).to(dev)
    d = dict(mu=t("mu"), lv=t("lv"), rp=torch.from_numpy(np.concatenate(rp)).to(dev), j=t("j", np.int64),
             node=t("node"), n_nodes=t("n_nodes", np.int64), local=t("local"), a1=t("a1", np.float32))
    d["n"] = len(d["rp"])
    return d
