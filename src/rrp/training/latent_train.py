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

from rrp.contracts.provenance import resolve_zero_prev_action
from rrp.models.checkpoint import save_checkpoint, load_checkpoint
from rrp.data.packed import PackedChunkDataset
from rrp.models.batch import Batch
from rrp.models.flow import FlowPolicy, PolicyConfig, interpolate_target, masked_mse
from rrp.models.latent_probes import PacketProbe, probe_loss, probe_metrics
from rrp.models.semantic_latent import LatentConfig, TargetEncoder, assembly_tokens
from rrp.controllers.latent_realizer import LatentRealizer, REALIZER_RECURRENT_STATE
from rrp.controllers.bundles import load_representation  # noqa: F401  (moved to controllers, W4)
from rrp.data.latent import LatentData  # noqa: F401  (moved to data, W4)
from rrp.contracts.workload import CheckpointSignal


def _dev():
    from rrp.contracts.workload import select_device
    return select_device(on_cap_error="raise")


_PF_DATA = None


def _pf_fetch(sel, tgt):
    # forked worker: torch intra-op thread pools inherited from the parent can deadlock after fork (seen on the host
    # once fetch() gained torch ops: goal_effect_from_batch); run single-threaded in the worker.
    torch.set_num_threads(1)
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
        from rrp.models.binding_aug import augment
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
    l_sem, logs = probe_loss(out, lab, smask, lv_min=cfg.probe_lv_min)
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
    from rrp.models.binding_aug import augment
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


def _explicit_zpa(cfg_json: dict, out_dir: Path, last_name: str, where: str) -> dict:
    """B-1: the config must state zero_prev_action for a NEW run; a resumed legacy run warns and keeps False.
    The resolved value is written back so every checkpoint config carries it explicitly."""
    from rrp.contracts.provenance import resolve_zero_prev_action
    new_run = not (Path(out_dir) / last_name).exists()
    return dict(cfg_json, zero_prev_action=resolve_zero_prev_action(cfg_json, where=where, new_run=new_run))


def train_representation(cfg_json: dict, out_dir: Path) -> dict:
    cfg_json = _explicit_zpa(cfg_json, out_dir, "rep_last.pt", f"train_representation({out_dir})")
    dev = _dev()
    sig = CheckpointSignal()
    cfg = LatentConfig(**cfg_json["latent"])
    seed = cfg_json.get("seed", 0)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = LatentData(Path(cfg_json["packed_dir"]), zero_prev_action=cfg_json.get("zero_prev_action", False),
                      anchor=cfg_json.get("realizer_anchor", False))
    E, R = TargetEncoder(cfg).to(dev), LatentRealizer(cfg.dz, layers=cfg.realizer_layers).to(dev)
    R.anchor = cfg_json.get("realizer_anchor", False)
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
        exact = _restore_rng(st["extra"], rng)      # checkpoints written before 2026-09-27 carry no RNG state
        print(f"resumed {last} at step {step} ({'exact: RNG restored' if exact else 'INEXACT: no RNG state in checkpoint'})",
              flush=True)
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
                            config=cfg_json, extra=dict(sched=sched.state_dict(), **_rng_state(rng)))
    res = dict(steps=step, wall_s=time.time() - t0, interrupted=sig.requested, latent_space_version=cfg.version(),
               realizer_compat_version=f"rz-{cfg.version()}-{REALIZER_RECURRENT_STATE}",
               eval=evaluate_representation(E, R, P, data, cfg, dev) if not sig.requested else None)
    name = "representation.pt" if not sig.requested else "representation_interrupted.pt"
    save_checkpoint(out_dir / name, model=_bundle(E, R, P), optimizer=None, step=step,
                    versions=dict(latent=cfg.version()), config=cfg_json, extra=dict(result=res))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _rng_state(rng: random.Random) -> dict:
    """Resume state of every RNG the representation loop draws from (batch sampling: python rng; posterior noise and
    dropout: torch CPU/CUDA generators), so a resumed Stage A continues the uninterrupted run exactly."""
    st = dict(rng_py=rng.getstate(), torch_rng=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda_rng"] = torch.cuda.get_rng_state_all()
    return st


def _restore_rng(extra: dict, rng: random.Random) -> bool:
    if "rng_py" not in extra:
        return False
    rng.setstate(extra["rng_py"])
    torch.set_rng_state(extra["torch_rng"].cpu())
    if "cuda_rng" in extra and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in extra["cuda_rng"]])
    return True


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


def train_latent_flow(cfg_json: dict, out_dir: Path) -> dict:
    """Stage B: system-i flow generating z (knots x assemblies) toward the frozen encoder mean."""
    from rrp.models.latent_batch import assembly_batch
    cfg_json = _explicit_zpa(cfg_json, out_dir, "policy_last.pt", f"train_latent_flow({out_dir})")
    dev = _dev()
    sig = CheckpointSignal()
    seed = cfg_json.get("seed", 0)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    lcfg, E, R, P, rep_res = load_representation(Path(cfg_json["representation"]), dev)
    pack_free = cfg_json.get("gen_dagger_frac", 0.0) >= 1.0 and bool(cfg_json.get("init_from"))
    data = None if pack_free else LatentData(Path(cfg_json["packed_dir"]), zero_prev_action=cfg_json.get("zero_prev_action", False))
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

    # generator DAgger (ladder sprint): (public context at learner-visited states, z* = E(stateless expert chunk))
    gd_items = []
    for gp in cfg_json.get("gen_dagger") or []:
        import pickle
        with open(gp, "rb") as fh:
            gd_items += pickle.load(fh)["items"]
    Bg = int(round(B * cfg_json.get("gen_dagger_frac", 0.5))) if gd_items else 0
    grng = random.Random(seed + 23)
    Bp = B - Bg
    feed = _prefetch(data, Bp, rng, 0, dev) if (cfg_json.get("prefetch") and Bp) else None   # resume: rng runs ahead by <= depth
    while step < steps and not sig.requested:
        if Bp == 0:                 # pack-free generator DAgger (gen_dagger_frac 1.0, warm start): DAgger rows only
            loss, logs = torch.zeros((), device=dev), {}
        elif feed is not None:
            batch, a, v, lab, r, j = next(feed)
        else:
            sel, tgt, j = data.sample(Bp, rng, 0)
            batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        if Bp:
            with torch.no_grad():
                af, am, ai = assembly_tokens(batch)
                z_target, _ = E(batch, a, v, af, am, ai)             # clean target = frozen posterior mean
            ab = assembly_batch(batch)
            smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
            S = batch.bank_tokens["scene"].shape[1]
            pl_fn = (lambda zc: probe_loss(P(zc, am, S), lab, smask, lv_min=lcfg.probe_lv_min)) if w_sem > 0 else None
            valid = am[:, None, :].expand(-1, lcfg.knots, -1)
            loss, logs = model.loss(ab, z_target, valid, None, generator=gen, packet_loss_fn=pl_fn, packet_weight=w_sem,
                                    packet_tau_min=cfg_json.get("packet_tau_min", 0.0))
        if Bg:
            from rrp.models.batch import collate_inputs
            it = [gd_items[grng.randrange(len(gd_items))] for _ in range(Bg)]
            bd = collate_inputs([x[0] for x in it]).to(dev)
            afd, amd, aid = assembly_tokens(bd)
            zt = torch.zeros(Bg, lcfg.knots, amd.shape[1], lcfg.dz, device=dev)
            for i_, x in enumerate(it):
                m_ = x[1].shape[1]
                zt[i_, :, :m_] = torch.from_numpy(x[1]).to(dev)
            ld, _ = model.loss(assembly_batch(bd), zt, amd[:, None, :].expand(-1, lcfg.knots, -1), None, generator=gen)
            logs = dict(logs, gen_dagger=float(ld.detach()))
            loss = (Bp * loss + Bg * ld) / B
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
               eval=None if (sig.requested or data is None) else evaluate_generated(model, E, R, P, data, lcfg, dev))
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
    from rrp.models.latent_batch import assembly_batch
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
    rep_cfg = load_checkpoint(rep_path, map_location="cpu")["config"]
    data = LatentData(packed_dir, zero_prev_action=resolve_zero_prev_action(      # same inputs as E saw (B-1)
        rep_cfg, where=f"fit_probes_on_frozen: representation {rep_path}", new_run=False))
    pk = rep_cfg.get("probe", {})
    P = PacketProbe(lcfg.dz, lcfg.knots, metadata_only=metadata_only, seed=seed, **pk).to(dev)
    opt = torch.optim.AdamW(P.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = random.Random(seed)
    t0 = time.time()
    gcf = torch.Generator().manual_seed(seed)
    for step in range(steps):
        sel, tgt, j = data.sample(128, rng, 0)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        if binding_cf > 0:        # probe also sees counterfactual-binding packets (labels follow the binding)
            from rrp.models.binding_aug import augment
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
    from rrp.models.latent_batch import assembly_batch
    dev = _dev()
    st = load_checkpoint(flow_ckpt, map_location=dev)
    cfgj = st["config"]
    lcfg, E, R, P, rep_res = load_representation(Path(cfgj["representation"]), dev)
    pcfg = PolicyConfig(**dict(cfgj["policy"], horizon=lcfg.knots, latent_dim=lcfg.dz, aux=False))
    model = FlowPolicy(pcfg).to(dev)
    model.load_state_dict(st["model"])
    data = LatentData(target_packed_dir, zero_prev_action=resolve_zero_prev_action(   # as the source flow (B-1)
        cfgj, where=f"sft_latent_flow: source flow {flow_ckpt}", new_run=False))
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
        fn = (lambda zc: probe_loss(P(zc, am, S), lab, smask, lv_min=lcfg.probe_lv_min)) if w_sem > 0 else None
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
    cfg_json = _explicit_zpa(cfg_json, out_dir, "rz_last.pt", f"refit_realizer({out_dir})")
    dev = _dev()
    sig = CheckpointSignal()
    seed = cfg_json.get("seed", 0)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rep_path = Path(cfg_json["representation"])
    st0 = load_checkpoint(rep_path, map_location=dev)
    lcfg, E, R_old, P, rep_res = load_representation(rep_path, dev)
    data = LatentData(Path(cfg_json["packed_dir"]), zero_prev_action=cfg_json.get("zero_prev_action", False),
                      anchor=cfg_json.get("realizer_anchor", False), drop_qd=cfg_json.get("realizer_drop_qd", False))
    from rrp.controllers.latent_realizer import make_realizer
    arch = dict(st0["config"].get("realizer_arch") or {})
    for k_ in ("layers", "width", "z_norm"):                  # capacity / input-normalization overrides (ladder sprint)
        if f"realizer_{k_}" in cfg_json:
            arch[k_] = cfg_json[f"realizer_{k_}"]
    R = make_realizer(lcfg.dz, lcfg.realizer_layers, arch).to(dev)
    R.anchor = cfg_json.get("realizer_anchor", False)
    R.drop_qd = cfg_json.get("realizer_drop_qd", False)
    if cfg_json.get("init", "fresh") == "old":
        R.load_state_dict(R_old.state_dict())
    for p_ in R.parameters():
        p_.requires_grad_(True)
    if getattr(R, "z_norm", False) and cfg_json.get("init", "fresh") != "old":
        zs = []
        with torch.no_grad():
            for _ in range(cfg_json.get("z_norm_batches", 40)):
                sel_, tgt_, j_ = data.sample(64, rng, lcfg.max_phase_ticks)
                b_, a_, v_, lab_, r_ = data.fetch(sel_, tgt_, dev)
                af_, am_, ai_ = assembly_tokens(b_)
                mu_, _ = E(b_, a_, v_, af_, am_, ai_)
                zs.append(mu_[am_[:, None, :].expand(-1, mu_.shape[1], -1)].float())
        zc = torch.cat(zs)
        R.z_mean.copy_(zc.mean(0)); R.z_std.copy_(zc.std(0).clamp(min=1e-3))
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
    dag = _load_dagger(cfg_json.get("dagger") or [], dev, anchor=R.anchor, drop_qd=R.drop_qd)
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
            zn = cfg_json.get("z_noise_rel", 0.0)
            if zn > 0:              # robustness to generator error: isotropic noise with relative norm ~ U(0, zn) per sample
                rel = torch.rand(z.shape[0], 1, 1, 1, device=dev) * zn
                nz = torch.randn_like(z)
                z = z + nz * rel * z.flatten(1).norm(dim=1)[:, None, None, None] / nz.flatten(1).norm(dim=1)[:, None, None, None]
        phase = torch.as_tensor(j * lcfg.control_dt, dtype=z.dtype, device=dev)
        qdp = cfg_json.get("realizer_qd_dropout", 0.0)
        if qdp > 0:                 # per-sample dropout of the joint-velocity input (eval keeps full qd): packet reliance
            from rrp.controllers.latent_realizer import QD_COL
            keep = (torch.rand(r["node"].shape[0], 1, device=dev) >= qdp).to(r["node"].dtype)
            r["node"] = r["node"].clone(); r["node"][:, :, QD_COL] = r["node"][:, :, QD_COL] * keep
        pred = R(z, am, kt, phase, r["node"], r["node_mask"], r["local"], node_asm=r.get("node_asm"))
        m = (r["v1"] & r["node_mask"]).float()
        w0 = cfg_json.get("j0_weight", 1.0)
        if w0 != 1.0:                                  # extra weight on the first tick of a packet (phase j = 0)
            m = m * (1.0 + (w0 - 1.0) * (j == 0).float()[:, None])
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
            nd_ = dag["node"][idx].float()
            if cfg_json.get("realizer_qd_dropout", 0.0) > 0:
                from rrp.controllers.latent_realizer import QD_COL
                keep = (torch.rand(Bd, 1, device=dev) >= cfg_json["realizer_qd_dropout"]).float()
                nd_[:, :, QD_COL] = nd_[:, :, QD_COL] * keep
            pd = R(zd, amd, kt, dag["j"][idx].float() * lcfg.control_dt, nd_, nmask,
                   dag["local"][idx].float())
            md = nmask.float()
            if cfg_json.get("j0_weight", 1.0) != 1.0:
                md = md * (1.0 + (cfg_json["j0_weight"] - 1.0) * (dag["j"][idx] == 0).float()[:, None])
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
                    config=dict(st0["config"], refit=cfg_json, realizer_anchor=R.anchor, realizer_drop_qd=R.drop_qd,
                                realizer_arch=dict(layers=len(R.blocks), width=R.D, z_norm=bool(getattr(R, "z_norm", False))),
                                zero_prev_action=cfg_json.get("zero_prev_action", False)), extra=dict(result=res))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _load_dagger(paths, dev, anchor: bool = False, drop_qd: bool = False):
    """System-0 DAgger buffers written by rrp.evaluation.ladder.save_dagger (single-assembly bodies; M = 1).
    Buffers store the BASE node features (col 28 = 0). anchor: recompute col 28 = q(t+j) - q(packet state) from the
    j = 0 row of the same packet (as LatentData does for the pack; rows without a j = 0 row are dropped).
    drop_qd: zero the joint-velocity column (27) like LatentData(drop_qd)."""
    if not paths:
        return None
    from rrp.controllers.latent_realizer import Q_COL, ANCHOR_COL, QD_COL
    parts = []
    for p in paths:
        z_ = np.load(p)
        q = {k_: z_[k_] for k_ in ("mu", "lv", "rp", "j", "node", "n_nodes", "local", "a1")}
        if anchor or drop_qd:
            node = q["node"].astype(np.float32)
            keep = np.ones(len(node), bool)
            if anchor:
                j0 = {}
                for i, (rp_, j_) in enumerate(zip(q["rp"], q["j"])):
                    if j_ == 0:
                        j0[int(rp_)] = i
                idx0 = np.array([j0.get(int(r_), -1) for r_ in q["rp"]])
                keep = idx0 >= 0
                node[keep, :, ANCHOR_COL] = node[keep, :, Q_COL] - node[idx0[keep], :, Q_COL]
            if drop_qd:
                node[:, :, QD_COL] = 0
            q["node"] = node.astype(q["node"].dtype)
            for k_ in ("rp", "j", "node", "n_nodes", "local", "a1"):
                q[k_] = q[k_][keep]
        parts.append(q)
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
