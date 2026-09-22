"""Behavioral (flow-matching) training of the policy, and codec training.

Resource-aware: GPU memory is capped via rrp.ops.gpu.apply_cap; SIGUSR1/SIGTERM trigger a
checkpoint and clean exit (lease checkpoint request). All metrics go to a JSONL log.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from rrp.learning.checkpoint import save_checkpoint, load_checkpoint
from rrp.learning.data import ChunkDataset, load_episodes
from rrp.model.codec import ActionCodec, CodecConfig
from rrp.model.flow import FlowPolicy, PolicyConfig
from rrp.ops.jobs import CheckpointSignal

FEAT_VERSION = "feat-v1"


def device_setup():
    if torch.cuda.is_available():
        from rrp.ops.gpu import apply_cap
        info = apply_cap()
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda"), info
    return torch.device("cpu"), {"cuda": False}


def encode_targets(codec: ActionCodec | None, batch, a, v):
    """Clean flow target: codec latent (frozen codec) or direct normalized action."""
    if codec is None:
        return a
    with torch.no_grad():
        z, mu, _ = codec.encode(a[..., 0], batch.node_feats, batch.node_mask)
    return mu


def train_codec(cfg: dict, out_dir: Path) -> dict:
    dev, ginfo = device_setup()
    sig = CheckpointSignal()
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    eps = load_episodes(Path(cfg["dataset"]), robots=set(cfg["train_robots"]))
    ds = ChunkDataset(eps, cfg["horizon"], stride=cfg.get("stride", 2))
    hold = load_episodes(Path(cfg["dataset"]), robots=set(cfg.get("heldout_robots", [])), limit_per_robot=20) \
        if cfg.get("heldout_robots") else []
    dsh = ChunkDataset(hold, cfg["horizon"], stride=4) if hold else None
    ccfg = CodecConfig(**cfg["codec"])
    model = ActionCodec(ccfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=1e-4)
    log = open(out_dir / "train_log.jsonl", "a")
    step, t0 = 0, time.time()
    for epoch in range(cfg["epochs"]):
        for batch, a, v, lab, eff in ds.batches(cfg["batch_size"], rng):
            batch, a, v, eff = batch.to(dev), a.to(dev), v.to(dev), eff.to(dev)
            loss, logs = model.loss(a[..., 0], v, batch.node_feats, batch.node_mask, eff)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                log.write(json.dumps(dict(step=step, epoch=epoch, t=time.time() - t0, **logs)) + "\n")
                log.flush()
            if sig.requested:
                break
        if sig.requested:
            break
    res = dict(steps=step, wall_s=time.time() - t0, train_chunks=len(ds))
    if dsh:
        res["heldout_recon"] = evaluate_codec(model, dsh, dev)
    res["train_recon"] = evaluate_codec(model, ds, dev, max_batches=20)
    meta = save_checkpoint(out_dir / "codec.pt", model=model, optimizer=opt, step=step,
                           versions=dict(codec=ccfg.version, featurizer=FEAT_VERSION), config=cfg,
                           extra=dict(result=res, interrupted=sig.requested))
    res["checkpoint"] = meta
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


@torch.no_grad()
def evaluate_codec(model, ds, dev, max_batches=50) -> dict:
    """Held-out reconstruction + shuffled-latent control (latents must carry information)."""
    rng = random.Random(0)
    rec, shuf, zero, n = 0.0, 0.0, 0.0, 0
    for i, (batch, a, v, lab, eff) in enumerate(ds.batches(128, rng, shuffle=True, drop_last=False)):
        if i >= max_batches:
            break
        batch, a, v = batch.to(dev), a.to(dev), v.to(dev)
        z, mu, _ = model.encode(a[..., 0], batch.node_feats, batch.node_mask)
        m = (v & batch.node_mask[:, None, :]).float()
        err = lambda ah: float(((ah - a[..., 0]) ** 2 * m).sum() / m.sum())
        rec += err(model.decode(mu, batch.node_feats, batch.node_mask)[0])
        perm = torch.randperm(mu.shape[0], device=mu.device)
        shuf += err(model.decode(mu[perm], batch.node_feats, batch.node_mask)[0])
        zero += err(torch.zeros_like(a[..., 0]))
        n += 1
    return dict(mse=rec / max(n, 1), shuffled_latent_mse=shuf / max(n, 1), zero_action_mse=zero / max(n, 1))


def train_policy(cfg: dict, out_dir: Path) -> dict:
    dev, ginfo = device_setup()
    sig = CheckpointSignal()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    eps = load_episodes(Path(cfg["dataset"]), robots=set(cfg["train_robots"]),
                        limit_per_robot=cfg.get("episodes_per_robot"))
    ds = ChunkDataset(eps, cfg["horizon"], stride=cfg.get("stride", 1))
    codec = None
    if cfg.get("codec_checkpoint"):
        st = load_checkpoint(Path(cfg["codec_checkpoint"]), requested_versions=dict(featurizer=FEAT_VERSION))
        codec = ActionCodec(CodecConfig(**st["config"]["codec"])).to(dev)
        codec.load_state_dict(st["model"])
        codec.eval()
        for p in codec.parameters():
            p.requires_grad_(False)       # frozen codec for the policy experiment
    pcfg = PolicyConfig(**cfg["policy"])
    model = FlowPolicy(pcfg).to(dev)
    swap = cfg.get("swap_alignment")
    proj, pairs = None, []
    params = list(model.parameters())
    if swap:
        from rrp.learning.swap_alignment import SwapProjector, build_pairs, swap_alignment_loss
        pairs = build_pairs(eps, cfg["horizon"])
        proj = SwapProjector(pcfg.D, swap.get("rank", 16)).to(dev)
        params += list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("wd", 1e-4))
    total_steps = cfg["epochs"] * max(1, len(ds) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 3e-4), total_steps=max(total_steps, 10),
                                                pct_start=0.05)
    log = open(out_dir / "train_log.jsonl", "a")
    step, t0 = 0, time.time()
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    start_epoch = 0
    last = out_dir / "policy_last.pt"
    if last.exists():      # resume: model/optimizer/scheduler/epoch cursor
        st = load_checkpoint(last, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["extra"]["sched"])
        step, start_epoch = st["step"], st["data_cursor"]["epoch"] + 1
        rng = random.Random(cfg["seed"] + start_epoch)
    every = cfg.get("checkpoint_every_epochs", 1)
    for epoch in range(start_epoch, cfg["epochs"]):
        for batch, a, v, lab, eff in ds.batches(cfg["batch_size"], rng):
            batch, a, v = batch.to(dev), a.to(dev), v.to(dev)
            lab = {k: t.to(dev) for k, t in lab.items()}
            target = encode_targets(codec, batch, a, v)
            loss, logs = model.loss(batch, target, v, lab if pcfg.aux else None, aux_weight=cfg.get("aux_weight", 0.1),
                                    generator=gen)
            if proj is not None and len(pairs) >= 8:
                sl = swap_alignment_loss(model, proj, rng.sample(pairs, min(swap.get("pairs_per_step", 64), len(pairs))),
                                         dev)
                loss = loss + swap.get("weight", 0.1) * sl
                logs["swap_align"] = float(sl.detach())
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total_steps - 1:
                sched.step()
            step += 1
            if step % 100 == 0:
                log.write(json.dumps(dict(step=step, epoch=epoch, t=time.time() - t0, loss=float(loss.detach()),
                                          grad_norm=float(gn), lr=sched.get_last_lr()[0], **logs)) + "\n")
                log.flush()
            if sig.requested:
                break
        if sig.requested:
            break
        if (epoch + 1) % every == 0 and epoch + 1 < cfg["epochs"]:
            save_checkpoint(last, model=model, optimizer=opt, step=step,
                            versions=dict(policy=pcfg.name, featurizer=FEAT_VERSION,
                                          codec=(codec.cfg.version if codec else None)),
                            config=cfg, data_cursor=dict(epoch=epoch), extra=dict(sched=sched.state_dict()))
    res = dict(steps=step, wall_s=time.time() - t0, train_chunks=len(ds), swap_pairs=len(pairs),
               n_params=sum(p.numel() for p in model.parameters()),
               gpu=ginfo, interrupted=sig.requested)
    meta = save_checkpoint(out_dir / "policy.pt", model=model, optimizer=opt, step=step,
                           versions=dict(policy=pcfg.name, featurizer=FEAT_VERSION,
                                         codec=(codec.cfg.version if codec else None)),
                           config=cfg, extra=dict(result=res))
    res["checkpoint"] = meta
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res
