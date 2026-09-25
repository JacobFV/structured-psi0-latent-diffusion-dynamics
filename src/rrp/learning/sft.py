"""Supervised new-body adaptation from a frozen SOURCE checkpoint (independent init per
target/method/seed/budget). Counts demo control transitions and optimizer updates."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from rrp.evaluation.adaptation import nested_budget_indices
from rrp.learning.behavior import device_setup, encode_targets
from rrp.learning.checkpoint import load_checkpoint, save_checkpoint
from rrp.learning.data import ChunkDataset, load_episodes
from rrp.model.flow import FlowPolicy, PolicyConfig


def sft(source_ckpt: Path, dataset: Path, target_robot: str, budget: int, *, seed: int, out_dir: Path,
        steps: int = 300, lr: float = 1e-4, batch_size: int = 128, modules: str = "all",
        demo_pool_seeds=(1000000, 1100000), aux_weight: float = 0.1) -> dict:
    dev, _ = device_setup()
    out_dir.mkdir(parents=True, exist_ok=True)
    st = load_checkpoint(source_ckpt, map_location=dev)
    pcfg = PolicyConfig(**st["config"]["policy"])
    model = FlowPolicy(pcfg).to(dev)
    model.load_state_dict(st["model"])
    pool = load_episodes(dataset, robots={target_robot}, seeds=demo_pool_seeds)
    pool.sort(key=lambda e: e[0]["meta"]["seed"])
    idx = nested_budget_indices(len(pool), [budget], seed)[budget]
    eps = [pool[i] for i in idx]
    transitions = sum(len(p["inputs"]) for p, _ in eps)
    ds = ChunkDataset(eps, pcfg.horizon, stride=1)
    if modules == "action_expert":
        for p in model.context.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    step, t0 = 0, time.time()
    log = open(out_dir / "sft_log.jsonl", "w")
    while step < steps:
        for batch, a, v, lab, eff in ds.batches(min(batch_size, max(8, len(ds))), rng, drop_last=False):
            batch, a, v = batch.to(dev), a.to(dev), v.to(dev)
            lab = {k: x.to(dev) for k, x in lab.items()}
            loss, logs = model.loss(batch, a, v, lab if pcfg.aux else None, aux_weight=aux_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                log.write(json.dumps(dict(step=step, loss=float(loss.detach()), **logs)) + "\n")
            if step >= steps:
                break
    res = dict(target=target_robot, budget=budget, seed=seed, demo_episodes=len(eps), demo_control_transitions=transitions,
               optimizer_updates=step, chunk_presentations=step * batch_size, wall_s=time.time() - t0,
               changed_modules=modules, trainable_params=sum(p.numel() for p in params),
               source_checkpoint=str(source_ckpt), demo_episode_ids=[e[0]["meta"]["episode_id"] for e in eps])
    cfg = dict(st["config"], sft=res)
    save_checkpoint(out_dir / "policy.pt", model=model, optimizer=None, step=step,
                    versions=dict(st["versions"], adapted_for=target_robot), config=cfg, extra=dict(result=res))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res


def sft_packed(source_ckpt: Path, target_packed_dir: Path, budget: int, *, seed: int, out_dir: Path, steps: int,
               lr: float = 1e-4, batch_size: int = 128, aux_weight: float = 0.1) -> dict:
    """Baseline new-body SFT with EXACTLY the acquisition of the latent methods (sft_latent_flow): the same target
    pack (stride-1 rows), the same nested episode choice (nested_budget_indices over the sorted episode ids, seeded),
    the same row sampler, batch min(128, max(8, rows)), lr, and optimizer-update count. All FlowPolicy parameters are
    trained (the analogue of system i); a codec, if any, stays frozen (the analogue of the frozen realizer)."""
    import numpy as np
    from rrp.learning.packed import PackedChunkDataset
    from rrp.model.codec import ActionCodec, CodecConfig
    dev, _ = device_setup()
    st = load_checkpoint(source_ckpt, map_location=dev)
    cfgj = st["config"]
    pcfg = PolicyConfig(**cfgj["policy"])
    model = FlowPolicy(pcfg).to(dev)
    model.load_state_dict(st["model"])
    codec = None
    if cfgj.get("codec_checkpoint"):
        cs = load_checkpoint(Path(cfgj["codec_checkpoint"]), map_location=dev)
        codec = ActionCodec(CodecConfig(**cs["config"]["codec"])).to(dev).eval()
        codec.load_state_dict(cs["model"])
        for p in codec.parameters():
            p.requires_grad_(False)
    data = PackedChunkDataset(target_packed_dir)
    if data.meta["stride"] != 1:
        raise ValueError("target pack must be stride-1 (same rows as the latent methods)")
    ep = np.asarray(data.arr["ep_idx"])
    eps = sorted(set(ep.tolist()))
    chosen = [eps[i] for i in nested_budget_indices(len(eps), [budget], seed)[budget]]
    pool = [int(i) for i in np.nonzero(np.isin(ep, chosen))[0]]
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    B = min(batch_size, max(8, len(pool)))
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "sft_log.jsonl", "w")
    for step in range(steps):
        sel = np.array(sorted(rng.sample(pool, B) if len(pool) >= B else [rng.choice(pool) for _ in range(B)]))
        batch, a, v, lab, _ = data.collate(sel)
        batch, a, v = batch.to(dev), a.to(dev), v.to(dev)
        lab = {k: x.to(dev) for k, x in lab.items()}
        target = encode_targets(codec, batch, a, v)
        loss, logs = model.loss(batch, target, v, lab if pcfg.aux else None, aux_weight=aux_weight, generator=gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            log.write(json.dumps(dict(step=step + 1, loss=float(loss.detach()), **logs)) + "\n")
            log.flush()
    res = dict(target=Path(target_packed_dir).name, budget=budget, seed=seed, demo_episodes=len(chosen),
               demo_episode_pack_ids=chosen, demo_control_transitions=len(pool), optimizer_updates=steps,
               batch_rows=B, chunk_presentations=steps * B, lr=lr, wall_s=time.time() - t0,
               changed_modules="flow_policy_all", frozen=(["action_codec"] if codec is not None else []),
               trainable_params=sum(p.numel() for p in params), source_checkpoint=str(source_ckpt))
    save_checkpoint(out_dir / "policy.pt", model=model, optimizer=None, step=steps,
                    versions=dict(st["versions"], adapted_for=res["target"]), config=dict(cfgj, sft=res),
                    extra=dict(result=res))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res
