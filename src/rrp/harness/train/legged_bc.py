"""POSITIVE CONTROL for the legged latent path: plain behaviour cloning (no packet).

A flow policy maps the SAME public inputs the latent route uses (system i's public context + system 0's local
state: joint encoders, IMU, per-foot touch, osc-v1 phase, morphology tokens) to a chunk of native joint targets
a[t : t+H] (action units, H = 40 = the horizon E encodes). In closed loop it replans every `replan` ticks and
executes the chunk open-loop in between. It is STATELESS (the output depends only on the current observation and
the sampling noise), so it is a valid expert at any learner-visited state (D-050): it provides the stateless
oracle packet E(BC chunk) and the DAgger labels.

Checkpoints: `bc_last.pt` (exact resume: model, optimizer, scheduler, step, rng) every `ckpt_every` steps, plus
`snap_s<step>.pt` model-only snapshots at `snap_every`; `policy.pt` at the end.

usage: python -m rrp.harness.train.legged_bc train --config C.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rrp.policies.features.legged import NODE_STATIC_DIM, ASM_DIM, GLOBAL_DIM
from rrp.policies.nets.flow import MLP, sinusoidal
from rrp.policies.nets.legged_latent import block, run_block
from rrp.harness.train.legged_latent_train import LeggedData, _dev, _save, H, rng_state, restore_rng, cuda_peak_mb
from rrp.policies.nets.legged_bc import LeggedBC, _mha, build, load_bc  # noqa: F401  (moved to models, W4)


def bc_batch(data: LeggedData, i):
    b = data.ctx_batch(i)
    return b, data.beh(i), data.A["amask"][i]


@torch.no_grad()
def eval_bc(model, data, n_batches=20, seed=11, nfe=8):
    """Held-out episodes: chunk MSE of one sample vs demonstrated targets; also first-5-tick MSE and the
    hold-still reference (target = current joint position) for the first tick."""
    model.eval()
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=data.dev).manual_seed(seed)
    r = dict(chunk_mse=[], first5_mse=[], first_mse=[], hold_first_mse=[], zero_first_mse=[])
    for _ in range(n_batches):
        i = data.sample(256, rng, test=True)
        b, a, am = bc_batch(data, i)
        x = model.sample(b, nfe=nfe, generator=g)
        m = am[..., None].float()
        e = (x - a) ** 2 * m
        r["chunk_mse"].append(float(e.sum() / (m.sum() * a.shape[-1])))
        r["first5_mse"].append(float(e[..., :5].sum() / (m.sum() * 5)))
        r["first_mse"].append(float(e[..., 0].sum() / m.sum()))
        hold = data.hold_still(i)
        r["hold_first_mse"].append(float((((hold - a[..., 0]) ** 2) * am).sum() / am.sum()))
        r["zero_first_mse"].append(float(((a[..., 0] ** 2) * am).sum() / am.sum()))
    model.train()
    return {k: float(np.mean(v)) for k, v in r.items()}


def train(cfg, out: Path):
    dev = _dev()
    out.mkdir(parents=True, exist_ok=True)
    data = LeggedData(Path(cfg["data"]), cfg["bodies"], dev)
    model = build(cfg).to(dev)
    steps, lr = cfg["steps"], cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))
    step0 = 0
    last = out / "bc_last.pt"
    if last.exists():
        st = torch.load(str(last), map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        step0 = st["step"]; exact = restore_rng(st, rng)
        print(f"resumed at step {step0} ({'exact: RNG restored' if exact else 'INEXACT: no CUDA RNG state in checkpoint'})", flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "train_log.jsonl", "a")
    B = cfg.get("batch_size", 256)
    ck, sn = cfg.get("ckpt_every", 500), cfg.get("snap_every", 5000)
    t0 = time.time()
    for step in range(step0 + 1, steps + 1):
        i = data.sample(B, rng)
        b, a, am = bc_batch(data, i)
        loss = model.loss(b, a, am)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 200 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), gn=float(gn), cuda_peak_mb=cuda_peak_mb())) + "\n")
            log.flush()
        if step % ck == 0 or step == steps:
            _save(last, model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), step=step,
                  **rng_state(rng), cfg=cfg)
        if step % sn == 0 and step < steps:
            ev = eval_bc(model, data)
            log.write(json.dumps(dict(step=step, eval=ev)) + "\n"); log.flush()
            _save(out / f"snap_s{step}.pt", model=model.state_dict(), cfg=cfg, step=step, eval=ev)
    res = dict(steps=steps, wall_s=time.time() - t0, bodies=cfg["bodies"], n_rows=data.n,
               n_train_rows=len(data.train_idx), n_heldout_rows=len(data.test_idx), eval=eval_bc(model, data),
               source="learned (behaviour cloning of scripted_teacher -> frozen tracker targets; POSITIVE CONTROL)")
    _save(out / "policy.pt", model=model.state_dict(), cfg=cfg, step=steps, result=res)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    if a.steps:
        cfg["steps"] = a.steps
    print(json.dumps(train(cfg, Path(a.out)), indent=1)[:3000])


if __name__ == "__main__":
    main()
