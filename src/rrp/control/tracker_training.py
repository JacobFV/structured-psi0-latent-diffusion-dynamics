"""Compact PPO trainer for body-specific locomotion trackers (asymmetric actor-critic).

Actor: PUBLIC tracker observation only (IMU gyro/gravity, joint encoders, command, last action,
gait clock) -> joint-target offsets. Critic: public obs + PRIVILEGED extras (base linear
velocity, height, foot contacts, friction, push flag). The deployable artifact is the actor
plus its observation normaliser; the critic is discarded.

Rollouts: CPU worker processes, each owning a LeggedEnv (N MjData, one model, a fixed
friction scale sampled per worker). Policy inference + PPO updates in the parent (CPU by
default; `--device cuda` requires a GPU lease and calls rrp.ops.gpu.apply_cap()).
Checkpoints every `--ckpt-every` iterations (resumable with --resume).

usage: python -m rrp.control.tracker_training --body go2 --iters 1500 --workers 11 --envs 64 --out runs/trackers/go2
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402


from rrp.control.legged_vec import VecPool  # noqa: E402  (torch-free worker processes)


# ------------------------------------------------------------------ PPO
def train(args):
    import torch
    import torch.nn as nn
    from rrp.control.tracker_nets import ActorCritic
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    gpu_info = {"cuda": False}
    if dev.type == "cuda":
        from rrp.ops.gpu import apply_cap
        gpu_info = apply_cap()
    torch.set_num_threads(max(1, args.torch_threads))
    torch.manual_seed(args.seed)
    pool = VecPool(args.body, args.workers, args.envs, args.seed,
                   env_kw=dict(push=not args.no_push, episode_s=args.episode_s))
    sp = pool.spec
    N = args.workers * args.envs
    hidden = tuple(int(h) for h in args.hidden.split(","))
    ac = ActorCritic(sp["obs_dim"], sp["priv_dim"], sp["act_dim"], hidden=hidden, init_std=args.init_std).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    it0 = 0
    log_path = out / "train_log.jsonl"
    ck = out / "checkpoint.pt"
    if args.resume and ck.exists():
        st = torch.load(ck, map_location=dev)
        ac.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it0 = st["iter"] + 1
        print(f"resumed from iter {it0}", flush=True)
    meta = dict(body=args.body, obs_dim=sp["obs_dim"], priv_dim=sp["priv_dim"], act_dim=sp["act_dim"],
                control_dt=sp["dt"], hidden=list(hidden), kind=sp["kind"], algo="ppo_asymmetric_actor_critic",
                actor_inputs="public: imu gyro, imu gravity, command, joint pos/vel, last action, gait clock",
                critic_inputs="public + privileged: base lin vel, height, foot contacts, friction, push flag",
                source_label="learned_tracker (trained with privileged critic)", args=vars(args), gpu=gpu_info)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    obs, priv = pool.reset()
    H = args.horizon
    t_start = time.time()
    lr = args.lr
    for it in range(it0, args.iters):
        t0 = time.time()
        bo = torch.zeros(H, N, sp["obs_dim"])
        bp = torch.zeros(H, N, sp["priv_dim"])
        ba = torch.zeros(H, N, sp["act_dim"])
        blp = torch.zeros(H, N)
        bmu = torch.zeros(H, N, sp["act_dim"])
        br = torch.zeros(H, N)
        bd = torch.zeros(H, N)
        bv = torch.zeros(H, N)
        stats = []
        with torch.no_grad():
            for h in range(H):
                o = torch.as_tensor(obs, device=dev)
                p = torch.as_tensor(priv, device=dev)
                dist = ac.dist(o)
                a = dist.sample()
                v = ac.value(o, p)
                obs, priv, r, d, tmo, st = pool.step(a.cpu().numpy().astype(np.float64))
                stats += st
                r = torch.as_tensor(r, dtype=torch.float32)
                tmo = torch.as_tensor(tmo, dtype=torch.float32)
                r = r + args.gamma * v.cpu() * tmo      # bootstrap time-outs
                bo[h], bp[h], ba[h] = o.cpu(), p.cpu(), a.cpu()
                blp[h] = dist.log_prob(a).sum(-1).cpu()
                bmu[h] = dist.mean.cpu()
                br[h], bd[h], bv[h] = r, torch.as_tensor(d, dtype=torch.float32), v.cpu()
            last_v = ac.value(torch.as_tensor(obs, device=dev), torch.as_tensor(priv, device=dev)).cpu()
        t_roll = time.time() - t0
        adv = torch.zeros(H, N)
        gae = torch.zeros(N)
        for h in reversed(range(H)):
            nv = last_v if h == H - 1 else bv[h + 1]
            nonterm = 1.0 - bd[h]
            delta = br[h] + args.gamma * nv * nonterm - bv[h]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[h] = gae
        ret = adv + bv
        flat = lambda x: x.reshape(H * N, *x.shape[2:]).to(dev)
        fo, fp, fa, flp, fadv, fret, fv, fmu = map(flat, (bo, bp, ba, blp, adv, ret, bv, bmu))
        old_std = ac.log_std.detach().exp()
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        mb = H * N // args.minibatches
        kl_mean = 0.0
        for ep in range(args.epochs):
            perm = torch.randperm(H * N, device=dev)
            for k in range(args.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                dist = ac.dist(fo[idx])
                lp = dist.log_prob(fa[idx]).sum(-1)
                ratio = torch.exp(lp - flp[idx])
                s1 = ratio * fadv[idx]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * fadv[idx]
                v = ac.value(fo[idx], fp[idx])
                v_clip = fv[idx] + torch.clamp(v - fv[idx], -args.clip, args.clip)
                vl = torch.max((v - fret[idx]) ** 2, (v_clip - fret[idx]) ** 2).mean()
                loss = -torch.min(s1, s2).mean() + args.vf_coef * vl - args.ent_coef * dist.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    std = dist.stddev
                    kl = (torch.log(std / old_std) + (old_std ** 2 + (fmu[idx] - dist.mean) ** 2) / (2 * std ** 2)
                          - 0.5).sum(-1).mean().item()
                    kl_mean = kl
                    if args.desired_kl > 0:     # adaptive lr
                        if kl > 2 * args.desired_kl:
                            lr = max(1e-5, lr / 1.5)
                        elif 0 < kl < args.desired_kl / 2:
                            lr = min(1e-2, lr * 1.5)
                        for g in opt.param_groups:
                            g["lr"] = lr
        with torch.no_grad():
            ac.log_std.clamp_(math.log(0.05), math.log(1.5))
            # normaliser updated between iterations (frozen during rollout + update: consistent ratios)
            ac.obs_norm.update(fo)
            ac.priv_norm.update(fp)
        el = time.time() - t0
        rec = dict(iter=it, reward_per_step=float(br.mean()), episodes=len(stats),
                   ep_ret=float(np.mean([s["ret"] for s in stats])) if stats else None,
                   ep_len=float(np.mean([s["len"] for s in stats])) if stats else None,
                   fall_rate=float(np.mean([s["fell"] for s in stats])) if stats else None,
                   std=float(ac.log_std.detach().exp().mean()), lr=lr, kl=kl_mean, value_loss=float(vl.detach()), rollout_s=t_roll, iter_s=el,
                   samples=int((it + 1) * H * N), wall_s=time.time() - t_start)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % 10 == 0:
            print(json.dumps(rec), flush=True)
        if (it + 1) % args.ckpt_every == 0 or it == args.iters - 1:
            st = dict(model=ac.state_dict(), opt=opt.state_dict(), iter=it, meta=meta)
            torch.save(st, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
            export_actor(ac, meta, out / "actor.pt", it)
    pool.close()
    print("done", flush=True)


def export_actor(ac, meta: dict, path: Path, it: int):
    import torch
    st = dict(actor=ac.actor.state_dict(), obs_mean=ac.obs_norm.mean.cpu(), obs_var=ac.obs_norm.var.cpu(),
              log_std=ac.log_std.detach().cpu(), meta=dict(meta, iter=it))
    torch.save(st, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--desired-kl", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--init-std", type=float, default=0.6)
    ap.add_argument("--hidden", default="256,128,64")
    ap.add_argument("--episode-s", type=float, default=20.0)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
