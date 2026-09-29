"""GPU PPO for legged trackers on MuJoCo Warp (W13 P1b, D-138). Same algorithm as rrp.training.tracker_training (asymmetric
actor-critic, GAE, clipped surrogate + clipped value loss, adaptive-KL lr, running normalisers updated between iterations,
time-out bootstrap, performance-gated gait_v2 alpha schedule via AlphaGate), with the whole rollout on the GPU
(rrp.envs.warp_tracker_env). Exports `actor.pt` in the exact LearnedTracker format; the actor is then validated in C MuJoCo
with full self-collision by rrp.evaluation.tracker_validation (the gate is never measured in the training simulator).

Source label of the exported actor: `learned_tracker` (trained with a privileged critic; deployed without it).

usage (peer, GPU lease):
    PYTHONPATH=src:$HOME/work/ext/pylibs/mjwarp python -m rrp.training.warp_tracker_ppo --body t1 --out artifacts/runs/X \
        [--recipe <name|json>] [--nworld 4096] [--iters 1500] [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rrp.envs.tracker_nets import ActorCritic
from rrp.training.reward_schedule import AlphaGate

TRAINER_VERSION = "warp_tracker_ppo_v1"


def export_actor(ac, meta: dict, path: Path, it: int):
    st = dict(actor={k: v.cpu() for k, v in ac.actor.state_dict().items()}, obs_mean=ac.obs_norm.mean.cpu(),
              obs_var=ac.obs_norm.var.cpu(), log_std=ac.log_std.detach().cpu(), meta=dict(meta, iter=it))
    torch.save(st, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def _kv(spec: str) -> dict:
    out = {}
    for kv in filter(None, (spec or "").split(",")):
        k, v = kv.split("=")
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=None, help="name in rrp.training.humanoid_recipes or a JSON file of defaults")
    ap.add_argument("--body")
    ap.add_argument("--out")
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--desired-kl", type=float, default=0.01)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--max-lr", type=float, default=3e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--init-std", type=float, default=0.6)
    ap.add_argument("--hidden", default="512,256,128")
    ap.add_argument("--episode-s", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reward-set", default="")
    ap.add_argument("--cmd-mix", default="default")
    ap.add_argument("--turn-frac", type=float, default=0.25)
    ap.add_argument("--slow-frac", type=float, default=0.0)
    ap.add_argument("--alpha-schedule", default="gated")
    ap.add_argument("--alpha-warmup", type=int, default=300)
    ap.add_argument("--alpha-every", type=int, default=25)
    ap.add_argument("--alpha-step", type=float, default=0.1)
    ap.add_argument("--alpha-advance", default="")
    ap.add_argument("--alpha-backoff", default="")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--nconmax", type=int, default=48)
    ap.add_argument("--njmax", type=int, default=320)
    ap.add_argument("--groups", default=None, help="morph_v1 shared tracker: JSON list of [[body keys], nworld] (or a recipe key)")
    a0, _ = ap.parse_known_args(argv)
    if a0.recipe:
        from rrp.training.humanoid_recipes import recipe_record
        opts, rec = recipe_record(a0.recipe)
        ap.set_defaults(**opts)
        args = ap.parse_args(argv)
        args.recipe_record = rec
    else:
        args = ap.parse_args(argv)
        args.recipe_record = None
    if (not args.body and not args.groups) or not args.out:
        raise SystemExit("--body (or --groups) and --out are required (here or in --recipe)")
    return args


def main(argv=None):
    args = build_args(argv)
    from rrp.envs.warp_tracker_env import ENV_VERSION, MorphMultiEnv, WarpTrackerEnv, window_metrics
    import mujoco_warp
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ekw = dict(reward_overrides=_kv(args.reward_set), episode_s=args.episode_s, push=not args.no_push, cmd_mix=args.cmd_mix,
               turn_frac=args.turn_frac, slow_frac=args.slow_frac, nconmax=args.nconmax, njmax=args.njmax)
    groups = None
    if args.groups:
        groups = json.loads(args.groups) if isinstance(args.groups, str) else args.groups
        env = MorphMultiEnv(groups, seed=args.seed, **ekw)
    else:
        env = WarpTrackerEnv(args.body, args.nworld, seed=args.seed, **ekw)
    N, H = env.N, args.horizon
    hidden = tuple(int(h) for h in args.hidden.split(","))
    ac = ActorCritic(env.obs_dim, env.priv_dim, env.nA, hidden=hidden, init_std=args.init_std).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    gate = AlphaGate(step=args.alpha_step, every=args.alpha_every, warmup=args.alpha_warmup)
    for spec_s, dst in ((args.alpha_advance, gate.advance), (args.alpha_backoff, gate.backoff)):
        dst.update(_kv(spec_s))
    if args.alpha_schedule.startswith("fixed:"):
        gate.alpha = float(args.alpha_schedule.split(":")[1])
    weights = env.set_alpha(gate.alpha)
    it0, lr = 0, args.lr
    ck = out / "checkpoint.pt"
    if args.resume and ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        ac.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it0, lr = st["iter"] + 1, st.get("lr", args.lr)
        g = st.get("gate") or {}
        gate.alpha, gate.history = g.get("alpha", gate.alpha), g.get("history", [])
        weights = env.set_alpha(gate.alpha)
        print(f"resumed from iter {it0}", flush=True)
    meta = dict(body=args.body or "shared", obs_dim=env.obs_dim, priv_dim=env.priv_dim, act_dim=env.nA, control_dt=env.dt, hidden=list(hidden),
                kind=env.b.kind, algo="ppo_asymmetric_actor_critic",
                actor_inputs="public: imu gyro, imu gravity, command, joint pos/vel, last action, gait clock",
                critic_inputs="public + privileged: base lin vel, height, foot contacts, friction, push flag + reward-schedule alpha",
                source_label="learned_tracker (trained with privileged critic)", args={k: v for k, v in vars(args).items()
                                                                                       if k != "recipe_record"},
                contact_model=env.meta.get("contact_model"), reward_version="gait_v2", init_from=None, ref_ff=0.0,
                actuator_limits=env.meta.get("actuator_limits"), actuator=None,
                alpha_schedule=args.alpha_schedule, trainer=TRAINER_VERSION, env_version=ENV_VERSION,
                sim_engine=f"mujoco_warp {getattr(mujoco_warp, '__version__', '3.14.0')}", sim_adaptations=env.adaptations,
                env_differences="per-world randomisation resampled at reset; see rrp.envs.warp_tracker_env docstring",
                reward_options=env.cfg0.options(), recipe=args.recipe_record,
                gpu=torch.cuda.get_device_name(0))
    if groups is not None:
        from rrp.envs.morph_obs import OBS_FORMAT
        meta.update(obs_format=OBS_FORMAT, groups=groups, train_bodies=sorted({k for ks, _ in groups for k in ([ks] if isinstance(ks, str) else ks)}),
                    shared=True, source_label="learned_tracker:shared_morph_v1 (trained with privileged critic)")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log_path = out / "train_log.jsonl"
    obs = env.observe()
    _, priv = obs, env.privileged(torch.zeros(N, env.nf, dtype=torch.bool, device=dev))
    win, t_start = [], time.time()
    D = env.obs_dim
    for it in range(it0, args.iters):
        t0 = time.time()
        bo = torch.zeros(H, N, D, device=dev)
        bp = torch.zeros(H, N, env.priv_dim, device=dev)
        ba = torch.zeros(H, N, env.nA, device=dev)
        blp, br, bd, bv = (torch.zeros(H, N, device=dev) for _ in range(4))
        bmu = torch.zeros(H, N, env.nA, device=dev)
        with torch.no_grad():
            for h in range(H):
                dist = ac.dist(obs)
                a = dist.sample()
                v = ac.value(obs, priv)
                bo[h], bp[h], ba[h] = obs, priv, a
                blp[h] = dist.log_prob(a).sum(-1)
                bmu[h] = dist.mean
                obs, priv, r, d, tmo = env.step(a)
                br[h] = r + args.gamma * v * tmo.float()
                bd[h], bv[h] = d.float(), v
            last_v = ac.value(obs, priv)
        t_roll = time.time() - t0
        rec_env = env.pop_stats()
        win.append(rec_env)
        adv = torch.zeros(H, N, device=dev)
        gae = torch.zeros(N, device=dev)
        for h in reversed(range(H)):
            nv = last_v if h == H - 1 else bv[h + 1]
            nt = 1.0 - bd[h]
            delta = br[h] + args.gamma * nv * nt - bv[h]
            gae = delta + args.gamma * args.lam * nt * gae
            adv[h] = gae
        ret = adv + bv
        fl = lambda x: x.reshape(H * N, *x.shape[2:])
        fo, fp, fa, flp, fadv, fret, fv, fmu = map(fl, (bo, bp, ba, blp, adv, ret, bv, bmu))
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
                    if args.desired_kl > 0:
                        if kl > 2 * args.desired_kl:
                            lr = max(args.min_lr, lr / 1.5)
                        elif 0 < kl < args.desired_kl / 2:
                            lr = min(args.max_lr, lr * 1.5)
                        for g in opt.param_groups:
                            g["lr"] = lr
        with torch.no_grad():
            ac.log_std.clamp_(math.log(0.05), math.log(1.5))
            ac.obs_norm.update(fo)
            ac.priv_norm.update(fp)
        gate_rec = None
        if (it + 1) % gate.every == 0:
            wm = window_metrics(win)
            win = []
            act = gate.update(it, wm) if args.alpha_schedule == "gated" else "off"
            if act in ("advance", "backoff"):
                weights = env.set_alpha(gate.alpha)
            gate_rec = dict(action=act, **wm)
        eps = rec_env["episodes"]
        rec = dict(iter=it, reward_per_step=float(br.mean()), episodes=eps,
                   ep_ret=rec_env["ret_sum"] / eps if eps else None, ep_len=rec_env["len_sum"] / eps if eps else None,
                   fall_rate=rec_env["falls"] / eps if eps else None, std=float(ac.log_std.exp().mean()), lr=lr, kl=kl_mean,
                   value_loss=float(vl.detach()), rollout_s=t_roll, iter_s=time.time() - t0, samples=int((it + 1) * H * N),
                   wall_s=time.time() - t_start, alpha=gate.alpha)
        if rec_env.get("per_group"):
            rec["per_group"] = rec_env["per_group"]
        if gate_rec:
            rec["gate"] = gate_rec
            rec["weights"] = weights
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % 10 == 0 or gate_rec:
            print(json.dumps({k: v for k, v in rec.items() if k != "weights"}), flush=True)
        if (it + 1) % args.ckpt_every == 0 or it == args.iters - 1:
            st = dict(model=ac.state_dict(), opt=opt.state_dict(), iter=it, lr=lr, gate=gate.state())
            torch.save(st, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
            export_actor(ac, dict(meta, alpha=gate.alpha, reward_weights=weights, gate_history=gate.history[-50:]),
                         out / "actor.pt", it)
    print("done", flush=True)


if __name__ == "__main__":
    main()
