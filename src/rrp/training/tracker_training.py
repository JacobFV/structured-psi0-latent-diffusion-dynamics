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


from rrp.envs.legged_vec import VecPool  # noqa: E402  (torch-free worker processes)


# ------------------------------------------------------------------ PPO
def train(args):
    import torch
    import torch.nn as nn
    from rrp.envs.tracker_nets import ActorCritic
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    gpu_info = {"cuda": False}
    if dev.type == "cuda":
        from rrp.contracts.workload import apply_cap
        gpu_info = apply_cap()
    torch.set_num_threads(max(1, args.torch_threads))
    torch.manual_seed(args.seed)
    from rrp.training.reward_schedule import AlphaGate, window_metrics
    pool = VecPool(args.body, args.workers, args.envs, args.seed,
                   env_kw=dict(push=not args.no_push, episode_s=args.episode_s, contact=args.contact,
                               reward=args.reward, reward_overrides=_kv(args.reward_set), actuator=args.actuator))
    sp = pool.spec
    sched = sp["reward"] == "gait_v2" and args.alpha_schedule == "gated"     # fixed:<a> and off never move alpha
    gate = AlphaGate(step=args.alpha_step, every=args.alpha_every, warmup=args.alpha_warmup)
    for spec_s, dst in ((args.alpha_advance, gate.advance), (args.alpha_backoff, gate.backoff)):
        for kv in filter(None, (spec_s or "").split(",")):
            k, v = kv.split("=")
            dst[k] = float(v)
    if args.alpha_schedule.startswith("fixed:"):
        gate.alpha = float(args.alpha_schedule.split(":")[1])
    weights = pool.set_alpha(gate.alpha) if sp["reward"] == "gait_v2" else sp["reward_weights0"]
    win = []
    saved_alpha0 = False
    turn_scale = None
    if args.slow_frac > 0:
        pool.set_slow_frac(args.slow_frac)
    if args.turn_curriculum > 0:
        turn_scale = pool.set_turn_scale(args.turn_curriculum)
        pool.set_turn_frac(args.turn_frac)
        turn_vx = pool.set_turn_vx(args.turn_vx0)
        if args.resume and (out / "checkpoint.pt").exists():
            ts = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False).get("turn_scale")
            if ts is not None:
                turn_scale = pool.set_turn_scale(ts)
    N = args.workers * args.envs
    hidden = tuple(int(h) for h in args.hidden.split(","))
    ac = ActorCritic(sp["obs_dim"], sp["priv_dim"], sp["act_dim"], hidden=hidden, init_std=args.init_std).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    it0 = 0
    if args.init_actor and not (args.resume and (out / "checkpoint.pt").exists()):
        # warm start: actor + observation normaliser from an exported actor (e.g. the contact_v1 tracker); the
        # critic starts fresh (its privileged inputs differ). Recorded as meta["init_from"].
        ist = torch.load(args.init_actor, map_location=dev, weights_only=False)
        ac.actor.load_state_dict(ist["actor"])
        ac.obs_norm.mean.copy_(ist["obs_mean"])
        ac.obs_norm.var.copy_(ist["obs_var"])
        ac.obs_norm.count.fill_(1e6)       # keep the v1 normaliser nearly frozen at first
        with torch.no_grad():
            ac.log_std.fill_(math.log(args.init_std))
    log_path = out / "train_log.jsonl"
    ck = out / "checkpoint.pt"
    if args.resume and ck.exists():
        st = torch.load(ck, map_location=dev)
        ac.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it0 = st["iter"] + 1
        if st.get("gate"):
            g = st["gate"]
            gate.alpha, gate.history = g["alpha"], g["history"]
            saved_alpha0 = gate.alpha > 0
            if sp["reward"] == "gait_v2":
                weights = pool.set_alpha(gate.alpha)
        print(f"resumed from iter {it0}", flush=True)
    meta = dict(body=args.body, obs_dim=sp["obs_dim"], priv_dim=sp["priv_dim"], act_dim=sp["act_dim"],
                control_dt=sp["dt"], hidden=list(hidden), kind=sp["kind"], algo="ppo_asymmetric_actor_critic",
                actor_inputs="public: imu gyro, imu gravity, command, joint pos/vel, last action, gait clock",
                critic_inputs="public + privileged: base lin vel, height, foot contacts, friction, push flag",
                source_label="learned_tracker (trained with privileged critic)", args=vars(args), gpu=gpu_info,
                contact_model=sp["contact"], reward_version=sp["reward"], init_from=args.init_actor,
                alpha_schedule=("gated" if sched else args.alpha_schedule),
                critic_extras=("+ reward-schedule alpha" if sp["reward"] == "gait_v2" else ""))
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
                win += st
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
                            lr = max(args.min_lr, lr / 1.5)
                        elif 0 < kl < args.desired_kl / 2:
                            lr = min(args.max_lr, lr * 1.5)
                        for g in opt.param_groups:
                            g["lr"] = lr
        with torch.no_grad():
            ac.log_std.clamp_(math.log(0.05), math.log(1.5))
            # normaliser updated between iterations (frozen during rollout + update: consistent ratios)
            ac.obs_norm.update(fo)
            ac.priv_norm.update(fp)
        el = time.time() - t0
        gate_rec = None
        if sched and (it + 1) % gate.every == 0:
            wm = window_metrics(win)
            win = []
            a_before = gate.alpha
            act = gate.update(it, wm)
            if act in ("advance", "backoff"):
                if a_before == 0.0 and not saved_alpha0:     # keep the priors-only policy for the alpha=0 comparison
                    export_actor(ac, dict(meta, alpha=0.0), out / "actor_alpha0.pt", it)
                    saved_alpha0 = True
                weights = pool.set_alpha(gate.alpha)
            gate_rec = dict(action=act, **wm)
            if turn_scale is not None and it >= args.alpha_warmup and wm.get("turn_ratio") is not None:
                # turn-in-place curriculum: widen the pure-turn yaw-rate range when turning is tracked
                if wm["turn_ratio"] >= args.turn_advance and wm["fall_rate"] <= 0.2:
                    if turn_vx > 1e-6:      # arc-to-in-place: shrink the forward component first
                        turn_vx = pool.set_turn_vx(max(0.0, turn_vx - 0.25 * args.turn_vx0))
                        gate_rec["turn_action"] = "shrink_vx"
                    elif turn_scale < 1.0:
                        turn_scale = pool.set_turn_scale(turn_scale + 0.1)
                        gate_rec["turn_action"] = "widen"
            if turn_scale is not None:
                gate_rec["turn_scale"] = turn_scale
                gate_rec["turn_vx"] = turn_vx
        elif sp["reward"] == "gait_v2" and (it + 1) % gate.every == 0:
            gate_rec = dict(action="off", **window_metrics(win))
            win = []
        stats = [s_ for s_ in stats if "fell" in s_]      # episode records (gate accumulators are separate)
        rec = dict(iter=it, reward_per_step=float(br.mean()), episodes=len(stats),
                   ep_ret=float(np.mean([s["ret"] for s in stats])) if stats else None,
                   ep_len=float(np.mean([s["len"] for s in stats])) if stats else None,
                   fall_rate=float(np.mean([s["fell"] for s in stats])) if stats else None,
                   std=float(ac.log_std.detach().exp().mean()), lr=lr, kl=kl_mean, value_loss=float(vl.detach()), rollout_s=t_roll, iter_s=el,
                   samples=int((it + 1) * H * N), wall_s=time.time() - t_start)
        if sp["reward"] == "gait_v2":
            rec.update(alpha=gate.alpha, weights=weights, turn_scale=turn_scale)
            if gate_rec:
                rec["gate"] = gate_rec
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % 10 == 0:
            print(json.dumps(rec), flush=True)
        if (it + 1) % args.ckpt_every == 0 or it == args.iters - 1:
            st = dict(model=ac.state_dict(), opt=opt.state_dict(), iter=it, meta=meta, gate=gate.state(),
                      turn_scale=turn_scale, turn_vx=turn_vx if turn_scale is not None else None)
            torch.save(st, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
            export_actor(ac, dict(meta, alpha=gate.alpha, reward_weights=weights, gate_history=gate.history[-50:]),
                         out / "actor.pt", it)
    pool.close()
    print("done", flush=True)


def _kv(spec: str) -> dict:
    return {k: float(v) for k, v in (kv.split("=") for kv in filter(None, (spec or "").split(",")))}


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
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--desired-kl", type=float, default=0.02)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--max-lr", type=float, default=1e-2,
                    help="adaptive-lr ceiling; use about 1e-3 for warm starts (a converged policy has tiny KL, so the lr climbs to the "
                         "ceiling within one iteration and a few 1e-2 Adam steps can blow the policy up: hexapod6 contact_v2 run 1)")
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
    ap.add_argument("--turn-curriculum", type=float, default=0.0,
                    help="bipeds: start pure-turn yaw-rate scale (e.g. 0.4); 0 = off (full range)")
    ap.add_argument("--turn-vx0", type=float, default=0.0,
                    help="arc-to-in-place curriculum: initial forward speed (m/s) added to pure-turn commands, shrunk to 0 in 4 steps")
    ap.add_argument("--slow-frac", type=float, default=0.0, help="bipeds: fraction of walking commands rescaled to 0.05-0.2 m/s")
    ap.add_argument("--turn-frac", type=float, default=0.25, help="probability of a pure-turn command (bipeds)")
    ap.add_argument("--turn-advance", type=float, default=0.6, help="window turn ratio needed to widen the turn range")
    ap.add_argument("--actuator", default="v1", help="v1 ideal PD | v2 rrp.physics.actuator (randomised)")
    ap.add_argument("--reward-set", default="", help="override base reward weights, e.g. clearance_floor=-2,floor_frac=0.6")
    ap.add_argument("--init-actor", default=None, help="warm-start actor + obs normaliser from an exported actor.pt")
    ap.add_argument("--contact", default="v1", help="contact model: v1 (legacy) | v2 (rrp.morphology.contact)")
    ap.add_argument("--reward", default=None, help="gait_v1 | gait_v2 (default: gait_v2 iff --contact v2)")
    ap.add_argument("--alpha-schedule", default="gated", help="gated | off | fixed:<alpha> (gait_v2 only)")
    ap.add_argument("--alpha-step", type=float, default=0.1)
    ap.add_argument("--alpha-every", type=int, default=25)
    ap.add_argument("--alpha-warmup", type=int, default=300)
    ap.add_argument("--alpha-advance", default="", help="override advance thresholds, e.g. slip_ratio=0.2,fall_rate=0.1")
    ap.add_argument("--alpha-backoff", default="", help="override back-off thresholds")
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
