"""Compact PPO trainer for body-specific locomotion trackers (asymmetric actor-critic).

Actor: PUBLIC tracker observation only (IMU gyro/gravity, joint encoders, command, last action,
gait clock) -> joint-target offsets. Critic: public obs + PRIVILEGED extras (base linear
velocity, height, foot contacts, friction, push flag). The deployable artifact is the actor
plus its observation normaliser; the critic is discarded.

Rollouts: CPU worker processes, each owning a LeggedEnv (N MjData, one model, a fixed
friction scale sampled per worker). Policy inference + PPO updates in the parent (CPU by
default; `--device cuda` requires a GPU lease and calls rrp.ops.gpu.apply_cap()).
Checkpoints every `--ckpt-every` iterations (resumable with --resume).

usage: python -m rrp.cli train tracker-cpu --body go2 --iters 1500 --workers 11 --envs 64 --out runs/trackers/go2
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


from rrp.core.provenance import file_digest
from rrp.envs.mujoco.legged_vec import VecPool  # noqa: E402  (torch-free worker processes)


# ------------------------------------------------------------------ PPO
def train(args):
    import torch
    import torch.nn as nn
    from rrp.envs.mujoco.tracker_nets import ActorCritic
    from rrp.harness.train.tracker_recipes import assert_trainable
    assert_trainable(args)                      # sealed split: before anything is created
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    gpu_info = {"cuda": False}
    if dev.type == "cuda":
        from rrp.ops.workload import apply_cap
        gpu_info = apply_cap()
    torch.set_num_threads(max(1, args.torch_threads))
    torch.manual_seed(args.seed)
    from rrp.harness.train.reward_schedule import AlphaGate, TerrainCurriculum, window_metrics
    overrides = reward_overrides(args)
    env_kw = dict(push=not args.no_push, episode_s=args.episode_s, contact=args.contact,
                  reward=args.reward, reward_overrides=overrides, actuator=args.actuator)
    tcur = None
    if args.terrain_curriculum == "gated":      # D-126 #15 (default off: flat floor, env_kw unchanged)
        tcur = TerrainCurriculum(amp_max=args.terrain_amp_max, step=args.terrain_step, every=args.alpha_every,
                                 warmup=args.terrain_warmup, after_alpha=args.terrain_after_alpha)
        for spec_s, dst in ((args.terrain_advance, tcur.advance), (args.terrain_backoff, tcur.backoff)):
            for kv in filter(None, (spec_s or "").split(",")):
                k, v = kv.split("=")
                dst[k] = float(v)
        env_kw["terrain"] = dict(amp_max=args.terrain_amp_max, half_m=args.terrain_half_m)
    elif args.terrain_curriculum != "off":
        raise SystemExit(f"--terrain-curriculum {args.terrain_curriculum!r}: off | gated")
    pool = VecPool(args.body, args.workers, args.envs, args.seed, env_kw=env_kw)
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
    if args.ref_ff > 0:
        pool.set_ref_ff(args.ref_ff)
    if args.ref_ff_vmax is not None:
        pool.set_ref_ff_vmax(args.ref_ff_vmax)
    if args.cmd_mix != "default":
        pool.set_cmd_mix(args.cmd_mix)
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
        if getattr(args, "init_actor_sha256", None):
            got = file_digest(Path(args.init_actor), length=None)
            if got != args.init_actor_sha256:
                raise SystemExit(f"--init-actor {args.init_actor} sha256 {got} != declared {args.init_actor_sha256}")
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
        if tcur is not None and st.get("terrain"):
            tcur.level, tcur.history = st["terrain"]["level"], st["terrain"]["history"]
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
                contact_model=sp["contact"], reward_version=sp["reward"], init_from=args.init_actor, ref_ff=args.ref_ff,
                actuator_limits=sp.get("actuator_limits"), actuator=args.actuator,
                alpha_schedule=("gated" if sched else args.alpha_schedule),
                critic_extras=("+ reward-schedule alpha" if sp["reward"] == "gait_v2" else ""))
    meta.update(d126_meta(args, sp, tcur))      # D-126 options: keys only when an option is on (default meta unchanged)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    terrain_amps = pool.set_terrain(tcur.level) if tcur is not None else None
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
        if tcur is not None and (it + 1) % tcur.every == 0:
            # terrain curriculum on the same window as the alpha gate (gait_v2) or its own window (gait_v1)
            wm_t = gate_rec if gate_rec is not None else window_metrics(win)
            if gate_rec is None:
                win = []
            act_t = tcur.update(it, wm_t, alpha=gate.alpha if sp["reward"] == "gait_v2" else 1.0)
            if act_t in ("advance", "backoff"):
                terrain_amps = pool.set_terrain(tcur.level)
            if gate_rec is not None:
                gate_rec["terrain_action"] = act_t
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
        if tcur is not None:
            rec.update(terrain_level=tcur.level, terrain_amp_m=terrain_amps)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % 10 == 0:
            print(json.dumps(rec), flush=True)
        if (it + 1) % args.ckpt_every == 0 or it == args.iters - 1:
            st = dict(model=ac.state_dict(), opt=opt.state_dict(), iter=it, meta=meta, gate=gate.state(),
                      turn_scale=turn_scale, turn_vx=turn_vx if turn_scale is not None else None)
            if tcur is not None:
                st["terrain"] = tcur.state()
            torch.save(st, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
            export_actor(ac, dict(meta, alpha=gate.alpha, reward_weights=weights, gate_history=gate.history[-50:],
                                  **({"terrain_level": tcur.level, "terrain_history": tcur.history[-50:]} if tcur is not None else {})),
                         out / "actor.pt", it)
    pool.close()
    print("done", flush=True)


def _num_or_str(v: str):
    try:
        return float(v)
    except ValueError:
        return v            # D-126: string-valued reward options (ref_gait=clock, limit_margin_agg=max)


def _kv(spec: str) -> dict:
    return {k: _num_or_str(v) for k, v in (kv.split("=") for kv in filter(None, (spec or "").split(",")))}


def reward_overrides(args) -> dict:
    """--reward-set plus the explicit D-126 #13 flags (which win). Empty/unchanged for a pre-D-126 command line."""
    ov = _kv(args.reward_set)
    if args.ref_gait == "clock" and args.ref_gait_mode in ("reward", "both"):
        ov["ref_gait"] = "clock"
        ov.setdefault("ref_lift", 1.0)
        ov.setdefault("ref_contact", 0.5)
    for flag, key in (("yaw_progress_cap", "yaw_progress_cap"), ("yaw_overshoot", "yaw_overshoot"),
                      ("yaw_lin_all_max", "yaw_lin_all_max"), ("limit_margin", "limit_margin"),
                      ("limit_margin_agg", "limit_margin_agg")):
        v = getattr(args, flag)
        if v is not None:
            ov[key] = v
    return ov


def d126_meta(args, sp: dict, tcur) -> dict:
    """Provenance of the D-126 tracker options (only the ones that are on)."""
    from rrp.envs.mujoco.legged_core import TRACKER_OPTIONS_VERSION
    m = {}
    if sp.get("reward_options"):
        m["reward_options"] = sp["reward_options"]
    if args.ref_gait != "none":
        m["ref_gait"] = dict(kind=args.ref_gait, mode=args.ref_gait_mode)
    if args.ref_ff_vmax is not None:
        m["ref_ff_vmax"] = float(args.ref_ff_vmax)          # LearnedTracker applies ref_ff below this speed
    if sp.get("actuator_speed_estimated"):
        m["actuator_speed_estimated"] = sp["actuator_speed_estimated"]   # D-126 #14: joints whose max speed is an ESTIMATE
    if tcur is not None:
        m["terrain_curriculum"] = dict(tcur.state(), history=[], terrain=sp.get("terrain"))
    if getattr(args, "recipe_record", None):
        m["recipe"] = args.recipe_record
    if m:
        m["tracker_options_version"] = TRACKER_OPTIONS_VERSION
    return m


def export_actor(ac, meta: dict, path: Path, it: int):
    import torch
    st = dict(actor=ac.actor.state_dict(), obs_mean=ac.obs_norm.mean.cpu(), obs_var=ac.obs_norm.var.cpu(),
              log_std=ac.log_std.detach().cpu(), meta=dict(meta, iter=it))
    torch.save(st, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=None, help="D-126: JSON file of argument defaults (keys = option dests, e.g. "
                    "a JSON file such as the h1_clock_scratch recipe); explicit command-line options override it; recorded in meta")
    ap.add_argument("--body", default=None, help="(required, here or in --recipe)")
    ap.add_argument("--out", default=None, help="(required, here or in --recipe)")
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
    ap.add_argument("--cmd-mix", default="default", help="default | teacher[:stop_share] (W8 waypoint-teacher command mix, 70%%; e.g. teacher:0.10)")
    ap.add_argument("--ref-ff", type=float, default=0.0,
                    help="bipeds: feed-forward clock stepping reference amplitude (rad) added to targets; stored in actor meta")
    ap.add_argument("--slow-frac", type=float, default=0.0, help="bipeds: fraction of walking commands rescaled to 0.05-0.2 m/s")
    ap.add_argument("--turn-frac", type=float, default=0.25, help="probability of a pure-turn command (bipeds)")
    ap.add_argument("--turn-advance", type=float, default=0.6, help="window turn ratio needed to widen the turn range")
    ap.add_argument("--actuator", default=None, help="v1 (= ideal) PD | v1lat | v2 rrp.bodies.actuator (randomised); default "
                    "$RRP_ACTUATOR_MODE, else rrp.bodies.actuator.ACTUATOR_MODE_DEFAULT (ideal; recorded as v1)")
    ap.add_argument("--reward-set", default="", help="override base reward weights, e.g. clearance_floor=-2,floor_frac=0.6")
    ap.add_argument("--init-actor", default=None, help="warm-start actor + obs normaliser from an exported actor.pt")
    ap.add_argument("--init-actor-sha256", default=None, help="D-126: refuse unless --init-actor has this sha256")
    ap.add_argument("--contact", default="v1", help="contact model: v1 (legacy) | v2 (rrp.morphology.contact)")
    ap.add_argument("--reward", default=None, help="gait_v1 | gait_v2 (default: gait_v2 iff --contact v2)")
    ap.add_argument("--alpha-schedule", default="gated", help="gated | off | fixed:<alpha> (gait_v2 only)")
    ap.add_argument("--alpha-step", type=float, default=0.1)
    ap.add_argument("--alpha-every", type=int, default=25)
    ap.add_argument("--alpha-warmup", type=int, default=300)
    ap.add_argument("--alpha-advance", default="", help="override advance thresholds, e.g. slip_ratio=0.2,fall_rate=0.1")
    ap.add_argument("--alpha-backoff", default="", help="override back-off thresholds")
    # ---- D-126 #13 (all default off) ----
    ap.add_argument("--ref-gait", default="none", choices=["none", "clock"],
                    help="clock: phase-locked foot-lift reference gait (bipeds), see RewardCfg.ref_gait")
    ap.add_argument("--ref-gait-mode", default="reward", choices=["reward", "residual", "both"],
                    help="clock reference as reward terms (ref_lift 1.0 / ref_contact 0.5 unless --reward-set), as a feed-forward "
                         "residual (ref_ff, default 0.1 rad, at every speed), or both")
    ap.add_argument("--ref-ff-vmax", type=float, default=None,
                    help="apply ref_ff below this speed (m/s; default None = 0.25 as before); stored in actor meta")
    ap.add_argument("--yaw-progress-cap", type=float, default=None, help="turn_lin upper clip (default 1.2); g1: 1.0")
    ap.add_argument("--yaw-overshoot", type=float, default=None, help="penalty weight on yaw rate above the command (e.g. -1)")
    ap.add_argument("--yaw-lin-all-max", type=float, default=None, help="yaw_lin_all only for |wz| <= this (rad/s)")
    ap.add_argument("--limit-margin", type=float, default=None, help="joint-limit-margin weight (gait_v2 default -1.0, untuned)")
    ap.add_argument("--limit-margin-agg", default=None, choices=["mean", "max", "sum"], help="joint aggregation (default mean)")
    # ---- D-126 #15 terrain curriculum (default off) ----
    ap.add_argument("--terrain-curriculum", default="off", help="off | gated (bumps_v1 heightfield, gated like alpha)")
    ap.add_argument("--terrain-amp-max", type=float, default=0.10, help="amplitude at level 1 (m); D-112 break-point 0.08")
    ap.add_argument("--terrain-step", type=float, default=0.1)
    ap.add_argument("--terrain-warmup", type=int, default=300)
    ap.add_argument("--terrain-after-alpha", type=float, default=0.0, help="advance only once alpha >= this")
    ap.add_argument("--terrain-half-m", type=float, default=12.0, help="heightfield half-size (m); beyond it a plane")
    ap.add_argument("--terrain-advance", default="", help="e.g. fall_rate=0.1,track_rel_err=0.4")
    ap.add_argument("--terrain-backoff", default="", help="e.g. fall_rate=0.25,track_rel_err=0.6")
    train(parse_args(ap, argv))


def parse_args(ap, argv=None):
    """Parse with optional --recipe defaults (explicit options win). The recipe's path, sha256 and content go to meta."""
    a = ap.parse_args(argv)
    if a.recipe:
        from rrp.harness.train.tracker_recipes import recipe_record
        rec, record = recipe_record(a.recipe)
        dests = {x.dest for x in ap._actions}
        bad = sorted(set(rec) - dests)
        if bad:
            raise SystemExit(f"recipe {a.recipe}: unknown options {bad}")
        ap.set_defaults(**rec)
        a = ap.parse_args(argv)
        a.recipe_record = record
    if not a.body or not a.out:
        raise SystemExit("--body and --out are required (on the command line or in --recipe)")
    from rrp.bodies.actuator import legacy_mode_name
    a.actuator = legacy_mode_name(a.actuator)      # D-126 #14: canonical mode; default "v1" (ideal), as before
    if a.ref_gait == "clock" and a.ref_gait_mode in ("residual", "both"):
        if not a.ref_ff:
            a.ref_ff = 0.1
        if a.ref_ff_vmax is None:
            a.ref_ff_vmax = 1e3                       # every speed
    return a


# ------------------------------------------------------------------ install (HS2, D-146 R2)
def install(run: str | Path, validations: list, body: str, version: str, *, label: str, root: str | Path | None = None,
            decision: str = "accepted", task: str | None = None) -> Path:
    """Register a trained actor in the tracker store `<root>/<body>/<version>/` (root default artifacts/trackers).

    Refuses (raises ValueError, nothing written) unless: `<run>/actor.pt` and `<run>/meta.json`-equivalent actor meta exist; every
    validation JSON (rrp.harness.eval.tracker_validation output) carries tracker_sha == the sha256 of this actor AND
    w6_gate.verdict == "pass" (the D-112 gate; fail / incomplete are never installed, a rejected actor is kept in its run dir);
    the target version does not exist (installs are never overwritten: new version name). `task` (steps | gap | gait; D-147): the
    verdict is instead `gates.tracker_verdict(validation, task)` on the task's gating trials and an `exception` verdict (lab gate passes,
    only margin / force / CoT fail within the D-147 caps) is installed with decision `accepted_d147_exception` and its failing values
    appended to the label. Writes actor.pt, meta.json (the actor's
    own meta + sha256 pin, install_label, installed, validation summary, decision), train_log_every10.jsonl (when the run has a
    train_log.jsonl) and the validation files. Returns the store directory."""
    import shutil
    import torch
    run = Path(run)
    actor = run / "actor.pt"
    if not actor.exists():
        raise ValueError(f"install: {actor} not found")
    if not validations:
        raise ValueError("install: at least one --validation (tracker_validation output with the D-112 gate) is required")
    sha = file_digest(actor, length=None)
    vals = []
    for v in validations:
        vp = Path(v)
        if not vp.exists():
            raise ValueError(f"install: validation {vp} not found")
        rec = json.loads(vp.read_text())
        vs = str(rec.get("tracker_sha") or "")          # tracker_validation records file_digest's 16-hex prefix
        if len(vs) < 16 or not sha.startswith(vs):
            raise ValueError(f"install: {vp} validates sha {str(rec.get('tracker_sha'))[:12]}, not this actor {sha[:12]}")
        if task is not None:
            from rrp.harness.eval.gates import tracker_verdict
            tv = tracker_verdict(rec, task)
            if tv["verdict"] == "fail":
                raise ValueError(f"install: {vp} D-147 verdict for task {task!r} is 'fail' ({tv['label']}; lab {tv['lab']['checks']})")
            if tv["verdict"] == "exception":
                decision, label = "accepted_d147_exception", f"{label} [{tv['label']}]"
            rec = dict(rec, w6_gate=dict(rec.get("w6_gate") or {}, verdict=tv["verdict"], task=task))
        else:
            verdict = (rec.get("w6_gate") or {}).get("verdict")
            if verdict != "pass":
                raise ValueError(f"install: {vp} D-112 gate verdict is {verdict!r}, not 'pass' (failed: "
                                 f"{(rec.get('w6_gate') or {}).get('failed')})")
        vals.append((vp, rec))
    from rrp.envs.mujoco.legged_tracker import TRACKER_DIR
    store = Path(TRACKER_DIR if root is None else root) / body / version
    if store.exists():
        raise ValueError(f"install: {store} already exists (installs are never overwritten; pick a new version)")
    st = torch.load(str(actor), map_location="cpu", weights_only=False)
    from rrp.envs.mujoco.legged_tracker import extra_kind
    meta = dict(st["meta"], sha256=sha, install_label=label, decision=decision, extra_obs=extra_kind(st["meta"]),
                installed=time.strftime("%Y-%m-%d"), installed_from=str(run),
                validations=[dict(file=vp.name, body=r.get("body"), verdict=r["w6_gate"]["verdict"]) for vp, r in vals])
    store.mkdir(parents=True)
    shutil.copy2(actor, store / "actor.pt")
    (store / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if (run / "train_log.jsonl").exists():
        lines = (run / "train_log.jsonl").read_text().splitlines()
        (store / "train_log_every10.jsonl").write_text("\n".join(lines[::10]) + "\n")
    for vp, r in vals:
        name = f"validation_{r.get('body') or vp.stem}.json" if len(vals) > 1 else "validation_learned.json"
        shutil.copy2(vp, store / name)
    return store


def install_main(argv=None):
    ap = argparse.ArgumentParser(description="register a trained tracker (actor.pt from a run dir) in artifacts/trackers/<body>/<version>/ "
                                 "after its D-112 gate passed")
    ap.add_argument("run", help="training run directory holding actor.pt")
    ap.add_argument("--validation", action="append", required=True,
                    help="tracker_validation output JSON (w6_gate.verdict must be pass, tracker_sha must match); repeat per body")
    ap.add_argument("--body", required=True, help="store body key (per-body trackers: the body; shared trackers: e.g. shared)")
    ap.add_argument("--version", required=True)
    ap.add_argument("--label", required=True, help="one-line install label: what it is and what its gate showed")
    ap.add_argument("--decision", default="accepted")
    ap.add_argument("--root", default=None, help="tracker store root (default artifacts/trackers)")
    ap.add_argument("--task", default=None, choices=("steps", "gap", "gait"),
                    help="D-147: judge by the task's gating trials; an exception verdict installs with decision accepted_d147_exception")
    a = ap.parse_args(argv)
    try:
        store = install(a.run, a.validation, a.body, a.version, label=a.label, root=a.root, decision=a.decision, task=a.task)
    except ValueError as e:
        raise SystemExit(str(e))
    print(json.dumps(dict(installed=str(store), spec=f"{a.body}:{a.version}")))
