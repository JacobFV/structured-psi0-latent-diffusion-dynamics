"""GPU PPO for legged trackers on MuJoCo Warp (W13 P1b, D-138). Same algorithm as rrp.harness.train.tracker_training (asymmetric
actor-critic, GAE, clipped surrogate + clipped value loss, adaptive-KL lr, running normalisers updated between iterations,
time-out bootstrap, performance-gated gait_v2 alpha schedule via AlphaGate), with the whole rollout on the GPU
(rrp.envs.warp_tracker_env). Exports `actor.pt` in the exact LearnedTracker format; the actor is then validated in C MuJoCo
with full self-collision by rrp.harness.eval.tracker_validation (the gate is never measured in the training simulator).

Source label of the exported actor: `learned_tracker` (trained with a privileged critic; deployed without it).

usage (peer, GPU lease):
    PYTHONPATH=src:$HOME/work/ext/pylibs/mjwarp python -m rrp.cli train tracker-warp --body t1 --out artifacts/runs/X \
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

from rrp.core.provenance import file_digest
from rrp.envs.mujoco.tracker_nets import ActorCritic
from rrp.harness.train.reward_schedule import AlphaGate

TRAINER_VERSION = "warp_tracker_ppo_v1"


def export_actor(ac, meta: dict, path: Path, it: int):
    st = dict(actor={k: v.cpu() for k, v in ac.actor.state_dict().items()}, obs_mean=ac.obs_norm.mean.cpu(),
              obs_var=ac.obs_norm.var.cpu(), log_std=ac.log_std.detach().cpu(), meta=dict(meta, iter=it))
    torch.save(st, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def upper_ramp_value(it: int, iters: int, *, amp0: float, amp: float, payload_frac: float, ramp: float) -> tuple[float, float]:
    """(upper-body amplitude, payload fraction) at iteration `it`: linear from (amp0, 0) to (amp, payload_frac) over the first
    `ramp` share of `iters`, then constant; ramp 0 = no ramp (the full values from the start). The same function is used live and
    on --resume (it depends on `it` only)."""
    if ramp <= 0:
        return float(amp), float(payload_frac)
    f = min(1.0, max(0.0, it / max(1.0, ramp * iters)))
    return float(amp0 + (amp - amp0) * f), float(payload_frac * f)


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
    ap.add_argument("--recipe", default=None, help="name in rrp.harness.train.tracker_recipes.WARP_RECIPES or a JSON file of defaults")
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
    ap.add_argument("--clock-gate", action="store_true", help="zero the gait-clock inputs under a ~0 command (actor meta clock_gate)")
    ap.add_argument("--target-margin", type=float, default=0.0, help="clip joint targets to the range minus this fraction (meta)")
    ap.add_argument("--land-vel", type=float, default=0.0, help="touchdown downward foot speed^2 penalty weight (training only)")
    ap.add_argument("--force-cap", type=float, default=0.0, help="per-tick foot normal force above --force-cap-bw penalty weight")
    ap.add_argument("--force-cap-bw", type=float, default=2.5)
    ap.add_argument("--teacher-stop", type=float, default=0.10, help="zero-command share inside the teacher command mix")
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
    ap.add_argument("--task", default=None, help="None (tracker) | steps (h_steps expert: public terrain scan in the actor) | "
                    "gap (h_gap expert; privileged gap terms in the critic only), rrp.envs.warp.task_env")
    ap.add_argument("--terrain-scan", action="store_true", help="append the public terrain_scan_v1 (D-146) to the actor input "
                    "(always on for --task steps)")
    ap.add_argument("--upper-body", action="store_true", help="U1 wholebody training: random upper-body (arms, waist, head) targets "
                    "and a payload in the sim, the upper joint state appended to the actor input (actor meta upper_obs)")
    ap.add_argument("--upper-amp", type=float, default=0.4, help="upper-body goal offset, fraction of each joint's half range")
    ap.add_argument("--upper-speed", type=float, default=1.5, help="upper-body target slew limit, rad/s")
    ap.add_argument("--payload-frac", type=float, default=0.08, help="max payload as a fraction of the robot mass")
    ap.add_argument("--upper-amp0", type=float, default=0.1, help="upper-body amplitude at iteration 0 when --upper-ramp > 0")
    ap.add_argument("--upper-ramp", type=float, default=0.0, help="share of --iters over which the upper-body amplitude rises from "
                    "--upper-amp0 to --upper-amp and the payload from 0 to --payload-frac (0 = no ramp)")
    ap.add_argument("--level-every", type=int, default=25, help="task curriculum window (iterations)")
    ap.add_argument("--level-up", type=float, default=0.7, help="window success rate to raise the task level")
    ap.add_argument("--level-down", type=float, default=0.3)
    ap.add_argument("--level0", type=float, default=0.0, help="initial task curriculum level (e.g. resuming a pre-level-save run)")
    ap.add_argument("--init-shared", default=None, help="warm start the actor from an exported morph_v1 actor.pt (extra inputs zero-init)")
    ap.add_argument("--init-shared-sha256", default=None, help="refuse unless --init-shared has this sha256 (the actors in "
                    "tracker_recipes.INIT_PINS are checked without it)")
    ap.add_argument("--groups", default=None, help="morph_v1 shared tracker: JSON list of [[body keys], nworld] (or a recipe key)")
    a0, _ = ap.parse_known_args(argv)
    if a0.recipe:
        from rrp.harness.train.tracker_recipes import recipe_record
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


def make_env(args):
    """(env, groups) for the parsed args: WarpTrackerEnv / task env, or a MorphMultiEnv over --groups."""
    from rrp.envs.warp.tracker_env import MorphMultiEnv, WarpTrackerEnv
    ekw = dict(reward_overrides=_kv(args.reward_set), episode_s=args.episode_s, push=not args.no_push, cmd_mix=args.cmd_mix,
               turn_frac=args.turn_frac, slow_frac=args.slow_frac, nconmax=args.nconmax, njmax=args.njmax,
               teacher_stop=args.teacher_stop, clock_gate=args.clock_gate, target_margin=args.target_margin,
               land_vel=args.land_vel, force_cap=args.force_cap, force_cap_bw=args.force_cap_bw,
               terrain_scan=args.terrain_scan, upper_body=args.upper_body, upper_amp=args.upper_amp,
               upper_speed=args.upper_speed, payload_frac=args.payload_frac)
    groups = None
    env_cls = None
    if args.task == "steps":
        from rrp.envs.warp.task_env import WarpStepsEnv
        env_cls = WarpStepsEnv
    elif args.task == "gap":
        from rrp.envs.warp.task_env import WarpGapEnv
        env_cls = WarpGapEnv
    elif args.task:
        raise SystemExit(f"unknown --task {args.task}")
    if args.groups:
        groups = json.loads(args.groups) if isinstance(args.groups, str) else args.groups
        env = MorphMultiEnv(groups, seed=args.seed, env_cls=env_cls, **ekw)
    else:
        env = (env_cls or WarpTrackerEnv)(args.body, args.nworld, seed=args.seed, **ekw)
    return env, groups


def main(argv=None):
    args = build_args(argv)
    import mujoco_warp
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    env, groups = make_env(args)
    train(args, env, groups=groups, dev=dev, engine=f"mujoco_warp {getattr(mujoco_warp, '__version__', '3.14.0')}")


def public_extra_meta(env) -> dict:
    """The actor-meta keys that describe the env's PUBLIC extra input block, from the env's own declaration (`env.public_extra`:
    the sensors in input order; a group env reads its first group's). Keys: `extra_obs` (the `legged_tracker.PUBLIC_EXTRA` kind),
    `public_extra` ([{sensor, dim, version}] in order) and one sensor spec per declared sensor (`terrain_scan`, `range_ring`), exactly
    what `extra_kind` reads. Refuses an undeclared block (never a guess from the width) and one whose declared dims are not the env's
    `extra_dim`. The only implicit case is the base WarpTrackerEnv, whose extra block is the terrain scan by construction."""
    from rrp.envs.mujoco import legged_core as LC
    from rrp.envs.mujoco.legged_tracker import PUBLIC_EXTRA, extra_kind
    table = dict(terrain_scan=(LC.SCAN_DIM, LC.TERRAIN_SCAN_VERSION, LC.terrain_scan_spec),
                 range_ring=(LC.RING_N, LC.RANGE_RING_VERSION, LC.range_ring_spec))
    n = int(env.extra_dim)
    sensors = getattr(env, "public_extra", None)
    if sensors is None and hasattr(env, "envs"):
        sensors = getattr(env.envs[0], "public_extra", None)
    if sensors is None:
        if n not in (0, LC.SCAN_DIM):
            raise ValueError(f"the env has a {n}-wide extra block but declares no `public_extra`; refusing to label it from its width")
        sensors = ("terrain_scan",) if n else ()
    kind = "+".join(sensors) or "none"
    if kind not in PUBLIC_EXTRA or any(k not in table for k in sensors):
        raise ValueError(f"public_extra {tuple(sensors)} is not one of {sorted(PUBLIC_EXTRA)}")
    if sum(table[k][0] for k in sensors) != n:
        raise ValueError(f"public_extra {tuple(sensors)} declares dims {sum(table[k][0] for k in sensors)} but the env's extra_dim is {n}")
    meta = dict(extra_obs=kind, public_extra=[dict(sensor=k, dim=table[k][0], version=table[k][1]) for k in sensors])
    meta.update({k: table[k][2]() for k in sensors})
    if extra_kind(dict(meta, extra_obs_dim=n)) != kind:
        raise ValueError(f"declared public extra block {kind!r} does not read back through extra_kind")
    return meta


def train(args, env, *, groups=None, dev, engine: str):
    """Asymmetric PPO on `env` (the Warp env, or any object with the same surface), writing meta.json / train_log.jsonl /
    checkpoint.pt / actor.pt under args.out."""
    from rrp.envs.warp.tracker_env import ENV_VERSION, window_metrics
    from rrp.harness.train.tracker_recipes import assert_trainable, check_init_pin
    assert_trainable(args)                       # sealed split: before anything is written
    if args.init_shared:
        check_init_pin(args.init_shared, args.init_shared_sha256)      # T0 pins: a different warm-start actor is refused before any write
    public = public_extra_meta(env)              # the env's declared public extra block (refuses an undeclared one, before anything is written)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_envs = [e for e in getattr(env, "envs", [env]) if hasattr(e, "set_level")]
    level = float(args.level0)
    for e in task_envs:
        e.set_level(level)
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
    if args.init_shared:
        ist = torch.load(args.init_shared, map_location=dev, weights_only=False)
        sd = ist["actor"]
        w0 = sd["0.weight"]
        if w0.shape[1] > env.obs_dim:
            raise SystemExit("--init-shared actor has more inputs than this env")
        pad = env.obs_dim - w0.shape[1]
        sd = dict(sd)
        sd["0.weight"] = torch.cat([w0, torch.zeros(w0.shape[0], pad, device=w0.device)], 1)
        ac.actor.load_state_dict(sd)
        n0 = ist["obs_mean"].shape[0]
        ac.obs_norm.mean[:n0].copy_(ist["obs_mean"])
        ac.obs_norm.var[:n0].copy_(ist["obs_var"])
        ac.obs_norm.count.fill_(1e6)          # keep the warm-start normaliser nearly frozen at first (as tracker_training)
        with torch.no_grad():
            ac.log_std.fill_(math.log(args.init_std))
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
        if st.get("level") and task_envs:                 # task curriculum level (saved since 2026-09-29)
            level = float(st["level"])
            for e in task_envs:
                e.set_level(level)
        print(f"resumed from iter {it0}", flush=True)
    meta = dict(body=args.body or "shared", obs_dim=env.obs_dim, priv_dim=env.priv_dim, act_dim=env.nA, control_dt=env.dt, hidden=list(hidden),
                kind=env.b.kind, algo="ppo_asymmetric_actor_critic",
                actor_inputs="public: imu gyro, imu gravity, command, joint pos/vel, last action, gait clock"
                             + "".join(f", {d['version']} (sensor model: noise, dropout, 1-tick latency)" for d in public["public_extra"])
                             + ((", upper-body joint state (morph_v2: padded q - q0, qdot + static slot descriptors)" if groups is not None
                    else ", upper-body joint state (q - q0, qdot)") if env.upper_dim else ""),
                critic_inputs="public + privileged: base lin vel, height, foot contacts, friction, push flag + reward-schedule alpha"
                              + (", exact noise-free terrain scan" if env.extra_dim else "") + (", task terms" if args.task else "")
                              + (", payload fraction, upper-body target offset" if env.upper_dim else ""),
                source_label="learned_tracker (trained with privileged critic)", args={k: v for k, v in vars(args).items()
                                                                                       if k != "recipe_record"},
                contact_model=env.meta.get("contact_model"), reward_version="gait_v2", init_from=args.init_shared,
                init_sha256=(file_digest(Path(args.init_shared), length=None) if args.init_shared else None),
                ref_ff=0.0,
                actuator_limits=env.meta.get("actuator_limits"), actuator=None,
                alpha_schedule=args.alpha_schedule, trainer=TRAINER_VERSION, env_version=ENV_VERSION,
                sim_engine=engine, sim_adaptations=env.adaptations,
                env_differences="per-world randomisation resampled at reset; see rrp.envs.warp_tracker_env docstring",
                reward_options=env.cfg0.options(), recipe=args.recipe_record, task=args.task,
                clock_gate=bool(args.clock_gate), target_margin=float(args.target_margin), land_vel=float(args.land_vel), force_cap=float(args.force_cap),
                force_cap_bw=float(args.force_cap_bw),
                extra_obs_dim=int(env.extra_dim), gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev)))
    meta.update(public)
    if env.upper_dim:
        meta.update(upper_obs=True, upper_body=env.upper_meta(),
                    upper_ramp=dict(amp0=args.upper_amp0, amp=args.upper_amp, payload_frac=args.payload_frac, ramp=args.upper_ramp,
                                    iters=args.iters))
    if groups is not None:
        from rrp.envs.mujoco.morph_obs import obs_format
        fmt = obs_format(bool(env.upper_dim))
        meta.update(obs_format=fmt, groups=groups, train_bodies=sorted({k for ks, _ in groups for k in ([ks] if isinstance(ks, str) else ks)}),
                    shared=True, source_label=f"learned_tracker:shared_{fmt} (trained with privileged critic)")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log_path = out / "train_log.jsonl"
    obs = env.observe()
    _, priv = obs, env.privileged(torch.zeros(N, env.nf, dtype=torch.bool, device=dev))
    win, task_win, t_start = [], [], time.time()
    D = env.obs_dim
    for it in range(it0, args.iters):
        t0 = time.time()
        if env.upper_dim:
            up_amp, up_pl = upper_ramp_value(it, args.iters, amp0=args.upper_amp0, amp=args.upper_amp,
                                             payload_frac=args.payload_frac, ramp=args.upper_ramp)
            env.set_upper_ramp(up_amp, up_pl)
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
        task_win.append(rec_env)
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
        if task_envs and (it + 1) % args.level_every == 0:
            ws, task_win = task_win, []
            ne = sum(r["episodes"] for r in ws)
            sr = sum(r.get("successes", 0) for r in ws) / ne if ne else 0.0
            if sr > args.level_up:
                level = min(1.0, level + 0.1)
            elif sr < args.level_down and level > 0:
                level = max(0.0, level - 0.1)
            for e in task_envs:
                e.set_level(level)
            rec_level = dict(level=level, success_rate=sr)
        else:
            rec_level = None
        eps = rec_env["episodes"]
        rec = dict(iter=it, reward_per_step=float(br.mean()), episodes=eps,
                   ep_ret=rec_env["ret_sum"] / eps if eps else None, ep_len=rec_env["len_sum"] / eps if eps else None,
                   fall_rate=rec_env["falls"] / eps if eps else None, std=float(ac.log_std.exp().mean()), lr=lr, kl=kl_mean,
                   value_loss=float(vl.detach()), rollout_s=t_roll, iter_s=time.time() - t0, samples=int((it + 1) * H * N),
                   wall_s=time.time() - t_start, alpha=gate.alpha)
        if env.upper_dim:
            rec["upper_amp"], rec["payload_frac"] = up_amp, up_pl
        if task_envs:
            rec["successes"] = rec_env.get("successes", 0)
            rec["level"] = level
            if rec_level:
                rec["level_update"] = rec_level
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
            st = dict(model=ac.state_dict(), opt=opt.state_dict(), iter=it, lr=lr, gate=gate.state(), level=level)
            torch.save(st, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
            export_actor(ac, dict(meta, alpha=gate.alpha, reward_weights=weights, gate_history=gate.history[-50:]),
                         out / "actor.pt", it)
    print("done", flush=True)
