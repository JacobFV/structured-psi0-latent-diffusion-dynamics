"""Packet-policy GRPO for the corrected architecture (R38): RL fine-tunes SYSTEM I only.

What is optimized: the flow that emits the latent packet z[knots, assemblies, dz] (FlowPolicy over assemblies).
Frozen: system 0 (LatentRealizer), the target encoder and the packet probes (none of them is even loaded into the
optimizer), and by default the flow's context encoder (`trainable="action_expert"`, as in the old GRPO track).

Sampler: the flow-SDE of `flow_sde.py`, run in the flow's STANDARDIZED latent space (the space the velocity field
lives in); the emitted packet is denormalize(z_K). The fixed affine map has a constant Jacobian, so likelihood
ratios are identical in either space. Per packet we record the whole EM path z_0..z_K, the schedule and the
behavior version; the augmented path log-likelihood is recomputed with gradient on the SAME path and the SAME
observation (stored PolicyInput, re-collated with `assembly_batch`). Padded assemblies contribute exactly 0.

Objective: GRPO clipped surrogate with group-relative advantages over G rollouts of the SAME seed (same initial
state; only system-i sampling noise differs), plus an exact same-variance Gaussian path KL to the frozen reference
flow (`kl_coef`, default 0.05 here: trust region required for this track, a labelled departure from Z-1).

Reward (training signal ONLY, never an observation):
  success      = privileged simulator success evaluator (label `privileged_sim_success`)
  shaping_events (optional, public) = w * fraction of public task-runtime events with status succeeded
  shaping_dist   (optional, PRIVILEGED sim truth) = w * (1 - min(d_cube_zone / 0.3, 1))
Every result row carries the reward label and weights.

Accounting (for comparison with SFT budgets): episodes, env control ticks, system-i packets, velocity-field
evaluations (rollout + update), optimizer updates, wall time. Evaluation episodes are counted separately and never
feed the update.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from rrp.core.errors import ControllerRejection, StaleActionError
from rrp.core.latent_action import LatentActionChunk, AssemblyHandle, EntityHandle
from rrp.harness.train.flow_sde import SDEConfig, sample_sde
from rrp.harness.train.grpo import GRPOConfig, GRPOLearner, group_advantages
from rrp.policies.nets.batch import collate_inputs
from rrp.policies.nets.latent_batch import assembly_batch
from rrp.policies.latent import LatentPolicy
from rrp.policies.system0 import batched_ticks  # noqa: F401  (moved to controllers, W4)

REWARD_LABEL = "privileged_sim_success"
TARGET_BODIES = ("xarm7_pg2", "xarm7_tf3", "panda_tf3")      # D-025: never used during development


def latent_collate(pis):
    return assembly_batch(collate_inputs(pis))


def latent_valid(b, knots: int, dz: int) -> torch.Tensor:
    am = b.node_mask                                              # [B, M] assemblies
    B, M = am.shape
    return am[:, None, :, None].expand(B, knots, M, dz).clone()


class LatentSDEPolicy(LatentPolicy):
    """System i with the stochastic flow-SDE sampler; `last_records[i]` describes the packet for sessions[i]."""

    def __init__(self, model, *, sde: SDEConfig, version: str = "grpo@0", seed: int = 0, **kw):
        super().__init__(model, seed=seed, **kw)
        self.sde = sde.validate()
        self.version = version
        self.last_records: list[dict] = []
        self.velocity_evals = 0

    @torch.no_grad()
    def packets(self, sessions) -> list[LatentActionChunk]:
        t0 = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        feats, obs = [], []
        for s in sessions:
            o = s.observe()
            obs.append(o)
            feats.append(self.featurizer(s)(o))
        b = latent_collate(feats).to(self.device)
        cache = self.model.prepare(b)
        K, dz = len(self.knot_times), self.model.cfg.latent_dim
        valid = latent_valid(b, K, dz)
        noise = torch.randn(valid.shape, generator=self.gen, device=self.device, dtype=cache.ctx.dtype)
        path = sample_sde(lambda z, t: self.model.velocity(z, t, cache), noise, valid, self.sde,
                          behavior_version=self.version, generator=self.gen)
        z = (self.model.denormalize(path.action) * valid.to(path.action.dtype)).float().cpu().numpy()
        if was_training:
            self.model.train()
        out, self.last_records = [], []
        for i, (s, o, pi) in enumerate(zip(sessions, obs, feats)):
            f = self.featurizer(s)
            M = int(b.node_mask[i].sum())
            gasms = [a for a in f.spec.assemblies if a.kind in ("gripper", "hand")][:M]
            now = float(s.data.time)
            out.append(LatentActionChunk(
                latent_space_version=self.lsv, realizer_compat_version=self.rcv, z=z[i][:, :M].astype(np.float32),
                knot_times=self.knot_times,
                assemblies=[AssemblyHandle(handle=f"asm:{f.spec.spec_hash}:{a.frame.link}", robot_index=0) for a in gasms],
                assembly_mask=[True] * M,
                entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in o.object_descriptors],
                observation_id=o.observation_id, graph_version=s.runtime.graph_version,
                runtime_version=s.runtime.runtime_version, robot_spec_hash=f.spec.spec_hash, generated_at=time.time(),
                valid_from=now, valid_until=now + self.validity, source="learned", policy_version=self.version,
                sampling=dict(nfe=self.sde.nfe, sampler="flow_sde", noise_level=self.sde.noise_level)))
            self.last_records.append(dict(pi=pi, path=path.select(slice(i, i + 1)).to("cpu"),
                                          behavior_version=self.version, sim_time=now))
        self.calls += len(sessions)
        self.velocity_evals += len(sessions) * self.sde.nfe
        self.latencies.append(time.perf_counter() - t0)
        return out


# ------------------------------------------------------------------ rollouts
def _public_event_fraction(s) -> float:
    inst = s.runtime.instances
    return sum(v.status == "succeeded" for v in inst.values()) / max(1, len(inst))


def _cube_zone_d(s) -> float | None:
    try:
        c = s.data.xpos[s.model.body("cube").id]
        z = s.data.xpos[s.model.body("target_zone").id]
    except KeyError:
        return None
    return float(np.linalg.norm(c[:2] - z[:2]))


@dataclass
class RewardConfig:
    shaping_events: float = 0.0        # public task-runtime progress
    shaping_dist: float = 0.0          # PRIVILEGED sim-truth cube-to-zone distance (reward only)
    dist_scale: float = 0.3
    shaping_reach: float = 0.0         # PRIVILEGED sim-truth: 1 - min_t d(TCP, cube) / d at learned-control start

    def label(self) -> str:
        parts = [REWARD_LABEL]
        if self.shaping_events:
            parts.append(f"{self.shaping_events}*public_event_fraction")
        if self.shaping_dist:
            parts.append(f"{self.shaping_dist}*privileged_cube_zone_dist")
        if self.shaping_reach:
            parts.append(f"{self.shaping_reach}*privileged_min_tcp_cube_dist")
        return " + ".join(parts)


def _tcp_cube_d(s) -> float:
    from rrp.harness.eval.latent_eval import _tcp
    return float(np.linalg.norm(_tcp(s) - s.data.xpos[s.model.body("cube").id]))


def run_episodes(policy, realizer, robot_key: str, seeds: list[int], *, replan_ticks=8, max_steps=300,
                 task="pick_place", device="cpu", reward: RewardConfig | None = None, record=True,
                 prefix_steps: int = 0) -> list[dict]:
    """Lock-step closed loop (same rules as evaluate_latent): system i every `replan_ticks`, system 0 every tick.
    `seeds` may repeat (a GRPO group = the same seed G times). Returns one dict per episode incl. packet records
    of ACCEPTED packets (rejected packets never influenced control and are excluded from training).
    prefix_steps > 0: CURRICULUM — the SCRIPTED TEACHER (privileged planner, source=scripted_teacher) controls the
    first prefix_steps ticks (deterministic given the seed, so identical within a group), then system i/system 0
    take over for the remaining max_steps - prefix_steps ticks. Results report it; never a deployable score."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    from rrp.policies.system0 import LatentSystem0
    reward = reward or RewardConfig()
    robot = workbench_robots()[robot_key]()
    S, s0, meta = [], [], []
    for sd in seeds:
        s = Session(BUILDERS[task](robot, sd, n_distractors=sd % 3), seed=sd)
        f = policy.featurizer(s)
        S.append(s)
        s0.append(LatentSystem0(realizer, f, latent_space_version=policy.lsv, realizer_compat_version=policy.rcv,
                                device=device))
        meta.append(dict(done=False, outcome=None, steps=0, calls=0, recs=[], teacher_steps=0, min_reach=None))
    if prefix_steps > 0:
        # teacher prefix runs ONCE per distinct seed; group members get a snapshot restore (physics, controller,
        # task runtime, tracker, RNG) so the shared prefix is identical and counted once in accounting.
        from rrp.policies.teachers.arm import PickPlaceTeacher
        leader: dict = {}
        for k, s in enumerate(S):
            if seeds[k] in leader:
                j = leader[seeds[k]]
                if meta[j]["outcome"] != "teacher_prefix_terminal":
                    s.restore(S[j].snapshot())
                meta[k].update(teacher_steps=meta[j]["teacher_steps"], done=meta[j]["done"], outcome=meta[j]["outcome"],
                               shared_prefix=True)
                continue
            leader[seeds[k]] = k
            t = PickPlaceTeacher(s)
            for _ in range(prefix_steps):
                s.step(t.act())
                meta[k]["teacher_steps"] += 1
                if s.runtime.succeeded() or s.data.xpos[s.model.body("cube").id][2] < -0.05:
                    meta[k]["done"] = True
                    meta[k]["outcome"] = "teacher_prefix_terminal"
                    break
    for k, s in enumerate(S):
        meta[k]["min_reach"] = meta[k]["reach0"] = _tcp_cube_d(s)
    for step in range(max_steps - prefix_steps):
        act = [k for k, m in enumerate(meta) if not m["done"]]
        if not act:
            break
        need = [k for k in act if step % replan_ticks == 0 or s0[k].packet is None]
        if need:
            pk = policy.packets([S[k] for k in need])
            recs = getattr(policy, "last_records", None) if record else None
            for j, (k, p) in enumerate(zip(need, pk)):
                meta[k]["calls"] += 1
                try:
                    s0[k].receive(p, now=float(S[k].data.time), graph_version=S[k].runtime.graph_version)
                    if recs:
                        meta[k]["recs"].append(recs[j])
                except (ControllerRejection, StaleActionError):
                    pass
        cmds = batched_ticks([s0[k] for k in act], [S[k] for k in act])
        for k, cmd in zip(act, cmds):
            s = S[k]
            s.step(cmd)
            meta[k]["steps"] += 1
            if True:
                meta[k]["min_reach"] = min(meta[k]["min_reach"], _tcp_cube_d(s))
            if s.data.xpos[s.model.body("cube").id][2] < -0.05:
                meta[k].update(done=True, outcome="failure")
            elif s.runtime.succeeded():
                meta[k]["done"] = True
    out = []
    for k, s in enumerate(S):
        m = meta[k]
        priv = bool(s.privileged_success())
        if m["outcome"] is None:
            m["outcome"] = "success" if priv else ("timeout" if m["steps"] >= max_steps - prefix_steps else "failure")
        ev = _public_event_fraction(s)
        d = _cube_zone_d(s)
        r = float(priv)
        if reward.shaping_events:
            r += reward.shaping_events * ev
        if reward.shaping_dist and d is not None:
            r += reward.shaping_dist * (1 - min(d / reward.dist_scale, 1.0))
        if reward.shaping_reach:
            r += reward.shaping_reach * (1 - min(m["min_reach"] / max(m["reach0"], 1e-6), 1.0))
        out.append(dict(seed=seeds[k], outcome=m["outcome"], privileged_success=priv,
                        public_success=bool(s.runtime.succeeded()), steps=m["steps"], teacher_steps=m["teacher_steps"],
                        shared_prefix=m.get("shared_prefix", False),
                        min_tcp_cube_dist_priv=m["min_reach"], packets=m["calls"],
                        rejected=s0[k].stats.rejected, public_event_fraction=ev, cube_zone_dist_priv=d, reward=r,
                        reward_label=reward.label(), events={e: v.status for e, v in s.runtime.instances.items()},
                        recs=m["recs"]))
    return out


def feasible_seeds(robot_key: str, start: int, n: int, task="pick_place") -> list[int]:
    """The ladder's seed set; the one definition lives in rrp.evaluation.robustness.feasible_arm_seeds (D-126: the
    body was identical; moved down so the evaluation-layer ladder CLI can use it without importing training)."""
    from rrp.harness.eval.robustness import feasible_arm_seeds
    return feasible_arm_seeds(robot_key, start, n, task)


# ------------------------------------------------------------------ training loop
@dataclass
class LatentGRPORunConfig:
    checkpoint: str
    robot: str
    out_dir: str
    iters: int = 20
    groups_per_iter: int = 8
    train_seed_start: int = 3_100_000
    eval_seed_start: int = 3_000_000
    eval_episodes: int = 64
    eval_every: int = 5
    eval_batch: int = 32
    max_steps: int = 300
    replan_ticks: int = 8
    nfe: int = 8
    seed: int = 0
    prefix_steps: int = 0              # scripted-teacher curriculum prefix (labelled); 0 = learned from reset
    allow_target: bool = False
    # D-126 #6 (all default off = the historical run): the deployed system 0 (the lineage's final refit bundle) instead
    # of the flow config's Stage-A representation; a training budget in new control transitions (env steps incl. a
    # teacher prefix, counted once per group; `iters` is then the cap); a policy_it<N>.pt at every eval point; and
    # anchor / forgetting evaluations on the original training bodies (rrp.training.grpo_anchor.AnchorConfig fields).
    representation: str | None = None
    budget_env_steps: int | None = None
    snapshot_evals: bool = False
    anchor: dict | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=lambda: GRPOConfig(
        group_size=8, lr=1e-6, epochs=2, minibatch=64, kl_coef=0.05,
        sde=SDEConfig(nfe=8, noise_level=0.5, first_step="clamp", last_step="deterministic")))


D126_FIELDS = ("representation", "budget_env_steps", "snapshot_evals", "anchor")


def _eval(model, base, realizer, cfg: LatentGRPORunConfig, seeds, device, tag, out: Path, prefix: int = 0) -> dict:
    """Deployed sampler (ODE, nfe) on held-out seeds; no probes (control only). prefix>0: teacher-prefix curriculum
    eval (suffix success; episodes the teacher itself finished are excluded and counted)."""
    pol = LatentPolicy(model, knot_times=base.knot_times, latent_space_version=base.lsv,
                       realizer_compat_version=base.rcv, device=device, nfe=cfg.nfe, name=tag, seed=cfg.seed + 99)
    rows, t0 = [], time.time()
    for i in range(0, len(seeds), cfg.eval_batch):
        rows += run_episodes(pol, realizer, cfg.robot, seeds[i:i + cfg.eval_batch], replan_ticks=cfg.replan_ticks,
                             max_steps=cfg.max_steps, device=device, reward=cfg.reward, record=False,
                             prefix_steps=prefix)
    with open(out / "eval_episodes.jsonl", "a") as fh:
        for r in rows:
            fh.write(json.dumps(dict({k: v for k, v in r.items() if k != "recs"}, tag=tag, sampler="ode",
                                     teacher_prefix_steps=prefix)) + "\n")
    from rrp.harness.eval.statistics import wilson
    teacher_done = sum(r["outcome"] == "teacher_prefix_terminal" for r in rows)
    rows = [r for r in rows if r["outcome"] != "teacher_prefix_terminal"]
    k = sum(r["privileged_success"] for r in rows)
    model.train()
    return dict(tag=tag, teacher_prefix_steps=prefix, teacher_finished_excluded=teacher_done, episodes=len(rows), successes=k, rate=k / max(1, len(rows)), wilson95=wilson(k, len(rows)),
                mean_reward=float(np.mean([r["reward"] for r in rows])),
                mean_event_fraction=float(np.mean([r["public_event_fraction"] for r in rows])),
                mean_min_tcp_cube_dist_priv=float(np.mean([r["min_tcp_cube_dist_priv"] for r in rows])),
                env_steps=sum(r["steps"] for r in rows), wall_s=time.time() - t0)


def train_latent_grpo(cfg: LatentGRPORunConfig) -> dict:
    from rrp.policies.nets.checkpoint import load_checkpoint, save_checkpoint
    from rrp.harness.train.latent_train import load_representation
    from rrp.bodies.armdiv import is_armdiv_sealed
    if (cfg.robot in TARGET_BODIES or is_armdiv_sealed(cfg.robot)) and not cfg.allow_target:
        raise ValueError(f"{cfg.robot} is a sealed target body (D-025); pass allow_target for the campaign stage only")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.ops.workload import apply_cap
        apply_cap()
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_d = asdict(cfg)
    for k_ in D126_FIELDS:            # unset D-126 options are left out: historical runs write the same config.json
        if cfg_d.get(k_) in (None, False):
            cfg_d.pop(k_)
    (out / "config.json").write_text(json.dumps(cfg_d, indent=1, default=str))
    base = LatentPolicy.from_checkpoint(cfg.checkpoint, device=dev, nfe=cfg.nfe)
    model = base.model
    st = load_checkpoint(cfg.checkpoint, map_location="cpu")
    rep_path = cfg.representation or st["config"]["representation"]
    _, _, R, _, rep_res = load_representation(Path(rep_path), dev)   # system 0: frozen
    if cfg.representation:        # deployed (refit) system 0: same latent space, packets addressed to THIS realizer
        if rep_res["latent_space_version"] != base.lsv:
            raise ValueError(f"representation latent space {rep_res['latent_space_version']} != flow {base.lsv}")
        base.rcv = rep_res["realizer_compat_version"]
    tracker = None
    if cfg.anchor:
        from rrp.harness.train.grpo_anchor import AnchorConfig, AnchorTracker, eval_latent_anchor
        acfg = AnchorConfig.from_dict(cfg.anchor)
        tracker = AnchorTracker(acfg, out, lambda tag: eval_latent_anchor(
            model, base, rep_path, acfg, device=dev, out_rows=out / "anchor_episodes.jsonl", tag=tag))
    learner = GRPOLearner(model, cfg.grpo, dev, version_prefix="latent_grpo", collate_fn=latent_collate)
    actor = LatentSDEPolicy(model, sde=cfg.grpo.sde, version=learner.version, seed=cfg.seed,
                            knot_times=base.knot_times, latent_space_version=base.lsv,
                            realizer_compat_version=base.rcv, device=dev, name="latent_grpo_actor")
    G = cfg.grpo.group_size
    eval_seeds = feasible_seeds(cfg.robot, cfg.eval_seed_start, cfg.eval_episodes)
    train_pool = feasible_seeds(cfg.robot, cfg.train_seed_start, cfg.iters * cfg.groups_per_iter)
    if set(eval_seeds) & set(train_pool):
        raise ValueError("train/eval seed overlap")
    acct = dict(train_episodes=0, train_env_steps=0, train_packets=0, rollout_velocity_evals=0,
                update_velocity_evals=0, optimizer_updates=0, informative_groups=0, zero_variance_groups=0,
                eval_episodes=0, eval_env_steps=0)
    log = open(out / "train_log.jsonl", "a")
    evals = []

    def do_eval(tag):
        for pf in sorted({0, cfg.prefix_steps}):
            evals.append(_eval(model, base, R, cfg, eval_seeds, dev, tag, out, prefix=pf))
            acct["eval_episodes"] += evals[-1]["episodes"]; acct["eval_env_steps"] += evals[-1]["env_steps"]
            print(json.dumps(evals[-1]), flush=True)
    do_eval("reference@0")
    if tracker is not None:
        tracker.reference(model)
        acct["anchor_episodes"] = sum(v["n"] for v in tracker.ref["counts"].values())
    budget_hit = False
    t_start = time.time()
    for it in range(cfg.iters):
        if cfg.budget_env_steps is not None and (acct["train_env_steps"] + acct.get("train_teacher_prefix_steps", 0)
                                                 >= cfg.budget_env_steps):
            budget_hit = True
            break
        t0 = time.time()
        seeds = train_pool[it * cfg.groups_per_iter:(it + 1) * cfg.groups_per_iter]
        actor.version = learner.version
        rows = run_episodes(actor, R, cfg.robot, [s for s in seeds for _ in range(G)], replan_ticks=cfg.replan_ticks,
                            max_steps=cfg.max_steps, device=dev, reward=cfg.reward, prefix_steps=cfg.prefix_steps)
        teacher_done = sum(r["outcome"] == "teacher_prefix_terminal" for r in rows)
        t_roll = time.time() - t0
        samples, info = [], dict(informative_groups=0, zero_variance_groups=0)
        for gi in range(len(seeds)):
            grp = rows[gi * G:(gi + 1) * G]
            adv, ok = group_advantages(torch.tensor([r["reward"] for r in grp], dtype=torch.float64),
                                       cfg.grpo.adv_eps, cfg.grpo.zero_var_tol, cfg.grpo.all_equal)
            info["informative_groups" if ok else "zero_variance_groups"] += 1
            if ok:
                for a, r in zip(adv.tolist(), grp):
                    samples += [dict(pi=x["pi"], path=x["path"], adv=a, behavior_version=x["behavior_version"])
                                for x in r["recs"]]
        ust = learner.update(samples)
        acct["train_episodes"] += len(rows)
        acct["train_env_steps"] += sum(r["steps"] for r in rows)
        acct["train_teacher_prefix_steps"] = acct.get("train_teacher_prefix_steps", 0) + sum(r["teacher_steps"] for r in rows if not r["shared_prefix"])
        acct["train_packets"] += sum(r["packets"] for r in rows)
        acct["rollout_velocity_evals"] = actor.velocity_evals
        acct["update_velocity_evals"] = learner.update_velocity_evals
        acct["optimizer_updates"] = learner.opt_steps
        for k_ in ("informative_groups", "zero_variance_groups"):
            acct[k_] += info[k_]
        rec = dict(iter=it + 1, behavior=f"latent_grpo@{it}", sampler="flow_sde", seeds=seeds,
                   sde_success=float(np.mean([r["privileged_success"] for r in rows])),
                   mean_reward=float(np.mean([r["reward"] for r in rows])),
                   group_success=[sum(r["privileged_success"] for r in rows[g * G:(g + 1) * G]) for g in range(len(seeds))],
                   rejected=sum(r["rejected"] for r in rows), teacher_finished=teacher_done,
                   mean_min_reach=float(np.mean([r["min_tcp_cube_dist_priv"] for r in rows])),
                   grasp_rate=float(np.mean([r["events"].get("grasp") == "succeeded" for r in rows])), **info, update=ust, rollout_s=t_roll,
                   iter_s=time.time() - t0, accounting=dict(acct))
        log.write(json.dumps(rec, default=str) + "\n"); log.flush()
        print(json.dumps({k: rec[k] for k in ("iter", "sde_success", "mean_reward", "grasp_rate", "mean_min_reach", "group_success",
                                              "informative_groups", "rollout_s", "iter_s")}),
              json.dumps({k: ust.get(k) for k in ("chunks", "ratio_mean", "clip_frac", "kl", "grad_norm",
                                                  "first_pass_max_abs_ratio_minus_1")}), flush=True)
        spent = acct["train_env_steps"] + acct.get("train_teacher_prefix_steps", 0)
        last = it + 1 == cfg.iters or (cfg.budget_env_steps is not None and spent >= cfg.budget_env_steps)
        if (it + 1) % cfg.eval_every == 0 or last:
            do_eval(f"latent_grpo@{it + 1}")
            if cfg.snapshot_evals:
                save_checkpoint(out / f"policy_it{it + 1}.pt", model=model, optimizer=None, step=learner.opt_steps,
                                versions=dict(st["versions"], latent_grpo=learner.version), config=st["config"],
                                extra=dict(result=dict((st.get("extra") or {}).get("result") or {}),
                                           latent_grpo_iter=it + 1, accounting=dict(acct)))
            if tracker is not None:
                chk = tracker.check(model, f"latent_grpo@{it + 1}")
                acct["anchor_episodes"] += sum(v["n"] for v in chk["counts"].values())
                print(json.dumps(dict(anchor=chk["tag"], regressed=chk["regressed"], reasons=chk["reasons"])), flush=True)
                if tracker.stopped:
                    break
    res = dict(method="latent_grpo", source_checkpoint=cfg.checkpoint, robot=cfg.robot, reward_label=cfg.reward.label(),
               controller_source=f"learned:{cfg.checkpoint}+latent_grpo",
               curriculum=(f"scripted_teacher prefix {cfg.prefix_steps} ticks (privileged planner), learned suffix"
                           if cfg.prefix_steps else "none (learned from reset)"), changed_modules=learner.module_report(),
               frozen=["system0_realizer", "target_encoder", "packet_probes"] + sorted(learner.frozen),
               eval_seeds=[eval_seeds[0], eval_seeds[-1], len(eval_seeds)], evals=evals, accounting=acct,
               train_wall_s=time.time() - t_start, kl_coef=cfg.grpo.kl_coef,
               note="Eval: deployed ODE sampler on held-out seeds; reference@0 = unmodified checkpoint.")
    if cfg.representation:
        res["system0_representation"] = rep_path
    if cfg.budget_env_steps is not None:
        res["budget"] = dict(counter="new_control_transitions (train env steps + teacher prefix, once per group)",
                             budget=cfg.budget_env_steps, spent=acct["train_env_steps"] + acct.get("train_teacher_prefix_steps", 0),
                             reached=budget_hit or acct["train_env_steps"] + acct.get("train_teacher_prefix_steps", 0)
                             >= cfg.budget_env_steps)
    if tracker is not None:
        kept = tracker.restore_best(model) if tracker.stopped else None
        res["anchor"] = dict(tracker.summary(), restored=kept)
    save_checkpoint(out / "policy.pt", model=model, optimizer=None, step=learner.opt_steps,
                    versions=dict(st["versions"], latent_grpo=learner.version), config=st["config"],
                    extra=dict(result=dict((st.get("extra") or {}).get("result") or dict(
                        latent_space_version=base.lsv, realizer_compat_version=base.rcv), latent_grpo=res)))
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res
