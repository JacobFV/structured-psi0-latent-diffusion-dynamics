"""Online adaptation runs: GRPO (plain / shared-prefix) and EXPO-FT-inspired, at matched NEW control
transitions. Evaluation reuses rrp.evaluation.runner.evaluate (raw per-episode JSONL) on held-out seeds;
evaluation episodes are never counted as training experience and never used for updates."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from rrp.learning.checkpoint import load_checkpoint, save_checkpoint
from rrp.learning.flow_sde import SDEConfig


def _device():
    if torch.cuda.is_available():
        from rrp.ops.gpu import apply_cap
        info = apply_cap()
        torch.backends.cuda.matmul.allow_tf32 = False     # likelihood ratios: keep full fp32 matmuls
        return "cuda", info
    return "cpu", {"cuda": False}


def load_policy(path, device):
    from rrp.model.flow import FlowPolicy, PolicyConfig
    from rrp.model.codec import ActionCodec, CodecConfig
    st = load_checkpoint(path, map_location=device)
    cfg = st["config"]
    model = FlowPolicy(PolicyConfig(**cfg["policy"])).to(device)
    model.load_state_dict(st["model"])
    codec = None
    if cfg.get("codec_checkpoint"):
        cs = load_checkpoint(cfg["codec_checkpoint"], map_location=device)
        codec = ActionCodec(CodecConfig(**cs["config"]["codec"])).to(device)
        codec.load_state_dict(cs["model"])
        codec.eval()
        for p in codec.parameters():
            p.requires_grad_(False)          # frozen codec during adaptation
    return model, codec, st


def scenario_factory(robot_key: str, n_distractors_fn=lambda s: s % 3):
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    robot = workbench_robots()[robot_key]()
    return lambda seed: BUILDERS["pick_place"](robot, seed, n_distractors=n_distractors_fn(seed))


class SeedStream:
    """Training seeds (disjoint from evaluation seeds); infeasible scenarios are skipped and counted."""

    def __init__(self, make_scenario, start: int):
        self.make = make_scenario
        self.next = start
        self.skipped = 0

    def take(self, k: int) -> list[int]:
        from rrp.sim.native import Session
        from rrp.learning.rollout import feasible
        out = []
        while len(out) < k:
            sd = self.next
            self.next += 1
            if feasible(Session(self.make(sd), seed=sd)):
                out.append(sd)
            else:
                self.skipped += 1
        return out


def _evaluate(policy, cfg, out_dir: Path, tag: str, ckpt: str):
    from rrp.evaluation.runner import evaluate, summarize
    seeds = list(range(cfg["eval_seed_start"], cfg["eval_seed_start"] + cfg["eval_episodes"]))
    t0 = time.time()
    res = evaluate(policy, cfg["robot"], seeds, method=f"{cfg['method']}:{tag}", checkpoint=ckpt,
                   max_steps=cfg["max_steps"], batch=cfg.get("eval_batch", 32),
                   out_path=out_dir / "eval_episodes.jsonl")
    s = summarize(res)
    s["eval_wall_s"] = time.time() - t0
    return s


class _RunnerShim:
    """Lets the teacher prefix set LearnedPolicy's private featurizer/prev-action state."""

    def __init__(self, pol):
        self.pol = pol
        self.prev = pol._prev

    def featurizer(self, s):
        return self.pol.featurizer(s)


def _evaluate_suffix(policy, cfg, out_dir: Path, tag: str, ckpt: str):
    """Suffix evaluation: SCRIPTED TEACHER executes up to the public `grasp` boundary, then the evaluated
    policy runs to termination (same rules/max_steps as evaluate()). Held-out eval seeds; raw rows appended
    to eval_suffix_episodes.jsonl. Denominator = feasible episodes whose teacher prefix reached the boundary."""
    from rrp.evaluation.statistics import wilson
    from rrp.learning.rollout import EpisodeState, drive, finalize, event_boundary, teacher_prefix, feasible
    from rrp.sim.native import Session
    make = scenario_factory(cfg["robot"])
    adapter = policy if hasattr(policy, "prev") else _RunnerShim(policy)
    t0 = time.time()
    states, rows = [], []
    for sd in range(cfg["eval_seed_start"], cfg["eval_seed_start"] + cfg["eval_episodes"]):
        s = Session(make(sd), seed=sd)
        if not feasible(s):
            rows.append(dict(seed=sd, outcome="infeasible"))
            continue
        st = EpisodeState(s, sd, cfg["max_steps"], tag=dict(seed=sd))
        if teacher_prefix(adapter, st, event_boundary(cfg.get("branch_event", "grasp")), cfg["max_steps"]):
            st.paused = False
            states.append(st)
        else:
            rows.append(dict(seed=sd, outcome="prefix_failed", teacher_steps=st.steps))
    drive(policy, states)
    for st in states:
        fin = finalize(st)
        rows.append(dict(seed=st.seed, outcome=fin["outcome"], privileged_success=fin["privileged_success"],
                         public_success=fin["public_success"], steps=st.steps,
                         teacher_prefix_steps=st.tag.get("teacher_steps", 0), policy_calls=st.calls,
                         chunk_rejections=st.rejections, command_rejections=st.cmd_rejections, object_fell=st.fell,
                         events={e: v.status for e, v in st.session.runtime.instances.items()}, note=st.note))
    for r in rows:
        r.update(robot=cfg["robot"], method=f"{cfg['method']}:{tag}", checkpoint=ckpt,
                 protocol="suffix_after_scripted_teacher_grasp")
    with open(out_dir / "eval_suffix_episodes.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    att = [r for r in rows if r["outcome"] not in ("infeasible", "prefix_failed")]
    k = sum(bool(r["privileged_success"]) for r in att)
    lo, hi = wilson(k, len(att))
    return dict(protocol="suffix_after_scripted_teacher_grasp", attempted=len(att), successes=k,
                success_rate=k / len(att) if att else None, wilson95=[lo, hi],
                infeasible=sum(r["outcome"] == "infeasible" for r in rows),
                prefix_failed=sum(r["outcome"] == "prefix_failed" for r in rows),
                outcomes={o: sum(r["outcome"] == o for r in att) for o in sorted({r["outcome"] for r in att})},
                eval_wall_s=time.time() - t0)


def _evaluate_any(policy, cfg, out_dir, tag):
    """Full-episode evaluation (rrp.evaluation.runner.evaluate) and, when the run uses the teacher-prefix
    suffix protocol, the suffix evaluation as well (reported separately, never merged)."""
    out = dict(full_episode=_evaluate(policy, cfg, out_dir, tag, cfg["checkpoint"]) if cfg.get("eval_full", True)
               else None)
    if cfg.get("prefix_source") == "teacher" or cfg.get("eval_suffix"):
        out["suffix"] = _evaluate_suffix(policy, cfg, out_dir, tag, cfg["checkpoint"])
    prim = out.get("suffix") or out["full_episode"]
    out.update({k: prim[k] for k in ("attempted", "successes", "success_rate", "wilson95")}, primary=prim.get(
        "protocol", "full_episode"))
    return out


def run(cfg: dict) -> dict:
    method = cfg["method"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    dev, ginfo = _device()
    torch.manual_seed(cfg.get("seed", 0))
    model, codec, st = load_policy(cfg["checkpoint"], dev)
    make = scenario_factory(cfg["robot"])
    seeds = SeedStream(make, cfg["train_seed_start"])
    log = open(out_dir / "train_log.jsonl", "a")
    budgets = sorted(cfg["budgets"])
    results = dict(method=method, robot=cfg["robot"], checkpoint=cfg["checkpoint"], gpu=ginfo, evals=[])
    counters = dict(new_transitions=0, learned_transitions=0, reused_prefix_transitions=0, episodes=0, groups=0,
                    rollout_wall_s=0.0,
                    update_wall_s=0.0)
    execute_prefix, nfe = cfg.get("execute_prefix", 8), cfg.get("nfe", 8)
    if method in ("grpo", "grpo_shared_prefix"):
        from rrp.learning.grpo import GRPOConfig, GRPOLearner, build_samples
        from rrp.learning.branching import collect_plain, collect_shared_prefix
        from rrp.learning.rollout import SDEPolicy
        from rrp.policy.runner import LearnedPolicy
        gc = dict(cfg.get("grpo", {}))
        sde = SDEConfig(**gc.pop("sde", {"nfe": nfe}))
        gcfg = GRPOConfig(sde=sde, **gc)
        learner = GRPOLearner(model, gcfg, dev, version_prefix=method)
        results["modules"] = learner.module_report()
        actor = SDEPolicy(model, codec, dev, sde, execute_prefix=execute_prefix, version=learner.version,
                          seed=cfg.get("seed", 0))

        def evaluate_now(tag):
            pol = LearnedPolicy(model, codec, dev, nfe=nfe, execute_prefix=execute_prefix, name=f"{method}:{tag}",
                                seed=cfg.get("eval_policy_seed", 0))
            r = _evaluate_any(pol, cfg, out_dir, tag)
            model.train()
            return r

        def train_iteration():
            ids = seeds.take(cfg.get("groups_per_iter", 8))
            actor.version = learner.version
            t0 = time.time()
            if method == "grpo":
                groups = collect_plain(actor, make, ids, gcfg.group_size, cfg["max_steps"],
                                       cfg.get("shaping_grasp", 0.0), prefix_source=cfg.get("prefix_source", "none"),
                                       event=cfg.get("branch_event", "grasp"))
            else:
                groups = collect_shared_prefix(actor, make, ids, gcfg.group_size, cfg["max_steps"],
                                               cfg.get("branch_event", "grasp"), cfg.get("shaping_grasp", 0.0),
                                               prefix_source=cfg.get("prefix_source", "learned"))
            t1 = time.time()
            samples, info = build_samples([dict(returns=g.returns, members=g.members) for g in groups
                                           if len(g.returns) >= 2], gcfg)
            upd = learner.update(samples)
            t2 = time.time()
            counters["rollout_wall_s"] += t1 - t0
            counters["update_wall_s"] += t2 - t1
            counters["new_transitions"] += sum(g.new_transitions for g in groups)
            counters["learned_transitions"] += sum(g.learned_transitions for g in groups)
            counters["reused_prefix_transitions"] += sum(g.reused_prefix_transitions for g in groups)
            counters["groups"] += len(groups)
            counters["episodes"] += sum(len(g.returns) for g in groups)
            succ = [r for g in groups for r in g.returns]
            row = dict(iteration=learner.iteration, t=time.time(), **counters, **info,
                       train_success_rate=sum(succ) / max(len(succ), 1),
                       boundary_rate=(sum(bool(g.reached_boundary) for g in groups) / len(groups))
                       if method == "grpo_shared_prefix" else None,
                       snapshot_s=sum(g.snapshot_s + g.restore_s for g in groups),
                       rollout_policy_velocity_evals=actor.velocity_evals,
                       update_velocity_evals=learner.update_velocity_evals, opt_steps=learner.opt_steps,
                       skipped_infeasible_seeds=seeds.skipped, update=upd,
                       groups_detail=[g.summary() for g in groups])
            log.write(json.dumps(row, default=str) + "\n")
            log.flush()
            print(json.dumps({k: row[k] for k in ("iteration", "new_transitions", "train_success_rate",
                                                  "informative_groups", "zero_variance_groups")}), flush=True)

        def save(tag):
            save_checkpoint(out_dir / f"policy_{tag}.pt", model=model, optimizer=learner.opt, step=learner.opt_steps,
                            versions=dict(st["versions"], adaptation=learner.version), config=st["config"],
                            extra=dict(adaptation=cfg, counters=dict(counters)))
    elif method == "expo":
        from rrp.learning.expo import ExpoAgent, ExpoConfig, collect_expo_episodes
        from rrp.learning.replay_buffer import ReplayBuffer
        from rrp.sim.native import Session
        ecfg = ExpoConfig(nfe=nfe, **cfg.get("expo", {}))
        probe = Session(make(cfg["eval_seed_start"]), seed=0)
        from rrp.data.collect import featurizer_for
        n_nodes = len(featurizer_for(probe).aspace.node_group)
        agent = ExpoAgent(model, codec, dev, ecfg, execute_prefix=execute_prefix, action_nodes=n_nodes,
                          seed=cfg.get("seed", 0))
        buf = ReplayBuffer(controller_version=probe.controller_version(), capacity=ecfg.replay_capacity,
                           robot_spec_hash=probe.robots[0].spec.spec_hash)
        results["modules"] = dict(frozen_modules=sorted(agent.frozen) + ["codec", "controller", "critic_encoder"],
                                  base_trainable_params=sum(p.numel() for p in agent.base_params),
                                  critic_params=sum(p.numel() for p in agent.q.parameters()),
                                  edit_params=sum(p.numel() for p in agent.edit.parameters()))
        pending = dict(steps=0.0)

        def evaluate_now(tag):
            agent.deterministic = True
            r = _evaluate_any(agent, cfg, out_dir, tag)
            agent.deterministic = False
            return r

        def train_iteration():
            ids = seeds.take(cfg.get("episodes_per_iter", 16))
            t0 = time.time()
            eps = collect_expo_episodes(agent, make, ids, cfg["max_steps"], buf, model.cfg.horizon,
                                        prefix_source=cfg.get("prefix_source", "none"),
                                        event=cfg.get("branch_event", "grasp"))
            t1 = time.time()
            steps = sum(e["steps"] for e in eps)
            pending["steps"] += steps
            calls = int(pending["steps"] // ecfg.env_steps_per_update)
            pending["steps"] -= calls * ecfg.env_steps_per_update
            logs = agent.update(buf, calls)
            t2 = time.time()
            counters["rollout_wall_s"] += t1 - t0
            counters["update_wall_s"] += t2 - t1
            counters["new_transitions"] += steps
            counters["learned_transitions"] += steps - sum(e["teacher_steps"] for e in eps)
            counters["episodes"] += len(eps)
            row = dict(t=time.time(), **counters, train_success_rate=sum(e["privileged_success"] for e in eps) / len(eps),
                       replay=len(buf), bc_windows=len(buf.bc), update_calls_this_iter=calls, **agent.stats,
                       skipped_infeasible_seeds=seeds.skipped, update=logs, episodes_detail=eps)
            log.write(json.dumps(row, default=str) + "\n")
            log.flush()
            print(json.dumps({k: row[k] for k in ("new_transitions", "train_success_rate", "replay", "bc_windows",
                                                  "critic_steps")} | {"q": logs.get("q_mean"),
                                                                      "alpha": logs.get("alpha")}), flush=True)

        def save(tag):
            save_checkpoint(out_dir / f"policy_{tag}.pt", model=model, optimizer=agent.opt_b, step=agent.base_iter,
                            versions=dict(st["versions"], adaptation=agent.policy_version), config=st["config"],
                            extra=dict(adaptation=cfg, counters=dict(counters), stats=dict(agent.stats)))
            torch.save(dict(q=agent.q.state_dict(), q_t=agent.q_t.state_dict(), edit=agent.edit.state_dict(),
                            log_alpha=agent.log_alpha.detach().cpu(), edit_version=agent.edit_version),
                       out_dir / f"expo_heads_{tag}.pt")
    else:
        raise ValueError(method)

    t_start = time.time()
    for b in budgets:
        while counters[cfg.get("budget_counter", "new_transitions")] < b:
            train_iteration()
        tag = f"b{b}"
        if b > 0:
            save(tag)
        ev = evaluate_now(tag)
        ev.update(budget=b, budget_counter=cfg.get("budget_counter", "new_transitions"),
                  actual_new_transitions=counters["new_transitions"],
                  actual_learned_transitions=counters["learned_transitions"],
                  reused_prefix_transitions=counters["reused_prefix_transitions"], wall_s=time.time() - t_start,
                  counters=dict(counters))
        results["evals"].append(ev)
        print("EVAL", tag, json.dumps({k: ev[k] for k in ("attempted", "successes", "success_rate", "wilson95")}),
              flush=True)
        (out_dir / "result.json").write_text(json.dumps(results, indent=1, default=str))
    results["counters"] = counters
    results["wall_s"] = time.time() - t_start
    (out_dir / "result.json").write_text(json.dumps(results, indent=1, default=str))
    return results
