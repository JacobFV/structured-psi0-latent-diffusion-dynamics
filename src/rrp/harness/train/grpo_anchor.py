"""Anchor / forgetting evaluations for RL fine-tuning (D-126 #6, W7 "GRPO with anchor evals").

GRPO fine-tunes on one robot (or a few); a gain there is only worth something if competence on the ORIGINAL training
tasks and bodies is kept. The anchor evaluation runs the SAME deployable route and harness as the lineage's final
evaluations (rrp.harness.eval.ladder.run_ladder: route `generated` for the latent route with the deployed system 0,
route `learned` for BC; first `episodes` feasible dev seeds from `seed_start`, prev-action as the lineage, no oracle
comparison) on the pre-GRPO checkpoint (reference) and on every GRPO checkpoint, and applies a fixed regression rule:

  regressed  <=>  pooled success drop (reference - current, over all anchor robots) > max_drop
                  OR any single robot's drop > max_drop_robot.
  action     "flag": record and continue; "stop": stop training and keep the last checkpoint that did not regress.

The rule, thresholds, seeds and per-robot counts (with the Newcombe CI of the pooled difference) are written to
anchor_report.json. Evaluation episodes never feed an update and are counted separately from the training budget.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ANCHOR_VERSION = "grpo_anchor_v1"


@dataclass
class AnchorConfig:
    robots: list = field(default_factory=lambda: ["panda_pg2", "parm6_tf3"])
    seed_start: int = 3_000_000
    episodes: int = 30
    max_drop: float = 0.10                 # pooled absolute success-rate drop that counts as a regression
    max_drop_robot: float = 0.20           # per-robot absolute drop that counts as a regression
    action: str = "flag"                   # flag | stop
    prev_action: str = "zero"              # deployment input (the lineage flag zero_prev_action)
    replan_ticks: int = 8
    max_steps: int = 300
    nfe: int = 8
    batch: int = 16

    def __post_init__(self):
        if self.action not in ("flag", "stop"):
            raise ValueError(f"anchor action {self.action!r} (flag|stop)")
        from rrp.bodies.armdiv import is_sealed_target
        bad = [r for r in self.robots if is_sealed_target(r)]
        if bad:
            raise ValueError(f"sealed target bodies cannot be anchors: {bad}")
        if self.seed_start < 3_000_000:
            raise ValueError("anchor seeds are dev seeds (>= 3,000,000)")

    @classmethod
    def from_dict(cls, d: dict | None) -> "AnchorConfig | None":
        return None if not d else cls(**d)


def anchor_verdict(ref: dict, cur: dict, cfg: AnchorConfig) -> dict:
    """ref/cur: {robot: {"success": k, "n": n}} on identical seeds. Returns the rule's inputs and decision."""
    from rrp.harness.eval.statistics import newcombe_diff, wilson
    robots = sorted(ref)
    if set(robots) != set(cur):
        raise ValueError(f"anchor robots differ: {sorted(ref)} vs {sorted(cur)}")
    per = {}
    for r in robots:
        if ref[r]["n"] != cur[r]["n"]:
            raise ValueError(f"{r}: anchor episode counts differ ({ref[r]['n']} vs {cur[r]['n']})")
        n = ref[r]["n"]
        rr, cr = ref[r]["success"] / max(n, 1), cur[r]["success"] / max(n, 1)
        per[r] = dict(ref=ref[r]["success"], cur=cur[r]["success"], n=n, drop=rr - cr)
    K0, K1 = sum(ref[r]["success"] for r in robots), sum(cur[r]["success"] for r in robots)
    N = sum(ref[r]["n"] for r in robots)
    drop = (K0 - K1) / max(N, 1)
    worst = max((v["drop"] for v in per.values()), default=0.0)
    reasons = []
    if drop > cfg.max_drop + 1e-12:
        reasons.append(f"pooled drop {drop:.3f} > {cfg.max_drop}")
    for r, v in per.items():
        if v["drop"] > cfg.max_drop_robot + 1e-12:
            reasons.append(f"{r} drop {v['drop']:.3f} > {cfg.max_drop_robot}")
    return dict(per_robot=per, pooled=dict(ref=K0, cur=K1, n=N, drop=drop, ref_wilson95=wilson(K0, N),
                                           cur_wilson95=wilson(K1, N), diff_cur_minus_ref_ci95=newcombe_diff(K1, N, K0, N)),
                worst_robot_drop=worst, regressed=bool(reasons), reasons=reasons)


def _run(cfg: AnchorConfig, models: dict, ids: dict, route: str, out_rows: Path | None, tag: str, device: str) -> dict:
    from rrp.harness.eval.ladder import LadderConfig, run_ladder
    from rrp.harness.train.latent_grpo import feasible_seeds
    res = {}
    for r in cfg.robots:
        seeds = feasible_seeds(r, cfg.seed_start, cfg.episodes)
        lc = LadderConfig(route=route, robot=r, seeds=seeds, replan_ticks=cfg.replan_ticks, max_steps=cfg.max_steps,
                          nfe=cfg.nfe, compare_oracle=False, device=device, prev_action=cfg.prev_action)
        rows = []
        for i in range(0, len(seeds), cfg.batch):
            lc.seeds = seeds[i:i + cfg.batch]
            rows += run_ladder(lc, None, models, ids)
        if out_rows is not None:
            with open(out_rows, "a") as fh:
                for x in rows:
                    fh.write(json.dumps(dict({k: v for k, v in x.items() if k != "ticks"}, anchor_tag=tag),
                                        default=str) + "\n")
        res[r] = dict(success=int(sum(bool(x["privileged_success"]) for x in rows)), n=len(rows),
                      seeds=[seeds[0], seeds[-1], len(seeds)])
    return res


def eval_latent_anchor(flow_model, base_policy, representation: str, cfg: AnchorConfig, *, device: str,
                       out_rows: Path | None = None, tag: str = "") -> dict:
    """Latent route: the IN-MEMORY system-i flow `flow_model` (same knots / versions as `base_policy`) -> the deployed
    system 0 of `representation` (the lineage's final refit), ladder route `generated`."""
    from rrp.policies.bundles import load_representation
    from rrp.policies.latent import LatentPolicy
    lcfg, E, R, P, res = load_representation(Path(representation), device)
    if base_policy.lsv != res["latent_space_version"]:
        raise ValueError(f"flow latent space {base_policy.lsv} != representation {res['latent_space_version']}")
    pol = LatentPolicy(flow_model, knot_times=base_policy.knot_times, latent_space_version=base_policy.lsv,
                       realizer_compat_version=res["realizer_compat_version"], device=device, nfe=cfg.nfe,
                       name=f"anchor:{tag}", seed=0)
    was = flow_model.training
    flow_model.eval()
    try:
        models = dict(E=E, R=R, P=P, lcfg=lcfg, res=res, flow=pol, learned=None)
        ids = dict(representation=dict(path=str(representation)), anchor=tag)
        return _run(cfg, models, ids, "generated", out_rows, tag, device)
    finally:
        if was:
            flow_model.train()


def eval_bc_anchor(model, codec, cfg: AnchorConfig, *, device: str, out_rows: Path | None = None, tag: str = "") -> dict:
    """Direct-action BC (FlowPolicy [+ frozen codec]) in memory, ladder route `learned` (replan = execute prefix)."""
    from rrp.policies.bc import LearnedPolicy
    was = model.training
    lp = LearnedPolicy(model, codec, device, nfe=cfg.nfe, execute_prefix=cfg.replan_ticks, name=f"anchor:{tag}", seed=0)
    try:
        models = dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None, learned=lp)
        return _run(cfg, models, dict(anchor=tag), "learned", out_rows, tag, device)
    finally:
        if was:
            model.train()


class AnchorTracker:
    """Reference at step 0, then check() at every checkpoint; keeps the last non-regressed weights (CPU copy) so a
    'stop' can restore them. Writes <out>/anchor_report.json after every check."""

    def __init__(self, cfg: AnchorConfig, out_dir: Path, evaluate):
        self.cfg, self.out, self.evaluate = cfg, Path(out_dir), evaluate
        self.ref, self.checks, self.best_state, self.best_tag, self.stopped = None, [], None, None, False

    def _save(self):
        rep = dict(version=ANCHOR_VERSION, config=asdict(self.cfg), reference=self.ref, checks=self.checks,
                   kept_checkpoint=self.best_tag, stopped_on_regression=self.stopped,
                   rule="regressed iff pooled drop > max_drop or any robot drop > max_drop_robot (absolute rates)")
        (self.out / "anchor_report.json").write_text(json.dumps(rep, indent=1, default=str))

    def reference(self, model, tag: str = "reference@0"):
        t0 = time.time()
        self.ref = dict(tag=tag, counts=self.evaluate(tag), wall_s=time.time() - t0)
        self._keep(model, tag)
        self._save()
        return self.ref

    def _keep(self, model, tag):
        self.best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        self.best_tag = tag

    def check(self, model, tag: str) -> dict:
        t0 = time.time()
        counts = self.evaluate(tag)
        v = anchor_verdict(self.ref["counts"], counts, self.cfg)
        rec = dict(tag=tag, counts=counts, wall_s=time.time() - t0, **v)
        self.checks.append(rec)
        if not v["regressed"]:
            self._keep(model, tag)
        elif self.cfg.action == "stop":
            self.stopped = True
        self._save()
        return rec

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        self._save()
        return self.best_tag

    def summary(self) -> dict:
        return dict(version=ANCHOR_VERSION, reference=self.ref, kept_checkpoint=self.best_tag,
                    stopped_on_regression=self.stopped,
                    checks=[{k: c[k] for k in ("tag", "regressed", "reasons", "pooled")} for c in self.checks])
