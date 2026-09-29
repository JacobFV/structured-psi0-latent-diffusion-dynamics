"""Per-input-group privileged-information audit (D-126 roadmap #28).

Three parts:
1. RUNTIME ABLATION AUDIT of a trained legged route (latent flow, oracle rep, or BC): every declared input group of
   the policy's inputs (system-i public context, system-0 local state) is ablated in turn,
     zero     the group is set to 0,
     shuffle  the group takes the value recorded at the same call index in ANOTHER episode (across-episode shuffle),
     noise    the group is replaced by Gaussian noise with the recorded per-dimension mean/std,
   plus an `identity` control (the unablated run repeated: measures run-to-run nondeterminism). Episodes are paired
   by seed with the unablated baseline. Per (group, ablation): success/fall counts and paired deltas, trajectory
   deviation from the baseline (privileged truth, SCORING only), final-pose shift, with bootstrap CIs.
   Flags: `changes_behaviour` (deviation CI above the threshold), `expected_irrelevant_but_used` (a group the caller
   declares irrelevant changes behaviour), `privileged_derived_used` (a group whose provenance is derived from
   simulator truth, e.g. the truth+noise localization speed, changes behaviour; re-run with the estimator).
2. DYNAMIC TRIPWIRE: builds a session and computes every deployable input (observe, public_context, local_state,
   the estimator tick) with the privileged accessors (base_pose_truth, truth_predicate, privileged_success,
   _held_truth) replaced by functions that raise; the public observation must pass the public transport
   (rrp.contracts.channels).
3. STATIC CHECK: the functions that compute deployable inputs are parsed (ast) and must not reference privileged
   names (PrivilegedTruth, base_pose_truth, truth_predicate, privileged_success, object poses, ...) nor the free
   joint (binding.qa / binding.da = true base state). Declared exceptions (e.g. the declared truth+noise
   localization sensor) are listed with their reason and reported, never silently passed.

Probes/ablations are diagnostics: a group that changes behaviour when ablated is USED; whether its use is legitimate
is decided by its declared provenance, not by the audit.

usage:
  python -m rrp.harness.eval.privileged_audit run --flow F/policy.pt --bodies go2 --seeds 10000-10009 --out DIR
      [--groups ctx.speed,local.qd] [--ablations zero,shuffle,noise] [--expect-irrelevant ctx.osc] [--max-s 30]
      [--base-state-source estimator]
  python -m rrp.harness.eval.privileged_audit static
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

AUDIT_VERSION = "paudit-1"
ABLATIONS = ("zero", "shuffle", "noise")
TRUTH_DERIVED = "truth_noise"          # provenance class of channels computed from simulator truth plus noise


@dataclass(frozen=True)
class InputGroup:
    name: str
    where: str                         # "ctx" (system-i public context) | "local" (system-0 local state)
    index: tuple                       # ctx: (lo, hi) slice; local: (field,) in ("q", "qd", "imu", "touch")
    provenance: str                    # public_sensor | controller_state | task_view | declared_estimator | truth_noise
    note: str = ""


# rrp.features.legged.public_context layout (22): gyro 3 | gravity 3 | osc sin/cos 2 | waypoint a 4 | waypoint b 4 |
# active event one-hot 4 | speed estimate, speed valid 2
LEGGED_GROUPS = (
    InputGroup("ctx.gyro", "ctx", (0, 3), "public_sensor", "IMU gyro"),
    InputGroup("ctx.gravity", "ctx", (3, 6), "public_sensor", "IMU orientation -> gravity direction"),
    InputGroup("ctx.osc", "ctx", (6, 8), "controller_state", "free-running gait clock"),
    InputGroup("ctx.waypoint_a", "ctx", (8, 12), TRUTH_DERIVED,
               "detector track (truth+noise, visibility) relative to the declared localization (truth+noise)"),
    InputGroup("ctx.waypoint_b", "ctx", (12, 16), TRUTH_DERIVED, "as waypoint_a"),
    InputGroup("ctx.event", "ctx", (16, 20), "task_view", "public task runtime (active event)"),
    InputGroup("ctx.speed", "ctx", (20, 22), TRUTH_DERIVED,
               "1 s differences of the declared localization (truth+noise); bse-1 estimator with "
               "base_state_source=estimator"),
    InputGroup("local.q", "local", ("q",), "public_sensor", "joint encoders"),
    InputGroup("local.qd", "local", ("qd",), "public_sensor", "joint velocities"),
    InputGroup("local.imu", "local", ("imu",), "public_sensor", "gyro + gravity direction"),
    InputGroup("local.touch", "local", ("touch",), "public_sensor", "foot touch sensors"),
)
GROUPS = {g.name: g for g in LEGGED_GROUPS}
LOCAL_FIELDS = ("q", "qd", "imu", "touch")


def provenance_of(group: InputGroup, base_state_source: str = "truth_noise") -> str:
    if group.name == "ctx.speed" and base_state_source == "estimator":
        return "declared_estimator"
    return group.provenance


# ---------------------------------------------------------------------------------------------- value bank
class ValueBank:
    """Per-episode sequences of each group's value, recorded on the baseline runs (call index -> value)."""

    def __init__(self):
        self.ep: dict[int, dict[str, list[np.ndarray]]] = {}

    def recorder(self, seed: int):
        rec = self.ep.setdefault(seed, {})

        def ctx_rec(c, ad):
            for g in LEGGED_GROUPS:
                if g.where == "ctx":
                    rec.setdefault(g.name, []).append(np.array(c[g.index[0]:g.index[1]], np.float32))
            return c

        def local_rec(q, qd, imu, touch, ad):
            for g, v in zip(("local.q", "local.qd", "local.imu", "local.touch"), (q, qd, imu, touch)):
                rec.setdefault(g, []).append(np.array(v, np.float32))
            return q, qd, imu, touch
        return ctx_rec, local_rec

    def stats(self, group: str):
        vals = [v for e in self.ep.values() for v in e.get(group, [])]
        if not vals:
            return None
        a = np.stack(vals)
        return a.mean(0), a.std(0)

    def other(self, seed: int, group: str, k: int):
        """Value at call index k of the NEXT recorded episode (cyclic over seeds; index clamped to its length)."""
        seeds = sorted(self.ep)
        if len(seeds) < 2:
            raise ValueError("shuffle ablation needs >= 2 baseline episodes")
        o = seeds[(seeds.index(seed) + 1) % len(seeds)]
        seq = self.ep[o].get(group) or []
        return seq[min(k, len(seq) - 1)] if seq else None


def make_transforms(group: InputGroup, ablation: str, bank: ValueBank, seed: int, rng: np.random.Generator):
    """(ctx_transform, local_transform) for LatentLeggedController / BCController hooks."""
    counter = dict(n=0)

    def value(orig):
        if ablation == "zero":
            return np.zeros_like(orig)
        if ablation == "noise":
            st = bank.stats(group.name)
            mu, sd = (np.zeros_like(orig), np.ones_like(orig)) if st is None else st
            return (mu + sd * rng.standard_normal(orig.shape)).astype(orig.dtype)
        if ablation == "shuffle":
            v = bank.other(seed, group.name, counter["n"])
            return orig if v is None or v.shape != orig.shape else v.astype(orig.dtype)
        raise ValueError(ablation)

    if group.where == "ctx":
        lo, hi = group.index

        def ctx_t(c, ad):
            c[lo:hi] = value(c[lo:hi])
            counter["n"] += 1
            return c
        return ctx_t, None
    fi = LOCAL_FIELDS.index(group.index[0])

    def local_t(q, qd, imu, touch, ad):
        v = [q, qd, imu, touch]
        v[fi] = value(v[fi])
        counter["n"] += 1
        return tuple(v)
    return None, local_t


# ---------------------------------------------------------------------------------------------- behaviour metrics
def trajectory_deviation(base_row, row) -> float | None:
    """Mean xy distance between the two runs' base trajectories over their common length (privileged; scoring)."""
    a, b = base_row.get("trace") or [], row.get("trace") or []
    n = min(len(a), len(b))
    if n == 0:
        return None
    return float(np.mean([math.hypot(a[i]["pose"][0] - b[i]["pose"][0], a[i]["pose"][1] - b[i]["pose"][1])
                          for i in range(n)]))


def compare(base_rows: list[dict], rows: list[dict], threshold_m: float) -> dict:
    from rrp.harness.eval.statistics import boot_ci, mcnemar_exact
    bs = {r["seed"]: r for r in base_rows}
    pairs = [(bs[r["seed"]], r) for r in rows if r["seed"] in bs]
    dev = [trajectory_deviation(b, r) for b, r in pairs]
    fshift = [math.hypot(b["final_pose"][0] - r["final_pose"][0], b["final_pose"][1] - r["final_pose"][1])
              for b, r in pairs]
    lost = sum(b["success"] and not r["success"] for b, r in pairs)
    gained = sum(r["success"] and not b["success"] for b, r in pairs)
    dci = boot_ci([d for d in dev if d is not None])
    return dict(n=len(pairs), success=sum(r["success"] for _, r in pairs), base_success=sum(b["success"] for b, _ in pairs),
                fell=sum(r["fell"] for _, r in pairs), base_fell=sum(b["fell"] for b, _ in pairs),
                success_lost=lost, success_gained=gained, mcnemar_p=mcnemar_exact(lost, gained) if lost + gained else None,
                traj_dev_m=dci, final_shift_m=boot_ci(fshift),
                changes_behaviour=bool(dci["lo"] is not None and dci["lo"] > threshold_m) or lost + gained > 0 and (
                    mcnemar_exact(lost, gained) < 0.05))


def summarize(results: dict, *, threshold_m: float, expect_irrelevant=(), base_state_source="truth_noise") -> dict:
    """results: {(group, ablation): comparison}; adds the flags."""
    out = dict(version=AUDIT_VERSION, threshold_m=threshold_m, groups={}, flags=[])
    for (g, ab), c in sorted(results.items()):
        grp = GROUPS.get(g)
        prov = provenance_of(grp, base_state_source) if grp else "control"
        out["groups"].setdefault(g, dict(provenance=prov, note=grp.note if grp else "", ablations={}))
        out["groups"][g]["ablations"][ab] = c
        if g == "identity" or not c["changes_behaviour"]:
            continue
        if g in expect_irrelevant:
            out["flags"].append(dict(flag="expected_irrelevant_but_used", group=g, ablation=ab))
        if prov == TRUTH_DERIVED:
            out["flags"].append(dict(flag="privileged_derived_used", group=g, ablation=ab,
                                     note="behaviour depends on a channel computed from simulator truth + noise"))
    ident = results.get(("identity", "none"))
    if ident is not None and ident["changes_behaviour"]:
        out["flags"].append(dict(flag="nondeterministic_baseline", group="identity",
                                 note="the unablated rerun differs from the baseline: effects below this are noise"))
    return out


# ---------------------------------------------------------------------------------------------- runner
def run_audit(make_ctl, bodies, seeds, out_dir: Path, *, groups=None, ablations=ABLATIONS, max_s=30.0,
              threshold_m=0.1, expect_irrelevant=(), deploy=None, seed=0) -> dict:
    """make_ctl(seed) -> a fresh controller (LatentLeggedController or BCController). Writes rows/<cond>.jsonl and
    summary.json in out_dir."""
    from rrp.harness.eval.legged_latent_eval import run_episode
    groups = [GROUPS[g] for g in (groups or GROUPS)]
    out_dir = Path(out_dir)
    (out_dir / "rows").mkdir(parents=True, exist_ok=True)
    bss = getattr(deploy, "base_state_source", "truth_noise")
    kw = {} if deploy is None else dict(deploy=deploy)
    rng = np.random.default_rng(seed)
    results = {}
    for body in bodies:
        bank = ValueBank()

        def episodes(cond, setup):
            rows = []
            with open(out_dir / "rows" / f"{body}_{cond}.jsonl", "w") as f:
                for sd in seeds:
                    ctl = make_ctl(sd)
                    setup(ctl, sd)
                    row, _ = run_episode(ctl, body, sd, max_s, **kw)
                    row["audit"] = dict(version=AUDIT_VERSION, condition=cond)
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
            return rows

        def rec_setup(ctl, sd):
            ctl.ctx_transform, ctl.local_transform = bank.recorder(sd)
        base = episodes("baseline", rec_setup)
        ident = episodes("identity", lambda ctl, sd: None)
        results[(f"identity", "none")] = compare(base, ident, threshold_m)
        for g in groups:
            for ab in ablations:
                def setup(ctl, sd, g=g, ab=ab):
                    ctl.ctx_transform, ctl.local_transform = make_transforms(g, ab, bank, sd, rng)
                results[(g.name, ab)] = compare(base, episodes(f"{g.name}_{ab}", setup), threshold_m)
    summ = summarize(results, threshold_m=threshold_m, expect_irrelevant=expect_irrelevant, base_state_source=bss)
    summ.update(bodies=list(bodies), seeds=[int(s) for s in seeds], max_s=max_s, ablations=list(ablations),
                deploy=None if deploy is None else deploy.record())
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


# ---------------------------------------------------------------------------------------------- static / dynamic checks
PRIVILEGED_NAMES = frozenset({"PrivilegedTruth", "base_pose_truth", "truth_predicate", "privileged_success",
                              "_held_truth", "object_poses", "object_entity_map", "event_completion_truth", "held_by",
                              "PrivateBus", "priv", "privileged"})
FREE_JOINT_ATTRS = frozenset({"qa", "da"})       # LeggedBinding free-joint addresses (true base pose / velocity)

# functions that compute deployable controller inputs -> declared exceptions {name: reason}
DEPLOYABLE_FUNCTIONS = {
    "rrp.policies.features.legged:public_context": {},
    "rrp.policies.features.legged:local_state": {},
    "rrp.envs.mujoco.legged:LeggedSession.observe": {},
    "rrp.envs.mujoco.legged:LeggedSession.estimate": {},
    "rrp.envs.mujoco.legged:LeggedSession._imu": {},
    "rrp.envs.mujoco.legged:LeggedSession._touch": {},
    "rrp.envs.mujoco.legged:LeggedSession._track": {},
    "rrp.envs.mujoco.legged:LeggedSession._estimator_tick": {},
    "rrp.envs.mujoco.legged:LeggedSession._sense": {
        "qa": "declared localization sensor = true base x, y, yaw + Gaussian noise (TRUTH_DERIVED; see ctx.speed/"
              "waypoint groups). base_state_source=estimator replaces its speed with bse-1"},
    "rrp.envs.mujoco.legged_core:LeggedBinding.imu": {
        "qa": "IMU quaternion == the framequat sensor (IMU site == root frame; tests/unit/test_legged.py)",
        "da": "IMU gyro == the gyro sensor (tests/unit/test_legged.py)"},
    "rrp.envs.mujoco.state_estimator:BaseStateEstimator.update": {},
    "rrp.envs.mujoco.state_estimator:LegKinematics.feet": {
        "qa": "writes the PRIVATE kinematics copy's free joint to the identity pose (never reads the sim state)"},
    "rrp.policies.legged:System0Adapter.dyn_batch": {},
    "rrp.policies.legged:LatentLeggedController._ctx": {},
}


def _function_source(qual: str):
    import importlib
    mod, _, path = qual.partition(":")
    obj = importlib.import_module(mod)
    for part in path.split("."):
        obj = getattr(obj, part)
    return textwrap.dedent(inspect.getsource(obj))


def static_check(functions: dict | None = None) -> dict:
    """AST scan: privileged names / free-joint attributes referenced by deployable-input functions.
    Returns {violations: [...], declared: [...]} (declared = exceptions listed with a reason)."""
    functions = DEPLOYABLE_FUNCTIONS if functions is None else functions
    viol, declared = [], []
    for qual, allow in functions.items():
        tree = ast.parse(_function_source(qual))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name) and node.id in PRIVILEGED_NAMES:
                name = node.id
            elif isinstance(node, ast.Attribute) and (node.attr in PRIVILEGED_NAMES or node.attr in FREE_JOINT_ATTRS):
                name = node.attr
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in PRIVILEGED_NAMES:
                name = node.value
            if name is None:
                continue
            rec = dict(function=qual, name=name, line=getattr(node, "lineno", None))
            if name in allow:
                declared.append(dict(rec, reason=allow[name]))
            else:
                viol.append(rec)
    dedup = lambda xs: [dict(t) for t in {tuple(sorted((k, str(v)) for k, v in x.items())) for x in xs}]
    return dict(version=AUDIT_VERSION, violations=sorted(dedup(viol), key=str), declared=sorted(dedup(declared), key=str),
                functions=sorted(functions))


def dynamic_tripwire(body: str = "hexapod6", seed: int = 0, base_state_source: str = "truth_noise") -> dict:
    """Compute every deployable input with the privileged accessors disabled; the observation must pass the public
    transport. Raises on a violation; returns what was exercised."""
    from rrp.core.channels import serialize_public
    from rrp.envs.mujoco.legged import LeggedSession, build_waypoint_contact
    from rrp.policies.features.legged import LeggedMorph, local_state, public_context
    sc = build_waypoint_contact(body, seed)
    from rrp.harness.eval.legged_latent_eval import default_tracker_kind
    s = LeggedSession(sc, tracker_kind=default_tracker_kind(body), seed=seed, base_state_source=base_state_source)
    s.reset()

    def trip(name):
        def f(*a, **k):
            raise AssertionError(f"deployable input read privileged accessor {name}")
        return f
    saved = {n: getattr(s, n) for n in ("base_pose_truth", "truth_predicate", "privileged_success", "_held_truth")}
    try:
        for n in saved:
            setattr(s, n, trip(n))
        s._sense()
        if s.base_estimator is not None:
            s._estimator_tick(0.02)
        obs = s.observe()
        serialize_public(obs)
        morph = LeggedMorph(s.model, s.binding, sc.robots[0].robot_spec.spec_hash)
        ctx = public_context(s, 0.0)
        loc = local_state(s, morph)
    finally:
        for n, f in saved.items():
            setattr(s, n, f)
    return dict(body=body, base_state_source=base_state_source, ctx_dim=int(ctx.shape[0]),
                local_dims=[int(x.shape[0]) for x in loc], observation_public=True)


# ---------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--flow")
    r.add_argument("--rep")
    r.add_argument("--realizer")
    r.add_argument("--bc", help="BC route (positive control)")
    r.add_argument("--bodies", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--groups", default=None, help="comma list (default: all): " + ",".join(GROUPS))
    r.add_argument("--ablations", default=",".join(ABLATIONS))
    r.add_argument("--expect-irrelevant", default="", help="comma list of groups declared irrelevant")
    r.add_argument("--threshold-m", type=float, default=0.1)
    r.add_argument("--max-s", type=float, default=30.0)
    r.add_argument("--nfe", type=int, default=8)
    r.add_argument("--base-state-source", default="truth_noise", choices=["truth_noise", "estimator"])
    r.add_argument("--out", required=True)
    sp.add_parser("static")
    t = sp.add_parser("tripwire")
    t.add_argument("--body", default="hexapod6")
    t.add_argument("--base-state-source", default="truth_noise")
    a = ap.parse_args(argv)
    if a.cmd == "static":
        res = static_check()
        print(json.dumps(res, indent=1))
        raise SystemExit(1 if res["violations"] else 0)
    if a.cmd == "tripwire":
        print(json.dumps(dynamic_tripwire(a.body, base_state_source=a.base_state_source), indent=1))
        return
    import torch
    from rrp.core.runs import parse_seed_spec
    from rrp.harness.eval.deploy_eval import DeployOptions
    from rrp.policies.legged import BCController, LatentLeggedController
    torch.set_num_threads(2)
    dev = torch.device("cpu")
    if a.bc:
        make = lambda sd: BCController(Path(a.bc), dev, nfe=a.nfe, seed=sd)
    else:
        make = lambda sd: LatentLeggedController(Path(a.flow) if a.flow else None, dev, nfe=a.nfe, seed=sd, rep=a.rep,
                                                 realizer=a.realizer)
    dep = DeployOptions(base_state_source=a.base_state_source)
    summ = run_audit(make, a.bodies.split(","), list(parse_seed_spec(a.seeds)), Path(a.out),
                     groups=a.groups.split(",") if a.groups else None, ablations=a.ablations.split(","),
                     max_s=a.max_s, threshold_m=a.threshold_m,
                     expect_irrelevant=[x for x in a.expect_irrelevant.split(",") if x],
                     deploy=None if dep.is_default() else dep)
    print(json.dumps(dict(flags=summ["flags"]), indent=1))


if __name__ == "__main__":
    main()
