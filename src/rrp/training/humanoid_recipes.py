"""Named humanoid tracker recipes for the GPU trainer rrp.training.warp_tracker_ppo (W13 P1b, D-138).

Same contract as rrp.training.tracker_recipes: a recipe is a dict of trainer option defaults (argparse dests); keys starting
with "_" are documentation. The resolved recipe (name, sha256 of its canonical JSON, options) goes into the actor meta.
All recipes: contact_v2, sourced torque limits (body-model default), gait_v2 rewards, from scratch (no warm start).
"""
from __future__ import annotations

import hashlib
import json

_TURN = "clearance_floor=-2,yaw_slip=-2,turn_step=2,turn_lin=1.5,sigma_ang=0.03"
_CLOCK = ",stance_cap=-1,ref_step=1.0,ref_gait=clock,ref_lift=1.0,ref_contact=0.5,limit_margin=-1.0,limit_margin_agg=max"
_GATE = dict(alpha_schedule="gated", alpha_warmup=300,
             alpha_advance="track_rel_err=0.5,fall_rate=0.25,slip_ratio=0.3",
             alpha_backoff="track_rel_err=0.65,fall_rate=0.4,slip_ratio=0.45")


def _clock(body: str, **kw) -> dict:
    d = dict(_what=f"{body} from scratch on the GPU: clock reference gait + turn terms + worst-joint limit margin, W8 command mix",
             _why="contact.md: warm-started humanoids shuffle / never lift the feet; D-126 #13 h1_clock_scratch recipe ported to the GPU "
                  "trainer (W13 P1b)",
             _gate="D-112 tracker gate + lab gate (no-fall 1.0, fwd >= 0.8, turn >= 0.5, slip < 0.15), measured in C MuJoCo",
             body=body, nworld=4096, iters=1500, horizon=24, hidden="512,256,128", init_std=0.6, lr=1e-3, max_lr=3e-3, seed=1,
             reward_set=_TURN + _CLOCK, cmd_mix="teacher", turn_frac=0.35, slow_frac=0.2, **_GATE)
    d.update(kw)
    return d


HUMANOID_RECIPES: dict[str, dict] = {
    "t1_clock_gpu": _clock("t1"),
    "g1_clock_gpu": _clock("g1", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-1.0"),
    "h1_clock_gpu": _clock("h1"),
    "op3_clock_gpu": _clock("op3"),
    "apollo_clock_gpu": _clock("apollo"),
    "adam_lite_clock_gpu": _clock("adam_lite"),
}
HUMANOID_RECIPES["shared_morph_v1"] = None      # built lazily (phum topology selection runs the generator)


def phum_groups(n_topologies: int = 4, per_topology: int = 16, nworld: int = 512, seed_range=(0, 20000)) -> list:
    """The n most frequent phum topologies among training seeds, each with its first `per_topology` seeds (deterministic)."""
    from collections import defaultdict
    from rrp.bodies.humanoid_gen import sample_params
    by = defaultdict(list)
    for s in range(*seed_range):
        by[sample_params(s).topology].append(s)
    tops = sorted(by, key=lambda t: (-len(by[t]), t))[:n_topologies]
    return [[[f"phum_{s}" for s in by[t][:per_topology]], nworld] for t in tops]


SHARED_POOL_V1 = ["t1", "g1", "h1", "op3", "apollo", "adam_lite"]     # talos: no sourced/author torque limits in talos_position.xml


def _shared(**kw) -> dict:
    d = dict(_what="ONE morphology-conditioned tracker (obs morph_v1, 14 canonical leg slots + static morphology context) over the "
                   "menagerie training pool and 4 phum topologies x 16 bodies; first humanoid transfer test (sealed zero-shot)",
             _why="D-138 P1c; per-body trackers do not transfer by construction",
             _gate="per pool body: the same C-MuJoCo tracker gate as the per-body trackers; sealed bodies evaluated once",
             nworld=0, iters=3000, horizon=24, hidden="512,512,256", init_std=0.6, lr=1e-3, max_lr=3e-3, seed=1,
             reward_set=_TURN + _CLOCK, cmd_mix="teacher", turn_frac=0.35, slow_frac=0.2, minibatches=8, **_GATE)
    d.update(kw)
    return d


def recipe_record(name_or_path: str) -> tuple[dict, dict]:
    if name_or_path in HUMANOID_RECIPES:
        rec = HUMANOID_RECIPES[name_or_path]
        if rec is None and name_or_path == "shared_morph_v1":
            rec = _shared(groups=[[[b], 512] for b in SHARED_POOL_V1] + phum_groups())
        raw = json.dumps(rec, sort_keys=True).encode()
        where = dict(name=name_or_path, module="rrp.training.humanoid_recipes")
    else:
        from pathlib import Path
        raw = Path(name_or_path).read_bytes()
        rec = json.loads(raw)
        where = dict(path=name_or_path)
    opts = {k: v for k, v in rec.items() if not k.startswith("_")}
    return opts, dict(where, sha256=hashlib.sha256(raw).hexdigest(), options=opts,
                      notes={k: v for k, v in rec.items() if k.startswith("_")})
