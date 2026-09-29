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
    "g1_clock_gpu": _clock("g1", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"),
    "h1_clock_gpu": _clock("h1"),
    # r1 (t1/h1, 2026-09-29 00:00) over-rotated in pure turns (window turn ratio 1.5-2.1): later recipes add the g1 fix
    "op3_clock_gpu": _clock("op3", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"),
    "apollo_clock_gpu": _clock("apollo", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"),
    "adam_lite_clock_gpu": _clock("adam_lite", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"),
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
             reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0", cmd_mix="teacher",
             turn_frac=0.35, slow_frac=0.2, minibatches=8, **_GATE)
    d.update(kw)
    return d


def recipe_record(name_or_path: str) -> tuple[dict, dict]:
    if name_or_path in HUMANOID_RECIPES:
        rec = HUMANOID_RECIPES[name_or_path]
        if rec is None and name_or_path == "shared_morph_v2":
            rec = _shared_v2()
        if rec is None and name_or_path == "shared_morph_v1":
            # profiling (2026-09-29): per-group overhead ~28 ms/tick regardless of worlds, physics unsaturated at 512 worlds ->
            # fewer, larger groups: 6 menagerie x 1024 + 2 phum topologies x 32 bodies x 2048 worlds
            rec = _shared(groups=[[[b], 1024] for b in SHARED_POOL_V1] + phum_groups(2, 32, 2048), minibatches=16)
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


def _steps(body: str, **kw) -> dict:
    d = dict(_what=f"{body} h_steps privileged expert (height scan) from scratch; scripted heading command; step-height curriculum",
             _gate="P2: expert success >= 0.9 at h_frac 0.10-0.30 per body in C MuJoCo, dataset gates",
             body=body, task="steps", nworld=4096, iters=2000, horizon=24, hidden="512,256,128", init_std=0.6, lr=1e-3,
             max_lr=3e-3, seed=1, reward_set=_TURN + _CLOCK, cmd_mix="default", turn_frac=0.0, slow_frac=0.0, **_GATE)
    d.update(kw)
    return d


HUMANOID_RECIPES["t1_steps_gpu"] = _steps("t1")
HUMANOID_RECIPES["h1_steps_gpu"] = _steps("h1")


# ---- r2 fine-tunes (2026-09-29): r1 h1 interim (C MuJoCo) no-fall 1.0, fwd 0.89, slip 0.05, but pure-turn ratio 2.1 (turn_lin
# rewarded over-rotation while the sharp yaw kernel gave ~0), stand drift -0.06 m/s (waypoint halts 0/20) and peak foot force
# 3.43 BW (> 3.0). Fix: yaw cap 1.0 + overshoot penalty, stronger standing velocity term, alpha fixed at 1 (natural terms
# incl. impact at full weight), warm start from the r1 actor (--init-shared path, same dims).
def _ft(body: str, init: str, **kw) -> dict:
    d = dict(_what=f"{body} r2 fine-tune of r1: yaw overshoot fix, standing drift, alpha 1 (impact/power/smooth at max)",
             body=body, nworld=4096, iters=600, horizon=24, hidden="512,256,128", init_std=0.25, lr=5e-4, max_lr=1e-3, seed=2,
             init_shared=init, reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0",
             cmd_mix="teacher", turn_frac=0.35, slow_frac=0.2, alpha_schedule="fixed:1.0")
    d.update(kw)
    return d


HUMANOID_RECIPES["h1_clock_gpu_r2"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r1/actor_iter649.pt")
# r2 (above) FAILED at iter 282 (window fall 0.95, track err 1.86): trainer bug, the warm-start normaliser was replaced by
# the first batch (count 1e-4); fixed (count 1e6). r2b = same with the fix and alpha fixed at 0.5 (less abrupt).
HUMANOID_RECIPES["h1_clock_gpu_r2b"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r1/actor_iter649.pt", alpha_schedule="fixed:0.5")
HUMANOID_RECIPES["t1_clock_gpu_r2"] = _ft("t1", "artifacts/runs/humanoid_p1b_t1_r1/actor_r1final.pt")

# r3 (2026-09-29): h1 r2b in C MuJoCo (full self-collision): turn fixed (1.10) but push no-fall 0.5, waypoint 15/20 fell, joint
# margin -0.02, peak force 3.75 BW -- trained with no_self_collision (legs pass through each other). r3 = r2b + the new default
# GPU adaptation leg_cross_collision (code default since this commit) + joint-limit hinge x2.
HUMANOID_RECIPES["h1_clock_gpu_r3"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r1/actor_iter649.pt", alpha_schedule="fixed:0.5",
                                          reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0")
                                          + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0")
HUMANOID_RECIPES["t1_clock_gpu_r2"] = _ft("t1", "artifacts/runs/humanoid_p1b_t1_r1/actor_r1final.pt", alpha_schedule="fixed:0.5",
                                          reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0")
                                          + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0")

# t1 r1 in C MuJoCo (full self-collision): forward trial no-fall 0.2, fwd 0.0, peak force 5.6 BW -- the r1 gait needs legs to
# pass through each other; with leg_cross_collision the r1 actor falls at ~1 s (t1 r2 fine-tune: every episode, ep_len 57).
# r1/r2 are void; t1 v2 = the r1 clock recipe from scratch under leg_cross_collision + the yaw/standing/limit fixes.
_FIX = ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"
HUMANOID_RECIPES["t1_clock_gpu_v2"] = _clock("t1", reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX)
HUMANOID_RECIPES["g1_clock_gpu_v2"] = _clock("g1", reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX)

# r4 (2026-09-29): h1 r3 in C MuJoCo: no-fall 1.0 on every trial incl. push and in-range robustness, fwd 1.08, turn 0.96,
# slip 0.018, CoT 0.51, joint margin 0.032; FAILS peak foot force 3.72 BW; waypoint 0/20 = walking and turning fine but the
# halt never completes: under a zero command h1 keeps stepping and drifts ~0.06 m/s with yaw (scripts/humanoid_waypoint_diag.py).
# r4 = fine-tune r3: ~25% zero commands, stronger standing terms, explicit touchdown-impact penalty.
_STAND = ",stand_contact=2.0,stand_still=-1.0,impact=-0.5"
HUMANOID_RECIPES["h1_clock_gpu_r4"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r3/actor_r3final.pt", alpha_schedule="fixed:0.5",
                                          teacher_stop=0.3,
                                          reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX + _STAND)

# v3 from-scratch recipes for the remaining pool bodies (all lessons: leg_cross_collision default, yaw fixes, standing, impact)
for _b in ("g1", "op3", "apollo", "adam_lite"):
    HUMANOID_RECIPES[f"{_b}_clock_gpu_v3"] = _clock(_b, teacher_stop=0.3, reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0",
                                                                                                       "limit_margin=-2.0") + _FIX + _STAND)
HUMANOID_RECIPES["h1_steps_gpu_v1"] = _steps("h1", init_shared="artifacts/runs/humanoid_p1b_h1_r4/actor.pt", init_std=0.3, iters=1500,
                                             reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX,
                                             alpha_schedule="fixed:0.5")


def _shared_v2():
    return _shared(groups=[[[b], 1024] for b in SHARED_POOL_V1] + phum_groups(2, 32, 2048), minibatches=16, teacher_stop=0.3,
                   reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX + _STAND)


HUMANOID_RECIPES["shared_morph_v2"] = None      # lazy: phum selection runs the generator

# r5 (2026-09-29): h1 r4 in C MuJoCo: no-fall 1.0, fwd 1.06, turn 1.06, slip 0.005, but peak force 4.08 BW, joint margin -0.005,
# arc yaw ~0 (0.004 vs 0.24 commanded) and under a zero command it still steps (duty 0.5, 1.25 Hz) -> halts fail (0/20).
# r5 = fine-tune of r3 with the CLOCK GATE (clock inputs zeroed at zero command: an explicit stand mode), turn progress
# on arcs up to 0.5 rad/s (yaw_lin_all), impact -2, joint-limit hinge -4.
_R5 = (_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-4.0") + _FIX + _STAND.replace("impact=-0.5", "impact=-2.0")
       + ",yaw_lin_all=1,yaw_lin_all_max=0.5")
HUMANOID_RECIPES["h1_clock_gpu_r5"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r3/actor_r3final.pt", alpha_schedule="fixed:0.5",
                                          teacher_stop=0.3, clock_gate=True, reward_set=_R5)
for _b in ("g1", "op3", "apollo", "adam_lite", "t1"):
    HUMANOID_RECIPES[f"{_b}_clock_gpu_v4"] = _clock(_b, teacher_stop=0.3, clock_gate=True, reward_set=_R5)

# r6 (last h1 attempt; stop rule): h1 r5 in C MuJoCo: no-fall 1.0 everywhere, fwd 1.04, turn 0.98, arc yaw 0.236/0.24, slip
# 0.015, CoT 0.65, WAYPOINT 13/20 (first halts; 0 falls); fails peak force 3.37 BW (walking trials 3.1-3.4) and joint margin
# 0.0017 (turn_fast only). r6 = fine-tune r5: impact -4, joint-limit hinge -8, 400 iters.
HUMANOID_RECIPES["h1_clock_gpu_r6"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r5/actor_r5final.pt", alpha_schedule="fixed:0.5",
                                          teacher_stop=0.3, clock_gate=True, iters=400,
                                          reward_set=_R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0",
                                                                                                        "limit_margin=-8.0"))
