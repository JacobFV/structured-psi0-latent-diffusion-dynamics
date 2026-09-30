"""Named tracker-training recipes (D-126 #13 / #14 / #15): ready to launch, NOT run.

A recipe is a dict of rrp.harness.train.tracker_training option defaults (keys = argparse dests). Launch with
    python -m rrp.cli train tracker-cpu --recipe <name> [--out ...] [explicit overrides]
(`--recipe` also takes a JSON file path). The resolved recipe (name, sha256 of its canonical JSON, options) is stored in the
actor meta. Keys starting with "_" are documentation (cost, rationale, the decision/track reference) and are not options.
They live here as python dicts, not as JSON files.

Two registries, one module (one `_TURN`, one `recipe_record`): CPU_RECIPES for the CPU PPO trainer (`rrp train tracker-cpu`,
rrp.harness.train.tracker_training) and WARP_RECIPES for the GPU trainer (`rrp train tracker-warp`, rrp.harness.train.warp_tracker_ppo;
the W13 humanoid recipes: contact_v2, sourced torque limits, gait_v2 rewards, from scratch unless `init_shared`). A recipe's sha256 is
over its canonical JSON (the module and registry name are not part of it), so moving a recipe here does not change its recorded sha.
The steps expert's actor input carries the public terrain scan (D-146) by construction of its env; no recipe key is needed.

Every recipe keeps sourced torque limits (the body-model default, D-107), contact_v2 physics and gait_v2 rewards. Initial actors
are pinned by sha256 (`init_actor_sha256`; the trainer refuses a different file). Peer paths are relative to the peer code dir,
whose artifacts/ is the shared store.
"""
from __future__ import annotations

import hashlib
import json

_TURN = "clearance_floor=-2,yaw_slip=-2,turn_step=2,turn_lin=1.5,sigma_ang=0.03"

CPU_RECIPES: dict[str, dict] = {
    # ---------------------------------------------------------------- #13 h1: clock-driven reference gait FROM SCRATCH
    "h1_clock_scratch": dict(
        _what="h1 from scratch under sourced limits with a phase-clocked reference gait from the start (humanoid-gym style)",
        _why="contact.md 'h1: PARKED': 8 warm-started attempts never lifted the feet under turn/slow commands; the recommendation is "
             "a clock reference from the start with many envs instead of fine-tuning the shuffle-prone warm start (D-103, D-107)",
        _cost="~2048 envs (16 workers x 128); 6000 iters ~ 3e8 samples; peer CPU, many hours (measure it/s in the first 50 iters)",
        _gate="lead gate: no-fall 1.0, fwd >= 0.8, turn >= 0.5, slip < 0.15, joint margin >= 0.02; 50-episode fall checks",
        body="h1", contact="v2", reward="gait_v2", iters=6000, workers=16, envs=128, horizon=24, hidden="512,256,128",
        init_std=0.6, lr=1e-3, max_lr=3e-3, seed=1, ckpt_every=50,
        ref_gait="clock", ref_gait_mode="reward",
        reward_set=_TURN + ",stance_cap=-1,ref_step=1.0,ref_lift=1.0,ref_contact=0.5",
        limit_margin=-1.0, limit_margin_agg="max",
        slow_frac=0.3, turn_curriculum=0.5, turn_frac=0.35,
        alpha_schedule="gated", alpha_warmup=500,
        alpha_advance="track_rel_err=0.5,fall_rate=0.25,slip_ratio=0.3",
        alpha_backoff="track_rel_err=0.65,fall_rate=0.4,slip_ratio=0.45",
    ),
    # ---------------------------------------------------------------- #13 g1: yaw progress capped at the commanded rate
    "g1_yawcap_ft": dict(
        _what="fine-tune g1_src with the dense yaw-progress reward capped at 1.0x the command, no arc yaw bonus above 0.3 rad/s, "
              "and an explicit over-rotation penalty",
        _why="g1_src (0/50 falls) spun 2.4-4x faster than commanded (turn 3.11, arc 4.07, slip 0.37): turn_lin capped at 1.2 and "
             "applied to every arc (yaw_lin_all) rewarded over-rotation (contact.md 'g1 status'; D-114 g1_src stomps 4.09 BW)",
        _cost="800 iters x 4 workers x 48 envs (~ the g1_src run), peer CPU",
        body="g1", contact="v2", reward="gait_v2", iters=800, workers=4, envs=48, hidden="512,256,128", init_std=0.25,
        lr=1e-3, max_lr=1e-3, seed=1, torch_threads=1,
        init_actor="artifacts/runs/contact_g1_src/actor.pt",
        init_actor_sha256="f0a4daaa004da7cba9f3ccac1ff15daf02465608382c40452e1c245170d6d739",
        cmd_mix="teacher", ref_ff=0.08, turn_curriculum=1.0, turn_frac=0.3,
        reward_set=_TURN + ",stance_cap=-1,ref_step=1.0,yaw_lin_all=1,stand_vel=-4",
        yaw_progress_cap=1.0, yaw_lin_all_max=0.3, yaw_overshoot=-1.0,
        limit_margin=-1.0, limit_margin_agg="max",
        alpha_schedule="fixed:0.0",
    ),
    # ---------------------------------------------------------------- #13/#14 t1: joint turning + latency fine-tune
    "t1_turn_latency_ft": dict(
        _what="t1 (installed w8d) fine-tuned for turning and 0-30 ms latency together (actuator v1lat) under the W8 command mix",
        _why="D-103(3)/D-107(2): lat2 fixed latency but lost walk-and-turn yaw; w8d passes the waypoint check but fails fwd 0.72, "
             "arc no-fall 0.83 at 30 ms and joint margin -0.053 (D-114). This recipe trains turning, the teacher mix and latency "
             "jointly, with the worst-joint limit-margin hinge",
        _cost="1000 iters x 4 workers x 64 envs, peer CPU (w8d: 800 iters x 3 x 64)",
        body="t1", contact="v2", reward="gait_v2", iters=1000, workers=4, envs=64, hidden="512,256,128", init_std=0.2,
        lr=1e-3, max_lr=1e-3, seed=1, torch_threads=1,
        init_actor="artifacts/trackers/t1/contact_v2/actor.pt",
        init_actor_sha256="36e9146792743115878c34e0bbf7cc46ccb5419417921358da3658c8377fc591",
        actuator="v1lat", cmd_mix="teacher:0.10", turn_curriculum=1.0, turn_frac=0.25,
        reward_set=_TURN + ",slip=-2.0,yaw_lin_all=1",
        yaw_progress_cap=1.2, limit_margin=-1.0, limit_margin_agg="max",
        alpha_schedule="fixed:1.0",
    ),
    # ---------------------------------------------------------------- #15 terrain curriculum (anymal_c: the D-112 body)
    "anymal_c_terrain_ft": dict(
        _what="anymal_c (installed contact_v2) fine-tuned with the gated bumps_v1 terrain curriculum up to 12 cm",
        _why="D-108/D-112: every learned anymal_c route breaks at 8 cm terrain (BC 5 cm); terrain was eval-only (backlog #9)",
        _cost="1500 iters x 8 workers x 64 envs, peer CPU; then re-run the D-112 robustness grid on the new tracker",
        body="anymal_c", contact="v2", reward="gait_v2", iters=1500, workers=8, envs=64, init_std=0.3, lr=1e-3, max_lr=1e-3, seed=1,
        init_actor="artifacts/trackers/anymal_c/contact_v2/actor.pt",
        init_actor_sha256="2a16532bbd07f7abc662bdda10df6bfb70fca2ec11d59f9567142e92a2f22d95",
        reward_set="clearance_floor=-6", alpha_schedule="fixed:1.0",
        terrain_curriculum="gated", terrain_amp_max=0.12, terrain_step=0.1, terrain_warmup=100,
    ),
}


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


WARP_RECIPES: dict[str, dict] = {
    "t1_clock_gpu": _clock("t1"),
    "g1_clock_gpu": _clock("g1", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"),
    "h1_clock_gpu": _clock("h1"),
    # r1 (t1/h1, 2026-09-29 00:00) over-rotated in pure turns (window turn ratio 1.5-2.1): later recipes add the g1 fix
    "op3_clock_gpu": _clock("op3", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"),
    "apollo_clock_gpu": _clock("apollo", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"),
    "adam_lite_clock_gpu": _clock("adam_lite", reward_set=_TURN + _CLOCK + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"),
}
WARP_RECIPES["shared_morph_v1"] = None      # built lazily (phum topology selection runs the generator)


def phum_groups(n_topologies: int = 4, per_topology: int = 16, nworld: int = 512, seed_range=(0, 20000), min_arm_dof: int = 0) -> list:
    """The n most frequent phum topologies among training seeds, each with its first `per_topology` seeds (deterministic).
    `min_arm_dof` > 0 keeps only topologies whose arms have at least that many joints (the wholebody recipe: every group then has
    an upper body to randomise)."""
    from collections import defaultdict
    from rrp.bodies.humanoid_gen import sample_params
    by = defaultdict(list)
    for s in range(*seed_range):
        p = sample_params(s)
        if p.arm_dof < min_arm_dof:
            continue
        by[p.topology].append(s)
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


def _steps(body: str, **kw) -> dict:
    d = dict(_what=f"{body} h_steps privileged expert (height scan) from scratch; scripted heading command; step-height curriculum",
             _gate="P2: expert success >= 0.9 at h_frac 0.10-0.30 per body in C MuJoCo, dataset gates",
             body=body, task="steps", nworld=4096, iters=2000, horizon=24, hidden="512,256,128", init_std=0.6, lr=1e-3,
             max_lr=3e-3, seed=1, reward_set=_TURN + _CLOCK, cmd_mix="default", turn_frac=0.0, slow_frac=0.0, **_GATE)
    d.update(kw)
    return d


WARP_RECIPES["t1_steps_gpu"] = _steps("t1")
WARP_RECIPES["h1_steps_gpu"] = _steps("h1")


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


WARP_RECIPES["h1_clock_gpu_r2"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r1/actor_iter649.pt")
# r2 (above) FAILED at iter 282 (window fall 0.95, track err 1.86): trainer bug, the warm-start normaliser was replaced by
# the first batch (count 1e-4); fixed (count 1e6). r2b = same with the fix and alpha fixed at 0.5 (less abrupt).
WARP_RECIPES["h1_clock_gpu_r2b"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r1/actor_iter649.pt", alpha_schedule="fixed:0.5")
WARP_RECIPES["t1_clock_gpu_r2"] = _ft("t1", "artifacts/runs/humanoid_p1b_t1_r1/actor_r1final.pt")

# r3 (2026-09-29): h1 r2b in C MuJoCo (full self-collision): turn fixed (1.10) but push no-fall 0.5, waypoint 15/20 fell, joint
# margin -0.02, peak force 3.75 BW -- trained with no_self_collision (legs pass through each other). r3 = r2b + the new default
# GPU adaptation leg_cross_collision (code default since this commit) + joint-limit hinge x2.
WARP_RECIPES["h1_clock_gpu_r3"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r1/actor_iter649.pt", alpha_schedule="fixed:0.5",
                                          reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0")
                                          + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0")
WARP_RECIPES["t1_clock_gpu_r2"] = _ft("t1", "artifacts/runs/humanoid_p1b_t1_r1/actor_r1final.pt", alpha_schedule="fixed:0.5",
                                          reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0")
                                          + ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0")

# t1 r1 in C MuJoCo (full self-collision): forward trial no-fall 0.2, fwd 0.0, peak force 5.6 BW -- the r1 gait needs legs to
# pass through each other; with leg_cross_collision the r1 actor falls at ~1 s (t1 r2 fine-tune: every episode, ep_len 57).
# r1/r2 are void; t1 v2 = the r1 clock recipe from scratch under leg_cross_collision + the yaw/standing/limit fixes.
_FIX = ",yaw_progress_cap=1.0,yaw_overshoot=-2.0,stand_vel=-3.0"
WARP_RECIPES["t1_clock_gpu_v2"] = _clock("t1", reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX)
WARP_RECIPES["g1_clock_gpu_v2"] = _clock("g1", reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX)

# r4 (2026-09-29): h1 r3 in C MuJoCo: no-fall 1.0 on every trial incl. push and in-range robustness, fwd 1.08, turn 0.96,
# slip 0.018, CoT 0.51, joint margin 0.032; FAILS peak foot force 3.72 BW; waypoint 0/20 = walking and turning fine but the
# halt never completes: under a zero command h1 keeps stepping and drifts ~0.06 m/s with yaw (archived diagnostic script humanoid_waypoint_diag.py).
# r4 = fine-tune r3: ~25% zero commands, stronger standing terms, explicit touchdown-impact penalty.
_STAND = ",stand_contact=2.0,stand_still=-1.0,impact=-0.5"
WARP_RECIPES["h1_clock_gpu_r4"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r3/actor_r3final.pt", alpha_schedule="fixed:0.5",
                                          teacher_stop=0.3,
                                          reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX + _STAND)

# v3 from-scratch recipes for the remaining pool bodies (all lessons: leg_cross_collision default, yaw fixes, standing, impact)
for _b in ("g1", "op3", "apollo", "adam_lite"):
    WARP_RECIPES[f"{_b}_clock_gpu_v3"] = _clock(_b, teacher_stop=0.3, reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0",
                                                                                                       "limit_margin=-2.0") + _FIX + _STAND)
WARP_RECIPES["h1_steps_gpu_v1"] = _steps("h1", init_shared="artifacts/runs/humanoid_p1b_h1_r4/actor.pt", init_std=0.3, iters=1500,
                                             reward_set=_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-2.0") + _FIX,
                                             alpha_schedule="fixed:0.5")


def _shared_v1():
    # profiling (2026-09-29): per-group overhead ~28 ms/tick regardless of worlds, physics unsaturated at 512 worlds ->
    # fewer, larger groups: 6 menagerie x 1024 + 2 phum topologies x 32 bodies x 2048 worlds
    return _shared(groups=[[[b], 1024] for b in SHARED_POOL_V1] + phum_groups(2, 32, 2048), minibatches=16)


def _shared_v2():
    # all P1b lessons: leg_cross_collision (code default), clock gate, target margin, landing-velocity penalty, R5 reward set
    return _shared(groups=[[[b], 1024] for b in SHARED_POOL_V1] + phum_groups(2, 32, 2048), minibatches=16, teacher_stop=0.3,
                   clock_gate=True, target_margin=0.03, land_vel=-2.0,
                   reward_set=_R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0", "limit_margin=-8.0"))


WARP_RECIPES["shared_morph_v2"] = None      # lazy: phum selection runs the generator

# r5 (2026-09-29): h1 r4 in C MuJoCo: no-fall 1.0, fwd 1.06, turn 1.06, slip 0.005, but peak force 4.08 BW, joint margin -0.005,
# arc yaw ~0 (0.004 vs 0.24 commanded) and under a zero command it still steps (duty 0.5, 1.25 Hz) -> halts fail (0/20).
# r5 = fine-tune of r3 with the CLOCK GATE (clock inputs zeroed at zero command: an explicit stand mode), turn progress
# on arcs up to 0.5 rad/s (yaw_lin_all), impact -2, joint-limit hinge -4.
_R5 = (_TURN + _CLOCK.replace("limit_margin=-1.0", "limit_margin=-4.0") + _FIX + _STAND.replace("impact=-0.5", "impact=-2.0")
       + ",yaw_lin_all=1,yaw_lin_all_max=0.5")
WARP_RECIPES["h1_clock_gpu_r5"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r3/actor_r3final.pt", alpha_schedule="fixed:0.5",
                                          teacher_stop=0.3, clock_gate=True, reward_set=_R5)
for _b in ("g1", "op3", "apollo", "adam_lite", "t1"):
    WARP_RECIPES[f"{_b}_clock_gpu_v4"] = _clock(_b, teacher_stop=0.3, clock_gate=True, reward_set=_R5)

# r6 (last h1 attempt; stop rule): h1 r5 in C MuJoCo: no-fall 1.0 everywhere, fwd 1.04, turn 0.98, arc yaw 0.236/0.24, slip
# 0.015, CoT 0.65, WAYPOINT 13/20 (first halts; 0 falls); fails peak force 3.37 BW (walking trials 3.1-3.4) and joint margin
# 0.0017 (turn_fast only). r6 = fine-tune r5: impact -4, joint-limit hinge -8, 400 iters.
WARP_RECIPES["h1_clock_gpu_r6"] = _ft("h1", "artifacts/runs/humanoid_p1b_h1_r5/actor_r5final.pt", alpha_schedule="fixed:0.5",
                                          teacher_stop=0.3, clock_gate=True, iters=400,
                                          reward_set=_R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0",
                                                                                                        "limit_margin=-8.0"))

# t1 v2 (from scratch, leg_cross_collision, no clock gate) in C MuJoCo: no-fall 1.0 everywhere, fwd 0.93, turn 1.01, slip 0.071,
# CoT 0.45 (installed w8d: 2.13), joint margin 0.0175, peak force 3.80 BW, waypoint 0/20 (same halt defect as h1 r3/r4).
# t1 v2ft = the h1 r5->r6 fixes applied to t1 v2 (clock gate, impact -4, limit hinge -8), 600 iters.
WARP_RECIPES["t1_clock_gpu_v2ft"] = _ft("t1", "artifacts/runs/humanoid_p1b_t1_v2/actor_v2final.pt", alpha_schedule="fixed:0.5",
                                            teacher_stop=0.3, clock_gate=True,
                                            reward_set=_R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0",
                                                                                                          "limit_margin=-8.0"))

# h1 r6 (final h1 attempt) in C MuJoCo: WAYPOINT 20/20 (0 falls), no-fall 1.0 everywhere, fwd 1.11, turn 0.82, slip 0.012,
# CoT 0.71; D-112 still FAILS on peak force 3.74 BW and joint margin -0.0014: heavier impact / limit-hinge weights did not bite.
# Structural fixes for the next attempts: target_margin 0.03 (targets clipped inside the range, deployed identically) and a
# touchdown-velocity penalty (land_vel). t1 v2ft2 = t1 v2ft + both.
def _ft2(body, init, **kw):
    return _ft(body, init, alpha_schedule="fixed:0.5", teacher_stop=0.3, clock_gate=True, target_margin=0.03, land_vel=-2.0,
               reward_set=_R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0", "limit_margin=-8.0"), **kw)


WARP_RECIPES["t1_clock_gpu_v2ft2"] = _ft2("t1", "artifacts/runs/humanoid_p1b_t1_v2ft/actor_v2ftfinal.pt")
WARP_RECIPES["g1_clock_gpu_v4ft"] = _ft2("g1", "artifacts/runs/humanoid_p1b_g1_v4/actor_v4final.pt")

WARP_RECIPES["h1_steps_gpu_v2"] = _steps("h1", init_shared="artifacts/runs/humanoid_p1b_h1_r6/actor_r6final.pt", init_std=0.3,
                                             iters=1500, clock_gate=True, alpha_schedule="fixed:0.5",
                                             reward_set=_R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0",
                                                                                                          "limit_margin=-8.0"))

# t1 v2ft2 in C MuJoCo: waypoint 19/20 (0 falls), no-fall 1.0, fwd 1.09, turn 1.18, slip 0.073, CoT 0.59, joint margin 0.022
# (target_margin fixed it); D-112 fails ONLY peak force 3.63 BW (land_vel did not bound it). Deviation from the stop rule
# (recorded): one more t1 attempt within the sample budget, v2ft3 = v2ft2 + per-tick force cap (2.5 BW, weight -2).
WARP_RECIPES["t1_clock_gpu_v2ft3"] = _ft2("t1", "artifacts/runs/humanoid_p1b_t1_v2ft2/actor_v2ft2final.pt", force_cap=-2.0,
                                              force_cap_bw=2.5)

# g1 v4ft (target_margin 0.03 + land_vel) in C MuJoCo: no-fall 1.0 on trials, fwd 0.96, turn 0.86, slip 0.016, CoT 0.51, force
# 2.12 BW (pass); joint margin 0.018 (fail: the PD overshoots the 3% target band by ~1% of the range); waypoint 15/20, 5 falls.
# g1 v4ft2 = last g1 attempt: target_margin 0.05, 400 iters.
WARP_RECIPES["g1_clock_gpu_v4ft2"] = _ft("g1", "artifacts/runs/humanoid_p1b_g1_v4ft/actor_v4ftfinal.pt", alpha_schedule="fixed:0.5",
                                             teacher_stop=0.3, clock_gate=True, target_margin=0.05, land_vel=-2.0, iters=400,
                                             reward_set=_R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0",
                                                                                                          "limit_margin=-8.0"))

# t1 v2ft3 (+ force cap 2.5 BW): waypoint 20/20, 0 falls; no-fall 1.0, fwd 1.08, turn 0.95, slip 0.054, CoT 0.53, peak force
# 2.72 BW (PASS now); joint margin 0.0094 (fail; regressed from 0.022). t1 v2ft4 = + target_margin 0.05 (as g1 v4ft2),
# 400 iters; within the t1 sample budget (~3.3e8 of 6e8), attempt count deviation recorded.
WARP_RECIPES["t1_clock_gpu_v2ft4"] = _ft("t1", "artifacts/runs/humanoid_p1b_t1_v2ft3/actor_v2ft3final.pt", alpha_schedule="fixed:0.5",
                                             teacher_stop=0.3, clock_gate=True, target_margin=0.05, land_vel=-2.0, force_cap=-2.0,
                                             force_cap_bw=2.5, iters=400,
                                             reward_set=_R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0",
                                                                                                          "limit_margin=-8.0"))

# P2 L1 h_gap_sidestep expert (pre-declared stop rule: <= 2 reward designs, <= 3e8 samples per body)
WARP_RECIPES["h1_gap_gpu_v1"] = _steps("h1", task="gap", init_shared="artifacts/runs/humanoid_p1b_h1_r6/actor_r6final.pt",
                                           init_std=0.3, iters=2000, clock_gate=True, alpha_schedule="fixed:0.5", episode_s=20.0,
                                           level_up=0.6, reward_set=_R5.replace("impact=-2.0", "impact=-4.0")
                                           .replace("limit_margin=-4.0", "limit_margin=-8.0"))
WARP_RECIPES["t1_gap_gpu_v1"] = dict(WARP_RECIPES["h1_gap_gpu_v1"], body="t1",
                                         init_shared="artifacts/runs/humanoid_p1b_t1_v2ft4/actor.pt", target_margin=0.05,
                                         force_cap=-2.0, force_cap_bw=2.5, land_vel=-2.0)


# U1 wholebody trackers (`*_ub`; docs/architecture.md 14.4): the same gait recipe with the upper body (arms, waist, head) driven
# by RANDOM slew-limited targets and a random hand payload (<= 8% of the robot mass) so the legs learn to balance under
# manipulation-like disturbances; the actor also takes the upper joint state (obs tail, actor meta upper_obs). Warm starts pad
# the new input columns with zeros (init_shared), so a legs-only gait is the exact starting policy. Where no accepted gait exists
# (op3, apollo, adam_lite) the v4 clock recipe runs from scratch. NOT run: training is paused; the gate is the D-112 tracker gate
# measured with the upper body held AND moving, in C MuJoCo (LeggedSession control="wholebody" reads the same obs block).
# HS2 (D-146 R2): the upper-body amplitude and the payload RAMP from a small start (upper_amp0, payload 0) to the full values over the
# first `upper_ramp` share of the iterations (`upper_ramp_value`), so the legs learn the gait before the full disturbance
# (the flat full-amplitude start is what the first wholebody attempts lacked).
_UB = dict(upper_body=True, upper_amp=0.4, upper_speed=1.5, payload_frac=0.08, upper_amp0=0.1, upper_ramp=0.3)
_R6 = _R5.replace("impact=-2.0", "impact=-4.0").replace("limit_margin=-4.0", "limit_margin=-8.0")


def _ub_ft(body: str, init: str, **kw) -> dict:
    d = _ft(body, init, alpha_schedule="fixed:0.5", teacher_stop=0.3, clock_gate=True, land_vel=-2.0, iters=800, init_std=0.3,
            reward_set=_R6, **_UB, **kw)
    d["_what"] = f"{body}: the latest gait fine-tune with random upper-body targets + payload; upper joint state in the actor input"
    return d


WARP_RECIPES["t1_clock_gpu_ub"] = _ub_ft("t1", "artifacts/runs/humanoid_p1b_t1_v2ft4/actor.pt", target_margin=0.05, force_cap=-2.0,
                                         force_cap_bw=2.5)
WARP_RECIPES["g1_clock_gpu_ub"] = _ub_ft("g1", "artifacts/runs/humanoid_p1b_g1_v4ft/actor_v4ftfinal.pt", target_margin=0.05)
WARP_RECIPES["h1_clock_gpu_ub"] = _ub_ft("h1", "artifacts/runs/humanoid_p1b_h1_r6/actor_r6final.pt", target_margin=0.03)
for _b in ("op3", "apollo", "adam_lite"):
    WARP_RECIPES[f"{_b}_clock_gpu_ub"] = _clock(_b, teacher_stop=0.3, clock_gate=True, reward_set=_R5, **_UB)


# HS2 steps + upper body: the h_steps expert (public terrain scan in the actor) trained with random upper-body targets + payload on
# the same ramp, from scratch (no `*_steps_gpu` actor has the upper input block). Smoke target of HS2; the real runs are HR's.
for _b in ("t1", "g1", "h1"):
    WARP_RECIPES[f"{_b}_steps_ub"] = _steps(_b, iters=2000, clock_gate=True, alpha_schedule="fixed:0.5", reward_set=_R6, **_UB)


def _shared_ub():
    # HS2: ONE morphology-conditioned wholebody tracker (obs morph_v2 = morph_v1 + the canonical upper block). Same gait recipe as
    # shared_morph_v2 (P1b lessons) with random upper-body targets + payload on every group that has an upper body. The groups are
    # the menagerie pool (six humanoids; arms/waist/head) and the two most frequent phum topologies that HAVE arms; 1024 / 2048
    # worlds as shared_morph_v2. The amplitude / payload ramp is the `_UB` one.
    d = _shared_v2()
    d.update(groups=[[[b], 1024] for b in SHARED_POOL_V1] + phum_groups(2, 32, 2048, min_arm_dof=1), **_UB)
    d["_what"] = ("ONE morphology-conditioned wholebody tracker (obs morph_v2: legs + padded upper joint state + static slot "
                  "descriptors) over the menagerie pool and 2 phum topologies with arms; random upper-body targets + payload on "
                  "a ramp")
    d["_why"] = "D-146 R2 HS2: the shared tracker had no upper-body block, so wholebody (`*_ub`) existed per body only"
    d["_gate"] = "per pool body: the D-112 tracker gate with the upper body held AND moving, in C MuJoCo; sealed bodies once"
    return d


WARP_RECIPES["shared_morph_ub"] = None      # lazy: phum selection runs the generator
_LAZY = dict(shared_morph_v1=_shared_v1, shared_morph_v2=_shared_v2, shared_morph_ub=_shared_ub)
assert not set(CPU_RECIPES) & set(WARP_RECIPES)
RECIPES: dict[str, dict | None] = {**CPU_RECIPES, **WARP_RECIPES}


def training_bodies(args) -> list[str]:
    """Every body a tracker-training run trains on: --body, or all keys of --groups (JSON string or parsed list)."""
    groups = getattr(args, "groups", None)
    if groups:
        groups = json.loads(groups) if isinstance(groups, str) else groups
        return sorted({k for ks, _ in groups for k in ([ks] if isinstance(ks, str) else ks)})
    return [args.body]


def assert_trainable(args, what: str = "tracker training") -> None:
    """Sealed-split guard (core.sealed): a tracker trains on no sealed body and no phum seed outside the declared training range.
    Raises SealedSplitError BEFORE any simulation is built."""
    from rrp.core.sealed import SealedSplit
    SealedSplit.load().assert_train_allowed(training_bodies(args), what=what)


def recipe_record(name_or_path: str) -> tuple[dict, dict]:
    """(options, record) for a recipe name (either registry) or a JSON file path. record = name/path, sha256 of the canonical
    JSON, options, notes."""
    if name_or_path in RECIPES:
        rec = RECIPES[name_or_path]
        if rec is None:                     # lazy (phum topology selection runs the generator)
            rec = _LAZY[name_or_path]()
        raw = json.dumps(rec, sort_keys=True).encode()
        where = dict(name=name_or_path, module="rrp.harness.train.tracker_recipes")
    else:
        from pathlib import Path
        raw = Path(name_or_path).read_bytes()
        rec = json.loads(raw)
        where = dict(path=name_or_path)
    opts = {k: v for k, v in rec.items() if not k.startswith("_")}
    return opts, dict(where, sha256=hashlib.sha256(raw).hexdigest(), options=opts,
                      notes={k: v for k, v in rec.items() if k.startswith("_")})
