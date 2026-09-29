"""Named tracker-training recipes (D-126 #13 / #14 / #15): ready to launch, NOT run.

A recipe is a dict of rrp.training.tracker_training option defaults (keys = argparse dests). Launch with
    python -m rrp.harness.train.tracker_training --recipe <name> [--out ...] [explicit overrides]
(`--recipe` also takes a JSON file path). The resolved recipe (name, sha256 of its canonical JSON, options) is stored in the
actor meta. Keys starting with "_" are documentation (cost, rationale, the decision/track reference) and are not options.
They live here, not under configs/ (every configs/**/*.json must be a RunConfig; tests/unit/test_runconfig.py).

Every recipe keeps sourced torque limits (the body-model default, D-107), contact_v2 physics and gait_v2 rewards. Initial actors
are pinned by sha256 (`init_actor_sha256`; the trainer refuses a different file). Peer paths are relative to the peer code dir,
whose artifacts/ is the shared store.
"""
from __future__ import annotations

import hashlib
import json

_TURN = "clearance_floor=-2,yaw_slip=-2,turn_step=2,turn_lin=1.5,sigma_ang=0.03"

TRACKER_RECIPES: dict[str, dict] = {
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


def recipe_record(name_or_path: str) -> tuple[dict, dict]:
    """(options, record) for a recipe name or a JSON file path. record = name/path, sha256 of the canonical JSON, options."""
    if name_or_path in TRACKER_RECIPES:
        rec = TRACKER_RECIPES[name_or_path]
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
