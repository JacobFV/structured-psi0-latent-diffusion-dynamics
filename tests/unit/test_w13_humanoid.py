"""W13 humanoid program (D-138): sealed split integrity, recipes, new adapters, GPU modules import without their deps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPLIT = json.loads((ROOT / "research/splits/humanoid_v1.json").read_text())


def _sealed_bodies():
    out = set()
    for t in SPLIT["targets"].values():
        b = t["bodies"]
        if isinstance(b, list):
            out |= set(b)
    return out


def test_sealed_bodies_disjoint_from_training():
    sealed = _sealed_bodies()
    assert sealed == {"g1_hands", "n1", "berkeley", "toddlerbot_2xc", "toddlerbot_2xm"}
    assert not sealed & set(SPLIT["source_train_bodies"])


def test_recipes_train_only_pool_bodies_and_valid_reward_keys():
    from rrp.envs.mujoco.legged_core import RewardCfg
    from rrp.harness.train.tracker_recipes import WARP_RECIPES as HUMANOID_RECIPES, recipe_record
    pool = set(SPLIT["source_train_bodies"])
    for name, rec in HUMANOID_RECIPES.items():
        opts, record = recipe_record(name)
        bodies = [opts["body"]] if opts.get("body") else [k for ks, _ in opts["groups"] for k in ks]
        for b in bodies:
            assert b in pool or (b.startswith("phum_") and int(b[5:]) < 1_000_000), (name, b)
        assert len(record["sha256"]) == 64
        for kv in opts["reward_set"].split(","):
            assert hasattr(RewardCfg(), kv.split("=")[0]), (name, kv)


def test_morph_slot_rules():
    from rrp.envs.mujoco.morph_obs import slot_of
    assert slot_of("Left_Hip_Pitch") == ("left", "hip_pitch")
    assert slot_of("r_ank_roll_act") == ("right", "ankle_roll")
    assert slot_of("l_hip_ie") == ("left", "hip_yaw") and slot_of("r_ankle_pd") == ("right", "ankle_pitch")
    assert slot_of("hipPitch_Right") == ("right", "hip_pitch") and slot_of("left_ankle") == ("left", "ankle_pitch")
    assert slot_of("leg_right_4_joint_position") == ("right", "knee")
    assert slot_of("LR_FAA") == ("right", "ankle_roll") and slot_of("LL_HR") == ("left", "hip_yaw")
    assert slot_of("waist_yaw") is None


def test_sealed_adapter_flagged():
    from rrp.bodies.legged import LEGGED_ASSETS
    for k, info in LEGGED_ASSETS.items():
        if k in _sealed_bodies():
            assert info.get("sealed") is True, k
    assert not LEGGED_ASSETS["apollo"].get("sealed") and not LEGGED_ASSETS["adam_lite"].get("sealed")


def test_auto_gain_rule():
    from rrp.bodies.legged import _auto_gains
    assert _auto_gains(5.0) == (10.0, 0.25)
    assert _auto_gains(139.0) == (139.0, 0.025 * 139.0)
    assert _auto_gains(494.0) == (300.0, 7.5)


def test_gpu_modules_import_without_warp():
    import rrp.envs.warp.model as wl
    try:
        import mujoco_warp  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="mujoco_warp"):
            wl._wp()


@pytest.mark.menagerie
def test_new_adapters_build_with_pitch_mapping():
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.mujoco.legged_core import LeggedBinding
    for key, left in (("apollo", ["l_hip_fe", "l_knee_fe", "l_ankle_pd"]),
                      ("adam_lite", ["hipPitch_Left", "kneePitch_Left", "anklePitch_Left"])):
        m, _, meta = standalone_model(legged_body(key), contact="v2")
        b = LeggedBinding(m, meta)
        assert b.n == 12 and b.nf == 2
        acts = meta["legged"]["policy_actuators"]
        assert [acts[i] for i in b.pitch_idx()[0]] == left
        assert meta["limits_source"] == "menagerie_author" and meta["gain_rule"]
        assert 0.8 < b.nominal_height() < 1.2


def test_phum_sampling_deterministic_and_sealed_region_excluded_from_training():
    from rrp.bodies.humanoid_gen import in_sealed_region, sample_params, sealed_region_seeds, SEALED_SEED_MIN
    assert sample_params(17) == sample_params(17)
    assert not any(in_sealed_region(sample_params(s)) for s in range(3000))
    ss = sealed_region_seeds(2)
    assert all(s >= SEALED_SEED_MIN and in_sealed_region(sample_params(s)) for s in ss)


def test_phum_body_builds_with_legged_contract():
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.mujoco.legged_core import LeggedBinding
    for seed in (0, 1, 4):
        m, _, meta = standalone_model(legged_body(f"phum_{seed}"), contact="v2")
        b = LeggedBinding(m, meta)
        assert b.nf == 2 and b.n == 2 * (meta["params"]["leg_dof"])
        assert meta["synthetic"] and not meta["sealed"] and meta["limits_source"] == "procedural_scaling"
        acts = meta["legged"]["policy_actuators"]
        assert [acts[i] for i in b.pitch_idx()[0]] == ["left_hip_pitch", "left_knee", "left_ankle_pitch"]
        assert 0.3 < b.nominal_height() / meta["params"]["height"] < 0.7
    _, _, meta = standalone_model(legged_body("phum_9000029"), contact="v2")
    assert meta["sealed"]


@pytest.mark.menagerie
def test_h_steps_scenario_and_scan():
    import numpy as np
    from rrp.envs.mujoco.humanoid_scenes import build_h_steps, steps_height_at, steps_scan_np, steps_layout
    sc = build_h_steps("t1", 3, h_frac=0.2)
    L, h = sc.meta["L"], sc.meta["staircase"]["h"]
    assert abs(h - 0.2 * L) < 1e-9 and sc.meta["x_end"] == steps_layout(L, h)[1]
    assert steps_height_at(np.array([sc.meta["staircase"]["x0"] + 1e-3]), L, h)[0] == h
    root = np.array([sc.meta["staircase"]["x0"] - 0.1 * L, 0, L, 1, 0, 0, 0])   # standing, facing +x, just before step 1
    scan = steps_scan_np(root, L, h)
    assert scan.shape == (34,) and abs(scan[-1] - 0.2) < 1e-6
    assert scan[0] == 0.0 and np.isclose(scan[3 * 10], 2 * h / L)                # behind: flat; 1.2 L ahead: on step 2


@pytest.mark.menagerie
def test_h_gap_scenario_and_obs():
    import numpy as np
    from rrp.envs.mujoco.humanoid_scenes import GAP_X, build_h_gap, gap_obs_np
    sc = build_h_gap("h1", 5, level=1.0)
    mt = sc.meta
    assert 1.2 <= mt["gap_ratio"] <= 1.6 and abs(mt["y_c"]) <= 0.6 * mt["L"] and abs(mt["psi_f"]) <= np.pi / 2
    o = gap_obs_np(np.array([0, 0, mt["L"], 1, 0, 0, 0]), mt, 0.0)
    assert o.shape == (8,) and np.isclose(o[0], GAP_X) and np.isclose(o[1] * mt["L"], mt["y_c"])


def test_model_adaptations_are_explicit():
    """rrp.envs.warp.model applies only the named adaptations and records them (moved from the D-126 #15 prototype test)."""
    from rrp.envs.warp import model as mod
    with pytest.raises(KeyError, match="unknown model adaptation"):
        mod.build_model("pquad4", "v2", adapt=["not_an_adaptation"])
    m, _, b, done = mod.build_model("pquad4", "v2", adapt=["pyramidal_cone"])
    assert done[0]["name"] == "pyramidal_cone" and int(m.opt.cone) == 0
    assert mod.default_data(m, b).qpos.shape == (m.nq,) and mod.substeps_of(m) >= 1
