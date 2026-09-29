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
    from rrp.envs.legged_core import RewardCfg
    from rrp.training.humanoid_recipes import HUMANOID_RECIPES, recipe_record
    pool = set(SPLIT["source_train_bodies"])
    for name, rec in HUMANOID_RECIPES.items():
        assert rec["body"] in pool, name
        opts, record = recipe_record(name)
        assert len(record["sha256"]) == 64
        for kv in opts["reward_set"].split(","):
            assert hasattr(RewardCfg(), kv.split("=")[0]), (name, kv)


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
    import rrp.envs.warp_legged as wl
    try:
        import mujoco_warp  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="mujoco_warp"):
            wl._wp()


@pytest.mark.menagerie
def test_new_adapters_build_with_pitch_mapping():
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.legged_core import LeggedBinding
    for key, left in (("apollo", ["l_hip_fe", "l_knee_fe", "l_ankle_pd"]),
                      ("adam_lite", ["hipPitch_Left", "kneePitch_Left", "anklePitch_Left"])):
        m, _, meta = standalone_model(legged_body(key), contact="v2")
        b = LeggedBinding(m, meta)
        assert b.n == 12 and b.nf == 2
        acts = meta["legged"]["policy_actuators"]
        assert [acts[i] for i in b.pitch_idx()[0]] == left
        assert meta["limits_source"] == "menagerie_author" and meta["gain_rule"]
        assert 0.8 < b.nominal_height() < 1.2
