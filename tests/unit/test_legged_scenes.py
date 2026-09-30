"""D-126 #22 foothold_steps scene (rrp.envs.legged_scenes, default off). loco_pick and its teacher stub were retired (D-146, TK)."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rrp.core.task import TaskDefinition
from rrp.tasks.compiler import compile_task

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["foothold_steps"])
def test_task_json_validates_and_compiles(name):
    from rrp.envs.mujoco.scenario import load_task
    compile_task(TaskDefinition.model_validate(load_task(name)))


def test_existing_builders_unchanged_until_register():
    import rrp.envs.mujoco.legged  # noqa: F401  (registers waypoint_contact as before)
    from rrp.envs.mujoco import scenario as sc
    before = dict(sc.BUILDERS)
    import rrp.envs.mujoco.legged_scenes as ls
    assert "foothold_steps" not in sc.BUILDERS   # import alone registers nothing
    from rrp.bodies import legged as bl
    assert "spot_arm" not in bl.ALL_LEGGED
    try:
        ls.register()
        assert all(sc.BUILDERS[k] is v for k, v in before.items())
        assert sc.BUILDERS["foothold_steps"] is ls.build_foothold_steps
    finally:
        sc.BUILDERS.pop("foothold_steps", None)


def test_relative_footholds_round_trip():
    from rrp.envs.mujoco.legged_scenes import foothold_relative_from_world, foothold_world_from_relative
    rel = [[0.3, 0.1, 0.2], [0.25, -0.2, -0.1], [0.2, 0.0, 0.3]]
    w = foothold_world_from_relative(rel)
    assert np.allclose(foothold_relative_from_world(w), rel)
    assert np.allclose(w[0], [0.3, 0.1, 0.2])
    assert np.allclose(w[1][:2], [0.3 + math.cos(0.2) * 0.25 + math.sin(0.2) * 0.2,
                                  0.1 + math.sin(0.2) * 0.25 - math.cos(0.2) * 0.2])


def _check_foothold_scene(sc, n):
    m = sc.meta
    assert m["scene_version"] == "foothold_steps_v0" and m["contact_model"]
    assert len(m["footholds_world"]) == len(m["footholds_relative"]) == len(m["foot_order"]) == n
    from rrp.envs.mujoco.legged_scenes import foothold_world_from_relative
    assert np.allclose(foothold_world_from_relative(m["footholds_relative"]), m["footholds_world"])
    d = mujoco.MjData(sc.model)
    mujoco.mj_forward(sc.model, d)
    for k, (x, y, _) in enumerate(m["footholds_world"]):
        bid = mujoco.mj_name2id(sc.model, mujoco.mjtObj.mjOBJ_BODY, f"foothold_{k:02d}")
        assert np.allclose(d.xpos[bid][:2], [x, y])
        g = mujoco.mj_name2id(sc.model, mujoco.mjtObj.mjOBJ_GEOM, f"foothold_{k:02d}_disc")
        assert sc.model.geom_contype[g] == 0 and sc.model.geom_conaffinity[g] == 0      # markers never collide
    compile_task(TaskDefinition.model_validate(sc.task))
    for k, f in enumerate(m["foot_order"]):
        roles = [r["role"] for r in sc.task["events"][k]["roles"]]
        assert roles.count("actor") == (2 if f is not None else 1)          # null foot identity = no foot role


def test_foothold_scene_and_public_estimate_procedural():
    """pquad4 (procedural, CPG tracker): builds, and the public foot-distance estimate tracks the truth."""
    from rrp.core.action import NativeCommand
    from rrp.envs.mujoco.legged_scenes import FootholdSession, build_foothold_steps
    sc = build_foothold_steps("pquad4", 1, n_steps=4, any_foot_every=2)
    _check_foothold_scene(sc, 4)
    assert sc.meta["foot_order"][1] is None
    s = FootholdSession(sc, seed=1, tracker_kind="cpg")
    s.reset()
    errs = []
    for _ in range(6):
        s.step(NativeCommand(controller_version=s.controller_version(), groups={"base_velocity": [0.1, 0.0, 0.0]},
                             source="scripted_teacher"))
        v, known, _ = s.estimate("foot_distance_m", ["body", "foothold_00"])
        t = s.truth_predicate("foot_distance_m", ["body", "foothold_00"])
        if known and t is not None:
            errs.append(abs(v - t))
    assert errs and max(errs) < 0.1
    assert s.failure_reason() in (None, "timeout", "missed_foothold", "wrong_foot", "fell")


@pytest.mark.menagerie
def test_foothold_scene_go2():
    from rrp.envs.mujoco.legged_scenes import build_foothold_steps
    sc = build_foothold_steps("go2", 3, n_steps=6)
    _check_foothold_scene(sc, 6)
    assert sc.meta["foot_order"][:2] == ["FL_calf", "FR_calf"]
    sc2 = build_foothold_steps("go2", 3, n_steps=6, stones=True)
    assert sc2.meta["stones"] and sc2.meta["footholds_world"] == sc.meta["footholds_world"]
