"""D-126 #34 loco_pick and #22 foothold_steps scenes (rrp.envs.legged_scenes, default off) + the loco_pick teacher STUB."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rrp.contracts.task import TaskDefinition
from rrp.tasks.compiler import compile_task

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["loco_pick", "foothold_steps"])
def test_task_json_validates_and_compiles(name):
    from rrp.envs.scenario import load_task
    compile_task(TaskDefinition.model_validate(load_task(name)))


def test_existing_builders_unchanged_until_register():
    import rrp.envs.legged  # noqa: F401  (registers waypoint_contact as before)
    from rrp.envs import scenario as sc
    before = dict(sc.BUILDERS)
    import rrp.envs.legged_scenes as ls
    assert "loco_pick" not in sc.BUILDERS and "foothold_steps" not in sc.BUILDERS   # import alone registers nothing
    from rrp.bodies import legged as bl
    assert "spot_arm" not in bl.ALL_LEGGED
    try:
        ls.register()
        assert all(sc.BUILDERS[k] is v for k, v in before.items())
        assert sc.BUILDERS["loco_pick"] is ls.build_loco_pick
    finally:
        sc.BUILDERS.pop("loco_pick", None)
        sc.BUILDERS.pop("foothold_steps", None)


def test_relative_footholds_round_trip():
    from rrp.envs.legged_scenes import foothold_relative_from_world, foothold_world_from_relative
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
    from rrp.envs.legged_scenes import foothold_world_from_relative
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
    from rrp.contracts.action import NativeCommand
    from rrp.envs.legged_scenes import FootholdSession, build_foothold_steps
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
    from rrp.envs.legged_scenes import build_foothold_steps
    sc = build_foothold_steps("go2", 3, n_steps=6)
    _check_foothold_scene(sc, 6)
    assert sc.meta["foot_order"][:2] == ["FL_calf", "FR_calf"]
    sc2 = build_foothold_steps("go2", 3, n_steps=6, stones=True)
    assert sc2.meta["stones"] and sc2.meta["footholds_world"] == sc.meta["footholds_world"]


@pytest.mark.menagerie
def test_loco_pick_spot_arm_scene():
    from rrp.envs.legged_scenes import LOCO_PICK_VERSION, build_loco_pick
    sc = build_loco_pick("spot_arm", 5)
    m = sc.meta
    assert m["scene_version"] == LOCO_PICK_VERSION and m["body_key"] == "spot_arm" and m["contact_model"]
    assert "unsourced" in m["actuator_limits_note"]
    L = sc.robots[0].meta["legged"]
    assert len(L["policy_actuators"]) == 12 and len(L["held_actuators"]) == 7        # legs vs arm+gripper
    assert {a.id for a in sc.robots[0].robot_spec.assemblies} >= {"body", "arm"}
    assert 1.5 <= np.linalg.norm(m["table"]["xy"]) <= 3.0
    compile_task(TaskDefinition.model_validate(sc.task))
    d = mujoco.MjData(sc.model)
    from rrp.envs.legged_core import LeggedBinding
    b = LeggedBinding(sc.model, sc.robots[0].meta)
    b.set_default(d)
    for _ in range(250):
        mujoco.mj_step(sc.model, d)
    cz = d.xpos[mujoco.mj_name2id(sc.model, mujoco.mjtObj.mjOBJ_BODY, "cube")][2]
    assert abs(cz - m["cube"]["z"]) < 0.01                          # the cube rests on the table
    assert d.qpos[b.qa + 2] > 0.6 * L["nominal_height"]             # spot stands under the PD hold
    from rrp.bodies import legged as bl
    assert "spot_arm" not in bl.ALL_LEGGED


def test_loco_pick_teacher_stub_walks_then_stops_labelled():
    from rrp.teachers.legged_loco import LocoPickTeacherStub
    pose = [0.0, 0.0, 0.0]
    s = SimpleNamespace(robots=[SimpleNamespace(meta={"legged": {"command_ranges": {"vx": [-0.5, 1.0], "wz": [-0.8, 0.8]}}})],
                        scenario=SimpleNamespace(meta={"standoff": {"xy": [1.0, 0.0], "yaw": 0.5}}),
                        base_pose_truth=lambda: np.array(pose), controller_version=lambda: "v")
    t = LocoPickTeacherStub(s)
    v = t.command_values()
    assert v[0] > 0 and t.phase == "walk_to_standoff"
    pose[:] = [1.0, 0.0, 0.0]
    v = t.command_values()
    assert t.phase == "face_table" and v[0] == 0 and v[2] > 0
    pose[:] = [1.0, 0.0, 0.5]
    assert np.all(t.command_values() == 0) and t.done and t.outcome == "stub_manipulation_not_implemented"
    assert t.meta()["stub"] is True and t.act().source == "scripted_teacher"
    with pytest.raises(NotImplementedError):
        t.arm_command()
