"""W7 arm teacher versions: min-jerk math, version labels, and a tiny real v2 episode (procedural fixture arm)."""
import numpy as np
import pytest

from rrp.contracts.provenance import Source, parse_source
from rrp.teachers.arm_smooth import (DEFAULT_ARM_TEACHER, MJ_APEAK, MJ_VPEAK, minjerk, minjerk_duration,
                                     teacher_source)


def test_minjerk_boundary_and_peaks():
    t = np.linspace(0, 1, 20001)
    s = minjerk(t)
    h = t[1] - t[0]
    v, a = np.gradient(s, h), np.gradient(np.gradient(s, h), h)
    assert s[0] == 0 and abs(s[-1] - 1) < 1e-12
    assert abs(v[0]) < 1e-3 and abs(v[-1]) < 1e-3 and abs(a[2]) < 1e-2 and abs(a[-3]) < 1e-2
    assert abs(v.max() - MJ_VPEAK) < 1e-3 and abs(np.abs(a[5:-5]).max() - MJ_APEAK) < 1e-2
    assert np.all(np.diff(s) >= 0)
    assert minjerk(-1.0) == 0 and minjerk(2.0) == 1


def test_minjerk_duration_respects_limits():
    for d, vm, am in [(0.3, 0.5, 2.0), (0.01, 0.5, 2.0), (1.2, 1.5, 6.0)]:
        T = minjerk_duration(d, vm, am)
        assert MJ_VPEAK * d / T <= vm + 1e-9 and MJ_APEAK * d / T ** 2 <= am + 1e-9


def test_teacher_labels():
    assert DEFAULT_ARM_TEACHER == "v1"
    for v, detail in (("v1", "pick_place_v1_waypoint"), ("v2", "pick_place_v2_minjerk")):
        lab = parse_source(teacher_source(v), strict=True)
        assert lab.kind is Source.SCRIPTED_TEACHER and lab.detail == detail and not lab.deployable


def test_v2_episode_smoke():
    pytest.importorskip("mujoco")
    from rrp.evaluation.teacher_quality import run_quality_episode
    row = run_quality_episode("parm5_pg2", 1, "v2")
    assert row["source"] == "scripted_teacher:pick_place_v2_minjerk"
    assert row["feasible"] and row["success"], row.get("failure_stage")
    assert row["joint_cmd_jerk_peak"] < 150 and row["vel_jump_any_max"] < 0.6
