"""Replay recorder (rrp.viz.record / rrp.viz.replay, viz/CONTRACT.md, D-131). Host-light: a tiny synthetic model
(no simulation beyond 3 mj_steps) for the schema, and a 6-tick arm case for recorded == unrecorded actions.
The legged hook check runs a short legged episode and is peer-only (RRP_NODE=peer)."""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pytest

from rrp.viz import replay as RP

XML = """<mujoco><asset><mesh name="tet" vertex="0 0 0 0.1 0 0 0 0.1 0 0 0 0.1"/></asset>
<worldbody><geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
<body name="box" pos="0 0 0.2"><freejoint/><geom name="b" type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
<geom name="m" type="mesh" mesh="tet" rgba="0 1 0 1"/></body>
<body name="ball" pos="0.3 0 0.2"><geom type="sphere" size="0.03"/><geom type="capsule" size="0.01 0.05" group="3"/></body>
</worldbody></mujoco>"""


def _meta(**kw):
    m = dict(family="arm", task="pick_place", body="synthetic", route="teacher", source_label="mock:synthetic_test",
             ckpt_sha=None, variant=None, seed=0, condition="unit_test", success=True, failure_stage=None,
             physics=dict(contact_version=None, grasp_contact_version="grasp_v1", actuator_limits_version=None,
                          actuator_mode="ideal"), decision_refs=["D-131"])
    m.update(kw)
    return m


def _synthetic(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    geoms, bodies = RP.export_geoms(m)
    col = RP.FrameCollector(m, bodies, control_dt=0.02)            # 50 Hz control -> stride 2 -> 25 fps
    assert col.stride == 2 and col.fps == 25.0
    for k in range(6):
        if k < 3:
            mujoco.mj_step(m, d)
        if col.due():
            col.frame(d, joint_pos=d.qpos[:3], contacts=[bool(d.ncon)], phase=f"p{k}", slip=None,
                      probe=dict(halt=False, goal=[0.1, 0.2]))
            col.events(d.time, {"ev": "active" if k < 4 else "completed"})
        col.tick()
    return m, geoms, col


def test_decimate_budget_and_indices():
    th = np.linspace(0, 2 * np.pi, 80)
    ph = np.linspace(0, np.pi, 60)
    T, P = np.meshgrid(th, ph)
    V = np.stack([np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)], -1).reshape(-1, 3)
    idx = np.arange(V.shape[0]).reshape(60, 80)
    F = np.concatenate([np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:]], -1).reshape(-1, 3),
                        np.stack([idx[:-1, :-1], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 3)])
    assert len(F) > 9000
    v2, f2 = RP.decimate(V, F, 2000)
    assert 100 < len(f2) <= 2000
    assert f2.min() >= 0 and f2.max() < len(v2)
    assert np.all(np.abs(np.linalg.norm(v2, axis=1) - 1) < 0.25)       # still roughly the sphere
    small_v, small_f = RP.decimate(V[:4], np.array([[0, 1, 2], [0, 2, 3]]), 2000)
    assert len(small_f) == 2                                           # under budget: untouched


def test_schema_on_tiny_synthetic_recording(tmp_path):
    m, geoms, col = _synthetic(tmp_path)
    types = {g["type"] for g in geoms}
    assert {"plane", "box", "mesh", "sphere"} <= types and "capsule" not in types     # group-3 collision geom dropped
    mesh = next(g for g in geoms if g["type"] == "mesh")["mesh"]
    assert len(mesh["faces"]) // 3 == 4 and len(mesh["vertices"]) == 12
    res = col.result()
    assert res["n_frames"] == 3 and res["fps"] == 25.0
    assert "slip" not in res["signals"]                                 # all-None signal omitted, never faked
    doc = RP.build_replay("unit-synthetic-s0", _meta(), geoms, res)
    p = RP.write_replay(doc, tmp_path / "arm")
    back = RP.read_replay(p)
    RP.validate_replay(back)
    assert back["frames"]["body_pos"][0][back["bodies"].index("box")][2] == pytest.approx(0.2, abs=1e-3)
    assert [e["status"] for e in back["signals"]["task_events"]] == ["active", "completed"]
    idx = RP.write_index(tmp_path, "2026-09-28T00:00:00Z")
    assert idx["n"] == 1 and idx["replays"][0]["id"] == "unit-synthetic-s0"
    assert json.loads((tmp_path / "index.json").read_text())["schema"] == RP.INDEX_SCHEMA
    with gzip.open(p, "rt") as fh:
        assert json.load(fh)["schema"] == RP.SCHEMA


@pytest.mark.parametrize("breakage", ["fps", "frames", "signal", "meta", "mesh", "pca"])
def test_schema_rejects_broken_documents(tmp_path, breakage):
    m, geoms, col = _synthetic(tmp_path)
    doc = RP.build_replay("unit-bad", _meta(), geoms, col.result())
    if breakage == "fps":
        doc["fps"] = 60
    elif breakage == "frames":
        doc["frames"]["t"].append(1.0)
    elif breakage == "signal":
        doc["signals"]["joint_pos"].pop()
    elif breakage == "meta":
        del doc["meta"]["decision_refs"]
    elif breakage == "mesh":
        g = next(g for g in doc["geoms"] if g["type"] == "mesh")
        g["mesh"] = dict(vertices=[0.0] * 9, faces=[0, 1, 2] * 2001)
    elif breakage == "pca":
        doc["signals"]["packet_pca"] = [[0.0, 0.0, 0.0]] * doc["n_frames"]      # without a basis in meta
    with pytest.raises(RP.ReplaySchemaError):
        RP.validate_replay(doc)


def test_pca_roundtrip():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 3)) @ np.diag([5.0, 1.0, 0.1]) @ rng.normal(size=(3, 12))
    b = RP.fit_pca(Z)
    assert b["dim"] == 12 and len(b["components"]) == 3 and b["explained_variance_ratio"][0] > 0.8
    p = RP.project_pca(b, Z[0])
    assert len(p) == 3


def test_recorded_arm_episode_actions_match_unrecorded(tmp_path):
    """Recording must not change behaviour: the ladder's executed commands and outcome are identical with and
    without the recorder attached (smallest procedural arm, teacher route, 6 control ticks)."""
    pytest.importorskip("mujoco")
    from rrp.evaluation.ladder import LadderConfig, run_ladder
    from rrp.evaluation.robustness import feasible_arm_seeds
    from rrp.viz import record as R
    seeds = feasible_arm_seeds("parm5_pg2", 3_000_000, 1)
    base_log: dict = {}
    cfg = LadderConfig(route="teacher", robot="parm5_pg2", seeds=list(seeds), max_steps=6, device="cpu")
    rows0 = run_ladder(cfg, None, dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None), {}, cmd_log=base_log)
    rec_log: dict = {}
    e = dict(id="unit-arm", harness="ladder", family="arm", task="pick_place", body="parm5_pg2", route="teacher",
             source_label="scripted_teacher:ladder_shadow", decision_refs=["D-131"], seeds=list(seeds), condition="unit",
             args=dict(route="teacher", robot="parm5_pg2", seed_start=3_000_000, n=1, max_steps=6), _cmd_log=rec_log)
    res = R.run_ladder(e, tmp_path, {})
    assert len(base_log[0]) == len(rec_log[0]) == 6
    for a, b in zip(base_log[0], rec_log[0]):
        assert (a is None) == (b is None)
        if a is not None:
            assert a.keys() == b.keys() and all(np.array_equal(np.asarray(a[g]), np.asarray(b[g])) for g in a)
    doc = RP.read_replay(res[0]["file"])
    RP.validate_replay(doc)
    assert doc["n_frames"] == 6 and doc["meta"]["success"] == rows0[0]["privileged_success"]
    assert doc["meta"]["physics"]["grasp_contact_version"] == "grasp_v1"
    assert {"joint_target", "joint_pos", "object_pose", "phase"} <= set(doc["signals"])


@pytest.mark.skipif(os.environ.get("RRP_NODE") != "peer", reason="legged episode: peer only (D-127 host = unit suite)")
def test_recorded_legged_episode_matches_unrecorded(tmp_path):
    from rrp.evaluation.legged_latent_eval import run_episode
    from rrp.viz import record as R
    row0, _ = run_episode(None, "hexapod6", 10000, 0.6)
    e = dict(id="unit-legged", harness="legged", family="legged", task="waypoint", body="hexapod6", route="teacher",
             source_label="scripted_teacher:waypoint", decision_refs=["D-131"], seeds=[10000], condition="unit",
             args=dict(max_s=0.6))
    res = R.run_legged(e, tmp_path, {})
    doc = RP.read_replay(res[0]["file"])
    RP.validate_replay(doc)
    assert doc["meta"]["reproduction"]["compared"]["final_pose"]["rerun"] == np.round(row0["final_pose"], 3).tolist()
