"""Replay recorder (rrp.viz.record / rrp.viz.replay, viz/CONTRACT.md, D-131; RP5 / D-146: the recorder is a set of
rollout hooks). Host-light: a tiny synthetic model (no simulation beyond 3 mj_steps) for the schema, tiny
CPG-tracker / arm / dual / rig episodes for "recorded == unrecorded" and "energy from StepResult", and an AST check
that record.py steps nothing itself. The legged hook check runs a short legged episode and is peer-only (RRP_NODE=peer)."""
from __future__ import annotations

import ast
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from rrp.viz import record as RV
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


# ------------------------------------------------------------------ R21: relation-factor maps
def test_factor_maps_schema_and_provenance(tmp_path):
    """record_factor_maps / write_factor_map / read_factor_map (D-144 R21) on a REAL `FactorSite.contributions()`
    call (tiny fixture, CPU, no simulation) — the per-factor per-head logit maps and their source provenance that
    the room's factor-inspection panel reads."""
    from rrp.policies.relations.base import FactorDef, RelCtx, TokenSet, register_factor, resolve
    from rrp.policies.relations.ops import FactorSite
    register_factor(FactorDef("test.viz_pape", "1", field="pos3d", op="sqdiff+diff", form="aug",
                              sources=("given", "gt"), params=(("p", 3),)))
    g = torch.Generator().manual_seed(0)
    tq = TokenSet("q", torch.ones(1, 3, dtype=torch.bool), fields={"pos3d": torch.randn(1, 3, 3, generator=g)})
    tk = TokenSet("k", torch.ones(1, 4, dtype=torch.bool), fields={"pos3d": torch.randn(1, 4, 3, generator=g)})
    rc = RelCtx(sets={"q": tq, "k": tk})
    specs = resolve(["test.viz_pape"])
    site = FactorSite(2, 8, "q>k", specs, ("pos3d",))
    with torch.no_grad():
        for prm in site.parameters():
            prm.copy_(torch.randn(prm.shape, generator=g))
    x = torch.randn(1, 3, 8, generator=g)
    contrib = site.contributions(rc, x, x)
    assert set(contrib) == {"test.viz_pape"} and tuple(contrib["test.viz_pape"].shape) == (1, 2, 3, 4)   # [B,H,Q,K]

    doc = RV.record_factor_maps("unit-factormap-s0", [{"t": 0.0, "site": "q>k", "contributions": contrib}], specs)
    assert doc["schema"] == RV.FACTORMAP_SCHEMA and doc["id"] == "unit-factormap-s0" and len(doc["steps"]) == 1
    step = doc["steps"][0]
    assert step["site"] == "q>k" and step["t"] == 0.0
    arr = step["factors"]["test.viz_pape"]
    assert len(arr) == 2 and len(arr[0]) == 3 and len(arr[0][0]) == 4                                    # [H,Q,K]
    ref = contrib["test.viz_pape"][0].detach().numpy()
    assert np.allclose(np.asarray(arr), np.round(ref, 4), atol=1e-4)
    prov = {row["name"]: row for row in doc["provenance"]}
    assert prov["test.viz_pape"]["source"] == "given" and prov["test.viz_pape"]["privileged"] is False
    assert prov["test.viz_pape"].get("control", "on") == "on"        # default control: `to_dict()` omits it

    p = RV.write_factor_map(doc, tmp_path)
    assert p.name == "unit-factormap-s0.factormap.json.gz" and p.parent == tmp_path
    back = RV.read_factor_map(p)
    assert back == doc


def test_factor_maps_batch_zero_and_missing_contributions_stay_absent():
    """A factor whose control is `off` (or that never reaches `contributions()`) is simply absent from a step's
    `factors` — never a zero-filled placeholder — and only batch element 0 is kept."""
    doc = RV.record_factor_maps("unit-empty", [{"t": 1.5, "site": "a>b",
                                                "contributions": {"x": np.zeros((3, 2, 4, 4))}}], specs=())
    assert doc["provenance"] == []
    step = doc["steps"][0]
    assert set(step["factors"]) == {"x"} and len(step["factors"]["x"]) == 2       # head dim of batch element 0


def test_recorded_arm_episode_actions_match_unrecorded(tmp_path):
    """Recording must not change behaviour: the ladder's executed commands and outcome are identical with and
    without the recorder attached (smallest procedural arm, teacher route, 6 control ticks)."""
    pytest.importorskip("mujoco")
    from rrp.harness.eval.ladder import LadderConfig, run_ladder
    from rrp.harness.eval.robustness import feasible_arm_seeds
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
    assert {"joint_vel", "joint_torque", "power", "energy", "object_vel", "object_ang_vel", "grasp_state"} <= set(doc["signals"])  # v1.2
    assert len(doc["meta"]["joint_torque_names"]) == len(doc["signals"]["joint_torque"][0])
    assert all(len(f) == len(doc["meta"]["contact_bodies"]) for f in doc["signals"].get("contact_force", []))
    m = doc["meta"]                                                  # contract v1.1 column names / bodies
    assert len(m["joint_names"]) == len(doc["signals"]["joint_pos"][0])
    assert len(m["joint_target_names"]) == len(doc["signals"]["joint_target"][0])
    assert len(m["contact_bodies"]) == len(doc["signals"]["contacts"][0]) and set(m["contact_bodies"]) <= set(doc["bodies"])


@pytest.mark.skipif(os.environ.get("RRP_NODE") != "peer", reason="legged episode: peer only (D-127 host = unit suite)")
def test_recorded_legged_episode_matches_unrecorded(tmp_path):
    from rrp.harness.eval.legged_latent_eval import run_episode
    from rrp.viz import record as R
    row0, _ = run_episode(None, "hexapod6", 10000, 0.6)
    e = dict(id="unit-legged", harness="legged", family="legged", task="waypoint", body="hexapod6", route="teacher",
             source_label="scripted_teacher:waypoint", decision_refs=["D-131"], seeds=[10000], condition="unit",
             args=dict(max_s=0.6))
    res = R.run_legged(e, tmp_path, {})
    doc = RP.read_replay(res[0]["file"])
    RP.validate_replay(doc)
    assert doc["meta"]["reproduction"]["compared"]["final_pose"]["rerun"] == np.round(row0["final_pose"], 3).tolist()


# ------------------------------------------------------------------ RP5: the recorder is a set of rollout hooks
def _calls(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            yield (f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None), n


def test_record_has_no_private_step_loop():
    """record.py owns no stepping: no `mj_step` / `.step()` / `mj_forward` call, no patched stepping or
    recorder-class function. The only things it still wraps are the three packet generators (display taps that read and
    call the original)."""
    src = Path(RV.__file__).read_text()
    tree = ast.parse(src)
    bad = [name for name, _ in _calls(tree) if name in ("step", "mj_step", "mj_forward", "mj_kinematics")]
    assert not bad, f"record.py steps the physics itself: {bad}"
    patched = [c.args[1].value for n, c in _calls(tree) if n == "_patched" and len(c.args) > 1
               and isinstance(c.args[1], ast.Constant)]
    assert set(patched) <= {"generate", "packet", "receive"}, patched
    assert "_Proxy" not in src and "on_substep" not in src and "after_step" not in src


def _entry(**kw):
    e = dict(id="unit", family="legged", task="waypoint", body="hexapod6", route="teacher",
             source_label="scripted_teacher:unit", decision_refs=["D-146"], seeds=[1000], condition="unit")
    e.update(kw)
    return e


def test_recorded_tracker_trial_equals_unrecorded_and_energy_is_the_step_result(tmp_path, monkeypatch):
    """The tracker-validation recorder is a hook on the bench env: the row is identical with and without it, and the
    replay's cumulative energy at its last frame is the sum of the env's own per-step `StepResult.energy_j` (substep-exact)."""
    pytest.importorskip("mujoco")
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.mujoco.legged_core import LeggedBinding
    from rrp.envs.mujoco.legged_tracker import CPGTracker
    from rrp.harness.eval import tracker_validation as TV
    from rrp.harness.rollout import rollout
    model, _, meta = standalone_model(legged_body("hexapod6"), contact="v1")
    b = LeggedBinding(model, meta)
    trial = dict(T=0.8, cmd=[0.3 * b.cmd_ranges["vx"][1], 0, 0])
    row0 = TV.run_episode(model, b, CPGTracker(b, meta), trial, 1000)
    monkeypatch.setattr(TV, "scripts", lambda bb: {"t": trial})
    e = _entry(harness="tracker_val", args=dict(kind="cpg", trial="t"))
    res = RV.run_tracker_val(e, tmp_path, {})
    doc = RP.read_replay(res[0]["file"])
    RP.validate_replay(doc)
    assert doc["meta"]["success"] == (not row0["fell"])
    for k in ("dist_m", "mean_vx", "slip_mps", "duty_factor"):
        assert doc["meta"]["result"][k] == pytest.approx(row0[k], abs=1e-9)
    # energy: an independent rollout summing the env's StepResult.energy_j at the same frames
    seen = []

    class Sum:
        def on_step(self, i, env, act, step):
            seen.append(step.energy_j)
    TV.run_episode(model, b, CPGTracker(b, meta), trial, 1000, hooks=[Sum()])
    assert len(seen) == int(round(trial["T"] / TV._DT)) and all(x is not None and x >= 0 for x in seen)
    k = int(round(doc["frames"]["t"][-1] / TV._DT))                    # the last recorded frame is the k-th tick
    total = doc["meta"]["energy_total_j"]
    assert total == pytest.approx(sum(seen[:k]), rel=1e-3) and total > 0


def test_recorded_arm_teacher_equals_unrecorded(tmp_path):
    """The arm-teacher recorder is a hook of `run_quality_episode`: the quality row is identical with and without it and
    the replay's phase is the teacher's own label, delivered as `Act.info["phase"]`."""
    pytest.importorskip("mujoco")
    from rrp.harness.eval.teacher_quality import run_quality_episode
    row0 = run_quality_episode("parm5_pg2", 3, "v1", max_steps=14)
    e = _entry(harness="arm_teacher", family="arm", task="pick_place", body="parm5_pg2", seeds=[3],
               args=dict(robot="parm5_pg2", version="v1", max_steps=14))
    res = RV.run_arm_teacher(e, tmp_path, {})
    doc = RP.read_replay(res[0]["file"])
    RP.validate_replay(doc)
    assert doc["meta"]["success"] == bool(row0["success"])
    assert doc["meta"]["reproduction"]["compared"]["steps"]["rerun"] == row0["steps"]
    assert doc["n_frames"] >= 1 and all(isinstance(p, str) and p for p in doc["signals"]["phase"])


def test_recorded_dual_teacher_equals_unrecorded(tmp_path):
    """The dual-teacher recorder is a hook after the audit's own: same row with and without it; the replay's phase
    comes from `Act.info` (the teacher's label) and survives DART noise."""
    pytest.importorskip("mujoco")
    from rrp.harness.eval.dual_teacher_quality import run_audit_episode
    row0 = run_audit_episode("support_insert", DUAL_PAIR, 3, max_steps=12, noise=0.03)
    e = _entry(harness="dual_teacher", family="dual", task="support_insert", body=DUAL_PAIR, seeds=[3],
               args=dict(task="support_insert", pair=DUAL_PAIR, max_steps=12, noise=0.03))
    res = RV.run_dual_teacher(e, tmp_path, {})
    doc = RP.read_replay(res[0]["file"])
    RP.validate_replay(doc)
    m = doc["meta"]
    assert m["success"] == (row0["status"] == "success")
    assert m["reproduction"]["compared"]["steps"]["rerun"] == row0["steps"]
    assert doc["n_frames"] >= 1 and all(p is None or p.startswith("L:") for p in doc["signals"]["phase"])
    assert any(doc["signals"]["phase"])


def test_recorded_grasp_rig_equals_unrecorded(tmp_path):
    """The rig recorder is a hook on the rig env: the rig's result row is identical with and without it, and the
    replay's energy is the env's per-step StepResult.energy_j integrated over every physics step."""
    pytest.importorskip("mujoco")
    import rrp.harness.eval.grasp_rig as GR
    res0 = GR.run("v2", "pg2", 1.0)
    e = _entry(harness="grasp_rig", family="rig", task="grasp_rig", body="pg2", seeds=[0],
               args=dict(version="v2", gripper="pg2"))
    res = RV.run_grasp_rig(e, tmp_path, {})
    doc = RP.read_replay(res[0]["file"])
    RP.validate_replay(doc)
    assert doc["meta"]["result"] == res0
    assert doc["meta"]["energy_total_j"] > 0 and doc["n_frames"] > 10
    assert set(doc["signals"]["phase"]) >= {"close", "lift"}


DUAL_PAIR = "parm5l_pg2__parm6_pg2"
