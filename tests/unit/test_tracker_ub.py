"""HS2 (D-146 round 2): wholebody tracker recipes, amplitude ramp, sealed-split guard, tracker install. CPU only; needs no warp,
weights or assets (the fake-env PPO updates live in test_morph_multi.py)."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from rrp.harness.train import tracker_recipes as R
from rrp.harness.train.warp_tracker_ppo import upper_ramp_value


def test_ramp_is_linear_then_constant_and_off_at_zero():
    kw = dict(amp0=0.1, amp=0.4, payload_frac=0.08)
    assert upper_ramp_value(0, 1000, ramp=0.3, **kw) == (0.1, 0.0)
    a, p = upper_ramp_value(150, 1000, ramp=0.3, **kw)
    assert a == pytest.approx(0.25) and p == pytest.approx(0.04)
    assert upper_ramp_value(300, 1000, ramp=0.3, **kw) == (0.4, 0.08)
    assert upper_ramp_value(999, 1000, ramp=0.3, **kw) == (0.4, 0.08)
    assert upper_ramp_value(0, 1000, ramp=0.0, **kw) == (0.4, 0.08)             # no ramp: the full values from the start
    assert upper_ramp_value(3, 10, ramp=0.0001, **kw) == (0.4, 0.08)


def test_every_ub_recipe_ramps_and_the_shared_one_is_registered():
    ub = [k for k in R.WARP_RECIPES if k.endswith("_ub")]
    assert set(ub) >= {"t1_clock_gpu_ub", "g1_clock_gpu_ub", "h1_clock_gpu_ub", "op3_clock_gpu_ub", "t1_steps_ub"}
    assert R.recipe_record("t1_steps_ub")[0]["task"] == "steps"
    for k in ub:
        o, _ = R.recipe_record(k)
        assert o["upper_body"] and o["upper_ramp"] == 0.3 and o["upper_amp0"] == 0.1, k
    assert "shared_morph_ub" in R.WARP_RECIPES and "shared_morph_ub" in R._LAZY


def test_shared_ub_recipe_groups_all_have_arms(monkeypatch):
    seen = {}

    def fake(n, per, nw, seed_range=(0, 20000), min_arm_dof=0):
        seen.update(n=n, min_arm_dof=min_arm_dof)
        return [[["phum_1"], nw]]
    monkeypatch.setattr(R, "phum_groups", fake)
    o, rec = R.recipe_record("shared_morph_ub")
    assert seen["min_arm_dof"] >= 1 and o["upper_body"] and o["clock_gate"] and len(o["groups"]) == len(R.SHARED_POOL_V1) + 1
    assert o["hidden"] == "512,512,256" and rec["name"] == "shared_morph_ub"


def test_phum_groups_min_arm_dof_filters_topologies():
    from rrp.bodies.humanoid_gen import sample_params
    g = R.phum_groups(3, 2, 8, seed_range=(0, 400), min_arm_dof=1)
    assert g
    for keys, nw in g:
        assert nw == 8 and all(sample_params(int(k.split("_")[1])).arm_dof >= 1 for k in keys)


# ------------------------------------------------------------------------------------------------------ sealed-split guard
def test_trainers_refuse_a_sealed_body_before_building_anything(tmp_path):
    from rrp.core.sealed import SealedSplitError
    a = SimpleNamespace(body="phum_9000001", groups=None, out=str(tmp_path / "x"))
    with pytest.raises(SealedSplitError):
        R.assert_trainable(a)
    from rrp.harness.train.warp_tracker_ppo import train as gpu_train
    with pytest.raises(SealedSplitError):
        gpu_train(a, None, dev=torch.device("cpu"), engine="fake")
    from rrp.harness.train.tracker_training import train as cpu_train
    with pytest.raises(SealedSplitError):
        cpu_train(a)
    assert not (tmp_path / "x").exists()                                          # nothing was created
    with pytest.raises(SealedSplitError):                                          # also inside a shared-tracker group list
        R.assert_trainable(SimpleNamespace(body=None, groups=[[["t1"], 4], [["phum_9000001", "phum_1"], 4]]))
    R.assert_trainable(SimpleNamespace(body="t1", groups=None))
    R.assert_trainable(SimpleNamespace(body=None, groups=json.dumps([[["phum_1", "phum_2"], 4]])))


# ------------------------------------------------------------------------------------------------------ install
def _run(tmp_path, *, verdict="pass", sha=None):
    run = tmp_path / "run"
    run.mkdir()
    st = dict(actor={"0.weight": torch.zeros(2, 3)}, obs_mean=torch.zeros(3), obs_var=torch.ones(3), log_std=torch.zeros(2),
              meta=dict(body="tiny", obs_dim=3, act_dim=2, clock_gate=True, obs_format="morph_v2"))
    torch.save(st, run / "actor.pt")
    (run / "train_log.jsonl").write_text("\n".join(json.dumps(dict(iter=i)) for i in range(25)) + "\n")
    real = hashlib.sha256((run / "actor.pt").read_bytes()).hexdigest()
    v = tmp_path / "validation.json"
    v.write_text(json.dumps(dict(body="tiny", tracker_sha=sha or real, w6_gate=dict(verdict=verdict, failed=["force"] if verdict != "pass" else []))))
    return run, v, real


def test_install_round_trip_into_the_tracker_registry(tmp_path):
    from rrp.envs.mujoco.legged_tracker import scan_trackers
    from rrp.harness.train.tracker_training import install
    run, v, sha = _run(tmp_path)
    root = tmp_path / "store"
    store = install(run, [v], "tiny", "ub_v1", label="tiny test", root=root)
    assert store == root / "tiny" / "ub_v1" and (store / "actor.pt").read_bytes() == (run / "actor.pt").read_bytes()
    meta = json.loads((store / "meta.json").read_text())
    assert meta["sha256"] == sha and meta["install_label"] == "tiny test" and meta["obs_format"] == "morph_v2"
    assert meta["extra_obs"] == "none" and meta["validations"] == [dict(file="validation.json", body="tiny", verdict="pass")]
    assert (store / "validation_learned.json").exists() and len((store / "train_log_every10.jsonl").read_text().splitlines()) == 3
    e = scan_trackers(root)[("tiny", "ub_v1")]
    assert e.sha256 == sha and e.obs_format == "morph_v2" and e.gate and e.decision == "accepted" and e.actor.exists()
    with pytest.raises(ValueError, match="already exists"):                        # never overwritten
        install(run, [v], "tiny", "ub_v1", label="again", root=root)


def test_install_refuses_a_failed_gate_a_foreign_sha_and_no_validation(tmp_path):
    from rrp.harness.train.tracker_training import install
    root = tmp_path / "store"
    run, v, _ = _run(tmp_path, verdict="fail")
    with pytest.raises(ValueError, match="verdict is 'fail'"):
        install(run, [v], "tiny", "v", label="x", root=root)
    with pytest.raises(ValueError, match="at least one"):
        install(run, [], "tiny", "v", label="x", root=root)
    assert not root.exists()
    (tmp_path / "b").mkdir()
    run2, v2, _ = _run(tmp_path / "b", sha="0" * 64)
    with pytest.raises(ValueError, match="validates sha"):
        install(run2, [v2], "tiny", "v", label="x", root=root)
    assert not root.exists()                                                       # a refusal writes nothing


def test_the_install_command_is_registered():
    from rrp.cli.tools import TOOLS
    assert TOOLS[("train", "tracker-install")][0] == "rrp.harness.train.tracker_training:install_main"


def test_install_command_line(tmp_path, capsys):
    from rrp.harness.train.tracker_training import install_main
    run, v, _ = _run(tmp_path)
    install_main([str(run), "--validation", str(v), "--body", "tiny", "--version", "ub_v1", "--label", "cli", "--root", str(tmp_path / "s")])
    assert json.loads(capsys.readouterr().out)["spec"] == "tiny:ub_v1" and (tmp_path / "s/tiny/ub_v1/actor.pt").exists()
    with pytest.raises(SystemExit, match="already exists"):
        install_main([str(run), "--validation", str(v), "--body", "tiny", "--version", "ub_v1", "--label", "cli", "--root", str(tmp_path / "s")])
