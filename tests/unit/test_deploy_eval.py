"""D-126 deployment options in the legged closed loop (#27 estimator, #29 packet OOD, #30 safety, #32 long runs and
latency): defaults byte-identical to pre-D-126 rows; each option does what it declares. Tiny random-weight bundle
(tests/unit/_legged_tiny.py) on the procedural hexapod: plumbing only, no result."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rrp.evaluation.deploy_eval import DeployOptions
from rrp.evaluation.legged_latent_eval import LatentLeggedController, run_episode

from ._legged_tiny import row_digest, tiny_bundle

# goldens computed with the pre-D-126 code (origin/main 733b02a) by the same calls
GOLDEN_TEACHER = "7f250b18c57b2afdb1d3dbbf5c0177db896251ef4bc7362cc12a3ba8744e741f"
GOLDEN_LATENT = "3fdfe8beb4030db563e6fb5a0ea04bcfc11d6bdb104aa0a5821803e0902553a5"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    torch.set_num_threads(1)
    return tiny_bundle(tmp_path_factory.mktemp("tiny"))


def _ctl(bundle, **kw):
    return LatentLeggedController(bundle[1], torch.device("cpu"), nfe=2, seed=3, **kw)


def test_defaults_byte_identical_to_pre_d126(bundle):
    row, _ = run_episode(None, "hexapod6", 3, max_s=2.5)
    assert row_digest(row) == GOLDEN_TEACHER
    row, _ = run_episode(None, "hexapod6", 3, max_s=2.5, deploy=DeployOptions())      # default instance = None
    assert row_digest(row) == GOLDEN_TEACHER and "deploy" not in row
    row, _ = run_episode(_ctl(bundle), "hexapod6", 3, max_s=1.5)
    assert row_digest(row) == GOLDEN_LATENT


def test_options_validate():
    with pytest.raises(ValueError):
        DeployOptions(packet_ood="monitor")                       # needs a model
    with pytest.raises(ValueError):
        DeployOptions(packet_ood="enforce", ood_model="x", ood_fallback="safe_stop")   # needs safety=enforce
    with pytest.raises(ValueError):
        DeployOptions(safety="on")
    assert DeployOptions().is_default() and not DeployOptions(measure_latency=True).is_default()


def test_monitor_modes_do_not_change_behaviour(bundle, tmp_path):
    """safety=monitor and packet_ood=monitor observe only: same trajectory as the default run."""
    from rrp.controllers.packet_ood import PacketOODModel
    base, _ = run_episode(_ctl(bundle), "hexapod6", 3, max_s=1.5)
    rng = np.random.default_rng(0)
    ctl = _ctl(bundle)
    M = 7                                                         # hexapod6: 6 legs + body
    z = rng.normal(0, 1, (60, 4, M, 8))
    PacketOODModel.fit(z[:40], z[40:], latent_space_version=ctl.lsv, r=8).save(tmp_path / "ood")
    dep = DeployOptions(safety="monitor", packet_ood="monitor", ood_model=str(tmp_path / "ood"), measure_latency=True)
    row, _ = run_episode(ctl, "hexapod6", 3, max_s=1.5, deploy=dep)
    assert row["final_pose"] == base["final_pose"] and row["stats"] == base["stats"]
    assert row["deploy"]["version"] == "deploy-1" and row["deploy"]["component_versions"]["safety"] == "safety-1"
    assert row["packet_ood"]["n"] == row["stats"]["packets"] and row["packet_ood"]["rejected"] == 0
    assert row["safety"]["mode"] == "monitor" and row["safety"]["ticks"] > 0
    assert row["latency"]["tick"]["n"] > 0 and row["latency"]["system_i"]["n"] == row["stats"]["packets"]


def test_ood_enforce_rejects_and_falls_back(bundle, tmp_path):
    from rrp.controllers.packet_ood import PacketOODModel
    ctl = _ctl(bundle)
    z = np.random.default_rng(0).normal(0, 1e-3, (60, 4, 7, 8))   # tiny-variance "training" packets: all real ones are OOD
    PacketOODModel.fit(z[:40], z[40:], latent_space_version=ctl.lsv, r=8).save(tmp_path / "ood")
    for fb, safety in (("hold_measured", "off"), ("safe_stop", "enforce")):
        dep = DeployOptions(packet_ood="enforce", ood_model=str(tmp_path / "ood"), ood_fallback=fb, safety=safety)
        row, _ = run_episode(_ctl(bundle), "hexapod6", 3, max_s=0.9, deploy=dep)
        assert row["packet_ood"]["rejected"] == row["stats"]["rejected"] > 0
        assert row["stats"]["packets"] == 0
        assert any("packet_ood" in e["err"] for e in row["packet_log"])
        if fb == "safe_stop":
            assert row["safety"]["safe_stop_reason"] == "packet_ood"


def test_ood_model_refuses_other_bundle(bundle, tmp_path):
    from rrp.controllers.packet_ood import PacketOODModel
    z = np.random.default_rng(0).normal(0, 1, (30, 4, 7, 8))
    PacketOODModel.fit(z[:20], z[20:], latent_space_version="legged-ls-other-w000000000000", r=4).save(tmp_path / "o")
    with pytest.raises(ValueError, match="fitted for latent space"):
        run_episode(_ctl(bundle), "hexapod6", 3, max_s=0.5,
                    deploy=DeployOptions(packet_ood="monitor", ood_model=str(tmp_path / "o")))


def test_long_mode_estimator_and_safety_enforce_teacher():
    dep = DeployOptions(eval_mode="long", long_s=2.0, window_s=0.5, base_state_source="estimator", safety="enforce")
    row, _ = run_episode(None, "hexapod6", 3, max_s=60.0, deploy=dep)
    assert row["sim_time"] >= 1.9 or row["fell"]                  # long mode runs to long_s (unless it falls)
    lr = row["long_run"]
    assert lr["window_s"] == 0.5 and len(lr["windows"]) >= 3
    assert all("est_vel_rmse" in w for w in lr["windows"])
    # the estimator actually runs inside run_episode (perturb.install_legged replaces _tracker_tick): estimated speed
    # tracks the true speed (regression: the first peer smoke had est = 0 everywhere)
    moving = [w for w in lr["windows"] if w["mean_speed_true"] > 0.05]
    assert moving and all(w["mean_speed_est"] > 0.3 * w["mean_speed_true"] for w in moving
                          if w["mean_speed_est"] is not None)                # None: first 1 s (no baseline yet)
    assert np.mean([w["est_vel_rmse"] for w in moving]) < 0.5 * np.mean([w["mean_speed_true"] for w in moving])
    assert row["deploy"]["component_versions"]["estimator"] == "bse-1"
    assert row["safety"]["mode"] == "enforce" and row["safety"]["limit_source"]
    json.dumps(row)                                               # rows stay JSON-serializable


def test_record_packets_cli(bundle, tmp_path):
    from rrp.evaluation.legged_latent_eval import main
    out = tmp_path / "rows.jsonl"
    main(["--flow", str(bundle[1]), "--bodies", "hexapod6", "--seeds", "3-3", "--nfe", "2", "--max-s", "0.9",
          "--out", str(out), "--record-packets", str(tmp_path / "pk")])
    row = json.loads(out.read_text().splitlines()[0])
    f = np.load(row["packets_file"])
    assert f["z"].shape[1:] == (4, 7, 8) and len(f["t"]) == row["stats"]["packets"]
    assert "_packet_z" not in row and row["deploy"]["record_packets"] is True


def test_ood_fit_packets_and_score_cli(tmp_path):
    """#29 experiment tool on synthetic packet files in the --record-packets format (no simulation, D-127)."""
    from rrp.training.packet_ood_fit import auroc, main as ood_main
    rng = np.random.default_rng(0)
    pk = tmp_path / "pk"
    pk.mkdir()
    basis = rng.normal(0, 1, (3, 4 * 7 * 8))

    def z(n, big=False):
        x = rng.normal(0, 1, (n, 3)) @ basis + rng.normal(0, 0.05, (n, basis.shape[1]))
        return (x + (rng.normal(0, 3, x.shape) if big else 0)).reshape(n, 4, 7, 8)

    for k in range(10):
        np.savez(pk / f"hexapod6_s{k}_none.npz", t=np.arange(5) * 0.4, edit=np.array(["none"] * 5), z=z(5),
                 fell=False, success=True, source="learned:x", lsv="ls-x")
    for k in range(3):
        np.savez(pk / f"hexapod6_s{k}_zero.npz", t=np.arange(5) * 0.4, edit=np.array(["none"] * 2 + ["zero"] * 3),
                 z=np.concatenate([z(2), z(3, big=True)]), fell=True, success=False, source="learned:x", lsv="ls-x")
    ood_main(["fit-packets", "--packets", str(pk), "--body", "hexapod6", "--out", str(tmp_path / "m"), "--r", "3"])
    ood_main(["score", "--model", str(tmp_path / "m"), "--packets", str(pk), "--out", str(tmp_path / "s.json")])
    s = json.loads((tmp_path / "s.json").read_text())["summary"]
    assert s["n_edited"] == 3 and s["n_unedited"] == 10 and s["n_falls"] == 3
    assert s["auroc_edited_vs_unedited"] == 1.0 and s["auroc_fall"] == 1.0 and s["flagged_edited"] == 3
    assert auroc([1, 2], [0, 0]) == 1.0 and auroc([0], [0]) == 0.5 and auroc([], [1]) is None


def test_system2_flag_wires_harness(bundle, tmp_path):
    from rrp.evaluation.legged_latent_eval import main
    out = tmp_path / "rows.jsonl"
    main(["--flow", str(bundle[1]), "--bodies", "hexapod6", "--seeds", "3-3", "--nfe", "2", "--max-s", "0.5",
          "--out", str(out), "--system2", "oracle"])
    row = json.loads(out.read_text().splitlines()[0])
    assert row["system2"]["harness_version"] == "s2h-1" and row["system2"]["binding_correct"] is True
    assert row["system2"]["target"]["source"].startswith("oracle")


def test_legged_rows_canonical_source_label_when_switched_on(bundle, monkeypatch):
    monkeypatch.setenv("RRP_SOURCE_LABELS", "canonical")
    row, _ = run_episode(_ctl(bundle), "hexapod6", 3, max_s=0.3)
    assert row["source_label"].startswith("learned:flow/policy.pt") and row["source_label_version"] == "sl-1"
    assert row["source"] == _ctl(bundle).policy_version                      # legacy key unchanged
