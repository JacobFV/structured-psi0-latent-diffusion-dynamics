"""W11: the stable public core API (rrp.core, docs/core_api.md), its extension hooks (families from outside rrp,
robots from outside rrp, entry points), repo-path independence of an installed rrp, and the statistics math."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CORE_ONLY = ("torch", "mujoco", "fastapi", "jsonschema", "PIL", "imageio", "transformers")


def _py(code: str, *, env_extra: dict | None = None, cwd: Path | None = None, extra_path: list[str] = ()):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), *extra_path]))
    env.pop("RRP_HOME", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=env,
                          timeout=180, cwd=cwd)


# ------------------------------------------------------------------------------------------------ public names
def test_every_stable_name_imports():
    import rrp.core as core
    missing = []
    for name in core.STABLE:
        try:
            getattr(core, name)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{name}: {e!r}")
    assert not missing, missing
    assert set(core.__all__) >= set(core.STABLE) and core.CORE_API_VERSION.split(".")[0] == "1"
    with pytest.raises(AttributeError):
        core.not_a_public_name  # noqa: B018


def test_core_names_need_only_the_core_install():
    """numpy + pydantic only: every name except the torch/mujoco-backed ones imports with the heavy deps blocked."""
    heavy_modules = {"rrp.controllers.latent_realizer", "rrp.models.latent_probes", "rrp.evaluation.legged_latent_eval",
                     "rrp.bodies.catalog"}          # torch (ml extra) or mujoco (sim extra)
    r = _py(f"""
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {CORE_ONLY!r}:
                    raise ModuleNotFoundError("blocked: " + name, name=name)
        sys.meta_path.insert(0, Block())
        import rrp.core as core
        bad = []
        for n, m in core.STABLE.items():
            if m.split(":")[0] in {sorted(heavy_modules)!r}:
                continue
            try:
                getattr(core, n)
            except Exception as e:
                bad.append(f"{{n}} ({{m}}): {{e!r}}")
        print("BAD", bad)
        assert not bad
    """)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


# ------------------------------------------------------------------------------------------------ extension: family
FLAGS_NONE = dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None,
                  qd_dropout=None, contact_version=None)


@pytest.fixture
def dummy_family():
    from rrp.contracts.runconfig import register_family, unregister_family
    from rrp.pipelines.base import unregister_stages
    register_family("dummy_ext", flag_spec={"train_rep": {"probe_lv_min": "latent.probe_lv_min"}},
                    doc="test-only external family")
    yield "dummy_ext"
    unregister_stages("dummy_ext")
    unregister_family("dummy_ext")


def _rc(family="dummy_ext", stage="collect", flags=None, **kw):
    from rrp.core import RunConfig
    d = dict(schema_version="runconfig-1", family=family, stage=stage, variant="na", seed=1, lineage="ext",
             track="t", flags=flags or FLAGS_NONE, params={"n": 3})
    d.update(kw)
    return RunConfig.model_validate(d)


def test_unknown_family_is_refused():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="unknown family"):
        _rc(family="no_such_family")


def test_external_family_registers_stages_and_runs(dummy_family, tmp_path):
    from rrp.core import Pipeline, RunConfigError, register_family, register_stage, read_manifest, families
    assert "dummy_ext" in families()

    @register_stage("dummy_ext", "collect", source="scripted_teacher")
    def collect(ctx):
        """dummy collect"""
        ctx.out.mkdir(parents=True, exist_ok=True)
        (ctx.out / "data.json").write_text(json.dumps(ctx.native))
        return dict(outputs={"data": f"{ctx.rc.out}/data.json"}, metrics={"n": ctx.native["n"]})

    assert Pipeline("dummy_ext").stages() == ["collect"]
    rc = _rc()
    body = Pipeline("dummy_ext").run(rc, root=tmp_path)
    m = read_manifest(tmp_path / rc.out / "pipeline_manifest.json")
    assert m["hash_ok"] and body["metrics"] == {"n": 3} and body["config_hash"] == rc.config_hash()
    assert m["provenance"].source == "scripted_teacher"
    # the family's flag spec is enforced like the built-ins'
    with pytest.raises(Exception, match="probe_lv_min"):
        _rc(stage="train_rep")
    _rc(stage="train_rep", flags=dict(FLAGS_NONE, probe_lv_min=-4.0), params={"latent": {"semantic_weight": 0}})
    with pytest.raises(RunConfigError, match="different spec"):
        register_family("dummy_ext", flag_spec={})
    with pytest.raises(RunConfigError, match="built in"):
        register_family("arm")
    # the built-ins are unchanged by registering an extension family (dual stages: D-126 #33)
    assert Pipeline("dual").stages() == ["collect", "pack", "train_rep", "probes", "train_flow", "flow_ft",
                                         "dagger_collect", "refit", "eval_r2", "heldout", "edits"]


def test_external_family_in_run_dag(dummy_family):
    from rrp.core import plan_dag
    spec = dict(name="ext_dag", family="dummy_ext", track="t", lineage="ext", matrix={"seed": [1, 2]},
                base=dict(variant="na", flags={}), defaults=dict(resources=dict(cpu=1, mem="1G")),
                nodes={"c": dict(stage="collect", config=dict(params={"n": 1}))})
    plan = plan_dag(spec)
    assert len(plan.nodes) == 2 and {n.rc.family for n in plan.nodes.values()} == {"dummy_ext"}


def test_external_family_via_entry_point(tmp_path):
    """A package installed next to rrp declares [project.entry-points."rrp.families"]; rrp loads it lazily."""
    site = tmp_path / "site"
    (site / "dummy_rrp_ext-0.1.dist-info").mkdir(parents=True)
    (site / "dummy_rrp_ext-0.1.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: dummy-rrp-ext\n"
                                                                   "Version: 0.1\n")
    (site / "dummy_rrp_ext-0.1.dist-info" / "entry_points.txt").write_text(
        "[rrp.families]\ng1_dummy = dummy_rrp_ext:register\n[rrp.robots]\nbots = dummy_rrp_ext:robots\n")
    (site / "dummy_rrp_ext.py").write_text(textwrap.dedent("""
        def register():
            from rrp.contracts.runconfig import register_family
            from rrp.pipelines import register
            register_family("g1_dummy")

            @register("g1_dummy", "eval_r2", source="learned")
            def eval_r2(ctx):
                return {}

        def robots():
            return {"dummy_bot": lambda: "a robot"}
    """))
    r = _py("""
        from rrp.core import RunConfig, Pipeline, workbench_robots
        rc = RunConfig.model_validate(dict(schema_version="runconfig-1", family="g1_dummy", stage="eval_r2",
             variant="na", seed=0, lineage="x", flags=dict(zero_prev_action=None, realizer_anchor=None,
             realizer_drop_qd=None, probe_lv_min=None, qd_dropout=None, contact_version=None)))
        print(rc.family, Pipeline("g1_dummy").stages())
        print(workbench_robots()["dummy_bot"]())
    """, extra_path=[str(site)])
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.split("\n")[:2] == ["g1_dummy ['eval_r2']", "a robot"]


def test_register_robot_hook():
    from rrp.core import register_robot, unregister_robot, workbench_robots
    base = workbench_robots()
    register_robot("dummy_ext_bot", lambda: "bot")
    try:
        w = workbench_robots()
        assert w["dummy_ext_bot"]() == "bot" and set(w) - set(base) == {"dummy_ext_bot"}
        register_robot("parm5_pg2", lambda: None)
        with pytest.raises(ValueError, match="shadow"):
            workbench_robots()
        unregister_robot("parm5_pg2")
    finally:
        unregister_robot("dummy_ext_bot")
    assert workbench_robots().keys() == base.keys()


# ------------------------------------------------------------------------------------------------ system 0 base
def _packet(lsv="ls", rcv="rz", spec="a" * 16, now=0.0):
    from rrp.core import AssemblyHandle, LatentActionChunk
    return LatentActionChunk(latent_space_version=lsv, realizer_compat_version=rcv, z=np.zeros((2, 1, 4), np.float32),
                             knot_times=[0.1, 0.2], assemblies=[AssemblyHandle(handle=f"asm:{spec}:arm", robot_index=0)],
                             assembly_mask=[True], observation_id="o", graph_version=0, runtime_version=0,
                             robot_spec_hash=spec, generated_at=now, valid_from=now, valid_until=now + 1,
                             source="learned", policy_version="p")


def test_system0_base_protocol():
    from rrp.core import ControllerRejection, System0Base

    class Dummy(System0Base):
        robot_spec_hash = "a" * 16

    s0 = Dummy(latent_space_version="ls", realizer_compat_version="rz")
    s0.receive(_packet(), now=0.0, graph_version=0)
    assert s0.packet is not None and s0.stats.packets == 1
    with pytest.raises(ControllerRejection):
        s0.receive(_packet(lsv="other"), now=0.0, graph_version=0)
    assert s0.stats.rejected == 1 and s0.log[-1]["event"] == "packet_rejected"
    s0.invalidate("graph_edit", 0.5)
    assert s0.packet is None
    from rrp.controllers.latent_realizer import LatentSystem0
    assert issubclass(LatentSystem0, System0Base)


# ------------------------------------------------------------------------------------------------ paths
def test_rrp_home_checkout_env_and_installed(tmp_path):
    from rrp.core import data_path, is_checkout, rrp_home
    assert is_checkout() and rrp_home() == REPO and data_path("tasks") == REPO / "tasks"
    r = _py("from rrp.core import rrp_home; print(rrp_home())", env_extra={"RRP_HOME": str(tmp_path)})
    assert r.stdout.strip() == str(tmp_path)
    # a copy of the package outside any checkout = the installed case: cwd, packaged task specs, installed sha
    import shutil
    site = tmp_path / "site"
    shutil.copytree(REPO / "src" / "rrp", site / "rrp", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "tasks", site / "rrp" / "_data" / "tasks")
    work = tmp_path / "work"
    work.mkdir()
    r = subprocess.run([sys.executable, "-c", "from rrp.core import rrp_home, is_checkout, data_path, code_provenance;"
                        "from rrp.contracts.provenance import installed_revision as ir;"
                        "print(is_checkout(), rrp_home(), data_path('tasks', 'pick_place.json').is_file(),"
                        " code_provenance().git_sha == ir())"],
                       capture_output=True, text=True, cwd=work, timeout=120,
                       env={k: v for k, v in os.environ.items() if k not in ("RRP_HOME", "RRP_GIT_SHA")}
                       | {"PYTHONPATH": str(site)})
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["False", str(work), "True", "True"]      # never the cwd's git state


# ------------------------------------------------------------------------------------------------ statistics
def test_newcombe_matches_published_example():
    from rrp.core import newcombe_diff
    lo, hi = newcombe_diff(56, 70, 48, 80)           # Newcombe (1998), method 10 example: 0.0524, 0.3339
    assert abs(lo - 0.0524) < 5e-4 and abs(hi - 0.3339) < 5e-4
    assert newcombe_diff(1, 0, 1, 2) == (None, None)


def test_mcnemar_and_sign_flip_exact():
    from rrp.core import mcnemar_exact, paired_permutation_test
    assert mcnemar_exact(0, 5) == pytest.approx(2 * 0.5 ** 5) and mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 2 ** 10)
    r = paired_permutation_test([1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
    assert r["exact"] and r["p"] == pytest.approx(2 / 32) and r["mean_diff"] == 1.0
    r = paired_permutation_test([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], alternative="greater")
    assert r["p"] == pytest.approx(1 / 32)
    r = paired_permutation_test([0.0] * 4, [0.0] * 4)
    assert r["p"] == 1.0
    with pytest.raises(ValueError):
        paired_permutation_test([1, 2], [1])


def test_permutation_and_bootstrap():
    from rrp.core import boot_ci, boot_diff, paired_bootstrap_ci, permutation_test
    g = np.random.default_rng(0)
    a, b = g.normal(1.0, 1, 40), g.normal(0.0, 1, 40)
    p = permutation_test(a, b, n=2000)
    assert p["p"] < 0.01 and p["mean_diff"] == pytest.approx(a.mean() - b.mean())
    same = permutation_test(a, a, n=500)
    assert same["p"] == 1.0
    ci = paired_bootstrap_ci(a, b)
    assert ci["lo"] < ci["mean"] < ci["hi"] and ci["n"] == 40 and ci["mean"] == pytest.approx((a - b).mean())
    assert boot_ci([None, float("nan")])["n"] == 0
    d = boot_diff(a, b)
    assert d["lo"] < d["mean"] < d["hi"]
    # boot_ci is the latent_causal convention, bit for bit
    from rrp.evaluation.latent_causal import boot_ci as lc_boot_ci
    assert boot_ci(list(a)) == lc_boot_ci(list(a))


# ------------------------------------------------------------------------------------------------ edit harness
def test_edit_harness_suite_and_paired_effect(tmp_path):
    from rrp.core import EditCondition, matched_random, orthogonal_matched, paired_effect, read_rows, restamp, run_suite
    conds = [EditCondition("control", "control"), EditCondition("shift", "semantic", "moves +x"),
             EditCondition("irrelevant", "irrelevant_control", "no change")]
    with pytest.raises(ValueError):
        EditCondition("x", "bogus")

    def run_one(seed, c):
        if seed == 3 and c.name == "irrelevant":
            return dict(skipped="infeasible")
        return dict(dx=(0.05 if c.name == "shift" else 0.0) + 0.001 * seed)
    out = tmp_path / "rows.jsonl"
    rows = run_suite(range(8), conds, run_one, out_path=out, log=lambda m: None)
    assert read_rows(out) == rows and len(rows) == 24
    e = paired_effect(rows, "dx", "shift")
    assert e["n_pairs"] == 8 and e["mean_diff"] == pytest.approx(0.05) and e["p"] == pytest.approx(2 / 256)
    e0 = paired_effect(rows, "dx", "irrelevant")
    assert e0["n_pairs"] == 7 and e0["mean_diff"] == pytest.approx(0.0)
    p = _packet(now=0.0)
    q = restamp(p, now=5.0, z=np.ones((2, 1, 4)), intervention="shift")
    assert (q.valid_from, q.valid_until, q.source, q.sampling["intervention"]) == (5.0, 6.0, "debug", "shift")
    assert restamp(p, now=1.0).source == "learned"
    d = np.ones((2, 1, 4), np.float32)
    assert np.linalg.norm(matched_random(d, 0)) == pytest.approx(np.linalg.norm(d), rel=1e-5)
    J = np.eye(8)[:3]
    r, info = orthogonal_matched((2, 1, 4), 2.0, J, 0)
    assert np.abs(J @ r.reshape(-1)).max() < 1e-5 and np.linalg.norm(r) == pytest.approx(2.0, rel=1e-5)
    assert info["probe_span_dim"] == 3


def test_probe_guided_edit_moves_readout_and_anchors():
    torch = pytest.importorskip("torch")
    from rrp.core import probe_guided_edit, probe_jacobian
    W = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    readout = lambda z: W @ z.reshape(-1)                             # noqa: E731
    z0 = np.zeros(4, np.float32)
    z, info = probe_guided_edit(z0, lambda z: (readout(z)[0] - 1.0) ** 2,
                                anchor_loss=lambda z, z0: (readout(z)[1] - readout(z0)[1]) ** 2, steps=300, lr=0.05)
    assert abs(z[0] - 1.0) < 0.05 and abs(z[1]) < 0.05 and info["target_loss"] < 1e-2
    assert np.allclose(probe_jacobian(readout, z0), W.numpy())
    assert math.isfinite(info["anchor_residual"])
