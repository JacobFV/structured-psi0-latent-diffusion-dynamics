"""Ψ₀ data driver + SIMPLE plumbing (readiness unit P2, audit D20): per-item seeded realization ticks (one epoch identical for
any worker count), the one ext-dir resolver, the render profile as an env kwarg, the label driver on harness.rollout, and the
psi0 stages' heldout gate. No GPU, no simulator (the label driver runs against the fake worker of test_psi0)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rrp.policies.psi0.data import CachedDataset, collate  # noqa: E402

EPS, FRAMES = 3, 14
NOFLAGS = dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None, qd_dropout=None,
               contact_version=None)


def _fake_worker() -> str:
    """The simulator-free worker double of test_psi0 (one copy of it)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_test_psi0_double", Path(__file__).with_name("test_psi0.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m._FAKE_WORKER


@pytest.fixture()
def feat_dir(tmp_path):
    d = tmp_path / "feat"
    d.mkdir()
    n = EPS * FRAMES
    small = dict(state=[torch.zeros(36) for _ in range(n)], actions=[torch.zeros(30, 36) for _ in range(n)],
                 amask=[torch.ones(30, 36, dtype=torch.bool) for _ in range(n)], ent=[torch.zeros(0, dtype=torch.int32)] * n,
                 ep=[e for e in range(EPS) for _ in range(FRAMES)], fr=[f for _ in range(EPS) for f in range(FRAMES)],
                 length=[2] * n)
    for i in range(n):
        small["state"][i][0] = i
    torch.save(small, d / "small.pt")
    np.save(d / "hidden.npy", np.zeros((n, 2, 2048), np.uint16))
    (d / "meta.json").write_text(json.dumps(dict(episodes=EPS, shards=[])))
    return d


def _epochs(feat_dir, workers, n_epochs=2):
    """{(ep, fr): j} per epoch of a shuffled persistent-worker loader built the way `rrp train psi0` builds it."""
    ds = CachedDataset(feat_dir, seed=7)
    dl = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=True, collate_fn=collate, num_workers=workers,
                                     generator=torch.Generator().manual_seed(3), persistent_workers=workers > 0)
    out = []
    for _ in range(n_epochs):
        js = {}
        for b in dl:
            js.update({(int(e), int(f)): int(j) for e, f, j in zip(b["ep"], b["fr"], b["j"])})
        out.append(js)
    return out


def test_epoch_is_identical_across_worker_counts(feat_dir):
    ref = _epochs(feat_dir, 0)
    assert len(ref[0]) == EPS * FRAMES and any(ref[0].values())      # every frame once; ticks are not all 0
    for workers in (2, 3):
        assert _epochs(feat_dir, workers) == ref, f"num_workers={workers} changed the realization ticks"
    assert ref[0] != ref[1]                                          # a new epoch draws new ticks


def test_tick_is_a_function_of_seed_item_and_visit_only(feat_dir):
    ds = CachedDataset(feat_dir, seed=7)
    a = [ds.realization_tick(n, v) for n in range(len(ds)) for v in range(3)]
    assert a == [CachedDataset(feat_dir, seed=7).realization_tick(n, v) for n in range(len(ds)) for v in range(3)]
    assert a != [CachedDataset(feat_dir, seed=8).realization_tick(n, v) for n in range(len(ds)) for v in range(3)]
    sub = CachedDataset(feat_dir, seed=7, episodes={1})               # the same frame of a subset draws the same tick
    n_full = ds.index[(1, 4)]
    assert sub.realization_tick(sub.index[(1, 4)], 2) == ds.realization_tick(n_full, 2)
    first = [int(ds[n]["j"]) for n in range(len(ds))]
    ds.reset_epochs()
    assert first == [int(ds[n]["j"]) for n in range(len(ds))]


# ------------------------------------------------------------------ one ext-dir resolver, render profile as an env kwarg
def test_one_ext_dir_resolver(monkeypatch, tmp_path):
    import rrp.envs.simple as S
    from rrp.envs.simple import compat as C
    monkeypatch.setenv("RRP_PSI0_EXT", str(tmp_path / "ext"))
    monkeypatch.delenv("PSI_HOME", raising=False)
    assert C.ext_dir() == tmp_path / "ext" and S.ext_dir is C.ext_dir
    assert C.psi_home() == tmp_path / "ext/psi_home" and S.psi_home is C.psi_home
    assert C.simple_root() == tmp_path / "ext/psi0/third_party/SIMPLE"
    monkeypatch.setenv("PSI_HOME", str(tmp_path / "ph"))
    assert C.psi_home() == tmp_path / "ph"
    assert not hasattr(C, "EXT") and not hasattr(C, "SIMPLE_ROOT")   # no import-time constants left to go stale
    from rrp.policies.psi0 import data
    assert data.psi_home is C.psi_home


def test_render_profile_is_an_env_kwarg(monkeypatch, tmp_path):
    from rrp.envs.simple import DEFAULT_RENDER_PROFILE, RENDER_PROFILES, SimpleEnv, worker_env
    from rrp.envs.simple import compat as C
    monkeypatch.setenv("RRP_PSI0_EXT", str(tmp_path))
    assert DEFAULT_RENDER_PROFILE in RENDER_PROFILES
    env = worker_env(b"k" * 32)
    assert env["RRP_SIMPLE_RENDER"] == DEFAULT_RENDER_PROFILE and env["RRP_PSI0_EXT"] == str(tmp_path)
    other = sorted(set(RENDER_PROFILES) - {DEFAULT_RENDER_PROFILE})[0]
    assert worker_env(b"k" * 32, other)["RRP_SIMPLE_RENDER"] == other
    with pytest.raises(ValueError, match="render_profile"):
        worker_env(b"k" * 32, "no_such_profile")
    with pytest.raises(ValueError, match="render_profile"):          # refused before any process starts
        SimpleEnv("G1WholebodyTabletopGraspMP-v0", render_profile="no_such_profile", worker_cmd=["false"])
    monkeypatch.setenv("RRP_SIMPLE_RENDER", "no_such_profile")
    with pytest.raises(ValueError, match="unknown render profile"):
        C.render_profile()


def test_render_profile_reaches_the_worker(tmp_path, monkeypatch):
    _FAKE_WORKER = _fake_worker()
    from rrp.envs.simple import RENDER_PROFILES, SimpleEnv
    other = sorted(RENDER_PROFILES)[-1]
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER.replace("W.serve(", "open(os.environ['MARK'], 'w').write(os.environ['RRP_SIMPLE_RENDER'])\nW.serve("))
    monkeypatch.setenv("MARK", str(tmp_path / "mark"))
    env = SimpleEnv("G1WholebodyTabletopGraspMP-v0", worker_cmd=[sys.executable, str(script)], start_timeout=60,
                    render_profile=other)
    env.close()
    assert (tmp_path / "mark").read_text() == other


# ------------------------------------------------------------------ label driver on harness.rollout
def test_label_driver_records_through_rollout(tmp_path):
    _FAKE_WORKER = _fake_worker()
    from rrp.cli.data import _episodes, record_psi0_labels
    from rrp.policies.psi0 import Psi0Policy
    task = "G1WholebodyTabletopGraspMP-v0"
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER)
    orig = Psi0Policy.recorded_rows
    Psi0Policy.recorded_rows = lambda self, e: np.full((60, 36), e, np.float32)
    try:
        eps = record_psi0_labels(task, tmp_path / "labels", _episodes("0:2"), batch=2,
                                 env_kw=dict(worker_cmd=[sys.executable, str(script)], start_timeout=60))
    finally:
        Psi0Policy.recorded_rows = orig
    assert _episodes("3,5") == [3, 5] and [e.seed for e in eps] == [0, 1]
    rows = [json.loads(x) for x in (tmp_path / "labels/labels.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and {r["label_source"] for r in rows} == {"privileged:sim_replay"}
    assert (tmp_path / "labels/episode_000001.npz").exists()


# ------------------------------------------------------------------ pipeline: labels collect + heldout gate
def _ctx(tmp_path, stage, **kw):
    from rrp.core.runconfig import RunConfig, RunIndex
    from rrp.harness.pipelines import base as B
    B._load_families()
    rc = RunConfig.model_validate(dict(schema_version="runconfig-1", family="psi0", stage=stage, variant="na", seed=0,
                                       lineage="p2", track="psi0", flags=NOFLAGS, options=kw.get("options", {}),
                                       inputs=kw.get("inputs", {}), params=kw.get("params", {})))
    return B.StageContext(rc=rc, index=RunIndex(), root=tmp_path)




def test_structured_stage_is_blocked_by_the_gate(tmp_path, monkeypatch):
    from rrp.harness.pipelines import base as B
    from rrp.harness.pipelines.base import GateFailed, StageError
    calls = []
    monkeypatch.setattr(B.StageContext, "run", lambda self, argv, **kw: calls.append(argv))
    monkeypatch.setattr("rrp.harness.pipelines.psi0._run_dir", lambda ctx: "run")
    inputs = dict(features="in/feat", labels="in/labels", stage_a="in/stage_a.pt", gate="in/heldout.json")
    (tmp_path / "artifacts/in").mkdir(parents=True)
    stage = B._REGISTRY[("psi0", "train_flow")].fn
    opts = dict(task="G1WholebodyTabletopGraspMP-v0")
    for gate, exc in ((dict(gate=dict(gap=0.01, margin=0.05)), GateFailed), (dict(no="gate"), StageError)):
        (tmp_path / "artifacts/in/heldout.json").write_text(json.dumps(gate))
        with pytest.raises(exc):
            stage(_ctx(tmp_path, "train_flow", options=opts, inputs=inputs))
    assert not calls                                                   # the trainer never started
    (tmp_path / "artifacts/in/heldout.json").write_text(json.dumps(dict(gate=dict(gap=0.07, margin=0.05))))
    ctx = _ctx(tmp_path, "train_flow", options=opts, inputs=inputs)
    try:
        stage(ctx)
    except (FileNotFoundError, StageError):                            # no summary.json: the stub trainer wrote nothing
        pass
    assert calls and "--labels-dir" in calls[0] and json.loads((ctx.out / "gate_report.json").read_text())["verdict"] == "pass"


def test_collect_labels_stage_and_source(tmp_path, monkeypatch):
    from rrp.harness.pipelines import base as B
    from rrp.harness.pipelines.base import StageError
    calls = []
    monkeypatch.setattr(B.StageContext, "run", lambda self, argv, **kw: calls.append(argv))
    (tmp_path / "artifacts/in/feat").mkdir(parents=True)
    (tmp_path / "artifacts/in/feat/meta.json").write_text(json.dumps(dict(episodes=6)))
    ctx = _ctx(tmp_path, "collect", options=dict(task="simple/G1WholebodyTabletopGraspMP-v0", data="labels"),
               inputs=dict(features="in/feat"))
    res = B._REGISTRY[("psi0", "collect")].fn(ctx)
    a = calls[0]
    assert a[:4] == ["-m", "rrp.cli", "data", "psi0-labels"] and a[a.index("--episodes") + 1] == "0:6"
    assert a[a.index("--task") + 1] == "G1WholebodyTabletopGraspMP-v0" and res["source"] == "privileged_teacher:sim_replay"
    from rrp.core.provenance import make_provenance
    assert make_provenance(res["source"], flags={}, versions={}, notes="t").source.startswith("privileged_teacher")
    with pytest.raises(StageError, match="features or labels"):
        B._REGISTRY[("psi0", "collect")].fn(_ctx(tmp_path, "collect", options=dict(task="x", data="frames")))


def test_heldout_without_structured_is_the_gate(tmp_path, monkeypatch):
    from rrp.harness.pipelines import base as B
    from rrp.harness.pipelines.base import GateFailed
    monkeypatch.setattr("rrp.harness.pipelines.psi0._run_dir", lambda ctx: "run")
    (tmp_path / "artifacts/in").mkdir(parents=True)
    (tmp_path / "artifacts/in/summary.json").write_text(json.dumps(dict(val_eps=[1])))
    inputs = dict(features="in/feat", summary="in/summary.json", stage_a="in/stage_a.pt")

    def stub(gap):
        def run(self, argv, **kw):
            f = Path(argv[argv.index("--out") + 1]); f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(dict(gate=dict(gap=gap, margin=0.05))))
        return run
    fn = B._REGISTRY[("psi0", "heldout")].fn
    monkeypatch.setattr(B.StageContext, "run", stub(0.2))
    assert fn(_ctx(tmp_path, "heldout", options=dict(task="G1WholebodyTabletopGraspMP-v0"), inputs=inputs))["metrics"]["verdict"] == "pass"
    monkeypatch.setattr(B.StageContext, "run", stub(0.0))
    with pytest.raises(GateFailed):
        fn(_ctx(tmp_path, "heldout", options=dict(task="G1WholebodyTabletopGraspMP-v0"), inputs=inputs))
