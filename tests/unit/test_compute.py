"""Compute block (core/compute.py; unit prec): default is today's behaviour and hashes to the same config; a non-default
block hashes in and pins the outputs; precision resolution, fp32 islands, compile guard and the pipeline channel."""
from __future__ import annotations

import json

import pytest
import torch

from rrp.core import compute
from rrp.core.compute import Compute
from rrp.core.runconfig import RunConfig
from rrp.harness.pipelines import base as B


def _rc(**kw):
    d = dict(schema_version="runconfig-1", family="arm", stage="refit", variant="semfix", seed=2, lineage="arm-semfix",
             tag="t", inputs={"representation": "runs/x:representation.pt"},
             flags=dict(zero_prev_action=True, realizer_anchor=True, realizer_drop_qd=False, probe_lv_min=None,
                        qd_dropout=None, contact_version="contact_v1"), params={"steps": 10})
    d.update(kw)
    return RunConfig.model_validate(d)


@pytest.fixture(autouse=True)
def _clean():
    prev, act = compute.configure(None), compute._ACTIVE
    compute._ACTIVE = None
    yield
    compute.configure(prev)
    compute._ACTIVE = act


def test_default_compute_is_absent_from_the_serialised_config_and_hash():
    base = _rc()
    assert "compute" not in json.loads(base.to_json())
    assert _rc(compute=Compute()).config_hash() == base.config_hash()           # a default-equal block is the same config
    assert _rc(compute={"precision": None}).config_hash() == base.config_hash()


def test_non_default_compute_hashes_in_and_round_trips():
    base, bf = _rc(), _rc(compute=dict(precision="bf16"))
    assert bf.config_hash() != base.config_hash()
    assert bf.config_hash() != _rc(compute=dict(precision="fp32")).config_hash()
    assert json.loads(bf.to_json())["compute"]["precision"] == "bf16"
    assert RunConfig.model_validate_json(bf.to_json()).config_hash() == bf.config_hash()
    assert _rc(compute=dict(seeds_per_job=4)).config_hash() != base.config_hash()   # the 'seeds' unit's field, same block


def test_compute_validation():
    with pytest.raises(Exception, match="cuda_graphs needs"):
        Compute(cuda_graphs=True)
    with pytest.raises(Exception, match="reduce-overhead"):
        Compute(compile="reduce-overhead")
    with pytest.raises(Exception):
        Compute(precision="fp16")
    with pytest.raises(Exception):
        Compute(seeds_per_job=0)
    assert Compute(compile="reduce-overhead", cuda_graphs=True).compile == "reduce-overhead"
    assert Compute(compile="max-autotune").cuda_graphs is False


def test_stage_versions_pin_only_a_non_default_block():
    assert "compute" not in B.stage_versions(_rc())
    p = B.stage_versions(_rc(compute=dict(precision="bf16")))
    q = B.stage_versions(_rc(compute=dict(precision="fp32")))
    assert p["compute"] and p["compute"] != q["compute"]                        # outputs of different settings never adopt


def test_legacy_precision_table():
    cuda = {t: compute.resolve(t, "cuda") for t in ("behavior.policy", "pointer.rep", "psi0.train", "legged_bc", "latent.rep")}
    assert [c.precision for c in cuda.values()] == ["bf16" if t in ("pointer.rep", "psi0.train") else "fp32" for t in cuda]
    assert all(c.precision_source == "legacy" and not c.island for c in cuda.values())
    assert cuda["behavior.policy"].tf32 is True and cuda["legged_bc"].tf32 is None
    assert compute.resolve("pointer.rep", "cpu").precision == "fp32"              # the legacy autocast was CUDA-only
    ex = compute.resolve("pointer.rep", "cuda", Compute(precision="fp32"))
    assert ex.precision == "fp32" and ex.precision_source == "explicit" and not ex.island
    ex = compute.resolve("legged_bc", "cuda", Compute(precision="bf16", tf32=False))
    assert ex.precision == "bf16" and ex.island and ex.tf32 is False


def test_autocast_context_follows_the_effective_precision():
    x, w = torch.randn(4, 8), torch.nn.Linear(8, 8)
    with compute.resolve("legged_bc", "cpu", Compute(precision="bf16")).autocast():
        assert w(x).dtype == torch.bfloat16
    with compute.resolve("legged_bc", "cpu").autocast():
        assert w(x).dtype == torch.float32


def test_fp32_islands_are_the_identity_at_the_legacy_default_and_fp32_under_explicit_bf16():
    @compute.f32
    def red(x):
        return x.dtype, (x ** 2).sum()

    xb = torch.randn(6).bfloat16()
    compute.setup("pointer.rep", "cpu")                                         # legacy
    assert red(xb)[0] == torch.bfloat16 and compute.upcast(xb).dtype == torch.bfloat16
    cx = compute.setup("legged_bc", "cpu", Compute(precision="bf16"))
    with cx.autocast():
        assert red(xb)[0] == torch.float32                                      # upcast + autocast off inside
        a, b = compute.upcast(xb, xb)
        assert a.dtype == b.dtype == torch.float32
        assert compute.upcast(torch.arange(3)).dtype == torch.int64             # integer tensors are left alone


def test_islands_reduce_in_fp32_for_the_real_losses():
    from rrp.policies.relations.base import gaussian_nll
    pred = torch.randn(8, 8).bfloat16()
    tgt, m = torch.randn(8, 4), torch.ones(8)
    compute.setup("legged_bc", "cpu", Compute(precision="bf16"))
    with compute.active().autocast():
        out = gaussian_nll(pred, tgt, m, -8.0, 4.0)
    assert out.dtype == torch.float32
    assert torch.equal(out, gaussian_nll(pred.float(), tgt, m, -8.0, 4.0))      # bf16 net output, fp32 reduction: exact
    compute.setup("legged_bc", "cpu")                                           # legacy: the loss sees what it is given
    assert torch.equal(gaussian_nll(pred.float(), tgt, m, -8.0, 4.0), out)


def test_compile_off_is_untouched_and_a_failing_compile_falls_back_recorded(monkeypatch):
    cx = compute.resolve("legged_bc", "cuda")
    f = lambda x: x + 1  # noqa: E731
    assert cx.compile(f, "f") is f and cx.compiled["f"]["status"] == "off"
    cx = compute.resolve("legged_bc", "cuda", Compute(compile="max-autotune"))
    assert cx._mode() == "max-autotune-no-cudagraphs"
    assert compute.resolve("legged_bc", "cuda", Compute(compile="max-autotune", cuda_graphs=True))._mode() == "max-autotune"
    assert compute.resolve("legged_bc", "cuda", Compute(compile="reduce-overhead", cuda_graphs=True))._mode() == "reduce-overhead"
    assert compute.resolve("legged_bc", "cpu", Compute(compile="max-autotune"))._mode() is None   # CPU: never compiled

    def boom(fn, mode=None):
        def bad(*a, **k):
            raise RuntimeError("no compiler here")
        return bad
    monkeypatch.setattr(torch, "compile", boom)
    net = torch.nn.Linear(3, 3)
    sd = list(net.state_dict())
    cx.compile(net, "net")
    out = net(torch.ones(1, 3))                                                 # first call fails compiled, runs eager
    assert out.shape == (1, 3) and cx.compiled["net"]["status"] == "fallback_eager" and "no compiler" in cx.compiled["net"]["reason"]
    assert list(net.state_dict()) == sd                                         # no `_orig_mod.` prefixes


def test_stamp_file(tmp_path):
    cx = compute.setup("pointer.rep", "cpu", Compute(precision="bf16"))
    s = cx.write_stamp(tmp_path / "x.compute.json")
    d = json.loads((tmp_path / "x.compute.json").read_text())
    assert d == json.loads(json.dumps(s, default=str)) and d["effective"]["precision_source"] == "explicit"
    assert d["requested"]["precision"] == "bf16" and d["trainer"] == "pointer.rep"
    cx.write_stamp(tmp_path / "run")
    assert (tmp_path / "run" / "compute.json").exists()


def test_apply_run_context_configures_and_restores(tmp_path, monkeypatch):
    p = tmp_path / B.CONTEXT_FILE
    p.write_text(_rc(compute=dict(precision="bf16", tf32=True)).to_json())
    monkeypatch.setenv(B.CONTEXT_ENV, str(p))
    assert compute.current().is_default
    with B.apply_run_context():
        assert compute.current().precision == "bf16" and compute.current().tf32 is True
    assert compute.current().is_default
    p.write_text(_rc().to_json())
    with B.apply_run_context():
        assert compute.current().is_default


def test_dag_base_carries_a_compute_block():
    from rrp.harness import dag
    src = open(dag.__file__).read()
    assert 'compute=cfg.pop("compute", None)' in src


def test_no_trainer_keeps_its_own_autocast():
    """One shared helper: no trainer calls torch.autocast / sets TF32 itself (core/compute.py is the only place)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "rrp"
    files = list((root / "harness" / "train").rglob("*.py")) + [root / "policies" / "psi0" / "train.py"]
    bad = [str(f.relative_to(root)) for f in files if f.name not in ("adapt.py", "bench_compute.py")       # adapt.py forces TF32 OFF (likelihood ratios); the benchmark resets flags between runs
           and ("torch.autocast" in f.read_text() or "allow_tf32" in f.read_text())]
    assert not bad, bad


# ------------------------------------------------------------------------------------------- equivalence tool (bench_compute --equiv)
def test_equiv_compare_curves_gate():
    from rrp.harness.train import bench_compute as bc
    ref = [1.0 - 0.001 * i for i in range(200)]
    same = [x * 1.01 for x in ref]                       # 1% apart everywhere: inside the declared tolerance
    off = [x * 1.5 for x in ref]                         # 50% apart: outside
    nan = ref[:100] + [float("nan")] * 100
    assert bc.compare_curves(ref, same)["ok"]
    assert not bc.compare_curves(ref, off)["ok"]
    assert not bc.compare_curves(ref, nan)["ok"]
    assert not bc.compare_curves(ref, [])["ok"]


def test_equiv_compare_dev_gate():
    from rrp.harness.train import bench_compute as bc
    ok = bc.compare_dev({"eval.mse": 0.10, "eval.acc": 0.95}, {"eval.mse": 0.11, "eval.acc": 0.94})
    bad = bc.compare_dev({"eval.mse": 0.10}, {"eval.mse": 0.20})
    assert ok["ok"] and ok["n"] == 2
    assert not bad["ok"] and bad["failed"] == ["eval.mse"]
    assert bc.compare_dev({}, {"x": 1.0})["ok"] is None


def test_equiv_loss_tap_sums_backward_scalars_per_optimizer_step():
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    from rrp.harness.train import bench_compute as bc
    m = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    tap = bc._LossTap()
    h = register_optimizer_step_post_hook(tap.step)
    try:
        with tap:
            for _ in range(3):
                x = torch.ones(4, 2)
                (m(x).pow(2).mean()).backward()
                (m(x).pow(2).mean()).backward()          # two losses before one step: summed
                opt.step()
                opt.zero_grad()
    finally:
        h.remove()
    assert len(tap.losses) == 3 and all(l > 0 for l in tap.losses)
    assert torch.Tensor.backward.__name__ == "backward" and not hasattr(torch.Tensor.backward, "__wrapped__")
