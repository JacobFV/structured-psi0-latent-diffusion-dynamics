"""Every pointer trainer (rep, flow, bc) runs on a tiny synthetic pack (D-146 K2): `Demos.batch` supplies every key
the net's `_relctx` reads (`wpos3d` / `wcamuvd`), from the stored table when the pack has it and as zeros flagged
invalid (`wgeo_ok`) when the pack predates them. CPU, two steps each, no simulation."""
import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rrp.harness.train import pointer as T  # noqa: E402
from rrp.policies.pointer import LC, LI, NH, NW, WF, PointerGeometry  # noqa: E402

GEOM = PointerGeometry((640, 480), 0.001, 10.0)          # synthetic: the trainers read the env spec's geometry


def _pack(path, *, geometry: bool, seed0: int, n_ep=4, n_tick=8):
    rng = np.random.default_rng(seed0)
    n_tab, N = n_ep, n_ep * n_tick
    wmask = np.zeros((n_tab, NW), bool)
    wmask[:, :5] = True
    d = dict(
        wch=rng.integers(1, 20, (n_tab, NW, LC)).astype(np.int16), wrole=rng.integers(0, 4, (n_tab, NW)).astype(np.int8),
        wbound=rng.integers(0, 3, (n_tab, NW)).astype(np.int8),
        wf=(rng.normal(size=(n_tab, NW, WF)) * 0.3).astype(np.float16), wmask=wmask,
        tab=np.repeat(np.arange(n_tab), n_tick).astype(np.int32), ep=np.repeat(np.arange(n_ep), n_tick).astype(np.int32),
        ptr=rng.uniform(-1, 1, (N, 2)).astype(np.float32), btn=np.zeros(N, np.float32),
        hist=np.zeros((N, NH, 5), np.float32), cmd_xy=rng.uniform(-1, 1, (N, 2)).astype(np.float32),
        cmd_btn=np.zeros(N, np.float32), cmd_key=np.zeros(N, np.int16), slot=rng.integers(0, 5, N).astype(np.int16),
        txy=rng.uniform(-1, 1, (N, 2)).astype(np.float32), phase=rng.integers(0, 3, N).astype(np.int8),
        ep_seed=np.arange(n_ep, dtype=np.int64) + seed0, ep_start=np.arange(n_ep, dtype=np.int64) * n_tick,
        ep_end=(np.arange(n_ep, dtype=np.int64) + 1) * n_tick, ep_instr=rng.integers(1, 90, (n_ep, LI)).astype(np.int64),
        ep_goal=np.array(['{"a": 1, "b": 2}'] * n_ep), task=np.array("cw/calc_sum"))
    if geometry:
        d["wpos3d"] = (rng.normal(size=(n_tab, NW, 3)) * 0.1).astype(np.float32)
        d["wcamuvd"] = rng.normal(size=(n_tab, NW, 3)).astype(np.float32)
    np.savez_compressed(path, **d)


@pytest.fixture(autouse=True)
def _geometry(monkeypatch):
    monkeypatch.setattr("rrp.harness.train.pointer.train.pointer_geometry", lambda: GEOM)


def _args(tmp_path, files, **kw):
    return argparse.Namespace(data=[str(f) for f in files], steps=2, batch=8, lr=3e-4, seed=0, device="cpu",
                              log_every=1, split=T.SPLIT_PATH, **kw)


def test_demos_batch_supplies_geometry_stored_or_flagged_invalid(tmp_path):
    new, old = tmp_path / "new.npz", tmp_path / "old.npz"
    _pack(new, geometry=True, seed0=10)
    _pack(old, geometry=False, seed0=20)
    data = T.Demos([str(new), str(old)], "cpu", GEOM)
    assert data.half.tolist() == pytest.approx([0.32, 0.24])
    ix = torch.arange(data.N)
    b, _, _ = data.batch(ix)
    assert b["wpos3d"].shape == b["wcamuvd"].shape == (data.N, NW, 3)
    from_new = data.t["tab"][ix] < 4
    assert b["wgeo_ok"][from_new][:, :5].all() and not b["wgeo_ok"][~from_new].any()
    assert b["wpos3d"][from_new].abs().sum() > 0 and b["wpos3d"][~from_new].abs().sum() == 0
    assert not b["wgeo_ok"][:, 5:].any()                        # empty slots are never valid


def test_rep_flow_bc_two_steps_on_synthetic_pack(tmp_path):
    packs = []
    for i, geo in enumerate((True, False)):                    # one pack with stored geometry, one without
        packs.append(tmp_path / f"p{i}.npz")
        _pack(packs[-1], geometry=geo, seed0=10 * (i + 1))
    rep, flow, bc = (tmp_path / f"{k}.pt" for k in ("rep", "flow", "bc"))
    T.cmd_rep(_args(tmp_path, packs, out=str(rep), variant="semfix", dz=16, w_sem=1.0, lv_min=-4.0, beta=3e-3,
                    w_xy=5.0))
    T.cmd_flow(_args(tmp_path, packs, out=str(flow), target="latent", representation=str(rep), w_sem=0.5))
    T.cmd_bc(_args(tmp_path, packs, out=str(bc), w_xy=5.0))
    for path, kind in ((rep, "pointer_rep"), (flow, "pointer_flow"), (bc, "pointer_bc")):
        blob = torch.load(path, weights_only=False)
        assert blob["kind"] == kind and all(np.isfinite(v) for v in blob["metrics"].values()
                                            if isinstance(v, float))


def test_demos_refuse_a_pack_from_another_geometry(tmp_path):
    import json
    p = tmp_path / "p.npz"
    _pack(p, geometry=True, seed0=1)
    d = dict(np.load(p))
    np.savez_compressed(p, **d, geom=np.array(json.dumps(GEOM.as_dict())))
    T.Demos([str(p)], "cpu", GEOM)                                        # a stamped pack of the same geometry loads
    with pytest.raises(ValueError, match="different pointer geometry"):
        T.Demos([str(p)], "cpu", PointerGeometry((960, 640), 0.001, 10.0))


def test_fit_is_the_one_loop_and_tick_period_comes_from_the_geometry(tmp_path):
    """`fit` drives a step_fn / eval_fn pair; the realizer is queried at phases j * geom.dt (here 0.05 s, not 0.1)."""
    from rrp.harness.train.pointer.train import _tile_forward, fit
    _pack(tmp_path / "p.npz", geometry=True, seed0=3)
    data = T.Demos([str(tmp_path / "p.npz")], "cpu", GEOM)
    w = torch.nn.Parameter(torch.ones(1))
    seen = []
    log = fit([w], lambda ix: ((w * data.t["ptr"][ix].sum()).abs() * 0 + (w - 3) ** 2, dict(n=len(ix))), data, steps=5,
              batch=4, lr=0.1, eval_fn=lambda: seen.append(1) or dict(v=1.0), log_every=2)
    assert [r["step"] for r in log] == [0, 2, 4] and len(seen) == 3 and log[0]["n"] == 4 and log[0]["val_v"] == 1.0

    phases = []

    def R(zz, ph, ptr, pbtn):
        phases.append(ph)
        return torch.zeros(len(ph), 2), torch.zeros(len(ph)), torch.zeros(len(ph), 3)
    _tile_forward(R, torch.zeros(2, 4, 1, 5), GEOM.dt * 0.5, torch.zeros(2, 7, 2), torch.zeros(2, 7), "cpu")
    assert phases[0].reshape(2, 7)[0].tolist() == pytest.approx([j * GEOM.dt * 0.5 for j in range(7)])
