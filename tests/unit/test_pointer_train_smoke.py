"""Every pointer trainer (rep, flow, bc) runs on a tiny synthetic pack (D-146 K2): `Demos.batch` supplies every key
the net's `_relctx` reads (`wpos3d` / `wcamuvd`), from the stored table when the pack has it and as zeros flagged
invalid (`wgeo_ok`) when the pack predates them. CPU, two steps each, no simulation.

C1 (D-146): `--factors` reaches the nets, `_save` stamps the structure hash and the loader checks it, UI factors refuse
a pack without the public UI fields, `ui.drag_to` (probe source) is supervised, `--split-seed` is independent of
`--seed`; the collect -> `Demos.batch` -> net path and the live rollout share one featurizer (needs the CW extra);
flow noise is a function of (policy seed, env seed, packet index)."""
import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rrp.envs.computerworld import UI_REL_VOCAB  # noqa: E402
from rrp.harness.train import pointer as T  # noqa: E402
from rrp.policies.pointer import LC, LI, NH, NW, WF, PointerGeometry  # noqa: E402

GEOM = PointerGeometry((640, 480), 0.001, 10.0)          # synthetic: the trainers read the env spec's geometry


def _pack(path, *, geometry: bool, seed0: int, n_ep=4, n_tick=8, ui: bool = True):
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
    if geometry and ui:                                        # the public UI fields (`stack_tables`' on-disk layout)
        d["wzlayer"] = rng.integers(0, 3, (n_tab, NW)).astype(np.float32)
        d["wparent"] = rng.integers(-1, 2, (n_tab, NW)).astype(np.int64)
        foc = np.full((n_tab, NW), -1, np.int64)
        foc[:, 0] = 0
        d["wfocusrank"] = foc
        edges = rng.random((n_tab, NW, NW, len(UI_REL_VOCAB))) < 0.05
        d["wuiedges"] = np.stack([np.packbits(e.reshape(-1)) for e in edges])
        drags = np.arange(n_ep) % 2 == 0                        # even episodes drag (handle slot 1 -> target slot 3)
        d["ep_drags"] = drags
        d["ep_drag"] = np.where(drags[:, None], np.array([1, 3]), -1).astype(np.int16)
    np.savez_compressed(path, **d)


@pytest.fixture(autouse=True)
def _geometry(monkeypatch):
    monkeypatch.setattr("rrp.harness.train.pointer.train.pointer_geometry", lambda: GEOM)


def _args(tmp_path, files, **kw):
    return argparse.Namespace(**dict(dict(data=[str(f) for f in files], steps=2, batch=8, lr=3e-4, seed=0, split_seed=0,
                                          device="cpu", log_every=1, split=T.SPLIT_PATH_V2, factors=None), **kw))


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


# ---------------------------------------------------------------------------------------------------- C1
def test_batch_carries_the_public_ui_fields_and_old_packs_are_flagged(tmp_path):
    new, old = tmp_path / "new.npz", tmp_path / "old.npz"
    _pack(new, geometry=True, seed0=10)
    _pack(old, geometry=False, seed0=20)
    data = T.Demos([str(new), str(old)], "cpu", GEOM)
    assert not data.has_ui
    b, _, _ = data.batch(torch.arange(data.N))
    from_new = data.t["tab"][torch.arange(data.N)] < 4
    assert b["wuiedges"].shape == (data.N, NW, NW, len(UI_REL_VOCAB)) and b["wuiedges"].dtype == torch.bool
    assert b["wuiedges"][from_new].any() and not b["wuiedges"][~from_new].any()
    assert (b["wparent"][~from_new] == -1).all() and (b["wfocusrank"][~from_new] == -1).all()
    only_new = T.Demos([str(new)], "cpu", GEOM)
    assert only_new.has_ui
    raw = np.load(new)
    got = only_new.batch(torch.arange(only_new.N))[0]["wuiedges"][0]
    tab0 = int(only_new.t["tab"][0])
    assert np.array_equal(got.numpy(), np.unpackbits(raw["wuiedges"][tab0])[:NW * NW * len(UI_REL_VOCAB)].reshape(
        NW, NW, len(UI_REL_VOCAB)).astype(bool))


def test_split_seed_is_independent_of_seed(tmp_path):
    from rrp.harness.train.pointer.train import setup
    p = tmp_path / "p.npz"
    _pack(p, geometry=True, seed0=10, n_ep=20)
    val = lambda **kw: setup(_args(tmp_path, [p], **{"out": "x", **kw}))[1].val_mask.copy()  # noqa: E731
    assert (val(seed=0) == val(seed=7)).all()                    # a seed sweep keeps ONE held-out set
    assert not (val(split_seed=0) == val(split_seed=1)).all()    # the split seed changes it
    setup(_args(tmp_path, [p], out="x", seed=3))
    r1 = torch.rand(2)
    setup(_args(tmp_path, [p], out="x", seed=3))
    assert torch.equal(r1, torch.rand(2))                        # `--seed` still seeds torch


def test_factors_are_threaded_stamped_and_hash_checked_on_load(tmp_path):
    from rrp.policies.pointer import load_pointer_bundle
    from rrp.policies.relations.base import FactorError, compat_hash, resolve
    pack = tmp_path / "p.npz"
    _pack(pack, geometry=True, seed0=10)
    plain, ui = tmp_path / "plain.pt", tmp_path / "ui.pt"
    T.cmd_bc(_args(tmp_path, [pack], out=str(plain), w_xy=5.0))
    T.cmd_bc(_args(tmp_path, [pack], out=str(ui), w_xy=5.0, factors=["preset:ui"]))
    specs = resolve(["preset:ui"], default="none", family="pointer", training=True)
    assert torch.load(plain, weights_only=False)["versions"]["factors"] == compat_hash(resolve(None, default="none"))
    blob = torch.load(ui, weights_only=False)
    assert blob["versions"]["factors"] == compat_hash(specs) != torch.load(plain, weights_only=False)["versions"]["factors"]
    assert [s["name"] for s in blob["config"]["arch"]["BC"]["factors"]] == [s.name for s in specs]
    bundle = load_pointer_bundle(str(ui))                         # rebuilt WITH the factors, hash matches
    assert {s.name for s in bundle["modules"]["BC"].ctx.specs} == {s.name for s in specs}
    plain_keys = set(load_pointer_bundle(str(plain))["modules"]["BC"].state_dict())
    assert set(bundle["modules"]["BC"].state_dict()) > plain_keys              # the factors' parameters exist and load
    # a tampered hash / arch is refused; a deliberate ablation load is allowed
    blob["versions"]["factors"] = "fx-000000000000"
    torch.save(blob, tmp_path / "bad.pt")
    with pytest.raises(FactorError, match="factor structure"):
        load_pointer_bundle(str(tmp_path / "bad.pt"))
    load_pointer_bundle(str(tmp_path / "bad.pt"), allow_factor_mismatch=True)
    nohash = torch.load(plain, weights_only=False)
    del nohash["versions"]["factors"]
    torch.save(nohash, tmp_path / "nohash.pt")
    with pytest.raises(FactorError):
        load_pointer_bundle(str(tmp_path / "nohash.pt"))


@pytest.mark.slow                                                   # ~20 s on the host (three trainers)
def test_ui_factors_train_all_three_trainers_and_supervise_drag_to(tmp_path):
    pack = tmp_path / "p.npz"
    _pack(pack, geometry=True, seed0=10)
    rep, flow, bc = (tmp_path / f"{k}.pt" for k in ("rep", "flow", "bc"))
    fx = ["preset:ui"]
    T.cmd_rep(_args(tmp_path, [pack], out=str(rep), variant="semfix", dz=16, w_sem=1.0, lv_min=-4.0, beta=3e-3,
                    w_xy=5.0, factors=fx))
    T.cmd_flow(_args(tmp_path, [pack], out=str(flow), target="latent", representation=str(rep), w_sem=0.5, factors=fx))
    T.cmd_bc(_args(tmp_path, [pack], out=str(bc), w_xy=5.0, factors=fx))
    for path in (rep, flow, bc):
        m = torch.load(path, weights_only=False)["metrics"]
        assert np.isfinite(m["fx_probe_ui.drag_to"])              # the probe-source head has a supervised loss term
    from rrp.policies.pointer import load_pointer_bundle
    load_pointer_bundle(str(rep)), load_pointer_bundle(str(flow)), load_pointer_bundle(str(bc))


def test_ui_factors_refuse_a_pack_without_the_public_ui_fields_the_drag_target_and_a_curriculum(tmp_path):
    from rrp.policies.relations.base import FactorError
    old = tmp_path / "old.npz"
    _pack(old, geometry=True, seed0=10, ui=False)
    with pytest.raises(FactorError, match="public UI fields"):
        T.cmd_bc(_args(tmp_path, [old], out=str(tmp_path / "x.pt"), w_xy=5.0, factors=["preset:ui"]))
    T.cmd_bc(_args(tmp_path, [old], out=str(tmp_path / "y.pt"), w_xy=5.0))          # default factors: an old pack still trains
    new = tmp_path / "new.npz"
    _pack(new, geometry=True, seed0=11)
    with pytest.raises(FactorError, match="--curriculum"):                       # mix > 0 needs the relgen stream
        T.cmd_bc(_args(tmp_path, [new], out=str(tmp_path / "z.pt"), w_xy=5.0,
                       factors=['{"name": "ui.same_window", "mix": 0.5}']))
    nodrag = tmp_path / "nodrag.npz"                                              # UI fields but no stored teacher drag
    z = dict(np.load(new))
    z.pop("ep_drag"), z.pop("ep_drags")
    np.savez_compressed(nodrag, **z)
    with pytest.raises(FactorError, match="ep_drag"):
        T.cmd_bc(_args(tmp_path, [nodrag], out=str(tmp_path / "w.pt"), w_xy=5.0, factors=["preset:ui"]))


def test_flow_noise_is_a_function_of_policy_seed_env_seed_and_packet_index():
    from rrp.policies.pointer.runtime import flow_noise
    sh = (4, 1, 16)
    a = flow_noise(0, 5, 0, sh)
    assert np.array_equal(a, flow_noise(0, 5, 0, sh))
    for other in (flow_noise(1, 5, 0, sh), flow_noise(0, 6, 0, sh), flow_noise(0, 5, 1, sh)):
        assert not np.allclose(a, other)


def test_flow_rows_do_not_depend_on_the_batch_and_sample_takes_explicit_noise(tmp_path):
    """Identical z at batch 1 and batch 16 given the same per-row noise (synthetic batches, tiny flow)."""
    from rrp.policies.nets.pointer import PointerFlow
    from rrp.policies.pointer.runtime import flow_noise
    pack = tmp_path / "p.npz"
    _pack(pack, geometry=True, seed0=10)
    data = T.Demos([str(pack)], "cpu", GEOM)
    b, _, _ = data.batch(torch.arange(16))
    torch.manual_seed(0)
    S = PointerFlow(dz=8, D=32, heads=4, layers=1).eval()
    noise = torch.from_numpy(np.stack([flow_noise(0, i, 0, (4, 1, 8)) for i in range(16)]))
    with torch.no_grad():
        z16 = S.sample(b, nfe=3, noise=noise)
        for i in (0, 7, 15):
            z1 = S.sample({k: v[i:i + 1] for k, v in b.items()}, nfe=3, noise=noise[i:i + 1])
            assert torch.allclose(z1, z16[i:i + 1], atol=1e-5)
    with pytest.raises(ValueError, match="noise shape"):
        S.sample(b, nfe=1, noise=noise[:3])


@pytest.mark.computerworld
def test_collect_stores_what_live_rollout_featurizes_and_key_sets_are_equal(tmp_path):
    """One featurizer path (C1): a pack written by `collect_episode` + `write_pack` on a real CW episode, read back by
    `Demos.batch`, has exactly the key set (and, at tick 0, the values) of the live `public_features` +
    `env_widget_table` + `collate_public` the rollout feeds the net -- and stores real UI fields (focus rank, edges)."""
    import random

    from rrp.envs.base import make_env
    from rrp.harness.train.pointer.collect import collect_episode, write_pack
    from rrp.policies.pointer import EventHistory, collate_public, env_widget_table, public_features, screen_half
    task, seed = "cw/fill_form", 0
    ep = collect_episode(task, seed, dart_px=0.0, rng=random.Random(0), max_ticks=400)
    assert ep is not None and ep["tabs"]
    env = make_env("computerworld", task=task, body="cw_pointer", seed=seed)
    geom = PointerGeometry.from_spec(env.spec)
    ep["seed"] = seed
    write_pack(tmp_path / "p.npz", [ep], task, geom)
    data = T.Demos([str(tmp_path / "p.npz")], "cpu", geom)
    assert data.has_ui
    train_b = data.batch(torch.arange(len(ep["rows"])))[0]
    obs = env.observe()
    live = collate_public([public_features(obs, screen_half(env.spec), EventHistory(), 0, table=env_widget_table(env))],
                          "cpu")
    env.close()
    assert set(train_b) == set(live)
    for k, v in live.items():                                    # tick 0 (fresh history): every value and dtype agrees
        assert train_b[k].dtype == v.dtype, k
        if k not in ("wf",):                                     # wf is stored float16
            assert torch.allclose(train_b[k][0].float(), v[0].float()), k
    assert torch.allclose(train_b["wf"][0], live["wf"][0], atol=2e-3)
    assert (train_b["wuiedges"].any() and (train_b["wfocusrank"] == 0).any()
            and (train_b["wzlayer"] != 0).any()), "the pack must store real UI fields"


@pytest.mark.computerworld
def test_pointer_system_i_packets_are_identical_at_batch_1_and_batch_4_and_reset_replays(tmp_path):
    """Flow noise per (policy seed, env seed, packet index): env 0's z is the same alone and inside a batch of 4, the
    second packet differs from the first, `reset` replays the first, and another policy seed changes it."""
    import torch as th

    from rrp.envs.base import make_env
    from rrp.policies.nets.pointer import PointerFlow
    from rrp.policies.pointer import PointerSystemI
    th.manual_seed(0)
    S = PointerFlow(dz=8, D=32, heads=4, layers=1)
    mk = lambda seed=0: PointerSystemI(S, lsv="l", rcv="r", nfe=3, seed=seed)          # noqa: E731
    envs = [make_env("computerworld", task="cw/fill_form", body="cw_pointer", seed=s) for s in range(4)]
    z = lambda p: np.asarray(p.z, np.float32)                                          # noqa: E731
    solo, batch = mk(), mk()
    solo.reset(envs[:1])
    batch.reset(envs)
    a1, b4 = z(solo.packets(envs[:1])[0]), z(batch.packets(envs)[0])
    assert np.allclose(a1, b4, atol=1e-5)
    a2 = z(solo.packets(envs[:1])[0])
    assert not np.allclose(a1, a2)                                                      # packet index advances
    solo.reset(envs[:1])
    assert np.allclose(z(solo.packets(envs[:1])[0]), a1)                                # per-episode replay
    other = mk(seed=1)
    other.reset(envs[:1])
    assert not np.allclose(z(other.packets(envs[:1])[0]), a1)
    for e in envs:
        e.close()
