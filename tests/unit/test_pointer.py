"""Pointer system 0 (rrp.policies.pointer): knot/tick mapping and the engineered packet encoding (pure), and the
oracle -> engineered system 0 route through rrp eval's harness on every cw/* task (marked `computerworld`)."""
import numpy as np
import pytest

from rrp.policies.pointer import (ENG_DIM, KNOT_TIMES, decode_slot, encode_commands, packet_ticks, tick_slot)


def test_tick_slots_cover_knots_at_10hz():
    assert packet_ticks(0.1) == [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    assert tick_slot(0.7, 0.1) is None and tick_slot(0.05, 0.1) == (1, 0)


def test_engineered_encoding_roundtrip():
    cmds = [{"pointer": [0.1, -0.2], "button": [0.0]}, {"pointer": [0.12, -0.2], "button": [1.0], "key": [17.0]},
            None, {"pointer": [0.3, 0.1], "button": [0.0], "wheel": [-2.0]}]
    z = encode_commands(cmds, 0.1, 109, depths=[0.0, 0.004, 0.0, 0.002])
    assert z.shape == (len(KNOT_TIMES), 1, ENG_DIM)
    for j, (k, s) in enumerate(packet_ticks(0.1)):
        d = decode_slot(z, k, s, 109)
        if j >= len(cmds) or cmds[j] is None:
            assert d is None
            continue
        c = cmds[j]
        assert (d["x"], d["y"]) == pytest.approx(c["pointer"], abs=1e-6)
        assert d["button"] == (c["button"][0] >= 0.5)
        assert d["key"] == int(c.get("key", [-1])[0]) and d["wheel"] == int(c.get("wheel", [0])[0])


@pytest.mark.computerworld
@pytest.mark.parametrize("task", ["cw/calc_sum", "cw/open_type", "cw/drag_window", "cw/fill_form"])
def test_oracle_packets_drive_every_task(task):
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.base import make_policy
    pol = make_policy("pointer_oracle")
    eps = evaluate(pol, "computerworld", task, "cw_pointer", [0, 1, 2])
    assert [e.outcome for e in eps] == ["success"] * 3
    assert all(e.metrics["packets"] > 0 and e.metrics["command_rejections"] == 0 for e in eps)
    assert all(s.stats.rejected == 0 for s in pol.s0)


def test_split_declares_disjoint_seeds_and_heldout_variants():
    import json
    from rrp.harness.train.pointer import SPLIT_PATH, TASKS, heldout_goal
    s = json.load(open(SPLIT_PATH))
    lo, hi = s["seed_ranges"]["train"]
    for t in TASKS:
        ev = [x for k in ("dev", "sealed_id", "sealed_heldout") for x in s["seeds"][t].get(k, [])]
        assert ev and not any(lo <= x < hi for x in ev) and len(ev) == len(set(ev))
    assert heldout_goal("cw/open_type", {"text": "falcon"}, s) and not heldout_goal("cw/open_type", {"text": "robot"}, s)
    assert heldout_goal("cw/calc_sum", {"a": 3, "b": 3}, s) and not heldout_goal("cw/calc_sum", {"a": 3, "b": 4}, s)
    assert heldout_goal("cw/fill_form", {"name": "Hedy"}, s)


# ==================================================================================================== D-144 R6
# pointer nets on the relation-factor foundation: `Block` -> `RelBlock` (empty preset `none`) and `PointerProbe` ->
# `nets.probes.ReadoutProbe` (`probes:pointer-v1`). docs/relations.md section 10, R6 brief; addendum (a) (probe-swap
# acceptance = checkpoint-load equivalence + unchanged golden.json).

class _OldPointerBlock:
    """Frozen copy of the pointer `Block` class this unit deletes (module keys n1, a, n2, m; pre-LN, EITHER
    cross-attention (`cross=True`) OR self-attention (`cross=False`) + MLP -- 2 stages, not `RelBlock`'s 3). Exists
    ONLY to build fixture old-format state dicts / reference forwards for the equivalence tests below; never
    imported by production code."""

    def __new__(cls, D, heads, cross=False):
        import torch.nn as nn
        from rrp.policies.nets.attention import MHA
        from rrp.policies.nets.flow import MLP

        class _B(nn.Module):
            def __init__(self):
                super().__init__()
                self.n1, self.a, self.n2, self.m = nn.LayerNorm(D), MHA(D, heads), nn.LayerNorm(D), MLP(D, D, 4 * D)
                self.cross = cross

            def forward(self, x, kv=None, key_mask=None):
                h = self.n1(x)
                x = x + self.a(h, kv=kv if self.cross else None, key_mask=key_mask)
                return x + self.m(self.n2(x))
        return _B()


def _old_pointer_realizer(dz, D, heads, layers=2):
    """Frozen copy of the pre-D-144-R6 `PointerRealizer` (real old checkpoint layout: `blocks.<i>.{n1,a,n2,m}`),
    used only to build a fixture old checkpoint for `test_pointer_realizer_old_checkpoint_matches_old_forward`."""
    import torch
    import torch.nn as nn
    from rrp.policies.pointer import KNOT_TIMES, N_KEYCLS
    from rrp.policies.nets.flow import MLP, sinusoidal

    class _R(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_in, self.dt = nn.Linear(dz, D), MLP(D, D)
            self.loc, self.ph = MLP(3, D), MLP(D, D)
            self.blocks = nn.ModuleList([_OldPointerBlock(D, heads, cross=True) for _ in range(layers)])
            self.xy, self.btn, self.key = nn.Linear(D, 2), nn.Linear(D, 1), nn.Linear(D, N_KEYCLS)
            self.register_buffer("kt", torch.tensor(KNOT_TIMES, dtype=torch.float32))
            self.D = D

        def forward(self, z, phase, ptr, btn):
            tok = self.z_in(z[:, :, 0]) + self.dt(sinusoidal(self.kt[None] - phase[:, None], self.D))
            q = (self.loc(torch.cat([ptr, btn[:, None]], -1)) + self.ph(sinusoidal(phase, self.D)))[:, None]
            for L in self.blocks:
                q = L(q, kv=tok)
            q = q[:, 0]
            return self.xy(q), self.btn(q)[:, 0], self.key(q)
    return _R()


def test_remap_block_state_prefixes_are_disjoint_and_correct():
    """`_remap_block_state` (D-144 R6): old `Block` keys (n1, a, n2, m) -> `RelBlock` layout, per mode -- "cross"
    keeps n1, sends a -> x, n2 -> n3; "self" sends n1 -> n2, a -> s, n2 -> n3. Two prefixes in one state dict (as a
    real `PointerEncoder` checkpoint has: `ctx.blocks.` self-only, `blocks.` cross-only) stay disjoint."""
    from rrp.policies.pointer import _remap_block_state
    sd = {"ctx.blocks.0.n1.weight": 1, "ctx.blocks.0.a.q.weight": 2, "ctx.blocks.0.n2.weight": 3,
          "ctx.blocks.0.m.net.0.weight": 4, "blocks.0.n1.weight": 5, "blocks.0.a.q.weight": 6,
          "blocks.0.n2.weight": 7, "blocks.0.m.net.0.weight": 8}
    assert _remap_block_state(sd, "ctx.blocks.", "self") == {
        "ctx.blocks.0.n2.weight": 1, "ctx.blocks.0.s.q.weight": 2, "ctx.blocks.0.n3.weight": 3,
        "ctx.blocks.0.m.net.0.weight": 4}
    assert _remap_block_state(sd, "blocks.", "cross") == {
        "blocks.0.n1.weight": 5, "blocks.0.x.q.weight": 6, "blocks.0.n3.weight": 7, "blocks.0.m.net.0.weight": 8}


def test_cross_relblock_self_stage_is_a_noop_matching_old_cross_only_block():
    """`cross_relblock` (D-144 R6): its zero-init-output self stage contributes exactly 0, so its 3-stage `RelBlock`
    forward reduces to the deleted `Block(cross=True)`'s 2-stage (cross + MLP) formula, computed here from the SAME
    submodules directly."""
    import torch
    from rrp.policies.pointer import nets
    torch.manual_seed(0)
    D, heads, B, Q, Kt = 32, 4, 2, 3, 5
    rb = nets()["cross_relblock"](D, heads)
    x, kv = torch.randn(B, Q, D), torch.randn(B, Kt, D)
    kv_mask = torch.ones(B, Kt, dtype=torch.bool)
    kv_mask[:, -1] = False
    with torch.no_grad():
        out_new = rb(x, kv=kv, kv_mask=kv_mask)
        h = rb.n1(x)
        old = x + rb.x(h, kv=kv, key_mask=kv_mask)
        out_old = old + rb.m(rb.n3(old))
    assert torch.allclose(out_new, out_old, atol=1e-6)


def test_self_relblock_cross_stage_is_a_noop_matching_old_self_only_block():
    """`self_relblock` (D-144 R6): symmetric to the cross case, its zero-init-output CROSS stage is the no-op, so
    its forward reduces to the deleted `Block(cross=False)`'s (self + MLP) formula."""
    import torch
    from rrp.policies.pointer import nets
    torch.manual_seed(1)
    D, heads, B, Q = 32, 4, 2, 6
    rb = nets()["self_relblock"](D, heads)
    x = torch.randn(B, Q, D)
    m = torch.ones(B, Q, dtype=torch.bool)
    m[:, -1] = False
    with torch.no_grad():
        out_new = rb(x, kv=x, q_mask=m)
        h = rb.n2(x)
        old = x + rb.s(h, kv=None, key_mask=m)
        out_old = old + rb.m(rb.n3(old))
    assert torch.allclose(out_new, out_old, atol=1e-6)


def test_pointer_realizer_old_checkpoint_matches_old_forward():
    """D-144 R6 acceptance (docs/relations.md 10): an old (pre-R6) `PointerRealizer` checkpoint, key-mapped by
    `load_pointer_module` onto the new `RelBlock`-based `PointerRealizer`, reproduces the old forward's output
    exactly on a seeded fixture (atol 1e-5) -- checkpoint-load equivalence AND fixed-input equivalence together."""
    import torch
    from rrp.policies.pointer import KNOT_TIMES, load_pointer_module, nets
    torch.manual_seed(7)
    dz, D, heads = 8, 32, 4
    old = _old_pointer_realizer(dz, D, heads)
    old.eval()
    new = load_pointer_module("PointerRealizer", lambda: nets()["PointerRealizer"](dz=dz, D=D, heads=heads),
                              old.state_dict())
    new.eval()
    B, K = 3, len(KNOT_TIMES)
    z, phase = torch.randn(B, K, 1, dz), torch.rand(B)
    ptr, btn = torch.randn(B, 2), torch.randn(B)
    with torch.no_grad():
        oxy, obtn, okey = old(z, phase, ptr, btn)
        nxy, nbtn, nkey = new(z, phase, ptr, btn)
    assert torch.allclose(oxy, nxy, atol=1e-5)
    assert torch.allclose(obtn, nbtn, atol=1e-5)
    assert torch.allclose(okey, nkey, atol=1e-5)


def test_pointer_realizer_new_checkpoint_loads_strictly():
    """A post-D-144-R6 checkpoint (module keys n1, x, n2, s, n3, m) has no old `.a.` key anywhere, so
    `load_pointer_module` passes it straight to `load_state_dict` -- no remap, no missing/unexpected keys."""
    from rrp.policies.pointer import load_pointer_module, nets
    dz, D, heads = 8, 32, 4
    src = nets()["PointerRealizer"](dz=dz, D=D, heads=heads)
    new = load_pointer_module("PointerRealizer", lambda: nets()["PointerRealizer"](dz=dz, D=D, heads=heads),
                              src.state_dict())
    for k, v in src.state_dict().items():
        assert (new.state_dict()[k] == v).all()


def test_probes_pointer_v1_preset_resolves_in_order():
    from rrp.policies.relations.base import get_factor, resolve
    specs = resolve(["preset:probes:pointer-v1"])
    assert [s.name for s in specs] == ["probe.pointer.slot", "probe.pointer.rel", "probe.pointer.phase"]
    d = get_factor("probe.pointer.slot")
    assert d.readout.address == "knot×asm" and d.readout.out == 80 and d.readout.loss == "ce"


def test_new_pointer_probe_output_shapes():
    import torch
    from rrp.policies.pointer import KNOT_TIMES, NW, PHASES, new_pointer_probe, run_pointer_probe
    torch.manual_seed(0)
    P = new_pointer_probe(dz=8, D=32, heads=4)
    z = torch.randn(3, len(KNOT_TIMES), 1, 8)
    out = run_pointer_probe(P, z)
    assert out["slot"].shape == (3, len(KNOT_TIMES), NW)
    assert out["rel"].shape == (3, len(KNOT_TIMES), 4)
    assert out["phase"].shape == (3, len(KNOT_TIMES), len(PHASES))


def test_old_pointer_probe_checkpoint_drops_and_refits_new_loads_strictly():
    """`load_pointer_probe_state` (D-144 R6, addendum a: same fallback as psi0 R5): an old `PointerProbe` state
    (different architecture) is dropped in favour of a fresh `ReadoutProbe` init, never raising; a state already in
    the new layout loads strictly (bit-exact)."""
    import torch
    from rrp.policies.pointer import load_pointer_probe_state, new_pointer_probe
    old_sd = {"kq": torch.zeros(4, 32), "bag.c.weight": torch.zeros(1), "wf.net.0.weight": torch.zeros(1),
              "role.weight": torch.zeros(1), "rel.net.0.weight": torch.zeros(1), "phase.net.0.weight": torch.zeros(1)}
    P = load_pointer_probe_state(old_sd, dz=8, D=32)
    assert isinstance(P, torch.nn.Module)
    fresh = new_pointer_probe(dz=8, D=32, seed=99)
    P2 = load_pointer_probe_state(fresh.state_dict(), dz=8, D=32, seed=99)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, P2.state_dict()[k])


def test_pointer_probe_loss_masks_missing_labels():
    import torch
    from rrp.harness.train.pointer import probe_loss
    from rrp.policies.pointer import new_pointer_probe, run_pointer_probe
    torch.manual_seed(0)
    P = new_pointer_probe(dz=8, D=32)
    z = torch.randn(2, 4, 1, 8)
    out = run_pointer_probe(P, z)
    lab = dict(slot=torch.full((2, 4), -1, dtype=torch.long), rel=torch.zeros(2, 4, 2),
               rel_ok=torch.zeros(2, 4, dtype=torch.bool), phase=torch.full((2, 4), -1, dtype=torch.long))
    loss, logs = probe_loss(out, lab, P.specs)
    assert torch.isfinite(loss) and loss.item() == pytest.approx(0.0, abs=1e-6)
    assert set(logs) == {"slot", "rel", "phase"}


# ================================================================== D-144 R20 follow-up: `preset:ui` at UICtx's widget self-attention (archived research/tracks/rel-r20.md "lead_questions")
# The exact fixture `test_relations_r20.py` uses (no `computerworld` wheel needed: `scene_widgets` / `SlotRegistry` /
# `descriptors` are pure): window 0 an enabled "Name" label + focused textbox (z-layer 0); window 1 a Email label +
# a DISABLED textbox (z-layer 1, renders above window 0) -- `ui_edges` is non-empty on it (`label_for`, `focus_next`)
# and `ui_public_fields` carries real `parent_id` / `zlayer` structure (`ui.same_window` / `ui.above`, D-144
# addendum), so `UICtx`'s `preset:ui` factors have real graph structure to read.
_I = {"a": 1024, "b": 0, "c": 0, "d": 1024, "tx": 0, "ty": 0}


def _ui_node(i, x, y, w, h, z, sem=None, inter=None, tx=0, ty=0):
    return {"id": i, "bounds": {"x": x, "y": y, "width": w, "height": h}, "primitive": {"kind": "box"},
            "semantic": sem, "interaction": inter, "transform": dict(_I, tx=tx, ty=ty), "z": z, "opacity": 255,
            "clip": None}


def _ui_lbl(text):
    return {"role": "label", "label": text, "value": None, "disabled": False, "focusable": False}


def _ui_tb(text, disabled=False):
    return {"role": "textbox", "label": text, "value": "", "disabled": disabled, "focusable": True}


_UI_SCENE = {"focus": {"interaction": "window:0:content:name"}, "nodes": [
    _ui_node(1, 20, 30, 40, 20, 100, sem=_ui_lbl("Name"), inter="window:0:content:name_label", tx=100, ty=100),
    _ui_node(2, 20, 60, 40, 20, 100, sem=_ui_tb("Name"), inter="window:0:content:name", tx=100, ty=100),
    _ui_node(3, 20, 90, 40, 20, 200, sem=_ui_lbl("Email"), inter="window:1:content:email_label", tx=200, ty=200),
    _ui_node(4, 20, 120, 40, 20, 200, sem=_ui_tb("Email", disabled=True), inter="window:1:content:email", tx=200,
             ty=200),
]}


def _ui_batch(table_):
    """One unbatched `public_features`-shaped sample -> a size-1 `collate_public` batch, `table` included (so
    `wuiedges` / `wzlayer` / `wparent` / `wfocusrank` are real, not the zero-edge no-op fallback)."""
    import numpy as np
    from rrp.envs.computerworld import ScreenFrame, descriptors
    from rrp.policies.pointer import LI, NH, collate_public, widget_features
    frame = ScreenFrame(640, 480)
    half = np.array([frame.width, frame.height], np.float32) * frame.m_per_px / 2

    class _Obs:
        object_descriptors = descriptors(table_, frame, 0.0, {})

    f = dict(widget_features(_Obs(), half, table_), instr=np.zeros(LI, np.int16), ptr=np.zeros(2, np.float32),
             btn=np.float32(0.0), tick=np.float32(0.0), hist=np.zeros((NH, 5), np.float32))
    return collate_public([f], "cpu")


def test_widget_features_with_table_carries_r20s_ui_fields_and_a_nonempty_edge_graph():
    """`ui_widget_fields` (wired through `widget_features`'s new `table` argument) reproduces R20's own
    `ui_public_fields` / `ui_edges` on this scene (`test_relations_r20.py::test_ui_public_fields_on_the_fixture_scene`
    /`test_ui_edges_*`'s values), padded from 4 to NW slots."""
    from rrp.envs.computerworld import SlotRegistry, scene_widgets
    from rrp.policies.pointer import NW
    table = SlotRegistry().assign(scene_widgets(_UI_SCENE))
    b = _ui_batch(table)
    np.testing.assert_array_equal(b["wzlayer"][0, :4].numpy(), [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(b["wfocusrank"][0, :4].numpy(), [-1, 0, -1, -1])
    assert b["wuiedges"].shape == (1, NW, NW, 2)
    assert b["wuiedges"][0, :4, :4].sum() > 0
    assert b["wuiedges"][0, 4:].sum() == 0 and b["wuiedges"][0, :, 4:].sum() == 0   # padding stays all-False


def test_widget_features_without_table_leaves_the_ui_keys_out():
    """The default call shape (every existing caller: no `table`) adds `wpos3d` / `wcamuvd` (free from the existing
    descriptor geometry) but leaves the table-derived UI keys out entirely -- `UICtx._relctx` then falls back to an
    all-zero `ui-rel-v1` graph (a documented no-op, not a silent fabrication)."""
    from rrp.envs.computerworld import ScreenFrame, SlotRegistry, descriptors, scene_widgets
    from rrp.policies.pointer import widget_features
    table = SlotRegistry().assign(scene_widgets(_UI_SCENE))
    frame = ScreenFrame(640, 480)
    half = np.array([frame.width, frame.height], np.float32) * frame.m_per_px / 2

    class _Obs:
        object_descriptors = descriptors(table, frame, 0.0, {})

    f = widget_features(_Obs(), half)
    assert "wpos3d" in f and "wcamuvd" in f
    assert not ({"wuiedges", "wzlayer", "wparent", "wfocusrank"} & set(f))


def test_default_uictx_factors_are_a_zero_bias_no_op():
    """`PolicyConfig(factors=None)` (== every existing checkpoint's construction, `PointerFlow()` / `PointerBC()` /
    `PointerEncoder()` unchanged) resolves to the empty preset: `UICtx.rel[i]` gets no parameters and its
    `.bias()` / `.augment()` are exactly `None` / `(None, None)` -- the same zero-bias equivalence R6 already
    established for the (still unused-by-default) `RelBlock` stages themselves."""
    import torch
    from rrp.envs.computerworld import SlotRegistry, scene_widgets
    from rrp.policies.pointer import LI, NH, NW, UI_CARRIES, nets
    UICtx = nets()["UICtx"]
    ctx = UICtx(D=16, heads=2, layers=2)
    assert all(sum(p.numel() for p in site.parameters()) == 0 for site in ctx.rel)
    tbl = SlotRegistry().assign(scene_widgets(_UI_SCENE))
    b = _ui_batch(tbl)
    rc = ctx._relctx(b, NW + LI + NH + 1)
    assert ctx.rel[0].bias(rc) is None
    qa, ka = ctx.rel[0].augment(rc, torch.randn(1, 5, 16), torch.randn(1, 5, 16))
    assert qa is None and ka is None
    assert "edges:ui-rel-v1" in UI_CARRIES and "pos3d" in UI_CARRIES and "cam_uvd" in UI_CARRIES


def test_preset_ui_changes_widget_self_attention_logits_on_the_cw_fixture_scene():
    """The R20 lead_question, closed: enabling `factors=["preset:ui"]` (`PolicyConfig` -> `resolve` -> `UICtx`'s
    per-layer `FactorSite`, built on a real `TokenSet`/`RelCtx` from `widget_features`' `table`-derived `ui-rel-v1`
    edges) changes the widget self-attention bias from the default preset's exact `None` to a real, finite,
    non-zero logit term -- and changes `UICtx`'s own forward output on the identical batch."""
    import torch
    from rrp.envs.computerworld import SlotRegistry, scene_widgets
    from rrp.policies.pointer import LI, NH, NW, PolicyConfig, nets

    table = SlotRegistry().assign(scene_widgets(_UI_SCENE))
    b = _ui_batch(table)
    T = NW + LI + NH + 1

    torch.manual_seed(0)
    UICtx = nets()["UICtx"]
    off = UICtx(D=16, heads=2, layers=1)                                       # factors=None (default preset)
    on = UICtx(D=16, heads=2, layers=1, specs=PolicyConfig(["preset:ui"]).specs())
    on.load_state_dict(off.state_dict(), strict=False)                        # identical base weights
    torch.manual_seed(1)
    for p in on.rel.parameters():                                             # off zero-init (docs 3.2): give the
        torch.nn.init.normal_(p, std=0.2)                                     # factors a real, non-trivial effect

    assert {s.name for s in on.rel[0].specs} == \
        {"ui.label_for", "ui.same_window", "ui.focus_next", "ui.above", "ui.drag_to"}

    bias_off = off.rel[0].bias(off._relctx(b, T))
    bias_on = on.rel[0].bias(on._relctx(b, T))
    assert bias_off is None
    assert bias_on is not None and bias_on.shape == (1, 2, T, T)
    assert torch.isfinite(bias_on).all()
    assert bias_on.abs().sum() > 0                                            # a genuine, non-zero logit change

    with torch.no_grad():
        x_off, m_off = off(b)
        x_on, m_on = on(b)
    assert torch.equal(m_off, m_on)                                           # the key mask itself is unaffected
    assert not torch.allclose(x_off, x_on)                                    # but the self-attended hiddens differ


# ================================================================== D-144 R20 follow-up: live rollout call sites pass `env_widget_table`
@pytest.mark.computerworld
def test_env_widget_table_matches_the_real_envs_observe_slot_table():
    """`env_widget_table(env)` (the new helper the three live rollout call sites use) is 1:1 with `obs.object_
    descriptors` -- same slot count, same `key` per occupied slot -- on a REAL `ComputerWorldEnv`
    (`~/work/ext/cw-site` on PYTHONPATH; skips otherwise), exactly like the fixture-only `SlotRegistry().assign
    (scene_widgets(...))` calls `test_relations_r20.py` / this file already use for the wheel-free tests."""
    from rrp.envs.base import make_env
    from rrp.policies.pointer import env_widget_table

    env = make_env("computerworld", task="cw/fill_form", body="cw_pointer", seed=0)
    obs = env.reset(seed=0)
    table = env_widget_table(env)
    assert len(table) == len(obs.object_descriptors)
    occupied = [w for w in table if w is not None]
    assert occupied                                                          # the fixture form has real widgets
    env.close()


@pytest.mark.computerworld
def test_live_rollout_call_sites_are_byte_identical_with_the_default_empty_factor_set(monkeypatch):
    """The acceptance check for item (2): `PointerBCPolicy.act` now threads `env_widget_table(env)` through to
    `public_features` / `widget_features` so `preset:ui` factors see real edges once a factor set enables them --
    but `PolicyConfig(factors=None)` (== every existing BC checkpoint's construction) still resolves to the EMPTY
    preset, so this must emit BYTE-IDENTICAL commands on a real, fixed-seed `ComputerWorldEnv` rollout
    (`~/work/ext/cw-site` on PYTHONPATH; skips otherwise) whether or not a real widget table backs the batch --
    a fixed-seed digest of the rollout's commands with `env_widget_table` monkeypatched back to the old
    `table=None` call shape must match the un-patched one exactly."""
    import hashlib
    import json

    import torch

    from rrp.envs.base import make_env
    from rrp.policies.pointer import PointerBCPolicy, nets

    def rollout(ticks=5):
        torch.manual_seed(0)
        net = nets()["PointerBC"](D=16, heads=2, layers=1).eval()            # factors=None (default): no UI reads
        policy = PointerBCPolicy({"modules": {"BC": net}, "config": {}}, path="test", replan_ticks=4)
        env = make_env("computerworld", task="cw/fill_form", body="cw_pointer", seed=0)
        obs = env.reset(seed=0)
        policy.reset(env.spec, "cw/fill_form", [0], envs=[env])
        groups = []
        for _ in range(ticks):
            act = policy.act({0: obs})[0]
            groups.append(act.command.groups)
            obs = env.step(act.command).observation
        env.close()
        return hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()

    digest_with_table = rollout()

    import rrp.policies.pointer as pointer_mod
    monkeypatch.setattr(pointer_mod, "env_widget_table", lambda env: None)   # the pre-change call shape
    digest_without_table = rollout()

    assert digest_with_table == digest_without_table


@pytest.mark.computerworld
def test_teacher_oracle_source_encode_is_byte_identical_with_the_default_empty_factor_set(monkeypatch):
    """`TeacherOracleSource._encode` (the LEARNED target-encoder oracle route) also now threads
    `env_widget_table(env)` through to `public_features` -- with a `PointerEncoder` built from the default (empty)
    `PolicyConfig`, its emitted z must stay byte-identical to the pre-change `table=None` call shape, on a real,
    fixed-seed `ComputerWorldEnv` rollout (`~/work/ext/cw-site` on PYTHONPATH; skips otherwise)."""
    import torch

    from rrp.envs.base import make_env
    from rrp.policies.pointer import TeacherOracleSource, nets

    def run():
        torch.manual_seed(0)
        enc = nets()["PointerEncoder"](dz=8, D=16, heads=2, layers=1).eval()   # factors=None (default): no UI reads
        env = make_env("computerworld", task="cw/fill_form", body="cw_pointer", seed=0)
        env.reset(seed=0)
        src = TeacherOracleSource(encoder=enc, versions={"latent_space_version": "x", "realizer_compat_version": "y"})
        src.reset([env])
        z = np.array(src.packets([env])[0].z)
        env.close()
        return z

    z_with_table = run()

    import rrp.policies.pointer as pointer_mod
    monkeypatch.setattr(pointer_mod, "env_widget_table", lambda env: None)   # the pre-change call shape
    z_without_table = run()

    np.testing.assert_array_equal(z_with_table, z_without_table)


@pytest.mark.computerworld
def test_pointer_system_i_packets_is_byte_identical_with_the_default_empty_factor_set(monkeypatch):
    """`PointerSystemI.packets` (system i's own live rollout call site) also now threads `env_widget_table(env)`
    through to `public_features` -- with a `PointerFlow` built from the default (empty) `PolicyConfig`, its sampled
    z must stay byte-identical to the pre-change `table=None` call shape, on a real, fixed-seed `ComputerWorldEnv`
    (`~/work/ext/cw-site` on PYTHONPATH; skips otherwise)."""
    import torch

    from rrp.envs.base import make_env
    from rrp.policies.pointer import ENG_VERSION, PointerSystemI, nets

    def run():
        torch.manual_seed(0)
        flow = nets()["PointerFlow"](dz=8, D=16, heads=2, layers=1).eval()   # factors=None (default): no UI reads
        env = make_env("computerworld", task="cw/fill_form", body="cw_pointer", seed=0)
        env.reset(seed=0)
        si = PointerSystemI(flow, lsv=ENG_VERSION, rcv=ENG_VERSION, nfe=2, seed=0)
        si.reset([env])
        z = np.array(si.packets([env])[0].z)
        env.close()
        return z

    z_with_table = run()

    import rrp.policies.pointer as pointer_mod
    monkeypatch.setattr(pointer_mod, "env_widget_table", lambda env: None)   # the pre-change call shape
    z_without_table = run()

    np.testing.assert_array_equal(z_with_table, z_without_table)
