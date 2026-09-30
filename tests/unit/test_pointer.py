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
