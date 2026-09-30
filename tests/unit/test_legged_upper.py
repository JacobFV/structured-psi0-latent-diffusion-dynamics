"""HX (D13, architecture 14.4): the humanoid system 0 realizes the `upper` group (arms, waist, head) from the packet.

Tiny random-weight models (plumbing and causal routing, no result): the realizer output splits into the `legs` and `upper`
groups by the public node flag, an arm assembly's packet knots move only that arm's joints, and on t1 (Menagerie,
`control="wholebody"`) the latent policy emits both groups from one packet and an edited packet changes the upper command."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rrp.policies.nets import legged_latent as LL

from ._legged_tiny import tiny_bundle
from .test_legged_relations import ARM_L, ARM_R, BODY, DZ, D, batch
from .test_wholebody import _session, registry  # noqa: F401  (registry is a fixture)

# batch(): joints 0-2 leg L, 3-5 leg R, 6 body, 7-8 arm L, 9-10 arm R (assemblies 0..4)
LEGS, UPPER = list(range(6)), list(range(6, 11))


def _flagged(b):
    b["node_static"][..., LL.IS_POLICY_COL] = torch.tensor([1.0] * 6 + [0.0] * 5)
    return b


def test_realizer_groups_split_by_the_policy_flag_and_route_by_assembly():
    torch.manual_seed(0)
    b = _flagged(batch())
    R = LL.LeggedRealizer(dz=DZ, D=D, layers=1).eval()
    z = torch.randn(2, 4, 5, DZ)
    ph = torch.zeros(2)
    with torch.no_grad():
        out = R(z, b, ph)
        g = R.groups(z, b, ph)
        assert LL.REALIZER_GROUPS == ("legs", "upper") and set(g) == {"legs", "upper"}
        assert g["legs"][:, UPPER].abs().max() == 0 and g["upper"][:, LEGS].abs().max() == 0
        assert torch.equal(g["legs"] + g["upper"], out) and g["upper"][:, UPPER].abs().min() > 0
        # routing: an upper node reads only its own assembly's knots and the body assembly's (cross-attention bias) ...
        K, M = 4, 5
        bias = R._route_bias(b, K, M, b["asm_mask"][:, None].expand(2, K, M).reshape(2, K * M), z.device)[0, 0]   # [N, K*M]
        knot_asm = torch.arange(M).repeat(K)
        for node, asm in ((7, ARM_L), (9, ARM_R), (0, 0)):
            reads = set(knot_asm[torch.isfinite(bias[node])].tolist())
            assert reads == {asm, BODY}, (node, reads)
        # ... so the arm's own knots move that arm's joints most (the nodes' self-attention spreads a little to the rest)
        for arm, own, rest in ((ARM_L, [7, 8], LEGS + [6, 9, 10]), (ARM_R, [9, 10], LEGS + [6, 7, 8])):
            z2 = z.clone()
            z2[:, :, arm] += 1.0
            d = (R(z2, b, ph) - out).abs()
            assert d[:, own].min() > 1e-5 and d[:, own].mean() > 3 * d[:, rest].mean()
        z4 = z.clone()
        z4[:, :, BODY] += 1.0
        assert (R(z4, b, ph) - out).abs().min() > 0                      # the body assembly reaches every node


def test_group_masks_partition_the_node_mask():
    b = _flagged(batch())
    b["node_mask"][:, -1] = False                                           # a padded node belongs to no group
    m = LL.group_masks(b)
    assert not (m["legs"] & m["upper"]).any() and torch.equal(m["legs"] | m["upper"], b["node_mask"])
    assert m["legs"].sum() == 12 and m["upper"].sum() == 8


def test_realizer_state_dict_has_no_upper_head():
    """Legs-only checkpoints load unchanged: the groups share the one output head (no new key)."""
    keys = set(LL.LeggedRealizer(dz=DZ, D=D, layers=1).state_dict())
    assert not any("upper" in k for k in keys) and "out.weight" in keys


# ------------------------------------------------------------------ the policy on t1 (Menagerie)
def _policy(tmp_path, **kw):
    import torch as T
    from rrp.policies.legged import LatentLeggedController, LeggedLatentPolicy
    T.set_num_threads(1)
    _, flow = tiny_bundle(tmp_path, upper_trained=bool(kw.get("upper")))
    return LeggedLatentPolicy(LatentLeggedController(flow, T.device("cpu"), nfe=2, seed=3, **kw))


def _first_packet(pol, s, max_ticks=45):
    """Run the policy until it emits its first packet; return (upper command of that tick, the Act, the whole-body env)."""
    for _ in range(max_ticks):
        act = pol.act({0: None})[0]
        r = s.step(act.command)
        assert r.rejected is None
        if act.packet is not None:
            return act
    raise AssertionError("no packet emitted")


@pytest.mark.menagerie
def test_latent_policy_emits_legs_and_upper_from_one_packet(tmp_path, registry):
    s = _session(tmp_path, registry)
    s.reset(0)
    pol = _policy(tmp_path, upper=True)
    assert pol.info.requires.groups == {"legs", "upper"} and pol.info.version.endswith("|upper")
    pol.reset(None, None, [0], envs=[s])
    act = _first_packet(pol, s)
    b = s.binding
    assert set(act.command.groups) == {"legs", "upper"}
    up = np.asarray(act.command.groups["upper"])
    assert up.shape == (len(b.held_act),) and np.isfinite(up).all()
    assert (up >= b.held_lo - 1e-9).all() and (up <= b.held_hi + 1e-9).all()
    assert np.abs(up - b.q0_held).max() > 1e-4                             # realized from the packet, not the default hold
    kinds = [a.handle.split(":")[-1].rstrip("0123456789") for a in act.packet.assemblies]
    assert "arm" in kinds and "body" in kinds and len(act.packet.assemblies) == pol.ad.m.M    # arms join the assembly set


@pytest.mark.menagerie
def test_edited_packet_changes_the_upper_command(tmp_path, registry):
    ups = {}
    for edit in ("none", "zero"):
        s = _session(tmp_path, registry)
        s.reset(0)
        pol = _policy(tmp_path, upper=True, edit=edit, t_edit=0.0)
        pol.reset(None, None, [0], envs=[s])
        ups[edit] = np.asarray(_first_packet(pol, s).command.groups["upper"])
    assert np.abs(ups["none"] - ups["zero"]).max() > 1e-4


@pytest.mark.menagerie
def test_legs_only_policy_leaves_the_upper_group_out_and_upper_needs_wholebody(tmp_path, registry):
    s = _session(tmp_path, registry)
    s.reset(0)
    pol = _policy(tmp_path)                                                # upper=False: the legs-only contract
    pol.reset(None, None, [0], envs=[s])
    act = _first_packet(pol, s)
    assert set(act.command.groups) == {"legs"} and pol.info.requires.groups == {"legs"}
    legs_env = _session(tmp_path, registry, control="legs")
    up = _policy(tmp_path, upper=True)
    with pytest.raises(ValueError, match="control"):
        up.reset(None, None, [0], envs=[legs_env])
    with pytest.raises(ValueError, match="upper=True"):
        from rrp.policies.legged import LatentLeggedController, LeggedLatentPolicy
        _, flow = tiny_bundle(tmp_path, upper_trained=True)
        LeggedLatentPolicy(LatentLeggedController(flow, torch.device("cpu"), upper=True), oracle=True)
