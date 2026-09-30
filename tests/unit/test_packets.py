"""W5 dedup: ladder.make_packet == packets.arm_packet (layout-insensitive), and
the device helper keeps both old behaviours."""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest


def _fakes():
    asm = [NS(kind="gripper", frame=NS(link="hand")), NS(kind="arm", frame=NS(link="base"))]
    f = NS(spec=NS(assemblies=asm, spec_hash="0123456789abcdef"))
    o = NS(object_descriptors=[NS(slot=0), NS(slot=1)], observation_id="obs:0123456789abcdef")
    s = NS(data=NS(time=1.25), runtime=NS(graph_version=1, runtime_version=1), observe=lambda: o)
    return f, s, o


def test_make_packet_equals_arm_packet_on_strided_z():
    from rrp.harness.eval.ladder import make_packet
    from rrp.policies.packets import arm_packet
    f, s, o = _fakes()
    big = np.arange(4 * 2 * 16, dtype=np.float64).reshape(4, 2, 16) / 7.0
    z = big[:, :1, ::2]                                    # non-contiguous view
    kw = dict(lsv="ls-a", rcv="rz-b", knot_times=[0.1, 0.2, 0.3, 0.4])
    a = make_packet(s, f, z, kw["knot_times"], "ls-a", "rz-b", 0.8, source="target_encoder_oracle", policy_version="v")
    b = arm_packet(f, s, o, z, **kw, source="target_encoder_oracle", name="v", validity=0.8)
    for p in (b,):
        da, dp = a.model_dump(), p.model_dump()
        for d in (da, dp):
            d.pop("generated_at")
            d["z"] = np.asarray(d["z"]).tolist()
        assert da == dp
    assert np.asarray(a.z).dtype == np.float32 and np.array_equal(np.asarray(a.z), z.astype(np.float32))


def test_select_device(monkeypatch):
    import torch
    import rrp.ops.workload as w
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom():
        raise RuntimeError("cap")
    monkeypatch.setattr(w, "apply_cap", boom)
    with pytest.raises(RuntimeError):
        w.select_device(on_cap_error="raise")
    assert w.select_device(on_cap_error="ignore").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert w.select_device().type == "cpu"


# ---------------------------------------------------------------- D-146 P3: one packet-edit registry, Ψ₀ policy surface
def test_edit_registry_numpy_torch_agree_and_never_mutate():
    import torch
    from rrp.policies import packets as P
    z = np.random.default_rng(0).normal(size=(5, 3, 4)).astype(np.float32)
    mean = np.full((5, 3, 4), 0.5, np.float32)
    cases = dict(mean_packet=dict(mean=mean), zero_slot=dict(slot=1), swap_assembly=dict(a=0, b=2))
    assert set(cases) == set(P.EDITS)
    for name, kw in cases.items():
        z0 = z.copy()
        out_np = P.apply_edit(name, z, **kw)
        out_t = P.apply_edit(name, torch.from_numpy(z), **{k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                                                          for k, v in kw.items()})
        assert np.array_equal(z, z0) and not np.array_equal(out_np, z)              # pure, and it does something
        assert np.array_equal(out_np, out_t.numpy())
    assert (P.apply_edit("zero_slot", z, slot=1)[:, 1] == 0).all() and np.array_equal(P.apply_edit("zero_slot", z, slot=1)[:, 0], z[:, 0])
    sw = P.apply_edit("swap_assembly", z, a=0, b=2)
    assert np.array_equal(sw[:, 0], z[:, 2]) and np.array_equal(sw[:, 2], z[:, 0]) and np.array_equal(sw[:, 1], z[:, 1])
    with pytest.raises(KeyError):
        P.apply_edit("nope", z)


def test_chunk_hook_and_tensor_hook_apply_the_same_edit_and_label_the_packet():
    from rrp.policies import packets as P
    f, s, o = _fakes()
    f.spec.assemblies.append(NS(kind="gripper", frame=NS(link="hand2")))
    z = np.random.default_rng(1).normal(size=(4, 2, 8)).astype(np.float32)
    p = P.arm_packet(f, s, o, z, lsv="ls", rcv="rz", knot_times=[0.1, 0.2, 0.3, 0.4], source="learned", name="v",
                     sampling=dict(nfe=10))
    q = P.chunk_hook("swap_assembly", a=0, b=1)(0, p)
    assert np.array_equal(q.z, P.tensor_hook("swap_assembly", a=0, b=1)(0, z)) and np.array_equal(q.z, z[:, ::-1])
    assert q.source == "debug" and q.sampling == dict(nfe=10, intervention="swap_assembly") and p.source == "learned"
    assert q.assemblies == p.assemblies and np.array_equal(p.z, z)                  # original packet untouched


def _fake_ours(arm, tmp_z=None):
    """An OursModel without the VLM / upstream config: fixed VLM features, recording heads."""
    import torch
    from rrp.policies.psi0 import OursModel
    seen = {}

    class Head:
        z_mean = torch.full((5, 6, 3), 0.25)

        def sample(self, b, nfe=10, generator=None):
            seen["b"] = b
            return torch.zeros(b["state0"].shape[0], 30, 36)

        def sample_z(self, b, nfe=10, generator=None):
            seen["b"] = b
            return torch.arange(5 * 6 * 3, dtype=torch.float32).reshape(1, 5, 6, 3).repeat(b["state0"].shape[0], 1, 1, 1)

        def realize(self, z, state, phase):
            seen["z"] = z
            return torch.zeros(z.shape[0], 30, 36)
    m = object.__new__(OursModel)
    m.arm, m.head, m.entity_override, m.packet_hook, m.packet_edit, m.last_z, m.generated_z = arm, Head(), None, None, None, None, None
    m.vlm_features = lambda obs, ins: (torch.ones(len(ins), 7, 4), torch.ones(len(ins), 7, dtype=torch.bool),
                                       torch.zeros(len(ins), 7, dtype=torch.bool))
    return m, seen


def test_direct_and_structured_arms_receive_identical_inputs():
    import torch
    states = torch.arange(2 * 3 * 40, dtype=torch.float32).reshape(2, 3, 40)
    got = {}
    for arm in ("direct", "structured"):
        m, seen = _fake_ours(arm)
        m.predict_action([["img"]] * 2, states, ["pick up the cup", "pick up the box"])
        got[arm] = seen["b"]
    a, b = got["direct"], got["structured"]
    assert set(a) == set(b) == {"hidden", "mask", "ent", "state0"}
    assert all(torch.equal(a[k], b[k]) for k in a) and a["state0"].shape == (2, 36)
    assert torch.equal(a["state0"], states[:, -1, :36])                              # the last state, nothing later
    from rrp.policies.psi0 import INPUT_SPEC
    assert set(INPUT_SPEC) >= {"images", "instruction", "state", "vlm", "entity"}


def test_structured_packet_edit_reaches_system_0_and_generated_z_is_kept():
    import torch
    m, seen = _fake_ours("structured")
    st = torch.zeros(1, 1, 40)
    m.predict_action([["img"]], st, ["pick up the cup"])
    gen = seen["z"].clone()
    m.packet_edit = ("swap_assembly", dict(a=0, b=1))
    m.predict_action([["img"]], st, ["pick up the cup"])
    assert torch.equal(seen["z"][:, :, 0], gen[:, :, 1]) and torch.equal(m.generated_z, gen) and torch.equal(m.last_z, seen["z"])
    m.packet_edit = ("mean_packet", {})
    m.predict_action([["img"]], st, ["pick up the cup"])
    assert torch.equal(seen["z"], torch.full_like(gen, 0.25))                         # default mean = the head's z_stats mean
    m.packet_hook = lambda z: z
    with pytest.raises(ValueError, match="not both"):
        m.predict_action([["img"]], st, ["pick up the cup"])


def test_structured_policy_refuses_a_head_without_a_passed_gate_for_this_stage_a(tmp_path):
    from rrp.policies.psi0 import _digest_file, structured_provenance
    sa = tmp_path / "stage_a.pt"
    sa.write_bytes(b"stage a")
    sha = _digest_file(sa)
    ok = dict(config=dict(packet_gate=dict(passed=True, gap=0.2, margin=0.05, stage_a_sha256_16=sha)),
              versions=dict(factors="abc"))
    assert structured_provenance(ok, str(sa)) == dict(stage_a=sha, packet_gate=dict(gap=0.2, margin=0.05), factors="abc")
    for cfg in (dict(), dict(packet_gate=None), dict(packet_gate=dict(passed=False, gap=0.0, margin=0.05, stage_a_sha256_16=sha)),
                dict(packet_gate=dict(passed=True, gap=0.2, margin=0.05, stage_a_sha256_16="0" * 16))):
        with pytest.raises(ValueError, match="passed packet-use gate"):
            structured_provenance(dict(config=cfg), str(sa))


def test_psi0_policy_validates_packet_edit_at_construction():
    from rrp.policies.psi0 import Psi0Policy
    Psi0Policy("psi0_structured", weights=None, packet_edit=dict(name="zero_slot", slot=2))
    for kind, edit in (("psi0_structured", dict(name="nope")), ("psi0_direct", dict(name="zero_slot", slot=0))):
        with pytest.raises(ValueError, match="packet_edit"):
            Psi0Policy(kind, packet_edit=edit)
