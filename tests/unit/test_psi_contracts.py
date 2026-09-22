import numpy as np
import pytest

from rrp.control.psi_contracts import ContractBodyError, ContractWidthError, psi0_original, psi0_sonic


def test_layouts_match_audit():
    o, s = psi0_original(), psi0_sonic()
    assert (o.action_dim, o.state_dim, o.state_real_dim) == (36, 36, 32)
    assert (s.action_dim, s.state_dim) == (80, 45)
    og = {g.name: (g.start, g.stop) for g in o.action_groups}
    assert og["q_hand"] == (0, 14) and og["q_arm"] == (14, 28) and og["torso_rpy"] == (28, 31)
    assert og["base_height"] == (31, 32) and og["v_x"] == (32, 33) and og["p_yaw"] == (35, 36)
    sg = {g.name: (g.start, g.stop) for g in s.action_groups}
    assert sg == {"sonic_token": (0, 64), "hands": (64, 78), "neck": (78, 80)}
    ss = {g.name: (g.start, g.stop) for g in s.state_groups}
    assert ss["legs"] == (0, 12) and ss["hands"] == (29, 43) and ss["neck"] == (43, 45)


def test_rejects_wrong_width_and_cross_contract():
    o, s = psi0_original(), psi0_sonic()
    o.validated_bodies = {"h1_test"}
    s.validated_bodies = {"g1_test"}
    with pytest.raises(ContractWidthError):
        s.apply("g1_test", np.zeros(78))        # neckless 78-dim is NOT the 80-dim contract
    with pytest.raises(ContractWidthError):
        o.apply("h1_test", np.zeros(80))        # SONIC action into the AMO contract
    assert o.apply("h1_test", np.zeros(36))["v_x"].shape == (1,)
    with pytest.raises(ContractWidthError):
        s.pack_state({"legs": np.zeros(12)})


def test_refuses_unvalidated_body():
    s = psi0_sonic()
    with pytest.raises(ContractBodyError):
        s.apply("unitree_g1", np.zeros(80))     # nothing validated yet in our sim
    s.validated_bodies = {"unitree_g1"}
    with pytest.raises(ContractBodyError):
        s.apply("booster_t1", np.zeros(80))     # no silent G1 -> other body
