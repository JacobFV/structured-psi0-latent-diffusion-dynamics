"""Mechanical boundary tests for the transmitted latent packet (R38; tests 3 and 10 of the correction)."""
import numpy as np
import pytest
from pydantic import ValidationError

from rrp.contracts.latent_action import LatentActionChunk, AssemblyHandle, EntityHandle, check_packet
from rrp.contracts.errors import ControllerRejection, StaleActionError

H = "asm:0123456789abcdef:r0/0"


def pkt(**kw):
    d = dict(latent_space_version="ls1", realizer_compat_version="rz1", z=np.zeros((4, 1, 8), np.float32),
             knot_times=[0.0, 0.2, 0.4, 0.6], assemblies=[AssemblyHandle(handle=H, robot_index=0)],
             assembly_mask=[True], entity_registry=[EntityHandle(handle="ent:0")], observation_id="o",
             graph_version=1, runtime_version=3, robot_spec_hash="h", generated_at=1.0, valid_from=0.0,
             valid_until=0.8, source="learned", policy_version="p")
    d.update(kw)
    return LatentActionChunk(**d)


def test_roundtrip_is_exact_float32():
    p = pkt(z=np.random.default_rng(0).normal(size=(4, 1, 8)).astype(np.float32))
    q = LatentActionChunk.from_bytes(p.to_bytes())
    assert np.array_equal(p.z, q.z)


@pytest.mark.parametrize("bad", [
    dict(z=np.full((4, 1, 8), np.nan, np.float32)),
    dict(z=np.zeros((4, 8), np.float32)),
    dict(knot_times=[0.0, 0.2, 0.2, 0.6]),
    dict(knot_times=[0.0, 0.2]),
    dict(assembly_mask=[True, False]),
    dict(valid_until=0.0),
    dict(assemblies=[AssemblyHandle(handle=H, robot_index=0), AssemblyHandle(handle=H, robot_index=0)],
         assembly_mask=[True, True], z=np.zeros((4, 2, 8), np.float32)),
])
def test_malformed_packets_rejected(bad):
    with pytest.raises(ValidationError):
        pkt(**bad)


@pytest.mark.parametrize("field", ["focused_object", "active_task_description", "target_pose", "desired_effect",
                                   "dependencies"])
def test_no_semantic_metadata_fields(field):
    with pytest.raises(ValidationError):
        pkt(**{field: "cube"})


def test_entity_registry_is_opaque():
    with pytest.raises(ValidationError):
        EntityHandle(handle="red cube")
    with pytest.raises(ValidationError):
        AssemblyHandle(handle="gripper that should grasp the cube", robot_index=0)


def test_admission_checks():
    p = pkt()
    check_packet(p, latent_space_version="ls1", realizer_compat_version="rz1", robot_spec_hash="h", now=0.1,
                 graph_version=1, owned_assemblies={H})
    for kw, exc in [(dict(latent_space_version="ls2"), ControllerRejection),
                    (dict(realizer_compat_version="rz2"), ControllerRejection),
                    (dict(robot_spec_hash="other"), ControllerRejection),
                    (dict(now=0.9), StaleActionError),
                    (dict(graph_version=2), StaleActionError),
                    (dict(owned_assemblies={"asm:0123456789abcdef:r1/0"}), ControllerRejection)]:
        args = dict(latent_space_version="ls1", realizer_compat_version="rz1", robot_spec_hash="h", now=0.1,
                    graph_version=1, owned_assemblies={H})
        args.update(kw)
        with pytest.raises(exc):
            check_packet(p, **args)
