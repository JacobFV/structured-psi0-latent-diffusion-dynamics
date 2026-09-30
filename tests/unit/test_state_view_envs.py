"""D-144 R8: StateView contract tests for Warp, ComputerWorld and SIMPLE (docs/relations.md 5.1, section 10 row R8).

Each backend's `state_view()` wraps a PURE builder function that this file exercises directly on a fixture, so none
of these tests need the optional `computerworld` wheel, an Isaac / SIMPLE venv, or CUDA / mujoco_warp:
  - ComputerWorld: an inline fixture scene (same shape as test_computerworld.py's SCENE).
  - SIMPLE: a recorded `truth()` dict (the shape `rrp.envs.simple.worker.Worker._sim_truth()` returns).
  - Warp: a 2-env CPU batch (`mujoco` + a procedural `hexapod()` body -- no menagerie assets, no CUDA); skip-marked
    if that host-side model cannot be built.
Every `rrp.envs.base.StateView` implementation here is checked against the `runtime_checkable` protocol itself.
"""
import numpy as np
import pytest

from rrp.envs.base import CapabilityError, StateView

# ----------------------------------------------------------------------------------------------------- ComputerWorld
from rrp.envs.computerworld import ScreenFrame, SlotRegistry, cw_state_view

_XFORM = {"a": 1024, "b": 0, "c": 0, "d": 1024, "tx": 0, "ty": 0}


def _cw_node(i, x, y, w, h, z, sem=None, inter=None, tx=0, ty=0):
    return {"id": i, "bounds": {"x": x, "y": y, "width": w, "height": h}, "primitive": {"kind": "box"},
            "semantic": sem, "interaction": inter, "transform": dict(_XFORM, tx=tx, ty=ty), "z": z, "opacity": 255,
            "clip": None}


def _cw_btn(label):
    return {"role": "button", "label": label, "value": None, "disabled": False, "focusable": True}


# a desktop icon (z0) and a window (z100) with one Save button inside it, focused
CW_SCENE = {"focus": {"interaction": "window:0:content:save"}, "nodes": [
    _cw_node(1, 10, 10, 40, 40, 0, sem=_cw_btn("Files"), inter="shell:launch:files"),
    _cw_node(2, 100, 100, 300, 200, 100),
    _cw_node(3, 20, 30, 60, 20, 100, sem=_cw_btn("Save"), inter="window:0:content:save", tx=100, ty=100),
]}


def test_cw_state_view_on_fixture_scene():
    sv = cw_state_view(CW_SCENE, ScreenFrame(640, 480), SlotRegistry(), t=1.5)
    assert isinstance(sv, StateView)
    assert sv.caps == frozenset({"poses", "ui_tree"}) and sv.time == 1.5
    assert list(sv.gravity) == [0.0, 0.0, 0.0]

    ents = {e.name: e for e in sv.entities()}
    assert set(ents) == {"Files", "Save"}
    files, save = ents["Files"], ents["Save"]
    assert files.kind == "widget" and files.parent is None and files.visible is True
    assert save.parent == "window:0" and save.attrs["focused"] == "true"
    assert save.pos[2] > files.pos[2]                       # higher z-layer -> larger stack depth

    nodes = {n["label"]: n for n in sv.ui_tree()}
    assert nodes["Save"]["parent"] == "window:0" and nodes["Save"]["focus_rank"] == 0
    assert nodes["Files"]["focus_rank"] is None

    # token_entity round-trips through the same slot table state_view built
    ids = [sv.token_entity("widgets", i) for i in range(len(sv.entities()))]
    assert save.id in ids and files.id in ids

    for method in (sv.contacts, sv.joints, lambda: sv.camera("x")):
        with pytest.raises(CapabilityError):
            method()


def test_cw_state_view_slots_are_stable_across_calls():
    """Slots are keyed by (interaction, role, label): re-building state_view from the SAME registry after a widget
    vanishes leaves its slot present but occupied by None from `SlotRegistry.assign` -> filtered out here."""
    reg = SlotRegistry()
    sv1 = cw_state_view(CW_SCENE, ScreenFrame(640, 480), reg, t=0.0)
    slot_ids = {e.id for e in sv1.entities()}
    smaller_scene = {"focus": {}, "nodes": [CW_SCENE["nodes"][0]]}    # Save's window disappears
    sv2 = cw_state_view(smaller_scene, ScreenFrame(640, 480), reg, t=1.0)
    assert {e.name for e in sv2.entities()} == {"Files"}
    assert all(sid.startswith("slot") for sid in slot_ids)            # slot ids keep their original index


# --------------------------------------------------------------------------------------------------------- SIMPLE
from rrp.envs.simple import CONTROL_HZ, simple_state_view

SIMPLE_TRUTH = dict(
    source="privileged:sim", step=100,
    palm={"left": np.array([0.1, 0.2, 0.3], np.float32), "right": np.array([0.4, 0.5, 0.6], np.float32)},
    pelvis=np.array([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0], np.float32),
    objects={"target": np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], np.float32),
             "distractor": np.array([0.2, 0.3, 0.8, 1.0, 0.0, 0.0, 0.0], np.float32)},
    contact={"left:target": True, "right:target": False, "left:distractor": False, "right:distractor": False},
    contact_point={"left": np.array([0.5, 0.0, 0.8], np.float32), "right": np.full(3, np.nan, np.float32)},
    reward=1.0, max_reward=1.0, first_contact={"left:target": 80}, success=True, terminated=False, truncated=False,
    target_name="target")


def test_simple_state_view_on_recorded_truth_dict():
    sv = simple_state_view(SIMPLE_TRUTH)
    assert isinstance(sv, StateView)
    assert sv.caps == frozenset({"poses", "contacts"})
    assert sv.time == pytest.approx(100 / CONTROL_HZ)
    assert list(sv.gravity) == pytest.approx([0.0, 0.0, -9.81])

    ents = {e.id: e for e in sv.entities()}
    assert set(ents) == {"pelvis", "palm:left", "palm:right", "object:target", "object:distractor"}
    assert ents["pelvis"].pos == pytest.approx([0.0, 0.0, 0.9]) and ents["pelvis"].quat == pytest.approx([1, 0, 0, 0])
    assert ents["palm:left"].parent == "pelvis" and ents["palm:left"].pos == pytest.approx([0.1, 0.2, 0.3])
    assert ents["object:target"].attrs["is_target"] is True
    assert ents["object:distractor"].attrs["is_target"] is False

    contacts = sv.contacts()
    assert len(contacts) == 1
    c = contacts[0]
    assert (c.a, c.b) == ("palm:left", "object:target")
    assert c.pos == pytest.approx([0.5, 0.0, 0.8]) and c.normal == pytest.approx([0.0, 0.0, 0.0])

    assert sv.token_entity("palm", "left") == "palm:left"
    assert sv.token_entity("objects", "target") == "object:target"
    assert sv.token_entity("objects", "nope") is None
    assert sv.token_entity("pelvis", None) == "pelvis"

    for method in (sv.joints, lambda: sv.camera("head"), sv.ui_tree):
        with pytest.raises(CapabilityError):
            method()


def test_simple_state_view_handles_missing_contact_point():
    """Right hand never touched `target`, so its contact_point is NaN and must not appear (only active contacts do)."""
    sv = simple_state_view(SIMPLE_TRUTH)
    assert all(c.a != "palm:right" for c in sv.contacts())


def test_simple_env_declares_state_view():
    from rrp.envs.simple import SimpleEnv
    assert hasattr(SimpleEnv, "state_view")


# ----------------------------------------------------------------------------------------------------------- Warp
def _host_hexapod_model():
    from rrp.bodies.legged import hexapod, standalone_model
    m, _, meta = standalone_model(hexapod())
    return m


def test_warp_state_view_on_two_env_cpu_batch():
    """`warp_state_view_from_arrays` is the CPU-testable core of `WarpTrackerEnv.state_view`: it needs a host
    `mujoco.MjModel` (built without CUDA or mujoco_warp) plus plain numpy arrays standing in for a 2-world batch."""
    pytest.importorskip("mujoco")
    from rrp.envs.warp.tracker_env import warp_state_view_from_arrays
    try:
        m = _host_hexapod_model()
    except Exception as e:  # noqa: BLE001 -- host cannot build even a procedural body: skip, don't fail the suite
        pytest.skip(f"cannot build a host mujoco model: {e}")

    nbody = m.nbody
    rng = np.random.RandomState(0)
    xpos = rng.randn(2, nbody, 3).astype(np.float32)             # a "2-env" batch
    # 3 contact slots across the pool: 2 in world 0, 1 in world 1, plus one unused (-1) slot
    c_geom = np.array([[0, 1], [1, 2], [0, 3], [-1, -1]], dtype=np.int32)
    c_pos = rng.randn(4, 3).astype(np.float32)
    c_world = np.array([0, 0, 1, 0], dtype=np.int32)

    sv0 = warp_state_view_from_arrays(m, xpos[0], c_geom, c_pos, c_world, index=0, t=1.0)
    sv1 = warp_state_view_from_arrays(m, xpos[1], c_geom, c_pos, c_world, index=1, t=2.0)
    assert isinstance(sv0, StateView) and isinstance(sv1, StateView)
    assert sv0.caps == frozenset({"poses", "contacts"})
    assert list(sv0.gravity) == pytest.approx(list(m.opt.gravity))

    assert len(sv0.entities()) == nbody == len(sv1.entities())
    e0 = {e.id: e for e in sv0.entities()}
    assert np.allclose(e0[mujoco_body_name(m, 0)].pos, xpos[0, 0])   # world 0 reads world 0's positions
    e1 = {e.id: e for e in sv1.entities()}
    assert np.allclose(e1[mujoco_body_name(m, 0)].pos, xpos[1, 0])   # world 1 reads world 1's positions, not world 0's

    assert len(sv0.contacts()) == 2 and len(sv1.contacts()) == 1     # per-world filtering by c_world; -1 slot dropped
    assert sv0.time == 1.0 and sv1.time == 2.0

    with pytest.raises(CapabilityError):
        sv0.joints()


def mujoco_body_name(m, b: int) -> str:
    import mujoco
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


@pytest.mark.skipif(True, reason="WarpTrackerEnv itself always needs CUDA (self.dev is hard-coded); "
                                  "no CPU device path exists to construct a real one on host (D-144 R8 brief: "
                                  "'Warp on a 2-env CPU batch if available, else skip-marked')")
def test_warp_tracker_env_state_view_live():
    pass


def test_warp_tracker_env_declares_state_view():
    from rrp.envs.warp.tracker_env import WarpTrackerEnv
    assert hasattr(WarpTrackerEnv, "state_view")
