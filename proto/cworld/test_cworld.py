"""ComputerWorld env prototype tests. Pure mapping tests use an inline 5-node scene; the rest need the
`computerworld` wheel (optional extra) and skip without it. No network: the form site is simulated in-world."""
import importlib.util

import pytest

from cworld import (KEY_NAMES, KEY_VOCAB, PointerState, ScreenFrame, SlotRegistry, command_to_actions, cw_pointer_spec,
                    descriptors, key_index, node_box, scene_widgets)

I = {"a": 1024, "b": 0, "c": 0, "d": 1024, "tx": 0, "ty": 0}


def _node(i, x, y, w, h, z, kind="box", sem=None, inter=None, tx=0, ty=0, clip=None):
    return {"id": i, "bounds": {"x": x, "y": y, "width": w, "height": h}, "primitive": {"kind": kind},
            "semantic": sem, "interaction": inter, "transform": dict(I, tx=tx, ty=ty), "z": z, "opacity": 255,
            "clip": clip}


def _btn(label):
    return {"role": "button", "label": label, "value": None, "disabled": False, "focusable": True}


# desktop icon at z0; window 0 (z100) with a Save button in local coords; window 1 (z200) fully covers the Save button
SCENE = {"focus": {}, "nodes": [
    _node(1, 10, 10, 40, 40, 0, sem=_btn("Files"), inter="shell:launch:files"),
    _node(2, 100, 100, 300, 200, 100, kind="rounded_box"),
    _node(3, 20, 30, 60, 20, 100, sem=_btn("Save"), inter="window:0:content:save", tx=100, ty=100),
    _node(4, 110, 110, 200, 100, 200, kind="box"),
    _node(5, 150, 150, 50, 20, 200, kind="ui_text", sem=_btn("OK"), inter="window:1:content:ok"),
]}


def test_px_metre_mapping():
    f = ScreenFrame(640, 480)
    assert f.px_to_m(320, 240) == (0.0, 0.0)                       # origin at screen centre
    assert f.px_to_m(420, 140) == pytest.approx((0.1, 0.1))        # 1 px = 1 mm, y up
    for u, v in [(0, 0), (639, 479), (17, 301)]:
        assert f.m_to_px(*f.px_to_m(u, v)) == (u, v)
    assert f.m_to_px(10.0, -10.0) == (639, 479)                    # clamped to the viewport


def test_node_box_applies_transform():
    assert node_box(SCENE["nodes"][2]) == (120, 130, 180, 150)


def test_depth_modes_and_occlusion():
    ws = {w["label"]: w for w in scene_widgets(SCENE)}
    assert ws["Save"]["visible"] is False                          # covered by window 1's box: kept, not visible
    assert ws["Files"]["visible"] and ws["OK"]["visible"]          # text never occludes; OK sits on top
    stack = {d.slot: d for d in descriptors(SlotRegistry().assign(list(ws.values())), ScreenFrame(640, 480), 0.0)}
    zs = [d.position_estimate[2] for d in stack.values()]
    assert zs == pytest.approx([0.0, 0.002, 0.004])                # dense z-layer rank x 2 mm
    flat = descriptors(SlotRegistry().assign(list(ws.values())), ScreenFrame(640, 480, depth="constant"), 0.0)
    assert all(d.position_estimate[2] == 0.0 for d in flat)
    assert stack[1].position_estimate[:2] == pytest.approx([(150 - 320) * 1e-3, (240 - 140) * 1e-3])


def test_null_slots_stable():
    reg = SlotRegistry()
    first = reg.assign(scene_widgets(SCENE))
    later = dict(SCENE, nodes=[SCENE["nodes"][i] for i in (0, 1, 3, 4)] + [
        _node(6, 5, 400, 30, 30, 0, sem=_btn("Trash"), inter="shell:trash")])
    second = reg.assign(scene_widgets(later))
    assert [w["label"] for w in first] == ["Files", "Save", "OK"]
    assert second[0]["label"] == "Files" and second[1] is None and second[2]["label"] == "OK"
    assert second[3]["label"] == "Trash"
    d = descriptors(second, ScreenFrame(), 0.0)
    assert d[1].descriptor == "null" and d[1].visible is False and d[1].position_estimate is None


def test_duplicate_widgets_get_distinct_slots():
    dup = dict(SCENE, nodes=SCENE["nodes"] + [_node(7, 12, 12, 30, 30, 0, sem=_btn("Files"), inter="shell:launch:files")])
    keys = [w["key"] for w in scene_widgets(dup)]
    assert len(keys) == len(set(keys))


def test_button_edges_and_keys():
    f, st = ScreenFrame(640, 480), PointerState(320, 240)
    acts, ex, rej = command_to_actions({"pointer": [0.1, 0.0], "button": [0.0]}, st, f)
    assert rej is None and [a[1] for a in acts] == ["move"] and acts[0][2]["x"] == 420
    acts, *_ = command_to_actions({"pointer": [0.1, 0.0], "button": [1.0]}, st, f)
    assert [a[1] for a in acts] == ["down"]                        # no move when the pixel is unchanged
    acts, *_ = command_to_actions({"button": [0.9]}, st, f)
    assert acts == []                                             # held: no repeated edge
    acts, *_ = command_to_actions({"pointer": [0.12, 0.0], "button": [0.1]}, st, f)
    assert [a[1] for a in acts] == ["move", "up"]                 # drag then release, in that order
    acts, ex, _ = command_to_actions({"key": [key_index("a")], "wheel": [-2]}, st, f)
    assert acts == [("pointer.v1", "wheel", {"x": 440, "y": 240, "delta_y": -240, "width": 640, "height": 480}),
                    ("keyboard.v1", "type", {"text": "a"})]
    acts, *_ = command_to_actions({"key": [KEY_NAMES.index("Enter")]}, st, f)
    assert acts == [("keyboard.v1", "key", {"key": "Enter"})]
    assert command_to_actions({"key": [-1]}, st, f)[0] == []
    assert command_to_actions({"key": [len(KEY_VOCAB)]}, st, f)[2] == "key_out_of_vocab"
    assert command_to_actions({"arm": [0.0]}, st, f)[2] == "bad_group:arm"


def test_pointer_body():
    spec = cw_pointer_spec(ScreenFrame(640, 480))
    assert [j.type for j in spec.joints] == ["slide"] * 3 and spec.spec_hash
    assert spec.joints[0].range == pytest.approx([-0.32, 0.319])
    assert spec == cw_pointer_spec(ScreenFrame(640, 480))


# ----------------------------------------------------------------------------------------- need the wheel
from cworld import TASKS, NativeCommand, ScriptedTeacher, _click, make_env, run_episode  # noqa: E402

pytestmark_cw = pytest.mark.skipif(importlib.util.find_spec("computerworld") is None,
                                   reason="optional extra `computerworld` not installed")


@pytestmark_cw
@pytest.mark.parametrize("task", sorted(TASKS))
def test_teacher_solves(task):
    env = make_env(task=task, seed=0)
    for seed in (0, 1):
        r = run_episode(env, ScriptedTeacher(task), seed)
        assert (r["outcome"], r["failure_reason"], r["source"]) == ("success", None, "scripted_teacher"), r


@pytestmark_cw
@pytest.mark.parametrize("task,reason", [("cw/calc_sum", "no_result"), ("cw/open_type", "app_not_open"),
                                         ("cw/drag_window", "off_target"), ("cw/fill_form", "not_submitted")])
def test_idle_times_out_with_reason(task, reason):
    r = run_episode(make_env(task=task, seed=0), lambda o: None, 0)
    assert (r["outcome"], r["failure_reason"]) == ("timeout", reason)


def _drive(env, plan, seed=0, max_ticks=200):
    """Run a command-dict generator (built from the teacher helpers) until the judge is done."""
    env.reset(seed)
    it = plan(env)
    for _ in range(max_ticks):
        j = env.judge()
        if j.done:
            return j
        g = next(it, None)
        env.step(None if g is None else NativeCommand(controller_version="cw_pointer.v1", groups=g, source="mock"))
    raise AssertionError("judge never finished")


@pytestmark_cw
def test_wrong_text_fails():
    def plan(env):
        yield from _click(env, lambda w: w["interaction"] == "shell:launch:editor")
        for ch in "zzz":
            yield env.teacher_command(key=key_index(ch))
    j = _drive(make_env(task="cw/open_type", seed=0), plan)
    assert (j.outcome, j.failure_reason) == ("failure", "wrong_text")


@pytestmark_cw
def test_wrong_sum_and_wrong_field_fail():
    def calc(env):
        for label in ("1", "+", "1", "="):
            yield from _click(env, lambda w, l=label: w["interaction"].endswith(f":calc:{l}"))
    env = make_env(task="cw/calc_sum", seed=0)
    assert env.goal["a"] + env.goal["b"] != 2
    assert (_drive(env, calc).failure_reason) == "wrong_result"

    def form(env):                                      # submits with the email left empty
        yield from _click(env, lambda w: w["role"] == "textbox" and w["interaction"].endswith(":content:name"))
        for ch in env.goal["name"]:
            yield env.teacher_command(key=key_index(ch))
        yield from _click(env, lambda w: w["label"] == "Submit")
    assert _drive(make_env(task="cw/fill_form", seed=0), form).failure_reason == "wrong_value:email"


@pytestmark_cw
def test_reset_deterministic_and_snapshot():
    env = make_env(task="cw/fill_form", seed=3)
    h0, d0 = env.state_hash(), [d.model_dump(mode="json") for d in env.observe().object_descriptors]
    run_episode(env, ScriptedTeacher("cw/fill_form"), 3)
    assert env.state_hash() != h0
    env.reset(3)
    assert env.state_hash() == h0
    assert [d.model_dump(mode="json") for d in env.observe().object_descriptors] == d0
    snap = env.snapshot()
    env.step(NativeCommand(controller_version="cw_pointer.v1", groups={"pointer": [0.05, 0.05]}, source="mock"))
    env.restore(snap)
    assert env.state_hash() == h0 and env.pointer.u == env.frame.width // 2


@pytestmark_cw
def test_observation_public_and_truth_privileged():
    env = make_env(task="cw/drag_window", seed=0, images=True)
    obs = env.observe()
    assert obs.measured_node_state.qpos.tolist() == pytest.approx([0.0, 0.0, env.frame.hover_z])
    assert obs.sensor_images[0].pixels.shape[:2] == (env.frame.height, env.frame.width)
    assert any(d.bound_entity and d.bound_entity.id == "window" for d in obs.object_descriptors)
    tr = env.truth()
    assert tr.labels["goal"] == env.goal and len(tr.object_poses) >= sum(d.descriptor != "null"
                                                                         for d in obs.object_descriptors)
    assert "goal" not in obs.model_dump_json()                   # the goal only reaches the policy as the instruction
    assert "images" in env.spec.capabilities and "chunk_executor" not in env.spec.capabilities


@pytestmark_cw
def test_rejected_codes_not_silent():
    env = make_env(task="cw/open_type", seed=0)
    st = env.step(NativeCommand(controller_version="cw_pointer.v1", groups={"arm": [0.0]}, source="mock"))
    assert st.rejected == "bad_group:arm" and st.executed == {}
