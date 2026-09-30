"""HD1 (D-146 readiness round 2): both legged collectors run their episode as a `harness.rollout` with a recorder hook
(no private step loop), and record what training needs. The 20-tick stub-tracker episodes below were recorded from the
private loops BEFORE the port (`loop.legged.*` in tests/data/golden.json; floats rounded to 5 decimals); the digests
cover only the keys that existed then, so the keys HD1 adds are checked separately. Stub trackers are random-weight:
plumbing only, no number here is a result. Needs the Menagerie assets (skipped without them)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_legged_collect_tasks import BODY, stub_tracker   # noqa: E402,F401  (stub_tracker: fixture)
from test_trackers import registry   # noqa: E402,F401  (fixture)
from test_golden import golden   # noqa: E402,F401  (fixture)
from test_ladder_rollout import _digest   # noqa: E402

from rrp.harness.data import legged_collect as LCOL, legged_latent_collect as LLC   # noqa: E402

TASKS = ("waypoint_contact", "h_steps", "h_gap")
TICKS = 20
OLD_META = ("episode_id", "robot", "spec_hash", "lineage", "family", "synthetic", "controller_version", "tracker_source",
            "tracker_sha256", "task", "task_hash", "seed", "control_dt", "physics_dt", "tracker_hz", "steps", "max_steps",
            "status", "failure_reason", "public_runtime_success", "event_status", "source", "privileged_teacher", "teacher",
            "teacher_version", "teacher_variant", "waypoints", "split_lineage", "tracker_source_label", "physics")
OLD_LABELS = ("base_pose", "base_vel_body", "base_height", "foot_contact", "bad_contact", "predicates_truth",
              "event_completion_truth", "teacher_phase")
OLD_LATENT = ("q", "qd", "imu", "touch", "osc", "a", "ctx", "ev", "pose", "contact", "bad", "height", "cmd")
OLD_LATENT_META = ("body", "seed", "sigma", "task", "status", "failure_reason", "steps", "max_steps", "ticks",
                   "tracker_source", "tracker_version", "source", "teacher", "teacher_version", "tracker_sha256",
                   "tracker_run", "actuator", "privileged_teacher", "teacher_variant", "waypoints", "spec_hash",
                   "tracker_source_label", "physics", "motion")
LEGACY_CTX = 22          # the pre-HD1 context width; HD1 appends further target slots after it


def _pub(p: dict) -> dict:
    """The public record without observation ids (a process-global serial)."""
    return {k: v for k, v in p.items()}


def _collect_digest(rec) -> str:
    pub, prv = rec.public, rec.private
    return _digest([{k: pub["meta"][k] for k in OLD_META}, [_pub(i) for i in pub["inputs"]], pub["actions"],
                    pub["action_space"], [{k: lab[k] for k in OLD_LABELS} for lab in prv["labels"]], prv["phases"]])


def _latent_digest(arr, meta, morph) -> str:
    a = {k: (arr[k][:, :LEGACY_CTX] if k == "ctx" else arr[k]) for k in OLD_LATENT}
    return _digest([a, {k: meta[k] for k in OLD_LATENT_META}, morph.node_static, morph.node_asm, morph.asm_static,
                    morph.q0_all, morph.n_policy, morph.nf, morph.handles, morph.asm_kind])


@pytest.mark.menagerie
@pytest.mark.parametrize("task", TASKS)
def test_collect_golden(task, golden, stub_tracker):
    rec = LCOL.collect_episode(BODY, 0, task=task, max_steps=TICKS, tracker_id=stub_tracker(task))
    golden(f"loop.legged.collect.{task}", _collect_digest(rec))


@pytest.mark.menagerie
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("sigma", (0.0, 0.2))
def test_latent_collect_golden(task, sigma, golden, stub_tracker):
    arr, meta, morph = LLC.collect_episode(BODY, 1, sigma, task=task, max_steps=TICKS, tracker_id=stub_tracker(task))
    golden(f"loop.legged.latent.{task}.s{sigma}", _latent_digest(arr, meta, morph))


# ------------------------------------------------------------------ what the collector records (HD1)
import ast   # noqa: E402
import json   # noqa: E402
import inspect   # noqa: E402

from rrp.policies.features import legged as F   # noqa: E402
from rrp.tasks.spec import TASKS as REGISTRY   # noqa: E402
import rrp.tasks.humanoid   # noqa: E402,F401  (registers the h_* tasks)

BASE_TASKS = ("waypoint_contact", "h_steps", "h_gap")
WHOLEBODY_TASKS = ("h_walk", "h_turn", "h_reach", "h_squat_pick", "h_place", "h_carry", "h_loco_pick", "h_steps_carry",
                   "h_gap_cart")
HD1_KEYS = ("upper", "upper_valid", "terrain", "terrain_valid", "foothold_cell", "com_support", "com_support_valid")


def test_every_h_task_with_a_teacher_is_covered_here():
    h = {k for k, v in REGISTRY.items() if k.startswith("h_") and "mujoco/legged" in v.envs and v.teacher}
    assert h == (set(BASE_TASKS) | set(WHOLEBODY_TASKS)) - {"waypoint_contact"}


@pytest.mark.parametrize("task", sorted(k for k, v in REGISTRY.items() if k.startswith("h_") or k == "waypoint_contact"))
def test_task_view_holds_every_registered_h_graph(task):
    spec = REGISTRY[task]
    if spec.graph is None:
        pytest.skip(f"{task} has no graph")
    v = F.task_view_of(task)
    assert 1 <= len(v.events) <= F.EVENT_SLOTS and 1 <= len(v.targets) <= F.TARGET_SLOTS
    assert v.as_dict()["event_slots"] == F.EVENT_SLOTS and v.as_dict()["target_slots"] == F.TARGET_SLOTS


def test_widened_slots_keep_the_legacy_columns():
    """One constant (TARGET_SLOTS) widens the context; the first 22 columns and their meaning are unchanged."""
    assert F.GLOBAL_DIM == 22 + 4 * (F.TARGET_SLOTS - F.LEAD_TARGET_SLOTS) and (F.EVENT_SLOTS, F.TARGET_SLOTS) == (3, 3)


@pytest.mark.menagerie
@pytest.mark.parametrize("task", BASE_TASKS + WHOLEBODY_TASKS)
def test_latent_episode_writes_every_key(task, stub_tracker):
    arr, meta, morph = LLC.collect_episode(BODY, 0, 0.1, task=task, max_steps=TICKS, tracker_id=stub_tracker(task))
    T = len(arr["a"])
    assert T == meta["ticks"] > 0
    assert set(OLD_LATENT + HD1_KEYS) <= set(arr) and all(len(v) == T for v in arr.values())
    n_up = arr["q"].shape[1] - morph.n_policy
    assert arr["upper"].shape == (T, n_up) and arr["upper"].dtype == np.float32 and arr["upper_valid"].dtype == bool
    assert arr["terrain"].shape == (T, 77) and arr["terrain_valid"].shape == (T, 77) and arr["terrain_valid"].dtype == bool
    assert np.isfinite(arr["terrain"]).all()
    assert arr["foothold_cell"].shape == (T, morph.M) and arr["foothold_cell"].dtype == np.int64
    assert arr["foothold_cell"].min() >= -2 and arr["foothold_cell"].max() < 77
    assert (arr["foothold_cell"][:, morph.nf:] == -2).all()                     # non-foot assemblies: absent
    assert ((arr["foothold_cell"][:, :morph.nf] == -1) == arr["contact"]).all()  # planted <=> foot down
    assert arr["com_support"].shape == (T,) and np.isfinite(arr["com_support"]).all() and arr["com_support_valid"].all()
    assert arr["ctx"].shape == (T, F.GLOBAL_DIM)
    whole = meta["control"] == "wholebody"
    assert whole == (task in WHOLEBODY_TASKS) and bool(arr["upper_valid"].all()) == whole and bool(arr["upper_valid"].any()) == whole
    assert meta["terrain_source"] in ("session_scan", "collector_scan")
    assert (meta["motion"] is None) == whole
    if whole:
        assert meta["step_ticks"] == 1 and meta["steps"] <= T <= meta["steps"] + round(LLC.POST_SECONDS / F.TICK_DT)
    else:
        assert not arr["upper"].any() and meta["step_ticks"] == 5


@pytest.mark.menagerie
def test_wholebody_upper_targets_are_the_teachers(stub_tracker):
    """h_reach moves an arm: the recorded `upper` rows follow the session's applied upper target, in action units."""
    arr, meta, morph = LLC.collect_episode(BODY, 0, 0.0, task="h_reach", max_steps=TICKS, tracker_id=stub_tracker("h_reach"))
    assert np.abs(arr["upper"]).max() > 0 and arr["upper_valid"].all()
    assert meta["legs_source"] in ("rl_expert", "planned_com") and meta["source"] == "scripted_teacher"


@pytest.mark.menagerie
@pytest.mark.parametrize("task", BASE_TASKS + WHOLEBODY_TASKS)
def test_public_private_record_carries_com_support(task, stub_tracker):
    rec = LCOL.collect_episode(BODY, 0, task=task, max_steps=TICKS, tracker_id=stub_tracker(task))
    labs = rec.private["labels"]
    assert labs and all(np.isfinite(lab["com_support"]) and lab["com_support_valid"] is True for lab in labs)
    acts = rec.public["actions"]
    assert set(acts[0]) == ({"legs", "upper"} if task in WHOLEBODY_TASKS else {"base_velocity"})


@pytest.mark.menagerie
def test_com_support_matches_the_relgen_label(stub_tracker):
    from rrp.harness.data.relgen import TokenIndex
    from rrp.harness.data.relgen.body import com_support_fn
    spec, s = LCOL.open_episode("h_steps", BODY, 0, tracker_id=stub_tracker("h_steps"))
    s.reset()
    for _ in range(3):
        s.step(None)
        ids = ["body"] + [f"leg:{s.model.body(b).name}" for b in s.binding.foot_bids]
        lab = com_support_fn(s.state_view(), TokenIndex({"ctx": ids}))
        assert lab.valid[0] and abs(float(lab.value[0, 0]) - LCOL.com_support(s)) < 1e-4


def test_foothold_cells_hindsight():
    """Unit math: planted -1; in swing the cell of the next touchdown in the yaw frame of that tick; -2 without one."""
    nf, T = 2, 6
    contact = np.array([[1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [0, 0]], bool)
    xy = np.zeros((T, nf, 2))
    xy[4, 0] = [0.3, 0.0]                                   # foot 0 lands 0.3 m ahead of the origin at tick 4
    xy[3, 1] = [0.0, -0.3]                                  # foot 1 lands 0.3 m to the right at tick 3
    pose = np.zeros((T, 3), np.float32)
    out = LLC.foothold_cells(contact, xy, pose, M=4)
    assert out.shape == (T, 4) and (out[:, 2:] == -2).all()
    assert out[0, 0] == -1 and out[1, 0] == -1 and out[4, 0] == -1
    cell = lambda lx, ly: round((lx + 0.2) / 0.1) * 7 + round((ly + 0.3) / 0.1)
    assert out[2, 0] == out[3, 0] == cell(0.3, 0.0)
    assert out[1, 1] == out[2, 1] == cell(0.0, -0.3) and out[0, 1] == -1 and out[5, 0] == -2 and out[5, 1] == -2
    pose[2, 2] = np.pi / 2                                  # yawed 90 deg left: the landing at +x is now to the body's right
    assert LLC.foothold_cells(contact, xy, pose, M=4)[2, 0] == cell(0.0, -0.3)
    far = xy.copy(); far[4, 0] = [5.0, 0.0]
    assert LLC.foothold_cells(contact, far, np.zeros((T, 3), np.float32), M=4)[2, 0] == -2     # outside the scan


@pytest.mark.menagerie
@pytest.mark.parametrize("body", ["t1"])
def test_morph_parent_and_trunk_are_kinematic(body, stub_tracker):
    spec, s = LCOL.open_episode("h_steps", body, 0, tracker_id=stub_tracker("h_steps"))
    m = F.LeggedMorph(s.model, s.binding, s.scenario.robots[0].robot_spec.spec_hash)
    N = len(m.node_parent)
    assert m.node_parent.dtype == np.int64 and m.asm_trunk.shape == (m.M,)
    assert ((m.node_parent >= -1) & (m.node_parent < N)).all() and (m.node_parent != np.arange(N)).all()
    for k in range(N):                                      # acyclic: every chain ends at a root (-1) within N steps
        j, n = k, 0
        while j >= 0:
            j, n = int(m.node_parent[j]), n + 1
            assert n <= N
    assert (m.node_parent < 0).sum() >= 1
    assert m.asm_trunk[m.nf] == s.binding.root_bid          # the body assembly hangs off the root link
    jb = [int(s.model.jnt_bodyid[j]) for j in
          [int(s.model.actuator_trnid[a, 0]) for a in list(s.binding.pol_act) + list(s.binding.held_act)]]
    for k in range(N):                                      # a parent joint sits on the same body or one of its ancestors
        p = int(m.node_parent[k])
        if p >= 0:
            b, anc = jb[k], set()
            while b > 0:
                anc.add(b); b = int(s.model.body_parentid[b])
            assert jb[p] in anc


@pytest.mark.menagerie
def test_shard_has_the_new_keys_and_one_task_per_out(tmp_path, stub_tracker):
    out = tmp_path / "h_walk_out"
    LLC.main(["--body", BODY, "--seeds", "0-0", "--task", "h_walk", "--max-steps", "6", "--sigmas", "0",
              "--tracker-id", stub_tracker("h_walk"), "--out", str(out), "--shard", "t"])
    z = np.load(out / BODY / "t.npz")
    assert set(OLD_LATENT + HD1_KEYS + ("ep", "t", "node_static", "node_asm", "asm_static", "q0_all", "n_policy", "nf",
                                         "node_parent", "asm_trunk")) <= set(z.files)
    meta = json.loads((out / BODY / "t.json").read_text())
    assert meta["task_view"]["event_slots"] == F.EVENT_SLOTS and meta["episodes"][0]["control"] == "wholebody"
    with pytest.raises(SystemExit, match="own directory"):
        LLC.main(["--body", BODY, "--seeds", "0-0", "--task", "h_steps", "--out", str(out), "--shard", "u"])


def test_one_directory_one_task_looks_into_body_subdirectories(tmp_path):
    (tmp_path / "t1").mkdir()
    (tmp_path / "t1" / "s0_manifest.json").write_text(json.dumps(dict(task="h_gap")))
    LCOL.assert_one_task(tmp_path, "h_gap")
    with pytest.raises(SystemExit, match="own directory"):
        LCOL.assert_one_task(tmp_path, "h_steps")


@pytest.mark.parametrize("mod", [LCOL, LLC])
def test_no_private_step_loop(mod):
    """Both collectors run their episode through harness.rollout: neither module calls a session's `step`."""
    tree = ast.parse(inspect.getsource(mod))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "step"]
    assert not calls
