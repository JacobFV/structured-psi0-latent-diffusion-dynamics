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
