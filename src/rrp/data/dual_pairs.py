"""Pair index for the manipulator-assignment family (assign_left / assign_right).

A pair = two episodes with the same robot pair, seed and execution-noise level whose task graphs differ ONLY in the
manipulator bound as actor of `take`/`place`. The index proves scene identity by rebuilding both initial scenes
and hashing the initial physics state (qpos, qvel, body poses, mocap target zone), records both demonstrations'
status (scripted_teacher, privileged evaluator), which manipulator actually held the bar (privileged labels,
diagnostic) and, when a pack is given, the ep_idx of each episode's rows.

Intervention semantics for the acceptance track: in the scene of a pair, rebinding the actor entity of take/place
from `left` to `right` (or back) is a VALID manipulator-assignment edit; the expected behaviour is the other
episode of the pair (the other arm grasps and places the bar; the non-assigned arm stays staged).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def scene_fingerprint(task: str, pair: str, seed: int) -> str:
    from rrp.control.dual_validate import make_session
    s = make_session(task, pair, seed)
    m, d = s.model, s.data
    h = hashlib.sha256()
    for x in (d.qpos, d.qvel, m.body_pos, m.body_quat, d.mocap_pos):
        h.update(np.round(np.asarray(x, np.float64), 7).tobytes())
    return h.hexdigest()[:16]


def task_graph_hash(task: str) -> str:
    from rrp.sim.dual_scenarios import assign_task
    return hashlib.sha256(json.dumps(assign_task(task.split("_", 1)[1]), sort_keys=True).encode()).hexdigest()[:16]


def build_pair_index(ds_dir: Path, packed_dir: Path | None = None, check_scene: bool = True) -> dict:
    from rrp.data.collect import read_episode
    man = json.loads((ds_dir / "manifest.json").read_text())
    ep_index = {}
    if packed_dir is not None and (packed_dir / "meta.json").exists():
        ep_index = json.loads((packed_dir / "meta.json").read_text()).get("episode_index", {})
    groups: dict[tuple, dict] = {}
    for m in man["episodes"]:
        if "episode_id" not in m or not m.get("task", "").startswith("assign_"):
            continue
        key = (m["robot_key"], m["seed"], float(m.get("exec_noise", 0.0)))
        groups.setdefault(key, {})[m["task"].split("_", 1)[1]] = m
    pairs = []
    for (rk, seed, noise), g in sorted(groups.items()):
        if set(g) != {"left", "right"}:
            continue
        row = dict(pair_id=f"assign:{rk}:s{seed}" + (f":dart{int(noise * 1000)}" if noise else ""), robot_pair=rk,
                   seed=seed, exec_noise=noise, split=g["left"].get("split_lineage", {}).get("split"),
                   source="scripted_teacher", variants={})
        for arm in ("left", "right"):
            m = g[arm]
            held = None
            if m["status"] in ("success", "failure"):
                prv = read_episode(ds_dir / "episodes" / f"{m['episode_id']}.private.pkl.gz")
                mans = list(prv["manipulators"])
                hs = np.array([lab["slot_held"][0] for lab in prv["labels"]])   # slot 0 = bar
                held = {e: int(hs[:, k].sum()) for k, e in enumerate(mans)} if len(hs) else {}
            row["variants"][arm] = dict(episode_id=m["episode_id"], task=m["task"], status=m["status"],
                                        public_runtime_success=m.get("public_runtime_success"), steps=m.get("steps"),
                                        task_hash=m.get("task_hash"), held_steps_by_manipulator=held,
                                        packed_ep_idx=ep_index.get(m["episode_id"]))
        row["both_success"] = all(v["status"] == "success" for v in row["variants"].values())
        row["holder_matches_assignment"] = all(
            v["held_steps_by_manipulator"] is not None and v["held_steps_by_manipulator"].get(a, 0) > 0 and
            v["held_steps_by_manipulator"].get("left" if a == "right" else "right", 0) == 0
            for a, v in row["variants"].items() if v["status"] == "success")
        if check_scene:
            fl, fr = scene_fingerprint("assign_left", rk, seed), scene_fingerprint("assign_right", rk, seed)
            row.update(scene_fingerprint=fl, scene_identical=fl == fr)
        pairs.append(row)
    summ = dict(n_pairs=len(pairs), both_success=sum(p["both_success"] for p in pairs),
                scene_identical=sum(bool(p.get("scene_identical")) for p in pairs),
                holder_matches_assignment=sum(p["holder_matches_assignment"] for p in pairs))
    by = {}
    for p in pairs:
        k = f"{p['split']}|{p['robot_pair']}|{'dart' if p['exec_noise'] else 'clean'}"
        b = by.setdefault(k, dict(pairs=0, both_success=0))
        b["pairs"] += 1
        b["both_success"] += p["both_success"]
    return dict(dataset=str(ds_dir), packed=str(packed_dir) if packed_dir else None, family="assign_pick_place",
                task_graph_hashes={t: task_graph_hash(t) for t in ("assign_left", "assign_right")},
                intervention="rebind actor of take/place: left <-> right (valid manipulator-assignment edit); "
                             "expected behaviour = the other variant of the pair",
                summary=summ, by_group=by, pairs=pairs)
