"""Closed-loop batched evaluation. Every attempted episode is recorded (success, timeout,
refusal/rejection, crash) with raw counts; the denominator is all attempted episodes."""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from rrp.contracts.errors import StaleActionError


@dataclass
class EpisodeResult:
    robot: str
    seed: int
    method: str
    checkpoint: str
    outcome: str                 # success | failure | timeout | crash | infeasible
    privileged_success: bool
    public_success: bool
    steps: int
    policy_calls: int
    chunk_rejections: int
    command_rejections: int
    sim_time: float
    wall_s: float
    events: dict
    object_fell: bool = False
    note: str = ""


def evaluate(policy, robot_key: str, seeds: list[int], *, method: str, checkpoint: str, max_steps: int = 300,
             batch: int = 16, n_distractors_fn=lambda s: s % 3, out_path: Path | None = None,
             task: str = "pick_place", check_feasible: bool = True) -> list[EpisodeResult]:
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.control.teachers import PickPlaceTeacher
    robot = workbench_robots()[robot_key]()
    results = []
    for i in range(0, len(seeds), batch):
        group = seeds[i:i + batch]
        sessions, meta = [], []
        for sd in group:
            kw = {"n_distractors": n_distractors_fn(sd)} if task == "pick_place" else {}
            s = Session(BUILDERS[task](robot, sd, **kw), seed=sd)
            feas = PickPlaceTeacher(s).feasibility()["feasible"] if (check_feasible and task == "pick_place") else True
            sessions.append(s)
            meta.append(dict(seed=sd, t0=time.time(), done=not feas, outcome="infeasible" if not feas else None,
                             calls=0, rej=0, cmd_rej=0, steps=0, fell=False))
        for step in range(max_steps):
            active = [k for k, m in enumerate(meta) if not m["done"]]
            if not active:
                break
            need = [k for k in active if not sessions[k].executor.queue]
            if need:
                try:
                    chunks = policy.chunks([sessions[k] for k in need])
                    for k, ch in zip(need, chunks):
                        meta[k]["calls"] += 1
                        try:
                            sessions[k].submit_chunk(ch, execute_prefix=policy.execute_prefix)
                        except StaleActionError:
                            meta[k]["rej"] += 1
                except Exception as e:  # noqa: BLE001
                    for k in need:
                        meta[k].update(done=True, outcome="crash", note=repr(e)[:200])
                    continue
            for k in active:
                if meta[k]["done"]:
                    continue
                s = sessions[k]
                try:
                    r = s.step(None)
                except FloatingPointError as e:
                    meta[k].update(done=True, outcome="crash", note=str(e))
                    continue
                meta[k]["steps"] += 1
                if r.rejected:
                    meta[k]["cmd_rej"] += 1
                cube_z = s.data.xpos[s.model.body("cube").id][2] if task == "pick_place" else 0.0
                if cube_z < -0.05:
                    meta[k].update(done=True, outcome="failure", fell=True)
                elif s.runtime.succeeded():
                    meta[k].update(done=True)
        for k, s in enumerate(sessions):
            m = meta[k]
            priv = bool(s.privileged_success()) if m["outcome"] != "infeasible" else False
            pub = bool(s.runtime.succeeded())
            if m["outcome"] is None:
                m["outcome"] = "success" if priv else ("timeout" if m["steps"] >= max_steps else "failure")
            results.append(EpisodeResult(robot_key, m["seed"], method, checkpoint, m["outcome"], priv, pub, m["steps"],
                                         m["calls"], m["rej"], m["cmd_rej"], float(s.data.time),
                                         time.time() - m["t0"], {e: v.status for e, v in s.runtime.instances.items()},
                                         m["fell"], m.get("note", "")))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
    return results


def summarize(results: list[EpisodeResult]) -> dict:
    att = [r for r in results if r.outcome != "infeasible"]
    n = len(att)
    k = sum(r.privileged_success for r in att)
    from rrp.evaluation.statistics import wilson
    lo, hi = wilson(k, n)
    return dict(attempted=n, successes=k, success_rate=k / n if n else None, wilson95=[lo, hi],
                infeasible=sum(r.outcome == "infeasible" for r in results),
                outcomes={o: sum(r.outcome == o for r in results) for o in sorted({r.outcome for r in results})},
                public_private_agreement=sum(r.privileged_success == r.public_success for r in att) / n if n else None)
