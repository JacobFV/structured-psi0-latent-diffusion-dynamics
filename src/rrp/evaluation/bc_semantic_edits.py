"""Semantic-edit suite on the plain behaviour-cloning positive control (sprint_bc).

Same scenes, conditions, context edits and physical measurements as rrp.evaluation.latent_semantic_edits, but the
controller is a plain FlowPolicy (LearnedPolicy: direct-action or action-only-codec baseline). BC has no packet: the
edit is applied to the public context BC observes when it generates each chunk (snapshot -> context_edit -> chunk ->
restore), exactly where the generated route regenerates its packet. The physical scene is never changed. Flow noise is
re-seeded per (seed, call) so control and edit episodes are paired. Packet-only conditions (orthogonal_matched) and
control_replay are n/a. Source label: learned:<ckpt>.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rrp.control.teachers import PickPlaceTeacher
from rrp.evaluation import latent_semantic_edits as se
from rrp.evaluation.latent_causal import _body_pos, _key, _tcp


class BCSource:
    label = "learned"

    def __init__(self, policy, name):
        self.pol, self.name = policy, name

    def chunk(self, s, cond, goal_off, key):
        self.pol.gen.manual_seed(int(key))
        snap = s.snapshot()
        try:
            se.context_edit(cond, s, goal_off)
            ch = self.pol.chunks([s])[0]
        finally:
            s.restore(snap)
        return ch


def _scene(robot, key, scene):
    if hasattr(se, "_scene"):
        return se._scene(robot, key, scene)
    if scene != "pick_place":
        raise ValueError("paired scenes need the acceptance track's latent_semantic_edits")
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    return Session(BUILDERS["pick_place"](robot, key, n_distractors=max(1, key % 3)), seed=key)


def run_bc_condition(src: BCSource, robot, robot_key, seed, cond, *, max_steps=300, replan=8, g=0.12,
                     scene="pick_place"):
    s = _scene(robot, seed, scene)
    goal_off = se.goal_offset(s, g)
    base = dict(robot=robot_key, seed=seed, condition=cond, scene=scene)
    if not PickPlaceTeacher(s).feasibility()["feasible"]:
        return dict(base, skipped="infeasible_control")
    if cond in ("orthogonal_matched", "control_replay"):
        return dict(base, skipped="n/a_no_packet")
    z0 = {b: _body_pos(s, b) for b in ("cube", "distractor0", "target_zone")}
    goal_new = z0["target_zone"] + goal_off
    lift = {"cube": 0.0, "distractor0": 0.0}
    objs = {s.model.body(b).id: b for b in lift}
    contacts = getattr(se, "robot_contacts", None)
    rb = se.robot_body_ids(s) if hasattr(se, "robot_body_ids") else None
    fcon = {b: None for b in lift}
    tcp_tr, calls, rejected = [_tcp(s)], 0, 0
    for step in range(max_steps):
        if step % replan == 0 or not s.executor.queue:
            ch = src.chunk(s, cond, goal_off, _key(seed, calls))
            calls += 1
            try:
                s.submit_chunk(ch, execute_prefix=replan)
            except Exception:                          # stale chunk: hold (counted)
                rejected += 1
        s.step(None)
        for b in lift:
            lift[b] = max(lift[b], float(_body_pos(s, b)[2] - z0[b][2]))
        if contacts is not None:
            for b, _ in contacts(s, objs, rb):
                if fcon[b] is None:
                    fcon[b] = step
        tcp_tr.append(_tcp(s))
        if s.runtime.succeeded() or _body_pos(s, "cube")[2] < -0.05:
            break
    held = {b: any(b in v for v in s.truth().held_by.values()) for b in lift}
    fin = {b: _body_pos(s, b) for b in lift}
    xy = lambda a, b: float(np.linalg.norm((a - b)[:2]))
    placed = lambda b, goal: bool(xy(fin[b], goal) < se.ZONE_R and fin[b][2] < 0.06 and not held[b])
    tcp_tr = np.array(tcp_tr)
    row = dict(base, route=src.label, source=f"learned:{src.name}", steps=step + 1, system_i_calls=calls,
               packets_rejected=rejected, privileged_success=bool(s.privileged_success()),
               lifted={b: v > 0.03 for b, v in lift.items()}, max_lift_m=lift, held_at_end=held,
               moved_xy_m={b: xy(fin[b], z0[b]) for b in lift},
               final_xy_to_zone={b: xy(fin[b], z0["target_zone"]) for b in lift},
               final_xy_to_shifted_goal={b: xy(fin[b], goal_new) for b in lift},
               placed_in_zone={b: placed(b, z0["target_zone"]) for b in lift},
               placed_at_shifted_goal={b: placed(b, goal_new) for b in lift},
               min_tcp_dist={b: float(np.linalg.norm(tcp_tr - z0[b], axis=1).min()) for b in lift},
               goal_offset=goal_off.tolist(), tcp=tcp_tr[::4].round(4).tolist(),
               objects0={b: z0[b].round(4).tolist() for b in z0})
    if hasattr(se, "approach_metrics"):
        row.update(se.approach_metrics(tcp_tr, {b: z0[b] for b in lift}, fcon))
    if scene == "paired":
        row["patient_color"] = s.scenario.meta["cube_color"]
        row["rebind_color"] = s.scenario.object("distractor0").descriptor
    row["followed"] = se.followed(row)
    return row


def bc_semantic_suite(src, robot_key, seeds, conditions, *, max_steps=300, scene="pick_place", out_path=None,
                      log=print):
    from rrp.morphology.catalog import workbench_robots
    robot = workbench_robots()[robot_key]()
    rows = []
    for sd in seeds:
        for c in conditions:
            t0 = time.time()
            r = run_bc_condition(src, robot, robot_key, sd, c, max_steps=max_steps, scene=scene)
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            if out_path:
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            log(f"[bc-semantic] {robot_key} {sd} {c}: " + (r.get("skipped") or
                f"followed={r['followed']} lifted={r['lifted']} placed_zone={r['placed_in_zone']} "
                f"approached_first={r.get('approached_first')} min_d={ {k: round(v, 3) for k, v in r['min_tcp_dist'].items()} }"),
                flush=True)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--robots", default="panda_pg2")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=3_000_000)
    ap.add_argument("--scene", choices=["pick_place", "paired"], default="pick_place")
    ap.add_argument("--conditions", default="control,rebind_obj,goal_shift,irrelevant_distractor")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed_start < 3_000_000:
        raise SystemExit("dev seeds must be >= 3,000,000")
    from rrp.policy.runner import LearnedPolicy
    pol = LearnedPolicy.from_checkpoint(a.policy, device="cpu", nfe=8, execute_prefix=8)
    src = BCSource(pol, a.label)
    seeds = (se.paired_edit_keys(a.seed_start, a.episodes) if a.scene == "paired"
             else list(range(a.seed_start, a.seed_start + a.episodes)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rp = out / f"semantic_rows_learned_{a.scene}.jsonl"
    if rp.exists():
        rp.unlink()
    rows = []
    for r in a.robots.split(","):
        rows += bc_semantic_suite(src, r, seeds, tuple(a.conditions.split(",")), max_steps=a.max_steps,
                                  scene=a.scene, out_path=rp)
    summ = dict(meta=dict(route="learned", source=f"learned:{a.label}", checkpoint=a.policy, robots=a.robots,
                          scene=a.scene, seeds=[seeds[0], seeds[-1]], conditions=a.conditions,
                          edit="context edit applied to BC's public observation at each chunk (no packet)"),
                summary=se.summarize_semantic(rows))
    (out / f"semantic_summary_learned_{a.scene}.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ["summary"], indent=1, default=str))


if __name__ == "__main__":
    main()
