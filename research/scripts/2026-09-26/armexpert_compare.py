"""W7: old (v1) vs new (v2) arm teacher on identical bodies/seeds -> markdown table + JSON (raw rows as input)."""
import collections
import json
import sys

import numpy as np

v1p, v2p, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for p in (v1p, v2p) for l in open(p)]
g = collections.defaultdict(list)
for r in rows:
    g[(r["robot"], r["version"])].append(r)
bodies = sorted({r["robot"] for r in rows})
TRAIN = {"panda_pg2", "parm6_tf3", "parm5_pg2", "parm5_tf3", "parm5l_tf3", "parm5s_pg2", "parm6_pg2", "parm7_pg2",
         "parm7_tf3", "sawyer_pg2", "sawyer_tf3", "ur5e_pg2", "ur5e_tf3"}
HELD = {"parm5s_tf3", "parm5l_pg2"}
keys = [("joint_cmd_jerk_peak", "jerk cmd"), ("joint_meas_jerk_peak", "jerk meas"), ("tcp_meas_jerk_peak", "TCP jerk"),
        ("joint_cmd_acc_peak", "acc cmd"), ("vel_jump_switch_max", "dv@switch"), ("joint_limit_margin_min", "lim margin"),
        ("time_s", "time"), ("pen_hand_max_grasp_m", "pen"), ("slip_max_m", "slip")]
res = {}
for b in bodies:
    e = {}
    for v in ("v1", "v2"):
        rs = [r for r in g[(b, v)] if r.get("feasible")]
        s = [r for r in rs if r["success"]]
        d = dict(n_seeds=len(g[(b, v)]), feasible=len(rs), success=len(s),
                 failure_stages=dict(collections.Counter(r["failure_stage"] for r in rs if not r["success"])),
                 lift_before_held=sum(bool(r.get("lift_before_held")) for r in rs),
                 held_lost_before_open=sum(1 for r in rs if (r.get("held_lost_before_open") or 0) > 0))
        for k, _ in keys:
            vals = np.array([r[k] for r in rs if r.get(k) is not None], float)
            d[k] = dict(median=float(np.median(vals)), p90=float(np.percentile(vals, 90)), max=float(vals.max()),
                        min=float(vals.min())) if vals.size else None
        e[v] = d
    res[b] = e
tot = {v: [sum(res[b][v][x] for b in bodies) for x in ("success", "feasible")] for v in ("v1", "v2")}
json.dump(dict(inputs=[v1p, v2p], per_body=res, totals=tot), open(out, "w"), indent=1)
m = lambda d, k, f="{:.0f}": (f.format(d[k]["median"]) if d[k] else "-")
print("| body | split | success v1 | success v2 | v1 failure stages | jerk cmd med v1 -> v2 (rad/s^3) | jerk meas med | TCP jerk meas med (m/s^3) | dv at switch med (rad/s per tick) | joint-limit margin min | time med (s) | grasp pen med (mm) | slip med (mm) | lift before held v1/v2 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for b in bodies:
    a, c = res[b]["v1"], res[b]["v2"]
    sp = "train" if b in TRAIN else ("held-out source" if b in HELD else "target (demos only)")
    print(f"| {b} | {sp} | {a['success']}/{a['feasible']} | {c['success']}/{c['feasible']} | "
          f"{', '.join(f'{k} {v}' for k, v in a['failure_stages'].items()) or '-'} | "
          f"{m(a,'joint_cmd_jerk_peak')} -> {m(c,'joint_cmd_jerk_peak')} | {m(a,'joint_meas_jerk_peak')} -> {m(c,'joint_meas_jerk_peak')} | "
          f"{m(a,'tcp_meas_jerk_peak','{:.1f}')} -> {m(c,'tcp_meas_jerk_peak','{:.1f}')} | "
          f"{m(a,'vel_jump_switch_max','{:.2f}')} -> {m(c,'vel_jump_switch_max','{:.2f}')} | "
          f"{a['joint_limit_margin_min']['min']:.3f} -> {c['joint_limit_margin_min']['min']:.3f} | "
          f"{m(a,'time_s','{:.1f}')} -> {m(c,'time_s','{:.1f}')} | "
          f"{1000*a['pen_hand_max_grasp_m']['median']:.0f} -> {1000*c['pen_hand_max_grasp_m']['median']:.0f} | "
          f"{1000*a['slip_max_m']['median']:.0f} -> {1000*c['slip_max_m']['median']:.0f} | {a['lift_before_held']}/{c['lift_before_held']} |")
print(f"\nTOTAL success v1 {tot['v1'][0]}/{tot['v1'][1]}, v2 {tot['v2'][0]}/{tot['v2'][1]}")
