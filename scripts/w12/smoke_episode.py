import sys, time, json
sys.path.insert(0, "src")
from rrp.evaluation.dual_teacher_quality import run_audit_episode
task, pair, seed, ver = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
t0 = time.time()
r = run_audit_episode(task, pair, seed, max_steps=int(sys.argv[5]) if len(sys.argv) > 5 else 1200,
                      teacher_version=None if ver == "v2" else ver)
keep = {k: r.get(k) for k in ("task", "pair", "seed", "status", "failure_phase", "steps", "penetration_max_m", "gate", "teacher_limits", "wall_s", "statuses")}
keep["per_arm"] = {e: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for e, p in (r.get("per_arm") or {}).items()}
c = r.get("contact") or {}
keep["contact"] = {k: v for k, v in c.items() if k in ("cf_held_pos_drift_grip_max_m", "cf_held_rot_drift_grip_max_rad", "cf_support_anchor_slip_max_m", "cf_contact_order_error", "cf_contact_event_times")}
print(json.dumps(keep, default=str))
