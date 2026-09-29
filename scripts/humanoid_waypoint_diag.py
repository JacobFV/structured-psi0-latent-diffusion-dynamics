"""W13 diag (peer CPU): one waypoint_contact episode with a given actor; prints event statuses, base pose and command every 2 s."""
import sys
import numpy as np
import rrp.envs.legged as _L
from rrp.envs.legged import LeggedSession, build_waypoint_contact
from rrp.envs.legged_tracker import LearnedTracker
from rrp.teachers.legged import WaypointTeacher
body, actor, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
_L.load_tracker = lambda key, binding, meta, kind="auto": LearnedTracker(actor, binding, key)
sc = build_waypoint_contact(body, seed)
s = LeggedSession(sc, tracker_kind="learned", seed=seed)
s.reset(seed)
te = WaypointTeacher(s)
print("waypoints", sc.meta["waypoints"])
for k in range(420):
    cmd = te.act()
    s.step(cmd)
    if k % 20 == 0 or te.done or s.fell:
        st = {e: i.status for e, i in s.runtime.instances.items()}
        x, y, yaw = s.base_pose_truth()
        print(f"t={k/10:5.1f} phase={te.phase:9s} cmd={np.round(cmd.groups['base_velocity'],2)} pose=({x:.2f},{y:.2f},{yaw:.2f}) {st}")
    if te.done or s.fell:
        break
print("privileged_success", s.privileged_success(), "fell", s.fell)
