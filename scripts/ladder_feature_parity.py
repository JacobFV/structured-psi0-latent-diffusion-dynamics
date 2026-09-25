"""Feature parity: replay a stored dataset episode (same robot/seed/n_distractors/exec_noise) through the CURRENT
featurizer + teacher and compare every PolicyInput field with the stored one (catches train/deploy featurization drift)."""
import sys, json
from pathlib import Path
import numpy as np
from rrp.data.collect import read_episode, collect_teacher_episode
from rrp.learning.packed import local_sensors
from rrp.morphology.catalog import workbench_robots
from rrp.sim.scenario import BUILDERS
from rrp.sim.native import Session
ds, eid = Path(sys.argv[1]), sys.argv[2]
pub = read_episode(ds / "episodes" / f"{eid}.public.pkl.gz")
m = pub["meta"]
robot = workbench_robots()[m["robot_key"]]()
s = Session(BUILDERS["pick_place"](robot, m["seed"], n_distractors=m["n_distractors"]), seed=m["seed"])
rec = collect_teacher_episode(s, exec_noise=m["exec_noise"], noise_seed=m["seed"])
new = rec.public
print("steps stored/new", len(pub["inputs"]), len(new["inputs"]), "status", m["status"], new["meta"]["status"])
T = min(len(pub["inputs"]), len(new["inputs"]))
worst = {}
for t in range(T):
    a, b = pub["inputs"][t], new["inputs"][t]
    fields = {"act_node_feats": (a.act_node_feats, b.act_node_feats), "q0": (pub["q0"][t], new["q0"][t]),
              "local": (local_sensors(a), local_sensors(b))}
    for bank in a.tokens:
        fields[f"tok_{bank}"] = (a.tokens[bank], b.tokens[bank])
    fields["relations"] = (np.asarray(a.relations), np.asarray(b.relations))
    fields["arm_action"] = (np.asarray(pub["actions"][t]["arm"]), np.asarray(new["actions"][t]["arm"]))
    fields["grip_action"] = (np.asarray(pub["actions"][t]["gripper"]), np.asarray(new["actions"][t]["gripper"]))
    for k, (x, y) in fields.items():
        if np.shape(x) != np.shape(y):
            worst[k] = ("shape", np.shape(x), np.shape(y), t)
            continue
        d = float(np.abs(np.asarray(x, float) - np.asarray(y, float)).max()) if np.size(x) else 0.0
        if d > worst.get(k, (0,))[0] if isinstance(worst.get(k, (0,))[0], float) else False or k not in worst:
            worst[k] = (d, t)
for k, v in worst.items():
    print(k, v)
if T:
    a, b = pub["inputs"][0], new["inputs"][0]
    d = np.abs(a.act_node_feats - b.act_node_feats) if a.act_node_feats.shape == b.act_node_feats.shape else None
    if d is not None:
        print("node feat cols differing at t=0:", sorted(set(np.nonzero(d > 1e-4)[1].tolist())))
    for bank in a.tokens:
        if a.tokens[bank].shape == b.tokens[bank].shape:
            dd = np.abs(a.tokens[bank] - b.tokens[bank])
            print(bank, "cols differing t=0:", sorted(set(np.nonzero(dd > 1e-4)[1].tolist()))[:40])
        else:
            print(bank, "shape", a.tokens[bank].shape, b.tokens[bank].shape)
cols = {}
for t in range(1, T):
    a, b = pub["inputs"][t], new["inputs"][t]
    for nm, x, y in (("node", a.act_node_feats, b.act_node_feats), *[(bk, a.tokens[bk], b.tokens[bk]) for bk in a.tokens]):
        if x.shape == y.shape:
            for c in np.nonzero((np.abs(x - y) > 1e-4).any(0))[0].tolist():
                cols.setdefault(nm, set()).add(c)
        else:
            cols.setdefault(nm, set()).add(("shape", x.shape, y.shape))
print("differing columns over t>=1:", {k: sorted(v, key=str)[:30] for k, v in cols.items()})
