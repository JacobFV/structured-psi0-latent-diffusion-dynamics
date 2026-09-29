#!/usr/bin/env bash
# W13 P1 tracker gate (peer, CPU; run inside an ops lease): C-MuJoCo validation of a GPU-trained actor with full self-collision,
# contact_v2, sourced limits. Writes <out>/val.json (tracker_validation, 10 seeds, --robust), <out>/gate/gate_report.json (D-112
# tracker gate, rrp.evaluation.gates), <out>/waypoint.json (scripted_teacher on waypoint_contact, 20 dev seeds) and
# <out>/lab_gate.json (no-fall 1.0, forward >= 0.8, turn >= 0.5, slip < 0.15). Exit 0 always; verdicts are in the JSON.
# usage: scripts/humanoid_tracker_gate.sh <body> <actor.pt> <out_dir>
set -uo pipefail
body=$1; actor=$2; out=$3
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
mkdir -p "$out/gate"
export PYTHONPATH=src RRP_CONTACT_MODEL=v2
$PY -m rrp.harness.eval.tracker_validation --body "$body" --kind learned --actor "$actor" --seeds 10 --contact v2 --robust \
    --gate-dir "$out/gate" --out "$out/val.json" > "$out/val.log" 2>&1; echo "validation rc=$?"
$PY scripts/contact_waypoint_eval.py "$body" "$actor" 10000 20 "$out/waypoint.json" > "$out/waypoint.log" 2>&1; echo "waypoint rc=$?"
$PY - "$out" <<'PYEOF'
import json, sys
out = sys.argv[1]
v = json.load(open(f"{out}/val.json"))
g = v["gate"]
lab = dict(no_fall=g.get("no_fall_rate"), forward=g.get("forward_ratio"), turn=g.get("turn_ratio"),
           slip=(g.get("contact_gate") or {}).get("slip_ratio"))
lab["passed"] = bool(lab["no_fall"] == 1.0 and (lab["forward"] or 0) >= 0.8 and (lab["turn"] or 0) >= 0.5
                     and (lab["slip"] if lab["slip"] is not None else 1) < 0.15)
try:
    rep = json.load(open(f"{out}/gate/gate_report.json"))
    lab["d112_verdict"] = rep.get("verdict")
    lab["d112_criteria"] = {c["name"]: [c.get("value"), c.get("status")] for c in rep.get("criteria", [])}
except Exception as e:
    lab["d112_verdict"] = f"missing: {e}"
try:
    w = json.load(open(f"{out}/waypoint.json"))
    lab.update(waypoint_success=w["success"], waypoint_fell=w["fell"], waypoint_n=w["n"])
except Exception as e:
    lab["waypoint"] = f"missing: {e}"
json.dump(lab, open(f"{out}/lab_gate.json", "w"), indent=1, default=str)
print(json.dumps(lab, default=str))
PYEOF
