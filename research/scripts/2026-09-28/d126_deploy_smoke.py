"""D-126 deploy smoke (peer, lowest priority): estimator vs truth on a real body (go2, learned tracker, teacher route),
safety enforce + long mode + tripwire + static check. Plumbing/sanity only: 1 seed, not a result."""
import json
import sys
import time
from pathlib import Path

import torch

from rrp.evaluation.deploy_eval import DeployOptions
from rrp.evaluation.legged_latent_eval import run_episode
from rrp.evaluation import privileged_audit as pa

torch.set_num_threads(1)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/runs/d126deploy_smoke")
out.mkdir(parents=True, exist_ok=True)
res = dict(static=pa.static_check(), tripwire=[pa.dynamic_tripwire("go2", 0, b) for b in ("truth_noise", "estimator")])
for name, dep in (("est_long", DeployOptions(base_state_source="estimator", eval_mode="long", long_s=12.0, window_s=2.0)),
                  ("est_safety", DeployOptions(base_state_source="estimator", safety="enforce")),
                  ("default", None)):
    t0 = time.time()
    row, _ = run_episode(None, "go2", 10000, max_s=20.0, **({} if dep is None else dict(deploy=dep)))
    keep = {k: row.get(k) for k in ("source", "success", "fell", "sim_time", "failure_stage", "safety", "long_run",
                                     "base_state", "deploy")}
    keep["wall_s"] = time.time() - t0
    res[name] = keep
    print(name, json.dumps({k: keep[k] for k in ("success", "fell", "sim_time", "wall_s")}), flush=True)
(out / "smoke.json").write_text(json.dumps(res, indent=1, default=str))
for name in ("est_long", "est_safety"):
    b = res[name].get("long_run") or res[name].get("base_state") or {}
    print(name, "est_vel_rmse per window:", [round(w.get("est_vel_rmse", -1), 3) for w in b.get("windows", [])],
          "true speed:", [round(w["mean_speed_true"], 2) for w in b.get("windows", [])],
          "est speed:", [None if w["mean_speed_est"] is None else round(w["mean_speed_est"], 2) for w in b.get("windows", [])])
print("static violations:", res["static"]["violations"])
