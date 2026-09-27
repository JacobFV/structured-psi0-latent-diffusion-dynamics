"""Stage-A shared-clip starvation check (D-085 / lead request for W8): from each run's train_log.jsonl, the logged
pre-clip grad norm `gn` (clip_grad_norm_ at 1.0) -> median grad norm and mean effective update scale min(1, 1/gn),
overall and per third of training (same statistic as scripts/legged_fixsem_geom.py).
usage: python scripts/legged8_clipscale.py OUT.json LABEL=RUN_DIR ..."""
import json
import sys
from pathlib import Path

import numpy as np

res = {}
for arg in sys.argv[2:]:
    lab, d = arg.split("=", 1)
    f = Path(d) / "train_log.jsonl"
    if not f.exists():
        res[lab] = dict(run=d, missing=True)
        continue
    rows = [json.loads(l) for l in open(f) if '"gn"' in l]
    gn = np.asarray([r["gn"] for r in rows], float)
    st = np.asarray([r["step"] for r in rows])
    thirds = np.array_split(np.arange(len(gn)), 3)
    res[lab] = dict(run=d, n_logged=len(gn), last_step=int(st[-1]), median_grad_norm=round(float(np.median(gn)), 3),
                    mean_update_scale=round(float(np.minimum(1, 1 / gn).mean()), 4),
                    by_third=[dict(steps=f"{int(st[i[0]])}-{int(st[i[-1]])}", median_grad_norm=round(float(np.median(gn[i])), 3),
                                   mean_update_scale=round(float(np.minimum(1, 1 / gn[i]).mean()), 4)) for i in thirds if len(i)])
    print(f"{lab:28s} steps<={res[lab]['last_step']:6d} median gn {res[lab]['median_grad_norm']:9.3f} scale {res[lab]['mean_update_scale']:.4f} "
          + " | ".join(f"{t['steps']}: {t['median_grad_norm']:.2f}/{t['mean_update_scale']:.3f}" for t in res[lab]["by_third"]))
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
