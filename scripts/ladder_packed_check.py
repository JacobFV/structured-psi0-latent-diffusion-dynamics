"""Stage-A realization error on the training pack for one robot, split arm/gripper and by phase j."""
import json, sys
import torch
from rrp.evaluation.ladder import packed_realization_check
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    from rrp.ops.gpu import apply_cap
    apply_cap()
rep, robot, out = sys.argv[1], sys.argv[2], sys.argv[3]
r = packed_realization_check(rep, "artifacts/packed/latent_pp_v3dart_s1_H16", robot, dev)
open(out, "w").write(json.dumps(r, indent=1))
for k, v in r["by_node_and_j"].items():
    print(k, {j: (round(x["err"], 4), round(x["zero_action"], 4)) for j, x in v.items()})
