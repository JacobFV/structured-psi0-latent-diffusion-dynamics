"""Held-out source competence of a BC snapshot with the sealed-protocol harness (as sprint_bc did for direct1701_u12000:
rrp.training.baseline_campaign.evaluate_checkpoint, parm5s_tf3 + parm5l_pg2, 50 seeds from the protocol seed_start)."""
import json
import sys
from pathlib import Path

from rrp.training.baseline_campaign import evaluate_checkpoint

ck, out, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
proto = json.loads(Path("configs/eval/latent_slice1.json").read_text())
out.parent.mkdir(parents=True, exist_ok=True)
s = evaluate_checkpoint(ck, proto["source_heldout_eval_robots"], out, protocol=proto, method="baseline_direct_action",
                        seed=seed, n=50, device="cpu")
print(json.dumps({k: (v.get("successes"), v.get("attempted")) for k, v in s.items() if k != "_meta"}))
