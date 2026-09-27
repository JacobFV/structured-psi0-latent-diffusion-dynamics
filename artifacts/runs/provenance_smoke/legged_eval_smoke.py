"""W3 smoke: legged checkpoints (fingerprinted, legacy bare, refit realizer) load; compat IDs; 1 short episode each."""
import json, sys
from pathlib import Path
import torch
from rrp.evaluation.legged_latent_eval import LatentLeggedController, run_episode
from rrp.learning.legged_latent_train import _save
D = Path("artifacts/runs/provenance_smoke")
rep, flow = D / "legged_rep/representation.pt", D / "legged_flow/policy.pt"
st = torch.load(rep, weights_only=False)
legacy = D / "legged_rep_legacy/representation.pt"; legacy.parent.mkdir(exist_ok=True)
torch.save({k: v for k, v in st.items() if k != "_provenance"}, legacy)          # exactly the pre-W3 _save format
R2 = {k: v + 0.01 for k, v in st["R"].items()}                                    # stand-in "refit" system 0
_save(D / "legged_refit/realizer.pt", R=R2, cfg=dict(representation=str(rep)), step=1, representation=str(rep)) \
    if (D / "legged_refit").mkdir(exist_ok=True) is None else None
out = []
for name, kw in [("fingerprinted", dict(rep=rep)), ("legacy", dict(rep=legacy)),
                 ("refit", dict(rep=rep, realizer=D / "legged_refit/realizer.pt"))]:
    ctl = LatentLeggedController(flow, "cpu", seed=0, **{k: v for k, v in kw.items()})
    ctl.cfg = dict(ctl.cfg, representation=str(kw["rep"]))
    row, _ = run_episode(ctl, "go2", 10000, max_s=3.0)
    out.append(dict(case=name, source=row["source"], lsv=ctl.lsv, rcv=ctl.rcv, stats=row["stats"],
                    checkpoint_provenance=row["checkpoint_provenance"], success=row["success"]))
    print(json.dumps(out[-1]), flush=True)
assert out[0]["lsv"] == out[1]["lsv"] and out[0]["rcv"] == out[1]["rcv"], "legacy vs fingerprinted IDs differ"
assert out[2]["rcv"] != out[0]["rcv"] and out[2]["lsv"] == out[0]["lsv"], "refit not distinguishable"
assert all(o["stats"]["rejected"] == 0 and o["stats"]["packets"] > 0 for o in out)
(D / "legged_eval_smoke.json").write_text(json.dumps(out, indent=1))
print("OK")
