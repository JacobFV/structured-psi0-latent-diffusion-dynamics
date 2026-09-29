"""Closed-loop failure-localization ladder: thin wrapper over rrp.evaluation.ladder_cli (moved there, D-126; same CLI,
prints and outputs). Examples:
  ladder.py --route teacher   --robot panda_pg2 --n 30 --out artifacts/runs/ladder_v1/panda_pg2
  ladder.py --route oracle    --robot panda_pg2 --n 30 --rep artifacts/runs/latent_sem_v1/representation.pt --out ...
  ladder.py --route generated --robot panda_pg2 --n 30 --flow artifacts/runs/flow_latent_sem_v2/policy.pt --out ...
  ladder.py --disturbance --route oracle ...   (fixed-packet joint disturbance via latent_eval.disturbance_test)
Writes <out>/<route>[_tag].jsonl (one row per episode) and <out>/<route>[_tag].summary.json."""
from rrp.harness.eval.ladder_cli import main

if __name__ == "__main__":
    main()
