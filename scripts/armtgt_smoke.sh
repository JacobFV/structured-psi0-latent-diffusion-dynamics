#!/usr/bin/env bash
# ARM V6 TARGET BODIES smoke (plumbing only, NEVER reported; protocol dev_rule: no target data or eval scenes touched):
# (1) target_eval --smoke on the NON-target source body parm6_tf3, dev seeds, 2 episodes, both routes (latent / BC);
# (2) the adaptation functions the target_adapt stage calls (flow SFT, budgeted system-0 refit, BC SFT) for a few steps
#     on the NON-target v6 source pack (the stage itself only accepts target packs, which stay untouched).
# Runs inside ONE peer lease: RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armtgt scripts/peer_run.sh ... -- bash scripts/armtgt_smoke.sh
set -euo pipefail
export RRP_GRASP_CONTACT=v2.1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
R=artifacts/runs; O=$R/ladder_smoke/armtgt_smoke; mkdir -p $O
F=$R/armv6/arm6-semfix/flow_ft-gdag2h_s1/policy.pt
REP=$R/armv6/arm6-semfix/refit-gendag3_noqd_s1/representation.pt
BC=$R/armexpert_bcv6/baseline_direct_action/seed1701/source/policy.pt
PK=artifacts/packed/latent_pp_v6dart_s1_H16
$PY -m rrp.evaluation.target_eval --protocol configs/eval/latent_slice1.json --robot parm6_tf3 --route generated \
  --prev-action zero --kind zero_shot --tag smoke_latent --out $O/eval_latent --flow $F --rep $REP --smoke --smoke-episodes 2
$PY -m rrp.evaluation.target_eval --protocol configs/eval/latent_slice1.json --robot parm6_tf3 --route learned \
  --prev-action zero --kind zero_shot --tag smoke_bc --out $O/eval_bc --policy $BC --policy-label bcv6_direct1701_final --smoke --smoke-episodes 2
$PY - <<PY
import json
from pathlib import Path
from rrp.training.latent_train import sft_latent_flow, refit_realizer
from rrp.training.sft import sft_packed
from rrp.models.checkpoint import load_checkpoint
O = Path("$O")
r1 = sft_latent_flow(Path("$F"), Path("$PK"), 5, seed=1701, out_dir=O / "flow_sft", steps=5, lr=1e-4)
src = load_checkpoint(Path("$REP"), map_location="cpu")["config"]
cfg = dict(representation="$REP", packed_dir="$PK", steps=5, batch_size=128, lr=1e-4, seed=1701, init="old",
           episode_budget=5, budget_seed=1701, prefetch_workers=0, zero_prev_action=bool(src.get("zero_prev_action")),
           realizer_anchor=bool(src.get("realizer_anchor")), realizer_drop_qd=bool(src.get("realizer_drop_qd")), name="armtgt_smoke_rz")
(O / "rz").mkdir(parents=True, exist_ok=True)
r2 = refit_realizer(cfg, O / "rz")
r3 = sft_packed(Path("$BC"), Path("$PK"), 5, seed=1701, out_dir=O / "bc_sft", steps=5, lr=1e-4)
print("SMOKE_ADAPT_OK", json.dumps(dict(flow_sft=str(r1)[:200], refit=str(r2)[:200], bc_sft=str(r3)[:200])))
PY
for f in $O/eval_latent/generated_smoke_latent.summary.json $O/eval_bc/learned_smoke_bc.summary.json $O/flow_sft/policy.pt $O/rz/representation.pt $O/bc_sft/policy.pt; do test -s $f || { echo "MISSING $f"; exit 1; }; done
echo SMOKE_ALL_OK
