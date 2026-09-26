#!/usr/bin/env bash
# Host-side driver (ladder sprint): full recipe for one binding-v4 bundle (v = sem|nosem) on the peer:
# r3 system 0 -> R2 rollouts with its own flow collecting system-0 + generator DAgger -> system-0 refit (GPU) ->
# flow DAgger fine-tune (GPU) -> R1 stateless + R2 evals (CPU). Usage: ladder_bindv4_chain.sh sem|nosem
set -uo pipefail
v=$1
cd ~/work/rrp-wt/ladder; export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/ladder
R=/dev/shm/rrp-brandonin/repo/artifacts/runs
PYP=/dev/shm/rrp-brandonin/venv/bin/python
wait_f() { until ssh gb10-direct "[ -f $1 ]"; do sleep 30; done; }
wait_n() { until [ "$(ssh gb10-direct "ls $1 2>/dev/null | wc -l")" -ge $2 ]; do sleep 30; done; }
wait_f $R/ladder_rz_bindv4${v}_bcdag3_noqd/representation.pt
echo "$(date) r3 ready"
REP=artifacts/runs/ladder_rz_bindv4${v}_bcdag3_noqd/representation.pt
F=artifacts/runs/ladder_flow_bindv4${v}/policy.pt
for g in "panda_pg2 parm5_pg2 parm5_tf3 parm5l_tf3 parm5s_pg2 parm6_pg2" "parm6_tf3 parm7_pg2 parm7_tf3 sawyer_pg2 sawyer_tf3 ur5e_pg2 ur5e_tf3"; do
  scripts/peer_run.sh --cpu 3 --mem 8G --label ladder_gdag_bv4$v --max-seconds 10800 --detach -- env EXPERT=bc FLOW=$F GENCTX=1 \
    SEED=3900000 bash scripts/ladder_dagger_collect.sh $REP artifacts/runs/ladder_dagger_bv4${v}_gdag1 24 $g
done
scripts/peer_run.sh --cpu 3 --mem 10G --label ladder_orcbc_bv4${v}r3 --max-seconds 7200 --detach -- bash scripts/ladder_eval_orcbc.sh $REP bindv4${v}bcdag3noqd
wait_n "$R/ladder_dagger_bv4${v}_gdag1/*.pkl" 13
echo "$(date) gdag collected"
scripts/peer_run.sh --gpu --gpu-mem 8G --cpu 3 --mem 20G --label ladder_rz_bindv4${v}_gendag1 --max-seconds 14400 -- $PYP scripts/ladder_refit.py configs/ladder/rz_bindv4${v}_gendag1_noqd.json
echo "$(date) gendag1 refit done"
scripts/peer_run.sh --cpu 3 --mem 10G --label ladder_eval_bv4${v}gd1 --max-seconds 10800 --detach -- bash -c "export CUDA_VISIBLE_DEVICES=
  bash scripts/ladder_eval_orcbc.sh artifacts/runs/ladder_rz_bindv4${v}_gendag1_noqd/representation.pt bindv4${v}gendag1noqd
  for r in parm6_tf3 panda_pg2; do $PYP scripts/ladder.py --route generated --flow $F --rep artifacts/runs/ladder_rz_bindv4${v}_gendag1_noqd/representation.pt --prev-action zero --robot \$r --n 30 --tag zero_flowbv4${v}12k_rzbv4${v}gendag1noqd --out artifacts/runs/ladder_v1/\$r; done"
scripts/peer_run.sh --gpu --gpu-mem 16G --cpu 4 --mem 24G --label ladder_flow_bindv4${v}_gdag1 --max-seconds 14400 -- $PYP -m rrp.cli latent train-flow --config configs/ladder/flow_bindv4${v}_gdag1.json
echo "$(date) flow gdag1 done"
scripts/peer_run.sh --cpu 3 --mem 10G --label ladder_r2_bv4${v}_fgdag --max-seconds 10800 --detach -- bash -c "export CUDA_VISIBLE_DEVICES=
  for r in parm6_tf3 panda_pg2; do $PYP scripts/ladder.py --route generated --flow artifacts/runs/ladder_flow_bindv4${v}_gdag1/policy.pt --rep artifacts/runs/ladder_rz_bindv4${v}_gendag1_noqd/representation.pt --prev-action zero --robot \$r --n 30 --tag zero_flowbv4${v}gdag1_rzbv4${v}gendag1noqd --out artifacts/runs/ladder_v1/\$r; done"
echo "$(date) chain done"
