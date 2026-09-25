#!/usr/bin/env bash
# Track `binding`, whole-pipeline binding diversity (v3): Stage A on paired scenes -> post-hoc probes ->
# counterfactual tests -> Stage B flow + dev eval + disturbance + video (latent_chain_v2.sh) -> closed-loop
# paired-binding eval. One variant per invocation (sem | nosem); idempotent (done-files); serial foreground leases.
# Run from the track's peer dir:  RRP_PEER_REPO=$PWD scripts/binding_chain_v3.sh sem
set -uo pipefail
V=${1:?variant sem|nosem}
P=/dev/shm/rrp-brandonin
cd "${RRP_PEER_REPO:-$P/repo}"
export PATH=$P/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD RRP_OPS_ROOT=${RRP_OPS_ROOT:-$P/repo} MUJOCO_GL=egl
PY=$P/venv/bin/python
run() { python3 -m rrp.cli ops run "$@"; }
N=binding_paired_${V}_v3
R=artifacts/runs/$N
PK=artifacts/packed/binding_combined_v1_H16
[ -f $PK/meta.json ] || { echo "missing $PK"; exit 1; }
for attempt in 1 2 3; do          # 6 h lease cap: training resumes from rep_last.pt
  [ -f $R/representation.pt ] || run --gpu --gpu-mem 16G --cpu 5 --mem 24G --label rep_$N --max-seconds 21600 -- \
    $PY -m rrp.cli latent train-representation --config configs/latent/rep-$N.json
done
[ -f $R/representation.pt ] || { echo "rep $N did not finish"; exit 1; }
[ -f $R/probe_bindcf.pt ] || run --gpu --gpu-mem 6G --cpu 3 --mem 12G --label probe_$N --max-seconds 7200 -- \
  $PY -m rrp.cli latent fit-probes --representation $R/representation.pt --packed-dir $PK --binding-cf 0.5 --out $R/probe_bindcf.pt
if [ $V = sem ] && [ ! -f $R/probe_metadata_bindcf.pt ]; then
  run --gpu --gpu-mem 6G --cpu 3 --mem 12G --label probemeta_$N --max-seconds 7200 -- \
    $PY -m rrp.cli latent fit-probes --representation $R/representation.pt --packed-dir $PK --binding-cf 0.5 \
      --metadata-only --out $R/probe_metadata_bindcf.pt
fi
for ds in pick_place_primary_v3dart pick_place_paired_v1; do
  [ -f $R/counterfactuals_$ds.json ] || run --gpu --gpu-mem 4G --cpu 2 --mem 8G --label cf_$N --max-seconds 3600 -- \
    $PY -m rrp.cli latent counterfactuals --representation $R/representation.pt --probe $R/probe_bindcf.pt --gpu \
      --dataset artifacts/datasets/$ds --n 60 --per-episode 2 --out $R/counterfactuals_$ds.json
done
RRP_PEER_REPO=$PWD bash scripts/latent_chain_v2.sh $N
F=artifacts/runs/flow_$N
if [ -f $F/policy.pt ] && [ ! -f $F/eval_binding.done ]; then
  rm -f $F/eval_binding.jsonl
  run --gpu --gpu-mem 8G --cpu 6 --mem 16G --label evalbind_$N --max-seconds 10800 -- \
    $PY -m rrp.cli latent eval-binding --checkpoint $F/policy.pt --robots panda_pg2,parm6_tf3 --scenes 10 \
      --seed-start 3100000 --method flow_$N --out $F/eval_binding.jsonl > $F/eval_binding.summary.txt 2>&1 \
    && touch $F/eval_binding.done
fi
echo binding-chain-v3-$V-done
